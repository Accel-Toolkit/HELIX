"""M3 — envelope-mode surrogate hook: verify the seam in
``_fitted_matrix_slice_at`` consults the registry, routes through
the surrogate when registered, and falls back on OutOfScopeError."""
import numpy as np

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import (
    MlpHead, OutOfScopeError, Scope, SurrogateFieldMap, SurrogateMetadata,
)
from linac_gen.tracking.envelope import _fitted_matrix_slice_at


# ---------------------------------------------------------------------------
class _MockElem(FieldMapElement):
    """A minimal FieldMapElement whose fitted_matrix_slice returns a
    sentinel matrix so the test can tell which code path ran."""

    SENTINEL = 999.0

    def __init__(self, name: str = "MOCK_FM"):
        super().__init__(name=name, length=100.0, aperture=10.0, n_steps=1)

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        return np.eye(6) * 7.0

    def fitted_matrix_slice(self, ref, ds_mm):
        # Return a sentinel matrix the test can pattern-match on.
        return np.eye(6) * self.SENTINEL


class _MockRef:
    def __init__(self, w_kin: float = 3.0):
        self.w_kin = float(w_kin); self.beta = 0.1; self.gamma = 1.005

    def copy(self):
        return _MockRef(self.w_kin)


def _build_surrogate(name: str, *, in_scope: bool = True) -> SurrogateFieldMap:
    """A SurrogateFieldMap whose fitted_matrix_slice returns a recognisable
    sentinel matrix (or raises OutOfScopeError if ``in_scope`` is False)."""
    wrapped = _MockElem(name=name)
    mlp = MlpHead(input_dim=3, output_dim=36, hidden_dims=(4,))
    meta = SurrogateMetadata(
        element_key=name, element_class="_MockElem",
        architecture={"input_dim": 3, "output_dim": 36,
                      "hidden_dims": [4], "activation": "silu",
                      "param_names": []},
        scope=Scope(input_names=["w_kin", "beta", "gamma"],
                    input_lo=np.array([2.0, 0.0, 1.0]) if in_scope
                             else np.array([99.0, 0.0, 1.0]),  # forces OOD
                    input_hi=np.array([10.0, 1.0, 100.0])),
        input_norm={"mean": [0.0]*3, "std": [1.0]*3},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="lh", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    # Override fitted_matrix_slice to return a sentinel matrix (so the
    # test can tell that the surrogate path -- not the wrapped -- ran).
    surr.fitted_matrix_slice = lambda ref, ds_mm: np.eye(6) * 42.0
    if not in_scope:
        # Force the seam to raise OutOfScopeError, exercising fall-back.
        def _raise(ref, ds_mm):
            raise OutOfScopeError("test out-of-scope")
        surr.fitted_matrix_slice = _raise
    return surr


# ---------------------------------------------------------------------------
def test_seam_falls_through_when_no_surrogate_registered():
    """With an empty registry the helper hits the wrapped element."""
    registry.clear()
    elem = _MockElem()
    M = _fitted_matrix_slice_at(elem, _MockRef(), ds_mm=1.0, z_from_mm=0.0)
    assert M[0, 0] == _MockElem.SENTINEL   # wrapped element ran
    registry.clear()


def test_seam_routes_through_registered_surrogate():
    """When a surrogate is registered by name, the helper uses it."""
    registry.clear()
    elem = _MockElem(name="EL_X")
    surr = _build_surrogate("EL_X", in_scope=True)
    registry.register(surr)
    M = _fitted_matrix_slice_at(elem, _MockRef(), ds_mm=1.0, z_from_mm=0.0)
    assert M[0, 0] == 42.0   # surrogate ran
    registry.clear()


def test_seam_falls_back_on_out_of_scope():
    """OutOfScopeError from the surrogate is caught; wrapped runs instead."""
    registry.clear()
    elem = _MockElem(name="EL_Y")
    surr = _build_surrogate("EL_Y", in_scope=False)  # raises OOD
    registry.register(surr)
    M = _fitted_matrix_slice_at(elem, _MockRef(), ds_mm=1.0, z_from_mm=0.0)
    assert M[0, 0] == _MockElem.SENTINEL   # fell back to wrapped element
    registry.clear()


def test_by_name_lookup_drops_when_lattice_hash_unregistered():
    """unregister() removes from _BY_NAME when no other reg keeps the name alive."""
    registry.clear()
    surr = _build_surrogate("EL_Z", in_scope=True)
    registry.register(surr)
    assert registry.get_by_element_name("EL_Z") is not None
    registry.unregister(surr.metadata.lattice_hash, "EL_Z")
    assert registry.get_by_element_name("EL_Z") is None


# ---------------------------------------------------------------------------
# Default fitted_matrix_slice on SurrogateFieldMap actually uses the
# surrogate NN (via _slice_matrix), not the wrapped element's RK4.
# Regression test for the "Use checkbox does nothing for envelope" bug.
# ---------------------------------------------------------------------------
class _CountingWrapped(FieldMapElement):
    """FieldMapElement that counts how many times its fitted_matrix_slice
    was called.  Lets us assert the surrogate path does NOT fall back to
    wrapped RK4 on in-scope inputs."""

    SENTINEL = 555.0

    def __init__(self, name: str = "CTW"):
        super().__init__(name=name, length=100.0, aperture=10.0, n_steps=1)
        self.n_slice_calls = 0

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        # An identity-like matrix that the surrogate's logm/expm can
        # exponentiate cleanly.
        return np.eye(6)

    def fitted_matrix_slice(self, ref, ds_mm):
        self.n_slice_calls += 1
        return np.eye(6) * self.SENTINEL


def _build_real_surrogate(name: str, wrapped) -> SurrogateFieldMap:
    """Real SurrogateFieldMap (no monkey-patch on fitted_matrix_slice).

    The MLP is a tiny 3-input -> 36-output network with random weights;
    we override the surrogate's fitted_matrix() to return identity so
    _slice_matrix's logm/expm produces identity slices regardless of
    weights -- the test is about CODE PATH, not numerical accuracy.
    """
    mlp = MlpHead(input_dim=3, output_dim=36, hidden_dims=(4,))
    meta = SurrogateMetadata(
        element_key=name, element_class="_CountingWrapped",
        architecture={"input_dim": 3, "output_dim": 36,
                      "hidden_dims": [4], "activation": "silu",
                      "param_names": []},
        scope=Scope(input_names=["w_kin", "beta", "gamma"],
                    input_lo=np.array([2.0, 0.0, 1.0]),
                    input_hi=np.array([10.0, 1.0, 100.0])),
        input_norm={"mean": [0.0]*3, "std": [1.0]*3},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="lh", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    # Pin fitted_matrix to identity so _slice_matrix's expm/logm doesn't
    # depend on the random NN weights for the slice-path code-path test.
    surr.fitted_matrix = lambda ref: np.eye(6)
    return surr


def test_surrogate_fitted_matrix_slice_uses_NN_for_full_element_slice():
    """When ds_mm == element.length (the full-element transport, which
    is what a no-SC envelope pass requests), fitted_matrix_slice must
    use the surrogate's NN prediction via fitted_matrix(ref) -- NOT
    delegate to the wrapped RK4.  This is the only path where the
    matrix-log-exp slice composition is exact, so it's the only place
    the surrogate is allowed to override RK4.
    """
    registry.clear()
    wrapped = _CountingWrapped(name="FULL_SURR")
    surr = _build_real_surrogate("FULL_SURR", wrapped)
    M = surr.fitted_matrix_slice(_MockRef(), ds_mm=wrapped.length)
    assert wrapped.n_slice_calls == 0, (
        f"Full-element slice should use NN, not RK4.  wrapped was called "
        f"{wrapped.n_slice_calls} time(s)."
    )
    np.testing.assert_allclose(M, np.eye(6), atol=1e-9)
    registry.clear()


def test_surrogate_fitted_matrix_slice_delegates_for_partial_slice():
    """When ds_mm < element.length (envelope SC bundle splitting an
    element), fitted_matrix_slice MUST delegate to wrapped RK4.

    Why: the surrogate's NN gives M_full (end-to-end linear matrix).
    Composing a partial slice via expm(logm(M_full) × ratio) only
    equals the true RK4 slice when the per-substep dynamics is
    constant-coefficient.  For FieldMap RF cavities (z-varying
    sin(ωt) field) the log-exp slice gives a time-averaged
    approximation that diverges from RK4.  RK4 is the honest answer.

    Regression test against re-enabling a too-aggressive surrogate
    slice that breaks SC-active envelope accuracy.
    """
    registry.clear()
    wrapped = _CountingWrapped(name="PART_SURR")
    surr = _build_real_surrogate("PART_SURR", wrapped)
    # ds_mm well below element.length -- partial slice.
    M = surr.fitted_matrix_slice(_MockRef(), ds_mm=wrapped.length / 5.0)
    assert wrapped.n_slice_calls == 1, (
        f"Partial slice must delegate to wrapped.fitted_matrix_slice; "
        f"got {wrapped.n_slice_calls} call(s) to wrapped."
    )
    assert M[0, 0] == _CountingWrapped.SENTINEL
    registry.clear()


def test_surrogate_fitted_matrix_slice_falls_back_on_OOD_for_full_element():
    """OOD on a full-element slice falls back to wrapped RK4
    (same OOD safety the envelope hook in envelope.py enforces)."""
    registry.clear()
    wrapped = _CountingWrapped(name="OOD_SURR")
    surr = _build_real_surrogate("OOD_SURR", wrapped)
    def _ood(ref):
        raise OutOfScopeError("w_kin outside scope")
    surr.fitted_matrix = _ood
    M = surr.fitted_matrix_slice(_MockRef(), ds_mm=wrapped.length)
    assert wrapped.n_slice_calls == 1
    assert M[0, 0] == _CountingWrapped.SENTINEL
    registry.clear()


# ---------------------------------------------------------------------------
# END-TO-END PHYSICS REGRESSION
#
# A unit test on the slice method's call counter is necessary but not
# sufficient: it can pass even if the slice method returns the WRONG
# matrix.  This higher-level test verifies that envelope tracking
# through a z-varying FieldMap with SC-style slicing gives the SAME
# end-of-line emittance whether the surrogate is registered or not.
#
# This catches the family of bugs where a "speedup" hook inadvertently
# replaces RK4 with a less-accurate approximation (e.g. expm(logm(M))
# slicing of a time-varying RF field's full matrix).
# ---------------------------------------------------------------------------
class _ZVaryingMock(FieldMapElement):
    """Mock FieldMapElement whose RK4 slice produces a z-position-
    dependent matrix (mimicking RF cavity behaviour).  Its
    ``fitted_matrix`` (full element) returns identity, so any
    expm(logm(I) × ratio) slice would be identity -- whereas the
    true RK4 slice depends on ds.  If the surrogate's hook ever
    routes partial slices through expm(logm), the envelope output
    will diverge measurably from the no-surrogate run.
    """

    def __init__(self, name: str = "ZVAR_FM"):
        super().__init__(name=name, length=100.0, aperture=10.0, n_steps=1)

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        # Full-element matrix is identity; the z-varying physics is
        # entirely in fitted_matrix_slice (so partial slices and the
        # full slice differ in their composition).
        return np.eye(6)

    def fitted_matrix_slice(self, ref, ds_mm, **kw):
        # Synthetic z-varying force: every slice gives a small rotation
        # proportional to ds_mm.  Five slices of length/5 compose to
        # rotation(theta) which is DIFFERENT from the full element
        # matrix (identity) -- exactly the kind of z-varying behaviour
        # an RF cavity exhibits.
        theta = 0.01 * float(ds_mm)
        c, s = np.cos(theta), np.sin(theta)
        M = np.eye(6)
        M[0:2, 0:2] = np.array([[c, s], [-s, c]])
        # x' rotation pair to keep the matrix symplectic-ish.
        M[1, 0] = -s
        M[1, 1] = c
        return M


def test_envelope_with_surrogate_matches_envelope_without_surrogate_endtoend():
    """End-to-end physics regression: envelope tracking through a
    z-varying mock FieldMap must produce the same end-of-line sigma
    matrix whether the surrogate is registered or not.

    Catches the eaa4a07 family of bugs where the surrogate's slice
    method returns an approximation (e.g. expm(logm(M_full) × ratio))
    that diverges from RK4 for time-varying fields.
    """
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.tracking.envelope import _fitted_matrix_slice_at

    # Run the slice through the actual seam helper -- the same path
    # the EnvelopeSolver uses.  Compose 5 partial slices and check the
    # composition equals the wrapped element's 5-slice composition.
    registry.clear()
    elem = _ZVaryingMock(name="ZTEST")

    # No-surrogate baseline: 5 sequential calls give 5 rotations of
    # theta = 0.01 * length/5 each, composing to rotation(0.01 * length).
    M_baseline = np.eye(6)
    ref = _MockRef()
    for _ in range(5):
        M_step = _fitted_matrix_slice_at(
            elem, ref, ds_mm=elem.length / 5.0, z_from_mm=0.0)
        M_baseline = M_step @ M_baseline

    # Register a surrogate wrapping the same element.  fitted_matrix
    # (full element) is pinned to identity by the mock above; any bug
    # that routes partial slices through expm/logm(identity) would
    # give identity for every slice -> composed = identity, NOT
    # rotation(0.01 * length).  The wrapped path correctly produces
    # rotation(0.01 * length).
    surr = _build_real_surrogate("ZTEST", elem)
    # surr.fitted_matrix is pinned to identity by _build_real_surrogate;
    # the mock's RK4 slice has the z-varying physics.
    registry.register(surr)

    M_with_surr = np.eye(6)
    for _ in range(5):
        M_step = _fitted_matrix_slice_at(
            elem, ref, ds_mm=elem.length / 5.0, z_from_mm=0.0)
        M_with_surr = M_step @ M_with_surr
    registry.clear()

    # With the current (correct) implementation: partial slices delegate
    # to wrapped RK4 -> M_with_surr equals M_baseline exactly.
    # With the eaa4a07 bug: surrogate's slice would return identity for
    # every partial slice -> M_with_surr would be identity, far from
    # M_baseline = rotation(0.01 * length).
    np.testing.assert_allclose(
        M_with_surr, M_baseline, atol=1e-12,
        err_msg=(
            "Envelope through a z-varying FieldMap diverged between "
            "surrogate-registered and non-surrogate runs.  The "
            "surrogate's partial-slice path is no longer delegating "
            "to the wrapped element's RK4 -- regression of eaa4a07."
        ),
    )


def test_ood_wrapper_in_lattice_falls_back_not_aborts():
    """External-review find (2026-07-20): a SurrogateFieldMap placed IN
    the lattice (compare flows) used to ABORT envelope runs with an
    uncaught OutOfScopeError from get_element_matrix — the registry
    contract is 'caller falls back to the wrapped element', so the
    matrix path must honour it too.  Pinned at I=0 AND I>0 with a REAL
    FieldMap as the wrapped element."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.io.field_map_reader import FieldMapData
    from linac_gen.tracking.envelope import EnvelopeSolver

    z = np.linspace(0.0, 0.12, 13)
    cav = FieldMap("CAV1", length=120.0,
                   field_data=FieldMapData(
                       z=z, Ez=2.0 * np.sin(np.pi * z / 0.12),
                       symmetry="1d"),
                   phase=-25.0, frequency=352.21, aperture=30.0,
                   n_steps=60)

    class _OOSSurr(SurrogateFieldMap):
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self._enabled = True
            self._fast_init_failed = False
            self.name = wrapped.name

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_wrapped"),
                           item)

        def fitted_matrix(self, ref):
            raise OutOfScopeError("always OOD (test)")

    lat = Lattice()
    lat.add(Drift("D1", length=50.0, aperture=30.0))
    lat.add(_OOSSurr(cav))            # wrapper IS the lattice element
    lat.add(Drift("D2", length=50.0, aperture=30.0))
    init = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
                alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                alpha_z=0.0, beta_z=10.0, emit_z=0.3)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    for cur in (0.0, 20.0):
        res = EnvelopeSolver(lat, ref, dict(init), current=cur).run()
        assert res.sigma_x[-1] > 0            # completed, no abort
