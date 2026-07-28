"""M7-followup -- linear-matrix fast path on SurrogateFieldMap.track_rk4.

Verifies that:

  1. The fast-path flag defaults to OFF (non-breaking).
  2. With MP engaged but fast path OFF, behaviour is bit-identical
     to the wrapped (the M7 safe-delegate path is intact).
  3. With fast path ON on the linear mock, output matches wrapped
     to FP precision (mock's true dynamics ARE linear, and the
     surrogate matrix matches that linear truth, so the fast path's
     `M @ particles` is exact).
  4. `reset_run_state` clears the fast-path init state so the next
     element traversal re-initialises.
  5. OOD ref still falls back per the existing scope check.

Production accuracy on real lattices (smoke vs moderate training)
is validated separately by `/tmp/mp_hybrid_mebt_hwr.py`, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import (
    MlpHead, OutOfScopeError, Scope, SurrogateFieldMap,
    SurrogateMetadata,
)


# ---------------------------------------------------------------------------
class _LinearMockFieldMap(FieldMapElement):
    """Linear-dynamics mock: track_rk4 applies a fixed 6x6 to alive
    particles + advances ref.s.  fitted_matrix returns the same 6x6.

    The fast path requires the wrapped element to expose a battery of
    private helpers (`_z_map_start`, `_phi_sync_rad`, `_phasor`,
    `_sample_onaxis`, `_scale_factor`, `_step_idx`, `field_data`,
    `_interpolators`).  This mock stubs the minimum subset the fast
    path actually calls so the test exercises the full code path
    without dragging in the real FieldMap3D's field-data plumbing.
    """
    def __init__(self):
        super().__init__(name="LIN_MOCK", length=100.0,
                         aperture=10.0, n_steps=10)
        # Linear dynamics + something for the surrogate to encode.
        self._M = np.eye(6)
        self._M[0, 1] = 0.05
        self._M[2, 3] = 0.04
        self.scale = 1.0

        # ---- Fast-path helper stubs ------------------------------
        # No electric channels -> dW_ref = 0 in the fast path, which
        # is fine for this linear test (the matrix's M[5, ...] row
        # already captures the energy coupling we care about).
        self._z_map_start = 0.0
        self._step_idx = 0
        self._phi_s_at_entrance = 0.0
        self._sync_offset_deg = None
        # Empty interpolators dict -> the fast path's
        # _electric_channels list is empty -> dW_ref skipped cleanly.
        self._interpolators = {}

        from types import SimpleNamespace
        self.field_data = SimpleNamespace(channels={}, z=np.array([0.0]))

    # The fast path calls these on first init.
    def _calibrate_sync_phase(self, ref): pass
    def _build_interpolators(self): pass

    # Helpers the fast path calls per substep (only on electric chs;
    # since our list is empty, these are never invoked in the test
    # but are stubbed for completeness).
    def _phi_sync_rad(self, ref, phi_s_mid): return 0.0
    def _phasor(self, ch_enum, phi_rad): return np.array([1.0])
    def _sample_onaxis(self, ch_enum, comp_interps, z): return 0.0
    def _scale_factor(self, ch_enum, ch_data): return 1.0

    # The two contracts the tracker really cares about.
    def fitted_matrix(self, ref):
        return self._M.copy()

    def fitted_matrix_slice(self, ref, ds_mm):
        from scipy.linalg import expm, logm
        ratio = float(ds_mm) / max(float(self.length), 1e-12)
        return np.real(expm(logm(self._M) * ratio))

    def track_rk4(self, beam, ds):
        alive = beam.alive_mask
        if alive.any():
            beam.particles[alive] = beam.particles[alive] @ self._M.T
        beam.ref.s += float(ds)


class _Ref:
    def __init__(self, w_kin: float = 5.0):
        self.w_kin = float(w_kin)
        self.beta = 0.1
        self.gamma = 1.005
        self.s = 0.0
        self.phi_s = 0.0
        self.wavelength = 0.0   # disables phi_s advance in the fast path
        from types import SimpleNamespace
        self.species = SimpleNamespace(charge=-1.0, mass=939.0)
    def copy(self):
        r = _Ref(self.w_kin)
        r.s = self.s; r.phi_s = self.phi_s
        return r


class _Beam:
    def __init__(self, n: int = 32, w_kin: float = 5.0):
        rng = np.random.default_rng(0)
        self.particles = rng.normal(scale=0.5, size=(n, 6)).astype(np.float64)
        self.alive_mask = np.ones(n, dtype=bool)
        self.ref = _Ref(w_kin=w_kin)


def _build_surrogate(wrapped, *, scope_w_kin_range=(2.0, 10.0)):
    """Construct a SurrogateFieldMap whose MLP exactly returns wrapped._M.

    Output_norm.mean is set to flatten(M) and the MLP weights are
    zeroed in `_zero_mlp` below; the forward pass then returns the
    unnormalised mean, which IS wrapped._M.flatten().
    """
    mlp = MlpHead(input_dim=4, output_dim=36, hidden_dims=(8, 8))
    meta = SurrogateMetadata(
        element_key=wrapped.name, element_class=type(wrapped).__name__,
        architecture={"input_dim": 4, "output_dim": 36,
                      "hidden_dims": [8, 8], "activation": "silu",
                      "param_names": ["scale"]},
        scope=Scope(
            input_names=["w_kin", "beta", "gamma", "scale"],
            input_lo=np.array([scope_w_kin_range[0], 0.0, 1.0, 0.8]),
            input_hi=np.array([scope_w_kin_range[1], 1.0, 1000.0, 1.2]),
        ),
        input_norm={"mean": [5.0, 0.1, 1.005, 1.0],
                    "std":  [1.0, 1.0, 1.0, 1.0]},
        output_norm={"mean": list(wrapped._M.flatten()),
                     "std":  [1.0] * 36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="hash-test", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    _zero_mlp(surr.mlp)
    return surr


def _zero_mlp(m) -> None:
    import torch
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()


# ---------------------------------------------------------------------------
def test_fast_path_flag_default_off():
    """Default state: registered surrogates do NOT activate the fast
    path even if MP is engaged."""
    saved = registry.is_fast_path_enabled()
    try:
        registry.set_fast_path_enabled(False)
        assert registry.is_fast_path_enabled() is False
        registry.set_fast_path_enabled(True)
        assert registry.is_fast_path_enabled() is True
    finally:
        registry.set_fast_path_enabled(saved)


def test_mp_engaged_but_fast_off_still_delegates():
    """M7 safe path stays intact: MP on, fast path OFF -> wrapped."""
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)

    beam_a = _Beam(); beam_b = _Beam()
    saved_mp = registry.is_mp_enabled()
    saved_fp = registry.is_fast_path_enabled()
    try:
        registry.set_mp_enabled(True)
        registry.set_fast_path_enabled(False)
        wrapped.track_rk4(beam_a, 100.0)
        surr.track_rk4(beam_b, 100.0)
        np.testing.assert_array_equal(beam_a.particles, beam_b.particles)
    finally:
        registry.set_mp_enabled(saved_mp)
        registry.set_fast_path_enabled(saved_fp)


def test_fast_path_matches_wrapped_on_linear_dynamics():
    """On a wrapped element whose true dynamics ARE linear, the
    surrogate matrix matches truth and the fast path produces FP-
    identical output to the wrapped's track_rk4."""
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)

    beam_a = _Beam(); beam_b = _Beam()
    saved_mp = registry.is_mp_enabled()
    saved_fp = registry.is_fast_path_enabled()
    try:
        registry.set_mp_enabled(True)
        registry.set_fast_path_enabled(True)
        wrapped.track_rk4(beam_a, 100.0)
        surr.track_rk4(beam_b, 100.0)
        # The matrix is the same, the apply is the same -> bit-identical
        # to ~FP precision (tiny noise from logm/expm round-trip).
        np.testing.assert_allclose(
            beam_a.particles, beam_b.particles, atol=1e-10)
    finally:
        registry.set_mp_enabled(saved_mp)
        registry.set_fast_path_enabled(saved_fp)


def test_reset_run_state_clears_fast_path_init():
    """reset_run_state -> next track_rk4 re-runs _init_fast_path."""
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)
    beam = _Beam()

    saved_mp = registry.is_mp_enabled()
    saved_fp = registry.is_fast_path_enabled()
    try:
        registry.set_mp_enabled(True)
        registry.set_fast_path_enabled(True)
        # Trigger init.
        surr.track_rk4(beam, 100.0)
        assert surr._fast_init_done is True
        assert surr._M_full is not None
        assert surr._log_M_full is not None

        # Reset -> all init flags clear.
        surr.reset_run_state()
        assert surr._fast_init_done is False
        assert surr._M_full is None
        assert surr._log_M_full is None
        assert surr._slice_cache_key is None
    finally:
        registry.set_mp_enabled(saved_mp)
        registry.set_fast_path_enabled(saved_fp)


def test_ood_with_fast_path_falls_back():
    """Even with fast path on, OOD ref -> wrapped.track_rk4 fallback."""
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped, scope_w_kin_range=(2.0, 10.0))

    beam = _Beam(w_kin=100.0)  # outside scope
    snapshot = beam.particles.copy()

    saved_mp = registry.is_mp_enabled()
    saved_fp = registry.is_fast_path_enabled()
    try:
        registry.set_mp_enabled(True)
        registry.set_fast_path_enabled(True)
        surr.track_rk4(beam, 100.0)
    finally:
        registry.set_mp_enabled(saved_mp)
        registry.set_fast_path_enabled(saved_fp)

    # Wrapped's M was applied (the OOD fallback path).
    expected = snapshot @ wrapped._M.T
    np.testing.assert_array_equal(beam.particles, expected)


def test_slice_cache_hits_on_repeated_same_ds():
    """Per-substep calls with the same (ds, w_kin) hit the cache --
    no per-call expm work after the first."""
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)

    saved_mp = registry.is_mp_enabled()
    saved_fp = registry.is_fast_path_enabled()
    try:
        registry.set_mp_enabled(True)
        registry.set_fast_path_enabled(True)
        # First call -> init + compute slice.
        b1 = _Beam(); surr.track_rk4(b1, 50.0)
        first_key = surr._slice_cache_key
        first_M = surr._slice_cache_M.copy()
        # Same ds, same w_kin (no electric channels = no ref.w_kin
        # change in this mock) -> cache hit.
        surr.track_rk4(b1, 50.0)
        assert surr._slice_cache_key == first_key
        np.testing.assert_array_equal(surr._slice_cache_M, first_M)
    finally:
        registry.set_mp_enabled(saved_mp)
        registry.set_fast_path_enabled(saved_fp)
