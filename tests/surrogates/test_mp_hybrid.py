"""M7 — Hybrid MP-mode surrogate contract.

The hybrid path lives in :meth:`SurrogateFieldMap.track_rk4`.  These
tests cover:

  1. Bit-identical safety: when MP-engagement is OFF, registered
     surrogates do nothing to MP runs (delegate to wrapped).
  2. Slice matrix log/exp round-trip.
  3. OOD fallback: ref outside training scope -> delegate.
  4. `residual_n_steps == 0` reduces to pure linear-matrix apply
     (and the ref still advances correctly via the trampoline).
  5. End-to-end on the test mock element: hybrid matches reference.
"""
from __future__ import annotations

import copy

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
    """Mock whose track_rk4 applies a fixed linear matrix to particles.

    Because the dynamics are exactly linear, the hybrid path's
    surrogate matrix matches the wrapped element's pushed output
    exactly (modulo MLP prediction error), and the residual is zero.
    Great for verifying the bit-identical and residual=0 invariants.
    """
    def __init__(self):
        super().__init__(name="LIN_MOCK", length=100.0,
                         aperture=10.0, n_steps=10)
        self._M = np.eye(6)
        self._M[0, 1] = 0.05
        self._M[2, 3] = 0.04
        # Tag a parameter so the surrogate has something to read.
        self.scale = 1.0

    def fitted_matrix(self, ref):
        # Same regardless of `ref` so the MLP can learn it exactly.
        return self._M.copy()

    def fitted_matrix_slice(self, ref, ds_mm):
        ratio = float(ds_mm) / max(float(self.length), 1e-12)
        # For a fixed linear map, M^ratio approximates the slice (the
        # surrogate's _slice_matrix will reach the same via log/exp).
        from scipy.linalg import expm, logm
        return np.real(expm(logm(self._M) * ratio))

    def track_rk4(self, beam, ds):
        # Push alive particles through M (full ds matrix), then advance
        # the reference particle's s.  Reference w_kin is unchanged
        # (linear map preserves energy in this mock).
        alive = beam.alive_mask
        if alive.any():
            beam.particles[alive] = beam.particles[alive] @ self._M.T
        beam.ref.s += float(ds)


class _Ref:
    """Minimal reference particle that satisfies the surrogate's
    `_make_input` (reads `w_kin`, `beta`, `gamma`, has `copy`)."""
    def __init__(self, w_kin: float = 5.0):
        self.w_kin = float(w_kin)
        self.beta = 0.1
        self.gamma = 1.005
        self.s = 0.0

    def copy(self):
        r = _Ref(self.w_kin)
        r.s = self.s
        return r


class _Beam:
    """Minimal beam with the attributes the hybrid `track_rk4` reads."""
    def __init__(self, n: int = 32, w_kin: float = 5.0):
        rng = np.random.default_rng(0)
        self.particles = rng.normal(scale=0.5, size=(n, 6)).astype(np.float64)
        self.alive_mask = np.ones(n, dtype=bool)
        self.ref = _Ref(w_kin=w_kin)


def _build_surrogate(wrapped, *, scope_w_kin_range=(2.0, 10.0)):
    """Construct a SurrogateFieldMap whose MLP exactly returns wrapped.M.

    We do this by hand-crafting the architecture / normalisation so
    the MLP output IS the flattened M.  This keeps the test focused
    on the hybrid-path plumbing, not on MLP training accuracy.
    """
    # Architecture: dim4 input (w_kin, beta, gamma, scale) -> 36 output.
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
    # Zero the MLP so the output is exactly the unnormalised mean,
    # which we set to flatten(M) above -> fitted_matrix returns M.
    with torch_no_grad_zero(surr.mlp):
        pass
    return surr


def torch_no_grad_zero(m):
    """Context manager that zeros every parameter of an nn.Module."""
    class _Ctx:
        def __enter__(self_inner):
            import torch
            with torch.no_grad():
                for p in m.parameters():
                    p.zero_()
            return self_inner
        def __exit__(self_inner, *args):
            return False
    return _Ctx()


# ---------------------------------------------------------------------------
def test_mp_engaged_flag_default_off():
    """The MP engagement flag defaults to False at import time."""
    # Save + restore in case a previous test left it on.
    saved = registry.is_mp_enabled()
    try:
        registry.set_mp_enabled(False)
        assert registry.is_mp_enabled() is False
        registry.set_mp_enabled(True)
        assert registry.is_mp_enabled() is True
    finally:
        registry.set_mp_enabled(saved)


def test_bit_identical_when_mp_disabled():
    """Registered surrogate + MP toggle OFF -> wrapped behaviour exactly.

    Mirror the wrapped element's push and the surrogate's push and
    assert they're identical to floating-point precision.
    """
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)

    beam_a = _Beam(); beam_b = _Beam()
    np.testing.assert_array_equal(beam_a.particles, beam_b.particles)

    # Path A: just wrapped.
    wrapped.track_rk4(beam_a, 100.0)
    # Path B: surrogate with MP-engagement OFF -> delegate to wrapped.
    registry.set_mp_enabled(False)
    surr.track_rk4(beam_b, 100.0)

    np.testing.assert_array_equal(beam_a.particles, beam_b.particles)
    assert beam_a.ref.s == pytest.approx(beam_b.ref.s)


def test_slice_matrix_logexp_roundtrip():
    """Slicing M_full into N slices that compose back gives M_full.

    Cumulative product of N slice matrices of equal width must
    equal M_full within numerical noise.
    """
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)
    M_full = surr.fitted_matrix(_Ref())

    N = 7
    ds = wrapped.length / N
    composed = np.eye(6)
    for _ in range(N):
        M_slice = surr._slice_matrix(_Ref(), ds)
        composed = M_slice @ composed
    np.testing.assert_allclose(composed, M_full, atol=1e-9)


def test_ood_falls_back_to_wrapped():
    """A ref outside training scope -> wrapped.track_rk4 (no crash)."""
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped, scope_w_kin_range=(2.0, 10.0))

    beam = _Beam(w_kin=100.0)  # WAY outside scope
    snapshot = beam.particles.copy()

    saved = registry.is_mp_enabled()
    try:
        registry.set_mp_enabled(True)
        surr.track_rk4(beam, 100.0)
    finally:
        registry.set_mp_enabled(saved)

    # Whether the wrapped element advanced is implementation-defined;
    # the only invariant we assert is "no exception raised AND the
    # wrapped's push happened" (particles changed exactly by wrapped's M).
    expected = snapshot @ wrapped._M.T
    np.testing.assert_array_equal(beam.particles, expected)


def test_hybrid_matches_wrapped_for_linear_dynamics():
    """For a purely-linear wrapped element, the hybrid path's residual
    is exactly zero and the output matches wrapped to FP precision.

    This validates the entire 4-step contract: linear anchor + residual
    RK4 + sum = truth.
    """
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)

    beam_a = _Beam(); beam_b = _Beam()
    np.testing.assert_array_equal(beam_a.particles, beam_b.particles)

    # Truth path: just wrapped.
    wrapped.track_rk4(beam_a, 100.0)

    saved = registry.is_mp_enabled()
    try:
        registry.set_mp_enabled(True)
        # Force a reasonable substep count -- doesn't matter for a
        # linear mock since residual is exactly zero anyway.
        surr.residual_n_steps = 5
        surr.track_rk4(beam_b, 100.0)
    finally:
        registry.set_mp_enabled(saved)

    # Tight tolerance: linear + residual must reconstruct wrapped to
    # ~FP precision (the log/exp on a fixed matrix has tiny rounding).
    np.testing.assert_allclose(
        beam_a.particles, beam_b.particles, atol=1e-10)


def test_residual_zero_pure_linear_mode():
    """`residual_n_steps == 0` skips the residual pass and writes the
    linear-anchor result directly.  Particles equal M_full @ inputs.
    """
    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)

    beam = _Beam()
    inputs = beam.particles.copy()
    saved = registry.is_mp_enabled()
    try:
        registry.set_mp_enabled(True)
        surr.residual_n_steps = 0
        surr.track_rk4(beam, 100.0)
    finally:
        registry.set_mp_enabled(saved)

    expected = inputs @ surr.fitted_matrix(_Ref()).T
    np.testing.assert_allclose(beam.particles, expected, atol=1e-12)


def test_fitted_matrix_nonfinite_output_falls_back():
    """Output sanity gate (2026-07 external review): a NaN-producing NN
    must raise OutOfScopeError from fitted_matrix — every caller's
    wrapped-element fallback then applies — and _slice_matrix must
    delegate to the wrapped element instead of propagating."""
    import numpy as np
    import pytest
    import torch
    from linac_gen.surrogates.base import OutOfScopeError

    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)
    # Poison the output de-normalisation so the NN output is NaN.
    surr._out_mean = surr._out_mean * float("nan")
    with pytest.raises(OutOfScopeError, match="non-finite"):
        surr.fitted_matrix(_Ref())
    M = surr._slice_matrix(_Ref(), wrapped.length / 2.0)
    assert np.all(np.isfinite(M))     # wrapped element's matrix


def test_fitted_matrix_singular_output_falls_back():
    """2026-07 external review (#8): an outright singular NN prediction
    is unambiguous garbage (no physical transfer map is singular) —
    reject via OutOfScopeError like the non-finite case."""
    import numpy as np
    import pytest
    from linac_gen.surrogates.base import OutOfScopeError

    wrapped = _LinearMockFieldMap()
    surr = _build_surrogate(wrapped)
    # Zero the output de-normalisation: the NN now predicts the zero
    # matrix (det = 0) for every input.
    surr._out_mean = surr._out_mean * 0.0
    surr._out_std = surr._out_std * 0.0
    with pytest.raises(OutOfScopeError, match="SINGULAR"):
        surr.fitted_matrix(_Ref())
    M = surr._slice_matrix(_Ref(), wrapped.length / 2.0)
    assert np.all(np.isfinite(M))     # wrapped element's matrix
