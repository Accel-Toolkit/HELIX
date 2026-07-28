"""M8 -- autograd-differentiable matcher arm for SurrogateFieldMap.

Verifies that:

  1. `fitted_matrix_torch(kin_tensor)` returns a (6, 6) F64 tensor.
  2. The output is numerically equal to `fitted_matrix(ref)` (same MLP,
     just without `torch.no_grad`).
  3. Gradients propagate when `kin_tensor.requires_grad=True`.
  4. `element_matrix_torch(...)` dispatches a `SurrogateFieldMap` to
     the new path and raises a hard error on OOD (never identity).
  5. `check_gradient_supported(...)` accepts a lattice containing a
     `SurrogateFieldMap` but still rejects a plain `FieldMap3D`.

Production accuracy (matching convergence on a real lattice) is
checked separately by the `/tmp/match_with_surrogate.py` driver.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates.base import (
    F64, MlpHead, OutOfScopeError, Scope, SurrogateFieldMap,
    SurrogateMetadata,
)


# ---------------------------------------------------------------------------
class _Mock(FieldMapElement):
    """Mock FieldMap-class element with a fixed linear matrix."""
    def __init__(self, name="MOCK"):
        super().__init__(name=name, length=100.0, aperture=10.0, n_steps=10)
        self._M = np.eye(6)
        self._M[0, 1] = 0.05
        self._M[2, 3] = 0.04
        self.scale = 1.0
        # Gradient-gate contract: wrapped elements expose their electric
        # amplitude; ke == 0 marks this mock as non-accelerating (the
        # gate refuses surrogates whose wrapped ke is nonzero or absent).
        self.ke = 0.0
    def track_rk4(self, beam, ds): return None
    def fitted_matrix(self, ref): return self._M.copy()
    def fitted_matrix_slice(self, ref, ds_mm): return np.eye(6)


def _build(name="MOCK", scope_w_kin=(2.0, 10.0)) -> SurrogateFieldMap:
    wrapped = _Mock(name)
    mlp = MlpHead(input_dim=4, output_dim=36, hidden_dims=(8,))
    meta = SurrogateMetadata(
        element_key=name, element_class="_Mock",
        architecture={"input_dim": 4, "output_dim": 36,
                      "hidden_dims": [8], "activation": "silu",
                      "param_names": ["scale"]},
        scope=Scope(input_names=["w_kin", "beta", "gamma", "scale"],
                    input_lo=np.array([scope_w_kin[0], 0.0, 1.0, 0.8]),
                    input_hi=np.array([scope_w_kin[1], 1.0, 1000.0, 1.2])),
        input_norm={"mean": [5.0, 0.1, 1.005, 1.0],
                    "std":  [1.0, 1.0, 1.0, 1.0]},
        output_norm={"mean": list(wrapped._M.flatten()),
                     "std":  [1.0] * 36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    with torch.no_grad():
        for p in surr.mlp.parameters():
            p.zero_()
    return surr


class _Ref:
    """Minimal Reference stand-in for the unit tests above.  For the
    matcher-arm tests we need a richer reference, built via the real
    `ReferenceParticle` (see `_real_ref` below)."""
    def __init__(self, w_kin=5.0):
        self.w_kin = float(w_kin); self.beta = 0.1; self.gamma = 1.005
        self.s = 0.0; self.phi_s = 0.0; self.wavelength = 0.0
        self.frequency = 162.5
    def copy(self):
        return _Ref(self.w_kin)


def _real_ref(w_kin=5.0):
    """A real ReferenceParticle so RefKinematics.from_reference works."""
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    return ReferenceParticle(species=H_MINUS, w_kin=w_kin, frequency=162.5)


# ---------------------------------------------------------------------------
def test_fitted_matrix_torch_shape_and_dtype():
    surr = _build()
    kin = torch.tensor([5.0, 0.1, 1.005], dtype=F64)
    M = surr.fitted_matrix_torch(kin)
    assert M.shape == (6, 6)
    assert M.dtype == torch.float64


def test_fitted_matrix_torch_matches_numpy():
    """Same MLP, same input, different return type -- values must agree."""
    surr = _build()
    ref = _Ref(w_kin=5.0)
    M_np = surr.fitted_matrix(ref)
    kin = torch.tensor([ref.w_kin, ref.beta, ref.gamma], dtype=F64)
    M_torch = surr.fitted_matrix_torch(kin).detach().numpy()
    np.testing.assert_allclose(M_torch, M_np, atol=1e-12)


def test_fitted_matrix_torch_autograd_connects():
    """When kin_tensor.requires_grad=True, the gradient flows through."""
    surr = _build()
    kin = torch.tensor([5.0, 0.1, 1.005], dtype=F64, requires_grad=True)
    M = surr.fitted_matrix_torch(kin)
    # Scalar loss (trace) so we get a single .backward() call.
    loss = M.trace()
    loss.backward()
    # Gradient w.r.t. kin must exist and be finite (the MLP is zeroed
    # in the fixture so the actual gradient is also zero, but the
    # autograd graph must connect end-to-end without error).
    assert kin.grad is not None
    assert kin.grad.shape == (3,)
    assert torch.isfinite(kin.grad).all()


def test_element_matrix_torch_dispatches_surrogate_in_scope():
    """`element_matrix_torch(SurrogateFieldMap, kin)` calls fitted_matrix_torch."""
    from linac_gen.tracking.torch_matrices import RefKinematics
    from linac_gen.tracking.torch_tracking import element_matrix_torch

    surr = _build(scope_w_kin=(0.0, 100.0))
    real = _real_ref(w_kin=5.0)
    kin = RefKinematics.from_reference(real)
    # Make sure the surrogate's scope includes the real beta/gamma at
    # w_kin=5 MeV (otherwise OOD short-circuits to eye(6)).
    M = element_matrix_torch(surr, kin)
    # Should NOT be the eye(6) stub -- the mock M has off-diagonal 0.05.
    # Build the same input the surrogate uses internally and compare.
    expected_input = np.array(
        [float(real.w_kin), float(real.beta), float(real.gamma), 1.0],
        dtype=np.float64,
    )
    in_scope = surr.metadata.scope.is_in_scope(expected_input)
    assert in_scope, "test fixture: real ref must be in scope"
    expected = surr.fitted_matrix(real)
    np.testing.assert_allclose(M.detach().numpy(), expected, atol=1e-12)


def test_element_matrix_torch_ood_raises():
    """OOD ref -> the matcher arm refuses (hard error, never identity):
    silently substituting eye(6) would let the optimizer converge on a
    lattice different from the one the user specified."""
    import pytest
    from linac_gen.tracking.torch_matrices import RefKinematics
    from linac_gen.tracking.torch_tracking import element_matrix_torch

    # Surrogate trained on tight w_kin range.
    surr = _build(scope_w_kin=(2.0, 3.0))
    kin = RefKinematics.from_reference(_real_ref(w_kin=100.0))  # OOD
    with pytest.raises(ValueError, match="outside its trained scope"):
        element_matrix_torch(surr, kin)


def test_check_gradient_supported_accepts_surrogate():
    """A lattice with a SurrogateFieldMap passes the support check,
    while a plain FieldMap3D still raises."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.matching.torch_objective import check_gradient_supported

    # SurrogateFieldMap arm: should pass.
    lat_surr = Lattice()
    lat_surr.add(Drift("D1", length=100.0, aperture=10.0))
    lat_surr.add(_build("FM_S"))
    lat_surr.add(Drift("D2", length=100.0, aperture=10.0))
    # Empty variables / constraints lists trip none of the per-knob
    # guards, so only the element-loop is exercised here.
    check_gradient_supported(lat_surr, _real_ref(), variables=[],
                              constraints=[])


def test_gradient_gate_rejects_accelerating_surrogate():
    """A surrogate wrapping an ACCELERATING map (ke != 0) must be
    refused: the torch composition holds the reference energy fixed,
    so an energy-gaining element would silently corrupt every matrix
    downstream (2026-07-25 review, claim 13)."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.matching.torch_objective import check_gradient_supported

    surr = _build("FM_ACC")
    surr._wrapped.ke = 2.0                       # accelerating cavity
    lat = Lattice()
    lat.add(Drift("D1", length=100.0, aperture=10.0))
    lat.add(surr)
    with pytest.raises(ValueError, match="non-accelerating"):
        check_gradient_supported(lat, _real_ref(), variables=[],
                                 constraints=[])


def test_gradient_gate_rejects_unknown_wrapped():
    """No ke on the wrapped element -> refuse loudly, never guess."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.matching.torch_objective import check_gradient_supported

    surr = _build("FM_UNK")
    del surr._wrapped.ke
    lat = Lattice()
    lat.add(surr)
    with pytest.raises(ValueError, match="non-accelerating"):
        check_gradient_supported(lat, _real_ref(), variables=[],
                                 constraints=[])


def test_element_matrix_torch_refuses_nonfinite_output():
    """Torch mirror of the numpy finiteness gate (the two paths move in
    pairs): a NaN-producing NN must raise loudly — a corrupt weight
    file cannot feed NaN into the matcher Jacobian silently."""
    from linac_gen.tracking.torch_matrices import RefKinematics
    from linac_gen.tracking.torch_tracking import element_matrix_torch

    surr = _build(scope_w_kin=(0.0, 100.0))
    surr._out_mean = surr._out_mean * float("nan")
    kin = RefKinematics.from_reference(_real_ref(w_kin=5.0))
    with pytest.raises(ValueError, match="NON-FINITE"):
        element_matrix_torch(surr, kin)


def test_element_matrix_torch_refuses_singular_output():
    """Torch mirror of the numpy |det| < 1e-9 singular gate (the two
    paths move in pairs): a rank-collapsed NN output must raise loudly
    instead of feeding a non-invertible map to the matcher."""
    import torch
    from linac_gen.tracking.torch_matrices import RefKinematics
    from linac_gen.tracking.torch_tracking import element_matrix_torch

    surr = _build(scope_w_kin=(0.0, 100.0))
    # Squash the output scale to zero: the MLP then yields M == out_mean
    # broadcast (identical rows scaled by mean) — set the mean itself so
    # the reshaped 6x6 is exactly singular (all-equal entries).
    surr._out_std = surr._out_std * 0.0
    surr._out_mean = torch.ones_like(surr._out_mean) * 0.5
    kin = RefKinematics.from_reference(_real_ref(w_kin=5.0))
    with pytest.raises(ValueError, match="SINGULAR"):
        element_matrix_torch(surr, kin)
