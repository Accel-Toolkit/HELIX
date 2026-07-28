"""M1 — base classes for surrogate elements: MLP head, Scope, metadata,
and the SurrogateFieldMap drop-in contract."""
import numpy as np
import torch

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates.base import (
    F64,
    MlpHead,
    OutOfScopeError,
    Scope,
    SurrogateFieldMap,
    SurrogateMetadata,
)


def test_mlp_head_forward_shape():
    mlp = MlpHead(input_dim=5, output_dim=36, hidden_dims=(16, 16))
    x = torch.zeros(7, 5, dtype=F64)
    y = mlp(x)
    assert y.shape == (7, 36)
    assert y.dtype == torch.float64


def test_scope_in_and_out():
    scope = Scope(
        input_names=["w_kin", "beta"],
        input_lo=np.array([2.0, 0.05]),
        input_hi=np.array([10.0, 0.20]),
    )
    assert scope.is_in_scope(np.array([3.0, 0.1]))
    assert scope.is_in_scope(np.array([2.0, 0.05]))    # boundary
    assert not scope.is_in_scope(np.array([1.9, 0.1]))  # below w_kin
    assert not scope.is_in_scope(np.array([3.0, 0.21]))  # above beta


def test_scope_dict_roundtrip():
    s = Scope(input_names=["a", "b"],
              input_lo=np.array([0.0, -1.0]),
              input_hi=np.array([1.0, 1.0]))
    s2 = Scope.from_dict(s.to_dict())
    assert s2.input_names == s.input_names
    assert np.allclose(s.input_lo, s2.input_lo)
    assert np.allclose(s.input_hi, s2.input_hi)


def test_metadata_json_roundtrip():
    meta = SurrogateMetadata(
        element_key="EL_1",
        element_class="MockFieldMap",
        architecture={"input_dim": 4, "output_dim": 36,
                      "hidden_dims": [16, 16], "activation": "silu",
                      "param_names": ["k1"]},
        scope=Scope(input_names=["w_kin", "beta", "gamma", "k1"],
                    input_lo=np.array([2.0, 0.05, 1.0, 0.8]),
                    input_hi=np.array([10.0, 0.2, 1.05, 1.2])),
        input_norm={"mean": [0.0]*4, "std": [1.0]*4},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=42,
        n_samples=1000,
        epochs=100,
        val_mape=0.005,
        helix_commit_sha="abc123",
        lattice_hash="def456",
        created_iso="2026-05-24T12:00:00Z",
    )
    d = meta.to_json()
    meta2 = SurrogateMetadata.from_json(d)
    assert meta2.element_key == meta.element_key
    assert meta2.architecture == meta.architecture
    assert np.allclose(meta2.scope.input_lo, meta.scope.input_lo)


# ---------------------------------------------------------------------------
class _MockFieldMap(FieldMapElement):
    """Mock FieldMapElement for hermetic unit tests."""

    def __init__(self, scale: float = 1.0):
        super().__init__(name="MOCK", length=100.0, aperture=10.0, n_steps=10)
        self.scale = scale

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        M = np.eye(6)
        M[0, 1] = 0.001 * ref.w_kin * self.scale
        M[2, 3] = 0.0008 * ref.w_kin * self.scale
        return M

    def fitted_matrix_slice(self, ref, ds_mm):
        return np.eye(6)


class _MockRef:
    def __init__(self, w_kin: float = 3.0):
        self.w_kin = float(w_kin)
        self.beta = 0.1
        self.gamma = 1.005

    def copy(self):
        return _MockRef(self.w_kin)


def test_surrogate_fieldmap_construction_and_track_fallback():
    """SurrogateFieldMap should construct cleanly via multi-inheritance
    and route track_rk4 to the wrapped element."""
    wrapped = _MockFieldMap()
    mlp = MlpHead(input_dim=4, output_dim=36, hidden_dims=(8,))
    meta = SurrogateMetadata(
        element_key="MOCK", element_class="_MockFieldMap",
        architecture={"input_dim": 4, "output_dim": 36,
                      "hidden_dims": [8], "activation": "silu",
                      "param_names": ["scale"]},
        scope=Scope(input_names=["w_kin", "beta", "gamma", "scale"],
                    input_lo=np.array([2.0, 0.05, 1.0, 0.8]),
                    input_hi=np.array([10.0, 0.2, 1.1, 1.2])),
        input_norm={"mean": [3.0, 0.1, 1.005, 1.0], "std": [1.0]*4},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=0, n_samples=10, epochs=1, val_mape=0.0,
        helix_commit_sha="", lattice_hash="", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    # Drop-in: it IS a FieldMapElement.
    assert isinstance(surr, FieldMapElement)
    # Drop-in: it IS a torch.nn.Module too.
    assert isinstance(surr, torch.nn.Module)
    # track_rk4 falls back (just calls the mock's no-op)
    surr.track_rk4(beam=None, ds=1.0)


def test_surrogate_fitted_matrix_returns_6x6():
    """fitted_matrix runs the MLP and returns a numpy 6x6."""
    wrapped = _MockFieldMap()
    mlp = MlpHead(input_dim=4, output_dim=36, hidden_dims=(8,))
    meta = SurrogateMetadata(
        element_key="MOCK", element_class="_MockFieldMap",
        architecture={"input_dim": 4, "output_dim": 36,
                      "hidden_dims": [8], "activation": "silu",
                      "param_names": ["scale"]},
        scope=Scope(input_names=["w_kin", "beta", "gamma", "scale"],
                    input_lo=np.array([2.0, 0.05, 1.0, 0.8]),
                    input_hi=np.array([10.0, 0.2, 1.1, 1.2])),
        input_norm={"mean": [3.0, 0.1, 1.005, 1.0], "std": [1.0]*4},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=0, n_samples=10, epochs=1, val_mape=0.0,
        helix_commit_sha="", lattice_hash="", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    M = surr.fitted_matrix(_MockRef(w_kin=3.0))
    assert isinstance(M, np.ndarray)
    assert M.shape == (6, 6)
    assert np.isfinite(M).all()


def test_surrogate_out_of_scope_raises():
    """Querying outside training scope raises OutOfScopeError."""
    wrapped = _MockFieldMap()
    mlp = MlpHead(input_dim=4, output_dim=36, hidden_dims=(8,))
    meta = SurrogateMetadata(
        element_key="MOCK", element_class="_MockFieldMap",
        architecture={"input_dim": 4, "output_dim": 36,
                      "hidden_dims": [8], "activation": "silu",
                      "param_names": ["scale"]},
        scope=Scope(input_names=["w_kin", "beta", "gamma", "scale"],
                    input_lo=np.array([2.0, 0.05, 1.0, 0.8]),
                    input_hi=np.array([10.0, 0.2, 1.1, 1.2])),
        input_norm={"mean": [3.0, 0.1, 1.005, 1.0], "std": [1.0]*4},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=0, n_samples=10, epochs=1, val_mape=0.0,
        helix_commit_sha="", lattice_hash="", created_iso="",
    )
    surr = SurrogateFieldMap(wrapped, mlp, meta)
    # w_kin below the scope.lo
    import pytest
    with pytest.raises(OutOfScopeError):
        surr.fitted_matrix(_MockRef(w_kin=1.0))
