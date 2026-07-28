"""M1 — training pipeline: data gen + train + save/load round-trip."""
import json
import warnings

import numpy as np
import pytest
import torch

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates.base import MlpHead, SurrogateMetadata
from linac_gen.surrogates.training import (
    discover_cached_surrogates,
    find_cached_surrogate,
    generate_training_data,
    load_surrogate,
    save_surrogate,
    train_surrogate,
    train_surrogate_for_element,
)


class _MockFieldMap(FieldMapElement):
    """Mock element with a known analytic fitted_matrix(ref)."""

    def __init__(self, scale: float = 1.0):
        super().__init__(name="MOCK", length=100.0, aperture=10.0, n_steps=10)
        self.scale = scale

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        M = np.eye(6)
        # Two parameter-dependent terms — enough for the MLP to learn.
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


def test_generate_training_data_shapes_and_scope():
    elem = _MockFieldMap()
    X, Y, info = generate_training_data(
        element=elem,
        ref_template=_MockRef(),
        n_samples=64,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        seed=0,
    )
    assert X.shape == (64, 4)        # [w_kin, beta, gamma, scale]
    assert Y.shape == (64, 36)
    assert info["param_names"] == ["scale"]
    assert info["input_names"][0] == "w_kin"
    assert len(info["input_lo"]) == 4
    # Sample values inside the intended ranges
    assert 2.0 <= X[:, 0].min() and X[:, 0].max() <= 10.0
    assert 0.8 <= X[:, 3].min() and X[:, 3].max() <= 1.2
    # Scope bounds are the USER-INTENDED ranges (not the observed
    # LHS min/max — otherwise edge queries would be rejected as OOD).
    assert info["input_lo"][0] == 2.0   # w_kin lo = intended
    assert info["input_hi"][0] == 10.0  # w_kin hi = intended
    assert info["input_lo"][3] == 0.8   # scale lo = intended
    assert info["input_hi"][3] == 1.2   # scale hi = intended


def test_train_surrogate_converges_on_simple_data():
    """The MLP should learn the mock's two-term linear map cleanly."""
    elem = _MockFieldMap()
    X, Y, _ = generate_training_data(
        element=elem,
        ref_template=_MockRef(),
        n_samples=512,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        seed=0,
    )
    mlp, norm, val_mape = train_surrogate(
        X, Y,
        hidden_dims=(32, 32),
        epochs=80,
        lr=3e-3,
        batch_size=64,
        seed=0,
    )
    assert isinstance(mlp, MlpHead)
    # The target is nearly-identity + two small terms — a 32x32 MLP
    # over 512 samples should reach a sub-1% val MAPE comfortably.
    assert val_mape < 0.05, f"val_mape={val_mape!r}"


def test_save_and_load_roundtrip(tmp_path):
    """save_surrogate + load_surrogate reproduces the MLP and metadata."""
    elem = _MockFieldMap()
    mlp, meta = train_surrogate_for_element(
        element=elem,
        ref_template=_MockRef(),
        n_samples=128,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(16, 16),
        epochs=20,
        lr=3e-3,
        batch_size=32,
        seed=0,
        out_dir=tmp_path / "surr",
        lattice_hash="hash-x",
        element_key="MOCK_1",
    )
    assert (tmp_path / "surr" / "weights.pt").exists()
    assert (tmp_path / "surr" / "metadata.json").exists()
    # metadata.json shape
    with open(tmp_path / "surr" / "metadata.json") as f:
        d = json.load(f)
    assert d["element_key"] == "MOCK_1"
    assert d["lattice_hash"] == "hash-x"
    assert d["architecture"]["input_dim"] == 4
    assert d["architecture"]["output_dim"] == 36
    # load + check the loaded MLP gives the same output as the in-memory one
    mlp2, meta2 = load_surrogate(tmp_path / "surr")
    x = torch.zeros(1, 4, dtype=torch.float64)
    with torch.no_grad():
        y1 = mlp(x)
        y2 = mlp2(x)
    assert torch.allclose(y1, y2, rtol=1e-12, atol=1e-12)
    assert meta2.val_mape == meta.val_mape


# ---- cache helpers --------------------------------------------------------
def _stage_cached_surrogate(tmp_path, element_key: str,
                             lattice_hash: str = "hash-c"):
    """Train + save a tiny surrogate; return its directory."""
    elem = _MockFieldMap()
    out = tmp_path / "weights" / lattice_hash[:16] / element_key
    train_surrogate_for_element(
        element=elem,
        ref_template=_MockRef(),
        n_samples=64,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(8, 8),
        epochs=5,
        lr=3e-3,
        batch_size=32,
        seed=0,
        out_dir=out,
        lattice_hash=lattice_hash,
        element_key=element_key,
    )
    return out


def test_find_cached_surrogate_missing_dir(tmp_path):
    """Missing directory -> None (does not raise)."""
    assert find_cached_surrogate(tmp_path / "nope") is None


def test_find_cached_surrogate_partial_dir(tmp_path):
    """metadata.json without weights.pt -> None."""
    d = tmp_path / "partial"
    d.mkdir()
    (d / "metadata.json").write_text("{}")
    assert find_cached_surrogate(d) is None


def test_find_cached_surrogate_full(tmp_path):
    """A real saved surrogate is rediscovered + loads cleanly."""
    out = _stage_cached_surrogate(tmp_path, "ELEM_A")
    loaded = find_cached_surrogate(out)
    assert loaded is not None
    mlp, meta = loaded
    assert isinstance(mlp, MlpHead)
    assert meta.element_key == "ELEM_A"


def test_discover_cached_surrogates_filters_by_element_names(tmp_path):
    """Only subdirs whose names are in the allow-list are loaded."""
    lh = "hash-d"
    _stage_cached_surrogate(tmp_path, "ELEM_A", lattice_hash=lh)
    _stage_cached_surrogate(tmp_path, "ELEM_B", lattice_hash=lh)
    _stage_cached_surrogate(tmp_path, "ELEM_C", lattice_hash=lh)
    root = tmp_path / "weights" / lh[:16]
    # No filter -> all three
    found = discover_cached_surrogates(root)
    assert set(found) == {"ELEM_A", "ELEM_B", "ELEM_C"}
    # Restrict to A + C
    found = discover_cached_surrogates(root, element_names=["ELEM_A", "ELEM_C"])
    assert set(found) == {"ELEM_A", "ELEM_C"}
    # Restrict to nothing -> empty
    found = discover_cached_surrogates(root, element_names=[])
    assert found == {}


def test_discover_cached_surrogates_missing_root(tmp_path):
    """A missing weights root is treated as empty, not an error."""
    assert discover_cached_surrogates(tmp_path / "missing") == {}


def test_discover_cached_surrogates_skips_broken(tmp_path):
    """A subdir with bad metadata is silently skipped."""
    lh = "hash-e"
    _stage_cached_surrogate(tmp_path, "GOOD", lattice_hash=lh)
    root = tmp_path / "weights" / lh[:16]
    broken = root / "BROKEN"
    broken.mkdir()
    (broken / "weights.pt").write_bytes(b"not a torch file")
    (broken / "metadata.json").write_text("not json")
    found = discover_cached_surrogates(root)
    # GOOD survives; BROKEN is silently skipped.
    assert set(found) == {"GOOD"}


# ---- Multiprocessing data generation -------------------------------------
def test_generate_training_data_n_workers_serial_path(tmp_path):
    """n_workers=None and n_workers=1 both run the serial path and
    produce identical output (smoke baseline)."""
    elem = _MockFieldMap()
    X1, Y1, info1 = generate_training_data(
        element=elem, ref_template=_MockRef(),
        n_samples=32,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        seed=7,
        n_workers=None,
    )
    X2, Y2, info2 = generate_training_data(
        element=elem, ref_template=_MockRef(),
        n_samples=32,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        seed=7,
        n_workers=1,
    )
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(Y1, Y2)
    assert info1 == info2


def test_generate_training_data_mp_matches_serial_bit_identical(tmp_path):
    """The multiprocessing path is bit-identical to the serial path.

    This is the load-bearing invariant: each worker runs the EXACT
    scalar fitted_matrix(ref) call on its own deepcopy of the element;
    only the dispatch is parallel.  For a fixed seed the (X, Y)
    training set must match the serial path byte-for-byte, so the
    trained MLP and val MAPE are reproducible regardless of worker
    count.
    """
    elem = _MockFieldMap()
    X_ser, Y_ser, info_ser = generate_training_data(
        element=elem, ref_template=_MockRef(),
        n_samples=64,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        seed=11,
        n_workers=None,
    )
    X_mp, Y_mp, info_mp = generate_training_data(
        element=elem, ref_template=_MockRef(),
        n_samples=64,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        seed=11,
        n_workers=2,
    )
    # Bit-identical -- not allclose, EQUAL.
    np.testing.assert_array_equal(X_ser, X_mp)
    np.testing.assert_array_equal(Y_ser, Y_mp)
    assert info_ser == info_mp


def test_generate_training_data_progress_callback_fires(tmp_path):
    """progress_callback receives data_gen events with the expected
    keys; no crash when None."""
    elem = _MockFieldMap()
    events = []
    def cb(info):
        events.append(info)
    X, Y, _ = generate_training_data(
        element=elem, ref_template=_MockRef(),
        n_samples=200, ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        seed=42, progress_callback=cb,
    )
    assert len(events) >= 1, "no progress events emitted"
    keys = set(events[-1].keys())
    assert {"stage", "done", "total", "elapsed_s"} <= keys
    assert events[-1]["stage"] == "data_gen"
    # The final event always reports done == total.
    assert events[-1]["done"] == events[-1]["total"] == 200


def test_train_surrogate_progress_callback_fires(tmp_path):
    """progress_callback receives one epoch event per epoch with
    train_loss / val_mape / per_entry_val_mape filled in."""
    import numpy as np
    elem = _MockFieldMap()
    X, Y, _ = generate_training_data(
        element=elem, ref_template=_MockRef(),
        n_samples=64, ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)}, seed=0,
    )
    events = []
    train_surrogate(
        X, Y, hidden_dims=(8, 8), epochs=5, lr=3e-3,
        batch_size=16, seed=0,
        progress_callback=lambda info: events.append(info),
    )
    assert len(events) == 5
    e = events[-1]
    assert e["stage"] == "epoch"
    assert {"epoch", "total", "train_loss", "val_mape", "best_val_mape",
            "per_entry_val_mape", "elapsed_s"} <= set(e.keys())
    assert e["epoch"] == 5
    assert e["total"] == 5
    arr = e["per_entry_val_mape"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6, 6)
    assert np.isfinite(arr).all()
    # best_val_mape monotonically non-increasing.
    bests = [ev["best_val_mape"] for ev in events]
    assert all(b2 <= b1 for b1, b2 in zip(bests, bests[1:]))


def test_train_surrogate_for_element_threads_workers(tmp_path):
    """train_surrogate_for_element forwards n_workers to data gen
    and the resulting surrogate is still trainable + loadable."""
    out_dir = tmp_path / "mp_surr"
    mlp, meta = train_surrogate_for_element(
        element=_MockFieldMap(),
        ref_template=_MockRef(),
        n_samples=32,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(8, 8),
        epochs=5,
        lr=3e-3,
        batch_size=16,
        seed=0,
        out_dir=out_dir,
        lattice_hash="hash-mp",
        element_key="MOCK_MP",
        n_workers=2,
    )
    assert (out_dir / "weights.pt").exists()
    assert (out_dir / "metadata.json").exists()
    # Reload to confirm the trained weights are usable.
    mlp_l, meta_l = load_surrogate(out_dir)
    assert meta_l.element_key == "MOCK_MP"


# ---------------------------------------------------------------------------
# Cooperative cancellation (should_stop)
# ---------------------------------------------------------------------------

def test_generate_training_data_cancel_at_sample():
    from linac_gen.core.cancelled import OperationCancelled

    polls = {"n": 0}
    def stop_after_10(_c=polls):
        _c["n"] += 1
        return _c["n"] > 10

    with pytest.raises(OperationCancelled):
        generate_training_data(
            element=_MockFieldMap(), ref_template=_MockRef(),
            n_samples=64, ref_w_kin_range=(2.0, 10.0),
            param_ranges={"scale": (0.8, 1.2)}, seed=0,
            should_stop=stop_after_10,
        )
    assert polls["n"] == 11        # polled once per sample, stopped at #11


def test_train_surrogate_cancel_at_epoch():
    from linac_gen.core.cancelled import OperationCancelled

    X, Y, _ = generate_training_data(
        element=_MockFieldMap(), ref_template=_MockRef(),
        n_samples=64, ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)}, seed=0,
    )
    epochs_seen = []
    with pytest.raises(OperationCancelled):
        train_surrogate(
            X, Y, hidden_dims=(8,), epochs=50, batch_size=32, seed=0,
            progress_callback=lambda info: epochs_seen.append(info["epoch"]),
            should_stop=lambda: len(epochs_seen) >= 3,
        )
    assert epochs_seen == [1, 2, 3]   # exactly three epochs ran


def test_cancelled_training_saves_nothing(tmp_path):
    from linac_gen.core.cancelled import OperationCancelled
    from linac_gen.surrogates.training import train_surrogate_for_element

    out_dir = tmp_path / "weights"
    with pytest.raises(OperationCancelled):
        train_surrogate_for_element(
            element=_MockFieldMap(), ref_template=_MockRef(),
            n_samples=32, ref_w_kin_range=(2.0, 10.0),
            param_ranges={"scale": (0.8, 1.2)},
            out_dir=out_dir, epochs=10,
            should_stop=lambda: True,
        )
    assert not out_dir.exists()       # no partial weights ever land


def test_mp_data_gen_cancel_terminates_pool():
    """Cancel during the multiprocessing branch must raise inside the
    Pool context (whose __exit__ terminates the spawn workers)."""
    from linac_gen.core.cancelled import OperationCancelled

    with pytest.raises(OperationCancelled):
        generate_training_data(
            element=_MockFieldMap(), ref_template=_MockRef(),
            n_samples=512, ref_w_kin_range=(2.0, 10.0),
            param_ranges={"scale": (0.8, 1.2)}, seed=0,
            n_workers=2,
            should_stop=lambda: True,
        )


def test_load_surrogate_computes_weights_sha256(tmp_path):
    """load_surrogate() hashes the ACTUAL weights.pt bytes onto the
    metadata — route-independent provenance (CLI/GUI/Python API alike),
    recomputed at read time and never trusted from metadata.json."""
    import hashlib

    from linac_gen.surrogates.training import load_surrogate

    elem = _MockFieldMap()
    _mlp, _meta = train_surrogate_for_element(
        element=elem, ref_template=_MockRef(), n_samples=64,
        ref_w_kin_range=(2.0, 10.0), param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(8,), epochs=5, lr=3e-3, batch_size=32, seed=0,
        out_dir=tmp_path / "surr", lattice_hash="hash-w",
        element_key="MOCK_W",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _mlp2, meta2 = load_surrogate(tmp_path / "surr")
    expect = hashlib.sha256(
        (tmp_path / "surr" / "weights.pt").read_bytes()).hexdigest()
    assert meta2.weights_sha256 == expect
    # never TRUSTED from metadata.json: plant a bogus stored hash and
    # re-load — the recomputed value must win
    mj = tmp_path / "surr" / "metadata.json"
    d = json.loads(mj.read_text())
    d["weights_sha256"] = "deadbeef" * 8
    mj.write_text(json.dumps(d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _mlp3, meta3 = load_surrogate(tmp_path / "surr")
    assert meta3.weights_sha256 == expect


def test_load_warns_on_stale_species_mass(tmp_path):
    """Kinematic admissibility: weights trained at a species mass that no
    longer matches the current table (e.g. pre-H⁻-mass-fix caches at
    938.272 MeV) must warn loudly on load — the lattice-hash cache key
    cannot see the species table."""
    import dataclasses
    import pytest
    from linac_gen.surrogates.training import save_surrogate, load_surrogate
    mlp, meta = train_surrogate_for_element(
        element=_MockFieldMap(),
        ref_template=_MockRef(),
        n_samples=64,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(8, 8), epochs=5, lr=3e-3, batch_size=32, seed=0,
        out_dir=tmp_path / "t", lattice_hash="h", element_key="MOCK_S",
    )
    stale = dataclasses.replace(meta, species_name="H-",
                                species_mass_mev=938.27208816)
    out = tmp_path / "stale"
    save_surrogate(mlp, stale, out)
    with pytest.warns(UserWarning, match="STALE kinematics"):
        load_surrogate(out)


def test_load_warns_on_unknown_species(tmp_path):
    """A species OUTSIDE the kinematics table cannot be mass-checked at
    all — the load must say so (it used to pass silently for any
    unrecognised name with a positive mass)."""
    import dataclasses
    import pytest
    from linac_gen.surrogates.training import save_surrogate, load_surrogate
    mlp, meta = train_surrogate_for_element(
        element=_MockFieldMap(),
        ref_template=_MockRef(),
        n_samples=64,
        ref_w_kin_range=(2.0, 10.0),
        param_ranges={"scale": (0.8, 1.2)},
        hidden_dims=(8, 8), epochs=5, lr=3e-3, batch_size=32, seed=0,
        out_dir=tmp_path / "t", lattice_hash="h", element_key="MOCK_U",
    )
    odd = dataclasses.replace(meta, species_name="muon",
                              species_mass_mev=1.0)
    out = tmp_path / "odd"
    save_surrogate(mlp, odd, out)
    with pytest.warns(UserWarning, match="UNVERIFIABLE"):
        load_surrogate(out)
