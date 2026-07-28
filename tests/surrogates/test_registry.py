"""M1 — registry: hash, register / get / unregister / clear."""
import numpy as np
import pytest

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import (
    MlpHead, Scope, SurrogateFieldMap, SurrogateMetadata,
)


class _Mock(FieldMapElement):
    def __init__(self, name="MOCK"):
        super().__init__(name=name, length=10.0, aperture=10.0, n_steps=1)

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        return np.eye(6)

    def fitted_matrix_slice(self, ref, ds_mm):
        return np.eye(6)


def _make_surrogate(element_key: str, lattice_hash: str = "lh") -> SurrogateFieldMap:
    mlp = MlpHead(input_dim=3, output_dim=36, hidden_dims=(4,))
    meta = SurrogateMetadata(
        element_key=element_key, element_class="_Mock",
        architecture={"input_dim": 3, "output_dim": 36,
                      "hidden_dims": [4], "activation": "silu",
                      "param_names": []},
        scope=Scope(input_names=["w_kin", "beta", "gamma"],
                    input_lo=np.array([0.0, 0.0, 1.0]),
                    input_hi=np.array([100.0, 1.0, 100.0])),
        input_norm={"mean": [0.0]*3, "std": [1.0]*3},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash=lattice_hash, created_iso="",
    )
    return SurrogateFieldMap(_Mock(name=element_key), mlp, meta)


def test_register_get_unregister_clear():
    registry.clear()
    surr = _make_surrogate("EL_1")
    registry.register(surr)
    assert registry.get("lh", "EL_1") is surr
    assert registry.get("lh", "absent") is None

    surr2 = _make_surrogate("EL_2", lattice_hash="other-lh")
    registry.register(surr2)
    assert set(registry.list_registered()) == {("lh", "EL_1"),
                                                ("other-lh", "EL_2")}

    registry.unregister("lh", "EL_1")
    assert registry.get("lh", "EL_1") is None
    assert registry.get("other-lh", "EL_2") is surr2

    registry.clear()
    assert registry.list_registered() == []


def test_cross_lattice_name_collision_warns():
    """Registering the same element NAME under a different lattice hash
    replaces it for every name-scoped tracking lookup — that must be
    loud (interim honesty until hash-scoped lookup lands).  Same-hash
    re-registration stays silent."""
    import warnings

    registry.clear()
    try:
        registry.register(_make_surrogate("CAV1", lattice_hash="lat-A"))
        with pytest.warns(UserWarning, match="DIFFERENT lattice"):
            registry.register(
                _make_surrogate("CAV1", lattice_hash="lat-B"))
        # last-write-wins is unchanged
        assert registry.get_by_element_name("CAV1").metadata.lattice_hash \
            == "lat-B"
        # same-hash re-register: silent
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            registry.register(
                _make_surrogate("CAV1", lattice_hash="lat-B"))
    finally:
        registry.clear()


def test_get_for_element_identity_and_fingerprint_paths():
    """The guarded lookup engages on identity (shipped register-then-
    track flow) and on a structurally-identical element (same deck
    reloaded), but never needs a lattice hash."""
    registry.clear()
    try:
        surr = _make_surrogate("FMAP_001")
        registry.register(surr)
        # identity: the very element the surrogate wraps
        assert registry.get_for_element(surr._wrapped) is surr
        # fingerprint: a fresh instance of the same element (reload)
        clone = _Mock(name="FMAP_001")
        assert registry.get_for_element(clone) is surr
    finally:
        registry.clear()


def test_get_for_element_cross_lattice_mismatch_skips_and_warns_once():
    """The silent-wrong-physics hazard: lattice B's FMAP_001 is a
    DIFFERENT element (auto-generated names collide across decks by
    construction).  The guard must skip the surrogate (native element
    tracks) and warn exactly once per name."""
    import warnings

    registry.clear()
    try:
        registry.register(_make_surrogate("FMAP_001"))
        other = _Mock(name="FMAP_001")
        other.length = 999.0                    # lattice B's geometry
        with pytest.warns(UserWarning, match="DIFFERENT element"):
            assert registry.get_for_element(other) is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # second call: silent
            assert registry.get_for_element(other) is None
        # unrelated names unaffected
        assert registry.get_for_element(_Mock(name="OTHER")) is None
    finally:
        registry.clear()


def test_get_for_element_kb_and_cache_key_attrs_fingerprinted():
    """Adversarial-review finding: the fingerprint must cover the
    repo's matrix-affecting attribute list (FieldMap._cache_keys) —
    headlined by kb, the drive parameter of every magnetic map — plus
    aperture (the surrogate element copies the wrapped aperture)."""
    import warnings

    registry.clear()
    try:
        for attr, other_val in (("kb", 5.0), ("scale", 2.0),
                                ("aperture", 30.0),
                                ("voltage_rel", 0.5)):
            registry.clear()
            registry.register(_make_surrogate("FMAP_001"))
            other = _Mock(name="FMAP_001")
            setattr(other, attr, other_val)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert registry.get_for_element(other) is None, attr
    finally:
        registry.clear()


def test_get_for_element_decision_pinned_within_generation():
    """The engage/skip decision is pinned per element per registry
    generation: mutating a drive parameter mid-run (ADJUST during a
    match) must NOT flip engagement (discontinuous objective); a
    registry change invalidates the pin immediately."""
    import warnings

    registry.clear()
    try:
        surr = _make_surrogate("FMAP_001")
        registry.register(surr)
        clone = _Mock(name="FMAP_001")          # fingerprint-equal
        assert registry.get_for_element(clone) is surr
        clone.ke = 1.234                        # ADJUST-style mutation
        assert registry.get_for_element(clone) is surr   # pinned
        # a registry change bumps the generation -> re-decide: the
        # mutated clone now mismatches and is skipped
        registry.register(_make_surrogate("OTHER"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert registry.get_for_element(clone) is None
    finally:
        registry.clear()


def test_get_for_element_wrapped_none_skips_with_accurate_warning():
    """A registration without a wrapped element cannot be verified —
    skip with a message that says so (not 'different element')."""
    from types import SimpleNamespace

    registry.clear()
    try:
        registry.register(SimpleNamespace(metadata=None, _wrapped=None),
                          lattice_hash="lh", element_key="FMAP_001")
        with pytest.warns(UserWarning, match="cannot be verified"):
            assert registry.get_for_element(_Mock(name="FMAP_001")) is None
    finally:
        registry.clear()


def test_get_for_element_mismatch_falls_through_at_tracking_seam():
    """End-to-end at the envelope seam: a registered surrogate wrapping
    lattice A's element must NOT drive lattice B's same-named element —
    the native element tracks instead."""
    import warnings

    from linac_gen.tracking.envelope import _fitted_matrix_slice_at

    class _Ref:
        w_kin = 3.0; beta = 0.08; gamma = 1.003; bg = 0.0802
        frequency = 352.21; wavelength = 851.2
        class species:
            mass = 938.272; charge = 1

    registry.clear()
    try:
        registry.register(_make_surrogate("FMAP_001"))
        lattice_b_elem = _Mock(name="FMAP_001")
        lattice_b_elem.length = 999.0           # different geometry
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            M = _fitted_matrix_slice_at(lattice_b_elem, _Ref(),
                                        ds_mm=1.0, z_from_mm=0.0)
        np.testing.assert_allclose(M, np.eye(6))   # _Mock's native slice
    finally:
        registry.clear()


def test_hash_lattice_file(tmp_path):
    """SHA256 hex digest is deterministic and content-keyed."""
    p1 = tmp_path / "lat1.dat"
    p2 = tmp_path / "lat2.dat"
    p1.write_text("hello world\n")
    p2.write_text("hello world\n")
    p3 = tmp_path / "lat3.dat"
    p3.write_text("hello world!\n")
    h1 = registry.hash_lattice_file(p1)
    h2 = registry.hash_lattice_file(p2)
    h3 = registry.hash_lattice_file(p3)
    assert h1 == h2          # same content -> same hash
    assert h1 != h3          # different content -> different hash
    assert len(h1) == 64     # SHA256 hex
