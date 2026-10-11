"""Named space-charge methods: registry integrity and the construction rules
of SpaceChargeConfig (linac_gen.pic.methods)."""
from __future__ import annotations

import dataclasses

import pytest

from linac_gen.core.config import SpaceChargeConfig
from linac_gen.pic import methods


# ---------------------------------------------------------------------------
# registry integrity
# ---------------------------------------------------------------------------
def test_names_unique_and_custom_last():
    names = methods.names()
    assert len(names) == len(set(names))
    assert names[-1] == "custom"
    assert "default" not in names


@pytest.mark.parametrize("name", [n for n in methods.names() if n != "custom"])
def test_every_method_resolves_validates_and_infers_back(name):
    cfg = SpaceChargeConfig(method=name)
    assert cfg.method == name
    spec = methods.get(name)
    for key, val in spec.pinned.items():
        assert getattr(cfg, key) == val
    for key, val in spec.params.items():
        assert getattr(cfg, key) == val
    assert methods.infer(methods.base_fields(name)) == name
    assert methods.effective_method(cfg) == name


def test_every_spec_has_plain_label_and_summary():
    for name in methods.names()[:-1]:
        spec = methods.get(name)
        assert spec.label and spec.summary.endswith(".")
        assert set(spec.pinned) | set(spec.params) <= set(methods.METHOD_FIELDS)


def test_default_method_exists_or_is_classic():
    assert methods.DEFAULT_METHOD is None or methods.DEFAULT_METHOD in methods.names()


def test_unknown_method_lists_valid_names():
    with pytest.raises(ValueError, match="valid names"):
        SpaceChargeConfig(method="cic_fixd")


# ---------------------------------------------------------------------------
# construction rules
# ---------------------------------------------------------------------------
def test_bare_config_is_the_default():
    cfg = SpaceChargeConfig()
    expect = methods.base_fields(methods.DEFAULT_METHOD)
    for key, val in expect.items():
        assert getattr(cfg, key) == val


def test_explicit_fields_keep_the_classic_base():
    """Existing scripts set some fields and rely on the others' classic
    values; that must not change when the default method changes."""
    cfg = SpaceChargeConfig(nx=64, grid_extent=6.0)
    assert (cfg.nx, cfg.ny, cfg.nz) == (64, 96, 96)
    assert cfg.grid_extent == 6.0 and cfg.grid_mode == "fixed"
    assert cfg.kernel == "cic" and cfg.method == "cic_fixed"


def test_method_params_can_be_overridden_and_label_kept():
    cfg = SpaceChargeConfig(method="tsc_adaptive", nx=64, ny=64, nz=64,
                            green_kind="point")
    assert cfg.method == "tsc_adaptive"
    assert (cfg.nx, cfg.grid_extent, cfg.green_kind) == (64, 6.0, "point")


def test_contradicting_a_pinned_setting_relabels():
    cfg = SpaceChargeConfig(method="tsc_adaptive", kernel="cic")
    assert cfg.kernel == "cic" and cfg.method == "cic_adaptive"


def test_custom_starts_from_the_classic_base_and_is_labelled_by_its_settings():
    cfg = SpaceChargeConfig(method="custom", kernel="tsc")
    assert cfg.kernel == "tsc" and cfg.grid_mode == "fixed" and cfg.nx == 96
    assert cfg.method == "tsc_fixed"
    # torch on a fixed grid matches no named method
    assert SpaceChargeConfig(method="custom", sc_backend="torch").method \
        == "custom"


def test_labels_describe_the_settings_after_replace():
    odd = SpaceChargeConfig(sc_backend="torch")          # matches nothing
    assert odd.method == "custom"
    assert dataclasses.replace(odd, grid_mode="adaptive").method \
        == "differentiable_pic"


def test_default_alias():
    a = SpaceChargeConfig(method="default")
    b = SpaceChargeConfig()
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_replace_reproduces_and_relabels():
    cfg = SpaceChargeConfig(method="cic_fixed", nx=40, ny=40, nz=40)
    same = dataclasses.replace(cfg)
    assert same == cfg
    companion = dataclasses.replace(cfg, nz=120, train_images=False)
    assert companion.method == "cic_fixed" and companion.nz == 120
    adaptive = dataclasses.replace(cfg, grid_mode="adaptive")
    assert adaptive.method == "cic_adaptive" and adaptive.nx == 40


def test_fields_set_after_construction_are_seen_by_effective_method():
    cfg = SpaceChargeConfig(method="cic_fixed")
    cfg.kernel = "tsc"
    assert methods.effective_method(cfg) == "tsc_fixed"


def test_torch_and_halo_engines_are_named():
    assert SpaceChargeConfig(sc_backend="torch", grid_mode="adaptive").method \
        == "differentiable_pic"
    assert SpaceChargeConfig(sc_backend="halo").method == "halo_pic"
    assert SpaceChargeConfig(sc_backend="torch").method == "custom"


# ---------------------------------------------------------------------------
# command-line value conversion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,raw,want", [
    ("nx", "64", 64), ("grid_extent", "6.5", 6.5), ("kernel", "tsc", "tsc"),
    ("csr_enabled", "on", True), ("csr_enabled", "false", False),
    ("train_images", "False", False), ("train_images", "true", True),
    ("train_images", "none", None), ("halo", '{"alpha": 0.5}', {"alpha": 0.5}),
])
def test_coerce_field(name, raw, want):
    assert methods.coerce_field(name, raw) == want


def test_coerce_field_rejects_bad_bool_and_unknown_field():
    with pytest.raises(ValueError):
        methods.coerce_field("csr_enabled", "maybe")
    with pytest.raises(ValueError):
        methods.coerce_field("no_such_field", "1")


@pytest.mark.parametrize("name,raw", [
    ("kernel", "auto"), ("grid_mode", "none"), ("green_kind", "null"),
])
def test_auto_or_none_is_not_a_way_to_unset_a_method_field(name, raw):
    """Only train_images takes None as a value ("auto"); for the method
    fields the text stays text and the config refuses it, as in 1.14.0."""
    assert methods.coerce_field(name, raw) == raw
    with pytest.raises(ValueError, match=name):
        methods.resolve_sc({}, {"sc": {name: raw}})


@pytest.mark.parametrize("raw", ["none", "3", "[1, 2]", "{bad"])
def test_halo_must_be_a_json_object(raw):
    with pytest.raises(ValueError, match="halo"):
        methods.coerce_field("halo", raw)


# ---------------------------------------------------------------------------
# after the default switch (DEFAULT_METHOD set): the documented rules
# ---------------------------------------------------------------------------
@pytest.fixture()
def default_tsc_adaptive(monkeypatch):
    monkeypatch.setattr(methods, "DEFAULT_METHOD", "tsc_adaptive")


def test_with_a_default_method_a_bare_config_gets_it(default_tsc_adaptive):
    cfg = SpaceChargeConfig()
    assert cfg.method == "tsc_adaptive" and cfg.grid_mode == "adaptive"
    assert SpaceChargeConfig(method="default").method == "tsc_adaptive"


def test_with_a_default_method_explicit_fields_keep_the_classic_base(
        default_tsc_adaptive):
    cfg = SpaceChargeConfig(nx=64, grid_extent=6.0)
    assert (cfg.ny, cfg.grid_mode, cfg.kernel) == (96, "fixed", "cic")
    assert cfg.method == "cic_fixed"


def test_with_a_default_method_projects(default_tsc_adaptive):
    # no space-charge keys: the default method, project device kept
    cfg = methods.resolve_sc({"env_solver": "matrix"}, {})
    assert cfg.method == "tsc_adaptive" and cfg.nx == 48
    # legacy keys: exactly as before
    legacy = methods.resolve_sc({"grid_nx": 64, "kernel": "tsc"}, {})
    assert (legacy.nx, legacy.grid_mode, legacy.kernel) == (64, "fixed", "tsc")
    # "default" in a project: the default method with the stored params
    cfg = methods.resolve_sc({"sc_method": "default", "grid_nx": 40}, {})
    assert cfg.method == "tsc_adaptive" and cfg.nx == 40
    # --sc-method default on a legacy project: the default, with a note
    notes = []
    cfg = methods.resolve_sc({"grid_nx": 64}, {"sc_method": "default"},
                             notes=notes)
    assert cfg.method == "tsc_adaptive" and cfg.nx == 48 and notes


@pytest.mark.parametrize("name,raw,want", [
    ("csr_bins", 100.0, 100), ("csr_enabled", 1, True), ("csr_enabled", 0, False),
    ("grid_extent", 6, 6.0), ("nx", 64.0, 64), ("train_images", False, False),
    ("kernel", None, None),
])
def test_coerce_field_converts_typed_job_file_values_as_before(name, raw, want):
    got = methods.coerce_field(name, raw)
    assert got == want and type(got) is type(want)


def test_coerce_field_refuses_a_dict_for_a_single_value():
    with pytest.raises(ValueError, match="kernel"):
        methods.coerce_field("kernel", {"a": 1})


def test_a_null_override_is_no_override():
    """A JSON null in a job file's "sc" dict leaves the project's value (it
    used to be an error; it must not silently mean the classic value)."""
    cfg = methods.resolve_sc({"grid_nx": 64, "kernel": "tsc"},
                             {"sc": {"kernel": None, "nx": None}})
    assert (cfg.kernel, cfg.nx) == ("tsc", 64)
    cfg = methods.resolve_sc({}, {"sc": {"train_images": None}})
    assert cfg.train_images is None


def test_contradicting_a_pinned_setting_gives_the_matching_method():
    """method="tsc_adaptive" with grid_mode="fixed" is method="tsc_fixed":
    the fields not given take tsc_fixed's defaults."""
    got = SpaceChargeConfig(method="tsc_adaptive", grid_mode="fixed")
    assert dataclasses.asdict(got) == dataclasses.asdict(
        SpaceChargeConfig(method="tsc_fixed"))
    # explicit params stay
    got = SpaceChargeConfig(method="tsc_adaptive", grid_mode="fixed", nx=64)
    assert (got.method, got.nx, got.grid_extent) == ("tsc_fixed", 64, 7.0)


def test_card_policy():
    assert methods.card_policy(SpaceChargeConfig()) == "apply"
    assert methods.card_policy(SpaceChargeConfig(method="tsc_adaptive")) \
        == "ignore"
    cfg = SpaceChargeConfig(method="cic_adaptive")
    assert methods.card_policy(dataclasses.replace(cfg, grid_mode="fixed")) \
        == "apply"
    assert methods.card_policy(SpaceChargeConfig(grid_card="ignore")) \
        == "ignore"
