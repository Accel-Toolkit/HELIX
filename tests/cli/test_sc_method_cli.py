"""``--sc-method`` on the command line, and the project ``sc_method`` key,
through the real resolver (linac_gen.pic.methods.resolve_sc)."""
from __future__ import annotations

import json

import pytest

from linac_gen.cli import common
from linac_gen.pic import methods


class _Beam:
    current = 5.0


def _cfg(conv, cli=None):
    return common.make_sc_config(_Beam(), conv, cli or {})


_LEGACY_TSC = {"grid_nx": 48, "grid_extent_sigma": 20.0, "kernel": "tsc",
               "grid_mode": "fixed", "green_kind": "igf", "backend": "auto",
               "dc_kernel": "uniform", "csr_enabled": False}


def test_project_without_method_is_labelled_but_unchanged():
    cfg = _cfg(_LEGACY_TSC)
    assert cfg.method == "tsc_fixed"
    assert (cfg.nx, cfg.grid_extent, cfg.kernel, cfg.grid_mode) == (48, 20.0, "tsc", "fixed")


def test_cli_naming_the_projects_own_method_keeps_its_settings():
    cfg = _cfg(_LEGACY_TSC, {"sc_method": "tsc_fixed"})
    assert (cfg.nx, cfg.grid_extent) == (48, 20.0)


def test_cli_naming_another_method_uses_its_defaults_and_notes_it():
    notes = []
    cfg = methods.resolve_sc(_LEGACY_TSC, {"sc_method": "tsc_adaptive"}, notes=notes)
    assert cfg.method == "tsc_adaptive"
    assert (cfg.nx, cfg.grid_extent, cfg.grid_mode) == (48, 6.0, "adaptive")
    assert cfg.use_gpu == "auto" and cfg.dc_kernel == "uniform"
    assert notes and "tsc_adaptive" in notes[0]


def test_cli_flags_override_a_method_param():
    cfg = _cfg({}, {"sc_method": "cic_adaptive", "nx": 64, "grid_extent": 5.0})
    assert (cfg.nx, cfg.ny, cfg.nz, cfg.grid_extent) == (64, 64, 64, 5.0)
    assert cfg.method == "cic_adaptive"


def test_generic_sc_method_override_works_like_the_flag():
    a = _cfg({}, {"sc": {"method": "tsc_fixed"}})
    b = _cfg({}, {"sc_method": "tsc_fixed"})
    assert a == b


def test_project_sc_method_key_round_trips():
    conv = {"sc_method": "cic_adaptive", "grid_nx": 40, "grid_extent_sigma": 5.5,
            "kernel": "cic", "grid_mode": "adaptive", "green_kind": "point"}
    cfg = _cfg(conv)
    assert cfg.method == "cic_adaptive"
    assert (cfg.nx, cfg.grid_extent, cfg.green_kind) == (40, 5.5, "point")


def test_cli_custom_keeps_the_projects_settings():
    """--sc-method custom on a project: its own settings, labelled custom,
    with individual --sc overrides on top."""
    notes = []
    cfg = methods.resolve_sc(_LEGACY_TSC, {"sc_method": "custom",
                                           "sc": {"green_kind": "point"}},
                             notes=notes)
    assert cfg.method == "tsc_fixed" and not notes
    assert (cfg.nx, cfg.grid_extent, cfg.kernel, cfg.grid_mode) == (48, 20.0, "tsc", "fixed")
    assert cfg.green_kind == "point"


def test_cli_custom_on_a_named_project_starts_from_its_method():
    """The project stores only some fields; custom keeps its method's
    pinned settings for the rest."""
    cfg = _cfg({"sc_method": "tsc_adaptive", "grid_nx": 64}, {"sc_method": "custom"})
    assert cfg.method == "tsc_adaptive"
    assert (cfg.kernel, cfg.grid_mode, cfg.nx, cfg.grid_extent) == ("tsc", "adaptive", 64, 6.0)


def test_project_with_sc_method_default():
    cfg = _cfg({"sc_method": "default", "grid_nx": 40})
    expect = methods.base_fields(methods.DEFAULT_METHOD)
    assert cfg.nx == 40
    assert (cfg.kernel, cfg.grid_mode, cfg.grid_extent) == (
        expect["kernel"], expect["grid_mode"], expect["grid_extent"])


def test_cli_default_replaces_project_settings_with_a_note():
    notes = []
    cfg = methods.resolve_sc(_LEGACY_TSC, {"sc_method": "default"}, notes=notes)
    expect = methods.base_fields(methods.DEFAULT_METHOD)
    for key in ("nx", "grid_extent", "kernel", "grid_mode"):
        assert getattr(cfg, key) == expect[key]
    assert notes and "'default'" in notes[0]


def test_project_values_stored_as_text_are_converted():
    cfg = _cfg({"sc_method": "cic_adaptive", "grid_nx": "40",
                "grid_extent_sigma": "5.5", "csr_enabled": "false",
                "train_images": "true"})
    assert (cfg.nx, cfg.grid_extent) == (40, 5.5)
    assert cfg.csr_enabled is False and cfg.train_images is True


def test_helix_keys_win_over_field_named_duplicates():
    """rfq_demo-style stray nx/ny/nz keys: the grid_nx key HELIX writes wins
    for all three sizes (the legacy rule ignores the stray keys too)."""
    cfg = _cfg({"sc_method": "tsc_adaptive", "grid_nx": 96,
                "nx": 32, "ny": 32, "nz": 32})
    assert (cfg.nx, cfg.ny, cfg.nz) == (96, 96, 96)


def test_naming_a_legacy_projects_own_adaptive_method_keeps_applying_cards():
    conv = {**_LEGACY_TSC, "grid_mode": "adaptive"}
    assert _cfg(conv).grid_card == "apply"
    cfg = _cfg(conv, {"sc_method": "tsc_adaptive"})
    assert cfg.method == "tsc_adaptive" and cfg.grid_card == "apply"
    assert (cfg.nx, cfg.grid_extent) == (48, 20.0)


def test_switching_to_an_adaptive_method_ignores_cards():
    cfg = _cfg(_LEGACY_TSC, {"sc_method": "tsc_adaptive"})
    assert cfg.grid_card == "auto"
    cfg = _cfg(_LEGACY_TSC, {"sc_method": "tsc_adaptive",
                             "sc": {"grid_card": "apply"}})
    assert cfg.method == "tsc_adaptive" and cfg.grid_card == "apply"


def test_unknown_method_name_is_an_error():
    with pytest.raises(ValueError, match="valid names"):
        _cfg({}, {"sc_method": "cic_fast"})


def test_train_images_false_is_false_not_a_truthy_string():
    """--sc train_images=False used to arrive as the string 'False'."""
    cfg = _cfg({}, {"sc": {"train_images": "False"}})
    assert cfg.train_images is False


@pytest.mark.parametrize("command", ["run", "scan", "backtrack"])
def test_sc_method_list_needs_no_input(command, capsys):
    """--sc-method list prints the methods and exits 0 at once, before the
    required input (and scan's --vary) are asked for."""
    from linac_gen.__main__ import main as cli_main_fn
    with pytest.raises(SystemExit) as exc:
        cli_main_fn([command, "--sc-method", "list"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in methods.names():
        assert name in out


def _deck(tmp_path):
    deck = tmp_path / "d.dat"
    deck.write_text("DRIFT 100 20 0\nEND\n")
    return deck


def _project_file(tmp_path, conv):
    deck = _deck(tmp_path)
    p = tmp_path / "p.lgproj"
    p.write_text(json.dumps({"__kind__": "linac_gen_project", "__version__": 1,
                             "lattice_path": deck.name,
                             "beam": {"current": 1.0, "n_particles": 50},
                             "convergence": conv}))
    return p


@pytest.mark.parametrize("args", [
    ["--sc", "method=nope"],
    ["--sc-method", "tsc_fixed", "--sc", "method=cic_fixed"],
])
def test_cli_run_rejects_bad_method_choices_with_exit_code_2(args, tmp_path,
                                                             capsys):
    from linac_gen.__main__ import main as cli_main_fn
    rc = cli_main_fn(["run", str(_deck(tmp_path)), "--mode", "mp", "--current",
                      "1", "-q", "--out", str(tmp_path)] + args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "valid names" in err or "conflicts" in err


def test_cli_run_rejects_a_project_with_an_unknown_method(tmp_path, capsys):
    from linac_gen.__main__ import main as cli_main_fn
    proj = _project_file(tmp_path, {"sc_method": "tsc_adapative"})
    rc = cli_main_fn(["run", str(proj), "--mode", "mp", "-q", "--out",
                      str(tmp_path), "--sc-method", "cic_fixed"])
    assert rc == 2
    assert "tsc_adapative" in capsys.readouterr().err


def test_cli_backtrack_checks_the_method_without_sc_on(tmp_path, capsys):
    from linac_gen.__main__ import main as cli_main_fn
    rc = cli_main_fn(["backtrack", str(_deck(tmp_path)), "--sc", "method=nope",
                      "-q", "--out", str(tmp_path)])
    assert rc == 2
    assert "valid names" in capsys.readouterr().err


def test_scan_kernel_flag_and_generic_override_follow_run(monkeypatch,
                                                         tmp_path):
    """--sc kernel= wins over --kernel in scan as in run, and the method
    reaches the scan point."""
    from linac_gen.__main__ import main as cli_main_fn
    seen = {}

    def capture(*a, **kw):
        seen.update(kw["sc_overrides"])
        raise ValueError("stop after the first point")

    monkeypatch.setattr(common, "build_scan_point", capture)
    rc = cli_main_fn(["scan", str(_deck(tmp_path)), "--vary", "nx=24",
                      "--kernel", "tsc", "--sc", "kernel=cic",
                      "--sc-method", "cic_adaptive", "-q", "--out",
                      str(tmp_path / "s.csv")])
    assert rc == 2
    assert seen["kernel"] == "cic" and seen["method"] == "cic_adaptive"


# ---------------------------------------------------------------------------
# projects that name a method
# ---------------------------------------------------------------------------
_GUI_BLOCK = {"grid_nx": 48, "grid_extent_sigma": 7.0, "kernel": "cic",
              "grid_mode": "fixed", "green_kind": "igf", "backend": "auto",
              "dc_kernel": "uniform", "csr_enabled": False,
              "sc_backend": "numpy"}


def test_project_method_wins_over_contradicting_stored_keys():
    """A GUI-style block whose sc_method was edited: the method decides the
    pinned settings, the stored grid size and extent stay, one note per
    contradicting key."""
    conv = {**_GUI_BLOCK, "sc_method": "tsc_adaptive"}
    for cli in ({}, {"sc_method": "tsc_adaptive"}):
        notes = []
        cfg = methods.resolve_sc(conv, cli, notes=notes)
        assert cfg.method == "tsc_adaptive"
        assert (cfg.kernel, cfg.grid_mode) == ("tsc", "adaptive")
        assert (cfg.nx, cfg.grid_extent) == (48, 7.0)
        assert len(notes) == 2 and all("contradicts" in n for n in notes)


def test_switching_method_keeps_the_projects_general_settings():
    conv = {"sc_method": "tsc_adaptive", "grid_nx": 64, "use_gpu": "cpu",
            "train_images": False, "csr_bins": 300, "csr_enabled": "false",
            "dc_kernel": "gaussian"}
    notes = []
    cfg = methods.resolve_sc(conv, {"sc_method": "cic_adaptive"}, notes=notes)
    assert cfg.method == "cic_adaptive" and cfg.nx == 48 and notes
    assert (cfg.use_gpu, cfg.train_images, cfg.csr_bins) == ("cpu", False, 300)
    assert cfg.csr_enabled is False and cfg.dc_kernel == "gaussian"
    # the same project without a switch agrees on every general setting
    same = methods.resolve_sc(conv, {})
    for key in ("use_gpu", "train_images", "csr_bins", "csr_enabled",
                "dc_kernel"):
        assert getattr(same, key) == getattr(cfg, key), key


def test_null_project_keys_are_skipped():
    cfg = methods.resolve_sc({"sc_method": "tsc_fixed", "backend": None,
                              "grid_nx": None}, {"sc_method": "cic_fixed"})
    assert cfg.use_gpu == "auto" and cfg.nx == 48


def test_project_with_only_general_keys_has_no_own_method():
    notes = []
    cfg = methods.resolve_sc({"backend": "cpu"}, {"sc_method": "tsc_fixed"},
                             notes=notes)
    assert cfg.method == "tsc_fixed" and cfg.nx == 48 and not notes
    assert cfg.use_gpu == "cpu"


def test_project_sc_method_custom_keeps_every_stored_key():
    cfg = methods.resolve_sc({"sc_method": "custom", "kernel": "tsc",
                              "grid_mode": "adaptive", "grid_nx": 40}, {})
    assert cfg.method == "tsc_adaptive"
    assert (cfg.kernel, cfg.grid_mode, cfg.nx) == ("tsc", "adaptive", 40)


def test_project_unknown_method_is_an_error_even_when_overridden():
    with pytest.raises(ValueError, match="valid names"):
        methods.resolve_sc({"sc_method": "cic_fast"},
                           {"sc_method": "cic_fixed"})


def test_cli_run_rejects_unknown_method_with_exit_code_2(tmp_path, capsys):
    from linac_gen.__main__ import main as cli_main_fn
    deck = tmp_path / "d.dat"
    deck.write_text("DRIFT 100 20 0\nEND\n")
    rc = cli_main_fn(["run", str(deck), "--mode", "mp", "--current", "1",
                        "--sc-method", "nope", "-q", "--out", str(tmp_path)])
    assert rc == 2
    assert "valid names" in capsys.readouterr().err


def test_no_contradiction_note_when_the_project_settings_are_not_used():
    conv = {**_GUI_BLOCK, "sc_method": "tsc_adaptive"}
    notes = []
    cfg = methods.resolve_sc(conv, {"sc_method": "cic_fixed"}, notes=notes)
    assert cfg.kernel == "cic"
    assert not any("contradicts" in n for n in notes)
    notes = []
    cfg = methods.resolve_sc(conv, {"kernel": "cic"}, notes=notes)
    assert cfg.kernel == "cic"
    assert [n for n in notes if "contradicts" in n] == [
        n for n in notes if "grid_mode" in n]


@pytest.mark.parametrize("name", ["tsc_adaptive", "custom"])
def test_naming_a_legacy_adaptive_projects_own_method_keeps_the_exact_build(
        name):
    """Naming the project's own method (or custom) changes nothing: the
    classic value of every setting the project does not store stays."""
    conv = {**_LEGACY_TSC, "grid_mode": "adaptive"}
    base = _cfg(conv)
    cfg = _cfg(conv, {"sc_method": name})
    assert base.green_build == cfg.green_build == "exact"
    assert cfg.grid_card == "apply"
    for key in ("nx", "grid_extent", "kernel", "grid_mode", "green_kind"):
        assert getattr(cfg, key) == getattr(base, key), key
