"""scripts/sc_method_migrate.py: pin, switch and restore a project's
space-charge method; nothing is written without --write."""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest

from linac_gen.pic import methods

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import sc_method_migrate as mig  # noqa: E402

_LEGACY = {"grid_nx": 48, "grid_extent_sigma": 7.0, "kernel": "tsc",
           "grid_mode": "fixed", "green_kind": "igf", "backend": "auto",
           "dc_kernel": "uniform", "csr_enabled": False, "step1_per_m": 100.0}

#: projects of every kind: no space-charge key (follows the default),
#: legacy keys, a method with its params, custom, and the default alias
_KINDS = [
    {},
    {"step1_per_m": 100.0},
    _LEGACY,
    {"grid_nx": 64, "grid_mode": "adaptive"},
    {"nx": 32, "ny": 32, "nz": 32, "kernel": "tsc", "grid_mode": "adaptive"},
    {"sc_method": "tsc_adaptive", "grid_nx": 64, "dc_kernel": "pic2d"},
    {"sc_method": "custom", "grid_nx": 40, "kernel": "tsc"},
    {"sc_method": "default", "grid_nx": 64},
]

#: command lines a pin must not change
_CLIS = ([{}, {"nx": 64}, {"grid_extent": 7.0}, {"kernel": "tsc"},
          {"backend": "cpu"}, {"sc": {"grid_mode": "adaptive"}},
          {"sc": {"method": "cic_fixed"}}, {"sc": {"reuse_every": "2"}}]
         + [{"sc_method": n} for n in methods.names()]
         + [{"sc_method": n, "nx": 40} for n in methods.names()])


def _fields(conv, cli=None):
    cfg = methods.resolve_sc(conv, cli or {})
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
            if f.name != "method"}


def _resolved(conv, cli):
    try:
        return _fields(conv, cli)
    except ValueError as exc:
        return f"error: {exc}"


def _write(tmp_path, conv, name="p.lgproj"):
    p = tmp_path / name
    p.write_text(json.dumps({"__kind__": "linac_gen_project",
                             "convergence": conv}, indent=2))
    return p


@pytest.mark.parametrize("conv", _KINDS)
def test_pin_keeps_the_configuration(conv):
    new = mig.pin(conv)
    assert new["sc_method"] == (methods.own_method(conv) or "custom")
    assert _fields(new) == _fields(conv)
    for key, val in conv.items():                  # other keys untouched
        if key not in mig._sc_keys(conv):
            assert new[key] == val


def test_a_project_without_space_charge_keys_is_pinned_as_custom():
    """Such a project follows the default method; naming a method on the
    command line gives that method's defaults (review: a pin under the
    matching name, cic_fixed, kept 96^3 +-5 sigma for --sc-method
    cic_fixed instead of its 48^3 +-7 sigma)."""
    new = mig.pin({"step1_per_m": 100.0})
    assert new["sc_method"] == "custom"
    cfg = methods.resolve_sc(new, {"sc_method": "cic_fixed"})
    assert (cfg.nx, cfg.grid_extent) == (48, 7.0)


@pytest.mark.parametrize("conv", _KINDS)
def test_pin_keeps_every_command_line_also_after_a_default_change(
        conv, monkeypatch):
    before = [_resolved(conv, cli) for cli in _CLIS]
    new = mig.pin(conv)
    assert [_resolved(new, cli) for cli in _CLIS] == before
    monkeypatch.setattr(methods, "DEFAULT_METHOD", "tsc_adaptive")
    assert [_resolved(new, cli) for cli in _CLIS] == before


def test_pinned_project_ignores_later_default_changes(monkeypatch):
    new = mig.pin(_LEGACY)
    monkeypatch.setattr(methods, "DEFAULT_METHOD", "cic_adaptive")
    assert _fields(new) == _fields(_LEGACY)


def test_pin_refuses_a_legacy_project_with_the_torch_engine():
    """The GUI runs it with the PyTorch PIC, the command line with the
    numpy PIC: no pin keeps both."""
    with pytest.raises(ValueError, match="torch"):
        mig.pin({"grid_nx": 32, "sc_backend": "torch"})
    new = mig.pin({"sc_method": "differentiable_pic", "grid_nx": 32})
    assert new["sc_backend"] == "torch"


def test_pin_keeps_the_order_of_the_entries():
    conv = {"step1_per_m": 100.0, "sc_method": "tsc_fixed", "grid_nx": 48,
            "grid_extent_sigma": 7.0, "backend": "auto",
            "sc_backend": "numpy", "green_kind": "igf", "kernel": "tsc",
            "grid_mode": "fixed", "integrator_kind": "rk4",
            "dc_kernel": "uniform", "csr_enabled": False}
    new = mig.pin(conv)
    assert list(new)[:len(conv)] == list(conv)
    assert set(new) - set(conv) == {"grid_card", "green_build",
                                    "reuse_every"}


def test_switch_and_restore_round_trip():
    sw = mig.switch(_LEGACY, "tsc_adaptive")
    assert sw["sc_method"] == "tsc_adaptive"
    assert _fields(sw) == _fields({"sc_method": "tsc_adaptive"})
    back = mig.restore(sw)
    assert back == _LEGACY and list(back) == list(_LEGACY)
    with pytest.raises(ValueError, match="sc_previous"):
        mig.switch(sw, "cic_fixed")
    with pytest.raises(ValueError, match="nothing to restore"):
        mig.restore(_LEGACY)
    with pytest.raises(ValueError, match="valid names"):
        mig.switch(_LEGACY, "cic_fast")


def test_switch_gives_the_methods_defaults_and_keeps_the_rest():
    """Also when the project already matches the method (review: 64^3
    +-4 sigma was kept)."""
    conv = {"grid_nx": 64, "grid_extent_sigma": 4.0, "kernel": "cic",
            "grid_mode": "fixed", "dc_kernel": "pic2d", "csr_enabled": True,
            "backend": "cpu"}
    sw = mig.switch(conv, "cic_fixed")
    cfg = methods.resolve_sc(sw, {})
    assert (cfg.nx, cfg.grid_extent, cfg.method) == (48, 7.0, "cic_fixed")
    assert (cfg.dc_kernel, cfg.csr_enabled, cfg.use_gpu) == ("pic2d", True,
                                                            "cpu")


def test_switch_to_default_follows_the_default(monkeypatch):
    sw = mig.switch(_LEGACY, "default")
    assert sw["sc_method"] == "default"
    assert "kernel" not in sw and "grid_nx" not in sw
    monkeypatch.setattr(methods, "DEFAULT_METHOD", "tsc_adaptive")
    assert methods.resolve_sc(sw, {}).method == "tsc_adaptive"


def test_cli_dry_run_writes_nothing_and_write_does(tmp_path, capsys):
    p = _write(tmp_path, _LEGACY)
    before = p.read_text()
    assert mig.main(["--pin", str(p)]) == 0
    assert p.read_text() == before
    assert "nothing written" in capsys.readouterr().out
    assert mig.main(["--pin", "--write", str(p)]) == 0
    conv = json.loads(p.read_text())["convergence"]
    assert conv["sc_method"] == "tsc_fixed"
    assert _fields(conv) == _fields(_LEGACY)


def test_switch_then_restore_of_a_project_without_a_block(tmp_path):
    """The block --switch adds goes away again: the file is as before."""
    p = tmp_path / "bare.lgproj"
    p.write_text(json.dumps({"__kind__": "linac_gen_project",
                             "lattice_path": "x.dat"}, indent=2) + "\n")
    before = p.read_text()
    assert mig.main(["--switch", "tsc_adaptive", "--write", str(p)]) == 0
    conv = json.loads(p.read_text())["convergence"]
    assert conv["sc_method"] == "tsc_adaptive" and conv["sc_previous"] is None
    assert mig.main(["--restore", "--write", str(p)]) == 0
    assert p.read_text() == before


def test_cli_reports_errors_with_exit_code_2_and_goes_on(tmp_path, capsys):
    bad_name = _write(tmp_path, {"sc_method": "cic_fast"}, "a.lgproj")
    top_list = tmp_path / "b.lgproj"
    top_list.write_text("[1, 2]")
    conv_list = tmp_path / "c.lgproj"
    conv_list.write_text(json.dumps({"convergence": [1]}))
    good = _write(tmp_path, _LEGACY, "d.lgproj")
    rc = mig.main(["--pin", "--write", str(bad_name), str(top_list),
                   str(conv_list), str(good)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "valid names" in err and "not a JSON object" in err
    assert "convergence entry" in err
    assert json.loads(good.read_text())["convergence"]["sc_method"] == \
        "tsc_fixed"
