"""StepConfig.drift_single_push through the CLI: precedence, the on/off flag on
run / scan / backtrack, the scan-point builder, and the provenance record."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pytest

from linac_gen.__main__ import main
from linac_gen.cli import common

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "fodo_cell.dat"


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), (1, True), (0, False),
    ("on", True), ("off", False), ("true", True), ("false", False), ("yes", True), ("no", False),
])
def test_as_bool(value, expected):
    assert common._as_bool(value) is expected


def test_as_bool_rejects_garbage():
    with pytest.raises(ValueError):
        common._as_bool("maybe")


@pytest.mark.parametrize("conv, cli, expected", [
    ({}, {}, True),
    ({"drift_single_push": False}, {}, False),
    ({"drift_single_push": "off"}, {}, False),
    ({"drift_single_push": False}, {"drift_single_push": True}, True),
    ({"drift_single_push": True}, {"drift_single_push": False}, False),
    ({}, {"drift_single_push": None}, True),
])
def test_make_step_config_precedence(conv, cli, expected):
    cfg = common.make_step_config(conv, cli)
    assert cfg.drift_single_push is expected
    assert (cfg.integration_steps_per_metre, cfg.sc_steps_per_metre) == (100.0, 50.0)


def test_build_scan_point_carries_the_option():
    from linac_gen.parallel.scan_pool import ScanPoint
    p = common.build_scan_point(str(_DAT))
    assert isinstance(p, ScanPoint) and p.drift_single_push is True
    p = common.build_scan_point(str(_DAT), cli={"drift_single_push": False})
    assert p.drift_single_push is False
    p = common.build_scan_point(str(_DAT), cli={"drift_single_push": "off"})
    assert p.drift_single_push is False


@pytest.mark.parametrize("module", ["run", "scan", "backtrack"])
def test_parsers_accept_the_flag(module):
    import importlib
    mod = importlib.import_module(f"linac_gen.cli.{module}")
    p = argparse.ArgumentParser()
    mod.add_arguments(p)
    base = ["deck.dat"] + (["--vary", "q.gradient=1:2:1"] if module == "scan" else [])
    assert p.parse_args(base).drift_single_push is None
    assert p.parse_args(base + ["--drift-single-push", "off"]).drift_single_push == "off"
    assert p.parse_args(base + ["--drift-single-push", "on"]).drift_single_push == "on"
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--drift-single-push", "maybe"])
    if module != "scan":
        ov = mod._cli_overrides(p.parse_args(base + ["--drift-single-push", "off"]))
        assert ov["drift_single_push"] is False
        assert mod._cli_overrides(p.parse_args(base))["drift_single_push"] is None


@pytest.mark.parametrize("flag, expected", [("off", False), ("on", True)])
def test_run_records_the_option_in_provenance(tmp_path, flag, expected):
    rc = main(["run", str(_DAT), "--mode", "mp", "--energy", "2.1226695", "--freq", "162.5",
               "--species", "H-", "--n-particles", "400", "--drift-single-push", flag,
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    h5 = list(tmp_path.glob("*_results.h5"))
    assert len(h5) == 1
    with h5py.File(h5[0], "r") as f:
        assert bool(f["provenance"].attrs["drift_single_push"]) is expected


def test_scan_can_sweep_the_option():
    """--vary drift_single_push=0:1:1 is a structural sweep (like step1), so a
    scan can A/B the two walks; the value reaches the scan point as a bool."""
    from linac_gen.cli import scan as scan_cli
    var, vals = scan_cli._parse_vary("drift_single_push=0:1:1")
    assert var == "drift_single_push" and vals == [0.0, 1.0]
    assert scan_cli._kind("drift_single_push") == "structural"
    p0 = common.build_scan_point(str(_DAT), cli={"drift_single_push": 0.0})
    p1 = common.build_scan_point(str(_DAT), cli={"drift_single_push": 1.0})
    assert p0.drift_single_push is False and p1.drift_single_push is True
