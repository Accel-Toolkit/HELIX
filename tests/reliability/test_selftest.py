"""The built-in self-test: PASS on this tree, the pinned baseline present
and within tolerance, the regression half refuses without a BEFORE
snapshot and is bit-identical against a snapshot of the same tree, and
the CLI exit codes."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.__main__ import main
from linac_gen.reliability.selftest import (BASELINE_PATH, PUBLIC_CONTROL_CASES,
                                            _load_harness, format_result, run_selftest)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run_selftest(quick=True, out_dir=tmp_path_factory.mktemp("selftest"))


def test_quick_selftest_passes_every_check(result):
    assert result["verdict"] == "PASS", format_result(result)
    assert result["n_failed"] == 0 and result["n_checks"] >= 50
    names = {r["name"] for r in result["checks"]}
    for must in ("deck.parser_warnings", "deck.generator_reproduces_files",
                 "legA.rephased.deficit.GAP_004", "legA.frozen.deficit.GAP_001",
                 "legA.pin.clock.GAP_003", "legA.ranking.top_cavity", "legA.compensation.k1_neighbours",
                 "legA.compensation.recovered", "legC.seed0.correction_converged",
                 "legD.foil.highland_theta", "legD.strip.anchor_600", "legB.closed.two_of_three",
                 "legB.mc.single_block_within_3.5sigma", "legB.bins.deterministic_recovery_one_bin",
                 "report.embedded_png"):
        assert must in names, must
    text = format_result(result)
    assert "VERDICT PASS" in text and "CALIBRATION point" in text


def test_pinned_baseline_present_and_compared(result):
    assert BASELINE_PATH.exists()
    base = [r for r in result["checks"] if r["name"].startswith("baseline.")]
    assert base and all(r["status"] == "PASS" for r in base), [r for r in base if r["status"] != "PASS"]
    assert any(r["name"] == "baseline.key_set" for r in base)
    with np.load(BASELINE_PATH, allow_pickle=False) as z:
        keys = set(z.files)
    assert {"__machine__", "__deck_sha256__", "__helix_version__", "__git_rev__"} <= keys
    assert "legA|GAP_004|re|metrics" in keys and "legC|seed0|kicks" in keys


def test_regression_refuses_without_before_and_is_identical_with_one(tmp_path):
    res = run_selftest(quick=True, regression=True, before_dir=None, out_dir=tmp_path / "a",
                       regression_cases=["fodo_cell"])
    assert res["verdict"] == "REFUSED" and "no BEFORE snapshot" in res["regression"]["note"]
    assert res["regression"]["n_run"] == 3                       # fodo_cell env / matrix / mp
    with pytest.raises(ValueError, match="unknown control case"):
        run_selftest(quick=True, regression=True, before_dir=None, out_dir=tmp_path / "x",
                     regression_cases=["nope"])
    # a BEFORE snapshot of THIS tree through the harness's own run_case
    mod = _load_harness()
    before = tmp_path / "before"
    before.mkdir()
    from linac_gen.reliability.selftest import REPO
    for name, spec in mod.CASES.items():
        if name.split(":")[0] == "fodo_cell":
            out = mod.run_case(REPO, name, spec)
            np.savez_compressed(before / (name.replace(":", "__") + ".npz"), **out)
    res2 = run_selftest(quick=True, regression=True, before_dir=before, out_dir=tmp_path / "b",
                        regression_cases=["fodo_cell"])
    assert res2["verdict"] == "PASS", format_result(res2)
    assert (res2["regression"]["n_identical"], res2["regression"]["n_changed"],
            res2["regression"]["n_missing"]) == (3, 0, 0)
    # a snapshot missing a case makes the comparison incomplete → REFUSED
    (before / "fodo_cell__mp.npz").unlink()
    res3 = run_selftest(quick=True, regression=True, before_dir=before, out_dir=tmp_path / "c",
                        regression_cases=["fodo_cell"])
    assert res3["verdict"] == "REFUSED" and res3["regression"]["n_missing"] == 1


def test_cli_exit_codes_and_json(tmp_path, capsys):
    assert main(["reliability", "selftest", "--quick", "--out", str(tmp_path / "q")]) == 0
    out = capsys.readouterr().out
    assert "VERDICT PASS" in out
    assert main(["reliability", "selftest", "--regression", "--cases", "fodo_cell",
                 "--out", str(tmp_path / "r")]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert main(["reliability", "selftest", "--quick", "--json", "--out", str(tmp_path / "j")]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["verdict"] == "PASS" and doc["n_failed"] == 0
    assert main(["reliability", "selftest", "--regression", "--cases", "nope", "--out", str(tmp_path / "z")]) == 2
    assert PUBLIC_CONTROL_CASES[0] == "fodo_cell"
