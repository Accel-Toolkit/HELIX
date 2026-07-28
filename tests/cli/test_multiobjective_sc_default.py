# tests/cli/test_multiobjective_sc_default.py
"""multiobjective CLI: space charge is automatic from current > 0.

The old opt-in --space-charge flag was the one entry point where SC
silently defaulted OFF for a current-carrying beam — inconsistent with
`linac_gen run`.  Now: current > 0 → SC on; --no-space-charge opts out;
--space-charge is a deprecated no-op.
"""
from __future__ import annotations

import argparse

import pytest

import linac_gen.cli.multiobjective as mo


@pytest.fixture()
def deck(tmp_path):
    # Valid for the REAL path too (bounded ADJUST; the objectives used
    # below exist) — the mocked tests only need it to parse, but the
    # end-to-end opt-out test runs it for real.
    dat = tmp_path / "mini.dat"
    dat.write_text("FREQ 352.21\n"
                   "ADJUST 1 2 0 0.5 20.0\n"
                   "QUAD 80 5.0 20 0 0 0 0 0\n"
                   "DRIFT 200 20 0\n"
                   "END\n")
    return str(dat)


def _run(deck, extra, monkeypatch):
    captured = {}

    def fake_pareto(lattice, beam_cfg, objectives, **kw):
        captured.update(kw, current=beam_cfg.current)
        raise SystemExit(0)      # stop before any real optimization

    monkeypatch.setattr(
        "linac_gen.matching.multiobjective.pareto_optimize", fake_pareto)
    p = argparse.ArgumentParser()
    mo.add_arguments(p)
    args = p.parse_args([deck, "--objective", "emit_x",
                         "--objective", "emit_y", *extra])
    with pytest.raises(SystemExit):
        mo.run(args)
    return captured


def test_current_gt_zero_enables_sc(deck, monkeypatch):
    got = _run(deck, ["--current", "10"], monkeypatch)
    assert got["space_charge"] is True


def test_no_space_charge_opts_out(deck, monkeypatch):
    got = _run(deck, ["--current", "10", "--no-space-charge"], monkeypatch)
    assert got["space_charge"] is False


def test_zero_current_means_no_sc(deck, monkeypatch):
    got = _run(deck, [], monkeypatch)
    assert got["space_charge"] is False


def test_deprecated_flag_accepted_and_noop(deck, monkeypatch, capsys):
    got = _run(deck, ["--current", "10", "--space-charge"], monkeypatch)
    assert got["space_charge"] is True
    assert "deprecated" in capsys.readouterr().err


def test_no_space_charge_optout_is_warning_free(deck, tmp_path):
    """END-TO-END regression (review CONFIRMED-BUG): the explicit
    opt-out must not fire NoSpaceChargeWarning on mp evaluations — the
    matcher threads the "off" sentinel through to every Simulation it
    builds.  Before the fix this warned once per evaluation."""
    import warnings

    from linac_gen.tracking.tracker import NoSpaceChargeWarning

    p = argparse.ArgumentParser()
    mo.add_arguments(p)
    args = p.parse_args([deck, "--objective", "exit_sigma_x",
                         "--objective", "exit_sigma_y",
                         "--current", "10", "--no-space-charge",
                         "--cost-solver", "mp", "--mp-n-particles", "50",
                         "--algorithm", "nsga2",
                         "--pop-size", "2", "--n-gen", "1",
                         "--out", str(tmp_path / "pareto.csv")])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rc = mo.run(args)
    assert rc == 0
    sc_warnings = [w for w in caught
                   if issubclass(w.category, NoSpaceChargeWarning)]
    assert not sc_warnings, [str(w.message)[:80] for w in sc_warnings]
