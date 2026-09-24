"""WP5 — SET_PHASE_OUT: parser/writer round trip, the wrapped one-sided
residual, the inert stub with one warning, the envelope clock history
(analytic on the demo deck, persisted to HDF5), and the compensation
objective that restores the arrival phase."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.cli import common
from linac_gen.core.lattice import Lattice
from linac_gen.elements.lattice_commands import COMMAND_CLASSES, SetPhaseOut
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin
from linac_gen.matching import constraints as C

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"


@pytest.fixture(scope="module")
def demo():
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, conv = common.load_input(str(DEMO / "reliability_demo.lgproj"))
    truth = json.loads((DEMO / "truth.json").read_text(encoding="utf-8"))
    return lat, cfg, conv, truth


def test_card_registered_and_round_trips(tmp_path):
    assert COMMAND_CLASSES["SET_PHASE_OUT"] is SetPhaseOut
    src = tmp_path / "in.dat"
    src.write_text("FREQ 162.5\nDRIFT 100 20\nSET_PHASE_OUT 123.4 2 0.5\n"
                   "DRIFT 50 20\nSET_PHASE_OUT -30\nEND\n", encoding="utf-8")
    lat, meta = parse_tracewin(str(src))
    assert meta.get("warnings") == []
    cards = [e for e in lat.elements if isinstance(e, SetPhaseOut)]
    assert [(c.phase_deg, c.weight, c.tol_deg) for c in cards] == [(123.4, 2.0, 0.5), (-30.0, 1.0, 0.0)]
    out = tmp_path / "out.dat"
    write_tracewin(lat, str(out))
    lines = [ln.strip() for ln in out.read_text(encoding="utf-8").splitlines()]
    assert "SET_PHASE_OUT 123.4 2 0.5" in lines and "SET_PHASE_OUT -30 1 0" in lines
    lat2, _ = parse_tracewin(str(out))
    assert [(c.phase_deg, c.weight, c.tol_deg) for c in lat2.elements
            if isinstance(c, SetPhaseOut)] == [(123.4, 2.0, 0.5), (-30.0, 1.0, 0.0)]


def _res(phi_end):
    return SimpleNamespace(ref_phi_s=[0.0, phi_end])


def test_residual_wraps_and_is_one_sided():
    ev = C._make_set_phase_out_evaluator(SetPhaseOut("p", phase_deg=-10.0))
    assert ev(_res(350.0), None)[0] == 0.0                      # 350 == -10 mod 360
    assert ev(_res(-10.0 + 720.0), None)[0] == 0.0
    assert ev(_res(26.0), None)[0] == pytest.approx(36.0 / 360.0)
    assert ev(_res(-46.0), None)[0] == pytest.approx(36.0 / 360.0)
    assert ev(_res(170.0), None)[0] == pytest.approx(180.0 / 360.0)   # the far side
    tol = C._make_set_phase_out_evaluator(SetPhaseOut("p", phase_deg=0.0, tol_deg=2.0))
    assert tol(_res(1.5), None)[0] == 0.0
    assert tol(_res(-3.0), None)[0] == pytest.approx(1.0 / 360.0)
    lat = Lattice(); lat.add(SetPhaseOut("p", phase_deg=0.0, weight=10.0))
    cs = C.collect_constraints(lat)
    assert len(cs) == 1 and cs[0].label == "SET_PHASE_OUT:0deg"
    assert cs[0].evaluate(_res(36.0), lat)[0] == pytest.approx(1.0)  # 0.1 x 10
    lat0 = Lattice(); lat0.add(SetPhaseOut("p", phase_deg=0.0, weight=0.0))
    assert C.collect_constraints(lat0) == []


def test_inert_without_clock_history_warns_once(capsys, monkeypatch):
    monkeypatch.setattr(C, "_SET_PHASE_OUT_WARNED", False)
    ev = C._make_set_phase_out_evaluator(SetPhaseOut("p", phase_deg=5.0))
    assert ev(SimpleNamespace(ref_w_kin=[1.0]), None)[0] == 0.0
    assert ev(SimpleNamespace(ref_phi_s=[]), None)[0] == 0.0
    err = capsys.readouterr().err
    assert err.count("SET_PHASE_OUT found but results carry no ref_phi_s") == 1
    assert C._SET_PHASE_OUT_WARNED is True


def test_envelope_records_the_clock_and_hdf5_persists_it(demo, tmp_path):
    lat, cfg, _conv, truth = demo
    res = common.run_envelope_sim(copy.deepcopy(lat), cfg)
    assert len(res.ref_phi_s) == len(res.s) == len(res.ref_w_kin)
    c = truth["constants"]; m = c["m_proton_mev"]; lam = c["lambda_mm"]
    def beta(w):
        g = 1.0 + w / m
        return math.sqrt(1.0 - 1.0 / (g * g))
    w4 = truth["nominal"]["w_after_gap_mev"]["GAP_004"]
    # after GAP_004: 300 mm to the foil at W4, the foil's mean loss, 100 mm
    want = (truth["nominal"]["clock_at_gap_entrance_deg"]["GAP_004"]
            + 360.0 * 300.0 / (beta(w4) * lam)
            + 360.0 * 100.0 / (beta(w4 - truth["foil"]["mean_loss_mev"]) * lam))
    assert res.ref_phi_s[-1] == pytest.approx(want, rel=1e-9)
    assert res.ref_phi_s[0] == 0.0
    out = tmp_path / "env.h5"
    common.write_results(res, str(out), "hdf5", cfg, lat)
    import h5py
    with h5py.File(out, "r") as f:
        phi = f["reference"]["phi_s"][...]
    assert phi.shape == (len(res.s),) and phi[-1] == res.ref_phi_s[-1]


def test_objective_card_and_compensation_restores_the_arrival_phase(demo):
    from linac_gen.failures import CompensationConfig, compensate, enumerate_scenarios
    from linac_gen.failures.compensation import _forward_metrics, objective_cards
    lat, cfg, _conv, _truth = demo
    lead, trail = objective_cards(CompensationConfig(objectives={
        "phase_out": {"phase_deg": 12.5, "weight": 3.0, "tol_deg": 0.25}}), {"ref_w_kin": 1.0}, 5)
    assert [type(c).__name__ for c in trail] == ["SetKeOutMin", "SetPhaseOut"]
    assert (trail[1].phase_deg, trail[1].weight, trail[1].tol_deg) == (12.5, 3.0, 0.25)
    _l, t2 = objective_cards(CompensationConfig(objectives={"phase_out": {}}), {"ref_w_kin": 1.0}, 5)
    assert [type(c).__name__ for c in t2] == ["SetKeOutMin"]      # no target: skipped
    # end to end on the demo deck: GAP_002 off, one neighbour on each side.
    # (With the LAST gap off every compensator is upstream: the beam is then
    # faster over the middle section and arrives earlier, and no knob can
    # slow it without decelerating — the arrival phase has a physical
    # floor there, ~15 deg on this deck.  A downstream neighbour absorbs
    # the timing.)
    phi0 = common.run_envelope_sim(copy.deepcopy(lat), cfg).ref_phi_s[-1]
    scen, n2c, _ = enumerate_scenarios(lat, types={"cavity"})
    s = next(x for x in scen if x.element_names == ("GAP_002",))
    baseline = _forward_metrics(copy.deepcopy(lat), cfg, "envelope")
    def run(objectives):
        comp = CompensationConfig(strategy="k_out_of_n", k=1, algorithm="least_squares",
                                  cost_solver="envelope", max_iter=120,
                                  family_of={g: "GAP" for g in n2c},
                                  amp_bounds_by_family={"GAP": (0.0, 1.5)},
                                  objectives=objectives)
        r = compensate(copy.deepcopy(lat), cfg, s, n2c, baseline, comp)
        work = copy.deepcopy(lat)
        common.apply_element_override(work, "GAP_002.voltage_rel", -1.0)
        for k, v in r.settings.items():
            common.apply_element_override(work, k, v)
        phi = common.run_envelope_sim(work, cfg).ref_phi_s[-1]
        return r, (phi - phi0 + 180.0) % 360.0 - 180.0
    r_e, d_e = run(None)
    r_p, d_p = run({"phase_out": {"phase_deg": phi0, "weight": 5.0, "tol_deg": 0.0}})
    assert r_e.compensator_names == ["GAP_001", "GAP_003"]
    assert r_e.recovered and r_p.recovered
    assert abs(d_p) < 0.5 and abs(d_p) <= abs(d_e) + 1e-9
