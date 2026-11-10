"""Fault scoring sees the orbit (2026-11-08).

The criticality rule looked at emittance growth, beam loss, exit energy and loss power: in
envelope mode, which loses no particle, a bend at 90 % that put the beam edge outside the aperture
at 327 elements of a transfer line scored non-critical.  A run now records its largest beam edge
over the aperture, max over elements of (|centroid| + 3 sigma) / half-aperture, and a fault is
critical when that passes the aperture and the baseline's own value."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.reliability.observables import criticality_rule, evaluate_run

BASE = {"emit_nx": 1.0, "emit_ny": 1.0, "emit_nz": 1.0, "ref_w_kin": 100.0}


@pytest.mark.parametrize("edge0, edge, critical", [
    (0.50, 1.30, True), (0.50, 0.90, False), (None, 1.05, True),
    (1.20, 1.25, True), (1.20, 1.10, False), (0.50, None, False)],
    ids=["outside", "inside", "no baseline value", "worse than a baseline already outside",
         "better than a baseline already outside", "not recorded"])
def test_a_fault_that_puts_the_beam_edge_outside_the_aperture_is_critical(edge0, edge, critical):
    got, terms = criticality_rule({**BASE, "edge_over_aperture": edge0},
                                  {**BASE, "edge_over_aperture": edge}, {})
    assert got is critical
    assert ("edge_over_aperture" in terms) is (edge is not None)


def test_the_term_can_be_switched_off():
    got, terms = criticality_rule({**BASE, "edge_over_aperture": 0.5},
                                  {**BASE, "edge_over_aperture": 3.0}, {"aperture_edge": None})
    assert got is False and terms["edge_over_aperture"] == 3.0


DECK = """FREQ 352.21
DRIFT 200 20
QUAD 100 6 20
DRIFT 200 20
EDGE 5 2000 30 0.45 2.8 20 0
BEND 8 2000 0 20 0
EDGE 5 2000 30 0.45 2.8 20 0
DRIFT 2000 20
QUAD 100 -6 20
DRIFT 2000 20
APERTURE 15 10 0
DRIFT 500 20
END
"""


def _run(tmp_path, tag, angle_rel=0.0):
    from linac_gen.cli.common import apply_element_override, run_envelope_sim
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.hdf5_output import save_results_hdf5
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "line.dat"
    deck.write_text(DECK)
    lat, _ = parse_tracewin(deck)
    if angle_rel:
        apply_element_override(lat, "BEND_001.field_rel", angle_rel)
    cfg = BeamConfig(species="proton", energy=800.0, frequency=352.21, current=0.0,
                     emit_nx=0.3, emit_ny=0.3, emit_z=0.3, beta_x=8.0, beta_y=8.0, beta_z=1.0)
    res = run_envelope_sim(lat, cfg)
    h5 = tmp_path / f"{tag}.h5"
    save_results_hdf5(res, str(h5))
    index = {getattr(e, "name", None): i for i, e in enumerate(lat.elements)}
    from linac_gen.reliability.campaign import ReliabilityCampaign
    apert = ReliabilityCampaign._apertures(None, lat)
    out = evaluate_run(str(h5), element_index=index, landmarks={}, sections=(), current_mA=0.0,
                       duty_pct=1.0, apertures=apert)
    return out, res, lat, apert


def test_the_edge_is_read_from_the_run_at_every_aperture(tmp_path):
    out, res, lat, apert = _run(tmp_path, "nominal", angle_rel=-0.1)
    cen = np.asarray(res.centroid, dtype=float)
    want = max(max((abs(cen[j, 0]) + 3 * res.sigma_x[j]) / ax, (abs(cen[j, 2]) + 3 * res.sigma_y[j]) / ay)
               for i, (ax, ay) in apert.items() for j in [int(res.element_exit_idx[i])])
    assert out["edge_over_aperture"] == pytest.approx(want, rel=1e-12)
    assert out["edge_element"] in {getattr(e, "name", None) for e in lat.elements}
    assert any(type(lat.elements[i]).__name__ == "Aperture" and v == (15.0, 10.0)
               for i, v in apert.items())


def test_a_bend_at_90_percent_scores_critical_in_envelope_mode(tmp_path):
    """No particle is lost in envelope mode, so only the orbit term can see it."""
    base, *_ = _run(tmp_path, "nominal")
    fault, *_ = _run(tmp_path, "bend90", angle_rel=-0.1)
    keys = ("emit_nx", "emit_ny", "emit_nz", "ref_w_kin")
    b = {**dict.fromkeys(keys, 1.0), "edge_over_aperture": base["edge_over_aperture"]}
    f = {**dict.fromkeys(keys, 1.0), "edge_over_aperture": fault["edge_over_aperture"]}
    assert base["edge_over_aperture"] < 1.0 < fault["edge_over_aperture"]
    assert criticality_rule(b, f, {})[0] is True
    assert criticality_rule(b, f, {"aperture_edge": None})[0] is False



def test_the_edge_is_the_formula_at_hand_numbers():
    """Rows by hand: element A's exit row has x = 3 mm, sigma_x 2 mm in a 10 mm half-aperture
    (0.9) and sigma_y 4 mm in a 10 mm one (1.2, the vertical plane leading); element B's row has
    x = 1, sigma_x 1 in a 20 x 5 bore (0.2) and sigma_y 1 (0.6)."""
    from linac_gen.analysis.aperture_edge import edge_over_aperture
    cen = np.zeros((3, 6))
    cen[1, 0], cen[2, 0] = 3.0, 1.0
    sx, sy = np.array([1.0, 2.0, 1.0]), np.array([1.0, 4.0, 1.0])
    worst, where, by = edge_over_aperture(cen, sx, sy, [0, 1, 2], {1: "A", 2: "B"},
                                          {1: (10.0, 10.0), 2: (20.0, 5.0)})
    assert where == "A" and worst == pytest.approx(1.2) and by == {"A": pytest.approx(1.2),
                                                                   "B": pytest.approx(0.6)}


def test_the_apertures_are_the_ones_the_tracker_scrapes():
    from types import SimpleNamespace

    from linac_gen.analysis.aperture_edge import half_apertures
    from linac_gen.elements.aperture import Aperture
    from linac_gen.elements.drift import Drift
    lat = SimpleNamespace(elements=[Drift("D1", length=100.0, aperture=20.0),
                                    Drift("D2", length=100.0, aperture=14.0, aperture_y=10.0),
                                    Aperture("A0", dx=15.0, dy=10.0, aperture_type=0),
                                    Aperture("A1", dx=12.0, dy=3.0, aperture_type=1),
                                    Aperture("A2", dx=12.0, dy=3.0, aperture_type=2),
                                    Drift("D3", length=100.0)])
    assert half_apertures(lat) == {0: (20.0, 20.0), 1: (14.0, 10.0), 2: (15.0, 10.0),
                                   3: (12.0, 12.0)}


@pytest.mark.parametrize("by, critical", [({"A": 1.24, "B": 1.10}, True), ({"A": 1.20, "B": 0.90}, False),
                                          ({"A": 1.30, "B": 0.30}, True)],
                         ids=["outside elsewhere", "inside elsewhere", "worse at the scraper"])
def test_a_scraper_the_baseline_fills_does_not_mask_the_rest_of_the_line(by, critical):
    """The baseline fills a collimator (1.24 there): a fault that pushes the edge out at another
    element is critical though the line's largest ratio stays 1.24."""
    base = {**BASE, "edge_over_aperture": 1.24, "edge_by_element": {"A": 1.24, "B": 0.30}}
    row = {**BASE, "edge_over_aperture": max(by.values()), "edge_by_element": by}
    got, terms = criticality_rule(base, row, {})
    assert got is critical
    if critical:
        assert terms["edge_out_element"] == max((n for n in by if by[n] > max(1.0, base["edge_by_element"][n])),
                                                key=by.get)


def test_a_configured_threshold_is_the_one_used():
    base = {**BASE, "edge_over_aperture": 0.5, "edge_by_element": {"A": 0.5}}
    row = {**BASE, "edge_over_aperture": 1.3, "edge_by_element": {"A": 1.3}}
    assert criticality_rule(base, row, {})[0] is True
    assert criticality_rule(base, row, {"aperture_edge": 1.5})[0] is False
    assert criticality_rule(base, row, {"aperture_edge": 1.2})[0] is True


def test_the_campaign_records_the_edge_of_a_run(tmp_path):
    """Through the campaign's own reading of a run (its extras), apertures from its lattice."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.project import write_project
    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.reliability.spec import default_spec
    deck = tmp_path / "line.dat"
    deck.write_text(DECK)
    proj = write_project(tmp_path / "line.lgproj", lattice_path=deck,
                         beam=BeamConfig(species="proton", energy=800.0, frequency=352.21,
                                         current=0.0, emit_nx=0.3, emit_ny=0.3, emit_z=0.3,
                                         beta_x=8.0, beta_y=8.0, beta_z=1.0))
    c = ReliabilityCampaign.create(tmp_path / "campaign", default_spec(str(proj), preset="quick",
                                                                        name="e"))
    item = c.plan()["faults"][0]
    out, res, lat, _a = _run(tmp_path, "rec", angle_rel=-0.1)
    from linac_gen.io.hdf5_output import save_results_hdf5
    c.item_dir(item).mkdir(parents=True, exist_ok=True)
    save_results_hdf5(res, str(c.item_dir(item) / "results.h5"))
    extras = c._extras(item)
    assert extras["edge_over_aperture"] == pytest.approx(out["edge_over_aperture"], rel=1e-12)
    assert extras["edge_by_element"] and extras["edge_element"] == out["edge_element"]


def test_the_compensated_run_carries_the_edge(tmp_path):
    """compensation's forward run reports it, so the rule's verdict on a compensated case sees
    the edge (it never did: an edge-only critical fault came back recovered by the rule)."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.failures.compensation import _forward_metrics
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "line.dat"
    deck.write_text(DECK)
    lat, _ = parse_tracewin(deck)
    m = _forward_metrics(lat, BeamConfig(species="proton", energy=800.0, frequency=352.21,
                                         current=0.0, emit_nx=0.3, emit_ny=0.3, emit_z=0.3,
                                         beta_x=8.0, beta_y=8.0, beta_z=1.0), "envelope")
    assert m["edge_over_aperture"] is not None and isinstance(m["edge_by_element"], dict)


def test_faults_csv_carries_the_edge():
    from linac_gen.reliability.summary import FAULT_COLS
    assert {"edge_over_aperture", "edge_element", "edge_out_element"} <= set(FAULT_COLS)
