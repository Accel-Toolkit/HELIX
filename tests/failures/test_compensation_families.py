"""Failure compensation stays within the failed element's family, and a bend is never part of it.

A bend sat in the "magnet" family with the quadrupoles and solenoids, so it compensated a failed
quadrupole through its angle, which moves the reference path; a failed bend was reported recovered
by quadrupoles; and an element with no family fell back to "magnet", so a failed steerer got
solenoids (2026-11-07; on a test line of two doublets around a 20° bend, 3 MeV protons at
162.5 MHz, a ±50 % angle window moved the exit RF phase by ±166°, and the bend at 90 %, its exit
orbit at +21.6 mm / +46.3 mrad, was "recovered" by quadrupoles left as they were or turned to
0 T/m, +28.2 mm / +64.9 mrad).  The families are RF (cavities, spares) and transverse focusing
(quadrupoles, solenoids); a failed bend is not compensable by re-tuning optics."""
from __future__ import annotations

import warnings
from dataclasses import asdict

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.failures import compensation as comp
from linac_gen.failures.compensation import CompensationConfig, compensate, select_zone
from linac_gen.failures.failure_mode import FailureKind
from linac_gen.failures.scenario import enumerate_scenarios
from linac_gen.io.tracewin_parser import parse_tracewin

DECK = """FREQ 352.21
DRIFT 100 20
GAP 300000 -30 20 0 0 0 0 0 0
DRIFT 100 20
QUAD 100 8 20
DRIFT 100 20
GAP 300000 -30 20 0 0 0 0 0 0
DRIFT 100 20
QUAD 100 -8 20
DRIFT 100 20
SOLENOID 100 0.2 20
DRIFT 100 20
EDGE 5 1000 30 0.45 2.8 20 0
BEND 10 1000 0 20 0
EDGE 5 1000 30 0.45 2.8 20 0
DRIFT 100 20
QUAD 100 8 20
THIN_STEERING 0.0001 0 20
DRIFT 100 20
GAP 300000 -30 20 0 0 0 0 0 0
DRIFT 100 20
QUAD 100 -8 20
DRIFT 100 20
SOLENOID 100 0.2 20
DRIFT 100 20
END
"""
CAVITIES = {"GAP_001", "GAP_002", "GAP_003"}
FOCUSING = {"QUAD_001", "QUAD_002", "QUAD_003", "QUAD_004", "SOL_001", "SOL_002"}
BEND = "BEND_001"


@pytest.fixture()
def deck(tmp_path):
    p = tmp_path / "line.dat"
    p.write_text(DECK)
    return p


def _beam():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21, current=0.0,
                      n_particles=1000, emit_nx=0.25, emit_ny=0.25, emit_z=0.3,
                      alpha_x=-0.5, beta_x=1.0, alpha_y=0.5, beta_y=1.0, beta_z=1.0)


def _scenario(lat, name, types=("cavity", "quad", "solenoid", "dipole")):
    scns, n2c, _names = enumerate_scenarios(lat, types=set(types), kind=FailureKind.OFF,
                                            combination="single")
    return next(s for s in scns if s.element_names == (name,)), n2c


_STRATEGIES = [CompensationConfig(strategy="k_out_of_n", k=3),
               CompensationConfig(strategy="l_neighboring_lattices", l=3),
               CompensationConfig(strategy="manual", manual_names=sorted(FOCUSING | CAVITIES | {BEND}))]


@pytest.mark.parametrize("cfg", _STRATEGIES, ids=["k_out_of_n", "l_neighboring_lattices", "manual"])
@pytest.mark.parametrize("failed", ["QUAD_003", "QUAD_002", "SOL_001", "GAP_002"])
def test_a_bend_is_never_a_compensator(deck, cfg, failed):
    lat, _ = parse_tracewin(deck)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zone = select_zone(lat, [failed], cfg)
    assert BEND not in zone and failed not in zone, zone


@pytest.mark.parametrize("failed, family", [("QUAD_003", FOCUSING), ("SOL_002", FOCUSING),
                                            ("GAP_002", CAVITIES)])
def test_a_failed_element_is_compensated_within_its_own_family(deck, failed, family):
    lat, _ = parse_tracewin(deck)
    zone = select_zone(lat, [failed], CompensationConfig(strategy="k_out_of_n", k=3))
    assert zone and set(zone) <= family - {failed}, zone


def test_a_failed_bend_is_not_compensable_and_no_match_is_run(deck, monkeypatch):
    lat, _ = parse_tracewin(deck)
    scenario, n2c = _scenario(lat, BEND)

    def no_match(*a, **k):
        raise AssertionError("the matcher ran for a failed bend")
    monkeypatch.setattr(comp, "match", no_match)
    res = compensate(lat, _beam(), scenario, n2c, comp._forward_metrics(lat, _beam(), "envelope"),
                     CompensationConfig(strategy="k_out_of_n", k=2, algorithm="least_squares"))
    assert res.recovered is False and res.compensator_names == [] and res.settings == {}
    assert "not compensable" in res.message and BEND in res.message
    assert res.metrics_after                     # the failed line's impact is still reported


def test_compensator_settings_never_move_the_reference_path(deck):
    """The knobs are cavity voltages and phases, quadrupole gradients and solenoid fields: no bend
    angle, so the reference path through the line is the design one."""
    lat, _ = parse_tracewin(deck)
    beam = _beam()
    base = comp._forward_metrics(lat, beam, "envelope")
    for failed in ("QUAD_003", "SOL_001"):
        scenario, n2c = _scenario(lat, failed)
        res = compensate(lat, beam, scenario, n2c, base,
                         CompensationConfig(strategy="k_out_of_n", k=3, algorithm="least_squares",
                                            max_iter=3))
        assert res.compensator_names and BEND not in res.compensator_names
        assert {k.split(".")[1] for k in res.settings} <= {"gradient", "field"}, res.settings
    assert "Dipole" not in comp._COMP_PARAMS


def test_a_manual_zone_drops_a_bend_and_says_so(deck):
    lat, _ = parse_tracewin(deck)
    with pytest.warns(UserWarning, match=f"{BEND}.*bend is never a compensator"):
        zone = select_zone(lat, ["QUAD_003"], CompensationConfig(strategy="manual",
                                                                 manual_names=[BEND, "QUAD_004"]))
    assert zone == ["QUAD_004"]


def test_an_element_with_no_family_gets_no_compensators(deck):
    """A steerer (and, in the default vocabulary, a multi-gap cavity) fell back to the "magnet"
    family: a failed steerer got solenoids."""
    lat, _ = parse_tracewin(deck)
    for cfg in (CompensationConfig(strategy="k_out_of_n", k=3),
                CompensationConfig(strategy="k_out_of_n", k=3, extra_types=("steerer",))):
        assert select_zone(lat, ["STEER_001"], cfg) == []


def test_a_failed_ncells_cavity_gets_no_focusing_magnets():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.ncells import NCells
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    for e in (Quadrupole("Q1", length=100.0, gradient=8.0), Drift("D1", length=100.0),
              NCells("NC1", mode=1, n_cells=4, beta_g=0.2, eot_v_per_m=1.0e6, theta_s_deg=-30.0,
                     aperture_mm=20.0, frequency_mhz=352.21),
              Drift("D2", length=100.0), Quadrupole("Q2", length=100.0, gradient=-8.0)):
        lat.add(e)
    assert not set(select_zone(lat, ["NC1"], CompensationConfig(strategy="k_out_of_n", k=2))) \
        & {"Q1", "Q2"}


def test_a_campaign_compensation_job_never_tunes_a_bend(deck):
    """The reliability campaign's own job (family headroom windows, its objectives), for a failed
    quadrupole next to the bend and for the bend itself."""
    from linac_gen.reliability.campaign import compensation_job
    lat, _ = parse_tracewin(deck)
    beam = _beam()
    base = comp._forward_metrics(lat, beam, "envelope")
    _scns, n2c, _names = enumerate_scenarios(lat, types={"cavity", "quad", "solenoid", "dipole"},
                                             kind=FailureKind.OFF, combination="single")
    fam = {n: "".join(ch for ch in n if ch.isalpha()) for n in n2c}
    job = {"lattice_path": str(deck), "beam_config": asdict(beam), "name_to_class": n2c,
           "comp": {"algorithm": "least_squares", "cost_solver": "envelope", "max_iter": 3, "k": 2},
           "baseline_metrics": {k: base.get(k) for k in ("ref_w_kin", "transmission")},
           "family_of": fam, "amp_bounds_by_family": {f: (0.0, 1.5) for f in set(fam.values())},
           "objectives": {"ke_out_min": True, "emit_growth": True}, "strategy": "k_out_of_n"}
    quad = compensation_job({**job, "case_id": "q", "failures": [["QUAD_003", "off", 0.0, 0.0]]})
    assert quad["error"] is None and quad["compensator_names"], quad
    assert BEND not in quad["compensator_names"]
    assert not any(k.startswith(BEND) for k in quad["settings"])
    bend = compensation_job({**job, "case_id": "b", "failures": [[BEND, "off", 0.0, 0.0]]})
    assert bend["error"] is None and bend["compensator_names"] == [] and not bend["recovered"]
    assert "not compensable" in bend["message"]


def test_the_cli_reports_a_failed_bend_as_not_compensable(deck, tmp_path, capsys):
    import argparse

    from linac_gen.cli import failures as cli
    out = tmp_path / "f.csv"
    p = argparse.ArgumentParser()
    cli.add_arguments(p)
    args = p.parse_args([str(deck), "--types", "dipole", "--mode", "off", "--forward", "envelope",
                         "--workers", "1", "--compensate", "--comp-algorithm", "least_squares",
                         "--top", "1", "--out", str(out)])
    assert cli.run(args) == 0
    stdout = capsys.readouterr().out
    assert "not compensable" in stdout and BEND in stdout, stdout[-2000:]
    import csv
    rows = [row for row in csv.DictReader(out.open()) if any(BEND in str(v) for v in row.values())]
    assert rows and all(row["recovered"] == "False" and row["compensators"] == "" for row in rows)


def _fieldmap(name, ke, kb):
    from linac_gen.elements.field_map import FieldMap
    fm = FieldMap.__new__(FieldMap)        # no field file: zone selection reads the name and scales
    fm.name, fm.ke, fm.kb = name, ke, kb
    return fm


def _spare_line():
    from types import SimpleNamespace

    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap
    return SimpleNamespace(elements=[
        RFGap("C1", 1.0, -20.0, 325.0), Quadrupole("Q1", length=100.0, gradient=8.0),
        _fieldmap("SP", 0.0, 0.0), _fieldmap("SOL", 0.0, 0.4),
        Quadrupole("Q2", length=100.0, gradient=-8.0), RFGap("C2", 1.0, -20.0, 325.0)])


@pytest.mark.parametrize("extra", [(), ("spare",)], ids=["default", "spares asked for"])
def test_an_unpowered_cavity_is_a_spare_not_a_solenoid(extra):
    """It was labelled a solenoid and offered to focusing faults (4 quadrupole faults of a
    2907-element linac-to-foil deck, 2026-11-07)."""
    lat = _spare_line()
    zone = select_zone(lat, ["Q1"], CompensationConfig(strategy="k_out_of_n", k=3,
                                                       extra_types=extra))
    assert "SP" not in zone and set(zone) <= {"Q2", "SOL"} and zone, zone


def test_a_spare_compensates_a_cavity_when_asked():
    lat = _spare_line()
    assert "SP" not in select_zone(lat, ["C1"], CompensationConfig(strategy="k_out_of_n", k=3))
    zone = select_zone(lat, ["C1"], CompensationConfig(strategy="k_out_of_n", k=3,
                                                       extra_types=("spare",)))
    assert "SP" in zone and set(zone) <= {"SP", "C2"}, zone


def test_a_failed_multigap_cavity_is_compensated_by_cavities():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.ncells import NCells
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap
    lat = Lattice()
    for e in (RFGap("G1", 0.3, -30.0, 352.21), Quadrupole("Q1", length=100.0, gradient=8.0),
              NCells("NC1", mode=1, n_cells=4, beta_g=0.2, eot_v_per_m=1.0e6, theta_s_deg=-30.0,
                     aperture_mm=20.0, frequency_mhz=352.21),
              Drift("D1", length=100.0), Quadrupole("Q2", length=100.0, gradient=-8.0),
              RFGap("G2", 0.3, -30.0, 352.21)):
        lat.add(e)
    assert select_zone(lat, ["NC1"], CompensationConfig(strategy="k_out_of_n", k=2)) \
        == ["G1", "G2"]


# --- C3: the knobs of a field-map compensator follow its field ---------------------------
def _map(cls_name, channels, ke, kb, name="FM"):
    """A field-map element with in-memory channels (no field file)."""
    from linac_gen.io.field_map_data import FieldMapData
    if cls_name == "FieldMap3D":
        from linac_gen.elements.field_map_3d import FieldMap3D as cls
    else:
        from linac_gen.elements.field_map import FieldMap as cls
    el = cls.__new__(cls)
    fd = FieldMapData.__new__(FieldMapData)
    fd.channels = dict(channels)
    el.name, el.ke, el.kb, el.phase, el.field_data = name, ke, kb, -30.0, fd
    return el


def _grid3(fx, fy, fz):
    """A 3-D static magnetic channel sampled from the functions given."""
    import numpy as np

    from linac_gen.io.field_map_data import FieldChannel
    x = y = np.linspace(-10.0, 10.0, 5)
    z = np.linspace(0.0, 200.0, 21)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return FieldChannel(geometry=7, x=x, y=y, z=z, Fx=fx(X, Y, Z), Fy=fy(X, Y, Z), Fz=fz(X, Y, Z))


def _bump(Z):
    import numpy as np
    return np.exp(-((Z - 100.0) / 40.0) ** 2)


def _solenoid3():
    from linac_gen.io.tracewin_geom import Channel
    # Bz on the axis, the radial field of its slope off it
    return {Channel.STAT_B: _grid3(lambda X, Y, Z: -0.5 * X * (-(Z - 100) / 800) * _bump(Z),
                                   lambda X, Y, Z: -0.5 * Y * (-(Z - 100) / 800) * _bump(Z),
                                   lambda X, Y, Z: _bump(Z))}


def _quadrupole3():
    from linac_gen.io.tracewin_geom import Channel
    return {Channel.STAT_B: _grid3(lambda X, Y, Z: 0.1 * Y * _bump(Z), lambda X, Y, Z: 0.1 * X * _bump(Z),
                                   lambda X, Y, Z: 0.0 * Z)}


def _dipole3():
    from linac_gen.io.tracewin_geom import Channel
    return {Channel.STAT_B: _grid3(lambda X, Y, Z: 0.0 * Z, lambda X, Y, Z: 1.0 + 0.0 * Z,
                                   lambda X, Y, Z: 0.01 * Y * _bump(Z))}


def _cavity1():
    import numpy as np

    from linac_gen.io.field_map_data import FieldChannel
    from linac_gen.io.tracewin_geom import Channel
    z = np.linspace(0.0, 200.0, 41)
    return {Channel.RF_E: FieldChannel(geometry=1, z=z, Fz=_bump(z)),
            Channel.RF_B: FieldChannel(geometry=1, z=z, Fz=0.0 * z)}


def test_a_static_solenoid_map_is_retuned_through_its_kb():
    """Its ke and phase do nothing to a static field: re-tuned through them, a failed magnet's
    compensation left the exit σ exactly as it was (39 faults of a linac-to-foil deck)."""
    import numpy as np

    from linac_gen.io.field_map_data import FieldChannel
    from linac_gen.io.tracewin_geom import Channel
    z = np.linspace(0.0, 200.0, 41)
    one_d = {Channel.STAT_B: FieldChannel(geometry=1, z=z, Fz=_bump(z))}
    assert comp.knobs(_map("FieldMap3D", _solenoid3(), 0.0, 3.47)) == [(5, "kb", "amp")]
    assert comp.knobs(_map("FieldMap", one_d, 0.0, 1.87)) == [(5, "kb", "amp")]


@pytest.mark.parametrize("field", [_quadrupole3, _dipole3], ids=["quadrupole", "dipole"])
def test_a_static_quadrupole_or_dipole_map_is_no_compensator(field):
    """Neither has a longitudinal field on the axis; a dipole map's kb would move the reference
    path, as a bend's angle does."""
    assert comp.knobs(_map("FieldMap3D", field(), 0.0, 1.0)) == []


def test_a_cavity_map_scales_its_electric_and_magnetic_fields_together(monkeypatch):
    """ke alone changed the ratio of the RF electric to magnetic field the map carries (kb = ke
    in most shipped decks); with kb in the same optimiser column they scale together."""
    assert comp.knobs(_map("FieldMap", _cavity1(), 2.0, 2.0)) == \
        [(6, "ke", "amp"), (5, "kb", "amp"), (2, "phase", "phase")]
    assert comp.knobs(_map("FieldMap", _cavity1(), 2.0, 0.0)) == [(6, "ke", "amp"), (2, "phase", "phase")]
    spare = comp.knobs(_map("FieldMap3D", _cavity1(), 0.0, 0.0))
    assert [a for (_p, a, r) in spare if r == "amp"] == ["ke", "kb"]


def test_a_cavity_maps_ke_and_kb_share_one_column_in_the_match(monkeypatch):
    from types import SimpleNamespace

    from linac_gen.elements.lattice_commands import Adjust
    seen = []

    def spy(work, *a, **k):
        seen.extend(e for e in work.elements if isinstance(e, Adjust))
        return SimpleNamespace(success=True, cost=0.0, message="spy")
    monkeypatch.setattr(comp, "match", spy)
    monkeypatch.setattr(comp, "_forward_metrics", lambda *a, **k: {})
    cav = _map("FieldMap", _cavity1(), 2.0, 2.0, name="CAV2")
    lat = SimpleNamespace(elements=[_map("FieldMap", _cavity1(), 2.0, 2.0, name="CAV1"), cav])
    scenario = SimpleNamespace(element_names=("CAV1",), element_overrides=lambda n2c: [])
    comp.compensate(lat, _beam(), scenario, {"CAV1": "FieldMap", "CAV2": "FieldMap"}, {},
                    CompensationConfig(strategy="manual", manual_names=["CAV2"]))
    amp = [a for a in seen if a.param_idx in (5, 6)]
    assert len(amp) == 2 and amp[0].link_group == amp[1].link_group != 0
    assert [a for a in seen if a.param_idx == 2][0].link_group == 0


# --- C4: "neighbouring cells" zones ----------------------------------------------------
@pytest.mark.parametrize("failed, family", [("QUAD_003", FOCUSING), ("SOL_001", FOCUSING),
                                            ("GAP_002", CAVITIES), ("QUAD_001", FOCUSING)])
def test_a_neighbouring_cells_zone_keeps_to_the_failed_elements_family(deck, failed, family):
    """It took every knob in the cells, so a failed quadrupole re-tuned cavities and a failed
    cavity quadrupoles; and the line's first focusing element, in no cell, got nothing."""
    lat, _ = parse_tracewin(deck)
    zone = select_zone(lat, [failed], CompensationConfig(strategy="l_neighboring_lattices", l=1))
    assert zone and set(zone) <= family - {failed}, zone


def test_the_first_focusing_element_is_in_the_first_cell(deck):
    lat, _ = parse_tracewin(deck)
    assert "QUAD_001" in select_zone(lat, ["QUAD_002"],
                                     CompensationConfig(strategy="l_neighboring_lattices", l=1))


def test_an_element_with_no_family_gets_no_neighbouring_cells(deck):
    lat, _ = parse_tracewin(deck)
    for failed in ("STEER_001", BEND):
        assert select_zone(lat, [failed],
                           CompensationConfig(strategy="l_neighboring_lattices", l=2)) == []


# --- C5: the entry points --------------------------------------------------------------
def test_the_cli_compensates_with_the_projects_beam(deck, tmp_path, monkeypatch):
    """It compensated with a default beam (3 MeV at 352.21 MHz), not the one the sweep ran: a
    failed cavity of one deck came out at 6.72 MeV against its 23.72 and not recovered."""
    import argparse

    import linac_gen.failures as failures
    from linac_gen.cli import failures as cli
    from linac_gen.io.project import write_project
    proj = write_project(tmp_path / "line.lgproj", lattice_path=deck,
                         beam=BeamConfig(species="proton", energy=4.25, frequency=352.21,
                                         current=0.0, n_particles=200, emit_nx=0.25,
                                         emit_ny=0.25, emit_z=0.3, beta_x=1.0, beta_y=1.0,
                                         beta_z=1.0))
    seen = []

    def spy(lattice, beam, scenario, n2c, base, cfg):
        seen.append((beam.energy, beam.frequency))
        assert cfg.objectives and cfg.objectives.get("emit_growth")    # the campaign's
        return comp.CompensationResult(False, [], message="spy")
    monkeypatch.setattr(failures, "compensate", spy)
    p = argparse.ArgumentParser()
    cli.add_arguments(p)
    args = p.parse_args([str(proj), "--types", "cavity", "--mode", "off", "--forward", "envelope",
                         "--workers", "1", "--compensate", "--top", "1",
                         "--out", str(tmp_path / "f.csv")])
    assert cli.run(args) == 0
    assert seen == [(4.25, 352.21)]


# --- review fixes ------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [
    CompensationConfig(strategy="manual", manual_names=["QUAD_003", "QUAD_004"]),
    CompensationConfig(strategy="manual", manual_names=["GAP_003"], extra_types=("spare",))],
    ids=["a manual zone", "the campaign's spares job"])
def test_a_failed_bend_runs_no_match_whatever_the_zone(deck, monkeypatch, cfg):
    lat, _ = parse_tracewin(deck)
    scenario, n2c = _scenario(lat, BEND)

    def no_match(*a, **k):
        raise AssertionError("the matcher ran for a failed bend")
    monkeypatch.setattr(comp, "match", no_match)
    res = compensate(lat, _beam(), scenario, n2c, {}, cfg)
    assert res.compensator_names == [] and "not compensable" in res.message


def test_a_manual_zone_keeps_to_the_failed_elements_family_and_says_what_it_drops(deck):
    """The campaign's spares job gave a quadrupole fault spare cavities, and the tab's picks
    compensated one family with another."""
    lat, _ = parse_tracewin(deck)
    with pytest.warns(UserWarning, match="GAP_003 dropped — not of the failed element's family"):
        zone = select_zone(lat, ["QUAD_003"], CompensationConfig(
            strategy="manual", manual_names=["GAP_003", "QUAD_004"]))
    assert zone == ["QUAD_004"]
    with pytest.warns(UserWarning, match="STEER_001 dropped — no compensation knob"):
        zone = select_zone(lat, ["QUAD_003"], CompensationConfig(
            strategy="manual", manual_names=["STEER_001", "SOL_002"]))
    assert zone == ["SOL_002"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert select_zone(lat, ["STEER_001"], CompensationConfig(
            strategy="manual", manual_names=["QUAD_004", "GAP_003"])) == []


def test_each_cavity_compensator_has_its_own_column(monkeypatch):
    """ke and kb share a column within one cavity map, never across two (one column for all
    would start the second map at the first one's value)."""
    from types import SimpleNamespace

    from linac_gen.elements.lattice_commands import Adjust
    seen = []

    def spy(work, *a, **k):
        seen.extend(e for e in work.elements if isinstance(e, Adjust))
        return SimpleNamespace(success=True, cost=0.0, message="spy")
    monkeypatch.setattr(comp, "match", spy)
    monkeypatch.setattr(comp, "_forward_metrics", lambda *a, **k: {})
    lat = SimpleNamespace(elements=[_map("FieldMap", _cavity1(), 2.0, 2.0, name="CAV1"),
                                    _map("FieldMap", _cavity1(), 2.0, 2.0, name="CAV2"),
                                    _map("FieldMap", _cavity1(), 3.0, 3.0, name="CAV3")])
    scenario = SimpleNamespace(element_names=("CAV1",), element_overrides=lambda n2c: [])
    comp.compensate(lat, _beam(), scenario, dict.fromkeys(("CAV1", "CAV2", "CAV3"), "FieldMap"), {},
                    CompensationConfig(strategy="manual", manual_names=["CAV2", "CAV3"]))
    amp = [(a.target, a.link_group) for a in seen if a.param_idx in (5, 6)]
    groups = {t: {g for (tt, g) in amp if tt == t} for (t, _g) in amp}
    assert len(groups) == 2 and all(len(g) == 1 for g in groups.values())
    assert len({next(iter(g)) for g in groups.values()}) == 2


def test_a_cavity_map_whose_kb_differs_varies_ke_alone_and_says_so(monkeypatch):
    from types import SimpleNamespace
    cav = _map("FieldMap", _cavity1(), 2.0, 3.0, name="CAV2")
    assert comp.knobs(cav) == [(6, "ke", "amp"), (2, "phase", "phase")]
    monkeypatch.setattr(comp, "match", lambda *a, **k: SimpleNamespace(success=True, cost=0.0,
                                                                      message="spy"))
    monkeypatch.setattr(comp, "_forward_metrics", lambda *a, **k: {})
    lat = SimpleNamespace(elements=[_map("FieldMap", _cavity1(), 2.0, 2.0, name="CAV1"), cav])
    scenario = SimpleNamespace(element_names=("CAV1",), element_overrides=lambda n2c: [])
    with pytest.warns(UserWarning, match="kb differs from its ke"):
        comp.compensate(lat, _beam(), scenario, {"CAV1": "FieldMap", "CAV2": "FieldMap"}, {},
                        CompensationConfig(strategy="manual", manual_names=["CAV2"]))


def test_a_map_read_without_field_data():
    """ke == 0 with kb != 0 is a solenoid's map, any other a cavity's (a spare's when both are
    zero)."""
    assert comp.knobs(_fieldmap("SOL", 0.0, 0.4)) == [(5, "kb", "amp")]
    assert comp.knobs(_fieldmap("CAV", 2.0, 2.0)) == [(6, "ke", "amp"), (5, "kb", "amp"),
                                                       (2, "phase", "phase")]
    assert [a for (_p, a, r) in comp.knobs(_fieldmap("SP", 0.0, 0.0)) if r == "amp"] == ["ke", "kb"]


def test_a_weak_longitudinal_field_on_the_axis_is_no_solenoid():
    """On the axis Bz beats the transverse field, but stays under a tenth of the map's peak (a
    quadrupole map with a small end-field Bz): no solenoid."""
    from linac_gen.io.tracewin_geom import Channel
    weak = {Channel.STAT_B: _grid3(lambda X, Y, Z: 0.5 * Y * _bump(Z), lambda X, Y, Z: 0.5 * X * _bump(Z),
                                   lambda X, Y, Z: 0.05 * _bump(Z))}
    assert comp.knobs(_map("FieldMap3D", weak, 0.0, 1.0)) == []


def _cavities_and_quads():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap
    lat = Lattice()
    for i, (cls, kw) in enumerate([(Quadrupole, dict(length=100.0, gradient=8.0)),
                                   (RFGap, dict(voltage=0.3, phase=-30.0, frequency=352.21)),
                                   (Quadrupole, dict(length=100.0, gradient=-8.0)),
                                   (RFGap, dict(voltage=0.3, phase=-30.0, frequency=352.21)),
                                   (Quadrupole, dict(length=100.0, gradient=8.0)),
                                   (Drift, dict(length=100.0)),
                                   (RFGap, dict(voltage=0.3, phase=-30.0, frequency=352.21))]):
        lat.add(cls(f"E{i}", **kw))
    return lat


def test_an_element_after_the_last_focusing_element_is_in_the_last_cell():
    """The cell rule ends a cell at focuser k+2, so the line's tail was in no cell: a cavity
    there got no "neighbouring cells" zone."""
    lat = _cavities_and_quads()
    zone = select_zone(lat, ["E6"], CompensationConfig(strategy="l_neighboring_lattices", l=1))
    assert zone and set(zone) <= {"E1", "E3"}, zone


def test_an_unpowered_spare_is_no_focusing_element_of_a_cell():
    """The cell finder counted an unpowered cavity (ke = kb = 0) as a solenoid, so the cells beside
    it were one element long and cut the quadrupoles' zones."""
    from types import SimpleNamespace

    from linac_gen.elements.quadrupole import Quadrupole
    quads = [Quadrupole(f"Q{i}", length=100.0, gradient=(-1) ** i * 8.0) for i in range(6)]
    els = quads[:3] + [_map("FieldMap3D", _cavity1(), 0.0, 0.0, name="SP")] + quads[3:]
    zone = select_zone(SimpleNamespace(elements=els), ["Q2"],
                       CompensationConfig(strategy="l_neighboring_lattices", l=1))
    assert {"Q1", "Q3", "Q4"} <= set(zone) and "SP" not in zone, zone


def test_section_rematch_keeps_the_spares_in_its_list(deck, monkeypatch):
    """Its list carries the downstream spares of a cavity fault; asked for no spares, the zone
    dropped them all (the spares were candidates only as mislabelled solenoids before)."""
    import linac_gen.failures as failures
    from linac_gen.reliability.campaign import compensation_job
    seen = {}

    def spy(lattice, beam, scenario, n2c, base, cfg):
        seen["cfg"] = cfg
        return comp.CompensationResult(False, [], message="spy")
    monkeypatch.setattr(failures, "compensate", spy)
    lat, _ = parse_tracewin(deck)
    out = compensation_job({"lattice_path": str(deck), "beam_config": asdict(_beam()),
                            "name_to_class": {}, "comp": {}, "baseline_metrics": {},
                            "strategy": "section_rematch", "manual_names": ["GAP_003"],
                            "case_id": "c", "failures": [["GAP_002", "off", 0.0, 0.0]]})
    assert out["error"] is None and "spare" in seen["cfg"].extra_types


def test_the_spares_strategy_is_planned_for_an_rf_fault_only(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from linac_gen.reliability.campaign import ReliabilityCampaign
    from linac_gen.reliability.spec import default_spec
    demo = Path(__file__).resolve().parents[2] / "examples" / "reliability_demo"
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json"):
        shutil.copy2(demo / name, tmp_path / name)
    spec = default_spec(str(tmp_path / "reliability_demo.lgproj"), preset="quick", name="s",
                        circuits="circuits.json", landmarks={"treaty_element": "BPM_003"},
                        compensation={"strategies": ["k_out_of_n", "spares"], "top_n": 2,
                                      "spare_names": ["GAP_001"]})
    c = ReliabilityCampaign.create(tmp_path / "campaign", spec)
    cases, _n2c = c._cases_cached()
    gap = next(k for k in cases if k.case_id.startswith("S1_GAP_") and k.case_id.endswith("_re"))
    sol = next(k for k in cases if k.case_id.startswith("S2_SOL_") and k.case_id.endswith("_re"))

    def row(k):
        return {"item": {"case_id": k.case_id, "cls": k.cls, "bracket": "rephased",
                         "label": k.case_id, "elements": list(k.element_names), "mode": "envelope"},
                "criticality": {"critical": True, "score": 1.0}}
    monkeypatch.setattr(c, "ranking", lambda bracket: [row(gap), row(sol)])
    planned = {(i["case_id"], i["strategy"]) for i in c._plan_compensation()}
    assert (gap.case_id, "spares") in planned and (sol.case_id, "spares") not in planned
    assert (sol.case_id, "k_out_of_n") in planned


def test_n_knobs_counts_a_cavity_maps_ke_and_kb_once():
    from linac_gen.reliability.summary import _n_knobs
    assert _n_knobs({"FMAP_006.ke": 0.74, "FMAP_006.kb": 0.74, "FMAP_006.phase": -39.9,
                     "QUAD_1.gradient": 5.0, "SOL_1.kb": 1.2}) == 4
