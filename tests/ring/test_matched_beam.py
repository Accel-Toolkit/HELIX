"""The beam matched to the ring at the start of a run, in the Beam tab's terms
(RingSimulation.matched_beam), and the run's check of the beam it starts
with: the Beam tab's beam made from the matched one is matched (BMAG 1, a
bunch length that does not breathe, a centroid that stays on the closed
orbit), a mismatched one is reported; a ramp's matched beam is that of its
first turn; earlier projects' "matched" source becomes the Beam tab's beam."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.steerer import Steerer
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring.config import migrate_source

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
REF = json.loads((ROOT / "tests" / "ring" / "fixtures" / "refs" /
                  "simple_booster_madx.json").read_text())
C_LIGHT = 299792458.0
#: the Beam tab's settings matched_beam gives
KEYS = ("alpha_x", "beta_x", "alpha_y", "beta_y", "disp_x", "disp_xp", "disp_y", "disp_yp",
        "alpha_z", "beta_z", "centroid_x", "centroid_xp", "centroid_y", "centroid_yp",
        "centroid_dphi", "centroid_dw")
#: the transition example's ramp (800 MeV -> 6 GeV at 15 Hz, the clock 300
#: turns before transition)
RAMP = {"energy": {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 6000.0,
                   "frequency_hz": 15.0, "t_min_s": 0.0005, "t_start_s": 0.0008},
        "voltage": {"kind": "table", "times_s": [0.0, 0.003, 0.03],
                    "values_mv": [0.4, 1.2, 1.2]},
        "gamma_t": 5.449167323, "t_start_s": 0.020106046391991875}


@pytest.fixture(scope="module")
def booster():
    # (through HELIX's loader, which keeps the deck's BEAM: 800 MeV protons)
    from linac_gen.cli.common import load_lattice
    return load_lattice(str(FIXTURE))


def _beam(**kw):
    d = dict(species="proton", energy=800.0, frequency=44.7044096, current=0.0, emit_nx=1.5,
             emit_ny=1.5, emit_z=2.0, beta_z=100.0, n_particles=2000, cutoff=4.0,
             distribution="gaussian")
    d.update(kw)
    return BeamConfig(**d)


def _matched(lat, beam, cfg=None):
    """(matched_beam's dict, the Beam tab's beam made from it)."""
    mb = RingSimulation(copy.deepcopy(lat), beam, cfg or RingConfig()).matched_beam()
    vals = {k: mb[k] for k in KEYS if mb[k] is not None}
    return mb, dataclasses.replace(beam, energy=mb["kinetic_mev"], **vals)


def _run(lat, beam, **kw):
    cfg = RingConfig(**dict(dict(n_turns=60, source="beam", tune_probes=0), **kw))
    return RingSimulation(copy.deepcopy(lat), beam, cfg).run()


def _bmag(b, a, bm, am):
    """BMAG of a beam of Twiss (b, a) in an ellipse (bm, am)."""
    return 0.5 * (b * (1 + am * am) / bm - 2 * a * am + (1 + a * a) / b * bm)


def _swing(v) -> float:
    v = np.asarray(v, dtype=float)
    return float((v.max() - v.min()) / v.mean())


def test_the_matched_beam_is_the_rings_optics_in_beam_tab_units(booster):
    mb, _b = _matched(booster, _beam())
    st = REF["start"]
    assert mb["species"] == "proton" and mb["kinetic_mev"] == pytest.approx(800.0, rel=1e-12)
    assert mb["rf_on"] and not mb["ramp"]
    assert mb["beta_x"] == pytest.approx(st["betx"], rel=1e-4)
    assert mb["beta_y"] == pytest.approx(st["bety"], rel=1e-4)
    assert mb["alpha_x"] == pytest.approx(st["alfx"], abs=1e-4)
    assert mb["alpha_y"] == pytest.approx(st["alfy"], abs=1e-4)
    # dispersion per MeV of kinetic energy: D/(beta0 p0c), m → mm
    per_mev = 1e3 / (REF["beta0"] * REF["pc_gev"] * 1e3)
    assert mb["disp_x"] == pytest.approx(st["dx_delta"] * per_mev, rel=1e-3)
    assert mb["disp_xp"] == pytest.approx(st["dpx_delta"] * per_mev, abs=1e-6)
    assert mb["disp_y"] == mb["disp_yp"] == 0.0
    # a ring without errors: on the axis (the search's residue made zero)
    assert all(mb[k] == 0.0 for k in KEYS if k.startswith("centroid"))
    # the RF of the first turn: harmonic 84 of the revolution frequency
    f_rev = REF["beta0"] * C_LIGHT / REF["length_m"]
    assert mb["rf_frequency_hz"] == pytest.approx(84 * f_rev, rel=1e-9)
    # the longitudinal Twiss: those of the smooth approximation, beta_zeta =
    # |eta| C / (2 pi Qs), within its thin-kick correction (2 pi Qs)^2/6 = 0.4 %
    g0 = 1.0 / math.sqrt(1.0 - REF["beta0"] ** 2)
    eta = REF["alfa"] - 1.0 / g0 ** 2
    beta_zeta = abs(eta) * REF["length_m"] / (2.0 * math.pi * mb["qs"])
    e0 = REF["pc_gev"] * 1e3 / REF["beta0"]
    deg_per_m = 360.0 * 44.7044096e6 / (REF["beta0"] * C_LIGHT)
    beta_z = beta_zeta * deg_per_m / (REF["beta0"] ** 2 * e0)
    assert mb["beta_z"] == pytest.approx(beta_z, rel=5e-3)
    assert mb["frequency_mhz"] == 44.7044096 and abs(mb["alpha_z"]) < 0.1
    # the phase in degrees of another frequency: beta_z scales with it, alpha_z not
    mb2 = RingSimulation(copy.deepcopy(booster), _beam(), RingConfig()).matched_beam(
        frequency_mhz=2 * 44.7044096)
    assert mb2["beta_z"] == pytest.approx(2 * mb["beta_z"], rel=1e-12)
    assert mb2["alpha_z"] == mb["alpha_z"] and mb2["beta_x"] == mb["beta_x"]
    with pytest.raises(ValueError, match="must be positive"):
        RingSimulation(copy.deepcopy(booster), _beam(), RingConfig()).matched_beam(
            frequency_mhz=0.0)


def test_the_beam_tabs_beam_made_matched_is_matched(booster):
    mb, beam = _matched(booster, _beam())
    res = _run(booster, beam)
    # BMAG 1 in every mode, no note; the bunch length does not breathe
    assert res.meta["start_mismatch"] == pytest.approx([1.0, 1.0, 1.0], abs=2e-4)
    assert not any("mismatched" in n for n in res.meta["notes"])
    sz = res.sigma[:, 4]
    assert _swing(sz) < 0.015                         # (0.6 %: the particles' noise)
    # the matched source of the same emittances: the same beam, statistically
    ref = RingSimulation(copy.deepcopy(booster), beam,
                         RingConfig(n_turns=60, n_particles=2000, source="matched",
                                    tune_probes=0)).run()
    assert ref.meta["start_mismatch"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)
    assert float(np.mean(res.sigma[:, 4])) == pytest.approx(float(np.mean(ref.sigma[:, 4])),
                                                             rel=0.03)
    assert res.meta["eps0_normal_modes"] == pytest.approx(ref.meta["eps0_normal_modes"],
                                                          rel=1e-3)
    # beta_z half as large again: reported, BMAG as the Twiss give it, and the
    # bunch length breathes between sqrt(BMAG -+ sqrt(BMAG^2 - 1)) times the
    # matched one (0.82 and 1.22 here)
    bad = dataclasses.replace(beam, beta_z=1.5 * beam.beta_z, beta_x=1.3 * beam.beta_x)
    res2 = _run(booster, bad)
    bm = res2.meta["start_mismatch"]
    assert bm[0] == pytest.approx(_bmag(bad.beta_x, bad.alpha_x, beam.beta_x, beam.alpha_x),
                                  rel=1e-3)
    assert bm[1] == pytest.approx(1.0, abs=2e-4)
    assert bm[2] == pytest.approx(_bmag(bad.beta_z, bad.alpha_z, beam.beta_z, beam.alpha_z),
                                  rel=1e-3)
    note = next(n for n in res2.meta["notes"] if "mismatched" in n)
    assert note.startswith("the beam at the start is mismatched to the ring's optics: BMAG x 1.0")
    assert ", y 1.0000, z 1.08" in note and note.endswith("(1 when matched)")
    s0, s2 = float(np.mean(sz)), res2.sigma[:, 4]
    w = math.sqrt(bm[2] ** 2 - 1.0)
    assert float(s2.min()) == pytest.approx(s0 * math.sqrt(bm[2] - w), rel=0.02)
    assert float(s2.max()) == pytest.approx(s0 * math.sqrt(bm[2] + w), rel=0.02)


def test_a_ramps_matched_beam_is_that_of_its_first_turn(booster):
    cfg = RingConfig(ramp=RAMP)
    mb, beam = _matched(booster, _beam(), cfg)
    assert mb["ramp"] and mb["kinetic_mev"] == pytest.approx(4059.111, abs=1e-3)
    assert any(n.startswith("the ramp starts at 4059.1111 MeV") for n in mb["notes"])
    # the RF of turn 0: harmonic 84 at the start's revolution frequency
    g = 1.0 + mb["kinetic_mev"] / 938.27208816
    b = math.sqrt(1.0 - 1.0 / g ** 2)
    assert mb["rf_frequency_hz"] == pytest.approx(84 * b * C_LIGHT / REF["length_m"],
                                                  rel=1e-6)
    # the dispersion per MeV at the start's momentum; the Twiss are the ring's
    assert mb["beta_x"] == pytest.approx(REF["start"]["betx"], rel=1e-4)
    assert mb["disp_x"] == pytest.approx(1e3 * REF["start"]["dx_delta"]
                                         / (mb["beta0"] * mb["p0c_mev"]), rel=1e-3)
    assert 0.0 < mb["beta_z"] < 5.0                   # (a short bunch near transition)
    # the run of the Beam tab's beam made from it starts matched
    res = _run(booster, beam, n_turns=3, ramp=RAMP)
    assert res.meta["start_mismatch"] == pytest.approx([1.0, 1.0, 1.0], abs=2e-4)


def test_a_distorted_closed_orbit_is_the_matched_beams_centroid(booster):
    lat = copy.deepcopy(booster)
    brho = REF["pc_gev"] * 1e9 / C_LIGHT
    lat.elements.insert(30, Steerer(name="st", by_l=5e-4 * brho))       # 0.5 mrad
    mb, beam = _matched(lat, _beam())
    opt = RingSimulation(copy.deepcopy(lat), beam, RingConfig())._start()[3]
    co = np.asarray(opt.closed_orbit)
    assert abs(co[0]) > 2e-3                             # (millimetres)
    assert mb["centroid_x"] == pytest.approx(co[0] * 1e3, rel=1e-9)
    assert mb["centroid_dphi"] == pytest.approx(-360.0 * 44.7044096e6 * co[4]
                                                / (REF["beta0"] * C_LIGHT), rel=1e-6, abs=1e-12)
    # on the closed orbit the beam's centre stays there; off it, it oscillates
    # about it.  (The Beam tab's generator sets the second moments exactly and
    # leaves the sample mean free: the centre starts within the noise of the
    # mean, sigma_x/sqrt(n) = 0.13 mm here, of the closed orbit.)
    on = _run(lat, beam, n_turns=30)
    off = _run(lat, dataclasses.replace(beam, centroid_x=0.0, centroid_xp=0.0, centroid_y=0.0,
                                        centroid_yp=0.0, centroid_dphi=0.0, centroid_dw=0.0),
               n_turns=30)
    noise = math.sqrt(on.cov[0, 0, 0] / on.n_alive[0])
    assert abs(on.mean[0, 0] - co[0]) < 3.0 * noise
    assert np.max(np.abs(on.mean[:, 0] - co[0])) < 0.15 * abs(co[0])
    assert np.max(np.abs(off.mean[:, 0] - co[0])) > 0.7 * abs(co[0])


def test_without_rf_the_matched_beam_coasts(booster):
    lat = copy.deepcopy(booster)
    for e in lat.elements:
        if isinstance(e, RFGap):
            e.voltage = 0.0
    mb, beam = _matched(lat, _beam(n_particles=500))
    assert not mb["rf_on"] and mb["rf_frequency_hz"] is None and mb["qs"] is None
    assert mb["beta_z"] is None and mb["alpha_z"] is None
    assert mb["centroid_dphi"] is None and mb["centroid_dw"] is None
    # (the Beam tab's longitudinal settings stay: a bunch, which the ring lets
    # coast; the transverse modes only are judged)
    res = _run(lat, beam, n_turns=2)
    assert res.meta["start_mismatch"] == pytest.approx([1.0, 1.0], abs=2e-4)


def test_an_earlier_projects_matched_source_becomes_the_beam_tabs_beam():
    assert migrate_source({"source": "matched", "n_particles": 5000, "n_turns": 3}) == (
        {"source": "beam", "n_turns": 3}, True, 5000)
    # (no source then meant the matched one)
    assert migrate_source({"n_particles": 700}) == ({"source": "beam"}, True, 700)
    # the beam or an empty start stay; "n_particles" goes (the Beam tab's)
    assert migrate_source({"source": "beam", "n_particles": 9}) == ({"source": "beam"}, False,
                                                                     None)
    assert migrate_source({"source": "empty"}) == ({"source": "empty"}, False, None)
    # (settings without a start: an earlier version's, which started matched —
    # this version always saves the start)
    assert migrate_source({"n_turns": 20}) == ({"n_turns": 20, "source": "beam"}, True, None)
    assert migrate_source({}) == ({"source": "beam"}, True, None)
    assert migrate_source({"source": "matched", "n_particles": True})[2] is None
    cfg = {"source": "matched", "n_particles": 10}
    migrate_source(cfg)
    assert cfg == {"source": "matched", "n_particles": 10}        # (a copy)


def test_a_stop_request_ends_the_matched_beam(booster):
    from linac_gen.core.cancelled import OperationCancelled
    sim = RingSimulation(copy.deepcopy(booster), _beam(), RingConfig())
    with pytest.raises(OperationCancelled):
        sim.matched_beam(should_stop=lambda: True)
    # (the simulation is left as it was: a run afterwards is not stopped)
    assert sim._should_stop is None
    assert sim.matched_beam()["beta_x"] == pytest.approx(REF["start"]["betx"], rel=1e-4)


def test_settings_with_an_injection_and_no_start_are_an_empty_ring():
    # (one rule for the GUI, the command line and the assistant)
    inj = {"turns": 2, "particles_per_turn": 10}
    assert migrate_source({"injection": inj, "n_particles": 5}) == (
        {"injection": inj, "source": "empty"}, False, None)
    assert migrate_source({"injection": inj, "source": "matched"})[0]["source"] == "beam"
    assert migrate_source({"injection": inj, "source": "beam"}) == (
        {"injection": inj, "source": "beam"}, False, None)
