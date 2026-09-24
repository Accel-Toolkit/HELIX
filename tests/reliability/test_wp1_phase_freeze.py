"""WP1 — frozen-phase pins.  Thin gap: the entrance-clock pin reproduces
the independent kinematic chain of truth.json in both brackets; None slot
bit-identical; both frequency branches; transport, reset and deepcopy.
Field map: harvest == the lazily calibrated psi of the NCells fixture.
Container children are refused."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.cli import common
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.reliability import PinSet, collect_pins, harvest_pins

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"


@pytest.fixture(scope="module")
def demo():
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, conv = common.load_input(str(DEMO / "reliability_demo.lgproj"))
    truth = json.loads((DEMO / "truth.json").read_text(encoding="utf-8"))
    return lat, cfg, conv, truth


def _pinned(lat, pins: PinSet):
    work = copy.deepcopy(lat)
    for sel, val in pins.overrides:
        common.apply_element_override(work, sel, val)
    return work


def _w_env(lat, cfg):
    return common.result_summary(common.run_envelope_sim(copy.deepcopy(lat), cfg))["ref_w_kin"]


# --------------------------------------------------------------------------
# harvest on the demo deck
# --------------------------------------------------------------------------
def test_harvest_leaves_the_live_lattice_alone_and_pins_equal_truth_clock(demo):
    lat, cfg, _, truth = demo
    pins = harvest_pins(lat, cfg)
    assert all(e.sync_phase_pin is None for e in lat.elements if isinstance(e, RFGap))
    assert set(pins.kinds.values()) == {"clock"} and len(pins) == 4
    assert pins.skipped == ()
    for name, want in truth["nominal"]["clock_at_gap_entrance_deg"].items():
        assert pins.pins[name] == pytest.approx(want, rel=1e-12)
    assert pins.overrides[0][0].startswith("@") and pins.overrides[0][0].endswith(".sync_phase_pin")


def test_pinned_nominal_is_bit_identical_to_unpinned(demo):
    """frozen == rephased on the design lattice (the pin IS the design pass)."""
    lat, cfg, conv, _ = demo
    pins = harvest_pins(lat, cfg)
    work = _pinned(lat, pins)
    a = common.result_summary(common.run_envelope_sim(copy.deepcopy(lat), cfg))
    b = common.result_summary(common.run_envelope_sim(work, cfg))
    assert a == b
    # (a zero-thickness foil: its scattering draws are unseeded until WP6)
    cfg2 = copy.deepcopy(cfg); cfg2.n_particles = 200
    sc = common.make_sc_config(cfg2, conv, {}); st = common.make_step_config(conv, {})
    la = copy.deepcopy(lat); common.apply_element_override(la, "FOIL_001.thickness_ug_cm2", 0.0)
    mp_pins = harvest_pins(lat, cfg2, mode="mp", sc_config=sc, step_config=st)
    lb = _pinned(la, mp_pins)
    rec_a, _ = common.run_mp_sim(la, cfg2, sc, st, seed=7)
    rec_b, _ = common.run_mp_sim(lb, cfg2, sc, st, seed=7)
    assert rec_a.ref_w_kin[-1] == rec_b.ref_w_kin[-1]
    assert np.array_equal(np.asarray(rec_a.sigma_x), np.asarray(rec_b.sigma_x))
    # the two engines round the clock differently: envelope pins on an MP
    # run agree only to ~1e-13 deg (why a bracket is pinned by its own engine)
    for name in pins.pins:
        assert mp_pins.pins[name] == pytest.approx(pins.pins[name], rel=1e-12)
    with pytest.raises(ValueError, match="mode"):
        harvest_pins(lat, cfg2, mode="matrix")


@pytest.mark.parametrize("gap", ["GAP_001", "GAP_002", "GAP_003", "GAP_004"])
def test_frozen_and_rephased_brackets_match_the_kinematic_truth(demo, gap):
    lat, cfg, conv, truth = demo
    pins = harvest_pins(lat, cfg)
    w0 = _w_env(lat, cfg)
    tol = truth["tolerances"]["energy_rel"]
    re = copy.deepcopy(lat); common.apply_element_override(re, f"{gap}.voltage", 0.0)
    fr = _pinned(lat, pins); common.apply_element_override(fr, f"{gap}.voltage", 0.0)
    d_re = w0 - _w_env(re, cfg)
    d_fr = w0 - _w_env(fr, cfg)
    want_re = truth["cavity_off_rephased"][gap]["d_energy_mev"]
    want_fr = truth["cavity_off_frozen"][gap]["d_energy_mev"]
    assert abs(d_re - want_re) <= tol * want_re
    assert abs(d_fr - want_fr) <= tol * want_fr
    if gap != "GAP_004":
        assert abs(d_fr - d_re) > 1e-3        # the brackets differ upstream of the last gap
    # the multiparticle reference sees the same frozen chain
    cfg2 = copy.deepcopy(cfg); cfg2.n_particles = 50
    sc = common.make_sc_config(cfg2, conv, {}); st = common.make_step_config(conv, {})
    rec, _ = common.run_mp_sim(fr, cfg2, sc, st, seed=1)
    d_mp = truth["nominal"]["w_end_mp_ref_mev"] - rec.ref_w_kin[-1]
    assert abs(d_mp - want_fr) <= tol * want_fr


# --------------------------------------------------------------------------
# the element itself
# --------------------------------------------------------------------------
def _ref(f=162.5, phi=0.0):
    return ReferenceParticle(species=PROTON, w_kin=20.0, frequency=f, phi_s=phi)


def test_operating_phase_wrap_and_frequency_jump_refused():
    g = RFGap("g", voltage=1.0, phase=-20.0, frequency=162.5)
    assert g._operating_phase(_ref(phi=123.0)) == -20.0            # unpinned: untouched
    g.sync_phase_pin = 100.0
    assert g._operating_phase(_ref(phi=110.0)) == pytest.approx(-10.0)       # f_gap == f_ref
    assert g._operating_phase(_ref(phi=100.0 + 350.0)) == pytest.approx(-30.0)   # wrap to -10
    # no frequency scaling inside the gap: a per-element frequency jump is
    # refused by the harvest instead (the engines switch the reference
    # frequency at different points of such a jump)
    g2 = RFGap("g2", voltage=1.0, phase=-20.0, frequency=325.0)
    g2.sync_phase_pin = 100.0
    assert g2._operating_phase(_ref(f=162.5, phi=110.0)) == pytest.approx(-10.0)
    lat = Lattice(); lat.add(Drift("d", length=100.0)); lat.add(g2)
    from linac_gen.core.config import BeamConfig
    with pytest.raises(NotImplementedError, match="frequency jump"):
        harvest_pins(lat, BeamConfig(species="proton", energy=20.0, frequency=162.5, current=0.0, n_particles=16))
    from linac_gen.elements.lattice_commands import Freq
    ok = Lattice(); ok.add(Drift("d", length=100.0)); ok.add(Freq("f", frequency_mhz=325.0))
    ok.add(RFGap("g3", voltage=1.0, phase=-20.0, frequency=325.0))
    ps = harvest_pins(ok, BeamConfig(species="proton", energy=20.0, frequency=162.5, current=0.0, n_particles=16))
    assert ps.kinds == {"g3": "clock"}


def test_unpinned_kick_is_bit_identical_and_records_the_entry_clock():
    def run(pin):
        g = RFGap("g", voltage=1.0, phase=-20.0, frequency=162.5)
        g.sync_phase_pin = pin
        ref = _ref(phi=57.25)
        beam = Beam(ref=ref, n_particles=8, current=0.0)
        beam.particles[:] = np.linspace(-1.0, 1.0, 48).reshape(8, 6)
        g.apply_kick(beam)
        return beam.particles.copy(), ref.w_kin, g._entry_phi_s
    pa, wa, ea = run(None)
    pb, wb, eb = run(57.25)          # pinned at exactly the entry clock: slip 0
    assert np.array_equal(pa, pb) and wa == wb
    assert ea == 57.25 and eb == 57.25
    pc, wc, _ = run(37.25)           # 20 deg late -> fires on crest
    assert wc == pytest.approx(20.0 + 1.0, rel=1e-12)
    assert wa == pytest.approx(20.0 + np.cos(np.radians(-20.0)), rel=1e-12)


def test_matrix_and_advance_ref_follow_the_pin():
    g = RFGap("g", voltage=1.0, phase=-20.0, frequency=162.5)
    ref = _ref(phi=10.0)
    m_free = g.kick_matrix(ref)
    g.sync_phase_pin = 10.0
    assert np.array_equal(g.kick_matrix(ref), m_free)
    g.sync_phase_pin = -10.0             # 20 deg late -> on crest: no longitudinal shear
    m = g.kick_matrix(ref)
    assert m[5, 4] == pytest.approx(0.0, abs=1e-15)
    r = _ref(phi=10.0); g.advance_ref(r)
    assert r.w_kin == pytest.approx(21.0, rel=1e-12) and g._entry_phi_s == 10.0


def test_inverse_kick_uses_the_entrance_clock():
    g = RFGap("g", voltage=1.0, phase=-20.0, frequency=162.5)
    g.sync_phase_pin = 30.0
    ref = _ref(phi=57.0)
    entry = ref.copy()
    beam = Beam(ref=ref, n_particles=6, current=0.0)
    beam.particles[:] = np.linspace(-0.5, 0.5, 36).reshape(6, 6)
    before = beam.particles.copy()
    g.apply_kick(beam)
    g.inverse_kick(beam, entry)
    assert np.allclose(beam.particles, before, rtol=0, atol=1e-12)


def test_pin_transport_reset_and_deepcopy():
    lat = Lattice()
    lat.add(Drift("d", length=100.0))
    lat.add(RFGap("g", voltage=1.0, phase=-20.0, frequency=162.5))
    common.apply_element_override(lat, "@2.sync_phase_pin", "123.5")   # None slot -> float
    g = lat.elements[1]
    assert g.sync_phase_pin == 123.5 and isinstance(g.sync_phase_pin, float)
    if hasattr(g, "reset_run_state"):
        g.reset_run_state()
    assert g.sync_phase_pin == 123.5
    assert copy.deepcopy(lat).elements[1].sync_phase_pin == 123.5
    assert RFGap("h", 1.0, -20.0, 162.5).sync_phase_pin is None       # class default untouched


# --------------------------------------------------------------------------
# field-map branch and the container refusal
# --------------------------------------------------------------------------
def test_field_map_branch_pins_the_lazily_calibrated_psi():
    from linac_gen.core.config import BeamConfig
    from linac_gen.elements.ncells import NCells
    F, BG = 804.96, 0.456630316
    gamma = 1.0 / np.sqrt(1.0 - BG * BG)
    w = (gamma - 1.0) * H_MINUS.mass
    def nc():
        return NCells("c", mode=1, n_cells=8, beta_g=BG, eot_v_per_m=6.8e6,
                      theta_s_deg=-28.0, aperture_mm=15.0, sync_phase=True,
                      frequency_mhz=F)
    lat = Lattice(); lat.add(Drift("d", length=50.0)); lat.add(nc())
    cfg = BeamConfig(species="H-", energy=w, frequency=F, current=0.0, n_particles=16)
    pins = harvest_pins(lat, cfg)
    assert pins.kinds == {"c": "psi"} and pins.overrides[0][0] == "@2.sync_phase_pin"
    assert lat.elements[1].sync_phase_pin is None
    # the value the lazy calibration writes on a tracked pass
    ref = ReferenceParticle(species=H_MINUS, w_kin=w, frequency=F)
    beam = Beam(ref=ref, n_particles=4, current=0.0); beam.particles[:] = 0.0
    c = nc(); c.reset_run_state(); c.track_rk4(beam, c.length)
    assert pins.pins["c"] == pytest.approx(c._sync_offset_deg, rel=1e-12)


def test_collect_pins_refuses_container_children():
    from linac_gen.elements.ncells import NCells
    child = NCells("kid", mode=1, n_cells=2, beta_g=0.4, eot_v_per_m=1e6,
                   theta_s_deg=-30.0, aperture_mm=15.0, sync_phase=True,
                   frequency_mhz=325.0)
    child._sync_offset_deg = 1.0
    lat = SimpleNamespace(elements=[Drift("d", length=10.0),
                                    SimpleNamespace(name="S", children=[(0.0, child)])])
    with pytest.raises(NotImplementedError, match="SuperposedFieldMap children"):
        collect_pins(lat)
    lat2 = SimpleNamespace(elements=[Drift("d", length=10.0),
                                     SimpleNamespace(name="S", children=[(0.0, Drift("x", 1.0))])])
    assert collect_pins(lat2) == PinSet()


def test_collect_pins_reports_uncalibrated_cavities_as_skipped():
    from linac_gen.elements.ncells import NCells
    c = NCells("c", mode=1, n_cells=2, beta_g=0.4, eot_v_per_m=1e6,
               theta_s_deg=-30.0, aperture_mm=15.0, sync_phase=True,
               frequency_mhz=325.0)
    g = RFGap("g", 1.0, -20.0, 325.0)
    lat = SimpleNamespace(elements=[c, g])
    ps = collect_pins(lat)                    # no design pass ran
    assert ps.skipped == ("c", "g") and len(ps) == 0
