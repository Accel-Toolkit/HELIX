"""A cavity on another clock than the beam's, with no FREQ card in front of it, changes the clock itself.

The change happens at one instant.  The reference phase is ω·t in degrees of the clock, so it is multiplied by
f_new / f_old, as a FREQ card does, and the particles' Δφ scale the same way.  A cavity that changes the clock
itself must therefore give what the same cavity gives behind a FREQ card: energy, reference phase, position, clock,
map, and a bunched beam's particles.  An LLRF frequency offset is not a change of clock: such a cavity beats against
the clock from its own entrance.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.constants import C_LIGHT
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.lattice_commands import Freq
from linac_gen.elements.ncells import NCells
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking import tracker as T
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, reference_at

CLOCK, F = 162.5, 325.0
INITIAL = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
           "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2}


def _ref(w=20.0, clock=CLOCK):
    return ReferenceParticle(species=PROTON, w_kin=w, frequency=clock)


def _ncells(p_flag=1, beta_g=0.2, sync=False, f=F):
    return NCells("NC", mode=1, n_cells=6, beta_g=beta_g, eot_v_per_m=2e6, theta_s_deg=-30.0,
                  aperture_mm=30.0, p_flag=p_flag, frequency_mhz=f, sync_phase=sync)


def _gap():
    return RFGap("GAP", voltage=0.5, phase=-30.0, frequency=F, aperture=30.0)


def _fieldmap():
    z = np.linspace(0.0, 120.0, 121)
    fd = FieldMapData(z=z, Ez=2.0 * np.sin(np.pi * z / 120.0), symmetry="1d")
    return FieldMap("FM", length=120.0, field_data=fd, phase=-25.0, frequency=F, aperture=30.0,
                    n_steps=125, p_flag=0)


def _fieldmap_3d():
    rng = np.random.default_rng(0)
    L = 240.0
    x = np.linspace(-20.0, 20.0, 9)
    z = np.linspace(0.0, L, 61)
    shape = (x.size, x.size, z.size)
    fz = 3.0 * np.sin(np.pi * z / L)[None, None, :] + 0.3 * rng.normal(size=shape)
    fd = FieldMapData(z=z, frequency=F)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=x, y=x, z=z, Fx=0.3 * rng.normal(size=shape), Fy=0.3 * rng.normal(size=shape), Fz=fz)
    return FieldMap3D(name="FM3", length=L, field_data=fd, scale=1.0, phase=-30.0, frequency=F,
                      n_steps=200, p_flag=0)


def _superposed():
    z = np.linspace(0.0, 300.0, 151)
    sol = FieldMap("SOL", length=300.0, aperture=30.0, n_steps=150,
                   field_data=FieldMapData(z=z, Bz=0.35 * (0.5 - 0.5 * np.cos(2 * np.pi * z / 300.0)),
                                           symmetry="1d"))
    cav = _fieldmap()
    cav.name = "CAV"
    return SuperposedFieldMap("SUP", [(0.0, sol), (90.0, cav)])


def _lines(make):
    """The cavity behind two drifts, alone and behind a FREQ card at its own frequency."""
    out = []
    for with_card in (False, True):
        lat = Lattice()
        lat.add(Drift("D1", 200.0))
        lat.add(Drift("D2", 150.0))
        if with_card:
            lat.add(Freq("FREQ", F))
        lat.add(make())
        lat.add(Drift("D3", 100.0))
        lat.step_config = StepConfig(100.0, 50.0)
        out.append(lat)
    return out


def _state(r):
    return np.array([r.w_kin, r.phi_s, r.s, r.frequency])


def _walk(lat):
    return _state(reference_at(lat, _ref(), len(lat.elements))), compute_transfer_matrix(lat, _ref())


def _envelope(lat, current=0.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = EnvelopeSolver(lat, _ref(), dict(INITIAL), current=current)
        res = solver.run()
    return _state(solver._ref), np.array([res.sigma_x[-1], res.sigma_y[-1], res.sigma_phi[-1], res.sigma_w[-1]])


def _tracker(lat):
    beam = Beam(ref=_ref(), n_particles=40, current=0.0)
    rng = np.random.default_rng(3)
    beam.particles[:] = rng.normal(size=(40, 6)) * [0.5, 0.5, 0.5, 0.5, 2.0, 1e-4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T.Tracker(lat, beam).run()
    return _state(beam.ref), beam.particles.copy()


ENGINES = {"matrix walk": _walk, "envelope": _envelope, "envelope with space charge":
           lambda lat: _envelope(lat, current=1.0), "tracker": _tracker}
NCELLS = [pytest.param(dict(p_flag=p, beta_g=bg, sync=s), id=f"P{p}-bg{bg}{'-sync' if s else ''}")
          for p in (0, 1, 2) for bg in (0.2, 0.0, -0.2) for s in (False, True)]


def _same(own, card):
    s_own, x_own = own
    s_card, x_card = card
    np.testing.assert_allclose(s_own, s_card, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x_own, x_card, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("kw", NCELLS)
def test_an_ncells_on_another_clock_switches_it_like_a_freq_card(engine, kw):
    own, card = _lines(lambda: _ncells(**kw))
    _same(ENGINES[engine](own), ENGINES[engine](card))


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("make", [_gap, _fieldmap, _fieldmap_3d, _superposed],
                         ids=["RFGap", "FieldMap P0", "FieldMap3D P0", "SuperposedFieldMap"])
def test_any_rf_element_on_another_clock_switches_it_like_a_freq_card(engine, make):
    own, card = _lines(make)
    _same(ENGINES[engine](own), ENGINES[engine](card))


def test_the_entrance_phase_is_the_arrival_time_on_the_cavity_clock():
    own, _ = _lines(_ncells)
    r = reference_at(own, _ref(), 2)               # after the two drifts, still on the beam's clock
    nc = own.elements[2]
    nc.reset_run_state()
    nc.advance_ref(r)
    lam = C_LIGHT / (F * 1e6) * 1000.0             # mm on the cavity's clock
    assert nc._phi_s_at_entrance == pytest.approx(360.0 * 350.0 / (_ref().beta * lam), rel=1e-12)


def test_a_line_of_cavities_slightly_off_the_beam_clock_needs_no_card():
    """Cavities a few hundred kHz above the beam's frequency, as decks often have: with or without a FREQ card in
    front of the first, the line is the same."""
    beam_clock, cav = 804.6, 804.96
    lines = []
    for with_card in (False, True):
        lat = Lattice()
        lat.add(Drift("D0", 300.0))
        if with_card:
            lat.add(Freq("FREQ", cav))
        for k in range(5):
            lat.add(NCells(f"C{k}", mode=1, n_cells=5, beta_g=0.6, eot_v_per_m=15e6, theta_s_deg=-20.0,
                           aperture_mm=40.0, p_flag=1, frequency_mhz=cav))
            lat.add(Drift(f"D{k + 1}", 400.0))
        lines.append(lat)
    end = [reference_at(lat, _ref(w=200.0, clock=beam_clock), len(lat.elements)) for lat in lines]
    assert end[0].w_kin == pytest.approx(end[1].w_kin, abs=1e-9)
    assert end[0].phi_s == pytest.approx(end[1].phi_s, rel=1e-12)


@pytest.mark.parametrize("make", [_ncells, _fieldmap, _fieldmap_3d, _superposed],
                         ids=["NCells", "FieldMap", "FieldMap3D", "SuperposedFieldMap"])
def test_putting_a_reference_on_a_cavitys_clock_moves_its_phase(make):
    """What a surrogate's training does with its template: the same instant on the cavity's clock."""
    r = _ref()
    r.phi_s = 100.0
    make()._propagate_frequency_to_ref(r)
    assert r.frequency == F and r.phi_s == 100.0 * F / CLOCK


def _entrance():
    """The reference at the cavity's entrance in the line above, still on the beam's clock, and the same instant on
    the cavity's."""
    own, _ = _lines(_ncells)
    r = reference_at(own, _ref(), 2)
    q = r.copy()
    q.switch_clock(F)
    return r, q


@pytest.mark.parametrize("beta_g", [0.0, -0.2])
def test_a_cavity_builds_its_running_geometry_on_its_own_clock(beta_g):
    """Cells that follow the running velocity are laid out by a forward pass, which reads the phase on the cavity's
    clock whichever clock the reference it is handed is on."""
    r, q = _entrance()
    a = _ncells(beta_g=beta_g)._build_gaps_running(r)
    b = _ncells(beta_g=beta_g)._build_gaps_running(q)
    assert [(g.z_center_mm, g.voltage_mv) for g in a] == [(g.z_center_mm, g.voltage_mv) for g in b]


def test_advancing_the_reference_over_a_cavity_starts_on_its_clock():
    """The envelope's bundles walk a cavity piece by piece: handed the reference on the beam's clock, the first
    piece switches it as the entrance does."""
    r, q = _entrance()
    a, b = _ncells(), _ncells()
    for cav, ref in ((a, r), (b, q)):
        cav.reset_run_state()
        cav.advance_ref_over(ref, 0.0, cav.length / 2)
        cav.advance_ref_over(ref, cav.length / 2, cav.length)
    np.testing.assert_array_equal(_state(r), _state(q))



@pytest.mark.parametrize("cavity_mhz", [F, 2 * F], ids=["behind the card", "on a clock of its own"])
def test_a_frequency_offset_does_not_depend_on_where_the_line_starts(cavity_mhz):
    """An LLRF frequency offset beats against the clock from the cavity's own entrance.  Lines that differ only by
    a whole number of RF periods of drift in front give the same energy, as they do with no offset: had the offset
    moved the reference phase like a change of clock, the extra 3 x 360 deg would shift the cavity by 3*360*df/f."""
    def line(extra_mm):
        cav = _fieldmap()
        cav.frequency = cavity_mhz
        cav.frequency_offset = 0.01                       # 10 kHz
        lat = Lattice()
        lat.add(Drift("D1", 200.0 + extra_mm))
        lat.add(Freq("FREQ", F))
        lat.add(cav)
        lat.add(Drift("D2", 100.0))
        nxt = _fieldmap()                                  # back on the clock: the phase it sees must not move
        nxt.name = "FM2"
        lat.add(nxt)
        lat.add(Drift("D3", 100.0))
        return lat
    lam = C_LIGHT / (F * 1e6) * 1000.0                     # mm on the card's clock
    periods = 3 * _ref().beta * lam                        # three RF periods of drift on that clock
    for engine in ("matrix walk", "envelope", "tracker"):
        w = [ENGINES[engine](line(x))[0][0] for x in (0.0, periods)]
        assert w[0] == pytest.approx(w[1], abs=1e-12), engine


@pytest.mark.parametrize("make", [_fieldmap, _fieldmap_3d, _superposed],
                         ids=["FieldMap", "FieldMap3D", "SuperposedFieldMap"])
def test_a_frequency_offset_switches_the_reference_but_not_its_phase(make):
    """The reference runs at the cavity's oscillator; its phase moves only for a change of the design clock."""
    el = make()
    el.frequency_offset = 0.01
    behind = _ref(clock=F)
    behind.phi_s = 100.0
    el._propagate_frequency_to_ref(behind)
    assert behind.frequency == F + 0.01 and behind.phi_s == 100.0
    own = _ref()
    own.phi_s = 100.0
    el._propagate_frequency_to_ref(own)
    assert own.frequency == F + 0.01 and own.phi_s == 100.0 * F / CLOCK


@pytest.mark.parametrize("path", ["advance_ref", "apply_kick"])
def test_a_gap_with_a_frequency_offset_keeps_the_phase(path):
    gap = _gap()
    gap.frequency_offset = 0.01
    r = _ref(clock=F)
    r.phi_s = 100.0
    if path == "advance_ref":
        gap.advance_ref(r)
    else:
        beam = Beam(ref=r, n_particles=4, current=0.0)
        gap.apply_kick(beam)
        r = beam.ref
    assert r.frequency == F + 0.01 and r.phi_s == 100.0


def test_a_boundary_state_keeps_the_clock_its_phase_is_counted_on():
    """The backtracker's design table, past a cavity with a frequency offset: the reference it rebuilds runs at the
    cavity's oscillator with its phase still counted on the clock, as the forward walk's was."""
    from linac_gen.tracking.backtrack import build_replay_table
    cav = _fieldmap()
    cav.frequency_offset = 0.01
    lat = Lattice()
    lat.add(Drift("D1", 200.0))
    lat.add(Freq("FREQ", F))
    lat.add(cav)
    lat.add(Drift("D2", 100.0))
    row = build_replay_table(lat, _ref(), end=len(lat.elements) - 1)[3]
    r = row.make_ref(PROTON)
    assert r.frequency == F + 0.01 and r._clock == F


def test_the_clock_survives_a_copy_and_a_reload():
    """Past an offset cavity the reference runs at its oscillator while its phase is counted on the clock; a copy
    and a reloaded state must say so too, or the next cavity would move the phase on its way back to the clock."""
    from linac_gen.tracking.tracker import load_reference_state
    r = _ref(clock=F)
    r.switch_clock(F, F + 0.01)
    c = r.copy()
    loaded = _ref()
    load_reference_state(loaded, r)
    for got in (r, c, loaded):
        assert got.frequency == F + 0.01 and got._clock == F
