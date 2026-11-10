"""A window that holds a change of clock.

Its exit side is in degrees of another clock than its entrance: a scaling of one coordinate, which no eigen-analysis
can absorb.  Brought to one clock, the window's longitudinal tune is the cell's own, as with no change in it, and its
Twiss is that of the cell expressed on the entrance clock.

Every result is checked against its twin: the same cavities with the beam already on their clock, where nothing
changes clock.  The same bunch on another clock carries the same charge, so its current scales with the clock, and so
do its phase extent and β_z.  Clocks are switched up and down, by 2 and by 2.5 (a ratio not exact in binary).
"""
import warnings

import numpy as np
import pytest

from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.analysis.phase_advance import (
    channel_phase_advance,
    run_phase_probe,
    structure_phase_advance,
)
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Freq
from linac_gen.elements.ncells import NCells
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching.periodic import find_matched_period_sigma

CAV, HALF, LOW = 325.0, 162.5, 130.0
INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2,
            alpha_z=0.0, beta_z=1.0, emit_z=0.2)


def _quiet(f, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **kw)


def _quiet_run(cls, *a, **kw):
    """``cls(*a, **kw).run()`` with its warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(*a, **kw).run()


def _cell(i, f=CAV, phase=-30.0):
    return [NCells(f"NC{i}", mode=1, n_cells=4, beta_g=0.2, eot_v_per_m=1.0e6, theta_s_deg=phase,
                   aperture_mm=15.0, p_flag=2, frequency_mhz=f),
            Drift(f"Da{i}", 100.0), Quadrupole(f"QF{i}", length=100.0, gradient=12.0, aperture=30.0),
            Drift(f"Db{i}", 100.0), Quadrupole(f"QD{i}", length=100.0, gradient=-12.0, aperture=30.0),
            Drift(f"Dc{i}", 100.0)]


def _clocks(case, low):
    """(cavity clock, beam clock).  ``own``: the beam on the cavities' clock.  ``jump``: the beam on ``low``, so the
    first cavity switches the clock up; ``card``: the same with a FREQ card opening the window.  ``half``: cavities on
    ``low`` with the beam on it; ``down``: the same cavities with the beam on CAV, so the first one switches it down."""
    return {"own": (CAV, CAV), "jump": (CAV, low), "card": (CAV, low), "half": (low, low), "down": (low, CAV)}[case]


def _twin(case, low=HALF):
    """The case's twin, and the factor taking the twin's phase to the case's entrance clock."""
    return ("own", low / CAV) if case in ("jump", "card") else ("half", CAV / low)


def _line(case, w=19.0, low=HALF, n=3, phase=-30.0):
    """``n`` cells, the window being the first."""
    f_cav, f_beam = _clocks(case, low)
    lat = Lattice()
    els = [Freq("F", CAV)] if case == "card" else []
    for i in range(n):
        els += _cell(i, f_cav, phase)
    for e in els:
        lat.add(e)
    lat.step_config = StepConfig(100.0, 50.0)
    m = 6 + (case == "card")
    per = PeriodicStructure(start=0, end=m, inner_period_length=m, inner_slice_end=m, n_repeats=1,
                            label="c", source="fallback")
    return lat, ReferenceParticle(species=PROTON, w_kin=w, frequency=f_beam), per


def _structure(case, w=19.0, low=HALF):
    return _quiet(structure_phase_advance, *_line(case, w, low))


def _same_bunch(beam, k, current=0.0):
    """``beam`` at ``current`` on another clock, ``k`` times the first: the same bunch."""
    return dict(beam, beta_z=beam["beta_z"] * k, emit_z=beam["emit_z"] * k), current * k


# ---------------------------------------------------------------------------
# the structure tune
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w", [19.0, 21.0])
@pytest.mark.parametrize("case, low", [("jump", HALF), ("card", HALF), ("down", HALF), ("jump", LOW), ("down", LOW)])
def test_a_window_with_a_clock_change_has_the_cells_own_tune(case, low, w):
    twin, k = _twin(case, low)
    ref, got = _structure(twin, w, low), _structure(case, w, low)
    assert got["stable_z"] and got["mu_z_deg"] == pytest.approx(ref["mu_z_deg"], rel=1e-9)
    assert got["beta_z"] == pytest.approx(ref["beta_z"] * k, rel=1e-9)
    assert got["alpha_z"] == pytest.approx(ref["alpha_z"], rel=1e-9, abs=1e-12)
    assert got["mu_x_deg"] == pytest.approx(ref["mu_x_deg"], rel=1e-12)
    assert got["clock_mhz"] == _clocks(case, low)[1]


def test_the_window_still_reports_its_raw_map():
    got = _structure("jump")
    D = np.eye(6)
    D[4, 4] = HALF / CAV
    np.testing.assert_array_equal(got["M_period_one_clock"], D @ got["M_period"])


def test_with_no_change_of_clock_nothing_moves():
    got = _structure("own")
    assert got["M_period_one_clock"] is got["M_period"]


def test_a_window_behind_a_card_holds_no_change_of_clock():
    """A FREQ card before the window: the window runs on the card's clock alone."""
    lat, ref, _ = _line("card")
    behind = _quiet(structure_phase_advance, lat, ref, PeriodicStructure(
        start=7, end=13, inner_period_length=6, inner_slice_end=13, n_repeats=1, label="c", source="fallback"))
    lat, ref, _ = _line("own")
    own = _quiet(structure_phase_advance, lat, ref, PeriodicStructure(
        start=6, end=12, inner_period_length=6, inner_slice_end=12, n_repeats=1, label="c", source="fallback"))
    assert behind["M_period_one_clock"] is behind["M_period"] and behind["clock_mhz"] == CAV
    for key in ("mu_z_deg", "beta_z", "alpha_z", "mu_x_deg"):
        assert behind[key] == pytest.approx(own[key], rel=1e-12, abs=1e-15), key


def _gap_cell(f_gap, offset):
    from linac_gen.elements.rf_gap import RFGap
    lat = Lattice()
    for e in [RFGap("G", voltage=0.3, phase=-60.0, frequency=f_gap, frequency_offset=offset), Drift("Da", 100.0),
              Quadrupole("QF", length=100.0, gradient=12.0, aperture=30.0), Drift("Db", 100.0),
              Quadrupole("QD", length=100.0, gradient=-12.0, aperture=30.0), Drift("Dc", 100.0)]:
        lat.add(e)
    per = PeriodicStructure(start=0, end=6, inner_period_length=6, inner_slice_end=6, n_repeats=1,
                            label="c", source="fallback")
    return lat, per


def test_a_window_past_a_frequency_offset_is_taken_on_one_clock():
    """A gap whose oscillator runs 50 kHz off: past it the reference runs at the oscillator, so the window's exit is
    on 325.05 MHz while its entrance is on 325.  On one clock the tune is that of the same cell all on 325.05."""
    lat, per = _gap_cell(CAV, 0.05)
    off = _quiet(structure_phase_advance, lat, ReferenceParticle(species=PROTON, w_kin=19.0, frequency=CAV), per)
    lat, per = _gap_cell(CAV + 0.05, 0.0)
    on = _quiet(structure_phase_advance, lat, ReferenceParticle(species=PROTON, w_kin=19.0, frequency=CAV + 0.05),
                per)
    assert off["mu_z_deg"] == pytest.approx(on["mu_z_deg"], rel=1e-12)


# ---------------------------------------------------------------------------
# the channel tunes, cell by cell, bare and depressed by space charge
# ---------------------------------------------------------------------------
def _channel(lat, ref, beam, current, spans):
    per = PeriodicStructure(start=spans[0][0], end=spans[-1][1], inner_period_length=spans[0][1] - spans[0][0],
                            inner_slice_end=spans[0][1], n_repeats=len(spans), label="c", source="fallback",
                            repeat_spans=tuple(spans))
    ch = _quiet(channel_phase_advance, _quiet(run_phase_probe, lat, ref, dict(beam), current=current), per)
    return np.concatenate([ch[f"mu_{p}_{t}_deg"] for p in "xz" for t in ("bare", "dep")])


CELLS = ((0, 6), (6, 12), (12, 18))


@pytest.mark.parametrize("case, low, current", [("jump", HALF, 0.0), ("jump", HALF, 8.0), ("down", HALF, 0.0),
                                                ("down", HALF, 8.0), ("jump", LOW, 8.0)])
def test_the_channel_tune_of_a_cell_holding_the_change_is_its_own(case, low, current):
    """Cell by cell, the line whose first cavity switches the clock against its twin, bare and depressed."""
    twin, k = _twin(case, low)
    beam, cur = _same_bunch(INIT, k, current)
    got = _channel(*_line(case, low=low)[:2], beam, cur, CELLS)
    ref = _channel(*_line(twin, low=low)[:2], INIT, current, CELLS)
    np.testing.assert_allclose(got, ref, rtol=1e-9)
    if current:
        assert not np.allclose(ref[3:6], ref[9:12])              # space charge does depress the z tunes


def test_a_card_closing_a_cell_and_a_cavity_switching_back():
    """The beam on the cavities' clock, a card at the end of the first cell taking it to half, and the next cavity
    switching it back: every cell's channel tune is that of the line with no change of clock."""
    lat = Lattice()
    for e in _cell(0) + [Freq("F", HALF)] + _cell(1) + _cell(2):
        lat.add(e)
    lat.step_config = StepConfig(100.0, 50.0)
    ref = ReferenceParticle(species=PROTON, w_kin=19.0, frequency=CAV)
    got = _channel(lat, ref, INIT, 8.0, ((0, 7), (7, 13), (13, 19)))
    own = _channel(*_line("own")[:2], INIT, 8.0, CELLS)
    np.testing.assert_allclose(got, own, rtol=1e-9)


# ---------------------------------------------------------------------------
# the matched period beam, and the mismatch across the window
# ---------------------------------------------------------------------------
def _matched_beam(case, low=HALF):
    """The twin's matched beam at no current, with its phase on the case's entrance clock."""
    twin, k = _twin(case, low)
    s = _structure(twin, low=low)
    beam = {f"{k_}_{p}": s[f"{k_}_{p}"] for p in "xyz" for k_ in ("alpha", "beta")}
    return dict(beam, beta_z=beam["beta_z"] * k, emit_x=0.2, emit_y=0.2, emit_z=0.2)


@pytest.mark.parametrize("case, low", [("jump", HALF), ("down", HALF), ("jump", LOW)])
def test_the_matched_beam_of_such_a_window_is_its_cells(case, low):
    """Launched as the cell's matched beam, at no current, the beam is a fixed point of the period pass and the matched
    solve keeps it.  (A start far from it would rest on the solver's final least-squares polish, whose end point
    depends on round-off for this cell.)"""
    beam = _matched_beam(case, low)
    m = _quiet(find_matched_period_sigma, *_line(case, low=low), 1e-9, beam)
    S = m["sigma_entry"][4:6, 4:6]
    assert m["converged"]
    assert S[0, 0] / np.sqrt(np.linalg.det(S)) == pytest.approx(beam["beta_z"], rel=1e-6)
    assert -S[0, 1] / np.sqrt(np.linalg.det(S)) == pytest.approx(beam["alpha_z"], rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("case, low", [("jump", HALF), ("down", HALF), ("jump", LOW)])
def test_a_matched_window_across_the_switch_reads_as_matched(case, low):
    """The cell's matched beam through the window: its exit, on the entrance clock, has the entrance Twiss."""
    from linac_gen.analysis.phase_advance import beam_phase_advance
    from linac_gen.tracking.envelope import EnvelopeSolver
    lat, ref, per = _line(case, low=low)
    res = _quiet_run(EnvelopeSolver, lat, ref.copy(), _matched_beam(case, low), current=0.0)
    b = _quiet(beam_phase_advance, res, per)
    assert b["mismatch_bmag_z"] == pytest.approx(1.0, abs=1e-9)
    assert b["mismatch_bmag_x"] == pytest.approx(1.0, abs=1e-9)


def test_the_mismatch_of_a_continuous_beam_ignores_a_card():
    """A FREQ card does not rescale a continuous beam's phase, so a window closed by one reads as without it."""
    from linac_gen.analysis.phase_advance import beam_phase_advance
    from linac_gen.core.beam import Beam
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    from linac_gen.distributions.gaussian import generate_gaussian
    from linac_gen.tracking.tracker import Tracker
    bmag = {}
    for card in (False, True):
        lat = Lattice()
        els = [Drift("Da", 100.0), Quadrupole("QF", length=100.0, gradient=4.0, aperture=60.0), Drift("Db", 100.0),
               Quadrupole("QD", length=100.0, gradient=-4.0, aperture=60.0), Drift("Dc", 100.0)]
        for e in els + ([Freq("F", CAV)] if card else []):
            lat.add(e)
        m = len(lat.elements)
        per = PeriodicStructure(start=0, end=m, inner_period_length=m, inner_slice_end=m, n_repeats=1, label="c",
                                source="fallback")
        rng = np.random.default_rng(1)
        p = generate_gaussian(2000, 0.2, 0.0, 1.0, 0.2, 0.0, 1.0, 0.1, 0.0, 1.0, seed=3)
        p[:, 4] = rng.uniform(-180.0, 180.0, 2000)
        p[:, 5] = rng.normal(0.0, 1e-3, 2000)
        beam = Beam(ReferenceParticle(species=PROTON, w_kin=2.1, frequency=HALF), n_particles=2000, current=0.0)
        beam.particles[:] = p
        beam.continuous = True
        rec = _quiet_run(Tracker, lat, beam, recorder=DiagnosticRecorder())
        assert rec.continuous_at[-1] and rec.ref_frequency[-1] == (CAV if card else HALF)
        bmag[card] = _quiet(beam_phase_advance, rec, per)["mismatch_bmag_z"]
    assert bmag[True] == pytest.approx(bmag[False], rel=1e-12)


# ---------------------------------------------------------------------------
# the tune footprint
# ---------------------------------------------------------------------------
def _r2_cavity(i, f):
    """A cavity whose field grows off axis as 1 + 0.02 r²/mm², so a particle away from the axis gains a phase and energy
    offset.  The growth is of second order: the cell's linear map stays uncoupled.  Its phase is set where it bunches
    on either clock."""
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen.io.field_map_data import FieldChannel, FieldMapData
    from linac_gen.io.tracewin_geom import Channel
    L = 240.0
    x = np.linspace(-8.0, 8.0, 33)
    z = np.linspace(0.0, L, 61)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    fz = 3.0 * np.sin(np.pi * Z / L) * (1.0 + 0.02 * (X ** 2 + Y ** 2))
    fd = FieldMapData(z=z, frequency=f)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=np.zeros_like(fz), Fy=np.zeros_like(fz),
                                             Fz=fz)
    return FieldMap3D(name=f"FM{i}", length=L, field_data=fd, scale=1.0, phase=-30.0 if f == CAV else -150.0,
                      frequency=f, n_steps=60, p_flag=0, aperture=8.0)


def _r2_line(case):
    f_cav, f_beam = _clocks(case, HALF)
    lat = Lattice()
    for i in range(2):
        for e in [_r2_cavity(i, f_cav)] + _cell(i)[1:]:
            lat.add(e)
    lat.step_config = StepConfig(100.0, 50.0)
    per = PeriodicStructure(start=0, end=6, inner_period_length=6, inner_slice_end=6, n_repeats=1, label="c",
                            source="fallback")
    return lat, ReferenceParticle(species=PROTON, w_kin=19.0, frequency=f_beam), per


@pytest.mark.parametrize("case", ["jump", "down"])
@pytest.mark.parametrize("current, emit_x", [(0.0, 0.2), (1.0, 0.8)])
def test_the_footprint_of_a_cell_that_switches_the_clock_is_the_cells(case, current, emit_x):
    """Every turn starts at the cell's entrance, so the particles must too, on its clock: a phase offset left on the
    exit clock would double (or halve) every turn.  The cavity gives the off-axis particles one.  At 1 mA the beam is
    kept far from round (ε_x = 4 ε_y): near round, the Gaussian field's elliptic formula loses its precision far from
    the axis on one side, and round-off then grows into large kicks."""
    from linac_gen.analysis.footprint import tune_footprint
    twin, k = _twin(case)
    s = _quiet(structure_phase_advance, *_r2_line(twin))
    beam = {f"{k_}_{p}": s[f"{k_}_{p}"] for p in "xyz" for k_ in ("alpha", "beta")}
    beam = dict(beam, emit_x=emit_x, emit_y=0.2, emit_z=0.2)
    q = {}
    for line_case, (b, cur) in ((twin, (beam, current)), (case, _same_bunch(beam, k, current))):
        r = _quiet(tune_footprint, *_r2_line(line_case), b, cur, n_turns=16, n_particles=4, seed=0)
        q[line_case] = np.concatenate([r["qx"][r["ax_sigma"] > 0], r["qy"][r["ay_sigma"] > 0],
                                       [r["qx_core"], r["qy_core"]]])
    np.testing.assert_allclose(q[case], q[twin], rtol=1e-9)


# ---------------------------------------------------------------------------
# what is measured at the period entries
# ---------------------------------------------------------------------------
N_CELLS = 8


def _period_entries(case, make_seed):
    """An envelope run through the case's line and its twin, each of ``N_CELLS`` cells at a -90 degree phase, which
    gain almost no energy (13 keV per cell), so that they nearly repeat; ``make_seed(floquet)`` gives the twin's
    starting Σ and centroid, converted for the case."""
    pytest.importorskip("linac_gen.analysis.envelope_floquet",
                        reason="Floquet analyzer not present in this distribution")
    from linac_gen.analysis.envelope_floquet import envelope_floquet
    from linac_gen.tracking.envelope import EnvelopeSolver
    twin, k = _twin(case)
    spans = tuple((6 * i, 6 * i + 6) for i in range(N_CELLS))
    per = PeriodicStructure(start=0, end=6 * N_CELLS, inner_period_length=6, inner_slice_end=6, n_repeats=N_CELLS,
                            label="c", source="fallback", repeat_spans=spans)
    lines = {c: _line(c, n=N_CELLS, phase=-90.0) for c in (twin, case)}
    s = _quiet(structure_phase_advance, lines[twin][0], lines[twin][1], per)
    beam = dict({f"{k_}_{p}": s[f"{k_}_{p}"] for p in "xyz" for k_ in ("alpha", "beta")}, emit_x=0.2, emit_y=0.2,
                emit_z=0.2)
    K = np.eye(6)
    K[4, 4] = k
    runs = {}
    for c, b in ((twin, beam), (case, _same_bunch(beam, k)[0])):
        lat, ref, _ = lines[c]
        f = _quiet(envelope_floquet, lat, ref, per, 0.0, b)
        if c == twin:
            seed, centroid = make_seed(f)
        S, cen = (seed, centroid) if c == twin else (K @ seed @ K.T, K @ centroid)
        res = _quiet_run(EnvelopeSolver, lat, ref.copy(), dict(b, centroid=list(cen)), current=0.0, initial_sigma=S)
        runs[c] = (res, per, f)
    assert float(runs[case][0].ref_frequency[0]) == _clocks(case, HALF)[1]
    return runs[twin], runs[case]


@pytest.mark.parametrize("case", ["jump", "down"])
def test_mode_growth_reads_every_period_entry_on_one_clock(case):
    pytest.importorskip("linac_gen.analysis.envelope_floquet",
                        reason="Floquet analyzer not present in this distribution")
    pytest.importorskip("linac_gen.analysis.mode_growth",
                        reason="mode-growth analysis not present in this distribution")
    from linac_gen.analysis.envelope_floquet import seed_mode_sigma
    from linac_gen.analysis.mode_growth import measure_mode_growth
    twin, got = _period_entries(case, lambda f: (seed_mode_sigma(f, 0, 1e-3), np.zeros(6)))
    amp = [np.sort(np.abs(measure_mode_growth(*r)["amplitudes"]), axis=0) for r in (twin, got)]
    np.testing.assert_allclose(amp[1], amp[0], rtol=1e-6, atol=1e-9 * amp[0].max())


@pytest.mark.parametrize("case", ["jump", "down"])
def test_the_centroid_spectrum_reads_every_period_entry_on_one_clock(case):
    pytest.importorskip("linac_gen.analysis.mode_growth",
                        reason="mode-growth analysis not present in this distribution")
    from linac_gen.analysis.mode_growth import centroid_spectrum
    twin, got = _period_entries(case, lambda f: (f["matched"]["sigma_entry"], np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])))
    ref, c = (centroid_spectrum(r[0], r[1], plane="z") for r in (twin, got))
    assert ref["reason"] is None and c["reason"] is None
    assert c["tune_per_period"] == pytest.approx(ref["tune_per_period"], rel=1e-9)
    assert c["growth_measured"] == pytest.approx(ref["growth_measured"], rel=1e-9)
