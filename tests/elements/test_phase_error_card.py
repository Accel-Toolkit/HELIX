"""SET_BEAM_PHASE_ERROR shifts the RF phase of the cavities after it, as the TraceWin manual defines it.

TraceWin manual (CEA, D. Uriot, 2021-12-31, p. 108): "The beam phase is not affected by this command"; with Dp = 0 it
shifts all the following cavity RF phases by the difference between the reference design and the beam phase observed
at the command, to "uncouple the RF phase of some part of the machine and start a linac part with a new RF phase";
with Dp different from 0 the RF shift is increased by Dp, or by a random value within +-Dp when RandomFlag is set.
HELIX read the card into a running offset that no cavity read, so it did nothing in any mode.

The shift is on the phase a cavity's beam sees.  A Dp = 0 card ends the shifts before it, and the cavities that see
the arrival (a field map with p_flag 0, an NCELLS with P = 1, a gap held at a pinned clock) take the error study's
correction for the seed: the error-free arrival minus the seed's, along one reference walk.  HELIX's relative-phase
cavities re-phase to the reference every run and need none.  In a plain run there is no correction.
"""
import copy
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.lattice_commands import REFERENCE_COMMANDS, Freq, SetBeamPhaseError
from linac_gen.elements.marker import Marker
from linac_gen.elements.ncells import NCells
from linac_gen.elements.rf_gap import RFGap
from linac_gen.io.field_map_data import FieldMapData
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, reference_at
from linac_gen.tracking.tracker import Tracker

F = 352.21
INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
            emit_z=0.2)


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=F)


def _field_map(p_flag, phase_offset=0.0, name="FM"):
    z = np.linspace(0.0, 120.0, 121)
    fd = FieldMapData(z=z, Ez=2.0 * np.sin(np.pi * z / 120.0), symmetry="1d")
    return FieldMap(name, length=120.0, field_data=fd, phase=-25.0, frequency=F, aperture=30.0, n_steps=125,
                    p_flag=p_flag, phase_offset=phase_offset)


def _ncells(p_flag, sync=False, phase_offset=0.0):
    c = NCells("NC", mode=0, n_cells=3, beta_g=0.08, eot_v_per_m=1.5e6, theta_s_deg=-30.0, frequency_mhz=F,
               p_flag=p_flag, sync_phase=sync, aperture_mm=20.0)
    c.phase_offset = phase_offset
    return c


def _fm3d(p_flag, off):
    from tests.tracking.test_envelope_reference_matches_tracker import _fm3d as make
    fm = make(name="FM3", freq=F, p_flag=p_flag, phase_offset=off)
    if p_flag == 1 and off:                       # as for the 1-D map: psi held at the fit without the error
        bare = _line(Marker("PE"), make(name="FM3", freq=F, p_flag=1))
        reference_at(bare, _ref(), len(bare.elements))
        fm.sync_phase_pin = bare.elements[3]._sync_offset_deg
    return fm


def _sync_field_map(off):
    """A SET_SYNC_PHASE field map: HELIX fits its offset psi with its own phase error in it, where the RF shift moves
    the RF after the fit; so the twin holds psi at the fit without the error."""
    fm = _field_map(1, off)
    if off:
        bare = _line(Marker("PE"), _field_map(1))
        reference_at(bare, _ref(), len(bare.elements))
        fm.sync_phase_pin = bare.elements[3]._sync_offset_deg
    return fm


def _superposed(p_flag, off):
    """A SUPERPOSE cluster of a solenoid and a cavity; the cluster's own phase error moves its cavity.  (With p_flag 1
    the cluster fits the cavity's psi with the other children at their phases, so both the shift and the cluster
    error come after the fit.)"""
    from linac_gen.elements.superposed_field_map import SuperposedFieldMap
    z = np.linspace(0.0, 120.0, 121)
    sol = FieldMap("SOL", length=120.0, aperture=30.0, n_steps=125,
                   field_data=FieldMapData(z=z, Bz=0.2 * np.sin(np.pi * z / 120.0), symmetry="1d"))
    return SuperposedFieldMap("SUP", [(0.0, sol), (0.0, _field_map(p_flag, name="CAV"))], phase_offset=off)


# each cavity, and the phase_offset that moves the phase its beam sees by +Dp
CAVITIES = {
    "gap": lambda off: RFGap("G", voltage=0.6, phase=-30.0, frequency=F, phase_offset=off),
    "field map p0": lambda off: _field_map(0, off),
    "field map p1": _sync_field_map,
    "3-D field map p0": lambda off: _fm3d(0, off),
    "3-D field map p1": lambda off: _fm3d(1, off),
    "ncells P0": lambda off: _ncells(0, phase_offset=off),
    "ncells P1": lambda off: _ncells(1, phase_offset=-off),          # its beam sees phi - theta
    "ncells P2": lambda off: _ncells(2, phase_offset=off),
    "ncells sync": lambda off: _ncells(0, sync=True, phase_offset=off),
    "superposed p0": lambda off: _superposed(0, off),
    "superposed p1": lambda off: _superposed(1, off),
}


def _line(card, cavity):
    lat = Lattice()
    for e in (Drift("D1", 80.0), card, Drift("D2", 60.0), cavity, Drift("D3", 80.0)):
        lat.add(e)
    return lat


def _runs(lat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beam = Beam(_ref(), n_particles=40, current=0.0)
        beam.particles[:] = np.random.default_rng(2).normal(size=(40, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        Tracker(copy.deepcopy(lat), beam).run()
        env = EnvelopeSolver(copy.deepcopy(lat), _ref(), dict(INIT), current=0.0).run()
        end = reference_at(copy.deepcopy(lat), _ref(), len(lat.elements))
        return {"tracker": np.concatenate([beam.particles.ravel(), [beam.ref.w_kin, beam.ref.phi_s]]),
                "envelope": np.concatenate([np.asarray(env.sigma_matrix[-1]).ravel(), [env.ref_w_kin[-1]]]),
                "matrix": compute_transfer_matrix(copy.deepcopy(lat), _ref()),
                "reference": np.array([end.w_kin, end.phi_s])}


@pytest.mark.parametrize("cavity", CAVITIES)
def test_a_shift_moves_the_phase_the_next_cavity_sees(cavity):
    """Dp = 7 degrees: the cavity after the card runs as if its own phase error moved its beam 7 degrees, in every mode;
    the beam's own phase is not touched by the card."""
    shifted = _runs(_line(SetBeamPhaseError("PE", dphi_deg=7.0), CAVITIES[cavity](0.0)))
    twin = _runs(_line(Marker("PE"), CAVITIES[cavity](7.0)))
    for mode in shifted:
        np.testing.assert_allclose(shifted[mode], twin[mode], rtol=1e-11, atol=1e-11, err_msg=mode)
    assert not np.allclose(shifted["reference"], _runs(_line(Marker("PE"), CAVITIES[cavity](0.0)))["reference"])


@pytest.mark.parametrize("cavity", CAVITIES)
def test_no_shift_in_a_plain_run_changes_nothing(cavity):
    """Dp = 0 cancels what static errors did to the beam's phase; a run without an error study has none."""
    a = _runs(_line(SetBeamPhaseError("PE", dphi_deg=0.0), CAVITIES[cavity](0.0)))
    b = _runs(_line(Marker("PE"), CAVITIES[cavity](0.0)))
    for mode in a:
        np.testing.assert_array_equal(a[mode], b[mode], err_msg=mode)


def test_shifts_add_up_and_follow_a_change_of_clock():
    """Two cards add; behind a FREQ card the shift is the same time in degrees of the new clock."""
    ref = _ref()
    from linac_gen.core.track_state import TrackState
    ts = TrackState(ref=ref)
    SetBeamPhaseError("A", dphi_deg=5.0).apply_command(ts)
    SetBeamPhaseError("B", dphi_deg=2.5).apply_command(ts)
    assert ref.rf_shift == pytest.approx(7.5)
    ref.rf_rephase = 3.0
    Freq("F2", frequency_mhz=2.0 * F).apply_command(ts)
    assert ref.rf_shift == pytest.approx(15.0) and ref.rf_rephase == pytest.approx(6.0)
    assert (ref.copy().rf_shift, ref.copy().rf_rephase) == (ref.rf_shift, ref.rf_rephase)
    assert isinstance(SetBeamPhaseError("C", dphi_deg=1.0), REFERENCE_COMMANDS)


def test_a_random_shift_outside_an_error_study_is_none_with_a_warning():
    lat_card = _line(SetBeamPhaseError("PE", dphi_deg=7.0, random_flag=1), CAVITIES["field map p0"](0.0))
    with pytest.warns(UserWarning, match="RandomFlag"):
        EnvelopeSolver(copy.deepcopy(lat_card), _ref(), dict(INIT), current=0.0).run()
    a, b = _runs(lat_card), _runs(_line(Marker("PE"), CAVITIES["field map p0"](0.0)))
    for mode in a:
        np.testing.assert_array_equal(a[mode], b[mode], err_msg=mode)


def test_the_matrix_cache_tells_shifts_apart():
    from linac_gen.tracking.matrix_tracking import get_element_matrix
    fm = _field_map(0)
    ref, shifted, rephased = _ref(), _ref(), _ref()
    shifted.rf_shift, rephased.rf_rephase = 7.0, 7.0
    cache = {}
    a = get_element_matrix(fm, ref, cache=cache)
    b = get_element_matrix(fm, shifted, cache=cache)
    c = get_element_matrix(fm, rephased, cache=cache)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


# --------------------------------------------------------------- an error study
def _study_line(card, cavity=None):
    """An upstream gap whose amplitude error makes the beam late, then the card right in front of a cavity that sees
    the arrival (by default a field map with p_flag 0).  (The card cancels the phase error at its position; a beam off
    energy goes on slipping over any drift after it.)"""
    lat = Lattice()
    for e in (RFGap("GAP_1", voltage=0.6, phase=-30.0, frequency=F), Drift("D1", 150.0), Drift("D2", 100.0), card,
              cavity if cavity is not None else _field_map(0, name="FM_2"), Drift("D3", 50.0)):
        lat.add(e)
    return lat


def _ncells_named(p_flag):
    c = _ncells(p_flag)
    c.name = "FM_2"
    return c


def _cfg():
    return BeamConfig(species="proton", energy=3.0, frequency=F, n_particles=50, current=0.0, distribution="waterbag",
                      emit_nx=0.2, alpha_x=0.0, beta_x=1.0, emit_ny=0.2, alpha_y=0.0, beta_y=1.0, emit_z=0.2,
                      alpha_z=0.0, beta_z=1.0)


def _seen_at_field_map(lat):
    """The phase the field map's beam sees at its entrance (deg)."""
    i = next(k for k, e in enumerate(lat.elements) if e.name == "FM_2")
    ref = reference_at(lat, _ref(), i)
    fm = lat.elements[i]
    return float(np.rad2deg(fm._phi_sync_rad(ref, ref.phi_s)))


def test_no_shift_in_an_error_study_puts_the_next_cavity_back_on_its_phase():
    from linac_gen.errors.error_model import ErrorStudy
    nominal = _seen_at_field_map(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0)))
    seen = {}
    for card in (SetBeamPhaseError("PE", dphi_deg=0.0), Marker("PE")):
        study = ErrorStudy(_study_line(card), _cfg(), n_seeds=1)
        study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            errored = study._apply_errors(3)
        seen[type(card).__name__] = _seen_at_field_map(errored)
    assert abs(seen["Marker"] - nominal) > 0.1                       # the error moved the phase it sees
    assert seen["SetBeamPhaseError"] == pytest.approx(nominal, abs=1e-9)


def test_a_random_shift_is_drawn_per_seed_and_leaves_the_error_draws_alone():
    from linac_gen.errors.error_model import ErrorStudy

    def study(card):
        s = ErrorStudy(_study_line(card), _cfg(), n_seeds=3)
        s.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
        return s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_card = study(SetBeamPhaseError("PE", dphi_deg=7.0, random_flag=1))
        plain = study(Marker("PE"))
        draws = []
        for seed in (0, 1, 1):
            lat = with_card._apply_errors(seed)
            card = next(e for e in lat.elements if isinstance(e, SetBeamPhaseError))
            draws.append((card.random_drawn, card.random_draw_deg))
            gap = next(e for e in lat.elements if e.name == "GAP_1")
            assert gap.voltage_rel == next(e for e in plain._apply_errors(seed).elements
                                           if e.name == "GAP_1").voltage_rel
    assert all(d[0] == 1 and abs(d[1]) <= 7.0 for d in draws)
    assert draws[1] == draws[2] and draws[0] != draws[1]


def test_a_seed_runs_its_draw_in_every_mode():
    """The draw stamped on the seed's copy is the shift every mode then applies."""
    from linac_gen.errors.error_model import ErrorStudy
    s = ErrorStudy(_study_line(SetBeamPhaseError("PE", dphi_deg=7.0, random_flag=1)), _cfg(), n_seeds=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat = s._apply_errors(4)
    card = next(e for e in lat.elements if isinstance(e, SetBeamPhaseError))
    twin = _study_line(Marker("PE"))
    twin.elements[4].phase_offset = card.random_draw_deg
    a, b = _runs(lat), _runs(twin)
    for mode in a:
        np.testing.assert_allclose(a[mode], b[mode], rtol=1e-11, atol=1e-11, err_msg=mode)


@pytest.mark.parametrize("cavity", ["gap", "field map p0", "field map p1"])
def test_the_backtracker_closes_across_a_shift(cavity):
    """Forward through the card and the cavity it shifts, then back: the beam it started with.  (An NCELLS is left
    out: the backtrack through one does not close even without the card.)"""
    from linac_gen.tracking.backtrack import backtrack_distribution
    lat = _line(SetBeamPhaseError("PE", dphi_deg=7.0), CAVITIES[cavity](0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beam = Beam(_ref(), n_particles=20, current=0.0)
        beam.particles[:] = np.random.default_rng(5).normal(size=(20, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        start = beam.particles.copy()
        Tracker(copy.deepcopy(lat), beam).run()
        backtrack_distribution(copy.deepcopy(lat), beam, _ref())
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-7)


def test_a_pinned_gap_sees_the_correction_and_a_free_one_does_not():
    """A gap held at a pinned clock sees the arrival, so the Dp = 0 correction; one that re-phases does not."""
    ref = _ref()
    ref.rf_rephase = 5.0
    free = RFGap("G", voltage=0.6, phase=-30.0, frequency=F)
    pinned = RFGap("G", voltage=0.6, phase=-30.0, frequency=F)
    pinned.sync_phase_pin = ref.phi_s
    assert free._operating_phase(ref) == -30.0
    assert pinned._operating_phase(ref) == pytest.approx(-25.0)


@pytest.mark.parametrize("make, sign", [(lambda: _field_map(0, name="FM_2"), 1.0),
                                        (lambda: _ncells_named(1), -1.0)], ids=["field map p0", "ncells P1"])
def test_a_seed_with_no_shift_runs_its_correction_in_every_mode(make, sign):
    """The correction a Dp = 0 card takes in an error study, carried on copies of the reference by every mode, equals
    the cavity after it given that phase error itself."""
    from linac_gen.errors.error_model import ErrorStudy
    study = ErrorStudy(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0), cavity=make()), _cfg(), n_seeds=1)
    study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errored = study._apply_errors(3)
    card = errored.elements[3]
    nominal = reference_at(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0), cavity=make()), _ref(), 3)
    correction = card.rephase_deg
    assert correction == pytest.approx(nominal.phi_s - reference_at(errored, _ref(), 3).phi_s, abs=1e-9)
    assert abs(correction) > 0.1
    twin = copy.deepcopy(errored)
    twin.elements[3] = Marker("PE")
    twin.elements[4].phase_offset = twin.elements[4].phase_offset + sign * correction
    a, b = _runs(errored), _runs(twin)
    for mode in a:
        np.testing.assert_allclose(a[mode], b[mode], rtol=1e-10, atol=1e-10, err_msg=mode)


def test_the_backtracker_closes_across_a_correction():
    from linac_gen.errors.error_model import ErrorStudy
    from linac_gen.tracking.backtrack import backtrack_distribution
    study = ErrorStudy(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0)), _cfg(), n_seeds=1)
    study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat = study._apply_errors(3)
        beam = Beam(_ref(), n_particles=20, current=0.0)
        beam.particles[:] = np.random.default_rng(5).normal(size=(20, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        start = beam.particles.copy()
        Tracker(copy.deepcopy(lat), beam).run()
        backtrack_distribution(copy.deepcopy(lat), beam, _ref())
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-7)


def test_a_negative_random_shift_draws_within_its_size():
    from linac_gen.errors.error_model import ErrorStudy
    s = ErrorStudy(_study_line(SetBeamPhaseError("PE", dphi_deg=-7.0, random_flag=1)), _cfg(), n_seeds=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = [next(e for e in s._apply_errors(k).elements if isinstance(e, SetBeamPhaseError)).random_draw_deg
                 for k in range(8)]
    assert all(abs(d) <= 7.0 for d in draws) and min(draws) < 0.0 < max(draws)


def _two_card_line(first, second):
    lat = Lattice()
    for e in (RFGap("GAP_1", voltage=0.6, phase=-30.0, frequency=F), Drift("D1", 100.0), first, Drift("D2", 100.0),
              second, _field_map(0, name="FM_2"), Drift("D3", 50.0)):
        lat.add(e)
    return lat


def test_each_random_card_draws_from_a_stream_of_its_own():
    """Two cards of one seed draw apart, and a study on another stream (pulse jitter's) apart from both."""
    from linac_gen.errors.error_model import ErrorStudy
    lat = _two_card_line(SetBeamPhaseError("PA", dphi_deg=7.0, random_flag=1),
                         SetBeamPhaseError("PB", dphi_deg=7.0, random_flag=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = ErrorStudy(lat, _cfg(), n_seeds=1)._apply_errors(2)
        other = ErrorStudy(lat, _cfg(), n_seeds=1, rng_stream=(3, 2))._apply_errors(2)
    a, b = (e.random_draw_deg for e in own.elements if isinstance(e, SetBeamPhaseError))
    c = next(e.random_draw_deg for e in other.elements if isinstance(e, SetBeamPhaseError))
    assert len({a, b, c}) == 3


_STUDY_CARDS = {"Dp 0": lambda: SetBeamPhaseError("PE", dphi_deg=0.0),
                "random": lambda: SetBeamPhaseError("PE", dphi_deg=7.0, random_flag=1)}


@pytest.mark.parametrize("card", _STUDY_CARDS)
def test_a_frozen_seed_keeps_its_stamps_under_pulse_jitter(card):
    """Pulse jitter runs a study on a seed's errored copy, which has no error-free pass of its own: its card keeps the
    seed's design phase or draw, so a pulse without jitter is the frozen seed."""
    from linac_gen.cli.common import run_envelope_sim
    from linac_gen.errors.ensemble import run_jitter
    from linac_gen.errors.error_model import ErrorStudy
    study = ErrorStudy(_study_line(_STUDY_CARDS[card]()), _cfg(), n_seeds=1)
    study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frozen = study._apply_errors(3)
        pulse = run_jitter(frozen, _cfg(), lambda s: None, 1, static_seed=3)[0]
        assert pulse["sigx_end_mm"] == float(run_envelope_sim(frozen, _cfg()).sigma_x[-1])


@pytest.mark.parametrize("card", _STUDY_CARDS)
def test_a_seeds_stamps_travel_with_its_overrides(card):
    """A reliability campaign sends a seed to the scan pool as overrides of the design lattice: the stamps go too."""
    from linac_gen.cli.common import apply_element_override
    from linac_gen.errors.error_model import ErrorStudy
    study = ErrorStudy(_study_line(_STUDY_CARDS[card]()), _cfg(), n_seeds=1)
    study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draw = study.draw_overrides(3)
        sent = copy.deepcopy(study.lattice)
        for selector, value in draw.overrides:
            apply_element_override(sent, selector, value)
        a, b = _runs(sent), _runs(draw.lattice)
    assert any(s.startswith("@4.") for s, _ in draw.overrides) and not draw.untransportable
    for mode in a:
        np.testing.assert_array_equal(a[mode], b[mode], err_msg=mode)


def test_the_error_free_pass_leaves_the_studied_lattice_alone():
    """The design phase is taken on a copy: the studied lattice's random card, which that pass crosses, still warns in
    a plain run afterwards."""
    from linac_gen.errors.error_model import ErrorStudy
    lat = _two_card_line(SetBeamPhaseError("PR", dphi_deg=3.0, random_flag=1), SetBeamPhaseError("PE", dphi_deg=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ErrorStudy(lat, _cfg(), n_seeds=1)._apply_errors(0)
    with pytest.warns(UserWarning, match="RandomFlag"):
        EnvelopeSolver(lat, _ref(), dict(INIT), current=0.0).run()


def test_loading_a_reference_state_takes_its_shifts():
    """The envelope and the backtracker load the tracker's reference states through a field map: the whole state."""
    from linac_gen.tracking.tracker import load_reference_state
    src, dst = _ref(), _ref()
    src.rf_shift, src.rf_rephase = 7.0, -3.0
    load_reference_state(dst, src)
    assert (dst.rf_shift, dst.rf_rephase) == (7.0, -3.0)


# ------------------------------------------------------- Dp = 0: a new part of the machine
@pytest.mark.parametrize("cavity", ["gap", "field map p0", "ncells P1"])
def test_no_shift_ends_the_shifts_before_it(cavity):
    """A plain run: Dp = 7, then Dp = 0, then the cavity, which fires as with neither card (bit for bit)."""
    def line(first, second):
        lat = Lattice()
        for e in (Drift("D0", 40.0), first, Drift("D1", 40.0), second, Drift("D2", 60.0), CAVITIES[cavity](0.0),
                  Drift("D3", 80.0)):
            lat.add(e)
        return lat
    a = _runs(line(SetBeamPhaseError("PA", dphi_deg=7.0), SetBeamPhaseError("PE", dphi_deg=0.0)))
    b = _runs(line(Marker("PA"), Marker("PE")))
    for mode in a:
        np.testing.assert_array_equal(a[mode], b[mode], err_msg=mode)


def test_each_part_after_a_new_start_fires_at_its_own_draw():
    """TraceWin's pattern, Dp = 0 then a random Dp, before each of two parts: each part's gap fires at its own draw,
    not at the running sum of the draws (free gaps, which need no correction)."""
    from linac_gen.errors.error_model import ErrorStudy

    def line(*parts):
        lat = Lattice()
        for e in (Drift("D0", 40.0), *parts[0], RFGap("G1", voltage=0.6, phase=-30.0, frequency=F),
                  Drift("D1", 80.0), *parts[1], RFGap("G2", voltage=0.6, phase=-30.0, frequency=F),
                  Drift("D2", 80.0)):
            lat.add(e)
        return lat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seed = ErrorStudy(line(*[(SetBeamPhaseError(f"P0_{k}", dphi_deg=0.0),
                                  SetBeamPhaseError(f"PR_{k}", dphi_deg=7.0, random_flag=1)) for k in (1, 2)]),
                          _cfg(), n_seeds=1)._apply_errors(5)
    d1, d2 = (e.random_draw_deg for e in seed.elements if isinstance(e, SetBeamPhaseError) and e.random_flag)
    twin = line(*[(Marker(f"P0_{k}"), Marker(f"PR_{k}")) for k in (1, 2)])
    twin.elements[3].phase_offset, twin.elements[7].phase_offset = d1, d2
    a, b = _runs(seed), _runs(twin)
    for mode in a:
        np.testing.assert_allclose(a[mode], b[mode], rtol=1e-11, atol=1e-11, err_msg=mode)


def test_an_error_free_seed_is_the_design_in_every_mode():
    """The correction is the error-free arrival minus the seed's along one walk: a seed without errors gets exactly
    none, whatever each engine's own walk gives."""
    from linac_gen.errors.error_model import ErrorStudy
    lat = _study_line(SetBeamPhaseError("PE", dphi_deg=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seed = ErrorStudy(lat, _cfg(), n_seeds=1)._apply_errors(1)
    assert (seed.elements[3].rephased, seed.elements[3].rephase_deg) == (1, 0.0)
    a, b = _runs(seed), _runs(lat)
    for mode in a:
        np.testing.assert_array_equal(a[mode], b[mode], err_msg=mode)


def test_an_input_energy_error_is_cancelled():
    """An energy error at the input moves the reference, so the seed's correction, walked from the seed's own
    entrance, cancels it at the cavity after the card (as the manual has it for input-beam errors)."""
    from linac_gen.cli.common import build_ref
    from linac_gen.errors.error_model import ErrorStudy

    def seen(card):
        study = ErrorStudy(_study_line(card), _cfg(), n_seeds=1)
        study.add_beam_error("energy_rel", sigma=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lat, cfg = study._apply_errors(4), study._apply_beam_errors(4)
        i = next(k for k, e in enumerate(lat.elements) if e.name == "FM_2")
        ref = reference_at(lat, build_ref(cfg), i)
        return float(np.rad2deg(lat.elements[i]._phi_sync_rad(ref, ref.phi_s)))
    nominal = _seen_at_field_map(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0)))
    assert abs(seen(Marker("PE")) - nominal) > 1.0
    assert seen(SetBeamPhaseError("PE", dphi_deg=0.0)) == pytest.approx(nominal, abs=1e-9)


def test_a_dst_beam_is_corrected_from_where_its_beam_starts(tmp_path):
    """A .dst source's beam starts at the file's energy (the file wins over the config, and over an energy error on
    it): the correction walks start there too, so the card corrects the lattice's errors and adds none of its own."""
    from linac_gen.distributions.factory import create_beam
    from linac_gen.errors.error_model import ErrorStudy
    from linac_gen.io.tracewin_dst import write_dst
    dst = tmp_path / "beam.dst"
    write_dst(str(dst), np.random.default_rng(0).normal(size=(50, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002],
              current_mA=0.0, frequency_MHz=F, mass_MeV=PROTON.mass, w_kin_ref=3.05)
    cfg = BeamConfig(species="proton", energy=3.0, frequency=F, n_particles=50, current=0.0, source="file",
                     distribution_file=str(dst))

    def seen(lat, ref):
        i = next(k for k, e in enumerate(lat.elements) if e.name == "FM_2")
        r = reference_at(lat, ref, i)
        return float(np.rad2deg(lat.elements[i]._phi_sync_rad(r, r.phi_s)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = create_beam(cfg).ref
        nominal = seen(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0)), start)
        study = ErrorStudy(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0)), cfg, n_seeds=1)
        study.add_beam_error("energy_rel", sigma=0.01)
        assert study._apply_errors(4).elements[3].rephase_deg == 0.0       # the beam ignores the energy error
        study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
        lat = study._apply_errors(4)
    assert abs(start.w_kin - cfg.energy) > 0.04 and abs(lat.elements[3].rephase_deg) > 0.1     # the file's energy
    assert seen(lat, create_beam(study._apply_beam_errors(4)).ref) == pytest.approx(nominal, abs=1e-9)


def test_pulse_jitter_on_a_lattice_as_it_stands_is_not_corrected():
    """A jitter study on a lattice no error study stamped: its cards are fixed once (no correction, one draw), so the
    pulses' errors stay uncorrected (dynamic), as with no card at all."""
    from linac_gen.errors.ensemble import run_jitter

    def pulses(card):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_jitter(_study_line(card), _cfg(),
                              lambda s: s.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05), 3,
                              static_seed=2)
    with_card, without = pulses(SetBeamPhaseError("PE", dphi_deg=0.0)), pulses(Marker("PE"))
    assert [r["sigx_end_mm"] for r in with_card] == [r["sigx_end_mm"] for r in without]
    assert len({r["sigx_end_mm"] for r in without}) == 3                  # the pulses differ


def test_a_frozen_lattice_keeps_one_draw_for_every_pulse():
    from linac_gen.errors.error_model import ErrorStudy, freeze_phase_error_cards
    lat = _study_line(SetBeamPhaseError("PR", dphi_deg=7.0, random_flag=1))
    freeze_phase_error_cards(lat, 3)
    drawn = lat.elements[3].random_draw_deg
    assert lat.elements[3].random_drawn == 1 and 0.0 < abs(drawn) <= 7.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = ErrorStudy(lat, _cfg(), n_seeds=0, rng_stream=(3, 3))
        assert {study._apply_errors(p).elements[3].random_draw_deg for p in range(3)} == {drawn}
        assert ErrorStudy(_study_line(SetBeamPhaseError("PR", dphi_deg=7.0, random_flag=1)), _cfg(),
                          n_seeds=1)._apply_errors(3).elements[3].random_draw_deg == drawn   # a study's seed 3


def test_the_correction_walk_leaves_the_seed_copy_as_it_was():
    """The walk that finds the seed's arrival runs on a copy of it: the seed's elements carry no walk state (which a
    campaign's transport would read as draws)."""
    from linac_gen.errors.error_model import ErrorStudy

    def copy_of(card):
        line = _study_line(card, cavity=_ncells_named(1))
        study = ErrorStudy(line, _cfg(), n_seeds=1)
        study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return study._apply_errors(3)

    def scalars(e):
        return {k: v for k, v in vars(e).items() if isinstance(v, (bool, int, float, str, type(None)))}
    a, b = copy_of(SetBeamPhaseError("PE", dphi_deg=0.0)), copy_of(Marker("PE"))
    assert a.elements[3].rephased == 1
    for k in (0, 1, 2, 4, 5):
        assert scalars(a.elements[k]) == scalars(b.elements[k]), a.elements[k].name


def test_a_fault_on_an_error_free_seed_is_the_fault_alone():
    """A reliability campaign puts faults on seeds' copies.  A seed's correction is for its own errors, so a fault on
    an error-free seed is the fault on the design: neither corrects it."""
    from linac_gen.errors.error_model import ErrorStudy
    lat = _study_line(SetBeamPhaseError("PE", dphi_deg=0.0))
    nominal = _seen_at_field_map(lat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seed = ErrorStudy(lat, _cfg(), n_seeds=1)._apply_errors(1)
    for machine in (seed, lat):
        machine.elements[0].voltage_rel = -0.1
    assert _seen_at_field_map(seed) == _seen_at_field_map(lat)
    assert abs(_seen_at_field_map(lat) - nominal) > 0.1


def test_a_study_without_a_beam_config_corrects_from_the_reference_it_is_given():
    """The Floquet error ensemble's study has no beam config (the card made it raise): it takes the caller's
    reference; a study with neither corrects nothing and says so."""
    from linac_gen.errors.error_model import ErrorStudy

    def seed(cfg, **kw):
        study = ErrorStudy(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0)), cfg, n_seeds=1, **kw)
        study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
        return study._apply_errors(3).elements[3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        given, from_cfg = seed(None, entrance_ref=_ref()), seed(_cfg())
    assert given.rephase_deg == from_cfg.rephase_deg and abs(given.rephase_deg) > 0.1
    with pytest.warns(UserWarning, match="no beam config"):
        assert seed(None).rephased == 0


def test_the_floquet_error_ensemble_runs_with_a_card():
    pytest.importorskip("linac_gen.analysis.floquet_uq",
                        reason="Floquet error ensemble not present in this distribution")
    from linac_gen.analysis.floquet_uq import floquet_error_ensemble
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    lat.add(SetBeamPhaseError("PE", dphi_deg=0.0))
    for _ in range(6):
        for e in (Quadrupole("QF", 50.0, gradient=40.0, aperture=20.0), Drift("D1", 100.0),
                  Quadrupole("QD", 50.0, gradient=-40.0, aperture=20.0), Drift("D2", 100.0)):
            lat.add(e)
    period = next(p for p in detect_periods(lat) if p.source != "fallback")
    init = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.3, alpha_y=0.0, beta_y=1.0, emit_y=0.3, alpha_z=0.0, beta_z=2.0,
                emit_z=0.3, continuous=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = floquet_error_ensemble(lat, _ref(), period, 0.0, init, n_seeds=2,
                                     errors=[{"pattern": "Q*", "parameter": "gradient_rel", "sigma": 0.01}])
    assert out["reason"] is None and out["n_seeds"] == 2
    assert not [w for w in caught if "no beam config" in str(w.message)]     # its study took the reference


_ON_2F = {
    "gap": lambda off: RFGap("G", voltage=0.6, phase=-30.0, frequency=2 * F, phase_offset=off),
    "field map p0": lambda off: FieldMap("FM", length=120.0, phase=-25.0, frequency=2 * F, aperture=30.0, n_steps=125,
                                         p_flag=0, phase_offset=off,
                                         field_data=FieldMapData(z=np.linspace(0.0, 120.0, 121), symmetry="1d",
                                                                 Ez=2.0 * np.sin(np.pi * np.linspace(0.0, 120.0, 121)
                                                                                 / 120.0))),
}


@pytest.mark.parametrize("cavity", _ON_2F)
def test_a_cavity_with_a_clock_of_its_own_takes_the_shift_on_that_clock(cavity):
    """Its own jump is a FREQ card in front of it: Dp = 7 at F is 14 degrees at 2F, in every mode (a gap read the
    shift before its jump in three modes and after it in the envelope)."""
    shifted = _runs(_line(SetBeamPhaseError("PE", dphi_deg=7.0), _ON_2F[cavity](0.0)))
    twin = _runs(_line(Marker("PE"), _ON_2F[cavity](14.0)))
    for mode in shifted:
        np.testing.assert_allclose(shifted[mode], twin[mode], rtol=1e-11, atol=1e-11, err_msg=mode)


def test_the_madx_export_writes_a_gap_at_the_phase_it_fires_at(tmp_path):
    from linac_gen.io.madx_writer import write_madx

    def cavity(path, card, phase):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_madx(_line(card, RFGap("G", voltage=0.6, phase=phase, frequency=F)), path, _ref())
        return next(ln for ln in path.read_text(encoding="utf-8").splitlines() if "RFCAVITY" in ln)
    assert cavity(tmp_path / "a.madx", SetBeamPhaseError("PE", dphi_deg=7.0), -30.0) \
        == cavity(tmp_path / "b.madx", Marker("PE"), -23.0)


def test_a_seeds_log_holds_its_draw_and_its_correction():
    from linac_gen.errors.error_model import ErrorStudy
    study = ErrorStudy(_two_card_line(SetBeamPhaseError("PR", dphi_deg=3.0, random_flag=1),
                                      SetBeamPhaseError("PE", dphi_deg=0.0)), _cfg(), n_seeds=1)
    study.add_error(pattern="GAP_1", parameter="voltage_rel", sigma=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draw = study.draw_overrides(2)
    logged = {(r["element"], r["parameter"]): r["value"] for r in draw.applied}
    cards = {e.name: e for e in draw.lattice.elements if isinstance(e, SetBeamPhaseError)}
    assert logged[("PR", "random_draw_deg")] == cards["PR"].random_draw_deg
    assert logged[("PE", "rephase_deg")] == cards["PE"].rephase_deg
    assert ("GAP_1", "voltage_rel") in logged


def test_the_assistant_does_not_set_or_scan_a_stamp(tmp_path):
    from linac_gen.assist.tools import TOOLS, WorkContext
    deck = tmp_path / "line.dat"
    deck.write_text("DRIFT 80 20\n")
    ctx = WorkContext(calc_dir=str(tmp_path))
    ctx.set_lattice(_study_line(SetBeamPhaseError("PE", dphi_deg=0.0)), str(deck))
    ctx.set_beam_config(_cfg())
    for stamp in SetBeamPhaseError.STUDY_STAMPS:
        r = TOOLS["set_element_param"].fn(ctx, element_name="PE", param=stamp, value=1.0)
        assert r["status"] == "refused" and "error study" in r["data"]["message"]
        r = TOOLS["parameter_scan"].fn(ctx, element="PE", attribute=stamp, start=0.0, stop=1.0)
        assert r["status"] == "refused" and "error study" in r["data"]["message"]
    assert TOOLS["set_element_param"].fn(ctx, element_name="PE", param="dphi_deg", value=2.0)["status"] == "ok"


_LOADED = {   # a beam-loaded cavity, and the phase its beam sees behind Dp = 7 at F
    "gap": (lambda: RFGap("G", voltage=0.6, phase=-30.0, frequency=F), -23.0),
    "gap at 2F": (lambda: RFGap("G", voltage=0.6, phase=-30.0, frequency=2 * F), -16.0),
    "ncells sync at 2F": (lambda: NCells("G_NC", mode=0, n_cells=3, beta_g=0.08, eot_v_per_m=1.5e6, theta_s_deg=-30.0,
                                         frequency_mhz=2 * F, p_flag=0, sync_phase=True, aperture_mm=20.0), -16.0),
}


@pytest.mark.parametrize("cavity", _LOADED)
def test_beam_loading_takes_the_phase_the_cavity_fires_at(tmp_path, cavity):
    """The beam-loading design pass derives a cavity's voltage from its gain and its synchronous phase: with the card
    that is theta plus the shift on the cavity's clock, as with its own phase error (a gap's voltage comes back)."""
    import json

    from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics, TrainRunner
    side = tmp_path / "cav.json"
    side.write_text(json.dumps({"G*": {"r_over_q": 200.0, "q_loaded": 5.0e6}}))
    cfg = BeamConfig(species="proton", energy=3.0, frequency=F, current=5.0, n_particles=200, distribution="waterbag",
                     emit_nx=0.25, alpha_x=0.0, beta_x=0.4, emit_ny=0.25, alpha_y=0.0, beta_y=0.4, emit_z=0.15,
                     alpha_z=0.0, beta_z=1.2)
    tc = TrainConfig(bunch_frequency_MHz=F, pattern=PulsePattern.from_rle("1*1"),
                     physics=TrainPhysics(beam_loading=True), cavity_params=str(side))
    make, phase = _LOADED[cavity]
    runner = TrainRunner(_line(SetBeamPhaseError("PE", dphi_deg=7.0), make()), cfg, tc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.run()
    st = next(iter(runner._loading.reg.items()))[1]
    assert st.phi_s_deg == pytest.approx(phase, abs=1e-12)
    if cavity.startswith("gap"):
        assert st.v_design_MV == pytest.approx(0.6, rel=1e-9)


def test_the_inspector_does_not_offer_what_an_error_study_stamps():
    """Nor do the Parameter Scan and the Study tab, which share one list of what can be scanned."""
    pytest.importorskip("PyQt6")
    from linac_gen_gui.interphase.dialogs.parameter_scan import _scannable_params
    from linac_gen_gui.interphase.panels.element_inspector import _introspect_command_schema
    card = SetBeamPhaseError("PE", dphi_deg=1.0)
    attrs = [a for a, _label, _unit in _introspect_command_schema(card)]
    assert attrs == ["dphi_deg", "random_flag"]
    offered = set(_scannable_params(card))
    assert {"dphi_deg", "random_flag"} <= offered
    assert not offered & set(SetBeamPhaseError.STUDY_STAMPS)
