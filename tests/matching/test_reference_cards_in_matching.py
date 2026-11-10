"""Matching and the period passes across a card that sets the reference.

The walks hold these cards, so what runs after one has to start from that reference,
with the beam the line hands it, and must not apply the card again.
"""
import numpy as np
import pytest

from linac_gen.analysis.footprint import tune_footprint
from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.analysis.phase_advance import channel_phase_advance_matched
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Freq, SetReferenceEnergy
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.matching.periodic import (
    _emittance_damping,
    find_matched_input_twiss,
    find_matched_period_sigma,
    find_sc_matched_input_twiss,
    make_period_sigma_map,
)
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import reference_at
from linac_gen.tracking.tracker import Tracker

BASE = {"emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2, "alpha_z": 0.0, "beta_z": 1.0}
INITIAL = dict(BASE, alpha_x=0.0, beta_x=1.0, alpha_y=0.0, beta_y=1.0)
TWISS = ("alpha_x", "beta_x", "alpha_y", "beta_y")
# the HELIX card (SET_BEAM_ENERGY and SET_BEAM_E0_P0 set and shifted the energy until
# 2026-11; in TraceWin they are matching criteria)
ENERGY_CARDS = [lambda: SetReferenceEnergy("E", energy_MeV=120.0),
                lambda: SetReferenceEnergy("E", energy_MeV=20.0, shift=True)]
IDS = ["HELIX_SET_ENERGY", "HELIX_SHIFT_ENERGY"]


def _line(*elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    lat.step_config = StepConfig(100.0, 50.0)
    return lat


def _ref(w=100.0, f=352.21):
    return ReferenceParticle(species=PROTON, w_kin=w, frequency=f)


def _cell(tag=""):
    return [Quadrupole(f"QF{tag}", length=100.0, gradient=8.0, aperture=30.0), Drift(f"Da{tag}", 300.0),
            Quadrupole(f"QD{tag}", length=100.0, gradient=-8.0, aperture=30.0), Drift(f"Db{tag}", 300.0)]


def _gap():
    return RFGap("G", voltage=5.0, phase=-30.0, frequency=352.21, ttf=1.0)


def _same_twiss(a, b, rel):
    for key in TWISS:
        assert a[key] == pytest.approx(b[key], rel=rel, abs=rel), key


# ---------------------------------------------------------------------------
# the space-charge matched input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("card", ENERGY_CARDS, ids=IDS)
def test_the_matched_cell_runs_at_the_energy_a_card_sets(card):
    a = find_sc_matched_input_twiss(_line(card(), *_cell()), _ref(100.0), 1, 4, 5.0, dict(BASE))
    b = find_sc_matched_input_twiss(_line(*_cell()), _ref(120.0), 0, 3, 5.0, dict(BASE))
    _same_twiss(a, b, 1e-6)
    assert a["mu_x"] == pytest.approx(b["mu_x"], rel=1e-12)


@pytest.mark.parametrize("front", [
    lambda: [Drift("D0", 100.0), ENERGY_CARDS[0]()],
    lambda: [Drift("D0", 100.0), ENERGY_CARDS[1]()],
    lambda: [_gap(), Drift("D0", 100.0)],
], ids=[*IDS, "a cavity"])
def test_at_low_current_the_match_tends_to_the_zero_current_one(front):
    els = front()
    lat = _line(*els, *_cell())
    n = len(els)
    zero = find_matched_input_twiss(lat, _ref(100.0), n, n + 3)
    low = find_sc_matched_input_twiss(lat, _ref(100.0), n, n + 3, 1e-6, dict(BASE), max_iter=100)
    assert low["mu_x"] == pytest.approx(zero["mu_x"], rel=1e-9)
    _same_twiss(low, zero, 2e-3)


def test_a_cavity_in_front_hands_the_cell_a_damped_beam():
    """The entrance state, through the front, reaches the match of the cell alone at
    the energy and the emittance it has there."""
    lat = _line(_gap(), *_cell())
    ref = _ref(100.0)
    a = find_sc_matched_input_twiss(lat, ref, 1, 4, 5.0, dict(BASE))
    at = reference_at(lat, ref, 1)
    damp = ref.bg / at.bg
    assert damp < 0.99
    alone = dict(BASE, emit_x=BASE["emit_x"] * damp, emit_y=BASE["emit_y"] * damp)
    b = find_sc_matched_input_twiss(_line(*_cell()), at, 0, 3, 5.0, alone)
    init = dict(BASE, **{k: a[k] for k in TWISS})
    end = np.asarray(EnvelopeSolver(_line(_gap()), ref.copy(), init, current=5.0).run().sigma_matrix[-1])
    for (i, j), alpha, beta in (((0, 1), "alpha_x", "beta_x"), ((2, 3), "alpha_y", "beta_y")):
        eps = (end[i, i] * end[j, j] - end[i, j] ** 2) ** 0.5
        assert end[i, i] / eps == pytest.approx(b[beta], rel=1e-3)
        assert -end[i, j] / eps == pytest.approx(b[alpha], rel=1e-3, abs=1e-3)


def test_a_freq_card_in_front_keeps_the_same_bunch():
    """Half the frequency and half the current on the card\'s far side: the same charge,
    and the same length of bunch in the degrees of each side."""
    a = find_sc_matched_input_twiss(_line(Drift("D0", 100.0), *_cell()), _ref(100.0, 704.42), 1, 4, 10.0,
                                    dict(BASE))
    half = dict(BASE, emit_z=0.1, beta_z=0.5)
    b = find_sc_matched_input_twiss(_line(Drift("D0", 100.0), Freq("F", frequency_mhz=704.42), *_cell()),
                                    _ref(100.0, 352.21), 2, 5, 5.0, half)
    _same_twiss(a, b, 1e-4)


@pytest.mark.parametrize("front, want", [
    (lambda: [ENERGY_CARDS[0]()], 1.0),
    (lambda: [ENERGY_CARDS[1]()], 1.0),
    (lambda: [Drift("D", 100.0)], 1.0),
], ids=[*IDS, "a drift"])
def test_only_a_cavity_damps_the_emittance(front, want):
    lat = _line(*front(), *_cell())
    n = len(front())
    assert _emittance_damping(lat, _ref(), n, reference_at(lat, _ref(), n)) == pytest.approx(want, rel=1e-14)


def test_a_cavity_counts_and_a_card_after_it_does_not():
    ref = _ref()
    gap_alone = _line(_gap())
    damp = ref.bg / reference_at(gap_alone, ref, 1).bg
    assert damp < 1.0
    lat = _line(_gap(), SetReferenceEnergy("E", energy_MeV=20.0, shift=True))
    assert _emittance_damping(lat, ref, 2, reference_at(lat, ref, 2)) == pytest.approx(damp, rel=1e-12)


# ---------------------------------------------------------------------------
# a card before a period, and the passes over that period
# ---------------------------------------------------------------------------
def _periodic(prefix, n_cells=4, front=True):
    els = [*([Drift("D00", 100.0)] if front else []), *prefix()]
    n0 = len(els)
    for i in range(n_cells):
        els += _cell(str(i))
    lat = _line(*els)
    return lat, PeriodicStructure(start=n0, end=len(lat.elements), inner_period_length=4,
                                  inner_slice_end=n0 + 4, n_repeats=n_cells, label="p", source="fallback")


PREFIXES = {
    "a shift, not idempotent": lambda: [SetReferenceEnergy("E", energy_MeV=20.0, shift=True)],
    "a set energy, then a cavity": lambda: [SetReferenceEnergy("E", energy_MeV=120.0), _gap()],
    "FREQ, then a cavity at another": lambda: [Freq("F", frequency_mhz=352.21),
                                               RFGap("G", voltage=1.0, phase=-30.0, frequency=704.42, ttf=1.0)],
}


@pytest.fixture
def seen(monkeypatch):
    """The reference each pass really starts from."""
    runs = []
    env_run, track_run = EnvelopeSolver.run, Tracker.run

    def on_env(self):
        runs.append((float(self._ref.w_kin), float(self._ref.frequency)))
        return env_run(self)

    def on_track(self, *a, **k):
        runs.append((float(self.beam.ref.w_kin), float(self.beam.ref.frequency)))
        return track_run(self, *a, **k)
    monkeypatch.setattr(EnvelopeSolver, "run", on_env)
    monkeypatch.setattr(Tracker, "run", on_track)
    return runs


@pytest.mark.parametrize("name", PREFIXES)
def test_the_period_pass_starts_where_the_line_has_got_to(name, seen):
    lat, period = _periodic(PREFIXES[name])
    ref = _ref()
    at = reference_at(lat, ref, period.start)
    sigma_map, state = make_period_sigma_map(lat, ref, period, 5.0, dict(INITIAL))
    seen.clear()
    sigma_map(state["sigma_seed"])
    assert len(seen) == 1
    assert seen[0] == pytest.approx((at.w_kin, at.frequency), rel=1e-12)


def test_a_card_before_a_period_gives_the_period_of_the_line_at_that_energy():
    """With nothing before the card: a drift at another energy would hand on another bunch.

    The cell has no RF, so a bunched beam has no longitudinal focusing and no matched bunch
    exists: the full iteration stops unconverged, where a 1e-15 change of the input moves its
    answer by 0.26 (measured 2026-11-07).  So the period's map is compared from one seed, and
    the matched beam with its bunch length held (``longitudinal="frozen"``, which converges):
    its jitter under 1e-15 to 1e-12 changes of the input stays below 3e-8 of max|Σ|, while the line
    at 140 MeV moves it by 5e-2 (the map by 0.34)."""
    with_card, p1 = _periodic(PREFIXES["a shift, not idempotent"], front=False)
    plain, p2 = _periodic(lambda: [], front=False)
    map_a, state_a = make_period_sigma_map(with_card, _ref(100.0), p1, 5.0, dict(INITIAL))
    map_b, state_b = make_period_sigma_map(plain, _ref(120.0), p2, 5.0, dict(INITIAL))
    seed = state_b["sigma_seed"]
    np.testing.assert_allclose(state_a["sigma_seed"], seed, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(map_a(seed), map_b(seed), rtol=1e-12, atol=1e-12)
    a = find_matched_period_sigma(with_card, _ref(100.0), p1, 5.0, dict(INITIAL), longitudinal="frozen")
    b = find_matched_period_sigma(plain, _ref(120.0), p2, 5.0, dict(INITIAL), longitudinal="frozen")
    assert a["converged"] and b["converged"]
    np.testing.assert_allclose(a["sigma_entry"], b["sigma_entry"], rtol=1e-6,
                               atol=1e-6 * np.abs(b["sigma_entry"]).max())


def test_the_channel_phase_advance_is_that_of_the_line_at_that_energy():
    """On the matched beam with its bunch length held (see above): the bare tunes agree to
    1e-9 and the depressed ones to 1e-6, against a measured jitter of 0 and 2.8e-9; the line at
    140 MeV moves both by 8e-2."""
    with_card, p1 = _periodic(PREFIXES["a shift, not idempotent"], front=False)
    plain, p2 = _periodic(lambda: [], front=False)
    a = channel_phase_advance_matched(with_card, _ref(100.0), p1, 5.0, dict(INITIAL), longitudinal="frozen",
                                      max_iter=8)
    b = channel_phase_advance_matched(plain, _ref(120.0), p2, 5.0, dict(INITIAL), longitudinal="frozen",
                                      max_iter=8)
    assert a["matched_state"]["converged"] and b["matched_state"]["converged"]
    for key in ("total_mu_x_bare", "total_mu_y_bare"):
        assert a[key] == pytest.approx(b[key], rel=1e-9), key
    for key in ("total_mu_x_dep", "total_mu_y_dep"):
        assert a[key] == pytest.approx(b[key], rel=1e-6), key


def test_the_footprint_tracks_at_the_energy_the_line_has(seen):
    lat, period = _periodic(PREFIXES["a shift, not idempotent"], n_cells=3)
    tune_footprint(lat, _ref(100.0), period, dict(INITIAL), 5.0, n_turns=4, n_particles=4)
    tracked = [w for w, _f in seen]
    assert tracked and set(np.round(tracked[-4:], 9)) == {120.0}
