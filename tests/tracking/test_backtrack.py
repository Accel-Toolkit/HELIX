# tests/tracking/test_backtrack.py
"""Backward tracking: replay table, round-trips, recorder shape, guards.

Round-trip pattern: forward-track a beam (no SC, no apertures), snapshot
the input, backtrack the exit distribution, compare.  Everything in this
file is exact algebra (thin kicks + matrix inverses), so tolerances are
tight (rtol=1e-8) — field maps and SC have their own test modules.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.foil import Foil
from linac_gen.tracking.tracker import Tracker
from linac_gen.tracking.backtrack import (
    BacktrackWarning, backtrack_distribution, build_replay_table,
)

W_KIN = 3.0        # MeV
FREQ = 352.21      # MHz


def _make_ref():
    return ReferenceParticle(species=PROTON, w_kin=W_KIN, frequency=FREQ)


def _make_beam(n=200, seed=42, current=0.0):
    beam = Beam(ref=_make_ref(), n_particles=n, current=current)
    beam.continuous = False
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0, 1.0, n)   # x mm
    beam.particles[:, 1] = rng.normal(0, 0.5, n)   # x' mrad
    beam.particles[:, 2] = rng.normal(0, 1.0, n)   # y mm
    beam.particles[:, 3] = rng.normal(0, 0.5, n)   # y' mrad
    beam.particles[:, 4] = rng.normal(0, 5.0, n)   # dphi deg
    beam.particles[:, 5] = rng.normal(0, 0.005, n)  # dW MeV
    return beam


def _fodo(aperture=0.0):
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, aperture=aperture, n_steps=5))
    lat.add(Drift("D1", 200.0, aperture=aperture))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, aperture=aperture,
                       n_steps=5))
    lat.add(Drift("D2", 200.0, aperture=aperture))
    return lat


def _fodo_rfgap(freq=FREQ):
    lat = _fodo()
    lat.add(RFGap("GAP", voltage=0.4, phase=-30.0, frequency=freq, ttf=0.85))
    lat.add(Drift("D3", 150.0))
    lat.add(Steerer("ST", bx_l=1e-4, by_l=-2e-4))
    lat.add(Drift("D4", 150.0))
    return lat


def _forward(lat, beam, **kw):
    return Tracker(lat, beam, **kw).run()


def _roundtrip(lat, *, start=0, end=None, seed=42):
    """Forward-track, snapshot input, backtrack exit; return the pieces."""
    beam = _make_beam(seed=seed)
    p_in = beam.particles.copy()
    _forward(lat, beam)
    rec = backtrack_distribution(lat, beam, _make_ref(), start=start, end=end)
    return p_in, beam, rec


# ---------------------------------------------------------------------------
# Replay table
# ---------------------------------------------------------------------------

def test_replay_table_matches_forward_run():
    lat = _fodo_rfgap()
    table = build_replay_table(lat, _make_ref(), end=len(lat.elements) - 1)
    assert len(table) == len(lat.elements) + 1
    # Boundaries: monotone s, entrance state exact.
    assert table[0].s == 0.0
    assert table[0].w_kin == W_KIN
    s_vals = [b.s for b in table]
    assert all(b >= a for a, b in zip(s_vals, s_vals[1:]))
    # Forward MP run must land exactly on the table's exit state.
    beam = _make_beam()
    _forward(lat, beam)
    assert beam.ref.w_kin == pytest.approx(table[-1].w_kin, rel=1e-12)
    assert beam.ref.s == pytest.approx(table[-1].s, rel=1e-12)
    # RFGap accelerated the reference:
    assert table[-1].w_kin > W_KIN


def test_replay_table_rf_seen_flag():
    lat = _fodo_rfgap()
    table = build_replay_table(lat, _make_ref(), end=len(lat.elements) - 1)
    names = [type(e).__name__ for e in lat.elements]
    i_gap = names.index("RFGap")
    assert table[i_gap].rf_seen is False       # entrance of the gap
    assert table[i_gap + 1].rf_seen is True    # exit of the gap
    assert table[-1].rf_seen is True


# ---------------------------------------------------------------------------
# Round trips (exact algebra)
# ---------------------------------------------------------------------------

def test_fodo_roundtrip_exact():
    p_in, beam, _ = _roundtrip(_fodo())
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-11)
    assert beam.ref.s == pytest.approx(0.0, abs=1e-12)
    assert beam.ref.w_kin == pytest.approx(W_KIN, rel=1e-12)


def test_fodo_rfgap_roundtrip_exact():
    """Damping, defocus, energy kick and steerer all undone exactly."""
    p_in, beam, _ = _roundtrip(_fodo_rfgap())
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-11)
    assert beam.ref.w_kin == pytest.approx(W_KIN, rel=1e-12)


def test_rfgap_freq_jump_roundtrip():
    """The gap doubles the RF frequency: forward rescales dphi by
    f_new/f_old; the inverse must rescale back exactly."""
    p_in, beam, _ = _roundtrip(_fodo_rfgap(freq=2 * FREQ))
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-11)
    assert beam.ref.frequency == pytest.approx(FREQ)


def test_subrange_backtrack_matches_forward_snapshot():
    """Backtracking [2, 4] must land on the forward state at the
    entrance of element 2."""
    lat = _fodo_rfgap()
    start, end = 2, 4
    beam = _make_beam()
    # Forward with a snapshot at the entrance of `start`:
    snap = None
    tracker = Tracker(lat, beam)
    for i, element in enumerate(lat.elements):
        if i == start:
            snap = beam.particles.copy()
        tracker._track_element(element)
        if i == end:
            break
    assert snap is not None
    backtrack_distribution(lat, beam, _make_ref(), start=start, end=end)
    np.testing.assert_allclose(beam.particles, snap, rtol=1e-8, atol=1e-11)


def test_misaligned_quad_roundtrip():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, n_steps=5,
                       dx=1.5, dy=-0.7, tilt_deg=3.0))
    lat.add(Drift("D1", 200.0))
    p_in, beam, _ = _roundtrip(lat)
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-11)


# ---------------------------------------------------------------------------
# Recorder shape
# ---------------------------------------------------------------------------

def test_recorder_reversed_monotone_s():
    lat = _fodo_rfgap()
    _, _, rec = _roundtrip(lat)
    assert rec.direction == "backward"
    assert rec.backtrack_range == (0, len(lat.elements) - 1)
    assert len(rec.s) == len(lat.elements) + 1
    assert all(b >= a for a, b in zip(rec.s, rec.s[1:]))
    # Index 0 = reconstructed entrance, last = the supplied exit beam.
    assert rec.s[0] == pytest.approx(0.0, abs=1e-12)
    assert rec.element_names[-1] == "INPUT"
    # Per-step series lengths all match after the in-place reversal.
    assert len(rec.sigma_x) == len(rec.s)
    assert len(rec.ref_w_kin) == len(rec.s)
    assert rec.ref_w_kin[-1] > rec.ref_w_kin[0]   # gap gain visible


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_invalid_range_raises():
    lat = _fodo()
    beam = _make_beam()
    with pytest.raises(ValueError, match="invalid backtrack range"):
        backtrack_distribution(lat, beam, _make_ref(), start=3, end=1)
    with pytest.raises(ValueError, match="invalid backtrack range"):
        backtrack_distribution(lat, beam, _make_ref(), start=0, end=99)


def test_energy_mismatch_warns_then_refuses():
    lat = _fodo()
    beam = _make_beam()
    _forward(lat, beam)
    beam.ref.w_kin *= 1.01          # 1 % off design: warn + re-center
    with pytest.warns(BacktrackWarning, match="re-centring"):
        backtrack_distribution(lat, beam, _make_ref())

    beam2 = _make_beam()
    _forward(lat, beam2)
    beam2.ref.w_kin *= 1.10         # 10 % off design: refuse
    with pytest.raises(ValueError, match="energy_hard_limit"):
        backtrack_distribution(lat, beam2, _make_ref())


def test_energy_recentring_shifts_dw():
    lat = _fodo()
    beam = _make_beam()
    _forward(lat, beam)
    dw_before = beam.particles[:, 5].copy()
    offset = W_KIN * 0.005
    beam.ref.w_kin += offset
    with pytest.warns(BacktrackWarning):
        rec = backtrack_distribution(lat, beam, _make_ref(),
                                     start=3, end=3)   # single drift
    # The dW re-centring happened before the walk:
    assert rec.direction == "backward"
    # A drift leaves dW untouched, so the shift is visible at the end.
    np.testing.assert_allclose(beam.particles[:, 5], dw_before + offset,
                               rtol=1e-10)


def test_dc_flagged_beam_downstream_of_gap_refused():
    """A beam still flagged continuous downstream of a bunching element
    is inconsistent input (a forward run would have bunched it)."""
    lat = _fodo_rfgap()
    beam = _make_beam()
    _forward(lat, beam)
    beam.continuous = True            # forge the inconsistency
    with pytest.raises(ValueError, match="flagged continuous"):
        backtrack_distribution(lat, beam, _make_ref())


def test_dc_upstream_opt_in_flips_flag():
    """allow_dc_crossing asserts the forward beam was DC upstream of
    the first buncher: the walk flips beam.continuous there, mirroring
    the forward tracker's flip.  The kick algebra is unchanged (no SC
    here), so the round trip stays exact."""
    lat = _fodo_rfgap()
    beam = _make_beam()
    p_in = beam.particles.copy()
    _forward(lat, beam)
    with pytest.warns(BacktrackWarning, match="DC↔bunched"):
        backtrack_distribution(lat, beam, _make_ref(),
                               allow_dc_crossing=True)
    assert beam.continuous is True    # DC upstream of the gap
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-11)


def test_bunched_beam_crosses_gap_without_opt_in():
    """The default walk treats the beam as bunched everywhere — a range
    containing RF elements needs no flag (the forward tracker only flips
    `continuous` for DC beams, so there is no transition to undo)."""
    lat = _fodo_rfgap()
    beam = _make_beam()
    _forward(lat, beam)
    backtrack_distribution(lat, beam, _make_ref())
    assert beam.continuous is False


def test_aperture_losses_warn_survivors_only():
    lat = _fodo(aperture=2.0)       # tight: guaranteed losses
    beam = _make_beam()
    _forward(lat, beam)
    assert beam.n_alive < beam.n_particles
    with pytest.warns(BacktrackWarning, match="surviving particles only"):
        backtrack_distribution(lat, beam, _make_ref())
    # No resurrection, no further cuts:
    assert beam.n_alive < beam.n_particles


def test_foil_refused_with_clear_message():
    lat = _fodo()
    lat.add(Foil("F1", thickness_ug_cm2=100.0, seed=7))
    lat.add(Drift("D3", 100.0))
    beam = _make_beam()
    _forward(lat, beam)
    with pytest.raises(NotImplementedError, match="Foil has no inverse_kick"):
        backtrack_distribution(lat, beam, _make_ref())
