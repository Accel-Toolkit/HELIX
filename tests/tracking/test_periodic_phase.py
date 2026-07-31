# tests/tracking/test_periodic_phase.py
"""Periodic phase coordinates for bunch-train beams.

An RFQ makes ONE BUNCH PER RF PERIOD, but HELIX seeds a single period.
Space charge then pushes ~20 % of the particles across a bucket
boundary and, with Δφ stored unwrapped, they become satellite clumps
one full bunch spacing away — inflating every reported σ_φ and ε_z
(183° for a bunch that is really 4°) and feeding the partran export,
SET_SIZE constraints, the IBS σ_z and the GUI plots.

``BeamConfig.periodic_phase`` folds Δφ into one bunch spacing during
TRACKING (the Toutatis convention) so the satellites never form.  It is
opt-in and default OFF; folding the STATISTICS instead was tried twice
and rejected with measurements (see
``diagnostics/moments.py::wrap_phase_column``).

These tests pin the four properties the fix rests on:

1. the fold is invisible to the physics (RF forces are 360°-periodic);
2. it only ever engages on a beam that was injected DC and then bunched;
3. the period is the BUNCH SPACING, not the local RF period, so it
   doubles across a 162.5 → 325 MHz jump;
4. it is refused where it would be wrong — non-integer frequency
   ratios, and backtracking.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Freq
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.tracker import PeriodicPhaseWarning

F0 = 162.5


def _lat(n_drift=8, gap_voltage=0.002, freq=F0):
    """Two drifts, a bunching RF gap, then more drifts — the minimum
    lattice that exercises the DC → bunched flip and then folds."""
    lat = Lattice()
    lat.add(Drift("D_in0", length=100.0, aperture=50.0))
    lat.add(Drift("D_in1", length=100.0, aperture=50.0))
    lat.add(RFGap("BUNCHER", voltage=gap_voltage, phase=-90.0,
                  frequency=freq, ttf=1.0, aperture=50.0))
    for i in range(n_drift):
        lat.add(Drift(f"D{i}", length=200.0, aperture=50.0))
    return lat


def _entrance_ref():
    return ReferenceParticle(species=PROTON, w_kin=0.03, frequency=F0)


def _dc_beam(periodic, n=400, current=0.0, seed=11):
    """DC beam: uniform phase over one RF period, as the factory makes
    it for ``continuous=True``."""
    ref = ReferenceParticle(species=PROTON, w_kin=0.03, frequency=F0)
    b = Beam(ref=ref, n_particles=n, current=current)
    b.continuous = True
    b.periodic_phase = periodic
    rng = np.random.default_rng(seed)
    b.particles[:, 0] = rng.normal(0, 2.0, n)
    b.particles[:, 1] = rng.normal(0, 3.0, n)
    b.particles[:, 2] = rng.normal(0, 2.0, n)
    b.particles[:, 3] = rng.normal(0, 3.0, n)
    b.particles[:, 4] = rng.uniform(-180.0, 180.0, n)
    return b


def _bunched_beam(periodic, n=400, seed=5):
    """Beam born bunched — the MEBT-to-Foil case.  Must never fold."""
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=F0)
    b = Beam(ref=ref, n_particles=n, current=0.0)
    b.periodic_phase = periodic          # flag set, marker absent
    rng = np.random.default_rng(seed)
    b.particles[:, 0] = rng.normal(0, 1.0, n)
    b.particles[:, 2] = rng.normal(0, 1.0, n)
    # Deliberately WIDE in phase: > half a bunch spacing, so an
    # unguarded fold would visibly move particles.
    b.particles[:, 4] = rng.uniform(-300.0, 300.0, n)
    b.particles[:, 5] = rng.normal(0, 0.01, n)
    return b


# ----------------------------------------------------------------------
# 1. The fold is invisible to the physics
# ----------------------------------------------------------------------

def test_fold_does_not_change_the_physics():
    """Same lattice, flag OFF vs ON: identical losses, identical
    transverse coordinates and energies, and Δφ differing only by whole
    multiples of the bunch spacing.

    Bit-identity is NOT claimed for the folded particles — cos(x+360k)
    differs from cos(x) in the last bits — hence the tight-but-finite
    tolerances.  Flag OFF is bit-identical to before the feature.
    """
    off = _dc_beam(False)
    on = _dc_beam(True)
    Simulation(_lat(), off, space_charge="off").run()
    Simulation(_lat(), on, space_charge="off").run()

    np.testing.assert_array_equal(off.lost, on.lost)
    alive = off.alive_mask
    for col, tol in ((0, 1e-9), (1, 1e-9), (2, 1e-9), (3, 1e-9), (5, 1e-12)):
        np.testing.assert_allclose(off.particles[alive, col],
                                   on.particles[alive, col],
                                   rtol=0, atol=tol)
    # Δφ: equal modulo the bunch spacing (360° here — f_local == f_bunch).
    resid = (off.particles[alive, 4] - on.particles[alive, 4] + 180.0) \
        % 360.0 - 180.0
    assert np.abs(resid).max() < 1e-8


def test_fold_actually_folds_and_bounds_the_phase():
    """The whole point: after bunching, no surviving particle is more
    than half a bunch spacing from the synchronous particle — and
    without the flag some are."""
    off = _dc_beam(False)
    on = _dc_beam(True)
    Simulation(_lat(), off, space_charge="off").run()
    Simulation(_lat(), on, space_charge="off").run()

    assert np.abs(off.particles[off.alive_mask, 4]).max() > 180.0, \
        "test lattice no longer produces out-of-bucket particles"
    assert np.abs(on.particles[on.alive_mask, 4]).max() <= 180.0


def test_fold_counter_reports_actual_bucket_crossings():
    """``Tracker._n_phase_folded`` is the honest measure of how much a
    run was changed — it must stay exactly 0 when the fold never fires,
    including for a flagged run whose beam never leaves its bucket."""
    from linac_gen.tracking.tracker import Tracker

    def _count(beam):
        tr = Tracker(_lat(), beam, pic_solver="off")
        tr.run()
        return tr._n_phase_folded

    assert _count(_dc_beam(False)) == 0          # flag off
    assert _count(_dc_beam(True)) > 0            # flag on, wide beam

    tight = _dc_beam(True, n=64)
    tight.particles[:, 4] = 0.0                  # never leaves the bucket
    tight.particles[:, 5] = 0.0
    assert _count(tight) == 0


def test_reported_sigma_phi_shrinks_with_no_diagnostic_change():
    """σ_φ must drop because the TRACKED coordinates changed, not
    because any statistics code was touched."""
    off = _dc_beam(False)
    on = _dc_beam(True)
    r_off = Simulation(_lat(), off, space_charge="off").run()
    r_on = Simulation(_lat(), on, space_charge="off").run()
    assert r_on.sigma_phi[-1] < r_off.sigma_phi[-1]


# ----------------------------------------------------------------------
# 2. Provenance: only a DC-injected, then-bunched beam may fold
# ----------------------------------------------------------------------

def test_beam_born_bunched_is_never_folded():
    """MEBT-to-Foil and every other bunched-input deck: the flag alone
    must not touch the beam, because such a beam is ONE bunch, not a
    seeded period of a train."""
    ref_b = _bunched_beam(False)
    on_b = _bunched_beam(True)
    lat = Lattice()
    for i in range(6):
        lat.add(Drift(f"D{i}", length=200.0, aperture=50.0))
    Simulation(lat, ref_b, space_charge="off").run()
    Simulation(lat, on_b, space_charge="off").run()
    np.testing.assert_array_equal(ref_b.particles, on_b.particles)
    assert np.abs(on_b.particles[:, 4]).max() > 180.0


def test_bunch_train_marker_set_only_at_the_dc_flip():
    b = _dc_beam(True)
    assert b.bunch_train is False
    Simulation(_lat(), b, space_charge="off").run()
    assert b.bunch_train is True
    assert b.continuous is False

    born = _bunched_beam(True)
    lat = Lattice()
    lat.add(Drift("D", length=200.0, aperture=50.0))
    Simulation(lat, born, space_charge="off").run()
    assert born.bunch_train is False


def test_dc_segment_is_not_folded():
    """Upstream of the buncher the beam really does occupy the whole RF
    period; folding there would be a lie (and a no-op, since ±180 is
    already the bucket).  Nothing may move before the gap."""
    b = _dc_beam(True)
    lat = Lattice()
    lat.add(Drift("D0", length=200.0, aperture=50.0))
    before = b.particles.copy()
    Simulation(lat, b, space_charge="off").run()
    np.testing.assert_array_equal(before[:, 4], b.particles[:, 4])


# ----------------------------------------------------------------------
# 3. The period is the BUNCH SPACING, not the local RF period
# ----------------------------------------------------------------------

def test_period_is_the_bunch_spacing_not_the_rf_period():
    """After 162.5 → 325 MHz the local degrees are half as big, so ONE
    BUNCH SPACING is 720 local degrees, not 360.

    ``bunch_frequency`` is pinned at beam creation and deliberately not
    updated by the FREQ card, which is exactly what makes this work.
    Folding with a 360° period here would move a particle into a bunch
    that does not exist.
    """
    from linac_gen.tracking.tracker import Tracker
    lat = Lattice()
    lat.add(RFGap("BUNCHER", voltage=0.001, phase=-90.0, frequency=F0,
                  ttf=1.0, aperture=50.0))
    lat.add(Freq("FREQ", frequency_mhz=2 * F0))
    b = _dc_beam(True, n=8)
    b.particles[:] = 0.0
    # Post-bunching state, as run() leaves it at the first RF element.
    # (That the flip really sets the marker is pinned separately by
    # test_bunch_train_marker_set_only_at_the_dc_flip — Tracker.run
    # sets it, and _track_element is called directly here.)
    b.continuous, b.bunch_train = False, True
    b.bunch_train_frequency = F0             # captured at the buncher
    tr = Tracker(lat, b, pic_solver="off")
    tr._track_element(lat.elements[1])       # the frequency jump
    assert b.ref.frequency == pytest.approx(2 * F0)
    assert b.bunch_frequency == pytest.approx(F0), \
        "bunch_frequency must stay pinned at the RFQ/buncher rate"

    # 300 local degrees out.  With the correct 720° bunch spacing this
    # is INSIDE the bucket and must not move; with the naive 360° RF
    # period it would be folded to −60° — into a bunch that does not
    # exist at this frequency.
    b.particles[0, 4] = 300.0
    # 800 local degrees out: genuinely one bunch spacing away, so this
    # one DOES fold, to 800 − 720 = 80°.
    b.particles[1, 4] = 800.0
    tr._fold_phase()
    assert b.particles[0, 4] == 300.0, \
        "folded with a 360° period instead of the 720° bunch spacing"
    assert b.particles[1, 4] == pytest.approx(80.0, abs=1e-9)


def test_fold_and_freq_rescale_commute():
    """The FREQ card multiplies Δφ by f_new/f_old and the fold period by
    the same ratio, so fold-then-jump and jump-then-fold agree.  That is
    why the period may be recomputed live and never cached."""
    from linac_gen.tracking.tracker import Tracker

    def _run(phi, fold_before):
        lat = Lattice()
        lat.add(RFGap("BUNCHER", voltage=0.001, phase=-90.0, frequency=F0,
                      ttf=1.0, aperture=50.0))
        lat.add(Freq("FREQ", frequency_mhz=2 * F0))
        b = _dc_beam(True, n=4)
        b.particles[:] = 0.0
        b.continuous, b.bunch_train = False, True
        b.bunch_train_frequency = F0
        tr = Tracker(lat, b, pic_solver="off")
        b.particles[0, 4] = phi              # pre-jump degrees
        if fold_before:
            tr._fold_phase()                 # P = 360 here
        tr._track_element(lat.elements[1])   # ×2
        tr._fold_phase()                     # P = 720 here
        return b.particles[0, 4]

    for phi in (300.0, 540.0, -900.0):
        assert _run(phi, True) == pytest.approx(_run(phi, False), abs=1e-9)


def test_non_integer_frequency_ratio_is_skipped_with_one_warning():
    """A sub-harmonic buncher (f_local < f_bunch) makes the fold change
    the RF phase a particle sees.  Skip it — loudly, once."""
    b = _dc_beam(True, n=8)
    lat = Lattice()
    lat.add(RFGap("BUNCHER", voltage=0.05, phase=-90.0, frequency=F0,
                  ttf=1.0, aperture=50.0))
    lat.add(Freq("FREQ", frequency_mhz=1.5 * F0))
    for i in range(4):
        lat.add(Drift(f"D{i}", length=100.0, aperture=50.0))
    b.particles[:, 4] = 0.0
    b.particles[0, 4] = 300.0
    b.particles[:, 5] = 0.0
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        Simulation(lat, b, space_charge="off").run()
    hits = [w for w in wl if issubclass(w.category, PeriodicPhaseWarning)]
    assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"
    assert "integer multiple" in str(hits[0].message)
    assert np.abs(b.particles[0, 4]) > 180.0, "folded despite the guard"


# ----------------------------------------------------------------------
# 4. Refusals
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_degenerate_frequency_bails_out_instead_of_crashing(bad):
    """A NaN frequency passes a plain ``<= 0`` test and would then blow
    up inside round() with 'cannot convert float NaN to integer'.  Bail
    out silently instead — a run in that state has bigger problems than
    the fold.  (0.0 is unreachable on ``ref``: the ReferenceParticle
    setter divides by it to get the wavelength and raises first.)"""
    from linac_gen.tracking.tracker import Tracker
    lat = Lattice()
    lat.add(Drift("D", length=100.0, aperture=50.0))
    b = _dc_beam(True, n=8)
    b.continuous, b.bunch_train = False, True
    b.particles[:, 4] = 400.0
    tr = Tracker(lat, b, pic_solver="off")
    b.ref.frequency = bad
    tr._fold_phase()                       # must not raise
    assert b.particles[0, 4] == 400.0      # and must not fold
    b.ref.frequency = F0
    for worse in (bad, 0.0):               # bunch_frequency is a plain attr
        b.bunch_frequency = worse
        tr._fold_phase()
        assert b.particles[0, 4] == 400.0


def test_period_comes_from_the_buncher_not_the_beam_config():
    """ADVERSARIAL FIND (2026-07-30).  ``bunch_frequency`` is pinned from
    BeamConfig.frequency at beam creation, which is NOT the bunch
    repetition rate when the buncher runs at a different frequency.
    Using it made the feature silently inert: a beam configured at
    162.5 MHz and bunched by a 325 MHz gap folded with a 720° period,
    so every satellite survived, |Δφ| reached 359° and not one warning
    was raised.  The period must come from the bunching element."""
    lat = Lattice()
    lat.add(Drift("D0", length=100.0, aperture=50.0))
    lat.add(RFGap("BUNCH325", voltage=0.002, phase=-90.0, frequency=2 * F0,
                  ttf=1.0, aperture=50.0))
    for i in range(8):
        lat.add(Drift(f"D{i}", length=200.0, aperture=50.0))
    b = _dc_beam(True)               # created at F0, bunched at 2*F0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Simulation(lat, b, space_charge="off").run()
    assert b.bunch_frequency == pytest.approx(F0)          # config value
    assert b.bunch_train_frequency == pytest.approx(2 * F0)  # the buncher
    assert np.abs(b.particles[b.alive_mask, 4]).max() <= 180.0


def test_static_electric_element_does_not_make_a_bunch_train():
    """ADVERSARIAL FIND.  ``_is_rf_bunching_element`` accepts any
    ELECTRIC channel, including ``Channel.STAT_E`` — an einzel lens or
    DC extraction column.  Such an element cannot bunch anything, so
    marking the beam a train and folding its genuinely uniform ±180°
    phase distribution would invent a bucket that does not exist."""
    from linac_gen.io.tracewin_geom import Channel
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.tracking.tracker import (_is_rf_bunching_element,
                                            _is_time_varying_rf_buncher)

    class _Chan:
        def __init__(self, ch):
            self.channels = {ch: type("C", (), {"geometry": 1})()}

    static = FieldMap.__new__(FieldMap)
    static.field_data = _Chan(Channel.STAT_E)
    rf = FieldMap.__new__(FieldMap)
    rf.field_data = _Chan(Channel.RF_E)

    # The looser predicate keeps its (pre-existing) behaviour so no
    # existing run's `continuous` flip changes...
    assert _is_rf_bunching_element(static) is True
    # ...but the marker predicate rejects the static element.
    assert _is_time_varying_rf_buncher(static) is False
    assert _is_time_varying_rf_buncher(rf) is True


def test_non_harmonic_frequency_after_folds_is_a_hard_error():
    """ADVERSARIAL FIND.  Skipping the fold at a non-integer ratio is
    only safe while nothing has been folded.  Folds already applied get
    rescaled by f_new/f_old at every downstream RF element and stop
    being whole RF periods — measured as a full crest-to-trough 2qV
    energy error.  Unrecoverable, so it must raise, not warn."""
    from linac_gen.tracking.tracker import Tracker
    lat = Lattice()
    lat.add(RFGap("BUNCHER", voltage=0.001, phase=-90.0, frequency=F0,
                  ttf=1.0, aperture=50.0))
    lat.add(Freq("SUBHARM", frequency_mhz=1.5 * F0))
    b = _dc_beam(True, n=8)
    b.particles[:] = 0.0
    b.continuous, b.bunch_train = False, True
    b.bunch_train_frequency = F0
    tr = Tracker(lat, b, pic_solver="off")
    b.particles[0, 4] = 400.0
    tr._fold_phase()                       # a real fold happens here
    assert tr._n_phase_folded == 1
    # Tracking the FREQ card moves the clock to 1.5*F0 and the fold hook
    # at the end of _track_element raises there — the earliest point at
    # which the corruption is detectable.
    with pytest.raises(ValueError, match="not recoverable"):
        tr._track_element(lat.elements[1])


def test_csr_and_periodic_phase_are_mutually_exclusive():
    """ADVERSARIAL FIND.  csr.py derives its histogram range, bin width
    and smoothing scale from z.min()/z.max() of the whole ensemble, so
    the wake is NOT periodic in the bunch spacing: folding changed the
    energy kick on core particles that were never folded (measured 5 %
    of σ_W).  Refuse the combination."""
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.pic.pic_solver import PicSolver
    from linac_gen.tracking.tracker import Tracker
    sc = SpaceChargeConfig(nx=16, ny=16, nz=16, use_gpu="cpu",
                           grid_mode="adaptive", csr_enabled=True)
    with pytest.raises(ValueError, match="csr_enabled"):
        Tracker(_lat(), _dc_beam(True), pic_solver=PicSolver(sc))
    # Same solver without CSR is fine.
    sc.csr_enabled = False
    Tracker(_lat(), _dc_beam(True), pic_solver=PicSolver(sc))


def test_emit_z_staircase_is_confined_to_beams_that_actually_cross():
    """The documented COST of the convention, pinned so it cannot grow
    silently: a fold is a step change in ε_z, so ε_z(s) is no longer
    exactly constant through identical drifts.  It must remain confined
    to beams whose particles are genuinely crossing buckets — a beam
    that stays in its bucket has to be exactly as flat as before."""
    def _spread(periodic, gap_v):
        lat = Lattice()
        lat.add(RFGap("B", voltage=gap_v, phase=-90.0, frequency=F0,
                      ttf=1.0, aperture=50.0))
        for i in range(40):
            lat.add(Drift(f"D{i}", length=50.0, aperture=50.0))
        res = Simulation(lat, _dc_beam(periodic, n=600),
                         space_charge="off").run()
        ez = np.asarray(res.emit_z)[-40:]
        ez = ez[np.isfinite(ez) & (ez > 0)]
        return (ez.max() - ez.min()) / ez.mean() * 100.0

    # A beam that never leaves its bucket: EXACTLY as flat with the flag
    # on as off (the mask means those particles are never touched).
    assert _spread(True, 0.0002) == pytest.approx(_spread(False, 0.0002),
                                                  abs=1e-9)
    # A badly debunched one does develop the staircase — that is the
    # convention, not a defect, and the docs say so.
    assert _spread(True, 0.002) > _spread(False, 0.002)


def test_realistic_cavity_detuning_does_not_disable_the_fold():
    """ADVERSARIAL FIND, second order.  Machine-epsilon strictness on the
    frequency ratio made the feature hostage to cavity detuning: an
    error-study seed perturbing a cavity by a few kHz would have turned
    the fold off (or, once folds were in flight, aborted the seed).  The
    tolerance is 1e-3 because the period is 360·round(ratio) — a whole
    number of LOCAL degrees, so the RF force is exactly unchanged — and
    the residual only misplaces a particle by ≤ 0.36° inside a bunch
    several degrees wide."""
    from linac_gen.tracking.tracker import Tracker
    for offset_hz in (0.0, 1e3, 10e3, 50e3):
        lat = Lattice()
        lat.add(RFGap("BUNCHER", voltage=0.001, phase=-90.0, frequency=F0,
                      ttf=1.0, aperture=50.0))
        lat.add(Freq("JUMP", frequency_mhz=2 * F0 + offset_hz * 1e-6))
        b = _dc_beam(True, n=8)
        b.particles[:] = 0.0
        b.continuous, b.bunch_train = False, True
        b.bunch_train_frequency = F0
        tr = Tracker(lat, b, pic_solver="off")
        b.particles[0, 4] = 800.0            # one spacing out at P=720
        with warnings.catch_warnings():
            warnings.simplefilter("error", PeriodicPhaseWarning)
            tr._track_element(lat.elements[1])
        assert tr._n_phase_folded == 1, f"fold lost at {offset_hz:g} Hz"
    # A genuine sub-harmonic is still caught, three orders of magnitude
    # away from the tolerance.
    lat = Lattice()
    lat.add(Freq("SUB", frequency_mhz=1.5 * F0))
    b = _dc_beam(True, n=8)
    b.particles[:] = 0.0
    b.continuous, b.bunch_train = False, True
    b.bunch_train_frequency = F0
    b.ref.frequency = 1.5 * F0
    tr = Tracker(lat, b, pic_solver="off")
    b.particles[0, 4] = 800.0
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        tr._fold_phase()
    assert any(issubclass(w.category, PeriodicPhaseWarning) for w in wl)
    assert tr._n_phase_folded == 0


def test_matrix_element_coupling_phase_is_refused():
    """ADVERSARIAL FIND.  An imported Elegant EMATRIX applies the full
    6×6, so a non-zero M[i,4] turns the fold into a deterministic shift
    of M[i,4]·P — measured at 1.85× the beam's own σ_W for a M[5,4] of
    just 1e-6 MeV/deg."""
    from linac_gen.elements.matrix_element import MatrixElement
    from linac_gen.tracking.tracker import Tracker
    m = np.eye(6)
    m[5, 4] = 1e-6
    lat = _lat()
    lat.add(MatrixElement("EMAT", matrix=m, length=0.0))
    with pytest.raises(ValueError, match="column 4"):
        Tracker(lat, _dc_beam(True), pic_solver="off")
    # An identity-column-4 matrix (everything HELIX generates) is fine.
    lat2 = _lat()
    lat2.add(MatrixElement("EMAT_OK", matrix=np.eye(6), length=0.0))
    Tracker(lat2, _dc_beam(True), pic_solver="off")


def test_run_provenance_is_stamped_on_the_recorder():
    """The GUI display fold must be able to tell a folded run from an
    unfolded one instead of guessing from a checkbox."""
    off = Simulation(_lat(), _dc_beam(False), space_charge="off").run()
    on = Simulation(_lat(), _dc_beam(True), space_charge="off").run()
    assert off.periodic_phase is False
    assert on.periodic_phase is True


def test_backtracking_accepts_a_fresh_beam_from_a_flagged_project():
    """ADVERSARIAL FIND, and a shipped regression: the refusal keyed on
    the FLAG alone rejected design-mode backtracking of a beam that had
    never been tracked and so had never been folded — including the
    LEBT+RFQ examples, which now ship with the flag on."""
    from linac_gen.tracking.backtrack import backtrack_distribution
    fresh = _dc_beam(True, n=32)
    assert fresh.bunch_train is False       # nothing has been folded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = backtrack_distribution(_lat(), fresh, _entrance_ref(), end=1)
    assert len(rec.s) > 0


def test_backtracking_refuses_a_folded_beam():
    """The fold discards WHICH bunch a particle landed in, so it cannot
    be undone: backtracking would be wrong by whole bunch spacings.
    Refuse rather than approximate — this module's contract is
    machine-precision closure."""
    from linac_gen.tracking.backtrack import backtrack_distribution
    lat = _lat()
    b = _dc_beam(True)
    Simulation(lat, b, space_charge="off").run()
    with pytest.raises(ValueError, match="non-invertible"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backtrack_distribution(lat, b, _entrance_ref())


def test_backtracker_class_refuses_directly_too():
    """ADVERSARIAL FIND: ``_Backtracker`` is driven directly by tests and
    by Simulation internals, bypassing the function-level guard."""
    from linac_gen.tracking.backtrack import _Backtracker
    lat = _lat()
    b = _dc_beam(True)
    Simulation(lat, b, space_charge="off").run()
    assert b.bunch_train is True
    bt = _Backtracker(lat, b, table=None, start=0, end=0, pic_solver="off")
    with pytest.raises(ValueError, match="cannot be undone"):
        bt.run()


def test_backward_dc_flip_clears_the_bunch_train_marker():
    """ADVERSARIAL FIND: the backward walk sets ``continuous = True``
    but used to leave ``bunch_train`` set, giving a beam that was both
    DC and a bunch train — and the fold tests only the marker, so a
    later forward run would have folded a genuinely uniform ±180°
    distribution."""
    b = _dc_beam(False)
    b.continuous, b.bunch_train = False, True
    b.bunch_train_frequency = F0
    lat = _lat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from linac_gen.tracking.backtrack import (backtrack_distribution,
                                                  build_replay_table)
        build_replay_table(lat, _entrance_ref(), end=8)
        backtrack_distribution(lat, b, _entrance_ref(), end=8,
                               allow_dc_crossing=True)
    assert b.continuous is True
    assert b.bunch_train is False
    assert b.bunch_train_frequency == 0.0


def test_backtracking_still_works_with_the_flag_off():
    """Control for the refusal above: the guard must not fire on a
    normal run."""
    from linac_gen.tracking.backtrack import backtrack_distribution
    lat = _lat()
    b = _dc_beam(False)
    Simulation(lat, b, space_charge="off").run()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = backtrack_distribution(lat, b, _entrance_ref())
    assert len(rec.s) > 0


# ----------------------------------------------------------------------
# 5. Plumbing: the flag reaches the Beam through the real factory
# ----------------------------------------------------------------------

def test_flag_flows_config_to_beam_and_defaults_off():
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam

    base = dict(species="proton", energy=0.03, frequency=F0, current=0.0,
                n_particles=32, distribution="gaussian",
                emit_nx=0.2, alpha_x=0.0, beta_x=1.0,
                emit_ny=0.2, alpha_y=0.0, beta_y=1.0,
                emit_z=0.1, alpha_z=0.0, beta_z=1.0, continuous=True)
    assert BeamConfig(**base).periodic_phase is False        # default OFF
    assert create_beam(BeamConfig(**base), seed=1).periodic_phase is False
    on = create_beam(BeamConfig(**base, periodic_phase=True), seed=1)
    assert on.periodic_phase is True


def test_lgproj_round_trip_carries_the_flag(tmp_path):
    """Older projects without the key must still load (default OFF)."""
    import json
    from linac_gen.io.project import _beam_from_dict

    d = dict(species="proton", energy=0.03, frequency=F0, current=0.0,
             n_particles=32, distribution="gaussian",
             emit_nx=0.2, alpha_x=0.0, beta_x=1.0,
             emit_ny=0.2, alpha_y=0.0, beta_y=1.0,
             emit_z=0.1, alpha_z=0.0, beta_z=1.0, continuous=True)
    assert _beam_from_dict(d).periodic_phase is False
    assert _beam_from_dict({**d, "periodic_phase": True}).periodic_phase
    # And the shipped RFQ examples actually enable it.
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for name in ("lebt_plus_rfq.lgproj", "lebt_plus_rfq_66kV.lgproj"):
        p = root / "examples" / "lebt_plus_rfq" / name
        if p.exists():
            assert json.loads(p.read_text())["beam"]["periodic_phase"] \
                is True, f"{name} should ship with periodic_phase on"
