
from tests.dataguard import needs, require  # noqa: E402

require("examples/pipii/mebt/mebt.dat", "Fields")
# tests/tracking/test_backtrack_fieldmap.py
"""Backward tracking through real field maps (PIP-II MEBT bunchers).

The v1 field-map inverse uses the envelope solver's fitted LINEAR slice
matrices against a nonlinear RK4 forward — these tests FREEZE that
fidelity floor (measured 2026-07 on buncher #1, tolerances set at ~2×
the measured max error) so any regression in the slice build, replay
table, or SET_SYNC_PHASE calibration reuse trips them.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.tracker import Tracker
from linac_gen.tracking.backtrack import backtrack_distribution

MEBT_DAT = (Path(__file__).resolve().parents[2]
            / "examples" / "pipii" / "mebt" / "mebt.dat")
BUNCHER = 36            # first FIELD_MAP element (SET_SYNC_PHASE-driven)

# Frozen v1 fidelity floor: max |reconstructed − true| per column over
# a σ=(0.8 mm, 0.25 mrad, 4°, 3 keV) Gaussian at 500 particles, seed 3.
FLOOR = np.array([6e-3,     # x   (mm)     measured 2.8e-3
                  5e-2,     # x'  (mrad)   measured 2.1e-2
                  4e-2,     # y   (mm)     measured 1.7e-2
                  3e-1,     # y'  (mrad)   measured 1.5e-1
                  8e-2,     # dphi (deg)   measured 3.4e-2
                  4e-4])    # dW  (MeV)    measured 1.7e-4


@pytest.fixture(scope="module")
def mebt():
    lattice, _ = parse_tracewin(str(MEBT_DAT))
    return lattice


def _make_ref():
    return ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)


def _forward_to(lattice, beam, end, snapshot_at):
    """Forward-track elements [0..end], returning a snapshot of the
    particle array at the entrance of ``snapshot_at``."""
    tracker = Tracker(lattice, beam)
    snap = None
    for i in range(end + 1):
        if i == snapshot_at:
            snap = beam.particles.copy()
        tracker._track_element(lattice.elements[i])
    return snap


def _fresh_beam(n=500, seed=3):
    beam = Beam(ref=_make_ref(), n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    for j, s in enumerate([0.8, 0.25, 0.8, 0.25, 4.0, 0.003]):
        beam.particles[:, j] = rng.normal(0, s, n)
    return beam


def test_buncher_roundtrip_within_frozen_floor(mebt):
    """RK4 forward vs linear-slice backward through buncher #1 must stay
    inside the frozen v1 fidelity floor."""
    beam = _fresh_beam()
    snap = _forward_to(mebt, beam, end=BUNCHER, snapshot_at=BUNCHER)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backtrack_distribution(mebt, beam, _make_ref(),
                               start=BUNCHER, end=BUNCHER,
                               field_map_mode="linear")
    err = np.abs(beam.particles - snap).max(axis=0)
    assert (err < FLOOR).all(), (
        f"buncher fidelity floor exceeded: err={err} floor={FLOOR}")


def test_buncher_backward_is_deterministic(mebt):
    """SET_SYNC_PHASE calibration reuse: two independent backward walks
    from identical forward states must agree bit-for-bit (the replay
    table recalibrates the cavity to the same forward semantics)."""
    outs = []
    for _ in range(2):
        beam = _fresh_beam()
        _forward_to(mebt, beam, end=BUNCHER, snapshot_at=BUNCHER)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backtrack_distribution(mebt, beam, _make_ref(),
                                   start=BUNCHER, end=BUNCHER,
                                   field_map_mode="linear")
        outs.append(beam.particles.copy())
    np.testing.assert_array_equal(outs[0], outs[1])


def test_replay_table_matches_forward_through_fieldmaps(mebt):
    """The replay table's exit boundary must track the forward tracker's
    reference through the buncher.

    Known integrator difference: the tracker advances the reference by
    RK4 sync-particle integration while the table (like the envelope /
    matrix paths) uses ``advance_ref`` — through MEBT buncher #1 they
    differ by ~2e-6 relative (4.7e-6 MeV at 2.1 MeV).  That residual is
    absorbed by backtrack's energy re-centring (it is ~3 orders below
    ``energy_tolerance``), so the assertion pins it at 1e-5 to catch
    real calibration regressions without failing on the scheme gap."""
    from linac_gen.tracking.backtrack import build_replay_table
    beam = _fresh_beam(n=10)
    tracker = Tracker(mebt, beam)
    for i in range(BUNCHER + 1):
        tracker._track_element(mebt.elements[i])
    table = build_replay_table(mebt, _make_ref(), end=BUNCHER)
    assert beam.ref.w_kin == pytest.approx(table[-1].w_kin, rel=1e-5)
    assert beam.ref.s == pytest.approx(table[-1].s, rel=1e-12)
    assert beam.ref.frequency == pytest.approx(table[-1].frequency)


def test_quad_channel_before_buncher_exact(mebt):
    """The buncher-free MEBT front (quads/steerers/apertures only) must
    round-trip at machine precision — pins the fidelity loss to the
    field maps alone."""
    end = BUNCHER - 1
    beam = _fresh_beam()
    p_in = beam.particles.copy()
    tracker = Tracker(mebt, beam)
    for i in range(end + 1):
        tracker._track_element(mebt.elements[i])
        tracker._check_aperture(mebt.elements[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backtrack_distribution(mebt, beam, _make_ref(), start=0, end=end)
    alive = beam.alive_mask
    np.testing.assert_allclose(beam.particles[alive], p_in[alive],
                               rtol=1e-9, atol=1e-11)
