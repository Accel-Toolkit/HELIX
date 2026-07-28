"""Regression: LatticeCommand.apply_command fires from every tracking entry point.

Builds a small lattice DRIFT → SET_BEAM_ENERGY → DRIFT and asserts that
``ref.w_kin`` jumps to the new value at the second drift's entrance in
both ``EnvelopeSolver.run`` and ``Tracker.run``.

Also confirms that ``SetBeamPhaseError`` accumulates ``phase_ref_shift``
on the live ``track_state`` and that ``SetSyncPhase`` flips
``sync_phase_mode``.
"""
from __future__ import annotations

import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    SetBeamEnergy, SetBeamPhaseError, SetSyncPhase,
)
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.tracker import Tracker


def _make_lattice(*commands):
    lat = Lattice()
    lat.add(Drift("D1", length=100.0, aperture=10.0))
    for cmd in commands:
        lat.add(cmd)
    lat.add(Drift("D2", length=100.0, aperture=10.0))
    return lat


def _ref(w_kin=3.0):
    return ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=352.21)


def _envelope_initial():
    return {
        "alpha_x": 0.0, "beta_x": 1.0, "emit_x": 1.0,
        "alpha_y": 0.0, "beta_y": 1.0, "emit_y": 1.0,
        "alpha_z": 0.0, "beta_z": 1.0, "emit_z": 1.0,
    }


# ---------------------------------------------------------------------------
# EnvelopeSolver hook
# ---------------------------------------------------------------------------
class TestEnvelopePath:
    def test_set_beam_energy_jumps_w_kin(self):
        lat = _make_lattice(SetBeamEnergy("CMD1", k=1, energy_MeV=7.5))
        solver = EnvelopeSolver(lat, _ref(3.0), _envelope_initial(), current=0.0)
        results = solver.run()
        # Final ref energy should reflect the SET_BEAM_ENERGY mutation.
        assert results.ref_w_kin[-1] == pytest.approx(7.5)
        # First record (D1 exit) is still at 3 MeV; SetBeamEnergy fires
        # before D2 → record after D2 is 7.5 MeV.
        elem_names = list(results.element_names)
        d2_idx = elem_names.index("D2")
        assert results.ref_w_kin[d2_idx] == pytest.approx(7.5)

    def test_phase_error_accumulates_on_track_state(self):
        lat = _make_lattice(
            SetBeamPhaseError("CMD1", dphi_deg=2.5),
            SetBeamPhaseError("CMD2", dphi_deg=1.5),
        )
        solver = EnvelopeSolver(lat, _ref(), _envelope_initial(), current=0.0)
        solver.run()
        assert solver.track_state.phase_ref_shift == pytest.approx(4.0)

    def test_sync_phase_sets_flag_on_track_state(self):
        lat = _make_lattice(SetSyncPhase("CMD1"))
        solver = EnvelopeSolver(lat, _ref(), _envelope_initial(), current=0.0)
        solver.run()
        assert solver.track_state.sync_phase_mode is True


# ---------------------------------------------------------------------------
# Tracker hook
# ---------------------------------------------------------------------------
class TestTrackerPath:
    def test_set_beam_energy_jumps_w_kin(self):
        lat = _make_lattice(SetBeamEnergy("CMD1", k=1, energy_MeV=6.0))
        beam = Beam(ref=_ref(2.0), n_particles=1, current=0.0)
        tracker = Tracker(lat, beam)
        tracker.run()
        assert beam.ref.w_kin == pytest.approx(6.0)

    def test_phase_error_accumulates_on_track_state(self):
        lat = _make_lattice(
            SetBeamPhaseError("CMD1", dphi_deg=3.0),
            SetBeamPhaseError("CMD2", dphi_deg=-1.0),
        )
        beam = Beam(ref=_ref(), n_particles=1, current=0.0)
        tracker = Tracker(lat, beam)
        tracker.run()
        assert tracker.track_state.phase_ref_shift == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Cross-path consistency
# ---------------------------------------------------------------------------
def test_command_free_lattice_unchanged():
    """Sanity: a lattice with no LatticeCommand still tracks identically
    on both paths (no off-by-one introduced by the hook)."""
    lat = Lattice()
    lat.add(Drift("D", length=100.0, aperture=10.0))

    # Envelope
    solver = EnvelopeSolver(lat, _ref(), _envelope_initial(), current=0.0)
    res = solver.run()
    assert res.ref_w_kin[-1] == pytest.approx(3.0)

    # Tracker
    beam = Beam(ref=_ref(), n_particles=1, current=0.0)
    Tracker(lat, beam).run()
    assert beam.ref.w_kin == pytest.approx(3.0)
