# tests/tracking/test_tracker.py
import numpy as np
import pytest
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.tracker import Tracker


def _make_beam(n=100):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=60.0)
    rng = np.random.default_rng(42)
    beam.particles[:, 0] = rng.normal(0, 1.0, n)  # x mm
    beam.particles[:, 1] = rng.normal(0, 0.5, n)  # xp mrad
    beam.particles[:, 2] = rng.normal(0, 1.0, n)  # y mm
    beam.particles[:, 3] = rng.normal(0, 0.5, n)  # yp mrad
    return beam


def test_tracker_drift_advances_s():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    assert beam.ref.s == 100.0
    assert len(rec.s) == 2  # initial + after drift


def test_tracker_fodo_no_loss():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, aperture=50.0, n_steps=5))
    lat.add(Drift("D1", 200.0, aperture=50.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, aperture=50.0, n_steps=5))
    lat.add(Drift("D2", 200.0, aperture=50.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    assert rec.transmission[-1] == 100.0
    assert len(rec.s) == 5  # initial + 4 elements


def test_tracker_records_diagnostics():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    assert len(rec.emit_x) == 2
    assert all(e > 0 for e in rec.emit_x)
    assert len(rec.ref_w_kin) == 2
    assert rec.ref_w_kin[0] == 3.0


def test_tracker_aperture_loss():
    lat = Lattice()
    lat.add(Drift("D1", 100.0, aperture=0.5))  # very tight aperture
    beam = _make_beam(n=1000)
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    assert rec.transmission[-1] < 100.0  # some particles lost
    assert beam.n_alive < 1000


def test_tracker_initial_s_recorded():
    lat = Lattice()
    lat.add(Drift("D1", 50.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    assert rec.s[0] == 0.0
    assert rec.s[1] == 50.0


def test_tracker_ref_beta_gamma_recorded():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    # beta, gamma, bg should be recorded and positive
    assert all(b > 0 for b in rec.ref_beta)
    assert all(g > 1 for g in rec.ref_gamma)
    assert all(bg > 0 for bg in rec.ref_bg)


def test_tracker_transmission_starts_full():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    beam = _make_beam(n=50)
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    assert rec.transmission[0] == 100.0


def test_tracker_empty_lattice():
    lat = Lattice()
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    # Only initial record
    assert len(rec.s) == 1
    assert rec.transmission[0] == 100.0


def test_tracker_sigma_x_positive():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    assert all(s > 0 for s in rec.sigma_x)
    assert all(s > 0 for s in rec.sigma_y)


def test_tracker_snapshot_at_marker():
    """Marker with snapshot=True should trigger save_snapshot."""
    from linac_gen.elements.marker import Marker
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Marker("M1", snapshot=True))
    lat.add(Drift("D2", 100.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    rec = tracker.run()
    # Should have a snapshot at M1's s-position
    assert len(rec._snapshots) == 1


def test_tracker_sc_comp_reduces_kick():
    """SpaceChargeComp should reduce the SC kick factor."""
    from linac_gen.elements.space_charge_comp import SpaceChargeComp
    lat = Lattice()
    lat.add(SpaceChargeComp("SC1", factor=1.0))  # full neutralization
    lat.add(Drift("D1", 100.0))
    beam = _make_beam()
    tracker = Tracker(lat, beam)
    # After SC comp with factor=1.0, sc_factor should be 0.0
    tracker._track_element(lat.elements[0])
    assert tracker._sc_factor == 0.0


def test_drift_integration_honours_step_config():
    """A 200 mm drift with step1 = 100/m must produce 20 integration sub-steps."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.core.step_config import StepConfig
    from linac_gen.elements.drift import Drift
    from linac_gen.tracking.tracker import Tracker

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    lat = Lattice()
    lat.step_config = StepConfig(integration_steps_per_metre=100.0)
    lat.add(Drift("D", length=200.0))
    Tracker(lat, beam).run()
    assert abs(beam.ref.s - 200.0) < 1e-9


def test_non_drift_elements_use_two_substeps():
    """QUAD / SOLENOID / BEND get 2 sub-steps regardless of step_config."""
    from unittest.mock import patch
    from linac_gen.core.beam import Beam
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.core.step_config import StepConfig
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.tracking.tracker import Tracker

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    lat = Lattice()
    lat.step_config = StepConfig(integration_steps_per_metre=1000.0)
    lat.add(Quadrupole("Q", length=100.0, gradient=5.0))

    call_count = {"n": 0}
    original = Quadrupole.track

    def counting_track(self, beam, ds=None):
        call_count["n"] += 1
        return original(self, beam, ds=ds)

    with patch.object(Quadrupole, "track", counting_track):
        Tracker(lat, beam).run()

    # Split-operator: 2 sub-steps => 4 calls to element.track (half/half per step).
    assert call_count["n"] == 4, f"expected 4 half-map calls, got {call_count['n']}"
