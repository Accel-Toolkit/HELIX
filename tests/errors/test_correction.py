# tests/errors/test_correction.py
"""Tests for the orbit correction module (Task 11.2)."""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.errors.correction import apply_correction
from linac_gen.tracking.tracker import Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _make_beam_factory(n=300, offset_x=0.0, offset_y=0.0, seed=42):
    """Return a beam_factory callable that creates a fresh beam each call."""
    def factory():
        ref = _ref()
        beam = Beam(ref=ref, n_particles=n, current=0.0)
        rng = np.random.default_rng(seed)
        # Small Gaussian halo
        beam.particles[:, 0] = rng.normal(offset_x, 0.5, n)  # x (mm)
        beam.particles[:, 1] = rng.normal(0, 0.2, n)           # xp (mrad)
        beam.particles[:, 2] = rng.normal(offset_y, 0.5, n)  # y (mm)
        beam.particles[:, 3] = rng.normal(0, 0.2, n)           # yp (mrad)
        return beam
    return factory


def _offset_lattice(offset_x_mm=2.0, offset_y_mm=1.0):
    """Lattice: drift + offset quad (sets centroid via dx/dy) + steerer + BPM.

    The quad is given an alignment offset; downstream we place a steerer and BPM.
    """
    lat = Lattice()
    lat.add(Drift("D0", 100.0))
    q = Quadrupole("QUAD_1", length=50.0, gradient=5.0, aperture=50.0, n_steps=3)
    q.dx = offset_x_mm
    q.dy = offset_y_mm
    lat.add(q)
    lat.add(Drift("D1", 100.0))
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D2", 100.0))
    lat.add(Marker("BPM_1"))
    return lat


def _simple_lattice():
    """Steerer → Drift → BPM (simplest case)."""
    lat = Lattice()
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D1", 500.0))
    lat.add(Marker("BPM_1"))
    return lat


# ---------------------------------------------------------------------------
# Test: apply_correction returns correct keys
# ---------------------------------------------------------------------------

def test_correction_returns_dict():
    lat = _simple_lattice()
    factory = _make_beam_factory(offset_x=2.0, offset_y=1.0)
    result = apply_correction(lat, factory)
    assert isinstance(result, dict)


def test_correction_keys_are_steerer_names():
    lat = _simple_lattice()
    factory = _make_beam_factory(offset_x=2.0)
    result = apply_correction(lat, factory)
    assert "STEER_1" in result
    assert "bx_l" in result["STEER_1"]
    assert "by_l" in result["STEER_1"]


# ---------------------------------------------------------------------------
# Test: No BPMs → no correction
# ---------------------------------------------------------------------------

def test_no_bpms_returns_empty():
    lat = Lattice()
    lat.add(Steerer("STEER_1"))
    lat.add(Drift("D1", 100.0))
    factory = _make_beam_factory()
    result = apply_correction(lat, factory, bpm_pattern="BPM_*")
    assert result == {}


def test_no_steerers_returns_empty():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Marker("BPM_1"))
    factory = _make_beam_factory()
    result = apply_correction(lat, factory, steerer_pattern="STEER_*")
    assert result == {}


# ---------------------------------------------------------------------------
# Test: One-to-one correction reduces centroid at BPM
# ---------------------------------------------------------------------------

def test_one_to_one_reduces_centroid_x():
    """After correction, |centroid_x| at BPM should be smaller than before."""
    offset = 3.0  # mm beam offset
    lat = _simple_lattice()
    factory = _make_beam_factory(offset_x=offset, n=500)

    # Measure centroid before correction
    beam_before = factory()
    rec_before = Tracker(lat, beam_before).run()
    # BPM_1 is element index 2 → recorder index 3
    bpm_idx = 2 + 1
    cx_before = abs(float(np.array(rec_before.centroid[bpm_idx])[0]))

    # Apply correction
    apply_correction(lat, factory, method="one_to_one")

    # Measure centroid after correction
    beam_after = factory()
    rec_after = Tracker(lat, beam_after).run()
    cx_after = abs(float(np.array(rec_after.centroid[bpm_idx])[0]))

    assert cx_after < cx_before, (
        f"Correction should reduce centroid: before={cx_before:.3f}, after={cx_after:.3f}"
    )


def test_one_to_one_reduces_centroid_y():
    """After correction, |centroid_y| at BPM should be smaller than before."""
    offset = 2.5  # mm
    lat = _simple_lattice()
    factory = _make_beam_factory(offset_y=offset, n=500)

    beam_before = factory()
    rec_before = Tracker(lat, beam_before).run()
    bpm_idx = 2 + 1
    cy_before = abs(float(np.array(rec_before.centroid[bpm_idx])[2]))

    apply_correction(lat, factory, method="one_to_one")

    beam_after = factory()
    rec_after = Tracker(lat, beam_after).run()
    cy_after = abs(float(np.array(rec_after.centroid[bpm_idx])[2]))

    assert cy_after < cy_before


# ---------------------------------------------------------------------------
# Test: Steerer settings change after correction
# ---------------------------------------------------------------------------

def test_steerer_by_l_changes_after_correction():
    """by_l should be non-zero after correcting a horizontal orbit offset."""
    lat = _simple_lattice()
    steer = lat.elements[0]  # STEER_1
    original_by_l = steer.by_l

    factory = _make_beam_factory(offset_x=2.0, n=300)
    apply_correction(lat, factory, method="one_to_one")

    assert steer.by_l != original_by_l


def test_steerer_bx_l_changes_after_correction():
    """bx_l should be non-zero after correcting a vertical orbit offset."""
    lat = _simple_lattice()
    steer = lat.elements[0]

    factory = _make_beam_factory(offset_y=2.0, n=300)
    apply_correction(lat, factory, method="one_to_one")

    assert steer.bx_l != 0.0


# ---------------------------------------------------------------------------
# Test: No orbit offset → no correction (steerers stay near zero)
# ---------------------------------------------------------------------------

def test_no_offset_no_large_correction():
    """Without an orbit offset, the corrective kick should be very small."""
    lat = _simple_lattice()
    steer = lat.elements[0]

    factory = _make_beam_factory(offset_x=0.0, offset_y=0.0, n=1000)
    apply_correction(lat, factory, method="one_to_one")

    # With zero centroid, corrections should be tiny (< 1e-3 T.m)
    assert abs(steer.by_l) < 1e-2
    assert abs(steer.bx_l) < 1e-2


# ---------------------------------------------------------------------------
# Test: Steerer downstream of BPM is not paired
# ---------------------------------------------------------------------------

def test_steerer_downstream_of_bpm_not_paired():
    """A steerer with no downstream BPM should not appear in corrections."""
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Marker("BPM_1"))  # BPM before steerer
    lat.add(Drift("D2", 100.0))
    lat.add(Steerer("STEER_1"))

    factory = _make_beam_factory(offset_x=2.0)
    result = apply_correction(lat, factory, method="one_to_one")
    # STEER_1 has no downstream BPM so it should not be in result
    assert "STEER_1" not in result


# ---------------------------------------------------------------------------
# Test: Multiple steerer-BPM pairs
# ---------------------------------------------------------------------------

def test_multiple_pairs_corrected():
    """Two steerer-BPM pairs are both corrected."""
    lat = Lattice()
    lat.add(Steerer("STEER_1"))
    lat.add(Drift("D1", 300.0))
    lat.add(Marker("BPM_1"))
    lat.add(Drift("D2", 300.0))
    lat.add(Steerer("STEER_2"))
    lat.add(Drift("D3", 300.0))
    lat.add(Marker("BPM_2"))

    factory = _make_beam_factory(offset_x=2.0, n=300)
    result = apply_correction(lat, factory, method="one_to_one")

    assert "STEER_1" in result
    assert "STEER_2" in result


# ---------------------------------------------------------------------------
# Test: SVD method basic correctness
# ---------------------------------------------------------------------------

def test_svd_returns_corrections():
    """SVD method returns a dict with steerer names."""
    lat = _simple_lattice()
    factory = _make_beam_factory(offset_x=2.0, n=300)
    result = apply_correction(lat, factory, method="svd")
    assert isinstance(result, dict)
    assert "STEER_1" in result


def test_svd_reduces_orbit():
    """SVD correction should reduce the centroid at the BPM."""
    offset = 3.0
    lat = _simple_lattice()
    factory = _make_beam_factory(offset_x=offset, n=500)

    # Before
    beam_before = factory()
    rec_before = Tracker(lat, beam_before).run()
    bpm_idx = 2 + 1
    cx_before = abs(float(np.array(rec_before.centroid[bpm_idx])[0]))

    apply_correction(lat, factory, method="svd")

    # After
    beam_after = factory()
    rec_after = Tracker(lat, beam_after).run()
    cx_after = abs(float(np.array(rec_after.centroid[bpm_idx])[0]))

    assert cx_after < cx_before, (
        f"SVD correction should reduce centroid: before={cx_before:.3f}, after={cx_after:.3f}"
    )


def test_invalid_method_raises():
    """Unsupported method raises ValueError."""
    lat = _simple_lattice()
    factory = _make_beam_factory()
    with pytest.raises(ValueError, match="Unknown method"):
        apply_correction(lat, factory, method="magic")


# ---------------------------------------------------------------------------
# Test: Corrections modify lattice in-place
# ---------------------------------------------------------------------------

def test_correction_modifies_lattice_in_place():
    """apply_correction modifies steerer settings directly in the passed lattice."""
    lat = _simple_lattice()
    steer = lat.elements[0]  # STEER_1
    assert steer.by_l == 0.0

    factory = _make_beam_factory(offset_x=3.0, n=300)
    apply_correction(lat, factory, method="one_to_one")

    # The steerer in the *original* lattice object should now have non-zero by_l
    assert steer.by_l != 0.0

