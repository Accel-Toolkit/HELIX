"""Tests for the centroid-offset, mismatch, and duty-cycle extensions to
:func:`linac_gen.distributions.factory.create_beam`.

The base generator tests in ``tests/distributions/`` cover distribution
shape and emittance; this file specifically exercises the post-generation
plumbing added for the GUI's Beam-Center / Mismatch / Duty-cycle fields.
"""
import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam


def _base_cfg(**overrides):
    cfg = dict(
        species="proton", energy=3.0, frequency=352.21, current=10.0,
        # Big enough that centroid standard-error is << any reasonable offset.
        n_particles=50_000, distribution="gaussian",
        emit_nx=0.25, emit_ny=0.25, emit_z=0.3,
        alpha_x=0.0, beta_x=2.0,
        alpha_y=0.0, beta_y=2.0,
        alpha_z=0.0, beta_z=1.0,
    )
    cfg.update(overrides)
    return BeamConfig(**cfg)


class TestCentroidOffsets:
    # Strategy: compare offset-applied beam against an offset-free beam with
    # the SAME seed.  The particle-level difference is exactly the centroid
    # vector, so we don't need to fight random-sampling noise at all.

    def test_zero_offsets_leave_particles_unchanged(self):
        ref = create_beam(_base_cfg(), seed=42)
        zero = create_beam(_base_cfg(
            centroid_x=0.0, centroid_xp=0.0,
            centroid_y=0.0, centroid_yp=0.0,
            centroid_dphi=0.0, centroid_dw=0.0,
        ), seed=42)
        np.testing.assert_array_equal(ref.particles, zero.particles)

    def test_offset_produces_exact_shift(self):
        """Particle-wise (offset-applied - baseline) must equal the offset vector."""
        ref = create_beam(_base_cfg(), seed=42)
        offsets = (1.0, 0.1, -2.0, -0.2, 3.0, 0.005)
        shifted = create_beam(_base_cfg(
            centroid_x=offsets[0], centroid_xp=offsets[1],
            centroid_y=offsets[2], centroid_yp=offsets[3],
            centroid_dphi=offsets[4], centroid_dw=offsets[5],
        ), seed=42)
        diff = shifted.particles - ref.particles
        for col, off in enumerate(offsets):
            np.testing.assert_allclose(diff[:, col], off, rtol=1e-12, atol=1e-12)

    def test_offsets_preserve_rms_width(self):
        ref = create_beam(_base_cfg(), seed=42)
        shifted = create_beam(_base_cfg(centroid_x=10.0, centroid_y=-5.0), seed=42)
        np.testing.assert_allclose(
            ref.particles.std(axis=0), shifted.particles.std(axis=0),
            rtol=1e-12,
        )

    def test_centroid_mean_reflects_offset(self):
        """With enough particles, the measured centroid should sit at the offset."""
        beam = create_beam(_base_cfg(centroid_x=5.0, centroid_y=-3.0), seed=42)
        # At n=50k with sigma_x ~2.5 mm, SE ~ 0.011 mm => atol=0.05 mm is
        # >4 sigma, safely loose.
        assert beam.particles[:, 0].mean() == pytest.approx(5.0, abs=0.05)
        assert beam.particles[:, 2].mean() == pytest.approx(-3.0, abs=0.05)


class TestMismatch:
    def test_zero_mismatch_matches_baseline(self):
        ref = create_beam(_base_cfg(), seed=42)
        zero = create_beam(_base_cfg(mismatch_x=0.0, mismatch_y=0.0,
                                     mismatch_z=0.0), seed=42)
        np.testing.assert_array_equal(ref.particles, zero.particles)

    def test_positive_mismatch_inflates_emittance(self):
        """+10% mismatch_x -> ~10% larger geometric emittance in x."""
        from linac_gen.diagnostics.moments import compute_emittance
        base = create_beam(_base_cfg(n_particles=20_000), seed=42)
        mis = create_beam(
            _base_cfg(n_particles=20_000, mismatch_x=10.0),
            seed=42,
        )
        e0 = compute_emittance(base.particles, "x")
        e1 = compute_emittance(mis.particles, "x")
        # Emittance scales linearly with the mismatch factor (1 + m/100)
        assert e1 / e0 == pytest.approx(1.10, rel=0.03)

    def test_mismatch_per_plane_independent(self):
        """mismatch_x inflates x emittance without touching y or z."""
        from linac_gen.diagnostics.moments import compute_emittance
        base = create_beam(_base_cfg(n_particles=20_000), seed=42)
        mis = create_beam(
            _base_cfg(n_particles=20_000, mismatch_x=20.0),
            seed=42,
        )
        ex0 = compute_emittance(base.particles, "x")
        ex1 = compute_emittance(mis.particles, "x")
        ey0 = compute_emittance(base.particles, "y")
        ey1 = compute_emittance(mis.particles, "y")
        assert ex1 > ex0 * 1.15
        assert ey1 == pytest.approx(ey0, rel=0.05)


class TestDutyCycle:
    def test_default_duty_is_100_percent(self):
        beam = create_beam(_base_cfg(), seed=1)
        assert beam.duty_cycle == 100.0
        assert beam.average_current == beam.current

    def test_duty_cycle_flows_through_to_beam(self):
        beam = create_beam(_base_cfg(duty_cycle=25.0), seed=1)
        assert beam.duty_cycle == 25.0
        assert beam.average_current == pytest.approx(beam.current * 0.25)

    def test_duty_cycle_does_not_affect_sc_path(self):
        """SC uses peak (beam.current); changing duty_cycle is informational."""
        b1 = create_beam(_base_cfg(current=10.0, duty_cycle=100.0), seed=1)
        b2 = create_beam(_base_cfg(current=10.0, duty_cycle=10.0), seed=1)
        # Particle distributions identical for same seed, same peak current
        np.testing.assert_array_equal(b1.particles, b2.particles)
        assert b1.current == b2.current == 10.0
        assert b1.average_current != b2.average_current
