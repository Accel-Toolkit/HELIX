"""Tests for :class:`DiagnosticRecorder`.

These cover the two data paths used by every tracking run -- :meth:`record`
and :meth:`save_snapshot` -- plus the edge case of an all-lost beam, which
hits the len(alive)==0 branch that is easy to break.
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.diagnostics.recorder import DiagnosticRecorder


def _seeded_beam(n=200, seed=7):
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=10.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, 2.0, n)
    beam.particles[:, 1] = rng.normal(0.0, 0.5, n)
    beam.particles[:, 2] = rng.normal(0.0, 2.0, n)
    beam.particles[:, 3] = rng.normal(0.0, 0.5, n)
    beam.particles[:, 4] = rng.normal(0.0, 5.0, n)
    beam.particles[:, 5] = rng.normal(0.0, 0.01, n)
    return beam


class TestRecord:
    def test_record_populates_all_arrays(self):
        rec = DiagnosticRecorder()
        beam = _seeded_beam()
        rec.record(beam, s_position=0.0)

        # All the diagnostic arrays should now have length 1
        for attr in ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                     "emit_x", "emit_y", "emit_z", "emit_nx", "emit_ny",
                     "alpha_x", "beta_x", "alpha_y", "beta_y",
                     "halo_x", "halo_y", "transmission",
                     "ref_w_kin", "ref_phi_s", "ref_beta",
                     "ref_gamma", "ref_bg"):
            arr = getattr(rec, attr)
            assert len(arr) == 1, f"{attr} did not grow to length 1"
            assert np.isfinite(arr[0]), f"{attr}[0] is not finite: {arr[0]}"

        assert len(rec.centroid) == 1
        assert rec.centroid[0].shape == (6,)

    def test_multiple_record_calls_append(self):
        rec = DiagnosticRecorder()
        beam = _seeded_beam()
        for s in (0.0, 10.0, 20.0, 30.0):
            rec.record(beam, s_position=s)
        assert rec.s == [0.0, 10.0, 20.0, 30.0]
        assert len(rec.sigma_x) == 4

    def test_record_all_lost_beam_is_finite(self):
        """A beam with every particle lost must not produce NaN diagnostics."""
        rec = DiagnosticRecorder()
        beam = _seeded_beam(n=10)
        for i in range(beam.n_particles):
            beam.record_loss(i, s=0.0, element_name="APERTURE")
        assert beam.n_alive == 0

        rec.record(beam, s_position=0.0)

        assert rec.transmission[0] == 0.0
        for attr in ("sigma_x", "sigma_y", "emit_x", "emit_y", "emit_z",
                     "alpha_x", "beta_x", "alpha_y", "beta_y",
                     "halo_x", "halo_y"):
            val = getattr(rec, attr)[0]
            assert np.isfinite(val), f"{attr} = {val} when all particles lost"

    def test_emit_normalized_equals_emit_geometric_times_bg(self):
        rec = DiagnosticRecorder()
        beam = _seeded_beam()
        rec.record(beam, s_position=0.0)
        assert rec.emit_nx[0] == pytest.approx(
            rec.emit_x[0] * beam.ref.bg, rel=1e-12,
        )
        assert rec.emit_ny[0] == pytest.approx(
            rec.emit_y[0] * beam.ref.bg, rel=1e-12,
        )


class TestSnapshot:
    def test_snapshot_roundtrip(self):
        rec = DiagnosticRecorder()
        beam = _seeded_beam(n=50)
        rec.save_snapshot(beam, s_position=5.0)

        particles, ref = rec.beam_at(5.0)
        np.testing.assert_array_equal(particles, beam.particles)
        assert ref.w_kin == beam.ref.w_kin
        assert ref.frequency == beam.ref.frequency

    def test_snapshot_is_a_copy_not_an_alias(self):
        """Mutating the beam after the snapshot must not rewrite history."""
        rec = DiagnosticRecorder()
        beam = _seeded_beam(n=20)
        rec.save_snapshot(beam, s_position=1.0)
        original = beam.particles.copy()

        beam.particles[:] = 0.0  # trash the current beam state

        particles, _ = rec.beam_at(1.0)
        np.testing.assert_array_equal(particles, original)

    def test_beam_at_missing_key_raises(self):
        rec = DiagnosticRecorder()
        with pytest.raises(KeyError):
            rec.beam_at(1234.5)
