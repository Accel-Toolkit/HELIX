"""End-to-end tests for the :class:`Simulation` facade.

These tests wire together Lattice + Beam + Simulation and check the run /
run_envelope methods complete and return sensible diagnostics. They do not
deep-check physics -- that's the job of element/tracking tests -- but they
do exercise the glue that only the top-level facade touches.
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap


def _fodo_lattice():
    lat = Lattice()
    lat.add(Drift("D_IN", length=50.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=+8.0))
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=-8.0))
    lat.add(Drift("D2", length=200.0))
    return lat


def _gaussian_beam(n=500, current=0.0, seed=1):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=current)
    rng = np.random.default_rng(seed)
    # Geometric RMS: sigma_x = 1 mm, sigma_xp = 0.3 mrad, uncorrelated
    beam.particles[:, 0] = rng.normal(0.0, 1.0, n)
    beam.particles[:, 1] = rng.normal(0.0, 0.3, n)
    beam.particles[:, 2] = rng.normal(0.0, 1.0, n)
    beam.particles[:, 3] = rng.normal(0.0, 0.3, n)
    beam.particles[:, 4] = rng.normal(0.0, 3.0, n)
    beam.particles[:, 5] = rng.normal(0.0, 0.005, n)
    return beam


class TestSimulationRun:
    def test_run_fodo_no_sc_completes(self):
        sim = Simulation(_fodo_lattice(), _gaussian_beam())
        res = sim.run()
        assert len(res.s) >= 5
        assert res.s[-1] > res.s[0]
        assert all(np.isfinite(v) for v in res.sigma_x)
        assert all(np.isfinite(v) for v in res.emit_x)

    def test_run_fodo_with_sc_completes(self):
        sc = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0)
        sim = Simulation(_fodo_lattice(), _gaussian_beam(current=20.0),
                         space_charge=sc)
        res = sim.run()
        assert all(np.isfinite(v) for v in res.sigma_x)
        # With space charge, transverse sizes should grow a bit relative
        # to no-sc reference -- but at least nothing blew up.
        assert res.sigma_x[-1] > 0

    def test_run_with_rf_gap_gains_energy(self):
        lat = _fodo_lattice()
        lat.add(RFGap("RF1", voltage=1.0, phase=-30.0, frequency=352.21))
        lat.add(Drift("D3", length=100.0))
        sim = Simulation(lat, _gaussian_beam())
        res = sim.run()
        assert res.ref_w_kin[-1] > res.ref_w_kin[0]

    def test_get_results_before_run_raises(self):
        sim = Simulation(_fodo_lattice(), _gaussian_beam())
        with pytest.raises(RuntimeError):
            sim.get_results()


class TestSimulationEnvelope:
    def test_run_envelope_completes(self):
        beam = _gaussian_beam()
        sim = Simulation(_fodo_lattice(), beam)
        sim.beam_envelope_params = {
            "alpha_x": 0.0, "beta_x": 2.0,
            "alpha_y": 0.0, "beta_y": 2.0,
            "alpha_z": 0.0, "beta_z": 1.0,
            "emit_x": 0.25, "emit_y": 0.25, "emit_z": 0.3,
        }
        res = sim.run_envelope()
        assert len(res.s) >= 5
        assert all(np.isfinite(v) for v in res.sigma_x)
        assert res.sigma_x[0] == pytest.approx(np.sqrt(2.0 * 0.25), rel=1e-6)

    def test_run_envelope_emit_conserved_in_drifts_and_quads(self):
        """Pure transfer-map tracking (no SC) conserves geometric emittance."""
        beam = _gaussian_beam(current=0.0)  # zero current => no SC
        sim = Simulation(_fodo_lattice(), beam)
        sim.beam_envelope_params = {
            "alpha_x": 0.0, "beta_x": 2.0,
            "alpha_y": 0.0, "beta_y": 2.0,
            "alpha_z": 0.0, "beta_z": 1.0,
            "emit_x": 0.25, "emit_y": 0.25, "emit_z": 0.3,
        }
        res = sim.run_envelope()
        # Emittance should be conserved to high precision across the
        # (non-accelerating, SC-off) lattice.
        assert res.emit_x[-1] == pytest.approx(res.emit_x[0], rel=1e-10)
        assert res.emit_y[-1] == pytest.approx(res.emit_y[0], rel=1e-10)
