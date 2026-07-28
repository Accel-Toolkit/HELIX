"""Tests for beam-input error perturbations (BeamErrorDef).

The ErrorStudy now produces a per-seed BeamConfig copy with random
centroid / emittance / mismatch / current perturbations applied
before each Monte Carlo realisation runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.errors.beam_error import BeamErrorDef
from linac_gen.errors.error_model import ErrorStudy


def _minimal_lattice() -> Lattice:
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    return lat


def _minimal_config(**overrides) -> BeamConfig:
    base = dict(
        species="proton", energy=10.0, frequency=325.0,
        n_particles=500, distribution="gaussian",
        emit_nx=0.25, beta_x=2.0,
        emit_ny=0.25, beta_y=2.0,
        emit_z=0.30, beta_z=3.0,
    )
    base.update(overrides)
    return BeamConfig(**base)


# ---------------------------------------------------------------------------
class TestBeamErrorDefShape:
    def test_default_distribution_is_gaussian(self):
        be = BeamErrorDef(parameter="centroid_x", sigma=0.1)
        assert be.distribution == "gaussian"
        assert be.cutoff == 3.0


# ---------------------------------------------------------------------------
class TestBeamErrorApply:
    def test_no_beam_errors_no_op(self):
        """ErrorStudy with no beam errors → original BeamConfig is reused."""
        cfg = _minimal_config(centroid_x=2.0)
        study = ErrorStudy(_minimal_lattice(), cfg, n_seeds=1)
        out = study._apply_beam_errors(seed=0)
        assert out is cfg

    def test_centroid_x_perturbation_applied(self):
        """add_beam_error('centroid_x', sigma=0.5) → seed-dependent shift."""
        cfg = _minimal_config(centroid_x=0.0)
        study = ErrorStudy(_minimal_lattice(), cfg, n_seeds=1)
        study.add_beam_error("centroid_x", distribution="gaussian", sigma=0.5)
        a = study._apply_beam_errors(seed=0)
        b = study._apply_beam_errors(seed=1)
        # Different seeds → different perturbations.
        assert a.centroid_x != b.centroid_x
        # Original config unchanged.
        assert cfg.centroid_x == 0.0

    def test_centroid_x_within_cutoff(self):
        """Gaussian draws are clipped at cutoff·σ."""
        cfg = _minimal_config(centroid_x=0.0)
        study = ErrorStudy(_minimal_lattice(), cfg, n_seeds=1)
        study.add_beam_error("centroid_x", sigma=0.1, cutoff=2.0)
        for s in range(50):
            cfg_s = study._apply_beam_errors(seed=s)
            # Clipped at ±2σ = ±0.2.
            assert -0.2001 <= cfg_s.centroid_x <= 0.2001

    def test_uniform_within_half_width(self):
        cfg = _minimal_config()
        study = ErrorStudy(_minimal_lattice(), cfg, n_seeds=1)
        study.add_beam_error("centroid_x", distribution="uniform",
                             half_width=0.5)
        for s in range(50):
            cfg_s = study._apply_beam_errors(seed=s)
            assert -0.5 <= cfg_s.centroid_x <= 0.5

    def test_emit_nx_rel_scales_emittance(self):
        cfg = _minimal_config(emit_nx=0.30)
        study = ErrorStudy(_minimal_lattice(), cfg, n_seeds=1)
        study.add_beam_error("emit_nx_rel", sigma=0.10)
        # Many seeds — average should be ~ design × 1.0 ± noise.
        emits = [study._apply_beam_errors(seed=s).emit_nx
                 for s in range(200)]
        # Mean should be close to design (within sampling noise σ/√N).
        assert abs(np.mean(emits) - 0.30) < 0.30 * 0.10 * 3 / np.sqrt(200)
        # Spread should be ~ design · σ = 0.030.
        assert abs(np.std(emits) - 0.030) < 0.015

    def test_mismatch_x_additive(self):
        cfg = _minimal_config(mismatch_x=5.0)
        study = ErrorStudy(_minimal_lattice(), cfg, n_seeds=1)
        study.add_beam_error("mismatch_x", sigma=2.0)
        cfg_s = study._apply_beam_errors(seed=42)
        # Result is design + draw, draw is bounded by 3σ = 6.
        assert -1.0 <= cfg_s.mismatch_x <= 11.0
        assert cfg_s.mismatch_x != 5.0  # actually got perturbed


# ---------------------------------------------------------------------------
class TestLatticeBeamErrorAbsorption:
    """An ErrorStudy auto-absorbs lattice.beam_errors at construction."""

    def test_lattice_attached_beam_error_used(self):
        lat = _minimal_lattice()
        lat.beam_errors = [BeamErrorDef(parameter="centroid_y", sigma=0.3)]
        cfg = _minimal_config()
        study = ErrorStudy(lat, cfg, n_seeds=1)
        a = study._apply_beam_errors(seed=0)
        b = study._apply_beam_errors(seed=1)
        assert a.centroid_y != b.centroid_y

    def test_lattice_attached_element_error_used(self):
        from linac_gen.errors.error_model import ErrorDef
        lat = Lattice()
        from linac_gen.elements.quadrupole import Quadrupole
        lat.add(Quadrupole("Q1", length=100.0, gradient=5.0))
        lat.errors = [ErrorDef(pattern="Q1", parameter="dx", sigma=0.1)]
        cfg = _minimal_config()
        study = ErrorStudy(lat, cfg, n_seeds=1)
        # _apply_errors mutates a deepcopy; verify dx is non-zero on the copy.
        lat_copy = study._apply_errors(seed=42)
        q = next(e for e in lat_copy.elements if e.name == "Q1")
        assert q.dx != 0.0


# ---------------------------------------------------------------------------
class TestMCConvergenceCentroid:
    """1000-seed sample: mean centroid → 0, std → input σ."""

    def test_mean_and_std_converge(self):
        """For σ_dx = 0.1mm and 1000 seeds, sample mean → 0, sample σ → 0.1."""
        cfg = _minimal_config(centroid_x=0.0)
        study = ErrorStudy(_minimal_lattice(), cfg, n_seeds=1)
        study.add_beam_error("centroid_x", sigma=0.1, cutoff=4.0)
        draws = np.array([
            study._apply_beam_errors(seed=s).centroid_x for s in range(1000)
        ])
        assert abs(draws.mean()) < 0.01
        assert abs(draws.std() - 0.1) < 0.02
