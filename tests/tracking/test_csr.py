"""1-D steady-state CSR kicker — physics and integration tests.

Covers:
  * net energy loss (the bunch radiates energy away)
  * the head-gains / tail-loses CSR signature
  * the R^(-2/3) bend-radius scaling law
  * no-op gates (zero current, too few particles, zero thickness bend)
  * SpaceChargeConfig validation of the csr_* fields
  * tracker integration — CSR fires inside Dipoles, not in drifts/quads
"""
import math

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.tracking.csr import CsrKicker

_DPHI, _DW = 4, 5


def _gaussian_bunch(n=40000, dphi_sigma=2.0, current_mA=4.84, seed=0):
    """A Gaussian bunch — only the longitudinal phase is populated."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=current_mA)
    rng = np.random.default_rng(seed)
    beam.particles[:, _DPHI] = rng.normal(0.0, dphi_sigma, n)
    return beam


def _z_of(beam):
    """Longitudinal coordinate (m) of each particle; larger z = head."""
    ref = beam.ref
    return (-beam.particles[:, _DPHI] * ref.beta
            * (ref.wavelength * 1e-3) / 360.0)


# ── physics ─────────────────────────────────────────────────────────────────

def test_csr_net_energy_loss():
    """The bunch as a whole must lose energy — CSR radiates power away."""
    beam = _gaussian_bunch(seed=1)
    dw0 = beam.particles[:, _DW].copy()
    CsrKicker(n_bins=200).apply(beam, Dipole("B", angle=10.0, rho=1000.0),
                                ds_mm=10.0)
    mean_shift = float(np.mean(beam.particles[:, _DW] - dw0))
    assert mean_shift < 0.0, (
        f"CSR must remove energy on average, got {mean_shift:+.3e} MeV"
    )


def test_csr_head_gains_tail_loses():
    """Classic CSR signature: the head gains (or is ~neutral), the tail
    loses the most, and head > tail."""
    beam = _gaussian_bunch(seed=2)
    z = _z_of(beam)
    dw0 = beam.particles[:, _DW].copy()
    CsrKicker(n_bins=200).apply(beam, Dipole("B", angle=10.0, rho=1000.0),
                                ds_mm=10.0)
    delta = beam.particles[:, _DW] - dw0
    head = delta[z > np.percentile(z, 75)].mean()
    tail = delta[z < np.percentile(z, 25)].mean()
    assert head > tail, f"head ({head:+.3e}) should exceed tail ({tail:+.3e})"
    assert tail < 0.0, f"tail must lose energy, got {tail:+.3e}"


def test_csr_r_scaling():
    """Energy kick scales as R^(-2/3): halving R multiplies it by 2^(2/3)."""
    def kick_std(rho_mm, seed=3):
        beam = _gaussian_bunch(seed=seed)
        dw0 = beam.particles[:, _DW].copy()
        CsrKicker(n_bins=200).apply(
            beam, Dipole("B", angle=10.0, rho=rho_mm), ds_mm=10.0)
        return float(np.std(beam.particles[:, _DW] - dw0))

    s_full = kick_std(1000.0)
    s_half = kick_std(500.0)
    ratio = s_half / s_full
    expected = 2.0 ** (2.0 / 3.0)
    assert ratio == pytest.approx(expected, rel=0.05), (
        f"R^(-2/3) scaling: got ratio {ratio:.4f}, expected {expected:.4f}"
    )


# ── no-op gates ─────────────────────────────────────────────────────────────

def test_csr_zero_current_is_noop():
    beam = _gaussian_bunch(current_mA=0.0, seed=4)
    dw0 = beam.particles[:, _DW].copy()
    CsrKicker(n_bins=200).apply(beam, Dipole("B", angle=10.0, rho=1000.0),
                                ds_mm=10.0)
    assert np.array_equal(beam.particles[:, _DW], dw0)


def test_csr_too_few_particles_is_noop():
    beam = _gaussian_bunch(n=8, seed=5)
    dw0 = beam.particles[:, _DW].copy()
    CsrKicker(n_bins=200).apply(beam, Dipole("B", angle=10.0, rho=1000.0),
                                ds_mm=10.0)
    assert np.array_equal(beam.particles[:, _DW], dw0)


def test_csr_zero_ds_is_noop():
    beam = _gaussian_bunch(seed=6)
    dw0 = beam.particles[:, _DW].copy()
    CsrKicker(n_bins=200).apply(beam, Dipole("B", angle=10.0, rho=1000.0),
                                ds_mm=0.0)
    assert np.array_equal(beam.particles[:, _DW], dw0)


# ── reproducibility / convergence ───────────────────────────────────────────

def test_csr_profile_converges_with_n_particles():
    """The mean energy loss should stabilise as N grows (smoothing keeps
    shot noise from dominating)."""
    losses = []
    for n in (10_000, 40_000, 160_000):
        beam = _gaussian_bunch(n=n, seed=7)
        dw0 = beam.particles[:, _DW].copy()
        CsrKicker(n_bins=200).apply(
            beam, Dipole("B", angle=10.0, rho=1000.0), ds_mm=10.0)
        losses.append(float(np.mean(beam.particles[:, _DW] - dw0)))
    # 40k vs 160k should agree to 10 % — the result is converged.
    assert losses[1] == pytest.approx(losses[2], rel=0.10)
    assert all(x < 0 for x in losses)


# ── config validation ───────────────────────────────────────────────────────

def test_config_accepts_csr_fields():
    cfg = SpaceChargeConfig(csr_enabled=True, csr_bins=256,
                            csr_model="1d_steady")
    assert cfg.csr_enabled is True
    assert cfg.csr_bins == 256


def test_config_rejects_bad_csr_bins():
    with pytest.raises(ValueError, match="csr_bins"):
        SpaceChargeConfig(csr_bins=0)


def test_config_rejects_bad_csr_model():
    with pytest.raises(ValueError, match="csr_model"):
        SpaceChargeConfig(csr_model="2d_transient")


def test_csrkicker_rejects_bad_model():
    with pytest.raises(ValueError, match="1d_steady"):
        CsrKicker(model="bogus")


# ── tracker integration ─────────────────────────────────────────────────────

def test_csr_fires_only_in_dipoles():
    """The tracker's _apply_csr_kick must touch the beam inside a Dipole
    and leave it untouched for a Drift."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.tracking.tracker import Tracker

    beam = _gaussian_bunch(seed=8)
    lat = Lattice()
    tracker = Tracker(lat, beam, csr_kicker=CsrKicker(n_bins=200))

    # Drift → no change.
    dw0 = beam.particles[:, _DW].copy()
    tracker._apply_csr_kick(Drift("D", length=100.0), ds_mm=10.0)
    assert np.array_equal(beam.particles[:, _DW], dw0)

    # Dipole → energy changes.
    tracker._apply_csr_kick(Dipole("B", angle=10.0, rho=1000.0), ds_mm=10.0)
    assert not np.array_equal(beam.particles[:, _DW], dw0)


def test_tracker_without_csr_kicker_is_noop():
    """A tracker built with csr_kicker=None never modifies energy via CSR."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.tracker import Tracker

    beam = _gaussian_bunch(seed=9)
    tracker = Tracker(Lattice(), beam, csr_kicker=None)
    dw0 = beam.particles[:, _DW].copy()
    tracker._apply_csr_kick(Dipole("B", angle=10.0, rho=1000.0), ds_mm=10.0)
    assert np.array_equal(beam.particles[:, _DW], dw0)


# ── bundled CSR example end-to-end ──────────────────────────────────────────

def _realistic_bunch(n: int, seed: int):
    """A proper 6-D bunch (all coords populated) — suitable for a full
    Simulation run, unlike the longitudinal-only _gaussian_bunch helper."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=5.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, 1.0, n)      # x  mm
    beam.particles[:, 1] = rng.normal(0.0, 0.3, n)      # xp mrad
    beam.particles[:, 2] = rng.normal(0.0, 1.0, n)      # y  mm
    beam.particles[:, 3] = rng.normal(0.0, 0.3, n)      # yp mrad
    beam.particles[:, 4] = rng.normal(0.0, 0.6, n)      # dphi deg
    beam.particles[:, 5] = rng.normal(0.0, 0.005, n)    # dw  MeV
    return beam


def test_csr_chicane_example_runs_end_to_end():
    """examples/csr_chicane.dat runs through a full multi-particle
    Simulation with CSR enabled.  Verifies the bundled example works and
    that CSR has a measurable, finite effect.  (The directional physics —
    net loss, head/tail signature — is covered by the unit tests above;
    a full chicane run also folds in space charge and dispersion, so this
    test only asserts the example is functional, not a precise delta.)"""
    from pathlib import Path

    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.io.tracewin_parser import parse_tracewin

    repo = Path(__file__).resolve().parents[2]
    dat = repo / "examples" / "csr_chicane.dat"
    assert dat.exists(), "examples/csr_chicane.dat is missing"

    def run(csr_on: bool):
        lat, _ = parse_tracewin(str(dat))
        sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=5.0,
                               kernel="cic", use_gpu="cpu",
                               csr_enabled=csr_on, csr_bins=200)
        return Simulation(lat, _realistic_bunch(12_000, seed=3),
                          space_charge=sc).run()

    res_off = run(csr_on=False)
    res_on = run(csr_on=True)
    # The example must run to completion with finite, sane diagnostics.
    assert len(res_on.emit_z) > 0
    assert np.isfinite(res_on.emit_z[-1]) and res_on.emit_z[-1] > 0
    assert np.isfinite(res_off.emit_z[-1]) and res_off.emit_z[-1] > 0
    # CSR must have had *some* measurable effect (same seed both runs).
    assert res_on.emit_z[-1] != res_off.emit_z[-1]
