"""Parallelisation regression tests.

Locks in bit-for-bit numerical output of the PIC pipeline so every
parallelisation change (FFT workers, scan-point pool, OpenMP CIC) can
be proven to preserve physics.

The numbers are captured with the same deterministic inputs the
convergence scan uses: FODO lattice, H- 2.12 MeV, 162.5 MHz, 5 mA,
5 000 Gaussian particles, seed=42, nx=ny=nz=48, grid_extent=5 σ,
step1=100/m, step2=50/m.  48³ picked so the test runs in ≤ 2 s.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.io.tracewin_parser import parse_tracewin


def _fodo_cfg():
    return BeamConfig(
        species="H-", energy=2.1226695, frequency=162.5,
        current=5.0, duty_cycle=100.0,
        n_particles=5000, distribution="gaussian", cutoff=4.0,
        emit_nx=0.21, alpha_x=1.228, beta_x=0.316,
        emit_ny=0.21, alpha_y=-0.095394, beta_y=0.113,
        emit_z=0.06231832, alpha_z=0.0, beta_z=819.05492,
    )


def _run_reference():
    """Produce the canonical end-of-lattice MP result used for parity checks."""
    lattice, _ = parse_tracewin("examples/fodo_cell.dat")
    cfg = _fodo_cfg()
    beam = create_beam(cfg, seed=42)
    sc = SpaceChargeConfig(nx=48, ny=48, nz=48, grid_extent=5.0)
    return Simulation(lattice, beam, space_charge=sc).run()


# --- Golden numbers captured on feat/interphase-and-sc-fixes (commit e23922c)
# Regenerate with:
#   pytest -q tests/parallelization/test_parity_baseline.py::test_capture_golden
# then copy the printed dict in here.  Any parallelisation change MUST keep
# these tolerances — the tight ones (1e-10 relative) catch anything that
# perturbs the numerics, not just visually-noticeable deviations.
GOLDEN = {
    # Re-pinned 2026-05-08 after adding per-plane Cholesky rescaling in
    # ``generate_gaussian`` so the *sample* covariance exactly matches the
    # requested Σ (no more 1/√N sampling drift in initial ε).  The
    # rescaling perturbs the input distribution by ~1/√N relative
    # (≈3 % at N=5 000), which propagates into a ~0.3 % shift in
    # downstream σ — captured here as the new baseline.  Earlier
    # re-pins:
    #   - Switched Hockney Poisson kernel from 1/r point-source to
    #     Qiang IGF (PRSTAB 9, 044204) — cell-volume-averaged, matches
    #     OPAL / Cheetah.  σ_x_end shifted ~−2.9 % (expected accuracy
    #     improvement, not a parallelisation bug).
    # Re-pinned 2026-07-17 for the FREQ-clock fix: the deck's ``FREQ
    # 352.21`` card now materializes as a Freq command element switching
    # the machine clock at the card (TraceWin semantics).  With the beam
    # defined at 162.5 MHz this re-expresses the longitudinal phase
    # coordinate in 352.21-MHz degrees at s=0: n_recorded +1 (the Freq
    # row), sigma_phi and emit_z scale by exactly 352.21/162.5 = 2.16745
    # (units, same physical bunch), and the transverse numbers are
    # unchanged to ~1e-13 relative (float noise from the coordinate
    # conversion) — proving the PIC physics is invariant.
    # Re-pinned 2026-07-18 for the H⁻ ion-mass fix (938.272 → 939.294 MeV
    # = m_p + 2m_e, TraceWin's value).  Every number moves by ~5e-4 relative
    # (σ_x +0.045%) — exactly the Bρ / β kinematic shift of the heavier ion,
    # validating that nothing else changed.
    "n_recorded":    23,
    "sigma_x_end":   6.301524561022442,
    "sigma_y_end":   8.336275142507612,
    "sigma_phi_end": 35.829401270129246,
    "emit_x_end":    4.432318898799213,
    "emit_y_end":    6.859666825826842,
    "emit_z_end":    0.1407073506836574,
}


def _collect(res):
    return {
        "n_recorded":    len(res.s),
        "sigma_x_end":   float(res.sigma_x[-1]),
        "sigma_y_end":   float(res.sigma_y[-1]),
        "sigma_phi_end": float(res.sigma_phi[-1]),
        "emit_x_end":    float(res.emit_x[-1]),
        "emit_y_end":    float(res.emit_y[-1]),
        "emit_z_end":    float(res.emit_z[-1]),
    }


def test_capture_golden():
    """Run and print current values — used once to freeze GOLDEN above."""
    res = _run_reference()
    vals = _collect(res)
    print("\nGOLDEN = {")
    for k, v in vals.items():
        print(f"    {k!r}: {v!r},")
    print("}")
    # Sanity: every recorded field has a value
    for k, v in vals.items():
        assert v is not None, k


@pytest.mark.skipif(
    GOLDEN["sigma_x_end"] is None,
    reason="GOLDEN not yet populated — run test_capture_golden first and pin the values",
)
def test_pic_output_bit_for_bit_matches_golden():
    """Parallelisation must preserve these values to 1e-10 relative."""
    res = _run_reference()
    vals = _collect(res)
    assert vals["n_recorded"] == GOLDEN["n_recorded"]
    for key in ("sigma_x_end", "sigma_y_end", "sigma_phi_end",
                "emit_x_end", "emit_y_end", "emit_z_end"):
        np.testing.assert_allclose(
            vals[key], GOLDEN[key], rtol=1e-10,
            err_msg=f"{key} drifted from golden — parallelisation broke parity",
        )
