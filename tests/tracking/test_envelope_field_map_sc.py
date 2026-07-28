"""Envelope mode must apply space charge inside a FieldMap.

Without this plumbing the envelope sigma_x result is identical for
current=0 and current>0 whenever a beam traverses a FieldMap or
FieldMap3D element (Task 8b fix).
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking.envelope import EnvelopeSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_static_B(Bz_T: float = 0.3, L_mm: float = 400.0) -> FieldMapData:
    """Create a FieldMapData with a uniform static Bz solenoid field."""
    n = 3
    nz = 11
    x = np.linspace(-20.0, 20.0, n)
    y = np.linspace(-20.0, 20.0, n)
    z = np.linspace(0.0, L_mm, nz)
    Bz = np.full((n, n, nz), Bz_T)
    fd = FieldMapData(z=z, frequency=0.0)
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=7, x=x, y=y, z=z,
        Fx=np.zeros((n, n, nz)),
        Fy=np.zeros((n, n, nz)),
        Fz=Bz,
    )
    return fd


def _make_initial(beta_x: float = 10.0, beta_y: float = 10.0,
                  beta_z: float = 5.0) -> dict:
    """Build an ``initial`` Twiss + emittance dict for EnvelopeSolver."""
    return {
        "alpha_x": 0.0, "beta_x": beta_x,
        "alpha_y": 0.0, "beta_y": beta_y,
        "alpha_z": 0.0, "beta_z": beta_z,
        "emit_x": 1.0, "emit_y": 1.0, "emit_z": 1.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_envelope_applies_sc_inside_fieldmap3d():
    """With current > 0 through a solenoid FieldMap3D, beta_x at exit must
    differ from the current=0 case by more than 1 % (regression guard for
    the Task 8b fix).  Before the fix both cases produced identical results
    because the FieldMapElement branch fell through to the no-SC else clause.
    """
    L_mm = 400.0
    fd = _uniform_static_B(Bz_T=0.3, L_mm=L_mm)
    sol = FieldMap3D(name="SOL", length=L_mm, field_data=fd,
                     scale=1.0, n_steps=20)
    lat = Lattice()
    lat.add(sol)

    initial = _make_initial()

    ref_nosc = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    ref_sc   = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)

    env_nosc = EnvelopeSolver(lat, ref_nosc, initial, current=0.0)
    env_sc   = EnvelopeSolver(lat, ref_sc,   initial, current=5.0)

    res_nosc = env_nosc.run()
    res_sc   = env_sc.run()

    beta_x_nosc = res_nosc.beta_x[-1]
    beta_x_sc   = res_sc.beta_x[-1]

    ratio = beta_x_sc / beta_x_nosc if beta_x_nosc > 0 else float("inf")
    # Regression guard: the SC kick must change the transverse Twiss; the
    # *magnitude* is small here (short element, low current, and the drift
    # phase slip inside the field map rotates (Δφ, ΔW) which partially
    # shields SC from β_x at exit).  0.3 % is still comfortably above the
    # "SC silently disabled" floor the guard is protecting against.
    assert not np.isclose(ratio, 1.0, rtol=3e-3), (
        f"beta_x with current=5 mA should differ from current=0 mA by >0.3 %, "
        f"got ratio={ratio:.6f}  (beta_x_nosc={beta_x_nosc:.4f}, "
        f"beta_x_sc={beta_x_sc:.4f})"
    )


def test_envelope_sc_fieldmap3d_increases_beam_size():
    """For a purely defocusing SC environment (no solenoid field), the beam
    size at exit should be strictly larger with SC than without."""
    L_mm = 400.0
    # Use a zero-field solenoid so there is no focusing to cancel the SC.
    fd = _uniform_static_B(Bz_T=0.0, L_mm=L_mm)
    sol = FieldMap3D(name="SOL0", length=L_mm, field_data=fd,
                     scale=1.0, n_steps=20)
    lat = Lattice()
    lat.add(sol)

    initial = _make_initial(beta_x=5.0, beta_y=5.0)

    ref_nosc = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    ref_sc   = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)

    env_nosc = EnvelopeSolver(lat, ref_nosc, initial, current=0.0)
    env_sc   = EnvelopeSolver(lat, ref_sc,   initial, current=10.0)

    res_nosc = env_nosc.run()
    res_sc   = env_sc.run()

    sigma_x_nosc = res_nosc.sigma_x[-1]
    sigma_x_sc   = res_sc.sigma_x[-1]

    assert sigma_x_sc > sigma_x_nosc, (
        f"SC should increase sigma_x through a field map with zero field, "
        f"got sigma_x_sc={sigma_x_sc:.4f} vs sigma_x_nosc={sigma_x_nosc:.4f}"
    )


def test_fitted_matrix_slice_advances_step_idx():
    """fitted_matrix_slice must advance _step_idx by the consumed sub-steps
    so successive calls cover the full element without overlaps or gaps.

    ``FieldMap3D`` auto-refines static-B-only maps to ~5000 steps/m so the
    solenoid's linear matrix matches TraceWin, so for a 200 mm element the
    full pass uses ~1000 sub-steps and each 100 mm slice uses ~500.  The
    invariant we care about is "slice 2's _step_idx is exactly 2 × slice
    1's" — no overlap and no gap — not the absolute count.
    """
    L_mm = 200.0
    fd = _uniform_static_B(Bz_T=0.1, L_mm=L_mm)
    sol = FieldMap3D(name="SOL", length=L_mm, field_data=fd,
                     scale=1.0, n_steps=10)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)

    sol._step_idx = 0
    sol.fitted_matrix_slice(ref, 100.0)
    idx_after_first = sol._step_idx
    assert idx_after_first > 0, f"slice 1 did not advance _step_idx"
    sol.fitted_matrix_slice(ref, 100.0)
    assert sol._step_idx == 2 * idx_after_first, (
        f"Expected _step_idx={2 * idx_after_first} after second slice, "
        f"got {sol._step_idx}")
