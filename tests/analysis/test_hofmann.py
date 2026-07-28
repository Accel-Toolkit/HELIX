"""Hofmann stability-chart overlay (analysis/hofmann.py).

The trajectory maps the per-cell channel tunes onto the chart axes
(abscissa k_z/k_x, ordinate k_x/k_0x); the resonance lines are exact
conditions k_z/k_x = m/n while the band WIDTHS are indicative only
(they scale with 1 − depression and vanish at zero current).  Most
logic is pure — driven by a ``channel_out`` dict — so it is tested with
synthetic channel outputs; one end-to-end case confirms the wiring off a
real uncoupled channel.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.analysis.hofmann import (
    RESONANCE_RATIOS, hofmann_trajectory, resonance_bands,
)


def _uncoupled_channel(mu_x_bare, mu_x_dep, mu_z_dep, cells=None):
    n = len(mu_x_bare)
    return {
        "cells": cells if cells is not None else list(range(n)),
        "coupled_xy": False,
        "mu_x_bare_deg": np.asarray(mu_x_bare, float),
        "mu_x_dep_deg": np.asarray(mu_x_dep, float),
        "mu_z_dep_deg": np.asarray(mu_z_dep, float),
    }


def test_uncoupled_trajectory_axes():
    """Ratio = k_z/k_x, depression = k_x_dep/k_x_bare, label 'x'."""
    ch = _uncoupled_channel([80.0, 80.0], [40.0, 40.0], [20.0, 20.0])
    traj = hofmann_trajectory(ch)
    assert traj["transverse_label"] == "x"
    assert np.allclose(traj["ratio"], 0.5)          # 20/40
    assert np.allclose(traj["depression"], 0.5)     # 40/80


def test_depression_unity_at_zero_current():
    """Zero current ⇒ depressed == bare ⇒ ordinate 1.0."""
    ch = _uncoupled_channel([90.0, 90.0], [90.0, 90.0], [45.0, 45.0])
    traj = hofmann_trajectory(ch)
    assert np.allclose(traj["depression"], 1.0)


def test_nan_where_tune_unavailable():
    """Missing/zero tunes propagate to NaN, never a fabricated point."""
    ch = _uncoupled_channel([80.0, 0.0], [40.0, np.nan], [20.0, 20.0])
    traj = hofmann_trajectory(ch)
    assert np.isfinite(traj["ratio"][0])
    assert np.isnan(traj["ratio"][1]) or np.isnan(traj["depression"][1])


def test_coupled_picks_larger_bare_mode_as_transverse():
    """Coupled channel: the normal mode with the larger bare tune plays
    the transverse (k_x) role and is reported."""
    ch = {
        "cells": [0, 1],
        "coupled_xy": True,
        "mu_I_bare_deg": np.array([30.0, 30.0]),
        "mu_I_dep_deg": np.array([20.0, 20.0]),
        "mu_II_bare_deg": np.array([80.0, 80.0]),
        "mu_II_dep_deg": np.array([50.0, 50.0]),
        "mu_z_dep_deg": np.array([25.0, 25.0]),
    }
    traj = hofmann_trajectory(ch)
    assert traj["transverse_label"] == "mode II"
    assert np.allclose(traj["depression"], 50.0 / 80.0)   # II is transverse
    assert np.allclose(traj["ratio"], 25.0 / 50.0)


def test_resonance_bands_topology():
    """Six principal ratios; widths shrink to zero as depression → 1 and
    are positive under strong space charge."""
    bands = resonance_bands(emit_ratio=1.0)
    assert [b["label"] for b in bands] == [f"{m}/{n}" for m, n in RESONANCE_RATIOS]
    for b in bands:
        assert b["width"](1.0) == pytest.approx(0.0, abs=1e-12)
        assert b["width"](0.0) > 0.0
        # Monotone in depression.
        assert b["width"](0.2) > b["width"](0.8)
    ratios = [b["ratio"] for b in bands]
    assert ratios == sorted(ratios)                       # ordered 1/3 … 2/1


def test_resonance_bands_handle_bad_emit_ratio():
    """A NaN/zero emittance ratio must not crash the width heuristic."""
    for er in (float("nan"), 0.0, -1.0):
        bands = resonance_bands(emit_ratio=er)
        assert len(bands) == len(RESONANCE_RATIOS)
        assert all(np.isfinite(b["width"](0.5)) for b in bands)


def test_end_to_end_uncoupled_fodo():
    """Wiring check off a real uncoupled channel: label 'x', depression
    strictly in (0, 1] under space charge, finite where tunes exist."""
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.analysis.phase_advance import channel_phase_advance
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.matching.periodic import find_periodic_twiss
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0, aperture=20.0))
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0, aperture=20.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    init = dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=15.0,
                         phase_probe=True).run()
    ch = channel_phase_advance(res, detect_periods(lat)[0])
    traj = hofmann_trajectory(ch, res)
    assert traj["transverse_label"] == "x"
    dep = traj["depression"][np.isfinite(traj["depression"])]
    assert dep.size > 0
    assert np.all(dep > 0.0) and np.all(dep <= 1.0 + 1e-9)
    assert np.median(dep) < 1.0            # space charge depresses
