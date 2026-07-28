"""Tests for :mod:`linac_gen.matching.constraints`."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    SetSize, SetSizeMax, SetSizeMin, SetTwiss, SetPosition,
)
from linac_gen.matching.constraints import collect_constraints


def _fake_results(*, sigma_x, sigma_y, sigma_phi=0.0,
                  alpha_x=0.0, beta_x=1.0, alpha_y=0.0, beta_y=1.0):
    """Stand-in for ``EnvelopeResults``: only the fields constraint
    evaluators read."""
    return SimpleNamespace(
        s=[0.0, 1000.0],
        sigma_x=[sigma_x, sigma_x],
        sigma_y=[sigma_y, sigma_y],
        sigma_phi=[sigma_phi, sigma_phi],
        alpha_x=[alpha_x, alpha_x], beta_x=[beta_x, beta_x],
        alpha_y=[alpha_y, alpha_y], beta_y=[beta_y, beta_y],
    )


# ---------------------------------------------------------------------------
def test_set_twiss_residual_zero_when_matched():
    lat = Lattice()
    lat.add(SetTwiss("C1", family="QUAD1",
                     alpha_x=1.5, beta_x=0.5,
                     alpha_y=-0.3, beta_y=0.8,
                     kax=1, kbx=1, kay=1, kby=1))
    cs = collect_constraints(lat)
    assert len(cs) == 1
    res = cs[0].evaluate(
        _fake_results(sigma_x=0, sigma_y=0,
                      alpha_x=1.5, beta_x=0.5,
                      alpha_y=-0.3, beta_y=0.8),
        lat,
    )
    assert np.allclose(res, 0.0)


def test_set_twiss_residual_picks_up_misalignment():
    lat = Lattice()
    lat.add(SetTwiss("C1", family="Q", alpha_x=2.0, beta_x=1.0, kax=1, kbx=1))
    cs = collect_constraints(lat)
    res = cs[0].evaluate(
        _fake_results(sigma_x=0, sigma_y=0, alpha_x=1.0, beta_x=2.0),
        lat,
    )
    # 1.0 - 2.0 = -1.0 ;  2.0 - 1.0 = 1.0
    assert np.allclose(sorted(res), [-1.0, 1.0])


# ---------------------------------------------------------------------------
def test_set_size_residual_at_target():
    lat = Lattice()
    lat.add(SetSize("C1", k=1, x_mm=2.0, y_mm=3.0))
    cs = collect_constraints(lat)
    res = cs[0].evaluate(_fake_results(sigma_x=2.0, sigma_y=3.0), lat)
    assert np.allclose(res, 0.0)


def test_set_size_max_inactive_when_under():
    lat = Lattice()
    lat.add(SetSizeMax("C1", k=1, n_elems=1, x_mm=5.0, y_mm=5.0))
    lat.add(Drift("D1", length=1000.0, aperture=30.0))
    cs = collect_constraints(lat)
    # σ < target → no penalty
    res = cs[0].evaluate(_fake_results(sigma_x=2.0, sigma_y=2.0), lat)
    assert np.allclose(res, 0.0)


def test_set_size_max_active_when_over():
    lat = Lattice()
    lat.add(SetSizeMax("C1", k=1, n_elems=1, x_mm=2.0, y_mm=2.0))
    lat.add(Drift("D1", length=1000.0, aperture=30.0))
    cs = collect_constraints(lat)
    res = cs[0].evaluate(_fake_results(sigma_x=4.0, sigma_y=2.0), lat)
    # σ_x: 4-2 = 2  positive ; σ_y: 0
    assert res[0] == pytest.approx(2.0)
    assert res[1] == pytest.approx(0.0)


def test_set_size_min_active_when_under():
    lat = Lattice()
    lat.add(SetSizeMin("C1", k=1, n_elems=1, x_mm=3.0, y_mm=3.0))
    lat.add(Drift("D1", length=1000.0, aperture=30.0))
    cs = collect_constraints(lat)
    # σ_x = 1.0 < 3.0  →  -1*(1-3) = 2.0 penalty
    res = cs[0].evaluate(_fake_results(sigma_x=1.0, sigma_y=5.0), lat)
    assert res[0] == pytest.approx(2.0)
    assert res[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
def test_set_position_is_mp_only():
    lat = Lattice()
    lat.add(SetPosition("C1", k=1, x_mm=0.0, y_mm=0.0))
    cs = collect_constraints(lat)
    assert cs[0].label == "SET_POSITION"
    assert cs[0].requires_mp
    # Envelope-style results carry no centroid → fixed-length inert zeros
    res = cs[0].evaluate(_fake_results(sigma_x=999, sigma_y=999), lat)
    assert res.shape == (4,)
    assert np.allclose(res, 0.0)


def test_inert_set_twiss_drops_out():
    """All k-flags zero ⇒ the constraint is filtered out at collection."""
    lat = Lattice()
    lat.add(SetTwiss("C1", family="Q", alpha_x=1.0, beta_x=1.0))
    cs = collect_constraints(lat)
    assert cs == []


# ── 2026-07 external review: window must be card-anchored ELEMENTS ───────
def test_set_size_max_window_is_element_indexed_not_rows():
    """The bound's window is 'the N elements following the card' (TW),
    mapped through element_exit_idx — extra recorder rows (substeps,
    interior markers) must NOT change which s-range is bounded.  The
    old evaluator sliced the last N recorder ROWS from the lattice end,
    so the objective depended on the recording cadence."""
    lat = Lattice()
    lat.add(SetSizeMax("C1", k=1, n_elems=1, x_mm=2.0, y_mm=99.0))
    lat.add(Drift("D1", length=500.0, aperture=30.0))   # the bounded one
    lat.add(Drift("D2", length=500.0, aperture=30.0))   # outside window
    cs = collect_constraints(lat)

    def _res(n_substep_rows):
        # rows: INPUT, D1-exit split into n rows..., D2 rows.  σ inside
        # D1's span = 1.0 (under target); σ in D2's span = 4.0 (over).
        sx = [1.0] + [1.0] * n_substep_rows + [4.0, 4.0]
        n = len(sx)
        return SimpleNamespace(
            s=list(np.linspace(0.0, 1000.0, n)),
            sigma_x=sx, sigma_y=[0.5] * n, sigma_phi=[0.0] * n,
            # one exit row per lattice element: C1 (zero-length, exits
            # at row 0), D1 (last of its substep rows), D2 (last row).
            element_exit_idx=[0, n_substep_rows, n - 1],
        )

    # Window covers only D1 (σ=1.0 < 2.0): no penalty — REGARDLESS of
    # how many substep rows D1 recorded.  Before the fix the 1-row
    # window grabbed the LAST row (D2's σ=4.0) and penalised 2.0.
    for rows in (1, 3, 7):
        res = cs[0].evaluate(_res(rows), lat)
        assert np.allclose(res, 0.0), (rows, res)


def test_set_size_max_trailing_card_disabled():
    import warnings as _w
    lat = Lattice()
    lat.add(Drift("D0", length=100.0, aperture=30.0))
    lat.add(SetSizeMax("C1", k=1, n_elems=1, x_mm=2.0))
    cs = collect_constraints(lat)
    with _w.catch_warnings(record=True) as wl:
        _w.simplefilter("always")
        res = cs[0].evaluate(_fake_results(sigma_x=9.0, sigma_y=9.0), lat)
    assert np.allclose(res, 0.0)
    assert any("no following elements" in str(w.message) for w in wl)
