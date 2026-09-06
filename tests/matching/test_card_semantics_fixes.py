"""Regressions for the 2026-07 matching card-semantics fixes.

Each test pins one verified defect from the deep review of the
card→variable/constraint wiring:

* SET_SIZE_MIN took ``max()`` over its window — envelope dips below the
  declared floor were invisible to the matcher.
* ``ADJUST_BEAM_*`` flag pattern ``1 2`` produced two INDEPENDENT
  variables instead of a coupled pair (only ``2 2`` coupled).
* Beam-couple link groups were seeded at 1 — colliding with an
  ``ADJUST … n=1`` family number and lock-stepping a quad gradient to a
  beam Twiss parameter in one optimizer column.
* ``ADJUST_STEERER diag_n`` was miscounted as "the diag_n-th Steerer
  forward of the card" (correction.py reads it as a BPM number; the
  matcher now targets the next Steerer after the card, mirroring the
  card driver).
* An out-of-range integer ADJUST target silently fell through to
  name-prefix matching (``"12"`` bound to ``12MEV_CAV``).
* ``MatchResult.report()`` zipped per-DoF variables against per-COLUMN
  arrays — wrong values shown for every entry after a linked pair.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    Adjust, AdjustBeamTwiss, AdjustSteerer,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.matching.constraints import _make_set_size_bound_evaluator
from linac_gen.matching.engine import MatchResult, _link_group_index
from linac_gen.matching.variables import (
    Variable, _BEAM_LINK_GROUP_BASE, _collect_flag_axes, collect_variables,
)


# ---------------------------------------------------------------------
# SET_SIZE_MIN sees dips
# ---------------------------------------------------------------------
def test_set_size_min_detects_dip_below_floor():
    cmd = SimpleNamespace(n_elems=2, x_mm=1.0, y_mm=0.0, phi_or_z=0.0)
    results = SimpleNamespace(
        s=[0.0, 100.0],
        sigma_x=[0.5, 2.0],      # dips to 0.5 — below the 1.0 floor
        sigma_y=[1.5, 1.5],
        sigma_phi=[1.0, 1.0],
    )
    res_min = _make_set_size_bound_evaluator(cmd, sign=-1)(results, None)
    assert res_min[0] == pytest.approx(0.5), \
        "MIN floor must be checked against the SMALLEST σ in the window"
    # MAX still uses the largest sample.
    res_max = _make_set_size_bound_evaluator(cmd, sign=+1)(results, None)
    assert res_max[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------
# ADJUST_BEAM_* flag semantics
# ---------------------------------------------------------------------
def _flags(flags):
    src = SimpleNamespace(KEYWORD="ADJUST_BEAM_TWISS")
    return _collect_flag_axes(["alpha_x", "beta_x"], flags, BeamConfig(),
                              link_seed=_BEAM_LINK_GROUP_BASE + 1,
                              source=src)


def test_flag_1_then_2_couples():
    out = _flags([1, 2])
    assert len(out) == 2
    assert out[0].link_group == out[1].link_group != 0, \
        "'1 2' must couple β to the preceding α (one optimizer column)"
    _, n_cols = _link_group_index(out)
    assert n_cols == 1


def test_flag_1_1_stays_independent():
    out = _flags([1, 1])
    _, n_cols = _link_group_index(out)
    assert n_cols == 2


def test_flag_2_2_still_couples():
    out = _flags([2, 2])
    _, n_cols = _link_group_index(out)
    assert n_cols == 1


# ---------------------------------------------------------------------
# Link-group namespace collision
# ---------------------------------------------------------------------
def test_adjust_family_1_does_not_merge_with_beam_couple():
    lat = Lattice()
    lat.add(Adjust("ADJ", target="QUAD_1", param_idx=2, link_group=1))
    lat.add(Quadrupole(name="QUAD_1", length=100.0, gradient=10.0))
    lat.add(AdjustBeamTwiss("ABT", 0, 2, 2, 0, 0, 0, 0))
    lat.add(Drift("D1", 100.0))

    variables = collect_variables(lat, BeamConfig())
    assert len(variables) == 3          # 1 quad DoF + 2 beam DoF
    _, n_cols = _link_group_index(variables)
    assert n_cols == 2, \
        ("ADJUST family n=1 must NOT share a column with the beam couple "
         "(quad gradient forced equal to a Twiss value)")


# ---------------------------------------------------------------------
# ADJUST_STEERER partner resolution
# ---------------------------------------------------------------------
def test_steerer_card_targets_next_steerer_not_diag_nth():
    lat = Lattice()
    lat.add(Drift("D0", 100.0))
    lat.add(AdjustSteerer("ADJS", diag_n=7, vmax=0.02))   # BPM number 7
    lat.add(Steerer("STEER_A"))
    lat.add(Drift("D1", 100.0))
    lat.add(Steerer("STEER_B"))

    variables = collect_variables(lat, BeamConfig())
    steer_vars = [v for v in variables if isinstance(v.target, Steerer)]
    assert steer_vars, "card must resolve (old code hunted a 7th steerer)"
    assert all(v.target.name == "STEER_A" for v in steer_vars), \
        "partner is the NEXT steerer after the card, not the diag_n-th"


# ---------------------------------------------------------------------
# Out-of-range integer target raises instead of prefix-binding
# ---------------------------------------------------------------------
def test_out_of_range_index_does_not_bind_by_name_prefix():
    lat = Lattice()
    lat.add(Adjust("ADJ", target="12", param_idx=2, link_group=0))
    lat.add(Quadrupole(name="12MEV_CAV", length=100.0, gradient=5.0))

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        variables = collect_variables(lat, BeamConfig())
    assert not variables, \
        'target "12" must not silently bind to element "12MEV_CAV"'
    assert any("diagnostic index 12 not found" in str(w.message)
               for w in rec)


# ---------------------------------------------------------------------
# report() with linked variables
# ---------------------------------------------------------------------
def test_report_maps_linked_variables_to_their_column():
    q1 = SimpleNamespace(name="Q1", gradient=1.0)
    q2 = SimpleNamespace(name="Q2", gradient=1.0)
    q3 = SimpleNamespace(name="Q3", gradient=9.0)
    vs = [
        Variable(target=q1, attr="gradient", vmin=0, vmax=10, x0=1.0,
                 link_group=5, label="Q1.gradient", source=None),
        Variable(target=q2, attr="gradient", vmin=0, vmax=10, x0=1.0,
                 link_group=5, label="Q2.gradient", source=None),
        Variable(target=q3, attr="gradient", vmin=0, vmax=10, x0=9.0,
                 link_group=0, label="Q3.gradient", source=None),
    ]
    res = MatchResult(
        success=True, message="ok", n_iter=1, elapsed_s=0.0,
        x0=np.array([1.0, 9.0]),          # per-COLUMN (2 cols, 3 DoF)
        x_final=np.array([2.0, 8.0]),
        residuals=np.array([0.0]), cost=0.0,
        variables=vs, constraints=[],
    )
    text = res.report()
    lines = [ln for ln in text.splitlines() if ".gradient" in ln]
    assert len(lines) == 3                       # tail no longer dropped
    # Both linked vars show column 0's values; the independent one col 1.
    assert "2" in lines[0] and "2" in lines[1]
    assert "8" in lines[2], \
        "independent variable must show ITS column, not a shifted one"


# ---------------------------------------------------------------------
# rows() with linked variables (display contract for GUI / examples)
# ---------------------------------------------------------------------
def test_rows_maps_linked_variables_to_their_column():
    q1 = SimpleNamespace(name="Q1", gradient=1.0)
    q2 = SimpleNamespace(name="Q2", gradient=1.0)
    q3 = SimpleNamespace(name="Q3", gradient=9.0)
    vs = [
        Variable(target=q1, attr="gradient", vmin=0, vmax=10, x0=1.0,
                 link_group=5, label="Q1.gradient", source=None),
        Variable(target=q2, attr="gradient", vmin=0, vmax=10, x0=1.0,
                 link_group=5, label="Q2.gradient", source=None),
        Variable(target=q3, attr="gradient", vmin=0, vmax=10, x0=9.0,
                 link_group=0, label="Q3.gradient", source=None),
    ]
    res = MatchResult(
        success=True, message="ok", n_iter=1, elapsed_s=0.0,
        x0=np.array([1.0, 9.0]),          # per-COLUMN (2 cols, 3 DoF)
        x_final=np.array([2.0, 8.0]),
        residuals=np.array([0.0]), cost=0.0,
        variables=vs, constraints=[],
    )
    rows = res.rows()
    assert len(rows) == 3                       # one row per ADJUST DoF
    for k in range(3):
        assert rows[k][0] is vs[k]              # collection order kept
    assert [c for _v, c, _a, _b in rows] == [0, 0, 1]
    assert [x for _v, _c, x, _b in rows] == [1.0, 1.0, 9.0]
    assert [x for _v, _c, _a, x in rows] == [2.0, 2.0, 8.0]

    # Empty/cancelled result: missing columns yield nan, never raise.
    res_empty = MatchResult(
        success=False, message="cancelled by user", n_iter=0,
        elapsed_s=0.0, x0=np.array([]), x_final=np.array([]),
        residuals=np.array([]), cost=0.0, variables=vs, constraints=[],
    )
    for _var, col, x0, xf in res_empty.rows():
        assert col in (0, 1)
        assert np.isnan(x0) and np.isnan(xf)


# ---------------------------------------------------------------------
# "No SET constraints" early return seeds per optimiser column
# ---------------------------------------------------------------------
_NOCON_LINKED = """\
ADJUST QUAD 2 1 -30 30 0.5 0
DRIFT 100 30
QUAD 80 5 20
ADJUST QUAD 2 1 -30 30 0.5 0
DRIFT 200 30
QUAD 80 -5 20
DRIFT 100 30
END
"""


def test_no_constraint_result_is_per_column(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.matching import match

    # Linked deck: 2 DoF, ONE column — x0/x_final must be per column
    # (the dataclass contract), seeded like the real run (last writer).
    deck = tmp_path / "linked_nocon.dat"
    deck.write_text(_NOCON_LINKED)
    lat, _meta = parse_tracewin(str(deck))
    res = match(lat, BeamConfig())
    assert res.success is False
    assert "No SET constraints" in res.message
    assert res.x0.shape == (1,)
    assert res.x0[0] == -5.0            # last-writer seed, real baseline
    assert np.array_equal(res.x_final, res.x0)
    # The seed respects the intersected bounds (_build_x0_and_bounds
    # clips into [vmin, vmax]).
    v = res.variables[0]
    assert v.vmin <= res.x0[0] <= v.vmax
    rep = res.report()
    grad_lines = [ln for ln in rep.splitlines() if ".gradient" in ln]
    assert len(grad_lines) == 2
    assert all("-5" in ln for ln in grad_lines)

    # Same deck with an out-of-bounds seed: [1, 30] excludes -5, so the
    # column seed is clipped to the bound exactly as the real run's.
    deck_oob = tmp_path / "linked_nocon_oob.dat"
    deck_oob.write_text(_NOCON_LINKED.replace("-30 30", "1 30"))
    lat_oob, _meta = parse_tracewin(str(deck_oob))
    res_oob = match(lat_oob, BeamConfig())
    assert res_oob.x0.shape == (1,)
    assert res_oob.x0[0] == 1.0

    # Unlinked deck: one column per variable — array unchanged.
    deck_u = tmp_path / "unlinked_nocon.dat"
    deck_u.write_text(_NOCON_LINKED.replace(" 2 1 ", " 2 0 "))
    lat_u, _meta = parse_tracewin(str(deck_u))
    res_u = match(lat_u, BeamConfig())
    assert res_u.x0.tolist() == [5.0, -5.0]
    assert res_u.x_final.tolist() == [5.0, -5.0]


def test_no_constraint_unlinked_oob_seed_clips_per_column(tmp_path):
    """Unlinked deck × out-of-bounds seeds: with one column PER variable
    the bound clip applies independently to EACH column — the regime
    ``test_no_constraint_result_is_per_column`` leaves untested (its
    unlinked deck's seeds are already in bounds)."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.matching import match

    deck = tmp_path / "unlinked_nocon_oob.dat"
    deck.write_text(_NOCON_LINKED.replace(" 2 1 ", " 2 0 ")
                                 .replace("-30 30", "1 30"))
    lat, _meta = parse_tracewin(str(deck))
    res = match(lat, BeamConfig())
    assert res.success is False
    assert "No SET constraints" in res.message
    assert res.x0.shape == (2,)
    # Column 0 (G=+5) sits inside [1, 30]; column 1 (G=-5) clips to the
    # lower bound — per column, both entries independently.
    assert res.x0.tolist() == [5.0, 1.0]
    assert np.array_equal(res.x_final, res.x0)
    for v, x in zip(res.variables, res.x0):
        assert v.vmin <= x <= v.vmax
