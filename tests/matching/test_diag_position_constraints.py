# tests/matching/test_diag_position_constraints.py
"""DIAG_POSITION targets as matching constraints (diagnostic matching).

Target-carrying ``Marker(is_bpm=True)`` elements mint position
constraints that fit under BOTH cost solvers (envelope results carry a
real first moment since 2026-07-19); families with no ADJUST card are
passive monitors (TraceWin semantics — fnalscl family 12); foreign
results without a ``centroid`` evaluate to fixed-length zeros.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust
from linac_gen.elements.marker import Marker
from linac_gen.elements.steerer import Steerer
from linac_gen.matching.constraints import collect_constraints
from linac_gen.matching.engine import match


def _lat(fam=1, x_t=0.1, y_t=-0.2, dm=1.0, with_adjust=True):
    lat = Lattice()
    lat.add(Drift("D1", 200.0))
    if with_adjust:
        lat.add(Adjust("A1", target=str(fam), param_idx=2))
        lat.add(Steerer("S1"))
    lat.add(Drift("D2", 200.0))
    lat.add(Marker("BPM1", is_bpm=True, diag_family=fam,
                   x_target_mm=x_t, y_target_mm=y_t, accuracy_mm=dm))
    return lat


def _mp_results(lat, cx=0.5, cy=-0.5):
    n = len(lat.elements)
    cent = [np.zeros(6)]
    for _ in range(n):
        cent.append(np.array([cx, 0.0, cy, 0.0, 0.0, 0.0]))
    return SimpleNamespace(centroid=cent,
                           element_exit_idx=list(range(1, n + 1)))


def _env_results():
    return SimpleNamespace(sigma_x=[1.0], sigma_y=[1.0])


def test_mints_per_marker_for_adjusted_family_only():
    lat = _lat(fam=11)
    lat.add(Marker("MON", is_bpm=True, diag_family=12,
                   x_target_mm=0.9, y_target_mm=0.9))   # no ADJUST 12
    cs = collect_constraints(lat)
    labels = [c.label for c in cs]
    assert "DIAG_POSITION:BPM1" in labels
    assert not any("MON" in l for l in labels)          # passive monitor
    assert all(c.requires_mp for c in cs)


def test_targetless_bpm_mints_nothing():
    lat = Lattice()
    lat.add(Marker("B", is_bpm=True))                    # bare BPM card
    lat.add(Marker("B2", is_bpm=True, origin_keyword="BPM"))
    assert collect_constraints(lat) == []


def test_weight_is_inverse_dm():
    cs = collect_constraints(_lat(dm=0.25))
    assert cs[0].weight == pytest.approx(4.0)


def test_evaluator_reads_marker_row_and_signs():
    lat = _lat(x_t=0.1, y_t=-0.2)
    (c,) = collect_constraints(lat)
    res = c.evaluate(_mp_results(lat, cx=0.5, cy=-0.5), lat)
    assert res == pytest.approx([0.5 - 0.1, -0.5 + 0.2])


def test_disabled_plane_shrinks_residual():
    lat = _lat(x_t=None, y_t=-0.2)                       # x unconstrained
    (c,) = collect_constraints(lat)
    res = c.evaluate(_mp_results(lat, cx=99.0, cy=-0.5), lat)
    assert res.shape == (1,)
    assert res == pytest.approx([-0.3])


def test_envelope_inert_fixed_length_zeros():
    lat = _lat(x_t=0.1, y_t=-0.2)
    (c,) = collect_constraints(lat)
    res = c.evaluate(_env_results(), lat)
    assert res.shape == (2,)
    assert np.allclose(res, 0.0)


def test_dead_beam_is_penalized_not_perfect():
    """A fully-lost beam records centroid zeros — that must read as a
    HUGE residual, never as a perfect on-target orbit (otherwise the
    optimizer can accept a beam-killing kick as an improvement)."""
    lat = _lat(x_t=0.0, y_t=0.0)                 # zero targets: the trap
    (c,) = collect_constraints(lat)
    r = _mp_results(lat, cx=0.0, cy=0.0)
    r.transmission = [100.0] * (len(lat.elements) + 1)
    r.transmission[-1] = 0.0                     # dead at the BPM row
    res = c.evaluate(r, lat)
    assert res.shape == (2,)
    assert np.all(res >= 1e3)
    # alive at the row → the zeros are a genuine on-target reading
    r.transmission[-1] = 42.0
    assert np.allclose(c.evaluate(r, lat), 0.0)


def test_file_override_beats_deck():
    lat = _lat(x_t=0.1, y_t=-0.2)
    m = next(e for e in lat.elements if getattr(e, "is_bpm", False))
    m.diag_target_override = (1.0, 2.0, None)
    (c,) = collect_constraints(lat)
    res = c.evaluate(_mp_results(lat, cx=1.0, cy=2.0), lat)
    assert np.allclose(res, 0.0)


def test_nan_nan_override_frees_both_planes():
    """A (None, None) override must NOT resurrect the deck targets —
    override PRESENCE wins, so the BPM mints no constraint at all."""
    lat = _lat(x_t=0.1, y_t=-0.2)
    m = next(e for e in lat.elements if getattr(e, "is_bpm", False))
    m.diag_target_override = (None, None, None)
    assert collect_constraints(lat) == []


def test_zero_file_weight_disables_constraint():
    lat = _lat(x_t=0.1, y_t=-0.2)
    m = next(e for e in lat.elements if getattr(e, "is_bpm", False))
    m.diag_target_override = (0.1, -0.2, 0.0)
    assert collect_constraints(lat) == []


def test_adjust_steerer_family_counts_as_adjusted():
    """ADJUST_STEERER N cards mint variables — their family's DIAG
    targets must mint constraints too (else match() silently no-ops)."""
    from linac_gen.elements.lattice_commands import AdjustSteerer
    lat = Lattice()
    lat.add(AdjustSteerer("AS", diag_n=5))
    lat.add(Steerer("S1"))
    lat.add(Drift("D1", 200.0))
    lat.add(Marker("B1", is_bpm=True, diag_family=5,
                   x_target_mm=0.5, y_target_mm=-0.3))
    cs = collect_constraints(lat)
    assert [c.label for c in cs] == ["DIAG_POSITION:B1"]


def test_envelope_cost_solver_fits_position_constraints():
    """Envelope results carry the centroid now — DIAG_POSITION targets
    fit under cost_solver='envelope' (noiseless, seconds-fast) instead
    of being refused by the audit."""
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, n_particles=50,
                     centroid_x=2.0, centroid_y=-1.5)
    lat = _steering_lattice(fam=1, x_t=0.5, y_t=-0.3)
    res = match(lat, cfg, algorithm="least_squares", max_iter=40,
                cost_solver="envelope")
    assert res.cost < 1e-6          # noiseless envelope → exact landing


# ---------------------------------------------------------------------------
# Dual-flavor end-to-end MP matches (house rule: test BOTH regimes).
# ---------------------------------------------------------------------------
def _steering_lattice(fam, x_t, y_t):
    lat = Lattice()
    lat.add(Drift("D1", 200.0, aperture=50.0))
    lat.add(Adjust("A1", target=str(fam), param_idx=1,
                   vmin=-0.05, vmax=0.05))               # bx_l → y plane
    lat.add(Adjust("A2", target=str(fam), param_idx=2,
                   vmin=-0.05, vmax=0.05))               # by_l → x plane
    lat.add(Steerer("S1"))
    lat.add(Drift("D2", 400.0, aperture=50.0))
    lat.add(Marker("BPM1", is_bpm=True, diag_family=fam,
                   x_target_mm=x_t, y_target_mm=y_t))
    return lat


def _offaxis_cfg():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      current=0.0, n_particles=1000,
                      distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                      emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                      emit_z=0.3, alpha_z=0.0, beta_z=10.0,
                      centroid_x=2.0, centroid_y=-1.5)


def _bpm_centroid(lat, res):
    idx = next(i for i, e in enumerate(lat.elements)
               if getattr(e, "is_bpm", False))
    row = res.element_exit_idx[idx]
    return res.centroid[row][0], res.centroid[row][2]


def test_match_mp_zero_target_bpm_flavor():
    """PIP-tool flavor: steer an off-axis beam to ZERO at the BPM."""
    lat = _steering_lattice(fam=1, x_t=0.0, y_t=0.0)
    res = match(lat, _offaxis_cfg(), algorithm="least_squares",
                cost_solver="mp", mp_n_particles=1000, mp_seed=7,
                max_iter=20)
    assert res.cost < 0.25          # (mm²) — well under the 2 mm launch
    kicks = sorted(abs(v) for v in res.x_final)
    assert all(k > 0 for k in kicks)          # steerers actually moved


def test_match_mp_nonzero_target_fnalscl_flavor():
    """fnalscl flavor: steer to a RECORDED orbit (+0.5, −0.3) mm."""
    lat = _steering_lattice(fam=1, x_t=0.5, y_t=-0.3)
    cfg = _offaxis_cfg()
    res = match(lat, cfg, algorithm="least_squares",
                cost_solver="mp", mp_n_particles=1000, mp_seed=7,
                max_iter=20)
    from copy import copy

    from linac_gen.cli.common import run_mp_sim
    cfg2 = copy(cfg)
    cfg2.n_particles = 4000
    rec, _ = run_mp_sim(lat, cfg2, None, None, seed=11)
    cx, cy = _bpm_centroid(lat, rec)
    assert cx == pytest.approx(0.5, abs=0.35)    # MP sampling noise floor
    assert cy == pytest.approx(-0.3, abs=0.35)
