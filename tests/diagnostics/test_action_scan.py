"""Halo action scan — kernel (:mod:`linac_gen.diagnostics.action_scan`)
and its opt-in recorder hook.

External anchors (analytic, not round trips):

* Gaussian beam: N_out(n)/N = exp(-n/2) in every plane;
* uniformly filled ellipse: N_out(n)/N = 1 - n/4 (pins the factor of 2
  between the ellipse emittance W and the action J = W/2);
* <n> = 2 identically, and eps equals the plane's rms emittance (raw) or
  the Schur-complement betatron emittance (corrected);
* the counts equal those of the validated Phase-A prototype
  (per-particle correction -> tail.cs_actions -> 2W/<W> -> sort).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.diagnostics.action_scan import (
    PLANES, ActionScan, action_scan_from_results, action_scan_row,
    counts_above, default_action_grid, ellipse_sizes, gaussian_level_n,
    gaussian_percent_outside, validate_grid,
)
from linac_gen.diagnostics.moments import compute_emittance, compute_moments
from linac_gen.diagnostics.recorder import DiagnosticRecorder
from linac_gen.diagnostics.tail import cs_actions

GRID = default_action_grid()
# Correlated 2x2 blocks (alpha != 0) per plane: (u, u') = L @ (z1, z2).
_L = {0: np.array([[2.0, 0.0], [0.9, 0.5]]),
      2: np.array([[1.5, 0.0], [-0.6, 0.7]]),
      4: np.array([[8.0, 0.0], [0.002, 0.003]])}


def _gauss(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.empty((n, 6))
    for i, L in _L.items():
        p[:, i:i + 2] = rng.standard_normal((n, 2)) @ L.T
    return p


def _binomial_ok(count: int, n: int, frac: float, k_sigma: float = 4.0):
    sig = math.sqrt(n * frac * (1.0 - frac))
    return abs(count - n * frac) <= k_sigma * max(sig, 1.0)


def _at(n_value: float) -> int:
    j = int(np.searchsorted(GRID, n_value))
    assert GRID[j] == n_value, f"{n_value} is not a grid value"
    return j


# --------------------------------------------------------------------------
class TestGrid:
    def test_default_grid(self):
        assert GRID.size == 551
        assert GRID[0] == 0.0 and GRID[200] == 50.0 and GRID[201] == 51.0
        assert GRID[-1] == 400.0
        np.testing.assert_array_equal(np.diff(GRID[:201]), 0.25)
        np.testing.assert_array_equal(np.diff(GRID[200:]), 1.0)

    @pytest.mark.parametrize("bad", [[], [[0.0, 1.0]], [0.0, np.nan],
                                     [-1.0, 1.0], [0.0, 2.0, 1.0],
                                     [1.0, 1.0]])
    def test_validate_grid_rejects(self, bad):
        with pytest.raises(ValueError):
            validate_grid(bad)

    def test_validate_grid_copies(self):
        src = np.array([0.0, 1.0, 2.0])
        g = validate_grid(src)
        g[0] = 5.0
        assert src[0] == 0.0


class TestCountsAbove:
    def test_matches_brute_force_including_ties(self):
        rng = np.random.default_rng(0)
        k = np.concatenate([rng.exponential(2.0, 5000),
                            GRID[::7],                   # exact ties
                            [0.0, 0.0, 400.0, 1e6]])      # zero, end, beyond
        ref = (k[None, :] > GRID[:, None]).sum(axis=1)
        got = counts_above(k, GRID)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, ref)

    def test_empty(self):
        got = counts_above(np.array([]), GRID)
        assert got.dtype == np.int32 and not got.any()
        assert got.size == GRID.size


# --------------------------------------------------------------------------
class TestAnalyticAnchors:
    N = 400_000

    @pytest.fixture(scope="class")
    def row(self):
        return action_scan_row(_gauss(self.N, 1), GRID)

    @pytest.mark.parametrize("plane", PLANES)
    @pytest.mark.parametrize("n_value", [1.0, 2.0, 4.5, 9.25, 13.75, 18.5])
    def test_gaussian_exp_minus_n_over_2(self, row, plane, n_value):
        c = int(row[f"count_{plane}"][_at(n_value)])
        assert _binomial_ok(c, self.N, math.exp(-n_value / 2.0)), (
            plane, n_value, c / self.N, math.exp(-n_value / 2.0))

    def test_gaussian_helpers(self):
        assert gaussian_percent_outside(0.0) == 100.0
        assert gaussian_level_n(1.0) == pytest.approx(9.2103, abs=1e-4)
        assert gaussian_level_n(0.1) == pytest.approx(13.8155, abs=1e-4)
        assert gaussian_level_n(0.01) == pytest.approx(18.4207, abs=1e-4)
        np.testing.assert_allclose(
            gaussian_percent_outside(gaussian_level_n(0.1)), 0.1)

    def test_uniform_ellipse_one_minus_n_over_4(self):
        """W uniform on [0, 4 eps] for a filled ellipse: the factor-2 pin
        (an action-based n = J/eps would give 1 - n/2)."""
        rng = np.random.default_rng(2)
        n = self.N
        r = np.sqrt(rng.uniform(0.0, 1.0, n))
        th = rng.uniform(0.0, 2.0 * np.pi, n)
        p = _gauss(n, 3)
        p[:, 0:2] = np.column_stack([r * np.cos(th), r * np.sin(th)]) @ \
            _L[0].T
        row = action_scan_row(p, GRID)
        for n_value in (0.5, 1.0, 2.0, 3.0, 3.75):
            c = int(row["count_x_raw"][_at(n_value)])
            assert _binomial_ok(c, n, 1.0 - n_value / 4.0), (n_value, c / n)
        assert row["count_x_raw"][_at(4.5)] == 0
        assert row["n_max_x_raw"] == pytest.approx(4.0, abs=0.02)

    @pytest.mark.parametrize("plane", PLANES)
    def test_mean_n_is_two_and_eps_is_rms(self, plane):
        p = _gauss(20_000, 4)
        p[:, 0] += 30.0 * p[:, 5]
        n, eps = ellipse_sizes(p, plane)
        assert n.mean() == pytest.approx(2.0, abs=1e-12)
        if plane in ("x_raw", "y_raw", "z"):
            assert eps == pytest.approx(
                compute_emittance(p, plane[0]), rel=1e-12)


# --------------------------------------------------------------------------
class TestDispersion:
    def _dispersive(self, n=400_000):
        """Gaussian betatron x-x' plus D*delta with a UNIFORM delta: the raw
        plane is non-Gaussian, the betatron plane Gaussian."""
        rng = np.random.default_rng(5)
        p = _gauss(n, 6)
        delta = rng.uniform(-1.0, 1.0, n) * 0.02
        p[:, 5] = delta
        p[:, 0] += 400.0 * delta
        p[:, 1] += 60.0 * delta
        return p

    def test_corrected_meets_pin_raw_does_not(self):
        p = self._dispersive()
        n = p.shape[0]
        row = action_scan_row(p, GRID)
        j = _at(9.25)
        frac = math.exp(-9.25 / 2.0)
        assert _binomial_ok(int(row["count_x"][j]), n, frac)
        sig = math.sqrt(n * frac * (1 - frac))
        assert abs(int(row["count_x_raw"][j]) - n * frac) > 10.0 * sig
        # y has no dispersion: corrected and raw agree to the statistics
        assert _binomial_ok(int(row["count_y"][j]), n, frac)

    def test_z_untouched_by_the_correction(self):
        p = self._dispersive(50_000)
        q = p.copy()
        q[:, 0:4] = _gauss(50_000, 7)[:, 0:4]        # other transverse data
        a, b = action_scan_row(p, GRID), action_scan_row(q, GRID)
        assert a["count_z"].any()                      # a real ellipse
        assert a["eps_z"] == pytest.approx(compute_emittance(p, "z"),
                                           rel=1e-12)
        np.testing.assert_array_equal(a["count_z"], b["count_z"])
        assert a["eps_z"] == b["eps_z"]

    def test_eps_is_the_schur_complement(self):
        p = self._dispersive(50_000)
        S = compute_moments(p)["sigma_matrix"]
        for plane, (i, j) in (("x", (0, 1)), ("y", (2, 3))):
            s11 = S[i, i] - S[i, 5] ** 2 / S[5, 5]
            s12 = S[i, j] - S[i, 5] * S[j, 5] / S[5, 5]
            s22 = S[j, j] - S[j, 5] ** 2 / S[5, 5]
            _n, eps = ellipse_sizes(p, plane)
            assert eps == pytest.approx(math.sqrt(s11 * s22 - s12 ** 2),
                                        rel=1e-12)

    def test_no_energy_spread_corrected_equals_raw_bitwise(self):
        p = _gauss(20_000, 8)
        p[:, 5] = 0.0                              # Sigma_55 = 0
        row = action_scan_row(p, GRID)
        for u in ("x", "y"):
            np.testing.assert_array_equal(row[f"count_{u}"],
                                          row[f"count_{u}_raw"])
            assert row[f"eps_{u}"] == row[f"eps_{u}_raw"]
            assert row[f"n_max_{u}"] == row[f"n_max_{u}_raw"]
        assert row["eps_z"] == 0.0 and not row["count_z"].any()


# --------------------------------------------------------------------------
class TestPrototypeAndContract:
    def test_equals_the_phase_a_prototype(self):
        """Same counts as the validated prototype (per-particle dispersion
        removal, tail.cs_actions, k = 2W/<W>, sort + searchsorted)."""
        p = _gauss(100_000, 9)
        p[:, 0] += 40.0 * p[:, 5]
        p[:, 1] += 7.0 * p[:, 5]
        row = action_scan_row(p, GRID)
        q = p - p.mean(axis=0)
        d = q[:, 5]
        dd = float((d * d).mean())
        for c in (0, 1, 2, 3):
            q[:, c] -= (float((q[:, c] * d).mean()) / dd) * d
        n = p.shape[0]
        for plane in ("x", "y", "z"):
            W = cs_actions(q, plane)
            ks = np.sort(2.0 * W / W.mean())
            ref = n - np.searchsorted(ks, GRID, side="right")
            np.testing.assert_array_equal(row[f"count_{plane}"], ref)
            np.testing.assert_allclose(ellipse_sizes(p, plane)[0],
                                       2.0 * W / W.mean(), rtol=1e-9)

    def test_input_never_modified(self):
        p = _gauss(5_000, 10)
        p[:, 0] += 5.0 * p[:, 5]
        before = p.copy()
        action_scan_row(p, GRID)
        for plane in PLANES:
            ellipse_sizes(p, plane)
        np.testing.assert_array_equal(p, before)

    def test_passing_the_moments_is_bit_identical(self):
        p = _gauss(5_000, 11)
        m = compute_moments(p)
        a = action_scan_row(p, GRID)
        b = action_scan_row(p, GRID, mean=m["mean"], sigma=m["sigma_matrix"])
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    @pytest.mark.parametrize("n_part", [0, 1, 2])
    def test_too_few_particles(self, n_part):
        row = action_scan_row(_gauss(5, 12)[:n_part], GRID)
        assert row["n_alive"] == n_part
        for plane in PLANES:
            assert not row[f"count_{plane}"].any()
            assert row[f"eps_{plane}"] == 0.0 and row[f"n_max_{plane}"] == 0.0
        assert ellipse_sizes(_gauss(5, 12)[:n_part], "x") == (None, 0.0)

    def test_degenerate_planes(self):
        p = _gauss(2_000, 13)
        p[:, 5] = 0.0
        p[:, 0:2] = 0.0                        # zero x plane
        p[:, 3] = 2.0 * p[:, 2]                # rank-1 y plane
        row = action_scan_row(p, GRID)
        for plane in ("x", "x_raw", "y", "y_raw"):
            assert row[f"eps_{plane}"] == 0.0
            assert not row[f"count_{plane}"].any()
        assert ellipse_sizes(p, "y") == (None, 0.0)

    def test_non_finite_moments_give_nan_eps(self):
        p = _gauss(2_000, 14)
        p[7, 0] = np.nan
        row = action_scan_row(p, GRID)
        assert math.isnan(row["eps_x_raw"]) and math.isnan(row["eps_x"])
        assert not row["count_x_raw"].any()
        assert row["eps_y_raw"] > 0.0

    def test_non_finite_energy_never_passes_as_the_raw_plane(self):
        """A NaN in W makes S_55 NaN: the betatron planes must report NaN,
        not silently fall back to the raw plane (S_55 > eps is False for
        NaN)."""
        p = _gauss(2_000, 15)
        p[:, 0] += 100.0 * p[:, 5]
        p[3, 5] = np.nan
        row = action_scan_row(p, GRID)
        for u in ("x", "y"):
            assert math.isnan(row[f"eps_{u}"]), u
            assert not row[f"count_{u}"].any()
        assert row["eps_x_raw"] > 0.0 and math.isnan(row["eps_z"])

    def test_unknown_plane(self):
        with pytest.raises(ValueError):
            ellipse_sizes(_gauss(10, 0), "q")


# --------------------------------------------------------------------------
def _beam(n=400, seed=7, lost=()):
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=0.0)
    beam.particles[:] = _gauss(n, seed)
    for i in lost:
        beam.lost[i] = True
    return beam


class TestRecorder:
    def test_opt_in(self):
        rec = DiagnosticRecorder()
        rec.record(_beam(), 0.0)
        assert not hasattr(rec, "action_scan")

    def test_configure_resets_and_validates(self):
        rec = DiagnosticRecorder()
        rec.configure_action_scan()
        rec.record(_beam(), 0.0)
        assert len(rec.action_scan["count_x"]) == 1
        rec.configure_action_scan()
        assert rec.action_scan["count_x"] == []
        np.testing.assert_array_equal(rec.action_scan["n"], GRID)
        with pytest.raises(ValueError):
            rec.configure_action_scan([2.0, 1.0])

    def test_aligned_through_normal_dc_and_all_lost_records(self):
        rec = DiagnosticRecorder()
        rec.configure_action_scan()
        rec.record(_beam(), 0.0, "A")
        dc = _beam(seed=8)
        dc.continuous = True
        rec.record(dc, 10.0, "B")
        rec.record(_beam(lost=range(400)), 20.0, "C")
        scan = rec.action_scan
        for key, rows in scan.items():
            if key != "n":
                assert len(rows) == len(rec.s) == 3, key
        assert scan["n_alive"] == [400, 400, 0]
        assert not scan["count_x"][2].any() and scan["eps_x"][2] == 0.0
        assert rec.continuous_at == [False, True, False]
        assert scan["count_z"][1].any()              # z recorded for DC too

    def test_normalized_emittances_match_the_recorder(self):
        rec = DiagnosticRecorder()
        rec.configure_action_scan()
        beam = _beam(n=3000)
        beam.particles[:, 0] += 50.0 * beam.particles[:, 5]
        rec.record(beam, 0.0)
        s = rec.action_scan
        assert s["eps_n_z"][0] == pytest.approx(rec.emit_nz[0], rel=1e-12)
        assert s["eps_n_x_raw"][0] == pytest.approx(rec.emit_nx[0], rel=1e-12)
        assert s["eps_n_y_raw"][0] == pytest.approx(rec.emit_ny[0], rel=1e-12)
        assert s["eps_n_x"][0] == pytest.approx(
            s["eps_x"][0] * beam.ref.bg, rel=1e-15)
        assert s["eps_x"][0] < s["eps_x_raw"][0]    # dispersion removed

    def test_backtrack_reversal_keeps_rows_and_the_grid(self):
        from linac_gen.tracking.backtrack import _reverse_recorder_in_place
        rec = DiagnosticRecorder()
        rec.configure_action_scan([0.0, 1.0, 2.0])   # G == S == 3
        for i, n in enumerate((500, 400, 300)):
            rec.record(_beam(n=n, seed=20 + i), float(i))
        before = {k: list(v) for k, v in rec.action_scan.items() if k != "n"}
        _reverse_recorder_in_place(rec)
        np.testing.assert_array_equal(rec.action_scan["n"], [0.0, 1.0, 2.0])
        for key, rows in before.items():
            for a, b in zip(rec.action_scan[key], rows[::-1]):
                np.testing.assert_array_equal(a, b)
        assert rec.action_scan["n_alive"] == [300, 400, 500]


# --------------------------------------------------------------------------
class TestReadBack:
    def _rec(self):
        rec = DiagnosticRecorder()
        rec.configure_action_scan()
        for i in range(4):
            rec.record(_beam(n=800, seed=30 + i), 100.0 * i, f"E{i}")
        return rec

    def test_live_and_dict_forms_agree(self):
        rec = self._rec()
        live = action_scan_from_results(rec)
        as_dict = {"s": np.asarray(rec.s),
                   "action_scan": {k: np.asarray(v)
                                   for k, v in rec.action_scan.items()}}
        loaded = action_scan_from_results(as_dict)
        assert live.planes == PLANES
        assert live.element == ("E0", "E1", "E2", "E3")
        for p in PLANES:
            np.testing.assert_array_equal(live.counts[p], loaded.counts[p])
            np.testing.assert_array_equal(live.eps_n[p], loaded.eps_n[p])
        assert live.counts["x"].shape == (4, GRID.size)

    def test_absent_and_misaligned(self):
        assert action_scan_from_results(DiagnosticRecorder()) is None
        assert action_scan_from_results({"s": [0.0]}) is None
        rec = self._rec()
        rec.s.append(400.0)
        with pytest.raises(ValueError):
            action_scan_from_results(rec)

    def test_percent_and_level_crossing(self):
        n = np.array([0.0, 1.0, 2.0, 3.0])
        counts = np.array([[1000, 100, 10, 0],
                           [1000, 50, 5, 1],
                           [10, 5, 1, 0],
                           [0, 0, 0, 0]], dtype=np.int32)
        sc = ActionScan(
            n=n, s=np.arange(4.0), n_alive=np.array([1000, 1000, 10, 0]),
            counts={"x": counts}, eps={"x": np.array([1.0, 1.0, 1.0, 0.0])},
            eps_n={"x": np.ones(4)}, n_max={"x": np.ones(4)},
            continuous=np.zeros(4, bool))
        pct = sc.percent_outside("x")
        np.testing.assert_allclose(pct[0], [100.0, 10.0, 1.0, 0.0])
        # each step is normalised by ITS OWN particle count (10, not 1000)
        np.testing.assert_allclose(pct[2], [100.0, 50.0, 10.0, 0.0])
        assert np.isnan(pct[3]).all()
        # fewer than 1 % (10 of 1000): row 0 -> n = 3 (count 0 < 10),
        # row 1 -> n = 2 (5 < 10); row 2: 1 % of 10 is < 1 particle
        lvl = sc.level_crossing("x", 1.0)
        assert lvl[0] == 3.0 and lvl[1] == 2.0
        assert np.isnan(lvl[2]) and np.isnan(lvl[3])
        # more than 5 % beyond the whole grid -> NaN
        assert np.isnan(sc.level_crossing("x", 0.1)[1])


def test_dc_records_are_undefined_in_z_only():
    n = np.array([0.0, 1.0])
    counts = np.array([[100, 30], [100, 40]], dtype=np.int32)
    ones = np.ones(2)
    sc = ActionScan(
        n=n, s=np.arange(2.0), n_alive=np.array([100, 100]),
        counts={"x": counts, "z": counts},
        eps={"x": ones, "z": ones}, eps_n={"x": ones, "z": ones},
        n_max={"x": ones, "z": ones},
        continuous=np.array([True, False]))
    assert np.isnan(sc.percent_outside("z")[0]).all()
    np.testing.assert_allclose(sc.percent_outside("z")[1], [100.0, 40.0])
    np.testing.assert_allclose(sc.percent_outside("x")[0], [100.0, 30.0])
    assert np.isnan(sc.level_crossing("z", 50.0)[0])
    assert sc.level_crossing("x", 50.0)[0] == 1.0
