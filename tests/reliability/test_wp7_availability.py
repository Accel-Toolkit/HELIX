"""WP7 — availability block-diagram Monte Carlo: closed forms, the Monte
Carlo within its statistics, exact trip bins for a deterministic recovery,
every recovery/repair branch, class splits, degraded events, CSV round
trip and the surrogate template."""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.reliability.availability import (
    DEFAULT_BUDGET_BINS_S, Block, FaultClass, analytic_availability,
    block_availability, read_blocks_csv, simulate, write_blocks_csv,
    write_blocks_template)

A1 = 100.0 / 101.0


def test_closed_forms():
    single = Block("S", 100.0, 1.0)
    assert block_availability(single) == pytest.approx(A1)
    assert analytic_availability([single, Block("T", 100.0, 1.0)]) == pytest.approx(A1 * A1)
    assert analytic_availability([Block("P", 100.0, 1.0, n_parallel=2)]) == pytest.approx(1.0 - (1.0 - A1) ** 2)
    assert analytic_availability([Block("K", 100.0, 1.0, n_parallel=3, k_required=2)]) == pytest.approx(
        3 * A1 ** 2 - 2 * A1 ** 3)
    assert analytic_availability([Block("D", 100.0, 1.0, fault_class="degraded")]) == 1.0
    ar = analytic_availability([Block("R", 100.0, 5.0, fault_class="auto_rephase")])
    assert ar == pytest.approx(100.0 / (100.0 + 10.0 / 3600.0))         # class recovery replaces the MTTR
    with pytest.raises(ValueError):
        Block("bad", 0.0, 1.0)
    with pytest.raises(ValueError):
        Block("bad", 10.0, 1.0, n_parallel=2, k_required=3)


@pytest.mark.parametrize("dist", ["exponential", "lognormal", "fixed"])
def test_monte_carlo_matches_the_closed_form_within_statistics(dist):
    blocks = [Block("S", 100.0, 1.0, mttr_dist=dist), Block("T", 300.0, 2.0, mttr_dist=dist)]
    r = simulate(blocks, hours_per_year=20000.0, n_trials=120, seed=1)
    a = analytic_availability(blocks)
    se = r.availability.std(ddof=1) / math.sqrt(r.n_trials)
    assert abs(r.mean - a) < 3.5 * se + 2e-4
    assert r.analytic == pytest.approx(a) and r.percentile(5) <= r.mean <= r.percentile(95)
    assert r.events_by_block["S"] == pytest.approx(200.0, rel=0.1)
    assert r.downtime_h_by_block["S"] == pytest.approx(200.0 * 1.0, rel=0.15)


def test_parallel_and_k_of_n_branches():
    two = [Block("P", 50.0, 5.0, n_parallel=2, k_required=1)]
    r = simulate(two, hours_per_year=50000.0, n_trials=60, seed=3)
    a = analytic_availability(two)
    assert abs(r.mean - a) < 5e-4 and r.mean > 100.0 / 110.0             # better than one unit
    three = [Block("K", 50.0, 5.0, n_parallel=3, k_required=2)]
    r3 = simulate(three, hours_per_year=50000.0, n_trials=60, seed=4)
    assert abs(r3.mean - analytic_availability(three)) < 8e-4


def test_deterministic_recovery_lands_in_exactly_one_bin_and_degraded_costs_nothing():
    trips = [Block("RF", 10.0, 1.0, fault_class="auto_rephase")]     # 10 s fixed recovery
    r = simulate(trips, hours_per_year=1000.0, n_trials=20, seed=7)
    hist = r.trip_histogram
    assert hist[0] == pytest.approx(r.trips_per_year["auto_rephase"]) and np.all(hist[1:] == 0.0)
    assert r.trips_per_year["auto_rephase"] == pytest.approx(100.0, rel=0.15)
    assert r.mean == pytest.approx(1.0 - 100.0 * 10.0 / 3600.0 / 1000.0, abs=5e-5)
    deg = simulate([Block("D", 10.0, 5.0, fault_class="degraded")], hours_per_year=1000.0, n_trials=10, seed=2)
    assert deg.mean == 1.0 and deg.trips_per_year["degraded"] == 0.0
    assert deg.events_by_block["D"] > 50.0 and deg.downtime_h_by_block["D"] == 0.0


def test_class_split_and_every_recovery_distribution():
    classes = {
        "auto_rephase": FaultClass("auto_rephase", {"dist": "fixed", "value_s": 10.0}),
        "operator_retune": FaultClass("operator_retune", {"dist": "lognormal", "mean_s": 600.0, "sigma_ln": 0.5}),
        "expo": FaultClass("expo", {"dist": "exponential", "mean_s": 120.0}),
        "emp": FaultClass("emp", {"dist": "empirical", "samples_s": [30.0, 90.0, 4000.0]}),
    }
    blocks = [Block("CAV", 20.0, 10.0)]
    split = {"CAV": {"auto_rephase": 0.5, "operator_retune": 0.2, "expo": 0.1, "emp": 0.1, "downtime": 0.1}}
    r = simulate(blocks, classes, hours_per_year=4000.0, n_trials=40, seed=9, class_split=split)
    tot = sum(r.trips_per_year.values())
    assert tot == pytest.approx(200.0, rel=0.15)
    assert r.trips_per_year["auto_rephase"] == pytest.approx(0.5 * tot, rel=0.15)
    assert r.trips_per_year["downtime"] == pytest.approx(0.1 * tot, rel=0.35)
    assert r.trip_histogram.sum() == pytest.approx(tot)
    assert r.trip_histogram[-1] > 0.0                                   # the 10 h repairs overflow
    bins = DEFAULT_BUDGET_BINS_S
    assert len(r.trip_histogram) == len(bins) + 1
    with pytest.raises(ValueError, match="empirical"):
        simulate(blocks, {"emp": FaultClass("emp", {"dist": "empirical", "samples_s": []})},
                 hours_per_year=100.0, n_trials=2, class_split={"CAV": {"emp": 1.0}})
    with pytest.raises(ValueError, match="recovery dist"):
        simulate(blocks, {"x": FaultClass("x", {"dist": "weibull", "mean_s": 1.0})},
                 hours_per_year=100.0, n_trials=2, class_split={"CAV": {"x": 1.0}})


def test_sensitivity_signs_and_reproducibility():
    blocks = [Block("A", 100.0, 1.0), Block("B", 1000.0, 10.0)]
    r1 = simulate(blocks, hours_per_year=3000.0, n_trials=10, seed=5)
    r2 = simulate(blocks, hours_per_year=3000.0, n_trials=10, seed=5)
    assert np.array_equal(r1.availability, r2.availability)
    for name, d_mtbf, d_mttr in r1.sensitivity:
        assert d_mtbf > 0.0 and d_mttr < 0.0, name
    s = r1.summary()
    assert set(s) >= {"availability_mean", "trips_per_year", "trip_histogram", "sensitivity", "bins_s"}
    import json; json.dumps(s)


def test_csv_round_trip_and_template(tmp_path):
    p = tmp_path / "blocks.csv"
    blocks = [Block("A", 100.0, 1.0, "S1"), Block("B", 250.5, 0.25, "S1", "lognormal", 2, 1, "operator_retune")]
    write_blocks_csv(blocks, p, header_note="two rows")
    back = read_blocks_csv(p)
    assert back == blocks
    t = tmp_path / "template.csv"
    write_blocks_template(t)
    rows = read_blocks_csv(t)
    assert len(rows) >= 8 and any(b.fault_class == "auto_rephase" for b in rows)
    assert t.read_text(encoding="utf-8").startswith("# Reliability block diagram — SURROGATE")
    empty = tmp_path / "empty.csv"; empty.write_text("name,mtbf_h,mttr_h\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no block rows"):
        read_blocks_csv(empty)
    r = simulate(rows, hours_per_year=5000.0, n_trials=5, seed=0)
    assert 0.0 < r.mean < 1.0


def test_steady_state_start_removes_the_finite_horizon_bias():
    """A short horizon with a long repair: units starting up at t = 0
    over-estimate availability by ~MTTR^2/(MTBF T); the steady-state start
    keeps the Monte Carlo on the closed form."""
    blocks = [Block("CM", 20000.0, 240.0)]
    r = simulate(blocks, hours_per_year=5000.0, n_trials=4000, seed=11)
    a = analytic_availability(blocks)
    se = r.availability.std(ddof=1) / math.sqrt(r.n_trials)
    assert abs(r.mean - a) < 3.0 * se + 1e-4
    rows = read_blocks_csv.__wrapped__ if hasattr(read_blocks_csv, "__wrapped__") else None
    assert Block("X", 1.0, 1.0, scenario_class="S1").scenario_class == "S1"
