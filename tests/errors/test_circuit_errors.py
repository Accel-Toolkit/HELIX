"""Power-supply circuit draws in ErrorStudy (A5, BTL error study 2026-10).

A "BdL within x % of each other" magnet specification only matters where
several magnets share one supply: a series string is set to its mean
transfer function, so only the spread about that mean survives, while a
magnet alone on its supply absorbs its own error.  These tests pin:

* the historical one-draw-per-element stream is unchanged for definitions
  without the new options (bit for bit);
* ``about_mean`` — zero mean per series string, zero on single / unlisted;
* ``common`` — one shared value per circuit (power-supply excursion);
* ``assignment="sorted"`` — outliers to single positions, near-equal
  magnets per string, measurably smaller string spread than random;
* ``into_slot`` — the design gradient is untouched, the relative slot
  carries the draw, the effective gradient is the same physics;
* the audit reports the grouping, and invalid combinations are refused.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.errors.error_model import (
    ErrorStudy, _draw_truncated_gaussian, sorted_assignment)

CIRCUITS = {"C1": ["Q1", "Q2", "Q3"], "C2": ["Q4", "Q5", "Q6", "Q7"],
            "S1": ["Q8"]}          # Q9, Q10 on no listed circuit


def _lattice():
    lat = Lattice()
    for k in range(1, 11):
        q = Quadrupole(f"QUAD_{k:03d}", length=200.0, gradient=6.0 * (-1) ** k)
        q.label = f"Q{k}"
        lat.add(q)
        lat.add(Drift(f"D{k}", length=500.0))
    return lat


def _study(**kw):
    st = ErrorStudy(_lattice(), BeamConfig(), n_seeds=1)
    st.add_error("QUAD_*", "gradient_rel", distribution="gaussian", sigma=4e-3,
                 cutoff=2.0, **kw)
    return st


def _rel(lat):
    """Relative gradient error per label (from gradient and gradient_rel)."""
    out = {}
    for e in lat.elements:
        if isinstance(e, Quadrupole):
            g0 = 6.0 * (-1) ** int(e.label[1:])
            out[e.label] = e.gradient * (1 + e.gradient_rel) / g0 - 1
    return out


def test_definitions_without_circuits_keep_the_historical_stream():
    st = _study()
    lat = st._apply_errors(7)
    rng = np.random.default_rng((7, 0))
    want = [_draw_truncated_gaussian(rng, 4e-3, 2.0) for _ in range(10)]
    got = [r["value"] for r in st.applied_log[7]]
    assert got == want                                  # bit for bit
    assert all(r["mode"] == "scale" for r in st.applied_log[7])
    np.testing.assert_allclose(list(_rel(lat).values()), want, rtol=1e-12, atol=1e-15)


def test_independent_mode_with_circuits_draws_the_same_values():
    a = _study()._apply_errors(3)
    b = _study(circuits=CIRCUITS, circuit_mode="independent")._apply_errors(3)
    assert _rel(a) == _rel(b)


def test_about_mean_zero_mean_per_string_and_absorbed_singles():
    for seed in range(20):
        r = _rel(_study(circuits=CIRCUITS, circuit_mode="about_mean")._apply_errors(seed))
        for c in ("C1", "C2"):
            assert abs(sum(r[m] for m in CIRCUITS[c])) < 1e-15
            assert np.std([r[m] for m in CIRCUITS[c]]) > 0
        for m in ("Q8", "Q9", "Q10"):
            assert r[m] == 0.0


def test_about_mean_keeps_the_drawn_deviations():
    """The draws are those of the independent stream, minus the string mean."""
    ind = _rel(_study()._apply_errors(5))
    am = _rel(_study(circuits=CIRCUITS, circuit_mode="about_mean")._apply_errors(5))
    for c in ("C1", "C2"):
        mean = np.mean([ind[m] for m in CIRCUITS[c]])
        for m in CIRCUITS[c]:
            assert am[m] == pytest.approx(ind[m] - mean, abs=1e-15)


def test_common_mode_one_value_per_circuit():
    r = _rel(_study(circuits=CIRCUITS, circuit_mode="common")._apply_errors(11))
    for c in ("C1", "C2"):
        vals = {round(r[m], 15) for m in CIRCUITS[c]}
        assert len(vals) == 1
    assert r["Q1"] != r["Q4"] and r["Q9"] != r["Q10"]   # unlisted: own draws


def test_sorted_assignment_rule():
    rng = np.random.default_rng(0)
    batch = np.array([0.0, 5.0, 0.1, -4.0, 0.2, 0.3, 0.05, 3.0, -0.1, 0.15])
    groups = [("C1", [0, 1, 2]), ("C2", [3, 4, 5, 6]), ("S1", [7])]
    out = sorted_assignment(batch, groups, [8, 9], rng)
    assert sorted(out) == sorted(batch)                 # a permutation
    singles = {out[7], out[8], out[9]}
    assert singles == {5.0, -4.0, 3.0}                  # the three outliers
    # contiguous runs of the sorted remainder, the largest string (C2)
    # first (with two strings the block order is (4, 3): largest, then
    # the next appended after it)
    rest = sorted(set(batch) - singles)                 # 7 values
    c1 = sorted(out[[0, 1, 2]]); c2 = sorted(out[[3, 4, 5, 6]])
    assert c2 == rest[:4] and c1 == rest[4:]


def test_centred_block_order():
    from linac_gen.errors.error_model import _centred_block_order
    o = _centred_block_order([8, 8, 5, 4, 4, 3])
    assert sorted(o) == [3, 4, 4, 5, 8, 8]
    mid = len(o) // 2
    assert 8 in (o[mid - 1], o[mid])                    # largest in the middle


def test_sorting_is_per_magnet_family():
    """A long-magnet outlier can never be installed in a short-magnet
    position: batches are per family (class + length)."""
    lat = Lattice()
    for k in range(1, 9):
        q = Quadrupole(f"QUAD_{k:03d}", length=200.0 if k <= 6 else 358.0, gradient=6.0)
        q.label = f"Q{k}"
        lat.add(q)
    circ = {"A": ["Q1", "Q2", "Q3"], "B": ["Q4", "Q5", "Q6"], "L1": ["Q7"], "L2": ["Q8"]}
    st = ErrorStudy(lat, BeamConfig(), n_seeds=1)
    st.add_error("QUAD_*", "gradient_rel", "uniform", half_width=2e-3, circuits=circ,
                 circuit_mode="about_mean", assignment="sorted", into_slot=True)
    for seed in range(10):
        r = {e.label: e.gradient_rel for e in st._apply_errors(seed).elements}
        assert r["Q7"] == 0.0 and r["Q8"] == 0.0       # single supplies
        for c in ("A", "B"):
            assert abs(sum(r[m] for m in circ[c])) < 1e-15


def test_listed_only_draws_only_the_listed_supplies():
    st = _study(circuits={"C1": ["Q1", "Q2", "Q3"]}, circuit_mode="common",
                listed_only=True)
    r = _rel(st._apply_errors(4))
    assert len({round(r[m], 15) for m in ("Q1", "Q2", "Q3")}) == 1 and r["Q1"] != 0
    assert all(r[f"Q{k}"] == 0.0 for k in range(4, 11))
    assert {x["element"] for x in st.applied_log[4]} == {"QUAD_001", "QUAD_002", "QUAD_003"}


def test_circuit_map_problems_warn():
    with pytest.warns(UserWarning, match="matches no element"):
        _study(circuits={"C1": ["Q1", "QX99"]}, circuit_mode="about_mean")
    with pytest.warns(UserWarning, match="listed on"):
        _study(circuits={"C1": ["Q1", "Q2"], "C2": ["Q2", "Q3"]}, circuit_mode="about_mean")
    st = ErrorStudy(_lattice(), BeamConfig(), n_seeds=1)
    with pytest.warns(UserWarning, match="only 1 of 3 members"):
        st.add_error("QUAD_001", "gradient_rel", sigma=1e-3, circuits=CIRCUITS,
                     circuit_mode="about_mean")


@pytest.mark.parametrize("key", [(0, 0), (0, 0, 0), 2, 5, (3, 0)])
def test_rng_streams_never_replay_the_default_ones(key):
    """SeedSequence zero-pads short entropy: an untagged (seed, *key) made
    key (0,0) replay the default element stream and key 2 the default beam
    stream (adversarial-review finding)."""
    from linac_gen.errors.error_model import stream_entropy
    ref = [np.random.default_rng((7, 0)).normal(), np.random.default_rng((7, 2)).normal(),
           np.random.default_rng(7).normal(), np.random.default_rng(stream_entropy(7, 5)).normal()]
    st = ErrorStudy(_lattice(), BeamConfig(), n_seeds=1, rng_stream=key)
    st.add_error("QUAD_*", "gradient_rel", sigma=1.0, cutoff=10.0)
    st.add_beam_error("centroid_x", sigma=1.0, cutoff=10.0)
    st._apply_errors(7); st._apply_beam_errors(7)
    first_el = st.applied_log[7][0]["value"]
    first_beam = st.beam_applied_log[7][0]["value"]
    assert first_el not in ref and first_beam not in ref and first_el != first_beam


def test_sorting_reduces_the_string_spread():
    def spread(assign):
        s = []
        for seed in range(200):
            r = _rel(_study(circuits=CIRCUITS, circuit_mode="about_mean",
                            assignment=assign)._apply_errors(seed))
            s += [np.std([r[m] for m in CIRCUITS[c]]) for c in ("C1", "C2")]
        return np.mean(s)
    assert spread("sorted") < 0.6 * spread("random")


def test_into_slot_keeps_design_gradient():
    st = _study(into_slot=True)
    lat = st._apply_errors(2)
    for e in lat.elements:
        if isinstance(e, Quadrupole):
            assert e.gradient == 6.0 * (-1) ** int(e.label[1:])
            assert e.gradient_rel != 0.0
    assert all(r["mode"] == "slot-add" for r in st.applied_log[2])
    # same physics as the historical scale path
    np.testing.assert_allclose(list(_rel(lat).values()),
                               list(_rel(_study()._apply_errors(2)).values()),
                               rtol=1e-12, atol=1e-15)


def test_audit_reports_the_grouping():
    st = _study(circuits=CIRCUITS, circuit_mode="about_mean")
    row = st.audit_errors()["defs"][0]
    assert row["circuits"]["mode"] == "about_mean"
    assert row["circuits"]["groups"]["C2"] == ["QUAD_004", "QUAD_005",
                                               "QUAD_006", "QUAD_007"]
    assert row["circuits"]["unlisted"] == ["QUAD_009", "QUAD_010"]


@pytest.mark.parametrize("kw", [
    dict(circuit_mode="about_mean"),                       # no circuits
    dict(circuits=CIRCUITS, circuit_mode="median"),
    dict(circuits=CIRCUITS, assignment="sorted"),          # needs about_mean
    dict(circuits=CIRCUITS, assignment="shuffled"),
])
def test_invalid_combinations_are_refused(kw):
    with pytest.raises(ValueError):
        _study(**kw)


def test_into_slot_needs_a_relative_parameter():
    st = ErrorStudy(_lattice(), BeamConfig(), n_seeds=1)
    with pytest.raises(ValueError):
        st.add_error("QUAD_*", "dx", sigma=0.1, into_slot=True)


def test_audit_honours_listed_only():
    """listed_only defs split one pattern into disjoint supply families:
    the audit must not report them as overlapping selectors."""
    st = ErrorStudy(_lattice(), BeamConfig(), n_seeds=1)
    st.add_error("QUAD_*", "gradient_rel", sigma=1e-3, circuits={"A": ["Q1", "Q2"]},
                 circuit_mode="common", listed_only=True)
    st.add_error("QUAD_*", "gradient_rel", sigma=1e-3, circuits={"B": ["Q3", "Q4"]},
                 circuit_mode="common", listed_only=True)
    a = st.audit_errors()
    assert a["multi"] == {}
    assert a["defs"][0]["applicable"] == ["QUAD_001", "QUAD_002"]
    assert a["defs"][1]["applicable"] == ["QUAD_003", "QUAD_004"]


def test_stream_entropy_needs_a_key():
    from linac_gen.errors.error_model import stream_entropy
    with pytest.raises(ValueError):
        stream_entropy(3)
    assert len(stream_entropy(3, 5)) >= 4


def test_sorting_on_measured_values():
    """sort_noise: decisions use true + measurement error, true values are
    installed.  σ = 0 is the perfect-measurement case bit for bit; a huge
    measurement error makes sorting no better than random installation."""
    from linac_gen.errors.error_model import sorted_assignment
    rng = np.random.default_rng(1)
    batch = rng.normal(size=10)
    groups = [("C1", [0, 1, 2]), ("C2", [3, 4, 5, 6])]
    a = sorted_assignment(batch, groups, [7, 8, 9], np.random.default_rng(5))
    b = sorted_assignment(batch, groups, [7, 8, 9], np.random.default_rng(5), keys=batch.copy())
    assert np.array_equal(a, b)
    assert sorted(sorted_assignment(batch, groups, [7, 8, 9], np.random.default_rng(5),
                                    keys=batch + 1.0 * rng.normal(size=10))) == sorted(batch)

    def spread(noise):
        s = []
        for seed in range(300):
            r = _rel(_study(circuits=CIRCUITS, circuit_mode="about_mean", assignment="sorted",
                            sort_noise=noise)._apply_errors(seed))
            s += [np.std([r[m] for m in CIRCUITS[c]]) for c in ("C1", "C2")]
        return np.mean(s)
    perfect, small, huge = spread(0.0), spread(1e-3), spread(1.0)
    rand = np.mean([np.std([_rel(_study(circuits=CIRCUITS, circuit_mode="about_mean")
                                  ._apply_errors(sd))[m] for m in CIRCUITS[c]])
                    for sd in range(300) for c in ("C1", "C2")])
    assert perfect < small < 0.9 * rand
    assert huge == pytest.approx(rand, rel=0.12)
    with pytest.raises(ValueError):
        _study(circuits=CIRCUITS, circuit_mode="about_mean", sort_noise=1e-3)


def test_sort_noise_zero_is_the_previous_stream():
    a = _rel(_study(circuits=CIRCUITS, circuit_mode="about_mean", assignment="sorted")._apply_errors(9))
    b = _rel(_study(circuits=CIRCUITS, circuit_mode="about_mean", assignment="sorted",
                    sort_noise=0.0)._apply_errors(9))
    assert a == b
