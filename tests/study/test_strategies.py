"""Strategy expansion: counts, determinism, and failure modes."""
import numpy as np
import pytest

from linac_gen.study.spec import ParamSpec, StudySpec
from linac_gen.study.strategies import expand_runs, param_values


def _spec(strategy="grid", repeats=1, n_samples=None, params=None):
    if params is None:
        params = [
            ParamSpec(selector="@2.gradient", start=6.0, stop=9.0, n=4,
                      baseline=7.5),
            ParamSpec(selector="current", start=0.0, stop=8.0, n=3,
                      baseline=5.0),
        ]
    return StudySpec(name="t", input="x.dat", strategy=strategy,
                     parameters=params, repeats=repeats,
                     n_samples=n_samples)


class TestValues:
    def test_linspace(self):
        p = ParamSpec(selector="a.b", start=1.0, stop=3.0, n=3)
        assert param_values(p) == [1.0, 2.0, 3.0]

    def test_explicit_list_wins(self):
        p = ParamSpec(selector="a.b", values=[4, 5])
        assert param_values(p) == [4.0, 5.0]

    def test_log_spacing(self):
        p = ParamSpec(selector="a.b", start=1.0, stop=100.0, n=3,
                      spacing="log")
        assert param_values(p) == pytest.approx([1.0, 10.0, 100.0])

    def test_log_rejects_nonpositive(self):
        p = ParamSpec(selector="a.b", start=0.0, stop=1.0, n=3,
                      spacing="log")
        with pytest.raises(ValueError):
            param_values(p)


class TestStrategies:
    def test_grid_count_and_order(self):
        runs = expand_runs(_spec("grid"))
        assert len(runs) == 4 * 3
        assert [r.index for r in runs] == list(range(12))
        # spec-order Cartesian: first param varies slowest (3 runs per
        # value of the 4-valued first parameter)
        assert runs[0].params == (("@2.gradient", 6.0), ("current", 0.0))
        assert runs[2].params == (("@2.gradient", 6.0), ("current", 8.0))
        assert runs[3].params == (("@2.gradient", 7.0), ("current", 0.0))

    def test_oat_reference_first(self):
        runs = expand_runs(_spec("oat"))
        assert len(runs) == 1 + 4 + 3
        ref = runs[0]
        assert dict(ref.params) == {"@2.gradient": 7.5, "current": 5.0}
        # every non-reference run varies exactly one parameter
        for r in runs[1:]:
            off = [1 for (sel, v) in r.params
                   if v != dict(ref.params)[sel]]
            assert sum(off) <= 1

    def test_oat_requires_baselines(self):
        s = _spec("oat")
        s.parameters[0].baseline = None
        with pytest.raises(ValueError, match="baseline"):
            expand_runs(s)

    def test_zip_equal_lengths(self):
        s = _spec("zip", params=[
            ParamSpec(selector="a.b", values=[1, 2, 3]),
            ParamSpec(selector="c.d", values=[10, 20, 30])])
        runs = expand_runs(s)
        assert len(runs) == 3
        assert runs[1].params == (("a.b", 2.0), ("c.d", 20.0))

    def test_zip_mismatch_raises(self):
        s = _spec("zip", params=[
            ParamSpec(selector="a.b", values=[1, 2, 3]),
            ParamSpec(selector="c.d", values=[10, 20])])
        with pytest.raises(ValueError, match="equal-length"):
            expand_runs(s)

    def test_lhs_reproducible_and_bounded(self):
        s = _spec("lhs", n_samples=16)
        r1 = expand_runs(s)
        r2 = expand_runs(s)
        assert r1 == r2                       # deterministic
        assert len(r1) == 16
        for r in r1:
            vals = dict(r.params)
            assert 6.0 <= vals["@2.gradient"] <= 9.0
            assert 0.0 <= vals["current"] <= 8.0

    def test_random_seed_changes_draw(self):
        s1 = _spec("random", n_samples=8)
        s2 = _spec("random", n_samples=8)
        s2.sampler_seed = 999
        assert expand_runs(s1) != expand_runs(s2)

    def test_repeats_expand_seeds(self):
        s = _spec("grid", repeats=3)
        runs = expand_runs(s)
        assert len(runs) == 12 * 3
        assert {r.seed for r in runs[:3]} == {42, 43, 44}
        assert [r.index for r in runs] == list(range(36))

    def test_tags_are_filesystem_safe(self):
        for r in expand_runs(_spec("grid")):
            assert "/" not in r.tag and " " not in r.tag
            assert len(r.tag) <= 70
