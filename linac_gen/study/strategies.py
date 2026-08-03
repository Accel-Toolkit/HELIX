"""Expand a StudySpec into the deterministic list of runs.

``expand_runs`` is PURE: the same spec always yields the identical
list (order included).  Resume depends on this — run directories are
keyed by the run index.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import numpy as np

from linac_gen.study.spec import ParamSpec, StudySpec


@dataclass(frozen=True)
class RunSpec:
    index: int
    params: tuple                 # ((selector, value), ...) in spec order
    seed: int
    repeat: int
    tag: str


def param_values(p: ParamSpec) -> list:
    if p.values is not None:
        return [float(v) for v in p.values]
    if p.spacing == "log":
        if p.start <= 0 or p.stop <= 0:
            raise ValueError(
                f"parameter {p.selector!r}: log spacing needs "
                "positive start/stop")
        return list(np.geomspace(p.start, p.stop, int(p.n)))
    return list(np.linspace(p.start, p.stop, int(p.n)))


_TAG_BAD = re.compile(r"[^A-Za-z0-9_.@+-]+")


def _tag(params: tuple, seed: int) -> str:
    parts = [f"{sel}={val:.6g}" for sel, val in params]
    tag = "_".join(parts) if parts else "baseline"
    tag = _TAG_BAD.sub("-", tag)
    if len(tag) > 60:
        tag = tag[:57] + "..."
    return f"{tag}_s{seed}"


def _combos(spec: StudySpec) -> list:
    """Parameter combinations (selector, value) per run, strategy-wise."""
    ps = spec.parameters
    if spec.strategy == "grid":
        val_lists = [param_values(p) for p in ps]
        return [tuple((p.selector, v) for p, v in zip(ps, combo))
                for combo in itertools.product(*val_lists)]

    if spec.strategy == "zip":
        val_lists = [param_values(p) for p in ps]
        lens = {len(v) for v in val_lists}
        if len(lens) != 1:
            raise ValueError(
                f"zip strategy needs equal-length value lists, got "
                f"{[len(v) for v in val_lists]}")
        return [tuple((p.selector, vl[i]) for p, vl in zip(ps, val_lists))
                for i in range(lens.pop())]

    if spec.strategy == "oat":
        for p in ps:
            if p.baseline is None:
                raise ValueError(
                    f"oat strategy: parameter {p.selector!r} has no "
                    "baseline (the all-nominal reference run needs one)")
        combos = [tuple((p.selector, float(p.baseline)) for p in ps)]
        for k, p in enumerate(ps):
            for v in param_values(p):
                combos.append(tuple(
                    (q.selector, v if j == k else float(q.baseline))
                    for j, q in enumerate(ps)))
        return combos

    if spec.strategy in ("random", "lhs"):
        for p in ps:
            if p.values is not None or None in (p.start, p.stop):
                raise ValueError(
                    f"{spec.strategy} strategy: parameter "
                    f"{p.selector!r} needs start/stop ranges")
        d = len(ps)
        n = int(spec.n_samples)
        lo = np.array([p.start for p in ps], dtype=float)
        hi = np.array([p.stop for p in ps], dtype=float)
        if spec.strategy == "lhs":
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=d, seed=spec.sampler_seed)
            unit = sampler.random(n)
            pts = qmc.scale(unit, lo, hi)
        else:
            rng = np.random.default_rng(spec.sampler_seed)
            pts = lo + (hi - lo) * rng.random((n, d))
        return [tuple((p.selector, float(row[j]))
                      for j, p in enumerate(ps))
                for row in pts]

    raise ValueError(f"unknown strategy {spec.strategy!r}")


def expand_runs(spec: StudySpec) -> list:
    """The full deterministic run list: combos × seed repeats."""
    spec.validate_shape()
    combos = _combos(spec)
    runs = []
    idx = 0
    for combo in combos:
        for r in range(spec.repeats):
            seed = spec.seed + r
            runs.append(RunSpec(index=idx, params=combo, seed=seed,
                                repeat=r, tag=_tag(combo, seed)))
            idx += 1
    return runs
