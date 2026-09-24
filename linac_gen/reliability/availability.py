"""Leg B: availability and the beam-trip budget from a reliability block
diagram (numpy only).

A machine is a series of ``Block`` rows — each a repairable unit (or
``n_parallel`` identical units of which ``k_required`` must be up) with
an MTBF, an MTTR distribution and a *fault class* that says how an
event ends: ``downtime`` (the block's repair time), ``auto_rephase``
(the RF re-phases itself, ~10 s), ``operator_retune`` (minutes), or
``degraded`` (the beam continues at reduced performance: the event is
counted, no unavailability).  A ``class_split`` — the recovered-by
fractions the fault leg measures — routes a block's events over several
classes.  :func:`simulate` is a Monte Carlo over ``n_trials`` years of
Poisson failures with drawn recoveries; :func:`analytic_availability`
is the closed form (series product of ``MTBF/(MTBF+MTTR)`` with the
k-of-n binomial) the tests pin the Monte Carlo against.  Nothing here
is a measurement: the block values come from the user's CSV (see
:func:`write_blocks_template` for the surrogate rows the campaign
starts from) and the report labels them as inputs.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Block", "FaultClass", "DEFAULT_FAULT_CLASSES", "AvailabilityResult",
           "simulate", "analytic_availability", "read_blocks_csv", "write_blocks_csv",
           "write_blocks_template", "DEFAULT_BUDGET_BINS_S"]

#: ESS-style trip-duration bins (s): the beam-trip budget counts events per bin.
DEFAULT_BUDGET_BINS_S = (10.0, 60.0, 300.0, 1200.0, 3600.0, 4.0 * 3600.0)

_MTTR_DISTS = ("exponential", "lognormal", "fixed")
_RECOVERY_DISTS = ("fixed", "exponential", "lognormal", "empirical")


@dataclass
class Block:
    name: str
    mtbf_h: float
    mttr_h: float
    parent: str = ""
    mttr_dist: str = "exponential"
    n_parallel: int = 1
    k_required: int = 1
    fault_class: str = "downtime"
    scenario_class: str = ""      # the fault-leg class (S1…S10) whose recovered-by split routes this block's events

    def __post_init__(self):
        if self.mtbf_h <= 0.0:
            raise ValueError(f"block {self.name!r}: mtbf_h must be > 0")
        if self.mttr_h < 0.0:
            raise ValueError(f"block {self.name!r}: mttr_h must be >= 0")
        if self.mttr_dist not in _MTTR_DISTS:
            raise ValueError(f"block {self.name!r}: mttr_dist {self.mttr_dist!r} not in {_MTTR_DISTS}")
        if self.n_parallel < 1 or not (1 <= self.k_required <= self.n_parallel):
            raise ValueError(f"block {self.name!r}: need 1 <= k_required <= n_parallel")

    @property
    def unit_availability(self) -> float:
        return self.mtbf_h / (self.mtbf_h + self.mttr_h)


@dataclass
class FaultClass:
    """How an event of this class ends.  ``recovery`` None = the block's
    own MTTR; else ``{"dist": "fixed"|"exponential"|"lognormal"|
    "empirical", "mean_s"|"value_s", "sigma_ln", "samples_s"}``.
    ``downtime`` False = counted, never unavailable (degraded running)."""
    name: str
    recovery: dict | None = None
    downtime: bool = True
    trip: bool = True          # counts in the beam-trip budget


DEFAULT_FAULT_CLASSES = {
    "downtime": FaultClass("downtime"),
    "auto_rephase": FaultClass("auto_rephase", {"dist": "fixed", "value_s": 10.0}),
    "operator_retune": FaultClass("operator_retune",
                                  {"dist": "lognormal", "mean_s": 600.0, "sigma_ln": 0.6}),
    "degraded": FaultClass("degraded", {"dist": "fixed", "value_s": 0.0}, downtime=False, trip=False),
}


@dataclass
class AvailabilityResult:
    hours_per_year: float
    n_trials: int
    availability: np.ndarray                  # per trial
    trips_per_year: dict                      # class -> mean events / year
    downtime_h_by_block: dict                 # block -> mean hours / year
    events_by_block: dict                     # block -> mean events / year
    bins_s: tuple
    trip_histogram: np.ndarray                # mean events / year per bin (+ overflow)
    analytic: float
    sensitivity: list = field(default_factory=list)   # (block, dA/dlnMTBF, dA/dlnMTTR)

    @property
    def mean(self) -> float:
        return float(np.mean(self.availability))

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.availability, p))

    def summary(self) -> dict:
        return {
            "hours_per_year": self.hours_per_year, "n_trials": self.n_trials,
            "availability_mean": self.mean, "availability_p05": self.percentile(5),
            "availability_p95": self.percentile(95), "availability_analytic": self.analytic,
            "trips_per_year": dict(self.trips_per_year),
            "downtime_h_by_block": dict(self.downtime_h_by_block),
            "events_by_block": dict(self.events_by_block),
            "bins_s": list(self.bins_s), "trip_histogram": self.trip_histogram.tolist(),
            "sensitivity": [list(r) for r in self.sensitivity],
        }


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------
def _k_of_n(a: float, n: int, k: int) -> float:
    return float(sum(math.comb(n, j) * a ** j * (1.0 - a) ** (n - j) for j in range(k, n + 1)))


def block_availability(b: Block) -> float:
    """Steady-state availability of one block (k-of-n of identical units)."""
    return _k_of_n(b.unit_availability, b.n_parallel, b.k_required)


def analytic_availability(blocks, fault_classes=None) -> float:
    """Series product of the block availabilities.  A block whose fault
    class carries its own recovery uses that mean as its MTTR; a
    non-downtime class contributes 1."""
    fc = fault_classes or DEFAULT_FAULT_CLASSES
    prod = 1.0
    for b in blocks:
        cls = fc.get(b.fault_class, FaultClass(b.fault_class))
        if not cls.downtime:
            continue
        mttr = b.mttr_h if cls.recovery is None else _recovery_mean_s(cls.recovery) / 3600.0
        a = _k_of_n(b.mtbf_h / (b.mtbf_h + mttr), b.n_parallel, b.k_required)
        prod *= a
    return float(prod)


def _recovery_mean_s(rec: dict) -> float:
    d = rec.get("dist", "fixed")
    if d == "fixed":
        return float(rec.get("value_s", rec.get("mean_s", 0.0)))
    if d == "empirical":
        s = np.asarray(rec.get("samples_s", []), dtype=float)
        return float(s.mean()) if s.size else 0.0
    return float(rec.get("mean_s", 0.0))


# ---------------------------------------------------------------------------
# draws
# ---------------------------------------------------------------------------
def _draw_repair_h(rng, b: Block, n: int) -> np.ndarray:
    if b.mttr_h <= 0.0 or n == 0:
        return np.zeros(n)
    if b.mttr_dist == "exponential":
        return rng.exponential(b.mttr_h, n)
    if b.mttr_dist == "fixed":
        return np.full(n, b.mttr_h)
    sigma = 0.6                                       # lognormal, mean pinned
    mu = math.log(b.mttr_h) - 0.5 * sigma * sigma
    return rng.lognormal(mu, sigma, n)


def _draw_recovery_s(rng, rec: dict, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0)
    d = rec.get("dist", "fixed")
    if d not in _RECOVERY_DISTS:
        raise ValueError(f"recovery dist {d!r} not in {_RECOVERY_DISTS}")
    if d == "fixed":
        return np.full(n, float(rec.get("value_s", rec.get("mean_s", 0.0))))
    if d == "exponential":
        return rng.exponential(float(rec["mean_s"]), n)
    if d == "empirical":
        s = np.asarray(rec["samples_s"], dtype=float)
        if s.size == 0:
            raise ValueError("empirical recovery needs samples_s")
        return s[rng.integers(0, s.size, n)]
    sigma = float(rec.get("sigma_ln", 0.6))
    mu = math.log(float(rec["mean_s"])) - 0.5 * sigma * sigma
    return rng.lognormal(mu, sigma, n)


def _union_length(starts: np.ndarray, ends: np.ndarray, t_end: float) -> float:
    """Total length of the union of [start, end) intervals clipped to [0, t_end)."""
    if starts.size == 0:
        return 0.0
    order = np.argsort(starts)
    s, e = starts[order], np.minimum(ends[order], t_end)
    total, cur_s, cur_e = 0.0, s[0], e[0]
    for a, b in zip(s[1:], e[1:]):
        if a > cur_e:
            total += max(0.0, cur_e - cur_s)
            cur_s, cur_e = a, b
        elif b > cur_e:
            cur_e = b
    total += max(0.0, cur_e - cur_s)
    return float(total)


def _k_of_n_down_intervals(unit_intervals: list, k: int, n: int, t_end: float):
    """Down intervals of a k-of-n block from its units' down intervals:
    times when more than n-k units are down simultaneously."""
    if n == 1:
        return unit_intervals[0]
    starts = np.concatenate([u[0] for u in unit_intervals]) if unit_intervals else np.zeros(0)
    ends = np.concatenate([u[1] for u in unit_intervals]) if unit_intervals else np.zeros(0)
    if starts.size == 0:
        return np.zeros(0), np.zeros(0)
    ev_t = np.concatenate([starts, np.minimum(ends, t_end)])
    ev_d = np.concatenate([np.ones(starts.size), -np.ones(ends.size)])
    order = np.lexsort((ev_d, ev_t))            # ends before starts at equal t
    ev_t, ev_d = ev_t[order], ev_d[order]
    down = np.cumsum(ev_d)
    need = n - k + 1
    out_s, out_e = [], []
    open_t = None
    for t, d in zip(ev_t, down):
        if d >= need and open_t is None:
            open_t = t
        elif d < need and open_t is not None:
            out_s.append(open_t); out_e.append(t); open_t = None
    if open_t is not None:
        out_s.append(open_t); out_e.append(t_end)
    return np.asarray(out_s), np.asarray(out_e)


def simulate(blocks, fault_classes=None, *, hours_per_year: float = 5000.0,
             n_trials: int = 200, seed: int = 0, class_split: dict | None = None,
             bins_s=DEFAULT_BUDGET_BINS_S) -> AvailabilityResult:
    """Monte Carlo of ``n_trials`` operating years.

    ``class_split`` — ``{block_name: {class: fraction}}`` routes a block's
    events over fault classes (the fractions the fault leg measured:
    recovered by auto re-phase / operator retune / not recoverable);
    without it every event of a block takes ``block.fault_class``.
    """
    fc = dict(DEFAULT_FAULT_CLASSES)
    fc.update(fault_classes or {})
    blocks = list(blocks)
    bins = tuple(float(b) for b in bins_s)
    rng = np.random.default_rng(seed)
    t_end = float(hours_per_year)
    avail = np.zeros(n_trials)
    trips = {name: np.zeros(n_trials) for name in fc}
    down_by_block = {b.name: np.zeros(n_trials) for b in blocks}
    events_by_block = {b.name: np.zeros(n_trials) for b in blocks}
    hist = np.zeros((n_trials, len(bins) + 1))
    for it in range(n_trials):
        sys_s, sys_e = [], []
        for b in blocks:
            split = (class_split or {}).get(b.name)
            unit_intervals = []
            if split:
                names = list(split)
                probs = np.asarray([split[k] for k in names], dtype=float)
                probs = probs / probs.sum()
            for _u in range(b.n_parallel):
                # Alternating renewal: a unit that is down cannot fail
                # again until it is back — the model behind the
                # MTBF/(MTBF+MTTR) closed form the results are checked
                # against.  Failures Exp(MTBF) apart, each followed by
                # its recovery.  Units start in their steady state (up
                # with probability MTBF/(MTBF+MTTR), else inside a
                # residual outage) so a finite horizon carries no
                # everything-starts-up bias.
                starts, ends = [], []
                t = 0.0
                cls0 = fc.get(b.fault_class, FaultClass(b.fault_class))
                if cls0.downtime:
                    # the outage a unit is found in at t = 0 follows the
                    # block's own class: its repair, or the class recovery
                    if cls0.recovery is None:
                        mean_out = b.mttr_h
                    else:
                        mean_out = _recovery_mean_s(cls0.recovery) / 3600.0
                    a_unit = b.mtbf_h / (b.mtbf_h + mean_out) if mean_out > 0.0 else 1.0
                    if mean_out > 0.0 and rng.uniform() >= a_unit:
                        if cls0.recovery is None and b.mttr_dist == "exponential":
                            resid = float(rng.exponential(b.mttr_h))
                        elif cls0.recovery is None:
                            resid = float(_draw_repair_h(rng, b, 1)[0]) * float(rng.uniform())
                        elif cls0.recovery.get("dist") == "exponential":
                            resid = float(_draw_recovery_s(rng, cls0.recovery, 1)[0]) / 3600.0
                        else:
                            resid = float(_draw_recovery_s(rng, cls0.recovery, 1)[0]) / 3600.0 * float(rng.uniform())
                        if resid > 0.0:
                            starts.append(0.0); ends.append(min(resid, t_end))
                        t = resid
                t += rng.exponential(b.mtbf_h)
                while t < t_end:
                    cname = (names[rng.choice(len(names), p=probs)] if split
                             else b.fault_class)
                    cls = fc.get(cname, FaultClass(cname))
                    if cls.recovery is None:
                        d_h = float(_draw_repair_h(rng, b, 1)[0])
                    else:
                        d_h = float(_draw_recovery_s(rng, cls.recovery, 1)[0]) / 3600.0
                    events_by_block[b.name][it] += 1
                    if cls.trip:
                        trips[cname][it] += 1
                        hist[it, int(np.searchsorted(bins, d_h * 3600.0, side="left"))] += 1
                    if cls.downtime and d_h > 0.0:
                        starts.append(t); ends.append(t + d_h)
                    t += d_h + rng.exponential(b.mtbf_h)
                unit_intervals.append((np.asarray(starts), np.asarray(ends)))
            bs, be = _k_of_n_down_intervals(unit_intervals, b.k_required, b.n_parallel, t_end)
            down_by_block[b.name][it] = _union_length(bs, be, t_end)
            sys_s.append(bs); sys_e.append(be)
        all_s = np.concatenate(sys_s) if sys_s else np.zeros(0)
        all_e = np.concatenate(sys_e) if sys_e else np.zeros(0)
        avail[it] = 1.0 - _union_length(all_s, all_e, t_end) / t_end
    sens = []
    for b in blocks:
        base = analytic_availability(blocks, fc)
        up = analytic_availability([Block(**{**b.__dict__, "mtbf_h": b.mtbf_h * math.e}) if x is b else x
                                    for x in blocks], fc)
        rp = analytic_availability([Block(**{**b.__dict__, "mttr_h": b.mttr_h * math.e}) if x is b else x
                                    for x in blocks], fc)
        sens.append((b.name, up - base, rp - base))
    return AvailabilityResult(
        hours_per_year=t_end, n_trials=int(n_trials), availability=avail,
        trips_per_year={k: float(v.mean()) for k, v in trips.items()},
        downtime_h_by_block={k: float(v.mean()) for k, v in down_by_block.items()},
        events_by_block={k: float(v.mean()) for k, v in events_by_block.items()},
        bins_s=bins, trip_histogram=hist.mean(axis=0),
        analytic=analytic_availability(blocks, fc), sensitivity=sens)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
_COLUMNS = ("name", "parent", "mtbf_h", "mttr_h", "mttr_dist", "n_parallel", "k_required", "fault_class",
            "scenario_class")


def read_blocks_csv(path) -> list[Block]:
    rows: list[Block] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(ln for ln in fh if not ln.startswith("#"))
        for r in reader:
            if not (r.get("name") or "").strip():
                continue
            rows.append(Block(
                name=r["name"].strip(), parent=(r.get("parent") or "").strip(),
                mtbf_h=float(r["mtbf_h"]), mttr_h=float(r.get("mttr_h") or 0.0),
                mttr_dist=(r.get("mttr_dist") or "exponential").strip(),
                n_parallel=int(float(r.get("n_parallel") or 1)),
                k_required=int(float(r.get("k_required") or 1)),
                fault_class=(r.get("fault_class") or "downtime").strip(),
                scenario_class=(r.get("scenario_class") or "").strip()))
    if not rows:
        raise ValueError(f"{path}: no block rows")
    return rows


def write_blocks_csv(blocks, path, header_note: str | None = None) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        if header_note:
            for line in header_note.splitlines():
                fh.write(f"# {line}\n")
        w = csv.writer(fh)
        w.writerow(_COLUMNS)
        for b in blocks:
            w.writerow([b.name, b.parent, f"{b.mtbf_h:g}", f"{b.mttr_h:g}", b.mttr_dist,
                        b.n_parallel, b.k_required, b.fault_class, b.scenario_class])


#: Surrogate rows (hours) a campaign starts from — placeholders in the
#: range of published linac component figures, NOT project data; replace
#: them with the project's MTBF/MTTR spreadsheet.
_TEMPLATE_ROWS = (
    Block("RF_STATION", 2000.0, 4.0, "RF", scenario_class="S8"),
    Block("SRF_CAVITY_TRIP", 60.0, 0.003, "RF", "fixed", fault_class="auto_rephase", scenario_class="S1"),
    Block("CRYOMODULE", 20000.0, 240.0, "CRYO", scenario_class="S7"),
    Block("CRYOPLANT", 8000.0, 24.0, "CRYO"),
    Block("MAGNET_PS", 40000.0, 3.0, "MAGNETS", scenario_class="S2"),
    Block("VACUUM", 15000.0, 8.0, "VACUUM"),
    Block("ION_SOURCE", 500.0, 1.0, "FRONT_END", n_parallel=2, k_required=1),
    Block("LLRF_CONTROLS", 5000.0, 0.5, "CONTROLS", fault_class="operator_retune", scenario_class="S3"),
    Block("MPS_FALSE_TRIP", 200.0, 0.02, "CONTROLS", "fixed", fault_class="auto_rephase"),
)


def write_blocks_template(path) -> None:
    write_blocks_csv(_TEMPLATE_ROWS, path, header_note=(
        "Reliability block diagram — SURROGATE values in the range of published "
        "linac component figures, not project data.\n"
        "Replace with the project's MTBF/MTTR spreadsheet.  Columns: name, parent "
        "(section/group), mtbf_h, mttr_h, mttr_dist (exponential|lognormal|fixed), "
        "n_parallel, k_required, fault_class (downtime|auto_rephase|operator_retune|degraded), "
        "scenario_class (the fault-leg class S1..S10 whose recovered-by fractions route this "
        "block's events; blank = the block's own fault_class)."))
    return None
