"""Failure study runner: sweep scenarios through the parallel scan pool.

Reuses :func:`linac_gen.cli.common.build_scan_point` (one
:class:`~linac_gen.parallel.scan_pool.ScanPoint` per scenario, failure injected
via ``element_overrides``) and
:func:`linac_gen.parallel.scan_pool.run_scan_points`, then scores each scenario
against the nominal baseline with :func:`linac_gen.failures.criticality`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import copy

from linac_gen.cli.common import (apply_element_override, build_scan_point,
                                  make_sc_config, make_step_config,
                                  result_summary, run_envelope_sim, run_mp_sim)
from linac_gen.failures.criticality import criticality_score
from linac_gen.failures.scenario import FailureScenario
from linac_gen.parallel.scan_pool import run_scan_points, run_scan_points_serial


@dataclass
class ScenarioImpact:
    scenario: FailureScenario
    metrics: dict                  # raw _scan_metrics dict for the scenario
    d_transmission: float | None   # baseline_T - scenario_T (percentage points)
    emit_growth_x: float
    emit_growth_y: float
    emit_growth_z: float | None
    d_energy_mev: float | None     # |baseline_E - scenario_E|
    beam_lost: bool
    criticality: float
    terms: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.scenario.label


@dataclass
class FailureStudyResults:
    baseline: dict
    impacts: list[ScenarioImpact]
    ranking: list[int]             # indices into impacts, descending criticality
    element_names: list[str]
    mode: str
    pair_matrix: np.ndarray | None = None     # NxN criticality (pairs only)
    pair_index: dict[str, int] = field(default_factory=dict)

    def top(self, k: int) -> list[ScenarioImpact]:
        return [self.impacts[i] for i in self.ranking[:k]]

    def to_rows(self) -> list[dict]:
        rows = []
        for im in self.impacts:
            t = im.metrics.get("transmission")
            rows.append({
                "scenario": im.label,
                "elements": "|".join(im.scenario.element_names),
                "criticality": round(im.criticality, 6),
                "transmission": t,
                "loss_pct": (None if t is None else round(max(0.0, 100.0 - t), 4)),
                "d_transmission": im.d_transmission,
                "emit_nx": im.metrics.get("emit_nx"),
                "emit_ny": im.metrics.get("emit_ny"),
                "emit_nz": im.metrics.get("emit_nz"),
                "emit_x": im.metrics.get("emit_x"),
                "emit_growth_x": round(im.emit_growth_x, 6),
                "emit_growth_y": round(im.emit_growth_y, 6),
                "emit_growth_z": (None if im.emit_growth_z is None
                                  else round(im.emit_growth_z, 6)),
                "ref_w_kin": im.metrics.get("ref_w_kin"),
                "d_energy_mev": im.d_energy_mev,
                "beam_lost": im.beam_lost,
            })
        return rows


def _growth(base, val):
    if base is None or val is None or base <= 0:
        return None if base is None or val is None else 0.0
    return float(val) / float(base) - 1.0


def _last_attr(res, name):
    arr = getattr(res, name, None)
    try:
        return float(arr[-1]) if arr is not None and len(arr) else None
    except (TypeError, IndexError, ValueError):
        return None


def recorder_metrics(res) -> dict:
    """``result_summary`` plus the NORMALISED RMS emittances εnx/εny/εnz.

    Read directly from the recorder when present (MP, and envelope when it
    records them); transverse falls back to εn = βγ·ε_geo from the recorder's
    ref_beta/ref_gamma when the normalised arrays are absent.
    """
    m = result_summary(res)
    b, g = _last_attr(res, "ref_beta"), _last_attr(res, "ref_gamma")
    bg = (b * g) if (b is not None and g is not None) else None
    enx = _last_attr(res, "emit_nx")
    eny = _last_attr(res, "emit_ny")
    enz = _last_attr(res, "emit_nz")
    # Envelope records only geometric emittances; reconstruct the normalised
    # ones the same way the MP recorder does: εn_xy = βγ·ε_geo, εnz = βγ·εz_mmmrad.
    if bg is not None:
        if enx is None and m.get("emit_x") is not None:
            enx = m["emit_x"] * bg
        if eny is None and m.get("emit_y") is not None:
            eny = m["emit_y"] * bg
        if enz is None:
            ez_mmmrad = _last_attr(res, "emit_z_mmmrad")
            if ez_mmmrad is not None:
                enz = ez_mmmrad * bg
    m["emit_nx"], m["emit_ny"], m["emit_nz"] = enx, eny, enz
    # Schema parity with scan_pool._scan_metrics (path mode): expose the same
    # keys regardless of forward executor, so any consumer sees one schema.
    m["ref_beta"], m["ref_gamma"] = b, g
    m["x_max"] = _last_attr(res, "x_max")
    m["y_max"] = _last_attr(res, "y_max")
    m.setdefault("elapsed", None)
    return m


class FailureStudy:
    """Run a failure sweep on a lattice file and score each scenario."""

    def __init__(self, lattice_path: str | None = None,
                 beam_overrides: dict | None = None, *,
                 lattice=None, beam_config=None,
                 mode: str = "mp", env_solver: str = "matrix",
                 seed: int = 42, cli: dict | None = None,
                 weights: dict | None = None,
                 lost_threshold_pct: float = 1.0):
        """Two modes:

        * **path mode** (CLI): pass ``lattice_path`` + ``beam_overrides``;
          scenarios run through the parallel scan pool (re-parsing the file).
        * **in-memory mode** (GUI): pass ``lattice`` (a parsed ``Lattice``) +
          ``beam_config`` (a ``BeamConfig``); scenarios run serially in-process
          on deep-copies, so edited / unsaved lattices and their element names
          are honoured exactly (no write→reparse round-trip).
        """
        self.lattice_path = str(lattice_path) if lattice_path else None
        self.beam_overrides = beam_overrides or {}
        self.lattice = lattice
        self.beam_config = beam_config
        self.in_memory = lattice is not None
        self.mode = mode
        self.env_solver = env_solver
        self.seed = seed
        self.cli = cli
        self.weights = weights
        self.lost_threshold_pct = lost_threshold_pct

    def _point(self, element_overrides):
        return build_scan_point(
            self.lattice_path, beam_overrides=self.beam_overrides,
            element_overrides=element_overrides, mode=self.mode,
            env_solver=self.env_solver, seed=self.seed, cli=self.cli)

    def _inmem_metrics(self, element_overrides) -> dict:
        work = copy.deepcopy(self.lattice)
        for sel, val in element_overrides:
            apply_element_override(work, sel, val)
        if self.mode == "mp":
            sc = make_sc_config(self.beam_config, {}, {})
            step = make_step_config({}, {})
            # run_mp_sim returns (recorder, beam) — take the recorder.
            res, _beam = run_mp_sim(work, copy.deepcopy(self.beam_config),
                                    sc, step, seed=self.seed)
        else:
            res = run_envelope_sim(work, copy.deepcopy(self.beam_config),
                                   self.env_solver)
        return recorder_metrics(res)

    def _impact(self, scenario, baseline, row) -> ScenarioImpact:
        score, terms, lost = criticality_score(
            baseline, row, weights=self.weights,
            lost_threshold_pct=self.lost_threshold_pct)
        t0, t = baseline.get("transmission"), row.get("transmission")
        dT = (t0 - t) if (t0 is not None and t is not None) else None
        e0, e = baseline.get("ref_w_kin"), row.get("ref_w_kin")
        dE = (abs(e0 - e) if (e0 is not None and e is not None
                              and e == e) else None)
        return ScenarioImpact(
            scenario=scenario, metrics=row, d_transmission=dT,
            # normalised-emittance growth (matches the criticality basis)
            emit_growth_x=_growth(baseline.get("emit_nx"), row.get("emit_nx")) or 0.0,
            emit_growth_y=_growth(baseline.get("emit_ny"), row.get("emit_ny")) or 0.0,
            emit_growth_z=_growth(baseline.get("emit_nz"), row.get("emit_nz")),
            d_energy_mev=dE, beam_lost=lost, criticality=score, terms=terms)

    def run(self, scenarios: list[FailureScenario], element_names: list[str],
            name_to_class: dict[str, str], *,
            combination: str = "single", max_workers: int | None = None,
            on_scenario_done=None, serial: bool = False) -> FailureStudyResults:
        if self.in_memory:
            # Serial, in-process on deep-copies — element names are exactly
            # the in-memory ones (no write→reparse rename), so this honours
            # edited / unsaved lattices.
            baseline = self._inmem_metrics(())
            impacts: list[ScenarioImpact] = []
            for i, s in enumerate(scenarios):
                row = self._inmem_metrics(s.element_overrides(name_to_class))
                im = self._impact(s, baseline, row)
                impacts.append(im)
                if on_scenario_done is not None:
                    on_scenario_done(i, im)
        else:
            # Path mode: baseline first, then the scenario sweep through the
            # parallel scan pool (criticality computed live in the callback).
            base_runner = (run_scan_points_serial if serial
                           else lambda pts, cb=None: run_scan_points(pts, cb, max_workers))
            baseline = base_runner([self._point(())])[0]
            pts = [self._point(s.element_overrides(name_to_class)) for s in scenarios]
            slots: list[ScenarioImpact | None] = [None] * len(scenarios)

            def _cb(i, row):
                im = self._impact(scenarios[i], baseline, row)
                slots[i] = im
                if on_scenario_done is not None:
                    on_scenario_done(i, im)

            if serial:
                run_scan_points_serial(pts, _cb)
            else:
                run_scan_points(pts, _cb, max_workers)
            impacts = [im for im in slots if im is not None]

        ranking = sorted(range(len(impacts)),
                         key=lambda i: impacts[i].criticality, reverse=True)

        pair_matrix = None
        pair_index: dict[str, int] = {}
        if combination == "pairs":
            names = list(element_names)
            pair_index = {n: i for i, n in enumerate(names)}
            n = len(names)
            pair_matrix = np.full((n, n), np.nan)
            for im in impacts:
                fnames = im.scenario.element_names
                if len(fnames) == 1 and fnames[0] in pair_index:
                    i = pair_index[fnames[0]]
                    pair_matrix[i, i] = im.criticality
                elif len(fnames) == 2 and all(f in pair_index for f in fnames):
                    a, b = pair_index[fnames[0]], pair_index[fnames[1]]
                    pair_matrix[a, b] = pair_matrix[b, a] = im.criticality

        return FailureStudyResults(
            baseline=baseline, impacts=impacts, ranking=ranking,
            element_names=list(element_names), mode=self.mode,
            pair_matrix=pair_matrix, pair_index=pair_index)
