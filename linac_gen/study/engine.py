"""StudyManager — folder-backed parameter-study execution with resume.

Source-of-truth hierarchy (crash-safe by construction):

  study.json + summary/runs_manifest.json   the PLAN
  runs/run_*/status.json (+ results.h5)     the FACTS (one per run;
                                            results written atomically
                                            .part -> os.replace by the
                                            scan_pool worker)
  summary/summary.csv                       a derived VIEW, rebuilt any
                                            time by summarize()

A run is COMPLETE iff its ``status.json`` exists with status ok|failed
(ok additionally requires ``results.h5`` present).  ``run()`` is
resume-by-default: completed runs are skipped, ``.part`` orphans are
swept, and the plan is re-expanded and diffed against the manifest —
a spec edited after a partial run refuses loudly instead of silently
mixing two studies.  The lattice is pinned by SHA-256 at create time;
runs execute against the ORIGINAL deck path (copying a deck breaks its
relative FIELD_MAP_PATH references), so the hash check on every load
is what guarantees the physics never drifts under a resumed study.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from linac_gen.study import observables as _obs
from linac_gen.study.spec import (ObservableSpec, StudySpec, load_spec,
                                  save_spec)
from linac_gen.study.strategies import RunSpec, expand_runs

_log = logging.getLogger(__name__)

#: fixed metric columns (mirrors scan_pool._scan_metrics keys)
METRIC_KEYS = ("elapsed", "sigma_x", "sigma_y", "sigma_phi", "emit_x",
               "emit_y", "sigma_w", "emit_z", "transmission",
               "ref_w_kin", "x_max", "y_max", "ref_beta", "ref_gamma",
               "emit_nx", "emit_ny", "emit_nz")


@dataclass
class StudyProgress:
    done: int
    failed: int
    total: int
    mean_elapsed: float | None
    eta_s: float | None


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_dirname(run: RunSpec) -> str:
    return f"run_{run.index:05d}_{run.tag}"


class StudyManager:
    def __init__(self, study_dir: Path, spec: StudySpec):
        self.study_dir = Path(study_dir)
        self.spec = spec
        self.runs_dir = self.study_dir / "runs"
        self.summary_dir = self.study_dir / "summary"
        self._input_path = self._resolve_input()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def create(cls, study_dir, spec: StudySpec) -> "StudyManager":
        study_dir = Path(study_dir)
        if (study_dir / "study.json").exists():
            raise FileExistsError(
                f"{study_dir}/study.json already exists -- use load() / "
                "`study run` to resume, or --force to wipe")
        mgr = cls(study_dir, spec)
        mgr._validate_against_lattice()
        study_dir.mkdir(parents=True, exist_ok=True)
        (study_dir / "lattice").mkdir(exist_ok=True)
        mgr.runs_dir.mkdir(exist_ok=True)
        mgr.summary_dir.mkdir(exist_ok=True)
        # provenance snapshot (NOT the run input -- relative
        # FIELD_MAP_PATH references would break; the sha pin below is
        # the integrity guarantee for the real input)
        try:
            shutil.copy2(mgr._input_path,
                         study_dir / "lattice" / mgr._input_path.name)
        except OSError as exc:
            _log.warning("study snapshot copy failed: %s", exc)
        spec.lattice_sha256 = _sha256(mgr._input_path)
        save_spec(spec, study_dir / "study.json")
        manifest = [{"index": r.index, "dir": _run_dirname(r),
                     "params": list(map(list, r.params)),
                     "seed": r.seed} for r in expand_runs(spec)]
        (mgr.summary_dir / "runs_manifest.json").write_text(
            json.dumps(manifest, indent=1) + "\n")
        return mgr

    @classmethod
    def load(cls, study_dir) -> "StudyManager":
        study_dir = Path(study_dir)
        spec = load_spec(study_dir / "study.json")
        mgr = cls(study_dir, spec)
        if spec.lattice_sha256:
            now = _sha256(mgr._input_path)
            if now != spec.lattice_sha256:
                raise RuntimeError(
                    f"lattice {mgr._input_path} changed since the study "
                    f"was created (sha {now[:12]} != pinned "
                    f"{spec.lattice_sha256[:12]}) -- results would mix "
                    "two different machines; refusing")
        mpath = mgr.summary_dir / "runs_manifest.json"
        if mpath.exists():
            manifest = json.loads(mpath.read_text())
            fresh = [{"index": r.index, "dir": _run_dirname(r),
                      "params": list(map(list, r.params)),
                      "seed": r.seed} for r in expand_runs(spec)]
            if fresh != manifest:
                raise RuntimeError(
                    "study.json no longer expands to the recorded run "
                    "plan (spec edited after runs started?) -- refusing "
                    "to mix plans.  Start a new study or restore the "
                    "spec.")
        return mgr

    def _resolve_input(self) -> Path:
        p = Path(self.spec.input)
        if not p.is_absolute():
            p = (self.study_dir / p).resolve() \
                if (self.study_dir / p).exists() else p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"study input not found: {p}")
        return p

    # ------------------------------------------------------------------
    # validation (create-time; catches bad selectors before any run)
    # ------------------------------------------------------------------
    def _validate_against_lattice(self) -> None:
        import copy

        from linac_gen.cli.common import (apply_element_override,
                                          load_input)
        from linac_gen.core.config import BeamConfig

        self.spec.validate_shape()
        lattice, beam_cfg, _conv = load_input(str(self._input_path))
        from linac_gen.study.strategies import param_values
        for p in self.spec.parameters:
            kind = p.resolved_kind()
            vals = param_values(p)
            if kind == "element":
                probe = copy.deepcopy(lattice)
                for v in (min(vals), max(vals)):
                    apply_element_override(probe, p.selector, v)
                sel, attr = p.selector.rsplit(".", 1)
                elem = self._resolve_element(lattice, sel)
                cur = getattr(elem, attr)
                if isinstance(cur, int) and not isinstance(cur, bool):
                    bad = [v for v in vals
                           if abs(v - round(v)) > 1e-9]
                    if bad:
                        raise ValueError(
                            f"{p.selector}: integer attribute swept "
                            f"with non-integral value(s) {bad[:3]} -- "
                            "coercion would silently truncate")
                if p.baseline is None:
                    p.baseline = float(cur)
                if p.elem_class is None:
                    p.elem_class = type(elem).__name__
            elif kind == "beam":
                if not hasattr(beam_cfg, p.selector):
                    raise ValueError(
                        f"unknown beam parameter {p.selector!r} "
                        f"(not a BeamConfig field)")
                if p.baseline is None:
                    base = self.spec.beam.get(
                        p.selector, getattr(beam_cfg, p.selector))
                    p.baseline = float(base)
                _ = BeamConfig  # imported for clarity of contract
            else:                                       # structural
                if p.baseline is None and \
                        p.selector in self.spec.numerics:
                    p.baseline = float(
                        self.spec.numerics[p.selector])
        # resolve element-position observables to s_m (exit face)
        for ob in self.spec.observables:
            if isinstance(ob.at, dict):
                if "s_m" in ob.at:
                    ob.s_m = float(ob.at["s_m"])
                elif "element" in ob.at:
                    s_mm = 0.0
                    target = None
                    for el in lattice.elements:
                        s_mm += getattr(el, "length", 0.0)
                        if getattr(el, "name", None) == ob.at["element"]:
                            target = s_mm
                            break
                    if target is None:
                        raise ValueError(
                            f"observable {ob.name!r}: element "
                            f"{ob.at['element']!r} not in the lattice")
                    ob.s_m = target * 1e-3

    @staticmethod
    def _resolve_element(lattice, sel: str):
        if sel.startswith("@"):
            return lattice.elements[int(sel[1:]) - 1]
        matches = [e for e in lattice.elements
                   if getattr(e, "name", None) == sel]
        if len(matches) != 1:
            raise ValueError(
                f"element selector {sel!r}: {len(matches)} matches")
        return matches[0]

    # ------------------------------------------------------------------
    # planning / resume state
    # ------------------------------------------------------------------
    def plan(self) -> list:
        return expand_runs(self.spec)

    def _run_dir(self, run: RunSpec) -> Path:
        return self.runs_dir / _run_dirname(run)

    def _status(self, run: RunSpec) -> dict | None:
        sp = self._run_dir(run) / "status.json"
        if not sp.exists():
            return None
        try:
            return json.loads(sp.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _is_complete(self, run: RunSpec, *, retry_failed=False) -> bool:
        st = self._status(run)
        if st is None:
            return False
        if st.get("status") == "ok":
            return (self._run_dir(run) / "results.h5").exists()
        if st.get("status") == "failed":
            return not retry_failed
        return False

    def pending(self, *, retry_failed: bool = False) -> list:
        return [r for r in self.plan()
                if not self._is_complete(r, retry_failed=retry_failed)]

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def _point_for(self, run: RunSpec):
        import dataclasses as _dc

        from linac_gen.cli.common import build_scan_point

        beam_ov = dict(self.spec.beam)
        elem_ov = []
        cli_ov = dict(self.spec.numerics)
        for p, (sel, val) in zip(self.spec.parameters, run.params):
            kind = p.resolved_kind()
            if kind == "beam":
                beam_ov[sel] = val
            elif kind == "structural":
                cli_ov[sel] = val
            else:
                elem_ov.append((sel, val))
        point = build_scan_point(
            str(self._input_path), beam_overrides=beam_ov,
            element_overrides=elem_ov,
            sc_overrides=dict(self.spec.sc) or None,
            mode=self.spec.mode, env_solver=self.spec.env_solver,
            seed=run.seed, cli=cli_ov)
        run_dir = self._run_dir(run)
        run_dir.mkdir(parents=True, exist_ok=True)
        return _dc.replace(point,
                           out_path=str(run_dir / "results.h5"),
                           capture_errors=True)

    def _sweep_orphans(self) -> None:
        for part in self.runs_dir.glob("*/results.h5.part"):
            try:
                part.unlink()
            except OSError:
                pass

    def run(self, *, max_workers: int | None = None, serial: bool = False,
            force: bool = False, retry_failed: bool = False,
            on_run_done=None, progress_cb=None,
            should_stop=None) -> Path:
        from linac_gen.parallel.scan_pool import (run_scan_points,
                                                  run_scan_points_serial)

        if force and self.runs_dir.exists():
            shutil.rmtree(self.runs_dir)
            self.runs_dir.mkdir()
        self._sweep_orphans()
        todo = self.pending(retry_failed=retry_failed)
        total = len(self.plan())
        already = total - len(todo)
        if already:
            _log.info("study %s: resuming -- %d/%d runs already "
                      "complete", self.spec.name, already, total)
        if not todo:
            return self.summarize()

        points = [self._point_for(r) for r in todo]
        elapsed_q: deque = deque(maxlen=10)
        counters = {"done": already, "failed": 0}

        def _on_done(i: int, row: dict) -> None:
            run = todo[i]
            ok = row.get("error") is None
            obs = {}
            if ok and self.spec.observables:
                try:
                    obs = _obs.evaluate(
                        str(self._run_dir(run) / "results.h5"),
                        self.spec.observables)
                except Exception as exc:      # noqa: BLE001
                    _log.warning("observables failed for run %d: %s",
                                 run.index, exc)
            status = {
                "status": "ok" if ok else "failed",
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed": row.get("elapsed"),
                "error": row.get("error"),
                "params": {sel: val for sel, val in run.params},
                "seed": run.seed,
                "metrics": {k: row.get(k) for k in METRIC_KEYS},
                "observables": obs,
            }
            spath = self._run_dir(run) / "status.json"
            spath.write_text(json.dumps(status, indent=1) + "\n")
            counters["done"] += 1
            if not ok:
                counters["failed"] += 1
                _log.warning("run %d (%s) FAILED: %s", run.index,
                             run.tag, row.get("error"))
            if row.get("elapsed"):
                elapsed_q.append(float(row["elapsed"]))
            if progress_cb is not None:
                mean = (sum(elapsed_q) / len(elapsed_q)
                        if elapsed_q else None)
                remaining = total - counters["done"]
                width = max(1, min(max_workers or 1, remaining or 1))
                eta = (remaining * mean / width
                       if mean is not None else None)
                progress_cb(StudyProgress(
                    done=counters["done"], failed=counters["failed"],
                    total=total, mean_elapsed=mean, eta_s=eta))
            if on_run_done is not None:
                on_run_done(run, row)

        if serial or (max_workers or 1) <= 1:
            run_scan_points_serial(points, on_done=_on_done,
                                   should_stop=should_stop)
        else:
            run_scan_points(points, on_done=_on_done,
                            max_workers=max_workers,
                            should_stop=should_stop)
        return self.summarize()

    # ------------------------------------------------------------------
    # summary (derived view, rebuilt from status.json facts)
    # ------------------------------------------------------------------
    def summarize(self) -> Path:
        self.summary_dir.mkdir(exist_ok=True)
        out = self.summary_dir / "summary.csv"
        param_cols = [p.selector for p in self.spec.parameters]
        obs_cols = [o.name for o in self.spec.observables]
        cols = (["index", "tag", "status", "seed"] + param_cols
                + list(METRIC_KEYS) + obs_cols
                + ["error", "results_path"])
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for run in self.plan():
                st = self._status(run)
                if st is None:
                    continue
                row = {"index": run.index, "tag": run.tag,
                       "status": st.get("status"),
                       "seed": run.seed,
                       "error": st.get("error"),
                       "results_path": str(
                           self._run_dir(run) / "results.h5")}
                row.update(st.get("params", {}))
                row.update(st.get("metrics", {}) or {})
                row.update(st.get("observables", {}) or {})
                w.writerow(row)
        return out
