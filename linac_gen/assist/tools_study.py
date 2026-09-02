"""``run_study`` — the assistant's door into the Parameter Study Manager.

Wires the assist tool registry to :mod:`linac_gen.study` (the same
engine behind the GUI's Param Study tab and ``python -m linac_gen
study``): folder-backed studies with one subprocess-or-serial run per
parameter point, resume-by-default, and a rebuilt ``summary/
summary.csv``.  Long-running (executes through the job manager).

Actions
-------
plan       expand the run table, execute nothing (cheap preview)
run        create-if-needed then execute; completed runs are skipped,
           so calling it again on the same study RESUMES it
summarize  rebuild summary.csv for an existing study directory
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from linac_gen.assist.tools import (
    TOOLS, WorkContext, _capture, _ctx_provenance, _err, _local_path,
    _need, _ok, _refused, _tool,
)

_ = (TOOLS, WorkContext)     # registry import side-effect anchors

_PARAM_ITEM = {
    "type": "object",
    "properties": {
        "selector": {"type": "string",
                     "description": "'NAME.attr' / '@N.attr' (element), "
                     "a BeamConfig field name, or nx/grid_extent/"
                     "step1/step2"},
        "start": {"type": "number"},
        "stop": {"type": "number"},
        "n": {"type": "integer"},
        "values": {"type": "array", "items": {"type": "number"},
                   "description": "explicit list (alternative to "
                   "start/stop/n)"},
        "spacing": {"type": "string", "enum": ["lin", "log"],
                    "default": "lin"},
        "baseline": {"type": "number",
                     "description": "nominal value (required for the "
                     "'oat' strategy)"},
    },
    "required": ["selector"],
}

_OBS_ITEM = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "quantity": {"type": "string",
                     "description": "any envelope/ array name "
                     "(sigma_x, emit_ny, transmission, ...)"},
        "at": {"description": "'end' (default), {\"s_m\": 12.5} or "
               "{\"element\": \"NAME\"}"},
    },
    "required": ["quantity"],
}


@_tool("run_study",
       "Parameter Study Manager (linac_gen.study): vary one or more "
       "lattice/beam/numerics parameters over a strategy (oat, zip, "
       "grid, random, lhs), run one headless simulation per point into "
       "a folder-backed study with per-run results.h5 + status, and "
       "build summary/summary.csv with the parameters, auto KPIs "
       "(transmission, final emittances/sizes/energy) and any user "
       "observables.  RESUME-BY-DEFAULT: re-running the same study_dir "
       "skips completed runs.  action='plan' previews the expanded run "
       "table without executing; 'summarize' rebuilds the summary for "
       "an existing directory.  The study executes the lattice FILE on "
       "disk (session lattice must be saved; unsaved edits are not "
       "seen).  Long-running background job.",
       {"type": "object",
        "properties": {
            "action": {"type": "string", "default": "run",
                       "enum": ["run", "plan", "summarize"]},
            "study_dir": {"type": "string",
                          "description": "existing study directory "
                          "(resume/summarize), or the directory to "
                          "create; default <calc_dir>/studies/<name>"},
            "name": {"type": "string",
                     "description": "new-study name (default "
                     "study_<timestamp>)"},
            "parameters": {"type": "array", "items": _PARAM_ITEM,
                           "description": "required for a NEW study"},
            "observables": {"type": "array", "items": _OBS_ITEM,
                            "default": []},
            "strategy": {"type": "string", "default": "grid",
                         "enum": ["oat", "zip", "grid", "random",
                                  "lhs"]},
            "n_samples": {"type": "integer",
                          "description": "random/lhs sample count"},
            "repeats": {"type": "integer", "default": 1,
                        "description": "seed repeats per point (MP "
                        "error bars)"},
            "seed": {"type": "integer", "default": 42},
            "mode": {"type": "string", "default": "envelope",
                     "enum": ["envelope", "mp"]},
            "lattice_path": {"type": "string",
                             "description": "deck or .lgproj to study "
                             "(default: the session lattice's file)"},
            "max_workers": {"type": "integer", "default": 1,
                            "description": "worker processes (1 = "
                            "serial in-process)"},
            "retry_failed": {"type": "boolean", "default": False,
                             "description": "re-queue runs whose "
                             "status is 'failed'"}},
        "required": []},
       "compute")
def _run_study(ctx, action: str = "run", study_dir: str = "",
               name: str = "", parameters=None, observables=None,
               strategy: str = "grid", n_samples: int | None = None,
               repeats: int = 1, seed: int = 42,
               mode: str = "envelope", lattice_path: str = "",
               max_workers: int = 1, retry_failed: bool = False,
               progress_callback=None, should_abort=None,
               _assist_prov=None):
    from linac_gen.study import (ObservableSpec, ParamSpec, StudySpec,
                                 StudyManager, expand_runs)

    # ---- existing-directory actions ---------------------------------
    if study_dir:
        try:
            study_dir = _local_path(study_dir)
        except ValueError as exc:
            return _refused(exc)

    def _load_existing():
        return StudyManager.load(Path(study_dir))

    if action == "summarize":
        if not study_dir:
            return _refused("summarize needs study_dir")
        mgr, refusal, _w = _capture(_load_existing)
        if refusal:
            return refusal
        out, refusal, _w = _capture(mgr.summarize)
        if refusal:
            return refusal
        return _ok({"study_dir": str(mgr.study_dir),
                    "summary_csv": str(out),
                    "n_runs": len(mgr.plan())},
                   _ctx_provenance(ctx))

    resuming = bool(study_dir) and \
        (Path(study_dir) / "study.json").exists()

    # ---- spec construction (new study) -------------------------------
    if not resuming:
        if not parameters:
            return _refused(
                "a NEW study needs 'parameters' (or pass study_dir of "
                "an existing study to resume/summarize)")
        if lattice_path:
            try:
                lattice_path = _local_path(lattice_path)
            except ValueError as exc:
                return _refused(exc)
            if not Path(lattice_path).is_file():
                return _err(f"lattice file not found: {lattice_path}")
            input_path = str(Path(lattice_path).resolve())
            gate = None
        else:
            gate = _need(ctx, "lattice", "beam_config")
            if gate:
                return gate
            if not ctx.lattice_path or \
                    not Path(ctx.lattice_path).exists():
                return _refused(
                    "the session lattice has no saved file — studies "
                    "execute the .dat on disk; save it first (or pass "
                    "lattice_path)")
            input_path = str(Path(ctx.lattice_path).resolve())

        def _build_spec():
            params = [ParamSpec(**{k: v for k, v in dict(p).items()
                                   if k in ("selector", "start", "stop",
                                            "n", "values", "spacing",
                                            "baseline")})
                      for p in (parameters or [])]
            obs = []
            for o in (observables or []):
                o = dict(o)
                obs.append(ObservableSpec(
                    name=str(o.get("name") or o["quantity"]),
                    quantity=str(o["quantity"]),
                    at=o.get("at", "end")))
            beam = asdict(ctx.beam_config) if ctx.beam_config else {}
            spec = StudySpec(
                name=(name or time.strftime("study_%Y%m%d_%H%M%S")),
                input=input_path, mode=mode, strategy=strategy,
                parameters=params, observables=obs, seed=int(seed),
                repeats=int(repeats), n_samples=n_samples, beam=beam)
            spec.validate_shape()
            return spec

        spec, refusal, _w = _capture(_build_spec)
        if refusal:
            return refusal

    if action == "plan":
        if resuming:
            mgr, refusal, _w = _capture(_load_existing)
            if refusal:
                return refusal
            spec = mgr.spec
        runs, refusal, _w = _capture(expand_runs, spec)
        if refusal:
            return refusal
        head = [{"index": r.index, "seed": r.seed,
                 "params": {s: v for s, v in r.params}}
                for r in runs[:20]]
        return _ok({"n_runs": len(runs), "strategy": spec.strategy,
                    "mode": spec.mode, "first_runs": head,
                    "truncated": len(runs) > 20},
                   _ctx_provenance(ctx))

    # ---- action == "run" ---------------------------------------------
    def _make_manager():
        if resuming:
            return StudyManager.load(Path(study_dir))
        root = Path(study_dir) if study_dir else \
            Path(getattr(ctx, "calc_dir", ".") or ".") / "studies" / \
            spec.name
        if (root / "study.json").exists():
            return StudyManager.load(root)
        return StudyManager.create(root, spec)

    mgr, refusal, _w = _capture(_make_manager)
    if refusal:
        return refusal

    total = len(mgr.plan())

    def _progress(p) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(float(p.done), p.done, p.total)
        except Exception:                               # noqa: BLE001
            pass

    def _execute():
        return mgr.run(max_workers=int(max_workers),
                       retry_failed=bool(retry_failed),
                       progress_cb=_progress,
                       should_stop=should_abort)

    out, refusal, _w = _capture(_execute)
    if refusal:
        return refusal
    resume_note = (["study_dir already contains a study — its stored "
                    "study.json governs; the 'parameters' argument was "
                    "ignored"]
                   if (resuming and parameters) else [])
    statuses = [(mgr._status(r) or {}).get("status")
                for r in mgr.plan()]
    failed = sum(1 for s in statuses if s == "failed")
    done = sum(1 for s in statuses if s == "ok")
    data = {"study_dir": str(mgr.study_dir), "n_runs": total,
            "completed": done, "failed": failed,
            "summary_csv": str(out),
            "observables": [o.name for o in mgr.spec.observables]}
    warnings = resume_note + \
        ([f"{failed} run(s) failed — re-run with retry_failed=true to "
          "re-queue them"] if failed else [])
    return _ok(data, _ctx_provenance(ctx), warnings)
