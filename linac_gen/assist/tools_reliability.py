"""``run_reliability`` — the assistant's door into the Reliability Study mode.

Wires the assist tool registry to :mod:`linac_gen.reliability` (the same
engine behind ``python -m linac_gen reliability`` and the GUI window):
a folder-backed, resume-by-default campaign over the four legs with a
summary and an HTML report.  Long-running (executes through the job
manager).

Actions
-------
plan       the static plan (items per leg) for a deck with a preset, no run
run        create-if-needed then execute (RESUMES an existing campaign_dir)
summarize  rebuild the CSVs, summary.json, figures and report
report     alias of summarize
export     write a job folder (job_dir) for another machine
import     bring a job folder's results back into campaign_dir
"""
from __future__ import annotations

import json
from pathlib import Path

from linac_gen.assist.tools import (
    TOOLS, WorkContext, _capture, _ctx_provenance, _err, _local_path,
    _need, _ok, _refused, _tool,
)

_ = (TOOLS, WorkContext)     # registry import side-effect anchors


@_tool("run_reliability",
       "Reliability Study mode (linac_gen.reliability): fault tolerance with "
       "compensation, imperfections with faults on error seeds, foil "
       "scenarios and availability, run as ONE resumable campaign on the "
       "lattice FILE on disk (per-item results.h5 + status), with per-leg "
       "CSVs, summary.json and a self-contained HTML report.  Presets "
       "'quick' (envelope, ~minutes) and 'full' (MP verification, 200 "
       "seeds).  RESUME-BY-DEFAULT: calling 'run' again on the same "
       "campaign_dir skips completed items.  action='plan' previews the "
       "static plan without executing; 'summarize' rebuilds the report; "
       "'export' / 'import' move the heavy waves to another machine.  "
       "Long-running background job.",
       {"type": "object",
        "properties": {
            "action": {"type": "string", "default": "run",
                       "enum": ["run", "plan", "summarize", "report", "export", "import"]},
            "campaign_dir": {"type": "string",
                             "description": "existing campaign directory (resume / summarize / "
                                            "export / import target), or the directory to create; "
                                            "default <deck dir>/reliability/<name>"},
            "lattice_path": {"type": "string",
                             "description": "deck or .lgproj (default: the session lattice's file)"},
            "preset": {"type": "string", "default": "quick", "enum": ["quick", "full", "custom"]},
            "name": {"type": "string", "description": "campaign name (default reliability_<preset>)"},
            "legs": {"type": "array", "items": {"type": "string"},
                     "description": "subset of faults, imperfections, foil, availability"},
            "circuits": {"type": "string", "description": "circuits.json (default: next to the deck)"},
            "max_workers": {"type": "integer", "default": 1},
            "retry_failed": {"type": "boolean", "default": False},
            "job_dir": {"type": "string", "description": "export: folder to write; import: folder to read"},
            "allow_partial": {"type": "boolean", "default": False}},
        "required": []},
       "compute")
def _run_reliability(ctx, action: str = "run", campaign_dir: str = "",
                     lattice_path: str = "", preset: str = "quick", name: str = "",
                     legs=None, circuits: str = "", max_workers: int = 1,
                     retry_failed: bool = False, job_dir: str = "",
                     allow_partial: bool = False, progress_callback=None,
                     should_abort=None, _assist_prov=None):
    from linac_gen.reliability.campaign import LEGS, ReliabilityCampaign
    from linac_gen.reliability.spec import default_spec
    if campaign_dir:
        try:
            campaign_dir = _local_path(campaign_dir)
        except ValueError as exc:
            return _refused(exc)
    if legs:
        bad = [lg for lg in legs if lg not in LEGS]
        if bad:
            return _refused(f"unknown leg(s) {bad}; known: {list(LEGS)}")
    if name and (any(ch in name for ch in "/\\") or ".." in name or name.startswith(".")):
        return _refused(f"name {name!r} must be a plain folder name (no separators or '..')")
    resuming = bool(campaign_dir) and (Path(campaign_dir) / "campaign.json").exists()

    def _load_existing():
        return ReliabilityCampaign.load(Path(campaign_dir))

    if action in ("summarize", "report", "export", "import"):
        if not resuming:
            return _refused(f"{action} needs campaign_dir of an existing campaign")
        c, refusal, _w = _capture(_load_existing)
        if refusal:
            return refusal
        if action in ("summarize", "report"):
            out, refusal, _w = _capture(c.summarize)
            if refusal:
                return refusal
            return _ok({"campaign_dir": str(c.dir), "summary_json": str(out),
                        "report_html": str(c.dir / "report.html"), "status": c.status()},
                       _ctx_provenance(ctx))
        if not job_dir:
            return _refused(f"{action} needs job_dir")
        try:
            job_dir = _local_path(job_dir)
        except ValueError as exc:
            return _refused(exc)
        if action == "export":
            from linac_gen.reliability.jobs import export_job
            job, refusal, _w = _capture(export_job, c, job_dir, legs=legs)
            if refusal:
                return refusal
            n = len(json.loads((Path(job) / "job.json").read_text(encoding="utf-8"))["expected"])
            return _ok({"job_dir": str(job), "n_items": n, "readme": str(Path(job) / "README.txt")},
                       _ctx_provenance(ctx))
        from linac_gen.reliability.jobs import import_results
        rec, refusal, _w = _capture(import_results, job_dir, c, allow_partial=bool(allow_partial))
        if refusal:
            return refusal
        out, refusal, _w = _capture(c.summarize)
        if refusal:
            return refusal
        return _ok({"campaign_dir": str(c.dir), "imported": len(rec["imported"]),
                    "skipped_ok": len(rec["skipped_ok"]), "missing": rec["missing"],
                    "summary_json": str(out)}, _ctx_provenance(ctx))

    # ---- plan / run --------------------------------------------------------
    if not resuming:
        if lattice_path:
            try:
                lattice_path = _local_path(lattice_path)
            except ValueError as exc:
                return _refused(exc)
            if not Path(lattice_path).is_file():
                return _err(f"lattice file not found: {lattice_path}")
            input_path = str(Path(lattice_path).resolve())
        else:
            gate = _need(ctx, "lattice", "beam_config")
            if gate:
                return gate
            if not ctx.lattice_path or not Path(ctx.lattice_path).exists():
                return _refused("the session lattice has no saved file — a campaign executes "
                                "the .dat on disk; save it first (or pass lattice_path)")
            input_path = str(Path(ctx.lattice_path).resolve())

        def _build_spec():
            spec = default_spec(input_path, preset=preset, name=name or None)
            circ = circuits
            if not circ and (Path(input_path).parent / "circuits.json").exists():
                circ = str(Path(input_path).parent / "circuits.json")
            if circ:
                spec.circuits = str(Path(_local_path(circ)).resolve())
            spec.validate_shape()
            return spec
        spec, refusal, _w = _capture(_build_spec)
        if refusal:
            return refusal
    if action == "plan":
        if resuming:
            c, refusal, _w = _capture(_load_existing)
            if refusal:
                return refusal
            plan = c.plan(legs)
            spec = c.spec
        else:
            c = ReliabilityCampaign(Path(campaign_dir) if campaign_dir else Path("."), spec)
            plan, refusal, _w = _capture(c._static_plan)
            if refusal:
                return refusal
            if legs:
                plan = {k: v for k, v in plan.items() if k in legs}
        return _ok({"preset": spec.preset, "classes": list(spec.classes),
                    "items_per_leg": {k: len(v) for k, v in plan.items()},
                    "first_items": {k: [it["id"] for it in v[:5]] for k, v in plan.items()},
                    "note": "compensation, MP verification and faults-on-seeds waves are planned "
                            "from the fault ranking at run time"}, _ctx_provenance(ctx))

    def _make_campaign():
        if resuming:
            return ReliabilityCampaign.load(Path(campaign_dir))
        root = Path(campaign_dir) if campaign_dir else \
            Path(spec.input).parent / "reliability" / spec.name
        if (root / "campaign.json").exists():
            return ReliabilityCampaign.load(root)
        return ReliabilityCampaign.create(root, spec)
    c, refusal, _w = _capture(_make_campaign)
    if refusal:
        return refusal
    total = sum(len(v) for v in c.plan(legs).values())

    def _progress(p) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(float(p.done), p.done, max(p.total, 1))
        except Exception:                               # noqa: BLE001
            pass

    def _execute():
        return c.run(legs, max_workers=int(max_workers), retry_failed=bool(retry_failed),
                     progress_cb=_progress, should_stop=should_abort)
    out, refusal, _w = _capture(_execute)
    if refusal:
        return refusal
    if out is None:
        return _err("campaign stopped before completion (resume with the same campaign_dir)")
    st = c.status()
    failed = sum(v["failed"] for v in st.values())
    summary = json.loads(Path(out).read_text(encoding="utf-8"))
    data = {"campaign_dir": str(c.dir), "n_items": total, "status": st,
            "summary_json": str(out), "report_html": str(c.dir / "report.html"),
            "faults": {k: summary["faults"].get(k) for k in ("n_cases", "n_critical", "recovered_by",
                                                              "unrecoverable")},
            "compensation": summary.get("compensation"),
            "imperfections": {k: summary["imperfections"].get(k) for k in
                              ("n_seeds", "transmission_mean", "transmission_min")},
            "availability": summary.get("availability")}
    warnings = ([f"{failed} item(s) failed — re-run with retry_failed=true to re-queue"]
                if failed else [])
    return _ok(data, _ctx_provenance(ctx), warnings=warnings)
