"""``python -m linac_gen reliability`` — the Reliability Study mode, headless.

Verbs
-----
plan       print the campaign plan (items per leg) for a deck / .lgproj
           with a preset, execute nothing; --write-spec saves the spec
run        create-if-needed, then execute; RESUME-BY-DEFAULT (completed
           items are skipped; --force wipes legs/; --retry-failed
           re-queues failures).  The target is a deck / .lgproj (a new
           campaign), a campaign directory, or an exported job folder
resume     alias of run for an existing campaign / job directory
summarize  rebuild the per-leg CSVs, summary.json, figures and report
report     alias of summarize (the report is part of the summary)
export     write a job folder for another machine: --to DIR [--legs]
           [--bundle-fields]
import     bring a job folder's results back: --into CAMPAIGN_DIR
           [--allow-partial]
selftest   the built-in self-test (linac_gen.reliability.selftest)

Exit codes: 0 success, 1 execution failure / self-test FAIL, 2 bad input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from linac_gen.cli import common

VERBS = ("plan", "run", "resume", "summarize", "report", "export", "import", "selftest")


def add_arguments(p) -> None:
    p.add_argument("verb", choices=VERBS)
    p.add_argument("target", nargs="?", default=None,
                   help="deck / .lgproj (plan, run), campaign directory (run, resume, "
                        "summarize, report, export) or job directory (run, import)")
    p.add_argument("--preset", choices=("quick", "full", "custom"), default="quick",
                   help="campaign preset for a NEW campaign (default quick)")
    p.add_argument("--spec", default=None,
                   help="a campaign.json to start from instead of the preset defaults")
    p.add_argument("--name", default=None, help="campaign name (default reliability_<preset>)")
    p.add_argument("--circuits", default=None,
                   help="circuits.json (default: <deck dir>/circuits.json when present)")
    p.add_argument("--dir", dest="dir_", default=None,
                   help="campaign directory (default: <input dir>/reliability/<name>/)")
    p.add_argument("--legs", default=None,
                   help="comma list of legs to run: faults,imperfections,foil,availability")
    p.add_argument("--parallel", type=int, default=None, metavar="N",
                   help="worker processes (default: the spec's execution.max_workers)")
    p.add_argument("--serial", action="store_true", help="force in-process serial execution")
    p.add_argument("--force", action="store_true", help="wipe legs/ and start over")
    p.add_argument("--retry-failed", action="store_true", help="re-queue failed items")
    p.add_argument("--write-spec", default=None, metavar="PATH",
                   help="plan: write the filled campaign spec here")
    p.add_argument("--to", default=None, metavar="DIR", help="export: the job folder to write")
    p.add_argument("--into", default=None, metavar="DIR", help="import: the campaign to import into")
    p.add_argument("--allow-partial", action="store_true",
                   help="import: accept a job with missing items")
    p.add_argument("--bundle-fields", action="store_true",
                   help="export: copy the field-map files into the job")
    p.add_argument("--quick", action="store_true", help="selftest: the quick checks only")
    p.add_argument("--regression", action="store_true",
                   help="selftest: also compare the public control cases with --before")
    p.add_argument("--before", default=None, metavar="DIR",
                   help="selftest: a physics_fix_report snapshot of the reference tree")
    p.add_argument("--out", default=None, metavar="DIR", help="selftest: work directory")
    p.add_argument("--cases", default=None,
                   help="selftest --regression: comma list of control cases (default: all public)")
    p.add_argument("--json", action="store_true", help="selftest: print the result as JSON")
    p.add_argument("-q", "--quiet", action="store_true")
    common.add_tracewin_ini_argument(p)


def _is_campaign(path: Path) -> bool:
    return path.is_dir() and (path / "campaign.json").exists()


def _new_spec(args):
    from linac_gen.reliability.spec import apply_preset, default_spec, load_spec
    target = Path(args.target)
    if args.spec:
        spec = load_spec(args.spec)
        spec.input = str(target.resolve())
        spec = apply_preset(spec)
    else:
        spec = default_spec(str(target.resolve()), preset=args.preset, name=args.name)
    if args.name:
        spec.name = args.name
    if args.tracewin_ini is not None:
        spec.tracewin_ini = args.tracewin_ini
    circuits = args.circuits
    if circuits is None and (target.parent / "circuits.json").exists():
        circuits = str(target.parent / "circuits.json")
    if circuits:
        spec.circuits = str(Path(circuits).resolve())
    return spec


def _legs(args):
    if not args.legs:
        return None
    from linac_gen.reliability.campaign import LEGS
    legs = [s.strip() for s in args.legs.split(",") if s.strip()]
    bad = [lg for lg in legs if lg not in LEGS]
    if bad:
        raise ValueError(f"unknown leg(s) {bad}; known: {list(LEGS)}")
    return legs


def run(args) -> int:
    if args.verb == "selftest":
        return _selftest(args)
    if not args.target:
        print(f"error: {args.verb} needs a target", file=sys.stderr)
        return 2
    target = Path(args.target)
    if not target.exists():
        print(f"error: target not found: {target}", file=sys.stderr)
        return 2
    from linac_gen.reliability.campaign import ReliabilityCampaign
    try:
        legs = _legs(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if legs is None and (target / "job.json").exists():
        # an exported job runs its own legs unless told otherwise
        try:
            legs = json.loads((target / "job.json").read_text(encoding="utf-8")).get("legs") or None
        except (OSError, json.JSONDecodeError):
            legs = None
    try:
        if args.verb == "plan":
            if _is_campaign(target):
                c = ReliabilityCampaign.load(target)
                spec = c.spec
                plan = c.plan(legs)
            else:
                spec = _new_spec(args)
                from linac_gen.reliability.spec import save_spec
                if args.write_spec:
                    save_spec(spec, args.write_spec)
                c = ReliabilityCampaign(Path(args.dir_ or "."), spec)
                plan = c._static_plan()
                if legs:
                    plan = {k: v for k, v in plan.items() if k in legs}
            n = sum(len(v) for v in plan.values())
            print(f"[reliability] {spec.name}: preset={spec.preset}, classes={','.join(spec.classes)}, "
                  f"{n} planned item(s) in the static plan")
            for lg, items in plan.items():
                print(f"  {lg:<14s} {len(items):5d}  "
                      + ", ".join(it["id"] for it in items[:4]) + (" …" if len(items) > 4 else ""))
            print("  (compensation, MP verification and faults-on-seeds waves are planned "
                  "from the fault ranking at run time)")
            if args.write_spec:
                print(f"  spec written: {args.write_spec}")
            return 0
        if args.verb in ("summarize", "report"):
            if not _is_campaign(target):
                print(f"error: {target} is not a campaign directory", file=sys.stderr)
                return 2
            c = ReliabilityCampaign.load(target)
            out = c.summarize()
            print(f"[reliability] summary written: {out}; report: {c.dir / 'report.html'}")
            return 0
        if args.verb == "export":
            if not _is_campaign(target) or not args.to:
                print("error: export needs a campaign directory and --to DIR", file=sys.stderr)
                return 2
            from linac_gen.reliability.jobs import export_job
            c = ReliabilityCampaign.load(target)
            job = export_job(c, args.to, legs=legs, bundle_fields=args.bundle_fields)
            n = len(json.loads((job / "job.json").read_text(encoding="utf-8"))["expected"])
            print(f"[reliability] job written: {job} ({n} item(s)); see {job / 'README.txt'}")
            return 0
        if args.verb == "import":
            if not args.into or not _is_campaign(Path(args.into)):
                print("error: import needs a job directory and --into CAMPAIGN_DIR", file=sys.stderr)
                return 2
            from linac_gen.reliability.jobs import import_results
            c = ReliabilityCampaign.load(args.into)
            rec = import_results(target, c, allow_partial=args.allow_partial)
            out = c.summarize()
            print(f"[reliability] imported {len(rec['imported'])} item(s), "
                  f"{len(rec['skipped_ok'])} already ok, {len(rec['missing'])} missing; summary: {out}")
            return 0
        # run / resume
        if _is_campaign(target):
            c = ReliabilityCampaign.load(target)
            if args.spec or args.name or args.preset != "quick" or args.circuits:
                print("[reliability] note: an existing campaign governs; --preset/--name/--spec/"
                      "--circuits were ignored", file=sys.stderr)
        else:
            if args.verb == "resume":
                print(f"error: resume needs an existing campaign directory, got {target}", file=sys.stderr)
                return 2
            spec = _new_spec(args)
            cdir = Path(args.dir_) if args.dir_ else target.resolve().parent / "reliability" / spec.name
            if _is_campaign(cdir):
                c = ReliabilityCampaign.load(cdir)
                print(f"[reliability] note: {cdir} already holds a campaign — its campaign.json "
                      "governs; --preset/--name/--spec/--circuits were ignored", file=sys.stderr)
            else:
                c = ReliabilityCampaign.create(cdir, spec)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    plan = c.plan(legs)
    total = sum(len(v) for v in plan.values())
    todo = len([it for it in c.pending(retry_failed=args.retry_failed) if not legs or it["leg"] in legs])
    if not args.quiet:
        print(f"[reliability] {c.spec.name}: {total} item(s) planned, {todo} to execute, "
              f"preset={c.spec.preset}, dir={c.dir}")

    def _progress(p) -> None:
        if args.quiet:
            return
        eta = f", ETA {p.eta_s / 60.0:.1f} min" if p.eta_s else ""
        print(f"[reliability] {p.leg}/{p.phase}: {p.done}/{p.total} ({p.failed} failed){eta}", flush=True)

    try:
        out = c.run(legs, max_workers=args.parallel, serial=args.serial, force=args.force,
                    retry_failed=args.retry_failed, progress_cb=_progress)
    except Exception as exc:                          # noqa: BLE001
        print(f"error: campaign execution failed: {exc}", file=sys.stderr)
        return 1
    if out is None:
        print("[reliability] stopped before completion", file=sys.stderr)
        return 1
    st = c.status()
    failed = sum(v["failed"] for v in st.values())
    if not args.quiet:
        print(f"[reliability] complete -- summary: {out}; report: {c.dir / 'report.html'}"
              + (f"  ({failed} item(s) failed; --retry-failed to re-queue)" if failed else ""))
    return 0


def _selftest(args) -> int:
    try:
        from linac_gen.reliability.selftest import run_selftest, format_result
    except ImportError as exc:
        print(f"error: self-test unavailable: {exc}", file=sys.stderr)
        return 2
    try:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
        res = run_selftest(quick=bool(args.quick) or not args.regression, regression=bool(args.regression),
                           before_dir=args.before, out_dir=args.out, regression_cases=cases)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res, indent=1, default=str))
    else:
        print(format_result(res))
    return {"PASS": 0, "FAIL": 1, "REFUSED": 1}.get(res.get("verdict"), 2)
