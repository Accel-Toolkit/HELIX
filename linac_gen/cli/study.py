"""``python -m linac_gen study`` — parameter studies with per-run folders.

Verbs
-----
plan       print the expanded run table, execute nothing
run        create-if-needed, then execute; RESUME-BY-DEFAULT (completed
           runs are skipped; --force wipes runs/; --retry-failed
           re-queues failures)
resume     alias of run (requires an existing study dir)
summarize  rebuild summary/summary.csv from the per-run status files

The study spec is a ``study.json`` (see linac_gen/study/spec.py); the
study directory defaults to ``<json_parent>/<spec.name>/`` or --dir.
Exit codes: 0 success, 1 execution failure, 2 bad input/spec.
"""
from __future__ import annotations

import sys
from pathlib import Path


def add_arguments(p) -> None:
    p.add_argument("verb",
                   choices=("plan", "run", "resume", "summarize"))
    p.add_argument("target",
                   help="study.json (plan/run) or study directory "
                        "(run/resume/summarize)")
    p.add_argument("--dir", dest="dir_", default=None,
                   help="study directory (default: <json dir>/<name>/)")
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="worker processes (default 1 = serial)")
    p.add_argument("--serial", action="store_true",
                   help="force in-process serial execution")
    p.add_argument("--force", action="store_true",
                   help="wipe runs/ and start over")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-queue runs whose status is 'failed'")
    p.add_argument("-q", "--quiet", action="store_true")


def _resolve(target: str, dir_: str | None):
    """Return (study_dir, spec_path_or_None)."""
    t = Path(target)
    if t.is_dir():
        return t, None
    if t.suffix == ".json" or t.name == "study.json":
        study_dir = Path(dir_) if dir_ else None
        return study_dir, t
    raise ValueError(
        f"{target}: expected a study.json or a study directory")


def run(args) -> int:
    from linac_gen.study.engine import StudyManager
    from linac_gen.study.spec import load_spec
    from linac_gen.study.strategies import expand_runs

    try:
        study_dir, spec_path = _resolve(args.target, args.dir_)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.verb == "plan":
        try:
            spec = load_spec(spec_path) if spec_path else \
                load_spec(Path(study_dir) / "study.json")
            runs = expand_runs(spec)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"[study] {spec.name}: {len(runs)} run(s), "
              f"strategy={spec.strategy}, mode={spec.mode}, "
              f"repeats={spec.repeats}")
        for r in runs:
            pv = "  ".join(f"{s}={v:.6g}" for s, v in r.params)
            print(f"  {r.index:5d}  seed={r.seed:<6d} {pv}")
        return 0

    if args.verb == "summarize":
        try:
            mgr = StudyManager.load(study_dir or Path(args.target))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        out = mgr.summarize()
        print(f"[study] summary written: {out}")
        return 0

    # run / resume
    try:
        if spec_path is not None:
            spec = load_spec(spec_path)
            if study_dir is None:
                study_dir = spec_path.parent / spec.name
            if (Path(study_dir) / "study.json").exists():
                mgr = StudyManager.load(study_dir)
            else:
                # resolve the input relative to the SPEC's directory
                inp = Path(spec.input)
                if not inp.is_absolute():
                    spec.input = str((spec_path.parent / inp).resolve())
                mgr = StudyManager.create(study_dir, spec)
        else:
            mgr = StudyManager.load(study_dir or Path(args.target))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    total = len(mgr.plan())
    todo = len(mgr.pending(retry_failed=args.retry_failed))
    if not args.quiet:
        print(f"[study] {mgr.spec.name}: {total} run(s), {todo} to "
              f"execute, mode={mgr.spec.mode}, "
              f"parallel={args.parallel}, dir={mgr.study_dir}")

    def _progress(p) -> None:
        if args.quiet:
            return
        eta = f", ETA {p.eta_s/60.0:.1f} min" if p.eta_s else ""
        print(f"[study] {p.done}/{p.total} done"
              f" ({p.failed} failed){eta}", flush=True)

    try:
        out = mgr.run(max_workers=args.parallel, serial=args.serial,
                      force=args.force, retry_failed=args.retry_failed,
                      progress_cb=_progress)
    except Exception as exc:                             # noqa: BLE001
        print(f"error: study execution failed: {exc}", file=sys.stderr)
        return 1
    failed = sum(1 for r in mgr.plan()
                 if (mgr._status(r) or {}).get("status") == "failed")
    if not args.quiet:
        print(f"[study] complete -- summary: {out}"
              + (f"  ({failed} run(s) failed; --retry-failed to "
                 f"re-queue)" if failed else ""))
    return 0
