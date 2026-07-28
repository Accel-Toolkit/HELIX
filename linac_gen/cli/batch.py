"""``python -m linac_gen batch`` — a multi-run campaign from a job file.

The job file is JSON: either a list of jobs or ``{"jobs": [ … ]}``.  Each
job describes one simulation::

    {"name": "low_I",  "input": "cell.dat", "mode": "envelope",
     "beam": {"current": 2.0}, "set": {"QF.gradient": 8.5}, "sc": {}}

All jobs run through the same process-pool engine as ``scan``; a
``batch_summary.csv`` collects one row of end-of-lattice metrics per job.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from linac_gen.cli import common

_METRICS = ("sigma_x", "sigma_y", "sigma_phi", "sigma_w", "emit_x", "emit_y",
            "emit_z", "transmission", "ref_w_kin", "x_max", "y_max",
            "elapsed")


def add_arguments(p) -> None:
    """Populate the ``batch`` sub-parser."""
    p.add_argument("jobfile", help="JSON job file")
    p.add_argument("--out", default=".",
                   help="output directory for batch_summary.csv (default .)")
    p.add_argument("--parallel", type=int, default=1,
                   help="worker processes (>1 → process pool)")
    p.add_argument("-q", "--quiet", action="store_true")


def _load_jobs(path: Path) -> list:
    data = json.loads(path.read_text())
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("job file must contain a non-empty list of jobs")
    return jobs


def run(args) -> int:
    """Execute the ``batch`` subcommand; return a process exit code."""
    jobfile = Path(args.jobfile)
    if not jobfile.is_file():
        print(f"error: job file not found: {args.jobfile}", file=sys.stderr)
        return 2
    try:
        jobs = _load_jobs(jobfile)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    names, inputs, modes, points = [], [], [], []
    for i, job in enumerate(jobs):
        try:
            name = str(job.get("name", f"job{i + 1}"))
            inp = job["input"]
            # A relative job input is resolved CWD-relative first, then
            # against the job-file directory (so job files stay portable).
            ipath = Path(inp)
            if not ipath.is_absolute():
                for base in (Path.cwd(), jobfile.parent):
                    if (base / inp).exists():
                        ipath = base / inp
                        break
                else:
                    ipath = jobfile.parent / inp
            mode = str(job.get("mode", "envelope"))
            elem = list(job.get("set", {}).items())
            point = common.build_scan_point(
                str(ipath),
                beam_overrides=dict(job.get("beam", {})),
                element_overrides=elem,
                sc_overrides=dict(job.get("sc", {})),
                mode=mode,
                env_solver=str(job.get("env_solver", "matrix")),
                seed=int(job.get("seed", 42)),
            )
        except (KeyError, ValueError) as exc:
            print(f"error: job {i + 1} ({job.get('name', '?')}): {exc}",
                  file=sys.stderr)
            return 2
        names.append(name)
        inputs.append(str(ipath))
        modes.append(mode)
        points.append(point)

    if not args.quiet:
        print(f"[batch] {len(points)} job(s), parallel={args.parallel}")
    try:
        from linac_gen.parallel.scan_pool import (
            run_scan_points, run_scan_points_serial,
        )
        if args.parallel > 1:
            results = run_scan_points(points, max_workers=args.parallel)
        else:
            results = run_scan_points_serial(points)
    except Exception as exc:                                # noqa: BLE001
        print(f"error: batch failed: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "batch_summary.csv"
    with summary.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "input", "mode", *_METRICS])
        for name, inp, mode, res in zip(names, inputs, modes, results):
            row = [name, inp, mode]
            for m in _METRICS:
                v = res.get(m)
                row.append("" if v is None else f"{v:.8g}")
            w.writerow(row)
    if not args.quiet:
        print(f"[batch] wrote {summary}  ({len(results)} jobs)")
    return 0
