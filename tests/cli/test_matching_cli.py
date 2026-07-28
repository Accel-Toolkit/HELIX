"""End-to-end smoke tests for ``python -m linac_gen.matching``."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE = REPO / "tests" / "io" / "fixtures" / "lattice_with_set_commands.dat"


def _run(*args, **kw) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "linac_gen.matching", *args],
        cwd=str(REPO),
        capture_output=True, text=True, env=env, **kw,
    )


def test_cli_runs_on_fixture(tmp_path):
    # The kitchen-sink fixture carries a constraint that is inert under the
    # default envelope cost-solver, so the pre-run audit refuses unless we
    # opt in with --allow-inert-constraints (see test_cli_refuses_* below).
    out = tmp_path / "matched.dat"
    cp = _run(str(FIXTURE), "--out", str(out), "--report",
              "--allow-inert-constraints")
    assert "Matching report" in cp.stdout
    assert out.exists()


def test_cli_refuses_inert_constraint(tmp_path):
    """Without --allow-inert-constraints, the CLI must refuse (non-zero
    exit) rather than silently ignore the fixture's envelope-inert
    constraint — the shell-level counterpart of the engine audit.  With
    the flag (see test_cli_runs_on_fixture) the same deck runs."""
    cp = _run(str(FIXTURE), "--no-write")
    assert cp.returncode != 0
    assert "silently ignore" in cp.stderr


def test_cli_no_write_skips_output(tmp_path):
    cp = _run(str(FIXTURE), "--no-write")
    assert "wrote" not in cp.stdout


def test_cli_missing_input_fails():
    cp = _run("/no/such/path.dat")
    assert cp.returncode != 0
    assert "not found" in cp.stderr.lower()


# -----------------------------------------------------------------
# Recent kwarg coverage -- these flags MUST reach match() and be
# echoed in the run header so users / scripts can verify routing.
# -----------------------------------------------------------------
def test_cli_cost_solver_envelope_default(tmp_path):
    """--cost-solver=envelope is the default; verify it shows in header."""
    cp = _run(str(FIXTURE), "--no-write", "--max-iter", "2")
    assert "cost_solver=envelope" in cp.stdout


def test_cli_cost_solver_mp_smoke(tmp_path):
    """Explicit --cost-solver=mp routes to the MP path.  Use a tiny
    particle count + short run so the test stays under a few seconds."""
    cp = _run(str(FIXTURE), "--no-write", "--max-iter", "2",
              "--cost-solver", "mp", "--mp-n-particles", "50")
    assert "cost_solver=mp" in cp.stdout


def test_cli_seqscan_all_kwargs_accepted(tmp_path):
    """Every seqscan_* flag should parse without error.  Smoke-only --
    the fixture may not match perfectly; we just need a clean exit
    proving the kwargs reached match()."""
    cp = _run(str(FIXTURE), "--no-write",
              "--algorithm", "sequential_scan",
              "--seqscan-passes", "1",
              "--seqscan-steps", "3",
              "--seqscan-step-frac", "0.05",
              "--seqscan-reversal", "any_grew",
              "--seqscan-threshold", "seed_exit")
    # Reaching the run header proves the parser accepted all the flags.
    assert "algorithm=sequential_scan" in cp.stdout


def test_cli_seqscan_threshold_invalid_rejected():
    """Unknown --seqscan-threshold value must fail argparse, not be
    silently passed through."""
    cp = _run(str(FIXTURE), "--no-write",
              "--algorithm", "sequential_scan",
              "--seqscan-threshold", "not_a_real_mode")
    assert cp.returncode != 0
    assert "seqscan-threshold" in cp.stderr or "invalid choice" in cp.stderr
