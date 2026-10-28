"""CI's ring shards (.github/workflows/tests.yml): every ring test file runs in
exactly one shard of the ring job, and the core legs run none of them.

The third shard is "everything the first two leave out" (``--ignore`` of
their files), so a new ring test file runs there without a change here; this
test catches a shard that names a missing file, a file run twice, or an
``--ignore`` list that no longer matches the explicit shards.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "tests.yml"
RING_PATHS = ("tests/ring", "tests/cli/test_ring_cli.py")


def _jobs() -> dict:
    yaml = pytest.importorskip("yaml")
    if not WORKFLOW.is_file():
        pytest.skip("no CI workflow in this tree")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _pytest_command(job: dict) -> str:
    return next(s["run"] for s in job["steps"] if "pytest" in str(s.get("run", "")))


def _test_files(path: str) -> set[str]:
    p = REPO / path
    if p.is_file():
        return {path}
    return {f.relative_to(REPO).as_posix() for f in p.rglob("test_*.py")}


def _shard_files(args: str) -> set[str]:
    """The test files ``pytest <args>`` runs (paths given, minus --ignore)."""
    tokens = shlex.split(args)
    ignored = {t.split("=", 1)[1] for t in tokens if t.startswith("--ignore=")}
    for t in ignored:
        assert (REPO / t).is_file(), f"--ignore of a missing file: {t}"
    paths = [t for t in tokens if not t.startswith("-")]
    files: set[str] = set()
    for p in paths:
        assert (REPO / p).exists(), f"a shard names a missing path: {p}"
        files |= _test_files(p)
    return files - ignored


def test_every_ring_test_file_runs_in_exactly_one_shard():
    jobs = _jobs()
    shards = jobs["ring"]["strategy"]["matrix"]["shard"]
    assert len(shards) >= 2
    assert "${{ matrix.shard.args }}" in _pytest_command(jobs["ring"])
    per_shard = [_shard_files(s["args"]) for s in shards]
    seen: dict[str, str] = {}
    for shard, files in zip(shards, per_shard, strict=True):
        assert files, f"shard {shard['name']} runs nothing"
        for f in files:
            assert f not in seen, f"{f} runs in shards {seen[f]} and {shard['name']}"
            seen[f] = shard["name"]
    expected = set().union(*(_test_files(p) for p in RING_PATHS))
    assert set(seen) == expected, sorted(expected ^ set(seen))


def test_the_core_legs_run_no_ring_tests():
    jobs = _jobs()
    for name in ("linux", "macos", "windows"):
        tokens = shlex.split(_pytest_command(jobs[name]).replace("\\", "/"))
        ring = [t for t in tokens if t.startswith(RING_PATHS)]
        assert not ring, f"the {name} leg runs ring tests: {ring}"
