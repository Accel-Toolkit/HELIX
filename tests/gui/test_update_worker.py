"""GitUpdateWorker against a local origin: ff-update and cancel."""
from __future__ import annotations

import shutil
import subprocess

import pytest

pytest.importorskip("PyQt6")

git_missing = shutil.which("git") is None


def _run(args, cwd):
    subprocess.run(args, cwd=str(cwd), check=True,
                   capture_output=True, text=True)


@pytest.fixture()
def behind_clone(tmp_path):
    """A clone whose origin/main is one commit ahead."""
    origin = tmp_path / "origin.git"
    _run(["git", "init", "--bare", "-b", "main", str(origin)], tmp_path)
    seed = tmp_path / "seed"
    _run(["git", "clone", str(origin), str(seed)], tmp_path)
    for msg in ("one", "two"):
        _run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
              "commit", "--allow-empty", "-m", msg], seed)
    _run(["git", "push", "origin", "HEAD:main"], seed)
    work = tmp_path / "user_clone"
    _run(["git", "clone", str(origin), str(work)], tmp_path)
    _run(["git", "checkout", "-B", "main", "origin/main"], work)
    _run(["git", "reset", "--hard", "HEAD~1"], work)
    return work


def _head(cwd) -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                         capture_output=True, text=True)
    return out.stdout.strip()


@pytest.mark.skipif(git_missing, reason="git not available")
def test_ff_update_advances_head(qapp, behind_clone):
    from PyQt6.QtCore import QCoreApplication

    from linac_gen_gui.interphase.dialogs.update_dialog import (
        GitUpdateWorker)

    before = _head(behind_clone)
    results = {}
    w = GitUpdateWorker(behind_clone)
    w.finished_ok.connect(lambda s: results.setdefault("ok", s))
    w.failed.connect(lambda s: results.setdefault("fail", s))
    w.start()
    assert w.wait(20000)
    QCoreApplication.processEvents()
    assert "ok" in results, results
    assert _head(behind_clone) != before
    st = subprocess.run(["git", "status", "--porcelain"],
                        cwd=str(behind_clone), capture_output=True,
                        text=True)
    assert st.stdout.strip() == ""          # worktree stays clean


@pytest.mark.skipif(git_missing, reason="git not available")
def test_pre_cancelled_leaves_tree_untouched(qapp, behind_clone):
    from PyQt6.QtCore import QCoreApplication

    from linac_gen_gui.interphase.dialogs.update_dialog import (
        GitUpdateWorker)

    before = _head(behind_clone)
    results = {}
    w = GitUpdateWorker(behind_clone)
    w.request_stop()                         # cancel before it starts
    w.cancelled.connect(lambda: results.setdefault("cancelled", True))
    w.finished_ok.connect(lambda s: results.setdefault("ok", s))
    w.start()
    assert w.wait(20000)
    QCoreApplication.processEvents()
    assert results.get("cancelled"), results
    assert _head(behind_clone) == before
