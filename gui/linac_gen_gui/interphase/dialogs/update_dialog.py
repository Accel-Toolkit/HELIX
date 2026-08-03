"""In-app updater: guarded git fetch + fast-forward with progress.

Only ever launched for a PUBLIC_CLEAN install (an unmodified clone of
the public repo on main — see services/update_check.py).  Two phases:

1. ``git fetch --progress origin main`` — the long part, streamed, its
   stderr percentages fed to the progress bar; CANCELLABLE (an aborted
   fetch never touches the worktree).
2. ``git merge --ff-only FETCH_HEAD`` — fast (the public release ladder
   is linear), Cancel is greyed out during it, and ``--ff-only``
   guarantees no merge state can ever be created.

Both run with ``GIT_TERMINAL_PROMPT=0`` so a misconfigured credential
helper fails instead of hanging invisibly.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QLabel, QProgressBar, QPushButton,
                             QVBoxLayout)

from ..services.update_check import parse_git_progress


class GitUpdateWorker(QThread):
    progress = pyqtSignal(str, int)      # (stage, percent)
    phase = pyqtSignal(str)              # "Fetching…" / "Applying…"
    finished_ok = pyqtSignal(str)        # summary text
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, repo_root, parent=None):
        super().__init__(parent)
        self._repo_root = str(repo_root)
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _env(self) -> dict:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def run(self) -> None:                              # noqa: C901
        try:
            self.phase.emit("Fetching update…")
            proc = subprocess.Popen(
                ["git", "fetch", "--progress", "origin", "main"],
                cwd=self._repo_root, env=self._env(),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, bufsize=0)
            buf = ""
            while True:
                if self._stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    self.cancelled.emit()
                    return
                chunk = proc.stderr.read(256)
                if chunk:
                    buf = (buf + chunk)[-4096:]
                    got = parse_git_progress(chunk)
                    if got:
                        self.progress.emit(*got)
                elif proc.poll() is not None:
                    break
            if proc.returncode != 0:
                self.failed.emit(
                    f"git fetch failed:\n{buf.strip()[-500:]}")
                return

            self.phase.emit("Applying update…")
            merge = subprocess.run(
                ["git", "merge", "--ff-only", "FETCH_HEAD"],
                cwd=self._repo_root, env=self._env(),
                capture_output=True, text=True, timeout=60)
            if merge.returncode != 0:
                self.failed.emit(
                    "fast-forward failed (local history diverged):\n"
                    f"{(merge.stderr or merge.stdout).strip()[-500:]}")
                return
            head = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self._repo_root, capture_output=True, text=True,
                timeout=5)
            summary = ("Already up to date."
                       if "Already up to date" in merge.stdout
                       else f"Updated to {head.stdout.strip()}.")
            self.finished_ok.emit(summary)
        except Exception as exc:                        # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class UpdateProgressDialog(QDialog):
    """Non-modal progress window for the updater (offscreen-safe)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Updating HELIX")
        self.cancel_cb = None            # bound to the worker by the app
        v = QVBoxLayout(self)
        self._phase = QLabel("Preparing…")
        self._stage = QLabel("")
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self._on_cancel)
        v.addWidget(self._phase)
        v.addWidget(self._bar)
        v.addWidget(self._stage)
        v.addWidget(self._cancel)
        self.setMinimumWidth(380)

    # slots wired by the app --------------------------------------------
    def on_phase(self, text: str) -> None:
        self._phase.setText(text)
        if text.startswith("Applying"):
            # the ff-merge must not be interrupted
            self._cancel.setEnabled(False)
            self._bar.setRange(0, 0)     # indeterminate, it's quick

    def on_progress(self, stage: str, pct: int) -> None:
        if self._bar.maximum() != 100:
            self._bar.setRange(0, 100)
        self._bar.setValue(int(pct))
        self._stage.setText(f"{stage}: {pct} %")

    def _on_cancel(self) -> None:
        cb = self.cancel_cb
        if cb is not None:
            self._cancel.setEnabled(False)
            self._stage.setText("cancelling…")
            cb()

    def closeEvent(self, ev) -> None:                   # noqa: N802
        self._on_cancel()
        super().closeEvent(ev)
