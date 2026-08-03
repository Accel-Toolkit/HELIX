"""Update-check logic: version compare, release lookup, install state.

Pure stdlib — no Qt — so every function unit-tests headlessly.  The GUI
(app.py) runs :func:`fetch_latest_release` on a daemon thread and
delivers the result through a queued signal; failures of any kind
return ``None`` and the launch path stays silent (a user who is offline
must never see an error about a feature they didn't invoke).

Install states drive what "update" means:

* ``DEV``          — origin is the maintainer's HELIX-dev clone: the
                     feature is inert (updates come from git directly).
* ``PUBLIC_CLEAN`` — unmodified clone of Accel-Toolkit/HELIX on main:
                     eligible for the guarded in-app ``git pull``.
* ``PUBLIC_DIRTY`` — public clone with local edits / detached HEAD /
                     side branch: never pulled over; browser fallback.
* ``NOT_GIT``      — zip download or foreign remote: browser fallback.
"""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from enum import Enum
from pathlib import Path

PUBLIC_REPO = "Accel-Toolkit/HELIX"
API_LATEST = ("https://api.github.com/repos/"
              f"{PUBLIC_REPO}/releases/latest")
RELEASES_PAGE = f"https://github.com/{PUBLIC_REPO}/releases/latest"


class InstallState(Enum):
    DEV = "dev"
    PUBLIC_CLEAN = "public_clean"
    PUBLIC_DIRTY = "public_dirty"
    NOT_GIT = "not_git"


# ----------------------------------------------------------------------
# versions
# ----------------------------------------------------------------------
def parse_version(s: str):
    """'v1.4' / '1.4' / '1.4.2' -> (1, 4) / (1, 4, 2); junk -> None."""
    if not s:
        return None
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", s.strip())
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def is_newer(latest_tag: str, installed: str) -> bool:
    """True iff latest_tag represents a strictly newer version.

    Unparseable input on either side -> False (never nag on garbage).
    """
    a = parse_version(latest_tag)
    b = parse_version(installed)
    if a is None or b is None:
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


# ----------------------------------------------------------------------
# release lookup
# ----------------------------------------------------------------------
def fetch_latest_release(url: str = API_LATEST, timeout: float = 3.0,
                         opener=None):
    """(tag_name, html_url) of the latest GitHub release, or None.

    ``opener(request, timeout)`` is injectable for tests; the default
    is ``urllib.request.urlopen``.  ANY failure — DNS, timeout,
    rate-limit 403, JSON drift — returns None: the silence contract.
    """
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "HELIX-update-check",
            "Accept": "application/vnd.github+json",
        })
        open_fn = opener or urllib.request.urlopen
        with open_fn(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tag = data["tag_name"]
        page = data.get("html_url") or RELEASES_PAGE
        if parse_version(tag) is None:
            return None
        return str(tag), str(page)
    except Exception:                                   # noqa: BLE001
        return None


# ----------------------------------------------------------------------
# install classification
# ----------------------------------------------------------------------
def _git(repo_root, args, runner=subprocess.run):
    """Guarded git call (style of io/hdf5_output._git_commit)."""
    try:
        out = runner(["git", *args], capture_output=True, text=True,
                     timeout=5, cwd=str(repo_root))
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:                                   # noqa: BLE001
        return None


def _normalize_remote(url: str) -> str:
    """https/ssh spellings of the same GitHub repo -> 'owner/repo'."""
    u = url.strip().rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    m = re.search(r"github\.com[:/]([^/]+/[^/]+)$", u)
    return m.group(1) if m else u


def classify_install(repo_root, runner=subprocess.run) -> InstallState:
    top = _git(repo_root, ["rev-parse", "--show-toplevel"], runner)
    if top is None or Path(top).resolve() != Path(repo_root).resolve():
        return InstallState.NOT_GIT
    origin = _git(repo_root, ["remote", "get-url", "origin"], runner)
    if origin is None:
        return InstallState.NOT_GIT
    repo = _normalize_remote(origin)
    if "HELIX-dev" in repo:
        return InstallState.DEV          # maintainer clone: always inert
    if repo.lower() != PUBLIC_REPO.lower():
        return InstallState.NOT_GIT
    dirty = _git(repo_root, ["status", "--porcelain"], runner)
    if dirty is None or dirty:
        return InstallState.PUBLIC_DIRTY
    branch = _git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"],
                  runner)
    if branch != "main":
        # detached HEAD / side branch: pulling would fail or surprise
        return InstallState.PUBLIC_DIRTY
    return InstallState.PUBLIC_CLEAN


def find_repo_root():
    """The git worktree this running code lives in, or None."""
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists():
            return p
    return None


# ----------------------------------------------------------------------
# git progress parsing (fetch/pull --progress stderr)
# ----------------------------------------------------------------------
_PROGRESS_RE = re.compile(
    r"(?:remote:\s*)?"
    r"(Counting objects|Compressing objects|Receiving objects|"
    r"Resolving deltas):\s+(\d+)%")


def parse_git_progress(chunk: str):
    """Latest ('stage', pct) in a stderr chunk, or None.

    git rewrites its progress lines with carriage returns, so a chunk
    may hold many updates — the LAST match is the current state.
    """
    last = None
    for piece in re.split(r"[\r\n]", chunk):
        m = _PROGRESS_RE.search(piece)
        if m:
            last = (m.group(1), int(m.group(2)))
    return last
