"""Verify every Python code block in the HELIX manual still runs.

Walks ``docs/manual/**/*.md``, extracts every fenced code block of the
form ```python or ```py, and executes it.  Blocks within one chapter
share a namespace and run IN ORDER — matching how a reader works
through a chapter (later blocks may use variables defined earlier,
exactly like a REPL session).  The namespace resets at each new file.
Reports failures with the file path so the problem is easy to locate.

Snippet conventions:

* Code in fences tagged ``python`` (or ``py``) is executed.
* Code in fences tagged ``python skip`` is parsed but not executed
  (use for snippets that legitimately can't run: API-signature
  pseudo-code, GUI-only flows, long-running cluster jobs).
* Code in fences tagged ``pycon`` (REPL output) is parsed but not
  executed.

Usage:
    python docs/manual/_build/verify_snippets.py             # run everything
    python docs/manual/_build/verify_snippets.py 03_elements # filter by path
    python docs/manual/_build/verify_snippets.py --list      # list snippets

CI integration: invoke from a pytest test file (see
``tests/docs/test_manual_snippets.py``) so failures show up alongside
regular tests.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path

MANUAL_ROOT = Path(__file__).resolve().parent.parent

# Snippets must exercise THIS repo's code — not whatever linac_gen an
# editable/site-packages install resolves to (a stale editable install
# once pointed at an old archive copy and silently validated the wrong
# codebase).  Repo root first on sys.path wins the import race.
_REPO_ROOT = MANUAL_ROOT.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "gui")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
EXEC_TAGS = {"python", "py"}
SKIP_TAGS = {"python skip", "py skip", "pycon", "pycon3", "console"}

_FENCE_RE = re.compile(
    r"^```(?P<tag>[^\n]*?)\n(?P<body>.*?)^```",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Snippet:
    file: Path
    line: int                  # 1-based line number of the opening fence
    tag: str
    body: str
    skip: bool = False


@dataclass
class FailedSnippet:
    snippet: Snippet
    excinfo: str
    stdout: str = ""
    stderr: str = ""


def collect_snippets(roots: list[Path]) -> list[Snippet]:
    """Walk markdown files under ``roots`` and harvest fenced code blocks."""
    out: list[Snippet] = []
    for root in roots:
        for path in root.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            offset = 0
            for m in _FENCE_RE.finditer(text):
                tag = m.group("tag").strip().lower()
                body = m.group("body")
                # 1-based line number of the ``` fence
                line = text.count("\n", 0, m.start()) + 1
                if tag in EXEC_TAGS:
                    out.append(Snippet(path, line, tag, body))
                elif tag in SKIP_TAGS or tag.startswith("python skip"):
                    out.append(Snippet(path, line, tag, body, skip=True))
                # else: not a Python snippet; ignore
                offset = m.end()
    return out


def run_snippet(s: Snippet, ns: dict) -> FailedSnippet | None:
    """Execute one snippet in ``ns``; return None on success.

    ``ns`` is the per-chapter namespace — the caller passes the same
    dict for every snippet of one file so blocks continue each other.
    """
    if s.skip:
        return None
    out, err = io.StringIO(), io.StringIO()
    cwd = os.getcwd()
    try:
        # Run from repo root so file paths in snippets resolve.
        os.chdir(MANUAL_ROOT.parent.parent)
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(s.body, str(s.file), "exec"), ns)  # noqa: S102
    except SystemExit:
        # ``sys.exit(0)`` in a snippet is fine.
        pass
    except Exception:                                             # noqa: BLE001
        return FailedSnippet(s, traceback.format_exc(),
                             out.getvalue(), err.getvalue())
    finally:
        os.chdir(cwd)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filters", nargs="*",
                        help="path-substring filters (e.g. '03_elements')")
    parser.add_argument("--list", action="store_true",
                        help="list snippets without running")
    args = parser.parse_args()

    snippets = collect_snippets([MANUAL_ROOT])
    if args.filters:
        snippets = [s for s in snippets
                    if any(f in str(s.file) for f in args.filters)]

    if args.list:
        for s in snippets:
            mark = "SKIP" if s.skip else "RUN "
            print(f"{mark} {s.file.relative_to(MANUAL_ROOT)}:{s.line}  ({s.tag})")
        return 0

    n_run = sum(1 for s in snippets if not s.skip)
    n_skip = len(snippets) - n_run
    print(f"Running {n_run} snippet(s) ({n_skip} skipped)…")

    failures: list[FailedSnippet] = []
    current_file: Path | None = None
    ns: dict = {}
    for s in snippets:
        if s.file != current_file:
            # New chapter → fresh namespace (blocks within one chapter
            # build on each other; chapters are independent).
            current_file = s.file
            ns = {"__name__": "__manual_snippet__"}
        if not s.skip:
            # Progress line BEFORE executing — with flush, so a snippet
            # that hard-crashes the interpreter is still identifiable.
            print(f"  … {s.file.relative_to(MANUAL_ROOT)}:{s.line}",
                  flush=True)
        f = run_snippet(s, ns)
        if f is not None:
            failures.append(f)

    if not failures:
        print(f"\nAll {n_run} snippets passed.")
        return 0

    print(f"\n{len(failures)} snippet(s) FAILED:\n")
    for f in failures:
        rel = f.snippet.file.relative_to(MANUAL_ROOT)
        print("─" * 72)
        print(f"FAIL {rel}:{f.snippet.line}")
        print(textwrap.indent(f.excinfo.rstrip(), "    "))
        if f.stdout.strip():
            print("    stdout:", textwrap.indent(f.stdout.rstrip(), "      ").lstrip())
        if f.stderr.strip():
            print("    stderr:", textwrap.indent(f.stderr.rstrip(), "      ").lstrip())
    return 1


if __name__ == "__main__":
    sys.exit(main())
