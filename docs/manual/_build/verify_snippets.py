"""Verify every Python code block in the HELIX manual still runs.

Walks ``docs/manual/**/*.md``, extracts every fenced code block of the
form ```python or ```py, and executes it.  Blocks within one chapter
share a namespace and run IN ORDER — matching how a reader works
through a chapter (later blocks may use variables defined earlier,
exactly like a REPL session).  The namespace resets at each new file.
Reports failures with the file path so the problem is easy to locate.

Snippet conventions:

* Code in fences tagged ``python`` (or ``py``) is executed.
* Code in fences tagged ``{.python .skip}`` (attr_list form — the
  manual's house style) is parsed but not executed: use it for
  snippets that legitimately can't run (API-signature pseudo-code,
  GUI-only flows, long-running cluster jobs).  The legacy space forms
  ``python skip``, ``pycon``, ``pycon3`` and ``console`` are also
  parsed-not-run.
* Code in fences tagged ``{.python data-needs="path …"}`` runs only
  when every listed path (relative to the repo root) exists; otherwise
  it is skipped with an honest ``data absent: …`` reason (same idea as
  ``tests/dataguard.py``).  Use it for snippets reading inputs that
  are not distributed with every checkout.
* Code in fences tagged ``{.python data-requires="module …"}`` runs
  only when every listed module is importable; otherwise it is skipped
  with ``module absent: …``.  Use it for optional dependencies.

Every python fence is therefore either executed or explicitly skipped
with a reason; the summary reports exact counts and lists the
data/module skips so a green run still shows what was not verified.

Usage:
    python docs/manual/_build/verify_snippets.py             # run everything
    python docs/manual/_build/verify_snippets.py 03_elements # filter by path
    python docs/manual/_build/verify_snippets.py --list      # list snippets

CI integration: the "Verify code snippets" step of
``.github/workflows/docs.yml`` runs this script (blocking), and
``tests/docs/test_manual_snippets.py`` replays the same loop inside
the regular test suite.
"""
from __future__ import annotations

import argparse
import importlib.util
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
# attr_list fence form: ```{.python .skip}, ```{.python data-needs="…"}
_ATTR_RE = re.compile(r"^\{(?P<body>[^}]*)\}$")
_KV_RE = re.compile(r'(?P<k>[\w-]+)="(?P<v>[^"]*)"')


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def classify_tag(tag: str) -> "tuple[bool, str] | None":
    """Classify one fence tag.

    Returns ``None`` for a non-Python fence (ignored), ``(False, "")``
    for a fence to execute, and ``(True, reason)`` for a fence that is
    collected but skipped.  ``tag`` is the raw (stripped) text after
    the opening triple-backtick.
    """
    m = _ATTR_RE.match(tag)
    if m is None:                       # legacy space form, unchanged
        low = tag.lower()
        if low in EXEC_TAGS:
            return (False, "")
        if low in SKIP_TAGS or low.startswith("python skip"):
            return (True, "tagged .skip")
        return None
    body = m.group("body")
    toks = body.split()
    classes = [t[1:] for t in toks if t.startswith(".")]
    if not classes or classes[0].lower() not in EXEC_TAGS:
        return None
    if any(c.lower() == "skip" for c in classes[1:]):
        return (True, "tagged .skip")
    kv = dict(_KV_RE.findall(body))
    missing = [p for p in kv.get("data-needs", "").split()
               if not (_REPO_ROOT / p).exists()]
    if missing:
        return (True, "data absent: " + ", ".join(missing))
    missing = [x for x in kv.get("data-requires", "").split()
               if not _module_present(x)]
    if missing:
        return (True, "module absent: " + ", ".join(missing))
    return (False, "")


@dataclass
class Snippet:
    file: Path
    line: int                  # 1-based line number of the opening fence
    tag: str
    body: str
    skip: bool = False
    reason: str = ""           # why a skipped snippet is skipped


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
            for m in _FENCE_RE.finditer(text):
                # Raw (not lowercased): data-needs paths are
                # case-sensitive.  classify_tag lowercases per branch.
                tag = m.group("tag").strip()
                body = m.group("body")
                # 1-based line number of the ``` fence
                line = text.count("\n", 0, m.start()) + 1
                c = classify_tag(tag)
                if c is None:
                    continue            # not a Python snippet; ignore
                skip, reason = c
                out.append(Snippet(path, line, tag, body,
                                   skip=skip, reason=reason))
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
            rel = s.file.relative_to(MANUAL_ROOT)
            if s.skip:
                print(f"SKIP {rel}:{s.line}  ({s.tag}) — {s.reason}")
            else:
                print(f"RUN  {rel}:{s.line}  ({s.tag})")
        return 0

    n_run = sum(1 for s in snippets if not s.skip)
    n_skip = len(snippets) - n_run
    n_tag = sum(1 for s in snippets if s.reason == "tagged .skip")
    n_data = sum(1 for s in snippets if s.reason.startswith("data absent"))
    n_mod = sum(1 for s in snippets if s.reason.startswith("module absent"))
    print(f"Running {n_run} snippet(s); skipping {n_skip} "
          f"({n_tag} tagged .skip, {n_data} data absent, "
          f"{n_mod} module absent)…")

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
        print(f"\nAll {n_run} snippets passed; {n_skip} skipped.")
        # A green run still shows what was NOT verified here.
        for s in snippets:
            if s.skip and s.reason != "tagged .skip":
                print(f"  skipped {s.file.relative_to(MANUAL_ROOT)}"
                      f":{s.line} — {s.reason}")
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
