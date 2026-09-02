"""Pytest wrapper for the manual snippet verifier.

This is the test file the verifier's docstring promises: it loads
``docs/manual/_build/verify_snippets.py`` (the ``_build`` directory is
not a package, so it is imported by file path), unit-tests the fence
classification in both regimes of every guard, structurally pins that
no python-looking fence in the manual is silently ignored, and replays
the verifier's own per-chapter loop end-to-end (slow, parametrized).

The docs CI step (``.github/workflows/docs.yml`` "Verify code
snippets") runs the same verifier over the whole manual; these tests
mirror it inside the regular test suite.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

# Snippets that plot must never try to open a native window under pytest.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
MANUAL = REPO / "docs" / "manual"
_VPATH = MANUAL / "_build" / "verify_snippets.py"

_spec = importlib.util.spec_from_file_location("verify_snippets", _VPATH)
_V = importlib.util.module_from_spec(_spec)
sys.modules["verify_snippets"] = _V   # dataclass decorator resolves via sys.modules
_spec.loader.exec_module(_V)

# Opening-fence tags that LOOK like Python.  ``pycon`` does not match
# (\b fails between "py" and "con") — that form is a legacy skip tag.
_PY_LOOKING = re.compile(r"^(?:\{\.)?(?:python|py)\b")


def _write_md(tmp_path: Path, name: str, *fences: tuple[str, str]) -> None:
    parts = ["Some prose.\n"]
    for tag, body in fences:
        parts.append(f"```{tag}\n{body}\n```\n")
    (tmp_path / name).write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# classification unit tests (both regimes of every guard)
# ---------------------------------------------------------------------------

def test_attr_list_skip_fence_is_collected_as_skip(tmp_path):
    _write_md(tmp_path, "a.md",
              ("{.python .skip}", 'raise RuntimeError("never run")'),
              ("python skip", 'raise RuntimeError("never run")'),
              ("pycon", ">>> 1 + 1"))
    snips = _V.collect_snippets([tmp_path])
    assert len(snips) == 3
    for s in snips:
        assert s.skip is True
        assert s.reason == "tagged .skip"
        assert _V.run_snippet(s, {}) is None      # never executed


def test_plain_python_fence_still_executes(tmp_path):
    # Behaviour-preservation pin for the fences that already pass.
    _write_md(tmp_path, "b.md", ("python", "x = 1"))
    snips = _V.collect_snippets([tmp_path])
    assert len(snips) == 1
    s = snips[0]
    assert s.skip is False
    assert getattr(s, "reason", "") == ""
    ns: dict = {}
    assert _V.run_snippet(s, ns) is None
    assert ns["x"] == 1

    _write_md(tmp_path, "c.md", ("python", 'raise ValueError("boom")'))
    bad = [s for s in _V.collect_snippets([tmp_path])
           if s.file.name == "c.md"]
    assert len(bad) == 1
    f = _V.run_snippet(bad[0], {})
    assert f is not None
    assert "ValueError" in f.excinfo


def test_data_needs_guard_both_regimes(tmp_path, monkeypatch):
    monkeypatch.setattr(_V, "_REPO_ROOT", tmp_path)
    (tmp_path / "present.dat").write_text("", encoding="utf-8")
    _write_md(tmp_path, "d.md",
              ('{.python data-needs="present.dat"}', "ran_a = True"),
              ('{.python data-needs="present.dat missing.dat"}',
               "ran_b = True"))
    snips = _V.collect_snippets([tmp_path])
    assert len(snips) == 2
    a, b = snips
    # REGIME present: executes exactly like a plain fence.
    assert a.skip is False and a.reason == ""
    ns: dict = {}
    assert _V.run_snippet(a, ns) is None
    assert ns["ran_a"] is True
    # REGIME absent: honest skip naming ONLY the missing path.
    assert b.skip is True
    assert b.reason == "data absent: missing.dat"
    assert _V.run_snippet(b, ns) is None
    assert "ran_b" not in ns


def test_data_requires_guard_both_regimes(tmp_path):
    _write_md(tmp_path, "e.md",
              ('{.python data-requires="os"}', "ran_a = True"),
              ('{.python data-requires="os no_such_module_xyz"}',
               "ran_b = True"))
    snips = _V.collect_snippets([tmp_path])
    assert len(snips) == 2
    a, b = snips
    assert a.skip is False
    ns: dict = {}
    assert _V.run_snippet(a, ns) is None
    assert ns["ran_a"] is True
    assert b.skip is True
    assert b.reason == "module absent: no_such_module_xyz"
    assert _V.run_snippet(b, ns) is None
    assert "ran_b" not in ns


# ---------------------------------------------------------------------------
# manual-wide structural pin
# ---------------------------------------------------------------------------

def test_every_python_fence_in_manual_is_exec_or_skip():
    """No python-looking fence in the manual is silently ignored, and
    the exec/skip counts the verifier reports cannot drift from the
    files."""
    n_exec = n_skip = 0
    unclassified: list[str] = []
    for path in sorted(MANUAL.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for m in _V._FENCE_RE.finditer(text):
            tag = m.group("tag").strip()
            c = _V.classify_tag(tag)
            if _PY_LOOKING.match(tag) and c is None:
                line = text.count("\n", 0, m.start()) + 1
                unclassified.append(
                    f"{path.relative_to(MANUAL)}:{line}  ```{tag}")
            if c is not None:
                if c[0]:
                    n_skip += 1
                else:
                    n_exec += 1
    assert not unclassified, (
        "python-looking fences the verifier ignores:\n"
        + "\n".join(unclassified))
    snips = _V.collect_snippets([MANUAL])
    assert sum(1 for s in snips if not s.skip) == n_exec
    assert sum(1 for s in snips if s.skip) == n_skip


# ---------------------------------------------------------------------------
# per-chapter end-to-end replay of the verifier loop
# ---------------------------------------------------------------------------

def _chapters() -> list[str]:
    """Every manual page with >= 1 python-looking fence (found without
    classify_tag, so the listing is identical before and after the
    classifier existed)."""
    out = []
    for path in sorted(MANUAL.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for m in _V._FENCE_RE.finditer(text):
            if _PY_LOOKING.match(m.group("tag").strip()):
                out.append(str(path.relative_to(MANUAL)))
                break
    return out


CHAPTERS = _chapters()


@pytest.mark.slow
@pytest.mark.parametrize("chapter", CHAPTERS, ids=str)
def test_chapter_snippets_pass(chapter):
    """Replay the verifier's own loop for one chapter: fresh namespace,
    snippets run in order, skips honoured, zero failures expected."""
    target = MANUAL / chapter
    snips = [s for s in _V.collect_snippets([MANUAL]) if s.file == target]
    assert snips, f"no snippets collected for {chapter}"
    ns = {"__name__": "__manual_snippet__"}
    fails = [f for s in snips if (f := _V.run_snippet(s, ns)) is not None]
    assert not fails, "\n".join(
        f"{f.snippet.file}:{f.snippet.line}\n{f.excinfo}" for f in fails)
