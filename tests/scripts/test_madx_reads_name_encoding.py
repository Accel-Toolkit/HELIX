"""Tests that handle text files HELIX writes name the encoding.

The MAD-X writer writes UTF-8 with non-ASCII text in its header ("Bρ", "T·m"),
and some example files carry non-ASCII too.  A test that reads such a file
with ``read_text()`` or ``open()`` and no encoding decodes it with the
platform's preferred encoding, which is cp1252 on the Windows CI runner, where
byte 0x81 of "ρ" has no mapping; one that copies such text with a
``write_text()`` naming no encoding writes cp1252 there, which a UTF-8 read
then refuses.  Either test fails on Windows and passes everywhere else.

Rules:

* in a test module that has HELIX write MAD-X (imports
  ``linac_gen.io.madx_writer``, drives the GUI's ``_export_madx`` or runs the
  CLI's ``export``), every text read (``Path.read_text``, ``open`` or
  ``Path.open`` in a read mode) names its encoding: UTF-8 for MAD-X and project
  files, latin-1 for TraceWin decks (what the TraceWin writer and parser use);
* anywhere under ``tests/``, a ``write_text`` that names no encoding does not
  write text that was decoded with a named encoding, or a non-ASCII literal.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WRITER = "linac_gen.io.madx_writer"


def _writes_madx(tree: ast.Module) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == WRITER:
            return True
        if isinstance(n, ast.Import) and any(a.name == WRITER for a in n.names):
            return True
        if isinstance(n, ast.Attribute) and n.attr == "_export_madx":
            return True
        if (isinstance(n, ast.Call) and n.args and isinstance(n.args[0], (ast.List, ast.Tuple))
                and n.args[0].elts and isinstance(n.args[0].elts[0], ast.Constant)
                and n.args[0].elts[0].value == "export"):
            return True                                   # main(["export", ...]): the CLI
    return False


def _name(call: ast.Call) -> str | None:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)


def _names_encoding(call: ast.Call) -> bool:
    """``encoding=`` given, or given by position (``read_text(enc)``, ``write_text(data, enc)``)."""
    if any(k.arg == "encoding" for k in call.keywords):
        return True
    name = _name(call)
    return (name == "read_text" and len(call.args) >= 1) or (name == "write_text" and len(call.args) >= 2)


def _mode(call: ast.Call) -> object:
    """The mode of an ``open`` call: argument 2 of ``open(path, mode)``, argument 1 of
    ``path.open(mode)``, or the ``mode=`` keyword; "r" when none is given."""
    f = call.func
    builtin = isinstance(f, ast.Name) or (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                                          and f.value.id in ("io", "builtins", "codecs"))
    pos = 1 if builtin else 0
    mode = call.args[pos] if len(call.args) > pos else None
    for k in call.keywords:
        if k.arg == "mode":
            mode = k.value
    if mode is None:
        return "r"
    return mode.value if isinstance(mode, ast.Constant) else None


def _reads_without_encoding(tree: ast.Module) -> list[int]:
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or _names_encoding(n):
            continue
        name = _name(n)
        if name == "read_text":
            out.append(n.lineno)
        elif name == "open":
            mode = _mode(n)
            if isinstance(mode, str) and "b" not in mode and not set(mode) & set("wax+"):
                out.append(n.lineno)
    return sorted(out)


def _decoded_or_non_ascii(node: ast.AST) -> bool:
    """Whether ``node`` holds a read that named an encoding, or a non-ASCII literal."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and _name(n) in ("read_text", "open") and _names_encoding(n):
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and not n.value.isascii():
            return True
    return False


def _lossy_writes(tree: ast.Module) -> list[int]:
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and _name(n) == "write_text" and not _names_encoding(n)
                and n.args and _decoded_or_non_ascii(n.args[0])):
            out.append(n.lineno)
    return sorted(out)


def _canary(src: str) -> ast.Module:
    return ast.parse(src)


def test_the_rules_flag_what_they_are_for():
    """Not vacuous: each rule flags a known violation and passes its fix."""
    bad = _canary('from linac_gen.io.madx_writer import write_madx\n'
                  'def t(p):\n    return p.read_text()\n')
    good = _canary('from linac_gen.io.madx_writer import write_madx\n'
                   'def t(p):\n    return p.read_text(encoding="utf-8"), p.read_text("utf-8")\n')
    cli = _canary('def t(p):\n    main(["export", "a.dat", "b.madx"])\n    return open(p).read()\n')
    assert _writes_madx(bad) and _reads_without_encoding(bad) == [3]
    assert _reads_without_encoding(good) == []
    assert _writes_madx(cli) and _reads_without_encoding(cli) == [3]
    copy = _canary('def t(a, b):\n    b.write_text(a.read_text(encoding="utf-8"))\n'
                   '    b.write_text("\\u03c1")\n    b.write_text("ascii")\n'
                   '    b.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")\n')
    assert _lossy_writes(copy) == [2, 3]
    tree = ast.parse((REPO / "tests" / "io" / "test_madx_writer.py").read_text(encoding="utf-8"))
    assert _writes_madx(tree)


def test_modules_that_write_madx_name_the_encoding_of_their_text_reads():
    bad = []
    for p in sorted((REPO / "tests").rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        if _writes_madx(tree):
            bad += [f"{p.relative_to(REPO).as_posix()}:{ln}" for ln in _reads_without_encoding(tree)]
    assert not bad, ("text reads with no encoding in tests that write MAD-X (they fail on a cp1252 "
                     "platform):\n" + "\n".join(bad))


def test_text_copies_keep_the_encoding_they_were_read_with():
    bad = []
    for p in sorted((REPO / "tests").rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        bad += [f"{p.relative_to(REPO).as_posix()}:{ln}" for ln in _lossy_writes(tree)]
    assert not bad, ("write_text with no encoding of decoded or non-ASCII text (it writes cp1252 on "
                     "Windows):\n" + "\n".join(bad))
