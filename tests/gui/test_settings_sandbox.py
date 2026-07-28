"""Guard: GUI tests must never touch the developer's real QSettings.

conftest.py sets HELIX_QSETTINGS_DIR, which the make_settings factory
honors by returning throwaway INI files.  If this ever breaks, tests
exercising the project save/load paths silently pollute the real store
again (recent-projects menu full of pytest-tmp 'broken.lgproj' entries).
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402

from linac_gen_gui.interphase.app_settings import make_settings  # noqa: E402


@pytest.mark.parametrize("org,app", [
    ("Linac_Gen", "Interphase"),      # app.py's store
    ("HELIX", "linac_gen_gui"),       # convergence/surrogates panel state
    ("linac_gen", "linac_gen_gui"),   # results-tab popups
    ("Helix", "HELIX"),               # beam tab
])
def test_qsettings_are_sandboxed(org, app):
    sandbox = os.environ.get("HELIX_QSETTINGS_DIR")
    assert sandbox, "conftest did not set HELIX_QSETTINGS_DIR"

    s = make_settings(org, app)
    assert s.format() == QSettings.Format.IniFormat
    assert s.fileName().startswith(sandbox), (
        f"make_settings({org!r}, {app!r}) resolves to {s.fileName()!r} — "
        "the settings sandbox is not in effect and tests will write "
        "into the real user settings store.")

    # And it actually round-trips inside the sandbox.
    s.setValue("sandbox_probe", "ok")
    s.sync()
    assert os.path.exists(s.fileName())


def test_no_direct_qsettings_constructions_in_gui():
    """Every GUI QSettings must go through make_settings — a direct
    QSettings(org, app) bypasses the sandbox (and on macOS cannot be
    redirected at all).  AST-based so comments/docstrings don't count."""
    import ast
    import pathlib

    gui_root = (pathlib.Path(__file__).resolve().parents[2]
                / "gui" / "linac_gen_gui")
    offenders = []
    for py in gui_root.rglob("*.py"):
        if py.name == "app_settings.py":
            continue   # the factory itself
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute)
                    else None)
            if name == "QSettings":
                offenders.append(
                    f"{py.relative_to(gui_root)}:{node.lineno}")
    assert not offenders, (
        "Direct QSettings construction bypasses the test sandbox:\n"
        + "\n".join(offenders))
