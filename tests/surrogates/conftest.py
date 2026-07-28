"""Shared setup for the surrogates test suite.

Sandboxes GUI settings exactly like tests/gui/conftest.py: the
SurrogatesTab / ConvergenceTab widgets persist UI state through the
make_settings factory, and GUI-level tests here must never write into
the developer's real settings store.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HELIX_QSETTINGS_DIR",
                      tempfile.mkdtemp(prefix="helix-test-qsettings-"))
