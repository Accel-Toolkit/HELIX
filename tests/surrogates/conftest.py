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


import pytest

from linac_gen.surrogates import registry as _registry


@pytest.fixture(autouse=True)
def _clean_surrogate_registry():
    """The surrogate registry is PROCESS-GLOBAL and, since the I = 0
    full-matrix seam (``envelope._full_matrix_at``), a registration
    leaked by one test changes sigma for every later envelope run whose
    lattice reuses the element name.  Clear before AND after every test
    in this suite so no test depends on, or pollutes, registry state.
    """
    _registry.clear()
    try:
        yield
    finally:
        _registry.clear()
