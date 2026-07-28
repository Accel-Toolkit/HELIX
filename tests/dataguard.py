"""Guards for tests that need reference data NOT distributed with the
public repository (third-party field maps, PIP-II lattice decks).

On the full development checkout everything exists and every guard is
inert.  On a public clone the guarded tests SKIP with an honest message
instead of failing — see examples/FIELD_MAPS.md.

Usage:
    from tests.dataguard import needs, require

    require("examples/pip2_misalignment_study")     # module-level skip

    @needs("examples/pipii/btl/btl.dat")            # per-test skip
    def test_btl_survey(): ...
"""
from __future__ import annotations

import os

import pytest

_REASON = ("undistributed reference data absent: {} — this test runs on "
           "the full development checkout; see examples/FIELD_MAPS.md")


def _missing(paths):
    return [p for p in paths if not os.path.exists(p)]


def require(*paths: str) -> None:
    """Module-level: skip the whole module when any path is absent."""
    miss = _missing(paths)
    if miss:
        pytest.skip(_REASON.format(", ".join(miss)),
                    allow_module_level=True)


def needs(*paths: str):
    """Decorator: skip one test when any path is absent."""
    miss = _missing(paths)
    return pytest.mark.skipif(
        bool(miss), reason=_REASON.format(", ".join(miss) or "?"))
