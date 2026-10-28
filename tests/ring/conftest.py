"""Ring-mode tests: the long benchmark campaigns (marker ``bench``) run only
when asked for, so a bare ``pytest tests`` stays fast.  They live under
``tests/ring/`` (this hook sees every collected test and skips those carrying
the ``bench`` marker itself — not tests that merely have "bench" in a name or
parameter id)."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("HELIX_RUN_BENCH") == "1":
        return
    skip = pytest.mark.skip(reason="benchmark campaign: set HELIX_RUN_BENCH=1 to run")
    for item in items:
        if item.get_closest_marker("bench") is not None:
            item.add_marker(skip)
