"""Checks every test of the suite shares."""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_madx_checkpoint_left_behind():
    """MAD-X TRACK writes checkpoint_restart.dat into its working directory: a test that runs it gives MAD-X a
    directory of its own (``cpymad.Madx(cwd=tmp_path)``), so the suite leaves nothing where it runs."""
    where = Path.cwd() / "checkpoint_restart.dat"
    before = where.exists()
    yield
    if not before and where.exists():
        pytest.fail(f"this test left {where} (MAD-X TRACK): give cpymad.Madx a cwd of its own", pytrace=False)
