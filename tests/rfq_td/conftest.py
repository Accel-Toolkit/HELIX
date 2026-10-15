"""Fixtures for the time-domain RFQ solver tests: field maps are cached in
a per-test directory (never the user's cache) and built in-process."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _rfqtd_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HELIX_RFQTD_WORKERS", "1")
    from linac_gen.rfq_td import build
    build._MEMORY.clear()
    yield
    build._MEMORY.clear()
