"""Shared fixtures for the RFQ benchmark suite.

Every ground-truth file lives outside git (``Tracewin_code/`` is in-repo
but never committed; the PXIE project folder is external) — fixtures
skip when a reference is absent so the suite stays green on any machine.
"""
from __future__ import annotations

import pytest

from tests.rfq import ref_loaders as rl


def _require(name: str):
    p = rl.reference_path(name)
    if p is None:
        pytest.skip(f"reference file {name!r} not present on this machine")
    return p


@pytest.fixture(scope="session")
def env_nosc():
    return rl.load_env_chart(_require("LEBT+RFQ_ENV+NOSC.txt"))


@pytest.fixture(scope="session")
def env_sc():
    # 2026-07-30 audit: the in-repo "SC" export is a BYTE-IDENTICAL copy
    # of the no-SC file (md5 2c1f81a5a514dfb1b4d0917fa5c096de for both)
    # — it is NOT space-charge ground truth.  Skip until a genuine
    # SC export replaces it, so no benchmark ever pins against the fake.
    pytest.skip("LEBT+RFQ_ENV+SC.txt is a mislabeled copy of the "
                "no-SC export — no genuine SC ground truth in repo")


@pytest.fixture(scope="session")
def transfer_ref():
    """(elem_no, s_m, cumulative 6×6 stack) from Transfer_matrix1.dat."""
    return rl.load_transfer_matrices(_require("Transfer_matrix1.dat"))


@pytest.fixture(scope="session")
def rfq_input_dst():
    """TraceWin's own RFQ input distribution (100k particles, 30 keV)."""
    from linac_gen.io.tracewin_dst import load_dst
    return load_dst(str(_require("part_rfq.dst")))


@pytest.fixture(scope="session")
def pxie_deck():
    if not rl.PXIE_DECK.is_file():
        pytest.skip("lebt_plus_rfq example deck not present")
    from linac_gen.io.tracewin_parser import parse_tracewin
    lattice, _ = parse_tracewin(str(rl.PXIE_DECK))
    return lattice
