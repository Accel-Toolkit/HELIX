"""Sanity of the RFQ reference loaders + the audited facts they encode.

These tests pin the 2026-07-30 axis audit (see ``ref_loaders`` module
docstring): which reference file covers which segment, and the exact
endpoint values later benchmark phases compare against.  If a reference
file is replaced by a re-export, these catch silent layout drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl


def test_env_chart_covers_full_lebt_rfq_line(env_nosc):
    s = env_nosc["s_m"]
    assert len(s) > 20_000                       # substep resolution
    assert s[-1] == pytest.approx(6.402673, abs=1e-4)
    # TW's own final energy on this deck — the anchor for every
    # longitudinal comparison (NOT the 2.1 MeV design-paper value).
    assert env_nosc["ref_W_MeV"][-1] == pytest.approx(1.955717, abs=2e-4)
    # rms sizes stay physical (mm scale, non-negative)
    assert np.all(env_nosc["sigma_x_mm"] >= 0)
    assert np.nanmax(env_nosc["sigma_x_mm"]) < 50.0


def test_transmission_chart_is_lebt_only():
    p = rl.reference_path("Chart_Transmission(%).txt")
    if p is None:
        pytest.skip("PXIE project charts not present")
    s, t = rl.load_chart_xy(p)
    assert s[-1] == pytest.approx(1.9561, abs=1e-3)   # ends at LEBT exit
    assert t[-1] == pytest.approx(77.191, abs=0.01)   # LEBT scraping only
    # every loss lies inside the LEBT scraper region (0.169–0.312 m)
    drops = s[1:][np.diff(t) < 0]
    assert drops.size and drops.min() > 0.15 and drops.max() < 0.35


def test_partran_out_is_lebt_only():
    p = rl.reference_path("partran1.out")
    if p is None:
        pytest.skip("PXIE project partran1.out not present")
    from linac_gen.io.tracewin_outputs import read_partran_out
    d = read_partran_out(p)
    assert d["s_m"][-1] == pytest.approx(1.9561, abs=1e-3)
    assert int(d["n_alive"][-1]) == 77_287


def test_transfer_matrix_blocks(transfer_ref):
    nums, s_m, cum = transfer_ref
    assert cum.shape == (242, 6, 6)
    assert nums[0] == 1 and nums[-1] == 242
    assert s_m[-1] == pytest.approx(6.402673, abs=1e-3)
    # first element is a 1e-11 m stub drift → cumulative ≈ identity
    assert np.allclose(cum[0], np.eye(6), atol=1e-6)
    per = rl.per_element_matrices(cum)
    assert per.shape == cum.shape
    # a genuine LEBT drift block: x/x' 2×2 is [[1, L], [0, 1]]
    # (element 4 sits at 0.01023 m after three stubs)
    d = per[3]
    assert d[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert d[1, 0] == pytest.approx(0.0, abs=1e-9)
    assert d[0, 1] == pytest.approx(0.01023, rel=1e-3)


def test_rfq_input_dst_reads(rfq_input_dst):
    particles, meta = rfq_input_dst
    assert meta["n_particles"] == 100_000
    assert meta["frequency_MHz"] == pytest.approx(162.5)
    assert meta["w_kin_ref"] == pytest.approx(0.030, abs=1e-4)
    assert len(particles) == 100_000


def test_pxie_deck_has_the_rfq(pxie_deck):
    from linac_gen.elements.rfq_cell import RfqCell
    cells = [e for e in pxie_deck.elements if isinstance(e, RfqCell)]
    assert len(cells) == 203
