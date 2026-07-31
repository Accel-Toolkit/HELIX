"""Phase-3 refinements: the −3 exit cell, live modulation checks.

Documented negative result (2026-07-30): per-slice coefficients
inverted from the ``pxie-rfq.vane`` tip table do NOT improve on the
card model — the tips are two-term to within the inversion's own noise
(A01 recovered as 1.000 ± 0.0013, A10 within ±3 % of the card), and
the near-singular mid-cell inversion injects more error than it
removes (shaper-region mean 0.054 → 0.197).  The residual difference
to TW's vane-based matrices lives in the Toutatis field SOLUTION
(vane shape between tips), not in the tip positions — unreachable
from card- or tip-level data.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl


def test_exit_cell_minus3_matches_tracewin(pxie_deck, transfer_ref,
                                           env_nosc):
    """The −3 exit cell: image275's middle sign is '+' (cos³ ramp).
    The '−' transcription gave a 22 % error; cos³ matches to 2e-4."""
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.rfq_cell import RfqCell
    nums, s_m, cum = transfer_ref
    per = rl.per_element_matrices(cum)
    cells = [e for e in pxie_deck.elements if isinstance(e, RfqCell)]
    blk = np.diff(np.concatenate([[0.0], s_m]))
    pairs = []
    bi = int(np.searchsorted(s_m, 1.9561 + 1e-6))
    for ci in range(len(cells)):
        while bi < len(s_m) and blk[bi] < 1e-9:
            bi += 1
        pairs.append((ci, bi))
        bi += 1
    ci = len(cells) - 1                      # RFQ_203, type -3
    assert cells[ci].cell_type == -3
    bi = dict(pairs)[ci]
    s_ent = s_m[bi] - blk[bi]
    gam = 1.0 + np.interp(s_ent, env_nosc["s_m"], env_nosc["gam1"])
    saved = cells[ci].field_model
    cells[ci].field_model = "tw2term"
    try:
        ref = ReferenceParticle(species=H_MINUS,
                                w_kin=(gam - 1) * H_MINUS.mass,
                                frequency=162.5)
        M = cells[ci].fitted_matrix(ref)
    finally:
        cells[ci].field_model = saved
    assert np.abs(M[:4, :4] - per[bi][:4, :4]).max() < 5e-3


def test_modulation_warnings():
    from linac_gen.elements.rfq_cell import RfqCell
    # internally consistent card (PXIE cell 150 values): silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RfqCell("ok", 60000.0, 5.576, 0.5998, 2.0725, 36.47, -33.0, -2)
    # inconsistent A10 vs (R0, m): warns
    with pytest.warns(UserWarning, match="internally inconsistent"):
        RfqCell("bad", 60000.0, 5.576, 0.30, 1.05, 7.4, -30.0, 2)
    # m beyond the documented two-term validity: warns
    with pytest.warns(UserWarning, match="> 3.2"):
        RfqCell("hot", 60000.0, 5.576, 0.0, 3.5, 60.0, -30.0, 2)


def test_pxie_deck_parses_without_modulation_warnings():
    """The real deck's (R0, m, A10) triplets are consistent — parsing
    must stay silent (guards against a future overeager threshold)."""
    if not rl.PXIE_DECK.is_file():
        pytest.skip("deck not present")
    from linac_gen.io.tracewin_parser import parse_tracewin
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_tracewin(str(rl.PXIE_DECK))
    bad = [x for x in w if "inconsistent" in str(x.message)
           or "> 3.2" in str(x.message)]
    assert not bad
