"""External anchor: HELIX's ``Edge`` against TraceWin's own exported transfer
matrices for the same deck.

``TraceWIn_Tools/Transfer_matrix1.dat`` is TraceWin's cumulative-matrix
export for ``TraceWIn_Tools/BTL_lattice.dat`` (H⁻, 800 MeV).  That deck
carries NEGATIVE pole-face angles on the two negative vertical bends BVDD
and ORB1 (the signed θ/2 convention inherited by every converted BTL deck
until 2026-09-06).  TraceWin applies the EDGE card literally — its manual:
"an edge focalizes if β < 0, whatever the curvature radius sign, the
bending angle sign and the particle charge state" — and so does HELIX, so
every edge matrix must agree, including the flipped ones.  This pins the
card-reading convention (the element), not the physics of that deck (whose
BVDD/ORB1 edges are edge-focusing, i.e. wrong for a rectangular magnet).

The export was taken after a TraceWin re-match of the SS05 trombone: its
eight quadrupoles (s = 98-140 m) differ from the deck cards by up to 1e-2,
so only the EDGE entries are a pure export of the deck — do not extend this
anchor to quadrupoles or cumulative matrices without re-exporting.
"""
from pathlib import Path

import numpy as np
import pytest

from tests.dataguard import needs

REPO = Path(__file__).resolve().parents[2]
_DECK = "TraceWIn_Tools/BTL_lattice.dat"
_TW = "TraceWIn_Tools/Transfer_matrix1.dat"


def _is_edge_like(M: np.ndarray) -> bool:
    """Identity except the two thin-lens entries (per-element matrices are
    recovered from 7-digit cumulative ones, so allow inversion noise)."""
    off = M - np.eye(6)
    off[1, 0] = 0.0
    off[3, 2] = 0.0
    return np.abs(off).max() < 1e-5 and (abs(M[1, 0]) > 0 or abs(M[3, 2]) > 0)


@needs(_DECK, _TW)
def test_edge_matrices_match_tracewin_export():
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.edge import Edge
    from linac_gen.io.tracewin_parser import parse_tracewin
    from tests.rfq.ref_loaders import load_transfer_matrices, per_element_matrices

    _nums, s_m, cum = load_transfer_matrices(REPO / _TW)
    per = per_element_matrices(cum)
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    lat, _ = parse_tracewin(str(REPO / _DECK))

    s = 0.0
    checked = 0
    flipped = []
    for el in lat.elements:
        s += float(getattr(el, "length", 0.0) or 0.0)
        if not isinstance(el, Edge):
            continue
        M_h = el.transfer_matrix(ref)
        # TraceWin lists each element by its END position printed with
        # ~5 significant digits (0.05 m resolution at 290 m); a zero-length
        # edge shares that s with its neighbours, so pick the edge-shaped
        # matrix among the entries near this s (never match by index: the
        # export has fewer entries than the deck has cards).
        s_here = s * 1e-3
        tol = max(2e-4, 1.2e-3 * s_here)
        cand = [per[i] for i in np.flatnonzero(np.abs(s_m - s_here) < tol)
                if _is_edge_like(per[i])]
        cand = [C for C in cand if np.sign(C[1, 0]) == np.sign(M_h[1, 0])
                and np.sign(C[3, 2]) == np.sign(M_h[3, 2])]
        assert cand, f"no TraceWin edge entry at s = {s * 1e-3:.4f} m"
        M_tw = min(cand, key=lambda C: np.abs(C[:4, :4] - M_h[:4, :4]).max())
        # measured 2026-09-06 over all 72 edges: worst relative error 1.0e-4
        # (main bends, |M10| = 2.7e-3), 2.2e-5 on the four small dogleg/orbump
        # edges; off-pattern inversion noise <= 1.8e-6
        np.testing.assert_allclose(M_h[:4, :4], M_tw[:4, :4], rtol=3e-4, atol=2e-6)
        checked += 1
        if el.pole_rotation < 0:
            flipped.append((el.name, M_h[1, 0], M_h[3, 2]))

    assert checked >= 60                      # every edge of the 36 bends
    # the four negative-β edges (BVDD, ORB1; hv = 1): the bend plane is y and
    # TraceWin, like HELIX, makes them FOCUSING there (M[3,2] < 0, M[1,0] > 0)
    assert len(flipped) == 4
    assert all(m10 > 0 and m32 < 0 for _n, m10, m32 in flipped)
