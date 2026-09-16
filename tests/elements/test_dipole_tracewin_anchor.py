"""External anchor: HELIX's bend path-length row against TraceWin's own export.

``TraceWIn_Tools/Transfer_matrix1.dat`` is TraceWin's cumulative-matrix export
for ``TraceWIn_Tools/BTL_lattice.dat`` (H-, 800 MeV).  Its per-element rows
carry the path-length coupling HELIX gained on 2026-09-07, so the 36 BTL
bends - horizontal, and the four vertical ones of both angle signs - pin the
new row against a second code, independently of MAD-X.

The comparison is on the three coefficients, NOT on ``matrix_to_tracewin``
output: the export's transverse coordinates are metres and radians, so the
transverse 4x4 is invariant under the mm/mrad rescaling but the off-diagonal
blocks are a factor 1000 away.  Extracting the coefficients keeps the
convention explicit:

    dL/dx  = -R51        dL/dx' = -R52 [m/rad]        dL/ddelta = L/gamma^2 - R56
"""
from pathlib import Path

import numpy as np
import pytest

from tests.dataguard import needs

REPO = Path(__file__).resolve().parents[2]
_DECK = "TraceWIn_Tools/BTL_lattice.dat"
_TW = "TraceWIn_Tools/Transfer_matrix1.dat"


@needs(_DECK, _TW)
def test_bend_path_length_row_matches_tracewin_export():
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.dipole import Dipole
    from linac_gen.io.tracewin_parser import parse_tracewin
    from tests.rfq.ref_loaders import load_transfer_matrices, per_element_matrices

    _nums, s_m, cum = load_transfer_matrices(REPO / _TW)
    per = per_element_matrices(cum)
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    lat, _ = parse_tracewin(str(REPO / _DECK))

    s = 0.0
    checked = 0
    vertical = 0
    worst = 0.0
    by_geometry: dict = {}
    for el in lat.elements:
        s += float(getattr(el, "length", 0.0) or 0.0)
        if not isinstance(el, Dipole):
            continue
        M = el.transfer_matrix(ref.copy())
        # TraceWin prints each element's END position with ~5 significant
        # digits, and a zero-length edge shares it; pick the entry whose own
        # (R12, R34) match this body's, which identifies it whatever its
        # length - the two 0.36 m orbit-bump bends are as thin as a drift.
        s_here = s * 1e-3
        idx = np.flatnonzero(np.abs(s_m - s_here) < max(2e-4, 1.2e-3 * s_here))
        assert idx.size, f"no TraceWin entry at s = {s_here:.4f} m for {el.name}"
        cand = min(idx, key=lambda i: (abs(per[i][0, 1] - M[0, 1])
                                       + abs(per[i][2, 3] - M[2, 3])))
        tw = per[cand]
        assert tw[0, 1] == pytest.approx(M[0, 1], abs=1e-3), f"{el.name}: wrong entry matched"

        b, g, m, wl = ref.beta, ref.gamma, ref.species.mass, ref.wavelength
        k = 360.0 / (b * wl)                       # deg per mm of path length
        L = el.length * 1e-3
        row = 2 if el.hv == 1 else 0               # the bend plane
        c1_h = M[4, row] / k                       # dimensionless
        c2_h = M[4, row + 1] / k                   # m/rad == mm/mrad
        c3_h = M[4, 5] * (b ** 3 * g * m * wl) / 360000.0 + L / (g * g)
        c1_t, c2_t, c3_t = -tw[4, row], -tw[4, row + 1], L / (g * g) - tw[4, 5]

        worst = max(worst, abs(c1_h - c1_t), abs(c2_h - c2_t), abs(c3_h - c3_t))
        by_geometry.setdefault((round(abs(el.angle), 6), round(L, 6), el.hv),
                               []).append((el.name, c3_h, c3_t))
        # An ABSOLUTE bound, because the export sets the floor, not HELIX.
        # The export carries 7 significant digits and these per-element
        # matrices come from inverting cumulative ones, so the error grows
        # with s.  Measured 2026-09-07: worst c1 2.9e-7, c2 1.7e-6, c3
        # 7.7e-6 (the 6.564 deg, 2.45 m horizontal bends - 0.14 % of c3).
        # A relative bound would be wrong here: the four 1.4 deg vertical
        # bends have c3 ~ 1e-4, where the same absolute export noise is
        # 1.4-5.5 %.  That noise is demonstrably the export's own -- see
        # the mirror-pair check after the loop.
        assert c1_h == pytest.approx(c1_t, abs=1e-5)
        assert c2_h == pytest.approx(c2_t, abs=1e-5)
        assert c3_h == pytest.approx(c3_t, abs=1e-5)
        # the sign follows the bend direction, and the compaction never does
        assert np.sign(c1_h) == np.sign(el.angle)
        assert c3_h > 0.0
        checked += 1
        vertical += el.hv == 1

    assert checked == 36 and vertical == 4
    assert worst < 1e-5

    # Momentum compaction goes as h^2, so two bends of identical geometry and
    # opposite sign must have exactly the same c3.  HELIX does, bit for bit.
    # TraceWin's export does not: BEND_033/034 (+-1.494 deg) differ by 1.3 %
    # in its own numbers and BEND_035/036 (+-1.409 deg) by 3.5 %, which is
    # what the 1e-5 absolute bound above is actually accommodating.
    pairs = [v for v in by_geometry.values() if len(v) > 1]
    assert pairs, "no mirror-image bend pairs found in the deck"
    for group in pairs:
        helix = {c3_h for _n, c3_h, _t in group}
        assert len(helix) == 1, (
            f"HELIX c3 differs across identical geometries: "
            f"{[(n, c) for n, c, _ in group]}")
