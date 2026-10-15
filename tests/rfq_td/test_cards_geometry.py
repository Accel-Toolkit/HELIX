"""Electrodes built from the RFQ_CELL cards (decks without a vane file)."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.rfq_coefficients import vane_apertures
from linac_gen.rfq_td import cards_geometry as cg

V, R0 = 80_000.0, 3.4


def _cells(tc=0.0):
    spec = [(0.0, 1.0, 5.4, 3), (0.0, 1.0, 5.4, 2)]
    spec += [(0.05 + 0.01 * k, 1.1, 5.4 + 0.2 * k, (-2, 2)[k % 2])
             for k in range(10)]
    spec += [(0.0, 1.0, 6.0, -3)]
    cells = []
    for i, (a10, m, L, t) in enumerate(spec):
        cells.append(RfqCell(name=f"C{i}", voltage_V=V, r0_mm=R0, A10=a10,
                             modulation=m, length_mm=L, phi_s_deg=-60.0,
                             cell_type=t, Tc_mm=tc))
    return cells


def test_tips_follow_the_cards():
    cells = _cells()
    el = cg.electrodes_from_cards(cells)
    edges = np.concatenate([[0.0], np.cumsum([c.length for c in cells])])
    # regular cells: the two-term tip condition, plane by plane
    for k in range(2, 12):
        c = cells[k]
        zl = np.linspace(0.3, c.length - 0.3, 7)
        want = np.array([vane_apertures(R0, c.A10, c.length, c.cell_type, z)
                         for z in zl])
        got_x = np.interp(edges[k] + zl, el.z, el.R_x)
        got_y = np.interp(edges[k] + zl, el.z, el.R_y)
        np.testing.assert_allclose(got_x, want[:, 0], rtol=3e-3)
        np.testing.assert_allclose(got_y, want[:, 1], rtol=3e-3)
    # +3: the quadrupole ramp sin^3 -> R = r0/sqrt(C1), flared at most to
    # R_MAX_FACTOR r0, round (x = y)
    z3 = np.linspace(2.0, 5.4, 9)
    c1 = np.sin(0.5 * np.pi * z3 / 5.4) ** 3
    want = R0 / np.sqrt(np.maximum(c1, cg.R_MAX_FACTOR ** -2))
    np.testing.assert_allclose(np.interp(z3, el.z, el.R_x), want, rtol=2e-3)
    np.testing.assert_array_equal(el.R_x[el.z < 5.4], el.R_y[el.z < 5.4])
    assert el.R_x.max() <= cg.R_MAX_FACTOR * R0 * (1 + 1e-12)
    # vanes start / end where the flares reach R_MAX_FACTOR r0
    assert 0.0 < el.z[0] < 2.0 and edges[-1] - 6.0 < el.z[-1] < edges[-1]
    np.testing.assert_allclose(el.R_x[[0, -1]], cg.R_MAX_FACTOR * R0,
                               rtol=1e-6)
    # voltages and end plates
    assert (el.V_x, el.V_y) == (-V / 2, V / 2)
    from linac_gen.rfq_td.geometry import end_plate_hole
    assert [p.hole_radius for p in el.plates] == [end_plate_hole(R0)] * 2
    assert end_plate_hole(5.576) == 6.0             # PXIE's, exactly
    # at unit voltage (the solver scales the field by the cards' voltage)
    el1 = cg.electrodes_from_cards(cells, voltage=1.0)
    assert (el1.V_x, el1.V_y) == (-0.5, 0.5)
    np.testing.assert_array_equal(el1.R_x, el.R_x)
    assert el.plates[1].z_face == pytest.approx(edges[-1])
    np.testing.assert_allclose(el.rho_x, 0.75 * R0)


def test_gaps_and_tip_radius_from_the_cards():
    cells = _cells(tc=2.9)
    el = cg.electrodes_from_cards(cells, gaps=(1.5, 2.0))
    L = sum(c.length for c in cells)
    assert el.z[0] == pytest.approx(1.5) and el.z[-1] == pytest.approx(L - 2.0)
    np.testing.assert_allclose(el.rho_y, 2.9)


def test_cards_that_do_not_fit_are_refused():
    with pytest.raises(ValueError, match="do not fit"):
        cg.electrodes_from_cards(_cells(), gaps=(40.0, 40.0))
    cells = _cells()
    cells[4].voltage_V = 1.1 * V
    with pytest.raises(ValueError, match="voltages"):
        cg.electrodes_from_cards(cells)


def test_pxie_cards_reproduce_the_vane_file():
    """Where the cards describe the vanes (all but cell 201, which the deck
    gives as unmodulated while the real vanes demodulate over it), the
    card-built tips match the PXIE vane table."""
    import warnings

    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_vane import parse_vane_file
    from linac_gen.rfq_td.geometry import RfqElectrodes
    from tests.rfq import ref_loaders as rl
    vane = rl.reference_path("pxie-rfq.vane")
    if vane is None:
        pytest.skip("PXIE vane file not present on this machine")
    deck = rl.REPO_ROOT / "examples" / "lebt_plus_rfq" / \
        "lebt2sol_plus_rfq_phifix.dat"
    from tests.dataguard import require
    require(str(deck))                  # not in the public repository
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
    cells = [e for e in lat.elements if isinstance(e, RfqCell)]
    el_c = cg.electrodes_from_cards(cells, gaps=lat.rfq_gaps_mm)
    el_v = RfqElectrodes.from_vane(parse_vane_file(str(vane)))
    edges = np.concatenate([[0.0], np.cumsum([c.length for c in cells])])
    z = el_v.z
    k = np.searchsorted(edges, z, side="right") - 1
    regular = (k >= 1) & (k <= 199)
    for a, b in ((el_c.R_x, el_v.R_x), (el_c.R_y, el_v.R_y)):
        dev = np.abs(np.interp(z, el_c.z, a) / b - 1.0)
        assert np.median(dev[regular]) < 1e-3
        assert np.percentile(dev[regular], 95) < 5e-3
        assert np.median(dev[k == 0]) < 0.05           # entrance flare
        assert np.median(dev[k == len(cells) - 1]) < 0.02   # exit flare
    assert el_c.z[0] == pytest.approx(el_v.z[0], abs=0.01)
    assert el_c.z[-1] == pytest.approx(el_v.z[-1], abs=0.02)
