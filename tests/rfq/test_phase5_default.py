"""Phase-5: tw2term is the production default everywhere.

The default flipped 2026-07-30 after the full benchmark ladder went
green (per-cell matrices, momentum invariant, exact ramp, LEBT
scraping, S-curve, measured emittances).  Legacy "2term" stays
available as an explicit opt-in fallback and bit-identical.
"""
from __future__ import annotations

import pytest

from tests.rfq import ref_loaders as rl


def test_constructor_default_is_tw2term():
    from linac_gen.elements.rfq_cell import RfqCell
    cell = RfqCell("c", 60000.0, 5.576, 0.1, 1.3, 10.0, -30.0, 2)
    assert cell.field_model == "tw2term"


def test_parser_built_cells_default_to_tw2term():
    if not rl.PXIE_DECK.is_file():
        pytest.skip("deck not present")
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(str(rl.PXIE_DECK))
    cells = [e for e in lat.elements if isinstance(e, RfqCell)]
    assert cells and all(c.field_model == "tw2term" for c in cells)


def test_gui_inspector_does_not_expose_field_model():
    """field_model is deliberately ABSENT from the RfqCell inspector:
    neither the TW RFQ_CELL card nor .lgproj can persist it, so a GUI
    edit would silently revert on save/reload (adversarial finding
    2026-07-30).  If you re-add the dropdown, solve persistence first."""
    import sys
    gui_dir = str(rl.REPO_ROOT / "gui")
    if gui_dir not in sys.path:
        sys.path.insert(0, gui_dir)
    ei = pytest.importorskip(
        "linac_gen_gui.interphase.panels.element_inspector")
    assert ("RfqCell", "field_model") not in ei._STR_CHOICES
    for name in dir(ei):
        obj = getattr(ei, name)
        if isinstance(obj, dict) and "RfqCell" in obj:
            rows = obj["RfqCell"]
            if isinstance(rows, list) and rows and isinstance(rows[0],
                                                              tuple):
                attrs = [r[0] for r in rows]
                if "voltage_V" in attrs:
                    assert "field_model" not in attrs
                    return
    pytest.fail("RfqCell inspector row table not found")


def test_gui_add_element_template_is_consistent():
    """The Add-Element RfqCell template triplet must not trip the
    Phase-3 modulation cross-check (it fired at 435 % deviation with
    the old toy A10 = 0.05)."""
    import sys
    import warnings
    gui_dir = str(rl.REPO_ROOT / "gui")
    if gui_dir not in sys.path:
        sys.path.insert(0, gui_dir)
    ef = pytest.importorskip("linac_gen_gui.interphase.element_factory")
    factory = ef.ELEMENT_FACTORY if hasattr(ef, "ELEMENT_FACTORY") else \
        next(getattr(ef, n) for n in dir(ef)
             if isinstance(getattr(ef, n), dict)
             and "RfqCell" in getattr(ef, n))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factory["RfqCell"]()


def test_legacy_2term_matrix_pinned_bit_stable():
    """'Kept bit-identical as fallback' is a claim — this pins it.
    Baseline generated 2026-07-30 from the PXIE cell-150 card (an
    internally consistent triplet, so construction stays silent)."""
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.rfq_cell import RfqCell
    cell = RfqCell("c", 60000.0, 5.576, 0.5998, 2.0725, 36.47, -33.0,
                   -2, field_model="2term")
    ref = ReferenceParticle(species=H_MINUS, w_kin=0.6, frequency=162.5)
    M = cell.fitted_matrix(ref)
    assert M[0, 0] == pytest.approx(0.8993220287232454, rel=1e-12)
    assert M[0, 1] == pytest.approx(0.03571472141189027, rel=1e-12)
    assert M[1, 0] == pytest.approx(-2.7257310755836923, rel=1e-12)
    assert M[1, 1] == pytest.approx(1.0037019500937652, rel=1e-12)
    assert M[2, 2] == pytest.approx(1.1616121466222533, rel=1e-12)
    assert M[2, 3] == pytest.approx(0.03722236581461788, rel=1e-12)
    assert M[3, 2] == pytest.approx(2.2498316467183193, rel=1e-12)
    assert M[3, 3] == pytest.approx(0.9329654994799831, rel=1e-12)
    assert M[4, 4] == pytest.approx(0.9719050935245518, rel=1e-12)
    assert M[4, 5] == pytest.approx(-164.709399866432, rel=1e-12)
    assert M[5, 4] == pytest.approx(0.00018705581204073404, rel=1e-12)
    assert M[5, 5] == pytest.approx(0.9972065749162169, rel=1e-12)
