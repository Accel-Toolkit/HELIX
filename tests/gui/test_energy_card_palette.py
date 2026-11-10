"""The Lattice tab's palette: "E-set" adds the HELIX energy card, among the commands; SET_BEAM_ENERGY is a criterion."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from linac_gen.elements.lattice_commands import SetBeamEnergy, SetReferenceEnergy  # noqa: E402


def test_the_palette_offers_the_helix_card_and_the_criterion():
    from linac_gen_gui.interphase.element_factory import make_default
    from linac_gen_gui.interphase.panels.element_palette import _LABELS
    assert isinstance(make_default("HELIX_SET_ENERGY"), SetReferenceEnergy) and _LABELS["HELIX_SET_ENERGY"] == "E-set"
    assert isinstance(make_default("SET_BEAM_ENERGY"), SetBeamEnergy) and _LABELS["SET_BEAM_ENERGY"] == "E-goal"


def test_the_palette_puts_the_helix_card_with_the_commands_and_it_does_nothing_until_set(qapp):
    """It fell in the palette's OTHER section, and its default set the energy to zero, which no run survives."""
    from linac_gen_gui.interphase.element_factory import make_default
    from linac_gen_gui.interphase.panels.element_palette import ElementPalette
    from PyQt6.QtWidgets import QLabel, QToolButton
    card = make_default("HELIX_SET_ENERGY")
    assert card.shift and card.energy_MeV == 0.0                  # a shift of nothing until it is edited
    palette = ElementPalette()
    try:
        section, where = None, {}
        for w in palette.findChildren((QLabel, QToolButton)):
            if isinstance(w, QLabel) and w.text().isupper() and w.text() != "PALETTE":
                section = w.text()
            elif isinstance(w, QToolButton) and getattr(w, "_type_name", None):
                where[w._type_name] = section
        assert where["HELIX_SET_ENERGY"] == where["SET_BEAM_ENERGY"] == "SET_* COMMANDS"
    finally:
        palette.deleteLater()
