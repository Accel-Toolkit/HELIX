"""Unit tests for the lattice-impression strip used in Results popups.

The strip widget renders coloured rectangles for finite-length elements
and thin vertical lines for physically-meaningful zero-length elements
(Foil, RFGap, Aperture, etc.).  Its x-axis links to a host plot via
``pg.ViewBox.setXLink`` so panning/zooming the host moves the strip
with it.

Tests cover:
  * widget builds without error on a known lattice
  * correct number of items per element category (rect vs line)
  * x-axis link to the host plot is established
  * fixed pixel height (no surprise growth)
  * empty / None lattice clears the strip cleanly
  * passive ``LatticeCommand`` elements are silently skipped
  * theme.EL_COLORS Foil entry is reachable
"""
import pyqtgraph as pg
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.foil import Foil
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.lattice_commands import SetSyncPhase


# ── helpers ──────────────────────────────────────────────────────────────────

def _strip_lattice() -> Lattice:
    """Build a Drift, Quad, Drift, Foil, RFGap, SET_SYNC_PHASE, Drift lattice."""
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="Q1", length=200.0, gradient=10.0, aperture=10.0))
    lat.add(Drift(name="D2", length=300.0, aperture=10.0))
    lat.add(Foil(name="STRIP", material="C", thickness_ug_cm2=600.0))
    lat.add(RFGap(name="G1", voltage=0.5, phase=-30.0, frequency=162.5))
    lat.add(SetSyncPhase(name="SET_SYNC_PHASE"))   # passive command — should be skipped
    lat.add(Drift(name="D3", length=200.0, aperture=10.0))
    return lat


# ── tests ────────────────────────────────────────────────────────────────────

def test_widget_builds_without_error(qapp):
    from linac_gen_gui.interphase.plots.lattice_strip import (
        LatticeStripWidget,
    )
    strip = LatticeStripWidget()
    assert strip is not None
    assert strip._items == []
    assert strip._total_length_mm == 0.0


def test_set_lattice_populates_correct_item_count(qapp):
    """4 finite-length elements → 4 rects;
    Foil (L=0, kept) + RFGap (L=0, kept) → 2 lines;
    SetSyncPhase (passive command) → skipped.
    Total: 6 items."""
    from linac_gen_gui.interphase.plots.lattice_strip import (
        LatticeStripWidget,
    )
    lat = _strip_lattice()
    strip = LatticeStripWidget()
    strip.set_lattice(lat)
    assert len(strip._items) == 6, (
        f"Expected 6 items (4 rects + 2 lines), got {len(strip._items)}"
    )
    # Total length = 100 + 200 + 300 + 0 + 0 + 0 + 200 = 800 mm
    assert strip._total_length_mm == pytest.approx(800.0)


def test_strip_is_fixed_height(qapp):
    from linac_gen_gui.interphase.plots.lattice_strip import (
        LatticeStripWidget,
    )
    strip = LatticeStripWidget()
    assert strip.minimumHeight() == LatticeStripWidget.STRIP_HEIGHT_PX
    assert strip.maximumHeight() == LatticeStripWidget.STRIP_HEIGHT_PX


def test_x_axis_links_to_host_plot(qapp):
    from linac_gen_gui.interphase.plots.lattice_strip import (
        make_lattice_strip,
    )
    host = pg.PlotWidget()
    strip = make_lattice_strip(None, host, _strip_lattice())
    linked = strip.getPlotItem().getViewBox().linkedView(0)
    assert linked is host.getPlotItem().getViewBox(), (
        "Strip x-axis should be linked to host plot's ViewBox"
    )


def test_passing_none_lattice_yields_empty_strip(qapp):
    from linac_gen_gui.interphase.plots.lattice_strip import (
        make_lattice_strip,
    )
    host = pg.PlotWidget()
    strip = make_lattice_strip(None, host, None)
    assert len(strip._items) == 0
    assert strip._total_length_mm == 0.0


def test_set_lattice_can_be_called_repeatedly(qapp):
    """Calling set_lattice a second time clears the previous items."""
    from linac_gen_gui.interphase.plots.lattice_strip import (
        LatticeStripWidget,
    )
    strip = LatticeStripWidget()
    strip.set_lattice(_strip_lattice())
    assert len(strip._items) == 6

    # Reset with a smaller lattice.
    small = Lattice()
    small.add(Drift(name="ONLY", length=50.0, aperture=10.0))
    strip.set_lattice(small)
    assert len(strip._items) == 1
    assert strip._total_length_mm == 50.0


def test_passive_lattice_command_is_skipped(qapp):
    """SetSyncPhase has length=0 and is a LatticeCommand — must be skipped."""
    from linac_gen_gui.interphase.plots.lattice_strip import (
        LatticeStripWidget,
    )
    lat = Lattice()
    lat.add(Drift(name="D", length=100.0, aperture=10.0))
    lat.add(SetSyncPhase(name="SYNC"))
    strip = LatticeStripWidget()
    strip.set_lattice(lat)
    # 1 drift rect + 0 (SetSyncPhase skipped) = 1 item.
    assert len(strip._items) == 1


def test_theme_has_foil_color_entry(qapp):
    """Foil element type must have a colour in EL_COLORS — added in this work."""
    from linac_gen_gui.interphase import theme
    assert "Foil" in theme.EL_COLORS
    # Sanity: looks like a hex colour string.
    assert theme.EL_COLORS["Foil"].startswith("#")
    assert len(theme.EL_COLORS["Foil"]) == 7


def test_rect_tooltip_has_element_info(qapp):
    """Each rectangle's tooltip should carry name + s-range for hover-info."""
    from linac_gen_gui.interphase.plots.lattice_strip import (
        LatticeStripWidget,
    )
    lat = Lattice()
    lat.add(Drift(name="DUMMY", length=100.0, aperture=10.0))
    strip = LatticeStripWidget()
    strip.set_lattice(lat)
    rect = strip._items[0]
    tip = rect.toolTip()
    assert "DUMMY" in tip
    assert "Drift" in tip
    assert "0.0" in tip
    assert "100" in tip
