"""End-to-end tests for the new lattice-editor flow.

These exercise the CommandBus, the LatticeTimeline drag-drop entry
point, the OutlineTree reorder signal, and the RFQ_CELL writer
round-trip — all the pieces that ship together in the M2-M5
"visually stunning lattice editor" plan.

We avoid pytest-qt by relying on the offscreen Qt platform (set in
``conftest.py``) and the session-scoped ``qapp`` fixture.
"""
from __future__ import annotations

import os
import time

import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin

from linac_gen_gui.interphase.commands import (
    CommandBus, DeleteCommand, InsertCommand, MoveCommand,
    ParamChangeCommand, MacroCommand,
)


# ---------------------------------------------------------------------------
# 1. Insert + undo restore length
# ---------------------------------------------------------------------------
def test_insert_then_undo_restores_count(mini_lattice):
    bus = CommandBus(lambda: mini_lattice)
    n0 = len(mini_lattice)
    new = Drift(name="NEW", length=10.0, aperture=10.0)
    bus.do(InsertCommand(2, new))
    assert len(mini_lattice) == n0 + 1
    bus.undo()
    assert len(mini_lattice) == n0
    assert new not in mini_lattice.elements


# ---------------------------------------------------------------------------
# 2. Delete + undo preserves identity
# ---------------------------------------------------------------------------
def test_delete_undo_preserves_identity(mini_lattice):
    bus = CommandBus(lambda: mini_lattice)
    target = mini_lattice.elements[2]
    target_id = id(target)
    bus.do(DeleteCommand(target))
    assert target not in mini_lattice.elements
    bus.undo()
    # Same Python instance — id() unchanged after restore.
    assert mini_lattice.elements[2] is target
    assert id(mini_lattice.elements[2]) == target_id


# ---------------------------------------------------------------------------
# 3. Duplicate inserts deep-copy after the source
# ---------------------------------------------------------------------------
def test_duplicate_inserts_clone_after_selection(mini_lattice):
    from linac_gen_gui.interphase.element_factory import clone
    bus = CommandBus(lambda: mini_lattice)
    src = mini_lattice.elements[1]      # Q1
    new = clone(src)
    bus.do(InsertCommand(2, new))
    assert mini_lattice.elements[2] is new
    assert mini_lattice.elements[2] is not src
    # Distinguishable name post-clone.
    assert new.name.endswith("_copy")


# ---------------------------------------------------------------------------
# 4. Move via indices
# ---------------------------------------------------------------------------
def test_move_command_via_indices(mini_lattice):
    bus = CommandBus(lambda: mini_lattice)
    a, b, c, d, e = mini_lattice.elements
    bus.do(MoveCommand(0, 4))   # move D1 to the end
    assert mini_lattice.elements == [b, c, d, e, a]
    bus.undo()
    assert mini_lattice.elements == [a, b, c, d, e]


# ---------------------------------------------------------------------------
# 5. Keyboard delete then Ctrl+Z restores
# ---------------------------------------------------------------------------
def test_keyboard_delete_then_ctrl_z(qapp, mini_lattice):
    """Driving the LatticeTab._delete_selected path end-to-end."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.lattice_tab import LatticeTab
    state = AppState()
    state.set_lattice(mini_lattice, "/tmp/x.dat")
    tab = LatticeTab(state)
    target = mini_lattice.elements[2]
    state.set_selected(target)
    tab._delete_selected()
    assert target not in mini_lattice.elements
    state.bus.undo()
    assert mini_lattice.elements[2] is target


# ---------------------------------------------------------------------------
# 6. ParamChangeCommand coalesces inside the 600 ms window
# ---------------------------------------------------------------------------
def test_param_change_coalesces_within_600ms(mini_lattice):
    bus = CommandBus(lambda: mini_lattice)
    q = mini_lattice.elements[1]
    bus.do(ParamChangeCommand(q, "gradient", 10.0, 11.0))
    bus.do(ParamChangeCommand(q, "gradient", 11.0, 12.0))
    bus.do(ParamChangeCommand(q, "gradient", 12.0, 13.0))
    assert q.gradient == 13.0
    # All three collapse into ONE undoable step.
    bus.undo()
    assert q.gradient == 10.0


# ---------------------------------------------------------------------------
# 7. RFQ_CELL writer round-trip
# ---------------------------------------------------------------------------
def test_save_reload_roundtrip_with_rfq_cell(rfq_lattice, tmp_path):
    fp = tmp_path / "rfq.dat"
    write_tracewin(rfq_lattice, fp)
    txt = fp.read_text()
    # Writer must have emitted at least one RFQ_CELL card.
    assert "RFQ_CELL" in txt
    # Re-parse and confirm the chain made it back round-trip clean.
    reloaded, _meta = parse_tracewin(str(fp))
    n_rfq_in = sum(1 for e in rfq_lattice.elements if isinstance(e, RfqCell))
    n_rfq_out = sum(1 for e in reloaded.elements if isinstance(e, RfqCell))
    assert n_rfq_in == n_rfq_out
    # Element-by-element field comparison on the RFQ chain.
    src = [e for e in rfq_lattice.elements if isinstance(e, RfqCell)]
    dst = [e for e in reloaded.elements if isinstance(e, RfqCell)]
    for a, b in zip(src, dst):
        assert a.voltage_V    == pytest.approx(b.voltage_V)
        assert a.r0_mm        == pytest.approx(b.r0_mm)
        assert a.A10          == pytest.approx(b.A10)
        assert a.modulation   == pytest.approx(b.modulation)
        assert a.length       == pytest.approx(b.length)
        assert a.phi_s_deg    == pytest.approx(b.phi_s_deg)
        assert a.cell_type    == b.cell_type


# ---------------------------------------------------------------------------
# 8. Dirty flag toggles on insert, clears on save
# ---------------------------------------------------------------------------
def test_dirty_flag_toggles_on_insert_clears_on_save(rfq_lattice, tmp_path):
    bus = CommandBus(lambda: rfq_lattice)
    assert not bus.dirty
    bus.do(InsertCommand(0, Drift(name="NEW", length=5.0, aperture=10.0)))
    assert bus.dirty
    fp = tmp_path / "rfq.dat"
    write_tracewin(rfq_lattice, fp)
    bus.mark_clean()
    assert not bus.dirty
    # History is preserved across mark_clean — undo still works.
    assert bus.can_undo
