"""Pure-Python tests for the CommandBus and the four command types.

These tests don't need pytest-qt — the only Qt machinery here is
``QObject`` / ``pyqtSignal`` which work without a running app.
"""
from __future__ import annotations

import time

import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen_gui.interphase.commands import (
    CommandBus, DeleteCommand, InsertCommand, MoveCommand,
    ParamChangeCommand,
)


def _mini_lattice() -> Lattice:
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="Q1", length=200.0, gradient=10.0, aperture=10.0))
    lat.add(Drift(name="D2", length=100.0, aperture=10.0))
    return lat


def test_lattice_insert_remove_helpers():
    lat = _mini_lattice()
    assert len(lat) == 3
    new = Drift(name="D_new", length=50.0, aperture=10.0)
    lat.insert(1, new)
    assert lat.elements[1] is new
    idx = lat.remove(new)
    assert idx == 1 and len(lat) == 3


def test_insert_undo_redo_preserves_identity():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    new = Drift(name="D_new", length=50.0, aperture=10.0)
    bus.do(InsertCommand(1, new))
    assert lat.elements[1] is new
    bus.undo()
    assert new not in lat.elements
    bus.redo()
    assert lat.elements[1] is new   # SAME instance, not a copy


def test_delete_undo_preserves_identity():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    target = lat.elements[1]
    bus.do(DeleteCommand(target))
    assert target not in lat.elements
    bus.undo()
    assert lat.elements[1] is target


def test_move_command_via_indices():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    a, b, c = lat.elements
    bus.do(MoveCommand(0, 2))   # move D1 to position 2
    assert lat.elements == [b, c, a]
    bus.undo()
    assert lat.elements == [a, b, c]


def test_param_change_command_round_trips():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    q = lat.elements[1]
    assert q.gradient == 10.0
    bus.do(ParamChangeCommand(q, "gradient", q.gradient, 25.0))
    assert q.gradient == 25.0
    bus.undo()
    assert q.gradient == 10.0
    bus.redo()
    assert q.gradient == 25.0


def test_param_change_coalesces_within_window():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    q = lat.elements[1]
    # Three quick edits within 600 ms — should collapse to one undo step.
    bus.do(ParamChangeCommand(q, "gradient", 10.0, 11.0))
    bus.do(ParamChangeCommand(q, "gradient", 11.0, 12.0))
    bus.do(ParamChangeCommand(q, "gradient", 12.0, 13.0))
    assert q.gradient == 13.0
    bus.undo()
    # Single undo restores all the way back to the original.
    assert q.gradient == 10.0


def test_param_change_does_not_coalesce_after_window():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    q = lat.elements[1]
    bus.do(ParamChangeCommand(q, "gradient", 10.0, 11.0))
    time.sleep(0.7)   # > 600 ms
    bus.do(ParamChangeCommand(q, "gradient", 11.0, 12.0))
    assert q.gradient == 12.0
    bus.undo()
    assert q.gradient == 11.0    # only the second edit was undone
    bus.undo()
    assert q.gradient == 10.0


def test_param_change_does_not_coalesce_different_attr():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    q = lat.elements[1]
    bus.do(ParamChangeCommand(q, "gradient", 10.0, 11.0))
    bus.do(ParamChangeCommand(q, "length", 200.0, 250.0))
    bus.undo()
    assert q.length == 200.0 and q.gradient == 11.0


def test_bus_drops_oldest_past_maxlen():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat, maxlen=3)
    # Push 5 distinct commands so coalescing doesn't fold them.
    for i in range(5):
        new = Drift(name=f"D_{i}", length=10.0, aperture=10.0)
        bus.do(InsertCommand(0, new))
    # Bus held only the most recent 3 — older two were dropped.
    assert len(bus._undo) == 3
    # Three undos remove the three remaining inserts.
    for _ in range(3):
        assert bus.undo()
    assert not bus.undo()


def test_redo_cleared_on_new_do():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    new = Drift(name="DX", length=10.0, aperture=10.0)
    bus.do(InsertCommand(0, new))
    bus.undo()
    assert bus.can_redo
    bus.do(InsertCommand(0, Drift(name="DY", length=10.0, aperture=10.0)))
    assert not bus.can_redo


def test_dirty_flag_clears_on_full_undo():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    assert not bus.dirty
    bus.do(InsertCommand(0, Drift(name="DX", length=10.0, aperture=10.0)))
    assert bus.dirty
    bus.undo()
    assert not bus.dirty   # bus is empty → clean


def test_reset_clears_stacks_and_dirty():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    bus.do(InsertCommand(0, Drift(name="DX", length=10.0, aperture=10.0)))
    bus.reset()
    assert not bus.can_undo and not bus.can_redo and not bus.dirty


def test_mark_clean_clears_dirty_keeps_history():
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    bus.do(InsertCommand(0, Drift(name="DX", length=10.0, aperture=10.0)))
    bus.mark_clean()
    assert not bus.dirty
    assert bus.can_undo   # history preserved (Save shouldn't lose it)


def test_forced_dirty_survives_edit_then_undo():
    """mark_dirty() records divergence undo cannot reach (e.g. a matcher
    result applied wholesale).  A subsequent edit+undo must not launder
    the flag back to clean."""
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat)
    bus.mark_dirty()
    assert bus.dirty
    bus.do(InsertCommand(0, Drift(name="DX", length=10.0, aperture=10.0)))
    bus.undo()
    assert bus.dirty          # was: cleared, because the stack is empty
    bus.mark_clean()
    assert not bus.dirty      # Save resolves the forced divergence
    bus.mark_dirty()
    bus.reset()
    assert not bus.dirty      # new lattice load resolves it too


def test_overflow_keeps_dirty_after_full_unwind():
    """When the bounded undo deque drops its oldest entry, undoing
    everything that remains cannot restore the on-disk state — the
    dirty flag must survive the full unwind."""
    lat = _mini_lattice()
    bus = CommandBus(lambda: lat, maxlen=2)
    for i in range(3):        # third do() evicts the first command
        bus.do(InsertCommand(0, Drift(name=f"D_{i}", length=10.0,
                                      aperture=10.0)))
    while bus.undo():
        pass
    assert bus.dirty          # was: clean, despite one unreachable edit
