"""Lattice command bus — every list mutation flows through here.

The ``CommandBus`` is the foundation for undo/redo, the dirty-flag, and
"unsaved changes" prompts.  Views must NOT mutate ``lattice.elements``
directly; instead they construct a ``LatticeCommand`` and push it via
``bus.do(cmd)``.

Element identity is preserved across undo/redo for ``InsertCommand``,
``DeleteCommand``, ``MoveCommand`` — i.e. the same Python object comes
back, so attribute edits made via the inspector after a delete-undo
remain attached to the correct element.  Only ``Cut`` / ``Copy``
explicitly deep-copy.

``ParamChangeCommand`` coalesces consecutive same-(element, attr) edits
within a 600 ms window into a single undoable step, so dragging a
spinbox does not flood the stack.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal


# ---------------------------------------------------------------------------
# Command base
# ---------------------------------------------------------------------------
class LatticeCommand(ABC):
    """A reversible mutation of a :class:`linac_gen.core.lattice.Lattice`."""

    @abstractmethod
    def do(self, lattice) -> None: ...

    @abstractmethod
    def undo(self, lattice) -> None: ...

    # Optional: human label for menus / status bar.
    label: str = "Edit"

    # Monotonic edit sequence stamped by CommandBus.do(); lets the bus
    # itemize "changes since the last save" (see describe_changes_since_clean).
    _seq: int = 0

    def describe(self) -> str:
        """One human-readable line for the unsaved-changes prompt."""
        return self.label


# ---------------------------------------------------------------------------
# Concrete commands
# ---------------------------------------------------------------------------
class InsertCommand(LatticeCommand):
    """Insert ``element`` at ``index`` (clamped to [0, len])."""

    label = "Insert element"

    def __init__(self, index: int, element: Any) -> None:
        self.index = int(index)
        self.element = element

    def do(self, lattice) -> None:
        lattice.insert(self.index, self.element)

    def undo(self, lattice) -> None:
        # Use identity-remove so we restore the exact instance only.
        lattice.remove(self.element)

    def describe(self) -> str:
        name = getattr(self.element, "name", "?")
        kind = type(self.element).__name__
        return f"inserted {kind} '{name}' at index {self.index}"


class DeleteCommand(LatticeCommand):
    """Delete ``element`` from the lattice; remembers its position for undo."""

    label = "Delete element"

    def __init__(self, element: Any) -> None:
        self.element = element
        self._index: Optional[int] = None  # populated on do()

    def do(self, lattice) -> None:
        self._index = lattice.remove(self.element)

    def undo(self, lattice) -> None:
        if self._index is None:
            raise RuntimeError("DeleteCommand.undo before do")
        lattice.insert(self._index, self.element)

    def describe(self) -> str:
        name = getattr(self.element, "name", "?")
        kind = type(self.element).__name__
        return f"deleted {kind} '{name}'"


class MoveCommand(LatticeCommand):
    """Move element from ``from_idx`` to ``to_idx`` (after the move)."""

    label = "Move element"

    def __init__(self, from_idx: int, to_idx: int) -> None:
        self.from_idx = int(from_idx)
        self.to_idx = int(to_idx)

    def do(self, lattice) -> None:
        if self.from_idx == self.to_idx:
            return
        el = lattice.elements.pop(self.from_idx)
        lattice.elements.insert(self.to_idx, el)

    def undo(self, lattice) -> None:
        if self.from_idx == self.to_idx:
            return
        el = lattice.elements.pop(self.to_idx)
        lattice.elements.insert(self.from_idx, el)

    def describe(self) -> str:
        return f"moved element from index {self.from_idx} to {self.to_idx}"


class ParamChangeCommand(LatticeCommand):
    """Set ``element.attr = new_value`` reversibly.

    Designed to be coalesced — see :meth:`CommandBus.do` and the
    ``coalesce_with`` method.  Coalescing keeps the original
    ``old_value`` so a long drag-edit collapses to one undo step.
    """

    label = "Change parameter"

    def __init__(self, element: Any, attr: str, old_value: Any, new_value: Any) -> None:
        self.element = element
        self.attr = attr
        self.old_value = old_value
        self.new_value = new_value
        self._stamp = time.monotonic()

    def do(self, lattice) -> None:
        setattr(self.element, self.attr, self.new_value)

    def undo(self, lattice) -> None:
        setattr(self.element, self.attr, self.old_value)

    def can_coalesce_with(self, other: "ParamChangeCommand", *,
                          window_s: float = 0.6) -> bool:
        return (
            isinstance(other, ParamChangeCommand)
            and other.element is self.element
            and other.attr == self.attr
            and (other._stamp - self._stamp) < window_s
        )

    def coalesce(self, other: "ParamChangeCommand") -> None:
        """Absorb ``other`` into self — keep our old_value, take their new_value."""
        self.new_value = other.new_value
        self._stamp = other._stamp

    def describe(self) -> str:
        name = getattr(self.element, "name", "?")
        kind = type(self.element).__name__

        def _fmt(v):
            return f"{v:.6g}" if isinstance(v, float) else repr(v)
        return (f"{kind} '{name}': {self.attr} "
                f"{_fmt(self.old_value)} → {_fmt(self.new_value)}")


class MacroCommand(LatticeCommand):
    """A grouped sequence of commands undone/redone as one unit."""

    label = "Edit"

    def __init__(self, commands: list[LatticeCommand], label: str = "Edit") -> None:
        self.commands = list(commands)
        self.label = label

    def do(self, lattice) -> None:
        for c in self.commands:
            c.do(lattice)

    def undo(self, lattice) -> None:
        for c in reversed(self.commands):
            c.undo(lattice)

    def describe(self) -> str:
        if not self.commands:
            return self.label
        inner = "; ".join(c.describe() for c in self.commands[:3])
        more = len(self.commands) - 3
        if more > 0:
            inner += f"; … {more} more"
        return f"{self.label} ({inner})"


# ---------------------------------------------------------------------------
# Command bus
# ---------------------------------------------------------------------------
class CommandBus(QObject):
    """Bounded undo/redo stack with a dirty flag and Qt signals.

    ``maxlen=50`` matches typical IDE undo histories — enough to roll
    back a session-worth of edits without unbounded memory.

    Signals
    -------
    changed
        Emitted after any successful do/undo/redo.  Views should listen
        to ``state.lattice_changed`` rather than directly to ``changed``;
        the bus calls into ``AppState`` which broadcasts.
    can_undo_changed / can_redo_changed
        Toolbar buttons toggle on these.
    dirty_changed
        Emitted when the dirty flag flips.
    """

    changed             = pyqtSignal()
    can_undo_changed    = pyqtSignal(bool)
    can_redo_changed    = pyqtSignal(bool)
    dirty_changed       = pyqtSignal(bool)

    def __init__(self, lattice_provider, maxlen: int = 50) -> None:
        """``lattice_provider`` is a zero-arg callable returning the live
        :class:`Lattice`; using a callable lets the bus track lattice
        replacement without holding a stale reference."""
        super().__init__()
        self._get_lattice = lattice_provider
        self._undo: deque[LatticeCommand] = deque(maxlen=maxlen)
        self._redo: deque[LatticeCommand] = deque(maxlen=maxlen)
        self._dirty: bool = False
        # Edit-sequence bookkeeping for "what changed since the last
        # save": every executed command is stamped with a monotonically
        # increasing _seq; mark_clean() records the sequence of the
        # stack top, and describe_changes_since_clean() itemizes the
        # commands stamped after it.
        self._next_seq: int = 1
        self._clean_seq: int = 0
        # Divergence that undo can no longer reach: set by mark_dirty()
        # (wholesale lattice replacement) and by undo-deque overflow
        # (the oldest edit fell off the stack).  While set, an empty
        # undo stack does NOT mean clean.
        self._forced_dirty: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def do(self, command: LatticeCommand) -> None:
        lattice = self._get_lattice()
        if lattice is None:
            return
        # Coalesce consecutive ParamChangeCommand on the same (element, attr)
        # — but never ACROSS the save point: merging into a command that
        # predates mark_clean() would hide the new edit from the
        # "changes since save" itemization and corrupt the clean marker.
        if (
            isinstance(command, ParamChangeCommand)
            and self._undo
            and isinstance(self._undo[-1], ParamChangeCommand)
            and self._undo[-1]._seq > self._clean_seq
            and self._undo[-1].can_coalesce_with(command)
        ):
            # Just apply and merge — no new stack entry.  A coalesced edit is
            # still a fresh user action that diverges from any redo history, so
            # the redo stack must be cleared here too — exactly as the
            # non-coalesce path does at ``self._redo.clear()`` below.  Without
            # this, Redo after a coalesced edit re-applies a change the user
            # had already undone, corrupting the lattice.
            command.do(lattice)
            self._undo[-1].coalesce(command)
            self._redo.clear()
            self._after_change()
            return
        command.do(lattice)
        if len(self._undo) == self._undo.maxlen:
            # The oldest command is about to fall off the bounded deque —
            # undoing everything left can no longer restore the on-disk
            # state, so the dirty flag must survive a full unwind.
            self._forced_dirty = True
        command._seq = self._next_seq
        self._next_seq += 1
        self._undo.append(command)
        self._redo.clear()
        self._after_change()

    def undo(self) -> bool:
        if not self._undo:
            return False
        lattice = self._get_lattice()
        if lattice is None:
            return False
        cmd = self._undo.pop()
        cmd.undo(lattice)
        self._redo.append(cmd)
        self._after_change()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        lattice = self._get_lattice()
        if lattice is None:
            return False
        cmd = self._redo.pop()
        cmd.do(lattice)
        self._undo.append(cmd)
        self._after_change()
        return True

    def reset(self) -> None:
        """Clear the stacks and mark clean (call when a new lattice loads)."""
        self._undo.clear()
        self._redo.clear()
        self._clean_seq = 0
        was_dirty = self._dirty
        self._dirty = False
        self._forced_dirty = False
        self.can_undo_changed.emit(False)
        self.can_redo_changed.emit(False)
        if was_dirty:
            self.dirty_changed.emit(False)
        self.changed.emit()

    def mark_clean(self) -> None:
        """Clear the dirty flag without touching the undo stack (after Save)."""
        self._forced_dirty = False
        self._clean_seq = self._undo[-1]._seq if self._undo else 0
        if self._dirty:
            self._dirty = False
            self.dirty_changed.emit(False)

    def describe_changes_since_clean(self, limit: int = 30) -> list[str]:
        """Human-readable lines for edits made after the last save point.

        Semantics mirror the dirty flag they explain:

        * commands on the undo stack stamped after ``mark_clean()`` are
          itemized via :meth:`LatticeCommand.describe`;
        * if the stack top sits *below* the save point (the user undid
          past it), the divergence is reported as such — the individual
          undone edits are gone from an itemization point of view;
        * a forced divergence (wholesale replacement via
          :meth:`mark_dirty`, or undo-deque overflow) gets its own line,
          because no command object describes it.

        Returns at most ``limit`` lines (+ an ellipsis line); empty when
        nothing changed.
        """
        lines: list[str] = []
        if self._forced_dirty:
            lines.append("lattice replaced or edited wholesale "
                         "(e.g. applied match result) — not itemizable")
        top_seq = self._undo[-1]._seq if self._undo else 0
        if top_seq < self._clean_seq:
            lines.append("edits undone PAST the last save point — the "
                         "lattice diverges from the saved file")
        fresh = [c for c in self._undo if c._seq > self._clean_seq]
        for c in fresh[:limit]:
            lines.append(c.describe())
        if len(fresh) > limit:
            lines.append(f"… and {len(fresh) - limit} more edits")
        if not lines and self._dirty:
            lines.append("lattice edits (details unavailable)")
        return lines

    def mark_dirty(self) -> None:
        """Force the dirty flag without going through the undo stack.

        Used when the lattice has been replaced wholesale (e.g. after
        applying a matcher result) and there's no command to record but
        the in-memory state still diverges from the on-disk .dat.
        The forced flag sticks until mark_clean()/reset() — a later
        edit-then-undo must not erase it (_after_change would otherwise
        recompute dirty from the empty undo stack).
        """
        self._forced_dirty = True
        if not self._dirty:
            self._dirty = True
            self.dirty_changed.emit(True)

    @property
    def can_undo(self) -> bool: return bool(self._undo)
    @property
    def can_redo(self) -> bool: return bool(self._redo)
    @property
    def dirty(self) -> bool: return self._dirty

    # ------------------------------------------------------------------
    def _after_change(self) -> None:
        # Dirty ⇔ the stack top is not the save point (either newer
        # edits exist, or the user undid PAST the save — both diverge
        # from the file on disk).  Redoing back to the exact save point
        # is clean again.  Forced divergence sticks regardless.
        # (The old `bool(self._undo)` rule missed the save-then-
        # undo-everything case: diverged from disk yet reported clean.)
        top_seq = self._undo[-1]._seq if self._undo else 0
        new_dirty = (top_seq != self._clean_seq) or self._forced_dirty
        if new_dirty != self._dirty:
            self._dirty = new_dirty
            self.dirty_changed.emit(new_dirty)
        self.can_undo_changed.emit(self.can_undo)
        self.can_redo_changed.emit(self.can_redo)
        self.changed.emit()
