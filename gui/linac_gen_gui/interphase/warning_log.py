"""HELIX's warnings for the status bar.

A GUI user does not see stderr, which is where they otherwise end up.  ``install``
sends them to ``AppState.log_event``, for ``chrome/statusbar.py`` to list, and leaves
the terminal as it was.

Two routes: what is logged under the ``linac_gen`` logger, and ``warnings.warn``.  Of
the second, only ``UserWarning`` and its subclasses from HELIX's own files count: numpy
attributes "invalid value encountered" to the line that called it, which may be ours.

``warnings.warn`` is hooked at ``warnings._showwarnmsg``, which every warning that
passes the filters goes through, and which ``catch_warnings`` does not touch.
``warnings.showwarning`` would not do: the app records warnings with ``catch_warnings``
in several threads, each such block takes it over for all of them, and two blocks that
overlap leave a dead recorder in place when the second one ends.

There is one handler for the process and it follows the newest window.  It holds the
window's state weakly, because that state can die first: a logger outlives every
window, and worker threads can still log after one closes.
"""
from __future__ import annotations

import contextlib
import logging
import warnings
import weakref

_state_ref = None
_next_showwarnmsg = None


def _send(text: str) -> None:
    state = _state_ref() if _state_ref is not None else None
    if state is not None:
        state.log_event.emit("warning", text)


class _LogForward(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):    # (its C++ side gone) a log call must never raise
            _send(record.getMessage())
        # Python prints through lastResort only while no handler exists at all, and
        # this is one: without this the terminal would go silent.
        if logging.lastResort is not None and not logging.getLogger().handlers:
            logging.lastResort.handle(record)


_handler = _LogForward()


def _hook(msg) -> None:
    with contextlib.suppress(Exception):        # as above
        if issubclass(msg.category, UserWarning) and "linac_gen" in str(msg.filename):
            _send(str(msg.message))
    _next_showwarnmsg(msg)


def install(state) -> None:
    """Send HELIX's warnings to ``state.log_event`` from now on."""
    global _state_ref, _next_showwarnmsg
    _state_ref = weakref.ref(state)
    logging.getLogger("linac_gen").addHandler(_handler)     # adds it once, at most
    current = getattr(warnings, "_showwarnmsg", None)       # a private name
    if current is not None and current is not _hook:
        _next_showwarnmsg = current
        warnings._showwarnmsg = _hook
