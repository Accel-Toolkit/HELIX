"""Keep macOS App Nap off while HELIX computes.

When the HELIX window is not in front, macOS puts the app into App Nap:
every thread drops to scheduling priority 4 and a run gets about 1.5 cores
instead of all of them (measured 2026-10-13: a 3-thread computation went
from 3.00 cores to 1.5 within a minute of the window losing focus).  The
command line is not affected.  An NSProcessInfo *activity* tells the system
that user-initiated work is in progress, which keeps the app awake; it is
the mechanism IPython uses through the appnope package (not a dependency:
this is the same few calls through ctypes).  The system may still idle-
sleep, as with any other app.

:class:`ComputeActivity` holds the activity while the app computes: while
``AppState.running`` is set (envelope, multiparticle, backtrack and bunch-
train runs, assistant turns) or while the process has used at least half a
core over the last poll interval (studies, scans, matching, ORM fits,
reliability campaigns, popup workers: every computation, whoever started
it).  It is released after two idle intervals.  Off macOS, or if the calls
fail, everything here does nothing.
"""
from __future__ import annotations

import logging
import sys
import time

from PyQt6.QtCore import QObject, QTimer

_log = logging.getLogger(__name__)

#: NSActivityUserInitiated with idle system sleep still allowed
#: (Foundation's NSActivityUserInitiatedAllowingIdleSystemSleep)
_NS_ACTIVITY_IDLE_SYSTEM_SLEEP_DISABLED = 1 << 20
_NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF | _NS_ACTIVITY_IDLE_SYSTEM_SLEEP_DISABLED
ACTIVITY_OPTIONS = (_NS_ACTIVITY_USER_INITIATED
                    & ~_NS_ACTIVITY_IDLE_SYSTEM_SLEEP_DISABLED)


class _NoBackend:
    """Off macOS: nothing to do."""

    available = False

    def begin(self, reason: str):
        return None

    def end(self, token) -> None:
        pass


class _CocoaBackend:
    """``[[NSProcessInfo processInfo] beginActivityWithOptions:reason:]`` and
    ``endActivity:`` through the Objective-C runtime.  The token is retained
    while the activity is held (the call returns it autoreleased, and Qt's
    event loop drains autorelease pools; a deallocated token ends the
    activity)."""

    available = True

    def __init__(self):
        import ctypes
        import ctypes.util
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        P = ctypes.c_void_p
        objc.objc_getClass.restype = P
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = P
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        addr = ctypes.cast(objc.objc_msgSend, P).value
        # one prototype per call shape: objc_msgSend is not variadic on
        # arm64, so every call needs its exact signature
        self._send = ctypes.CFUNCTYPE(P, P, P)(addr)
        self._send_str = ctypes.CFUNCTYPE(P, P, P, ctypes.c_char_p)(addr)
        self._send_begin = ctypes.CFUNCTYPE(P, P, P, ctypes.c_uint64, P)(addr)
        self._send_obj = ctypes.CFUNCTYPE(None, P, P, P)(addr)
        self._sel = objc.sel_registerName
        self._info = self._send(objc.objc_getClass(b"NSProcessInfo"),
                                self._sel(b"processInfo"))
        self._nsstring = objc.objc_getClass(b"NSString")
        if not (self._info and self._nsstring):
            raise OSError("NSProcessInfo / NSString not found")

    def begin(self, reason: str):
        text = self._send_str(self._nsstring,
                              self._sel(b"stringWithUTF8String:"),
                              reason.encode("utf-8"))
        token = self._send_begin(self._info,
                                 self._sel(b"beginActivityWithOptions:reason:"),
                                 ACTIVITY_OPTIONS, text)
        if not token:
            raise OSError("beginActivityWithOptions:reason: returned nil")
        self._send(token, self._sel(b"retain"))
        return token

    def end(self, token) -> None:
        if not token:
            return
        self._send_obj(self._info, self._sel(b"endActivity:"), token)
        self._send(token, self._sel(b"release"))


def _make_backend():
    if sys.platform != "darwin":
        return _NoBackend()
    try:
        return _CocoaBackend()
    except Exception as exc:                                 # noqa: BLE001
        _log.info("App Nap control unavailable (%s)", exc)
        return _NoBackend()


class AppNapBlocker:
    """Reference-counted hold of one activity: the first :meth:`hold`
    begins it, the matching last :meth:`release` ends it.  A failing call
    disables the blocker for good (logged once); it never raises."""

    REASON = "HELIX simulation running"

    def __init__(self, backend=None):
        self._backend = backend if backend is not None else _make_backend()
        self._count = 0
        self._token = None

    @property
    def held(self) -> bool:
        return self._count > 0

    def hold(self) -> None:
        self._count += 1
        if self._count == 1:
            try:
                self._token = self._backend.begin(self.REASON)
            except Exception as exc:                         # noqa: BLE001
                self._disable(exc)

    def release(self) -> None:
        if self._count == 0:
            return
        self._count -= 1
        if self._count == 0:
            token, self._token = self._token, None
            try:
                self._backend.end(token)
            except Exception as exc:                         # noqa: BLE001
                self._disable(exc)

    def _disable(self, exc) -> None:
        _log.warning("App Nap control disabled after a failed call: %s", exc)
        self._backend = _NoBackend()
        self._token = None


class ComputeActivity(QObject):
    """Holds an :class:`AppNapBlocker` while the app computes (see the
    module docstring).  ``cpu_clock`` returns the process's CPU time in
    seconds (all threads); tests pass their own clocks and backend."""

    INTERVAL_MS = 2000
    BUSY_CORES = 0.5            # CPU time per wall time that counts as work
    IDLE_TICKS = 2              # idle intervals before the hold is released

    def __init__(self, state, parent=None, *, blocker=None,
                 cpu_clock=time.process_time, wall_clock=time.monotonic):
        super().__init__(parent)
        self._state = state
        self._blocker = blocker if blocker is not None else AppNapBlocker()
        self._cpu_clock = cpu_clock
        self._wall_clock = wall_clock
        self._cpu0 = cpu_clock()
        self._wall0 = wall_clock()
        self._held = False
        self._idle = 0
        self._stopped = False
        self._timer = QTimer(self)
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        state.running_changed.connect(self._on_running_changed)

    @property
    def held(self) -> bool:
        return self._held

    def _on_running_changed(self, running: bool) -> None:
        try:
            if running and not self._stopped:
                self._idle = 0
                self._set_held(True)
        except RuntimeError:            # deleted during shutdown
            pass

    def _tick(self) -> None:
        if self._stopped:
            return
        try:
            cpu, wall = self._cpu_clock(), self._wall_clock()
            dt = wall - self._wall0
            busy_cpu = dt > 0 and (cpu - self._cpu0) >= self.BUSY_CORES * dt
            self._cpu0, self._wall0 = cpu, wall
            if busy_cpu or bool(getattr(self._state, "running", False)):
                self._idle = 0
                self._set_held(True)
            elif self._held:
                self._idle += 1
                if self._idle >= self.IDLE_TICKS:
                    self._set_held(False)
        except RuntimeError:            # deleted during shutdown
            pass

    def _set_held(self, on: bool) -> None:
        if on and not self._held:
            self._blocker.hold()
            self._held = True
        elif self._held and not on:
            self._blocker.release()
            self._held = False

    def stop(self) -> None:
        """Release the hold and stop for good (window teardown): a run flag
        raised afterwards takes no new hold."""
        self._stopped = True
        self._timer.stop()
        self._set_held(False)
