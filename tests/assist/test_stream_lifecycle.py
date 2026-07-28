"""Crash-regression: PortAudio input-stream lifecycle (SIGSEGV 2026-07-25).

A macOS sleep/wake wedged a blocking ``stream.read()``; teardown's join
timed out, the stream was closed anyway, and PortAudio's ring buffer was
freed under the in-flight read — memmove on freed memory, SIGSEGV inside
``PaUtil_ReadRingBuffer`` with three more readers piled up in
``ReadStream``.  The rules pinned here:

1. only the READER thread ever stops/closes a stream;
2. a wedged reader is PARKED (strong ref kept, stream left open) —
   never abandoned to GC, never closed over its head;
3. a parked owner refuses to open/start again (no stream swap under
   the wedged reader);
4. lifecycle transitions serialize through a timed process-wide lock
   that degrades to unserialized rather than freezing other threads.
"""
from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from linac_gen.assist import listen as L
from linac_gen.assist import voice as V


def _fake_sd(stream_cls, made):
    def factory(**kw):
        s = stream_cls(**kw)
        made.append(s)
        return s
    return types.SimpleNamespace(InputStream=factory)


class _DrainingStream:
    """Yields two blocks, then raises (ends the reader loop cleanly)."""

    def __init__(self, **kw):
        self._n = 0
        self.stopped_by = None
        self.closed_by = None

    def start(self): ...

    def read(self, n):
        self._n += 1
        if self._n > 2:
            raise RuntimeError("drained")
        return np.ones((n, 1), np.float32) * 0.1, False

    def stop(self):
        self.stopped_by = threading.current_thread()

    def close(self):
        self.closed_by = threading.current_thread()


class _WedgedStream:
    """read() blocks until released — CoreAudio's post-sleep hang."""

    def __init__(self, release, **kw):
        self._release = release
        self.stopped_by = None
        self.closed_by = None

    def start(self): ...

    def read(self, n):
        self._release.wait(timeout=10.0)
        raise RuntimeError("drained")           # unwedged → loop exits

    def stop(self):
        self.stopped_by = threading.current_thread()

    def close(self):
        self.closed_by = threading.current_thread()


# ---------------------------------------------------------------------------
# rule 1 — the reader closes, nobody else
# ---------------------------------------------------------------------------
def test_recorder_reader_owns_close(monkeypatch):
    made = []
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sd(_DrainingStream, made))
    rec = V.PushToTalkRecorder()
    rec.start()
    reader = rec._thread
    reader.join(timeout=2.0)
    rec.stop()
    assert made[0].closed_by is reader          # NOT the caller thread
    assert made[0].stopped_by is reader
    assert rec._stream is None


def test_micstream_reader_owns_close(monkeypatch):
    made = []
    monkeypatch.setitem(sys.modules, "sounddevice",
                        _fake_sd(_DrainingStream, made))
    ms = L.MicStream(blocksize=64)
    ms.open()
    reader = ms._thread
    reader.join(timeout=2.0)                    # drains, closes itself
    ms.close()
    assert made[0].closed_by is reader
    assert made[0].stopped_by is reader
    assert ms._stream is None


# ---------------------------------------------------------------------------
# rules 2+3 — wedged reader: park, never close, never reuse
# ---------------------------------------------------------------------------
def test_recorder_wedged_reader_parked_never_closed(monkeypatch):
    release = threading.Event()
    made = []
    fake = types.SimpleNamespace(
        InputStream=lambda **kw: made.append(_WedgedStream(release, **kw))
        or made[-1])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    rec = V.PushToTalkRecorder()
    rec._join_timeout_s = 0.2
    rec.start()
    reader = rec._thread
    audio = rec.stop()                          # join times out — wedged
    assert audio.shape == (0,)                  # honest empty capture
    assert rec in V.PushToTalkRecorder._ABANDONED   # parked, strong ref
    assert made[0].closed_by is None            # NEVER closed over the read
    # rule 3: a parked owner refuses to start again (no stream swap)
    rec.start()
    assert len(made) == 1
    # unwedge → the reader itself finishes the teardown
    release.set()
    reader.join(timeout=2.0)
    assert made[0].closed_by is reader


def test_micstream_wedged_close_parks_and_open_refuses(monkeypatch):
    release = threading.Event()
    made = []
    fake = types.SimpleNamespace(
        InputStream=lambda **kw: made.append(_WedgedStream(release, **kw))
        or made[-1])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)

    ms = L.MicStream(blocksize=64)
    ms._join_timeout_s = 0.2
    ms.open()
    reader = ms._thread
    ms.close()                                  # join times out — wedged
    assert ms in L._ABANDONED_STREAMS           # parked, strong ref
    assert made[0].closed_by is None            # NEVER closed over the read
    ms.open()                                   # rule 3: refused
    assert len(made) == 1
    release.set()
    reader.join(timeout=2.0)
    assert made[0].closed_by is reader


def test_recorder_double_start_is_single_stream(monkeypatch):
    release = threading.Event()
    made = []
    fake = types.SimpleNamespace(
        InputStream=lambda **kw: made.append(_WedgedStream(release, **kw))
        or made[-1])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    rec = V.PushToTalkRecorder()
    rec.start()
    reader = rec._thread
    rec.start()                                 # no-op while recording
    assert len(made) == 1
    release.set()
    reader.join(timeout=2.0)
    rec.stop()
    assert made[0].closed_by is reader


# ---------------------------------------------------------------------------
# rule 4 — the lifecycle lock can never freeze another thread
# ---------------------------------------------------------------------------
def test_pa_lifecycle_lock_times_out_instead_of_hanging():
    assert V._PA_LIFECYCLE_LOCK.acquire(timeout=1.0)
    try:
        entered = []
        with V.pa_lifecycle(timeout=0.05):      # held elsewhere → timed
            entered.append(True)                # proceed unserialized
        assert entered
        # and it must NOT have released the lock it never acquired
        assert V._PA_LIFECYCLE_LOCK.locked()
    finally:
        V._PA_LIFECYCLE_LOCK.release()


def test_pa_lifecycle_lock_normal_roundtrip():
    with V.pa_lifecycle():
        assert V._PA_LIFECYCLE_LOCK.locked()
    assert not V._PA_LIFECYCLE_LOCK.locked()
