"""Tests for the cross-screen geometry clamp (pure rect math).

When the window starts on — or is dragged to — a screen smaller than
its current geometry (laptop ↔ monitor switching), clamp_rect shrinks
and translates it fully on-screen; otherwise it returns the frame
unchanged so callers can no-op.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QRect  # noqa: E402

from linac_gen_gui.interphase.app import clamp_rect  # noqa: E402

AVAIL = QRect(0, 25, 1312, 823)   # scaled MacBook screen minus menu bar


def test_already_inside_is_unchanged():
    frame = QRect(100, 100, 1000, 700)
    out = clamp_rect(frame, AVAIL)
    assert out == frame


def test_oversized_shrinks_to_available():
    frame = QRect(-200, -100, 2560, 1400)   # monitor-sized geometry
    out = clamp_rect(frame, AVAIL)
    assert out.width() == AVAIL.width()
    assert out.height() == AVAIL.height()
    assert AVAIL.contains(out)


def test_off_left_and_above_is_translated_in():
    frame = QRect(-500, -300, 900, 600)
    out = clamp_rect(frame, AVAIL)
    assert out.size() == frame.size()       # no shrink needed
    assert out.left() == AVAIL.left()
    assert out.top() == AVAIL.top()
    assert AVAIL.contains(out)


def test_straddling_bottom_right_is_pulled_back():
    frame = QRect(1000, 700, 900, 600)      # hangs off bottom-right
    out = clamp_rect(frame, AVAIL)
    assert out.size() == frame.size()
    assert out.right() == AVAIL.right()
    assert out.bottom() == AVAIL.bottom()
    assert AVAIL.contains(out)


def test_idempotent():
    frame = QRect(-200, -100, 2560, 1400)
    once = clamp_rect(frame, AVAIL)
    twice = clamp_rect(once, AVAIL)
    assert once == twice
