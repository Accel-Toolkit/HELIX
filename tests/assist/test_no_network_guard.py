"""The offline guard itself: loopback allowed, everything else blocked.

Windows emulates ``socket.socketpair()`` (used by every asyncio event
loop) with a loopback TCP connect; the old blanket connect-block made
eight assist tests fail there on a false positive.  Loopback is local by
definition, so allowing it does not weaken the offline guarantee — the
guard still raises before any packet leaves for real destinations.
"""
from __future__ import annotations

import socket

import pytest


def test_loopback_connect_allowed():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cli:
            cli.settimeout(5.0)
            cli.connect(("127.0.0.1", port))           # must NOT raise


def test_socketpair_emulation_path_allowed():
    # the exact primitive Windows asyncio relies on
    a, b = socket.socketpair()
    a.close()
    b.close()


def test_external_connect_blocked():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(AssertionError, match="fully offline"):
            s.connect(("203.0.113.1", 80))             # TEST-NET-3: unroutable
