"""assist test fixtures — the ZERO-NETWORK guarantee.

Every test in this directory runs with the socket layer blocked: any
attempt to open a real connection fails the test.  The whole suite
therefore proves the assistant is testable — and the app unaffected —
with no API key and no network."""
from __future__ import annotations

import socket
import urllib.request

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError(
            "network access attempted inside tests/assist — the "
            "assistant suite must be fully offline")
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    yield


@pytest.fixture()
def ctx(tmp_path):
    """A WorkContext with a tiny real lattice + beam loaded."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.assist.tools import WorkContext

    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, aperture=20.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, aperture=20.0))
    lat.add(Drift("D2", 200.0))
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, duty_cycle=100.0, n_particles=200,
                     distribution="gaussian", cutoff=3.0,
                     emit_nx=0.25, alpha_x=0.0, beta_x=1.0,
                     emit_ny=0.25, alpha_y=0.0, beta_y=1.0,
                     emit_z=0.30, alpha_z=0.0, beta_z=5.0)
    c = WorkContext(calc_dir=str(tmp_path))
    c.set_lattice(lat, "<test-lattice>")
    c.set_beam_config(cfg)
    return c


@pytest.fixture()
def assist_config():
    from linac_gen.assist.config import AssistConfig
    return AssistConfig(provider="openai", model="test",
                        base_url="http://blocked.invalid/v1",
                        api_key="sk-TESTSECRET")
