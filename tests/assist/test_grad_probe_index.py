"""grad_sensitivities(at_index=k) probes after element k (0-based, the element included), as its schema says.

The beam was taken after element k, but the knobs and the reported probe stopped one element short: a quadrupole at
the probe had no sensitivity listed although it moved the answer, the probe was reported as k - 1, at_index 0 (after
the first element) was refused and one past the end accepted.  Anchored on the numpy envelope of the line cut after
element k, as the file of the end-of-line tests is.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from linac_gen.assist.tools import TOOLS, WorkContext  # noqa: E402
from linac_gen.cli.common import _envelope_initial, build_ref  # noqa: E402
from linac_gen.core.config import BeamConfig  # noqa: E402
from linac_gen.core.lattice import Lattice  # noqa: E402
from linac_gen.elements.drift import Drift  # noqa: E402
from linac_gen.elements.lattice_commands import Freq  # noqa: E402
from linac_gen.elements.quadrupole import Quadrupole  # noqa: E402
from linac_gen.tracking.envelope import EnvelopeSolver  # noqa: E402


def _line():
    lat = Lattice()
    for e in (Drift("D1", length=200.0), Quadrupole("QF", length=100.0, gradient=8.0), Freq("F", frequency_mhz=704.42),
              Drift("D2", length=400.0), Quadrupole("QD", length=100.0, gradient=-8.0), Drift("D3", length=200.0)):
        lat.add(e)
    return lat


def _cfg():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21, current=0.0, n_particles=1000,
                      alpha_x=0.0, beta_x=1.0, emit_nx=0.25, alpha_y=0.0, beta_y=1.0, emit_ny=0.25,
                      alpha_z=0.0, beta_z=0.8, emit_z=0.3)


def _ctx():
    ctx = WorkContext()
    ctx.lattice = _line()
    ctx.beam_config = _cfg()
    return ctx


def _sigma_after(k, kpi, name=None, dg=0.0):
    """The numpy envelope's sigma at the exit of element k (the line cut after it)."""
    lat = Lattice()
    for e in copy.deepcopy(_line().elements[:k + 1]):
        if name is not None and getattr(e, "name", "") == name:
            e.gradient += dg
        lat.add(e)
    ref = build_ref(_cfg())
    return float(getattr(EnvelopeSolver(lat, ref, _envelope_initial(_cfg(), ref), current=0.0).run(), kpi)[-1])


@pytest.mark.parametrize("k", range(6))
@pytest.mark.parametrize("kpi", ["sigma_x", "sigma_phi"])
def test_the_probe_sits_after_element_k_and_counts_its_knob(k, kpi):
    res = TOOLS["grad_sensitivities"].fn(_ctx(), kpi=kpi, at_index=k)
    if k == 0:                                        # a drift: no knob up to it
        assert res["status"] == "refused" and "no tunable" in res["data"]["message"]
        return
    assert res["status"] == "ok", res
    data = res["data"]
    assert data["probe_after_index"] == k
    assert data["kpi_value"] == pytest.approx(_sigma_after(k, kpi), abs=1e-6)       # the tool rounds to 6 digits
    knobs = {r["name"]: r["sens"] for r in data["ranked"]}
    assert set(knobs) == {e.name for e in _line().elements[:k + 1] if isinstance(e, Quadrupole)}
    for name, got in knobs.items():
        h = 1e-4
        fd = (_sigma_after(k, kpi, name, h) - _sigma_after(k, kpi, name, -h)) / (2.0 * h)
        assert got == pytest.approx(fd, rel=1e-6, abs=1e-9), (name, got, fd)


def test_the_end_of_the_line_is_the_last_element():
    ctx = _ctx()
    full = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y")["data"]
    last = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y", at_index=5)["data"]
    assert full["probe_after_index"] == 5 and full == last


@pytest.mark.parametrize("k", [-1, 6, 60])
def test_an_index_outside_the_line_is_refused(k):
    res = TOOLS["grad_sensitivities"].fn(_ctx(), kpi="sigma_y", at_index=k)
    assert res["status"] == "refused" and "0..5" in res["data"]["message"]


def test_an_empty_lattice_is_refused_for_what_it_is():
    ctx = _ctx()
    ctx.lattice = Lattice()
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y")
    assert res["status"] == "refused" and res["data"]["message"] == "the lattice has no elements"
