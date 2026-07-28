"""grad_sensitivities: autograd ranking anchored against the NUMPY
envelope finite difference (external anchor — round-tripping torch
against itself would cancel symmetric errors)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.assist.tools import TOOLS, WorkContext
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole


def _fodo():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=8.0))
    lat.add(Drift("D2", length=400.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=-8.0))
    lat.add(Drift("D3", length=200.0))
    return lat


def _cfg():
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=3.0, frequency=352.21, current=0.0,
                      n_particles=1000,
                      alpha_x=0.0, beta_x=1.0, emit_nx=0.25,
                      alpha_y=0.0, beta_y=1.0, emit_ny=0.25,
                      alpha_z=0.0, beta_z=0.8, emit_z=0.3)


def _ctx():
    ctx = WorkContext()
    ctx.lattice = _fodo()
    ctx.beam_config = _cfg()
    return ctx


def test_ranked_sensitivities_match_numpy_envelope_fd():
    ctx = _ctx()
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y", top_n=5)
    assert res["status"] == "ok", res
    data = res["data"]
    assert data["n_knobs"] == 2
    by_name = {r["name"]: r for r in data["ranked"]}
    assert set(by_name) == {"QF", "QD"}

    # external anchor: central finite difference through the NUMPY
    # envelope solver (I=0 -> matrix-exact)
    from linac_gen.cli.common import build_ref
    from linac_gen.tracking.envelope import EnvelopeSolver
    from linac_gen.cli.common import _envelope_initial as _ei
    def _sigma_y(gname, dg):
        lat = copy.deepcopy(ctx.lattice)
        for e in lat.elements:
            if getattr(e, "name", "") == gname:
                e.gradient += dg
        ref = build_ref(ctx.beam_config)
        out = EnvelopeSolver(lat, ref, _ei(ctx.beam_config, ref),
                             current=0.0).run()
        return float(out.sigma_y[-1])

    for gname in ("QF", "QD"):
        h = 1e-4
        fd = (_sigma_y(gname, +h) - _sigma_y(gname, -h)) / (2 * h)
        got = by_name[gname]["sens"]
        assert abs(fd - got) < max(1e-3 * abs(fd), 1e-6), (gname, fd, got)


def test_refuses_nonlinear_decks():
    ctx = _ctx()

    class FieldMapish:                    # name carries the fragment
        pass
    FieldMapish.__name__ = "FieldMap"
    ctx.lattice.add(FieldMapish())
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y")
    assert res["status"] == "refused"
    assert "FieldMap" in res["data"]["message"]


def test_knob_after_probe_has_zero_sensitivity():
    ctx = _ctx()
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_x", at_index=3)
    assert res["status"] == "ok", res
    rows = {r["name"]: r for r in res["data"]["ranked"]}
    assert rows["QF"]["sens"] != 0.0
    assert rows.get("QD", {"sens": 0.0})["sens"] == 0.0


def test_registered_as_compute_tier():
    assert TOOLS["grad_sensitivities"].tier == "compute"
