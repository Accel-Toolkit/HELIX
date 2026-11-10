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

    class FieldMapish:                    # a class it does not know, named like a field map
        pass
    FieldMapish.__name__ = "FieldMap"
    ctx.lattice.add(FieldMapish())
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y")
    assert res["status"] == "refused"
    assert "FieldMap" in res["data"]["message"]


@pytest.mark.parametrize("card", ["same", "other", "energy"])
def test_a_card_that_sets_the_reference_is_followed(card):
    """It refused every deck with a FREQ card, even the header one that changes
    nothing.  Anchored on the numpy envelope, as above."""
    from linac_gen.cli.common import _envelope_initial as _ei
    from linac_gen.cli.common import build_ref
    from linac_gen.elements.lattice_commands import Freq, SetReferenceEnergy
    from linac_gen.tracking.envelope import EnvelopeSolver

    make = {"same": lambda: Freq("F", frequency_mhz=352.21),
            "other": lambda: Freq("F", frequency_mhz=704.42),
            "energy": lambda: SetReferenceEnergy("E", energy_MeV=4.0)}[card]

    def line(dg=0.0):
        lat = Lattice()
        lat.add(Drift("D1", length=200.0))
        lat.add(make())
        lat.add(Quadrupole("QF", length=100.0, gradient=8.0 + dg))
        lat.add(Drift("D2", length=400.0))
        return lat
    ctx = _ctx()
    ctx.lattice = line()
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y")
    assert res["status"] == "ok", res
    got = {r["name"]: r for r in res["data"]["ranked"]}["QF"]["sens"]

    def sigma_y(dg):
        ref = build_ref(ctx.beam_config)
        return float(EnvelopeSolver(line(dg), ref, _ei(ctx.beam_config, ref), current=0.0).run().sigma_y[-1])
    h = 1e-4
    fd = (sigma_y(h) - sigma_y(-h)) / (2 * h)
    assert abs(fd - got) < max(1e-3 * abs(fd), 1e-6), (fd, got)


def test_knob_after_probe_has_zero_sensitivity():
    ctx = _ctx()
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_x", at_index=2)   # after D2, before QD
    assert res["status"] == "ok", res
    rows = {r["name"]: r for r in res["data"]["ranked"]}
    assert rows["QF"]["sens"] != 0.0
    assert rows.get("QD", {"sens": 0.0})["sens"] == 0.0


def test_registered_as_compute_tier():
    assert TOOLS["grad_sensitivities"].tier == "compute"


@pytest.mark.parametrize("kind", ["energy", "shift"],
                         ids=["a set to zero energy", "a shift down to zero"])
def test_a_card_that_leaves_no_energy_is_refused_not_nan(kind):
    from linac_gen.cli.common import build_ref
    from linac_gen.elements.lattice_commands import SetReferenceEnergy
    ctx = _ctx()
    w = build_ref(ctx.beam_config).w_kin
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(SetReferenceEnergy("E", energy_MeV=0.0) if kind == "energy"
            else SetReferenceEnergy("E", energy_MeV=-w, shift=True))
    lat.add(Quadrupole("QF", length=100.0, gradient=8.0))
    ctx.lattice = lat
    res = TOOLS["grad_sensitivities"].fn(ctx, kpi="sigma_y")
    assert res["status"] == "refused", res
    assert "no energy" in str(res)
