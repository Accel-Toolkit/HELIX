# tests/matching/test_engine_preview_hook.py
"""match() 4-arg callbacks receive info["results"] — the full per-eval
forward-sim result powering the GUI's live match preview."""
from __future__ import annotations

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching.engine import match


def _lat():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                   link_group=0, vmin=0.5, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0, y_mm=0.0, phi_or_z=0.0))
    return lat


def _cfg():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      n_particles=200, distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                      emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                      emit_z=0.3, alpha_z=0.0, beta_z=10.0, current=0.0)


def test_info_dict_carries_full_results_mp():
    seen = []

    def cb(it, x, cost, info=None):
        if info is not None:
            seen.append(info)

    match(_lat(), _cfg(), max_iter=2, algorithm="least_squares",
          cost_solver="mp", mp_n_particles=200, callback=cb)
    assert seen
    lr = seen[-1]["results"]
    # A DiagnosticRecorder: has centroid rows + exit-index map — exactly
    # what the popup fan-out consumes.
    assert lr is not None
    assert hasattr(lr, "centroid") and len(lr.centroid) > 0
    assert hasattr(lr, "element_exit_idx")
    # scalar summary keys unchanged alongside it
    assert "cost" in seen[-1] and "w_kin_out" in seen[-1]


def test_info_dict_carries_envelope_results_too():
    seen = []

    def cb(it, x, cost, info=None):
        if info is not None:
            seen.append(info)

    match(_lat(), _cfg(), max_iter=2, algorithm="least_squares",
          cost_solver="envelope", callback=cb)
    lr = seen[-1]["results"]
    assert lr is not None
    assert hasattr(lr, "sigma_x")            # EnvelopeResults
    # Envelope results now carry the first moment too — live previews
    # of envelope-cost matches show real orbits.
    assert hasattr(lr, "centroid")
    assert len(lr.centroid) == len(lr.s)


def test_three_arg_callback_still_supported():
    hits = []
    match(_lat(), _cfg(), max_iter=2, algorithm="least_squares",
          callback=lambda it, x, cost: hits.append(it))
    assert hits                               # legacy signature unharmed
