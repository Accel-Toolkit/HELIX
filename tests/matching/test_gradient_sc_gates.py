"""Gradient+SC path gates (claims 7 & 8, 2026-07-25 review).

* Identity-only passives (markers/apertures) must NOT abort the SC
  residual after the audit approved them (late abort, claim 7).
* Misaligned elements must be REFUSED at build time — the stepwise
  tracker would silently drop the roll (claim 8).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching.torch_objective import (
    build_torch_residual_sc, check_gradient_supported)


def _cfg():
    return BeamConfig(species="H-", energy=2.0, frequency=162.5,
                      current=5.0, n_particles=200,
                      emit_nx=0.2, alpha_x=0.0, beta_x=0.5,
                      emit_ny=0.2, alpha_y=0.0, beta_y=0.5,
                      emit_z=0.06, alpha_z=0.0, beta_z=500.0)


def _ref():
    return ReferenceParticle(species=H_MINUS, w_kin=2.0, frequency=162.5)


def _lat(extra=None):
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=20.0))
    lat.add(Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0))
    for e in (extra or []):
        lat.add(e)
    lat.add(Drift(name="D2", length=100.0, aperture=20.0))
    return lat


_SC = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0,
                        use_gpu="cpu", grid_mode="adaptive")


def test_identity_passives_do_not_abort_sc_residual():
    """Marker + Aperture pass the audit AND survive the residual — the
    pre-fix on_nonlinear='error' raised at the first evaluation, after
    the optimiser had already started."""
    lat = _lat([Marker("M1", snapshot=False),
                Aperture(name="A1", a=15.0, b=15.0)])
    ref = _ref()
    check_gradient_supported(lat, ref, variables=[], constraints=[])
    r = build_torch_residual_sc(lat, _cfg(), ref, variables=[],
                                constraints=[], col_for_var=[], n_cols=0,
                                sc_cfg=_SC, bunch_size=64, seed=1)
    out = r(torch.zeros(0, dtype=torch.float64))
    assert bool(torch.isfinite(out).all())


def test_misaligned_element_refused_at_build_time():
    q_rolled = Quadrupole(name="QR", length=100.0, gradient=5.0,
                          aperture=20.0, tilt_deg=1.0)
    lat = _lat([q_rolled])
    with pytest.raises(ValueError, match="misalign"):
        build_torch_residual_sc(lat, _cfg(), _ref(), variables=[],
                                constraints=[], col_for_var=[], n_cols=0,
                                sc_cfg=_SC, bunch_size=64, seed=1)


def test_pareto_refuses_mp_only_objective_under_envelope():
    """Claim 9: transmission_loss has no meaning in envelope results —
    the pairing must raise instead of producing a constant column."""
    from linac_gen.matching.multiobjective import pareto_optimize
    with pytest.raises(ValueError, match="cost_solver='mp'"):
        pareto_optimize(_lat(), _cfg(),
                        objective_names=["emit_nx_growth",
                                         "transmission_loss"],
                        cost_solver="envelope")
