"""M5 — gradient-based matching THROUGH non-linear PIC space charge.

The ``gradient`` algorithm with ``space_charge=True`` runs a Levenberg-
Marquardt solve whose residual is a macro-particle bunch tracked through
the differentiable PIC step tracker. It must recover a known quad gradient
from a wrong start.
"""
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching import match
from linac_gen.matching.constraints import collect_constraints
from linac_gen.matching.engine import _link_group_index
from linac_gen.matching.torch_objective import build_torch_residual_sc
from linac_gen.matching.variables import collect_variables

_TRUE_GRADIENT = 6.0

# The exact SC config match()'s gradient path builds internally — kept in
# sync so the target computed here matches what the matcher optimises.
_SC = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=4.0,
                        use_gpu="cpu", grid_mode="adaptive")


def _bcfg():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      n_particles=10, distribution="waterbag", current=5.0,
                      emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                      emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                      emit_z=0.3, alpha_z=0.0, beta_z=10.0)


def _fodo(quad_gradient, set_size_x_mm):
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0,
                    vmin=-30, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=quad_gradient,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=set_size_x_mm, y_mm=0.0,
                    phi_or_z=0.0))
    return lat


def _sigma_x_at(gradient):
    """End-of-lattice sigma_x at a quad gradient, with non-linear PIC SC —
    computed through the exact residual match() uses internally."""
    bcfg = _bcfg()
    lat = _fodo(quad_gradient=gradient, set_size_x_mm=1.0)
    ref = create_beam(bcfg, seed=42).ref
    variables = collect_variables(lat, bcfg)
    constraints = collect_constraints(lat)
    col, ncol = _link_group_index(variables)
    r = build_torch_residual_sc(lat, bcfg, ref, variables, constraints,
                                col, ncol, sc_cfg=_SC)
    # SetSize residual is sigma_x - x_mm, with x_mm = 1.0 here.
    return float(r(torch.tensor([gradient], dtype=torch.float64))[0]) + 1.0


def test_gradient_sc_recovers_known_quad():
    """Set the SET_SIZE target to sigma_x at a known quad gradient, start
    the quad wrong, and let gradient+SC matching recover the known value."""
    target = _sigma_x_at(_TRUE_GRADIENT)
    lat = _fodo(quad_gradient=9.0, set_size_x_mm=target)   # wrong start
    result = match(lat, _bcfg(), algorithm="gradient", space_charge=True,
                   max_iter=60)
    assert result.success, f"matching failed: cost={result.cost:.3e}"
    assert result.x_final[0] == pytest.approx(_TRUE_GRADIENT, abs=3e-3)
    assert result.cost < 1e-9
