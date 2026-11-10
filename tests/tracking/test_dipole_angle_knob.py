"""The differentiable dipole knob is the bend's angle at its radius, as everywhere else.

ADJUST on a dipole angle, the inspector, a scan and the Python API write the angle, and the arc |ρ|·|θ| follows (a bend
set to angle 0 has none, a pipe keeps the length it was given).  The torch arm, which the gradient matcher, the
differentiable lattice and the step tracker use, varied the field of a magnet of fixed length instead, so its residuals
and Jacobian were those of another knob, and the gradient matcher did not land where least_squares does.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.beam import Beam  # noqa: E402
from linac_gen.core.config import BeamConfig, SpaceChargeConfig  # noqa: E402
from linac_gen.core.lattice import Lattice  # noqa: E402
from linac_gen.core.particle import PROTON  # noqa: E402
from linac_gen.core.reference import ReferenceParticle  # noqa: E402
from linac_gen.elements.dipole import Dipole  # noqa: E402
from linac_gen.elements.drift import Drift  # noqa: E402
from linac_gen.elements.lattice_commands import Adjust, SetTwiss  # noqa: E402
from linac_gen.elements.quadrupole import Quadrupole  # noqa: E402
from linac_gen.matching import match  # noqa: E402
from linac_gen.tracking.autograd_api import DifferentiableLattice  # noqa: E402
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix  # noqa: E402
from linac_gen.tracking.torch_step_tracker import track_beam_torch_stepwise  # noqa: E402
from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch  # noqa: E402

F64 = torch.float64


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _line(**bend):
    lat = Lattice()
    lat.add(Drift("D1", 150.0))
    lat.add(Dipole("B", **dict(dict(angle=20.0, rho=1000.0), **bend)))
    lat.add(Quadrupole("Q", length=100.0, gradient=4.0))
    lat.add(Drift("D2", 150.0))
    return lat


def _pipe_line():
    lat = _line(angle=0.0)
    _bend(lat).length = 300.0
    return lat


def _bend(lat):
    return next(e for e in lat.elements if isinstance(e, Dipole))


def _edited(lat, angle):
    out = copy.deepcopy(lat)
    _bend(out).angle = angle
    return out


def _knob(lat, angle):
    return {id(_bend(lat)): torch.tensor(float(angle), dtype=F64)}


def _close(a, b, rtol=1e-12):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * float(np.abs(b).max()))


BENDS = [dict(), dict(field_rel=0.01), dict(field_rel=-1.0), dict(hv=1), dict(angle=-20.0),
         dict(e1=5.0, e2=-3.0, field_index=0.4)]
IDS = ["plain", "field error", "switched off", "vertical", "negative", "faces and index"]
# every branch of the body: elliptic, parabolic (the series, and just above it), hyperbolic, strong other-plane defocusing
BRANCHES = [dict(field_index=n, hv=hv) for n in (0.3, 1.0, 1.0 + 1e-9, 1.7, -2.0) for hv in (0, 1)]


@pytest.mark.parametrize("bend", BENDS, ids=IDS)
@pytest.mark.parametrize("delta", [-2.0, 0.5, 2.0])
def test_the_knob_is_an_edit_of_the_angle(bend, delta):
    lat = _line(**bend)
    angle = _bend(lat).angle + delta
    _close(compute_transfer_matrix_torch(lat, _ref(), overrides=_knob(lat, angle)).detach().numpy(),
           compute_transfer_matrix(_edited(lat, angle), _ref()))


@pytest.mark.parametrize("make, angle", [(_line, 0.0), (_pipe_line, 0.0), (_pipe_line, 3.0)],
                         ids=["bend to no angle", "pipe left a pipe", "pipe given an angle"])
def test_the_knob_follows_the_rule_at_no_angle(make, angle):
    """A bend set to angle 0 has no arc; a pipe keeps the length it was given and gets an arc with an angle."""
    lat = make()
    _close(compute_transfer_matrix_torch(lat, _ref(), overrides=_knob(lat, angle)).detach().numpy(),
           compute_transfer_matrix(_edited(lat, angle), _ref()))


@pytest.mark.parametrize("bend", BENDS + BRANCHES,
                         ids=IDS + [f"N={b['field_index']:.10g}, hv={b['hv']}" for b in BRANCHES])
def test_the_jacobian_of_the_knob_is_the_derivative_of_an_angle_edit(bend):
    """Also with the field off, where the bend is a pipe of its arc and the knob moves only its length."""
    lat = _line(**bend)
    a0 = _bend(lat).angle
    b_id = id(_bend(lat))

    def walk(a):
        return compute_transfer_matrix_torch(lat, _ref(), overrides={b_id: a})
    jac = torch.autograd.functional.jacobian(walk, torch.tensor(a0, dtype=F64)).numpy()
    h = 1e-4 * abs(a0)
    fd = (compute_transfer_matrix(_edited(lat, a0 + h), _ref())
          - compute_transfer_matrix(_edited(lat, a0 - h), _ref())) / (2.0 * h)
    _close(jac, fd, rtol=1e-7)


def test_a_number_as_the_knob_is_an_edit_of_the_angle():
    lat = _line(field_rel=0.01)
    for value in (22.0, np.float64(22.0)):
        _close(compute_transfer_matrix_torch(lat, _ref(), overrides={id(_bend(lat)): value}).numpy(),
               compute_transfer_matrix(_edited(lat, 22.0), _ref()))


def test_the_knobs_derivative_holds_down_to_a_tiny_angle():
    """Without a field index the vertical plane of a bend with faces is face, drift of the arc, face: its derivative
    is the same at any angle.  A ratio of tensors in the graph left round-off in it that grew as 1/angle (6 % at
    1e-12 degrees)."""
    lat = _line(e1=5.0, e2=5.0)
    b_id = id(_bend(lat))

    def d32(angle):
        t = torch.tensor(angle, dtype=F64, requires_grad=True)
        (g,) = torch.autograd.grad(compute_transfer_matrix_torch(lat, _ref(), overrides={b_id: t})[3, 2], t)
        return float(g)
    assert d32(10.0) != 0.0 and d32(1e-12) == pytest.approx(d32(10.0), rel=1e-9)


def test_the_differentiable_lattice_knob_is_the_angle():
    lat = _line()
    dl = DifferentiableLattice(lat, _ref())
    p = dl.set_tunables([("B", "angle")])[0]
    with torch.no_grad():
        p.tensor.fill_(22.0)
    _close(dl.transfer_matrix().detach().numpy(), compute_transfer_matrix(_edited(lat, 22.0), _ref()))


def _bcfg():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21, n_particles=10, distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=2.0, emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                      emit_z=0.3, alpha_z=0.0, beta_z=10.0)


def _bend_match(beta_x, field_rel=0.0):
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=20.0))
    lat.add(Adjust("CMD1", target="BEND", param_idx=1, link_group=0, vmin=5.0, vmax=40.0, start_step=0.5))
    lat.add(Dipole("BEND_001", angle=20.0, rho=1000.0, aperture=20.0, field_rel=field_rel))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=4.0, aperture=20.0))
    lat.add(Drift("D2", length=300.0, aperture=20.0))
    lat.add(SetTwiss("CT", family="END", beta_x=beta_x, kbx=1))
    return lat


@pytest.mark.parametrize("field_rel", [0.0, -1.0], ids=["bend", "switched off"])
def test_gradient_matching_of_a_bend_angle_lands_where_least_squares_does(field_rel):
    """The target is what an edit of the angle to 24 degrees gives; both algorithms find it.  The gradient one used to
    stop at 26.01 degrees (cost 6.5e-3), and with the field off it did not move from 20."""
    from linac_gen.cli.common import run_envelope_sim
    probe = _bend_match(1.0, field_rel)
    _bend(probe).angle = 24.0
    cfg = _bcfg()
    beta_end = float(run_envelope_sim(probe, cfg).beta_x[-1])
    r_lsq = match(_bend_match(beta_end, field_rel), cfg, algorithm="least_squares", max_iter=200)
    r_grad = match(_bend_match(beta_end, field_rel), cfg, algorithm="gradient", max_iter=200)
    assert r_lsq.x_final[0] == pytest.approx(24.0, abs=1e-3)
    assert r_grad.x_final[0] == pytest.approx(r_lsq.x_final[0], abs=1e-4)
    assert r_grad.cost == pytest.approx(r_lsq.cost, abs=1e-9)


def _bunch(n=200, current=12.0, seed=7):
    beam = Beam(ref=_ref(), n_particles=n, current=current)
    rng = np.random.default_rng(seed)
    beam.particles[:] = rng.normal(size=(n, 6)) * [1.0, 0.1, 1.0, 0.1, 5.0, 0.01]
    return beam


@pytest.mark.parametrize("start", [20.0, 0.0], ids=["bend", "bend with no angle"])
def test_the_step_tracker_takes_space_charge_over_the_knobs_arc(start):
    """With space charge the kicks are spread over the bend's length: the knob's own arc, also when the bend the knob
    starts from has none."""
    lat = _line(angle=start)
    beam = _bunch()
    mc = (beam.current * 1e-3) / (beam.bunch_frequency * 1e6) / beam.n_particles
    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0, use_gpu="cpu", grid_mode="adaptive")
    knob = track_beam_torch_stepwise(lat, _ref(), beam.particles.copy(), overrides=_knob(lat, 24.0), sc_cfg=cfg,
                                     macro_charge=mc)
    edit = track_beam_torch_stepwise(_edited(lat, 24.0), _ref(), beam.particles.copy(), sc_cfg=cfg, macro_charge=mc)
    _close(knob.detach().numpy(), edit.detach().numpy())


@pytest.mark.parametrize("field_rel", [0.0, -1.0], ids=["bend", "switched off"])
def test_the_step_trackers_gradient_is_the_derivative_of_an_angle_edit(field_rel):
    """Through the matrix and the space-charge kicks over the arc, checkpointed as a long line would be."""
    lat = _line(field_rel=field_rel)
    beam = _bunch()
    mc = (beam.current * 1e-3) / (beam.bunch_frequency * 1e6) / beam.n_particles
    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0, use_gpu="cpu", grid_mode="adaptive")
    weights = torch.as_tensor(np.random.default_rng(3).normal(size=beam.particles.shape), dtype=F64)

    def score(lattice, overrides=None, checkpoint=False):
        out = track_beam_torch_stepwise(lattice, _ref(), beam.particles.copy(), overrides=overrides, sc_cfg=cfg,
                                        macro_charge=mc, checkpoint=checkpoint)
        return (out * weights).sum()
    t = torch.tensor(20.0, dtype=F64, requires_grad=True)
    grads = [torch.autograd.grad(score(lat, {id(_bend(lat)): t}, ck), t)[0].item() for ck in (False, True)]
    h = 1e-4
    fd = (score(_edited(lat, 20.0 + h)).item() - score(_edited(lat, 20.0 - h)).item()) / (2.0 * h)
    assert grads[0] == grads[1]
    assert grads[0] == pytest.approx(fd, rel=1e-6)
