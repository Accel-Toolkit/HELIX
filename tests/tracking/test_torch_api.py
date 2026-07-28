"""DifferentiableLattice API behaviour."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.autograd_api import DifferentiableLattice


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _fodo():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=8.0))
    lat.add(Drift("D2", length=400.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=-8.0))
    lat.add(Drift("D3", length=200.0))
    return lat


def test_set_tunables_returns_leaf_params():
    dl = DifferentiableLattice(_fodo(), _ref())
    params = dl.set_tunables([("QF", "gradient"), ("QD", "gradient")])
    assert len(params) == 2
    for p in params:
        assert p.tensor.requires_grad
        assert p.tensor.dtype == torch.float64
    assert params[0].tensor.item() == 8.0
    assert params[1].tensor.item() == -8.0


def test_does_not_mutate_elements():
    lat = _fodo()
    qf = next(e for e in lat.elements if e.name == "QF")
    qd = next(e for e in lat.elements if e.name == "QD")
    dl = DifferentiableLattice(lat, _ref())
    dl.set_tunables([("QF", "gradient"), ("QD", "gradient")])
    dl.transfer_matrix()
    dl.twiss("x")
    dl.track(np.zeros((2, 6)))
    assert qf.gradient == 8.0 and qd.gradient == -8.0


def test_repeated_backward():
    dl = DifferentiableLattice(_fodo(), _ref())
    p = dl.set_tunables([("QF", "gradient")])[0]
    grads = []
    for _ in range(3):
        p.tensor.grad = None
        dl.transfer_matrix()[0, 0].backward()
        grads.append(float(p.tensor.grad))
    assert grads[0] == grads[1] == grads[2]
    assert np.isfinite(grads[0])


def test_resolve_by_element_object():
    lat = _fodo()
    qf = next(e for e in lat.elements if e.name == "QF")
    dl = DifferentiableLattice(lat, _ref())
    params = dl.set_tunables([(qf, "gradient")])
    assert params[0].element is qf


def test_unknown_element_name_raises():
    dl = DifferentiableLattice(_fodo(), _ref())
    with pytest.raises(KeyError, match="no element named"):
        dl.set_tunables([("NOPE", "gradient")])


def test_non_tunable_element_raises():
    dl = DifferentiableLattice(_fodo(), _ref())
    with pytest.raises(TypeError, match="no differentiable parameter"):
        dl.set_tunables([("D1", "length")])


def test_wrong_attr_raises():
    dl = DifferentiableLattice(_fodo(), _ref())
    with pytest.raises(ValueError, match="tunable attr must be"):
        dl.set_tunables([("QF", "length")])
