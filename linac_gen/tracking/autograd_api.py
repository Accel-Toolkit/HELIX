"""User-facing API for differentiable matrix tracking.

``DifferentiableLattice`` wraps a HELIX lattice + reference particle and
exposes its transfer matrix, Twiss parameters, tracked beam and sigma
propagation as PyTorch tensors that are autograd-differentiable with
respect to chosen tunable element parameters (quad gradient, solenoid
field, dipole angle).

The underlying lattice and its elements are **never mutated** — tunables
are injected through an ``overrides`` dict keyed on ``id(element)``.

Example
-------
::

    dl = DifferentiableLattice(lattice, ref)
    params = dl.set_tunables([("QF", "gradient"), ("QD", "gradient")])
    beta_x = dl.twiss("x")["beta"]
    beta_x.backward()
    print(params[0].tensor.grad)        # d beta_x / d gradient_QF
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from linac_gen.elements.dipole import Dipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.tracking.torch_matrices import F64
from linac_gen.tracking.torch_tracking import (
    compute_transfer_matrix_torch, compute_twiss_torch,
)

__all__ = ["TunableParam", "DifferentiableLattice"]

# Each tunable element type has exactly one differentiable design parameter.
_TUNABLE_ATTR = {Quadrupole: "gradient", Solenoid: "field", Dipole: "angle"}


@dataclass
class TunableParam:
    """One differentiable knob: the lattice element, the attribute name,
    and the ``requires_grad`` leaf tensor standing in for it.  Read
    ``tensor.grad`` after a ``backward()`` call."""

    element: object
    attr: str
    tensor: torch.Tensor


class DifferentiableLattice:
    """Differentiable view of a HELIX lattice.

    ``set_tunables`` turns chosen element parameters into autograd
    leaves; ``transfer_matrix``, ``twiss``, ``track`` and ``sigma``
    return torch tensors differentiable with respect to those leaves.
    """

    def __init__(self, lattice, ref, *, on_nonlinear: str = "identity"):
        self.lattice = lattice
        self.ref = ref
        self._on_nonlinear = on_nonlinear
        self._tunables: list[TunableParam] = []
        self._overrides: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def _resolve(self, elem_or_name):
        if isinstance(elem_or_name, str):
            for e in self.lattice.elements:
                if getattr(e, "name", None) == elem_or_name:
                    return e
            raise KeyError(f"no element named '{elem_or_name}' in lattice")
        return elem_or_name

    def set_tunables(self, specs) -> list[TunableParam]:
        """Declare differentiable parameters.

        ``specs`` is an iterable of ``(element_or_name, attr)``.  Valid
        attrs: ``'gradient'`` (Quadrupole), ``'field'`` (Solenoid),
        ``'angle'`` (Dipole).  Returns the list of :class:`TunableParam`;
        replaces any previously-declared tunables.
        """
        self._tunables = []
        self._overrides = {}
        for elem_or_name, attr in specs:
            elem = self._resolve(elem_or_name)
            etype = type(elem)
            expected = _TUNABLE_ATTR.get(etype)
            if expected is None:
                raise TypeError(
                    f"{etype.__name__} '{getattr(elem, 'name', '?')}' has no "
                    f"differentiable parameter (tunable types: Quadrupole, "
                    f"Solenoid, Dipole)"
                )
            if attr != expected:
                raise ValueError(
                    f"{etype.__name__} tunable attr must be '{expected}', "
                    f"got '{attr}'"
                )
            value = float(getattr(elem, attr))
            t = torch.tensor(value, dtype=F64, requires_grad=True)
            self._overrides[id(elem)] = t
            self._tunables.append(
                TunableParam(element=elem, attr=attr, tensor=t)
            )
        return list(self._tunables)

    @property
    def tunables(self) -> list[TunableParam]:
        return list(self._tunables)

    # ------------------------------------------------------------------
    def transfer_matrix(self) -> torch.Tensor:
        """6x6 torch transfer matrix, differentiable w.r.t. the tunables.

        A fresh autograd graph is built on every call, so ``backward()``
        may be called repeatedly (remember to clear ``param.tensor.grad``
        between independent backward passes)."""
        return compute_transfer_matrix_torch(
            self.lattice, self.ref, overrides=self._overrides,
            on_nonlinear=self._on_nonlinear,
        )

    def twiss(self, plane: str = "x") -> dict:
        """Twiss dict (alpha / beta / gamma_t / mu as tensors) for ``plane``."""
        return compute_twiss_torch(self.transfer_matrix(), plane)

    def track(self, particles) -> torch.Tensor:
        """Track an ``(N, 6)`` beam; returns an ``(N, 6)`` differentiable
        tensor (``M @ X.T).T``)."""
        M = self.transfer_matrix()
        X = (particles if isinstance(particles, torch.Tensor)
             else torch.as_tensor(particles, dtype=F64))
        return (M @ X.to(dtype=F64).T).T

    def sigma(self, sigma0) -> torch.Tensor:
        """Propagate a 6x6 sigma matrix: ``M @ sigma0 @ M.T``."""
        M = self.transfer_matrix()
        S0 = (sigma0 if isinstance(sigma0, torch.Tensor)
              else torch.as_tensor(sigma0, dtype=F64))
        return M @ S0.to(dtype=F64) @ M.T
