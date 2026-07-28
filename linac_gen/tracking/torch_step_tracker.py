"""Step-by-step differentiable (PyTorch autograd) particle tracker.

The existing differentiable path
(:func:`linac_gen.tracking.torch_tracking.compute_transfer_matrix_torch`)
pre-composes every element matrix into one 6x6 and applies it once. That is
incompatible with space charge: the SC kick depends on the *evolving*
distribution, so the beam must be advanced element by element with the kicks
interleaved.

:func:`track_beam_torch_stepwise` walks the lattice, applies each element's
torch transfer matrix to the beam tensor and — when a ``SpaceChargeConfig``
is supplied — interleaves the differentiable PIC space-charge kick with a
2nd-order Strang split. The whole forward pass is one autograd graph, so a
scalar formed from the tracked beam is differentiable with respect to
tunable element parameters.

Scope: the five linear element types (drift / quad / solenoid / dipole /
edge), exactly as :mod:`linac_gen.tracking.torch_tracking`. The
end-to-end-with-SC path is correct for linear lattices (e.g. the PIP-II BTL
transfer line); accelerating elements (RF gaps, field maps) are out of
scope and, with ``on_nonlinear="error"``, raise.
"""
from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.pic.torch.sc_kick import torch_pic_sc_kick
from linac_gen.tracking.autograd_api import TunableParam
from linac_gen.tracking.torch_matrices import (
    F64, RefKinematics, drift_matrix, quad_matrix, solenoid_matrix,
)
from linac_gen.tracking.torch_tracking import element_matrix_torch

__all__ = ["track_beam_torch_stepwise", "DifferentiableStepTracker"]

# Each tunable element type's single differentiable design parameter.
_TUNABLE_ATTR = {Quadrupole: "gradient", Solenoid: "field", Dipole: "angle"}


def _submatrix(element, kin, length_mm, overrides):
    """Transfer matrix of a length-composing element at a sub-length.

    Mirrors ``element_matrix_torch`` for Drift / Quadrupole / Solenoid but
    with a caller-chosen length — their matrices satisfy ``M(L) = M(L/n)^n``
    exactly, so the SC kick can be interleaved by splitting the *map*.
    Returns ``None`` for any other element type.
    """
    overrides = overrides or {}
    ov = overrides.get(id(element))
    if isinstance(element, Drift):
        return drift_matrix(length_mm, kin)
    if isinstance(element, Quadrupole):
        eff_g = (ov * (1.0 + element.gradient_rel) if ov is not None
                 else element.effective_gradient)
        return quad_matrix(length_mm, eff_g, element.skew_angle, kin)
    if isinstance(element, Solenoid):
        eff_b = (ov * (1.0 + element.field_rel) if ov is not None
                 else element.effective_field)
        return solenoid_matrix(length_mm, eff_b, kin)
    return None


def _apply(matrix: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Apply a 6x6 transfer matrix to an (N, 6) beam tensor."""
    return (matrix @ X.T).T


def _sc(X, sc_cfg, ds_mm, kin, macro_charge, dtype, device):
    return torch_pic_sc_kick(
        X, sc_cfg, ds_mm=ds_mm,
        gamma=kin.gamma, beta=kin.beta, mass_mev=kin.mass,
        wavelength_mm=kin.wavelength, macro_charge=macro_charge,
        charge_state=kin.charge,
        dtype=dtype, device=device)


def _advance_one_element(X, element, kin, overrides, sc_cfg, macro_charge,
                         step_config, on_nonlinear, dtype, device):
    """Advance the ``(N, 6)`` beam tensor ``X`` through one lattice element.

    With ``sc_cfg`` given and a length>0 element the SC kicks are
    interleaved with the numpy ``Tracker``'s cadence (drift -> step1/step2
    Strang bundles; quad/solenoid -> two sub-steps; dipole -> kick-drift-
    kick).  This is the unit a gradient-checkpoint segment wraps.
    """
    length = float(getattr(element, "length", 0.0) or 0.0)

    if (sc_cfg is None) or length <= 0.0:
        M = element_matrix_torch(element, kin, overrides=overrides,
                                 on_nonlinear=on_nonlinear)
        return _apply(M, X)

    if isinstance(element, Drift):
        # _track_drift: the step1/step2 grid -> Strang bundles.
        if step_config is not None:
            n_int = step_config.integration_steps_for_length_mm(length)
            n_sc = step_config.sc_steps_for_length_mm(length)
        else:
            n_int = n_sc = 2
        ds = length / n_int
        sc_every = max(1, n_int // n_sc)
        n_bundles = n_int // sc_every
        bundle_len = sc_every * ds
        half = drift_matrix(0.5 * bundle_len, kin)
        for _ in range(n_bundles):
            X = _apply(half, X)
            X = _sc(X, sc_cfg, bundle_len, kin, macro_charge, dtype, device)
            X = _apply(half, X)
        remainder = length - n_bundles * bundle_len
        if remainder > 1e-12:
            X = _apply(drift_matrix(remainder, kin), X)
        return X

    if isinstance(element, (Quadrupole, Solenoid)):
        # _track_transfer_map: two sub-steps, kick at each midpoint.
        m_quarter = _submatrix(element, kin, 0.25 * length, overrides)
        for _ in range(2):
            X = _apply(m_quarter, X)
            X = _sc(X, sc_cfg, 0.5 * length, kin, macro_charge, dtype, device)
            X = _apply(m_quarter, X)
        return X

    # Dipole / other length>0 element: kick-drift-kick Strang split —
    # the same total SC as the numpy 2-sub-step cadence.
    M = element_matrix_torch(element, kin, overrides=overrides,
                             on_nonlinear=on_nonlinear)
    X = _sc(X, sc_cfg, 0.5 * length, kin, macro_charge, dtype, device)
    X = _apply(M, X)
    X = _sc(X, sc_cfg, 0.5 * length, kin, macro_charge, dtype, device)
    return X


def track_beam_torch_stepwise(lattice, ref, particles, *,
                              overrides=None, sc_cfg=None, macro_charge=None,
                              on_nonlinear: str = "identity",
                              checkpoint: bool = False,
                              dtype: torch.dtype = F64,
                              device=None) -> torch.Tensor:
    """Track an ``(N, 6)`` beam element-by-element — autograd-differentiable.

    With ``sc_cfg=None`` this is the step-by-step equivalent of
    ``track_beam_torch`` (no space charge — for linear elements the result
    is identical to the pre-composed matrix up to round-off). With a
    ``SpaceChargeConfig`` it interleaves the differentiable PIC space-charge
    kick; ``macro_charge`` (charge per macro-particle, C) is then required.

    SC cadence — replicates the numpy ``Tracker``: a drift follows the
    step1/step2 grid (``lattice.step_config``) as ``n_sc`` Strang bundles
    of [drift, SC, drift] plus a no-SC remainder; a quad / solenoid is two
    sub-steps with a kick at each midpoint; a dipole uses a kick-drift-kick
    split (also 2nd-order Strang), so its edge focusing need not be
    factored out of the matrix.

    ``checkpoint`` — when True, each element is wrapped in
    :func:`torch.utils.checkpoint.checkpoint`: the forward pass stores only
    the small ``(N, 6)`` per-element input and recomputes the space-charge
    grids in the backward pass.  Peak autograd memory drops from
    ``O(n_kicks * grid^3)`` to ``O(grid^3)`` at ~2x forward cost — needed
    to backprop through a long lattice (the ~960-element BTL) without
    running out of memory.  The SC kick has no internal RNG (the bunch is
    sampled once, before tracking), so the recomputation is exact.
    """
    kin = RefKinematics.from_reference(ref)
    X = (particles if isinstance(particles, torch.Tensor)
         else torch.as_tensor(particles, dtype=dtype))
    X = X.to(dtype=dtype)

    sc_on = sc_cfg is not None
    if sc_on and macro_charge is None:
        raise ValueError("macro_charge is required when sc_cfg is given")

    step_config = getattr(lattice, "step_config", None)
    # Checkpoint only when a backward will actually happen — i.e. the beam
    # or some tunable override carries a gradient.
    needs_grad = X.requires_grad or any(
        t.requires_grad for t in (overrides or {}).values())
    use_ckpt = checkpoint and needs_grad

    for element in lattice.elements:
        if use_ckpt:
            segment = partial(
                _advance_one_element, element=element, kin=kin,
                overrides=overrides, sc_cfg=sc_cfg, macro_charge=macro_charge,
                step_config=step_config, on_nonlinear=on_nonlinear,
                dtype=dtype, device=device)
            X = _torch_checkpoint(segment, X, use_reentrant=False)
        else:
            X = _advance_one_element(
                X, element, kin, overrides, sc_cfg, macro_charge,
                step_config, on_nonlinear, dtype, device)
    return X


class DifferentiableStepTracker:
    """Differentiable step-by-step tracker with optional space charge.

    The lattice and its elements are never mutated — tunable parameters are
    injected through an ``overrides`` dict keyed on ``id(element)``, exactly
    as :class:`linac_gen.tracking.autograd_api.DifferentiableLattice`.
    """

    def __init__(self, lattice, ref, *, on_nonlinear: str = "identity"):
        self.lattice = lattice
        self.ref = ref
        self._on_nonlinear = on_nonlinear
        self._tunables: list[TunableParam] = []
        self._overrides: dict[int, torch.Tensor] = {}

    def _resolve(self, elem_or_name):
        if isinstance(elem_or_name, str):
            for e in self.lattice.elements:
                if getattr(e, "name", None) == elem_or_name:
                    return e
            raise KeyError(f"no element named '{elem_or_name}' in lattice")
        return elem_or_name

    def set_tunables(self, specs) -> list[TunableParam]:
        """Declare differentiable parameters — ``(element_or_name, attr)``
        pairs. Valid attrs: ``gradient`` (Quadrupole), ``field`` (Solenoid),
        ``angle`` (Dipole). Returns the :class:`TunableParam` list."""
        self._tunables = []
        self._overrides = {}
        for elem_or_name, attr in specs:
            elem = self._resolve(elem_or_name)
            expected = _TUNABLE_ATTR.get(type(elem))
            if expected is None:
                raise TypeError(
                    f"{type(elem).__name__} has no differentiable parameter "
                    f"(tunable types: Quadrupole, Solenoid, Dipole)")
            if attr != expected:
                raise ValueError(
                    f"{type(elem).__name__} tunable attr must be "
                    f"'{expected}', got '{attr}'")
            t = torch.tensor(float(getattr(elem, attr)), dtype=F64,
                             requires_grad=True)
            self._overrides[id(elem)] = t
            self._tunables.append(TunableParam(element=elem, attr=attr,
                                               tensor=t))
        return list(self._tunables)

    @property
    def tunables(self) -> list[TunableParam]:
        return list(self._tunables)

    def track(self, particles, *, sc_cfg=None, macro_charge=None,
              checkpoint: bool = False) -> torch.Tensor:
        """Track an ``(N, 6)`` beam; returns a differentiable ``(N, 6)``
        tensor. Pass ``sc_cfg`` (+ ``macro_charge``) for space charge, and
        ``checkpoint=True`` to gradient-checkpoint a long lattice."""
        return track_beam_torch_stepwise(
            self.lattice, self.ref, particles,
            overrides=self._overrides, sc_cfg=sc_cfg,
            macro_charge=macro_charge, on_nonlinear=self._on_nonlinear,
            checkpoint=checkpoint)
