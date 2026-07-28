"""Differentiable (PyTorch autograd) lattice composition.

A parallel, opt-in mirror of ``linac_gen/tracking/matrix_tracking.py``:
composes per-element torch transfer matrices into a lattice matrix,
tracks a beam, and extracts Twiss parameters — all autograd-capable.

Only the 5 linear element types (Drift, Quadrupole, Solenoid, Dipole,
Edge) are differentiable.  Non-linear elements (RFGap, FieldMap,
Multipole, Foil, …) are treated as the identity — with a one-shot
warning (``on_nonlinear="identity"``, the default), silently
(``on_nonlinear="ignore"``), or raising (``on_nonlinear="error"``).

The reference kinematics are built once and reused: a lattice of only
linear elements does not change the beam energy, so β/γ/Bρ are constant.
"""
from __future__ import annotations

import math
import warnings

import torch

from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.lattice_commands import LatticeCommand
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.tracking.torch_matrices import (
    F64, RefKinematics, dipole_matrix, drift_matrix, edge_matrix,
    quad_matrix, solenoid_matrix,
)


def tilt_rotation_matrix_torch(tilt_deg: float) -> torch.Tensor:
    """6x6 rotation around the longitudinal axis (M8 misalignment).

    Mirrors `tracker.py:208-219` and the numpy
    :func:`linac_gen.elements.mixins.Misalignment.tilt_rotation_matrix`.
    Identity when ``tilt_deg == 0``.  Operates on
    ``(x, x', y, y')`` block; (phi, dW) unchanged.
    """
    if abs(float(tilt_deg)) < 1e-12:
        return torch.eye(6, dtype=F64)
    th = math.radians(float(tilt_deg))
    c, s = math.cos(th), math.sin(th)
    R = torch.eye(6, dtype=F64)
    R[0, 0] = c; R[0, 2] = s
    R[1, 1] = c; R[1, 3] = s
    R[2, 0] = -s; R[2, 2] = c
    R[3, 1] = -s; R[3, 3] = c
    return R

__all__ = [
    "element_matrix_torch", "compute_transfer_matrix_torch",
    "track_beam_torch", "compute_twiss_torch",
]

_warned_nonlinear: set[str] = set()


def _warn_nonlinear(cls_name: str) -> None:
    """Warn once per non-linear element type."""
    if cls_name not in _warned_nonlinear:
        _warned_nonlinear.add(cls_name)
        warnings.warn(
            f"torch_tracking: '{cls_name}' is not a differentiable linear "
            f"element — treated as the identity in the torch transfer-matrix "
            f"path.", stacklevel=3,
        )


def element_matrix_torch(element, kin: RefKinematics, *,
                         overrides: dict | None = None,
                         on_nonlinear: str = "identity") -> torch.Tensor:
    """6x6 torch transfer matrix for one element.

    ``overrides`` maps ``id(element)`` to a tensor that replaces the
    element's tunable parameter (quad gradient / solenoid field / dipole
    angle — the design value; the ``*_rel`` error is folded in here just
    as the numpy ``effective_*`` properties do).
    """
    overrides = overrides or {}
    ov = overrides.get(id(element))

    if isinstance(element, Drift):
        return drift_matrix(element.length, kin)

    if isinstance(element, Quadrupole):
        eff_g = (ov * (1.0 + element.gradient_rel) if ov is not None
                 else element.effective_gradient)
        return quad_matrix(element.length, eff_g, element.skew_angle, kin)

    if isinstance(element, Solenoid):
        eff_b = (ov * (1.0 + element.field_rel) if ov is not None
                 else element.effective_field)
        return solenoid_matrix(element.length, eff_b, kin)

    if isinstance(element, Dipole):
        eff_a = (ov * (1.0 + element.field_rel) if ov is not None
                 else element.effective_angle)
        return dipole_matrix(eff_a, element.rho, element.length,
                             element.e1, element.e2, element.field_index,
                             element.hv, kin)

    if isinstance(element, Edge):
        return edge_matrix(element.pole_rotation, element.rho,
                           element.gap, element.k1, element.hv)

    if isinstance(element, LatticeCommand):
        # SET / ADJUST directive cards carry no optics — identity.
        return torch.eye(6, dtype=F64)

    # M8 matcher arm: SurrogateFieldMap exposes an autograd-differentiable
    # MLP via `fitted_matrix_torch(kin_tensor)`.  Must come BEFORE the
    # FieldMap-class fall-through which would treat the surrogate as
    # non-linear and stub it with eye(6).
    #
    # OOD safety: if the surrogate's input is outside its trained scope,
    # we don't want the MLP to extrapolate (matcher gradient on
    # out-of-distribution data is meaningless).  The gradient path cannot
    # execute the wrapped nonlinear element either, so the only safe
    # behavior is a hard error: silently substituting an identity map
    # would let the optimizer converge on a lattice different from the
    # one the user specified.
    from linac_gen.surrogates.base import SurrogateFieldMap   # lazy
    if isinstance(element, SurrogateFieldMap):
        import numpy as _np
        # `RefKinematics` carries beta/gamma/mass but not w_kin -- the
        # surrogate's input convention is (w_kin, beta, gamma, params)
        # so reconstruct w_kin = (gamma - 1) * mass.
        w_kin = float((kin.gamma - 1.0) * kin.mass)
        params_np = [float(getattr(element._wrapped, p))
                     for p in element._param_names]
        x_np = _np.array(
            [w_kin, float(kin.beta), float(kin.gamma), *params_np],
            dtype=_np.float64,
        )
        if not element.metadata.scope.is_in_scope(x_np):
            raise ValueError(
                f"element_matrix_torch: surrogate '{element.name}' received "
                f"an input outside its trained scope "
                f"(w_kin={w_kin:.6g} MeV, params={params_np}). Refusing to "
                f"extrapolate or substitute an identity map: disengage the "
                f"surrogate for this element or use "
                f"algorithm='least_squares' with the native model."
            )
        kin_t = torch.as_tensor(
            [w_kin, float(kin.beta), float(kin.gamma)],
            dtype=F64,
        )
        M = element.fitted_matrix_torch(kin_t)
        # Output sanity gate — torch mirror of the numpy
        # fitted_matrix finiteness check (the two paths move in
        # pairs).  The numpy side falls back to the wrapped element;
        # this arm's contract is refuse-loudly (same as the scope
        # check above), so a corrupt weight file cannot feed NaN into
        # the matcher Jacobian silently.
        if not bool(torch.isfinite(M).all()):
            raise ValueError(
                f"element_matrix_torch: surrogate '{element.name}' "
                f"produced a NON-FINITE transfer matrix (weights may "
                f"be corrupt) — disengage the surrogate for this "
                f"element or use algorithm='least_squares' with the "
                f"native model."
            )
        # Singular-matrix mirror of the numpy fitted_matrix |det| gate
        # (no physical transfer map is singular; Liouville: |det| ~ 1
        # up to adiabatic damping).  detach(): a gate, not a gradient.
        det = float(torch.linalg.det(M).detach())
        if abs(det) < 1e-9:
            raise ValueError(
                f"element_matrix_torch: surrogate '{element.name}' "
                f"produced a SINGULAR transfer matrix (|det|="
                f"{abs(det):.3g}) — disengage the surrogate for this "
                f"element or use algorithm='least_squares' with the "
                f"native model."
            )
        return M

    # Non-linear / unsupported element.
    cls_name = type(element).__name__
    if on_nonlinear == "error":
        raise TypeError(
            f"element_matrix_torch: '{cls_name}' ('{element.name}') is not a "
            f"differentiable linear element"
        )
    if on_nonlinear != "ignore":
        _warn_nonlinear(cls_name)
    return torch.eye(6, dtype=F64)


def compute_transfer_matrix_torch(lattice, ref, start: int = 0,
                                  end: int | None = None, *,
                                  overrides: dict | None = None,
                                  on_nonlinear: str = "identity"
                                  ) -> torch.Tensor:
    """Product of per-element torch transfer matrices from ``start`` to
    ``end`` (inclusive) — the differentiable mirror of
    :func:`linac_gen.tracking.matrix_tracking.compute_transfer_matrix`.

    The reference particle is *not* advanced per element: a linear
    lattice does not change the beam energy, so β/γ/Bρ are constant.
    """
    kin = RefKinematics.from_reference(ref)
    last = len(lattice.elements) - 1 if end is None else end
    if last < start:
        raise ValueError(f"end ({last}) must be >= start ({start})")
    M = torch.eye(6, dtype=F64)
    for element in lattice.elements[start:last + 1]:
        M_elem = element_matrix_torch(element, kin, overrides=overrides,
                                      on_nonlinear=on_nonlinear)
        # M8 misalignment wrap: tilt_deg rotates (x, x', y, y') around
        # the longitudinal axis.  Entry rotation = R(+tilt), exit =
        # R(−tilt) — the MP tracker's convention (tracker.py:
        # x_new = c·x + s·y with θ = +tilt) and, since the envelope
        # tilt-sign fix, the envelope solver's too.  The previous
        # R(−tilt)/R(+tilt) pair modelled −tilt: invisible to symmetric
        # Σ, sign-flipping the coupled plane of the composed map.
        # dx / dy are no-ops in Sigma propagation (Sigma is centroid-
        # invariant); only tilt couples into the matrix.
        tilt = getattr(element, "tilt_deg", 0.0)
        if abs(float(tilt)) > 1e-12:
            R_in = tilt_rotation_matrix_torch(tilt)
            R_out = tilt_rotation_matrix_torch(-tilt)
            M_elem = R_out @ M_elem @ R_in
        M = M_elem @ M
    return M


def track_beam_torch(lattice, ref, particles, *,
                     overrides: dict | None = None,
                     on_nonlinear: str = "identity") -> torch.Tensor:
    """Track a beam through the lattice with the torch transfer matrix.

    ``particles`` is an ``(N, 6)`` array / tensor; returns an ``(N, 6)``
    float64 tensor.  Mirrors the numpy ``(M @ X.T).T`` application; no
    aperture / loss masking (out of scope for the differentiable path).
    """
    M = compute_transfer_matrix_torch(lattice, ref, overrides=overrides,
                                      on_nonlinear=on_nonlinear)
    X = particles if isinstance(particles, torch.Tensor) \
        else torch.as_tensor(particles, dtype=F64)
    X = X.to(dtype=F64)
    return (M @ X.T).T


def compute_twiss_torch(M: torch.Tensor, plane: str = "x",
                        coupling_tol: float = 1e-8) -> dict:
    """Extract decoupled Twiss parameters from a one-period 6x6 torch
    transfer matrix — the differentiable mirror of
    :func:`linac_gen.tracking.matrix_tracking.compute_twiss`.

    Returns alpha / beta / gamma_t / mu (degrees) as torch tensors.
    Raises on coupled or unstable optics, exactly as the numpy version.
    """
    if plane == "x":
        i, j = 0, 1
        off_planes = [(2, 3)]
    elif plane == "y":
        i, j = 2, 3
        off_planes = [(0, 1)]
    elif plane == "z":
        i, j = 4, 5
        off_planes = [(0, 1), (2, 3)]
    else:
        raise ValueError(f"plane must be 'x', 'y', or 'z', got '{plane}'")

    off_diag = torch.zeros((), dtype=F64)
    for (oi, oj) in off_planes:
        off_diag = off_diag + (torch.abs(M[i, oi]) + torch.abs(M[i, oj])
                               + torch.abs(M[j, oi]) + torch.abs(M[j, oj]))
    # Detach for the raise/branch gates — these are validity decisions,
    # not differentiated quantities (avoids a requires_grad->scalar warning).
    off_diag_val = float(off_diag.detach())
    if off_diag_val > coupling_tol:
        raise ValueError(
            f"Lattice is coupled to plane {plane} (off-plane |M_ij| sum = "
            f"{off_diag_val:.2e} > coupling_tol={coupling_tol:.0e}). "
            f"compute_twiss_torch handles decoupled optics only."
        )

    m11 = M[i, i]
    m12 = M[i, j]
    m21 = M[j, i]
    m22 = M[j, j]
    det = m11 * m22 - m12 * m21
    det_val = float(det.detach())
    if det_val <= 0.0:
        raise ValueError(
            f"{plane}-block determinant {det_val:.3e} <= 0 — not a "
            f"transport matrix"
        )
    if abs(det_val - 1.0) <= 1e-9:
        # Symplectic fast path — mirrors the numpy version bit-for-bit
        # (no √det rounding on magnetostatic lattices).
        sq = torch.ones((), dtype=F64)
        cos_mu = 0.5 * (m11 + m22)
    else:
        # Conformally symplectic (accelerating) block: λ = √det·e^{±iμ}.
        sq = torch.sqrt(det)
        cos_mu = 0.5 * (m11 + m22) / sq
    cos_mu_val = float(cos_mu.detach())
    if abs(cos_mu_val) >= 1.0:
        raise ValueError(f"Unstable: cos(mu) = {cos_mu_val}")
    mu = torch.acos(cos_mu)
    sin_mu = torch.sin(mu)
    if float(m12.detach()) < 0:
        mu = 2.0 * math.pi - mu
        sin_mu = torch.sin(mu)
    beta = m12 / (sq * sin_mu)
    alpha = (m11 - m22) / (2.0 * sq * sin_mu)
    gamma_t = (1.0 + alpha ** 2) / beta
    mu_deg = mu * (180.0 / math.pi)
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma_t": gamma_t,
        "mu": mu_deg,
        "mu_folded": mu_deg if float(mu_deg.detach()) <= 180.0
                     else 360.0 - mu_deg,
        "damping": sq,
    }
