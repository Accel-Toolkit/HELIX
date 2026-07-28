"""Differentiable (PyTorch autograd) transfer matrices for HELIX's linear
elements.

A parallel, opt-in reimplementation of the 6x6 transfer matrices built by
the numpy ``transfer_matrix()`` methods on Drift, Quadrupole, Solenoid,
Dipole and Edge.  Each builder returns a ``(6, 6)`` ``torch.float64``
tensor; when a tunable argument (quad gradient, solenoid field, dipole
angle) is passed as a ``requires_grad`` tensor the whole matrix is
autograd-differentiable with respect to it.

The numpy path (``linac_gen/tracking/matrix_tracking.py`` and the element
classes) is untouched — this module only ever reads element attributes.

Coordinates: ``[x mm, x' mrad, y mm, y' mrad, dphi deg, dW MeV]``.
All tensors are float64 on CPU — fp64 is required to reproduce the numpy
path to ~1e-10 (Apple MPS has no fp64).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "RefKinematics",
    "drift_matrix", "quad_matrix", "solenoid_matrix",
    "dipole_matrix", "edge_matrix",
]

F64 = torch.float64
_DEG2RAD = math.pi / 180.0


@dataclass(frozen=True)
class RefKinematics:
    """Constant reference-particle kinematics for a non-accelerating
    linear lattice.

    Mirrors the fields the numpy matrices read off ``ReferenceParticle``;
    constant because the linear elements (drift / quad / solenoid /
    dipole / edge) do not change the beam energy.
    """

    beta: float
    gamma: float
    brho: float
    mass: float
    charge: int
    wavelength: float

    @classmethod
    def from_reference(cls, ref) -> "RefKinematics":
        return cls(
            beta=float(ref.beta),
            gamma=float(ref.gamma),
            brho=float(ref.brho),
            mass=float(ref.species.mass),
            charge=int(ref.species.charge),
            wavelength=float(ref.wavelength),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _t(x) -> torch.Tensor:
    """Coerce a scalar (python number or tensor) to a 0-dim float64 tensor.

    A grad-tracked tensor passes through with its graph intact; a python
    number becomes a constant.  This is the numpy<->torch boundary.
    """
    if isinstance(x, torch.Tensor):
        return x.to(dtype=F64)
    return torch.as_tensor(float(x), dtype=F64)


def _assemble(entries: dict) -> torch.Tensor:
    """Build a 6x6 float64 tensor from an ``{(i, j): value}`` dict.

    Unspecified entries default to the identity (1 on the diagonal, 0
    elsewhere).  Autograd-safe: every entry is stacked, never assigned
    in place into a grad-tracked tensor.
    """
    one = torch.ones((), dtype=F64)
    zero = torch.zeros((), dtype=F64)
    rows = []
    for i in range(6):
        row = [_t(entries.get((i, j), one if i == j else zero))
               for j in range(6)]
        rows.append(torch.stack(row))
    return torch.stack(rows)


def _phase_slip(length_mm, kin: RefKinematics) -> torch.Tensor:
    """Longitudinal M[4,5] phase slip — identical formula for drift, quad,
    solenoid and the dipole body (see drift.py for the derivation)."""
    denom = kin.beta ** 3 * kin.gamma ** 3 * kin.mass * kin.wavelength
    return -360.0 * _t(length_mm) / denom


# --- entire-function helpers (keep the quad matrix differentiable at 0) ----
# cos(√u) and sin(√u)/√u, analytically continued to all real u.  The naive
# numpy form sin(kL)/k with k = √|k1| has a *singular autograd gradient* at
# k1 = 0 (a quad at zero gradient); these branch-free forms do not.  They
# are computed by the closed form away from 0 and a short Taylor series
# near it — numerically identical to cos/cosh & sin/sinh for |u| ≥ 1e-4.
_SERIES_EPS = 1e-4


def _cosc(u: torch.Tensor) -> torch.Tensor:
    """cos(√u) for u≥0, cosh(√−u) for u<0, 1 at u=0 — autograd-safe at 0."""
    au = torch.abs(u)
    s = torch.sqrt(torch.clamp(au, min=1e-30))
    closed = torch.where(u >= 0.0, torch.cos(s), torch.cosh(s))
    taylor = 1.0 - u / 2.0 + u * u / 24.0 - u * u * u / 720.0
    return torch.where(au < _SERIES_EPS, taylor, closed)


def _sincs(u: torch.Tensor) -> torch.Tensor:
    """sin(√u)/√u for u>0, sinh(√−u)/√−u for u<0, 1 at u=0 — autograd-safe."""
    au = torch.abs(u)
    s = torch.sqrt(torch.clamp(au, min=1e-30))
    closed = torch.where(u >= 0.0, torch.sin(s) / s, torch.sinh(s) / s)
    taylor = 1.0 - u / 6.0 + u * u / 120.0 - u * u * u / 5040.0
    return torch.where(au < _SERIES_EPS, taylor, closed)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
def drift_matrix(length_mm: float, kin: RefKinematics) -> torch.Tensor:
    """6x6 transfer matrix of a field-free drift (mirrors Drift.transfer_matrix)."""
    L_m = length_mm * 1e-3
    return _assemble({
        (0, 1): L_m,
        (2, 3): L_m,
        (4, 5): _phase_slip(length_mm, kin),
    })


# ---------------------------------------------------------------------------
# Quadrupole
# ---------------------------------------------------------------------------
def _transverse_rotation(theta_rad: float) -> torch.Tensor:
    """6x6 rotation about s by theta_rad — mirrors quadrupole._transverse_rotation."""
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return _assemble({
        (0, 0): c, (0, 2): -s,
        (1, 1): c, (1, 3): -s,
        (2, 0): s, (2, 2): c,
        (3, 1): s, (3, 3): c,
    })


def _quad_normal_matrix(length_mm: float, eff_gradient,
                        kin: RefKinematics) -> torch.Tensor:
    """The un-rotated (normal) quadrupole matrix — mirrors
    Quadrupole._normal_transfer_matrix.

    Branch-free entire-function form (``_cosc`` / ``_sincs``): the matrix
    and its autograd gradient stay correct through zero gradient, where
    the numpy ``sin(kL)/k`` form (k = √|k1|) would 0/0.  For a non-zero
    gradient it is numerically identical to the numpy matrix.
    """
    L_m = length_mm * 1e-3
    eff_G = _t(eff_gradient)
    charge_sign = 1.0 if kin.charge > 0 else -1.0
    k1 = charge_sign * eff_G / kin.brho            # signed, 1/m^2
    ux = k1 * (L_m * L_m)                          # x-plane argument
    uy = -ux                                       # y-plane argument
    cx, sx = _cosc(ux), _sincs(ux)
    cy, sy = _cosc(uy), _sincs(uy)
    return _assemble({
        (0, 0): cx,                 (0, 1): L_m * sx,
        (1, 0): -k1 * L_m * sx,     (1, 1): cx,
        (2, 2): cy,                 (2, 3): L_m * sy,
        (3, 2): k1 * L_m * sy,      (3, 3): cy,
        (4, 5): _phase_slip(length_mm, kin),
    })


def quad_matrix(length_mm: float, eff_gradient, skew_angle_deg: float,
                kin: RefKinematics) -> torch.Tensor:
    """6x6 transfer matrix of a quadrupole (mirrors Quadrupole.transfer_matrix).

    ``eff_gradient`` is the *effective* gradient (T/m) — the design
    gradient with ``gradient_rel`` folded in by the caller; pass a
    ``requires_grad`` tensor to differentiate through it.
    """
    M = _quad_normal_matrix(length_mm, eff_gradient, kin)
    if skew_angle_deg == 0.0:
        return M
    theta = math.radians(skew_angle_deg)
    R = _transverse_rotation(theta)
    R_inv = _transverse_rotation(-theta)
    return R @ M @ R_inv


# ---------------------------------------------------------------------------
# Solenoid
# ---------------------------------------------------------------------------
def solenoid_matrix(length_mm: float, eff_field,
                    kin: RefKinematics) -> torch.Tensor:
    """6x6 transfer matrix of a solenoid (mirrors Solenoid.transfer_matrix).

    ``eff_field`` is the *effective* on-axis field (T); pass a
    ``requires_grad`` tensor to differentiate through it.  The
    ``sin(φ)/k_s`` couplings are written via ``torch.sinc`` so the matrix
    and its autograd gradient stay correct through zero field.
    """
    L_m = length_mm * 1e-3
    eff_B = _t(eff_field)
    charge_sign = 1.0 if kin.charge > 0 else -1.0
    k_s = charge_sign * eff_B / (2.0 * kin.brho)   # 1/m, signed, linear in B
    phi = k_s * L_m
    C = torch.cos(phi)
    S = torch.sin(phi)
    # sk == S / k_s == sin(phi)/k_s == L_m·sinc(phi/pi); finite at k_s = 0.
    sk = L_m * torch.sinc(phi / math.pi)
    return _assemble({
        (0, 0): C * C,        (0, 1): C * sk,
        (0, 2): S * C,        (0, 3): S * sk,
        (1, 0): -k_s * S * C, (1, 1): C * C,
        (1, 2): -k_s * S * S, (1, 3): S * C,
        (2, 0): -S * C,       (2, 1): -S * sk,
        (2, 2): C * C,        (2, 3): C * sk,
        (3, 0): k_s * S * S,  (3, 1): -S * C,
        (3, 2): -k_s * S * C, (3, 3): C * C,
        (4, 5): _phase_slip(length_mm, kin),
    })


# ---------------------------------------------------------------------------
# Dipole
# ---------------------------------------------------------------------------
def _dipole_edge_matrix(e_deg: float, rho_m: float) -> torch.Tensor:
    """Dipole-internal thin edge focus (mirrors Dipole._edge_matrix).

    Distinct from the standalone Edge element — no fringe ``psi`` term.
    """
    if e_deg == 0.0 or abs(rho_m) < 1e-12:
        return _assemble({})
    tan_e = math.tan(math.radians(e_deg))
    return _assemble({(1, 0): tan_e / rho_m, (3, 2): -tan_e / rho_m})


def _dipole_body_matrix(theta_deg, rho_mm: float, field_index: float,
                        kin: RefKinematics) -> torch.Tensor:
    """Dipole sector-bend body (mirrors Dipole._body_matrix)."""
    theta_t = _t(theta_deg)
    if bool(theta_t == 0.0):
        return _assemble({})
    theta = theta_t * _DEG2RAD
    rho_m = rho_mm * 1e-3
    L_m = abs(rho_m) * torch.abs(theta)
    N = field_index
    beta2gm = kin.beta ** 2 * kin.gamma * kin.mass

    if N == 0.0:
        # ---- pure sector bend ----
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        return _assemble({
            (0, 0): cos_t,             (0, 1): rho_m * sin_t,
            (1, 0): -sin_t / rho_m,    (1, 1): cos_t,
            (0, 5): 1000.0 * rho_m * (1.0 - cos_t) / beta2gm,
            (1, 5): 1000.0 * sin_t / beta2gm,
            (2, 3): L_m,
            (4, 5): _phase_slip(L_m * 1e3, kin),
        })

    # ---- combined-function bend (N != 0) ----
    kx2 = (1.0 - N) / (rho_m * rho_m)
    ky2 = N / (rho_m * rho_m)
    L = L_m
    sign = 1.0 if bool(theta_t >= 0.0) else -1.0
    entries: dict = {}

    # horizontal plane
    if kx2 > 1e-30:
        kx = math.sqrt(kx2)
        cx = torch.cos(kx * L)
        sx = torch.sin(kx * L)
        entries[(0, 0)] = cx
        entries[(0, 1)] = sx / kx
        entries[(1, 0)] = -kx * sx
        entries[(1, 1)] = cx
        entries[(0, 5)] = 1000.0 * (1.0 - cx) / (rho_m * kx2) / beta2gm
        entries[(1, 5)] = 1000.0 * sign * sx / (rho_m * kx) / beta2gm
    elif kx2 < -1e-30:
        kx = math.sqrt(-kx2)
        ch = torch.cosh(kx * L)
        sh = torch.sinh(kx * L)
        entries[(0, 0)] = ch
        entries[(0, 1)] = sh / kx
        entries[(1, 0)] = kx * sh
        entries[(1, 1)] = ch
        entries[(0, 5)] = (ch - 1.0) / (rho_m * kx2) / beta2gm
        entries[(1, 5)] = sign * sh / (rho_m * kx) / beta2gm
    else:
        entries[(0, 1)] = L

    # vertical plane
    if ky2 > 1e-30:
        ky = math.sqrt(ky2)
        cy = torch.cos(ky * L)
        sy = torch.sin(ky * L)
        entries[(2, 2)] = cy
        entries[(2, 3)] = sy / ky
        entries[(3, 2)] = -ky * sy
        entries[(3, 3)] = cy
    elif ky2 < -1e-30:
        ky = math.sqrt(-ky2)
        ch = torch.cosh(ky * L)
        sh = torch.sinh(ky * L)
        entries[(2, 2)] = ch
        entries[(2, 3)] = sh / ky
        entries[(3, 2)] = ky * sh
        entries[(3, 3)] = ch
    else:
        entries[(2, 3)] = L

    entries[(4, 5)] = _phase_slip(L * 1e3, kin)
    return _assemble(entries)


def dipole_matrix(eff_angle_deg, rho_mm: float, length_mm: float,
                  e1_deg: float, e2_deg: float, field_index: float,
                  hv: int, kin: RefKinematics,
                  ds_mm: float | None = None) -> torch.Tensor:
    """6x6 transfer matrix of a dipole (mirrors Dipole.transfer_matrix).

    ``eff_angle_deg`` is the effective bend angle (deg) — pass a
    ``requires_grad`` tensor to differentiate.  ``ds_mm=None`` builds the
    full element including the e1/e2 edges; a slice (``ds_mm`` given)
    scales the angle proportionally and omits the edges.
    """
    eff_angle = _t(eff_angle_deg)
    use_edges = ds_mm is None
    if ds_mm is not None:
        theta_deg = (eff_angle * (ds_mm / length_mm)
                     if length_mm != 0.0 else _t(0.0))
    else:
        theta_deg = eff_angle

    body_theta = torch.abs(theta_deg) if hv == 1 else theta_deg
    M_body = _dipole_body_matrix(body_theta, rho_mm, field_index, kin)

    if use_edges:
        rho_m = rho_mm * 1e-3
        M_ent = _dipole_edge_matrix(e1_deg, rho_m)
        M_ext = _dipole_edge_matrix(e2_deg, rho_m)
        M = M_ext @ M_body @ M_ent
    else:
        M = M_body

    if hv == 1:
        # Swap (x, x') <-> (y, y') planes.
        P = _assemble({
            (0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0, (3, 3): 0.0,
            (0, 2): 1.0, (1, 3): 1.0, (2, 0): 1.0, (3, 1): 1.0,
        })
        M = P @ M @ P
        if bool(theta_deg < 0.0):
            # Bend goes "down": dispersion in y flips sign.
            flip = torch.ones((6, 6), dtype=F64)
            flip[2, 5] = -1.0
            flip[3, 5] = -1.0
            M = M * flip
    return M


# ---------------------------------------------------------------------------
# Edge (standalone element)
# ---------------------------------------------------------------------------
def edge_matrix(pole_rotation_deg: float, rho_mm: float, gap_mm: float,
                k1: float, hv: int) -> torch.Tensor:
    """6x6 transfer matrix of a standalone Edge element (mirrors
    Edge.transfer_matrix).

    Edge has no tunable parameter — the matrix is a constant.
    """
    if pole_rotation_deg == 0.0 or rho_mm == 0.0:
        return _assemble({})
    beta_rad = math.radians(pole_rotation_deg)
    rho_m = rho_mm * 1e-3
    tan_b = math.tan(beta_rad)
    if gap_mm > 0.0 and k1 != 0.0:
        gap_m = gap_mm * 1e-3
        psi = (k1 * gap_m * (1.0 + math.sin(beta_rad) ** 2)
               / (rho_m * max(math.cos(beta_rad), 1e-12)))
    else:
        psi = 0.0
    if hv == 0:                       # horizontal bend plane
        return _assemble({
            (1, 0): tan_b / rho_m,
            (3, 2): -math.tan(beta_rad - psi) / rho_m,
        })
    return _assemble({                # vertical bend plane
        (3, 2): tan_b / rho_m,
        (1, 0): -math.tan(beta_rad - psi) / rho_m,
    })
