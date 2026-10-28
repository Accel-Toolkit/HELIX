# magnet_config (below) is adapted from Xsuite: xtrack v0.103.5,
# beam_elements/elements_src/track_magnet_configure.h (configure_tracking_model).
# Copyright (c) CERN — the Xsuite authors.  Licensed under the Apache License,
# Version 2.0; a copy is in licenses/Xsuite-Apache-2.0.txt and at
# http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.
# Modified for HELIX (2026): translated from C to Python for the five models
# ring mode uses, returning the strengths as a dict.  The rest of this file is
# HELIX's own.
"""Parameter blocks of the ring nodes (one builder per node kind).

Each builder returns the list of floats the kernels read at the node's offset
(layouts in :mod:`linac_gen.ring.kernels.nodes`, :mod:`.cartesian` and
:mod:`.curvilinear`).  They are also what the kernel tests use to set up one
element exactly as Xsuite or ImpactX would.
"""
from __future__ import annotations

import math

from linac_gen.ring.kernels import cartesian as cart
from linac_gen.ring.kernels import curvilinear as curv
from linac_gen.ring.kernels import nodes as nd

INTEGRATORS = {"teapot": cart.INT_TEAPOT, "yoshida4": cart.INT_YOSHIDA, "uniform": cart.INT_UNIFORM,
               "symp2": cart.INT_SYMP2, "symp4": cart.INT_SYMP4, "symp6": cart.INT_SYMP6}


def _trim(knl, ksl) -> tuple[list[float], list[float]]:
    """knl/ksl padded to the same length, trailing zero orders dropped."""
    n = max(len(knl), len(ksl))
    kn = [float(v) for v in knl] + [0.0] * (n - len(knl))
    ks = [float(v) for v in ksl] + [0.0] * (n - len(ksl))
    while kn and kn[-1] == 0.0 and ks[-1] == 0.0:
        kn.pop()
        ks.pop()
    return kn, ks


def drift(length: float, exact: bool) -> list[float]:
    return [float(length), 1.0 if exact else 0.0]


def magnet(length: float, *, integrator: int, n_kicks: int, drift_model: int,
           k0_drift=0.0, k1_drift=0.0, h_drift=0.0, k0_kick=0.0, k1_kick=0.0, h_kick=0.0,
           hxl=0.0, k0_hcorr=0.0, k1_hcorr=0.0, rot_frame=False,
           k2=0.0, k3=0.0, k0s=0.0, k1s=0.0, k2s=0.0, k3s=0.0,
           knl=(), ksl=()) -> list[float]:
    """A thick magnet, already configured (Xsuite ``configure_tracking_model``
    done by the caller, see :func:`magnet_config`)."""
    kn, ks = _trim(knl, ksl)
    order = len(kn) - 1
    b = [0.0] * cart.MG_SIZE_FIXED
    b[cart.MG_LENGTH] = float(length)
    b[cart.MG_INTEGRATOR] = float(integrator)
    b[cart.MG_NKICKS] = float(n_kicks)
    b[cart.MG_DRIFT_MODEL] = float(drift_model)
    b[cart.MG_K0_DRIFT] = float(k0_drift)
    b[cart.MG_K1_DRIFT] = float(k1_drift)
    b[cart.MG_H_DRIFT] = float(h_drift)
    b[cart.MG_K0_KICK] = float(k0_kick)
    b[cart.MG_K1_KICK] = float(k1_kick)
    b[cart.MG_H_KICK] = float(h_kick)
    b[cart.MG_HXL] = float(hxl)
    b[cart.MG_K0_HCORR] = float(k0_hcorr)
    b[cart.MG_K1_HCORR] = float(k1_hcorr)
    b[cart.MG_ROT_FRAME] = 1.0 if rot_frame else 0.0
    b[cart.MG_K2] = float(k2)
    b[cart.MG_K3] = float(k3)
    b[cart.MG_K0S] = float(k0s)
    b[cart.MG_K1S] = float(k1s)
    b[cart.MG_K2S] = float(k2s)
    b[cart.MG_K3S] = float(k3s)
    b[cart.MG_ORDER] = float(order)
    b[cart.MG_INV_FACT] = 1.0 / math.factorial(order) if order >= 0 else 1.0
    b[cart.MG_FACTOR] = 1.0
    return b + kn + ks


def magnet_config(model: str, k0: float, k1: float, h: float) -> dict:
    """Xsuite ``configure_tracking_model`` for the models used here: which
    strengths go into the drift and which into the kicks.

    ``model``: "bend-kick-bend" (exact bend for k0, h; k1 in the kicks),
    "rot-kick-rot" (Xsuite's Bend default: polar drift with k0 kicks nested
    in the drift), "mat-kick-mat" (MAD-X's expanded thick map for k0, k1,
    h), "drift-kick-drift-exact" / "drift-kick-drift-expanded" (everything
    in the kicks)."""
    h_zero = abs(h) < 1e-8
    if model == "bend-kick-bend":
        dm = cart.DM_STRAIGHT_BEND if h_zero else cart.DM_CURVED_BEND
    elif model == "rot-kick-rot":
        dm = cart.DM_DRIFT_EXACT if h_zero else cart.DM_POLAR_YOSHIDA
    elif model == "mat-kick-mat":
        dm = cart.DM_EXPANDED_CFD
    elif model == "drift-kick-drift-exact":
        dm = cart.DM_DRIFT_EXACT
    elif model == "drift-kick-drift-expanded":
        dm = cart.DM_DRIFT_EXPANDED
    else:
        raise ValueError(f"unknown magnet model {model!r}")
    if dm in (cart.DM_DRIFT_EXPANDED, cart.DM_DRIFT_EXACT):
        return dict(drift_model=dm, k0_drift=0.0, k1_drift=0.0, h_drift=0.0, k0_kick=k0,
                    k1_kick=k1, h_kick=h, k0_hcorr=k0, k1_hcorr=k1, rot_frame=True)
    if dm == cart.DM_EXPANDED_CFD:
        return dict(drift_model=dm, k0_drift=k0, k1_drift=k1, h_drift=h, k0_kick=0.0,
                    k1_kick=0.0, h_kick=h, k0_hcorr=0.0, k1_hcorr=k1, rot_frame=False)
    if dm == cart.DM_CURVED_BEND:
        return dict(drift_model=dm, k0_drift=k0, k1_drift=0.0, h_drift=h, k0_kick=0.0,
                    k1_kick=k1, h_kick=h, k0_hcorr=0.0, k1_hcorr=k1, rot_frame=False)
    if dm == cart.DM_STRAIGHT_BEND:
        return dict(drift_model=dm, k0_drift=k0, k1_drift=0.0, h_drift=0.0, k0_kick=0.0,
                    k1_kick=k1, h_kick=0.0, k0_hcorr=0.0, k1_hcorr=0.0, rot_frame=False)
    # DM_POLAR_YOSHIDA
    return dict(drift_model=dm, k0_drift=k0, k1_drift=0.0, h_drift=h, k0_kick=0.0,
                k1_kick=k1, h_kick=h, k0_hcorr=k0, k1_hcorr=k1, rot_frame=False)


def multipole(knl, ksl) -> list[float]:
    kn, ks = _trim(knl, ksl)
    if not kn:
        kn, ks = [0.0], [0.0]
    order = len(kn) - 1
    return [float(order), 1.0 / math.factorial(order)] + kn + ks


def kick(dpx: float, dpy: float) -> list[float]:
    return [float(dpx), float(dpy)]


def thin_lens(inv_fx: float, inv_fy: float) -> list[float]:
    return [float(inv_fx), float(inv_fy)]


def edge_linear(r21: float, r43: float) -> list[float]:
    return [float(r21), float(r43)]


def edge_exact(*, is_exit: bool, angle: float, k0_fringe: float, k0_wedge: float, k1: float,
               fint: float, hgap: float, quad_wedge: int, mult_fringe: bool,
               kn=(0.0, 0.0, 0.0, 0.0), ks=(0.0, 0.0, 0.0, 0.0), length=0.0,
               knl=(), ksl=()) -> list[float]:
    knl_, ksl_ = _trim(knl, ksl)
    order = len(knl_) - 1
    b = [0.0] * nd.EE_KNL
    b[nd.EE_IS_EXIT] = 1.0 if is_exit else 0.0
    b[nd.EE_ANGLE] = float(angle)
    b[nd.EE_SIN] = math.sin(angle)
    b[nd.EE_COS] = math.cos(angle)
    b[nd.EE_TAN] = math.tan(angle)
    b[nd.EE_K0_FRINGE] = float(k0_fringe)
    b[nd.EE_K0_WEDGE] = float(k0_wedge)
    b[nd.EE_K1] = float(k1)
    b[nd.EE_FINT] = float(fint)
    b[nd.EE_HGAP] = float(hgap)
    b[nd.EE_QUAD_WEDGE] = float(int(quad_wedge))      # 0 none, 1 Xsuite, 2 MAD8
    b[nd.EE_MULT_FRINGE] = 1.0 if mult_fringe else 0.0
    for i in range(4):
        b[nd.EE_KN + i] = float(kn[i])
        b[nd.EE_KN + 4 + i] = float(ks[i])
    b[nd.EE_LENGTH] = float(length)
    b[nd.EE_ORDER] = float(order)
    return b + knl_ + ksl_


def cavity(voltage_mv: float, frequency_hz: float, lag_deg: float) -> list[float]:
    return [float(voltage_mv), float(frequency_hz), float(lag_deg)]


def srotation(angle_rad: float) -> list[float]:
    """The frame rotation by ``angle_rad`` about s (exact for multiples of 90 deg)."""
    q = angle_rad / (0.5 * math.pi)
    if abs(q - round(q)) < 1e-15:
        s, c = ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))[int(round(q)) % 4]
    else:
        s, c = math.sin(angle_rad), math.cos(angle_rad)
    return [s, c]


def xyshift(dx: float, dy: float) -> list[float]:
    return [float(dx), float(dy)]


def aperture(shape: str, a=0.0, b=0.0, c=0.0, d=0.0, x0=0.0, y0=0.0,
             vertices=None) -> list[float]:
    code = {"circle": 1, "rectangle": 2, "ellipse": 3, "rectellipse": 4, "polygon": 5}[shape]
    blk = [float(code), float(a), float(b), float(c), float(d), float(x0), float(y0)]
    if shape == "polygon":
        xs_ = [float(v[0]) for v in vertices]
        ys_ = [float(v[1]) for v in vertices]
        blk += [float(len(xs_))] + xs_ + ys_
    else:
        blk += [0.0]
    return blk


def matrix(R, d=(0.0,) * 6) -> list[float]:
    return [float(v) for row in R for v in row] + [float(v) for v in d]


def curvilinear_element(kind: int, length: float, *, integrator: int, nsteps: int,
                    g: float = 0.0, k_normal=(), k_skew=(), m0: int = 1) -> list[float]:
    kn, ks = list(k_normal), list(k_skew)
    n = max(len(kn), len(ks))
    kn += [0.0] * (n - len(kn))
    ks += [0.0] * (n - len(ks))
    return [float(length), float(integrator), float(nsteps), float(kind), float(g),
            float(n), float(m0)] + [float(v) for v in kn] + [float(v) for v in ks]


def curvilinear_quad(length, k1, *, integrator, nsteps):
    return curvilinear_element(curv.KIND_QUAD, length, integrator=integrator, nsteps=nsteps, g=k1)


def curvilinear_multipole(length, k_normal, k_skew, *, integrator, nsteps):
    return curvilinear_element(curv.KIND_MULTIPOLE, length, integrator=integrator, nsteps=nsteps,
                           k_normal=k_normal, k_skew=k_skew)


def curvilinear_cfbend(length, rc, k_normal, k_skew, *, integrator, nsteps):
    """``k_normal[0]`` is the field error k0 - 1/rc (zero for a bend whose
    field matches its curvature, as ImpactX assumes)."""
    m0 = 0 if (k_normal and k_normal[0] != 0.0) or (k_skew and k_skew[0] != 0.0) else 1
    return curvilinear_element(curv.KIND_CFBEND, length, integrator=integrator, nsteps=nsteps, g=rc,
                           k_normal=k_normal, k_skew=k_skew, m0=m0)
