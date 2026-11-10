"""Sacherer / Kapchinsky-Vladimirsky RMS envelope ODE solver.

Continuous-beam envelope equations integrated with scipy ``solve_ivp``,
following the textbook formulation (Reiser §5; Sacherer 1971; KENV's
``solver.py``).  For an unbunched (DC) beam:

    σ_x'' + κ_x(s) σ_x − ε_x²/σ_x³ − K/[2(σ_x + σ_y)] = 0
    σ_y'' + κ_y(s) σ_y − ε_y²/σ_y³ − K/[2(σ_x + σ_y)] = 0

in rms quantities (σ rms size, ε rms emittance; the equivalent
edge-radius form X'' + κX − ε_X²/X³ − 2K/(X + Y) = 0 uses X = 2σ and
total emittance ε_X = 4ε),
with generalised perveance ``K = q·I / (2π ε₀ m₀ c³ (βγ)³)`` and a
focusing function κ(s) read off from the lattice element by element.

Scope of this implementation
----------------------------
Designed as a textbook benchmark for the existing matrix-plus-discrete-
SC EnvelopeSolver, not a replacement for general-purpose tracking.

Supported elements:

* **Drift** — κ = 0.
* **Quadrupole** (hard-edge) — κ_x = ±k₁, κ_y = ∓k₁ from
  ``element.gradient`` and the reference momentum.
* **Solenoid** (hard-edge SOLENOID card) — constant Larmor focusing
  κ = (B/(2Bρ))² in both planes from ``element.field``.
* **Solenoid field maps** — a FieldMap (1-D B_z, 2-D cylindrical) or
  FieldMap3D whose static magnetic field on the axis is longitudinal
  (``field_map_data.on_axis_field_kind``, the failure compensation's
  rule): Larmor focusing κ(s) = (B_z(s) / (2 Bρ))² in both planes, with
  B_z the on-axis field (a 3-D map's interpolated at x = 0, y = 0, as its
  own sampler gives it there, zero outside the map) times the amplitude
  every other path uses: scale × kb × (1 + voltage_rel) / norm_factor
  (``FieldMap._scale`` / ``FieldMap3D._scale_factor``), so kb = 0 or
  voltage_rel = −1 (a failure's OFF) is no field.
* **Steering correctors** — a powered 3-D map whose field on the axis,
  read there, is a dipole's with no quadrupole gradient of a quadrupole's
  size is left out, as a STEERER card is: it steers the beam, which the
  solver does not follow (a B_z it holds on the axis focuses the beam as
  a solenoid map's does).  The correctors of one map, or a cluster's
  corrector children together (their fields summed on the cluster's z),
  are left out while they bend the beam by at most 20 mrad (θ = the
  largest |∫ amplitude × (B_x, B_y) dz| along the element, over Bρ) and
  focus it, as the beam sees it, by at most 1/f = 0.0088 /m: the largest
  |∫ amplitude × (G_n + i G_s) exp(−2i θ_L) dz| along the element, over
  Bρ, of their normal and skew quadrupole gradient on the axis turned by
  the Larmor angle θ_L = sign(q) ∫ B_z dz / (2 Bρ) the beam turns by in
  the element's on-axis B_z, integrated in steps over which the beam
  turns the gradient by at most 2 deg.  Each of two opposite gradient
  lobes at a corrector's ends counts, and a solenoid that turns the beam
  between them keeps them from cancelling.  Beyond either bound they are
  refused: the repository's LEBT corrector maps of the first two pairs
  from 2.4 to 2.5 mrad (standing alone, superposed on their solenoid, or
  both of a pair together; their ADJUST limit is 16 mrad), the others at
  20 mrad.
* **RF field maps** — their RF channels are left out (DC-beam
  approximation: no synchronous-phase focus), so an RF map is a drift;
  one that also carries a solenoid channel keeps that solenoid's focusing.
* **SUPERPOSE_MAP clusters** (SuperposedFieldMap) — the sum of the
  children's on-axis B_z, each child read over its window z0 … z0 + L of
  the cluster with its own amplitude, times the cluster's own amplitude
  factor (1 + its voltage_rel); its steering-corrector children are
  bounded together and left out.
* **Markers / Apertures / Steerers / SpaceChargeComp / RF gaps** — pass
  through with κ = 0 (no transverse focus contribution).  RF gaps are
  treated as drifts (DC-beam approximation: no synchronous-phase focus).

Refused with a ValueError that names the element: a field map with a
curved reference trajectory (SUPERPOSE_MAP_OUT, a bending cluster), and
any powered static map the solver has no model of — a 1-D quadrupole
gradient G(z); a 3-D map whose quadrupole gradient on the axis makes more
than a quarter of its peak field at the grid's half-width (a quadrupole
whose magnetic centre lies on its grid, at any strength); a 3-D map whose
field on the axis is a solenoid's that its grid point nearest the axis
does not show, or no solenoid's (a dipole's, or one that vanishes there)
where that point shows a solenoid's; a 3-D map whose field vanishes on
the axis with no gradient there (a multipole); steering correctors that
bend the beam by more than 20 mrad or focus it, as the beam sees it, by
more than 1/f = 0.0088 /m (a dipole with a gradient part, a quadrupole
whose magnetic centre lies off its grid, a corrector powered past that); a
static electric map — each pointing to ``env_solver='matrix'``; and a 2-D
Cartesian map, which no HELIX mode tracks.  A solenoid map is read by its
B_z on the axis alone: a dipole field or a gradient it also holds there is
neither seen nor bounded.

The reference particle's β, γ are held constant per call (DC beam
through low-energy structures); for accelerated beams the existing
matrix tracker remains the right tool.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
from scipy.integrate import solve_ivp

from linac_gen.core.constants import EPSILON_0, PI, E_CHARGE, C_LIGHT
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.field_map import FieldMap as FieldMap1D
from linac_gen.elements.field_map import _normalize_2d_grid
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.field_map_data import on_axis_field_kind


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class SachererResults:
    """Output of the Sacherer ODE solver."""
    s: List[float] = field(default_factory=list)            # mm
    sigma_x: List[float] = field(default_factory=list)      # mm
    sigma_y: List[float] = field(default_factory=list)
    sigma_xp: List[float] = field(default_factory=list)     # mrad
    sigma_yp: List[float] = field(default_factory=list)
    element_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _perveance(current_mA: float, ref: ReferenceParticle) -> float:
    """Generalised perveance K = qI / (2π ε₀ m c³ (βγ)³).

    K is dimensionless when σ is in metres and σ' in radians; we then
    rescale below to the mm/mrad system used internally.
    """
    if current_mA == 0:
        return 0.0
    q = abs(ref.species.charge) * E_CHARGE  # |q| in Coulombs
    I_A = abs(current_mA) * 1e-3
    m_kg = ref.species.mass * 1e6 * E_CHARGE / C_LIGHT ** 2
    bg = ref.beta * ref.gamma
    return q * I_A / (
        2.0 * PI * EPSILON_0 * m_kg * C_LIGHT ** 3 * bg ** 3
    )


#: the field maps whose on-axis field the solver reads
_FIELD_MAPS = (FieldMap1D, FieldMap3D, SuperposedFieldMap)

#: the largest deflection (rad) of the steering correctors of one element,
#: or of one SUPERPOSE_MAP cluster together: powered 3-D maps whose field on
#: the axis is a dipole's are left out while their summed dipole field bends
#: the beam by at most this angle, and refused beyond it (the repository's
#: corrector maps but the first two LEBT pairs are refused from here)
_CORRECTOR_MAX_DEFLECTION = 0.020

#: the largest focusing (1/m) of the steering correctors of one element, or
#: of one cluster together: they are left out while their summed quadrupole
#: gradient on the axis, as the beam sees it (turned by the beam's Larmor
#: rotation in the element's B_z), focuses the beam by at most this 1/f at
#: every point of the element, and refused beyond it.  The bound lies where
#: leaving their gradient out costs 0.5 % of an rms size: the repository's
#: first two LEBT corrector pairs reach it at 2.4 to 2.5 mrad (at their
#: 16 mrad ADJUST limit they focus the beam by 0.057 to 0.059 /m), where,
#: superposed on the first solenoid, their gradient moves a 30 keV H- beam's
#: σ_x and σ_y by up to 0.50 % in a 20000-particle run, as in the matrix
#: envelope
_CORRECTOR_MAX_FOCUSING = 0.0088

#: the largest angle (rad) the gradient the beam sees turns by between two
#: points at which it is integrated: each cell of a corrector's grid is cut
#: into equal parts short enough for the B_z on the element's axis
_TURN_STEP = np.radians(2.0)

#: a 3-D map has a quadrupole gradient on its axis when that gradient, times
#: the grid's half-width, exceeds this share of the map's peak field (an ideal
#: quadrupole whose magnetic centre lies on the grid makes at least 1/(2√2))
_QUADRUPOLE_SHARE = 0.25

#: a grid point whose distance from 0 is at most this share of its cell's width
#: lies on the axis (a grid built from a file's bounds in metres can place the
#: point meant for the axis about 1e-15 mm off it)
_ON_THE_AXIS = 1e-9

#: what a refusal calls a static magnetic map the solver has no model of
_NO_MODEL = {
    "gradient": "a 1-D quadrupole-gradient map G(z)",
    "quadrupole": "a 3-D map with a quadrupole gradient on the axis",
    "solenoid": "a 3-D map whose field on the axis is a solenoid's, which "
                "its grid point nearest the axis does not show",
    "dipole": "a 3-D map whose field on the axis is a dipole's, which its "
              "grid point nearest the axis does not show",
    "vanishing": "a 3-D map whose field vanishes on the axis, with no "
                 "gradient there (a multipole)",
    "other": "a 2-D Cartesian field map",
}

#: what a refusal points to: the matrix envelope tracks every map refused
#: here but a 2-D Cartesian one, which no mode tracks
_MATRIX = "use env_solver='matrix'"
_NO_MODE = "no HELIX mode tracks one"


def _label(elem) -> str:
    return getattr(elem, "label", None) or getattr(elem, "name", None) or "?"


def _named(elem) -> str:
    """``elem``'s label or name, quoted, and its field file's name when it
    has no label of its own (the parser names a straight SUPERPOSE_MAP
    cluster's children FMAP_001, FMAP_002, … whatever the deck calls them)."""
    text = f"'{_label(elem)}'"
    path = getattr(elem, "field_file", None)
    if path and not getattr(elem, "label", None):
        text += f" ({os.path.basename(str(path))})"
    return text


def _refuse(what: str, elem, cluster=None, instead=_MATRIX):
    """Refuse ``what``, naming the element ``elem`` (or each element of a
    list of them, judged together; ``_named``) and the cluster it is in."""
    names = [_named(e) for e in (elem if isinstance(elem, list) else [elem])]
    where = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    if cluster is not None:
        where += f" in the SUPERPOSE_MAP cluster '{_label(cluster)}'"
    raise ValueError(f"the Sacherer envelope solver has no model of {what} "
                     f"({where}) — {instead}")


def _past(value, bound, digits, kind):
    """``value``, beyond ``bound``, as text: ``digits`` digits of the format
    ``kind`` ("f" decimals or "g" significant digits), more while the text
    would read no more than ``bound``."""
    text = f"{value:.{digits}{kind}}"
    while float(text) <= bound and digits < 17:
        digits += 1
        text = f"{value:.{digits}{kind}}"
    return text


def _amplitude(elem, ch_enum, ch) -> float:
    """A channel's amplitude as every physics path takes it: scale × k_e or
    k_b × (1 + voltage_rel) / norm_factor."""
    if isinstance(elem, FieldMap3D):
        return float(elem._scale_factor(ch_enum, ch))
    return float(elem._scale(ch_enum, ch))


def _axis_weights(a):
    """``[(index, weight)]`` of the linear interpolation at 0 along the grid
    axis ``a`` (between the point nearest 0 and its neighbour across 0, or
    that point alone when its distance from 0 is at most ``_ON_THE_AXIS``
    times its cell's width); None when 0 lies outside the grid."""
    a = np.asarray(a, dtype=np.float64).ravel()
    i = int(np.argmin(np.abs(a)))
    cell = min((abs(float(a[k] - a[i])) for k in (i - 1, i + 1) if 0 <= k < a.size),
               default=0.0)
    if abs(float(a[i])) <= _ON_THE_AXIS * cell:
        return [(i, 1.0)]
    j = next((k for k in (i - 1, i + 1)
              if 0 <= k < a.size and a[k] * a[i] < 0.0), None)
    if j is None:
        return None
    t = float(a[i] / (a[i] - a[j]))
    return [(i, 1.0 - t), (j, t)]


def _axis_slopes(a):
    """``[(index, weight)]`` of the slope at 0 along the grid axis ``a`` of
    the interpolation ``_axis_weights`` reads: across the two points around
    0 or, on a point at 0, across its neighbours (one-sided at the grid's
    edge); ``[]`` on a single point.  Call it where ``_axis_weights`` is not
    None."""
    a = np.asarray(a, dtype=np.float64).ravel()
    w = _axis_weights(a)
    if len(w) == 2:
        (i, _wi), (j, _wj) = w
    else:
        k = w[0][0]
        i, j = max(k - 1, 0), min(k + 1, a.size - 1)
        if i == j:
            return []
    d = float(a[j] - a[i])
    return [(i, -1.0 / d), (j, 1.0 / d)]


def _on_axis_3d(ch):
    """``(B_x, B_y, B_z, G_n, G_s)`` of a 3-D channel on the axis x = 0,
    y = 0 at its grid's z nodes (unscaled), or None when its grid misses the
    axis.  Each component is interpolated there as ``_axis_bz_nodes`` reads
    B_z; G_n and G_s (T/mm) are the normal and the skew quadrupole gradient
    there, (∂B_x/∂y + ∂B_y/∂x)/2 and (∂B_x/∂x − ∂B_y/∂y)/2: the traceless
    symmetric part of d(B_x, B_y)/d(x, y), taken across the grid points
    around the axis.  A uniform dipole field and a solenoid's fringe (the
    trace) have none; a quadrupole has it whether its magnetic centre is on
    the axis or off it."""
    wx, wy = _axis_weights(ch.x), _axis_weights(ch.y)
    if wx is None or wy is None:
        return None
    sx, sy = _axis_slopes(ch.x), _axis_slopes(ch.y)
    n_z = len(np.asarray(ch.z))

    def at(f, w_x, w_y):
        out = np.zeros(n_z)
        if f is None:
            return out
        f = np.asarray(f, dtype=np.float64)
        for i, w_i in w_x:
            for j, w_j in w_y:
                out = out + (w_i * w_j) * f[i, j, :]
        return out

    bx, by, bz = (at(f, wx, wy) for f in (ch.Fx, ch.Fy, ch.Fz))
    normal = 0.5 * (at(ch.Fx, wx, sy) + at(ch.Fy, sx, wy))
    skew = 0.5 * (at(ch.Fx, sx, wy) - at(ch.Fy, wx, sy))
    return bx, by, bz, normal, skew


def _axis_fields(ch, axis):
    """``(peak, bt, bl)`` of a 3-D channel: its peak field anywhere on its
    grid, and the largest transverse and longitudinal field on its axis
    (``axis``, as ``_on_axis_3d`` reads it there)."""
    bx, by, bz = axis[:3]
    peak = max(float(np.max(np.abs(f))) for f in (ch.Fx, ch.Fy, ch.Fz)
               if f is not None)
    bt = max(float(np.max(np.abs(bx))), float(np.max(np.abs(by))))
    return peak, bt, float(np.max(np.abs(bz)))


def _corrector_or_refuse(ch, axis, elem, cluster):
    """Refuse a powered 3-D channel whose field on the axis is not a
    solenoid's unless it is a steering corrector's.  A steering corrector's
    field on the axis, read there (``axis``, ``_on_axis_3d``'s), is a
    dipole's: transverse, at least its B_z and above a tenth of the map's
    peak (``on_axis_field_kind``'s rule), with no quadrupole gradient of a
    quadrupole's size (the gradient |(G_n, G_s)| times the grid's half-width
    within ``_QUADRUPOLE_SHARE`` of the peak); ``_steering_or_refuse``
    bounds what it bends and focuses.  Refused: a quadrupole whose magnetic
    centre lies on its grid (at any strength), a solenoid's field on the axis
    that the grid point nearest the axis, which ``on_axis_field_kind`` reads,
    does not show, a field that vanishes on the axis with no gradient there
    (a multipole)."""
    normal, skew = axis[3:]
    peak, bt, bl = _axis_fields(ch, axis)
    half = max(0.5 * float(np.ptp(np.asarray(a, dtype=np.float64)))
               for a in (ch.x, ch.y))
    grad = np.hypot(normal, skew)
    if float(np.max(grad)) * half > _QUADRUPOLE_SHARE * peak:
        _refuse(_NO_MODEL["quadrupole"], elem, cluster)
    if bl > bt and bl > 0.1 * peak:
        _refuse(_NO_MODEL["solenoid"], elem, cluster)
    if not (bt >= bl and bt > 0.1 * peak):
        _refuse(_NO_MODEL["vanishing"], elem, cluster)


def _points(z, lo, hi, step=np.inf):
    """The points (mm) at which a field sampled at the nodes ``z`` is
    integrated over the window [lo, hi], held within the nodes: lo, each
    node inside it and hi, each interval between them cut into equal parts
    no longer than ``step``; lo alone when the window is empty."""
    lo, hi = max(float(lo), float(z[0])), min(float(hi), float(z[-1]))
    if hi <= lo:
        return np.array([lo])
    at = np.concatenate(([lo], z[(z > lo) & (z < hi)], [hi]))
    if np.isfinite(step):
        parts = np.ceil(np.diff(at) / step).astype(int)
        at = np.concatenate([a + (b - a) * np.arange(n) / n for a, b, n
                             in zip(at[:-1], at[1:], parts, strict=True)]
                            + [at[-1:]])
    return at


def _integral_to(z, f, at):
    """∫ f dz from the first node of ``z`` to each point of ``at``, of the
    samples ``f`` at the nodes ``z``, linear between them and zero outside
    the nodes."""
    at = np.clip(np.asarray(at, dtype=np.float64), z[0], z[-1])
    if z.size < 2:
        return np.zeros_like(at)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(z))))
    k = np.clip(np.searchsorted(z, at, side="right") - 1, 0, z.size - 2)
    return cum[k] + 0.5 * (f[k] + np.interp(at, z, f)) * (at - z[k])


def _larmor_angle(pieces, z_local, brho, charge):
    """The angle θ_L (rad) the beam turns by from the element's entrance to
    each element-local z of ``z_local`` (mm), in the on-axis B_z of
    ``pieces`` (``[(z_end, z0, z_start, z_nodes, b_nodes)]``, each read from
    the entrance up to its z_end as ``_on_axis_bz`` reads it): θ_L =
    sign(q) × ∫ B_z dz / (2 Bρ), for a beam of rigidity ``brho`` (T·m) and
    charge ``charge`` (units of e)."""
    z_local = np.asarray(z_local, dtype=np.float64)
    turned = np.zeros_like(z_local)
    for z_end, z0, z_start, zn, bn in pieces:
        upto = np.clip(z_local, 0.0, max(z_end, 0.0))
        turned = turned + (_integral_to(zn, bn, z_start + (upto - z0))
                           - _integral_to(zn, bn, z_start - z0))
    return float(np.sign(charge)) * turned * 1e-3 / (2.0 * brho)


def _largest(runs):
    """max_z |Σ ∫_0^z f dz|: ``runs`` holds ``(points, f)`` on the
    element-local z, the samples f (real or complex) of an integrand at its
    points, linear between them and zero outside them (its integral zero
    before its first point and held past its last).  Between two points of
    the runs the summed integrand is linear (one-sided next to a run's
    ends), the summed integral I quadratic and |I|² a quartic: the sum is
    read at every point and where, within a cell, d|I|²/dz = 2 Re(conj(I)
    dI/dz) vanishes, at the roots of that cubic, so a field that changes
    sign or turns within a cell of a coarse grid is read in full, in one
    plane or in two."""
    at = np.unique(np.concatenate([p for p, _f in runs]))
    total = sum(_integral_to(p, f, at) for p, f in runs)
    if at.size > 1:
        # the summed integrand just after each point and just before the next
        after = sum(np.where((at >= p[0]) & (at < p[-1]), np.interp(at, p, f), 0.0)
                    for p, f in runs)
        before = sum(np.where((at > p[0]) & (at <= p[-1]), np.interp(at, p, f), 0.0)
                     for p, f in runs)
        # a cell's share t of the way along: I = c + b t + a t², so
        # Re(conj(I) dI/dt) = 2|a|² t³ + 3 Re(a b̄) t² + (|b|² + 2 Re(a c̄)) t + Re(b c̄)
        h = np.diff(at)
        c, b = total[:-1], h * after[:-1]
        a = 0.5 * h * (before[1:] - after[:-1])
        cubic = np.stack([2.0 * np.abs(a) ** 2, 3.0 * np.real(a * np.conj(b)),
                          np.abs(b) ** 2 + 2.0 * np.real(a * np.conj(c)),
                          np.real(b * np.conj(c))], axis=1)
        # a cell whose integrand is (all but) constant has no largest |I| inside
        k = np.flatnonzero(cubic[:, 0] > 1e-12 * np.max(np.abs(cubic), axis=1))
        if k.size:
            companion = np.zeros((k.size, 3, 3))
            companion[:, 0, :] = -cubic[k, 1:] / cubic[k, :1]
            companion[:, 1, 0] = companion[:, 2, 1] = 1.0
            t = np.real(np.linalg.eigvals(companion))
            inside = (at[k, None] + h[k, None] * t)[(t > 0.0) & (t < 1.0)]
            total = np.concatenate([total, sum(_integral_to(p, f, inside) for p, f in runs)])
    return float(np.max(np.abs(total)))


def _steering_or_refuse(correctors, pieces, brho, charge, cluster=None):
    """Leave out the steering correctors ``correctors`` of one element, or of
    one SUPERPOSE_MAP cluster (``cluster``) together, or refuse them.  Each
    is ``(elem, amp, z_end, z0, z_start, lo, hi, z, axis)`` (``_map_pieces``
    gives them): its map and amplitude, where it sits and ends in the element,
    its window [lo, hi] and z nodes (mm in the map's frame) and its field on
    the axis there (``_on_axis_3d``'s).  Summed on the element's z, each over
    its window, for a beam of rigidity ``brho`` (T·m) and charge ``charge``
    (units of e), they bend the beam by θ = max_z |∫_0^z Σ amp (B_x, B_y) dz|
    / Bρ ≤ ``_CORRECTOR_MAX_DEFLECTION`` (the largest angle the beam reaches,
    both planes together) and focus it by 1/f = max_z |∫_0^z Σ amp (G_n +
    i G_s) exp(−2i θ_L) dz| / Bρ ≤ ``_CORRECTOR_MAX_FOCUSING``: their
    quadrupole gradient on the axis as the beam sees it, turned by the Larmor
    angle θ_L (``_larmor_angle``) of the on-axis B_z of ``pieces`` (the
    element's: its solenoid channels' and any the correctors hold), and the
    largest value along z, so that each of two opposite lobes counts and a
    solenoid that turns the beam between them keeps them from cancelling.
    That gradient is integrated over each corrector's nodes, a cell cut into
    equal parts over which the beam turns it by at most ``_TURN_STEP`` (the
    gradient linear between the nodes, θ_L read at every point), and both
    largest values are read where ``_largest`` reads them.  They steer
    the beam, which the solver does not follow, and are given no rms
    focusing, as STEERER cards are; refused when they bend or focus the beam
    by more, the refusal giving the angle or the focusing with as many
    digits as it takes to read past the bound."""
    if not correctors:
        return
    # the beam sees the gradient turn by at most this angle per mm: twice
    # dθ_L/dz, bounded by the sum of each field's largest |B_z| over Bρ
    rate = sum(float(np.max(np.abs(b))) for *_piece, b in pieces) * 1e-3 / brho
    step = _TURN_STEP / rate if rate > 0.0 else np.inf
    bends, focus, elems = [], [], []
    for elem, amp, _z_end, z0, z_start, lo, hi, z, axis in correctors:
        bx, by, _bz, normal, skew = axis
        shift = z0 - z_start                  # from the map's frame to the element's z
        at = _points(z, lo, hi)
        bends.append((at + shift, amp * np.interp(at, z, bx + 1j * by)))
        at = _points(z, lo, hi, step)
        seen = np.interp(at, z, normal + 1j * skew) * np.exp(
            -2j * _larmor_angle(pieces, at + shift, brho, charge))
        focus.append((at + shift, amp * seen))
        if not any(elem is e for e in elems):
            elems.append(elem)
    one = len(elems) == 1
    named = elems[0] if one else elems
    theta = _largest(bends) * 1e-3 / brho
    if theta > _CORRECTOR_MAX_DEFLECTION:
        bound = _CORRECTOR_MAX_DEFLECTION * 1e3
        angle = _past(theta * 1e3, bound, 1, "f")
        _refuse(f"a 3-D map whose dipole field on the axis bends the beam by "
                f"{angle} mrad (a steering corrector, up to {bound:g} "
                f"mrad, is left out)" if one else
                f"3-D maps whose dipole fields on the axis bend the beam by "
                f"{angle} mrad together (steering correctors, up to "
                f"{bound:g} mrad together, are left out)", named, cluster)
    inv_f = _largest(focus) / brho
    if inv_f > _CORRECTOR_MAX_FOCUSING:
        bound = _CORRECTOR_MAX_FOCUSING
        focusing = _past(inv_f, bound, 3, "g")
        _refuse(f"a 3-D map whose quadrupole gradient on the axis, as the beam "
                f"sees it, focuses the beam by up to 1/f = {focusing} /m (a "
                f"steering corrector, up to {bound:g} /m, is left out)" if one else
                f"3-D maps whose quadrupole gradients on the axis, as the beam "
                f"sees them, focus the beam by up to 1/f = {focusing} /m "
                f"together (steering correctors, up to {bound:g} /m together, "
                f"are left out)", named, cluster)


def _axis_bz_nodes(ch, elem, cluster):
    """``(z, B_z)`` on the axis of a longitudinal static magnetic channel at
    its grid's z nodes (mm in the map's own frame; unscaled): a 1-D map's
    B_z, a 2-D cylindrical map's r = 0 row, a 3-D map's B_z interpolated at
    x = 0, y = 0 in every z plane (what its trilinear sampler gives there)."""
    z = np.asarray(ch.z, dtype=np.float64)
    fz = np.asarray(ch.Fz, dtype=np.float64)
    if ch.geometry == 1:
        return z, fz
    if ch.geometry in (4, 5):
        rows = _normalize_2d_grid(fz, np.asarray(ch.r, dtype=np.float64), z)
        return z, np.asarray(rows, dtype=np.float64)[0, :]
    wx, wy = _axis_weights(ch.x), _axis_weights(ch.y)
    if wx is None or wy is None:
        _refuse("a 3-D map whose grid does not reach the axis x = 0, y = 0",
                elem, cluster)
    bz = np.zeros(fz.shape[2])
    for i, w_i in wx:
        for j, w_j in wy:
            bz = bz + (w_i * w_j) * fz[i, j, :]
    return z, bz


def _map_pieces(elem, correctors, cluster=None, z0=0.0, factor=1.0) -> list:
    """The on-axis B_z of one FieldMap / FieldMap3D as
    ``[(z_end, z0, z_start, z_nodes, b_nodes)]``: z_nodes in the map's own
    frame, b_nodes in T (the channel amplitude × ``factor`` applied).  In a
    cluster the map sits at ``z0`` and ends at ``z_end`` = z0 + its length
    (it reads no field before its start, nor past its card's length: the
    tracker's window), unless the cluster ends there too (the ODE's last
    stage sits on the cluster's exit).

    Its RF channels are left out (the DC approximation), as is a channel at
    zero amplitude (no field); a steering corrector (``_corrector_or_refuse``)
    goes to the list ``correctors``, for ``_steering_or_refuse``, and a B_z
    it holds on the axis to the pieces, as a solenoid's; a powered channel
    the solver has no model of is refused, as is a 3-D
    channel whose grid point nearest the axis reads a solenoid's field where
    the field on the axis, read there (``_on_axis_3d``), is not a solenoid's
    (a dipole's, or one that vanishes there)."""
    z_end = np.inf
    if (cluster is not None
            and float(z0) + float(elem.length) < float(cluster.length)):
        z_end = float(z0) + float(elem.length)
    # the window the solver reads, in the map's own frame
    span = float(cluster.length) if cluster is not None else float(elem.length)
    lo = float(elem._z_map_start) - float(z0)
    hi = lo + min(z_end, span)
    pieces = []
    for ch_enum, ch in elem.field_data.channels.items():
        if ch_enum.is_rf:
            continue
        amp = _amplitude(elem, ch_enum, ch) * factor
        if amp == 0.0:
            continue
        if ch_enum.is_electric:
            _refuse("a static electric field map", elem, cluster)
        kind = on_axis_field_kind(ch)
        if kind == "none":
            continue
        if ch.geometry == 7:
            axis = _on_axis_3d(ch)
            if axis is None:
                _refuse("a 3-D map whose grid does not reach the axis x = 0, "
                        "y = 0", elem, cluster)
            if kind != "longitudinal":
                _corrector_or_refuse(ch, axis, elem, cluster)
                correctors.append((elem, amp, z_end, float(z0),
                                   float(elem._z_map_start), lo, hi,
                                   np.asarray(ch.z, dtype=np.float64), axis))
                if np.any(axis[2] != 0.0):
                    # its B_z on the axis focuses the beam as a solenoid's
                    # does (the LEBT corrector maps hold none, the SCL decks'
                    # up to 0.2 % of their dipole field)
                    pieces.append((z_end, float(z0), float(elem._z_map_start),
                                   np.asarray(ch.z, dtype=np.float64),
                                   amp * axis[2]))
                continue
            peak, bt, bl = _axis_fields(ch, axis)
            if not (bl > bt and bl > 0.1 * peak):
                _refuse(_NO_MODEL["dipole" if bt >= bl and bt > 0.1 * peak
                                  else "vanishing"], elem, cluster)
        if kind != "longitudinal":
            _refuse(_NO_MODEL[kind], elem, cluster,
                    _NO_MODE if kind == "other" else _MATRIX)
        z, bz = _axis_bz_nodes(ch, elem, cluster)
        pieces.append((z_end, float(z0), float(elem._z_map_start), z,
                       amp * bz))
    return pieces


def _on_axis_bz(elem, brho, charge) -> Callable[[float], float] | None:
    """B_z (T) on the axis of a field map or SUPERPOSE_MAP cluster as a
    function of the element-local z (mm), or None when it has no static
    magnetic field the solver reads (an RF map, a map switched off, steering
    correctors with no B_z on the axis, for a beam of rigidity ``brho``,
    T·m, and charge ``charge``, units of e).

    A cluster's is the sum of its children's, each over its own window and
    at its own amplitude, times the cluster's amplitude factor (1 + its
    voltage_rel, where an error study's draw on the cluster or an ERROR_CAV
    card lands); the tracker sums them the same way.  The steering
    correctors of a map, or of a cluster's children together, are bounded
    by ``_steering_or_refuse``.  Raises ValueError for a map the solver has
    no model of."""
    correctors = []
    if isinstance(elem, SuperposedFieldMap):
        factor = float(elem._child_amp())
        pieces = [p for z0, child in elem.children
                  for p in _map_pieces(child, correctors, elem, z0, factor)]
        _steering_or_refuse(correctors, pieces, brho, charge, elem)
    else:
        pieces = _map_pieces(elem, correctors)
        _steering_or_refuse(correctors, pieces, brho, charge)
    if not pieces:
        return None

    def bz_of_z(z_local_mm: float, _pieces=tuple(pieces)) -> float:
        b = 0.0
        for z_end, z0, z_start, zn, bn in _pieces:
            if z_local_mm < z_end:
                b += float(np.interp(z_start + (z_local_mm - z0), zn, bn,
                                     left=0.0, right=0.0))
        return b

    return bz_of_z


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
class SachererSolver:
    """Integrate the KV/Sacherer envelope ODE element by element.

    Parameters
    ----------
    lattice : Lattice
        Linac_Gen lattice.
    ref : ReferenceParticle
        Synchronous particle (β, γ held constant for DC operation).
    init : dict
        Initial Twiss parameters with keys ``alpha_x, beta_x, emit_x``
        and the y-counterparts — the SAME ``initial`` dialect
        ``EnvelopeSolver`` consumes (``cli/common.py:_envelope_initial``):
        ``emit_*`` GEOMETRIC in **mm·mrad**, ``beta_*`` in m (numerically
        ≡ mm/mrad).  The two solvers are dispatch-interchangeable behind
        ``run_envelope_sim``, so they must share one dict contract.
        (Before 2026-07-25 this boundary expected m·rad, so every
        production caller was 1e6 off in ε and σ came out 1000× large.)
    current : float
        Beam current in mA.
    n_record_per_element : int
        Number of σ samples written between element entry and exit (≥ 2).
    """

    def __init__(self, lattice, ref: ReferenceParticle, init: dict,
                 current: float = 0.0, n_record_per_element: int = 21):
        self.lattice = lattice
        for _e in getattr(lattice, "elements", []):
            if getattr(_e, "is_curved", False):
                raise ValueError(
                    "the Sacherer envelope solver has no model of a field map "
                    "with a curved reference trajectory (SUPERPOSE_MAP_OUT, "
                    f"'{getattr(_e, 'label', None) or _e.name}') — use "
                    "env_solver='matrix'")
        # a field map the solver has no model of is refused before the run
        for _e in getattr(lattice, "elements", []):
            if (isinstance(_e, _FIELD_MAPS)
                    and float(getattr(_e, "length", 0) or 0) > 0):
                _on_axis_bz(_e, ref.brho, ref.species.charge)
        self.ref = ref
        self.init = init
        self.current = float(current)
        self.n_record = max(2, int(n_record_per_element))
        # Base perveance at full current; the effective K used inside each
        # element is scaled by (1 − SCC.factor) to mirror the Tracker's
        # SPACE_CHARGE_COMP convention.
        self._K_full = _perveance(self.current, ref)
        self._sc_factor = 1.0  # current effective SC fraction (1 − comp)

        # Convert input twiss → σ at element 0 entrance.  The dict
        # carries ε in mm·mrad (ecosystem dialect); the ODE runs in
        # m/rad, so scale by 1e-6 here — and ONLY here, so the ε²/σ³
        # pressure term and the σ seed stay consistent.
        ax = float(init["alpha_x"]); bx = float(init["beta_x"])
        ay = float(init["alpha_y"]); by_ = float(init["beta_y"])
        ex = float(init["emit_x"]) * 1e-6      # mm·mrad → m·rad
        ey = float(init["emit_y"]) * 1e-6
        # σ_x in metres, σ_x' in radians.  Sacherer ODE uses m/rad
        # internally; we record in mm/mrad for parity with EnvelopeSolver.
        self.sx0 = float(np.sqrt(bx * ex))
        self.sxp0 = -ax * np.sqrt(ex / bx) if bx > 0 else 0.0
        self.sy0 = float(np.sqrt(by_ * ey))
        self.syp0 = -ay * np.sqrt(ey / by_) if by_ > 0 else 0.0
        self.ex_g = float(ex)
        self.ey_g = float(ey)
        if any(float(init.get(k, 0.0) or 0.0)
               for k in ("disp_x", "disp_xp", "disp_y", "disp_yp")):
            import warnings
            warnings.warn("the Sacherer solver does not model an input dispersion: it starts "
                          "from the betatron beam (the matrix envelope and a multiparticle "
                          "run apply the dispersion)", UserWarning, stacklevel=2)

    # ------------------------------------------------------------------
    def _kappa_of_element(self, elem):
        """Return (κ_x(z), κ_y(z), L_m) for the given element.

        z is measured from element entrance, in metres.  Functions return
        focusing strength in 1/m².
        """
        L_mm = float(getattr(elem, "length", 0) or 0)
        L_m = L_mm * 1e-3

        if isinstance(elem, Drift) or L_mm <= 0:
            return (lambda z: 0.0), (lambda z: 0.0), L_m

        if isinstance(elem, Quadrupole):
            # Hard-edge: use element.gradient (T/m) and reference Bρ.
            G = float(getattr(elem, "gradient", 0.0))
            p_SI = self.ref.beta * self.ref.gamma * \
                self.ref.species.mass * 1e6 * E_CHARGE / C_LIGHT
            Brho = p_SI / E_CHARGE  # T·m for unit charge
            k1 = G / Brho  # 1/m²
            return (lambda z, k=k1: +k), (lambda z, k=k1: -k), L_m

        if isinstance(elem, Solenoid):
            # Hard-edge SOLENOID card: constant Larmor focusing
            # κ = (B / (2 Bρ))² in BOTH planes.  Was silently treated
            # as a drift before 2026-07-25 (review claim 1) — for a
            # solver whose stated niche is DC LEBT benchmarking, the
            # solenoid is the element that matters most.
            B = float(getattr(elem, "effective_field", None)
                      or getattr(elem, "field", 0.0) or 0.0)
            p_SI = self.ref.beta * self.ref.gamma * \
                self.ref.species.mass * 1e6 * E_CHARGE / C_LIGHT
            Brho = p_SI / E_CHARGE  # T·m
            k_sol = (B / (2.0 * Brho)) ** 2
            return (lambda z, k=k_sol: k), (lambda z, k=k_sol: k), L_m

        if isinstance(elem, _FIELD_MAPS):
            # the on-axis B_z of a solenoid map or SUPERPOSE_MAP cluster
            # (raises for a map the solver has no model of)
            bz_fn = _on_axis_bz(elem, self.ref.brho,
                                self.ref.species.charge)
            if bz_fn is not None:
                # Larmor focusing: κ = (q Bz / (2 p))² ≡ (Bz / (2 Bρ))²
                p_SI = self.ref.beta * self.ref.gamma * \
                    self.ref.species.mass * 1e6 * E_CHARGE / C_LIGHT
                Brho = p_SI / E_CHARGE  # T·m
                def kappa(z_m, _bz=bz_fn, _Brho=Brho, _L=L_mm):
                    # z within the element (the last RK stage may round
                    # past its end by an ulp)
                    bz = _bz(min(max(z_m * 1e3, 0.0), _L))  # mm
                    return (bz / (2.0 * _Brho)) ** 2
                return kappa, kappa, L_m
            # No static magnetic field — a drift (RF cavities: the DC
            # approximation; a map switched off).
            return (lambda z: 0.0), (lambda z: 0.0), L_m

        # Anything else (markers, apertures, steerers, SpaceChargeComp):
        # zero focusing, just consume length (which is 0 for thin
        # markers / kicks anyway).
        return (lambda z: 0.0), (lambda z: 0.0), L_m

    # ------------------------------------------------------------------
    def _rhs(self, z, X, kx_fn, ky_fn):
        """Right-hand side of the coupled KV envelope ODE."""
        sx, sxp, sy, syp = X
        # Avoid σ → 0 singularities.
        sx_safe = sx if abs(sx) > 1e-12 else 1e-12
        sy_safe = sy if abs(sy) > 1e-12 else 1e-12
        kx = float(kx_fn(z))
        ky = float(ky_fn(z))
        K = self._K_full * self._sc_factor
        # σ in m, σ' in rad, K dimensionless, ε² in (m·rad)² → 1/m³
        # rms form: K/[2(σx+σy)] (edge-radius 2K/(X+Y) with X=2σ, ε_X=4ε)
        sc_term = 0.5 * K / (sx_safe + sy_safe) if K else 0.0
        d_sx = sxp
        d_sxp = -kx * sx + self.ex_g ** 2 / sx_safe ** 3 + sc_term
        d_sy = syp
        d_syp = -ky * sy + self.ey_g ** 2 / sy_safe ** 3 + sc_term
        return [d_sx, d_sxp, d_sy, d_syp]

    # ------------------------------------------------------------------
    def run(self) -> SachererResults:
        res = SachererResults()

        s_global_mm = 0.0  # accumulated path length in mm
        sx, sxp, sy, syp = self.sx0, self.sxp0, self.sy0, self.syp0

        # Record initial state.
        res.s.append(s_global_mm)
        res.sigma_x.append(sx * 1e3)        # mm
        res.sigma_y.append(sy * 1e3)
        res.sigma_xp.append(sxp * 1e3)      # mrad
        res.sigma_yp.append(syp * 1e3)
        res.element_names.append("ENTRANCE")

        for elem in self.lattice.elements:
            # SPACE_CHARGE_COMP markers update the effective K factor.
            if isinstance(elem, SpaceChargeComp):
                self._sc_factor = max(0.0, 1.0 - float(elem.factor))
                # Still record so element_names line up with envelope solver.
                res.s.append(s_global_mm)
                res.sigma_x.append(sx * 1e3)
                res.sigma_y.append(sy * 1e3)
                res.sigma_xp.append(sxp * 1e3)
                res.sigma_yp.append(syp * 1e3)
                res.element_names.append(elem.name)
                continue

            kx_fn, ky_fn, L_m = self._kappa_of_element(elem)
            if L_m <= 0:
                # Thin element — record current state and continue.
                res.s.append(s_global_mm)
                res.sigma_x.append(sx * 1e3)
                res.sigma_y.append(sy * 1e3)
                res.sigma_xp.append(sxp * 1e3)
                res.sigma_yp.append(syp * 1e3)
                res.element_names.append(elem.name)
                continue

            t_eval = np.linspace(0.0, L_m, self.n_record)
            sol = solve_ivp(
                self._rhs,
                (0.0, L_m),
                [sx, sxp, sy, syp],
                args=(kx_fn, ky_fn),
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-12,
                max_step=L_m / 50.0,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Sacherer ODE failed in {elem.name}: {sol.message}"
                )
            # Append all but the first point (which equals the previous
            # element's exit, already recorded).
            for i, t in enumerate(sol.t[1:], start=1):
                res.s.append(s_global_mm + t * 1e3)
                res.sigma_x.append(sol.y[0, i] * 1e3)
                res.sigma_y.append(sol.y[2, i] * 1e3)
                res.sigma_xp.append(sol.y[1, i] * 1e3)
                res.sigma_yp.append(sol.y[3, i] * 1e3)
                res.element_names.append(elem.name)
            sx, sxp, sy, syp = sol.y[:, -1]
            s_global_mm += L_m * 1e3

        return res
