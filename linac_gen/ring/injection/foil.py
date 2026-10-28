"""The stripping foil in ring mode: which particles cross it, and what it does
to them.

The physics is HELIX's foil element's (:class:`linac_gen.elements.foil.Foil`):
multiple Coulomb scattering as a Gaussian of the Highland width in each plane
(the core of the distribution — single large-angle and nuclear scattering are
not modelled), the mean ionization loss (the tabulated minimum-ionizing rate,
or Bethe's formula) with Landau or Gaussian straggling, and the two-step
H⁻ → H⁰ → p stripping.  A particle crosses the foil when its absolute
position (x, y) is inside the foil's rectangle (a side left open: no limit
there); the foil is thin, so the position is kept and the angles get the
scattering kicks (added to the slopes, as the linac foil does), then the
energy its loss.  The formulas are evaluated for a proton at the ring's
reference energy of the moment (the stripped ion has the same velocity).

Each pass draws its numbers from its own stream — the run's seed, the
purpose, the turn — in the order of the particles' slots (the scattering in
x, in y, the energy losses, then the stripping), so a CPU and a GPU run, and
a run restarted from a checkpoint, draw the same numbers.

The energy the crossings leave in the foil: each crossing's sampled
ionization loss (the deposit map).  Apart from it, each electron stripped
from an injected ion carries the kinetic energy (gamma - 1) m_e c^2 of its
ion's velocity (:meth:`RingFoil.electron_mev`) — where it goes (the foil, a
catcher) depends on the field around the foil, which is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linac_gen.ring import coords as C

#: outcome of an injected ion's first crossing
STRIPPED = 0
H0 = 1
HMINUS = 2


@dataclass
class FoilSettings:
    """The resolved foil (the settings, filled from the lattice's foil element
    or the defaults)."""

    name: str
    node: int                       # the turn is split here: the foil acts before node `node`
    material: str
    thickness_ug_cm2: float
    straggling: str
    dedx_model: str
    strip_model: str
    x_min_m: float | None
    x_max_m: float | None
    y_min_m: float | None
    y_max_m: float | None
    map_bins: tuple

    @property
    def closed(self) -> bool:
        return None not in (self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m)

    def to_dict(self) -> dict:
        return {"name": self.name, "node": int(self.node), "material": self.material,
                "thickness_ug_cm2": self.thickness_ug_cm2, "straggling": self.straggling,
                "dedx_model": self.dedx_model, "strip_model": self.strip_model,
                "x_min_m": self.x_min_m, "x_max_m": self.x_max_m, "y_min_m": self.y_min_m,
                "y_max_m": self.y_max_m, "map_bins": list(self.map_bins)}


class RingFoil:
    """The foil of a ring run: :meth:`inside`, :meth:`cross` (scattering and
    loss), :meth:`strip` (an injected ion's charge after its first crossing)
    and the hit and deposit maps."""

    def __init__(self, settings: FoilSettings, *, mass_mev: float):
        from linac_gen.elements.foil import Foil
        self.s = settings
        self.mass = float(mass_mev)
        self.elem = Foil(name=settings.name, material=settings.material,
                         thickness_ug_cm2=settings.thickness_ug_cm2,
                         straggling=settings.straggling, strip_model=settings.strip_model,
                         dedx_model=settings.dedx_model)
        self._phys_key = None
        self._phys = None
        nx, ny = settings.map_bins
        self.hit_map = np.zeros((nx, ny)) if settings.closed else None
        self.deposit_map = np.zeros((nx, ny)) if settings.closed else None

    # -- physics at the reference -------------------------------------------
    def _linac_ref(self, kinetic_mev: float, species: str = "proton"):
        from linac_gen.core.particle import H_MINUS, PROTON
        from linac_gen.core.reference import ReferenceParticle
        return ReferenceParticle(H_MINUS if species == "H-" else PROTON, float(kinetic_mev),
                                 100.0)

    def physics(self, kinetic_mev: float) -> dict:
        """theta0 [rad], the mean loss [MeV] and the straggling model for a
        proton of kinetic energy ``kinetic_mev`` (cached for the last
        energy)."""
        key = float(kinetic_mev)
        if self._phys_key != key:
            ref = self._linac_ref(key)
            # (Highland's logarithmic term takes the width below zero for a
            # foil thinner than 4e-12 radiation lengths: no scattering there,
            # at any crossing)
            # (Highland's theta0 is (13.6 MeV / beta c p) |z| sqrt(x/X0) (1 + 0.038
            # ln(x z^2 / (X0 beta^2))), as the foil element computes it: its
            # prefactor, the foil's x/X0 and z^2/beta^2 kept for the widths of
            # repeated crossings)
            self._phys = {"theta0": max(0.0, float(self.elem._highland_theta_rms(ref))),
                          "highland_scale": 13.6 / (ref.beta * ref.bg * ref.species.mass)
                          * abs(ref.species.charge),
                          "x_over_X0": float(self.elem._thickness_over_X0()),
                          "log_scale": ref.species.charge ** 2 / ref.beta ** 2,
                          "mean_loss_mev": float(self.elem._mean_energy_loss_MeV(ref)),
                          "straggling": self.elem.resolved_straggling_model(ref),
                          "ref": ref}
            self._phys_key = key
        return self._phys

    def widths(self, kinetic_mev: float, crossings, n: int) -> np.ndarray:
        """The rms angle kick of each of n particles' next crossing, after
        ``crossings`` (None: none) of the foil already: the growth of
        Highland's theta0^2 from k to k + 1 foil thicknesses, so that a
        particle's crossings add up to Highland at their total thickness
        (theta0^2 is not additive over layers: its logarithm grows with the
        thickness).  A first crossing gets the foil's own theta0; a foil too
        thin for the formula (theta0 0: a foil that only counts its hits)
        scatters at no crossing."""
        ph = self.physics(kinetic_mev)
        sig = np.full(n, ph["theta0"])
        if crossings is None or ph["theta0"] <= 0.0:
            return sig
        k = np.asarray(crossings, dtype=float)
        more = k > 0
        if np.any(more):
            a, A, s = ph["x_over_X0"], ph["highland_scale"], ph["log_scale"]

            def theta2(m):
                # (Highland's theta0^2 at m thicknesses, m >= 1)
                x = m * a
                f = 1.0 + 0.038 * np.log(np.where(x > 0.0, x * s, 1.0))
                return np.where((x > 0.0) & (f > 0.0), A * A * x * f * f, 0.0)

            km = k[more]
            sig[more] = np.sqrt(np.maximum(theta2(km + 1.0) - theta2(km), 0.0))
        return sig

    def electron_mev(self, ion_kinetic_mev: float, species: str = "H-") -> float:
        """The kinetic energy [MeV] of an electron stripped from an ion of
        kinetic energy ``ion_kinetic_mev``: (gamma - 1) m_e c^2, gamma the
        ion's (the electron leaves with its velocity)."""
        from linac_gen.core.constants import M_ELECTRON
        return float((self._linac_ref(ion_kinetic_mev, species).gamma - 1.0) * M_ELECTRON)

    def stripping_fractions(self, kinetic_mev: float) -> tuple[float, float, float]:
        """(f_p, f_H0, f_H-) of an H⁻ of kinetic energy ``kinetic_mev`` after the
        foil (all protons without the two-step model)."""
        if self.s.strip_model != "two_step":
            return (1.0, 0.0, 0.0)
        return tuple(float(v) for v in self.elem.stripping_fractions(
            self._linac_ref(kinetic_mev, "H-")))

    # -- geometry ---------------------------------------------------------------
    @property
    def bounds(self) -> tuple:
        """(x_min, x_max, y_min, y_max) [m], None an open side."""
        s = self.s
        return (s.x_min_m, s.x_max_m, s.y_min_m, s.y_max_m)

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # (kernels.cuda._foil_mask_kernel makes the same comparisons on a GPU:
        # change both together)
        s = self.s
        m = np.ones(np.shape(x), dtype=bool)
        if s.x_min_m is not None:
            m &= x >= s.x_min_m
        if s.x_max_m is not None:
            m &= x <= s.x_max_m
        if s.y_min_m is not None:
            m &= y >= s.y_min_m
        if s.y_max_m is not None:
            m &= y <= s.y_max_m
        return m

    # -- a crossing ---------------------------------------------------------------
    def cross(self, X: np.ndarray, rng, *, p0c: float, beta0: float, kinetic_mev: float,
              crossings=None):
        """Scatter and slow down the particles X (6, n) in place: the slopes
        get Gaussian kicks of the Highland width (of each particle's next
        crossing after ``crossings`` of the foil, None: a first crossing —
        :meth:`widths`), then the energy its sampled loss (the slopes kept).
        Returns the losses [MeV] (n,)."""
        n = X.shape[1]
        ph = self.physics(kinetic_mev)
        sig = self.widths(kinetic_mev, crossings, n)
        draw = bool(np.any(sig > 0.0))
        dxp = rng.normal(0.0, sig) if draw else np.zeros(n)
        dyp = rng.normal(0.0, sig) if draw else np.zeros(n)
        if self.s.thickness_ug_cm2 > 0.0:
            loss = np.asarray(self.elem._sample_energy_loss_MeV(ph["ref"], n, rng), dtype=float)
        else:
            loss = np.zeros(n)
        xp, yp = C.slopes_from_momenta(X[1], X[3], X[5])
        xp = xp + dxp
        yp = yp + dyp
        ptau = C.ptau_from_delta(X[5], beta0) - loss / p0c
        delta = C.delta_from_ptau(ptau, beta0)
        X[1], X[3] = C.momenta_from_slopes(xp, yp, delta)
        X[5] = delta
        return loss

    def strip(self, n: int, rng, *, kinetic_mev: float, species: str) -> np.ndarray:
        """The charge of n injected ions after their first crossing (STRIPPED,
        H0 or HMINUS), one uniform number each; protons, and ions without the
        two-step model, are all STRIPPED."""
        if species != "H-" or self.s.strip_model != "two_step":
            return np.full(n, STRIPPED, dtype=np.int8)
        f_p, f_h0, _f = self.stripping_fractions(kinetic_mev)
        u = rng.uniform(0.0, 1.0, size=n)
        out = np.full(n, STRIPPED, dtype=np.int8)
        out[(u >= f_p) & (u < f_p + f_h0)] = H0
        out[u >= f_p + f_h0] = HMINUS
        return out

    # -- maps -------------------------------------------------------------------------
    def record(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray,
               deposit_mev: np.ndarray) -> None:
        """Add crossings at (x, y) of ``weight`` real particles each, leaving
        ``deposit_mev`` each, to the maps (a closed rectangle only)."""
        if self.hit_map is None or not len(x):
            return
        s = self.s
        nx, ny = s.map_bins
        rng_ = [[s.x_min_m, s.x_max_m], [s.y_min_m, s.y_max_m]]
        h, _, _ = np.histogram2d(x, y, bins=(nx, ny), range=rng_, weights=weight)
        self.hit_map += h
        d, _, _ = np.histogram2d(x, y, bins=(nx, ny), range=rng_, weights=weight * deposit_mev)
        self.deposit_map += d

    def map_extent(self):
        s = self.s
        return None if not s.closed else [s.x_min_m, s.x_max_m, s.y_min_m, s.y_max_m]


def resolve(foil_cfg, rl, *, lattice, species_injected: str | None) -> FoilSettings:
    """The foil of a ring run from the injection's foil settings (or None) and
    the compiled ring: at the element ``at`` — the foil acts at its exit — or
    at the lattice's foil element; a setting left out takes that element's
    value, else the default (carbon; the straggling "auto", the MIP mean loss;
    the two-step stripping for an H⁻ beam).  A lattice foil element's extent
    (half widths about its offset) gives the rectangle's open sides."""
    from linac_gen.elements.foil import Foil
    from linac_gen.ring.injection.config import FoilConfig, InjectionError
    fc = foil_cfg if foil_cfg is not None else FoilConfig()
    foils = [k for k, t in enumerate(rl.element_types) if t == "Foil"]
    # (one foil per ring: a foil element elsewhere would not act)
    elems = list(lattice.elements) if lattice is not None else []
    elem = None
    if fc.at is None:
        if not foils:
            raise InjectionError("injection: the lattice has no foil element — name the element "
                                 "the foil is at (foil 'at')")
        if len(foils) > 1:
            raise InjectionError("injection: the lattice has several foil elements ("
                                 + ", ".join(rl.element_names[k] for k in foils[:4])
                                 + ") — name the one to use (foil 'at')")
        k = foils[0]
    else:
        low = fc.at.lower()
        hits = [k for k, n in enumerate(rl.element_names) if n.lower() == low]
        if not hits:
            raise InjectionError(f"injection: the foil's element {fc.at!r} is not in the ring")
        if len(hits) > 1:
            raise InjectionError(f"injection: the ring has {len(hits)} elements named {fc.at!r} "
                                 "— the foil's must be unique")
        k = hits[0]
    idle = [rl.element_names[j] for j in foils if j != k]
    if idle:
        raise InjectionError(
            "injection: the lattice's foil element(s) " + ", ".join(repr(n) for n in idle[:4])
            + f" would not act — a ring has one foil, here at {rl.element_names[k]!r} (remove "
            "the other foil elements, or put the foil at one of them)")
    if rl.element_types[k] == "Foil":
        named = [e for e in elems if isinstance(e, Foil) and str(e.name) == rl.element_names[k]]
        elem = named[0] if named else None
    if rl.element_last_node[k] >= 0:
        node = int(rl.element_last_node[k]) + 1
    else:
        # (an element with no node of its own: the foil acts after the
        # elements before it)
        prev = [int(j) for j in rl.element_last_node[:k] if j >= 0]
        node = (max(prev) + 1) if prev else 0
    ext = getattr(elem, "extent_mm", None) if elem is not None else None
    ox = float(getattr(elem, "dx", 0.0) or 0.0) * 1e-3 if elem is not None else 0.0
    oy = float(getattr(elem, "dy", 0.0) or 0.0) * 1e-3 if elem is not None else 0.0

    def side(v, default):
        return v if v is not None else default

    rect = (side(fc.x_min_m, None if ext is None else ox - ext[0] * 1e-3),
            side(fc.x_max_m, None if ext is None else ox + ext[0] * 1e-3),
            side(fc.y_min_m, None if ext is None else oy - ext[1] * 1e-3),
            side(fc.y_max_m, None if ext is None else oy + ext[1] * 1e-3))
    for lo, hi, nm in ((rect[0], rect[1], "x"), (rect[2], rect[3], "y")):
        if lo is not None and hi is not None and not hi > lo:
            raise InjectionError(f"injection: the foil's {nm} range is empty "
                                 f"({lo:g} to {hi:g} m)")
    thick = fc.thickness_ug_cm2 if fc.thickness_ug_cm2 is not None else (
        float(elem.thickness_ug_cm2) if elem is not None else None)
    if thick is None:
        raise InjectionError("injection: give the foil's thickness_ug_cm2 (the ring has no foil "
                             "element to take it from)")
    strip_default = "two_step" if species_injected == "H-" else "off"
    return FoilSettings(
        name=rl.element_names[k], node=int(node),
        material=fc.material or (elem.material if elem is not None else "C"),
        thickness_ug_cm2=float(thick),
        straggling=fc.straggling or (elem.straggling if elem is not None else "auto"),
        dedx_model=fc.dedx_model or (getattr(elem, "dedx_model", "mip") if elem is not None
                                     else "mip"),
        strip_model=fc.strip_model or (getattr(elem, "strip_model", "off") if elem is not None
                                       else strip_default),
        x_min_m=rect[0], x_max_m=rect[1], y_min_m=rect[2], y_max_m=rect[3],
        map_bins=tuple(fc.map_bins))


def emittance_growth(theta0: float, beta_m: float) -> float:
    """The rms emittance a beam gains from random, zero-mean angle kicks of
    rms ``theta0`` in a plane with beta function ``beta_m`` at the foil: beta
    theta0^2 / 2 [m rad], whatever alpha is.  After N crossings theta0 is
    Highland's at N times the thickness (:meth:`RingFoil.widths`)."""
    return 0.5 * beta_m * theta0 * theta0
