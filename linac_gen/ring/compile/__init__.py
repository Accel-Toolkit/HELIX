"""HELIX lattice → ring node table (:class:`RingLattice`).

The compiler reads a HELIX :class:`~linac_gen.core.lattice.Lattice` (never
modifying it) and emits, for each element, the kernel nodes that track it:
the element itself, its pole faces, the frame changes of a misalignment or a
vertical bend, and an aperture check at its exit.

**Strengths.**  Magnets are normalized at the rigidity HELIX itself uses for
the lattice's fields (the lattice beam's species and energy, HELIX
constants): a quadrupole of gradient G gets k1 = G/Bρ exactly as HELIX's
linac matrices compute it, so a MAD-X deck's normalized strengths come back
unchanged whatever constants profile the ring reference uses.  Bends are
geometric already (h = 1/ρ, k1 = -N/ρ² from the field index N, k2 as
imported); a field error B(1+δ) scales all of a bend's fields by (1+δ).

**Models.**  ``model = "cartesian"`` (a combined-function bend integrates an
exact curved-frame dipole with the multipoles expanded in Cartesian
coordinates, MAD-X's convention, curvature terms included; the pole faces
are exact by default — see ``edge_model``); ``model = "curvilinear"`` (exact
elements with the bend multipoles in the curved frame, and linear pole
faces).  The names of earlier versions, "madx" and "impactx", are read too.

**Bump magnets** (an injection's ``"dipole"`` magnet groups): a thick kicker
(the importers' half drift, thin kick, half drift), a bend with no angle, or
a drift (a magnet the deck holds with no field) becomes an exact dipole in a
straight frame — the hard-edge dipole fringe at
each face (exact, with the face's angle and gap), the exact map through the
uniform field between — its field set from the kick, k0 = -kick/L, and
written turn by turn by the magnet's waveform.  A stripping foil element is
a marker where the foil acts (:mod:`linac_gen.ring.injection`).

**Refused** (a clear error): solenoids, field maps, RFQ cells, NCELLS and
electrostatic steerers (not modelled in ring mode yet), and a rolled bend.
Lattice commands (matching directives, SET_*), space-charge compensation
cards and grid directives are passive and are skipped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from linac_gen.ring.compile import blocks as B
from linac_gen.ring.config import declared_beam  # noqa: F401  (re-exported)
from linac_gen.ring.kernels import cartesian as cart
from linac_gen.ring.kernels import curvilinear as curv
from linac_gen.ring.kernels import nodes as nd
from linac_gen.ring.reference import RingReference

C_LIGHT = 299792458.0


class RingCompileError(ValueError):
    """A lattice element ring mode cannot track (the message names it)."""


@dataclass
class RingLattice:
    """A ring as kernel nodes.  Positions in metres; ``node_s[j]`` is where
    node j starts, ``node_element[j]`` the HELIX element it belongs to
    (index into ``element_names``)."""

    kinds: np.ndarray
    offsets: np.ndarray
    params: np.ndarray
    node_s: np.ndarray
    node_length: np.ndarray
    node_element: np.ndarray
    element_names: list[str]
    element_types: list[str]
    element_s: np.ndarray            # start of each element [m]
    element_length: np.ndarray       # [m]
    element_last_node: np.ndarray    # last node of each element (-1: none)
    length: float                    # circumference [m]
    reference: RingReference
    brho_norm: float                 # rigidity the strengths are normalized with [T m]
    options: dict
    cavities: list[dict]
    harmonic: int | None
    ref_beta0: np.ndarray
    ref_p0c: np.ndarray
    ref_bg2: np.ndarray
    warnings: list[str] = field(default_factory=list)
    # correctors (steerers): their kick nodes and the kicks at the lattice's
    # rigidity; reference_following: each cavity followed by a reference-change
    # node (compiled for a ramp)
    correctors: list[dict] = field(default_factory=list)
    reference_following: bool = False
    # bump magnets compiled as exact dipoles (an injection's "dipole" groups):
    # element, its nodes (entry face, body, exit face), length, plane, kick, k0
    bumps: list[dict] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return int(self.kinds.size)

    @property
    def revolution_time(self) -> float:
        """Revolution time of the reference particle [s]."""
        return self.length / (self.reference.beta0 * C_LIGHT)

    def node_table(self):
        """(kinds, offsets, params, ref_beta0, ref_p0c, ref_bg2) for the drivers."""
        return (self.kinds, self.offsets, self.params, self.ref_beta0, self.ref_p0c,
                self.ref_bg2)

    def cavity_voltage(self, cavity: dict) -> float:
        """The voltage [MV] a cavity of ``cavities`` has in this ring as it is
        tracked (0 after :meth:`with_rf` (False); its own otherwise)."""
        return float(self.params[self.offsets[cavity["node"]]])

    @property
    def rf_on(self) -> bool:
        """Whether any cavity has a voltage, as tracked."""
        return any(self.cavity_voltage(c) != 0.0 for c in self.cavities)

    def with_rf(self, on: bool) -> RingLattice:
        """A copy whose cavities have their voltage (``on``) or none."""
        p = self.params.copy()
        for c in self.cavities:
            p[self.offsets[c["node"]]] = c["voltage_mv"] if on else 0.0
        return _replace(self, params=p)


def _replace(rl: RingLattice, **kw) -> RingLattice:
    import dataclasses
    return dataclasses.replace(rl, **kw)


# ---------------------------------------------------------------------------
_MODELS = ("cartesian", "curvilinear")
_EDGE_MODELS = ("exact", "full", "linear")


@dataclass
class _Options:
    model: str = "cartesian"
    edge_model: str | None = None        # None: the model's default
    drift_model: str = "exact"           # "exact" | "expanded" (cartesian model)
    bend_model: str = "exact"            # "exact" | "expanded" (cartesian model)
    integrator: str | None = None        # bends; None: yoshida4 (cartesian) / symp6 (curvilinear)
    magnet_integrator: str | None = None  # straight thick magnets; None: uniform / symp6
    bend_steps: int = 0                  # 0: automatic
    magnet_steps: int = 1

    def resolved(self):
        from linac_gen.ring.config import MODEL_ALIASES
        if isinstance(self.model, str):
            self.model = MODEL_ALIASES.get(self.model, self.model)
        if self.model not in _MODELS:
            raise ValueError(f"ring model {self.model!r} is not one of {list(_MODELS)}")
        em = self.edge_model or ("exact" if self.model == "cartesian" else "linear")
        if em not in _EDGE_MODELS:
            raise ValueError(f"edge model {em!r} is not one of {list(_EDGE_MODELS)}")
        if self.model == "curvilinear" and em != "linear":
            raise ValueError("the curvilinear model has linear pole faces only "
                             "(edge_model='linear')")
        integ = self.integrator or ("yoshida4" if self.model == "cartesian" else "symp6")
        minteg = self.magnet_integrator or ("uniform" if self.model == "cartesian" else "symp6")
        for nm, v in (("integrator", integ), ("magnet_integrator", minteg)):
            if v not in B.INTEGRATORS:
                raise ValueError(f"{nm} {v!r} is not one of {sorted(B.INTEGRATORS)}")
        if self.model == "curvilinear" and (integ not in ("symp2", "symp4", "symp6")
                                            or minteg not in ("symp2", "symp4", "symp6")):
            raise ValueError("the curvilinear model integrates with symp2, symp4 or symp6")
        if self.drift_model not in ("exact", "expanded"):
            raise ValueError(f"drift_model {self.drift_model!r} is not 'exact' or 'expanded'")
        if self.bend_model not in ("exact", "exact-bend", "expanded"):
            raise ValueError(f"bend_model {self.bend_model!r} is not one of 'exact', "
                             "'exact-bend', 'expanded'")
        if int(self.bend_steps) < 0 or int(self.magnet_steps) < 1:
            raise ValueError("bend_steps >= 0 (0 = automatic) and magnet_steps >= 1")
        return dict(model=self.model, edge_model=em,
                    drift_model="exact" if self.model == "curvilinear" else self.drift_model,
                    bend_model=self.bend_model, integrator=integ, magnet_integrator=minteg,
                    bend_steps=int(self.bend_steps), magnet_steps=int(self.magnet_steps))


def auto_bend_steps(length: float, h: float, k1: float, integrator: str) -> int:
    """Integration steps of a bend when none are given: enough that the
    focusing phase and the bend angle per step stay small (about 0.05 rad and
    10 mrad), for an integrator error far below the tune accuracy of the
    optics (pinned by tests/ring/test_optics_booster.py)."""
    phase = abs(length) * math.sqrt(abs(k1) + h * h)
    n = max(1, math.ceil(phase / 0.05), math.ceil(abs(h * length) / 0.01))
    if integrator in ("teapot", "uniform", "symp2"):
        n *= 4                       # (second order)
    return int(n)


def _n_kicks(integrator: str, steps: int) -> int:
    """Xsuite's num_multipole_kicks for ``steps`` integration steps."""
    return 7 * steps if integrator == "yoshida4" else steps


def lattice_rigidity(species: str, w_kin: float) -> float:
    """The signed rigidity HELIX uses for a lattice's fields: HELIX constants,
    the lattice beam's species and kinetic energy [MeV] (T m)."""
    from linac_gen.core.particle import H_MINUS, PROTON
    from linac_gen.core.reference import ReferenceParticle
    sp = {"proton": PROTON, "p": PROTON, "h-": H_MINUS}.get(str(species).lower())
    if sp is None:
        raise ValueError(f"ring mode: unknown lattice species {species!r}")
    ref = ReferenceParticle(sp, float(w_kin), 1.0)
    return math.copysign(ref.brho, sp.charge)


# ---------------------------------------------------------------------------
def lattice_beam(lattice, beam=None) -> tuple[str, float]:
    """(species, kinetic energy [MeV]) of the beam the lattice's fields were
    set for: a MAD-X deck's BEAM (``lattice.ring_reference_beam``, kept by the
    GUI and CLI loaders), else ``beam`` (a BeamConfig: the Beam tab's)."""
    rb = getattr(lattice, "ring_reference_beam", None)
    if rb:
        return str(rb["species"]), float(rb["energy"])
    if beam is not None:
        return str(beam.species), float(beam.energy)
    raise ValueError("the lattice carries no reference beam (a MAD-X deck's BEAM): "
                     "give the beam its magnets were set for")


def compile_from_config(lattice, config, *, beam=None) -> RingLattice:
    """Compile ``lattice`` with the settings of a
    :class:`~linac_gen.ring.config.RingConfig`: the ring reference is the
    config's species at ``design_energy_mev`` (default: the lattice beam's
    energy — for a deck that states no beam, ``beam``'s) with the config's
    constants; the magnets keep the normalized strengths of the lattice beam
    (they follow the energy)."""
    from linac_gen.ring.detect import detect_ring
    species, w_lat = lattice_beam(lattice, beam)
    declared = (getattr(lattice, "ring_reference_beam", None) is None
                or declared_beam(lattice) is not None)
    w_run = w_lat
    if not declared and not config.design_energy_mev:
        # (the importer's default beam only scaled the magnets: the ring runs
        # at the beam it is given)
        if beam is None:
            raise ValueError("the deck states no beam (its magnets keep the normalized "
                             "strengths it gives): give the beam the ring runs at")
        w_run = float(beam.energy)
    w = float(config.design_energy_mev) if config.design_energy_mev else w_run
    ref = RingReference.from_kinetic(config.species, w, constants=config.constants)
    harmonic = config.harmonic
    if harmonic is None:
        lg = dict(getattr(lattice, "lg_options", {}) or {})
        if "harmonic" in lg:
            harmonic = detect_ring(lattice).harmonic
    return compile_ring(lattice, ref, lattice_species=species, lattice_energy=w_lat,
                        lattice_declared=declared, harmonic=harmonic,
                        reference_following=getattr(config, "ramp", None) is not None,
                        corrector_mode=getattr(config, "corrector_mode", "follow"),
                        bumps=bump_specs(getattr(config, "injection", None)),
                        **config.element_options())


def bump_specs(injection) -> dict | None:
    """The bump magnets an injection compiles as exact dipoles: element name
    (lower case) → its group's plane, kick, fringe and face angles (None
    without a "dipole" group)."""
    if not injection:
        return None
    from linac_gen.ring.injection.config import InjectionConfig
    ic = injection if isinstance(injection, InjectionConfig) else InjectionConfig.from_dict(
        injection)
    out = {}
    for g in ic.magnets:
        if g.model != "dipole":
            continue
        for k, name in enumerate(g.elements):
            out[name.lower()] = {"group": g.name, "plane": g.plane,
                                 "kick": None if g.kick_rad is None else g.kick_rad[k],
                                 "fint": g.fint, "hgap_m": g.hgap_m, "e1_rad": g.e1_rad,
                                 "e2_rad": g.e2_rad}
    return out or None


# ---------------------------------------------------------------------------
def compile_ring(lattice, reference: RingReference, *, lattice_species: str = "proton",
                 lattice_energy: float | None = None, lattice_declared: bool = True,
                 harmonic: int | None = None,
                 reference_following: bool = False, corrector_mode: str = "follow",
                 bumps: dict | None = None, **options) -> RingLattice:
    """Compile ``lattice`` for the ring reference ``reference``.

    ``lattice_species``/``lattice_energy`` describe the beam the lattice's
    fields were set for (default: the ring reference's energy);
    ``lattice_declared`` False — the deck states no beam, those are its
    importer's defaults (no note about them); ``harmonic``
    locks every cavity to h·f_rev (default: each cavity's own harmonic number
    when it has one, else its frequency); ``reference_following`` puts a
    reference-change node after every cavity (a ramp: the reference particle
    gains energy there); ``corrector_mode`` — "follow": a steerer keeps its
    normalized kick (its field follows the reference momentum), or
    "fixed_field": it keeps its field, so its kick scales as the lattice
    beam's momentum over the reference's.  ``bumps`` — element name (lower
    case) → plane, kick, fint, hgap_m, e1_rad, e2_rad of the bump magnets to
    compile as exact dipoles (:func:`bump_specs`).  ``options`` — model,
    edge_model, drift_model, bend_model, integrator, magnet_integrator,
    bend_steps, magnet_steps — see :class:`_Options`.
    """
    if corrector_mode not in ("follow", "fixed_field"):
        raise ValueError(f"corrector_mode {corrector_mode!r} is not 'follow' or 'fixed_field'")
    opts = _Options(**options).resolved()
    w_lat = reference.kinetic0 if lattice_energy is None else float(lattice_energy)
    brho = lattice_rigidity(lattice_species, w_lat)
    warnings: list[str] = []
    if lattice_declared and str(lattice_species).lower() not in ("proton", "p"):
        warnings.append(
            f"the lattice's magnets are scaled for {lattice_species} at {w_lat:g} MeV; ring mode "
            "tracks protons with the same normalized strengths (k = field / signed rigidity)")
    b = _Builder(opts, reference, brho, warnings, reference_following=bool(reference_following),
                 corrector_mode=corrector_mode, bumps=bumps)
    elements = list(lattice.elements)
    i = 0
    while i < len(elements):
        consumed = b.element(elements, i)
        i += consumed
    rl = b.finish(elements, harmonic)
    if bumps:
        done = {d["element"].lower() for d in rl.bumps}
        missing = sorted(n for n in bumps if n not in done)
        if missing:
            raise RingCompileError(
                "bump magnet(s) " + ", ".join(repr(n) for n in missing[:5])
                + " not found in the ring as a thick kicker, a bend with no angle or a drift "
                "(the 'dipole' model needs one of these)")
    return rl


class _Builder:
    def __init__(self, opts, reference, brho, warnings, *, reference_following=False,
                 corrector_mode="follow", bumps=None):
        self.o = opts
        self.ref = reference
        self.brho = brho
        self.warnings = warnings
        self.reference_following = reference_following
        self.corrector_mode = corrector_mode
        self.bump_specs = dict(bumps or {})
        self.bumps: list[dict] = []
        self.correctors: list[dict] = []
        self.kinds: list[int] = []
        self.offsets: list[int] = []
        self.params: list[float] = []
        self.node_len: list[float] = []
        self.node_elem: list[int] = []
        self.elem_names: list[str] = []
        self.elem_types: list[str] = []
        self.elem_len: list[float] = []
        self.elem_last: list[int] = []
        self.cavities: list[dict] = []
        self._ignored: dict[str, int] = {}
        self._pitch_yaw: list[str] = []
        self._cur = -1

    # ---- node emission ------------------------------------------------------
    def _node(self, kind, blk, length=0.0):
        self.kinds.append(int(kind))
        self.offsets.append(len(self.params))
        self.params.extend(float(v) for v in blk)
        self.node_len.append(float(length))
        self.node_elem.append(self._cur)
        return len(self.kinds) - 1

    def _begin(self, e, length_m):
        self.elem_names.append(str(e.name))
        self.elem_types.append(type(e).__name__)
        self.elem_len.append(float(length_m))
        self.elem_last.append(-1)
        self._cur = len(self.elem_names) - 1

    def _end(self):
        if self.kinds and self.node_elem[-1] == self._cur:
            self.elem_last[self._cur] = len(self.kinds) - 1

    # ---- misalignment and apertures -----------------------------------------
    def _misalign_in(self, dx_m, dy_m, roll):
        if dx_m or dy_m:
            self._node(nd.N_XYSHIFT, B.xyshift(dx_m, dy_m))
        if roll:
            self._node(nd.N_SROTATION, B.srotation(roll))

    def _misalign_out(self, dx_m, dy_m, roll):
        if roll:
            self._node(nd.N_SROTATION, B.srotation(-roll))
        if dx_m or dy_m:
            self._node(nd.N_XYSHIFT, B.xyshift(-dx_m, -dy_m))

    def _check_unmodelled_dofs(self, e):
        if (getattr(e, "dz", 0.0) or getattr(e, "pitch_deg", 0.0)
                or getattr(e, "yaw_deg", 0.0)):
            self._pitch_yaw.append(str(e.name))

    def _element_aperture(self, e, x0=0.0, y0=0.0):
        """The element's aperture (centred at ``x0``, ``y0`` [m]: a shifted pipe)."""
        r = float(getattr(e, "aperture", 0.0) or 0.0)
        if r <= 0.0:
            return
        ry = getattr(e, "aperture_y", None)
        if ry is not None and ry > 0.0:
            self._node(nd.N_APERTURE, B.aperture("rectangle", r * 1e-3, float(ry) * 1e-3,
                                                 x0=x0, y0=y0))
        else:
            self._node(nd.N_APERTURE, B.aperture("circle", r * 1e-3, x0=x0, y0=y0))

    # ---- the elements ---------------------------------------------------------
    def element(self, elements, i) -> int:
        from linac_gen.elements.aperture import Aperture
        from linac_gen.elements.dipole import Dipole
        from linac_gen.elements.drift import Drift
        from linac_gen.elements.edge import Edge
        from linac_gen.elements.marker import Marker
        from linac_gen.elements.matrix_element import MatrixElement
        from linac_gen.elements.multipole import Multipole
        from linac_gen.elements.quadrupole import Quadrupole
        from linac_gen.elements.rf_gap import RFGap
        from linac_gen.elements.steerer import Steerer
        from linac_gen.elements.thin_lens import ThinLens

        e = elements[i]
        o = self.o
        L = float(getattr(e, "length", 0.0) or 0.0) * 1e-3
        if self.bump_specs:
            n = self._bump_magnet(elements, i)
            if n:
                return n
        if type(e).__name__ == "Foil":
            # (the stripping foil: where the turn stops for it — linac_gen.ring.injection)
            self._begin(e, L)
            if L:
                raise RingCompileError(f"foil {e.name!r} has a length: a foil is thin")
            self._node(nd.N_MARKER, [])
            self._element_aperture(e)
            self._end()
            return 1
        # (curvilinear model) a thick multipole the importer wrote as half drift,
        # thin kick, half drift
        if o["model"] == "curvilinear" and isinstance(e, Drift) and str(e.name).endswith("_din"):
            n = self._thick_multipole(elements, i)
            if n:
                return n
        if isinstance(e, Drift):
            self._begin(e, L)
            self._node(nd.N_DRIFT, B.drift(L, o["drift_model"] == "exact"), L)
            # (a drift's offset and roll move its pipe: the linac checks the
            # aperture in that frame; the drift itself does not see them)
            dx = float(getattr(e, "dx", 0.0) or 0.0) * 1e-3
            dy = float(getattr(e, "dy", 0.0) or 0.0) * 1e-3
            roll = math.radians(float(getattr(e, "tilt_deg", 0.0) or 0.0))
            rolled = roll and float(getattr(e, "aperture", 0.0) or 0.0) > 0.0
            if rolled:
                self._misalign_in(dx, dy, roll)
                self._element_aperture(e)
                self._misalign_out(dx, dy, roll)
            else:
                self._element_aperture(e, x0=dx, y0=dy)
            self._end()
            return 1
        if isinstance(e, Quadrupole):
            self._quadrupole(e, L)
            return 1
        if isinstance(e, Dipole):
            self._dipole(e, L, elements, i)
            return 1
        if isinstance(e, Edge):
            self._edge(e, elements, i)
            return 1
        if isinstance(e, Multipole):
            self._begin(e, 0.0)
            dx, dy = e.dx * 1e-3, e.dy * 1e-3
            roll = math.radians(e.tilt_deg)
            self._misalign_in(dx, dy, roll)
            self._node(nd.N_MULTIPOLE, B.multipole(e.knl, e.ksl))
            self._misalign_out(dx, dy, roll)
            self._element_aperture(e)
            self._end()
            return 1
        if isinstance(e, RFGap):
            self._cavity(e)
            return 1
        if isinstance(e, Steerer):
            if getattr(e, "elec", False):
                raise RingCompileError(
                    f"element {e.name!r}: electrostatic steerers are not modelled in ring mode")
            self._begin(e, 0.0)
            kx, ky = e.by_l / self.brho, e.bx_l / self.brho
            # (a corrector that keeps its field: its kick at the ring's
            # reference scales as the lattice beam's rigidity over the ring's)
            scale = (abs(self.brho) / self.ref.brho
                     if self.corrector_mode == "fixed_field" else 1.0)
            j = self._node(nd.N_KICK, B.kick(kx * scale, ky * scale))
            self.correctors.append({"node": j, "name": str(e.name), "kick": (kx, ky)})
            self._end()
            return 1
        if isinstance(e, ThinLens):
            self._begin(e, 0.0)
            ifx = 0.0 if math.isinf(e.fx) else 1.0 / e.fx
            ify = 0.0 if math.isinf(e.fy) else 1.0 / e.fy
            self._node(nd.N_THINLENS, B.thin_lens(ifx, ify))
            self._element_aperture(e)
            self._end()
            return 1
        if isinstance(e, Aperture):
            self._begin(e, 0.0)
            t = int(e.aperture_type)
            if t == Aperture.CIRCULAR:
                self._node(nd.N_APERTURE, B.aperture("circle", e.dx * 1e-3))
            elif t in (Aperture.RECTANGULAR, Aperture.FRACTION, Aperture.FINGER_H,
                       Aperture.FINGER_V):
                self._node(nd.N_APERTURE, B.aperture("rectangle", e.dx * 1e-3, e.dy * 1e-3))
            else:
                self._ignored[f"aperture type {t} (not simulated, as in linac tracking)"] = \
                    self._ignored.get(f"aperture type {t} (not simulated, as in linac tracking)", 0) + 1
                self._node(nd.N_MARKER, [])
            self._end()
            return 1
        if isinstance(e, Marker):
            self._begin(e, 0.0)
            self._node(nd.N_MARKER, [])
            self._end()
            return 1
        if isinstance(e, MatrixElement):
            self._matrix(e, L)
            return 1
        return self._other(e)

    def _other(self, e) -> int:
        from linac_gen.elements.base import PassiveElement
        from linac_gen.elements.lattice_commands import LatticeCommand
        name = type(e).__name__
        refused = {
            "Solenoid": "solenoids", "FieldMap": "field maps", "FieldMap3D": "field maps",
            "SuperposedFieldMap": "field maps", "RfqCell": "RFQ cells", "VaneRFQ": "RFQs",
            "NCells": "NCELLS cavities",
        }
        if name in refused:
            raise RingCompileError(
                f"element {e.name!r} ({name}): {refused[name]} are not modelled in ring mode")
        if isinstance(e, (LatticeCommand, PassiveElement)):
            self._begin(e, 0.0)
            self._node(nd.N_MARKER, [])
            self._end()
            self._ignored[name] = self._ignored.get(name, 0) + 1
            if float(getattr(e, "length", 0.0) or 0.0):
                raise RingCompileError(f"element {e.name!r} ({name}) has a length but no map")
            return 1
        raise RingCompileError(f"element {e.name!r}: {name} is not modelled in ring mode")

    # ---- quadrupole -----------------------------------------------------------
    def _quadrupole(self, e, L):
        o = self.o
        self._begin(e, L)
        self._check_unmodelled_dofs(e)
        k1 = e.effective_gradient / self.brho
        # higher multipoles: k_n = g_{n+1} / Bρ (HELIX's quadrupole convention)
        kh = [e.g3 / self.brho, e.g4 / self.brho, e.g5 / self.brho, e.g6 / self.brho]
        roll = math.radians(float(e.skew_angle) + float(e.tilt_deg))
        dx, dy = e.dx * 1e-3, e.dy * 1e-3
        self._misalign_in(dx, dy, roll)
        if L == 0.0:
            self.warnings.append(f"quadrupole {e.name!r} has no length: no focusing "
                                 "(as in linac tracking)")
            self._node(nd.N_MARKER, [])
        elif o["model"] == "curvilinear":
            integ = B.INTEGRATORS[o["magnet_integrator"]]
            if any(kh):
                self._node(nd.N_CURVILINEAR, B.curvilinear_multipole(
                    L, [0.0, k1] + kh, [0.0] * 6, integrator=integ, nsteps=o["magnet_steps"]), L)
            else:
                self._node(nd.N_CURVILINEAR, B.curvilinear_quad(
                    L, k1, integrator=integ, nsteps=o["magnet_steps"]), L)
        else:
            cfg = B.magnet_config("mat-kick-mat", 0.0, k1, 0.0)
            integ = o["magnet_integrator"]
            self._node(nd.N_MAGNET, B.magnet(
                L, integrator=B.INTEGRATORS[integ],
                n_kicks=_n_kicks(integ, o["magnet_steps"]),
                knl=[0.0, 0.0] + [k * L for k in kh], **cfg), L)
        # (the aperture moves with the magnet: checked in its frame, as the
        # linac checks it)
        self._element_aperture(e)
        self._misalign_out(dx, dy, roll)
        self._end()

    # ---- bends and pole faces -------------------------------------------------
    def _bend_strengths(self, e):
        """(L, h, k0, k1, k2) of a HELIX Dipole, ring units: h = θ/L signed by
        the angle; the field k0 = h unless the deck gave its own k0 (MAD-X
        K0); a field error B(1+δ) scales k0, k1 and k2."""
        L = float(e.length) * 1e-3
        if L == 0.0 or e.angle == 0.0:
            return L, 0.0, 0.0, 0.0, 0.0
        rho = abs(float(e.rho)) * 1e-3
        theta = math.radians(float(e.angle))
        h = theta / L
        f = 1.0 + float(getattr(e, "field_rel", 0.0) or 0.0)
        k0 = getattr(e, "k0", None)
        k0 = (float(k0) if k0 is not None else h) * f
        k1 = -float(e.field_index) / (rho * rho) * f
        k2 = float(getattr(e, "k2", 0.0) or 0.0) * f
        return L, h, k0, k1, k2

    def _dipole(self, e, L, elements, i):
        from linac_gen.elements.edge import Edge
        o = self.o
        self._begin(e, L)
        if getattr(e, "tilt_deg", 0.0):
            raise RingCompileError(
                f"element {e.name!r}: a rolled bend (tilt {e.tilt_deg} deg) is not modelled in "
                "ring mode yet")
        self._check_unmodelled_dofs(e)
        L, h, k0, k1, k2 = self._bend_strengths(e)
        vertical = int(getattr(e, "hv", 0)) == 1
        dx, dy = e.dx * 1e-3, e.dy * 1e-3
        self._misalign_in(dx, dy, 0.0)
        if vertical:
            self._node(nd.N_SROTATION, B.srotation(0.5 * math.pi))
        # the element's own pole faces: HELIX's tan(e)/ρ with the signed ρ
        # scaled by a field error (Dipole.transfer_matrix)
        eff = e.effective_angle
        if eff == e.angle:
            rho_face = float(e.rho) * 1e-3
        elif eff == 0.0:
            rho_face = 0.0                    # (no field: no pole-face kicks)
        else:
            rho_face = float(e.rho) * (float(e.angle) / eff) * 1e-3
        # (exact pole faces carry the hard-edge dipole fringe, also at a face
        # angle of zero: a bend with no pole face of its own and no EDGE
        # element next to it gets zero-angle faces)
        fringe_only = (o["model"] == "cartesian" and o["edge_model"] != "linear" and h != 0.0
                       and k0 != 0.0)
        edge_before = i > 0 and isinstance(elements[i - 1], Edge)
        edge_after = i + 1 < len(elements) and isinstance(elements[i + 1], Edge)
        if e.e1 and rho_face:
            self._face(math.radians(e.e1), rho_face, 0.0, 0.0, False, k0, k1, k2, L)
        elif fringe_only and not edge_before:
            self._face(0.0, 1.0 / h, 0.0, 0.0, False, k0, k1, k2, L)
        if L == 0.0:
            pass
        elif h == 0.0:
            self._node(nd.N_DRIFT, B.drift(L, o["drift_model"] == "exact"), L)
        elif o["model"] == "curvilinear":
            steps = o["bend_steps"] or auto_bend_steps(L, h, k1, o["integrator"])
            self._node(nd.N_CURVILINEAR, B.curvilinear_cfbend(
                L, 1.0 / h, [k0 - h, k1, k2], [0.0, 0.0, 0.0],
                integrator=B.INTEGRATORS[o["integrator"]], nsteps=steps), L)
        else:
            mdl = {"exact": "rot-kick-rot", "exact-bend": "bend-kick-bend",
                   "expanded": "mat-kick-mat"}[o["bend_model"]]
            integ = o["integrator"]
            steps = o["bend_steps"] or auto_bend_steps(L, h, k1, integ)
            cfg = B.magnet_config(mdl, k0, k1, h)
            self._node(nd.N_MAGNET, B.magnet(
                L, integrator=B.INTEGRATORS[integ], n_kicks=_n_kicks(integ, steps), k2=k2,
                **cfg), L)
        if e.e2 and rho_face:
            self._face(math.radians(e.e2), rho_face, 0.0, 0.0, True, k0, k1, k2, L)
        elif fringe_only and not edge_after:
            self._face(0.0, 1.0 / h, 0.0, 0.0, True, k0, k1, k2, L)
        if vertical:
            self._node(nd.N_SROTATION, B.srotation(-0.5 * math.pi))
        self._element_aperture(e)          # (in the magnet's frame, as in the linac)
        self._misalign_out(dx, dy, 0.0)
        self._end()

    def _face(self, beta, rho_m, gap_m, fint, is_exit, k0, k1, k2, mag_len):
        """One pole face in HELIX's convention: kick tan(beta)/rho with the
        signed radius ``rho_m`` [m] (the importers write beta = sign(θ)·e and
        rho > 0, i.e. MAD-X's h·tan e), full gap ``gap_m`` [m] and fringe
        integral ``fint``; ``k0, k1, k2, mag_len``: the bend it belongs to
        (k0 = 0: none)."""
        o = self.o
        kface = 1.0 / rho_m
        if o["model"] == "curvilinear":
            r21, r43 = curv.dipedge_linear_coefficients(beta, rho_m, gap_m, fint)
            self._node(nd.N_EDGE_LINEAR, B.edge_linear(r21, r43))
        elif o["edge_model"] == "linear" or not k0:
            r21, r43 = cart.edge_linear_coefficients(kface, beta, 0.0, 0.5 * gap_m, fint)
            self._node(nd.N_EDGE_LINEAR, B.edge_linear(r21, r43))
        else:
            # Xsuite's face angle e and field k0 (MAD-X signs): k0 tan e = tan(beta)/rho
            e_angle = beta if k0 * rho_m > 0.0 else -beta
            full = o["edge_model"] == "full"
            self._node(nd.N_EDGE_EXACT, B.edge_exact(
                is_exit=is_exit, angle=e_angle, k0_fringe=k0, k0_wedge=k0, k1=k1, fint=fint,
                hgap=0.5 * gap_m, quad_wedge=1 if full else 2, mult_fringe=full,
                kn=(k0, k1, k2, 0.0), length=mag_len))

    def _edge(self, e, elements, i):
        from linac_gen.elements.dipole import Dipole
        self._begin(e, 0.0)
        nxt = elements[i + 1] if i + 1 < len(elements) else None
        prv = elements[i - 1] if i > 0 else None
        if isinstance(nxt, Dipole):
            mag, is_exit = nxt, False
        elif isinstance(prv, Dipole):
            mag, is_exit = prv, True
        else:
            mag, is_exit = None, False
        k0 = k1 = k2 = L = 0.0
        if mag is not None:
            L, _h, k0, k1, k2 = self._bend_strengths(mag)
        elif self.o["model"] == "cartesian" and self.o["edge_model"] != "linear":
            self.warnings.append(
                f"edge {e.name!r} is not next to a bend: tracked as a linear pole face")
        vertical = int(getattr(e, "hv", 0)) == 1
        if vertical:
            self._node(nd.N_SROTATION, B.srotation(0.5 * math.pi))
        exact_face = (mag is not None and k0 != 0.0 and self.o["model"] == "cartesian"
                      and self.o["edge_model"] != "linear")
        rho = float(e.rho) * 1e-3 if e.rho else (1.0 / k0 if k0 else 0.0)
        gap = float(e.gap) * 1e-3
        fint = float(e.k1) if gap > 0.0 else 0.0
        # (a face at zero angle with a gap and a fringe integral still acts
        # — weakly defocusing across the bend plane — in every edge model:
        # MAD-X's -h tan(-psi))
        if (e.pole_rotation or exact_face or fint) and rho:
            self._face(math.radians(e.pole_rotation), rho, gap, fint, is_exit, k0, k1, k2, L)
        else:
            self._node(nd.N_MARKER, [])
        if vertical:
            self._node(nd.N_SROTATION, B.srotation(-0.5 * math.pi))
        self._end()

    # ---- thick multipole (curvilinear model) --------------------------------
    def _thick_multipole(self, elements, i) -> int:
        """Half drift, thin multipole, half drift (how the importers write a
        thick MAD-X multipole) → one thick ImpactX multipole, as ImpactX models
        it.  The three HELIX elements report the thick element's exit."""
        from linac_gen.elements.drift import Drift
        from linac_gen.elements.multipole import Multipole
        if i + 2 >= len(elements):
            return 0
        din, mid, dout = elements[i], elements[i + 1], elements[i + 2]
        base = str(din.name)[:-4]
        if not (isinstance(mid, Multipole) and isinstance(dout, Drift)
                and str(mid.name) == base and str(dout.name) == base + "_dout"
                and abs(din.length - dout.length) <= 1e-12 * max(1.0, din.length)
                and not (din.aperture or dout.aperture or mid.aperture)):
            return 0
        L = (din.length + dout.length) * 1e-3
        self._begin(din, din.length * 1e-3)
        first = self._cur
        self._begin(mid, 0.0)
        dx, dy = mid.dx * 1e-3, mid.dy * 1e-3
        roll = math.radians(mid.tilt_deg)
        self._misalign_in(dx, dy, roll)
        kn = [v / L for v in mid.knl]
        ks = [v / L for v in mid.ksl]
        n = max(len(kn), len(ks), 1)
        kn += [0.0] * (n - len(kn))
        ks += [0.0] * (n - len(ks))
        self._node(nd.N_CURVILINEAR, B.curvilinear_multipole(
            L, kn, ks, integrator=B.INTEGRATORS[self.o["magnet_integrator"]],
            nsteps=self.o["magnet_steps"]), L)
        self._misalign_out(dx, dy, roll)
        self._begin(dout, dout.length * 1e-3)
        last = len(self.kinds) - 1
        for k in (first, first + 1, first + 2):
            self.elem_last[k] = last
        return 3

    # ---- bump magnets (an injection's "dipole" groups) ----------------------------
    def _bump_magnet(self, elements, i) -> int:
        """A bump magnet at elements[i] compiled as an exact dipole (the number
        of HELIX elements it covers), or 0 when elements[i] is not one."""
        from linac_gen.elements.dipole import Dipole
        from linac_gen.elements.drift import Drift
        from linac_gen.elements.steerer import Steerer
        e = elements[i]
        name = str(e.name)
        if isinstance(e, Drift) and name.endswith("_din") and i + 2 < len(elements):
            mid, dout = elements[i + 1], elements[i + 2]
            base = name[:-4]
            spec = self.bump_specs.get(base.lower())
            if (spec is not None and isinstance(mid, Steerer) and str(mid.name) == base
                    and isinstance(dout, Drift) and str(dout.name) == base + "_dout"
                    and abs(e.length - dout.length) <= 1e-12 * max(1.0, e.length)):
                if getattr(mid, "elec", False):
                    raise RingCompileError(f"element {base!r}: an electrostatic kicker is not a "
                                           "bump dipole")
                L = (e.length + dout.length) * 1e-3
                self._begin(e, e.length * 1e-3)
                first = self._cur
                self._begin(mid, 0.0)
                kx, ky = mid.by_l / self.brho, mid.bx_l / self.brho
                self._emit_bump(mid, L, kx, ky, spec)
                for el in (e, mid, dout):
                    self._element_aperture(el)
                self._begin(dout, dout.length * 1e-3)
                last = len(self.kinds) - 1
                for k in (first, first + 1, first + 2):
                    self.elem_last[k] = last
                return 3
        spec = self.bump_specs.get(name.lower())
        if spec is None:
            return 0
        if isinstance(e, Dipole) and float(e.angle) == 0.0 and float(e.length) > 0.0:
            if getattr(e, "tilt_deg", 0.0):
                raise RingCompileError(f"element {name!r}: a rolled bump dipole is not "
                                       "modelled in ring mode yet")
            L = float(e.length) * 1e-3
            self._begin(e, L)
            k0 = getattr(e, "k0", None)
            kick = -float(k0) * L if k0 is not None else 0.0
            vertical = int(getattr(e, "hv", 0)) == 1
            self._emit_bump(e, L, 0.0 if vertical else kick, kick if vertical else 0.0, spec,
                            plane_default="y" if vertical else "x")
            self._element_aperture(e)
            self._end()
            return 1
        if isinstance(e, Drift) and float(e.length) > 0.0:
            # (a magnet the deck holds with no field — a drift, or a bend with
            # neither angle nor field read as one: its kick from the group)
            if spec.get("kick") is None or spec.get("plane") is None:
                raise RingCompileError(f"element {name!r} is a drift: as a bump dipole it needs "
                                       "the group's plane and kick_rad (it has no field of its "
                                       "own)")
            L = float(e.length) * 1e-3
            self._begin(e, L)
            self._emit_bump(e, L, 0.0, 0.0, spec)
            self._element_aperture(e)
            self._end()
            return 1
        if isinstance(e, Steerer):
            raise RingCompileError(f"element {name!r}: a kicker with no length cannot be a bump "
                                   "dipole (the 'dipole' model needs its length)")
        raise RingCompileError(f"element {name!r} ({type(e).__name__}): the 'dipole' model needs "
                               "a thick kicker, a bend with no angle or a drift")

    def _emit_bump(self, e, L, kx, ky, spec, plane_default=None):
        """Entry face, exact straight-frame dipole, exit face for a deflection
        ``kick`` in ``plane`` (x: k0 = -kick/L; y: the same in the frame turned
        by 90 degrees, where the field deflects in "x")."""
        plane = spec.get("plane") or plane_default
        if plane is None:
            if kx and ky:
                raise RingCompileError(f"element {e.name!r}: a bump dipole deflects in one plane "
                                       "(this kicker kicks in both) — give the group's plane")
            if not kx and not ky:
                raise RingCompileError(f"element {e.name!r} kicks in neither plane — give the "
                                       "group's plane and kick_rad")
            plane = "x" if kx else "y"
        kick = spec.get("kick")
        if kick is None:
            kick = kx if plane == "x" else ky
            other = ky if plane == "x" else kx
            if other:
                raise RingCompileError(
                    f"element {e.name!r} kicks in the {'y' if plane == 'x' else 'x'} plane — a "
                    f"bump dipole deflects in its group's plane only ({plane}): give kick_rad")
            if not kick:
                raise RingCompileError(f"element {e.name!r} has no field of its own — as a bump "
                                       "dipole it needs the group's kick_rad")
        kick = float(kick)
        k0 = -kick / L
        fint, hgap = float(spec.get("fint", 0.0)), float(spec.get("hgap_m", 0.0))
        e1, e2 = float(spec.get("e1_rad", 0.0)), float(spec.get("e2_rad", 0.0))
        if plane == "y":
            self._node(nd.N_SROTATION, B.srotation(0.5 * math.pi))
        j_in = self._node(nd.N_EDGE_EXACT, B.edge_exact(
            is_exit=False, angle=e1, k0_fringe=k0, k0_wedge=k0, k1=0.0, fint=fint, hgap=hgap,
            quad_wedge=0, mult_fringe=False, kn=(k0, 0.0, 0.0, 0.0), length=L))
        cfg = B.magnet_config("bend-kick-bend", k0, 0.0, 0.0)
        j_body = self._node(nd.N_MAGNET, B.magnet(L, integrator=cart.INT_UNIFORM, n_kicks=0,
                                                  **cfg), L)
        j_out = self._node(nd.N_EDGE_EXACT, B.edge_exact(
            is_exit=True, angle=e2, k0_fringe=k0, k0_wedge=k0, k1=0.0, fint=fint, hgap=hgap,
            quad_wedge=0, mult_fringe=False, kn=(k0, 0.0, 0.0, 0.0), length=L))
        if plane == "y":
            self._node(nd.N_SROTATION, B.srotation(-0.5 * math.pi))
        self.bumps.append({"element": str(e.name), "element_index": self._cur,
                           "nodes": [j_in, j_body, j_out], "length": L, "plane": plane,
                           "kick": kick, "k0": k0, "group": spec.get("group")})

    # ---- RF ---------------------------------------------------------------------
    def _cavity(self, e):
        self._begin(e, 0.0)
        # (the energy gain of the lattice's beam: its charge signs the
        # voltage, as the signed rigidity signs the magnets and as RFGap's
        # charge * V does)
        ttf = getattr(e, "ttf", None)
        v = float(e.voltage) * (1.0 + float(getattr(e, "voltage_rel", 0.0) or 0.0)) \
            * (1.0 if ttf is None else float(ttf)) * math.copysign(1.0, self.brho)
        lag = float(e.phase) + float(getattr(e, "phase_offset", 0.0) or 0.0) + 90.0
        f_hz = (float(e.frequency) + float(getattr(e, "frequency_offset", 0.0) or 0.0)) * 1e6
        h = getattr(e, "harmonic", None)
        j = self._node(nd.N_CAVITY, B.cavity(v, f_hz, lag))
        cav = {"node": j, "name": str(e.name), "voltage_mv": v, "frequency_hz": f_hz,
               "lag_deg": lag, "harmonic": int(h) if h else None}
        if self.reference_following:
            # (the reference's gain there, written by the ramp every turn; no
            # gain to start with)
            cav["refshift_node"] = self._node(nd.N_REFSHIFT,
                                              [0.0, self.ref.p0c, self.ref.beta0])
        self.cavities.append(cav)
        self._element_aperture(e)
        self._end()

    # ---- matrix -----------------------------------------------------------------
    def _matrix(self, e, L):
        """A HELIX matrix element: only a transverse map with no coupling to
        the longitudinal plane (HELIX's matrix does not record the RF
        frequency of its phase units, and a dispersion column without its
        path-length row would not be symplectic)."""
        M = np.asarray(e.matrix, dtype=float)
        lon = np.zeros((6, 6), dtype=bool)
        lon[:4, 4:] = lon[4:, :4] = True
        if (np.any(M[lon] != 0.0) or not np.array_equal(M[4:, 4:], np.eye(2))
                or (e.offset is not None and np.any(e.offset[4:] != 0.0))):
            raise RingCompileError(
                f"element {e.name!r}: a matrix element coupled to the longitudinal plane is "
                "not modelled in ring mode (only a transverse map)")
        T = M[:4, :4]                      # (mm, mrad) blocks are dimensionless / m alike
        S = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)
        err = float(np.abs(T.T @ S @ T - S).max())
        if err > 1e-10:
            raise RingCompileError(
                f"element {e.name!r}: its transverse matrix is not symplectic "
                f"(|M^T S M - S| = {err:.2e} > 1e-10)")
        R = np.eye(6)
        R[:4, :4] = T
        d = np.zeros(6)
        if e.offset is not None:
            d[:4] = np.asarray(e.offset[:4]) * 1e-3
        self._begin(e, L)
        if L:
            self.warnings.append(f"matrix element {e.name!r}: its {L:g} m are covered by the "
                                 "matrix (no drift added), as in linac tracking")
        self._node(nd.N_MATRIX, B.matrix(R, d), L)
        self._element_aperture(e)
        self._end()

    # ---- assembly ---------------------------------------------------------------
    def finish(self, elements, harmonic) -> RingLattice:
        n = len(self.kinds)
        node_len = np.array(self.node_len)
        node_s = np.concatenate([[0.0], np.cumsum(node_len)[:-1]]) if n else np.zeros(0)
        elem_len = np.array(self.elem_len)
        elem_s = np.concatenate([[0.0], np.cumsum(elem_len)[:-1]]) if elem_len.size else np.zeros(0)
        C = float(elem_len.sum())
        if not n or C <= 0.0:
            raise RingCompileError("the lattice has no length")
        ref = self.ref
        T0 = C / (ref.beta0 * C_LIGHT)
        # cavities: harmonic-locked frequency, from the ring's own revolution time
        h_ring = harmonic
        for c in self.cavities:
            h = harmonic if harmonic is not None else c["harmonic"]
            if h:
                c["harmonic"] = int(h)
                c["frequency_hz"] = h / T0
                self.params[self.offsets[c["node"]] + 1] = c["frequency_hz"]
                h_ring = h_ring or int(h)
        if self._pitch_yaw:
            self.warnings.append(
                f"{len(self._pitch_yaw)} element(s) carry dz/pitch/yaw misalignments, which ring "
                f"mode (like linac tracking) does not model: {', '.join(self._pitch_yaw[:5])}"
                + (" ..." if len(self._pitch_yaw) > 5 else ""))
        if self._ignored:
            self.warnings.append("skipped (no map in ring mode): " + ", ".join(
                f"{k} × {v}" for k, v in sorted(self._ignored.items())))
        bg2 = (ref.p0c / ref.mass) ** 2
        return RingLattice(
            kinds=np.array(self.kinds, dtype=np.int64),
            offsets=np.array(self.offsets, dtype=np.int64),
            params=np.array(self.params, dtype=np.float64),
            node_s=node_s, node_length=node_len,
            node_element=np.array(self.node_elem, dtype=np.int64),
            element_names=self.elem_names, element_types=self.elem_types,
            element_s=elem_s, element_length=elem_len,
            element_last_node=np.array(self.elem_last, dtype=np.int64),
            length=C, reference=ref.copy(), brho_norm=self.brho, options=dict(self.o),
            cavities=self.cavities, harmonic=h_ring,
            ref_beta0=np.full(n, ref.beta0), ref_p0c=np.full(n, ref.p0c),
            ref_bg2=np.full(n, bg2), warnings=self.warnings, correctors=self.correctors,
            reference_following=self.reference_following, bumps=self.bumps)
