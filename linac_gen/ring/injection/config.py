"""Injection settings of a ring run (:class:`InjectionConfig`).

Validated loudly at construction, like the ring, ramp and space-charge
settings: a value out of range or a key this HELIX does not know is an error
naming what is accepted.  ``to_dict``/``from_dict`` give the ``"injection"``
entry of a ring configuration (and of a project's ``"ring"`` section);
numbers may be given as text, as YAML reads ``2e-3``.

* ``foil`` (:class:`FoilConfig`) — where the stripping foil is (``at``: an
  element; the foil acts at its exit — None: the lattice's own foil
  element), its material and thickness, the straggling, mean-loss and
  stripping models of HELIX's foil element, and its rectangle in absolute
  ring coordinates (a side left out is open).  A setting left out takes the
  lattice foil element's value, else the default.
* ``source`` (:class:`SourceConfig`) — the injected beam: the Beam tab's
  distribution (``"beam"``), a particle file (``"file"``) or a Gaussian with
  Twiss parameters at the foil (``"gaussian"``); its species (an H⁻ beam is
  stripped to protons of the same velocity), energy, the frequency its phases
  are counted in and the linac's bunch rate; its centroid at the foil in ring
  coordinates; how the particles are drawn turn by turn.
* ``turns``, ``start_turn``, ``particles_per_turn`` — the turns that inject
  and the macroparticles each brings (at most :data:`PARTICLES_TOTAL_MAX` in
  all); ``current_ma`` — the average current
  injected during the pulse (after the chopper), which gives each
  macroparticle its real particles; ``notch_buckets`` — buckets left empty.
* ``chopper_window_deg``, ``chopper_phase_deg`` — the linac bunches kept
  (within the window, in degrees of the ring's RF, about the bucket's centre)
  and the RF phase of the first one.
* ``magnets`` (:class:`MagnetGroup`) — bump and painting magnets whose
  strengths follow a waveform (:class:`Waveform`) turn by turn.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

SOURCES = ("beam", "file", "gaussian")
SPECIES = ("H-", "proton")
DRAWS = ("shuffled", "in_order")
MODELS = ("kick", "dipole", "scale")
PLANES = (None, "x", "y")
WAVEFORMS = ("constant", "table", "linear", "sqrt", "sqrt_end")
STRAGGLING = ("auto", "landau", "gaussian")
STRIP_MODELS = ("off", "two_step")
DEDX_MODELS = ("mip", "bethe")

#: the most injection turns and macroparticles per turn a run may ask for
TURNS_MAX = 10_000_000
PARTICLES_PER_TURN_MAX = 10_000_000
#: the most macroparticles an injection may bring in all (about 120 bytes each)
PARTICLES_TOTAL_MAX = 100_000_000
#: the largest map side (bins) of the foil's hit and deposit maps
MAP_BINS_MAX = 1000
#: the smallest cut of a Gaussian source [sigma]: a pair is drawn again until
#: it falls inside the cut, which keeps 1 - exp(-c^2/2) of them — 39 % at 1
#: sigma (the Beam tab's smallest cutoff too), 5e-9 at 1e-4
CUTOFF_MIN = 1.0


class InjectionError(ValueError):
    """Injection settings that cannot be run (the message says why)."""


def _number(v, what: str, *, none_ok: bool = False):
    """A finite float (text as YAML reads '2e-3' too); None when allowed."""
    if v is None and none_ok:
        return None
    if isinstance(v, bool):
        raise InjectionError(f"injection: {what}={v!r} is not a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise InjectionError(f"injection: {what}={v!r} is not a number") from None
    if not math.isfinite(x):
        raise InjectionError(f"injection: {what}={v!r} is not a finite number")
    return x


def _whole(v, what: str, *, none_ok: bool = False):
    if v is None and none_ok:
        return None
    x = _number(v, what)
    if x != int(x):
        raise InjectionError(f"injection: {what}={v!r} is not a whole number")
    return int(x)


def _choice(v, allowed, what: str):
    if isinstance(v, str):
        low = v.strip()
        for a in allowed:
            if isinstance(a, str) and a.lower() == low.lower():
                return a
    if v in allowed:
        return v
    raise InjectionError(f"injection: {what}={v!r} is not one of {list(allowed)}")


def _from_dict(cls, d, what: str):
    if isinstance(d, cls):
        return cls(**d.to_dict())
    if not isinstance(d, dict):
        raise InjectionError(f"injection: {what} {d!r} is not a table of settings")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise InjectionError(f"injection: {what}: unknown key(s) {unknown} "
                             f"(known: {sorted(known)})")
    return cls(**d)


# ---------------------------------------------------------------------------
# waveforms
# ---------------------------------------------------------------------------
@dataclass
class Waveform:
    """A magnet's strength factor versus time t [s], counted from the start of
    the run:

    * ``"constant"`` — ``value``;
    * ``"table"`` — linear between the points (``times_s``, ``values``), the
      end values held outside them;
    * ``"linear"`` — ``start`` until ``t_start_s``, then linear to ``end`` at
      ``t_end_s``, held after;
    * ``"sqrt"`` — start + (end - start) sqrt(u), u = (t - t_start)/(t_end -
      t_start) clamped to [0, 1] (a closed orbit moved as the square root of
      time: the beam painted uniformly in action);
    * ``"sqrt_end"`` — end + (start - end) sqrt(1 - u): from ``start`` to
      ``end`` with the square root's steep part at the end (the other plane
      of anti-correlated painting).
    """

    kind: str = "constant"
    value: float = 1.0
    times_s: tuple = ()
    values: tuple = ()
    t_start_s: float = 0.0
    t_end_s: float | None = None
    start: float = 1.0
    end: float = 0.0

    def __post_init__(self):
        self.kind = _choice(self.kind, WAVEFORMS, "waveform kind")
        self.value = _number(self.value, "waveform value")
        self.times_s = tuple(_number(v, "waveform times_s") for v in (self.times_s or ()))
        self.values = tuple(_number(v, "waveform values") for v in (self.values or ()))
        self.t_start_s = _number(self.t_start_s, "waveform t_start_s")
        self.t_end_s = _number(self.t_end_s, "waveform t_end_s", none_ok=True)
        self.start = _number(self.start, "waveform start")
        self.end = _number(self.end, "waveform end")
        if self.kind == "table":
            if not self.times_s or len(self.times_s) != len(self.values):
                raise InjectionError("injection: a table waveform needs as many values as times "
                                     "(at least one point)")
            if any(b <= a for a, b in zip(self.times_s, self.times_s[1:], strict=False)):
                raise InjectionError("injection: a table waveform's times must increase")
        elif self.kind in ("linear", "sqrt", "sqrt_end"):
            if self.t_end_s is None or not self.t_end_s > self.t_start_s:
                raise InjectionError(f"injection: a {self.kind} waveform needs t_end_s after "
                                     "t_start_s")

    def __call__(self, t: float) -> float:
        t = float(t)
        if self.kind == "constant":
            return self.value
        if self.kind == "table":
            import numpy as np
            return float(np.interp(t, self.times_s, self.values))
        u = min(1.0, max(0.0, (t - self.t_start_s) / (self.t_end_s - self.t_start_s)))
        if self.kind == "linear":
            return self.start + (self.end - self.start) * u
        if self.kind == "sqrt":
            return self.start + (self.end - self.start) * math.sqrt(u)
        return self.end + (self.start - self.end) * math.sqrt(1.0 - u)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["times_s"] = list(self.times_s)
        d["values"] = list(self.values)
        return d

    @classmethod
    def from_dict(cls, d) -> Waveform:
        return _from_dict(cls, d, "waveform")


# ---------------------------------------------------------------------------
# the foil, the source, the magnets
# ---------------------------------------------------------------------------
@dataclass
class FoilConfig:
    at: str | None = None                    # element name; None: the lattice's foil element
    material: str | None = None              # None: the foil element's, else "C"
    thickness_ug_cm2: float | None = None    # None: the foil element's (required without one)
    straggling: str | None = None            # None: the foil element's, else "auto"
    dedx_model: str | None = None            # None: the foil element's, else "mip"
    strip_model: str | None = None           # None: the foil element's, else "two_step"
    x_min_m: float | None = None             # the foil's rectangle, absolute ring coordinates
    x_max_m: float | None = None             # (a side left out: open, or the foil element's
    y_min_m: float | None = None             # extent about its offset)
    y_max_m: float | None = None
    map_bins: tuple = (40, 40)               # the hit and deposit maps (closed rectangle only)

    def __post_init__(self):
        from linac_gen.elements.foil import supported_materials
        if self.at is not None:
            self.at = str(self.at).strip()
            if not self.at:
                raise InjectionError("injection: foil 'at' is an empty element name")
        if self.material is not None:
            mats = supported_materials()
            if self.material not in mats:
                raise InjectionError(f"injection: foil material {self.material!r} is not one of "
                                     f"{mats}")
        self.thickness_ug_cm2 = _number(self.thickness_ug_cm2, "foil thickness_ug_cm2",
                                        none_ok=True)
        if self.thickness_ug_cm2 is not None and not self.thickness_ug_cm2 > 0.0:
            raise InjectionError("injection: foil thickness_ug_cm2 must be positive")
        if self.straggling is not None:
            self.straggling = _choice(self.straggling, STRAGGLING, "foil straggling")
        if self.dedx_model is not None:
            self.dedx_model = _choice(self.dedx_model, DEDX_MODELS, "foil dedx_model")
        if self.strip_model is not None:
            self.strip_model = _choice(self.strip_model, STRIP_MODELS, "foil strip_model")
        for k in ("x_min_m", "x_max_m", "y_min_m", "y_max_m"):
            setattr(self, k, _number(getattr(self, k), f"foil {k}", none_ok=True))
        for lo, hi in (("x_min_m", "x_max_m"), ("y_min_m", "y_max_m")):
            a, b = getattr(self, lo), getattr(self, hi)
            if a is not None and b is not None and not b > a:
                raise InjectionError(f"injection: foil {hi} must be above {lo}")
        mb = self.map_bins
        if isinstance(mb, str):
            mb = mb.lower().replace("x", ",").split(",")
        try:
            mb = [_whole(v, "foil map_bins") for v in mb]
        except TypeError:
            raise InjectionError(f"injection: foil map_bins={self.map_bins!r} is not two "
                                 "numbers") from None
        if len(mb) != 2 or min(mb) < 1 or max(mb) > MAP_BINS_MAX:
            raise InjectionError(f"injection: foil map_bins must be two numbers from 1 to "
                                 f"{MAP_BINS_MAX}")
        self.map_bins = (mb[0], mb[1])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["map_bins"] = list(self.map_bins)
        return d

    @classmethod
    def from_dict(cls, d) -> FoilConfig:
        return _from_dict(cls, d, "foil")


@dataclass
class SourceConfig:
    kind: str = "beam"
    path: str | None = None                  # "file": DST, HELIX ASCII or TraceWin particle ASCII
    species: str | None = None               # None: the beam's or the file's ("gaussian": "H-")
    # the beam's energy: None, the beam's or the file's ("gaussian": required); else the
    # beam moved there (its reference energy; the particles' offsets from it kept)
    kinetic_mev: float | None = None
    frequency_mhz: float | None = None       # the unit of the source's phases; None: the beam's
    #                                          / the file's
    bunch_frequency_mhz: float | None = None  # the linac's bunch rate (the chopper); None: the
    #                                           frequency of the phases
    # "gaussian": normalized rms emittances [um = pi mm mrad], Twiss at the foil, the
    # relative momentum spread, a linac bunch's rms length in zeta [m], a cut [sigma]
    emit_nx_um: float = 0.0
    emit_ny_um: float = 0.0
    beta_x_m: float = 1.0
    alpha_x: float = 0.0
    beta_y_m: float = 1.0
    alpha_y: float = 0.0
    sigma_dp: float = 0.0
    sigma_z_m: float = 0.0
    cutoff: float | None = None
    # the injected beam's centroid at the foil, absolute ring coordinates
    x_m: float = 0.0
    px: float = 0.0
    y_m: float = 0.0
    py: float = 0.0
    draw: str = "shuffled"                   # the pool's order: a seeded shuffle, or the file's

    def __post_init__(self):
        self.kind = _choice(self.kind, SOURCES, "source kind")
        if self.species is not None:
            self.species = _choice(self.species, SPECIES, "source species")
        self.kinetic_mev = _number(self.kinetic_mev, "source kinetic_mev", none_ok=True)
        if self.kinetic_mev is not None and not self.kinetic_mev > 0.0:
            raise InjectionError("injection: source kinetic_mev must be positive")
        self.frequency_mhz = _number(self.frequency_mhz, "source frequency_mhz", none_ok=True)
        if self.frequency_mhz is not None and not self.frequency_mhz > 0.0:
            raise InjectionError("injection: source frequency_mhz must be positive")
        self.bunch_frequency_mhz = _number(self.bunch_frequency_mhz,
                                           "source bunch_frequency_mhz", none_ok=True)
        if self.bunch_frequency_mhz is not None and not self.bunch_frequency_mhz > 0.0:
            raise InjectionError("injection: source bunch_frequency_mhz must be positive")
        if self.kind == "file":
            if not self.path or not str(self.path).strip():
                raise InjectionError("injection: a file source needs its path")
            self.path = str(self.path)
        for k in ("emit_nx_um", "emit_ny_um", "sigma_dp", "sigma_z_m"):
            v = _number(getattr(self, k), f"source {k}")
            if v < 0.0:
                raise InjectionError(f"injection: source {k} must be 0 or more")
            setattr(self, k, v)
        for k in ("beta_x_m", "beta_y_m"):
            v = _number(getattr(self, k), f"source {k}")
            if not v > 0.0:
                raise InjectionError(f"injection: source {k} must be positive")
            setattr(self, k, v)
        for k in ("alpha_x", "alpha_y", "x_m", "px", "y_m", "py"):
            setattr(self, k, _number(getattr(self, k), f"source {k}"))
        self.cutoff = _number(self.cutoff, "source cutoff", none_ok=True)
        if self.cutoff is not None and not self.cutoff >= CUTOFF_MIN:
            raise InjectionError(f"injection: source cutoff must be at least {CUTOFF_MIN:g} "
                                 "(sigma; leave it out for no cut)")
        if abs(self.px) >= 1.0 or abs(self.py) >= 1.0:
            raise InjectionError("injection: the source's centroid px, py must be below 1 "
                                 "(P/P0)")
        if self.kind == "gaussian" and self.kinetic_mev is None:
            raise InjectionError("injection: a gaussian source needs its kinetic_mev")
        self.draw = _choice(self.draw, DRAWS, "source draw")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> SourceConfig:
        return _from_dict(cls, d, "source")


@dataclass
class MagnetGroup:
    name: str = ""
    elements: tuple = ()
    model: str = "kick"
    plane: str | None = None                 # "kick"/"dipole": "x" or "y" (None: the element's)
    kick_rad: tuple | None = None            # the deflection of each element at waveform 1
    #                                          (needed for a drift as a "dipole")
    fint: float = 0.0                        # "dipole": the fringe integral and half gap
    hgap_m: float = 0.0
    e1_rad: float = 0.0                      # "dipole": pole-face angles [rad]
    e2_rad: float = 0.0
    waveform: Waveform = field(default_factory=Waveform)

    def __post_init__(self):
        self.name = str(self.name or "").strip()
        if not self.name:
            raise InjectionError("injection: a magnet group needs a name")
        if isinstance(self.elements, str):
            self.elements = tuple(v for v in self.elements.replace(";", ",").split(","))
        names = tuple(str(v).strip() for v in (self.elements or ()) if str(v).strip())
        if not names:
            raise InjectionError(f"injection: magnet group {self.name!r} names no elements")
        low = [n.lower() for n in names]
        if len(set(low)) != len(low):
            raise InjectionError(f"injection: magnet group {self.name!r} names an element twice")
        self.elements = names
        self.model = _choice(self.model, MODELS, f"magnet group {self.name!r} model")
        if self.plane is not None:
            self.plane = _choice(self.plane, PLANES, f"magnet group {self.name!r} plane")
        if self.kick_rad is not None:
            k = self.kick_rad
            if isinstance(k, (int, float, str)) and not isinstance(k, bool):
                k = [k] * len(names)
            k = tuple(_number(v, f"magnet group {self.name!r} kick_rad") for v in k)
            if len(k) != len(names):
                raise InjectionError(f"injection: magnet group {self.name!r}: give one kick_rad "
                                     f"per element ({len(names)}), or one for all")
            if any(abs(v) >= 1.0 for v in k):
                raise InjectionError(f"injection: magnet group {self.name!r}: a kick_rad of 1 "
                                     "rad or more is not a kick")
            if self.model == "scale":
                raise InjectionError(f"injection: magnet group {self.name!r}: a 'scale' group "
                                     "scales the elements' own strengths — no kick_rad")
            self.kick_rad = k
        for k in ("fint", "hgap_m"):
            v = _number(getattr(self, k), f"magnet group {self.name!r} {k}")
            if v < 0.0:
                raise InjectionError(f"injection: magnet group {self.name!r}: {k} must be 0 or "
                                     "more")
            setattr(self, k, v)
        for k in ("e1_rad", "e2_rad"):
            v = _number(getattr(self, k), f"magnet group {self.name!r} {k}")
            if not abs(v) < 0.5 * math.pi:
                raise InjectionError(f"injection: magnet group {self.name!r}: {k} must be "
                                     "within +-pi/2")
            setattr(self, k, v)
        if self.model != "dipole" and (self.fint or self.hgap_m or self.e1_rad or self.e2_rad):
            raise InjectionError(f"injection: magnet group {self.name!r}: fint, hgap_m, e1_rad "
                                 "and e2_rad belong to the 'dipole' model")
        if isinstance(self.waveform, dict):
            self.waveform = Waveform.from_dict(self.waveform)
        elif not isinstance(self.waveform, Waveform):
            raise InjectionError(f"injection: magnet group {self.name!r}: the waveform must be "
                                 "a table of settings")

    def to_dict(self) -> dict:
        return {"name": self.name, "elements": list(self.elements), "model": self.model,
                "plane": self.plane,
                "kick_rad": None if self.kick_rad is None else list(self.kick_rad),
                "fint": self.fint, "hgap_m": self.hgap_m, "e1_rad": self.e1_rad,
                "e2_rad": self.e2_rad, "waveform": self.waveform.to_dict()}

    @classmethod
    def from_dict(cls, d) -> MagnetGroup:
        return _from_dict(cls, d, "magnet group")


@dataclass
class InjectionConfig:
    foil: FoilConfig = field(default_factory=FoilConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    turns: int = 1
    start_turn: int = 0
    particles_per_turn: int = 100
    current_ma: float | None = None
    notch_buckets: int = 0
    chopper_window_deg: float = 180.0
    chopper_phase_deg: float = 0.0
    magnets: tuple = ()

    def __post_init__(self):
        if isinstance(self.foil, dict):
            self.foil = FoilConfig.from_dict(self.foil)
        elif not isinstance(self.foil, FoilConfig):
            raise InjectionError("injection: foil must be a table of settings")
        if isinstance(self.source, dict):
            self.source = SourceConfig.from_dict(self.source)
        elif not isinstance(self.source, SourceConfig):
            raise InjectionError("injection: source must be a table of settings")
        self.turns = _whole(self.turns, "turns")
        if not 1 <= self.turns <= TURNS_MAX:
            raise InjectionError(f"injection: turns must be from 1 to {TURNS_MAX}")
        self.start_turn = _whole(self.start_turn, "start_turn")
        if not 0 <= self.start_turn <= TURNS_MAX:
            raise InjectionError(f"injection: start_turn must be from 0 to {TURNS_MAX}")
        self.particles_per_turn = _whole(self.particles_per_turn, "particles_per_turn")
        if not 1 <= self.particles_per_turn <= PARTICLES_PER_TURN_MAX:
            raise InjectionError(f"injection: particles_per_turn must be from 1 to "
                                 f"{PARTICLES_PER_TURN_MAX}")
        if self.turns * self.particles_per_turn > PARTICLES_TOTAL_MAX:
            raise InjectionError(f"injection: {self.turns} turns of {self.particles_per_turn} "
                                 f"macroparticles are more than {PARTICLES_TOTAL_MAX} "
                                 "macroparticles in all")
        self.current_ma = _number(self.current_ma, "current_ma", none_ok=True)
        if self.current_ma is None or not self.current_ma > 0.0:
            raise InjectionError("injection: give the injected current, current_ma > 0 (the "
                                 "average current during the pulse, after the chopper)")
        self.notch_buckets = _whole(self.notch_buckets, "notch_buckets")
        if self.notch_buckets < 0:
            raise InjectionError("injection: notch_buckets must be 0 or more")
        self.chopper_window_deg = _number(self.chopper_window_deg, "chopper_window_deg")
        if not 0.0 < self.chopper_window_deg <= 180.0:
            raise InjectionError("injection: chopper_window_deg must be above 0 and at most 180 "
                                 "(180: every linac bunch kept)")
        self.chopper_phase_deg = _number(self.chopper_phase_deg, "chopper_phase_deg")
        groups = []
        for g in self.magnets or ():
            groups.append(g if isinstance(g, MagnetGroup) else MagnetGroup.from_dict(g))
        names = [g.name.lower() for g in groups]
        if len(set(names)) != len(names):
            raise InjectionError("injection: two magnet groups have the same name")
        seen: dict = {}
        for g in groups:
            for e in g.elements:
                other = seen.get(e.lower())
                if other is not None:
                    raise InjectionError(f"injection: element {e!r} is in magnet groups "
                                         f"{other!r} and {g.name!r} — one group per element")
                seen[e.lower()] = g.name
        self.magnets = tuple(groups)

    @property
    def end_turn(self) -> int:
        """The first turn after the injection."""
        return self.start_turn + self.turns

    @property
    def particles_total(self) -> int:
        return self.turns * self.particles_per_turn

    def to_dict(self) -> dict:
        return {"foil": self.foil.to_dict(), "source": self.source.to_dict(),
                "turns": self.turns, "start_turn": self.start_turn,
                "particles_per_turn": self.particles_per_turn, "current_ma": self.current_ma,
                "notch_buckets": self.notch_buckets,
                "chopper_window_deg": self.chopper_window_deg,
                "chopper_phase_deg": self.chopper_phase_deg,
                "magnets": [g.to_dict() for g in self.magnets]}

    @classmethod
    def from_dict(cls, d) -> InjectionConfig:
        return _from_dict(cls, d, "settings")


# ---------------------------------------------------------------------------
# settings files
# ---------------------------------------------------------------------------
def read_waveform_csv(path) -> tuple[tuple, tuple]:
    """A waveform's table from a CSV file: a header row ``t_s,value``, then
    one row per point; blank lines and lines starting with ``#`` are skipped."""
    import csv
    from pathlib import Path
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise InjectionError(f"waveform table: cannot read {p}: {exc}") from None
    try:
        rows = [r for r in csv.reader(line for line in text.splitlines()
                                      if line.strip() and not line.lstrip().startswith("#"))]
    except csv.Error as exc:
        raise InjectionError(f"waveform table {p.name}: not a CSV table ({exc})") from None
    if not rows:
        raise InjectionError(f"waveform table {p.name}: the file is empty")
    head = [h.strip().lower() for h in rows[0]]
    if head != ["t_s", "value"]:
        raise InjectionError(f"waveform table {p.name}: the header must be 't_s,value' "
                             f"(got {rows[0]})")
    times, values = [], []
    for k, r in enumerate(rows[1:], start=2):
        if len(r) != 2:
            raise InjectionError(f"waveform table {p.name}: row {k} has {len(r)} values, not 2")
        try:
            times.append(float(r[0]))
            values.append(float(r[1]))
        except ValueError:
            raise InjectionError(f"waveform table {p.name}: row {k} is not two numbers "
                                 f"({r})") from None
    return tuple(times), tuple(values)


def write_waveform_csv(path, times, values) -> None:
    """Write a waveform's table as :func:`read_waveform_csv` reads it."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "value"])
        for t, v in zip(times, values, strict=True):
            w.writerow([repr(float(t)), repr(float(v))])


def injection_from_file(path) -> InjectionConfig:
    """Injection settings from a JSON file (or YAML, with PyYAML installed) in
    the layout of :meth:`InjectionConfig.to_dict`.  A magnet group's waveform
    table may be given as ``"file": "name.csv"`` (relative to the settings
    file; see :func:`read_waveform_csv`), and a file source's relative path
    is taken relative to the settings file too (kept as an absolute path, so
    the settings do not depend on how the file's path was spelled)."""
    import json
    from pathlib import Path
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise InjectionError(f"injection file: cannot read {p}: {exc}") from None
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise InjectionError(f"injection file {p.name}: YAML needs PyYAML, which is not "
                                 "installed — write the settings as JSON") from None
        try:
            d = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise InjectionError(f"injection file {p.name}: not valid YAML ({exc})") from None
    else:
        try:
            d = json.loads(text)
        except ValueError as exc:
            raise InjectionError(f"injection file {p.name}: not valid JSON ({exc})") from None
    if not isinstance(d, dict):
        raise InjectionError(f"injection file {p.name}: expected a mapping of the settings")
    d = dict(d)
    src = d.get("source")
    if isinstance(src, dict) and src.get("path") and not Path(src["path"]).is_absolute():
        d["source"] = dict(src, path=str((p.parent / src["path"]).resolve()))
    groups = []
    for g in d.get("magnets") or ():
        if isinstance(g, dict) and isinstance(g.get("waveform"), dict) \
                and "file" in g["waveform"]:
            wf = dict(g["waveform"])
            f = wf.pop("file")
            if "times_s" in wf or "values" in wf:
                raise InjectionError(f"injection file {p.name}: magnet group "
                                     f"{g.get('name')!r} gives both a waveform file and a table")
            if wf.get("kind", "table") != "table":
                raise InjectionError(f"injection file {p.name}: magnet group "
                                     f"{g.get('name')!r} gives a waveform file but is of kind "
                                     f"{wf.get('kind')!r}")
            csv_path = Path(f) if Path(f).is_absolute() else p.parent / f
            times, values = read_waveform_csv(csv_path)
            wf.update(kind="table", times_s=list(times), values=list(values))
            g = dict(g, waveform=wf)
        groups.append(g)
    if groups:
        d["magnets"] = groups
    return InjectionConfig.from_dict(d)
