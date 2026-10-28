"""Ring-mode configuration.

Validated loudly at construction, like the multibunch study: an unknown value
is an error naming the accepted ones, and a knob that no implemented part of
ring mode consumes is not offered at all (a later milestone adds injection
when it is implemented).
``to_dict``/``from_dict`` give the ``"ring"`` section of a ``.lgproj``; an
unknown key there is refused rather than ignored.

The element model (:mod:`linac_gen.ring.compile`):

* ``model`` — ``"cartesian"``: bend multipoles expanded in Cartesian
  coordinates about the curved reference (as MAD-X decks define them); ``"curvilinear"``:
  exact elements with the bend multipoles in the curved frame.  The names
  of earlier versions, ``"madx"`` and ``"impactx"``, are still read.
* ``edge_model`` — pole faces: ``"exact"`` (frame rotation, hard-edge dipole
  fringe, quadrupole and dipole wedges), ``"full"`` (also the quadrupole
  fringe, and the wedge's quadrupole field to first order), ``"linear"`` (thin
  linear kicks with the fringe correction); None
  = the model's own (exact for cartesian, linear for curvilinear).
* ``drift_model``, ``bend_model`` — exact or expanded maps (cartesian model);
* ``integrator`` (bends) and ``magnet_integrator`` (straight thick magnets)
  — ``yoshida4`` (Yoshida 1990), ``teapot`` (Schachinger & Talman 1987),
  ``uniform``, or ``symp2``/``symp4``/``symp6`` (Yoshida's symplectic orders 2,
  4, 6); None = the model's own;
* ``bend_steps`` (0 = automatic) and ``magnet_steps`` — integration steps.

The ramp (:mod:`linac_gen.ring.programs`): ``ramp`` — None (a storage ring at
fixed energy) or the programs as a dict (``RampConfig.to_dict``): the
reference momentum, the RF voltage and phase versus time, paraphasing groups,
gamma_t and the clock at turn 0; ``corrector_mode`` — "follow" (a steerer's
kick follows the reference momentum) or "fixed_field" (it keeps its field).

Space charge (:mod:`linac_gen.ring.sc`): ``space_charge`` — None (none) or the
settings as a dict (``SpaceChargeConfig.to_dict``): the 2.5-D model or a
frozen Gaussian, the bunch's intensity, the kicks per turn, the grid, the
pipe and the longitudinal kick.

Injection (:mod:`linac_gen.ring.injection`): ``injection`` — None (none) or
the settings as a dict (``InjectionConfig.to_dict``): the foil, the injected
beam, the turns that inject, the current, the chopper and the bump and
painting magnets.  With ``source = "empty"`` the ring starts with no beam
(the injection fills it); ``capacity`` (None: the particles of the start
and of the injection) bounds the particles a run holds.

The run: ``snapshot_turns`` holds whole numbers and ``"last"`` (the last
turn, whatever the number of turns); :func:`parse_snapshot_turns` reads them
from text.  :func:`auto_stats_every` is the statistics interval the GUI and
the command line use when none is given: every turn up to
``STATS_MAX_ROWS`` rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

#: the most statistics rows :func:`auto_stats_every` gives a run (about 600
#: bytes each in memory)
STATS_MAX_ROWS = 100_000

#: model names of earlier versions, still read from saved projects
MODEL_ALIASES = {"madx": "cartesian", "impactx": "curvilinear"}

_CHOICES = {
    "constants": ("helix", "codata2010", "codata2022"),
    "species": ("proton",),
    "model": ("cartesian", "curvilinear"),
    "edge_model": (None, "exact", "full", "linear"),
    "drift_model": ("exact", "expanded"),
    "bend_model": ("exact", "exact-bend", "expanded"),
    "integrator": (None, "yoshida4", "teapot", "uniform", "symp2", "symp4", "symp6"),
    "magnet_integrator": (None, "yoshida4", "teapot", "uniform", "symp2", "symp4", "symp6"),
    "corrector_mode": ("follow", "fixed_field"),
    "device": ("cpu", "cuda"),
    "source": ("beam", "matched", "empty"),
    "zeta_wrap": ("auto", "off"),
}


@dataclass
class RingConfig:
    n_turns: int = 1
    harmonic: int | None = None            # None = from the lattice (RF HARMON / ;@LG harmonic)
    constants: str = "helix"
    species: str = "proton"
    design_energy_mev: float | None = None  # None = the lattice's reference energy
    model: str = "cartesian"
    edge_model: str | None = None
    drift_model: str = "exact"
    bend_model: str = "exact"
    integrator: str | None = None
    magnet_integrator: str | None = None
    bend_steps: int = 0                    # 0 = automatic (from the focusing and angle)
    magnet_steps: int = 1                  # straight thick magnets
    corrector_mode: str = "follow"         # correctors follow the ramp, or keep their field
    device: str = "cpu"
    threads: int | None = None
    n_particles: int = 1000                # the "matched" source's (the "beam" source: the beam's)
    capacity: int | None = None            # None = the start's particles and the injection's
    seed: int = 1
    stats_every: int = 1                   # per-turn diagnostics cadence
    snapshot_turns: list = field(default_factory=list)   # whole numbers and "last"
    # the particles: a Gaussian of n_particles matched to the ring's periodic
    # optics with the Beam tab's emittances ("matched": the command line's
    # for a bare lattice file, and the API's default), the Beam tab's
    # distribution ("beam", generated at the ring's species and energy: the
    # GUI's and a project's), or none ("empty": the injection fills the ring)
    source: str = "matched"
    tune_probes: int = 1                   # 0 none, 1 the tunes, n >= 2: an n x n footprint
    probe_max_sigma: float = 3.0           # largest footprint amplitude, rms beam sizes
    checkpoint_every: int = 0              # turns between checkpoints (0: none)
    zeta_wrap: str = "auto"                # zeta into the RF period (C/h), or "off"
    ramp: dict | None = None               # None: fixed energy; else RampConfig.to_dict()
    space_charge: dict | None = None       # None: none; else SpaceChargeConfig.to_dict()
    injection: dict | None = None          # None: none; else InjectionConfig.to_dict()

    def __post_init__(self):
        # (a project saved with the model names of earlier versions)
        if isinstance(self.model, str):
            self.model = MODEL_ALIASES.get(self.model, self.model)
        for name, allowed in _CHOICES.items():
            v = getattr(self, name)
            if v not in allowed:
                raise ValueError(f"ring config: {name}={v!r} is not one of {list(allowed)}")
        if int(self.n_turns) < 1:
            raise ValueError("ring config: n_turns must be at least 1")
        if self.harmonic is not None and int(self.harmonic) < 1:
            raise ValueError("ring config: harmonic must be a positive integer")
        if self.design_energy_mev is not None and not self.design_energy_mev > 0:
            raise ValueError("ring config: design_energy_mev must be positive")
        if int(self.bend_steps) < 0 or int(self.magnet_steps) < 1:
            raise ValueError("ring config: bend_steps >= 0 (0 = automatic) and magnet_steps >= 1")
        if self.model == "curvilinear":
            if self.edge_model not in (None, "linear"):
                raise ValueError("ring config: the curvilinear model has linear pole faces "
                                 "only (edge_model='linear')")
            for nm in ("integrator", "magnet_integrator"):
                if getattr(self, nm) not in (None, "symp2", "symp4", "symp6"):
                    raise ValueError(f"ring config: the curvilinear model integrates with symp2, "
                                     f"symp4 or symp6 ({nm}={getattr(self, nm)!r})")
        if int(self.n_particles) < 1:
            raise ValueError("ring config: n_particles must be at least 1")
        if (self.capacity is not None and self.source != "empty"
                and int(self.capacity) < int(self.n_particles)):
            raise ValueError("ring config: capacity must be at least n_particles")
        if self.threads is not None and int(self.threads) < 1:
            raise ValueError("ring config: threads must be at least 1")
        if int(self.stats_every) < 1:
            raise ValueError("ring config: stats_every must be at least 1")
        self.snapshot_turns = [_snapshot_token(t) for t in self.snapshot_turns]
        if int(self.tune_probes) < 0:
            raise ValueError("ring config: tune_probes must be 0 or more")
        if not float(self.probe_max_sigma) > 0.0:
            raise ValueError("ring config: probe_max_sigma must be positive")
        if int(self.checkpoint_every) < 0:
            raise ValueError("ring config: checkpoint_every must be 0 (none) or more")
        if self.ramp is not None:
            from linac_gen.ring.programs import RampConfig
            try:
                rc = self.ramp if isinstance(self.ramp, RampConfig) else \
                    RampConfig.from_dict(dict(self.ramp))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ring config: ramp: {exc}") from None
            self.ramp = rc.to_dict()
        if self.space_charge is not None:
            from linac_gen.ring.sc.config import SpaceChargeConfig
            try:
                self.space_charge = SpaceChargeConfig.from_dict(self.space_charge).to_dict()
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ring config: {exc}") from None
        if self.injection is not None:
            from linac_gen.ring.injection.config import InjectionConfig
            try:
                ic = InjectionConfig.from_dict(self.injection)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ring config: {exc}") from None
            self.injection = ic.to_dict()
            if self.space_charge is not None and self.space_charge.get("intensity") is not None:
                raise ValueError("ring config: with injection the bunch's intensity comes from "
                                 "the injected current — leave the space-charge intensity out")
            n0 = 0 if self.source == "empty" else int(self.n_particles)
            if self.capacity is not None and int(self.capacity) < n0 + ic.particles_total:
                raise ValueError(f"ring config: capacity {self.capacity} is below the particles "
                                 f"of the start and the injection ({n0 + ic.particles_total})")
        elif self.source == "empty":
            raise ValueError("ring config: source 'empty' (no beam at the start) needs "
                             "injection settings")

    def element_options(self) -> dict:
        """The compiler's element-model options."""
        return {k: getattr(self, k) for k in ("model", "edge_model", "drift_model", "bend_model",
                                              "integrator", "magnet_integrator", "bend_steps",
                                              "magnet_steps")}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RingConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"ring config: unknown key(s) {unknown} — this HELIX does not "
                             f"implement them (known: {sorted(known)})")
        return cls(**d)


def migrate_source(cfg: dict) -> tuple[dict, bool, int | None]:
    """A project's ring settings as an earlier version saved them, whose run
    started from a Gaussian matched to the ring with its own number of
    particles ("source": "matched" — or no source, the default then; this
    version always saves the start): a project's run now starts from its Beam
    tab's beam.  Returns (the settings with "source": "beam" and without
    "n_particles", whether the source changed, the old number of particles or
    None); other settings come back unchanged ("n_particles" dropped: the Beam
    tab's).  Settings with an injection and no start are an empty ring the
    injection fills ("source": "empty", nothing changed).  The GUI, the
    command line and the assistant read a project through it alike."""
    out = dict(cfg or {})
    n = out.pop("n_particles", None)
    if out.get("source") is None and out.get("injection"):
        out["source"] = "empty"
        return out, False, None
    if out.get("source") not in (None, "matched"):
        return out, False, None
    out["source"] = "beam"
    return out, True, (n if isinstance(n, int) and not isinstance(n, bool) else None)


def declared_beam(lattice) -> dict | None:
    """The beam the lattice's deck declares — a MAD-X or MAD8 BEAM that gives
    the energy (or momentum, gamma, beta, rigidity) — or None.

    ``lattice.ring_reference_beam`` (kept by the GUI and CLI loaders) is
    always the beam the importer scaled the magnets with: the deck's, or the
    importer's default when the deck states none (every elegant deck, a BEAM
    without an energy).  Only a declared one sets the ring's species and
    energy; with none the ring runs at the Beam tab's (``--energy`` on the
    command line) while the magnets keep the deck's normalized strengths.
    (Here, not in the compiler, so that the GUI thread can use it without
    loading the compiler and numba.)"""
    rb = getattr(lattice, "ring_reference_beam", None)
    if not rb or not rb.get("declared", True):
        return None
    return rb


def _snapshot_token(t):
    """A snapshot turn: a whole number >= 0, or "last"."""
    if isinstance(t, str):
        tok = t.strip().lower()
        if tok == "last":
            return "last"
        if tok.isascii() and tok.isdigit():
            return int(tok)
    elif not isinstance(t, bool):
        try:
            if int(t) == t and int(t) >= 0:
                return int(t)
        except (TypeError, ValueError, OverflowError):
            pass
    raise ValueError(f"ring config: snapshot turn {t!r} is not a whole number >= 0 or 'last'")


def parse_snapshot_turns(text) -> list:
    """'0, 100, last' → [0, 100, 'last'] (commas, semicolons or spaces between;
    anything but whole numbers >= 0 and 'last' is refused)."""
    toks = str(text or "").replace(";", ",").replace(" ", ",").split(",")
    out = []
    for tok in toks:
        if tok.strip():
            t = _snapshot_token(tok)
            if t not in out:
                out.append(t)
    return out


def snapshot_turn_set(tokens, n_turns: int) -> set:
    """The turns of ``tokens`` in a run of ``n_turns`` ('last' = n_turns)."""
    return {int(n_turns) if t == "last" else int(t) for t in (tokens or [])}


def auto_stats_every(n_turns: int, max_rows: int = STATS_MAX_ROWS) -> int:
    """Statistics every turn, or every k turns when a row per turn would be
    more than ``max_rows`` rows."""
    return max(1, -(-int(n_turns) // int(max_rows)))
