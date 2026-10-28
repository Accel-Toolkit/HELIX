"""Space-charge settings of a ring run (:class:`SpaceChargeConfig`).

Validated loudly at construction, like the ring and ramp settings: a value
out of range or a key this HELIX does not know is an error naming what is
accepted.  ``to_dict``/``from_dict`` give the ``"space_charge"`` entry of a
ring configuration (and of a project's ``"ring"`` section); numbers may be
given as text, as YAML reads ``8e10``.

* ``model`` — ``"pic"``: the 2.5-D particle-in-cell model, the transverse
  field solved from the beam itself at every kick; ``"frozen"``: the field of
  a Gaussian with the beam's centroid and rms sizes at each kick, measured
  every ``refresh_turns`` turns (0: only at the start).
* ``intensity`` — real particles in the bunch, positive (None: the Beam
  tab's current over its frequency — one bunch — or, for a coasting beam, the
  current times the revolution time — the whole ring).
* ``kicks_per_turn`` — about this many kicks, at element boundaries.
* ``grid`` (nx, ny), ``deposit`` ("tsc" or "cic"), ``box_sigma`` — the grid
  covers the beam out to its extent or this many rms sizes, the smaller.
* ``pipe`` — "open", "round" (``pipe_a_m`` the radius), "rectangle" or
  "ellipse" (``pipe_a_m``, ``pipe_b_m`` the half width and half height):
  the grounded wall, centred on the reference orbit; ``reference_radius_m``
  — with no pipe, where the potential is zero (the longitudinal kick needs
  one); ``boundary_order`` — harmonics fitted to a rectangle or an ellipse.
* ``longitudinal`` — "potential" (the kick of the line density's slope with
  the potential of the transverse distribution), "g_factor" (the same kick
  with a constant geometry factor ``g_factor`` for every particle), "off";
  ``z_bins`` and ``z_smoothing_bins`` — the line density's bins over the RF
  period ("auto", the default: 16 bins per rms of the bunch's core, re-sized
  as the bunch changes, the line density smoothed over a width that shrinks
  as the particles grow — or a fixed number from 8 to 65536) and the width of
  its Gaussian filter (in bins; "auto" widens it as the particles need).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

MODELS = ("pic", "frozen")
DEPOSITS = ("tsc", "cic")
PIPES = ("open", "round", "rectangle", "ellipse")
LONGITUDINAL = ("potential", "g_factor", "off")

#: the largest grid side (a run allocates (2 nx) x (2 ny) complex FFT arrays, and
#: the Green's functions of one grid take 6 x 16 x nx x ny bytes: 96 MiB at 1024)
GRID_MAX = 1024
#: the most kicks a turn may ask for (a kick at every element boundary needs far fewer)
KICKS_MAX = 1_000_000
#: the largest refresh interval (turns) of the frozen model
REFRESH_MAX = 2_000_000_000
#: the largest box_sigma (rms sizes) and z_smoothing_bins (bins)
SPAN_MAX = 1.0e6


def _number(v, what: str, *, none_ok: bool = False):
    """A finite float (text as YAML reads '8e10' too); None when allowed."""
    if v is None and none_ok:
        return None
    if isinstance(v, bool):
        raise ValueError(f"space charge: {what}={v!r} is not a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"space charge: {what}={v!r} is not a number") from None
    if not math.isfinite(x):
        raise ValueError(f"space charge: {what}={v!r} is not a finite number")
    return x


def _whole(v, what: str) -> int:
    x = _number(v, what)
    if x != int(x):
        raise ValueError(f"space charge: {what}={v!r} is not a whole number")
    return int(x)


@dataclass
class SpaceChargeConfig:
    model: str = "pic"
    intensity: float | None = None          # real particles in the bunch
    kicks_per_turn: int = 100
    grid: tuple = (64, 64)
    deposit: str = "tsc"
    box_sigma: float = 4.0
    pipe: str = "open"
    pipe_a_m: float | None = None            # radius, or half width
    pipe_b_m: float | None = None            # half height
    reference_radius_m: float = 1.0
    boundary_order: int = 24
    longitudinal: str = "potential"
    g_factor: float | None = None
    z_bins: int | str = "auto"
    z_smoothing_bins: float = 1.0
    refresh_turns: int = 1

    def __post_init__(self):
        for name, allowed in (("model", MODELS), ("deposit", DEPOSITS), ("pipe", PIPES),
                              ("longitudinal", LONGITUDINAL)):
            v = getattr(self, name)
            if isinstance(v, str):
                v = v.strip().lower()
                setattr(self, name, v)
            if v not in allowed:
                raise ValueError(f"space charge: {name}={v!r} is not one of {list(allowed)}")
        self.intensity = _number(self.intensity, "intensity", none_ok=True)
        if self.intensity is not None and not self.intensity > 0.0:
            raise ValueError("space charge: intensity (real particles in the bunch) must be "
                             "positive — leave it out to take the Beam tab's current")
        self.kicks_per_turn = _whole(self.kicks_per_turn, "kicks_per_turn")
        if not 1 <= self.kicks_per_turn <= KICKS_MAX:
            raise ValueError(f"space charge: kicks_per_turn must be from 1 to {KICKS_MAX} "
                             "(a kick at every element boundary needs no more)")
        g = self.grid
        if isinstance(g, str):
            g = g.lower().replace("x", ",").replace("*", ",").split(",")
        try:
            g = [_whole(v, "grid") for v in g]
        except TypeError:
            raise ValueError(f"space charge: grid={self.grid!r} is not two numbers "
                             "(nx, ny)") from None
        if len(g) != 2 or min(g) < 8 or max(g) > GRID_MAX:
            raise ValueError(f"space charge: grid={self.grid!r} must be two numbers (nx, ny) "
                             f"from 8 to {GRID_MAX}")
        self.grid = (g[0], g[1])
        self.box_sigma = _number(self.box_sigma, "box_sigma")
        if not 1.0 <= self.box_sigma <= SPAN_MAX:
            raise ValueError(f"space charge: box_sigma must be from 1 to {SPAN_MAX:g} "
                             "(rms sizes)")
        self.pipe_a_m = _number(self.pipe_a_m, "pipe_a_m", none_ok=True)
        self.pipe_b_m = _number(self.pipe_b_m, "pipe_b_m", none_ok=True)
        if self.pipe != "open":
            if self.pipe_a_m is None or not self.pipe_a_m > 0.0:
                raise ValueError(f"space charge: a {self.pipe} pipe needs pipe_a_m > 0 "
                                 + ("(its radius)" if self.pipe == "round" else
                                    "(its half width)"))
            if self.pipe in ("rectangle", "ellipse") and (
                    self.pipe_b_m is None or not self.pipe_b_m > 0.0):
                raise ValueError(f"space charge: a {self.pipe} pipe needs pipe_b_m > 0 "
                                 "(its half height)")
        self.reference_radius_m = _number(self.reference_radius_m, "reference_radius_m")
        if not self.reference_radius_m > 0.0:
            raise ValueError("space charge: reference_radius_m must be positive")
        self.boundary_order = _whole(self.boundary_order, "boundary_order")
        if not 2 <= self.boundary_order <= 64:
            raise ValueError("space charge: boundary_order must be from 2 to 64")
        self.g_factor = _number(self.g_factor, "g_factor", none_ok=True)
        if self.longitudinal == "g_factor" and (self.g_factor is None
                                                or not self.g_factor > 0.0):
            raise ValueError("space charge: longitudinal='g_factor' needs g_factor > 0")
        if isinstance(self.z_bins, str) and self.z_bins.strip().lower() == "auto":
            self.z_bins = "auto"
        else:
            self.z_bins = _whole(self.z_bins, "z_bins")
            if not 8 <= self.z_bins <= 1 << 16:
                raise ValueError("space charge: z_bins must be 'auto' or from 8 to 65536")
        self.z_smoothing_bins = _number(self.z_smoothing_bins, "z_smoothing_bins")
        if not 0.0 <= self.z_smoothing_bins <= SPAN_MAX:
            raise ValueError(f"space charge: z_smoothing_bins must be from 0 (none) to "
                             f"{SPAN_MAX:g}")
        self.refresh_turns = _whole(self.refresh_turns, "refresh_turns")
        if not 0 <= self.refresh_turns <= REFRESH_MAX:
            raise ValueError(f"space charge: refresh_turns must be from 0 (never) to "
                             f"{REFRESH_MAX}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid"] = list(self.grid)
        return d

    @classmethod
    def from_dict(cls, d) -> SpaceChargeConfig:
        if isinstance(d, SpaceChargeConfig):
            return cls(**d.to_dict())
        if not isinstance(d, dict):
            raise ValueError(f"space charge: {d!r} is not a table of settings")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"space charge: unknown key(s) {unknown} (known: {sorted(known)})")
        return cls(**d)


def space_charge_from_file(path) -> SpaceChargeConfig:
    """Space-charge settings from a JSON file (or YAML, with PyYAML installed):
    the keys of :class:`SpaceChargeConfig`."""
    import json
    from pathlib import Path
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"space-charge file: cannot read {p}: {exc}") from None
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ValueError(f"space-charge file {p.name}: YAML needs PyYAML, which is not "
                             "installed — write the settings as JSON") from None
        try:
            d = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"space-charge file {p.name}: not valid YAML ({exc})") from None
    else:
        try:
            d = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"space-charge file {p.name}: not valid JSON ({exc})") from None
    if not isinstance(d, dict):
        raise ValueError(f"space-charge file {p.name}: expected a mapping of the settings")
    return SpaceChargeConfig.from_dict(d)
