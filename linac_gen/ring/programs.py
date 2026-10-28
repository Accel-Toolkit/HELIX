"""Ring-mode programs (milestone M3): the reference energy, the RF voltage and
the RF phase, turn by turn.

The magnets follow the reference particle — their normalized strengths stay
the same — so a ramp is set by the reference energy alone.  Turn by turn, at
the start of each turn (t: the reference particle's own time since the start
of the cycle, the sum of node length / (beta c) over the turns before):

* the energy gain the turn needs: ``gain = E_prog(t + C/(beta c)) - E_ref``,
  the program's energy one revolution later minus the reference's energy now
  (the reference catches up with the program every turn);
* the RF voltage ``V`` from the voltage program, shared by the cavities in
  proportion to their own voltages (evenly when the lattice gives them none);
* the RF phase from the phase rule: ``phi_s = asin(gain/V)`` (sine
  convention: a cavity gives the reference ``V sin(phi_s)``), mirrored to
  ``pi - phi_s`` above transition (gamma at the start of the turn above
  gamma_t) — a gain beyond the voltage is capped at +-90 deg and counted, or
  refused; with paraphasing groups (each cavity's phase offset theta_i) the
  rule uses ``R = |sum V_i exp(i theta_i)|`` and subtracts its argument (a
  cavity whose voltage is negative — the lattice beam's charge, or the
  deck's sign — counts with theta_i + pi);
* the RF frequency ``f = h beta c / C``, which keeps every cavity locked to
  the reference particle (zeta wrapped by C/h after a turn stays exact; every
  cavity must be at the ring's harmonic).

Each cavity then gives every particle ``V_k sin(lag_k - 2 pi f zeta/(beta0
c))`` and the reference ``V_k sin(lag_k)`` (with a phase table, its share of
the program's gain: the reference follows the energy program); a
reference-change node after it re-expresses the particles relative to the
new reference (exact; the transverse momenta shrink as the momentum grows —
adiabatic damping).

Momentum programs: ``flat`` (the reference stays); ``sine``, the momentum of
a resonant magnet cycle, ``p = p_mid - p_amp cos(2 pi f (t - t_min))``,
lowest at t_min, highest (the top energy) half a cycle later, and passing
through the start momentum at ``t_start`` (held there before it);
``table`` (times and kinetic energies, momenta or total energies, linear
between the points, the end values held).

Program files (:func:`ramp_from_file`): JSON, or YAML, in the layout of
:meth:`RampConfig.to_dict`, each table inline or in a CSV file whose header
names its columns: ``t_s`` and one of ``kinetic_mev``, ``momentum_mevc``,
``total_mev`` (energy), ``voltage_mv`` or ``phi_deg``.
"""
from __future__ import annotations

import fnmatch
import math
from dataclasses import asdict, dataclass, field

import numpy as np

C_LIGHT = 299792458.0

ENERGY_KINDS = ("flat", "sine", "table")
TABLE_QUANTITIES = ("kinetic_mev", "momentum_mevc", "total_mev")
VOLTAGE_KINDS = ("constant", "table")
PHASE_KINDS = ("auto", "table")
ON_EXCESS = ("cap", "refuse")


class RampError(ValueError):
    """A program that cannot be run (the message says why)."""


def _table(times, values, what: str) -> tuple[tuple, tuple]:
    t = tuple(float(v) for v in (times or ()))
    v = tuple(float(x) for x in (values or ()))
    if len(t) < 1 or len(t) != len(v):
        raise RampError(f"{what}: give as many values as times (at least one point)")
    if any(not math.isfinite(x) for x in t + v):
        raise RampError(f"{what}: the table holds a value that is not finite")
    if any(b <= a for a, b in zip(t, t[1:], strict=False)):
        raise RampError(f"{what}: the times must increase")
    return t, v


def _interp(t: float, times: tuple, values: tuple) -> float:
    """Linear between the points, the end values held outside them."""
    return float(np.interp(t, times, values))


def _given(x) -> bool:
    """A table column with points in it (a list, tuple or array)."""
    return x is not None and len(x) > 0


def _number(v, what: str, *, none_ok: bool = False):
    """A finite float (a number written as text, as YAML reads '6e3', too);
    None when allowed."""
    if v is None and none_ok:
        return None
    if isinstance(v, bool):
        raise RampError(f"{what}: {v!r} is not a number")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise RampError(f"{what}: {v!r} is not a number") from None
    if not math.isfinite(x):
        raise RampError(f"{what}: {v!r} is not a finite number")
    return x


@dataclass
class EnergyProgram:
    """The reference momentum versus time (see the module docstring)."""

    kind: str = "flat"
    start_kinetic_mev: float | None = None     # sine: None = the ring's reference
    end_kinetic_mev: float | None = None       # sine: the top of the cycle
    frequency_hz: float = 0.0                  # sine: the cycle's frequency
    t_min_s: float = 0.0                       # sine: when the field is lowest
    t_start_s: float = 0.0                     # sine: when it reaches the start energy
    times_s: tuple = ()                        # table
    values: tuple = ()                         # table
    quantity: str = "kinetic_mev"              # table: what the values are

    def __post_init__(self):
        if self.kind not in ENERGY_KINDS:
            raise RampError(f"energy program: kind {self.kind!r} is not one of {list(ENERGY_KINDS)}")
        what = "energy program"
        self.start_kinetic_mev = _number(self.start_kinetic_mev, f"{what}: start_kinetic_mev",
                                         none_ok=True)
        self.end_kinetic_mev = _number(self.end_kinetic_mev, f"{what}: end_kinetic_mev",
                                       none_ok=True)
        self.frequency_hz = _number(self.frequency_hz, f"{what}: frequency_hz")
        self.t_min_s = _number(self.t_min_s, f"{what}: t_min_s")
        self.t_start_s = _number(self.t_start_s, f"{what}: t_start_s")
        if self.quantity not in TABLE_QUANTITIES:
            raise RampError(f"energy program: quantity {self.quantity!r} is not one of "
                            f"{list(TABLE_QUANTITIES)}")
        if self.kind == "sine":
            if not self.frequency_hz > 0.0:
                raise RampError("energy program: a sine cycle needs a frequency above 0")
            if self.end_kinetic_mev is None or not self.end_kinetic_mev > 0.0:
                raise RampError("energy program: a sine cycle needs its top kinetic energy "
                                "(above 0)")
            if self.start_kinetic_mev is not None and not self.start_kinetic_mev > 0.0:
                raise RampError("energy program: the start kinetic energy must be above 0")
            if self.t_start_s < self.t_min_s:
                raise RampError("energy program: t_start must not be before t_min (the field's "
                                "lowest point)")
            if 2.0 * math.pi * self.frequency_hz * (self.t_start_s - self.t_min_s) >= math.pi:
                raise RampError("energy program: t_start must be within half a cycle of t_min")
        if self.kind == "table" or _given(self.times_s) or _given(self.values):
            # (a table kept with another kind is checked too: it must load back)
            self.times_s, self.values = _table(self.times_s, self.values, "energy table")
            if any(not v > 0.0 for v in self.values):
                raise RampError("energy table: the values must be positive")

    # -- evaluation -------------------------------------------------------------
    def _cycle(self, mass: float, p_ref: float) -> tuple[float, float, float]:
        """(start momentum, centre, amplitude) of the resonant cycle
        p(t) = centre - amplitude cos(w (t - t_min)): it passes through the
        start momentum at t_start and reaches the top momentum (centre +
        amplitude) half a cycle after t_min."""
        p_start = (p_ref if self.start_kinetic_mev is None
                   else _p_from_kinetic(self.start_kinetic_mev, mass))
        p_top = _p_from_kinetic(self.end_kinetic_mev, mass)
        phase0 = 2.0 * math.pi * self.frequency_hz * (self.t_start_s - self.t_min_s)
        # centre - amplitude cos(phase0) = p_start, centre + amplitude = p_top
        amplitude = (p_top - p_start) / (1.0 + math.cos(phase0))
        return p_start, p_top - amplitude, amplitude

    def momentum(self, t: float, mass: float, p_ref: float) -> float:
        """The reference momentum [MeV/c] at time ``t`` [s] (``p_ref``: the
        ring's reference momentum, the flat program's — and a sine cycle's
        start when it gives none)."""
        if self.kind == "flat":
            return float(p_ref)
        if self.kind == "sine":
            p_start, centre, amplitude = self._cycle(mass, p_ref)
            if t < self.t_start_s:
                return float(p_start)
            return centre - amplitude * math.cos(2.0 * math.pi * self.frequency_hz
                                                 * (t - self.t_min_s))
        v = _interp(t, self.times_s, self.values)
        if self.quantity == "kinetic_mev":
            return _p_from_kinetic(v, mass)
        if self.quantity == "momentum_mevc":
            return v
        if v <= mass:
            raise RampError(f"energy table: a total energy of {v:g} MeV is below the rest "
                            f"energy ({mass:g} MeV)")
        return math.sqrt(v * v - mass * mass)

    def total_energy(self, t: float, mass: float, p_ref: float) -> float:
        p = self.momentum(t, mass, p_ref)
        return math.sqrt(p * p + mass * mass)

    def t_end(self) -> float | None:
        """When the program ends: the top of a sine cycle, a table's last
        time; None for a flat program."""
        if self.kind == "sine":
            return self.t_min_s + 1.0 / (2.0 * self.frequency_hz)
        if self.kind == "table":
            return float(self.times_s[-1])
        return None


@dataclass
class VoltageProgram:
    """The total RF voltage [MV] versus time: ``constant`` (the cavities' own
    voltages) or ``table`` (linear between the points, the end values held)."""

    kind: str = "constant"
    times_s: tuple = ()
    values_mv: tuple = ()

    def __post_init__(self):
        if self.kind not in VOLTAGE_KINDS:
            raise RampError(f"voltage program: kind {self.kind!r} is not one of "
                            f"{list(VOLTAGE_KINDS)}")
        if self.kind == "table" or _given(self.times_s) or _given(self.values_mv):
            # (a table kept with the constant kind is checked too: it must load back)
            self.times_s, self.values_mv = _table(self.times_s, self.values_mv, "voltage table")
            if any(v < 0.0 for v in self.values_mv):
                raise RampError("voltage table: the voltages must not be negative")

    def total(self, t: float, v_design: float) -> float:
        if self.kind == "constant":
            return float(v_design)
        return _interp(t, self.times_s, self.values_mv)


@dataclass
class PhaseProgram:
    """The RF phase: ``auto`` (from the gain the turn needs, see the module
    docstring) or ``table`` (the synchronous phase [deg, sine convention]
    versus time, used as it is: the table carries its own transition jump).
    ``on_excess``: a gain beyond the voltage is capped at +-90 deg ("cap",
    each such turn counted) or refused ("refuse")."""

    kind: str = "auto"
    times_s: tuple = ()
    values_deg: tuple = ()
    on_excess: str = "cap"
    # (with a table the reference particle — and the magnets — follow the
    # energy program, and the beam's energy follows the table's phase: a
    # mismatch moves the beam off the reference momentum, as in a machine
    # whose radial loop is open)

    def __post_init__(self):
        if self.kind not in PHASE_KINDS:
            raise RampError(f"phase program: kind {self.kind!r} is not one of {list(PHASE_KINDS)}")
        if self.on_excess not in ON_EXCESS:
            raise RampError(f"phase program: on_excess {self.on_excess!r} is not one of "
                            f"{list(ON_EXCESS)}")
        if self.kind == "table" or _given(self.times_s) or _given(self.values_deg):
            self.times_s, self.values_deg = _table(self.times_s, self.values_deg, "phase table")


@dataclass
class ParaphasingGroup:
    """Cavities whose phase is offset by ``angle_deg``: each entry a name
    pattern (fnmatch) or a whole number, the cavity's place in the ring
    (0 = the first after s = 0) — for cavities that share a name."""

    cavities: tuple = ()
    angle_deg: float = 0.0

    def __post_init__(self):
        out = []
        for c in (self.cavities or ()):
            if isinstance(c, bool):
                raise RampError(f"paraphasing group: {c!r} is not a cavity name or number")
            if isinstance(c, (int, np.integer)):
                if int(c) < 0:
                    raise RampError(f"paraphasing group: cavity number {c} is negative")
                out.append(int(c))
            else:
                out.append(str(c))
        self.cavities = tuple(out)
        if not self.cavities:
            raise RampError("paraphasing group: name at least one cavity (pattern or number)")
        self.angle_deg = _number(self.angle_deg, "paraphasing group: angle_deg")


@dataclass
class RampConfig:
    """A ramp: the programs, the transition energy and the clock."""

    energy: EnergyProgram = field(default_factory=EnergyProgram)
    voltage: VoltageProgram = field(default_factory=VoltageProgram)
    phase: PhaseProgram = field(default_factory=PhaseProgram)
    paraphasing: tuple = ()
    gamma_t: float | None = None        # None: from the ring's optics
    t_start_s: float = 0.0              # the clock at turn 0 [s]

    def __post_init__(self):
        for name, cls in (("energy", EnergyProgram), ("voltage", VoltageProgram),
                          ("phase", PhaseProgram)):
            v = getattr(self, name)
            if isinstance(v, dict):
                setattr(self, name, _from_dict(cls, v, f"ramp {name}"))
            elif not isinstance(v, cls):
                raise RampError(f"ramp: {name} must be a {cls.__name__} or a dict")
        groups = []
        for g in self.paraphasing or ():
            groups.append(_from_dict(ParaphasingGroup, g, "paraphasing group")
                          if isinstance(g, dict) else g)
        self.paraphasing = tuple(groups)
        self.gamma_t = _number(self.gamma_t, "ramp: gamma_t", none_ok=True)
        if self.gamma_t is not None and not self.gamma_t > 1.0:
            raise RampError("ramp: gamma_t must be above 1")
        self.t_start_s = _number(self.t_start_s, "ramp: t_start_s")

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("energy", "voltage", "phase"):
            for kk, vv in d[k].items():
                if isinstance(vv, tuple):
                    d[k][kk] = list(vv)
        d["paraphasing"] = [{"cavities": list(g["cavities"]), "angle_deg": g["angle_deg"]}
                            for g in d["paraphasing"]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RampConfig:
        return _from_dict(cls, d, "ramp")


def _from_dict(cls, d, what):
    import dataclasses
    if not isinstance(d, dict):
        raise RampError(f"{what}: expected a mapping, got {type(d).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise RampError(f"{what}: unknown key(s) {unknown} (known: {sorted(known)})")
    return cls(**d)


def _p_from_kinetic(w: float, mass: float) -> float:
    return math.sqrt(w * w + 2.0 * w * mass)


# ---------------------------------------------------------------------------
# program files: a ramp as JSON (or YAML), its tables inline or in CSV files
# ---------------------------------------------------------------------------
#: the value columns a program's CSV table may have (its header names them;
#: the first column is the time, ``t_s``)
TABLE_COLUMNS = {"energy": TABLE_QUANTITIES, "voltage": ("voltage_mv",),
                 "phase": ("phi_deg",)}
#: where each program keeps its table's values
_VALUE_KEY = {"energy": "values", "voltage": "values_mv", "phase": "values_deg"}


def read_table_csv(path, program: str) -> tuple[tuple, tuple, str]:
    """A program's table from a CSV file: a header row ``t_s,<column>`` (the
    column one of :data:`TABLE_COLUMNS` for the program), then one row per
    point; blank lines and lines starting with ``#`` are skipped.  Returns
    (times [s], values, column)."""
    import csv
    from pathlib import Path
    allowed = TABLE_COLUMNS[program]
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RampError(f"{program} table: cannot read {p}: {exc}") from None
    try:
        rows = [r for r in csv.reader(line for line in text.splitlines()
                                      if line.strip() and not line.lstrip().startswith("#"))]
    except csv.Error as exc:
        # (e.g. not a CSV file: a field longer than the reader's limit)
        raise RampError(f"{program} table {p.name}: not a CSV table ({exc})") from None
    if not rows:
        raise RampError(f"{program} table {p.name}: the file is empty")
    head = [h.strip().lower() for h in rows[0]]
    if len(head) != 2 or head[0] != "t_s" or head[1] not in allowed:
        raise RampError(f"{program} table {p.name}: the header must be 't_s,<column>' with "
                        f"the column one of {list(allowed)} (got {rows[0]})")
    times, values = [], []
    for k, r in enumerate(rows[1:], start=2):
        if len(r) != 2:
            raise RampError(f"{program} table {p.name}: row {k} has {len(r)} values, not 2")
        try:
            times.append(float(r[0]))
            values.append(float(r[1]))
        except ValueError:
            raise RampError(f"{program} table {p.name}: row {k} is not two numbers "
                            f"({r})") from None
    t, v = _table(times, values, f"{program} table {p.name}")
    return t, v, head[1]


def write_table_csv(path, times, values, column: str) -> None:
    """Write a program's table as :func:`read_table_csv` reads it (the numbers
    in full precision)."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", column])
        for t, v in zip(times, values, strict=True):
            w.writerow([repr(float(t)), repr(float(v))])


def ramp_from_file(path) -> RampConfig:
    """A ramp from a JSON file (or YAML, with PyYAML installed) in the layout
    of :meth:`RampConfig.to_dict`.  A table may be given as ``"file":
    "name.csv"`` (relative to the ramp file; see :func:`read_table_csv`)
    instead of its times and values; for the energy, the CSV's column sets
    the quantity."""
    import json
    from pathlib import Path
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise RampError(f"ramp file: cannot read {p}: {exc}") from None
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise RampError(f"ramp file {p.name}: YAML needs PyYAML, which is not installed "
                            "— write the ramp as JSON") from None
        try:
            d = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise RampError(f"ramp file {p.name}: not valid YAML ({exc})") from None
    else:
        try:
            d = json.loads(text)
        except ValueError as exc:
            raise RampError(f"ramp file {p.name}: not valid JSON ({exc})") from None
    if not isinstance(d, dict):
        raise RampError(f"ramp file {p.name}: expected a mapping of the programs")
    d = dict(d)
    for program, key in _VALUE_KEY.items():
        sub = d.get(program)
        if not isinstance(sub, dict) or "file" not in sub:
            continue
        sub = dict(sub)
        f = sub.pop("file")
        if "times_s" in sub or key in sub:
            raise RampError(f"ramp file {p.name}: the {program} program gives both a file and "
                            "a table")
        if sub.get("kind", "table") != "table":
            raise RampError(f"ramp file {p.name}: the {program} program gives a file but is "
                            f"of kind {sub.get('kind')!r}")
        csv_path = Path(f) if Path(f).is_absolute() else p.parent / f
        times, values, column = read_table_csv(csv_path, program)
        if program == "energy":
            if "quantity" in sub and sub["quantity"] != column:
                raise RampError(f"ramp file {p.name}: the energy table's quantity "
                                f"{sub['quantity']!r} is not its file's column {column!r}")
            sub["quantity"] = column
        sub.update(kind="table", times_s=list(times), **{key: list(values)})
        d[program] = sub
    return RampConfig.from_dict(d)


# ---------------------------------------------------------------------------
# turn-by-turn evaluation on a compiled ring
# ---------------------------------------------------------------------------
@dataclass
class TurnPlan:
    """What one turn applies (see :meth:`Ramp.plan`)."""

    turn: int
    t_s: float                 # the reference's time at the start of the turn
    energy_mev: float          # its total energy then
    gain_mev: float            # the gain the turn needs
    voltage_mv: float          # the total voltage
    phi_deg: float             # the phase rule's phase (before the groups' offsets)
    frequency_hz: float
    above_transition: bool
    capped: bool
    lags_deg: np.ndarray       # per cavity
    volts_mv: np.ndarray       # per cavity
    dE_mev: np.ndarray         # the reference's gain at each cavity
    energy_after_mev: np.ndarray   # the reference's energy after each cavity
    turn_time_s: float         # the reference's time for the turn
    energy_end_mev: float      # its energy at the end of the turn


class Ramp:
    """A :class:`RampConfig` on a compiled ring (compiled with its cavities
    followed by reference-change nodes): :meth:`plan` evaluates a turn,
    :meth:`apply` writes it into the ring's node table, :meth:`orbit` runs the
    reference particle alone."""

    def __init__(self, cfg: RampConfig, rl, *, gamma_t: float | None = None,
                 corrector_mode: str = "follow"):
        self.cfg = cfg
        self.rl = rl
        if corrector_mode not in ("follow", "fixed_field"):
            raise RampError(f"corrector_mode {corrector_mode!r} is not 'follow' or 'fixed_field'")
        self.corrector_mode = corrector_mode
        # steerers that keep their field: kick = (field / the lattice beam's
        # rigidity) x (the lattice beam's momentum / the reference's)
        self.p_lattice = abs(float(rl.brho_norm)) * C_LIGHT * 1e-6
        corr = list(getattr(rl, "correctors", []) or [])
        self.corr_nodes = np.array([c["node"] for c in corr], dtype=np.int64)
        self.corr_kicks = np.array([c["kick"] for c in corr], dtype=float).reshape(-1, 2)
        ref = rl.reference
        self.mass = float(ref.mass)
        self.p_ref = float(ref.p0c)
        self.length = float(rl.length)
        self.gamma_t = (float(cfg.gamma_t) if cfg.gamma_t is not None
                        else (None if gamma_t is None else float(gamma_t)))
        cav = list(rl.cavities)
        if not cav:
            raise RampError("a ramp needs RF cavities in the ring")
        if any("refshift_node" not in c for c in cav):
            raise RampError("the ring was compiled without reference-change nodes (compile it "
                            "with reference_following=True for a ramp)")
        self.h = rl.harmonic
        if not self.h:
            raise RampError("a ramp needs the cavities at a harmonic of the revolution frequency "
                            "(HARMON, or the ring's harmonic number)")
        # (f = h beta c / C every turn: one harmonic for all, none at a fixed
        # frequency)
        fixed = [str(c["name"]) for c in cav if not c.get("harmonic")]
        if fixed:
            raise RampError(f"cavities at a fixed frequency (no harmonic number): {fixed[:4]} — "
                            "a ramp needs every cavity at a harmonic of the revolution frequency "
                            "(HARMON, or the ring's harmonic number)")
        others = sorted({int(c["harmonic"]) for c in cav} - {int(self.h)})
        if others:
            raise RampError(f"cavities at harmonics {others} besides {int(self.h)}: a ramp sets "
                            "every cavity at the ring's one harmonic")
        self.cav_nodes = np.array([c["node"] for c in cav], dtype=np.int64)
        self.shift_nodes = np.array([c["refshift_node"] for c in cav], dtype=np.int64)
        order = np.argsort(self.shift_nodes, kind="stable")
        if not np.array_equal(order, np.arange(len(cav))):
            raise RampError("internal: the cavities are not in node order")
        v = np.array([abs(float(c["voltage_mv"])) for c in cav])
        self.v_design = float(v.sum())
        self.share = v / v.sum() if v.sum() > 0.0 else np.full(len(cav), 1.0 / len(cav))
        # the charge signs the voltage (an H- deck's cavities)
        self.sign = np.array([math.copysign(1.0, float(c["voltage_mv"])) if c["voltage_mv"]
                              else 1.0 for c in cav])
        theta = np.zeros(len(cav))
        grouped = np.zeros(len(cav), dtype=bool)
        names = [str(c["name"]) for c in cav]
        for g in cfg.paraphasing:
            bad = [c for c in g.cavities if isinstance(c, int) and c >= len(names)]
            if bad:
                raise RampError(f"paraphasing: cavity number(s) {bad} beyond the ring's "
                                f"{len(names)} cavities")
            hit = [k for k, nm in enumerate(names)
                   if any((pat == k) if isinstance(pat, int) else fnmatch.fnmatchcase(nm, pat)
                          for pat in g.cavities)]
            if not hit:
                raise RampError(f"paraphasing: no cavity matches {list(g.cavities)} "
                                f"(cavities: {names[:6]}{' ...' if len(names) > 6 else ''})")
            twice = [k for k in hit if grouped[k]]
            if twice:
                raise RampError(f"paraphasing: cavity number(s) {twice} are in two groups")
            grouped[hit] = True
            theta[hit] = math.radians(g.angle_deg)
        self.theta = theta
        # (a negative voltage gives the reference V_k sin(lag_k + pi): the rule
        # counts that cavity with its offset plus pi)
        theta_eff = theta + np.where(self.sign < 0.0, math.pi, 0.0)
        A = float(np.sum(self.share * np.cos(theta_eff)))
        B = float(np.sum(self.share * np.sin(theta_eff)))
        self.r_unit = math.hypot(A, B)            # R / V
        self.arg = math.atan2(B, A)
        if self.r_unit <= 1e-12:
            raise RampError("the cavities' voltages cancel (no net voltage): their signs or "
                            "the paraphasing groups' angles")
        # node lengths and the segment of each node (0 before the first
        # reference change, k after the k-th)
        self.node_len = np.asarray(rl.node_length, dtype=float)
        seg = np.zeros(rl.n_nodes, dtype=np.int64)
        for k, j in enumerate(self.shift_nodes):
            seg[j + 1:] = k + 1
        self.node_seg = seg
        self.seg_len = np.bincount(seg, weights=self.node_len, minlength=len(cav) + 1)
        self.n_capped = 0

    # -- the reference --------------------------------------------------------
    def start(self) -> tuple[float, float]:
        """(t [s], total energy [MeV]) of the reference at turn 0."""
        t0 = self.cfg.t_start_s
        return t0, self.cfg.energy.total_energy(t0, self.mass, self.p_ref)

    def _beta(self, energy: float) -> float:
        gamma = energy / self.mass
        return math.sqrt(1.0 - 1.0 / (gamma * gamma))

    def plan(self, turn: int, t: float, energy: float) -> TurnPlan:
        """The turn starting at time ``t`` with the reference at total energy
        ``energy`` (see the module docstring)."""
        cfg = self.cfg
        beta = self._beta(energy)
        gamma = energy / self.mass
        dt = self.length / (beta * C_LIGHT)
        e_next = cfg.energy.total_energy(t + dt, self.mass, self.p_ref)
        gain = e_next - energy
        V = cfg.voltage.total(t, self.v_design)
        above = self.gamma_t is not None and gamma > self.gamma_t
        capped = False
        if cfg.phase.kind == "auto":
            R = V * self.r_unit
            # (no voltage: any gain is beyond it)
            x = (gain / R if R > 0.0
                 else (0.0 if gain == 0.0 else math.copysign(math.inf, gain)))
            if abs(x) > 1.0:
                if cfg.phase.on_excess == "refuse":
                    raise RampError(f"turn {turn} (t = {t * 1e3:.6g} ms): the ramp needs "
                                    f"{gain:.6g} MeV per turn, more than the RF gives "
                                    f"({R:.6g} MV)")
                capped = True
                self.n_capped += 1
                x = math.copysign(1.0, x)
            phi = math.asin(x)
            if above:
                phi = math.pi - phi
            phi -= self.arg
        else:
            # (the table's phase is that of the net voltage)
            phi = math.radians(_interp(t, cfg.phase.times_s, cfg.phase.values_deg)) - self.arg
        lags = np.degrees(phi + self.theta)
        volts = V * self.share * self.sign
        # auto: the reference is the particle at zeta = 0, it gains what its
        # cavity gives at the lag; a phase table: the reference (and the
        # magnets) follow the energy program, and a beam at another phase
        # gains or loses energy against it
        dE = (volts * np.sin(np.radians(lags)) if cfg.phase.kind == "auto"
              else gain * self.share)
        e_after = energy + np.cumsum(dE)
        # the reference's time over the turn: each segment at its own velocity
        e_seg = np.concatenate([[energy], e_after])
        g = e_seg / self.mass
        b_seg = np.sqrt(1.0 - 1.0 / (g * g))
        turn_time = float(np.sum(self.seg_len / b_seg)) / C_LIGHT
        f = self.h * beta * C_LIGHT / self.length
        return TurnPlan(turn=int(turn), t_s=float(t), energy_mev=float(energy),
                        gain_mev=float(gain), voltage_mv=float(V), phi_deg=math.degrees(phi),
                        frequency_hz=float(f), above_transition=bool(above), capped=capped,
                        lags_deg=lags, volts_mv=volts, dE_mev=dE, energy_after_mev=e_after,
                        turn_time_s=turn_time, energy_end_mev=float(e_after[-1]))

    def apply(self, plan: TurnPlan, params, ref_beta0, ref_p0c, ref_bg2) -> None:
        """Write the turn into the node table (in place): each cavity's
        voltage, frequency and lag, each reference change's gain and new
        reference, and every node's reference."""
        rl = self.rl
        e_seg = np.concatenate([[plan.energy_mev], plan.energy_after_mev])
        g = e_seg / self.mass
        bg = np.sqrt(g * g - 1.0)
        p0c = bg * self.mass
        b0 = bg / g
        for k in range(len(self.cav_nodes)):
            o = rl.offsets[self.cav_nodes[k]]
            params[o] = plan.volts_mv[k]
            params[o + 1] = plan.frequency_hz
            params[o + 2] = plan.lags_deg[k]
            o = rl.offsets[self.shift_nodes[k]]
            params[o] = plan.dE_mev[k]
            params[o + 1] = p0c[k + 1]
            params[o + 2] = b0[k + 1]
        s = self.node_seg
        ref_beta0[:] = b0[s]
        ref_p0c[:] = p0c[s]
        ref_bg2[:] = (bg * bg)[s]
        if self.corrector_mode == "fixed_field" and self.corr_nodes.size:
            scale = self.p_lattice / ref_p0c[self.corr_nodes]
            for i, j in enumerate(self.corr_nodes):
                o = rl.offsets[j]
                params[o] = self.corr_kicks[i, 0] * scale[i]
                params[o + 1] = self.corr_kicks[i, 1] * scale[i]

    def stationary_equivalent(self, plan: TurnPlan):
        """A copy of the ring at the energy of the start of ``plan``'s turn
        with each cavity at lag 0 and voltage V_k cos(lag_k), at the turn's
        frequency, the reference changes doing nothing: the linear RF focusing
        of the accelerating bucket about a stationary closed orbit (the
        optics of the turn, and the matched beam's)."""
        import dataclasses
        rl = self.rl
        g = plan.energy_mev / self.mass
        bg = math.sqrt(g * g - 1.0)
        p0c, b0 = bg * self.mass, bg / g
        p = rl.params.copy()
        for k in range(len(self.cav_nodes)):
            o = rl.offsets[self.cav_nodes[k]]
            p[o] = plan.volts_mv[k] * math.cos(math.radians(plan.lags_deg[k]))
            p[o + 1] = plan.frequency_hz
            p[o + 2] = 0.0
            o = rl.offsets[self.shift_nodes[k]]
            p[o], p[o + 1], p[o + 2] = 0.0, p0c, b0
        n = rl.n_nodes
        rb, rp, rg = np.full(n, b0), np.full(n, p0c), np.full(n, bg * bg)
        if self.corrector_mode == "fixed_field" and self.corr_nodes.size:
            for i, j in enumerate(self.corr_nodes):
                o = rl.offsets[j]
                p[o] = self.corr_kicks[i, 0] * self.p_lattice / p0c
                p[o + 1] = self.corr_kicks[i, 1] * self.p_lattice / p0c
        ref = dataclasses.replace(rl.reference, p0c=p0c)
        return dataclasses.replace(rl, params=p, ref_beta0=rb, ref_p0c=rp, ref_bg2=rg,
                                   reference=ref)

    def turns_to_end(self, max_turns: int = 50_000_000, should_stop=None) -> int | None:
        """Turns from t_start to the end of the program on the program's
        own clock (t += C/(beta(E_prog(t)) c) until the end); None for a flat
        program.  More than ``max_turns`` is refused; ``should_stop()``, when
        given, is asked every 4096 turns (OperationCancelled when it says
        yes)."""
        t_end = self.cfg.energy.t_end()
        if t_end is None:
            return None
        t, n = self.cfg.t_start_s, 0
        while t < t_end:
            e = self.cfg.energy.total_energy(t, self.mass, self.p_ref)
            t += self.length / (self._beta(e) * C_LIGHT)
            n += 1
            if n > max_turns:
                raise RampError(f"the program needs more than {max_turns:,} turns")
            if should_stop is not None and n % 4096 == 0:
                _poll(should_stop)
        return n

    def orbit(self, n_turns: int, should_stop=None) -> dict:
        """The reference particle alone for ``n_turns`` turns: per turn (at its
        start) the time, energy, gain, voltage, phase, frequency and whether
        it is above transition or capped; and the end, the turn transition is
        crossed (None: not crossed; ``above_transition_at_start``).
        ``should_stop()``, when given, is asked every 4096 turns
        (OperationCancelled when it says yes)."""
        t, e = self.start()
        if n_turns < 1:
            end = self.cfg.energy.t_end()
            raise RampError("no turns to run" + (
                f": the clock at turn 0 (t = {t * 1e3:.9g} ms) is at or after the end of the "
                f"energy program (t = {end * 1e3:.9g} ms)" if end is not None and t >= end
                else ""))
        rows = {k: np.empty(n_turns) for k in ("t_s", "energy_mev", "gain_mev", "voltage_mv",
                                               "phi_deg", "frequency_hz")}
        above = np.zeros(n_turns, dtype=bool)
        capped = np.zeros(n_turns, dtype=bool)
        for n in range(n_turns):
            if should_stop is not None and n % 4096 == 4095:
                _poll(should_stop)
            pl = self.plan(n, t, e)
            rows["t_s"][n], rows["energy_mev"][n] = t, e
            rows["gain_mev"][n], rows["voltage_mv"][n] = pl.gain_mev, pl.voltage_mv
            rows["phi_deg"][n], rows["frequency_hz"][n] = pl.phi_deg, pl.frequency_hz
            above[n], capped[n] = pl.above_transition, pl.capped
            t += pl.turn_time_s
            e = pl.energy_end_mev
        rows.update(above_transition=above, capped=capped, t_end_s=t, energy_end_mev=e,
                    gamma_end=e / self.mass, transition_turn=transition_crossing(above),
                    above_transition_at_start=bool(above[0]))
        return rows


def transition_crossing(above) -> int | None:
    """The first turn on the other side of transition from the turn before
    it (up or down); None when the turns never cross it."""
    a = np.asarray(above, dtype=bool)
    k = np.flatnonzero(a[1:] != a[:-1])
    return int(k[0]) + 1 if k.size else None


def _poll(should_stop) -> None:
    if should_stop():
        from linac_gen.core.cancelled import OperationCancelled
        raise OperationCancelled("ramp cancelled")
