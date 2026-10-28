"""The injected beam: a pool of particles, drawn turn by turn and put into ring
coordinates at the foil.

The pool holds the beam in ring coordinates relative to its own reference —
its species at its own kinetic energy (:class:`SourcePool`).  Each turn takes
the next ``n`` particles of the pool — in a seeded shuffle of it, or in its
own order — index (t n + i) mod N for the t-th injection turn, so a turn's
particles depend on the turn alone (a restart draws them again).  At the
foil an H⁻ is stripped to a proton of the same velocity and direction (the
momentum scales by m_p/m_H⁻), a proton beam is re-expressed relative to the
ring's reference, and the injected beam's centroid (its reference
trajectory, in ring coordinates) is added.

Sources: the Beam tab's distribution (HELIX's generators, or its file), a
particle file — a TraceWin ``.dst``, HELIX's ASCII distribution (its ``#
w_kin_ref``, ``# species:`` and ``# frequency:`` header lines) or TraceWin's
particle ASCII (the header line ``Number_of_particle Mass(MeV)
Beam_energy(MeV) Beam_Frequency(MHz) ...``) — or a Gaussian with Twiss
parameters at the foil.  The settings' ``kinetic_mev`` moves a beam to that
energy: its reference energy, the particles' offsets from it kept.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from linac_gen.ring import coords as C
from linac_gen.ring.injection.config import CUTOFF_MIN, InjectionError
from linac_gen.ring.reference import RingReference

C_LIGHT = 299792458.0
#: the random streams of an injection (default_rng([seed, PURPOSE, ...]))
PURPOSE_POOL = 101
PURPOSE_BUNCH = 102
PURPOSE_FOIL = 103
PURPOSE_FOIL_INJECTED = 104
PURPOSE_GAUSSIAN = 105
#: the most particles a generated pool may hold (48 bytes each)
POOL_MAX = 20_000_000


@dataclass
class SourcePool:
    """The injected beam: ``X`` (6, N) in ring coordinates relative to
    ``reference`` (the beam's species at its own kinetic energy, the ring's
    constants); ``frequency_mhz`` its bunch frequency; ``origin`` where it
    came from (for the run's summary)."""

    X: np.ndarray
    reference: RingReference
    species: str
    frequency_mhz: float
    origin: str

    @property
    def n(self) -> int:
        return int(self.X.shape[1])

    def order(self, seed: int, draw: str) -> np.ndarray | None:
        """The pool's order: a seeded permutation ("shuffled"), or None (its
        own order)."""
        if draw == "in_order":
            return None
        return np.random.default_rng([int(seed), PURPOSE_POOL]).permutation(self.n)

    def indices(self, k: int, n: int, perm) -> np.ndarray:
        """The pool indices of the k-th injection turn's n particles."""
        i = (np.arange(n, dtype=np.int64) + np.int64(k) * np.int64(n)) % self.n
        return i if perm is None else perm[i]


def _species_of_mass(mass: float) -> str:
    """'H-' or 'proton' from a file's rest mass [MeV] (within 0.2 MeV)."""
    from linac_gen.core.particle import H_MINUS, PROTON
    if abs(mass - H_MINUS.mass) < 0.2:
        return "H-"
    if abs(mass - PROTON.mass) < 0.2:
        return "proton"
    raise InjectionError(f"injection: the source's rest mass {mass:.6g} MeV is neither an H- "
                         "nor a proton — give its species")


def _species_of_name(name: str, origin: str) -> str:
    """'H-' or 'proton' from a file's species name."""
    low = name.strip().lower()
    if low in ("h-", "hminus", "h_minus"):
        return "H-"
    if low in ("proton", "p"):
        return "proton"
    raise InjectionError(f"injection: the source ({origin}) holds {name.strip()!r} — ring "
                         "injection takes an H- or a proton beam")


def _helix_header(path) -> dict:
    """The ``# species:`` and ``# frequency:`` lines of HELIX's ASCII
    distribution (as :func:`linac_gen.io.distribution_io.export_distribution`
    writes them): {'species': name, 'frequency_mhz': value} for those found."""
    import re
    out = {}
    with open(path, encoding="latin-1") as fh:
        for ln in fh:
            t = ln.strip()
            if not t:
                continue
            if not t.startswith("#"):
                break
            m = re.match(r"#\s*species\s*:\s*(\S+)", t, re.IGNORECASE)
            if m:
                out["species"] = m.group(1)
            m = re.match(r"#\s*frequency\s*:\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)", t,
                         re.IGNORECASE)
            if m:
                out["frequency_mhz"] = float(m.group(1))
    return out


def _reference(species: str, kinetic: float, constants) -> RingReference:
    return RingReference.from_kinetic("h-" if species == "H-" else "proton", float(kinetic),
                                      constants=constants)


def _from_linac(P: np.ndarray, *, species: str, kinetic: float, frequency: float,
                constants, origin: str) -> SourcePool:
    """A pool from HELIX linac coordinates (N, 6) [mm, mrad, mm, mrad, deg, MeV]
    relative to (species, kinetic) at ``frequency`` [MHz]."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 6 or P.shape[0] == 0:
        raise InjectionError(f"injection: the source ({origin}) has no particles")
    if not np.all(np.isfinite(P)):
        raise InjectionError(f"injection: the source ({origin}) holds a value that is not "
                             "finite")
    ref = _reference(species, kinetic, constants)
    X = C.from_helix(P, w_ref=float(kinetic), mass=ref.mass, frequency_mhz=float(frequency),
                     ring_ref=ref)
    return SourcePool(np.ascontiguousarray(X), ref, species, float(frequency), origin)


def read_tracewin_particles(path) -> tuple[np.ndarray, dict]:
    """TraceWin's particle ASCII file: a line naming ``Number_of_particle
    Mass(MeV) Beam_energy(MeV) Beam_Frequency(MHz) Beam_courant(A)
    Beam_Charge`` and the line of their values, the column names, then one
    particle per line with x(mm) x'(mrad) y(mm) y'(mrad) z(mm) z'(mrad)
    Phase(deg) Time(s) Energy(MeV) Loss.  Returns the particles kept (Loss 0)
    as HELIX linac coordinates (N, 6) — the phase is the delay at the beam
    frequency (> 0 late), dW the energy minus the beam energy — and the
    header (mass_mev, kinetic_mev, frequency_mhz, current_a, charge, n_file,
    n_lost)."""
    from pathlib import Path
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="latin-1").splitlines() if ln.strip()]
    if len(lines) < 4 or not lines[0].lower().startswith("number_of_particle"):
        raise InjectionError(f"injection: {p.name} is not a TraceWin particle file")
    try:
        vals = [float(v) for v in lines[1].split()]
        n_file, mass, energy, freq = int(vals[0]), vals[1], vals[2], vals[3]
        current = vals[4] if len(vals) > 4 else 0.0
        charge = vals[5] if len(vals) > 5 else 1.0
    except (ValueError, IndexError):
        raise InjectionError(f"injection: {p.name}: the header values are not numbers") from None
    cols = [c.lower() for c in lines[2].split()]
    want = {"x(mm)": None, "x'(mrad)": None, "y(mm)": None, "y'(mrad)": None,
            "phase(deg)": None, "energy(mev)": None, "loss": None}
    for k in want:
        if k in cols:
            want[k] = cols.index(k)
    missing = [k for k, v in want.items() if v is None and k != "loss"]
    if missing:
        raise InjectionError(f"injection: {p.name}: columns {missing} are missing")
    try:
        data = np.array([[float(v) for v in ln.split()] for ln in lines[3:]], dtype=float)
    except ValueError:
        raise InjectionError(f"injection: {p.name}: a particle line is not numbers") from None
    if data.ndim != 2 or data.shape[1] < len(cols):
        raise InjectionError(f"injection: {p.name}: the particle lines do not have "
                             f"{len(cols)} columns")
    keep = np.ones(data.shape[0], dtype=bool)
    if want["loss"] is not None:
        keep = data[:, want["loss"]] == 0.0
    d = data[keep]
    P = np.empty((d.shape[0], 6))
    P[:, 0] = d[:, want["x(mm)"]]
    P[:, 1] = d[:, want["x'(mrad)"]]
    P[:, 2] = d[:, want["y(mm)"]]
    P[:, 3] = d[:, want["y'(mrad)"]]
    P[:, 4] = d[:, want["phase(deg)"]]
    P[:, 5] = d[:, want["energy(mev)"]] - energy
    return P, {"mass_mev": mass, "kinetic_mev": energy, "frequency_mhz": freq,
               "current_a": current, "charge": charge, "n_file": n_file,
               "n_lost": int((~keep).sum())}


def _file_pool(cfg, constants) -> SourcePool:
    from pathlib import Path
    p = Path(cfg.path)
    if not p.is_file():
        raise InjectionError(f"injection: the source file {cfg.path} does not exist")
    origin = f"file {p.name}"
    if p.suffix.lower() == ".dst":
        from linac_gen.io.tracewin_dst import load_dst
        try:
            P, head = load_dst(str(p))
        except (OSError, ValueError) as exc:
            raise InjectionError(f"injection: {p.name}: {exc}") from None
        species = cfg.species or _species_of_mass(float(head["mass_MeV"]))
        kinetic = cfg.kinetic_mev if cfg.kinetic_mev is not None else float(head["w_kin_ref"])
        freq = cfg.frequency_mhz or float(head["frequency_MHz"])
        return _from_linac(P, species=species, kinetic=kinetic, frequency=freq,
                           constants=constants, origin=origin)
    first = ""
    with open(p, encoding="latin-1") as fh:
        for ln in fh:
            if ln.strip():
                first = ln.strip().lower()
                break
    if first.startswith("number_of_particle"):
        P, head = read_tracewin_particles(p)
        if cfg.species:
            species = cfg.species
        elif head["charge"] < 0:
            species = "H-"
        else:
            species = _species_of_mass(head["mass_mev"])
        kinetic = head["kinetic_mev"] if cfg.kinetic_mev is None else cfg.kinetic_mev
        freq = cfg.frequency_mhz or head["frequency_mhz"]
        return _from_linac(P, species=species, kinetic=kinetic, frequency=freq,
                           constants=constants, origin=origin)
    # HELIX's ASCII distribution: absolute phases and energies, offsets from its
    # # w_kin_ref (without one, from kinetic_mev)
    from linac_gen.distributions.from_file import load_distribution
    try:
        head = _helix_header(p)
        P, ref_head = load_distribution(str(p))
        w_file = ref_head.get("w_kin_ref")
        if w_file is None and cfg.kinetic_mev is not None:
            P, ref_head = load_distribution(str(p), ref_w_kin=cfg.kinetic_mev)
    except (OSError, ValueError) as exc:
        raise InjectionError(f"injection: {p.name}: {exc}") from None
    if w_file is None and cfg.kinetic_mev is None:
        raise InjectionError(f"injection: {p.name} gives no reference energy (# w_kin_ref) — "
                             "give the source's kinetic_mev")
    kinetic = cfg.kinetic_mev if cfg.kinetic_mev is not None else w_file
    if cfg.species:
        species = cfg.species
    elif head.get("species"):
        species = _species_of_name(head["species"], origin)
    else:
        raise InjectionError(f"injection: {p.name} does not say which particles it holds (no "
                             "# species: line) — give the source's species")
    freq = cfg.frequency_mhz or head.get("frequency_mhz")
    if not freq:
        raise InjectionError(f"injection: {p.name} gives no frequency (# frequency:) — give "
                             "the source's frequency_mhz (the unit of its phases)")
    return _from_linac(P, species=species, kinetic=float(kinetic), frequency=float(freq),
                       constants=constants, origin=origin)


def _beam_pool(cfg, beam_config, n_needed: int, seed: int, constants) -> SourcePool:
    import dataclasses

    from linac_gen.distributions.factory import create_beam
    if beam_config is None:
        raise InjectionError("injection: the 'beam' source needs the Beam tab's settings")
    species = cfg.species or ("H-" if str(beam_config.species).lower() in ("h-", "hminus")
                              else "proton")
    if str(beam_config.species).lower() not in ("h-", "proton", "p"):
        raise InjectionError(f"injection: the Beam tab's species {beam_config.species!r} is not "
                             "an H- or a proton beam")
    bc = beam_config
    if getattr(bc, "source", None) != "file":
        if n_needed > POOL_MAX:
            raise InjectionError(f"injection: {n_needed} particles to generate are more than "
                                 f"{POOL_MAX}")
        bc = dataclasses.replace(bc, n_particles=int(n_needed))
    beam = create_beam(bc, seed=int(seed))
    alive = ~np.asarray(beam.lost, dtype=bool)
    P = np.asarray(beam.particles, dtype=float)[alive]
    kinetic = float(beam.ref.w_kin) if cfg.kinetic_mev is None else cfg.kinetic_mev
    freq = cfg.frequency_mhz or float(beam.ref.frequency)
    origin = ("the Beam tab's file" if getattr(bc, "source", None) == "file"
              else f"the Beam tab's {bc.distribution} distribution")
    return _from_linac(P, species=species, kinetic=kinetic, frequency=freq,
                       constants=constants, origin=origin)


def _gaussian_pool(cfg, n_needed: int, seed: int, constants) -> SourcePool:
    """A Gaussian at the foil, in ring coordinates relative to the beam's own
    reference: x = sqrt(eps beta) u, x' = sqrt(eps/beta) (v - alpha u) (slopes,
    converted to px exactly), delta and zeta Gaussian; each pair cut at
    ``cutoff`` sigma in amplitude."""
    if n_needed > POOL_MAX:
        raise InjectionError(f"injection: {n_needed} particles to generate are more than "
                             f"{POOL_MAX}")
    cut = cfg.cutoff
    if cut is not None and not cut >= CUTOFF_MIN:
        # (as the settings refuse it: a pair falls inside the cut with
        # probability 1 - exp(-cut^2/2), and is drawn again until it does)
        raise InjectionError(f"injection: source cutoff must be at least {CUTOFF_MIN:g} "
                             "(sigma; leave it out for no cut)")
    species = cfg.species or "H-"
    ref = _reference(species, cfg.kinetic_mev, constants)
    rng = np.random.default_rng([int(seed), PURPOSE_GAUSSIAN])
    n = int(n_needed)
    U = np.empty((6, n))
    for k in range(3):
        got = 0
        while got < n:
            u = rng.standard_normal((2, max(2 * (n - got), 64)))
            if cut:
                u = u[:, (u[0] ** 2 + u[1] ** 2) < cut * cut]
            m = min(u.shape[1], n - got)
            U[2 * k:2 * k + 2, got:got + m] = u[:, :m]
            got += m
    bg = ref.beta0gamma0
    ex, ey = cfg.emit_nx_um * 1e-6 / bg, cfg.emit_ny_um * 1e-6 / bg
    x = math.sqrt(ex * cfg.beta_x_m) * U[0]
    xp = math.sqrt(ex / cfg.beta_x_m) * (U[1] - cfg.alpha_x * U[0])
    y = math.sqrt(ey * cfg.beta_y_m) * U[2]
    yp = math.sqrt(ey / cfg.beta_y_m) * (U[3] - cfg.alpha_y * U[2])
    delta = cfg.sigma_dp * U[5]
    zeta = cfg.sigma_z_m * U[4]
    px, py = C.momenta_from_slopes(xp, yp, delta)
    X = np.vstack([x, px, y, py, zeta, delta])
    freq = cfg.frequency_mhz or 0.0
    return SourcePool(np.ascontiguousarray(X), ref, species, float(freq), "a Gaussian")


def load_source(cfg, *, beam_config=None, n_needed: int, seed: int, constants) -> SourcePool:
    """The pool of the source settings ``cfg`` (:class:`~linac_gen.ring.
    injection.config.SourceConfig`): ``n_needed`` particles for a generated
    one, the file's for a file; ``constants`` — the ring's constants
    profile."""
    if cfg.kind == "file":
        return _file_pool(cfg, constants)
    if cfg.kind == "gaussian":
        return _gaussian_pool(cfg, n_needed, seed, constants)
    return _beam_pool(cfg, beam_config, n_needed, seed, constants)


def to_ring(pool: SourcePool, idx: np.ndarray, *, ring_ref: RingReference,
            centroid=(0.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    """Pool particles ``idx`` → ring coordinates (6, n) relative to the ring's
    reference at the foil ``ring_ref`` (a proton): an H⁻ stripped to a proton
    of the same velocity, a proton re-expressed relative to the ring's
    momentum; then the centroid (x, px, y, py) of the injected beam added."""
    X = pool.X[:, idx]
    if pool.species == "H-":
        X = C.strip_to_proton(X, hminus_ref=pool.reference, ring_ref=ring_ref)
    else:
        X = C.change_reference(X, old=pool.reference, new=ring_ref)
    X[0] += centroid[0]
    X[1] += centroid[1]
    X[2] += centroid[2]
    X[3] += centroid[3]
    return X


def mean_delta(pool: SourcePool, ring_ref: RingReference) -> float:
    """The injected beam's mean delta in the ring (after stripping)."""
    idx = np.arange(pool.n)
    return float(np.mean(to_ring(pool, idx, ring_ref=ring_ref)[5]))
