"""TraceWin project options file (``.ini``) reader.

TraceWin keeps every project-level setting — the two input beams, the
particle table, run and space-charge options, window state — in
``<project>.ini``, a binary dump of its options structure that begins
with the ASCII magic ``TraceWin_options_file``.  The format is not
documented anywhere (the manual describes ``.dst`` but not ``.ini``),
so the layout below was reverse-engineered by value-matching 54 project
files from two TraceWin generations (Fermilab PIP-II projects 2017–2026,
the LightWin/ADS sample) against their known run inputs.  Every field in
:data:`FIELDS` carries a status:

``verified``
    the decoded value reproduced the project's known input in at least
    two independent projects (and, for the longitudinal block, the
    LightWin ``ads.ini`` ↔ ``lightwin.toml`` σ-matrix identity and a
    TraceWin ``tracewin.out`` σ_φ);
``probable``
    the slot sits where the C struct implies (beam-2 twins of a verified
    beam-1 slot, or a value that is constant and plausible across all
    files) but no project varied it, so it is reported and recorded, never
    applied to a beam.

Binary layout (little-endian; offsets in bytes)
-----------------------------------------------

    0x0000  char[]   magic ``TraceWin_options_file``
    0x0068  int32    layout tag: the options-struct size of the TraceWin
                     that last touched the file (31624 / 44824).  Informative,
                     NOT authoritative — a project upgraded by a newer TraceWin
                     keeps its ``<project>.old.ini`` at the OLD size with the
                     NEW tag (7 such files on this machine).  The layout is
                     therefore keyed on the file size alone (31624: 2017
                     layout, 44824: 2019+ layout; prefix-identical, the newer
                     one appends 13,200 bytes) and a differing tag is a
                     warning.
    0x0859  char[]   last project directory (NUL-terminated)
    0x0c41  char[]   last project directory (second slot)
    0x17f9  char[]   project name (the ``.dat``/``.ini`` stem)
    0x2ed0  int32    nbr_part1      macro-particles, beam 1
    0x2ed4  int32    nbr_part2      (beam 2)
    0x2ed8  int32    particle_index1  row of the particle table (3 = H-)
    0x2edc  int32    particle_index2
    0x2ee8  int32    nbr_thread
    0x2efc  int32    picnic_r_mesh
    0x2f00  int32    picnic_z_mesh
    0x2f04  int32    picnic_xy_mesh (x)
    0x2f08  int32    picnic_xy_mesh (y)
    0x2f24  f64      freq1     MHz            (+8: freq2)
    0x2f34  f64      current1  A              (+8: current2)
    0x2f44  f64      energy1   eV             (+8: energy2)
    0x2f54  f64      etnx1     π m rad, rms normalised   (+8: etnx2)
    0x2f64  f64      etny1     π m rad, rms normalised   (+8: etny2)
    0x2f74  f64      eps_z1    π m rad, rms normalised longitudinal
                     (= βγ · ε_zδ, the (z, δ=Δp/p) emittance)   (+8: eps_z2)
    0x30dc  f64      alpx1                    (+16: alpx2)
    0x30e4  f64      betx1     m              (+16: betx2)
    0x30fc  f64      alpy1                    (+16: alpy2)
    0x3104  f64      bety1     m              (+16: bety2)
    0x311c  f64      alpz1     (z, z′=dz/ds) plane   (+16: alpz2)
    0x3124  f64      betz1     m/rad = γ² · β_zδ     (+16: betz2)
    0x31fc  12 × 44-byte particle records: f64 mass (eV), int32 charge,
                     char[32] name — 0 Positron, 1 Electron, 2 Proton,
                     3 H-, 4 Deuton, 5 H2+, 6 H3+, 7–11 user rows

Slots that exist as TraceWin options but whose offsets are NOT located
(they are named in the report as "not decoded" and never guessed):
``input_dist_type``, ``duty1``, ``spreadw1``/``dw1``, the centroid
offsets ``x1 … zp1`` (a single 2017 file shows five small doubles at
0x3024–0x3044 that may be them), ``dst_file1``/``use_dst_file``,
``part_step``, ``random_seed``, ``vfac``, the loss/emittance limits.
``mass1``/``charge1`` are taken from the particle table row instead.

Conversion to HELIX units
-------------------------

HELIX's :class:`~linac_gen.core.config.BeamConfig` uses normalised
transverse emittances in π mm mrad, β_x/β_y in m, and a longitudinal
(Δφ [deg], ΔW [MeV]) plane with ``emit_z`` in π deg MeV and ``beta_z``
in deg/MeV.  With ``f`` the beam frequency in Hz, ``mc²`` the rest
energy in MeV, ``c`` in m/s and βγ at the input energy:

    emit_nx [π mm mrad] = etnx1 × 1e6
    emit_z  [π deg MeV] = eps_z1 · 360 · f · mc² / c            (energy-independent)
    beta_z  [deg/MeV]   = betz1 · 360 · f / ((βγ)³ · c · mc²)
    alpha_z             = − alpz1

The α_z negation is HELIX's documented TraceWin convention (a late
particle has Δφ > 0 but z < 0); it enters in exactly one line here and
must not be re-applied by a caller.  Derivation: Δφ = −360 f z/(βc) and
ΔW = mc² β² γ δ, so ε_ΔφΔW = ε_zδ · 360 f mc² βγ / c = eps_z1 · 360 f
mc²/c and β_Δφ = σ_φ²/ε_ΔφΔW = β_zδ · 360 f/(β³ γ c mc²) with
β_zδ = betz1/γ².  Checked against the LightWin ``ads.ini`` σ-matrix
(σ_φ and σ_W agree to < 1e-3) and a PIP-II TraceWin run (``betz1`` = 8
m/rad at 116.1 MeV, 804.96 MHz converts to 61.219 deg/MeV, the value that
reproduces the run's σ_φ = 7.455° in ``tracewin.out`` to 1e-4).

A DC beam (``eps_z1 == 0`` — LEBT projects) maps to ``continuous=True``;
its energy spread is not decoded and stays at the caller's value.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional

from linac_gen.core.config import BeamConfig
from linac_gen.core.constants import C_LIGHT
from linac_gen.core.particle import DEUTERON, H_MINUS, PROTON, Particle
from linac_gen.core.reference import ReferenceParticle

__all__ = [
    "MAGIC", "TAG_OFFSET", "KNOWN_SIZES", "PARTICLE_TABLE_OFFSET",
    "FieldSpec", "FIELDS", "FIELD_BY_NAME", "ParticleRecord", "TraceWinIni",
    "load_tracewin_ini", "sibling_ini", "resolve_tracewin_ini",
    "species_for", "to_beam_config", "beam_header", "report",
    "to_project_extras", "lattice_first_frequency", "NOT_DECODED",
]

MAGIC = b"TraceWin_options_file"
TAG_OFFSET = 0x68
KNOWN_SIZES = {31624: "TraceWin 2017 layout", 44824: "TraceWin 2019+ layout"}
PARTICLE_TABLE_OFFSET = 0x31FC
PARTICLE_RECORD_SIZE = 44
N_PARTICLE_ROWS = 12
# Windows scanned for the raw/unknown-slot report: the int32 block that
# precedes the beam doubles and the double block up to the last Twiss slot.
RAW_INT_WINDOW = (0x2ED0, 0x2F24)
RAW_F64_WINDOW = (0x2F24, 0x313C)
# Doubles smaller than this are byte junk (denormals from non-double
# slots), not settings; they are dropped from the raw scan.
_F64_NOISE = 1e-200
_STRING_MAX = 1000

# TraceWin option names (LightWin's specs.py list) that exist in the file
# but whose offsets are not located.  Reported, never guessed.
NOT_DECODED = (
    "input_dist_type", "duty1/duty2", "spreadw1/dw1", "x1 xp1 y1 yp1 z1 zp1",
    "dst_file1/use_dst_file", "part_step", "random_seed", "vfac",
    "lost_p_limit/lost_e_limit/emit_p_limit/emit_e_limit",
    "mass1/charge1 (read from the particle table row instead)",
)


@dataclass(frozen=True)
class FieldSpec:
    """One decoded slot of the options file."""
    name: str
    offset: int
    fmt: str            # "<i" int32, "<d" float64, "s" NUL-terminated string
    unit: str
    beam: int           # 1 / 2 for per-beam slots, 0 otherwise
    status: str         # "verified" | "probable"
    note: str = ""

    @property
    def size(self) -> int:
        return _STRING_MAX if self.fmt == "s" else struct.calcsize(self.fmt)


def _beam_pair(name, off1, fmt, unit, note1, stride):
    return (
        FieldSpec(f"{name}1", off1, fmt, unit, 1, "verified", note1),
        FieldSpec(f"{name}2", off1 + stride, fmt, unit, 2, "probable",
                  "beam-2 twin of the verified beam-1 slot; no sample "
                  "project populates beam 2"),
    )


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("last_dir_1", 0x0859, "s", "", 0, "probable",
              "last project directory"),
    FieldSpec("last_dir_2", 0x0C41, "s", "", 0, "probable",
              "last project directory (second slot)"),
    FieldSpec("project_name", 0x17F9, "s", "", 0, "verified",
              "equals the .dat/.ini stem in every sample"),
    *_beam_pair("nbr_part", 0x2ED0, "<i", "", "macro-particle count", 4),
    *_beam_pair("particle_index", 0x2ED8, "<i", "",
                "row of the particle table (2 = Proton, 3 = H-)", 4),
    FieldSpec("nbr_thread", 0x2EE8, "<i", "", 0, "probable",
              "8 in every sample; never varied"),
    FieldSpec("picnic_r_mesh", 0x2EFC, "<i", "", 0, "probable",
              "PICNIC 2-D radial mesh (20 in every sample)"),
    FieldSpec("picnic_z_mesh", 0x2F00, "<i", "", 0, "probable",
              "PICNIC 2-D longitudinal mesh (40 in every sample)"),
    FieldSpec("picnic_xy_mesh_x", 0x2F04, "<i", "", 0, "probable",
              "PICNIC 3-D transverse mesh (9 or 20 in the samples)"),
    FieldSpec("picnic_xy_mesh_y", 0x2F08, "<i", "", 0, "probable",
              "PICNIC 3-D transverse mesh (9 or 20 in the samples)"),
    *_beam_pair("freq", 0x2F24, "<d", "MHz", "beam frequency", 8),
    *_beam_pair("current", 0x2F34, "<d", "A", "beam current", 8),
    *_beam_pair("energy", 0x2F44, "<d", "eV", "input kinetic energy", 8),
    *_beam_pair("etnx", 0x2F54, "<d", "pi m rad",
                "rms normalised horizontal emittance", 8),
    *_beam_pair("etny", 0x2F64, "<d", "pi m rad",
                "rms normalised vertical emittance", 8),
    *_beam_pair("eps_z", 0x2F74, "<d", "pi m rad",
                "rms normalised longitudinal emittance (= beta*gamma * eps_z_delta)", 8),
    *_beam_pair("alpx", 0x30DC, "<d", "", "horizontal Twiss alpha", 16),
    *_beam_pair("betx", 0x30E4, "<d", "m", "horizontal Twiss beta", 16),
    *_beam_pair("alpy", 0x30FC, "<d", "", "vertical Twiss alpha", 16),
    *_beam_pair("bety", 0x3104, "<d", "m", "vertical Twiss beta", 16),
    *_beam_pair("alpz", 0x311C, "<d", "",
                "longitudinal Twiss alpha in the (z, z') plane", 16),
    *_beam_pair("betz", 0x3124, "<d", "m/rad",
                "longitudinal Twiss beta in the (z, z') plane (= gamma^2 * beta_z_delta)", 16),
)
FIELD_BY_NAME = {f.name: f for f in FIELDS}
_FIELD_BY_SLOT = {(f.fmt, f.offset): f for f in FIELDS if f.fmt != "s"}

# The beam-1 slots that are applied to a BeamConfig (beam-2 uses the twins).
_BEAM_SLOTS = ("nbr_part", "particle_index", "freq", "current", "energy",
               "etnx", "etny", "eps_z", "alpx", "betx", "alpy", "bety",
               "alpz", "betz")
_HELIX_SPECIES: tuple[tuple[str, Particle], ...] = (
    ("proton", PROTON), ("deuteron", DEUTERON), ("H-", H_MINUS),
)
_SPECIES_MASS_TOL_MEV = 0.5


@dataclass(frozen=True)
class ParticleRecord:
    """One row of the 12-row particle table."""
    index: int
    name: str
    mass_eV: float
    charge: int

    @property
    def mass_MeV(self) -> float:
        return self.mass_eV * 1e-6


@dataclass
class TraceWinIni:
    """Decoded contents of a ``.ini`` file (see the module docstring)."""
    path: str
    size: int
    format_tag: int
    version: str
    fields: dict = field(default_factory=dict)
    particles: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    unknown: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def get(self, name: str, default=None):
        return self.fields.get(name, default)

    def beam_fields(self, beam: int = 1) -> dict:
        """The per-beam slots of ``beam`` keyed by their generic name
        (``freq``, ``energy``, … without the trailing 1/2)."""
        if beam not in (1, 2):
            raise ValueError(f"beam must be 1 or 2, got {beam!r}")
        return {k: self.fields[f"{k}{beam}"] for k in _BEAM_SLOTS}

    def particle_for_beam(self, beam: int = 1) -> Optional[ParticleRecord]:
        idx = int(self.fields.get(f"particle_index{beam}", -1))
        if 0 <= idx < len(self.particles):
            return self.particles[idx]
        return None

    def identified_not_applied(self) -> dict:
        """Run settings that are decoded but never written to a beam."""
        return {k: self.fields.get(k) for k in
                ("nbr_thread", "picnic_r_mesh", "picnic_z_mesh",
                 "picnic_xy_mesh_x", "picnic_xy_mesh_y")}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _read_field(buf: bytes, spec: FieldSpec):
    if spec.fmt == "s":
        chunk = buf[spec.offset:spec.offset + _STRING_MAX]
        return chunk.split(b"\0", 1)[0].decode("latin-1", errors="replace")
    return struct.unpack_from(spec.fmt, buf, spec.offset)[0]


def _read_particles(buf: bytes) -> list[ParticleRecord]:
    rows = []
    for r in range(N_PARTICLE_ROWS):
        o = PARTICLE_TABLE_OFFSET + PARTICLE_RECORD_SIZE * r
        mass, charge = struct.unpack_from("<di", buf, o)
        name = buf[o + 12:o + 44].split(b"\0", 1)[0].decode(
            "latin-1", errors="replace")
        rows.append(ParticleRecord(r, name, float(mass), int(charge)))
    return rows


def _scan_raw(buf: bytes) -> dict:
    raw = {}
    lo, hi = RAW_INT_WINDOW
    for o in range(lo, hi, 4):
        v = struct.unpack_from("<i", buf, o)[0]
        if v:
            raw[f"i32@0x{o:04x}"] = v
    lo, hi = RAW_F64_WINDOW
    for o in range(lo, hi, 8):
        v = struct.unpack_from("<d", buf, o)[0]
        if v and (math.isnan(v) or abs(v) >= _F64_NOISE):
            raw[f"f64@0x{o:04x}"] = v
    return raw


def load_tracewin_ini(path) -> TraceWinIni:
    """Read and decode a TraceWin ``.ini`` options file.

    Raises
    ------
    FileNotFoundError
        ``path`` does not exist.
    ValueError
        The file is not a TraceWin options file (magic) or has a size
        other than the two known layouts (a layout this reader has not
        seen).  A layout tag that differs from the size is a warning.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(path))
    buf = p.read_bytes()
    if not buf.startswith(MAGIC):
        raise ValueError(
            f"{p.name} is not a TraceWin options file (expected the magic "
            f"{MAGIC.decode()!r} at offset 0)")
    if len(buf) not in KNOWN_SIZES:
        known = ", ".join(f"{s} bytes ({v})" for s, v in KNOWN_SIZES.items())
        raise ValueError(
            f"{p.name} is {len(buf)} bytes; this reader knows the TraceWin "
            f".ini layouts of {known}.  Refusing to decode fixed offsets "
            f"from an unknown layout — map it first with "
            f"scripts/tracewin_ini_probe.py")
    tag = struct.unpack_from("<i", buf, TAG_OFFSET)[0]
    ini = TraceWinIni(path=str(p), size=len(buf), format_tag=int(tag),
                      version=KNOWN_SIZES[len(buf)])
    if tag != len(buf):
        ini.warnings.append(
            f"layout tag at 0x{TAG_OFFSET:x} is {tag} while the file is "
            f"{len(buf)} bytes; the {KNOWN_SIZES[len(buf)]} offsets are used "
            f"(a project re-saved by a newer TraceWin keeps its *.old.ini at "
            f"the old size with the new tag)")
    for spec in FIELDS:
        v = _read_field(buf, spec)
        ini.fields[spec.name] = int(v) if spec.fmt == "<i" else v
    ini.particles = _read_particles(buf)
    ini.raw = _scan_raw(buf)
    ini.unknown = {k: v for k, v in ini.raw.items()
                   if (("<i" if k.startswith("i32") else "<d"),
                       int(k.split("@", 1)[1], 16)) not in _FIELD_BY_SLOT}
    stem = p.stem
    pn = ini.fields.get("project_name", "")
    if pn and pn != stem:
        ini.warnings.append(
            f"project name stored in the file is {pn!r}, file stem is "
            f"{stem!r} (the file was renamed after TraceWin saved it)")
    return ini


def sibling_ini(lattice_path) -> Optional[Path]:
    """``<stem>.ini`` (or ``.INI``) next to a lattice file, if it exists."""
    lp = Path(lattice_path)
    for suffix in (".ini", ".INI"):
        cand = lp.with_suffix(suffix)
        if cand.is_file():
            return cand
    return None


def resolve_tracewin_ini(lattice_path, spec) -> Optional[Path]:
    """Turn a ``--tracewin-ini`` style request into a path.

    ``spec`` is ``None``/``False`` (no import → ``None``), ``True`` or
    ``"auto"`` (the sibling ``.ini`` of ``lattice_path``; ``ValueError``
    when there is none) or an explicit path (``FileNotFoundError`` when
    missing).
    """
    if spec is None or spec is False:
        return None
    if spec is True or (isinstance(spec, str) and spec.strip().lower() == "auto"):
        sib = sibling_ini(lattice_path)
        if sib is None:
            lp = Path(lattice_path)
            raise ValueError(
                f"no {lp.stem}.ini next to {lp.name} — pass the options "
                f"file path explicitly")
        return sib
    p = Path(spec)
    if not p.is_file():
        raise ValueError(f"TraceWin options file not found: {spec}")
    return p


# ---------------------------------------------------------------------------
# Species and conversion
# ---------------------------------------------------------------------------
def species_for(record: Optional[ParticleRecord]) -> tuple[Optional[str], Optional[str]]:
    """Map a particle-table row to a HELIX species name.

    Returns ``(species, warning)``: ``species`` is ``"proton"``,
    ``"deuteron"`` or ``"H-"`` when the row's charge and rest mass match
    one of them (mass within 0.5 MeV), else ``None`` with a warning that
    names the row.
    """
    if record is None:
        return None, "the beam's particle index points outside the particle table"
    for name, p in _HELIX_SPECIES:
        if record.charge == p.charge and abs(record.mass_MeV - p.mass) < _SPECIES_MASS_TOL_MEV:
            return name, None
    return None, (
        f"particle row {record.index} {record.name!r} ({record.mass_MeV:.3f} "
        f"MeV, q = {record.charge:+d}) has no HELIX species (proton, "
        f"deuteron, H-)")


def _resolve_species(ini: TraceWinIni, beam: int, species: Optional[str],
                     base: BeamConfig, warnings: list) -> str:
    rec = ini.particle_for_beam(beam)
    found, warn = species_for(rec)
    if species is not None:
        if species not in dict(_HELIX_SPECIES):
            raise ValueError(
                f"species must be one of proton, deuteron, H- (got {species!r})")
        if found is not None and found != species:
            warnings.append(
                f"species {species!r} requested; the .ini particle row is "
                f"{rec.name!r} which maps to {found!r}")
        return species
    if found is not None:
        return found
    fallback = base.species if base.species in dict(_HELIX_SPECIES) else "proton"
    warnings.append(
        f"{warn}; using {fallback!r} instead — the .ini rest mass "
        f"was NOT used, choose the species explicitly if this is wrong")
    return fallback


def _species_particle(species: str) -> Particle:
    return dict(_HELIX_SPECIES).get(species, PROTON)


def lattice_first_frequency(lattice) -> Optional[float]:
    """The first ``FREQ`` card of a parsed lattice (MHz), or ``None``."""
    from linac_gen.elements.lattice_commands import Freq
    for el in getattr(lattice, "elements", []) or []:
        if isinstance(el, Freq) and el.frequency_mhz > 0:
            return float(el.frequency_mhz)
    return None


def to_beam_config(ini: TraceWinIni, *, beam: int = 1,
                   species: Optional[str] = None,
                   base: Optional[BeamConfig] = None,
                   lattice=None) -> tuple[BeamConfig, list[str]]:
    """Convert beam ``beam`` (1 or 2) of the options file to a HELIX beam.

    Fields the file does not describe (distribution type, cut-off, DC
    energy spread, centroid offsets, …) are taken from ``base`` (default:
    a stock :class:`BeamConfig`).  ``species`` overrides the particle-table
    match (a user-defined ion has no HELIX species).  ``lattice`` is an
    optional parsed lattice whose first ``FREQ`` card is compared with the
    beam frequency.

    Returns ``(config, warnings)``; ``warnings`` lists every explicit
    degradation (species fallback, DC regime, frequency mismatch, …).

    Raises
    ------
    ValueError
        The beam is not populated (zero energy or frequency), has a zero
        transverse emittance or beta, or is bunched with ``betz == 0``.
    """
    base = base if base is not None else BeamConfig()
    warnings: list[str] = []
    bf = ini.beam_fields(beam)
    name = ini.name
    for k in ("freq", "current", "energy", "etnx", "etny", "eps_z",
              "alpx", "betx", "alpy", "bety", "alpz", "betz"):
        if not math.isfinite(float(bf[k])):
            raise ValueError(
                f"beam {beam} of {name}: slot {k}{beam} is {bf[k]} (not finite)")
    energy_MeV = float(bf["energy"]) * 1e-6
    freq_MHz = float(bf["freq"])
    if energy_MeV <= 0.0:
        raise ValueError(f"beam {beam} of {name} is not populated (energy = 0)")
    if freq_MHz <= 0.0:
        raise ValueError(f"beam {beam} of {name} has no frequency (freq = 0)")
    etnx, etny = float(bf["etnx"]), float(bf["etny"])
    if etnx <= 0.0 or etny <= 0.0:
        raise ValueError(
            f"beam {beam} of {name} has a zero transverse emittance "
            f"(etnx = {etnx:g}, etny = {etny:g})")
    betx, bety = float(bf["betx"]), float(bf["bety"])
    if betx <= 0.0 or bety <= 0.0:
        raise ValueError(
            f"beam {beam} of {name} has a non-positive transverse beta "
            f"(betx = {betx:g}, bety = {bety:g})")

    sp = _resolve_species(ini, beam, species, base, warnings)
    particle = _species_particle(sp)
    ref = ReferenceParticle(species=particle, w_kin=energy_MeV, frequency=freq_MHz)

    updates = dict(
        species=sp,
        energy=energy_MeV,
        frequency=freq_MHz,
        current=float(bf["current"]) * 1e3,
        emit_nx=etnx * 1e6,
        alpha_x=float(bf["alpx"]),
        beta_x=betx,
        emit_ny=etny * 1e6,
        alpha_y=float(bf["alpy"]),
        beta_y=bety,
        source="generate",
        distribution_file=None,
    )
    npart = int(bf["nbr_part"])
    if npart > 0:
        updates["n_particles"] = npart
    else:
        warnings.append(
            f"nbr_part{beam} is {npart}; keeping n_particles = {base.n_particles}")

    eps_z = float(bf["eps_z"])
    betz = float(bf["betz"])
    f_hz = freq_MHz * 1e6
    if eps_z < 0.0:
        raise ValueError(
            f"beam {beam} of {name} has a negative longitudinal emittance "
            f"(eps_z{beam} = {eps_z:g})")
    if eps_z == 0.0:
        updates.update(continuous=True, alpha_z=0.0)
        warnings.append(
            f"eps_z{beam} = 0: DC (unbunched) beam -> continuous = True; the "
            f"energy spread is not decoded from the .ini (dc_energy_spread_keV "
            f"stays {base.dc_energy_spread_keV:g}); emit_z/beta_z are ignored "
            f"in this mode")
    else:
        if betz <= 0.0:
            raise ValueError(
                f"beam {beam} of {name} is bunched (eps_z = {eps_z:g}) but "
                f"betz = {betz:g}")
        updates.update(
            continuous=False,
            emit_z=eps_z * 360.0 * f_hz * particle.mass / C_LIGHT,
            beta_z=betz * 360.0 * f_hz / (ref.bg ** 3 * C_LIGHT * particle.mass),
            alpha_z=-float(bf["alpz"]),
        )
    if base.source == "file":
        warnings.append(
            "the previous beam came from a distribution file; the imported "
            "Twiss parameters describe a generated beam (source = generate)")
    if lattice is not None:
        f_lat = lattice_first_frequency(lattice)
        if f_lat is not None and abs(f_lat - freq_MHz) > 1e-6 * max(1.0, f_lat):
            warnings.append(
                f"beam frequency {freq_MHz:g} MHz differs from the lattice's "
                f"first FREQ card ({f_lat:g} MHz); HELIX rescales the "
                f"longitudinal plane at the card")
    cfg = replace(base, **updates)
    return cfg, warnings


def beam_header(ini: TraceWinIni, *, beam: int = 1,
                species: Optional[str] = None,
                base: Optional[BeamConfig] = None) -> dict:
    """A ``load_dst``-style header dict for GUI reuse: the ``.dst`` keys
    (``n_particles``, ``current_mA``, ``frequency_MHz``, ``mass_MeV``,
    ``w_kin_ref``) plus every converted beam scalar and ``warnings``."""
    cfg, warns = to_beam_config(ini, beam=beam, species=species, base=base)
    rec = ini.particle_for_beam(beam)
    particle = _species_particle(cfg.species)
    return dict(
        n_particles=cfg.n_particles,
        current_mA=cfg.current,
        frequency_MHz=cfg.frequency,
        mass_MeV=particle.mass,
        w_kin_ref=cfg.energy,
        species=cfg.species,
        particle_name=rec.name if rec is not None else "",
        particle_index=rec.index if rec is not None else -1,
        emit_nx=cfg.emit_nx, alpha_x=cfg.alpha_x, beta_x=cfg.beta_x,
        emit_ny=cfg.emit_ny, alpha_y=cfg.alpha_y, beta_y=cfg.beta_y,
        emit_z=cfg.emit_z, alpha_z=cfg.alpha_z, beta_z=cfg.beta_z,
        continuous=cfg.continuous,
        source_file=ini.path,
        version=ini.version,
        warnings=list(warns),
    )


def to_project_extras(ini: TraceWinIni) -> dict:
    """The block recorded under ``convergence["tracewin_ini"]`` of a
    written ``.lgproj``: identified run settings, recorded not applied."""
    d = ini.identified_not_applied()
    return {"tracewin_ini": {
        "file": ini.name,
        "layout": ini.version,
        "nbr_thread": d["nbr_thread"],
        "picnic_r_mesh": d["picnic_r_mesh"],
        "picnic_z_mesh": d["picnic_z_mesh"],
        "picnic_xy_mesh": [d["picnic_xy_mesh_x"], d["picnic_xy_mesh_y"]],
        "status": "recorded, not applied (PICNIC meshes are not the HELIX "
                  "space-charge grid)",
    }}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.7g}"
    return str(v)


def report(ini: TraceWinIni, *, beam: int = 1, species: Optional[str] = None,
           base: Optional[BeamConfig] = None, lattice=None) -> str:
    """Human-readable summary: applied fields (TraceWin and HELIX values
    side by side), identified-not-applied settings, unidentified non-zero
    slots, options not decoded, and the conversion warnings."""
    lines = [
        f"TraceWin options file: {ini.name}  ({ini.size:,} bytes, {ini.version})",
        f"project name: {ini.fields.get('project_name') or '(empty)'}",
    ]
    rec = ini.particle_for_beam(beam)
    if rec is not None:
        lines.append(
            f"beam {beam}: particle row {rec.index} {rec.name!r} "
            f"({rec.mass_MeV:.6f} MeV, q = {rec.charge:+d})")
    else:
        lines.append(f"beam {beam}: particle index "
                     f"{ini.fields.get(f'particle_index{beam}')} (no table row)")
    err = None
    try:
        cfg, warns = to_beam_config(ini, beam=beam, species=species,
                                    base=base, lattice=lattice)
    except ValueError as exc:
        cfg, warns, err = None, [], str(exc)
    bf = ini.beam_fields(beam)
    lines.append("")
    lines.append("Applied to the HELIX beam" if cfg is not None
                 else f"NOT convertible: {err}")
    if cfg is not None:
        rows = [
            ("species", f"row {bf['particle_index']}", cfg.species),
            ("energy", f"{_fmt(bf['energy'])} eV", f"{cfg.energy:.6g} MeV"),
            ("frequency", f"{_fmt(bf['freq'])} MHz", f"{cfg.frequency:.6g} MHz"),
            ("current", f"{_fmt(bf['current'])} A", f"{cfg.current:.6g} mA"),
            ("n_particles", _fmt(bf['nbr_part']), str(cfg.n_particles)),
            ("emit_nx", f"{_fmt(bf['etnx'])} pi m rad", f"{cfg.emit_nx:.6g} pi mm mrad"),
            ("alpha_x", _fmt(bf['alpx']), f"{cfg.alpha_x:.6g}"),
            ("beta_x", f"{_fmt(bf['betx'])} m", f"{cfg.beta_x:.6g} m"),
            ("emit_ny", f"{_fmt(bf['etny'])} pi m rad", f"{cfg.emit_ny:.6g} pi mm mrad"),
            ("alpha_y", _fmt(bf['alpy']), f"{cfg.alpha_y:.6g}"),
            ("beta_y", f"{_fmt(bf['bety'])} m", f"{cfg.beta_y:.6g} m"),
        ]
        if cfg.continuous:
            rows.append(("longitudinal", f"eps_z = {_fmt(bf['eps_z'])}",
                         "DC beam (continuous = True)"))
        else:
            rows += [
                ("emit_z", f"{_fmt(bf['eps_z'])} pi m rad (norm.)",
                 f"{cfg.emit_z:.6g} pi deg MeV"),
                ("alpha_z", _fmt(bf['alpz']),
                 f"{cfg.alpha_z:.6g}  (sign flipped: HELIX (dphi, dW) convention)"),
                ("beta_z", f"{_fmt(bf['betz'])} m/rad", f"{cfg.beta_z:.6g} deg/MeV"),
            ]
        w = max(len(r[0]) for r in rows)
        w1 = max(len(r[1]) for r in rows)
        lines.append(f"  {'field':<{w}}  {'TraceWin (.ini)':<{w1}}  HELIX")
        for r in rows:
            lines.append(f"  {r[0]:<{w}}  {r[1]:<{w1}}  {r[2]}")
    lines.append("")
    lines.append("Identified, not applied")
    for k, v in ini.identified_not_applied().items():
        lines.append(f"  {k} = {v}  ({FIELD_BY_NAME[k].status})")
    lines.append("")
    lines.append(f"Unidentified non-zero slots (0x{RAW_INT_WINDOW[0]:04x}-"
                 f"0x{RAW_F64_WINDOW[1]:04x})")
    if ini.unknown:
        for k, v in ini.unknown.items():
            lines.append(f"  {k} = {_fmt(v)}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Not decoded (TraceWin options with no located offset)")
    for n in NOT_DECODED:
        lines.append(f"  {n}")
    all_warns = list(ini.warnings) + list(warns)
    lines.append("")
    lines.append("Warnings" if all_warns else "Warnings: none")
    for wmsg in all_warns:
        lines.append(f"  - {wmsg}")
    return "\n".join(lines)
