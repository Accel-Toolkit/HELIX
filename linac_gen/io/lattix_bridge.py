"""Import Bmad, SciBmad, PALS and lattix-JSON lattices through ``lattix``.

``lattix`` is the standalone accelerator-lattice translator (BSD-3); it
parses the deck into its own intermediate representation and its HELIX
adapter builds a :class:`~linac_gen.core.lattice.Lattice` from that.  Both
steps produce a fidelity report — every element that could only be
approximated or had to be dropped is listed by name — and the bridge turns
those entries into the same ``metadata["warnings"]`` list the native
importers return, so the CLI echoes them and the GUI counts them.

Locating lattix (in this order):

1. an installed ``lattix`` package (any environment where ``import lattix``
   works — lattix is not yet published on PyPI);
2. ``HELIX_LATTIX_ROOT`` — a checkout of the translator;
3. a ``Translator`` checkout next to the HELIX repository (a one-line note
   on stderr says which path was used).

Nothing here runs Bmad, Julia or any other engine: the translation is a
pure-Python parse of the lattice file.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

from linac_gen.io.formats import LATTIX_SUFFIXES  # noqa: F401  (re-exported)

class LattixUnavailable(ImportError, ValueError):
    """lattix is missing, too old for this format, or fails to import.

    Also a ValueError so the CLI's ``except (ValueError, KeyError)`` around
    ``load_input`` prints ``error: ...`` instead of a traceback."""


#: lattix format name expected for each suffix family (checked against the
#: FORMATS registry of the lattix that was found, so an older checkout gives a
#: clear message instead of lattix's own "cannot guess the lattice format")
_EXPECTED_FORMAT = {".bmad": "bmad", ".jl": "scibmad", ".scibmad": "scibmad",
                    ".pals.yaml": "pals", ".pals.yml": "pals", ".pals.json": "pals",
                    ".lattix.json": "lattix"}

_HELIX_ROOT = Path(__file__).resolve().parents[2]
#: ``Translator`` checkouts looked for near the HELIX repository: next to it,
#: and up to two directories above (a workspace that holds both projects)
_SIBLINGS = tuple(dict.fromkeys(p / "Translator" for p in _HELIX_ROOT.parents[:3]))


def _install_hint() -> str:
    looked = ", ".join(str(p) for p in _SIBLINGS)
    return ("Bmad/SciBmad/PALS import needs the optional 'lattix' package "
            "(not yet published): install lattix into this environment, or "
            "set HELIX_LATTIX_ROOT to a checkout "
            f"(also looked for a Translator checkout at: {looked})")

#: species names HELIX can track; everything else (electron, positron, ...)
#: is refused with a message that names the deck's reference
_HELIX_SPECIES = {"proton", "h-", "deuteron"}
_SPECIES_ALIASES = {"p": "proton", "hminus": "h-", "h_minus": "h-", "h^-": "h-",
                    "#1h-": "h-", "d": "deuteron"}


def _import_lattix():
    """Return the ``lattix`` package, importing it from an installed package,
    ``HELIX_LATTIX_ROOT`` or the sibling checkout."""
    try:
        import lattix                       # installed (or already on sys.path)
        return lattix
    except ImportError:
        pass
    candidates = []
    env = os.environ.get("HELIX_LATTIX_ROOT")
    if env:
        candidates.append((Path(env).expanduser(), "HELIX_LATTIX_ROOT"))
    candidates += [(p, "sibling checkout") for p in _SIBLINGS]
    for root, how in candidates:
        if (root / "lattix" / "__init__.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                import lattix
            except ImportError as exc:      # present but its own deps missing
                raise LattixUnavailable(f"{_install_hint()}\n  found {root} ({how}) but it "
                                        f"failed to import: {exc}") from exc
            if how == "sibling checkout":
                print(f"[lattice import] using lattix from {root}", file=sys.stderr)
            return lattix
    raise LattixUnavailable(_install_hint())


def _expected_format(path) -> str | None:
    name = Path(path).name.lower()
    for suf, fmt in _EXPECTED_FORMAT.items():
        if name.endswith(suf):
            return fmt
    return None


def _norm_species(name) -> str:
    key = str(name).strip().lower()
    return _SPECIES_ALIASES.get(key, key)


def _fidelity_lines(rep, stage: str) -> list[str]:
    """Every non-EXACT fidelity entry as one warning line."""
    out = []
    for e in rep.entries:
        cls = getattr(e.cls, "value", str(e.cls))
        if cls == "EXACT":
            continue
        where = f" {e.element}" if getattr(e, "element", None) else ""
        out.append(f"{stage} {cls}:{e.code}{where}: {e.message}")
    return out


def parse_with_lattix(path, *, species: str | None = None, w_kin: float | None = None,
                      frequency: float | None = None):
    """Parse *path* with lattix and convert it to a HELIX ``Lattice``.

    Parameters
    ----------
    path : str
        A ``.bmad``, ``.jl``/``.scibmad``, ``.pals.yaml``/``.pals.json`` or
        ``.lattix.json`` file.
    species : str, optional
        ``proton`` / ``h-`` / ``deuteron``: the particle every normalised
        strength (``k1``, ``Kn1``, ...) is converted with.  Defaults to the
        deck's own reference particle; a deck whose reference HELIX cannot
        track (Bmad's default is the positron) must be given one.
    w_kin : float, optional
        Kinetic energy in MeV, used only when the deck carries no reference
        energy (lattix reports ``REFERENCE_ASSUMED`` otherwise).
    frequency : float, optional
        RF frequency in MHz for the same case.

    Returns ``(lattice, metadata)`` with ``metadata`` keys ``title``,
    ``warnings`` (list of str), ``reference`` (HELIX ``ReferenceParticle``),
    ``fidelity`` (both reports as dicts) and ``format``.
    """
    lattix = _import_lattix()
    import lattix.ir.reference  # noqa: F401  (species lookup below)
    from lattix.formats.base import FORMATS, guess_format
    from lattix.formats.helix import to_helix

    fp = str(path)
    fmt = _expected_format(fp)
    if fmt is not None and fmt not in FORMATS:
        root = Path(lattix.__file__).resolve().parents[1]
        raise LattixUnavailable(
            f"the lattix at {root} (version {getattr(lattix, '__version__', '?')}) has no "
            f"{fmt!r} format, which {Path(fp).name} needs - update that checkout (or point "
            "HELIX_LATTIX_ROOT at one that has it)")
    fmt = fmt or guess_format(fp)
    reader = FORMATS[fmt].reader()
    accepts = inspect.signature(reader.read).parameters
    options = {}
    if "kinetic_energy_eV" in accepts and w_kin is not None:
        options["kinetic_energy_eV"] = float(w_kin) * 1e6
    from lattix.formats import read as lattix_read
    # First read with the deck's own particle, so the message below can say
    # what the deck's reference is; a different requested species re-reads
    # the deck with it (a Bmad p0c then gives that particle's energy).
    lat, rep = lattix_read(fp, **options)
    deck_species = _norm_species(lat.reference.species.name)
    deck_w_mev = lat.reference.kinetic_energy_eV / 1e6
    use_species = _norm_species(species) if species else deck_species
    warnings = []
    if use_species != deck_species:
        if "species" in accepts:
            lat, rep = lattix_read(fp, species=use_species, **options)
        else:
            lat.reference.species = lattix.ir.reference.species(use_species)
        warnings.append(f"{fmt} import: deck reference is {deck_species} at "
                        f"{deck_w_mev:.6g} MeV; strengths converted for {use_species} "
                        f"as requested")
    if frequency is not None and lat.reference.rf_frequency_Hz is None:
        lat.reference.rf_frequency_Hz = float(frequency) * 1e6
    if use_species not in _HELIX_SPECIES:
        raise ValueError(
            f"HELIX has no {use_species} model; this deck's reference particle is "
            f"{deck_species} at {deck_w_mev:.6g} MeV "
            f"({FORMATS[fmt].description}). Re-open it with species=proton, h- or "
            f"deuteron (CLI/assistant: parse_with_lattix(..., species=...)).")
    warnings += _fidelity_lines(rep, f"{fmt} read")
    hlat, rep2 = to_helix(lat, species=use_species)
    warnings += _fidelity_lines(rep2, "to HELIX")
    warnings += [f"lattix: {w}" for w in getattr(lat, "warnings", [])]

    from linac_gen.core.particle import DEUTERON, H_MINUS, PROTON
    from linac_gen.core.reference import ReferenceParticle
    particle = {"proton": PROTON, "h-": H_MINUS, "deuteron": DEUTERON}[use_species]
    w_mev = lat.reference.kinetic_energy_eV / 1e6
    # No RF in the deck: the same benign placeholder the MAD-X importer uses
    # (madx_parser.py:771) — a ReferenceParticle needs a finite wavelength.
    f_mhz = (lat.reference.rf_frequency_Hz or 352.21e6) / 1e6
    ref = ReferenceParticle(particle, w_mev, f_mhz)
    warnings.append(
        f"{fmt} import: strengths converted with {particle.name} at W = {w_mev:.4f} MeV "
        f"— set the Beam tab to match before running.")
    meta = {
        "title": lat.name,
        "warnings": warnings,
        "reference": ref,
        "fidelity": {"read": rep.model_dump(mode="json"),
                     "to_helix": rep2.model_dump(mode="json")},
        "format": fmt,
    }
    return hlat, meta


__all__ = ["LATTIX_SUFFIXES", "parse_with_lattix"]
