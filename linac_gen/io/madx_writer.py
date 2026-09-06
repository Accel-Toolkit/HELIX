"""MAD-X exporter — the exact inverse of :mod:`linac_gen.io.madx_parser`.

``write_madx(lattice, filepath, ref)`` serialises a HELIX lattice as a
MAD-X input file: a ``BEAM`` line built from the reference particle,
one element definition per HELIX element, and a single
``SEQUENCE … ENDSEQUENCE`` with ``refer=entry`` positions taken from the
lattice's cumulative *s* (no centre rounding), followed by ``USE``.

Conversions (MAD-X units are metres, radians, MV, MHz; HELIX uses mm,
degrees, MV, MHz; Bρ is the charge-*signed* rigidity so that MAD-X
strengths — which are normalised to the reference charge — come out
right for H⁻ as well as protons):

    Drift                 -> DRIFT       l
    Quadrupole            -> QUADRUPOLE  l, k1 = G/Bρ, tilt = skew [rad]
    Edge+Dipole+Edge      -> SBEND       l (arc), angle, e1, e2,
                                         k1 = -n/ρ², tilt=π/2 if vertical,
                                         hgap = gap/2, fint = K1
    Dipole (no edges)     -> SBEND       e1 = e2 = 0
    Solenoid              -> SOLENOID    l, ks = B/Bρ
    RFGap                 -> RFCAVITY    volt = V0·T [MV], freq,
                                         lag = (φs + 90°)/360
                                         (MAD-X gain q·V·sin 2π·lag; HELIX
                                         q·V·T·cos φs)
    Steerer (magnetic)    -> KICKER      hkick = ∫By dl / Bρ, vkick = ∫Bx dl / Bρ
    Multipole             -> MULTIPOLE   knl, ksl (MAD-X convention already)
    Marker / BPM marker   -> MARKER / MONITOR
    Aperture              -> RCOLLIMATOR / ECOLLIMATOR (half-apertures)
    FreqCommand           -> comment (RFCAVITY carries its own freq)
    SET_* / ADJUST_* / ERROR_* / DIAG_* / LATTICE / SPACE_CHARGE_COMP
                          -> ``! HELIX: <card>`` comment lines (nothing is
                             dropped silently)

Elements MAD-X cannot represent — field maps, ``NCELLS`` cavities, RFQ
cells, foils, explicit matrices, thin lenses, electric steerers — are
handled by ``on_unsupported``: ``"marker"`` (default) keeps the geometry
with a ``MARKER`` plus a body ``DRIFT`` of the element's length and
records a warning naming the element; ``"error"`` raises ``ValueError``.
Quadrupole multipole content (``g3..g6``, ``gfr``), edge ``K2`` and
misalignments have no counterpart on the MAD-X element and are reported
as warnings.

Every warning is returned as a list (the same channel the importer uses
in ``metadata["warnings"]``).
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path

import numpy as np

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.base import FieldMapElement, ThinKickElement
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.lattice_commands import Freq as FreqCommand
from linac_gen.elements.lattice_commands import LatticeCommand
from linac_gen.elements.marker import Marker
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.sc_grid import ScGridDirective
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.thin_lens import ThinLens
from linac_gen.io.madx_parser import _signed_brho

_MM_TO_M = 1.0e-3
_DEG_TO_RAD = math.pi / 180.0

# MAD-X reserved words that must not be used as element names.
_RESERVED = {
    "drift", "quadrupole", "sbend", "rbend", "solenoid", "multipole",
    "sextupole", "octupole", "rfcavity", "kicker", "hkicker", "vkicker",
    "tkicker", "marker", "monitor", "hmonitor", "vmonitor", "rcollimator",
    "ecollimator", "collimator", "instrument", "placeholder", "sequence",
    "endsequence", "beam", "use", "title", "line", "matrix", "dipedge",
    "twiss", "track", "match", "call", "option", "select", "value",
    "const", "real", "int", "if", "while", "macro", "at", "from",
    "refer", "l", "k1", "k2", "angle", "e1", "e2", "tilt", "ks", "volt",
    "lag", "freq", "hkick", "vkick", "kick", "knl", "ksl", "xsize",
    "ysize", "aperture", "true", "false", "pi", "twopi", "e", "clight",
    # MAD-X commands — an element with one of these names is fatal
    "start", "stop", "exit", "quit", "run", "save", "plot", "set", "print",
    "write", "create", "delete", "remove", "cycle", "flatten", "emit",
    "survey", "eoption", "ealign", "efcomp", "error", "seqedit", "endedit",
    "install", "move", "makethin", "sodd", "ibs", "aperture", "resplot",
    "help", "show", "system", "return", "exec", "readtable", "readmytable",
    "fill", "shrink", "setvars", "copyfile", "assign", "endmatch", "vary",
    "constraint", "lmdif", "migrad", "simplex", "jacobian", "global",
    "weight", "gweight", "rmatrix", "ptc_create_universe", "ptc_twiss",
    "ptc_track", "ptc_end", "dumpsequ", "extract", "seqedit", "reflect",
    "beta0", "savebeta", "sxfread", "sxfwrite", "touschek", "sectormap",
    "endtrack", "observe", "dynap", "coguess", "correct", "usekick",
    "usemonitor", "cmatrix", "setcorr", "getcorr", "esave", "select_ptc",
    "ptc_align", "ptc_setswitch", "ptc_normal", "ptc_moments", "printf",
    "proton", "antiproton", "electron", "positron", "ion", "posmuon",
    "negmuon",
    # MAD-X element classes — an element with one of these names is
    # silently turned into that class (a quad named "wire" becomes a wire)
    "wire", "translation", "srotation", "xrotation", "yrotation",
    "crabcavity", "rfmultipole", "nllens", "twcavity", "beambeam",
    "elseparator", "changeref", "hacdipole", "vacdipole", "sixmarker",
    "decapole", "dodecapole", "yrot", "xrot", "srot", "tdipole",
}
_MAX_NAME_LEN = 40          # MAD-X 5.09 is fatal on element names > 41 chars
_NAME_RE = re.compile(r"[^A-Za-z0-9_.]")


def _fmt(x: float) -> str:
    """Exact shortest repr; never 'nan'/'inf'."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot write non-finite number {x!r} to MAD-X")
    if x == 0.0:
        return "0"
    # Shortest repr that round-trips the double exactly (MAD-X reads it
    # verbatim), so the export→import path loses nothing but arithmetic.
    return repr(x)


def _name_sanitiser(reserved_names=()):
    """Element-name sanitiser.  ``reserved_names`` (e.g. the sequence
    name) are pre-registered: MAD-X crashes outright — no error message —
    when an element and the sequence share a name."""
    seen: dict[str, int] = {n.lower(): 1 for n in reserved_names}

    def sanitise(raw: str, fallback: str) -> str:
        base = (raw or "").strip() or fallback
        base = _NAME_RE.sub("_", base)
        if not re.match(r"^[A-Za-z]", base):
            base = "e_" + base
        if base.lower() in _RESERVED:
            base = base + "_el"
        if len(base) > _MAX_NAME_LEN:
            base = base[:_MAX_NAME_LEN]
        key = base.lower()
        n = seen.get(key, 0)
        seen[key] = n + 1
        if n == 0:
            return base
        # uniquify; keep re-checking so a suffixed name cannot collide
        # with a later raw name
        while True:
            suffix = f"_{n}"
            cand = base[:_MAX_NAME_LEN - len(suffix)] + suffix
            if cand.lower() not in seen:
                seen[cand.lower()] = 1
                return cand
            n += 1

    return sanitise


def _beam_line(ref: ReferenceParticle) -> str:
    sp = ref.species
    total_GeV = (ref.w_kin + sp.mass) / 1000.0
    name = (getattr(sp, "name", "") or "").lower()
    if name == "proton" or (sp.charge == 1 and abs(sp.mass - PROTON.mass) < 1e-6):
        return f"BEAM, particle=proton, energy={_fmt(total_GeV)};"
    # H⁻, deuteron and anything else: generic ion with explicit mass/charge.
    return (f"BEAM, particle=ion, mass={_fmt(sp.mass / 1000.0)}, "
            f"charge={int(sp.charge)}, energy={_fmt(total_GeV)};")


def _command_comment(elem) -> str:
    """One-line ``! HELIX: …`` reproduction of a TraceWin command card."""
    kw = getattr(elem, "KEYWORD", None) or type(elem).__name__
    try:
        args = elem.to_tracewin_args()
    except Exception:  # pragma: no cover - defensive
        args = []
    text = kw if not args else kw + " " + " ".join(str(a) for a in args)
    return f"! HELIX: {text}"


def _misalignment_warning(elem, warnings: list) -> None:
    for attr in ("dx", "dy", "tilt_deg"):
        v = getattr(elem, attr, 0.0)
        if v:
            warnings.append(
                f"{elem.name!r}: misalignment {attr}={v:g} has no MAD-X "
                "element attribute (use EALIGN) — exported aligned")
            return


def write_madx(lattice: Lattice, filepath, ref: ReferenceParticle, *,
               sequence_name: str | None = None,
               on_unsupported: str = "marker",
               linearize: bool = False,
               title: str | None = None) -> list[str]:
    """Write *lattice* as a MAD-X file.  Returns the list of warnings.

    Parameters
    ----------
    lattice : Lattice
    filepath : str or Path
    ref : ReferenceParticle
        Species and kinetic energy at the lattice entrance; sets ``BEAM``
        and the rigidity used for every k1 / ks / kick conversion.
    sequence_name : str, optional
        Defaults to the file stem.
    on_unsupported : {"marker", "error"}
        What to do with elements MAD-X cannot represent.
    linearize : bool
        Export field maps, multi-gap cavities, RFQ cells, thin lenses and
        explicit-matrix elements as MAD-X ``MATRIX`` elements built from
        their first-order map at the local reference energy (MAD-X
        canonical basis, see :mod:`linac_gen.tracking.longitudinal_coords`).
        A reference energy gain is emitted as ``kick6`` (a ``pt`` shift —
        MAD-X cannot move its reference energy) and warns.  Foils and
        electric steerers still fall under ``on_unsupported``.
    title : str, optional
        ``TITLE`` text; defaults to the file stem.
    """
    if on_unsupported not in ("marker", "error"):
        raise ValueError("on_unsupported must be 'marker' or 'error'")
    filepath = Path(filepath)
    seq_name = _NAME_RE.sub("_", (sequence_name or filepath.stem or "").strip()) or "helix"
    if not re.match(r"^[A-Za-z]", seq_name):
        seq_name = "seq_" + seq_name
    if seq_name.lower() in _RESERVED:
        seq_name = seq_name + "_seq"
    # TITLE text: MAD-X reads it verbatim between quotes, but ``!`` / ``;``
    # / quotes / newlines would break the HELIX importer's line splitter.
    title = re.sub(r'[!;"\r\n]+', " ", str(title or filepath.stem)).strip() or "helix"
    warnings: list[str] = []
    brho = _signed_brho(ref)
    sanitise = _name_sanitiser(reserved_names=(seq_name,))

    defs: list[str] = []          # element definition lines
    members: list[str] = []       # sequence member lines (incl. comments)
    s_mm = 0.0                    # running entry position
    current_freq = float(getattr(ref, "frequency", 0.0) or 0.0)

    elems = list(lattice.elements)
    i = 0
    n = len(elems)

    # Local reference replay (same rules as compute_transfer_matrix) —
    # only consulted by the linearised MATRIX path.
    ref_local = ref.copy()
    adv = [0]                     # elements [0, adv) already replayed
    rf_gain = [0.0]               # MeV: design gain through exported RFCAVITYs
    n_rfcav = [0]
    lin_gain = [0.0]              # MeV: reference gain emitted as MATRIX kick6
    n_lin = [0]
    vis_shift = [0.0]             # MeV: explicit-matrix energy offsets already
                                  # emitted as kick6 (they shift the MAD-X-
                                  # visible energy but not HELIX's reference)

    def _advance(r, e, *, reset: bool = True) -> None:
        if isinstance(e, FieldMapElement):
            if reset:
                e.reset_run_state()
            e.advance_ref(r)
        else:
            L = float(getattr(e, "length", 0.0) or 0.0)
            r.s += L
            if L > 0:
                r.phi_s += 360.0 * L / (r.beta * r.wavelength)
            if isinstance(e, ThinKickElement):
                e.advance_ref(r)

    dead = [None]                 # (index, name) where the reference died

    def ref_at(k):
        """Local reference before element k, or None once the replay has
        failed (a deck whose cavities decelerate this species below zero
        energy — e.g. a proton DTL exported with an H⁻ reference)."""
        while adv[0] < k and dead[0] is None:
            try:
                _advance(ref_local, elems[adv[0]])
            except (ValueError, ZeroDivisionError, OverflowError):
                dead[0] = (adv[0], elems[adv[0]].name)
                warnings.append(
                    f"reference particle lost at {elems[adv[0]].name!r} "
                    f"(element {adv[0] + 1}) — this species decelerates "
                    "below zero energy in the deck's RF; the energy-"
                    "dependent notes (line gain, linearised maps) stop "
                    "there, the static conversions are unaffected")
            adv[0] += 1
        return None if dead[0] is not None else ref_local.copy()

    def linearised(elem) -> bool:
        """Emit *elem* as a MAD-X MATRIX; False if it cannot be linearised."""
        nonlocal s_mm
        if not linearize or not isinstance(
                elem, (FieldMapElement, MatrixElement, ThinLens)):
            return False
        from linac_gen.tracking.longitudinal_coords import (
            matrix_to_madx, vector_to_madx)
        from linac_gen.tracking.matrix_tracking import get_element_matrix
        r_true = ref_at(i)            # i: the loop's current index
        if r_true is None:
            return False              # no local energy: MARKER + DRIFT path
        if isinstance(elem, FieldMapElement):
            elem.reset_run_state()    # same order as compute_transfer_matrix
        try:
            M = np.asarray(get_element_matrix(elem, r_true.copy()), dtype=float)
            r_after = r_true.copy()
            _advance(r_after, elem, reset=False)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            warnings.append(
                f"{elem.name!r} ({type(elem).__name__}): first-order map "
                f"unavailable at W_in={r_true.w_kin:.6g} MeV ({exc}) — "
                "exported as MARKER + body DRIFT instead of MATRIX")
            return False
        dW = r_after.w_kin - r_true.w_kin
        offset = getattr(elem, "offset", None)
        dW_off = float(offset[5]) if offset is not None else 0.0
        # MAD-X-visible energies: HELIX's replay plus every energy offset
        # already written as a kick6 (the importer replays exactly this).
        r_in = r_true.copy()
        r_in.w_kin = r_true.w_kin + vis_shift[0]
        r_out = r_in.copy()
        r_out.w_kin = r_in.w_kin + dW + dW_off
        R = matrix_to_madx(M, ref, r_in, r_out)
        kick = np.zeros(6)
        if offset is not None:
            kick += vector_to_madx(offset, ref, r_out)
        p0c = ref.bg * ref.species.mass
        kick[5] += dW / p0c
        vis_shift[0] += dW_off
        lin_gain[0] += dW
        n_lin[0] += 1
        nm = sanitise(elem.name, type(elem).__name__.lower())
        length = float(getattr(elem, "length", 0.0) or 0.0)
        parts = [f"l={_fmt(length * _MM_TO_M)}"]
        parts += [f"kick{a + 1}={_fmt(kick[a])}" for a in range(6) if kick[a] != 0.0]
        rows = [", ".join(f"rm{a + 1}{b + 1}={_fmt(R[a, b])}" for b in range(6))
                for a in range(6)]
        defs.append(f"{nm}: MATRIX, " + ", ".join(parts) + ",\n    "
                    + ",\n    ".join(rows) + ";")
        members.append(f"  {nm}, at={_fmt(s_mm * _MM_TO_M)};")
        note = (f"{elem.name!r} ({type(elem).__name__}): linearised as MAD-X "
                f"MATRIX — first-order map at W_in={r_true.w_kin:.6g} MeV")
        A = R[:4, :4]
        J = np.zeros((4, 4)); J[0, 1] = J[2, 3] = 1.0; J[1, 0] = J[3, 2] = -1.0
        symp_err = float(np.abs(A.T @ J @ A - J).max())
        if symp_err > 1e-6:
            note += (f"; transverse symplectic error {symp_err:.2e} — MAD-X "
                     "TWISS symplectifies MATRIX elements (TRACK applies "
                     "the map verbatim)")
        if abs(dW) > 0.0:
            note += (f"; reference energy gain ΔW={dW:.6g} MeV emitted as "
                     f"kick6={kick[5]:.6g} (pt) — MAD-X keeps the BEAM "
                     "energy fixed, so downstream strengths stay normalised "
                     "by the entrance rigidity")
        warnings.append(note)
        s_mm += length
        return True

    def unsupported(elem, why: str) -> None:
        nonlocal s_mm
        if linearised(elem):
            return
        msg = f"{elem.name!r} ({type(elem).__name__}): {why}"
        if on_unsupported == "error":
            raise ValueError("MAD-X export refused: " + msg)
        warnings.append(msg + " — exported as MARKER + body DRIFT")
        nm = sanitise(elem.name, type(elem).__name__.lower())
        defs.append(f"{nm}: MARKER;")
        members.append(f"  {nm}, at={_fmt(s_mm * _MM_TO_M)};")
        length = float(getattr(elem, "length", 0.0) or 0.0)
        if length > 0:
            bn = sanitise(elem.name + "_body", "body")
            defs.append(f"{bn}: DRIFT, l={_fmt(length * _MM_TO_M)};")
            members.append(f"  {bn}, at={_fmt(s_mm * _MM_TO_M)};")
            s_mm += length

    while i < n:
        elem = elems[i]
        length = float(getattr(elem, "length", 0.0) or 0.0)
        at = _fmt(s_mm * _MM_TO_M)

        # ---- commands and markers that become comments -------------------
        if isinstance(elem, FreqCommand):
            f = float(getattr(elem, "frequency_mhz", 0.0) or 0.0)
            if f > 0:
                current_freq = f
            members.append(f"  ! HELIX: FREQ {_fmt(f)}")
            i += 1
            continue
        if isinstance(elem, LatticeCommand):
            members.append("  " + _command_comment(elem))
            i += 1
            continue
        if isinstance(elem, SpaceChargeComp):
            members.append(f"  ! HELIX: SPACE_CHARGE_COMP {_fmt(elem.factor)}")
            i += 1
            continue
        if isinstance(elem, ScGridDirective):
            members.append("  " + _command_comment(elem))
            i += 1
            continue

        # ---- Edge + Dipole(s) + Edge -> SBEND(s) ------------------------
        # The parser writes Edge, Dipole, Edge; TraceWin decks may also
        # carry SET_*/ADJUST_* cards between them and split a magnet into
        # consecutive BEND cards sharing one pair of EDGEs.  Commands in
        # between still become comments at their position.
        if isinstance(elem, Edge):
            group = _bend_group(elems, i)
            if group is not None:
                e_in, bodies, e_out, next_i = group
                for k in range(i, next_i):
                    e = elems[k]
                    if isinstance(e, FreqCommand):
                        f = float(getattr(e, "frequency_mhz", 0.0) or 0.0)
                        if f > 0:
                            current_freq = f
                        members.append(f"  ! HELIX: FREQ {_fmt(f)}")
                    elif isinstance(e, (LatticeCommand, ScGridDirective)):
                        members.append("  " + _command_comment(e))
                    elif isinstance(e, SpaceChargeComp):
                        members.append(f"  ! HELIX: SPACE_CHARGE_COMP {_fmt(e.factor)}")
                for j, body in enumerate(bodies):
                    _emit_sbend(body,
                                e_in if j == 0 else None,
                                e_out if j == len(bodies) - 1 else None,
                                brho, sanitise, defs, members,
                                _fmt(s_mm * _MM_TO_M), warnings)
                    _misalignment_warning(body, warnings)
                    s_mm += float(body.length)
                if len(bodies) > 1:
                    warnings.append(
                        f"{bodies[0].name!r}..{bodies[-1].name!r}: "
                        f"{len(bodies)} consecutive BEND cards share one "
                        "EDGE pair — exported as adjacent SBENDs with the "
                        "entrance edge on the first and the exit edge on "
                        "the last")
                i = next_i
                continue
        if isinstance(elem, Dipole):
            _emit_sbend(elem, None, None, brho, sanitise, defs, members,
                        at, warnings)
            _misalignment_warning(elem, warnings)
            s_mm += length
            i += 1
            continue
        if isinstance(elem, Edge):
            unsupported(elem, "a pole-face rotation without its bend "
                              "(MAD-X DIPEDGE is not emitted)")
            i += 1
            continue

        # ---- ordinary elements ------------------------------------------
        if length < 0:
            raise ValueError(
                f"MAD-X export refused: {elem.name!r} ({type(elem).__name__}) "
                f"has negative length {length:g} mm at s={s_mm:g} mm — a MAD-X "
                "sequence cannot step backwards (TraceWin negative drifts "
                "overlap elements); remove or fold it first")
        if isinstance(elem, Drift):
            nm = sanitise(elem.name, "drift")
            defs.append(f"{nm}: DRIFT, l={_fmt(length * _MM_TO_M)};")
            members.append(f"  {nm}, at={at};")
        elif isinstance(elem, Quadrupole):
            nm = sanitise(elem.name, "quad")
            k1 = float(elem.gradient) / brho
            parts = [f"l={_fmt(length * _MM_TO_M)}", f"k1={_fmt(k1)}"]
            skew = float(getattr(elem, "skew_angle", 0.0) or 0.0)
            if skew:
                parts.append(f"tilt={_fmt(skew * _DEG_TO_RAD)}")
            for attr in ("g3", "g4", "g5", "g6", "gfr"):
                if getattr(elem, attr, 0.0):
                    warnings.append(
                        f"{elem.name!r}: quadrupole {attr}={getattr(elem, attr):g} "
                        "has no MAD-X QUADRUPOLE attribute — dropped")
            defs.append(f"{nm}: QUADRUPOLE, " + ", ".join(parts) + ";")
            members.append(f"  {nm}, at={at};")
            _misalignment_warning(elem, warnings)
        elif isinstance(elem, Solenoid):
            nm = sanitise(elem.name, "sol")
            ks = float(elem.field) / brho
            defs.append(f"{nm}: SOLENOID, l={_fmt(length * _MM_TO_M)}, "
                        f"ks={_fmt(ks)};")
            members.append(f"  {nm}, at={at};")
            _misalignment_warning(elem, warnings)
        elif isinstance(elem, RFGap):
            nm = sanitise(elem.name, "cav")
            ttf = float(getattr(elem, "ttf", 1.0) or 1.0)
            volt = float(elem.voltage) * ttf
            if abs(ttf - 1.0) > 1e-12:
                warnings.append(
                    f"{elem.name!r}: transit-time factor T={ttf:g} folded "
                    "into RFCAVITY volt (MAD-X has no TTF)")
            if getattr(elem, "p_flag", 0):
                warnings.append(
                    f"{elem.name!r}: absolute-phase gap (p_flag="
                    f"{elem.p_flag}) exported with its design phase")
            freq = float(elem.frequency or current_freq or 0.0)
            if freq > 0:
                current_freq = freq
            lag = (float(elem.phase) + 90.0) / 360.0
            # MAD-X does not multiply the RF kick by the beam charge;
            # HELIX gains q·V·T·cos φs.  A signed volt carries sign(q) so
            # the reference gain AND the phase slope match for H⁻ / ions.
            volt *= 1.0 if float(ref.species.charge) >= 0 else -1.0
            defs.append(f"{nm}: RFCAVITY, l=0, volt={_fmt(volt)}, "
                        f"lag={_fmt(lag)}, freq={_fmt(freq)};")
            members.append(f"  {nm}, at={at};")
            _misalignment_warning(elem, warnings)
            r_gap = ref_at(i)
            if r_gap is not None:
                g = r_gap.copy()
                try:
                    elem.advance_ref(g)
                    rf_gain[0] += g.w_kin - r_gap.w_kin
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass                      # reported by ref_at next time
            n_rfcav[0] += 1
        elif isinstance(elem, Steerer):
            if getattr(elem, "elec", False):
                unsupported(elem, "electrostatic steerer has no MAD-X "
                                  "kicker equivalent")
                i += 1
                continue
            nm = sanitise(elem.name, "kick")
            hk = float(elem.by_l) / brho
            vk = float(elem.bx_l) / brho
            defs.append(f"{nm}: KICKER, hkick={_fmt(hk)}, vkick={_fmt(vk)};")
            members.append(f"  {nm}, at={at};")
        elif isinstance(elem, Multipole):
            nm = sanitise(elem.name, "mult")
            knl = [float(x) for x in (elem.knl or [0.0])]
            ksl = [float(x) for x in (elem.ksl or [])]
            parts = ["knl={" + ", ".join(_fmt(x) for x in knl) + "}"]
            if any(ksl):
                parts.append("ksl={" + ", ".join(_fmt(x) for x in ksl) + "}")
            tilt_deg = float(getattr(elem, "tilt_deg", 0.0) or 0.0)
            if tilt_deg:
                # Multipole.tilt_deg rotates in the OPPOSITE sense to
                # MAD-X's tilt (and to Quadrupole.skew_angle, which
                # follows MAD-X) — pinned against MAD-X's R-matrix in
                # tests/io/test_madx_conventions.py; flip here so the
                # exported optics are MAD-X's.
                parts.append(f"tilt={_fmt(-tilt_deg * _DEG_TO_RAD)}")
            defs.append(f"{nm}: MULTIPOLE, " + ", ".join(parts) + ";")
            members.append(f"  {nm}, at={at};")
        elif isinstance(elem, Aperture):
            nm = sanitise(elem.name, "coll")
            t = int(getattr(elem, "aperture_type", 0))
            dx_m, dy_m = elem.dx * _MM_TO_M, elem.dy * _MM_TO_M
            # xsize/ysize feed MAD-X TRACK's collimator losses; apertype/
            # aperture feed its APERTURE module — emit both.
            if t in (Aperture.RECTANGULAR, Aperture.FRACTION,
                     Aperture.FINGER_H, Aperture.FINGER_V):
                if t != Aperture.RECTANGULAR:
                    warnings.append(
                        f"{elem.name!r}: aperture type {t} is a rectangular "
                        "cut in HELIX tracking — exported as RCOLLIMATOR")
                defs.append(f"{nm}: RCOLLIMATOR, xsize={_fmt(dx_m)}, "
                            f"ysize={_fmt(dy_m)}, apertype=rectangle, "
                            f"aperture={{{_fmt(dx_m)}, {_fmt(dy_m)}}};")
                members.append(f"  {nm}, at={at};")
            elif t == Aperture.CIRCULAR:
                defs.append(f"{nm}: ECOLLIMATOR, xsize={_fmt(dx_m)}, "
                            f"ysize={_fmt(dx_m)}, apertype=circle, "
                            f"aperture={{{_fmt(dx_m)}}};")
                members.append(f"  {nm}, at={at};")
            else:
                unsupported(elem, f"aperture type {t} has no MAD-X collimator")
                i += 1
                continue
        elif isinstance(elem, Marker):
            card_args = getattr(elem, "lattice_card_args", None)
            if card_args:
                members.append(f"  ! HELIX: LATTICE {' '.join(str(a) for a in card_args)}")
            fm = float(getattr(elem, "frequency_MHz", 0.0) or 0.0)
            if fm > 0:
                current_freq = fm
                members.append(f"  ! HELIX: FREQ {_fmt(fm)}")
                i += 1
                continue
            nm = sanitise(elem.name, "mark")
            kind = "MONITOR" if getattr(elem, "is_bpm", False) else "MARKER"
            defs.append(f"{nm}: {kind};")
            members.append(f"  {nm}, at={at};")
        elif isinstance(elem, ThinLens):
            unsupported(elem, "asymmetric thin lens has no MAD-X element")
            i += 1
            continue
        else:
            unsupported(elem, "no MAD-X representation")
            i += 1
            continue

        s_mm += length
        i += 1

    if n_rfcav[0] and abs(rf_gain[0]) > 1e-9 * max(float(ref.w_kin), 1e-9):
        warnings.append(
            f"line accelerates: design gain ΔW={rf_gain[0]:.6g} MeV over "
            f"{n_rfcav[0]} RFCAVITY element(s).  MAD-X keeps the BEAM energy "
            "fixed along a sequence — every k1/ks is normalised by the "
            "entrance rigidity (the exact inverse of the importer); MAD-X "
            "TWISS does not follow the energy gain (TRACK sees it as pt) and "
            "its RFCAVITY carries no transverse RF (de)focusing")

    if n_lin[0] and abs(lin_gain[0]) > 0.0:
        # pt = ΔE/(p₀c) is a deviation coordinate about the ENTRANCE
        # momentum.  MAD-X's kinematics are exact in pt, but its TWISS
        # closed-orbit search and the linear MATRIX maps assume it stays
        # small; measured on a 2→10 MeV linac (cpymad, 2026-09-03): TWISS
        # fails once the accumulated pt reaches a few 1e-2 and TRACK's pt
        # overflows to NaN by the end.  Say how far the line goes.
        p0c = ref.bg * ref.species.mass
        pt_total = lin_gain[0] / p0c
        level = "" if abs(pt_total) < 1e-2 else (
            "MAD-X TWISS may fail and TRACK may diverge — " if abs(pt_total) < 0.3
            else "MAD-X TWISS and TRACK will NOT reproduce this line — ")
        warnings.append(
            f"linearised export: {n_lin[0]} MATRIX element(s) carry a total "
            f"reference gain ΔW={lin_gain[0]:.6g} MeV = pt {pt_total:.3g} of the "
            f"entrance momentum ({p0c:.6g} MeV/c).  {level}the MATRIX "
            "representation is valid only while the accumulated pt stays "
            "small; for a full linac export a section at a time from its "
            "own entrance energy")

    n_err = len(getattr(lattice, "errors", None) or [])
    n_berr = len(getattr(lattice, "beam_errors", None) or [])
    if n_err or n_berr:
        # Same contract as write_tracewin: an error STUDY (ERROR_* cards)
        # is not a lattice element and is not serialised — MAD-X carries
        # tolerances through EALIGN/EFCOMP + a seed, a different model.
        warnings.append(
            f"this lattice carries {n_err} element + {n_berr} beam ERROR_* "
            "definitions (an error study) — not exported: MAD-X models "
            "tolerances through EALIGN/EFCOMP, which this writer does not "
            "emit; keep the original deck for error work")
        members.append(f"  ! HELIX: {n_err} ERROR_* element + {n_berr} beam "
                       "error definitions were attached to this lattice "
                       "(not representable here)")

    total_m = s_mm * _MM_TO_M
    if not total_m > 0:
        raise ValueError(
            "MAD-X export refused: the lattice has zero total length — MAD-X "
            "requires a sequence of positive length (add a drift)")
    stamp = _dt.date.today().isoformat()
    try:
        from linac_gen import __version__ as _ver
    except Exception:  # pragma: no cover
        _ver = "?"
    sp = ref.species
    lines = [
        f"! HELIX {_ver} MAD-X export — {title}  ({stamp})",
        f"! reference: {getattr(sp, 'name', 'proton')} W_kin={_fmt(ref.w_kin)} MeV, "
        f"f={_fmt(getattr(ref, 'frequency', 0.0) or 0.0)} MHz, Bρ={_fmt(abs(brho))} T·m",
        "! lengths in m, angles in rad, k1 in 1/m², ks in 1/m, volt in MV, freq in MHz",
        f'TITLE, "{title}";',
        _beam_line(ref),
        "",
        *defs,
        "",
        f"{seq_name}: SEQUENCE, refer=entry, l={_fmt(total_m)};",
        *members,
        "ENDSEQUENCE;",
        "",
        f"USE, sequence={seq_name};",
        "",
    ]
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return warnings


def _bend_group(elems, i):
    """From an Edge at ``elems[i]``: (edge_in, [bodies], edge_out, next_i)
    when the run Edge, {commands}, Dipole, {commands, Dipole}…, Edge
    closes; ``None`` otherwise.  Commands between are comment-only."""
    e_in = elems[i]
    bodies = []
    k = i + 1
    n = len(elems)
    while k < n:
        e = elems[k]
        if isinstance(e, (LatticeCommand, SpaceChargeComp, ScGridDirective)):
            k += 1
            continue
        if isinstance(e, Dipole):
            bodies.append(e)
            k += 1
            continue
        if isinstance(e, Edge) and bodies:
            return e_in, bodies, e, k + 1
        return None
    return None


def _edge_angle_madx(edge, body, warnings) -> float:
    """HELIX pole rotation β [deg] → MAD-X e [rad].

    HELIX's Edge focusing is ``tan β / ρ`` with the edge's own ρ and a
    sign fixed by β alone (β > 0 defocuses horizontally whichever way the
    magnet bends); MAD-X uses ``h·tan e`` with the SIGNED curvature
    h = angle/l.  So e = sign(angle·ρ)·β, and an edge whose ρ differs
    from the body's is re-expressed through the body's h (exact for the
    thin-edge term; the fringe-field term keeps the body's ρ)."""
    beta = float(edge.pole_rotation) * _DEG_TO_RAD
    rho_e = float(edge.rho)
    rho_b = float(body.rho)
    sgn = 1.0 if float(body.angle) * rho_b >= 0 else -1.0
    if rho_e and abs(abs(rho_e) - abs(rho_b)) > 1e-9 * max(abs(rho_b), 1.0):
        warnings.append(
            f"{body.name!r}: edge {edge.name!r} has ρ={rho_e:g} mm but the "
            f"body has ρ={rho_b:g} mm — pole-face angle re-expressed "
            "through the body's curvature (exact for the thin-edge term)")
        beta = math.atan(math.tan(beta) * abs(rho_b) / abs(rho_e))
    return sgn * beta


def _emit_sbend(body: Dipole, e_in, e_out, brho: float, sanitise, defs,
                members, at: str, warnings: list) -> None:
    nm = sanitise(body.name, "bend")
    angle_rad = float(body.angle) * _DEG_TO_RAD
    rho_mm = float(body.rho)
    l_m = float(body.length) * _MM_TO_M
    rho_m = abs(rho_mm) * _MM_TO_M
    hv = int(getattr(body, "hv", 0) or 0)
    # MAD-X: signed angle, positive arc length; a HELIX body with ρ < 0
    # (legacy signed-ρ convention) bends the way its angle says.
    parts = [f"l={_fmt(l_m)}", f"angle={_fmt(angle_rad)}"]
    sgn = 1.0 if angle_rad * rho_mm >= 0 else -1.0
    e1_body = float(getattr(body, "e1", 0.0) or 0.0) * _DEG_TO_RAD * sgn
    e2_body = float(getattr(body, "e2", 0.0) or 0.0) * _DEG_TO_RAD * sgn
    e1 = e1_body
    e2 = e2_body
    for edge, which in ((e_in, "e1"), (e_out, "e2")):
        if edge is None:
            continue
        if int(getattr(edge, "hv", 0) or 0) != hv:
            warnings.append(
                f"{body.name!r}: edge {edge.name!r} is in the other bend "
                "plane (hv differs) — folded into the SBEND anyway")
        e_edge = _edge_angle_madx(edge, body, warnings)
        e_prev = e1_body if which == "e1" else e2_body
        if e_prev:
            warnings.append(
                f"{body.name!r}: both the body's {which} and an EDGE element "
                "carry a pole-face angle — combined as one MAD-X edge")
            e_edge = math.atan(math.tan(e_edge) + math.tan(e_prev))
        if which == "e1":
            e1 = e_edge
        else:
            e2 = e_edge
    if e1:
        parts.append(f"e1={_fmt(e1)}")
    if e2:
        parts.append(f"e2={_fmt(e2)}")
    n_idx = float(getattr(body, "field_index", 0.0) or 0.0)
    if n_idx and rho_m:
        parts.append(f"k1={_fmt(-n_idx / (rho_m * rho_m))}")
    if hv == 1:
        parts.append("tilt=pi/2")
    gap_in = float(getattr(e_in, "gap", 0.0) or 0.0) if e_in is not None else 0.0
    gap_out = float(getattr(e_out, "gap", 0.0) or 0.0) if e_out is not None else 0.0
    gap = gap_in or gap_out
    if gap > 0:
        parts.append(f"hgap={_fmt(gap / 2.0 * _MM_TO_M)}")
        k1_in = float(getattr(e_in, "k1", 0.45)) if e_in is not None else 0.0
        k1_out = float(getattr(e_out, "k1", 0.45)) if e_out is not None else 0.0
        # MAD-X: fint applies to both faces unless fintx is given
        parts.append(f"fint={_fmt(k1_in if e_in is not None else k1_out)}")
        if e_in is not None and e_out is not None and abs(k1_out - k1_in) > 1e-12:
            parts.append(f"fintx={_fmt(k1_out)}")
        if gap_in and gap_out and abs(gap_out - gap_in) > 1e-9:
            warnings.append(
                f"{body.name!r}: entrance/exit edges differ in gap — MAD-X "
                "SBEND carries one hgap (entrance value used)")
        for edge in (e_in, e_out):
            if edge is not None and abs(float(getattr(edge, "k2", 2.80)) - 2.80) > 1e-12:
                warnings.append(
                    f"{body.name!r}: edge K2={edge.k2:g} has no MAD-X "
                    "counterpart (only fint/hgap) — dropped")
                break
    defs.append(f"{nm}: SBEND, " + ", ".join(parts) + ";")
    members.append(f"  {nm}, at={at};")


__all__ = ["write_madx"]
