"""In-house MAD-X lattice importer (subset parser).

Reads a MAD-X ``.madx`` / ``.seq`` file and builds a HELIX
:class:`~linac_gen.core.lattice.Lattice`, mirroring the
``(Lattice, metadata)`` return shape of
:func:`linac_gen.io.tracewin_parser.parse_tracewin`.  The existing
TraceWin parser is **not** touched — this is a parallel importer.

Supported subset
----------------
* Comments: ``! …``, ``// …``, and ``/* … */`` blocks.
* Variable assignments: ``name = expr ;`` and ``name := expr ;`` with
  literal numbers, ``+ - * / ^`` arithmetic, parentheses, the constants
  ``pi``/``twopi``/``e``/``clight``, ``sqrt``/``abs``/``sin``/``cos``,
  and references to previously-defined variables.
* Element definitions: ``name : TYPE, attr=val, … ;``
* ``SEQUENCE … ENDSEQUENCE`` blocks with ``elem, at=POS;`` members
  (``REFER`` defaults to ``centre``).  Gaps between placed elements are
  filled with ``Drift`` elements so the result is a contiguous HELIX
  element list.
* ``BEAM`` command — ``particle``, ``energy`` / ``pc`` / ``gamma``.
* ``USE, SEQUENCE=name`` — selects which sequence to expand.

Element type mapping (all lengths converted metres → millimetres):

    DRIFT       -> Drift
    QUADRUPOLE  -> Quadrupole          (k1 -> gradient via Bρ)
    SBEND       -> Edge + Dipole + Edge
    RBEND       -> Edge + Dipole + Edge (e1,e2 get +angle/2)
    SEXTUPOLE   -> Multipole           (thin; drift-padded if l>0)
    MULTIPOLE   -> Multipole
    RFCAVITY    -> RFGap               (thin; drift-padded if l>0)
    SOLENOID    -> Solenoid            (ks -> field via Bρ)
    MARKER      -> Marker
    MONITOR/HMONITOR/VMONITOR -> Marker(is_bpm=True)

Not supported (skipped, with a warning in ``metadata["warnings"]``):
``MACRO``, ``LINE = (...)`` expansion, ``IF``/``WHILE``, ``MATCH`` and
``TRACK`` blocks, and deferred expressions that reference
later-defined variables.
"""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path

from linac_gen.core.constants import C_LIGHT
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import DEUTERON, H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.marker import Marker
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid

_M_TO_MM = 1000.0
_RAD_TO_DEG = 180.0 / math.pi

# ---------------------------------------------------------------------------
# Safe arithmetic expression evaluator
# ---------------------------------------------------------------------------

_CONSTANTS = {
    "pi": math.pi,
    "twopi": 2.0 * math.pi,
    "e": math.e,
    "clight": C_LIGHT,
    "true": 1.0,
    "false": 0.0,
}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "exp": math.exp, "log": math.log,
}


def _eval_expr(expr: str, variables: dict) -> float:
    """Evaluate a MAD-X arithmetic expression to a float.

    Only numeric literals, the whitelisted constants/functions, basic
    arithmetic, and references into ``variables`` are permitted — there
    is no general ``eval``.  Raises ValueError on anything else.
    """
    src = expr.strip().replace("^", "**")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse MAD-X expression {expr!r}: {exc}")
    return _eval_node(tree.body, variables)


def _eval_node(node, variables: dict) -> float:
    if isinstance(node, ast.Constant):           # 3.8+ numeric literal
        if isinstance(node.value, bool):
            return 1.0 if node.value else 0.0
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported literal {node.value!r}")
    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in _CONSTANTS:
            return _CONSTANTS[key]
        if key in variables:
            return float(variables[key])
        raise ValueError(f"unknown MAD-X identifier {node.id!r}")
    if isinstance(node, ast.BinOp):
        lhs = _eval_node(node.left, variables)
        rhs = _eval_node(node.right, variables)
        op = node.op
        if isinstance(op, ast.Add):
            return lhs + rhs
        if isinstance(op, ast.Sub):
            return lhs - rhs
        if isinstance(op, ast.Mult):
            return lhs * rhs
        if isinstance(op, ast.Div):
            return lhs / rhs
        if isinstance(op, ast.Pow):
            return lhs ** rhs
        raise ValueError("unsupported binary operator in MAD-X expression")
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise ValueError("unsupported unary operator in MAD-X expression")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id.lower()
        if fname not in _FUNCS:
            raise ValueError(f"unsupported MAD-X function {fname!r}")
        args = [_eval_node(a, variables) for a in node.args]
        return float(_FUNCS[fname](*args))
    raise ValueError("unsupported construct in MAD-X expression")


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove ``! …``, ``// …`` line comments and ``/* … */`` blocks."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    out_lines = []
    for line in text.splitlines():
        line = re.split(r"!|//", line, maxsplit=1)[0]
        out_lines.append(line)
    return "\n".join(out_lines)


def _split_statements(text: str) -> list[str]:
    """Split a comment-stripped MAD-X body into ``;``-terminated statements."""
    stmts = []
    for raw in text.split(";"):
        s = raw.strip()
        if s:
            stmts.append(s)
    return stmts


def _split_attributes(body: str) -> list[str]:
    """Split a comma-separated attribute list, respecting brace/paren depth
    so vectors like ``knl={0,0,1.2}`` stay intact."""
    parts = []
    depth = 0
    cur = []
    for ch in body:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        tail = "".join(cur).strip()
        if tail:
            parts.append(tail)
    return parts


def _parse_attributes(body: str, variables: dict) -> dict:
    """Parse ``attr=val, attr=val, flag`` into a dict.

    Bare flags (no ``=``) map to ``True``.  Vector values
    ``{a,b,c}`` become lists of floats.  Quoted strings keep their text.
    """
    attrs: dict = {}
    for part in _split_attributes(body):
        if not part:
            continue
        if "=" not in part:
            attrs[part.strip().lower()] = True
            continue
        key, _, val = part.partition("=")
        # ``:=`` deferred form — the colon belongs to the operator.
        key = key.rstrip(":").strip().lower()
        val = val.strip()
        if val.startswith('"') and val.endswith('"'):
            attrs[key] = val[1:-1]
        elif val.startswith("{") and val.endswith("}"):
            inner = val[1:-1]
            attrs[key] = [
                _eval_expr(tok, variables) for tok in inner.split(",")
                if tok.strip()
            ]
        else:
            try:
                attrs[key] = _eval_expr(val, variables)
            except ValueError:
                attrs[key] = val      # keep as raw string (e.g. particle name)
    return attrs


# ---------------------------------------------------------------------------
# Reference particle / rigidity
# ---------------------------------------------------------------------------

def _species_from_name(name: str):
    n = (name or "proton").strip().lower()
    if n in ("proton", "p"):
        return PROTON
    if n in ("deuteron", "d"):
        return DEUTERON
    if n in ("h-", "hminus", "h_minus", "ion"):
        return H_MINUS
    return None        # caller emits a warning and falls back to PROTON


def _reference_from_beam(beam_attrs: dict, frequency_MHz: float,
                          warnings: list) -> ReferenceParticle:
    """Build a ReferenceParticle from a parsed MAD-X BEAM command."""
    species = _species_from_name(str(beam_attrs.get("particle", "proton")))
    if species is None:
        warnings.append(
            f"BEAM particle {beam_attrs.get('particle')!r} not modelled — "
            "defaulting to proton"
        )
        species = PROTON
    mass_MeV = species.mass

    # MAD-X energy keywords (GeV): energy = total, pc = momentum·c,
    # gamma = Lorentz factor.  HELIX wants kinetic energy in MeV.
    if "energy" in beam_attrs:
        total_MeV = float(beam_attrs["energy"]) * 1000.0
    elif "pc" in beam_attrs:
        pc_MeV = float(beam_attrs["pc"]) * 1000.0
        total_MeV = math.sqrt(pc_MeV ** 2 + mass_MeV ** 2)
    elif "gamma" in beam_attrs:
        total_MeV = float(beam_attrs["gamma"]) * mass_MeV
    else:
        warnings.append(
            "BEAM command has no energy/pc/gamma — defaulting to "
            "1 GeV kinetic"
        )
        total_MeV = mass_MeV + 1000.0
    w_kin = max(total_MeV - mass_MeV, 1e-9)
    return ReferenceParticle(species=species, w_kin=w_kin,
                             frequency=frequency_MHz)


def _brho(ref: ReferenceParticle) -> float:
    """Magnetic rigidity Bρ [T·m] for the reference particle.

    Bρ = p / |q|.  With p[kg·m/s] = βγ · m · c and |q| = e,
    Bρ = βγ · m[MeV] · 1e6 / c_light  (the e's cancel).
    """
    return ref.bg * ref.species.mass * 1e6 / C_LIGHT


def _signed_brho(ref: ReferenceParticle) -> float:
    """sign(q) · |Bρ| — the conversion factor from MAD normalized
    strengths to HELIX lab-frame fields.

    MAD's K1 (and solenoid KS) are normalized by q·p: K1 = (q/p)·∂B_y/∂x,
    i.e. K1 > 0 is horizontally focusing for the *reference charge*.
    HELIX stores the physical lab gradient G = ∂B_y/∂x and applies the
    beam's charge sign at track time (Quadrupole.transfer_matrix:
    k1 = sign(q)·G/Bρ).  So G = sign(q)·K1·|Bρ| — for H⁻ the gradient
    sign flips relative to a proton import.  External anchor: the legacy
    PIP-II BTL conversion header ``variable mad2tw -4.8828922`` (negative
    Bρ for 800 MeV H⁻).
    """
    q = float(getattr(ref.species, "charge", 1) or 1)
    return math.copysign(_brho(ref), q)


# ---------------------------------------------------------------------------
# Element builders — each returns (list[Element], total_length_mm)
# ---------------------------------------------------------------------------

def _gf(attrs: dict, key: str, default: float = 0.0) -> float:
    """Fetch a numeric attribute as float, with a default."""
    v = attrs.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _build_element(name: str, etype: str, attrs: dict, brho: float,
                   warnings: list) -> tuple[list, float]:
    """Map one MAD-X element to HELIX element(s).  Lengths returned in mm."""
    etype = etype.lower()
    l_mm = _gf(attrs, "l") * _M_TO_MM
    aperture_mm = _gf(attrs, "aperture") * _M_TO_MM   # MAD-X aperture rarely used

    if etype == "drift":
        return [Drift(name=name, length=l_mm, aperture=aperture_mm)], l_mm

    if etype == "quadrupole":
        # k1 [1/m²] -> gradient [T/m] = k1 · Bρ
        gradient = _gf(attrs, "k1") * brho
        return [Quadrupole(name=name, length=l_mm, gradient=gradient,
                           aperture=aperture_mm)], l_mm

    if etype in ("sbend", "rbend"):
        angle_rad = _gf(attrs, "angle")
        e1 = _gf(attrs, "e1")
        e2 = _gf(attrs, "e2")
        if etype == "rbend":
            # Rectangular bend: pole faces are parallel to the chord, so
            # each edge picks up an extra angle/2.  l is taken as the arc
            # length (MAD-X default RBARC=TRUE).
            e1 += angle_rad / 2.0
            e2 += angle_rad / 2.0
        if abs(angle_rad) < 1e-12:
            # Zero-angle bend is just a drift.
            return [Drift(name=name, length=l_mm,
                          aperture=aperture_mm)], l_mm
        angle_deg = angle_rad * _RAD_TO_DEG
        # Arc length l = |rho|·|angle| ⇒ rho = l / angle.
        rho_mm = l_mm / angle_rad
        edge_in = Edge(name=f"{name}_e1", pole_rotation=e1 * _RAD_TO_DEG,
                       rho=rho_mm, aperture=aperture_mm)
        body = Dipole(name=name, angle=angle_deg, rho=rho_mm,
                      aperture=aperture_mm)
        edge_out = Edge(name=f"{name}_e2", pole_rotation=e2 * _RAD_TO_DEG,
                        rho=rho_mm, aperture=aperture_mm)
        # Edges are zero-length; the Dipole carries the full arc length.
        return [edge_in, body, edge_out], body.length

    if etype == "solenoid":
        # ks [1/m] -> on-axis field B [T] = ks · Bρ.
        field = _gf(attrs, "ks") * brho
        return [Solenoid(name=name, length=l_mm, field=field,
                         aperture=aperture_mm)], l_mm

    if etype == "sextupole":
        # k2 [1/m³], integrated k2l = k2·l (HELIX/MAD-X knl convention).
        k2l = _gf(attrs, "k2") * _gf(attrs, "l")
        mp = Multipole(name=name, knl=[0.0, 0.0, k2l], aperture=aperture_mm)
        return _drift_pad(mp, l_mm, name)

    if etype == "multipole":
        knl = attrs.get("knl", []) or []
        ksl = attrs.get("ksl", []) or []
        knl = [float(x) for x in knl] if isinstance(knl, list) else []
        ksl = [float(x) for x in ksl] if isinstance(ksl, list) else []
        mp = Multipole(name=name, knl=knl or [0.0], ksl=ksl or [0.0],
                       aperture=aperture_mm)
        return _drift_pad(mp, l_mm, name)

    if etype == "rfcavity":
        volt_MV = _gf(attrs, "volt")            # MAD-X volt is in MV
        lag = _gf(attrs, "lag")                 # phase in units of 2π
        freq_MHz = _gf(attrs, "freq")           # MAD-X freq is in MHz
        gap = RFGap(name=name, voltage=volt_MV, phase=lag * 360.0,
                    frequency=freq_MHz)
        return _drift_pad(gap, l_mm, name)

    if etype == "marker":
        return [Marker(name=name)], 0.0

    if etype in ("monitor", "hmonitor", "vmonitor"):
        # Beam-position monitor → a zero-length Marker flagged as a BPM.
        # If the MAD-X monitor carried a length, drift-pad it.
        return _drift_pad(Marker(name=name, is_bpm=True), l_mm, name)

    if etype in ("instrument", "placeholder"):
        # Generic instrument / placeholder → plain zero-length Marker.
        return _drift_pad(Marker(name=name, is_bpm=False), l_mm, name)

    # Unknown element type — emit a zero-length marker so the sequence
    # geometry is preserved, and warn.
    warnings.append(f"element {name!r}: unsupported type {etype!r} — "
                    "replaced with a marker")
    return [Marker(name=name)], 0.0


def _drift_pad(thin_elem, l_mm: float, name: str) -> tuple[list, float]:
    """Wrap a zero-length HELIX element in half-drifts when the MAD-X
    element carried a non-zero length, so the sequence geometry holds."""
    if l_mm <= 1e-9:
        return [thin_elem], 0.0
    half = l_mm / 2.0
    return (
        [Drift(name=f"{name}_din", length=half),
         thin_elem,
         Drift(name=f"{name}_dout", length=half)],
        l_mm,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_madx(filepath: str) -> tuple[Lattice, dict]:
    """Parse a MAD-X ``.madx`` / ``.seq`` file into a HELIX Lattice.

    Returns ``(lattice, metadata)`` where ``metadata`` has keys
    ``"title"`` (str), ``"warnings"`` (list[str]) and ``"reference"``
    (the :class:`ReferenceParticle` built from the BEAM command).
    """
    text = _strip_comments(Path(filepath).read_text())
    statements = _split_statements(text)

    variables: dict = {}
    element_defs: dict = {}        # name -> (type, attrs)
    sequences: dict = {}          # name -> {"l": float, "members": [...]}
    beam_attrs: dict = {}
    use_sequence: str | None = None
    title = ""
    warnings: list[str] = []

    cur_seq: str | None = None    # name of the SEQUENCE block being read

    _ASSIGN = re.compile(r"^([A-Za-z_][\w.]*)\s*:?=\s*(.+)$", re.DOTALL)
    _DEFINE = re.compile(r"^([A-Za-z_][\w.]*)\s*:\s*([A-Za-z_]\w*)\s*"
                         r"(?:,(.*))?$", re.DOTALL)

    for stmt in statements:
        low = stmt.lower().strip()

        # --- inside a SEQUENCE block ----------------------------------
        if cur_seq is not None:
            if low == "endsequence" or low.startswith("endsequence"):
                cur_seq = None
                continue
            # A member line: ``elemname, at=POS [, attrs]`` or an inline
            # definition ``name: TYPE, at=POS, …``.
            mdef = _DEFINE.match(stmt)
            if mdef:
                ename, etype, body = mdef.group(1), mdef.group(2), \
                    mdef.group(3) or ""
                attrs = _parse_attributes(body, variables)
                element_defs[ename.lower()] = (etype.lower(), attrs)
                at = _gf(attrs, "at", float("nan"))
                sequences[cur_seq]["members"].append((ename.lower(), at))
                continue
            # Reference to an already-defined element.
            head, _, body = stmt.partition(",")
            ref_name = head.strip().lower()
            attrs = _parse_attributes(body, variables)
            at = _gf(attrs, "at", float("nan"))
            sequences[cur_seq]["members"].append((ref_name, at))
            continue

        # --- variable assignment (incl. deferred :=) ------------------
        # Only treat as an assignment when the LHS is a single identifier
        # and there is no comma before the '=' (which would make it a
        # command with attributes).
        massign = _ASSIGN.match(stmt)
        if massign and "," not in stmt.split("=", 1)[0] \
                and ":" not in stmt.split("=", 1)[0].rstrip(":"):
            name, expr = massign.group(1), massign.group(2)
            try:
                variables[name.lower()] = _eval_expr(expr, variables)
            except ValueError as exc:
                warnings.append(f"skipped variable {name!r}: {exc}")
            continue

        # --- definition: ``name : TYPE, …`` --------------------------
        mdef = _DEFINE.match(stmt)
        if mdef and ":=" not in stmt:
            ename, etype, body = mdef.group(1), mdef.group(2), \
                mdef.group(3) or ""
            etype_low = etype.lower()
            attrs = _parse_attributes(body, variables)
            if etype_low == "sequence":
                cur_seq = ename.lower()
                sequences[cur_seq] = {
                    "l": _gf(attrs, "l"),
                    "refer": str(attrs.get("refer", "centre")).lower(),
                    "members": [],
                }
            elif etype_low == "line":
                warnings.append(
                    f"sequence {ename!r}: LINE=(...) form not supported "
                    "(only SEQUENCE) — skipped"
                )
            else:
                element_defs[ename.lower()] = (etype_low, attrs)
            continue

        # --- commands -------------------------------------------------
        head, _, body = stmt.partition(",")
        cmd = head.strip().lower()
        if cmd == "beam":
            beam_attrs = _parse_attributes(body, variables)
        elif cmd == "title":
            title = body.strip().strip('"') or title
        elif cmd == "use":
            use_attrs = _parse_attributes(body, variables)
            seq = use_attrs.get("sequence") or use_attrs.get("period")
            if seq:
                use_sequence = str(seq).lower()
        elif cmd in ("option", "set", "select", "exec", "return",
                     "value", "show", "stop", "system", "print"):
            pass    # benign, ignored silently
        elif cmd in ("macro",) or low.endswith("macro"):
            warnings.append("MACRO definitions are not supported — skipped")
        elif cmd in ("match", "track", "twiss", "survey", "plot",
                     "sodd", "emit", "ibs", "aperture", "makethin"):
            warnings.append(f"{cmd!r} command not supported — skipped")
        elif cmd in ("ealign", "efcomp", "error", "seqedit", "endedit",
                     "flatten", "install", "move", "remove", "eoption"):
            # Alignment / field-error / sequence-editing commands change
            # the MACHINE — silently ignoring them imports a perfect
            # lattice where the file declared an imperfect one.
            warnings.append(
                f"{cmd!r} (alignment / field-error / sequence editing) "
                "not supported — the declared errors/edits are NOT "
                "imported; the lattice is built error-free.")
        elif cmd == "call":
            # File inclusion is how real MAD-X decks are organised
            # (elements / sequences / strengths in separate files).
            # Silently dropping it imported an INCOMPLETE lattice with
            # no signal at all (2026-07-25 review, claim 6).
            warnings.append(
                "CALL is not supported — the included file is NOT read; "
                "every element, sequence and strength it defines is "
                "MISSING from this import.  Inline the file or export a "
                "flattened deck from MAD-X before importing.")
        # Anything else: silently ignore (constants, etc.).

    # --- choose the sequence to expand --------------------------------
    if not sequences:
        raise ValueError(
            "MAD-X file contains no SEQUENCE block — nothing to import"
        )
    if use_sequence and use_sequence in sequences:
        seq_name = use_sequence
    else:
        if use_sequence:
            warnings.append(
                f"USE sequence {use_sequence!r} not found — using "
                f"the last-defined sequence instead"
            )
        seq_name = list(sequences.keys())[-1]
    seq = sequences[seq_name]

    # --- reference particle + rigidity --------------------------------
    # RF frequency for the ReferenceParticle: take the first RFCAVITY's
    # freq if present, else a harmless default (MAD-X lattices are often
    # magnet-only).
    freq_MHz = 0.0
    for etype, attrs in element_defs.values():
        if etype == "rfcavity" and _gf(attrs, "freq") > 0:
            freq_MHz = _gf(attrs, "freq")
            break
    if freq_MHz <= 0:
        freq_MHz = 352.21        # benign placeholder; no RF ⇒ unused
    reference = _reference_from_beam(beam_attrs, freq_MHz, warnings)
    # Charge-signed: K1/KS -> lab-frame G/B needs sign(q) (see
    # _signed_brho).  For negative species (H-) this flips every
    # imported gradient relative to the old unsigned conversion — the
    # old behaviour was wrong for H- decks; proton decks are unchanged.
    brho = _signed_brho(reference)

    # --- resolve the sequence into a contiguous element list ----------
    lattice = Lattice()
    placed: list[tuple[float, float, list]] = []   # (entry_mm, exit_mm, elems)
    for ename, at in seq["members"]:
        if ename not in element_defs:
            warnings.append(f"sequence member {ename!r} is undefined — skipped")
            continue
        etype, attrs = element_defs[ename]
        elems, total_mm = _build_element(ename, etype, attrs, brho, warnings)
        if math.isnan(at):
            warnings.append(
                f"member {ename!r} has no at= position — skipped"
            )
            continue
        at_mm = at * _M_TO_MM
        refer = seq.get("refer", "centre")
        if refer == "entry":
            entry = at_mm
        elif refer == "exit":
            entry = at_mm - total_mm
        else:                                   # centre (MAD-X default)
            entry = at_mm - total_mm / 2.0
        placed.append((entry, entry + total_mm, elems))

    placed.sort(key=lambda p: p[0])
    seq_len_mm = seq["l"] * _M_TO_MM
    cursor = 0.0
    drift_n = 0
    for entry, exit_, elems in placed:
        gap = entry - cursor
        if gap > 1e-6:
            drift_n += 1
            lattice.add(Drift(name=f"DRIFT_{drift_n:04d}", length=gap))
        elif gap < -1e-6:
            warnings.append(
                f"elements overlap by {-gap:.3f} mm near s={cursor:.1f} mm "
                "— geometry clamped"
            )
        for e in elems:
            lattice.add(e)
        cursor = max(cursor, exit_)
    # Trailing drift to the declared sequence length.
    if seq_len_mm - cursor > 1e-6:
        drift_n += 1
        lattice.add(Drift(name=f"DRIFT_{drift_n:04d}",
                          length=seq_len_mm - cursor))

    metadata = {
        "title": title or seq_name,
        "warnings": warnings,
        "reference": reference,
    }
    return lattice, metadata


__all__ = ["parse_madx"]
