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
    QUADRUPOLE  -> Quadrupole          (k1 -> gradient via Bρ; tilt -> skew)
    KICKER/HKICKER/VKICKER -> Steerer  (hkick/vkick -> ∫B·dl via signed Bρ)
    RCOLLIMATOR/ECOLLIMATOR -> Aperture (rectangular / circular)
    SBEND       -> Edge + Dipole + Edge
    RBEND       -> Edge + Dipole + Edge (e1,e2 get +angle/2; L is the
                                        chord, arc = L·(θ/2)/sin(θ/2))
                                       (k1 -> field index n = -k1·ρ²;
                                        tilt=±π/2 -> vertical bend;
                                        hgap/fint -> Edge gap/K1)
    SEXTUPOLE   -> Multipole           (thin; drift-padded if l>0)
    MULTIPOLE   -> Multipole
    RFCAVITY    -> RFGap               (thin; drift-padded if l>0;
                                        lag -> φs = 360·lag − 90°, see
                                        the RFCAVITY branch for why)
    SOLENOID    -> Solenoid            (ks -> field via Bρ)
    MARKER      -> Marker
    MONITOR/HMONITOR/VMONITOR -> Marker(is_bpm=True)
    MATRIX      -> MatrixElement       (rm/kick converted from MAD-X's
                                        canonical basis at the local energy)

Not supported (skipped, with a warning in ``metadata["warnings"]``):
``MACRO``, ``LINE = (...)`` expansion, ``IF``/``WHILE``, ``MATCH`` and
``TRACK`` blocks, and deferred expressions that reference
later-defined variables.
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np
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
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.matrix_element import MatrixElement

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

def _strip_line_comment(line: str) -> str:
    """Cut a ``!`` / ``//`` comment, ignoring markers inside "…" / '…'."""
    quote = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "!" or line.startswith("//", i):
            return line[:i]
        i += 1
    return line


def _strip_comments(text: str) -> str:
    """Remove ``! …``, ``// …`` line comments and ``/* … */`` blocks
    (comment markers inside quoted strings, e.g. a ``TITLE``, are kept)."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return "\n".join(_strip_line_comment(line) for line in text.splitlines())


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
    # ``particle=ion`` with an explicit mass/charge (what HELIX's own
    # exporter writes for H⁻ and deuterons): resolve by mass and charge.
    if str(beam_attrs.get("particle", "")).strip().lower() == "ion" \
            and "mass" in beam_attrs:
        mass_MeV = float(beam_attrs["mass"]) * 1000.0
        charge = int(round(float(beam_attrs.get("charge", 1))))   # MAD-X default
        best = None
        for cand in (PROTON, DEUTERON, H_MINUS):
            if cand.charge == charge and abs(cand.mass - mass_MeV) < 0.5:
                best = cand
        if best is not None:
            species = best
        else:
            species = H_MINUS if charge < 0 else PROTON
            warnings.append(
                f"BEAM ion mass={beam_attrs['mass']} GeV charge={charge} "
                f"matches no HELIX species — using {species.name} (same "
                "charge sign; the rigidity conversions use its mass)")
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


def _tilt_rad(attrs: dict, natural: float) -> float:
    """MAD-X ``tilt``: a value in radians, or a bare flag meaning the
    element's *natural* skew (π/4 for a quadrupole, π/2 for a bend)."""
    v = attrs.get("tilt", 0.0)
    if v is True:
        return natural
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _build_element(name: str, etype: str, attrs: dict, brho: float,
                   warnings: list, *, rbend_chord: bool = True
                   ) -> tuple[list, float]:
    """Map one MAD-X element to HELIX element(s).  Lengths returned in mm.

    ``rbend_chord`` — MAD-X ``OPTION, RBARC`` (default true): an RBEND's
    ``l`` is the straight chord and the arc is ``l·(θ/2)/sin(θ/2)``.
    ``OPTION, RBARC=false`` (LHC-style sequences) and the MAD8 importer
    pass ``False`` so ``l`` is taken as the arc length directly.
    """
    etype = etype.lower()
    l_mm = _gf(attrs, "l") * _M_TO_MM
    aperture_mm = _gf(attrs, "aperture") * _M_TO_MM   # MAD-X aperture rarely used

    if etype == "drift":
        return [Drift(name=name, length=l_mm, aperture=aperture_mm)], l_mm

    if etype == "quadrupole":
        # k1 [1/m²] -> gradient [T/m] = k1 · Bρ
        gradient = _gf(attrs, "k1") * brho
        # tilt [rad] about s -> HELIX skew angle [deg]
        skew_deg = _tilt_rad(attrs, math.pi / 4.0) * _RAD_TO_DEG
        return [Quadrupole(name=name, length=l_mm, gradient=gradient,
                           skew_angle=skew_deg, aperture=aperture_mm)], l_mm

    if etype in ("sbend", "rbend"):
        angle_rad = _gf(attrs, "angle")
        e1 = _gf(attrs, "e1")
        e2 = _gf(attrs, "e2")
        if etype == "rbend":
            # Rectangular bend: pole faces are parallel to the chord, so
            # each edge picks up an extra angle/2.  With MAD-X's default
            # RBARC=TRUE the given L is the STRAIGHT (chord) length and the
            # arc length is L·(θ/2)/sin(θ/2) — pinned against MAD-X's own
            # twiss s-positions in tests/io/test_madx_conventions.py.
            # (Before 2026-09-03 L was always taken as the arc, making
            # every MAD-X RBEND imported under the default option short.)
            e1 += angle_rad / 2.0
            e2 += angle_rad / 2.0
            if rbend_chord and abs(angle_rad) > 1e-12:
                l_mm = l_mm * (angle_rad / 2.0) / math.sin(angle_rad / 2.0)
        if abs(angle_rad) < 1e-12:
            # Zero-angle bend is just a drift — unless it carries k1,
            # which a straight HELIX Dipole cannot hold (ρ = ∞).
            if abs(_gf(attrs, "k1")) > 0.0:
                warnings.append(
                    f"bend {name!r}: angle=0 with k1={_gf(attrs, 'k1'):g} "
                    "— imported as a plain drift, the k1 focusing is lost")
            return [Drift(name=name, length=l_mm,
                          aperture=aperture_mm)], l_mm
        # tilt = ±π/2 is a vertical bend (HELIX hv=1): +π/2 bends towards
        # +y, −π/2 towards −y — the sign goes into the angle.  Any other
        # tilt has no HELIX counterpart.
        tilt = _tilt_rad(attrs, math.pi / 2.0)
        hv = 0
        if abs(abs(tilt) - math.pi / 2.0) < 1e-9:
            hv = 1
            if tilt < 0:
                angle_rad = -angle_rad
                e1, e2 = -e1, -e2
        elif abs(tilt) > 1e-12:
            warnings.append(
                f"bend {name!r}: tilt={tilt:g} rad is neither 0 nor ±π/2 "
                "— imported as a horizontal bend (hv=0)")
        angle_deg = angle_rad * _RAD_TO_DEG
        # TraceWin/HELIX convention: ρ > 0, the bend direction is the sign
        # of the angle (the MAD8 importer does the same).  Arc l = ρ·|θ|.
        rho_mm = abs(l_mm / angle_rad)
        # Combined-function bend: MAD-X k_x² = h² + k1, k_y² = -k1 (h = 1/ρ)
        # vs HELIX k_x² = (1-n)/ρ², k_y² = n/ρ²  ⇒  n = -k1·ρ²  (ρ in m).
        rho_m = rho_mm / _M_TO_MM
        field_index = -_gf(attrs, "k1") * rho_m * rho_m
        # Pole-face angles: MAD-X's edge focusing is h·tan(e) with the
        # SIGNED curvature h; HELIX's Edge is tan(β)/ρ with ρ > 0, so
        # β = sign(angle)·e (a rectangular magnet is defocusing in x for
        # either bend direction in both codes).
        sgn = 1.0 if angle_rad > 0 else -1.0
        beta1 = sgn * e1
        beta2 = sgn * e2
        # Fringe field: MAD-X hgap is the HALF gap [m], fint the fringe
        # integral (default 0; fintx overrides the exit face); HELIX Edge
        # takes the FULL gap [mm] and K1 (= fint).
        edge_kw_in: dict = {}
        edge_kw_out: dict = {}
        if "hgap" in attrs:
            gap_mm = 2.0 * _gf(attrs, "hgap") * _M_TO_MM
            fint = _gf(attrs, "fint") if "fint" in attrs else 0.0
            fintx = _gf(attrs, "fintx") if "fintx" in attrs else fint
            edge_kw_in = {"gap": gap_mm, "k1": fint}
            edge_kw_out = {"gap": gap_mm, "k1": fintx}
        elif "fint" in attrs or "fintx" in attrs:
            warnings.append(
                f"bend {name!r}: fint/fintx without hgap — the fringe "
                "integral has no effect without a gap in either code")
        edge_in = Edge(name=f"{name}_e1", pole_rotation=beta1 * _RAD_TO_DEG,
                       rho=rho_mm, aperture=aperture_mm, hv=hv, **edge_kw_in)
        body = Dipole(name=name, angle=angle_deg, rho=rho_mm,
                      field_index=field_index, aperture=aperture_mm, hv=hv)
        edge_out = Edge(name=f"{name}_e2", pole_rotation=beta2 * _RAD_TO_DEG,
                        rho=rho_mm, aperture=aperture_mm, hv=hv, **edge_kw_out)
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
        # HELIX's Multipole.tilt_deg rotates in the opposite sense to
        # MAD-X's tilt (see madx_writer) — negate so the optics match.
        tilt_raw = attrs.get("tilt", 0.0)
        tilt_deg = 0.0 if isinstance(tilt_raw, bool) else -float(tilt_raw or 0.0) * _RAD_TO_DEG
        mp = Multipole(name=name, knl=knl or [0.0], ksl=ksl or [0.0],
                       aperture=aperture_mm, tilt_deg=tilt_deg)
        return _drift_pad(mp, l_mm, name)

    if etype == "rfcavity":
        volt_MV = _gf(attrs, "volt")            # MAD-X volt is in MV
        lag = _gf(attrs, "lag")                 # phase lag in units of 2π
        freq_MHz = _gf(attrs, "freq")           # MAD-X freq is in MHz
        # MAD-X reference-particle energy gain is q·VOLT·sin(2π·lag)
        # (confirmed against MAD-X TRACK, tests/io/test_madx_conventions.py);
        # HELIX's RFGap gains q·V·T·cos(φs).  Hence φs = 360·lag − 90°.
        # Before 2026-09-03 this read ``phase = lag*360`` — a 90° error
        # that put every imported cavity at zero reference gain for lag=0.
        # MAD-X does not multiply the RF kick by the charge; HELIX does
        # (q·V·T·cos φs).  A volt whose sign disagrees with the charge is
        # a 180° phase shift in HELIX terms (the exporter writes
        # volt = sign(q)·V so the pair is an exact inverse).
        phase_deg = lag * 360.0 - 90.0
        if volt_MV * brho < 0:          # brho is charge-signed
            phase_deg += 180.0
        volt_MV = abs(volt_MV)
        gap = RFGap(name=name, voltage=volt_MV, phase=phase_deg,
                    frequency=freq_MHz, ttf=1.0)
        return _drift_pad(gap, l_mm, name)

    if etype in ("kicker", "hkicker", "vkicker", "tkicker"):
        # MAD-X kicks are deflection angles [rad] for the REFERENCE charge.
        # HELIX stores ∫B·dl [T·m] and kicks by sign(q)·(∫By dl)/Bρ in x
        # and sign(q)·(∫Bx dl)/Bρ in y (Steerer._kick_mrad), so folding
        # the signed rigidity in reproduces the MAD-X angle for both
        # proton and H⁻ beams.  `brho` here is already sign(q)·|Bρ|.
        if etype == "hkicker":
            hkick, vkick = _gf(attrs, "kick"), 0.0
        elif etype == "vkicker":
            hkick, vkick = 0.0, _gf(attrs, "kick")
        else:
            hkick, vkick = _gf(attrs, "hkick"), _gf(attrs, "vkick")
        st = Steerer(name=name, by_l=hkick * brho, bx_l=vkick * brho)
        return _drift_pad(st, l_mm, name)

    if etype in ("rcollimator", "ecollimator"):
        # Half-apertures [m] -> HELIX Aperture half-sizes [mm].
        xs = _gf(attrs, "xsize") * _M_TO_MM
        ys = _gf(attrs, "ysize") * _M_TO_MM
        if etype == "rcollimator":
            ap = Aperture(name=name, dx=xs, dy=ys, aperture_type=0)
        else:
            if ys > 0 and abs(ys - xs) > 1e-9:
                warnings.append(
                    f"collimator {name!r}: elliptical xsize≠ysize — HELIX "
                    "apertures are circular; using xsize as the radius")
            ap = Aperture(name=name, dx=xs, dy=xs, aperture_type=1)
        return _drift_pad(ap, l_mm, name)

    if etype == "matrix":
        # Explicit first-order map in MAD-X canonical coordinates.  The
        # basis change needs the LOCAL reference energy at the placement,
        # so the raw rm/kick data ride along and parse_madx converts
        # them in sequence order (see _convert_raw_matrices).
        R = np.eye(6)
        for a in range(6):
            for b in range(6):
                key = f"rm{a + 1}{b + 1}"
                if key in attrs:
                    R[a, b] = _gf(attrs, key)
        kick = np.array([_gf(attrs, f"kick{a + 1}") for a in range(6)])
        el = MatrixElement(name, np.eye(6), length=l_mm)
        el._madx_raw = (R, kick)
        return [el], l_mm

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
    text = _strip_comments(Path(filepath).read_text(encoding="latin-1"))
    statements = _split_statements(text)

    variables: dict = {}
    element_defs: dict = {}        # name -> (type, attrs)
    sequences: dict = {}          # name -> {"l": float, "members": [...]}
    beam_attrs: dict = {}
    rbarc = True            # MAD-X OPTION, RBARC (default true)
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
        elif cmd == "option":
            # Only RBARC changes the imported machine: it decides whether
            # an RBEND's ``l`` is the chord (true, default) or the arc.
            # Forms: ``rbarc``, ``-rbarc``, ``rbarc=true|false``.
            for part in _split_attributes(body):
                m_opt = re.fullmatch(r"\s*(-?)\s*rbarc\s*(?:=\s*(\w+))?\s*",
                                     part, flags=re.IGNORECASE)
                if m_opt:
                    val = m_opt.group(2)
                    if m_opt.group(1) == "-":
                        rbarc = False
                    elif val is None:
                        rbarc = True
                    else:
                        rbarc = val.lower() not in ("false", "0", "no")
        elif cmd in ("set", "select", "exec", "return",
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
        elems, total_mm = _build_element(ename, etype, attrs, brho, warnings,
                                         rbend_chord=rbarc)
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

    _convert_raw_matrices(lattice, reference, warnings)

    metadata = {
        "title": title or seq_name,
        "warnings": warnings,
        "reference": reference,
    }
    return lattice, metadata


def _convert_raw_matrices(lattice, reference, warnings) -> None:
    """Turn the raw MAD-X ``MATRIX`` data into HELIX-basis maps.

    MAD-X keeps one reference momentum p₀ for the whole sequence; a map
    that changes the energy is written about p₀ with the exit angles
    rescaled by p₀/p_out and the gain as ``kick6`` (pt).  The local
    energy at each placement is replayed exactly as the exporter does:
    RFCAVITY design gains (q·V·sin 2π·lag — the ``RFGap`` reference
    advance) and earlier ``MATRIX`` kick6 terms.
    """
    from linac_gen.tracking.longitudinal_coords import (
        matrix_from_madx, vector_from_madx)
    ref_local = reference.copy()
    p0c = reference.bg * reference.species.mass
    n_gain = 0
    lost_at = None
    for e in lattice.elements:
        raw = getattr(e, "_madx_raw", None)
        if raw is not None:
            R, kick = raw
            del e._madx_raw
            if lost_at is not None:
                # No local energy any more: keep the map about the BEAM
                # reference (a warning below says so).
                e.matrix = matrix_from_madx(R, reference)
                off = vector_from_madx(kick, reference)
                e.offset = off if np.any(off) else None
                continue
            r_in = ref_local.copy()
            dW = float(kick[5]) * p0c
            r_out = r_in.copy()
            try:
                r_out.w_kin = r_in.w_kin + dW
            except (ValueError, ZeroDivisionError, OverflowError):
                lost_at = e.name
                e.matrix = matrix_from_madx(R, reference)
                off = vector_from_madx(kick, reference)
                e.offset = off if np.any(off) else None
                continue
            e.matrix = matrix_from_madx(R, reference, r_in, r_out)
            off = vector_from_madx(kick, reference, r_out)
            e.offset = off if np.any(off) else None
            ref_local.w_kin = r_out.w_kin
            if dW != 0.0:
                n_gain += 1
        elif isinstance(e, RFGap) and lost_at is None:
            try:
                e.advance_ref(ref_local)
            except (ValueError, ZeroDivisionError, OverflowError):
                lost_at = e.name
    if lost_at is not None:
        warnings.append(
            f"reference particle lost at {lost_at!r}: the BEAM species "
            "decelerates below zero energy in this sequence's RF; any "
            "MATRIX element after that point is converted about the BEAM "
            "energy instead of the local one")
    if n_gain:
        warnings.append(
            f"{n_gain} MATRIX element(s) carry an energy kick (kick6): "
            "imported as a ΔW offset on every particle — HELIX's "
            "reference energy stays at the BEAM value through them")


__all__ = ["parse_madx"]
