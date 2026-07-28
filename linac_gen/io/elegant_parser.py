"""Elegant ``.lte`` lattice importer.

Reads an Elegant lattice-element file into a HELIX :class:`Lattice`.
Elegant is a MAD-family text format, so the strength/unit conversions and
thin-element padding are reused verbatim from :mod:`linac_gen.io.madx_parser`
(``_signed_brho``, ``_drift_pad``, ``_eval_expr``, the mm/deg constants).
Only the tokenizer dialect and the element keyword→HELIX mapping are
Elegant-specific.

Import-only; there is no Elegant exporter.

Elegant conventions honoured
----------------------------
* lengths ``l`` in metres, angles in radians, ``k1``/``k2`` in Elegant's
  (== MAD) normalized units, cavity ``volt`` in volts / ``phase`` in
  degrees (90° = crest) / ``freq`` in Hz;
* ``line=(...)`` beamlines with leading-``-`` reversal and ``N*elem``
  repetition; element templates (a type token that is itself a defined
  element name inherits its attributes); ``name[prop]=value`` overrides;
* the fortran-namelist comment/continuation rules (``!``/``#`` comments,
  ``&`` and trailing-``,`` line continuation).

Elements with no HELIX target degrade explicitly (never silently): CSR/LSC
drifts → plain ``Drift`` (+warning); ``charge``/``wake`` → ``Marker``
(beam data dropped, +warning); unknown types → ``Drift`` (+warning).
``ematrix`` (order 1) imports faithfully as a :class:`MatrixElement`.
"""
from __future__ import annotations

import math
import re

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.edge import Edge
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.marker import Marker
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.io.madx_parser import (
    _M_TO_MM, _RAD_TO_DEG, _brho, _signed_brho, _drift_pad, _eval_expr,
)

_SPECIES = {"proton": PROTON, "h-": H_MINUS, "hminus": H_MINUS,
            "deuteron": DEUTERON}

# Elegant's default cavity frequency when a cavity omits FREQ (matches
# Cheetah's from_elegant default of 500 MHz).
_DEFAULT_CAV_FREQ_HZ = 500e6


# ---------------------------------------------------------------------------
# Tokenizer (fortran-namelist dialect)
# ---------------------------------------------------------------------------
def _logical_statements(text: str) -> list[str]:
    """Split raw file text into logical statements.

    Handles ``!``/``#`` comments (to end of line), ``&`` continuation and
    trailing-``,``/``(`` continuation (an attribute list or beamline split
    over lines), then splits the joined stream on ``;`` and newlines.
    """
    joined: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].split("#", 1)[0].rstrip()
        if not line.strip():
            if buf:
                continue
            continue
        cont = line.rstrip().endswith(("&", ",", "("))
        line = line.rstrip()
        if line.endswith("&"):
            line = line[:-1]
        buf += (" " if buf else "") + line.strip()
        if not cont:
            joined.append(buf)
            buf = ""
    if buf:
        joined.append(buf)
    # each logical line may still carry multiple ';'-separated statements
    out: list[str] = []
    for lg in joined:
        for part in lg.split(";"):
            if part.strip():
                out.append(part.strip())
    return out


_NAME = r'(?:"[^"]+"|[^\s:=,\[\]]+)'
_ELEMENT_RE = re.compile(rf"^\s*({_NAME})\s*:\s*([A-Za-z][\w]*)\s*(?:,(.*))?$",
                         re.IGNORECASE)
_LINE_RE = re.compile(rf"^\s*({_NAME})\s*:\s*line\s*=\s*\((.*)\)\s*$",
                      re.IGNORECASE)
_OVERRIDE_RE = re.compile(rf"^\s*({_NAME})\s*\[\s*(\w+)\s*\]\s*=\s*(.+)$")
_VAR_RE = re.compile(rf"^\s*({_NAME})\s*=\s*(.+)$")
_USE_RE = re.compile(r"^\s*use\s*,\s*(\S+)\s*$", re.IGNORECASE)


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return s[1:-1]
    return s


def _split_attrs(body: str) -> dict:
    """Parse ``key=value, key="v", ...`` into a lowercase-keyed dict.

    Values are left as raw strings (evaluated lazily against variables).
    """
    attrs: dict = {}
    if not body or not body.strip():
        return attrs
    # split on commas that are not inside quotes
    for piece in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', body):
        piece = piece.strip()
        if not piece or "=" not in piece:
            continue
        key, val = piece.split("=", 1)
        attrs[key.strip().lower()] = val.strip()
    return attrs


# ---------------------------------------------------------------------------
# Parsed-file model
# ---------------------------------------------------------------------------
class _ElegantFile:
    def __init__(self):
        self.variables: dict = {}          # name(lower) -> float
        self.elements: dict = {}           # name(lower) -> (etype, attrs)
        self.lines: dict = {}              # name(lower) -> list[str members]
        self.order: list[str] = []         # element/line definition order
        self.use_root: str | None = None
        self.unresolved: set = set()       # expressions that would not eval

    # -- value evaluation ------------------------------------------------
    def num(self, attrs: dict, key: str, default: float = 0.0) -> float:
        if key not in attrs:
            return default
        v = self._eval(attrs[key])
        if not math.isfinite(v):
            # An unresolved variable/expression must not poison geometry
            # (a NaN length propagates to total_length): default it and
            # record it for a single summary warning.
            self.unresolved.add(str(attrs[key]))
            return default
        return v

    def _eval(self, expr: str) -> float:
        expr = _unquote(str(expr))
        try:
            return _eval_expr(expr, self.variables)
        except (ValueError, KeyError, TypeError):
            # Elegant RPN fallback (space-separated postfix).
            try:
                return _rpn_eval(expr, self.variables)
            except Exception:
                return float("nan")


def _rpn_eval(expr: str, variables: dict) -> float:
    """Minimal RPN evaluator for Elegant's native postfix notation."""
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
           "*": lambda a, b: a * b, "/": lambda a, b: a / b,
           "^": lambda a, b: a ** b}
    stack: list[float] = []
    for tok in expr.split():
        if tok in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[tok](a, b))
        elif tok == "sqrt":
            stack.append(math.sqrt(stack.pop()))
        else:
            key = tok.lower()
            stack.append(float(variables[key]) if key in variables
                         else float(tok))
    if len(stack) != 1:
        raise ValueError(f"bad RPN {expr!r}")
    return stack[0]


def _parse_file(text: str, warnings: list) -> _ElegantFile:
    ef = _ElegantFile()
    for stmt in _logical_statements(text):
        if stmt.startswith("%"):
            # Elegant RPN command line, e.g. "% 500 sto N_CSR_BIN" — stores
            # an RPN-evaluated value into a variable.  (Commented "!%"
            # lines are already stripped as comments.)
            toks = stmt[1:].split()
            if "sto" in toks:
                i = toks.index("sto")
                try:
                    ef.variables[toks[i + 1].lower()] = _rpn_eval(
                        " ".join(toks[:i]), ef.variables)
                except Exception:
                    warnings.append(f"could not evaluate RPN store {stmt!r}")
            continue
        m = _USE_RE.match(stmt)
        if m:
            ef.use_root = _unquote(m.group(1)).lower()
            continue
        m = _LINE_RE.match(stmt)
        if m:
            name = _unquote(m.group(1)).lower()
            members = [_unquote(t).lower() for t in
                       re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', m.group(2))
                       if t.strip()]
            ef.lines[name] = members
            ef.order.append(name)
            continue
        m = _OVERRIDE_RE.match(stmt)
        if m:
            name = _unquote(m.group(1)).lower()
            prop, val = m.group(2).strip().lower(), m.group(3).strip()
            if name in ef.elements:
                ef.elements[name][1][prop] = val
            else:
                warnings.append(f"override on unknown element {name!r} ignored")
            continue
        m = _ELEMENT_RE.match(stmt)
        if m:
            name = _unquote(m.group(1)).lower()
            etype = m.group(2).lower()
            attrs = _split_attrs(m.group(3) or "")
            # element template: type token is a previously-defined element
            if etype in ef.elements:
                base_type, base_attrs = ef.elements[etype]
                merged = dict(base_attrs)
                merged.update(attrs)
                ef.elements[name] = (base_type, merged)
            else:
                ef.elements[name] = (etype, attrs)
            ef.order.append(name)
            continue
        m = _VAR_RE.match(stmt)
        if m:
            name = _unquote(m.group(1)).lower()
            try:
                ef.variables[name] = _eval_expr(m.group(2), ef.variables)
            except Exception:
                warnings.append(f"could not evaluate variable {name!r}")
            continue
        warnings.append(f"unrecognized statement skipped: {stmt!r}")
    return ef


# ---------------------------------------------------------------------------
# Beamline expansion  (name -> ordered list of element names, signs handled)
# ---------------------------------------------------------------------------
_REP_RE = re.compile(r"^(\d+)\s*\*\s*(-?)(.+)$")


def _expand(name: str, ef: _ElegantFile, stack: tuple = ()) -> list[str]:
    """Expand a line/element name to a flat list of element names.

    Supports ``-name`` reversal and ``N*name`` repetition.  Elements
    (leaves) return ``[name]``.
    """
    reverse = False
    if name.startswith("-"):
        reverse, name = True, name[1:]
    rep = 1
    m = _REP_RE.match(name)
    if m:
        rep = int(m.group(1))
        reverse = reverse ^ bool(m.group(2))
        name = m.group(3)
    name = name.strip().lower()
    if name in stack:
        raise ValueError(f"cyclic beamline reference at {name!r}")
    if name in ef.lines:
        seq: list[str] = []
        members = ef.lines[name]
        for mem in members:
            seq.extend(_expand(mem, ef, stack + (name,)))
        if reverse:
            seq = seq[::-1]
        return seq * rep
    # leaf element
    return [name] * rep


# ---------------------------------------------------------------------------
# Element mapping  (Elegant m/rad/V/Hz -> HELIX mm/deg/MV/MHz)
# ---------------------------------------------------------------------------
def _build_elegant_element(name, etype, ef, ref, sbrho, brho, warnings):
    """Map one Elegant element to HELIX element(s); lengths in mm."""
    def N(k, d=0.0):
        return ef.num(ef.elements[name][1], k, d)
    attrs = ef.elements[name][1]
    l_mm = N("l") * _M_TO_MM

    if etype in ("drift", "drif"):
        return [Drift(name=name, length=l_mm)], l_mm
    if etype in ("csrdrift", "csrdrif", "lscdrift", "lscdrif"):
        warnings.append(f"{name}: {etype} collective effect dropped -> drift")
        return [Drift(name=name, length=l_mm)], l_mm

    if etype in ("quad", "quadrupole", "kquad"):
        gradient = N("k1") * sbrho
        skew = N("tilt") * _RAD_TO_DEG
        return [Quadrupole(name=name, length=l_mm, gradient=gradient,
                           skew_angle=skew)], l_mm

    if etype in ("sext", "sextupole", "ksext"):
        k2l = N("k2") * N("l")
        return _drift_pad(Multipole(name=name, knl=[0.0, 0.0, k2l]),
                          l_mm, name)

    if etype in ("mult", "multipole"):
        # Elegant MULT: integrated strength KNL at pole ORDER
        # (0=dipole, 1=quad, 2=sext, ...) — HELIX knl uses the same index.
        order = int(N("order", 0))
        knl = [0.0] * order + [N("knl")]
        return _drift_pad(Multipole(name=name, knl=knl), l_mm, name)

    if etype in ("sole", "solenoid"):
        field = N("ks") * brho          # ks normalized -> B [T]
        return [Solenoid(name=name, length=l_mm, field=field)], l_mm

    if etype in ("sben", "sbend", "csbend", "csrcsbend", "csrcsben"):
        return _build_bend(name, ef, l_mm, ref, sbrho, warnings, rect=False)
    if etype in ("rben", "rbend"):
        return _build_bend(name, ef, l_mm, ref, sbrho, warnings, rect=True)

    if etype in ("rfca", "rfcw"):
        volt_MV = N("volt") * 1e-6
        phase_deg = N("phase") - 90.0          # Elegant 90deg = crest
        freq_MHz = N("freq", _DEFAULT_CAV_FREQ_HZ) * 1e-6
        return _drift_pad(RFGap(name=name, voltage=volt_MV, phase=phase_deg,
                                frequency=freq_MHz), l_mm, name)

    if etype in ("hkick", "hkic"):
        return _drift_pad(Steerer(name=name, by_l=N("kick") * sbrho),
                          l_mm, name)
    if etype in ("vkick", "vkic"):
        return _drift_pad(Steerer(name=name, bx_l=N("kick") * sbrho),
                          l_mm, name)
    if etype in ("kick", "kicker"):
        return _drift_pad(Steerer(name=name, by_l=N("hkick") * sbrho,
                                  bx_l=N("vkick") * sbrho), l_mm, name)

    if etype in ("mark", "marker", "watch"):
        return [Marker(name=name)], 0.0
    if etype in ("moni", "monitor"):
        return _drift_pad(Marker(name=name, is_bpm=True), l_mm, name)

    if etype in ("ecol", "rcol"):
        ap = Aperture(name=f"{name}_ap",
                      aperture_type=(1 if etype == "ecol" else 0),
                      dx=N("x_max", 0.0) * _M_TO_MM,
                      dy=N("y_max", 0.0) * _M_TO_MM)
        elems, _ = _drift_pad(ap, l_mm, name)
        return elems, l_mm

    if etype == "ematrix":
        return _build_ematrix(name, ef, l_mm, warnings)

    if etype in ("charge", "wake"):
        warnings.append(f"{name}: {etype} beam properties not imported")
        return [Marker(name=name)], 0.0

    warnings.append(f"{name}: unsupported Elegant type {etype!r} -> drift")
    return [Drift(name=name, length=l_mm)], l_mm


def _build_bend(name, ef, l_mm, ref, sbrho, warnings, rect):
    def N(k, d=0.0):
        return ef.num(ef.elements[name][1], k, d)
    angle = N("angle")
    if abs(angle) < 1e-12:
        return [Drift(name=name, length=l_mm)], l_mm
    e1, e2 = N("e1"), N("e2")
    if rect:                                    # RBEND pole faces || chord
        e1 += angle / 2.0
        e2 += angle / 2.0
    rho_mm = l_mm / angle                        # arc = rho * angle
    rho_m = rho_mm * 1e-3
    # combined-function K1 -> HELIX field index N = -K1 * rho^2
    field_index = -N("k1") * rho_m ** 2
    gap_mm = 2.0 * N("hgap") * _M_TO_MM
    fint = N("fint", 0.45) if "fint" in ef.elements[name][1] else 0.45
    edge_in = Edge(name=f"{name}_e1", pole_rotation=e1 * _RAD_TO_DEG,
                   rho=rho_mm, gap=gap_mm, k1=fint)
    body = Dipole(name=name, angle=angle * _RAD_TO_DEG, rho=rho_mm,
                  field_index=field_index)
    edge_out = Edge(name=f"{name}_e2", pole_rotation=e2 * _RAD_TO_DEG,
                    rho=rho_mm, gap=gap_mm, k1=fint)
    return [edge_in, body, edge_out], body.length


def _build_ematrix(name, ef, l_mm, warnings):
    import numpy as np
    attrs = ef.elements[name][1]
    order = int(ef.num(attrs, "order", 1))
    if order != 1:
        warnings.append(f"{name}: EMATRIX order={order} unsupported "
                        "(only linear order 1) -> marker")
        return [Marker(name=name)], 0.0
    M = np.eye(6)
    offset = np.zeros(6)
    has_offset = False
    for i in range(1, 7):
        for j in range(1, 7):
            key = f"r{i}{j}"
            if key in attrs:
                M[i - 1, j - 1] = ef.num(attrs, key)
        ckey = f"c{i}"
        if ckey in attrs:
            offset[i - 1] = ef.num(attrs, ckey)
            has_offset = True
    el = MatrixElement(name=name, matrix=M, length=l_mm,
                       offset=offset if has_offset else None)
    return [el], l_mm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def parse_elegant(filepath, name: str | None = None,
                  species: str = "H-", w_kin: float = 1000.0,
                  frequency: float = 352.21):
    """Parse an Elegant ``.lte`` file into ``(Lattice, metadata)``.

    Parameters
    ----------
    filepath : str | Path
    name : optional root beamline name (else the last-defined ``line``).
    species, w_kin, frequency : reference particle — Elegant ``.lte`` files
        carry no beam energy (it lives in the run file), so supply it here;
        the strength conversion is self-consistent regardless of the value
        chosen (``k1`` is preserved through gradient = sign(q)·Bρ·k1).
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    warnings: list[str] = []
    ef = _parse_file(text, warnings)

    sp = _SPECIES.get(str(species).lower(), H_MINUS)
    ref = ReferenceParticle(species=sp, w_kin=float(w_kin),
                            frequency=float(frequency))
    brho = _brho(ref)
    sbrho = _signed_brho(ref)

    # choose the root beamline
    root = (name or ef.use_root
            or (ef.order[-1] if ef.lines and ef.order[-1] in ef.lines
                else next(reversed(ef.lines)) if ef.lines else None))
    if root is None:
        raise ValueError("no beamline (line=(...)) found in Elegant file")
    root = _unquote(str(root)).lower()
    if root not in ef.lines:
        raise ValueError(f"root beamline {root!r} not defined")

    lattice = Lattice()
    for elem_name in _expand(root, ef):
        if elem_name not in ef.elements:
            warnings.append(f"undefined element {elem_name!r} in line -> skipped")
            continue
        etype = ef.elements[elem_name][0]
        elems, _ = _build_elegant_element(elem_name, etype, ef, ref,
                                          sbrho, brho, warnings)
        for e in elems:
            lattice.add(e)

    if ef.unresolved:
        sample = sorted(ef.unresolved)[:5]
        warnings.append(
            f"{len(ef.unresolved)} expression(s) referenced undefined "
            f"variables and were defaulted to 0 (e.g. {sample})")

    meta = {"title": root, "warnings": warnings, "reference": ref}
    return lattice, meta
