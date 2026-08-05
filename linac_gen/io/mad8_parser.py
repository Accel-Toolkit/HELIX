"""In-house MAD8 flat-file (``.lat`` / ``.flat``) lattice importer.

Reads a MAD8-style flat lattice (SAVELINE / hand-written LINE format, the
dialect used by the PIP-II BTL exports) and builds a HELIX
:class:`~linac_gen.core.lattice.Lattice`, mirroring the
``(Lattice, metadata)`` return shape of :func:`parse_tracewin` /
:func:`parse_madx`.  The TraceWin and MAD-X parsers are **not** touched —
this is a parallel importer that reuses :mod:`linac_gen.io.madx_parser`'s
safe expression evaluator and element factory.

Supported subset
----------------
* ``!`` line comments; ``&`` end-of-line continuations.
* Deferred parameters ``name := expr`` (and plain ``name = expr``) with
  references to other parameters and ``NAME[ATTR]`` element-attribute
  references, resolved lazily with cycle detection.  Identifiers
  containing apostrophes (MAD8 ``QX' := …``) are stored but skipped with
  a warning if anything tries to use them in arithmetic.
* Element definitions ``name: TYPE, attr=expr, …`` — mapped through
  :func:`madx_parser._build_element` (drift, quadrupole, sbend/rbend →
  Edge+Dipole+Edge, solenoid, sextupole, multipole, rfcavity, marker,
  monitors).  MAD8 additions handled here:
    - ``KICKER/HKICKER/VKICKER`` → Marker + full-length body Drift
      (geometry preserved; a non-zero kick warns — orbit kicks are not
      imported).
    - plain ``MONITOR`` → Marker (not a BPM); ``H/VMONITOR`` → BPM marker.
    - ``TILT`` on a quadrupole → ``Quadrupole.skew_angle`` (degrees).
* ``name: LINE = (A, B, -C, 2*D)`` with recursive expansion, reversal
  and integer repetition.  The root line is the unreferenced LINE with
  the largest expansion (ambiguity warns; ``strict=True`` raises).

Rigidity / charge convention
----------------------------
MAD strengths are normalized (K1 = (q/p)·∂B_y/∂x); HELIX stores the
lab-frame gradient and applies the beam's charge at track time, so the
conversion is G = sign(q)·K1·|Bρ| (see ``madx_parser._signed_brho``).
Bρ resolution order: the ``brho=`` argument → a ``BRHO := …`` file
parameter → a ``BEAM`` statement → hard ``ValueError`` (no silent
default: silently mis-scaled gradients are the worst failure mode).

Periodicity
-----------
The LINE hierarchy declares the machine's cell structure — information a
flat TraceWin file loses.  With ``auto_periods=True`` (default) the
importer identifies FODO-type cells (LINE-valued grandchildren of the
root), groups consecutive cells with identical transport signatures and
identical significant-element counts, and declares them as
``LATTICE n1 0`` / ``LATTICE_END`` marker pairs that
:func:`linac_gen.analysis.period_detect.detect_periods` picks up
unchanged.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import DEUTERON, H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.madx_parser import (
    _CONSTANTS,
    _M_TO_MM,
    _RAD_TO_DEG,
    _brho,
    _build_element,
    _eval_expr,
    _reference_from_beam,
    _signed_brho,
    _species_from_name,
    _split_attributes,
)

_IDENT = r"[A-Za-z_][\w'.]*"
_KICKER_TYPES = ("kicker", "hkicker", "vkicker")


# ---------------------------------------------------------------------------
# Front-end: logical lines → params / element defs / LINE defs
# ---------------------------------------------------------------------------

def _logical_lines(text: str) -> list[str]:
    """Strip ``!`` comments and join ``&`` continuations."""
    out, buf = [], ""
    for raw in text.splitlines():
        raw = raw.split("!", 1)[0].rstrip()
        if not raw.strip() and not buf:
            continue
        buf += raw
        if buf.rstrip().endswith("&"):
            buf = buf.rstrip()[:-1]
            continue
        if buf.strip():
            out.append(buf.strip())
        buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


class _Mad8File:
    """Parsed statements + lazy, memoized expression resolution."""

    def __init__(self, text: str, warnings: list):
        self.params: dict[str, str] = {}       # UPPER name -> raw expr
        self.elems: dict[str, tuple[str, dict]] = {}   # name -> (type, raw attrs)
        self.lines: dict[str, list[str]] = {}   # name -> entries
        self.beam_attrs: dict = {}
        self.warnings = warnings
        self._memo: dict[str, float] = {}
        self._resolving: set[str] = set()
        self._parse(text)

    # -- statement classification ------------------------------------
    def _parse(self, text: str) -> None:
        for ln in _logical_lines(text):
            up = ln.upper()
            if up.startswith("RETURN"):
                break
            if up.startswith(("TITLE", "USE,", "USE ", "SAVELINE")):
                continue
            if up.startswith("BEAM"):
                body = ln.partition(",")[2]
                self.beam_attrs = self._raw_attrs(body)
                continue
            m = re.match(rf"^({_IDENT})\s*:?=\s*(.+)$", ln)
            if m and ":" not in m.group(1) and not re.match(
                    rf"^{_IDENT}\s*:\s*[A-Za-z]", ln):
                self.params[m.group(1).upper()] = m.group(2).strip()
                continue
            m = re.match(rf"^({_IDENT})\s*:\s*LINE\s*=\s*\((.*)\)\s*$",
                         ln, re.IGNORECASE)
            if m:
                self.lines[m.group(1).upper()] = [
                    e.strip().upper() for e in m.group(2).split(",")
                    if e.strip()]
                continue
            m = re.match(rf"^({_IDENT})\s*:\s*([A-Za-z]+)\s*(?:,\s*(.*))?$", ln)
            if m:
                self.elems[m.group(1).upper()] = (
                    m.group(2).lower(), self._raw_attrs(m.group(3) or ""))
                continue
            self.warnings.append(f"MAD8: unrecognised statement skipped: "
                                 f"{ln[:70]!r}")

    @staticmethod
    def _raw_attrs(body: str) -> dict:
        """``attr=expr`` pairs kept as RAW strings (evaluated lazily —
        eager evaluation would silently coerce unresolved refs to 0)."""
        attrs: dict = {}
        for part in _split_attributes(body):
            if not part:
                continue
            if "=" not in part:
                attrs[part.strip().lower()] = True
                continue
            k, _, v = part.partition("=")
            attrs[k.rstrip(":").strip().lower()] = v.strip()
        return attrs

    # -- lazy numeric resolution ---------------------------------------
    def resolve(self, expr: str) -> float:
        """Evaluate a MAD8 expression: numbers, arithmetic, parameter
        references, NAME[ATTR] element-attribute references."""
        e = re.sub(rf"({_IDENT})\[(\w+)\]",
                   lambda m: repr(self.attr_val(m.group(1).upper(),
                                                m.group(2).lower())),
                   str(expr))

        def sub_name(m):
            tok = m.group(0)
            if tok.lower() in _CONSTANTS:
                return tok            # let _eval_expr supply pi, clight, …
            return repr(self.param_val(tok.upper()))
        e = re.sub(rf"(?<![\d.]){_IDENT}", sub_name, e)
        return _eval_expr(e, {})

    def param_val(self, name: str) -> float:
        name = name.upper()
        if name in self._memo:
            return self._memo[name]
        if name not in self.params:
            raise ValueError(f"MAD8: unknown identifier {name!r}")
        if name in self._resolving:
            raise ValueError(f"MAD8: circular parameter reference {name!r}")
        self._resolving.add(name)
        try:
            v = self.resolve(self.params[name])
        finally:
            self._resolving.discard(name)
        self._memo[name] = v
        return v

    def attr_val(self, ename: str, attr: str) -> float:
        if ename not in self.elems:
            raise ValueError(f"MAD8: {ename}[{attr}] — unknown element")
        raw = self.elems[ename][1].get(attr)
        if raw is None or raw is True:
            return 0.0
        return self.resolve(raw)

    def numeric_attrs(self, ename: str, strict: bool) -> dict:
        """Element attributes with every value resolved to a float.
        Unresolvable values warn (or raise when strict) instead of the
        silent-0.0 coercion the MAD-X `_gf` path would apply."""
        etype, raw = self.elems[ename]
        out: dict = {}
        for k, v in raw.items():
            if v is True:
                out[k] = True
                continue
            try:
                out[k] = self.resolve(v)
            except ValueError as exc:
                msg = (f"MAD8: {ename}.{k} = {v!r} could not be "
                       f"evaluated ({exc})")
                if strict:
                    raise ValueError(msg) from exc
                self.warnings.append(msg + " — treated as 0")
                out[k] = 0.0
        return out


# ---------------------------------------------------------------------------
# LINE expansion (with provenance for the periodicity pass)
# ---------------------------------------------------------------------------

_REP = re.compile(r"^(\d+)\s*\*\s*(-?)(\w[\w'.]*)$")


def _expand(f: _Mad8File, lname: str, out: list[str],
            reverse: bool = False) -> None:
    entries = f.lines[lname]
    if reverse:
        entries = list(reversed(entries))
    for entry in entries:
        rep, neg, name = 1, False, entry
        m = _REP.match(entry)
        if m:
            rep, neg, name = int(m.group(1)), m.group(2) == "-", m.group(3)
        elif entry.startswith("-"):
            neg, name = True, entry[1:].strip()
        name = name.upper()
        for _ in range(rep):
            if name in f.lines:
                _expand(f, name, out, reverse=(neg ^ reverse))
            else:
                out.append(name)


def _root_line(f: _Mad8File, strict: bool) -> str:
    referenced: set[str] = set()
    for entries in f.lines.values():
        for e in entries:
            n = e.lstrip("-").upper()
            m = _REP.match(e)
            if m:
                n = m.group(3).upper()
            referenced.add(n)
    roots = [n for n in f.lines if n not in referenced]
    if not roots:
        raise ValueError("MAD8: no top-level LINE found (all lines are "
                         "referenced by other lines)")
    if len(roots) == 1:
        return roots[0]
    # Several unreferenced lines (saved sub-lines are common) — take the
    # one with the largest expansion.
    sized = []
    for r in roots:
        flat: list[str] = []
        _expand(f, r, flat)
        sized.append((len(flat), r))
    sized.sort(reverse=True)
    msg = (f"MAD8: {len(roots)} top-level LINEs; using the largest, "
           f"{sized[0][1]!r} ({sized[0][0]} entries); others: "
           f"{[r for _, r in sized[1:]]}")
    if strict:
        raise ValueError(msg)
    f.warnings.append(msg)
    return sized[0][1]


# ---------------------------------------------------------------------------
# Element construction (wraps madx_parser._build_element)
# ---------------------------------------------------------------------------

def _build_mad8_element(f: _Mad8File, name: str, brho_signed: float,
                        strict: bool) -> tuple[list, float]:
    etype, _raw = f.elems[name]
    attrs = f.numeric_attrs(name, strict)
    l_mm = float(attrs.get("l", 0.0) or 0.0) * _M_TO_MM

    if etype in _KICKER_TYPES:
        # Marker (instrument position) + full-length body drift so the
        # geometry is exact.  Orbit kicks are not imported — warn when
        # a file carries non-zero ones.
        kicks = [abs(float(attrs.get(k, 0.0) or 0.0))
                 for k in ("kick", "hkick", "vkick")]
        if max(kicks, default=0.0) > 0.0:
            f.warnings.append(f"MAD8: {name} carries a non-zero kick — "
                              "orbit kicks are not imported (set to 0)")
        elems: list = [Marker(name=name)]
        if l_mm != 0.0:
            elems.append(Drift(name=f"{name}_body", length=l_mm))
        return elems, l_mm

    if etype == "monitor":
        # Plain MONITOR = generic instrument (ion pump, collimator flag,
        # …), not a beam-position monitor.  H/VMONITOR stay BPMs via the
        # madx factory below.
        elems = [Marker(name=name, is_bpm=False)]
        if l_mm != 0.0:
            elems.append(Drift(name=f"{name}_body", length=l_mm))
        return elems, l_mm

    elems, total_mm = _build_element(name, etype, attrs, brho_signed,
                                     f.warnings)
    if etype == "quadrupole" and attrs.get("tilt"):
        for el in elems:
            if isinstance(el, Quadrupole):
                el.skew_angle = float(attrs["tilt"]) * _RAD_TO_DEG
    if etype in ("sbend", "rbend"):
        # MAD8 encodes vertical bends as TILT=±π/2; _build_element drops
        # TILT, and its ρ = L/θ carries the angle's sign.  HELIX/TraceWin
        # convention is ρ > 0, sign in the angle, plane in ``hv`` — the
        # vertical edge matrix genuinely differs if either is left as-is.
        tilt = float(attrs.get("tilt", 0.0) or 0.0)
        vertical = abs(abs(tilt) - math.pi / 2.0) < 1e-6
        if not vertical and abs(tilt) > 1e-9:
            f.warnings.append(
                f"MAD8: {name} has TILT={tilt:.4f} rad — only 0 (horizontal)"
                " and ±π/2 (vertical) bends are supported; tilt dropped")
        for el in elems:
            if isinstance(el, (Dipole, Edge)):
                el.rho = abs(float(el.rho))
                if vertical:
                    el.hv = 1
    return elems, total_mm


# ---------------------------------------------------------------------------
# Rigidity resolution
# ---------------------------------------------------------------------------

def _resolve_reference(f: _Mad8File, brho_arg, species_name: str):
    """(ReferenceParticle, |brho|) from arg → BRHO param → error.
    (The BEAM-statement path is handled by the caller.)"""
    species = _species_from_name(species_name) or H_MINUS
    brho_abs = None
    if brho_arg is not None:
        brho_abs = abs(float(brho_arg))
    elif "BRHO" in f.params:
        brho_abs = abs(f.param_val("BRHO"))
    if brho_abs is None:
        raise ValueError(
            "MAD8: no rigidity available — supply it as parse_mad8(..., "
            "brho=<T·m>), or declare `BRHO := <T·m>` in the file, or add "
            "a `BEAM, PARTICLE=..., ENERGY=...` statement.  Refusing to "
            "guess: a wrong Bρ silently mis-scales every magnet.")
    # kinetic energy back from Bρ:  βγ = Bρ / (m·1e6/c)
    from linac_gen.core.constants import C_LIGHT
    bg = brho_abs * C_LIGHT / (species.mass * 1e6)
    gamma = math.sqrt(1.0 + bg * bg)
    w_kin = species.mass * (gamma - 1.0)
    ref = ReferenceParticle(species=species, w_kin=w_kin, frequency=162.5)
    return ref, brho_abs


# ---------------------------------------------------------------------------
# Auto-declared periodicity
# ---------------------------------------------------------------------------

_SIG_TOL_MM = 2e-3            # 2 µm


def _cell_signature(elements: list) -> tuple:
    """Transport signature: consecutive drifts merged, zero-length
    non-transport elements skipped."""
    sig: list[list] = []
    for el in elements:
        if isinstance(el, Drift):
            L = float(el.length)
            if sig and sig[-1][0] == "D":
                sig[-1][1] += L
            else:
                sig.append(["D", L])
        elif isinstance(el, Quadrupole):
            sig.append(["Q", float(el.length), float(el.gradient)])
        elif isinstance(el, Dipole):
            sig.append(["B", float(el.length), float(el.angle)])
        elif isinstance(el, Edge):
            sig.append(["E", float(el.pole_rotation), float(el.rho)])
        # markers etc: no transport
    return tuple(tuple(s) for s in sig)


def _sig_equal(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x[0] != y[0] or any(abs(p - q) > _SIG_TOL_MM
                               for p, q in zip(x[1:], y[1:])):
            return False
    return True


def _declare_periods(elements: list, cells: list[dict],
                     warnings: list) -> list[dict]:
    """Insert LATTICE/LATTICE_END marker pairs around maximal consecutive
    runs of cells with identical signatures and significant counts.
    ``cells``: [{name, parent, i0, i1}] with element index ranges.
    Returns the list of declared periods (for metadata)."""
    from linac_gen.analysis.period_detect import _is_significant

    for c in cells:
        seg = elements[c["i0"]:c["i1"]]
        c["sig"] = _cell_signature(seg)
        c["n_sig"] = sum(1 for el in seg if _is_significant(el))

    declared: list[dict] = []
    by_parent: dict[str, list[dict]] = {}
    for c in cells:
        by_parent.setdefault(c["parent"], []).append(c)

    for parent, group in by_parent.items():
        group.sort(key=lambda c: c["i0"])
        # a cell is bracketable iff its signature repeats within the
        # parent AND it actually focuses (a quad or dipole in the cell) —
        # matching wire-scanner wrapper LINEs are not transport periods
        def repeats(c):
            if not any(item[0] in ("Q", "B") for item in c["sig"]):
                return False
            return sum(1 for o in group if _sig_equal(c["sig"], o["sig"])) >= 2
        runs: list[list[dict]] = []
        for c in group:
            if not repeats(c):
                continue
            if (runs and runs[-1][-1]["i1"] == c["i0"]
                    and _sig_equal(runs[-1][-1]["sig"], c["sig"])
                    and runs[-1][-1]["n_sig"] == c["n_sig"]):
                runs[-1].append(c)
            else:
                runs.append([c])
        for run in runs:
            declared.append({
                "cells": [c["name"] for c in run],
                "parent": parent,
                "n_repeats": len(run),
                "n_sig": run[0]["n_sig"],
                "i0": run[0]["i0"],
                "i1": run[-1]["i1"],
            })

    # insert bracket markers back-to-front so indices stay valid
    for k, d in enumerate(sorted(declared, key=lambda d: d["i0"],
                                 reverse=True)):
        open_m = Marker(name=f"LATTICE_{len(declared) - k}")
        open_m.lattice_card_args = [float(d["n_sig"]), 0.0]
        close_m = Marker(name=f"LATTICE_END_{len(declared) - k}")
        elements.insert(d["i1"], close_m)
        elements.insert(d["i0"], open_m)
    return sorted(declared, key=lambda d: d["i0"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_mad8(filepath: str, strict: bool = False, brho=None,
               species: str = "H-",
               auto_periods: bool = True) -> tuple[Lattice, dict]:
    """Parse a MAD8 flat file into a HELIX Lattice.

    Returns ``(lattice, metadata)`` with metadata keys ``"title"``,
    ``"warnings"``, ``"reference"`` (ReferenceParticle used for the
    strength conversion) and ``"periods"`` (auto-declared LATTICE
    brackets).  See the module docstring for the supported subset and
    the Bρ / charge-sign conventions.
    """
    path = Path(filepath)
    warnings: list[str] = []
    f = _Mad8File(path.read_text(encoding="latin-1", errors="replace"), warnings)

    # --- reference / rigidity -----------------------------------------
    if f.beam_attrs:
        num_beam = {}
        for k, v in f.beam_attrs.items():
            if v is True or k == "particle":
                num_beam[k] = v
            else:
                try:
                    num_beam[k] = f.resolve(v)
                except ValueError:
                    num_beam[k] = v
        ref = _reference_from_beam(num_beam, 162.5, warnings)
        if brho is not None and abs(abs(float(brho)) - _brho(ref)) > 1e-3:
            warnings.append(
                f"MAD8: brho argument {float(brho):.4f} T·m disagrees with "
                f"the file BEAM statement ({_brho(ref):.4f} T·m) — using "
                "the BEAM statement")
    else:
        ref, _ = _resolve_reference(f, brho, species)
    brho_signed = _signed_brho(ref)
    warnings.append(
        f"MAD8 import: strengths converted with {ref.species.name} at "
        f"W = {ref.w_kin:.2f} MeV (Bρ = {_brho(ref):.4f} T·m, "
        f"G = sign(q)·K1·Bρ) — set the Beam tab to match before running.")

    # --- expand the root line with cell provenance ---------------------
    # Depth 0 = children of the root (machine sections); depth 1 = their
    # LINE-valued entries = candidate periodic cells.  Reversal (-NAME)
    # and repetition (N*NAME) are honoured at every depth.
    root = _root_line(f, strict)
    elements: list = []
    cells: list[dict] = []

    def _entry_parts(entry: str) -> tuple[int, bool, str]:
        m = _REP.match(entry)
        if m:
            return int(m.group(1)), m.group(2) == "-", m.group(3).upper()
        if entry.startswith("-"):
            return 1, True, entry[1:].strip().upper()
        return 1, False, entry.upper()

    def emit(name: str) -> None:
        if name not in f.elems:
            msg = f"MAD8: {name!r} referenced but never defined"
            if strict:
                raise ValueError(msg)
            warnings.append(msg + " — skipped")
            return
        elems, _L = _build_mad8_element(f, name, brho_signed, strict)
        elements.extend(elems)

    def walk(lname: str, depth: int, parent: str, reverse: bool) -> None:
        entries = f.lines[lname]
        for entry in (reversed(entries) if reverse else entries):
            rep, neg, name = _entry_parts(entry)
            eff_rev = neg ^ reverse
            for _ in range(rep):
                if name in f.lines:
                    if depth == 1:
                        i0 = len(elements)
                        walk(name, depth + 1, parent, eff_rev)
                        cells.append({"name": name, "parent": parent,
                                      "i0": i0, "i1": len(elements)})
                    else:
                        walk(name, depth + 1,
                             name if depth == 0 else parent, eff_rev)
                else:
                    emit(name)

    walk(root, 0, root, False)

    periods: list[dict] = []
    if auto_periods and cells:
        periods = _declare_periods(elements, cells, warnings)

    lattice = Lattice()
    for el in elements:
        lattice.add(el)

    metadata = {
        "title": root,
        "warnings": warnings,
        "reference": ref,
        "periods": periods,
    }
    return lattice, metadata


__all__ = ["parse_mad8"]
