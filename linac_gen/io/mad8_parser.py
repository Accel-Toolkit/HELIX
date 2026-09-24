"""In-house MAD8 lattice importer (``.lat`` / ``.flat`` / ``.mad8``).

Reads a MAD8 lattice — SAVELINE flat files, hand-written decks, the
PIP-II BTL/BAL exports, Synergia / PyORBIT / Bmad-written MAD8 — and
builds a HELIX :class:`~linac_gen.core.lattice.Lattice`, mirroring the
``(Lattice, metadata)`` return shape of :func:`parse_tracewin` /
:func:`parse_madx`.  The TraceWin and MAD-X parsers are **not** touched —
this is a parallel importer that reuses :mod:`linac_gen.io.madx_parser`'s
element factory (with its own MAD8 expression evaluator).

Language covered
----------------
* ``!`` comments; ``&`` continuation (anything after the ``&`` on that
  line is ignored, as in MAD8); ``;`` statement separators;
  ``COMMENT`` … ``ENDCOMMENT`` blocks; ``CALL, FILENAME=…`` (resolved
  against the calling file's folder); ``RETURN`` / ``STOP``.
* Parameters ``name := expr`` / ``name = expr`` /
  ``name: CONSTANT = expr`` / ``SET, name, expr``, resolved lazily with
  cycle detection (the last definition wins; a redefinition warns).
  Expressions: + − * / ^ (**), unary signs, Fortran ``D`` exponents,
  ``NAME[ATTR]`` element (and ``BEAM[…]``) attribute references, the
  MAD8 functions SQRT LOG EXP SIN COS TAN ASIN ABS MAX MIN (+ ACOS ATAN
  SINH COSH TANH LOG10) and the constants PI TWOPI DEGRAD RADDEG E EMASS
  PMASS CLIGHT (MAD8 values).  RANF/GAUSS/TGAUSS evaluate to their mean
  with a warning (HELIX imports the nominal machine).
* Element definitions ``name: CLASS, attr=expr, …`` with MAD8 keyword
  abbreviations (QUAD, SEXT, RFCAV, …), class inheritance
  (``QF2: QF, K1=…``) and attribute changes (``QF, K1=…``).  String
  attributes (``TYPE=…``) are kept out of the arithmetic.
* ``name: LINE = (…)`` with ``-NAME``, ``N*NAME``, nested groups
  ``N*(A, -(B, C))``, and lines with formal arguments
  ``CELL(X, Y): LINE = (X, D, Y)`` called as ``CELL(QF, QD)``.
* ``name: SEQUENCE[, REFER=…, L=…]`` … ``ENDSEQUENCE[, AT=…]`` with
  ``AT`` / ``FROM`` placement (drifts generated between members).
* ``USE, name`` selects the beam line; without it the unreferenced LINE
  with the largest expansion is used (ambiguity warns; ``strict=True``
  raises); ``line=`` overrides both.
* Action commands (TWISS, MATCH … ENDMATCH, EALIGN, SELECT, …) are
  skipped and reported in one summary warning.

Element mapping
---------------
Through :func:`madx_parser._build_element` (drift, quadrupole,
sbend/rbend → Edge+Dipole+Edge, solenoid, sextupole, rfcavity, monitors,
collimators, instrument, matrix, marker) plus the MAD8-specific cases:

* ``KICKER/HKICKER/VKICKER`` with zero kick → Marker + full-length body
  Drift (the BTL/BAL corrector layout); a non-zero kick → ``Steerer``
  between half-length drifts, ``TILT`` rotating the kick.
* plain ``MONITOR`` (and IMONITOR/BLMONITOR/WIRE/PROFILE) → Marker (not a
  BPM) + body Drift; ``H/VMONITOR`` → BPM marker.
* ``TILT`` on a quadrupole → ``Quadrupole.skew_angle`` (bare TILT = 45°).
* ``OCTUPOLE`` → thin ``Multipole`` (k3·L) between half drifts;
  ``MULTIPOLE`` with MAD8 ``KnL`` / ``Tn`` → normal/skew coefficients
  (knl_n = KnL·cos((n+1)Tn), ksl_n = −KnL·sin((n+1)Tn), MAD-X sense).
* a zero-angle ``SBEND/RBEND`` carrying ``K1`` → ``Quadrupole`` (exact:
  with h = 0 the bend map is the quadrupole map).
* ``RFCAVITY`` with ``HARMON`` and no ``FREQ`` → frequency
  HARMON·βc/C, C = the length of the used line.
* ``LCAVITY`` (MAD8 linac cavity: DELTAE [MeV], PHI0 [2π], FREQ [MHz])
  → ``RFGap`` (gain DELTAE·cos 2πPHI0); magnets downstream are converted
  with the local rigidity, as MAD8 normalises them.
* ``LUMP, LINE=…`` → the referenced line, expanded in place.
* Anything else (ELSEPARATOR, WIGGLER, BEAMBEAM, SROT, YROT, ELENS, …) →
  a drift of its length (or a marker when L = 0); one warning per class.

Rigidity / charge convention
----------------------------
MAD strengths are normalized (K1 = (q/p)·∂B_y/∂x); HELIX stores the
lab-frame gradient and applies the beam's charge at track time, so the
conversion is G = sign(q)·K1·|Bρ| (see ``madx_parser._signed_brho``).
Bρ resolution order: a ``BEAM`` statement that names the particle or its
energy (PARTICLE and/or MASS+CHARGE, ENERGY/PC/GAMMA) → the ``brho=``
argument → a ``BRHO := …`` file parameter, whose species is ``species=``,
else the fallback beam's, else H⁻ (for H⁻ it is rejected when it implies
an impossible energy — the ×1000 unit slip of some BAL exports) → the
caller's ``fallback_beam`` (the project / Beam-tab beam, with a warning)
→ hard ``ValueError``.  A
BEAM particle HELIX does not model (electron, positron, antiproton, …)
keeps its own rigidity; the tracking reference becomes a proton at that
rigidity, and a warning says so.

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

import ast
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
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.steerer import Steerer
from linac_gen.io.madx_parser import (
    _M_TO_MM,
    _RAD_TO_DEG,
    _brho,
    _build_element,
    _convert_raw_matrices,
    _drift_pad,
    _signed_brho,
    _species_from_name,
    _split_attributes,
)

_IDENT = r"[A-Za-z_][\w'.]*"
_KICKER_TYPES = ("kicker", "hkicker", "vkicker", "tkicker")
_PLAIN_MONITORS = ("monitor", "imonitor", "blmonitor", "wire", "profile",
                   "slmonitor")

# MAD8 predefined constants (MAD8 values: EMASS/PMASS in GeV, the
# 1990s PDG masses MAD8 ships with — a BEAM ``ENERGY = 8 + PMASS`` means
# exactly what its author computed).  TRUE/FALSE kept from the MAD-X
# table the pre-2026-09-24 importer used.
_CONST8 = {
    "pi": math.pi, "twopi": 2.0 * math.pi,
    "degrad": 180.0 / math.pi, "raddeg": math.pi / 180.0,
    "e": math.e, "emass": 0.51099906e-3, "pmass": 0.93827231,
    "clight": 2.99792458e8, "true": 1.0, "false": 0.0,
}
_FUNC8 = {
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "sinh": math.sinh, "cosh": math.cosh,
    "tanh": math.tanh, "log10": math.log10, "abs": abs,
    "max": max, "min": min,
}
_RANDOM8 = {"ranf": 0.5, "gauss": 0.0, "tgauss": 0.0}

# MAD8 element classes (full names; abbreviations resolve by unique prefix)
_CLASSES = (
    "drift", "sbend", "rbend", "quadrupole", "sextupole", "octupole",
    "multipole", "solenoid", "hkicker", "vkicker", "kicker", "rfcavity",
    "lcavity", "elseparator", "hmonitor", "vmonitor", "monitor",
    "instrument", "marker", "ecollimator", "rcollimator", "yrot", "srot",
    "beambeam", "matrix", "lump", "wiggler", "imonitor", "blmonitor",
    "wire", "profile", "slmonitor", "tkicker", "gkick", "placeholder",
)
# action / control commands — parsed past, never built
_COMMANDS = (
    "title", "use", "saveline", "save", "option", "assign", "print",
    "twiss", "emit", "survey", "select", "ealign", "efield", "efcomp",
    "eopt", "eprint", "esave", "split", "value", "show", "help", "beta0",
    "bmpm", "match", "endmatch", "vary", "constraint", "couple", "weight",
    "lmdif", "simplex", "migrad", "cell", "fix", "rmatrix", "tmatrix",
    "track", "start", "run", "endtrack", "observe", "tsave", "plot",
    "setplot", "resplot", "archive", "retrieve", "static", "dynamic",
    "normal", "harmon", "hcell", "htune", "hvary", "hweight", "hchrom",
    "hresonance", "hfunctions", "endharm", "correct", "getorbit",
    "putorbit", "micado", "usekick", "usemonitor", "ibs", "excite",
    "increment", "envelope", "system", "list", "exit", "quit", "stop",
    "beam", "call", "return", "set", "comment", "endcomment", "line",
    "sequence", "endsequence", "constant", "subroutine", "endsubroutine",
    "do", "enddo", "packmemory", "optics", "table", "seqedit",
    "endedit", "remove", "cycle", "flatten", "reflect", "move",
)
_ERROR_COMMANDS = ("ealign", "efield", "efcomp", "eopt", "esave")
_KEYWORDS = _CLASSES + _COMMANDS
# attribute values that are words, not arithmetic
_STRING_ATTRS = ("type", "particle", "filename", "file", "refer", "from",
                 "line", "apertype", "period", "sequence", "range", "label")
_NUM_D = re.compile(r"(?<![\w.'])(\d+\.?\d*|\.\d+)[dD]([+-]?\d+)")
_TOKEN = re.compile(rf"(?<![\w.']){_IDENT}")


def _keyword(word: str) -> str | None:
    """Full MAD8 keyword for *word* (exact, else unique prefix ≥ 3)."""
    w = word.lower()
    if w in _KEYWORDS:
        return w
    if len(w) < 3:
        return None
    hits = [k for k in _KEYWORDS if k.startswith(w)]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# Front-end: physical text → statements
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Drop ``!`` comments (never inside a double-quoted string)."""
    inq = False
    for i, ch in enumerate(line):
        if ch == '"':
            inq = not inq
        elif ch == "!" and not inq:
            return line[:i]
    return line


def _statements(text: str) -> list[str]:
    """Physical lines → MAD8 statements: comments stripped, ``&``
    continuations joined (text after the ``&`` is ignored), ``;``
    separators split."""
    out, buf = [], ""
    for raw in text.splitlines():
        raw = _strip_comment(raw)
        amp = _find_unquoted(raw, "&")
        if amp >= 0:
            buf += raw[:amp] + " "
            continue
        buf += raw
        for st in _split_unquoted(buf, ";"):
            if st.strip():
                out.append(st.strip())
        buf = ""
    for st in _split_unquoted(buf, ";"):
        if st.strip():
            out.append(st.strip())
    return out


def _find_unquoted(s: str, ch: str) -> int:
    inq = False
    for i, c in enumerate(s):
        if c == '"':
            inq = not inq
        elif c == ch and not inq:
            return i
    return -1


def _split_unquoted(s: str, ch: str) -> list[str]:
    parts, cur, inq = [], [], False
    for c in s:
        if c == '"':
            inq = not inq
        if c == ch and not inq:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return parts


def _split_top(s: str) -> list[str]:
    """Split on commas at parenthesis depth 0."""
    parts, cur, depth = [], [], 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _split_quoted_attributes(body: str) -> list[str]:
    """``_split_attributes`` that also leaves commas inside "…" alone."""
    parts, cur, depth, inq = [], [], 0, False
    for ch in body:
        if ch == '"':
            inq = not inq
        elif not inq and ch in "{(":
            depth += 1
        elif not inq and ch in "})":
            depth -= 1
        if ch == "," and depth == 0 and not inq:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _unquote(v) -> str:
    v = str(v).strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


# ---------------------------------------------------------------------------
# LINE bodies: items → nodes
#   ("ref", NAME, rep, neg, args|None)     args = list of item lists
#   ("group", items, rep, neg)
# ---------------------------------------------------------------------------

def _parse_items(body: str) -> list[tuple]:
    return [_parse_item(it) for it in _split_top(body)]


def _parse_item(item: str) -> tuple:
    s = item.strip()
    rep, neg = 1, False
    while True:
        m = re.match(r"^(\d+)\s*\*\s*(.*)$", s)
        if m:
            rep *= int(m.group(1))
            s = m.group(2).strip()
            continue
        if s.startswith("-"):
            neg = not neg
            s = s[1:].strip()
            continue
        break
    if s.startswith("(") and s.endswith(")"):
        return ("group", _parse_items(s[1:-1]), rep, neg)
    m = re.match(rf"^({_IDENT})\s*\((.*)\)$", s)
    if m:
        args = [[_parse_item(a)] for a in _split_top(m.group(2))]
        return ("ref", m.group(1).upper(), rep, neg, args)
    return ("ref", s.upper(), rep, neg, None)


def _refs(items: list[tuple], out: set) -> None:
    for it in items:
        if it[0] == "group":
            _refs(it[1], out)
        else:
            out.add(it[1])
            for a in it[4] or ():
                _refs(a, out)


# ---------------------------------------------------------------------------
# The parsed file
# ---------------------------------------------------------------------------

class _Mad8File:
    """Parsed statements + lazy, memoized expression resolution."""

    def __init__(self, text: str, warnings: list, *, base_dir=None,
                 strict: bool = False):
        self.params: dict[str, str] = {}       # UPPER name -> raw expr
        self.elems: dict[str, tuple[str, dict]] = {}   # name -> (base, raw attrs)
        self.lines: dict[str, tuple[list, list]] = {}  # name -> (formals, items)
        self.sequences: dict[str, dict] = {}
        self.beam_attrs: dict = {}
        self.use: str | None = None
        self.warnings = warnings
        self.strict = strict
        self._memo: dict[str, float] = {}
        self._resolving: set[str] = set()
        self._rtype_memo: dict[str, tuple[str, dict]] = {}
        self._redefined: list[str] = []
        self._ignored: dict[str, int] = {}
        self._unrecognised: list[str] = []
        self._random_used: set[str] = set()
        self._seq: dict | None = None
        self._skip_until: str | None = None
        self._stopped = False
        self._calls: list[str] = []
        self._parse(text, Path(base_dir) if base_dir else Path.cwd(), 0)
        self._summarise()

    # -- statement classification ------------------------------------
    def _parse(self, text: str, base: Path, depth: int) -> None:
        for st in _statements(text):
            if self._stopped:
                return
            if self._handle(st, base, depth) == "return":
                return

    def _handle(self, st: str, base: Path, depth: int):
        up = st.upper()
        first = re.match(rf"^\s*({_IDENT})", st)
        word = first.group(1) if first else ""
        kw_first = _keyword(word) if word else None
        if self._skip_until is not None:
            if kw_first == self._skip_until:
                self._skip_until = None
            return None
        # -- parameters:  NAME := expr   /  NAME = expr ------------------
        m = re.match(rf"^({_IDENT})\s*:?=\s*(.+)$", st)
        if m and not re.match(rf"^{_IDENT}\s*:\s*[A-Za-z]", st):
            self._set_param(m.group(1), m.group(2))
            return None
        # -- label: …  (a label may carry formal arguments) ---------------
        m = re.match(rf"^({_IDENT})\s*(\(([^()]*)\))?\s*:(?!=)\s*(.*)$", st)
        if m:
            return self._labelled(m.group(1).upper(),
                                  m.group(3), m.group(4).strip(), st)
        # -- unlabelled: KEYWORD[, …] / NAME, attr=… ---------------------
        m = re.match(rf"^({_IDENT})\s*(?:,\s*(.*)|\s+(.*))?$", st)
        if not m:
            self._unrecognised.append(st)
            return None
        name, rest = m.group(1), (m.group(2) or m.group(3) or "").strip()
        nup = name.upper()
        if self._seq is not None and (nup in self.elems
                                      or nup in self.sequences):
            attrs = self._raw_attrs(rest)
            self._place(nup, attrs)
            if set(attrs) - {"at", "from"}:
                self.warnings.append(
                    f"MAD8: {nup} placed in sequence {self._seq['name']} with "
                    "extra attributes — only AT/FROM are used for a placement")
            return None
        if nup in self.elems:
            # attribute change:  QF, K1=0.5
            base_t, raw = self.elems[nup]
            raw = dict(raw)
            raw.update(self._raw_attrs(rest))
            self.elems[nup] = (base_t, raw)
            self._rtype_memo.clear()
            return None
        kw = _keyword(name)
        if kw is None:
            self._unrecognised.append(st)
            return None
        return self._command(kw, rest, base, depth, st)

    def _labelled(self, label: str, formals, rest: str, st: str):
        m = re.match(r"^LINE\s*=\s*\((.*)\)\s*$", rest, re.IGNORECASE)
        if m:
            fl = [a.strip().upper() for a in (formals or "").split(",")
                  if a.strip()]
            if label in self.lines and self.lines[label] != (fl, _parse_items(m.group(1))):
                self._redefined.append(label)
            self.lines[label] = (fl, _parse_items(m.group(1)))
            return None
        m = re.match(r"^CONST(?:ANT)?\s*=\s*(.+)$", rest, re.IGNORECASE)
        if m:
            self._set_param(label, m.group(1))
            return None
        m = re.match(rf"^({_IDENT})\s*(?:,\s*(.*))?$", rest)
        if not m:
            self._unrecognised.append(st)
            return None
        cls, body = m.group(1), (m.group(2) or "")
        cup = cls.upper()
        if cup == label and cup in self.elems:
            # "QF: QF, K1=…" re-states an existing element: merge
            base_t, raw0 = self.elems[label]
            raw0 = dict(raw0)
            raw0.update(self._raw_attrs(body))
            self.elems[label] = (base_t, raw0)
            self._rtype_memo.clear()
            return None
        if cup in self.elems or cup in self.sequences:
            base = cup                         # class inheritance / sequence ref
        else:
            kw = _keyword(cls)
            if kw == "sequence":
                self._open_sequence(label, self._raw_attrs(body))
                return None
            if kw == "beam":
                self._beam(body)
                return None
            if kw in _COMMANDS:
                self._ignore(kw)
                if kw in ("match", "subroutine"):
                    self._skip_until = "end" + kw
                return None
            base = kw if kw is not None else cls.lower()
        raw = self._raw_attrs(body)
        at = {k: raw.pop(k) for k in ("at", "from") if k in raw}
        if label in self.elems and self.elems[label] != (base, raw):
            self._redefined.append(label)
        self.elems[label] = (base, raw)
        self._rtype_memo.clear()
        if self._seq is not None:
            self._place(label, at)
        elif at:
            self.warnings.append(f"MAD8: {label} has AT/FROM outside a "
                                 "SEQUENCE — ignored")
        return None

    def _command(self, kw: str, rest: str, base: Path, depth: int, st: str):
        if kw == "return":
            return "return"
        if kw in ("stop", "exit", "quit"):
            self._stopped = True
            return "return"
        if kw == "beam":
            self._beam(rest)
        elif kw == "use":
            self._use(rest)
        elif kw == "call":
            self._call(rest, base, depth)
        elif kw == "set":
            parts = _split_top(rest)
            ma = (re.match(rf"^({_IDENT})\s*\[\s*(\w+)\s*\]$", parts[0])
                  if len(parts) == 2 else None)
            if len(parts) == 2 and re.match(rf"^{_IDENT}$", parts[0]):
                self._set_param(parts[0], parts[1])
            elif ma and ma.group(1).upper() in self.elems:
                base_t, raw = self.elems[ma.group(1).upper()]
                raw = dict(raw)
                raw[ma.group(2).lower()] = parts[1]
                self.elems[ma.group(1).upper()] = (base_t, raw)
                self._rtype_memo.clear()
            else:
                self._unrecognised.append(st)
        elif kw == "comment":
            self._skip_until = "endcomment"
        elif kw in ("match", "subroutine"):
            self._ignore(kw)
            self._skip_until = "end" + kw
        elif kw == "endsequence":
            self._close_sequence(self._raw_attrs(rest))
        elif kw in _COMMANDS:
            self._ignore(kw)
        else:
            # an element class with no label ("QUADRUPOLE, L=…") defines
            # nothing
            self._unrecognised.append(st)
        return None

    def _ignore(self, kw: str) -> None:
        self._ignored[kw] = self._ignored.get(kw, 0) + 1

    def _set_param(self, name: str, expr: str) -> None:
        key = name.upper()
        expr = expr.strip()
        if key in self.params and self.params[key] != expr:
            self._redefined.append(key)
        if key.lower() in _CONST8:
            self.warnings.append(
                f"MAD8: the file defines {key}, which is a MAD8 built-in "
                "constant — expressions use the built-in value")
        self.params[key] = expr

    def _beam(self, body: str) -> None:
        # later BEAM statements update the earlier ones attribute-wise
        self.beam_attrs.update(self._raw_attrs(body))

    def _use(self, rest: str) -> None:
        attrs = self._raw_attrs(rest)
        for k in ("period", "sequence"):
            if k in attrs and attrs[k] is not True:
                self.use = _unquote(attrs[k]).upper()
                return
        for k, v in attrs.items():
            if v is True:
                self.use = k.upper()
                return

    def _call(self, rest: str, base: Path, depth: int) -> None:
        attrs = self._raw_attrs(rest)
        fn = None
        for k in ("filename", "file"):
            if k in attrs and attrs[k] is not True:
                fn = _unquote(attrs[k])
        if fn is None:
            # a bare file name (``CALL, lattice.mad8``) — keep its case
            fn = _unquote(_split_top(rest)[0]) if rest.strip() else ""
        cands = [Path(fn)] if Path(fn).is_absolute() else [
            base / fn, Path.cwd() / fn]
        path = next((c for c in cands if c.is_file()), None)
        if path is None:
            msg = f"MAD8: CALL {fn!r} — file not found (looked in {base})"
            if self.strict:
                raise ValueError(msg)
            self.warnings.append(msg + " — skipped")
            return
        if depth >= 20:
            raise ValueError(f"MAD8: CALL nesting deeper than 20 at {fn!r}")
        self._calls.append(str(path))
        self._parse(path.read_text(encoding="latin-1", errors="replace"),
                    path.parent, depth + 1)

    # -- sequences -------------------------------------------------------
    def _open_sequence(self, name: str, attrs: dict) -> None:
        if self._seq is not None:
            self.warnings.append(f"MAD8: SEQUENCE {name} opened inside "
                                 f"{self._seq['name']} — the outer one is closed")
        refer = str(attrs.get("refer", "centre")).lower() \
            if attrs.get("refer") is not True else "centre"
        self._seq = {"name": name, "refer": _unquote(refer),
                     "length": attrs.get("l"), "items": []}
        self.sequences[name] = self._seq

    def _close_sequence(self, attrs: dict) -> None:
        if self._seq is None:
            self.warnings.append("MAD8: ENDSEQUENCE without SEQUENCE — ignored")
            return
        if attrs.get("at") not in (None, True):
            self._seq["length"] = attrs["at"]
        self._seq = None

    def _place(self, name: str, attrs: dict) -> None:
        if "at" not in attrs or attrs["at"] is True:
            self.warnings.append(f"MAD8: {name} in sequence {self._seq['name']} "
                                 "has no AT — placed at 0")
        self._seq["items"].append(
            (name, attrs.get("at", "0") if attrs.get("at") is not True else "0",
             _unquote(attrs["from"]).upper()
             if attrs.get("from") not in (None, True) else None))

    # -- bookkeeping -----------------------------------------------------
    def _summarise(self) -> None:
        if self._seq is not None:
            self.warnings.append(f"MAD8: SEQUENCE {self._seq['name']} never "
                                 "closed with ENDSEQUENCE")
        if self._redefined:
            names = sorted(set(self._redefined))
            self.warnings.append(
                f"MAD8: {len(names)} name(s) defined more than once — the last "
                f"definition is used: {names[:12]}")
        errs = sorted(k for k in self._ignored if k in _ERROR_COMMANDS)
        if errs:
            self.warnings.append(
                f"MAD8: machine-error commands ignored ({', '.join(e.upper() for e in errs)}) "
                "— HELIX imports the nominal machine; model errors with the "
                "Error Study")
        rest = {k: n for k, n in self._ignored.items() if k not in _ERROR_COMMANDS
                and k not in ("title", "saveline", "save", "option", "use")}
        if rest:
            self.warnings.append(
                "MAD8: action commands skipped (not lattice definitions): "
                + ", ".join(f"{k.upper()}×{n}" for k, n in sorted(rest.items())))
        for st in self._unrecognised:
            self.warnings.append(f"MAD8: unrecognised statement skipped: "
                                 f"{st[:70]!r}")

    @staticmethod
    def _raw_attrs(body: str) -> dict:
        """``attr=expr`` pairs kept as RAW strings (evaluated lazily —
        eager evaluation would silently coerce unresolved refs to 0)."""
        attrs: dict = {}
        for part in (_split_attributes(body) if '"' not in body
                     else _split_quoted_attributes(body)):
            if not part:
                continue
            if "=" not in part:
                attrs[part.strip().lower()] = True
                continue
            k, _, v = part.partition("=")
            k = k.rstrip(":").strip().lower()
            if "(" in k:                      # RM(1,2) / KICK(5) / TM(1,2,3)
                k = re.sub(r"[\s(),]", "", k)
            attrs[k] = v.strip()
        return attrs

    # -- element class resolution (inheritance) --------------------------
    def rtype(self, name: str, _seen=None) -> tuple[str, dict]:
        """(MAD8 class, merged raw attrs) of an element, following
        ``QF2: QF, …`` inheritance chains."""
        if name in self._rtype_memo:
            return self._rtype_memo[name]
        seen = _seen or set()
        if name in seen:
            raise ValueError(f"MAD8: circular class definition at {name!r}")
        seen.add(name)
        base, raw = self.elems[name]
        if base.upper() in self.elems and base.upper() != name:
            ptype, praw = self.rtype(base.upper(), seen)
            merged = dict(praw)
            merged.update(raw)
            out = (ptype, merged)
        else:
            out = (base, raw)
        self._rtype_memo[name] = out
        return out

    # -- lazy numeric resolution ---------------------------------------
    def resolve(self, expr: str) -> float:
        """Evaluate a MAD8 expression: numbers, arithmetic, parameter
        references, NAME[ATTR] element-attribute references, MAD8
        functions and constants."""
        s = _NUM_D.sub(r"\1e\2", str(expr).strip())
        s = re.sub(r"(?<![\w.'])0+(?=\d)", "", s)      # 010 → 10 (Fortran-legal)
        vals: dict[str, float] = {}

        def hold(v: float) -> str:
            k = f"_v{len(vals)}"
            vals[k] = float(v)
            return k

        out, i = [], 0
        while True:
            m = _TOKEN.search(s, i)
            if m is None:
                break
            out.append(s[i:m.start()])
            i = m.end()
            tok = m.group(0)
            tail = s[i:]
            ma = re.match(r"\s*\[\s*(\w+)\s*\]", tail)
            if ma:
                out.append(hold(self.attr_val(tok.upper(), ma.group(1).lower())))
                i += ma.end()
                continue
            low = tok.lower()
            if re.match(r"\s*\(", tail):
                if low in _RANDOM8:
                    if low not in self._random_used:
                        self._random_used.add(low)
                        self.warnings.append(
                            f"MAD8: random function {tok.upper()}() evaluated at "
                            f"its mean ({_RANDOM8[low]}) — HELIX imports the "
                            "nominal machine")
                    mc = re.match(r"\s*\(([^()]*)\)", tail)
                    i += mc.end() if mc else 0
                    out.append(hold(_RANDOM8[low]))
                    continue
                if low not in _FUNC8:
                    raise ValueError(f"MAD8: unsupported function {tok.upper()!r}")
                out.append(f"__f_{low}")
                continue
            if low in _CONST8:
                out.append(hold(_CONST8[low]))
                continue
            out.append(hold(self.param_val(tok.upper())))
        out.append(s[i:])
        src = "".join(out).replace("^", "**")
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"MAD8: cannot parse expression {expr!r}: {exc}")
        return _eval8(tree.body, vals)

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
        if ename == "BEAM" and ename not in self.elems:
            raw = self.beam_attrs.get(attr)
            if (raw is None or raw is True) and attr in (
                    "mass", "charge", "energy", "pc", "gamma", "beta"):
                # derived BEAM quantities (MAD8 fills them in from the others)
                if attr in ("mass", "charge"):
                    return float(_beam_particle(self)[0 if attr == "mass" else 1])
                if getattr(self, "_beam_busy", False):
                    raise ValueError(f"MAD8: circular BEAM reference BEAM[{attr.upper()}]")
                self._beam_busy = True
                try:
                    return float(_beam_kinematics(self, [])[attr])
                finally:
                    self._beam_busy = False
        elif ename not in self.elems:
            raise ValueError(f"MAD8: {ename}[{attr}] — unknown element")
        else:
            raw = self.rtype(ename)[1].get(attr)
        if raw is None or raw is True:
            return 0.0
        return self.resolve(raw)

    def numeric_attrs(self, ename: str, strict: bool) -> dict:
        """Element attributes with every value resolved to a float.
        Unresolvable values warn (or raise when strict) instead of the
        silent-0.0 coercion the MAD-X `_gf` path would apply.  Word
        attributes (TYPE, LINE, …) stay strings."""
        _etype, raw = self.rtype(ename)
        out: dict = {}
        for k, v in raw.items():
            if v is True:
                out[k] = True
                continue
            if k in _STRING_ATTRS or str(v).strip()[:1] in ('"', "'"):
                out[k] = _unquote(v)
                continue
            try:
                out[k] = self.resolve(v)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
                msg = (f"MAD8: {ename}.{k} = {v!r} could not be "
                       f"evaluated ({exc})")
                if strict:
                    raise ValueError(msg) from exc
                self.warnings.append(msg + " — treated as 0")
                out[k] = 0.0
        return out


def _eval8(node, vals: dict) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return 1.0 if node.value else 0.0
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"MAD8: unsupported literal {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in vals:
            return vals[node.id]
        raise ValueError(f"MAD8: unknown identifier {node.id!r}")
    if isinstance(node, ast.BinOp):
        lhs, rhs, op = _eval8(node.left, vals), _eval8(node.right, vals), node.op
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
        raise ValueError("MAD8: unsupported operator in expression")
    if isinstance(node, ast.UnaryOp):
        v = _eval8(node.operand, vals)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        raise ValueError("MAD8: unsupported unary operator in expression")
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id.startswith("__f_")):
        fn = _FUNC8[node.func.id[4:]]
        return float(fn(*[_eval8(a, vals) for a in node.args]))
    raise ValueError("MAD8: unsupported construct in expression")


# ---------------------------------------------------------------------------
# Element construction (wraps madx_parser._build_element)
# ---------------------------------------------------------------------------

_GENERIC_TYPES = {
    "drift", "quadrupole", "sbend", "rbend", "solenoid", "sextupole",
    "rfcavity", "hmonitor", "vmonitor", "instrument", "placeholder",
    "marker", "ecollimator", "rcollimator", "matrix",
}


def _multipole_mad8(name: str, attrs: dict) -> Multipole | None:
    """MAD8 ``KnL`` / ``Tn`` → normal/skew coefficients, or None when the
    element carries MAD-X-style KNL/KSL arrays instead."""
    orders = [n for n in range(21) if f"k{n}l" in attrs]
    if not orders:
        return None
    nmax = max(orders)
    knl, ksl = [0.0] * (nmax + 1), [0.0] * (nmax + 1)
    for n in orders:
        k = float(attrs[f"k{n}l"] or 0.0)
        t = attrs.get(f"t{n}", 0.0)
        t = math.pi / (2.0 * (n + 1)) if t is True else float(t or 0.0)
        knl[n] = k * math.cos((n + 1) * t) + 0.0
        ksl[n] = -k * math.sin((n + 1) * t) + 0.0
    return Multipole(name=name, knl=knl, ksl=ksl)


def _build_mad8_element(f: _Mad8File, name: str, brho_signed: float,
                        strict: bool, unsupported: dict | None = None
                        ) -> tuple[list, float]:
    etype, _raw = f.rtype(name)
    attrs = f.numeric_attrs(name, strict)
    l_mm = float(attrs.get("l", 0.0) or 0.0) * _M_TO_MM

    if etype in _KICKER_TYPES:
        if etype == "hkicker":
            hk, vk = float(attrs.get("kick", 0.0) or 0.0), 0.0
        elif etype == "vkicker":
            hk, vk = 0.0, float(attrs.get("kick", 0.0) or 0.0)
        else:
            hk = float(attrs.get("hkick", 0.0) or 0.0)
            vk = float(attrs.get("vkick", 0.0) or 0.0)
        if hk == 0.0 and vk == 0.0:
            # Marker (instrument position) + full-length body drift so the
            # geometry is exact — the BTL/BAL corrector layout.
            elems: list = [Marker(name=name)]
            if l_mm != 0.0:
                elems.append(Drift(name=f"{name}_body", length=l_mm))
            return elems, l_mm
        # A set kick: thin Steerer at the centre (exact for the linear
        # orbit of a uniform-field kicker).  TILT rotates the kick vector
        # the MAD way (positive = right-handed screw about s).
        tilt = attrs.get("tilt", 0.0)
        tilt = (math.pi / 2.0) if tilt is True else float(tilt or 0.0)
        c, s = math.cos(tilt), math.sin(tilt)
        hk, vk = hk * c - vk * s, hk * s + vk * c
        st = Steerer(name=name, by_l=hk * brho_signed, bx_l=vk * brho_signed)
        return _drift_pad(st, l_mm, name)

    if etype in _PLAIN_MONITORS:
        # Plain MONITOR = generic instrument (ion pump, collimator flag,
        # …), not a beam-position monitor.  H/VMONITOR stay BPMs via the
        # madx factory below.
        elems = [Marker(name=name, is_bpm=False)]
        if l_mm != 0.0:
            elems.append(Drift(name=f"{name}_body", length=l_mm))
        return elems, l_mm

    if etype == "octupole":
        k3l = float(attrs.get("k3", 0.0) or 0.0) * float(attrs.get("l", 0.0) or 0.0)
        tilt = attrs.get("tilt", 0.0)
        tilt = (math.pi / 8.0) if tilt is True else float(tilt or 0.0)
        mp = Multipole(name=name, knl=[0.0, 0.0, 0.0, k3l],
                       tilt_deg=tilt * _RAD_TO_DEG)
        return _drift_pad(mp, l_mm, name)

    if etype == "multipole":
        mp = _multipole_mad8(name, attrs)
        if mp is not None and float(attrs.get("k0l", 0.0) or 0.0) != 0.0:
            f.warnings.append(
                f"MAD8: {name}: K0L = {float(attrs['k0l']):g} is applied as an orbit "
                "kick about a straight reference (as HELIX's MAD-X importer does); "
                "MAD-X instead bends the reference orbit by K0L — check which the "
                "source intends")
        if mp is not None:
            if attrs.get("tilt") not in (None, 0.0, True):
                mp.tilt_deg = float(attrs["tilt"]) * _RAD_TO_DEG
            return _drift_pad(mp, l_mm, name)
        return _build_element(name, "multipole", attrs, brho_signed,
                              f.warnings, rbend_chord=False)

    if etype == "lcavity":
        # MAD8 linac cavity: DELTAE = on-crest energy gain [MeV], PHI0 in
        # units of 2π (0 = crest), FREQ [MHz].  HELIX RFGap gains
        # q·V·cos φ, so φ = 360·PHI0 (+180° for a negative charge).
        dE = float(attrs.get("deltae", 0.0) or 0.0)
        phase = 360.0 * float(attrs.get("phi0", 0.0) or 0.0)
        if brho_signed < 0:
            phase += 180.0
        if dE < 0:
            dE, phase = -dE, phase + 180.0
        gap = RFGap(name=name, voltage=dE, phase=phase,
                    frequency=float(attrs.get("freq", 0.0) or 0.0), ttf=1.0)
        if gap.frequency <= 0.0 and dE != 0.0:
            f.warnings.append(f"MAD8: LCAVITY {name} has no FREQ — frequency 0")
        return _drift_pad(gap, l_mm, name)

    if etype in ("sbend", "rbend"):
        for k in ("k2", "h1", "h2"):
            if float(attrs.get(k, 0.0) or 0.0) != 0.0:
                f.warnings.append(f"MAD8: {name}.{k.upper()} = {attrs[k]:g} — "
                                  "not modelled by the HELIX bend, dropped")
        if abs(float(attrs.get("angle", 0.0) or 0.0)) < 1e-12 \
                and float(attrs.get("k1", 0.0) or 0.0) != 0.0:
            # h = 0: the bend map IS the quadrupole map (edges vanish).  The
            # bend's TILT rotates it (a bare TILT on a bend means π/2).
            q = dict(attrs)
            if q.get("tilt") is True:
                q["tilt"] = math.pi / 2.0
            return _build_element(name, "quadrupole", q, brho_signed,
                                  f.warnings, rbend_chord=False)

    if etype not in _GENERIC_TYPES:
        # ELSEPARATOR, WIGGLER, BEAMBEAM, SROT/YROT, ELENS, … : keep the
        # geometry, say what is lost (one summary per class, see caller)
        if unsupported is not None:
            unsupported.setdefault(etype, []).append((name, l_mm, attrs))
        if l_mm != 0.0:
            return [Drift(name=name, length=l_mm)], l_mm
        return [Marker(name=name)], 0.0

    if etype == "rfcavity":
        freq = float(attrs.get("freq", 0.0) or 0.0)
        harmon = float(attrs.get("harmon", 0.0) or 0.0)
        if freq <= 0.0 and harmon <= 0.0:
            # MAD8 derives the RF frequency from HARMON; with neither
            # HARMON nor FREQ the cavity has no frequency — a drift of its
            # length (a voltage cannot be applied without a frequency).
            if unsupported is not None:
                unsupported.setdefault("rfcavity without HARMON/FREQ", []).append(
                    (name, l_mm, attrs))
            if l_mm != 0.0:
                return [Drift(name=name, length=l_mm)], l_mm
            return [Marker(name=name)], 0.0
        if freq <= 0.0:
            attrs = dict(attrs)
            attrs["freq"] = 0.0             # set from HARMON after the line is built

    # MAD8 has no RBARC option; its RBEND ``L`` is taken as the arc length
    # (the pre-2026-09-03 behaviour of the shared builder, kept explicitly —
    # not verified against a MAD8 binary, unlike the MAD-X chord rule).
    elems, total_mm = _build_element(name, etype, attrs, brho_signed,
                                     f.warnings, rbend_chord=False)
    if etype == "quadrupole" and attrs.get("tilt") and attrs["tilt"] is not True:
        for el in elems:
            if isinstance(el, Quadrupole):
                el.skew_angle = float(attrs["tilt"]) * _RAD_TO_DEG
    if etype in ("sbend", "rbend"):
        # MAD8 encodes vertical bends as TILT=±π/2 (a bare TILT = +π/2).
        # The shared _build_element already folds ±π/2 into ``hv`` and the
        # angle sign (TILT=-π/2 bends towards -y, MAD-X-verified) and emits
        # ρ > 0; the pass below only warns about other tilts and re-asserts
        # the HELIX/TraceWin convention (ρ > 0, sign in the angle, plane in
        # ``hv``) so a future builder change cannot silently break it.
        tilt = attrs.get("tilt", 0.0)
        tilt = (math.pi / 2.0) if tilt is True else float(tilt or 0.0)
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
    if etype == "rfcavity" and float(attrs.get("freq", 0.0) or 0.0) == 0.0:
        for el in elems:
            if isinstance(el, RFGap):
                el._mad8_harmon = float(attrs["harmon"])
    if etype == "matrix":
        n_tm = sum(1 for k in attrs if k.startswith("tm"))
        if n_tm:
            f.warnings.append(f"MAD8: MATRIX {name}: {n_tm} second-order TM "
                              "terms dropped (first-order map imported)")
    return elems, total_mm


# ---------------------------------------------------------------------------
# Rigidity resolution
# ---------------------------------------------------------------------------

# MAD8 particle names → (mass [GeV], charge)
_MAD8_PARTICLES = {
    "positron": (0.51099906e-3, 1), "electron": (0.51099906e-3, -1),
    "proton": (0.93827231, 1), "antiproton": (0.93827231, -1),
    "posmuon": (0.1056583755, 1), "negmuon": (0.1056583755, -1),
}
# H⁻ cannot survive beyond a few GeV (field and blackbody stripping); a
# BRHO parameter implying more is a unit slip, not a design.
_HMINUS_MAX_W_MEV = 20000.0


def _reference_at_brho(species, brho_abs: float) -> ReferenceParticle:
    from linac_gen.core.constants import C_LIGHT
    bg = brho_abs * C_LIGHT / (species.mass * 1e6)
    gamma = math.sqrt(1.0 + bg * bg)
    return ReferenceParticle(species=species, w_kin=species.mass * (gamma - 1.0),
                             frequency=162.5)


_BEAM_RIGIDITY_KEYS = ("particle", "mass", "charge", "energy", "pc", "gamma")


def _beam_declares_particle(f: _Mad8File) -> bool:
    """A BEAM statement is a rigidity source only when it names the particle
    (PARTICLE / MASS) or its energy (ENERGY / PC / GAMMA).  One carrying only
    emittances, NPART or a CHARGE is not — it must not override a BRHO."""
    return any(k in f.beam_attrs and f.beam_attrs[k] is not True
               for k in ("particle", "mass", "energy", "pc", "gamma"))


def _beam_particle(f: _Mad8File) -> tuple:
    """(mass [GeV], charge) of the BEAM particle alone — no energy needed, so
    ``ENERGY = BEAM[MASS] + …`` is well defined."""
    raw = f.beam_attrs
    pname = (_unquote(raw["particle"]).lower()
             if raw.get("particle") not in (None, True) else "")
    helix = _species_from_name(pname) if pname else None
    if pname in _MAD8_PARTICLES:
        m_gev, q = _MAD8_PARTICLES[pname]
    elif helix is not None:
        m_gev, q = helix.mass / 1000.0, helix.charge
    else:
        m_gev, q = _MAD8_PARTICLES["positron"]
    if raw.get("mass") not in (None, True):
        m_gev = f.resolve(raw["mass"])
    if raw.get("charge") not in (None, True):
        q = int(round(f.resolve(raw["charge"])))
    return m_gev, q


def _beam_kinematics(f: _Mad8File, warnings: list) -> dict:
    """The MAD8 BEAM particle and energy, MAD8 units (GeV, e): ``particle``
    (name or ""), ``mass``, ``charge``, ``energy`` (total), ``pc``,
    ``gamma``, ``beta``, plus the raw ``num`` values and the defaults
    applied (MAD8: positron, ENERGY = 1 GeV)."""
    num: dict = {}
    for k, v in f.beam_attrs.items():
        if k == "particle" and v is not True:
            num[k] = _unquote(v)
            continue
        if k not in _BEAM_RIGIDITY_KEYS or v is True:
            continue                  # emittances, NPART, flags: not needed
        try:
            num[k] = f.resolve(v)
        except (ValueError, ZeroDivisionError, TypeError, RecursionError) as exc:
            raise ValueError(f"MAD8: BEAM {k.upper()} = {v!r} could not be "
                             f"evaluated ({exc})") from exc
    pname = str(num.get("particle", "") or "").strip().strip('"').lower()
    helix = _species_from_name(pname) if pname else None
    if pname in _MAD8_PARTICLES:
        m_gev, q = _MAD8_PARTICLES[pname]
    elif helix is not None:
        m_gev, q = helix.mass / 1000.0, helix.charge
    else:
        m_gev, q = None, 1
    if "mass" in num:
        m_gev = float(num["mass"])
    if "charge" in num:
        q = int(round(float(num["charge"])))
    label = pname.upper()
    if m_gev is None:
        if pname and pname != "ion":
            raise ValueError(f"MAD8: BEAM PARTICLE={pname.upper()} is not a MAD8 "
                             "particle and no MASS is given")
        m_gev = _MAD8_PARTICLES["positron"][0]
        label = "POSITRON (MAD8 default)"
        warnings.append("MAD8: BEAM gives neither PARTICLE nor MASS — MAD8's "
                        "default particle, the positron, is assumed")
    if q == 0:
        raise ValueError("MAD8: BEAM CHARGE = 0 — a neutral beam has no rigidity")
    m_mev = m_gev * 1000.0
    if "energy" in num:
        total = float(num["energy"]) * 1000.0
    elif "pc" in num:
        total = math.hypot(float(num["pc"]) * 1000.0, m_mev)
    elif "gamma" in num:
        total = float(num["gamma"]) * m_mev
    else:
        total = 1000.0
        warnings.append("MAD8: BEAM has no ENERGY/PC/GAMMA — MAD8's default "
                        "ENERGY = 1 GeV (total) is assumed")
    if total <= m_mev:
        raise ValueError(f"MAD8: BEAM total energy {total / 1000:g} GeV is "
                         f"not above the particle mass {m_gev:g} GeV")
    pc = math.sqrt(total ** 2 - m_mev ** 2)
    return {"particle": label or "ION", "mass": m_gev, "charge": q,
            "energy": total / 1000.0, "pc": pc / 1000.0,
            "gamma": total / m_mev, "beta": pc / total, "num": num}


def _mad8_beam_reference(f: _Mad8File, warnings: list):
    """(ReferenceParticle, file_particle) from a MAD8 BEAM statement.
    ``file_particle`` is None when the BEAM particle is a HELIX species;
    otherwise ``{"mass_mev", "charge", "total_mev"}`` of the real particle,
    whose rigidity the magnets are converted with."""
    kin = _beam_kinematics(f, warnings)
    num, q, m_mev = kin["num"], kin["charge"], kin["mass"] * 1000.0
    match = None
    for cand in (PROTON, DEUTERON, H_MINUS):
        if cand.charge == q and abs(cand.mass - m_mev) < 0.5:
            match = cand
    if match is not None:
        # a HELIX species: its own mass (bit-compatible with the earlier
        # MAD-X-table path)
        if "energy" in num:
            total = float(num["energy"]) * 1000.0
        elif "pc" in num:
            total = math.hypot(float(num["pc"]) * 1000.0, match.mass)
        elif "gamma" in num:
            total = float(num["gamma"]) * match.mass
        else:
            total = kin["energy"] * 1000.0
        if total <= match.mass:
            raise ValueError(f"MAD8: BEAM total energy {total / 1000:g} GeV is "
                             f"not above the {match.name} mass")
        return (ReferenceParticle(species=match, w_kin=total - match.mass,
                                  frequency=162.5), None)
    brho_abs = kin["pc"] * 1e9 / 2.99792458e8 / abs(q)
    ref = _reference_at_brho(PROTON, brho_abs)
    warnings.append(
        f"MAD8: BEAM particle {kin['particle']} (m = {kin['mass']:.6g} GeV, "
        f"q = {q:+d}) is not a HELIX species — magnet strengths use its own "
        f"rigidity Bρ = {brho_abs:.6g} T·m; the tracking reference is a proton "
        f"at that rigidity (W = {ref.w_kin:.6g} MeV), so the magnetic optics "
        "match MAD8 while RF, time-of-flight and space-charge physics are "
        "those of a proton")
    return ref, {"mass_mev": m_mev, "charge": q,
                 "total_mev": kin["energy"] * 1000.0}


def _fallback_reference(fallback_beam) -> ReferenceParticle | None:
    if fallback_beam is None:
        return None
    if isinstance(fallback_beam, ReferenceParticle):
        return fallback_beam.copy()
    if isinstance(fallback_beam, (tuple, list)):
        sp_name, w = fallback_beam[0], fallback_beam[1]
    elif isinstance(fallback_beam, dict):
        sp_name, w = fallback_beam.get("species"), fallback_beam.get("energy")
    else:
        sp_name = getattr(fallback_beam, "species", None)
        w = getattr(fallback_beam, "energy", None)
    sp = sp_name if hasattr(sp_name, "mass") else _species_from_name(str(sp_name))
    if sp is None or w is None or float(w) <= 0.0:
        return None
    return ReferenceParticle(species=sp, w_kin=float(w), frequency=162.5)


def _resolve_reference(f: _Mad8File, brho_arg, species_name: str,
                       fallback_beam=None, warnings: list | None = None):
    """(ReferenceParticle, |brho|, source) from arg → BRHO param → fallback
    beam → error.  (The BEAM-statement path is handled by the caller.)"""
    warnings = f.warnings if warnings is None else warnings
    # species for a bare rigidity: the caller's choice, else the fallback
    # beam's (the beam that will be tracked), else H⁻ (the BTL lineage)
    fb0 = _fallback_reference(fallback_beam)
    if species_name:
        species = _species_from_name(species_name) or H_MINUS
    elif fb0 is not None:
        species = fb0.species
    else:
        species = H_MINUS
    brho_abs = None
    if brho_arg is not None:
        brho_abs = abs(float(brho_arg))
    elif "BRHO" in f.params:
        brho_abs = abs(f.param_val("BRHO"))
        ref = _reference_at_brho(species, brho_abs)
        if species is H_MINUS and ref.w_kin > _HMINUS_MAX_W_MEV:
            fb = _fallback_reference(fallback_beam)
            msg = (f"MAD8: the file's BRHO := {f.params['BRHO']} evaluates to "
                   f"{brho_abs:.6g} T·m, i.e. H⁻ at W = {ref.w_kin / 1000:.4g} GeV "
                   "— impossible for H⁻ (a unit slip, e.g. momentum in MeV/c with "
                   "c in cm/s)")
            if fb is None:
                raise ValueError(
                    msg + ".  Refusing to use it: pass the rigidity as "
                    "parse_mad8(..., brho=<T·m>), open the lattice from a "
                    "project whose beam is right, or fix BRHO in the file.")
            warnings.append(msg + f" — ignored; strengths converted with the "
                            f"caller's beam instead ({fb.species.name} at "
                            f"W = {fb.w_kin:.6g} MeV, Bρ = {_brho(fb):.6g} T·m)")
            return fb, _brho(fb), "fallback_beam"
    if brho_abs is None:
        fb = _fallback_reference(fallback_beam)
        if fb is not None:
            warnings.append(
                f"MAD8: the file declares no rigidity (no BEAM statement, no "
                f"BRHO parameter) — strengths converted with the caller's beam: "
                f"{fb.species.name} at W = {fb.w_kin:.6g} MeV "
                f"(Bρ = {_brho(fb):.6g} T·m).  Check that this is the beam the "
                "lattice was designed for.")
            return fb, _brho(fb), "fallback_beam"
        raise ValueError(
            "MAD8: no rigidity available — supply it as parse_mad8(..., "
            "brho=<T·m>), or declare `BRHO := <T·m>` in the file, or add "
            "a `BEAM, PARTICLE=..., ENERGY=...` statement, or open the "
            "lattice from a project (its beam is then used).  Refusing to "
            "guess: a wrong Bρ silently mis-scales every magnet.")
    return (_reference_at_brho(species, brho_abs), brho_abs,
            "argument" if brho_arg is not None else "brho_parameter")


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
# Beam-line selection and expansion
# ---------------------------------------------------------------------------

def _root_line(f: _Mad8File, strict: bool, line: str | None = None) -> str:
    if line is not None:
        name = line.upper()
        if name not in f.lines and name not in f.sequences:
            raise ValueError(f"MAD8: line {line!r} is not defined in the file")
        return name
    if f.use is not None:
        if f.use in f.lines or f.use in f.sequences:
            return f.use
        msg = f"MAD8: USE names {f.use!r}, which is not a LINE or SEQUENCE"
        if strict:
            raise ValueError(msg)
        f.warnings.append(msg + " — choosing the beam line automatically")
    referenced: set[str] = set()
    for _formals, items in f.lines.values():
        _refs(items, referenced)
    for seq in f.sequences.values():
        referenced.update(n for n, _at, _fr in seq["items"])
    for name, (base, raw) in f.elems.items():
        if f.rtype(name)[0] == "lump" and raw.get("line") not in (None, True):
            referenced.add(_unquote(raw["line"]).upper())
    roots = [n for n, (formals, _) in f.lines.items()
             if n not in referenced and not formals]
    roots += [n for n in f.sequences if n not in referenced]
    if not roots:
        raise ValueError("MAD8: no top-level LINE found (all lines are "
                         "referenced by other lines)")
    if len(roots) == 1:
        return roots[0]
    # Several unreferenced lines (saved sub-lines are common) — take the
    # one with the largest expansion.
    sized = []
    for r in roots:
        try:
            sized.append((_count(f, r), r))
        except (ValueError, RecursionError) as exc:
            f.warnings.append(f"MAD8: top-level line {r!r} cannot be expanded "
                              f"({exc}) — not a candidate")
    if not sized:
        raise ValueError("MAD8: no top-level LINE can be expanded")
    sized.sort(reverse=True)
    msg = (f"MAD8: {len(roots)} top-level LINEs; using the largest, "
           f"{sized[0][1]!r} ({sized[0][0]} entries); others: "
           f"{[r for _, r in sized[1:]]}")
    if strict:
        raise ValueError(msg)
    f.warnings.append(msg)
    return sized[0][1]


def _count(f: _Mad8File, name: str) -> int:
    """Number of element entries a line/sequence expands to."""
    names: list[str] = []
    _Expander(f, lambda n: names.append(n), strict=False, quiet=True).line(name)
    return len(names)


class _Expander:
    """Walks the beam-line tree.  ``emit(name)`` receives each element name
    in beam order; ``on_cell(name, i0_fn)`` bookkeeping is done by the
    caller through ``depth``/``parent`` hooks."""

    def __init__(self, f: _Mad8File, emit, *, strict: bool, quiet: bool = False,
                 on_line=None, seq_emit=None):
        self.f, self.emit, self.strict, self.quiet = f, emit, strict, quiet
        self.on_line = on_line          # (name, depth, parent, run) -> None
        self.seq_emit = seq_emit        # (drift_name, length_m) -> None
        self._stack: list[str] = []

    def line(self, name: str) -> None:
        self._ref(name, 1, False, None, {}, 0, name)

    def _ref(self, name, rep, neg, args, env, depth, parent):
        f = self.f
        if name in env:                               # formal line argument
            for _ in range(rep):
                self._items(env[name], neg, {}, depth, parent)
            return
        if name in f.lines:
            formals, items = f.lines[name]
            if name in self._stack:
                raise ValueError(f"MAD8: LINE {name!r} contains itself")
            sub_env = {}
            if formals:
                if not args or len(args) != len(formals):
                    raise ValueError(
                        f"MAD8: LINE {name} takes {len(formals)} argument(s), "
                        f"called with {len(args or [])}")
                # actual arguments are resolved in the CALLER's environment
                sub_env = {fm: self._bind(a, env) for fm, a in zip(formals, args)}
            for _ in range(rep):
                self._stack.append(name)
                try:
                    if self.on_line is not None:
                        self.on_line(name, depth, parent,
                                     lambda: self._items(items, neg, sub_env,
                                                         depth + 1,
                                                         name if depth == 1 else parent))
                    else:
                        self._items(items, neg, sub_env, depth + 1,
                                    name if depth == 1 else parent)
                finally:
                    self._stack.pop()
            return
        if name in f.sequences:
            for _ in range(rep):
                self._sequence(name, neg)
            return
        if name in f.elems:
            if f.rtype(name)[0] == "lump":
                ln = f.rtype(name)[1].get("line")
                if ln not in (None, True) and _unquote(ln).upper() in f.lines:
                    self._ref(_unquote(ln).upper(), rep, neg, None, {}, depth, parent)
                    return
            for _ in range(rep):
                self.emit(name)
            return
        if self.quiet:
            return
        msg = f"MAD8: {name!r} referenced but never defined"
        if self.strict:
            raise ValueError(msg)
        f.warnings.append(msg + " — skipped")

    def _bind(self, items, env):
        # substitute formal names inside an actual argument now
        out = []
        for it in items:
            if it[0] == "ref" and it[1] in env and it[4] is None:
                sub = env[it[1]]
                out.append(("group", sub, it[2], it[3]))
            else:
                out.append(it)
        return out

    def _items(self, items, reverse, env, depth, parent):
        for it in (reversed(items) if reverse else items):
            if it[0] == "group":
                _k, sub, rep, neg = it
                for _ in range(rep):
                    self._items(sub, neg ^ reverse, env, depth, parent)
            else:
                _k, name, rep, neg, args = it
                self._ref(name, rep, neg ^ reverse, args, env, depth, parent)

    def _sequence(self, name: str, reverse: bool) -> None:
        """Expand a SEQUENCE: members at their AT positions, drifts between."""
        f = self.f
        seq = f.sequences[name]
        if name in self._stack:
            raise ValueError(f"MAD8: SEQUENCE {name!r} contains itself")
        refer = seq["refer"]
        placed = []                                   # (start, length, member)
        centre_of: dict[str, float] = {}
        for member, at_raw, frm in seq["items"]:
            at = f.resolve(at_raw)
            if frm is not None:
                # FROM = relative to the earlier member's CENTRE, whatever
                # REFER is (MAD-X, pinned by tests/io/test_mad8_parser.py)
                if frm not in centre_of:
                    raise ValueError(f"MAD8: {member} FROM={frm} — {frm} is not "
                                     f"placed earlier in {name}")
                at += centre_of[frm]
            length = self._length(member)
            if refer == "entry":
                start = at
            elif refer == "exit":
                start = at - length
            else:
                start = at - length / 2.0
            centre_of[member] = start + length / 2.0
            placed.append((start, length, member))
        total = None
        if seq["length"] not in (None, True):
            total = f.resolve(seq["length"])
        pieces: list[tuple] = []
        cursor, k = 0.0, 0
        for start, length, member in placed:
            gap = start - cursor
            if gap < -1e-9 and not self.quiet:
                f.warnings.append(f"MAD8: SEQUENCE {name}: {member} overlaps the "
                                  f"previous member by {-gap:.6g} m — a negative "
                                  "drift is inserted")
            if abs(gap) > 1e-12:
                k += 1
                pieces.append(("drift", f"{name}_D{k}", gap))
            pieces.append(("member", member))
            cursor = start + length
        if total is not None and abs(total - cursor) > 1e-12:
            k += 1
            pieces.append(("drift", f"{name}_D{k}", total - cursor))
        self._stack.append(name)
        try:
            for p in (reversed(pieces) if reverse else pieces):
                if p[0] == "drift":
                    if self.seq_emit is not None:
                        self.seq_emit(p[1], p[2])
                    else:
                        self.emit(p[1])
                else:
                    self._ref(p[1], 1, reverse, None, {}, 99, name)
        finally:
            self._stack.pop()

    def _length(self, member: str) -> float:
        f = self.f
        if member in f.sequences:
            seq = f.sequences[member]
            if seq["length"] not in (None, True):
                return f.resolve(seq["length"])
            raise ValueError(f"MAD8: nested SEQUENCE {member} needs L= or "
                             "ENDSEQUENCE, AT=")
        if member in f.elems:
            raw = f.rtype(member)[1].get("l")
            return 0.0 if raw in (None, True) else f.resolve(raw)
        if member in f.lines:
            raise ValueError(f"MAD8: LINE {member} placed in a SEQUENCE — "
                             "not supported")
        raise ValueError(f"MAD8: {member!r} placed in a SEQUENCE but never defined")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_mad8(filepath: str, strict: bool = False, brho=None,
               species: str | None = None,
               auto_periods: bool = True, *, line: str | None = None,
               fallback_beam=None) -> tuple[Lattice, dict]:
    """Parse a MAD8 lattice file into a HELIX Lattice.

    Returns ``(lattice, metadata)`` with metadata keys ``"title"`` (the
    beam line used), ``"warnings"``, ``"reference"`` (ReferenceParticle
    used for the strength conversion), ``"rigidity_source"`` (``"beam"``,
    ``"argument"``, ``"brho_parameter"`` or ``"fallback_beam"``) and
    ``"periods"`` (auto-declared LATTICE brackets).

    ``line`` selects the beam line (default: the file's ``USE``, else the
    largest unreferenced LINE).  ``fallback_beam`` (a ``BeamConfig``, a
    ``(species, W_MeV)`` pair or a ``ReferenceParticle``) supplies the
    rigidity only when the file declares none — the GUI and the CLI pass
    the project's beam.  See the module docstring for the language subset
    and the Bρ / charge-sign conventions.
    """
    path = Path(filepath)
    warnings: list[str] = []
    f = _Mad8File(path.read_text(encoding="latin-1", errors="replace"),
                  warnings, base_dir=path.parent, strict=strict)

    # --- reference / rigidity -----------------------------------------
    file_particle = None
    if f.beam_attrs and not _beam_declares_particle(f):
        warnings.append("MAD8: the BEAM statement names no particle or energy "
                        "— not used for the rigidity")
    if _beam_declares_particle(f):
        ref, file_particle = _mad8_beam_reference(f, warnings)
        source = "beam"
        if brho is not None and abs(abs(float(brho)) - _brho(ref)) > 1e-3:
            warnings.append(
                f"MAD8: brho argument {float(brho):.4f} T·m disagrees with "
                f"the file BEAM statement ({_brho(ref):.4f} T·m) — using "
                "the BEAM statement")
    else:
        ref, _, source = _resolve_reference(f, brho, species, fallback_beam,
                                            warnings)
    brho_signed = _signed_brho(ref)
    warnings.append(
        f"MAD8 import: strengths converted with {ref.species.name} at "
        f"W = {ref.w_kin:.2f} MeV (Bρ = {_brho(ref):.4f} T·m, "
        f"G = sign(q)·K1·Bρ) — set the Beam tab to match before running.")

    # --- expand the root line with cell provenance ---------------------
    # Depth 0 = children of the root (machine sections); depth 1 = their
    # LINE-valued entries = candidate periodic cells.  Reversal (-NAME),
    # repetition (N*NAME), anonymous groups (transparent) and line
    # arguments are honoured at every depth.
    root = _root_line(f, strict, line)
    elements: list = []
    cells: list[dict] = []
    unsupported: dict[str, list] = {}
    local = {"ref": ref.copy(), "brho": brho_signed, "lcav": 0,
             "file": dict(file_particle) if file_particle else None}

    def emit(name: str) -> None:
        elems, _L = _build_mad8_element(f, name, local["brho"], strict,
                                        unsupported)
        elements.extend(elems)
        if f.rtype(name)[0] == "lcavity":
            # MAD8 normalises downstream strengths to the LOCAL momentum
            a = f.numeric_attrs(name, strict)
            dE = float(a.get("deltae", 0.0) or 0.0) * math.cos(
                2.0 * math.pi * float(a.get("phi0", 0.0) or 0.0))
            local["lcav"] += 1
            fp = local["file"]
            if fp is not None:
                # a substituted species: follow the REAL particle's momentum
                fp["total_mev"] += dE
                pc = math.sqrt(max(fp["total_mev"] ** 2 - fp["mass_mev"] ** 2, 0.0))
                local["brho"] = pc * 1e6 / 2.99792458e8 / abs(fp["charge"])
            else:
                r = local["ref"]
                r.w_kin = r.w_kin + dE
                local["brho"] = _signed_brho(r)

    def seq_drift(name: str, length_m: float) -> None:
        elements.append(Drift(name=name, length=length_m * _M_TO_MM))

    def on_line(name: str, depth: int, parent: str, run) -> None:
        if depth == 2:
            i0 = len(elements)
            run()
            cells.append({"name": name, "parent": parent,
                          "i0": i0, "i1": len(elements)})
        else:
            run()

    ex = _Expander(f, emit, strict=strict, on_line=on_line, seq_emit=seq_drift)
    ex.line(root)
    if local["lcav"] and file_particle is not None:
        warnings.append(
            "MAD8: LCAVITY with a substituted BEAM species — the magnets follow "
            "the real particle's momentum as in MAD8, but a tracked proton gains "
            "the same energy at a different momentum, so tracking through the "
            "cavities will not reproduce the MAD8 optics")
    if local["lcav"]:
        warnings.append(
            f"MAD8: {local['lcav']} LCAVITY element(s) change the reference "
            f"energy; magnets after each "
            "were converted with the local rigidity")

    for etype, items in unsupported.items():
        names = [n for n, _l, _a in items]
        thick = sum(1 for _n, l_mm, _a in items if l_mm != 0.0)
        detail = ""
        if etype in ("srot", "yrot") and any(
                float(a.get("angle", 0.0) or 0.0) != 0.0 for _n, _l, a in items):
            detail = (" — their coordinate rotations are NOT applied; the "
                      "elements after them are imported in the unrotated frame")
        warnings.append(
            f"MAD8: {len(items)} element(s) of unsupported type {etype!r} "
            f"({', '.join(sorted(set(names))[:8])}{'…' if len(set(names)) > 8 else ''}) "
            f"— {thick} imported as drifts of their length, "
            f"{len(items) - thick} as markers; their fields are not modelled{detail}")

    # RFCAVITY with HARMON: f = HARMON·βc/C over the used line
    gaps = [el for el in elements if hasattr(el, "_mad8_harmon")]
    if gaps:
        circ_m = sum(float(getattr(el, "length", 0.0) or 0.0)
                     for el in elements) / _M_TO_MM
        if file_particle is not None:          # the real particle's speed
            beta = math.sqrt(1.0 - (file_particle["mass_mev"]
                                    / (_beam_kinematics(f, [])["energy"] * 1000.0)) ** 2)
        else:
            beta = ref.bg / math.sqrt(1.0 + ref.bg ** 2)
        for el in gaps:
            h = el._mad8_harmon
            del el._mad8_harmon
            el.frequency = h * beta * 2.99792458e8 / circ_m / 1e6 if circ_m > 0 else 0.0
        warnings.append(
            f"MAD8: {len(gaps)} RFCAVITY element(s) set by HARMON — frequency "
            f"= HARMON·βc/C with C = {circ_m:.6g} m (the used line)")
    if any(hasattr(el, "_madx_raw") for el in elements):
        _convert_raw_matrices(elements_as_lattice(elements), ref, warnings)

    periods: list[dict] = []
    if auto_periods and cells:
        periods = _declare_periods(elements, cells, warnings)

    lattice = Lattice()
    for el in elements:
        lattice.add(el)
    # the beam the magnets were converted with — scans that re-parse the file
    # (GUI convergence tab) must use it, not whatever the Beam tab says later
    lattice.mad8_import_beam = {"species": ref.species.name, "energy": ref.w_kin}

    metadata = {
        "title": root,
        "warnings": list(dict.fromkeys(warnings)),
        "reference": ref,
        "rigidity_source": source,
        "periods": periods,
    }
    if f._calls:
        metadata["called_files"] = list(f._calls)
    return lattice, metadata


def elements_as_lattice(elements: list):
    """Minimal ``.elements`` holder for :func:`_convert_raw_matrices`."""
    class _L:
        pass
    h = _L()
    h.elements = elements
    return h


__all__ = ["parse_mad8"]
