"""In-house MAD-X lattice importer (subset parser).

Reads a MAD-X ``.madx`` / ``.seq`` file and builds a HELIX
:class:`~linac_gen.core.lattice.Lattice`, mirroring the
``(Lattice, metadata)`` return shape of
:func:`linac_gen.io.tracewin_parser.parse_tracewin`.  The existing
TraceWin parser is **not** touched — this is a parallel importer.

Supported subset (each rule follows MAD-X 5.09)
-----------------------------------------------
* Comments: ``! …``, ``// …``, and ``/* … */`` blocks (one left-to-right
  pass; comment marks inside quoted strings are text).
* Variables: ``name = expr ;`` (evaluated where written) and
  ``name := expr ;`` (evaluated where used), ``real``/``int``/``const``
  declarations; ``+ - * / ^`` arithmetic (``^`` left to right), comparisons
  in IF/WHILE conditions, MAD-X's constants (``pi``, ``twopi``, ``e``,
  ``clight``, ``degrad``, ``raddeg``, the particle masses, ``qelect``,
  ``hbar``, ``erad``, ``prad``, ``amu0``, ``version``) and deterministic
  functions (``sqrt``, ``abs``, ``exp``, ``log``, ``log10``, the
  trigonometric and hyperbolic functions, ``erf``, ``erfc``, ``floor``,
  ``ceil``, ``round``, ``frac``, ``sinc``), and ``elem->attr`` /
  ``beam->attr`` references.  A name never defined is 0 and a division by
  zero is 0, as in MAD-X (both reported); ``ranf``/``gauss``/``tgauss``
  are taken at their mean and ``table(…)`` reads 0 (reported).
* Element definitions ``name : TYPE, attr=val, … ;`` with ``:=``
  attributes, inheritance (``qd2: qd, k1=…`` copies ``qd`` when defined),
  and later changes (``qf->k1 = …``, ``qf, k1 := …``).
* ``SEQUENCE … ENDSEQUENCE`` with ``at=``/``at :=``, ``from=`` and
  ``refer=entry|centre|exit``; gaps between members become ``Drift``
  elements.  ``LINE = (…)`` beam lines: nesting, ``N*item``, reversal
  ``-item``, bracketed groups.
* ``BEAM`` (``particle``, ``mass``, ``charge``, ``energy`` / ``pc`` /
  ``gamma`` / ``beta`` / ``brho``; a later BEAM changes only what it
  gives), ``USE`` (positions and lengths as at the last USE of the line
  imported), ``OPTION, RBARC``.
* ``IF``/``ELSEIF``/``ELSE``, ``WHILE`` and ``MACRO``/``EXEC`` run as
  MAD-X runs them (a WHILE loop that iterates on TWISS, MATCH … or
  ``table(…)`` results is undone and reported); ``RETURN``, ``STOP``,
  ``EXIT`` and ``QUIT`` end the reading.

Element type mapping (all lengths converted metres → millimetres):

    DRIFT       -> Drift
    QUADRUPOLE  -> Quadrupole          (k1 -> gradient via Bρ; tilt -> skew;
                                        k1s turns it: sqrt(k1²+k1s²) at
                                        tilt − atan2(k1s, k1)/2)
    KICKER/HKICKER/VKICKER/TKICKER -> Steerer  (kick, hkick/vkick, chkick/
                                        cvkick -> ∫B·dl via signed Bρ; tilt
                                        rolls the kick)
    RCOLLIMATOR/ECOLLIMATOR -> Aperture (rectangular / circular)
    SBEND       -> Edge + Dipole + Edge
    RBEND       -> Edge + Dipole + Edge (e1,e2 get +angle/2; L is the
                                        chord, arc = L·(θ/2)/sin(θ/2))
                                       (k1 -> field index n = -k1·ρ²;
                                        tilt=±π/2 -> vertical bend;
                                        hgap/fint/fintx -> Edge gap/K1;
                                        k0/k2 kept for ring mode; k1s/k2s
                                        not modelled, reported)
    SEXTUPOLE   -> Multipole           (thin; drift-padded if l>0; tilt;
                                        k2s turns it: sqrt(k2²+k2s²) at
                                        tilt − atan2(k2s, k2)/3)
    MULTIPOLE   -> Multipole           (knl, ksl, tilt; a dipole component
                                        or ANGLE, which bends the reference
                                        orbit in MAD-X, is a kick about a
                                        straight reference here — reported)
    RFCAVITY    -> RFGap               (thin; drift-padded if l>0;
                                        lag -> φs = 360·lag − 90°, see
                                        the RFCAVITY branch for why;
                                        HARMON sets the frequency)
    SOLENOID    -> Solenoid            (ks -> field via Bρ)
    MARKER      -> Marker
    MONITOR/HMONITOR/VMONITOR -> Marker(is_bpm=True)
    MATRIX      -> MatrixElement       (rm/kick converted from MAD-X's
                                        canonical basis at the local energy)

Not supported (with a warning in ``metadata["warnings"]``): ``CALL`` (the
file is not read), ``LINE`` with formal arguments, sequences inside
sequences, the commands that compute (``TWISS``, ``MATCH``, ``TRACK``,
``SURVEY``, … — not run) and those that edit the machine or give it errors
(``SEQEDIT``, ``EALIGN``, ``EFCOMP``, … — not applied); an element type
without a HELIX counterpart keeps its length as a drift or marker.
"""
from __future__ import annotations

import ast
import contextlib
import keyword
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
from linac_gen.elements.rf_gap import RFFrequencyError, RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.matrix_element import MatrixElement

_M_TO_MM = 1000.0
_RAD_TO_DEG = 180.0 / math.pi
# a bend tilt this close to ±π/2 or ±π is taken as exact (decks write 1.5708)
_TILT_TOL = 1e-4

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

# MAD-X's other built-in constants (values of MAD-X 5.09, CODATA 2018 — the
# same constants HELIX uses; masses in GeV, as in MAD-X) and deterministic
# functions (its random-number functions RANF, GAUSS and TGAUSS cannot be
# reproduced).  Only the MAD-X importer knows them: the elegant importer
# shares this evaluator, and there these are ordinary names.
_MADX_CONSTANTS = {
    "degrad": 180.0 / math.pi,
    "raddeg": math.pi / 180.0,
    "pmass": 0.93827208816,
    "emass": 0.00051099895,
    "nmass": 0.93956542052,
    "umass": 0.93149410242,
    "mumass": 0.1056583755,
    "qelect": 1.602176634e-19,
    "hbar": 6.582119569e-25,
    "erad": 2.8179403262e-15,
    "prad": 2.8179403262e-15 * 0.00051099895 / 0.93827208816,
    "amu0": 4e-7 * math.pi,
    # the MAD-X release whose rules this importer follows (CERN decks test it:
    # ``if (version >= 50401) { option, -rbarc; }``)
    "version": 50902.0,
}
_MADX_FUNCS = {
    "log10": math.log10, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "erf": math.erf, "erfc": math.erfc, "floor": math.floor, "ceil": math.ceil,
    "round": lambda x: float(np.rint(x)),          # MAD-X rounds halves to even
    "frac": lambda x: math.modf(x)[0],             # keeps the sign: frac(-2.75) = -0.75
    "sinc": lambda x: math.sin(x) / x if x != 0.0 else 1.0,
}

# ``elem->attr`` references inside expressions; resolved through a callable
# that parse_madx stores in ``variables`` under a key no MAD-X name can take.
_ELEM_ATTR = re.compile(r"([A-Za-z_][\w.]*)->([A-Za-z_]\w*)")      # (no blanks: MAD-X)
_RESOLVER_KEY = "\x00elem_attr"
_STACK_KEY = "\x00deferred_stack"     # names of deferred variables being evaluated
_MADX_KEY = "\x00madx"                # present in parse_madx's scope: MAD-X's own names
# names MAD-X refuses to redefine ("attempt to redefine constant")
_MADX_PROTECTED = frozenset(("pi", "twopi", "e", "clight", *_MADX_CONSTANTS))
# commands parse_madx reads or reports: never taken for the name of an element
# whose attributes a ``name, attr=…`` statement changes
_COMMAND_WORDS = frozenset((
    "beam", "title", "use", "option", "set", "select", "exec", "return", "value",
    "show", "stop", "system", "print", "macro", "match", "track", "twiss", "survey",
    "plot", "sodd", "emit", "ibs", "aperture", "makethin", "ealign", "efcomp", "error",
    "seqedit", "endedit", "flatten", "install", "move", "remove", "eoption", "call",
    "setvars", "setvars_lin", "readtable", "readmytable", "fill", "assign", "save",
    "endsequence"))
# A MAD-X name may contain dots (``kqf.a12``); Python's parser cannot read
# them, so they are spelled with this marker while an expression is parsed.
_DOTTED = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.\w+)+)")
_DOT_MARK = "__dot__"


_NOTES_KEY = "\x00notes"              # parse_madx's scope: things to report once


def _madx_scope() -> dict:
    """A fresh variable table that knows MAD-X's built-in constants and
    functions (what :func:`parse_madx` evaluates expressions in), with the
    variables MAD-X predefines."""
    return {_MADX_KEY: True, _NOTES_KEY: {"div0": [], "random": set(), "table": [], "table0": []},
            "twiss_tol": 1e-6}


# MAD-X's random-number functions cannot be reproduced: they are taken at
# their mean (as the MAD8 importer does) and reported.
_MADX_RANDOM = {"ranf": 0.5, "gauss": 0.0, "tgauss": 0.0}
_KEYWORD_MARK = "__kw__"
_MACRO_DEPTH = 50                     # EXEC nesting refused beyond this (a macro calling itself)
_WHILE_LIMIT = 100_000                # passes of a WHILE loop before it is undone
_BEAM_ENERGY_KEYS = ("energy", "pc", "gamma", "beta", "brho")
_IDENT = re.compile(r"(?<![\w.])([A-Za-z_]\w*)")
# ``exec, name`` — also ``exec name`` and ``exec , name`` (MAD-X reads all three;
# ``exec.k = …`` is a variable)
_EXEC = re.compile(r"^\s*exec(?:\s*,\s*|\s+)(?![=:])(.+)$", re.IGNORECASE | re.DOTALL)
# one MAD-X token: ``elem->attr``, ``$name``, a number, a name, a quoted string or
# a single character
_MADX_TOKEN = re.compile(r"\s*([A-Za-z_][\w.]*->[A-Za-z_]\w*|\$[A-Za-z_][\w.]*"
                         r"|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][\w.$]*"
                         r"|\"[^\"]*\"|'[^']*'|\S)")
# commands whose results (tables, matched strengths) a deck may read back; the
# importer runs none of them
_RESULT_COMMANDS = frozenset((
    "twiss", "match", "survey", "track", "run", "emit", "dynap", "ibs", "aperture", "sodd",
    "touschek", "vary", "constraint", "lmdif", "migrad", "simplex", "jacobian", "endmatch",
    "correct", "usekick", "usemonitor", "coguess", "readtable", "readmytable", "setvars",
    "setvars_lin", "fill"))
_LOOP_KEY = "\x00loop"                 # parse_madx's scope: [WHILE loops under way]
_TABLES_KEY = "\x00tables"             # parse_madx's scope: [a table may exist in MAD-X]
# commands after which MAD-X has a table that table(…) reads
_TABLE_COMMANDS = frozenset((
    "twiss", "survey", "emit", "track", "dynap", "aperture", "touschek", "ibs", "sodd",
    "match", "create", "fill", "readtable", "readmytable", "call"))


class _NeedsMadx(Exception):
    """Raised inside a WHILE loop that would run TWISS, MATCH … or read
    ``table(…)``: what the loop iterates on is not computed here, so the loop
    is undone (deliberately not a ValueError, which the importer reports and
    passes over)."""


def _madx_int(x: float) -> float:
    """An INT as MAD-X makes one: truncated towards zero and held in 32 bits
    (``int k = 1e400`` is 2147483647)."""
    if math.isnan(x):
        return x
    return float(math.trunc(min(max(x, -2147483648.0), 2147483647.0)))


class _Deferred(str):
    """An ``attr := expr`` value (or a ``name := expr`` variable) kept as text
    and evaluated where it is used, with the values current at that moment
    (MAD-X semantics)."""


class _DeferredInt(_Deferred):
    """``int name := expr``: evaluated at use and truncated towards zero (MAD-X)."""


def _macro_arguments(text: str, value) -> list[str]:
    """EXEC's arguments as MAD-X 5.09 passes them: each parameter takes the
    next token — ``elem->attr``, a number, a name, a quoted string or one
    character — with the commas skipped: ``mk(0.02)`` passes ``0.02``,
    ``mk(1+1)`` passes ``1``, ``mk(qd->k1)`` passes ``qd->k1``.  A list
    that starts with a sign (or is empty) starts with 0: ``mk(-0.5)`` passes
    ``0``, ``mk(-0.5, 2)`` passes ``0`` and ``-``; a sign elsewhere passes
    itself (``mk(2, -0.5)`` passes ``2`` and ``-``).  ``$name`` passes the
    value of ``name`` as an INT (``value`` evaluates it)."""
    toks = [m.group(1) for m in _MADX_TOKEN.finditer(text) if m.group(1) != ","]
    if not toks and text.strip():
        return []                            # (``mk(,)``: MAD-X replaces nothing)
    if not toks or toks[0] in ("+", "-"):
        toks.insert(0, "0")
    out = []
    for t in toks:
        if t.startswith("$"):
            v = _madx_int(value(t[1:]))
            t = "0" if math.isnan(v) else str(int(v))
        out.append(t)
    return out


def _madx_source(expr: str, variables: dict) -> tuple[str, dict]:
    """A MAD-X expression rewritten for Python's parser: whitespace (line
    breaks included) collapsed, ``^`` as ``**``, ``elem->attr`` resolved,
    names that are Python keywords (``lambda``, ``in`` …) and dotted names
    (``kqf.a12``) spelled with markers that :func:`_eval_node` undoes."""
    src = " ".join(expr.split()).replace("^", "**")
    if "table" in src.lower():
        # table(twiss, r$end, betx): its arguments name a table, a row and a
        # column, never values (and may hold a '$')
        src = re.sub(r"(?i)(?<![\w.])table\s*\([^()]*\)", "table()", src)
    if "->" in src:
        resolver = variables.get(_RESOLVER_KEY)
        if resolver is None:
            raise ValueError(f"element-attribute reference in {expr!r} cannot be resolved here")
        local: dict = {}

        def _sub(m):
            key = f"__ea{len(local)}__"
            local[key] = resolver(m.group(1).lower(), m.group(2).lower())
            return key

        src = _ELEM_ATTR.sub(_sub, src)
        variables = {**variables, **local}
    src = _IDENT.sub(lambda m: m.group(1) + _KEYWORD_MARK if keyword.iskeyword(m.group(1))
                     else m.group(1), src)
    if "." in src:
        src = _DOTTED.sub(lambda m: m.group(1).replace(".", _DOT_MARK), src)
    return src, variables


def _in_parens(node, src: str) -> bool:
    j = node.col_offset - 1
    while j >= 0 and src[j] == " ":
        j -= 1
    return j >= 0 and src[j] == "("


def _madx_pow(node, src: str):
    """``a^b^c`` is ``(a^b)^c`` in MAD-X; Python's ``**`` groups the other
    way.  Chains written without parentheses are regrouped left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        chain = [node.left]
        r = node.right
        while isinstance(r, ast.BinOp) and isinstance(r.op, ast.Pow) and not _in_parens(r, src):
            chain.append(r.left)
            r = r.right
        chain.append(r)
        chain = [_madx_pow(c, src) for c in chain]
        out = chain[0]
        for c in chain[1:]:
            out = ast.BinOp(left=out, op=ast.Pow(), right=c)
        return out
    if isinstance(node, ast.BinOp):
        node.left, node.right = _madx_pow(node.left, src), _madx_pow(node.right, src)
    elif isinstance(node, ast.UnaryOp):
        node.operand = _madx_pow(node.operand, src)
    elif isinstance(node, ast.Call):
        node.args = [_madx_pow(a, src) for a in node.args]
    elif isinstance(node, ast.Compare):
        node.left = _madx_pow(node.left, src)
        node.comparators = [_madx_pow(c, src) for c in node.comparators]
    elif isinstance(node, ast.BoolOp):
        node.values = [_madx_pow(v, src) for v in node.values]
    return node


_LOGIC_OP = re.compile(r"&&|\|\|")
_TABLE_CALL = re.compile(r"(?i)(?<![\w.])table\s*\([^()]*\)")
_REL_OP = re.compile(r"==|<>|<=|>=|<|(?<!-)>")          # (``->`` is not a comparison)
_REL_FN = {"==": lambda a, b: a == b, "<>": lambda a, b: a != b,
           "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def _comparison(text: str, variables: dict, undefined: dict | None) -> bool:
    """One MAD-X comparison: exactly one of ==, <>, <, >, <=, >= between two
    expressions (``1 < a < 3``, ``(a == 0)`` and a bare ``a`` are not)."""
    ops = list(_REL_OP.finditer(text))
    if len(ops) != 1:
        raise ValueError(f"{text.strip()!r} is not one comparison")
    op = ops[0]
    lhs, rhs = text[:op.start()], text[op.end():]
    if not lhs.strip() or not rhs.strip():
        raise ValueError(f"{text.strip()!r} is not one comparison")
    return _REL_FN[op.group()](_eval_expr(lhs, variables, undefined),
                               _eval_expr(rhs, variables, undefined))


def _eval_condition(cond: str, variables: dict, undefined: dict | None = None,
                    notes: list | None = None) -> bool:
    """An IF/ELSEIF/WHILE condition, read as MAD-X 5.09 reads it (blanks
    between operator characters dropped: ``< =`` is ``<=``, ``| |`` is
    ``||``; ``a b`` stays two names): one comparison, or two
    joined by ``&&`` or ``||``.  The first part is taken first — ``false &&
    …`` is false and ``true || …`` true whatever follows; a first part that
    is not one comparison (``(a == 1)``, ``a``) counts as not false, so it
    makes ``… || …`` true and leaves ``… && …`` to the second (said in
    ``notes``).  Otherwise what follows must be one comparison.  Anything
    else (three comparisons the first does not decide, a lone ``(a == 0)``
    or ``a``, ``1 < a < 3``) is what MAD-X calls an illegal IF construct and
    sets false: a ValueError here, which the caller takes as false.
    A ``table(…)`` written in the condition reads 0, as in MAD-X, whatever
    tables exist (one reached through a variable or an element attribute is
    the table's value, as anywhere)."""
    cond = re.sub(r"(?<=[<>=&|])\s+(?=[<>=&|])", "", cond)
    if "table" in cond.lower():
        written = cond
        cond, n_table = _TABLE_CALL.subn("0", cond)
        log = (variables.get(_NOTES_KEY) or {}).get("table0") if n_table else None
        if log is not None:
            log.append(written)
    m = _LOGIC_OP.search(cond)
    if m is None:
        return _comparison(cond, variables, undefined)
    try:
        first = _comparison(cond[:m.start()], variables, undefined)
    except ValueError as exc:
        first = None
        if notes is not None:
            notes.append(f"{exc}: MAD-X takes the part before {m.group()} as not false")
    if m.group() == "||" and first is not False:
        return True
    if m.group() == "&&" and first is False:
        return False
    rest = cond[m.end():]
    if _LOGIC_OP.search(rest):
        raise ValueError("more than two comparisons that the first does not decide")
    return _comparison(rest, variables, undefined)


_IF_HEAD = re.compile(r"^\s*if\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
_ELSEIF_HEAD = re.compile(r"^\s*else\s*if\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
_ELSE_HEAD = re.compile(r"^\s*else\s*$", re.IGNORECASE)
_WHILE_HEAD = re.compile(r"^\s*while\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
_MACRO_HEAD = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(?:\(([^)]*)\))?\s*:\s*macro\s*=?\s*$",
                         re.IGNORECASE | re.DOTALL)        # (``name() : macro {`` too)


def _count_statements(nodes: list) -> int:
    n = 0
    for node in nodes:
        if node[0] in ("stmt", "unterminated"):
            n += 1
        elif node[0] == "if":
            n += sum(_count_statements(body) for _c, body in node[1])
        elif node[0] == "while":
            n += _count_statements(node[2])
        elif node[0] == "macro":
            n += 1
    return n


# MAD-X particle names: (mass in GeV, charge) — what ``beam->mass`` etc. read
_MADX_PARTICLES = {
    "proton": (0.93827208816, 1.0), "antiproton": (0.93827208816, -1.0),
    "electron": (0.00051099895, -1.0), "positron": (0.00051099895, 1.0),
    "posmuon": (0.1056583755, 1.0), "negmuon": (0.1056583755, -1.0),
    "ion": (0.93956542052, 1.0),           # (MAD-X 5.09: NMASS, not the atomic mass unit)
}


def _beam_value(beam: dict, attr: str, variables: dict, undefined: dict) -> float:
    """``beam->attr`` from the BEAM as read so far (MAD-X's default beam — a
    1 GeV positron — before any BEAM): mass, charge, energy, pc, gamma, beta,
    brho and any attribute the BEAM gave."""
    particle = str(beam.get("particle", "positron")).strip().lower()
    mass0, charge0 = _MADX_PARTICLES.get(particle, (0.93827208816, 1.0))

    def num(key, default):
        v = beam.get(key, default)
        if isinstance(v, _Deferred):
            return _eval_counted(str(v), variables, undefined)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    mass = num("mass", mass0)
    charge = num("charge", charge0)
    if "energy" in beam:
        energy = num("energy", 1.0)
    elif "pc" in beam:
        energy = math.hypot(num("pc", 0.0), mass)
    elif "gamma" in beam:
        energy = num("gamma", 1.0) * mass
    elif "beta" in beam:
        energy = mass / math.sqrt(1.0 - num("beta", 0.0) ** 2)
    elif "brho" in beam:
        energy = math.hypot(num("brho", 0.0) * C_LIGHT * 1e-9 * abs(charge), mass)
    else:
        energy = 1.0
    pc = math.sqrt(max(energy * energy - mass * mass, 0.0))
    derived = {"mass": mass, "charge": charge, "energy": energy, "pc": pc,
               "gamma": energy / mass, "beta": pc / energy,
               "brho": pc / (C_LIGHT * 1e-9 * abs(charge)) if charge else 0.0}
    if attr in derived:
        return derived[attr]
    return num(attr, 0.0)


def _matching_brace(text: str, i: int) -> int:
    depth = 0
    for m in re.finditer(r"[{}]", text[i:]):
        depth += 1 if m.group() == "{" else -1
        if depth == 0:
            return i + m.start()
    return len(text)


def _parse_program(text: str) -> list:
    """Statements and control blocks of a comment-stripped MAD-X file or
    macro body: ``("stmt", text)``, ``("if", [(condition, nodes), …, (None,
    nodes)])``, ``("while", condition, nodes)``, ``("macro", name, params,
    body text)``; ``("unterminated", text)`` for text that no ``;`` ends
    before a closing brace or the end — MAD-X does not run it;
    ``("unclosed", head)`` for a block never closed (MAD-X waits for its
    brace: it never runs); ``("stray_close",)`` for a ``}`` that closes no
    block (MAD-X reads nothing after it).  Without braces the statements are
    exactly :func:`_split_statements`'s but for a last piece no ``;`` ends."""
    return _parse_nodes(text, 0, False)[0]


_SPECIAL = re.compile(r"[;{}]")


def _parse_nodes(text: str, i: int, closing: bool) -> tuple[list, int]:
    nodes: list = []
    buf: list[str] = []
    vec = 0                                  # braces of a vector, knl={…}

    def flush(terminated: bool = True) -> None:
        stmt = "".join(buf).strip()
        buf.clear()
        if stmt:
            nodes.append(("stmt" if terminated else "unterminated", stmt))

    n = len(text)
    while i < n:
        m = _SPECIAL.search(text, i)         # (jump from one ; { } to the next)
        if m is None:
            buf.append(text[i:])
            i = n
            break
        j = m.start()
        buf.append(text[i:j])
        ch = text[j]
        if ch == ";":
            flush()
            vec = 0
            i = j + 1
            continue
        if ch == "{" and vec == 0:
            head = "".join(buf)
            mm = _MACRO_HEAD.match(head)
            if mm:
                k = _matching_brace(text, j)
                params = [q.strip().lower() for q in (mm.group(2) or "").split(",") if q.strip()]
                nodes.append(("macro", mm.group(1).lower(), params, text[j + 1:k]))
                if k >= n:
                    nodes.append(("unclosed", head.strip()))
                buf.clear()
                i = k + 1
                continue
            m_if, m_ei = _IF_HEAD.match(head), _ELSEIF_HEAD.match(head)
            m_el, m_wh = _ELSE_HEAD.match(head), _WHILE_HEAD.match(head)
            if m_if or m_ei or m_el or m_wh:
                body, i = _parse_nodes(text, j + 1, True)
                if i > n:
                    i = n
                buf.clear()
                if body and body[-1][0] == "eof":
                    # never closed: MAD-X waits for the brace to the end of the
                    # file, so the block — and the IF construct it ends — never runs
                    if (m_ei or m_el) and nodes and nodes[-1][0] == "if" \
                            and nodes[-1][1][-1][0] is not None:
                        nodes.pop()
                    nodes.append(("unclosed", head.strip()))
                    continue
                if m_ei or m_el:
                    if nodes and nodes[-1][0] == "if" and nodes[-1][1][-1][0] is not None:
                        nodes[-1][1].append((m_ei.group(1) if m_ei else None, body))
                    else:
                        nodes.append(("stray", head.strip()))
                else:
                    nodes.append(("while", m_wh.group(1), body) if m_wh
                                 else ("if", [(m_if.group(1), body)]))
                continue
        if ch == "{":
            vec += 1
        elif ch == "}":
            if vec > 0:
                vec -= 1
            elif closing:
                flush(terminated=False)      # (no ';' before the brace: MAD-X drops it)
                return nodes, j + 1
            else:
                end = text.find(";", j)
                whole = "".join(buf) + text[j:n if end < 0 else end]
                if whole.count("{") != whole.count("}"):
                    flush(terminated=False)
                    nodes.append(("stray_close",))   # (a '}' that closes no block)
                    i = j + 1
                    continue
                # (MAD-X reads on when the statement holds as many '{' as '}':
                # title, "a}b{c")
        buf.append(ch)
        i = j + 1
    flush(terminated=False)                  # (MAD-X runs no statement that ';' does not end)
    if closing:
        nodes.append(("eof",))               # (the block was never closed)
    return nodes, i


# MAD-X element classes.  A definition whose type is not one of these but the
# name of an earlier element (``qd2: qd, k1=…``) inherits that element's
# attributes, the explicit ones overriding.
_BASE_TYPES = frozenset((
    "drift", "quadrupole", "sbend", "rbend", "dipedge", "sextupole", "octupole",
    "multipole", "solenoid", "kicker", "hkicker", "vkicker", "tkicker",
    "rfcavity", "twcavity", "crabcavity", "rfmultipole", "marker", "monitor",
    "hmonitor", "vmonitor", "instrument", "placeholder", "rcollimator",
    "ecollimator", "collimator", "matrix", "elseparator", "srotation",
    "xrotation", "yrotation", "translation", "changeref", "beambeam", "nllens",
    "wire", "sequence", "line"))


def _eval_expr(expr: str, variables: dict, undefined: dict | None = None) -> float:
    """Evaluate a MAD-X arithmetic expression to a float.

    Only numeric literals, the whitelisted constants/functions, basic
    arithmetic, and references into ``variables`` are permitted — there
    is no general ``eval``.  Raises ValueError on anything else.
    ``elem->attr`` references are resolved through the element table when
    the caller provides one (inside :func:`parse_madx`).

    ``undefined``: with a dict, a name that is not defined is 0 — MAD-X's
    rule — and is counted there (name → uses) for the caller to report;
    without one, an unknown name raises ValueError.
    """
    madx = _MADX_KEY in variables
    if madx:
        src, variables = _madx_source(expr, variables)
        notes = variables.get(_NOTES_KEY)
        if notes is not None:
            notes["_current"] = expr
        if src.strip() in ("+", "-"):
            return 0.0                       # (MAD-X: ``kq = -`` sets 0)
    else:                    # (the elegant importer: the evaluator as it always was)
        src = expr.strip().replace("^", "**")
        if "->" in src:
            raise ValueError(f"element-attribute reference in {expr!r} cannot be resolved here")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse MAD-X expression {expr!r}: {exc}")
    return _eval_node(_madx_pow(tree.body, src) if madx else tree.body, variables, undefined)


def _eval_counted(expr: str, variables: dict, undefined: dict) -> float:
    """:func:`_eval_expr` under MAD-X's undefined-name rule; the undefined
    names are added to ``undefined`` only when the whole expression evaluates."""
    local: dict = {}
    val = _eval_expr(expr, variables, local)
    for name, n in local.items():
        undefined[name] = undefined.get(name, 0) + n
    return val


def _eval_node(node, variables: dict, undefined: dict | None = None) -> float:
    if isinstance(node, ast.Constant):           # 3.8+ numeric literal
        if isinstance(node.value, bool):
            return 1.0 if node.value else 0.0
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"unsupported literal {node.value!r}")
    if isinstance(node, ast.Name):
        key = node.id.lower().replace(_DOT_MARK, ".").replace(_KEYWORD_MARK, "")
        if key in _CONSTANTS:
            return _CONSTANTS[key]
        if key in _MADX_CONSTANTS and _MADX_KEY in variables:
            return _MADX_CONSTANTS[key]
        if key in variables:
            v = variables[key]
            if isinstance(v, _Deferred):            # ``name := expr``: evaluate at use
                stack = variables.setdefault(_STACK_KEY, set())
                if key in stack:
                    raise ValueError(f"deferred variable {key!r} refers to itself")
                stack.add(key)
                try:
                    val = _eval_expr(str(v), variables, undefined)
                finally:
                    stack.discard(key)
                return _madx_int(val) if isinstance(v, _DeferredInt) else val
            return float(v)
        if undefined is not None:
            undefined[key] = undefined.get(key, 0) + 1
            return 0.0
        raise ValueError(f"unknown MAD-X identifier "
                         f"{node.id.replace(_DOT_MARK, '.').replace(_KEYWORD_MARK, '')!r}")
    if isinstance(node, ast.BinOp):
        lhs = _eval_node(node.left, variables, undefined)
        rhs = _eval_node(node.right, variables, undefined)
        op = node.op
        if isinstance(op, ast.Add):
            return lhs + rhs
        if isinstance(op, ast.Sub):
            return lhs - rhs
        if isinstance(op, ast.Mult):
            return lhs * rhs
        # (the MAD-X importer reports arithmetic failures as ValueError; the
        # elegant importer, which shares this evaluator, keeps Python's own)
        madx = _MADX_KEY in variables
        if isinstance(op, ast.Div):
            if madx and rhs == 0.0:
                # MAD-X: "division by zero, result set to zero" (1/k with k never set)
                notes = variables.get(_NOTES_KEY)
                if notes is not None:
                    notes["div0"].append(notes.get("_current", ""))
                return 0.0
            return lhs / rhs
        if isinstance(op, ast.Pow):
            if not madx:
                return lhs ** rhs
            try:
                val = lhs ** rhs
            except (OverflowError, ZeroDivisionError) as exc:
                raise ValueError(f"MAD-X expression: {exc}") from exc
            if isinstance(val, complex):
                raise ValueError("MAD-X expression: negative number to a fractional power")
            return val
        raise ValueError("unsupported binary operator in MAD-X expression")
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, variables, undefined)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
        raise ValueError("unsupported unary operator in MAD-X expression")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id.lower()
        if fname in _MADX_RANDOM and _MADX_KEY in variables:
            notes = variables.get(_NOTES_KEY)
            if notes is not None:
                notes["random"].add(fname)
            return _MADX_RANDOM[fname]
        if fname == "table" and _MADX_KEY in variables:
            # table(summ, q1): a value from a table that TWISS, SURVEY, FILL …
            # make — this importer keeps none.  Before any such command MAD-X
            # reads 0 too (the table does not exist); after one the value is
            # unknown here, and a WHILE loop that would iterate on it is
            # undone instead (both reported).
            made = (variables.get(_TABLES_KEY) or [False])[0]
            if made and (variables.get(_LOOP_KEY) or [0])[0]:
                raise _NeedsMadx("it reads table(…)")
            notes = variables.get(_NOTES_KEY)
            if notes is not None:
                notes.setdefault("table" if made else "table0", []).append(
                    notes.get("_current", ""))
            return 0.0
        func = _FUNCS.get(fname)
        if func is None and _MADX_KEY in variables:
            func = _MADX_FUNCS.get(fname)
        if func is None:
            raise ValueError(f"unsupported MAD-X function {fname!r}")
        args = [_eval_node(a, variables, undefined) for a in node.args]
        if _MADX_KEY not in variables:
            return float(func(*args))
        try:
            return float(func(*args))
        except (TypeError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError(f"MAD-X function {fname}: {exc}") from exc
    raise ValueError("unsupported construct in MAD-X expression")


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove ``! …``, ``// …`` line comments and ``/* … */`` blocks
    (comment markers inside quoted strings, e.g. a ``TITLE``, are kept).

    One left-to-right pass, as MAD-X reads: whichever marker comes first
    wins, so a ``//****`` line is a line comment and cannot open a ``/*``
    block that would swallow the following lines."""
    out = []
    i, n = 0, len(text)
    quote = None
    while i < n:
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == quote or ch == "\n":     # (an unclosed quote ends with its line)
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "!" or text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j            # keep the newline
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
            continue
        if ch == "\n":
            quote = None                      # quotes never span lines
        out.append(ch)
        i += 1
    return "".join(out)


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


def _parse_attributes(body: str, variables: dict, *, defer: bool = False,
                      undefined: dict | None = None) -> dict:
    """Parse ``attr=val, attr=val, flag`` into a dict.

    Bare flags (no ``=``) map to ``True``.  Vector values
    ``{a,b,c}`` become lists of floats.  Quoted strings keep their text.
    With ``defer``, scalar ``attr := expr`` values are kept as
    :class:`_Deferred` text for evaluation at use.

    ``undefined`` (element definitions): the numeric attributes — all but
    the text ones in ``_TEXT_ATTRS`` — follow MAD-X's rule that a name never
    defined is 0, and such names are counted in it.  Without it (commands
    such as BEAM or USE, whose values may be names) a value that is not a
    number is kept as text.
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
        deferred = key.rstrip().endswith(":")
        key = key.rstrip(":").strip().lower()
        val = val.strip()
        if defer and deferred and not val.startswith('"'):
            attrs[key] = _Deferred(val)      # scalars and {vectors}: evaluated at use
            continue
        if val.startswith('"') and val.endswith('"'):
            attrs[key] = val[1:-1]
        elif val.startswith("{") and val.endswith("}"):
            inner = val[1:-1]
            attrs[key] = [
                _eval_expr(tok, variables) if undefined is None
                else _eval_counted(tok, variables, undefined)
                for tok in inner.split(",") if tok.strip()
            ]
        else:
            try:
                if undefined is not None and key not in _TEXT_ATTRS:
                    attrs[key] = _eval_counted(val, variables, undefined)
                else:
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
    elif "beta" in beam_attrs:
        total_MeV = mass_MeV / math.sqrt(1.0 - float(beam_attrs["beta"]) ** 2)
    elif "brho" in beam_attrs:
        pc_MeV = float(beam_attrs["brho"]) * C_LIGHT * 1e-6 * abs(species.charge or 1)
        total_MeV = math.sqrt(pc_MeV ** 2 + mass_MeV ** 2)
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


def _fold_skew(k: float, ks: float, tilt: float, order: int) -> tuple[float, float]:
    """A normal and a skew component (k1/k1s of a quadrupole, ``order`` 2;
    k2/k2s of a sextupole, ``order`` 3) as one normal component turned about
    s, as MAD-X takes them: sqrt(k² + ks²) at tilt − atan2(ks, k)/order.
    For k < 0 the same field is written as −sqrt(k² + ks²) at tilt −
    atan2(−ks, −k)/order (a component turned by π/order changes sign): the
    sign of k is kept and a small ks moves the tilt a little."""
    r = math.hypot(k, ks)
    if k < 0.0:
        return -r, tilt - math.atan2(-ks, -k) / order
    return r, tilt - math.atan2(ks, k) / order


def _thin_bend_note(names: list) -> str:
    """The warning for MULTIPOLEs that bend the reference orbit in MAD-X."""
    uniq = list(dict.fromkeys(names))
    which = (f"MULTIPOLE {uniq[0]!r} has" if len(uniq) == 1 else
             f"{len(uniq)} MULTIPOLEs (e.g. {', '.join(uniq[:4])}"
             f"{' …' if len(uniq) > 4 else ''}) have")
    return (f"{which} a dipole component or an ANGLE: MAD-X bends the reference orbit there "
            "(by ANGLE, else by KNL[0]; in TWISS also by KSL[0], vertically) — a thin bend.  "
            "HELIX's Multipole has no curvature: its KNL[0]/KSL[0] kick acts about a straight "
            "reference and ANGLE is not used, so the reference path does not bend there (the "
            "closed orbit and the dispersion differ from MAD-X's, and the survey for ANGLE or "
            "KNL[0])")


def _build_element(name: str, etype: str, attrs: dict, brho: float,
                   warnings: list, *, rbend_chord: bool = True,
                   k2_sink: list | None = None,
                   thin_bend_sink: list | None = None) -> tuple[list, float]:
    """Map one MAD-X element to HELIX element(s).  Lengths returned in mm.

    ``rbend_chord`` — MAD-X ``OPTION, RBARC`` (default true): an RBEND's
    ``l`` is the straight chord and the arc is ``l·(θ/2)/sin(θ/2)``.
    ``OPTION, RBARC=false`` (LHC-style sequences) and the MAD8 importer
    pass ``False`` so ``l`` is taken as the arc length directly.

    A bend's sextupole component ``K2`` and dipole component ``K0`` are kept
    on the HELIX ``Dipole`` as ``k2``/``k0`` (set only when given) for ring
    mode; the linac maps ignore them.  ``k2_sink`` collects the names of such
    bends so the caller can report them once; ``thin_bend_sink`` the names
    of MULTIPOLEs that MAD-X takes as thin bends (without it each is
    reported here).
    """
    etype = etype.lower()
    l_mm = _gf(attrs, "l") * _M_TO_MM
    aperture_mm = _gf(attrs, "aperture") * _M_TO_MM   # MAD-X aperture rarely used

    if etype == "drift":
        return [Drift(name=name, length=l_mm, aperture=aperture_mm)], l_mm

    if etype == "quadrupole":
        # k1 [1/m²] -> gradient [T/m] = k1 · Bρ; a skew component k1s turns
        # the quadrupole (_fold_skew; pinned against MAD-X's R matrix)
        k1, tilt = _gf(attrs, "k1"), _tilt_rad(attrs, math.pi / 4.0)
        if _gf(attrs, "k1s"):
            k1, tilt = _fold_skew(k1, _gf(attrs, "k1s"), tilt, 2)
        gradient = k1 * brho
        # tilt [rad] about s -> HELIX skew angle [deg]
        skew_deg = tilt * _RAD_TO_DEG
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
        # skew components: no HELIX counterpart (MAD-X 5.09's TWISS leaves a
        # bend's K1S and K2S out as well: the same map and orbit without them)
        skew = [f"{k.upper()}={_gf(attrs, k):g}" for k in ("k1s", "k2s") if _gf(attrs, k)]
        if skew:
            many = len(skew) > 1
            _warn_once(warnings, f"bend {name!r}: {', '.join(skew)} (skew "
                                 f"component{'s' if many else ''}) not modelled — dropped "
                                 f"(MAD-X's TWISS leaves {'them' if many else 'it'} out too)")
        if abs(angle_rad) < 1e-12:
            # Zero-angle bend is just a drift — unless it carries k1,
            # which a straight HELIX Dipole cannot hold (ρ = ∞).
            if abs(_gf(attrs, "k1")) > 0.0:
                warnings.append(
                    f"bend {name!r}: angle=0 with k1={_gf(attrs, 'k1'):g} "
                    "— imported as a plain drift, the k1 focusing is lost")
            if k2_sink is not None and _gf(attrs, "k2") != 0.0:
                warnings.append(
                    f"bend {name!r}: angle=0 with k2={_gf(attrs, 'k2'):g} "
                    "— imported as a plain drift, the k2 is lost")
            return [Drift(name=name, length=l_mm,
                          aperture=aperture_mm)], l_mm
        # tilt = ±π/2 is a vertical bend (HELIX hv=1): +π/2 bends towards
        # +y, −π/2 towards −y — the sign goes into the angle; tilt = ±π
        # bends the other way in the horizontal plane.  Decks write these
        # rounded (tilt = 1.5708, 3.1416): within _TILT_TOL they are taken
        # as exact, with a note.  Any other tilt has no HELIX counterpart.
        tilt = _tilt_rad(attrs, math.pi / 2.0)
        hv = 0
        flip = False                           # the angle's sign turned by the tilt
        d90, d180 = abs(abs(tilt) - math.pi / 2.0), abs(abs(tilt) - math.pi)
        if d90 <= _TILT_TOL:
            hv = 1
            flip = tilt < 0
        elif d180 <= _TILT_TOL:
            flip = True
        elif abs(tilt) > 1e-12:
            warnings.append(
                f"bend {name!r}: tilt={tilt:g} rad is none of 0, ±π/2, ±π "
                "— imported as a horizontal bend (hv=0): a bend rolled by any other "
                "angle has no HELIX counterpart")
        if min(d90, d180) <= _TILT_TOL and min(d90, d180) >= 1e-9:
            target = "π/2" if d90 <= _TILT_TOL else "π"
            warnings.append(
                f"bend {name!r}: tilt={tilt:.9g} rad taken as ±{target} (off by "
                f"{min(d90, d180):.2g} rad; that residual roll is not modelled)")
        if flip:
            angle_rad = -angle_rad
            e1, e2 = -e1, -e2
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
        # integral (default 0; fintx >= 0 overrides the exit face, a negative
        # fintx — MAD-X's default, -1 — is fint); HELIX Edge takes the FULL
        # gap [mm] and K1 (= fint).
        edge_kw_in: dict = {}
        edge_kw_out: dict = {}
        if "hgap" in attrs:
            gap_mm = 2.0 * _gf(attrs, "hgap") * _M_TO_MM
            fint = _gf(attrs, "fint") if "fint" in attrs else 0.0
            fintx = _gf(attrs, "fintx") if "fintx" in attrs else fint
            if fintx < 0.0:
                fintx = fint
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
        # Ring-mode-only components (attributes set only when present, so a
        # bend without them is the same object as before).
        # (a tilt that turned the angle's sign turns these too)
        k2 = _gf(attrs, "k2")
        if k2:
            body.k2 = -k2 if flip else k2
            if k2_sink is not None:
                k2_sink.append(name)
        if "k0" in attrs and not isinstance(attrs["k0"], (str, bool)):
            body.k0 = -_gf(attrs, "k0") if flip else _gf(attrs, "k0")
            if k2_sink is not None:
                k2_sink.append(name)
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
        # k2 [1/m³], integrated k2l = k2·l (HELIX/MAD-X knl convention); a
        # skew component k2s turns the sextupole (_fold_skew; pinned against
        # MAD-X TRACK).  tilt [rad] -> tilt_deg (same sense as MULTIPOLE /
        # QUADRUPOLE); a bare ``tilt`` is the sextupole's natural skew, pi/6
        k2, tilt = _gf(attrs, "k2"), _tilt_rad(attrs, math.pi / 6.0)
        if _gf(attrs, "k2s"):
            k2, tilt = _fold_skew(k2, _gf(attrs, "k2s"), tilt, 3)
        k2l = k2 * _gf(attrs, "l")
        tilt_deg = tilt * _RAD_TO_DEG
        mp = Multipole(name=name, knl=[0.0, 0.0, k2l], aperture=aperture_mm,
                       tilt_deg=tilt_deg)
        return _drift_pad(mp, l_mm, name)

    if etype == "multipole":
        knl = attrs.get("knl", []) or []
        ksl = attrs.get("ksl", []) or []
        knl = [float(x) for x in knl] if isinstance(knl, list) else []
        ksl = [float(x) for x in ksl] if isinstance(ksl, list) else []
        # MAD-X tilt [rad] -> Multipole.tilt_deg: the same rotation sense
        # (positive = the sense of Quadrupole.skew_angle).  A bare ``tilt``
        # flag has no natural angle for a general multipole and is ignored.
        tilt_raw = attrs.get("tilt", 0.0)
        if tilt_raw is True:
            warnings.append(f"MAD-X: {name}: a bare 'tilt' on a MULTIPOLE has no "
                            "natural angle for a general multipole — ignored")
        tilt_deg = 0.0 if isinstance(tilt_raw, bool) else _gf(attrs, "tilt") * _RAD_TO_DEG
        # A dipole component or an ANGLE makes the multipole a thin bend in
        # MAD-X (_thin_bend_note); the HELIX Multipole has no curvature
        if _gf(attrs, "angle") or _gf(attrs, "hxl") or (knl and knl[0]) or (ksl and ksl[0]):
            if thin_bend_sink is not None:
                thin_bend_sink.append(name)
            else:
                _warn_once(warnings, _thin_bend_note([name]))
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
        # As MAD-X: an HKICKER (VKICKER) kicks by KICK, or by HKICK (VKICK)
        # when KICK is zero — a deck may give either; KICKER and TKICKER by
        # HKICK and VKICK; CHKICK (CVKICK), a corrector's setting, adds to the
        # horizontal (vertical) kick; TILT rolls the kick about s.  A bare
        # flag — KICK, TILT, ... without a value — is 0.
        def kick(key):
            if attrs.get(key) is True:
                _warn_once(warnings, f"kicker {name!r}: {key.upper()} without a value is 0 "
                                     "(as MAD-X reads it)")
                return 0.0
            return _gf(attrs, key)
        refused = {"hkicker": ("vkick", "cvkick"), "vkicker": ("hkick", "chkick"),
                   "kicker": ("kick",), "tkicker": ("kick",)}[etype]
        for key in refused:
            if key in attrs:
                _warn_once(warnings, f"kicker {name!r}: MAD-X refuses {key.upper()} on "
                                     f"{etype.upper()} — not read")
        if etype == "hkicker":
            hkick, vkick = kick("kick"), 0.0
            if not hkick and "hkick" in attrs:
                hkick = kick("hkick")
            if "chkick" in attrs:
                hkick += kick("chkick")
        elif etype == "vkicker":
            hkick, vkick = 0.0, kick("kick")
            if not vkick and "vkick" in attrs:
                vkick = kick("vkick")
            if "cvkick" in attrs:
                vkick += kick("cvkick")
        else:
            hkick, vkick = kick("hkick"), kick("vkick")
            if "chkick" in attrs:
                hkick += kick("chkick")
            if "cvkick" in attrs:
                vkick += kick("cvkick")
        if attrs.get("tilt") is True:
            _warn_once(warnings, f"kicker {name!r}: TILT without a value is 0 (as MAD-X reads it)")
        tilt = _tilt_rad(attrs, 0.0)
        if tilt:
            c, s = math.cos(tilt), math.sin(tilt)
            hkick, vkick = hkick * c - vkick * s, hkick * s + vkick * c
            # (what a tilt of pi/2 leaves of the other plane is rounding: no kick)
            big = max(abs(hkick), abs(vkick))
            hkick = 0.0 if abs(hkick) <= 1e-15 * big else hkick
            vkick = 0.0 if abs(vkick) <= 1e-15 * big else vkick
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


def _resolve_def(name: str, element_defs: dict, _seen: tuple = ()) -> tuple[str, dict]:
    """(base MAD-X class, attributes) of element ``name``, following the
    parent chain of ``child: parent, attr=…`` definitions; the child's
    explicit attributes override the parent's."""
    etype, attrs = element_defs[name]
    if etype in _BASE_TYPES or etype not in element_defs:
        return etype, attrs
    if etype in _seen or etype == name:
        # a cycle: leave it unresolved — the builder reports the unknown type
        # and keeps the geometry, as the importer always did
        return etype, attrs
    ptype, pattrs = _resolve_def(etype, element_defs, _seen + (name,))
    merged = dict(pattrs)
    merged.update(attrs)
    return ptype, merged


def _expand_line(name: str, lines: dict, element_defs: dict, warnings: list) -> list[str]:
    """Flat element-name list of MAD-X ``LINE`` ``name``: nested lines,
    ``N*item``, ``-item`` (reversed order) and parenthesised groups.  MAD-X
    reverses the ORDER of a reflected line; it does not turn elements round."""
    out: list[str] = []
    stack: list[str] = []

    def ref(nm: str, rep: int, neg: bool) -> None:
        if nm in lines:
            if nm in stack:
                raise ValueError(f"MAD-X: LINE {nm!r} contains itself")
            stack.append(nm)
            try:
                for _ in range(rep):
                    items(lines[nm], neg)
            finally:
                stack.pop()
        elif nm in element_defs:
            out.extend([nm] * rep)
        else:
            warnings.append(f"LINE member {nm!r} is undefined — skipped")

    def items(its, reverse: bool) -> None:
        # MAD-X 5.09 drops a bracketed group written right after a sub-line
        # (``cell, (c, a)`` is ``cell``: it reads the group as arguments of that
        # line); the same machine is built here, and the deck is pointed at
        kept, prev = [], None
        for it in its:
            if it[0] == "group" and prev is not None and it[2] == 1 and not it[3]:
                _warn_once(warnings, f"LINE: the bracketed group after sub-line {prev!r} is "
                                     "dropped, as MAD-X does (it reads the group as arguments "
                                     "of that line) — check the deck")
                prev = None
                continue
            kept.append(it)
            prev = it[1].lower() if it[0] != "group" and it[1].lower() in lines else None
        for it in (reversed(kept) if reverse else kept):
            if it[0] == "group":
                _k, sub, rep, neg = it
                for _ in range(rep):
                    items(sub, neg ^ reverse)
            else:
                _k, nm, rep, neg, args = it
                if args:
                    warnings.append(f"LINE member {nm.lower()!r}: line arguments are not "
                                    "supported — skipped")
                    continue
                ref(nm.lower(), rep, neg ^ reverse)

    ref(name, 1, False)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_madx(filepath: str) -> tuple[Lattice, dict]:
    """Parse a MAD-X ``.madx`` / ``.seq`` file into a HELIX Lattice.

    Returns ``(lattice, metadata)`` where ``metadata`` has keys
    ``"title"`` (str), ``"warnings"`` (list[str]), ``"reference"``
    (the :class:`ReferenceParticle` built from the BEAM command, the
    importer's default without one — the beam the magnets are scaled with)
    and ``"reference_declared"`` (True when the deck's BEAM gives the
    energy: ENERGY, PC, GAMMA, BETA or BRHO).
    """
    text = _strip_comments(Path(filepath).read_text(encoding="latin-1"))
    program = _parse_program(text)

    variables: dict = _madx_scope()
    notes: dict = variables[_NOTES_KEY]
    loop_depth = variables[_LOOP_KEY] = [0]      # WHILE loops under way
    tables_made = variables[_TABLES_KEY] = [False]   # a TWISS, CREATE … (or CALL) read
    element_defs: dict = {}        # name -> (type, attrs)
    sequences: dict = {}          # name -> {"l": float, "members": [...]}
    beam_attrs: dict = {}          # the BEAM, as later BEAM commands update it
    state = {"rbarc": True,        # MAD-X OPTION, RBARC (default true)
             "use": None, "title": "", "cur_seq": None, "call": False, "exec_depth": 0,
             "return": False}      # a RETURN read: no statement after the current one runs
    # the deck as it was at the last USE of each sequence/line (variables,
    # element definitions, OPTION RBARC): MAD-X expands the sequence there,
    # freezing member positions and element lengths
    use_vars: dict = {}
    warnings: list[str] = []
    lines: dict = {}               # name -> LINE items (mad8_parser._parse_items form)
    # names used but never defined (MAD-X takes them as 0): name -> uses
    undef_names: dict = {}
    # cavity definitions the reference RF frequency is read from (see below)
    rf_seen: dict = {}
    macros: dict = {}              # name -> (parameters, body text)
    consts: set = set()            # names declared CONST (MAD-X keeps their first value)
    redefined: set = set()         # implicit re-definitions inside a SEQUENCE (ignored)

    _attr_stack: set = set()

    def _elem_attr(ename: str, attr: str) -> float:
        # ``elem->attr`` inside an expression (MAD-X: an unset numeric attribute is 0)
        if ename == "beam" and ename not in element_defs:
            return _beam_value(beam_attrs, attr, variables, undef_names)
        if ename not in element_defs:
            # MAD-X: an element never defined reads as 0
            undef_names[f"{ename}->{attr}"] = undef_names.get(f"{ename}->{attr}", 0) + 1
            return 0.0
        etype_, a = _resolve_def(ename, element_defs)
        v = a.get(attr, 0.0)
        if isinstance(v, _Deferred):
            key = (ename, attr)
            if key in _attr_stack:
                raise ValueError(f"{ename}->{attr} refers to itself through deferred expressions")
            _attr_stack.add(key)
            try:
                v = _eval_counted(str(v), variables, undef_names)   # deferred: evaluated at use
            finally:
                _attr_stack.discard(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{ename}->{attr} is not numeric ({v!r})")
        if attr == "l" and etype_ == "rbend" and state["rbarc"]:
            # MAD-X keeps an RBEND's length as its arc under RBARC (the default)
            ang = a.get("angle", 0.0)
            ang = (_eval_counted(str(ang), variables, undef_names) if isinstance(ang, _Deferred)
                   else float(ang or 0.0))
            if abs(ang) > 1e-12:
                v = float(v) * (ang / 2.0) / math.sin(ang / 2.0)
        return float(v)

    variables[_RESOLVER_KEY] = _elem_attr

    _ASSIGN = re.compile(r"^([A-Za-z_][\w.]*)\s*:?=\s*(.+)$", re.DOTALL)
    # ``real``/``int``/``const`` declarations: ``const lq = 5;``, ``real const k = 2;``
    _DECL = re.compile(r"^\s*(?:(real|int)\s+)?(const\s+)?(?=[A-Za-z_][\w.]*\s*:?=)",
                       re.IGNORECASE)
    # (a class may be a dotted name: ``qf.1: qf.c, k1=…`` inherits from ``qf.c``)
    _DEFINE = re.compile(r"^([A-Za-z_][\w.]*)\s*:\s*([A-Za-z_][\w.]*)\s*"
                         r"(?:,(.*))?$", re.DOTALL)
    # ``name : CLASS attr = …`` without the comma after the class (MAD-X reads it)
    _DEFINE_NOCOMMA = re.compile(r"^([A-Za-z_][\w.]*)\s*:\s*([A-Za-z_][\w.]*)\s+"
                                 r"([A-Za-z_]\w*\s*:?=.*)$", re.DOTALL)

    def _match_define(s: str):
        m = _DEFINE.match(s)
        if m:
            return m.group(1), m.group(2), m.group(3) or ""
        m = _DEFINE_NOCOMMA.match(s)
        if m and m.group(2).lower() != "line":
            return m.group(1), m.group(2), m.group(3)
        return None

    def _define(ename: str, etype: str, attrs: dict) -> None:
        """Store a definition.  ``child: parent`` copies the parent's
        attributes NOW (MAD-X clones the parent when the child is defined;
        later changes to the parent do not reach the child)."""
        en, et = ename.lower(), etype.lower()
        if et not in _BASE_TYPES and et in element_defs and et != en:
            pt, pa = _resolve_def(et, element_defs)
            merged = dict(pa)
            merged.update(attrs)
            element_defs[en] = (pt, merged)
        else:
            element_defs[en] = (et, attrs)

    _LINE_DEF = re.compile(r"^([A-Za-z_][\w.]*)\s*:\s*line\s*=\s*\((.*)\)\s*$",
                           re.IGNORECASE | re.DOTALL)
    _LINE_ARGS = re.compile(r"^[A-Za-z_][\w.]*\s*\([^)]*\)\s*:\s*line\b", re.IGNORECASE)
    _ATTR_SET = re.compile(r"^([A-Za-z_][\w.]*)->([A-Za-z_]\w*)\s*:?=\s*(.+)$",
                           re.DOTALL)

    class _EndOfInput(Exception):
        """STOP, EXIT or QUIT: MAD-X reads no further."""

    def _member(ename: str, attrs: dict) -> None:
        # a SEQUENCE member: its position (``at`` may be deferred) and ``from``
        sequences[state["cur_seq"]]["members"].append(
            (ename.lower(), attrs.get("at", float("nan")),
             str(attrs["from"]).lower() if "from" in attrs else None))

    def _is_assignment(stmt: str) -> bool:
        # ``x = …``, ``x := …``, ``real/int/const x = …`` (a comma or a colon
        # before the '=' makes it a member or a definition instead)
        mdecl = _DECL.match(stmt)
        s = stmt[mdecl.end():] if mdecl else stmt
        lhs = s.split("=", 1)[0]
        return bool(_ASSIGN.match(s)) and "," not in lhs and ":" not in lhs.rstrip(":")

    def _seq_command(stmt: str) -> bool:
        # a command (VALUE, PRINT, OPTION …) written inside a SEQUENCE block
        m = re.match(r"\s*([A-Za-z_][\w.]*)\s*(?:,|$)", stmt)
        w = m.group(1).lower() if m else ""
        return w in _COMMAND_WORDS and w != "endsequence" and w not in element_defs

    def _statement(stmt: str) -> None:
        low = stmt.lower().strip()
        mend = re.fullmatch(r"(return|stop|exit|quit)\s*(?:,.*)?", low, re.DOTALL)
        if mend and mend.group(1) == "return":
            # MAD-X finishes the statement under way — the rest of an IF body,
            # every pass of a WHILE, a macro body — and then reads no further
            state["return"] = True
            return
        if mend:
            raise _EndOfInput(mend.group(1))
        mexec = _EXEC.match(stmt)
        if mexec:
            _exec(mexec.group(1).strip())
            return

        # --- inside a SEQUENCE block ----------------------------------
        # (assignments, ``elem->attr = …`` and commands run there as they do
        # anywhere — MAD-X; everything else is a member)
        if state["cur_seq"] is not None:
            if low == "endsequence" or low.startswith("endsequence"):
                state["cur_seq"] = None
                return
            if not (_is_assignment(stmt) or _ATTR_SET.match(stmt) or _seq_command(stmt)):
                # A member line: ``elemname, at=POS [, attrs]`` or an inline
                # definition ``name: TYPE, at=POS, …``.
                mdef = _match_define(stmt)
                if mdef:
                    ename, etype, body = mdef
                    attrs = _parse_attributes(body, variables, defer=True,
                                              undefined=undef_names)
                    en = ename.lower()
                    if etype.lower() == en and en in element_defs:
                        pass          # ``name: name, at=…`` places the element already defined
                    elif en in element_defs:
                        # MAD-X: "implicit element re-definition ignored"
                        if en not in redefined:
                            redefined.add(en)
                            warnings.append(f"implicit re-definition of {ename!r} inside a "
                                            "SEQUENCE ignored (as in MAD-X): the first "
                                            "definition is placed")
                    else:
                        _define(ename, etype, attrs)
                        rf_seen[en] = etype.lower() == "rfcavity"
                    _member(ename, attrs)
                    return
                # Reference to an already-defined element.
                head, _, body = stmt.partition(",")
                _member(head.strip(), _parse_attributes(body, variables, defer=True,
                                                        undefined=undef_names))
                return

        # --- beam line: ``name: LINE = (a, 2*b, -(c, d), …)`` ----------
        mline = _LINE_DEF.match(stmt)
        if mline:
            from linac_gen.io.mad8_parser import _parse_items
            body = mline.group(2).replace("&", " ")       # MAD8 continuation mark: ignored
            # members separated only by blanks are separate members (as in MAD-X)
            body = re.sub(r"(?<=[\w.'])\s+(?=[A-Za-z_])", ",", body)
            body = re.sub(r"\s+", " ", body)               # groups may span lines
            lines[mline.group(1).lower()] = _parse_items(body)
            return
        if _LINE_ARGS.match(stmt):
            warnings.append(f"LINE with formal arguments is not supported — skipped: "
                            f"{stmt[:60]!r}")
            return

        # --- attribute assignment: ``elem->attr = value`` --------------
        mattr = _ATTR_SET.match(stmt)
        if mattr:
            ename, attr = mattr.group(1).lower(), mattr.group(2).lower()
            if ename not in element_defs:
                warnings.append(f"{ename}->{attr} assignment: element {ename!r} is not "
                                "defined at this point — skipped")
                return
            expr = mattr.group(3)
            self_ref = any(m.group(1).lower() == ename and m.group(2).lower() == attr
                           for m in _ELEM_ATTR.finditer(expr))
            if re.match(r"^[A-Za-z_][\w.]*\s*->\s*[A-Za-z_]\w*\s*:=", stmt) and not self_ref:
                val = _Deferred(expr.strip())
            else:
                try:
                    val = _eval_counted(expr, variables, undef_names)
                except ValueError as exc:
                    warnings.append(f"skipped {ename}->{attr} assignment: {exc}")
                    return
            et, at_ = element_defs[ename]
            at_ = dict(at_)
            at_[attr] = val
            element_defs[ename] = (et, at_)
            return

        # --- variable assignment (incl. deferred :=) ------------------
        # Only treat as an assignment when the LHS is a single identifier
        # and there is no comma before the '=' (which would make it a
        # command with attributes).
        mdecl = _DECL.match(stmt)
        is_const = bool(mdecl and mdecl.group(2))
        is_int = bool(mdecl and (mdecl.group(1) or "").lower() == "int")
        if mdecl and mdecl.end():
            stmt = stmt[mdecl.end():]
        massign = _ASSIGN.match(stmt)
        if massign and "," not in stmt.split("=", 1)[0] \
                and ":" not in stmt.split("=", 1)[0].rstrip(":"):
            name, expr = massign.group(1), massign.group(2)
            key = name.lower()
            if key in _MADX_PROTECTED or key in consts:
                # (the evaluator always reads the built-in or first value)
                kind = "built-in constant" if key in _MADX_PROTECTED else "constant"
                warnings.append(f"{name} = {expr.strip()[:40]}: attempt to redefine the "
                                f"{kind} ignored (as in MAD-X)")
                return
            if re.match(r"^\s*[A-Za-z_][\w.]*\s*:=", stmt) and not is_const:
                # evaluated at use (an INT then truncated)
                variables[key] = (_DeferredInt if is_int else _Deferred)(expr.strip())
                return
            try:
                val = _eval_counted(expr, variables, undef_names)
            except ValueError as exc:
                warnings.append(f"skipped variable {name!r}: {exc}")
                return
            # ``int n = 3.7`` is 3, ``int n = -3.7`` is -3, ``int n = 1e400`` is
            # 2147483647 (MAD-X: truncated, 32 bits; a later ``n = 4.6`` is 4.6)
            variables[key] = _madx_int(val) if is_int else val
            if is_const:
                consts.add(key)
            return

        # --- definition: ``name : TYPE, …`` --------------------------
        # (``attr := expr`` deferred attributes included: they used to make
        # the whole definition vanish without a word.  A plain ``name := expr``
        # never matches _DEFINE — TYPE must be an identifier.)
        mdef = _match_define(stmt)
        if mdef:
            ename, etype, body = mdef
            etype_low = etype.lower()
            if etype_low == ename.lower() and ename.lower() in element_defs:
                # ``qf: qf, k1=…`` at top level: MAD-X keeps the element as it was
                warnings.append(f"definition of {ename!r} as itself is ignored (as in MAD-X)")
                return
            attrs = _parse_attributes(body, variables, defer=etype_low != "line",
                                      undefined=undef_names)
            if etype_low == "sequence":
                state["cur_seq"] = ename.lower()
                sequences[state["cur_seq"]] = {
                    "l": attrs.get("l", 0.0),           # (``l := circ``: evaluated at use)
                    "refer": str(attrs.get("refer", "centre")).lower(),
                    "members": [],
                }
            elif etype_low == "line":
                warnings.append(
                    f"sequence {ename!r}: LINE=(...) form not supported "
                    "(only SEQUENCE) — skipped"
                )
            else:
                _define(ename, etype_low, attrs)
                rf_seen[ename.lower()] = etype_low == "rfcavity" and ":=" not in stmt
            return

        # --- commands -------------------------------------------------
        head, _, body = stmt.partition(",")
        cmd = head.strip().lower()
        if cmd in _TABLE_COMMANDS or cmd.startswith("ptc_"):
            tables_made[0] = True        # (from here on MAD-X may have a table to read)
        if loop_depth[0] and (cmd in _RESULT_COMMANDS or cmd.startswith("ptc_")):
            # inside a WHILE loop: MAD-X would iterate on what this computes
            raise _NeedsMadx(f"it runs {cmd.upper()}")
        if cmd == "beam":
            # a later BEAM changes only what it gives (MAD-X): an energy
            # replaces the earlier one, a particle its mass and charge
            new = _parse_attributes(body, variables)
            if any(k in new for k in _BEAM_ENERGY_KEYS):
                for k in _BEAM_ENERGY_KEYS:
                    beam_attrs.pop(k, None)
            if "particle" in new:
                for k in ("mass", "charge"):
                    if k not in new:
                        beam_attrs.pop(k, None)
            beam_attrs.update(new)
        elif cmd == "title":
            state["title"] = body.strip().strip('"') or state["title"]
        elif cmd == "use":
            use_attrs = _parse_attributes(body, variables)
            seq = use_attrs.get("sequence") or use_attrs.get("period")
            if seq and str(seq).lower() not in sequences and str(seq).lower() not in lines:
                warnings.append(f"USE of {str(seq)!r} before it is defined — MAD-X skips it "
                                "(\"unknown sequence skipped\"): it fixes no positions"
                                + ("; it still names the line imported" if state["use"] is None
                                   else ""))
                if state["use"] is None:
                    state["use"] = str(seq).lower()
            elif seq:
                state["use"] = str(seq).lower()
                use_vars[state["use"]] = (dict(variables), dict(element_defs), state["rbarc"])
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
                        state["rbarc"] = False
                    elif val is None:
                        state["rbarc"] = True
                    else:
                        state["rbarc"] = val.lower() not in ("false", "0", "no")
        elif cmd in element_defs and cmd not in _COMMAND_WORDS:
            # ``qf, k1 := kq, l = 0.5``: new values for attributes of an element
            # already defined, as ``qf->k1 := kq; qf->l = 0.5`` (CERN decks set
            # their strengths this way); elements cloned from it earlier keep
            # what they copied
            et, at_ = element_defs[cmd]
            at_ = dict(at_)
            at_.update(_parse_attributes(body, variables, defer=True, undefined=undef_names))
            element_defs[cmd] = (et, at_)
        elif cmd in ("set", "select", "value", "show", "system", "print"):
            pass    # benign, ignored silently
        elif cmd == "exec":
            _exec(body.strip())
        elif cmd in ("setvars", "setvars_lin"):
            warnings.append(f"{cmd.upper()} (variables from a table row) is not executed — the "
                            "variables it sets are missing here")
        elif cmd in ("macro",) or low.endswith("macro"):
            warnings.append("MACRO definition without a {…} body — skipped")
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
            state["call"] = True
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

    def _exec(spec: str) -> None:
        # ``exec, name;`` / ``exec, name(a, b);``: the MACRO's statements with
        # every occurrence of each parameter replaced by its argument — plain
        # text, inside names too (``lhcBIM`` with BIM = b1 is ``lhcb1``), one
        # parameter after the other, as MAD-X does
        m = re.match(r"^\$?([A-Za-z_][\w.]*)\s*(?:\((.*)\))?\s*$", spec, re.DOTALL)
        name = m.group(1).lower() if m else spec
        if name not in macros:
            warnings.append(f"EXEC {spec!r}: no such MACRO in this deck — not executed")
            return
        if state["exec_depth"] >= _MACRO_DEPTH:
            warnings.append(f"EXEC {spec!r}: macros nested more than {_MACRO_DEPTH} deep — "
                            "not executed")
            return
        params, body = macros[name]
        args = []
        if m and m.group(2) is not None:
            args = _macro_arguments(m.group(2),
                                    lambda nm: _eval_counted(nm, variables, undef_names))
        if len(args) < len(params):
            warnings.append(f"EXEC {spec!r}: MACRO {name!r} takes {len(params)} argument(s)")
        elif len(args) > len(params) and params:
            warnings.append(f"EXEC {spec!r}: MAD-X passes one name or number per parameter — "
                            f"{' '.join(args[len(params):])!r} ignored")
        for prm, arg in zip(params, args, strict=False):
            if prm:
                body = re.sub(re.escape(prm), lambda _m, a=arg: a, body, flags=re.IGNORECASE)
        state["exec_depth"] += 1
        try:
            _execute(_parse_program(body))
        finally:
            state["exec_depth"] -= 1

    def _save() -> tuple:
        return (dict(variables), dict(element_defs),
                {k: {**v, "members": list(v["members"])} for k, v in sequences.items()},
                dict(beam_attrs), dict(lines), dict(macros), dict(rf_seen), dict(state),
                dict(use_vars), dict(undef_names), set(consts), set(redefined),
                {k: (type(v)(v) if isinstance(v, (list, set)) else v) for k, v in notes.items()})

    def _restore(saved: tuple) -> None:
        for live, kept in zip((variables, element_defs, sequences, beam_attrs, lines, macros,
                               rf_seen, state, use_vars, undef_names, consts, redefined, notes),
                              saved, strict=True):
            live.clear()
            live.update(kept)

    def _while(cond: str, body: list) -> None:
        # passes run until the condition is false; a loop that would run TWISS,
        # MATCH … or read table(…) — what MAD-X iterates on and this importer
        # does not compute — or that is still true after _WHILE_LIMIT passes
        # is undone: every value is as it was before the loop
        saved = _save()
        n_warn = len(warnings)
        passes = 0
        loop_depth[0] += 1
        try:
            while True:
                try:
                    cond_notes: list = []
                    ok = _eval_condition(cond, variables, undef_names, cond_notes)
                    for msg in cond_notes:
                        _warn_once(warnings, f"WHILE condition {cond.strip()!r}: {msg}")
                except ValueError as exc:
                    warnings.append(f"WHILE condition {cond.strip()!r}: {exc} — taken as "
                                    "false, as MAD-X does")
                    break
                if not ok:
                    break
                if passes == _WHILE_LIMIT:
                    _restore(saved)
                    del warnings[n_warn:]
                    warnings.append(f"WHILE ({cond.strip()}) still true after {_WHILE_LIMIT} "
                                    "passes: the loop is undone — the values are those before it")
                    return
                _execute(body)
                passes += 1
        except _NeedsMadx as why:
            _restore(saved)
            del warnings[n_warn:]
            warnings.append(f"WHILE ({cond.strip()}) not run: {why}, which this importer does "
                            "not compute — the values are those before the loop (MAD-X "
                            "iterates)")
            return
        finally:
            loop_depth[0] -= 1
        # one copy of each warning the passes repeated
        seen = set(warnings[:n_warn])
        tail = []
        for w in warnings[n_warn:]:
            if w not in seen:
                seen.add(w)
                tail.append(w)
        warnings[n_warn:] = tail

    def _execute(nodes: list) -> None:
        for node in nodes:
            kind = node[0]
            if kind == "stmt":
                _statement(node[1])
            elif kind == "if":
                for cond, body in node[1]:
                    if cond is not None:
                        try:
                            cond_notes: list = []
                            ok = _eval_condition(cond, variables, undef_names, cond_notes)
                            for msg in cond_notes:
                                _warn_once(warnings, f"IF condition {cond.strip()!r}: {msg}")
                        except ValueError as exc:
                            # MAD-X: "illegal if construct set false" — the next
                            # branch (ELSEIF, ELSE) is then tried
                            warnings.append(f"IF condition {cond.strip()!r}: {exc} — taken as "
                                            "false, as MAD-X does")
                            ok = False
                        if not ok:
                            continue
                    _execute(body)
                    break
            elif kind == "while":
                _while(node[1], node[2])
            elif kind == "macro":
                macros[node[1]] = (node[2], node[3])
            elif kind == "stray":
                warnings.append(f"{node[1]!r} block without a preceding IF — skipped")
            elif kind == "stray_close":
                raise _EndOfInput("}")
            elif kind == "unterminated":
                warnings.append(f"{node[1][:60]!r} is not ended by ';' — not run, as in MAD-X")
            elif kind == "unclosed":
                warnings.append(f"the block opened by {node[1]!r} is never closed — it is not "
                                "run (MAD-X waits for its closing brace to the end of the file)")

    def _not_read(k_node: int, what: str) -> None:
        rest = _count_statements(program[k_node + 1:])
        if rest:
            warnings.append(f"{what}: the {rest} statement(s) after it are not read")

    for k_node, node in enumerate(program):
        try:
            _execute([node])
        except _EndOfInput as stop:
            _not_read(k_node, "a '}' that closes no block ends reading in MAD-X"
                      if str(stop) == "}" else f"{str(stop).upper()} ends MAD-X")
            break
        if state["return"]:
            _not_read(k_node, "RETURN ends the file in MAD-X")
            break
    rbarc = state["rbarc"]
    use_sequence = state["use"]
    title = state["title"]
    has_call = state["call"]

    # --- choose the sequence (or beam LINE) to expand ------------------
    line_name: str | None = None
    if use_sequence and use_sequence not in sequences and use_sequence in lines:
        line_name = use_sequence
    elif not sequences:
        if not lines:
            raise ValueError(
                "MAD-X file contains no SEQUENCE block — nothing to import"
            )
        line_name = list(lines.keys())[-1]
        warnings.append(
            (f"USE {use_sequence!r} not found — " if use_sequence else "no USE command — ")
            + f"expanding the last-defined LINE {line_name!r}")
    if line_name is None:
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
    # positions and lengths are those of the last USE of the line imported
    # (MAD-X expands it there: a later ``pq = …`` does not move ``at := pq``);
    # without a USE, those at the end of the deck
    # Positions and lengths are those of the last USE of the line imported:
    # MAD-X expands it there, and a later ``pq = …``, ``q->l = …``, bend
    # angle or OPTION RBARC moves nothing.  Strengths are those at the end of
    # the deck.  Without a USE, the end of the deck is used for everything.
    use_snap = use_vars.get(line_name if line_name is not None else seq_name)

    @contextlib.contextmanager
    def _at_use():
        # the deck as it was at the USE while positions and lengths are read
        if use_snap is None:
            yield
            return
        now = (dict(variables), dict(element_defs), state["rbarc"])
        for live, kept in zip((variables, element_defs), use_snap[:2], strict=True):
            live.clear()
            live.update(kept)
        state["rbarc"] = use_snap[2]
        try:
            yield
        finally:
            for live, kept in zip((variables, element_defs), now[:2], strict=True):
                live.clear()
                live.update(kept)
            state["rbarc"] = now[2]

    def _length_m(ename: str):
        # an element's length as MAD-X's expanded line has it (an RBEND's arc
        # under RBARC), or None
        if ename not in element_defs:
            return None
        try:
            return _elem_attr(ename, "l")
        except ValueError:
            return None

    use_len: dict = {}                   # name -> length at USE, where it differs from the end
    if use_snap is not None:
        if line_name is None:
            names = {n for n, _a, _f in sequences[seq_name]["members"]}
        else:
            names = set(_expand_line(line_name, lines, element_defs, []))
        end_len = {n: _length_m(n) for n in names}
        with _at_use():
            for n in names:
                lu = _length_m(n)
                if lu is not None and end_len[n] is not None and lu != end_len[n]:
                    use_len[n] = lu

    def _defn(ename: str, sink: list | None = None) -> tuple[str, dict]:
        # inherited attributes, then the deferred (``:=``) expressions,
        # evaluated now that every variable is known (MAD-X evaluates
        # deferred expressions at use); the length is that of the USE
        etype, attrs = _resolve_def(ename, element_defs)
        out = _late_eval(ename, attrs, variables, warnings if sink is None else sink,
                         undef_names)
        if ename in use_len:
            out = {**out, "l": use_len[ename]}
        return etype, out

    def _chord(ename: str) -> bool:
        # (a length from the USE is already an RBEND's arc)
        return rbarc and ename not in use_len

    # --- reference particle + rigidity --------------------------------
    # RF frequency for the ReferenceParticle: take the first RFCAVITY's
    # freq if present, else a harmless default (MAD-X lattices are often
    # magnet-only).
    # It is read, as it always was, from the first RFCAVITY defined with a
    # FREQ — among the definitions this importer always read: written as
    # RFCAVITY (not through a parent), and at top level without ``:=`` (it
    # skipped those statements whole).  A deck without one takes the first
    # cavity of the beam line (_finish_rf_and_k2).
    freq_MHz = 0.0
    ref_cav_attrs = None
    for en, (_etype, attrs) in element_defs.items():
        if not rf_seen.get(en):
            continue
        a = _late_eval(en, attrs, variables, [], {})    # (an unused cavity reports nothing)
        if _gf(a, "freq") > 0:
            freq_MHz = _gf(a, "freq")
            ref_cav_attrs = a
            break
    freq_placeholder = freq_MHz <= 0
    if freq_placeholder:
        freq_MHz = 352.21        # benign placeholder; no RF ⇒ unused
    reference = _reference_from_beam(beam_attrs, freq_MHz, warnings)
    # (a BEAM with only a particle, or no BEAM, leaves the energy a default)
    reference_declared = any(k in beam_attrs for k in _BEAM_ENERGY_KEYS)
    # Charge-signed: K1/KS -> lab-frame G/B needs sign(q) (see
    # _signed_brho).  For negative species (H-) this flips every
    # imported gradient relative to the old unsigned conversion — the
    # old behaviour was wrong for H- decks; proton decks are unchanged.
    brho = _signed_brho(reference)

    lattice = Lattice()
    k2_bends: list[str] = []
    thin_bends: list[str] = []          # MULTIPOLEs MAD-X takes as thin bends
    rf_built: list[tuple] = []          # (RFGap, harmon, freq as written)

    def _report_import_notes() -> None:
        if thin_bends:
            warnings.append(_thin_bend_note(thin_bends))
        if notes["div0"]:
            ex = list(dict.fromkeys(e.strip() for e in notes["div0"] if e.strip()))
            warnings.append(
                f"{len(notes['div0'])} division(s) by zero set to 0, as MAD-X does "
                f"(e.g. {', '.join(repr(e) for e in ex[:3])})")
        if notes["random"]:
            warnings.append(
                f"random-number function(s) {', '.join(sorted(notes['random']))} taken at their "
                "mean (RANF 0.5, GAUSS and TGAUSS 0): MAD-X draws random numbers that cannot "
                "be reproduced — the design machine is built")
        if notes["table"]:
            ex = list(dict.fromkeys(e.strip() for e in notes["table"] if e.strip()))
            warnings.append(
                f"{len(notes['table'])} table(…) value(s) read as 0 after TWISS, SURVEY, "
                "CREATE … or a CALL: this importer keeps no tables, so MAD-X's value is unknown "
                f"here (e.g. {', '.join(repr(e) for e in ex[:3])})")
        if notes["table0"]:
            ex = list(dict.fromkeys(e.strip() for e in notes["table0"] if e.strip()))
            warnings.append(
                f"{len(notes['table0'])} table(…) value(s) read as 0, as MAD-X does there: no "
                "table made yet, or an IF/WHILE condition (where MAD-X reads 0) "
                f"(e.g. {', '.join(repr(e) for e in ex[:3])})")
        if not undef_names:
            return
        names = list(undef_names)
        hint = (" — the deck CALLs files this importer does not read, which may define them"
                if has_call else "")
        warnings.append(
            f"{len(names)} name(s) used where not (yet) defined, taken as 0 there as MAD-X "
            f"does: "
            f"{', '.join(names[:12])}{' …' if len(names) > 12 else ''}{hint}")

    def _note_rf(etype: str, attrs: dict, elems: list) -> None:
        if etype == "rfcavity":
            for el in elems:
                if isinstance(el, RFGap):
                    rf_built.append((el, _gf(attrs, "harmon"), _gf(attrs, "freq")))

    if line_name is not None:
        # --- beam LINE: elements end to end -----------------------------
        line_w: list[str] = []
        names = _expand_line(line_name, lines, element_defs, line_w)
        if any("line arguments are not supported" in w for w in line_w):
            raise ValueError(
                f"MAD-X LINE {line_name!r} calls a line with arguments "
                f"({[w.split(chr(39))[1] for w in line_w if 'arguments' in w][0]!r}) — not "
                "supported by this importer; expand the call in the deck")
        missing = sorted({w.split("'")[1] for w in line_w if "is undefined" in w})
        if missing:
            # MAD-X stops on an undefined line member; a beam line with holes
            # would import as a silently shorter machine (e.g. a deck whose
            # magnets live in a CALLed file, which this importer does not read)
            raise ValueError(
                f"MAD-X LINE {line_name!r} uses {len(missing)} undefined element(s): "
                f"{', '.join(missing[:10])}{' …' if len(missing) > 10 else ''} — defined in a "
                "CALLed file?  CALL is not read: inline the file or export a flattened deck")
        warnings.extend(line_w)
        if not names:
            raise ValueError(f"MAD-X LINE {line_name!r} expands to no elements")
        seen_w: set = set()
        for ename in names:
            local_w: list[str] = []
            etype, attrs = _defn(ename, local_w)
            if etype == "multipole" and _gf(attrs, "l") != 0.0:
                # MAD-X: a MULTIPOLE is thin — its L is set to 0 (with a warning)
                local_w.append(f"multipole {ename!r}: L = {_gf(attrs, 'l'):g} m ignored — "
                               "a MULTIPOLE is thin (as in MAD-X)")
                attrs = dict(attrs)
                attrs["l"] = 0.0
            elems, _total = _build_element(ename, etype, attrs, brho, local_w,
                                           rbend_chord=_chord(ename), k2_sink=k2_bends,
                                           thin_bend_sink=thin_bends)
            l_attr = _gf(attrs, "l")
            if _total == 0.0 and l_attr > 0.0:
                # an element type the builder does not model came back as a
                # zero-length marker; end to end, that would shorten the
                # machine (a SEQUENCE keeps the geometry through its gaps)
                elems = list(elems) + [Drift(name=f"{ename}_body", length=l_attr * _M_TO_MM)]
                local_w.append(f"element {ename!r}: type {etype!r} is not modelled — kept as a "
                               f"drift of its length ({l_attr:g} m)")
            for w in local_w:                   # one copy per repeated element
                if w not in seen_w:
                    seen_w.add(w)
                    warnings.append(w)
            _note_rf(etype, attrs, elems)
            for e in elems:
                lattice.add(e)
        _report_import_notes()
        _finish_rf_and_k2(lattice, reference, rf_built, k2_bends, freq_placeholder, warnings,
                          ref_cav_attrs)
        _convert_raw_matrices(lattice, reference, warnings)
        return lattice, {"title": title or line_name, "warnings": warnings,
                         "reference": reference, "reference_declared": reference_declared}

    # --- resolve the sequence into a contiguous element list ----------
    def _position(v, what: str) -> float:
        # ``at`` and ``l`` are evaluated where the sequence is used (MAD-X:
        # ``at := pq`` with pq set after ENDSEQUENCE places the member at pq —
        # the pq of the USE, not a later one)
        if isinstance(v, _Deferred):
            try:
                return _eval_counted(str(v), variables, undef_names)
            except ValueError as exc:
                warnings.append(f"{what} = {str(v)!r} could not be evaluated ({exc})")
                return float("nan")
        if isinstance(v, bool):
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    members = seq["members"]
    refer = seq.get("refer", "centre")
    # every member built first: a FROM= position needs the length of the
    # member it counts from
    built: list = []                    # (elems, total_mm, etype, attrs) or None
    for ename, _at, _frm in members:
        if ename not in element_defs:
            if ename in sequences:
                warnings.append(
                    f"sequence member {ename!r} is itself a sequence — sequences inside "
                    "sequences are not read: its elements are MISSING from this import "
                    "(flatten it in MAD-X: SEQEDIT, FLATTEN; or save a flat deck)")
            else:
                warnings.append(f"sequence member {ename!r} is undefined — skipped")
            built.append(None)
            continue
        etype, attrs = _defn(ename)
        if etype == "multipole" and _gf(attrs, "l") != 0.0:
            # MAD-X: a MULTIPOLE is thin — its L is set to 0 (with a warning)
            _warn_once(warnings, f"multipole {ename!r}: L = {_gf(attrs, 'l'):g} m ignored — "
                                 "a MULTIPOLE is thin (as in MAD-X)")
            attrs = dict(attrs)
            attrs["l"] = 0.0
        elems, total_mm = _build_element(ename, etype, attrs, brho, warnings,
                                         rbend_chord=_chord(ename), k2_sink=k2_bends,
                                         thin_bend_sink=thin_bends)
        built.append((elems, total_mm, etype, attrs))

    def _centre(i: int, at: float) -> float:
        # a member's centre from its AT under the sequence's REFER (an element
        # the builder does not model keeps its declared length here)
        half = 0.0
        if built[i] is not None:
            total_mm = built[i][1] or max(_gf(built[i][3], "l"), 0.0) * _M_TO_MM
            half = total_mm / _M_TO_MM / 2.0
        return at + half if refer == "entry" else at - half if refer == "exit" else at

    with _at_use():
        seq_len = _position(seq["l"], f"sequence {seq_name!r}: l")
        ats = [_position(at, f"member {n!r}: at") for n, at, _f in members]
    # ``at = …, from = name``: counted from the CENTRE of that member — its
    # first occurrence, ``name[k]`` the k-th — whatever REFER says, or from
    # ``seq$start`` / ``seq$end`` (MAD-X)
    centre_of: list = [None] * len(members)
    occurrences: dict = {}
    for i, ((n, _a, frm), at) in enumerate(zip(members, ats, strict=True)):
        occurrences.setdefault(n, []).append(i)
        if frm is None and not math.isnan(at):
            centre_of[i] = _centre(i, at)

    def _from(frm: str):
        # the centre FROM= counts from; None while it is not known yet, False
        # when there is no such member
        if frm == f"{seq_name}$start":
            return 0.0
        if frm == f"{seq_name}$end":
            return 0.0 if math.isnan(seq_len) else seq_len
        mk = re.fullmatch(r"(.+?)\[\s*(\d+)\s*\]", frm)
        name, k = (mk.group(1).strip(), int(mk.group(2))) if mk else (frm, 1)
        idx = occurrences.get(name, [])
        if not 1 <= k <= len(idx):
            return False
        return centre_of[idx[k - 1]]

    pending = [i for i, (_n, _a, frm) in enumerate(members) if frm is not None]
    while pending:
        left = []
        for i in pending:
            c = _from(members[i][2])
            if c is None:
                left.append(i)
                continue
            if c is False:
                warnings.append(f"member {members[i][0]!r}: from={members[i][2]!r} is not a "
                                "member of the sequence — position counted from its start")
            else:
                ats[i] += c
            if not math.isnan(ats[i]):
                centre_of[i] = _centre(i, ats[i])
        if len(left) == len(pending):
            for i in left:
                warnings.append(f"member {members[i][0]!r}: from={members[i][2]!r} has no "
                                "position (its own FROM= cannot be resolved) — position counted "
                                "from the sequence start")
            break
        pending = left

    placed: list[tuple[float, float, list]] = []   # (entry_mm, exit_mm, elems)
    for i, ((ename, _at, _frm), at) in enumerate(zip(members, ats, strict=True)):
        if built[i] is None:
            continue
        elems, total_mm, etype, attrs = built[i]
        if math.isnan(at):
            warnings.append(
                f"member {ename!r} has no at= position — skipped"
            )
            continue
        _note_rf(etype, attrs, elems)
        at_mm = at * _M_TO_MM
        if refer == "entry":
            entry = at_mm
        elif refer == "exit":
            entry = at_mm - total_mm
        else:                                   # centre (MAD-X default)
            entry = at_mm - total_mm / 2.0
        placed.append((entry, entry + total_mm, elems))

    # MAD-X keeps the members in the order the deck lists them and puts a
    # drift between two of them only when the gap between their written
    # positions exceeds 1 µm; it refuses a gap below -1 µm ("negative
    # drift"), and in between the next member follows directly.  A deck MAD-X
    # accepts is placed that way.  It differs from the importer's older rule
    # only for members less than 1 µm apart: sorting put a thin kicker listed
    # before the bend it sits at (the bend's entry rounded 0.2 µm earlier)
    # behind the bend, and 0.2 µm drifts from rounded positions added up
    # (7 µm round the SLS).  A deck MAD-X would refuse keeps the older rule:
    # sorted by entry, every overlap clamped and reported.
    _MADX_TOL_MM = 1e-3
    in_order, end = True, 0.0
    for entry, exit_, _e in placed:
        if entry < end - _MADX_TOL_MM:
            in_order = False
            break
        end = max(end, exit_)
    if not in_order:
        placed.sort(key=lambda p: p[0])
    tol_mm = _MADX_TOL_MM if in_order else 1e-6
    seq_len_mm = (0.0 if math.isnan(seq_len) else seq_len) * _M_TO_MM
    cursor = 0.0
    drift_n = 0
    for entry, exit_, elems in placed:
        gap = entry - cursor
        if gap > tol_mm:
            drift_n += 1
            lattice.add(Drift(name=f"DRIFT_{drift_n:04d}", length=gap))
        elif gap < -1e-6 and not in_order:
            warnings.append(
                f"elements overlap by {-gap:.3f} mm near s={cursor:.1f} mm "
                "— geometry clamped"
            )
        for e in elems:
            lattice.add(e)
        cursor = exit_ if in_order else max(cursor, exit_)
    # Trailing drift to the declared sequence length.
    if seq_len_mm - cursor > tol_mm:
        drift_n += 1
        lattice.add(Drift(name=f"DRIFT_{drift_n:04d}",
                          length=seq_len_mm - cursor))

    _report_import_notes()
    _finish_rf_and_k2(lattice, reference, rf_built, k2_bends, freq_placeholder, warnings,
                      ref_cav_attrs)
    _convert_raw_matrices(lattice, reference, warnings)

    metadata = {
        "title": title or seq_name,
        "warnings": warnings,
        "reference": reference,
        "reference_declared": reference_declared,
    }
    return lattice, metadata


# Attribute values that are text by nature (never evaluated as numbers).
_TEXT_ATTRS = frozenset(("type", "apertype", "particle", "refer", "refpos", "next_sequ",
                         "from", "comments", "mech_sep", "slot_id", "assembly_id",
                         "lfile", "tfile", "sequence"))
# Numeric attributes the builder reads: one that cannot be evaluated is reported.
_USED_NUMERIC = frozenset(("l", "angle", "e1", "e2", "k0", "k1", "k1s", "k2", "k2s", "k3",
                           "tilt", "hgap", "fint", "fintx", "ks", "volt", "lag", "freq",
                           "harmon", "kick", "hkick", "vkick", "chkick", "cvkick", "aperture",
                           "at", "xsize", "ysize"))


def _warn_once(warnings: list, msg: str) -> None:
    if msg not in warnings:
        warnings.append(msg)


def _late_eval(name: str, attrs: dict, variables: dict, warnings: list,
               undefined: dict | None = None) -> dict:
    """Evaluate the deferred (``:=``) attributes of an element where it is used,
    with the values current at the end of the deck (MAD-X semantics; with
    ``undefined``, a name never defined is 0 and is counted there).
    Attributes given with ``=`` are untouched; one that could not be
    evaluated where it was written is reported."""
    out = None
    for k, v in attrs.items():
        if k in _TEXT_ATTRS:
            continue
        # only ``:=`` values: an ``=`` value was evaluated where it was written
        # (a forward reference there is 0, in MAD-X as in this importer)
        if not isinstance(v, _Deferred):
            if isinstance(v, str) and k in _USED_NUMERIC:
                _warn_once(warnings, f"element {name!r}: {k} = {v!r} could not be "
                                     "evaluated — taken as 0")
            continue
        s = str(v).strip()
        try:
            if s.startswith("{") and s.endswith("}"):
                toks = [tok for tok in _split_attributes(s[1:-1]) if tok.strip()]
                val = [_eval_expr(tok, variables) if undefined is None
                       else _eval_counted(tok, variables, undefined) for tok in toks]
            elif undefined is None:
                val = _eval_expr(s, variables)
            else:
                val = _eval_counted(s, variables, undefined)
        except ValueError as exc:
            if k in _USED_NUMERIC or s.startswith("{"):
                _warn_once(warnings, f"element {name!r}: {k} = {s!r} could not be evaluated "
                                     f"({exc}) — taken as 0")
            continue
        if out is None:
            out = dict(attrs)
        out[k] = val
    return attrs if out is None else out


def _finish_rf_and_k2(lattice, reference, rf_built, k2_bends, freq_placeholder, warnings,
                      ref_cavity_attrs: dict | None = None) -> None:
    """RF cavities with ``HARMON`` get f = h·βc/C, C the length of the expanded
    beam line — MAD-X's own rule, which also overrides a ``FREQ`` given with
    it.  h is the HARMON as written, as MAD-X takes it for the frequency (a
    HARMON of 84.7 gives 84.7·βc/C there too); it is stored on the gap as
    ``harmonic`` (an int when integral).  The reference frequency follows:
    the 352.21 MHz placeholder (no cavity with a written FREQ) is replaced by
    the first cavity of the line that has a frequency.  Bends that carry
    K0/K2 are reported once."""
    length_m = sum(float(getattr(e, "length", 0.0) or 0.0) for e in lattice.elements) / _M_TO_MM
    first_f = None
    # the cavity the reference frequency was taken from (first with FREQ > 0):
    # when it also has HARMON, its frequency is the harmonic one
    harmon_ref = None
    if ref_cavity_attrs is not None and _gf(ref_cavity_attrs, "harmon") > 0 and length_m > 0:
        harmon_ref = _gf(ref_cavity_attrs, "harmon") * reference.beta * C_LIGHT / length_m / 1e6
    for gap, harmon, _freq in rf_built:
        if harmon > 0:                             # MAD-X: HARMON overrides FREQ
            gap.harmonic = int(harmon) if float(harmon).is_integer() else harmon
            if length_m <= 0:
                warnings.append(f"RF cavity {gap.name!r}: HARMON={harmon:g} but the beam line "
                                "has no length — frequency left at 0")
                continue
            gap.frequency = harmon * reference.beta * C_LIGHT / length_m / 1e6
            if first_f is None:
                first_f = gap.frequency
                warnings.append(
                    f"RF frequency from HARMON={harmon:g}: f = h·βc/C = {gap.frequency:.9g} MHz "
                    f"(C = {length_m:.9g} m, β = {reference.beta:.9g})")
    if freq_placeholder:
        # the 352.21 MHz placeholder was never real: take the first cavity of
        # the beam line that has a frequency (HARMON, deferred or inherited FREQ)
        f_line = next((g.frequency for g, _h, _f in rf_built if g.frequency > 0), None)
        if f_line is not None:
            reference.frequency = f_line
    elif first_f is not None and harmon_ref is not None:
        reference.frequency = harmon_ref
    def _names(gaps):
        return f"{', '.join(gaps[:4])}{' …' if len(gaps) > 4 else ''}"

    idle = list(dict.fromkeys(g.name for g, _h, _f in rf_built
                              if not g.frequency > 0 and g.voltage == 0.0))
    if idle:
        warnings.append(
            f"{len(idle)} RF cavit{'y has' if len(idle) == 1 else 'ies have'} no VOLT, FREQ or "
            f"HARMON ({_names(idle)}): they do nothing, as in MAD-X (RF settings in a CALLed "
            "file?)")
    active = list(dict.fromkeys(g.name for g, _h, _f in rf_built
                                if not g.frequency > 0 and g.voltage != 0.0))
    if active:
        warnings.append(
            f"{len(active)} RF cavit{'y has' if len(active) == 1 else 'ies have'} a voltage but "
            f"neither FREQ nor HARMON ({_names(active)}): frequency 0 — envelope and matrix runs "
            "refuse such a gap, multiparticle runs keep the beam's frequency there; give the "
            "cavity a FREQ")
    if k2_bends:
        uniq = list(dict.fromkeys(k2_bends))
        warnings.append(
            f"{len(uniq)} bend(s) carry a K0/K2 component (e.g. {', '.join(uniq[:4])}): kept on "
            "the bend for ring mode; linac tracking ignores it")


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
    no_freq_at = False
    n_after = 0                       # MATRIX elements converted about the BEAM energy
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
                n_after += 1
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
                n_after += 1
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
            except RFFrequencyError:
                lost_at, no_freq_at = e.name, True
            except (ValueError, ZeroDivisionError, OverflowError):
                lost_at = e.name
    if lost_at is not None and (n_after or not no_freq_at):
        # (a cavity with no frequency matters here only for a MATRIX after it;
        # _finish_rf_and_k2 reports the cavity itself)
        why = ("the cavity has a voltage but no frequency, so the energy after it is unknown"
               if no_freq_at else
               "the BEAM species decelerates below zero energy in this sequence's RF")
        warnings.append(
            f"reference particle lost at {lost_at!r}: {why}; any "
            "MATRIX element after that point is converted about the BEAM "
            "energy instead of the local one")
    if n_gain:
        warnings.append(
            f"{n_gain} MATRIX element(s) carry an energy kick (kick6): "
            "imported as a ΔW offset on every particle — HELIX's "
            "reference energy stays at the BEAM value through them")


__all__ = ["parse_madx"]
