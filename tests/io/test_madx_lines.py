"""MAD-X importer: beam LINEs, element inheritance, deferred attributes,
``elem->attr``, MAD-X constants and functions, control blocks, HARMON
frequencies, bend K0/K2, sequence placement.

Ring decks (the simple Booster in tests/ring/fixtures among them) are written
with ``LINE=(…)`` and ``child: parent`` definitions; before these fixes the
importer dropped LINEs and inherited definitions silently (or turned an
inherited quadrupole into a marker), dropped every definition with a ``:=``
attribute, gave HARMON-only cavities frequency 0 and lost bend K2.

Every rule is pinned to MAD-X 5.09 itself: the value MAD-X gives is written
into the test (so the rule is checked wherever the tests run), and where
cpymad is installed the same deck is also run through MAD-X and compared.
"""
from __future__ import annotations

import contextlib
import functools
import math
from pathlib import Path

import pytest

from linac_gen.core.constants import C_LIGHT
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.io.madx_parser import parse_madx

FIXTURE = Path(__file__).resolve().parents[1] / "ring" / "fixtures" / "simple_booster.madx"


@functools.lru_cache(maxsize=1)
def _madx_release() -> tuple:
    import cpymad.madx
    m = cpymad.madx.Madx(stdout=False)
    try:
        return tuple(int(x) for x in str(m.version.release).split(".")[:2])
    finally:
        m.quit()


def _madx_or_none():
    """cpymad's MAD-X module — the external truth — or None: not installed, or
    older than 5.09 (its built-in constants predate CODATA 2018, which HELIX
    and MAD-X 5.09 share, so 1e-15 pins would fail for that reason alone)."""
    try:
        import cpymad.madx as cpymad
    except ImportError:
        return None
    return cpymad if _madx_release() >= (5, 9) else None


def _madx_module():
    """As :func:`_madx_or_none`, for a test that needs MAD-X itself: a skip."""
    cpymad = _madx_or_none()
    if cpymad is None:
        pytest.skip("cpymad with MAD-X >= 5.09 is not available")
    return cpymad


def _write(tmp_path, text: str, name="deck.madx"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


BEAM = "beam, particle=proton, energy=0.8+pmass;\n"


def _names(lat):
    return [e.name for e in lat.elements]


LINE_ELEMENTS = "a: drift, l=1; b: drift, l=2; c: drift, l=3; m: marker;\ncell: line=(a, b, m);\n"


@pytest.mark.parametrize("spec, names", [
    ("2*cell, -cell, a, (c, -(a, b))", "a b m a b m m b a a c b a"),
    ("a, (b, (c, m))", "a b c m"),
    ("(c, a), cell", "c a a b m"),
    # MAD-X drops a bracketed group written right after a sub-line (it reads
    # it as that line's arguments): pinned to MAD-X 5.09, and reported
    ("2*cell, -cell, (c, -(a, b))", "a b m a b m m b a"),
    ("cell, (c, a)", "a b m"),
])
def test_line_repetition_reflection_and_nesting(tmp_path, spec, names):
    deck = BEAM + LINE_ELEMENTS + f"ring: line=({spec});\nuse, period=ring;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert " ".join(_names(lat)) == names
    dropped = any("the bracketed group after sub-line 'cell' is dropped" in w
                  for w in meta["warnings"])
    assert dropped == (spec.startswith("2*cell, -cell, (") or spec.startswith("cell, ("))
    mad = _madx_values(tmp_path, deck, "ring")
    if mad is not None:
        cpymad = _madx_or_none()
        m = cpymad.Madx(stdout=False)
        try:
            m.input(deck + "survey;")
            seq = " ".join(n.split(":")[0] for n in m.table.survey.name if "$" not in n)
        finally:
            m.quit()
        assert seq == names


def test_a_sequence_inside_a_sequence_is_named_not_undefined(tmp_path):
    # (not read yet: the warning says what is missing and how to flatten it;
    # a member that names nothing keeps its own message)
    deck = BEAM + ("q1: quadrupole, l=0.5, k1=0.3;\n"
                   "inner: sequence, l=2.0; q1, at=1.0; endsequence;\n"
                   "outer: sequence, l=10.0; inner, at=5.0; ghost, at=8.0; endsequence;\n"
                   "use, sequence=outer;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    w = meta["warnings"]
    assert any("'inner' is itself a sequence" in x and "MISSING" in x and "FLATTEN" in x
               for x in w)
    assert any("'ghost' is undefined — skipped" in x for x in w)
    assert not any("'inner' is undefined" in x for x in w)


def test_line_without_use_takes_the_last_line_and_says_so(tmp_path):
    deck = BEAM + "a: drift, l=1; l1: line=(a); l2: line=(a, a);"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _names(lat) == ["a", "a"]
    assert any("last-defined LINE 'l2'" in w for w in meta["warnings"])


def test_line_errors_are_loud(tmp_path):
    with pytest.raises(ValueError, match="contains itself"):
        parse_madx(_write(tmp_path, BEAM + "a: drift, l=1; x: line=(a, y); y: line=(x); use, period=x;"))
    with pytest.raises(ValueError, match="undefined element.*ghost"):
        parse_madx(_write(tmp_path, BEAM + "a: drift, l=1; x: line=(a, ghost); use, period=x;", "b.madx"))


def test_inheritance_child_overrides_parent(tmp_path):
    deck = BEAM + """
    qf: quadrupole, l=0.5, k1=0.4;
    qd: qf, k1=-0.4;
    qd2: qd;
    r: line=(qf, qd, qd2); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    q = [e for e in lat.elements if isinstance(e, Quadrupole)]
    assert [e.name for e in q] == ["qf", "qd", "qd2"]
    assert [e.length for e in q] == [500.0, 500.0, 500.0]
    assert _k1(lat, meta) == pytest.approx({"qf": 0.4, "qd": -0.4, "qd2": -0.4}, abs=1e-15)


def test_inherited_member_in_a_sequence_is_no_longer_a_marker(tmp_path):
    # before: "element 'q2': unsupported type 'qd' — replaced with a marker"
    deck = BEAM + """
    qf: quadrupole, l=0.5, k1=0.5; qd: qf, k1=-0.5;
    s: sequence, l=3, refer=entry; qf, at=0; q2: qd, at=1; endsequence;
    use, sequence=s;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    q2 = [e for e in lat.elements if e.name == "q2"]
    assert len(q2) == 1 and isinstance(q2[0], Quadrupole)
    assert not any("unsupported type" in w for w in meta["warnings"])


def test_deferred_attributes_and_forward_references(tmp_path):
    # a definition with ':=' used to vanish without a word; a forward reference
    # is evaluated where the element is used (MAD-X semantics)
    deck = BEAM + """
    qf: quadrupole, l:=lq, k1:=kq;
    lq = 0.3; kq = 0.25;
    r: line=(qf); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    (q,) = lat.elements
    assert isinstance(q, Quadrupole) and q.length == pytest.approx(300.0)
    assert _k1(lat, meta)["qf"] == pytest.approx(0.25, abs=1e-15)
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert _k1(lat, meta)["qf"] == pytest.approx(mad["qf"]["k1"], abs=1e-15)
        assert q.length / 1000 == pytest.approx(mad["qf"]["l"], abs=1e-15)


def test_a_length_set_after_use_is_not_used(tmp_path):
    # MAD-X expands the line at USE: a length given later does not reach the
    # machine (SURVEY puts qf's exit at 0), a strength given later does
    deck = BEAM + """
    qf: quadrupole, l:=lq, k1:=kq;
    r: line=(qf); use, period=r;
    lq = 0.3; kq = 0.25;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    (q,) = lat.elements
    assert q.length == 0.0
    assert _k1(lat, meta)["qf"] == pytest.approx(0.25, abs=1e-15)
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        assert survey == [("qf", 0.0)]


def test_deferred_variables_are_evaluated_at_use(tmp_path):
    # a := b*2 with b set later: MAD-X 2.0 (an immediate reading would give 0)
    deck = BEAM + "a := b*2;\nb = 1;\nqf: quadrupole, l=0.5, k1:=a;\nr: line=(qf); use, period=r;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(2.0, abs=1e-15)
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert mad["qf"]["k1"] == pytest.approx(2.0, abs=1e-15)


def test_random_functions_are_taken_at_their_mean(tmp_path):
    # RANF/GAUSS/TGAUSS (MAD-X's random numbers) cannot be reproduced: the
    # design machine is built (ranf 0.5, gauss 0, tgauss 0) and that is said
    deck = BEAM + """
    qf: quadrupole, l=0.3, k1:=0.1*ranf();
    qg: quadrupole, l=0.3, k1=0.2 + 1e-3*gauss();
    mp: multipole, knl={0, 1e-4*tgauss(3)};
    r: line=(qf, qg, mp); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta) == pytest.approx({"qf": 0.05, "qg": 0.2}, abs=1e-15)
    assert any("random" in w and "gauss" in w and "ranf" in w and "tgauss" in w
               for w in meta["warnings"])


def test_unevaluable_numeric_attribute_is_reported_not_silently_zero(tmp_path):
    # ATAN2 is not a MAD-X function either ("illegal expression set to 0")
    deck = BEAM + """
    qf: quadrupole, l=0.3, k1:=0.1*atan2(1, 2);
    qg: quadrupole, l=0.3, k1=0.1*atan2(1, 2);
    r: line=(qf, qg); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert [e.gradient for e in lat.elements] == [0.0, 0.0]
    ws = meta["warnings"]
    assert any("'qf': k1 = '0.1*atan2(1, 2)' could not be evaluated" in w and "'atan2'" in w
               for w in ws)
    assert any("'qg': k1 = '0.1*atan2(1, 2)' could not be evaluated" in w for w in ws)


def _madx_values(tmp_path, text: str, line: str, name: str = "truth.madx") -> dict | None:
    """{element: {attribute: value}} of the expanded beam line ``line`` as
    MAD-X itself has it (cpymad), or None where MAD-X is not available (the
    tests then check the literal MAD-X values alone)."""
    cpymad = _madx_or_none()
    if cpymad is None:
        return None
    (tmp_path / name).write_text(text)
    m = cpymad.Madx(stdout=False)
    try:
        m.chdir(str(tmp_path))
        m.call(name)
        out: dict = {}
        for e in m.sequence[line].expanded_elements:
            nm = e.name.split(":")[0]
            if "$" in nm:
                continue
            d: dict = {}
            for a in ("l", "k1", "angle", "tilt", "e1", "e2"):
                with contextlib.suppress(Exception):      # (absent for this class)
                    d[a] = float(e[a])
            for a in ("knl", "ksl", "aperture"):
                with contextlib.suppress(Exception):
                    d[a] = [float(x) for x in e[a]]
            out[nm] = d
        return out
    finally:
        m.quit()


def _k1(lat, meta) -> dict:
    from linac_gen.io.madx_parser import _signed_brho
    brho = _signed_brho(meta["reference"])
    return {e.name: e.gradient / brho for e in lat.elements if isinstance(e, Quadrupole)}


def _same_vector(helix, madx) -> bool:
    n = max(len(helix), len(madx))
    a = list(helix) + [0.0] * (n - len(helix))
    b = list(madx) + [0.0] * (n - len(madx))
    return a == pytest.approx(b, abs=1e-15)


UNDEFINED_DECK = BEAM + """
kv = 0.1 + undef_a;
qa: quadrupole, l=0.5, k1=0.2 + undef_b;
qb: quadrupole, l=0.5, k1:=0.3 + undef_c;
qc: quadrupole, l=0.5, k1=kv;
mp: multipole, knl={0, undef_d, 0.01};
mq: multipole, knl:={0, 0.02 + undef_e};
qd: quadrupole, l=0.5, k1=0.1;
qd->k1 = 0.4 + undef_f;
qe: quadrupole, l=0.5, k1=0.5 - nosuch->k1;
qf: quadrupole, l=0.5, k1=kfwd;
kfwd = 0.7;
r: line=(qa, qb, qc, mp, mq, qd, qe, qf); use, period=r;
"""


def test_undefined_names_are_zero_as_in_madx(tmp_path):
    # MAD-X takes a name not (yet) defined as 0, without a word.  The importer
    # used to raise on one inside {…} (a multipole error strength set in
    # another file), take a whole scalar as 0 (k1 = 0.2 + undef gave 0, not
    # 0.2) and skip such a variable; it now follows MAD-X and lists the names
    # once.
    lat, meta = parse_madx(_write(tmp_path, UNDEFINED_DECK))
    k1 = _k1(lat, meta)
    assert [k1[q] for q in ("qa", "qb", "qc", "qd", "qe", "qf")] == pytest.approx(
        [0.2, 0.3, 0.1, 0.4, 0.5, 0.0], abs=1e-15)
    mps = {e.name: e for e in lat.elements if isinstance(e, Multipole)}
    assert _same_vector(mps["mp"].knl, [0.0, 0.0, 0.01])
    assert _same_vector(mps["mq"].knl, [0.0, 0.02])
    mad = _madx_values(tmp_path, UNDEFINED_DECK, "r")
    if mad is not None:
        for q in ("qa", "qb", "qc", "qd", "qe", "qf"):
            assert k1[q] == pytest.approx(mad[q]["k1"], abs=1e-15), q
        assert _same_vector(mps["mp"].knl, mad["mp"]["knl"])
        assert _same_vector(mps["mq"].knl, mad["mq"]["knl"])
    (w,) = [w for w in meta["warnings"] if "not (yet) defined" in w]
    for n in ("undef_a", "undef_b", "undef_c", "undef_d", "undef_e", "undef_f", "nosuch->k1", "kfwd"):
        assert n in w, n


def test_a_division_by_zero_is_zero_as_in_madx(tmp_path):
    # MAD-X: "division by zero, result set to zero" — only the division
    deck = BEAM + """
    kq = 0.3 + 0.01/kerr;
    qf: quadrupole, l=0.5, k1:=kq;
    qd: quadrupole, l=0.5, k1=-0.2 + 0.01/kerr2;
    mp: multipole, knl={0, 0.1/kd_err};
    r: line=(qf, qd, mp); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta) == pytest.approx({"qf": 0.3, "qd": -0.2}, abs=1e-15)
    (mp,) = [e for e in lat.elements if isinstance(e, Multipole)]
    assert _same_vector(mp.knl, [0.0, 0.0])
    assert any("3 division(s) by zero set to 0" in w for w in meta["warnings"])
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert _k1(lat, meta) == pytest.approx({"qf": mad["qf"]["k1"], "qd": mad["qd"]["k1"]},
                                               abs=1e-15)


DEFERRED_DECK = BEAM + """
d: drift, l=1.25;
d2: drift, l:=2*d->l;
d3: drift, l=2*d->l;
d->l = 0.5;
r: line=(d, d2, d3); use, period=r;
"""


def test_element_attribute_reference_and_assignment(tmp_path):
    # ``:=`` is evaluated at use (d2 sees the later d->l = 0.5); ``=`` at once (d3)
    lat, _ = parse_madx(_write(tmp_path, DEFERRED_DECK))
    assert [e.length for e in lat.elements] == [500.0, 1000.0, 2500.0]


def test_deferred_semantics_match_madx(tmp_path):
    madx_mod = _madx_module()
    deck = _write(tmp_path, DEFERRED_DECK, "deferred.madx")
    m = madx_mod.Madx(stdout=False)
    try:
        m.chdir(str(tmp_path))
        m.call("deferred.madx")
        m.input("twiss, betx=1, bety=1;")
        mad = [float(x) for x in m.table.twiss.l if float(x) > 0]
    finally:
        m.quit()
    lat, _ = parse_madx(deck)
    assert [e.length / 1000.0 for e in lat.elements] == pytest.approx(mad, abs=1e-15)


MADX_EXPRESSIONS = ("asin(0.5)", "acos(0.3)", "atan(-1.5)", "sinh(0.7)", "cosh(0.7)", "tanh(0.7)",
                    "log10(250)", "erf(0.4)", "erfc(0.4)", "floor(-2.5)", "ceil(-2.5)",
                    "round(2.5)", "round(-1.5)", "round(0.5)", "round(2.6)", "frac(-2.75)",
                    "frac(2.75)", "sinc(0)", "sinc(0.3)", "2*x.y.z + 1", "x.y.z^2")


MADX_EXPRESSION_VALUES = {           # MAD-X 5.09's own values (pinned through cpymad)
    "round(2.5)": 2.0, "round(-1.5)": -2.0, "round(0.5)": 0.0, "round(2.6)": 3.0,
    "frac(-2.75)": -0.75, "frac(2.75)": 0.75, "sinc(0)": 1.0, "floor(-2.5)": -3.0,
    "ceil(-2.5)": -2.0, "log10(1000)": 3.0, "2^3^2": 64.0, "-2^2": -4.0, "(2^3)^2": 64.0,
    "2*3^2": 18.0, "2*x.y.z + 1": 1.5, "lambda*in + class": 4.6,
    "0.1\n + 0.2": 0.30000000000000004,
}


@pytest.mark.parametrize("expr", sorted(MADX_EXPRESSION_VALUES))
def test_madx_expression_values(expr):
    # functions, rounding (halves to even), ^ grouped left to right, dotted
    # names, names that are Python keywords, an expression over two lines
    from linac_gen.io import madx_parser as mp
    scope = mp._madx_scope()
    scope.update({"x.y.z": 0.25, "lambda": 0.3, "in": 2.0, "class": 4.0})
    assert mp._eval_expr(expr, scope) == MADX_EXPRESSION_VALUES[expr]


def test_madx_builtin_constants_and_functions_match_madx():
    from linac_gen.io import madx_parser as mp
    cpymad = _madx_module()
    m = cpymad.Madx(stdout=False)
    try:
        scope = mp._madx_scope()
        for k in ("pmass", "emass", "nmass", "umass", "mumass", "qelect", "hbar", "erad", "clight",
                  "degrad", "raddeg", "amu0"):
            assert mp._eval_expr(k, scope) == pytest.approx(float(m.eval(k)), rel=1e-15), k
        assert mp._eval_expr("prad", scope) == pytest.approx(float(m.eval("prad")), rel=1e-12)
        # functions, MAD-X's rounding (halves to even) and dotted names
        m.input("x.y.z = 0.25;")
        scope["x.y.z"] = 0.25
        for i, expr in enumerate(MADX_EXPRESSIONS):
            m.input(f"r_{i} = {expr};")
            assert mp._eval_expr(expr, scope) == pytest.approx(float(m.eval(f"r_{i}")),
                                                              rel=1e-15, abs=1e-15), expr
        m.input("lambda = 0.3; in = 2; class = 4;")
        scope.update({"lambda": 0.3, "in": 2.0, "class": 4.0})
        for i, expr in enumerate(sorted(MADX_EXPRESSION_VALUES)):
            m.input(f"v_{i} = {expr};")
            assert float(m.eval(f"v_{i}")) == MADX_EXPRESSION_VALUES[expr], expr
    finally:
        m.quit()


def test_madx_constant_names_are_ordinary_names_in_elegant(tmp_path):
    # the elegant importer shares the evaluator: an elegant deck's own
    # variables named like MAD-X constants keep their values
    from linac_gen.io.elegant_parser import parse_elegant
    deck = tmp_path / "e.lte"
    deck.write_text('% 0.5 sto degrad\nerad = 0.1\nq: kquad, l=0.2, k1="degrad*4"\n'
                    'd: drift, l="erad*10"\ncell: line=(q, d)\n')
    lat, meta = parse_elegant(str(deck), name="cell", species="proton", w_kin=100.0)
    from linac_gen.io.madx_parser import _signed_brho
    q, d = lat.elements
    assert q.gradient / _signed_brho(meta["reference"]) == pytest.approx(2.0, rel=1e-12)
    assert d.length == pytest.approx(1000.0)


def test_redefining_a_builtin_constant_is_ignored_with_a_warning(tmp_path):
    deck = "pmass = 0.9;\nbeam, particle=proton, energy=pmass+0.8;\nqf: quadrupole, l=0.5, k1=0.3;\n" \
           "r: line=(qf); use, period=r;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    ref = meta["reference"]
    assert ref.w_kin == pytest.approx(800.0, rel=1e-12)
    assert any("pmass = 0.9: attempt to redefine the built-in constant ignored" in w
               for w in meta["warnings"])
    cpymad = _madx_or_none()
    if cpymad is not None:
        m = cpymad.Madx(stdout=False)
        try:
            m.input(deck)
            energy_gev = float(m.eval("beam->energy"))
        finally:
            m.quit()
        assert (ref.w_kin + ref.species.mass) / 1000.0 == pytest.approx(energy_gev, rel=1e-15)


def test_harmon_gives_the_ring_frequency(tmp_path):
    deck = BEAM + """
    d: drift, l=10; rf: rfcavity, l=0, volt=0.1, harmon=7, lag=0;
    r: line=(d, rf, d); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    (gap,) = [e for e in lat.elements if isinstance(e, RFGap)]
    ref = meta["reference"]
    f = 7 * ref.beta * C_LIGHT / 20.0 / 1e6
    assert gap.frequency == pytest.approx(f, rel=1e-14)
    assert gap.harmonic == 7
    assert ref.frequency == pytest.approx(f, rel=1e-14)       # the 352.21 placeholder is replaced
    assert any("RF frequency from HARMON=7" in w for w in meta["warnings"])


def test_harmon_overrides_freq_as_in_madx_track(tmp_path):
    # External truth: MAD-X TRACK kicks a particle at t = 0.5 m.  With FREQ and
    # HARMON both given the kick is the HARMON-only one, and its ratio to the
    # FREQ-only kick is sin(2π f t/c) / sin(2π FREQ t/c) at the frequency the
    # importer gives the gap, f = h·βc/C.
    cpymad = _madx_module()
    both = "cav: rfcavity, l=0, volt=10, lag=0, freq=52.8, harmon=84;"

    def kick(cav: str) -> float:
        m = cpymad.Madx(stdout=False)
        try:
            m.input(BEAM + cav + "\nd: drift, l=474.2;\nr: line=(cav, d);\nuse, period=r;\n"
                    "track, onepass, onetable, dump=false;\nstart, t=0.5;\nrun, turns=1;\n"
                    "endtrack;\n")
            return float(m.table.trackone.pt[-1])
        finally:
            m.quit()

    k_both = kick(both)
    k_freq = kick("cav: rfcavity, l=0, volt=10, lag=0, freq=52.8;")
    assert k_both == kick("cav: rfcavity, l=0, volt=10, lag=0, harmon=84;")
    lat, meta = parse_madx(_write(tmp_path, BEAM + both + "\nd: drift, l=474.2;\n"
                                  "r: line=(cav, d); use, period=r;\n"))
    (gap,) = [e for e in lat.elements if isinstance(e, RFGap)]
    ref = meta["reference"]
    assert gap.harmonic == 84
    assert gap.frequency == pytest.approx(84 * ref.beta * C_LIGHT / 474.2 / 1e6, rel=1e-14)
    phase = lambda f_mhz: math.sin(2 * math.pi * f_mhz * 1e6 * 0.5 / C_LIGHT)   # noqa: E731
    assert k_both / k_freq == pytest.approx(phase(gap.frequency) / phase(52.8), rel=1e-12)
    assert ref.frequency == pytest.approx(gap.frequency, rel=1e-14)   # (the reference follows)


def test_reference_frequency_is_not_taken_from_an_unused_deferred_cavity(tmp_path):
    # the reference RF frequency is read, as it always was, from the first
    # RFCAVITY defined with a FREQ among the definitions the importer always
    # read; an unused cavity written with ':=' (once skipped whole) is not one
    deck = BEAM + """
    v1 = 0.1;
    cav1: rfcavity, l=0, freq=201.25, volt:=v1, lag=0;
    cav2: rfcavity, l=0, freq=805, volt=0.1, lag=0;
    qf: quadrupole, l=0.5, k1=0.3;
    s: sequence, l=4; cav2, at=1; qf, at=3; endsequence;
    use, sequence=s;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert meta["reference"].frequency == 805.0


def test_bend_k2_is_kept_only_when_given(tmp_path):
    deck = BEAM + """
    b1: sbend, l=1, angle=0.1, k1=0.01, k2=0.5;
    b2: sbend, l=1, angle=0.1, k1=0.01;
    r: line=(b1, b2); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    d1, d2 = [e for e in lat.elements if isinstance(e, Dipole)]
    assert d1.k2 == 0.5
    assert not hasattr(d2, "k2") and not hasattr(d2, "k0")   # an ordinary bend is unchanged
    assert sum("carry a K0/K2 component" in w for w in meta["warnings"]) == 1


def test_simple_booster_line_import_matches_madx(tmp_path):
    """External truth: the simple Booster deck (24 cells written as LINEs,
    inherited magnets) imported directly; one-turn optics equal MAD-X's.
    The RF voltage is set to zero in both codes: HELIX's linac RF gap adds a
    transverse RF focusing term that a ring cavity does not have."""
    madx_mod = _madx_module()
    from linac_gen.matching.periodic import find_periodic_twiss

    text = FIXTURE.read_text(encoding="latin-1").replace("total_voltage = 0.2;",
                                                       "total_voltage = 0.0;")
    deck = _write(tmp_path, text, "simple_booster.madx")
    lat, meta = parse_madx(deck)
    assert len(lat.elements) == 976
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(474.202752, abs=1e-9)
    gaps = [e for e in lat.elements if isinstance(e, RFGap)]
    assert len(gaps) == 22 and {g.harmonic for g in gaps} == {84}

    m = madx_mod.Madx(stdout=False)
    try:
        m.chdir(str(tmp_path))
        m.call("simple_booster.madx")
        m.input("use, period=booster; twiss;")
        tw, summ = m.table.twiss, m.table.summ
        beta_rel = float(m.eval("beam->beta"))
        q = (float(summ.q1[0]), float(summ.q2[0]))
        start = {k: float(tw[k][0]) for k in ("betx", "bety", "alfx", "alfy", "dx")}
    finally:
        m.quit()
    ref = meta["reference"]
    pt = find_periodic_twiss(lat, ref)
    assert pt["beta_x"] == pytest.approx(start["betx"], abs=1e-9)
    assert pt["beta_y"] == pytest.approx(start["bety"], abs=1e-9)
    assert pt["alpha_x"] == pytest.approx(start["alfx"], abs=1e-9)
    assert pt["alpha_y"] == pytest.approx(start["alfy"], abs=1e-9)
    eta_m = pt["disp_x"] * ref.beta ** 2 * (ref.species.mass + ref.w_kin) * 1e-3
    assert eta_m == pytest.approx(beta_rel * start["dx"], abs=1e-9)
    # fractional tunes from the one-turn map (the integer part is 6 in both planes)
    assert pt["mu_x"] / 360.0 == pytest.approx(q[0] % 1.0, abs=1e-9)
    assert pt["mu_y"] / 360.0 == pytest.approx(q[1] % 1.0, abs=1e-9)


def test_line_with_an_undefined_member_is_refused(tmp_path):
    # MAD-X stops here too; importing the rest would build a shorter machine
    # (the usual cause: the magnets are in a CALLed file, which is not read)
    deck = BEAM + "a: drift, l=1; x: line=(a, ghost, ghost2); use, period=x;"
    with pytest.raises(ValueError, match="2 undefined element"):
        parse_madx(_write(tmp_path, deck))


def test_ampersand_continuation_inside_a_line_is_ignored(tmp_path):
    deck = BEAM + "a: drift, l=1; b: drift, l=2;\nx: line=(a, &\n   b, &\n a);\nuse, period=x;"
    lat, _ = parse_madx(_write(tmp_path, deck))
    assert _names(lat) == ["a", "b", "a"]


def test_unmodelled_element_keeps_its_length_in_a_line(tmp_path):
    deck = BEAM + "a: drift, l=1; c: collimator, l=1.0668; x: line=(a, c, a); use, period=x;"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert sum(e.length for e in lat.elements) == pytest.approx(3066.8)
    assert any("'c': type 'collimator' is not modelled — kept as a drift" in w for w in meta["warnings"])


def test_sequence_self_placement_is_the_defined_element(tmp_path):
    # ``q: q, at=…`` inside a SEQUENCE places the element defined before
    deck = BEAM + """
    q: quadrupole, l=0.5, k1=0.3; mk: marker;
    s: sequence, l=3, refer=entry; mk: mk, at=0; q: q, at=1; endsequence;
    use, sequence=s;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    (quad,) = [e for e in lat.elements if isinstance(e, Quadrupole)]
    assert quad.length == 500.0
    assert _k1(lat, meta)["q"] == pytest.approx(0.3, abs=1e-15)
    assert not any("unsupported type" in w for w in meta["warnings"])
    mad = _madx_values(tmp_path, deck, "s")
    if mad is not None:
        assert _k1(lat, meta)["q"] == pytest.approx(mad["q"]["k1"], abs=1e-15)


MODIFY_DECK = BEAM + """
kqf.a12 = 0.25;
qf.a12: quadrupole, l=0.5, k1:=kqf.a12*2;
qd: quadrupole, l=0.4, k1=-0.1;
qd2: qd;
qd, k1 := -kqf.a12, l = 0.6;
mb: sbend, l=1, angle=0.0;
mb, angle := theta.mb;
theta.mb = 0.02;
r: line=(qf.a12, qd, qd2, mb); use, period=r;
"""


def test_element_modification_and_dotted_names_match_madx(tmp_path):
    # ``qd, k1 := …`` changes an element already defined (CERN decks set
    # their strengths this way: SPS ``MBA.10030, ANGLE := kMBA``); names may
    # contain dots.  Both used to be lost without a word (the SPS bends had
    # no angle).  qd2 was cloned before the change and keeps what it copied.
    lat, meta = parse_madx(_write(tmp_path, MODIFY_DECK))
    k1 = _k1(lat, meta)
    assert k1 == pytest.approx({"qf.a12": 0.5, "qd": -0.25, "qd2": -0.1}, abs=1e-15)
    lengths = {e.name: e.length / 1000 for e in lat.elements if isinstance(e, Quadrupole)}
    assert lengths == pytest.approx({"qf.a12": 0.5, "qd": 0.6, "qd2": 0.4}, abs=1e-15)
    (b,) = [e for e in lat.elements if isinstance(e, Dipole)]
    assert math.radians(b.angle) == pytest.approx(0.02, abs=1e-15)
    assert not meta["warnings"]
    mad = _madx_values(tmp_path, MODIFY_DECK, "r")
    if mad is not None:
        assert k1 == pytest.approx({n: mad[n]["k1"] for n in ("qf.a12", "qd", "qd2")}, abs=1e-15)
        assert lengths == pytest.approx({n: mad[n]["l"] for n in ("qf.a12", "qd", "qd2")},
                                        abs=1e-15)
        assert math.radians(b.angle) == pytest.approx(mad["mb"]["angle"], abs=1e-15)


def test_definitions_with_deferred_attributes_never_crash_the_import(tmp_path):
    # such definitions used to be skipped whole; once read, a tilt or an
    # aperture vector that is not (yet) defined must not crash the import
    deck = BEAM + """
    mp: multipole, knl:={0, 0.1}, tilt:=t_later;
    mq: multipole, lrad:=0.1, knl={0, 0.1}, tilt=t_undefined;
    qf: quadrupole, l=0.5, k1:=kq, apertype=ellipse, aperture={ax, ay};
    kq = 0.3; ax = 0.03; ay = 0.02;
    r: line=(mp, mq, qf); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    mps = {e.name: e for e in lat.elements if isinstance(e, Multipole)}
    assert mps["mp"].tilt_deg == mps["mq"].tilt_deg == 0.0
    assert _k1(lat, meta)["qf"] == pytest.approx(0.3, abs=1e-15)
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert mad["mp"]["tilt"] == mad["mq"]["tilt"] == 0.0
        assert _k1(lat, meta)["qf"] == pytest.approx(mad["qf"]["k1"], abs=1e-15)
        assert mad["qf"]["aperture"][:2] == [0.0, 0.0]      # (= is evaluated where written)


INHERIT_CASES = {
    "parent_arrow_later": "qf: quadrupole, l=0.5, k1=0.4; qd: qf; qf->k1 = 0.7; qf->l = 0.9;",
    "parent_restated": "qf: quadrupole, l=0.5, k1=0.4; qd: qf; qf, k1=0.7, l=0.9;",
    "parent_redefined": "qf: quadrupole, l=0.5, k1=0.4; qd: qf; qf: quadrupole, l=0.9, k1=0.7;",
    "self_label_top_level": "qf: quadrupole, l=0.5, k1=0.4; qd: qf; qf: qf, k1=0.7, l=0.9;",
    "child_set_then_parent": "qf: quadrupole, l=0.5, k1=0.4; qd: qf, k1=-0.4; qf->k1 = 0.7;",
    "parent_becomes_drift": "qf: quadrupole, l=0.5, k1=0.4; qd: qf; qf: drift, l=0.9;",
}


INHERIT_MADX = {                       # (l, k1) of qf and qd in MAD-X 5.09
    "parent_arrow_later": {"qf": (0.9, 0.7), "qd": (0.5, 0.4)},
    "parent_restated": {"qf": (0.9, 0.7), "qd": (0.5, 0.4)},
    "parent_redefined": {"qf": (0.9, 0.7), "qd": (0.5, 0.4)},
    "self_label_top_level": {"qf": (0.5, 0.4), "qd": (0.5, 0.4)},
    "child_set_then_parent": {"qf": (0.5, 0.7), "qd": (0.5, -0.4)},
    "parent_becomes_drift": {"qf": (0.9, None), "qd": (0.5, 0.4)},
}


@pytest.mark.parametrize("case", sorted(INHERIT_CASES))
def test_inheritance_copies_the_parent_when_the_child_is_defined(tmp_path, case):
    # MAD-X clones the parent at the child's definition: later changes to
    # the parent (by ->, a restatement or a redefinition) do not reach it
    deck = BEAM + INHERIT_CASES[case] + "\nr: line=(qf, qd); use, period=r;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    lengths = {e.name: e.length / 1000 for e in lat.elements}
    k1 = _k1(lat, meta)
    for n, (length, strength) in INHERIT_MADX[case].items():
        assert lengths[n] == pytest.approx(length, abs=1e-15), n
        assert k1.get(n) == (None if strength is None else pytest.approx(strength, abs=1e-15)), n
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        for n, (length, strength) in INHERIT_MADX[case].items():
            assert mad[n]["l"] == pytest.approx(length, abs=1e-15), n
            if strength is not None:
                assert mad[n]["k1"] == pytest.approx(strength, abs=1e-15), n


BLOCK_CASES = {                          # deck → (qf k1, qd k1) in MAD-X 5.09
    "macro_not_exec": ("dk = 0.05;\nbump(): macro = {\n  qf->k1 = qf->k1 + dk;\n"
                       "  qd->k1 = qd->k1 - dk;\n};", (0.3, -0.3)),
    "macro_exec_twice": ("dk = 0.05;\nbump(): macro = {\n  qf->k1 = qf->k1 + dk;\n"
                         "  qd->k1 = qd->k1 - dk;\n};\nexec, bump;\nexec, bump;", (0.4, -0.4)),
    "macro_args": ("settune(dk): macro = {\n  qf->k1 = 0.3 + dk;\n  qd->k1 = -0.3 - dk;\n};\n"
                   "exec, settune(0.02);", (0.32, -0.32)),
    "if_false": ("flag = 0;\nif (flag == 1) {\n  qf->k1 = 0.5;\n  qd->k1 = -0.5;\n};",
                 (0.3, -0.3)),
    "if_true": ("flag = 1;\nif (flag == 1) { qf->k1 = 0.5; qd->k1 = -0.5; }", (0.5, -0.5)),
    "elseif": ("flag = 2;\nif (flag == 1) { qf->k1 = 0.5; } elseif (flag == 2) { qf->k1 = 0.6; } "
               "else { qf->k1 = 0.7; }", (0.6, -0.3)),
    "else_if_spaced": ("flag = 3;\nif (flag == 1) { qf->k1 = 0.5; } else if (flag == 2) "
                       "{ qf->k1 = 0.6; } else { qf->k1 = 0.7; }", (0.7, -0.3)),
    "nested_and_or": ("a = 1; b = 0;\nif (a == 1 && b < 1) { if (b == 1 || a > 5) "
                      "{ qf->k1 = 0.1; } else { qf->k1 = 0.2; } }", (0.2, -0.3)),
    "undefined_in_condition": ("if (nothere == 0) { qd->k1 = -0.9; }", (0.3, -0.9)),
    "brace_without_semicolon": ("flag = 0;\nif (flag == 1) { qf->k1 = 0.5; }\nqf->k1 = 0.2;",
                                (0.2, -0.3)),
    "while": ("n = 0;\nwhile (n < 3) { qf->k1 = qf->k1 + 0.1; n = n + 1; }", (0.6, -0.3)),
    "macro_without_equals": ("bump() : macro {\n  qf->k1 = 0.9;\n};\nexec, bump;", (0.9, -0.3)),
}


@pytest.mark.parametrize("case", sorted(BLOCK_CASES))
def test_control_blocks_run_as_in_madx(tmp_path, case):
    # IF/ELSEIF/ELSE run the branch whose condition holds, a MACRO body runs
    # on EXEC (its parameters replaced by the arguments), WHILE loops run
    body, (kqf, kqd) = BLOCK_CASES[case]
    deck = (BEAM + "qf: quadrupole, l=0.5, k1=0.3;\nqd: quadrupole, l=0.5, k1=-0.3;\n"
            + body + "\nr: line=(qf, qd); use, period=r;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta) == pytest.approx({"qf": kqf, "qd": kqd}, abs=1e-15)
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert (mad["qf"]["k1"], mad["qd"]["k1"]) == pytest.approx((kqf, kqd), abs=1e-15)


def test_circular_definitions_are_reported_not_a_crash(tmp_path):
    # MAD-X stops on these ("circular call"); the importer takes 0 and says so
    deck = BEAM + """
    a: drift, l:=b->l; b: drift, l:=a->l;
    kq = 0.3; kq := kq*1.1; qf: quadrupole, l=0.5, k1:=kq;
    r: line=(a, b, qf); use, period=r;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert [e.length for e in lat.elements] == [0.0, 0.0, 500.0]
    assert lat.elements[2].gradient == 0.0
    ws = meta["warnings"]
    assert any("refers to itself" in w and "'a'" in w for w in ws)
    assert any("deferred variable 'kq' refers to itself" in w for w in ws)


def test_self_scaling_attribute_assignment_matches_madx(tmp_path):
    deck = BEAM + "qf: quadrupole, l=0.5, k1=0.3; qf->k1 := qf->k1*1.1;\nr: line=(qf); use, period=r;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(0.33, rel=1e-15)
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert _k1(lat, meta)["qf"] == pytest.approx(mad["qf"]["k1"], rel=1e-15)


def test_return_ends_the_file_as_in_madx(tmp_path):
    deck = BEAM + """
    qf: quadrupole, l=0.5, k1=0.3;
    r: line=(qf); use, period=r;
    return;
    qf: quadrupole, l=0.9, k1=0.7;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    (q,) = lat.elements
    assert q.length / 1000 == 0.5
    assert any("RETURN ends the file in MAD-X: the 1 statement(s) after it are not read" in w
               for w in meta["warnings"])
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert mad["qf"]["l"] == 0.5


def test_stop_ends_reading_as_in_madx(tmp_path):
    deck = BEAM + "qf: quadrupole, l=0.5, k1=0.3;\nr: line=(qf); use, period=r;\nstop;\nqf->k1 = 0.9;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(0.3, abs=1e-15)
    assert any("STOP ends MAD-X: the 1 statement(s) after it are not read" in w
               for w in meta["warnings"])


PLACEMENT_DECK = BEAM + """
b: sbend, l=0.0137784, angle=0.001; k: hkicker; q: quadrupole, l=0.2287216, k1=0.1;
s: sequence, l=2, refer=centre;
  k1st: k, at=0.4855;
  b, at=0.492389;
  k2nd: k, at=0.499278;
  q, at=0.613639;
  k3rd: k, at=0.728;
endsequence;
use, sequence=s;
"""


def test_members_less_than_a_micron_apart_are_placed_as_madx_does(tmp_path):
    # (from the SLS deck) the positions are rounded to 1 µm: the bend's entry
    # is 0.2 µm before the kicker listed ahead of it, and 0.2 µm gaps follow.
    # MAD-X keeps the deck order and lays no drift under 1 µm; the importer
    # used to sort the kicker behind the bend and to lay 0.2 µm drifts
    expected = [("k1st", 0.4855), ("b", 0.4992784), ("k2nd", 0.4992784), ("q", 0.728),
                ("k3rd", 0.728)]                          # MAD-X 5.09 SURVEY
    lat, meta = parse_madx(_write(tmp_path, PLACEMENT_DECK))
    got = _exits(lat)
    assert [n for n, _ in got] == [n for n, _ in expected]
    assert [x for _, x in got] == pytest.approx([x for _, x in expected], abs=1e-12)
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(2.0, abs=1e-12)
    assert not meta["warnings"]
    survey = _madx_survey(tmp_path, PLACEMENT_DECK)
    if survey is not None:
        assert survey == [(n, pytest.approx(x, abs=1e-12)) for n, x in expected]


def _exits(lat) -> list:
    """(name, exit position in m) of every named element, bend edges merged."""
    s, out = 0.0, []
    for e in lat.elements:
        s += e.length / 1000
        if not e.name.startswith("DRIFT_") and not e.name.endswith(("_e1", "_e2", "_din", "_dout")):
            out.append((e.name, s))
    return out


def _madx_survey(tmp_path, deck: str) -> list | None:
    cpymad = _madx_or_none()
    if cpymad is None:
        return None
    m = cpymad.Madx(stdout=False)
    try:
        m.input(deck + "\nsurvey;")
        return [(n.split(":")[0], float(s))
                for n, s in zip(m.table.survey.name, m.table.survey.s, strict=True)
                if not n.startswith("drift") and "$" not in n]
    finally:
        m.quit()


PLACEMENT_CASES = {                 # deck body → MAD-X 5.09 SURVEY exits (name, s) and length
    "from_elem": ("qf: quadrupole, l=0.5, k1=0.3; ip: marker;\n"
                  "s: sequence, l=6, refer=centre; ip, at=2; qf, at=1, from=ip; endsequence;",
                  [("ip", 2.0), ("qf", 3.25)], 6.0),
    "from_refer_entry": ("qf: quadrupole, l=0.5, k1=0.3; ip: quadrupole, l=1, k1=0;\n"
                         "s: sequence, l=6, refer=entry; ip, at=2; qf, at=1, from=ip; endsequence;",
                         [("ip", 3.0), ("qf", 4.0)], 6.0),
    "at_deferred": ("qf: quadrupole, l=0.5, k1=0.3;\n"
                    "s: sequence, l=4, refer=centre; qf, at:=pq; endsequence;\npq = 1.5;",
                    [("qf", 1.75)], 4.0),
    "l_deferred": ("qf: quadrupole, l=0.5, k1=0.3;\n"
                   "s: sequence, l:=circ, refer=centre; qf, at=1; endsequence;\ncirc = 5;",
                   [("qf", 1.25)], 5.0),
    "sequence_without_comma": ("qf: quadrupole, l=0.5, k1=0.3;\n"
                               "s: sequence l=3, refer=centre; qf, at=1; endsequence;",
                               [("qf", 1.25)], 3.0),
    "thin_multipole": ("mp: multipole, l=0.2, knl={0, 0.01}; qf: quadrupole, l=0.5, k1=0.3;\n"
                       "s: sequence, l=4, refer=centre; mp, at=1; qf, at=1.5; endsequence;",
                       [("mp", 1.0), ("qf", 1.75)], 4.0),
    "dotted_class": ("qf.c: quadrupole, l=0.5, k1=0.4;\nqf.1: qf.c, k1=0.45;\n"
                     "s: sequence, l=4, refer=centre; qf.1, at=1; q.2: qf.c, at=3; endsequence;",
                     [("qf.1", 1.25), ("q.2", 3.25)], 4.0),
}


@pytest.mark.parametrize("case", sorted(PLACEMENT_CASES))
def test_sequence_positions_as_in_madx(tmp_path, case):
    # FROM= counts from another member; AT:= and L:= are read at USE; a
    # SEQUENCE header may lack its comma; a MULTIPOLE is thin; a class may be
    # a dotted name
    body, expected, length = PLACEMENT_CASES[case]
    deck = BEAM + body + "\nuse, sequence=s;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _exits(lat) == [(n, pytest.approx(x, abs=1e-12)) for n, x in expected]
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(length, abs=1e-12)
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        assert survey == [(n, pytest.approx(x, abs=1e-12)) for n, x in expected]


def test_dotted_class_names_keep_their_strengths(tmp_path):
    body = PLACEMENT_CASES["dotted_class"][0]
    lat, meta = parse_madx(_write(tmp_path, BEAM + body + "\nuse, sequence=s;\n"))
    assert _k1(lat, meta) == pytest.approx({"qf.1": 0.45, "q.2": 0.4}, abs=1e-15)


def test_an_implicit_redefinition_in_a_sequence_is_ignored(tmp_path):
    # MAD-X: "implicit element re-definition ignored" — the first one is placed
    deck = BEAM + ("qf: quadrupole, l=0.5, k1=0.3;\n"
                   "s: sequence, l=4, refer=centre; qf: quadrupole, at=1, k1=0.7; endsequence;\n"
                   "use, sequence=s;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(0.3, abs=1e-15)
    assert any("implicit re-definition of 'qf'" in w for w in meta["warnings"])


def test_const_real_and_int_declarations(tmp_path):
    # MAD-X keeps a CONST's first value ("attempt to redefine constant")
    deck = BEAM + ("const lc = 0.5;\nreal const rc = 0.2;\nreal rr = 0.3;\nint ii = 4;\nlc = 0.9;\n"
                   "qa: quadrupole, l:=lc, k1:=rc; qb: quadrupole, l:=rr, k1:=ii;\n"
                   "r: line=(qa, qb); use, period=r;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert [e.length / 1000 for e in lat.elements] == pytest.approx([0.5, 0.3], abs=1e-15)
    assert _k1(lat, meta) == pytest.approx({"qa": 0.2, "qb": 4.0}, abs=1e-15)
    assert any("lc = 0.9: attempt to redefine the constant ignored" in w for w in meta["warnings"])


def test_beam_attributes_in_expressions(tmp_path):
    # beam->brho etc. as MAD-X computes them (a proton of 0.8 GeV kinetic)
    deck = BEAM + "qf: quadrupole, l=0.5, k1:=5/beam->brho;\nr: line=(qf); use, period=r;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(5 / 4.881030241916092, rel=1e-14)


def test_a_later_beam_changes_only_what_it_gives(tmp_path):
    # MAD-X: the second BEAM keeps the H- charge and mass and takes the new
    # momentum (the importer used to replace the whole BEAM: a proton, every
    # gradient sign flipped)
    deck = ("beam, particle=ion, mass=0.938272+2*0.000510999, charge=-1, energy=1.738;\n"
            "beam, pc=1.5;\nqf: quadrupole, l=0.5, k1=0.3;\nr: line=(qf); use, period=r;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    ref = meta["reference"]
    assert ref.species.charge == -1
    pc = 1500.0
    assert ref.w_kin == pytest.approx(math.hypot(pc, ref.species.mass) - ref.species.mass,
                                      rel=1e-12)


@pytest.mark.parametrize("tilt, hv, sign, note", [
    ("pi/2", 1, 1, False), ("1.5708", 1, 1, True), ("-1.5708", 1, -1, True),
    ("pi", 0, -1, False), ("3.1416", 0, -1, True), ("0", 0, 1, False),
])
def test_bend_tilts_written_rounded_are_recognised(tmp_path, tilt, hv, sign, note):
    # MAD-X's SURVEY: tilt = 1.5708 is a vertical bend to 4e-7, tilt = pi bends
    # the other way in the same plane (decks write the values rounded)
    deck = BEAM + (f"b: sbend, l=1, angle=0.1, tilt={tilt}, e1=0.02, k2=0.5;\n"
                   "r: line=(b); use, period=r;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    (b,) = [e for e in lat.elements if isinstance(e, Dipole)]
    assert b.hv == hv
    assert math.radians(b.angle) == pytest.approx(sign * 0.1, abs=1e-15)
    assert b.k2 == pytest.approx(sign * 0.5, abs=1e-15)
    assert any("taken as ±" in w for w in meta["warnings"]) == note


def test_a_deck_madx_would_refuse_keeps_the_sorted_placement(tmp_path):
    # members out of order: MAD-X stops ("negative drift"); the importer
    # keeps its older rule — sorted by position, overlaps clamped and reported
    deck = BEAM + """
    qf: quadrupole, l=0.5, k1=0.3; mk: marker;
    s: sequence, l=4, refer=centre; qf, at=3; mk, at=1; endsequence;
    use, sequence=s;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    s, pos = 0.0, {}
    for e in lat.elements:
        s += e.length
        pos[e.name] = s
    assert pos["mk"] == pytest.approx(1000.0) and pos["qf"] == pytest.approx(3250.0)
    assert sum(e.length for e in lat.elements) == pytest.approx(4000.0)


def test_an_unclosed_block_is_reported(tmp_path):
    deck = (BEAM + "qf: quadrupole, l=0.5, k1=0.3;\nr: line=(qf); use, period=r;\n"
            "flag = 0;\nif (flag == 1) { qf->k1 = 0.5;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(0.3, abs=1e-15)
    assert any("the block opened by 'if (flag == 1)' is never closed" in w for w in meta["warnings"])


# ---------------------------------------------------------------------------
# IF/WHILE conditions, RETURN, macro arguments, WHILE loops MAD-X computes,
# USE, statements inside a SEQUENCE, FROM=, INT, particle=ion — each value is
# MAD-X 5.09's own.  Every deck ends with a statement of its own: MAD-X drops
# a block that follows another statement on its line when nothing comes after
# it in the file (an end-of-file quirk the importer does not copy).
# ---------------------------------------------------------------------------

QUADS = ("qf: quadrupole, l=0.5, k1=0.3;\nqd: quadrupole, l=0.5, k1=-0.3;\n"
         "qe: quadrupole, l=0.5, k1=0;\nr: line=(qf, qd, qe);\nuse, period=r;\n")


def _strengths(tmp_path, body: str, tail: str = "\nzz_end = 1;\n", *, madx: bool = True):
    """(qf, qd, qe) k1 as the importer reads ``body`` after QUADS, its
    warnings, and the same as MAD-X has them (None without cpymad, or with
    ``madx=False`` for a deck MAD-X would never finish)."""
    deck = BEAM + QUADS + body + tail
    lat, meta = parse_madx(_write(tmp_path, deck))
    got = _k1(lat, meta)
    mad = _madx_values(tmp_path, deck, "r") if madx else None
    return ((got["qf"], got["qd"], got["qe"]), meta["warnings"],
            None if mad is None else (mad["qf"]["k1"], mad["qd"]["k1"], mad["qe"]["k1"]))


CONDITIONS = {    # a = 0, b = 1, c = 1 → (MAD-X runs the IF branch, an illegal construct)
    "a==1 && b==1 || c==1": (False, False),      # false && …: false, whatever follows
    "c==1 || a==1 && b==1": (True, False),       # true || …: true, whatever follows
    "a==1 && b": (False, False),
    "a==0 || b": (True, False),
    "a==1 && b==1 && c==1": (False, False),
    "c==1 || a==0 || b==0": (True, False),
    "a==0 && b==1 || c==0": (False, True),       # not decided by the first: illegal, false
    "a == 0 && b == 1 && c == 1": (False, True),
    "a == 1 || b == 0 || c == 1": (False, True),
    "(a==1 && b==1) || c==1": (False, True),     # a comparison in parentheses
    "a==0 && (b==1)": (False, True),
    "((a==0))": (False, True),
    "1 < b < 3": (False, True),                  # two comparison operators
    "b": (False, True),                          # no comparison
    "a = 0": (False, True),
    "(a+1) == 1": (True, False),                 # parentheses in the arithmetic are fine
    "a == (b-1)": (True, False),
    "-a == 0": (True, False),
    "a <> 1": (True, False),
    "a >= 0 && b <= 1": (True, False),
    "a==0&&b==1": (True, False),
    "qf->k1 > 0.25": (True, False),
    "table(summ, q1) > 0.3": (False, False),     # no table: MAD-X reads 0
    "table(summ, q1) < 0.3": (True, False),
    "(a == 1) || (b == 0)": (True, False),       # a first part that is not one comparison
    "(a==1) && b==1": (True, False),             # counts as not false
    "(a==1) && b==0": (False, False),
    "a || b == 0": (True, False),
    "1 < b < 3 && b == 1": (True, False),
    "(a==1) && (b==1)": (False, True),           # (the second must still be one comparison)
    "a < = 1": (True, False),                    # blanks are dropped
    "a > = 0": (True, False),
    "a = = 0": (True, False),
    "a < > 1": (True, False),
    "a == 1 | | b == 1": (True, False),
    "a == 0 & & b == 1": (True, False),
    "a b == 3": (False, True),                   # (only operator characters merge)
    "a == 0 && a b == 3": (False, True),
    "qf -> k1 > 0.25": (False, True),            # ``->`` takes no blanks
}


@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_if_conditions_are_read_as_madx_reads_them(tmp_path, cond):
    taken, illegal = CONDITIONS[cond]
    body = f"a = 0; b = 1; c = 1;\nif ({cond}) {{\n  qd->k1 = -0.1;\n}} else {{\n  qd->k1 = -0.2;\n}}"
    (_kqf, kqd, _kqe), ws, mad = _strengths(tmp_path, body)
    assert kqd == pytest.approx(-0.1 if taken else -0.2, abs=1e-15)
    assert any("taken as false, as MAD-X does" in w for w in ws) == illegal
    if mad is not None:
        assert mad[1] == pytest.approx(kqd, abs=1e-15)


def test_an_illegal_elseif_is_false_and_the_else_runs(tmp_path):
    body = ("a = 0;\nif (a == 1) {\n  qd->k1 = -0.1;\n} elseif (a) {\n  qd->k1 = -0.15;\n}"
            " else {\n  qd->k1 = -0.2;\n}")
    (_kqf, kqd, _kqe), _ws, mad = _strengths(tmp_path, body)
    assert kqd == pytest.approx(-0.2, abs=1e-15)
    if mad is not None:
        assert mad[1] == pytest.approx(-0.2, abs=1e-15)


@pytest.mark.parametrize("cond, passes", [("n < 3 && m < 10", 3), ("m > 100 || n < 3", 3),
                                          ("n < 3 && m < 10 && n >= 0", 0), ("n", 0),
                                          ("(n < 3) && m < 10", 10)])
def test_while_conditions_are_read_as_madx_reads_them(tmp_path, cond, passes):
    body = (f"n = 0; m = 0;\nwhile ({cond}) {{\n  n = n + 1;\n  m = m + 1;\n"
            "  qd->k1 = qd->k1 - 0.1;\n}")
    (_kqf, kqd, _kqe), _ws, mad = _strengths(tmp_path, body)
    assert kqd == pytest.approx(-0.3 - 0.1 * passes, abs=1e-14)
    if mad is not None:
        assert mad[1] == pytest.approx(kqd, abs=1e-14)


@pytest.mark.parametrize("body", [
    "a = 1;\nif (a == 1) {\n  qf->k1 = 0.1;\n  qd->k1 = -0.12 }\nqe->k1 = 0.3;",
    "a = 0;\nif (a == 1) {\n  qd->k1 = -0.5;\n} else {\n  qf->k1 = 0.1;\n  qd->k1 = -0.12 }\n"
    "qe->k1 = 0.3;",
    "mm: macro = {\n  qf->k1 = 0.1;\n  qd->k1 = -0.12 };\nexec, mm;\nqe->k1 = 0.3;",
], ids=["if", "else", "macro"])
def test_a_statement_with_no_semicolon_before_the_brace_does_not_run(tmp_path, body):
    # MAD-X drops it; the statement after the block runs
    strengths, ws, mad = _strengths(tmp_path, body)
    assert strengths == pytest.approx((0.1, -0.3, 0.3), abs=1e-15)
    assert any("'qd->k1 = -0.12' is not ended by ';' — not run" in w for w in ws)
    if mad is not None:
        assert mad == pytest.approx((0.1, -0.3, 0.3), abs=1e-15)


def test_a_statement_with_no_semicolon_ends_no_while_pass(tmp_path):
    body = "n = 0;\nwhile (n < 3) {\n  n = n + 1;\n  qd->k1 = qd->k1 - 0.1 }\nqe->k1 = 0.3;"
    strengths, _ws, mad = _strengths(tmp_path, body)
    assert strengths == pytest.approx((0.3, -0.3, 0.3), abs=1e-15)
    if mad is not None:
        assert mad == pytest.approx((0.3, -0.3, 0.3), abs=1e-15)


@pytest.mark.parametrize("body", [
    "a = 1;\nif (a == 1) {\n  qf->k1 = 0.1;\n  return;\n  qd->k1 = -0.1;\n}\nqe->k1 = 0.5;",
    "mm: macro = {\n  qf->k1 = 0.1;\n  return;\n  qd->k1 = -0.1;\n};\nexec, mm;\nqe->k1 = 0.5;",
    "inner: macro = {\n  qf->k1 = 0.1;\n  return;\n};\nouter: macro = {\n  exec, inner;\n"
    "  qd->k1 = -0.1;\n};\nexec, outer;\nqe->k1 = 0.5;",
], ids=["if", "macro", "nested_macro"])
def test_return_in_a_block_finishes_the_statement_then_ends_the_file(tmp_path, body):
    strengths, ws, mad = _strengths(tmp_path, body)
    assert strengths == pytest.approx((0.1, -0.1, 0.0), abs=1e-15)
    assert any("RETURN ends the file in MAD-X: the 2 statement(s) after it are not read" in w
               for w in ws)
    if mad is not None:
        assert mad == pytest.approx((0.1, -0.1, 0.0), abs=1e-15)


def test_return_in_a_while_loop_lets_every_pass_run(tmp_path):
    body = ("n = 0;\nwhile (n < 5) {\n  n = n + 1;\n  qd->k1 = qd->k1 - 0.1;\n"
            "  if (n == 2) {\n    return;\n  }\n}\nqe->k1 = 0.5;")
    strengths, _ws, mad = _strengths(tmp_path, body)
    assert strengths == pytest.approx((0.3, -0.8, 0.0), abs=1e-14)
    if mad is not None:
        assert mad == pytest.approx((0.3, -0.8, 0.0), abs=1e-14)


MACRO_CASES = {          # body (defines the members of line m) → MAD-X k1 of them
    "inside_names": ("mk(nn): macro = {\n  qnn: quadrupole, l=0.5, k1=0.3;\n};\nexec, mk(1);\n"
                     "m: line=(q1);", {"q1": 0.3}),
    "in_order": ("mk(n, nn): macro = {\n  qnn: quadrupole, l=0.5, k1=n*0.1;\n};\nexec, mk(2, 7);\n"
                 "m: line=(q22);", {"q22": 0.2}),
    "in_order_reversed": ("mk(nn, n): macro = {\n  qnn: quadrupole, l=0.5, k1=n*0.1;\n};\n"
                          "exec, mk(7, 2);\nm: line=(q7);", {"q7": 0.2}),
    "any_case": ("mk(BIM): macro = {\n  lhcBIM: quadrupole, l=0.5, k1=0.4;\n};\nexec, mk(b1);\n"
                 "m: line=(lhcb1);", {"lhcb1": 0.4}),
    "one_token": ("mk(dk): macro = {\n  qa: quadrupole, l=0.5, k1=2*dk;\n};\nexec, mk(0.1+0.1);\n"
                  "m: line=(qa);", {"qa": 0.2}),
    "one_name": ("x = 0.05; y = 0.2;\nmk(dk): macro = {\n  qa: quadrupole, l=0.5, k1=2*dk;\n};\n"
                 "exec, mk(x+y);\nm: line=(qa);", {"qa": 0.1}),
    "exponent": ("mk(dk): macro = {\n  qa: quadrupole, l=0.5, k1=2*dk;\n};\nexec, mk(2.5e-1);\n"
                 "m: line=(qa);", {"qa": 0.5}),
    "cascade": ("mk(a1, b2): macro = {\n  qa: quadrupole, l=0.5, k1=a1;\n"
                "  qb: quadrupole, l=0.5, k1=b2;\n};\nexec, mk(b2, 0.5);\nm: line=(qa, qb);",
                {"qa": 0.5, "qb": 0.5}),
    "without_comma": ("mk(v): macro = {\n  qa: quadrupole, l=0.5, k1=v;\n};\nexec mk(0.3);\n"
                      "m: line=(qa);", {"qa": 0.3}),
    "without_comma_or_arguments": ("mm: macro = {\n  qa: quadrupole, l=0.5, k1=0.4;\n};\nexec mm;\n"
                                   "m: line=(qa);", {"qa": 0.4}),
    "comma_spaced": ("mm: macro = {\n  qa: quadrupole, l=0.5, k1=0.5;\n};\nexec , mm;\n"
                     "m: line=(qa);", {"qa": 0.5}),
    "dollar": ("mk(nn): macro = {\n  qnn: quadrupole, l=0.5, k1=0.1*nn;\n};\ni = 1;\n"
               "while (i <= 3) {\n  exec, mk($i);\n  i = i + 1;\n}\nm: line=(q1, q2, q3);",
               {"q1": 0.1, "q2": 0.2, "q3": 0.3}),
    "dollar_truncated": ("x = 2.7;\nmk(nn): macro = {\n  qnn: quadrupole, l=0.5, k1=0.3;\n};\n"
                         "exec, mk($x);\nm: line=(q2);", {"q2": 0.3}),
    "signed": ("mk(v): macro = {\n  qa: quadrupole, l=0.5, k1=0.4+v;\n};\nexec, mk(-0.5);\n"
               "m: line=(qa);", {"qa": 0.4}),
    "signed_later": ("mk(pa, pb, pc): macro = {\n  qa: quadrupole, l=0.5, k1=pa pb pc;\n};\n"
                     "exec, mk(0.2, -0.5, 0.1);\nm: line=(qa);", {"qa": 0.2 - 0.5}),
    "signed_first": ("mk(pa, pb, pc): macro = {\n  qa: quadrupole, l=0.5, k1=0.3 + pa pb pc;\n};\n"
                     "exec, mk(-0.5, 2, 3);\nm: line=(qa);", {"qa": 0.3 + 0 - 0.5}),
    "arrow_with_blanks": ("qd: quadrupole, l=0.5, k1=-0.25;\nmk(v): macro = {\n"
                          "  qa: quadrupole, l=0.5, k1=0.1 + 0*v;\n};\nexec, mk(qd -> k1);\n"
                          "m: line=(qa);", {"qa": 0.1}),
    "element_attribute": ("qd: quadrupole, l=0.5, k1=-0.25;\nmk(v): macro = {\n"
                          "  qa: quadrupole, l=0.5, k1=-v;\n};\nexec, mk(qd->k1);\nm: line=(qa);",
                          {"qa": 0.25}),
    "empty": ("x1 = 0.2; x2 = 0.5;\nmk(x1, x2): macro = {\n  qa: quadrupole, l=0.5, k1=x1 + x2;\n};\n"
              "exec, mk();\nm: line=(qa);", {"qa": 0.5}),
}


@pytest.mark.parametrize("case", sorted(MACRO_CASES))
def test_macro_arguments_are_replaced_as_madx_does(tmp_path, case):
    # each parameter is replaced as text wherever it appears (the LHC macros
    # build lhcb1 from lhcBIM), one after the other; an argument is one token
    body, expected = MACRO_CASES[case]
    deck = BEAM + body + "\nuse, period=m;\nzz_end = 1;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta) == pytest.approx(expected, abs=1e-15)
    assert any("one name or number per parameter" in w for w in meta["warnings"]) == \
        (case in ("one_token", "one_name", "signed", "signed_first", "signed_later",
                  "arrow_with_blanks"))
    mad = _madx_values(tmp_path, deck, "m")
    if mad is not None:
        assert {n: mad[n]["k1"] for n in expected} == pytest.approx(expected, abs=1e-15)


def test_a_while_loop_that_needs_twiss_is_not_run(tmp_path):
    # MAD-X runs TWISS on each pass until the tune reaches 0.05 (kq = 0.27
    # after 17 passes); the importer computes no TWISS: the loop is not run,
    # and said so (it used to run to its pass limit: k1 = 1000.1)
    deck = BEAM + ("kq = 0.1; qx = 0;\nqf: quadrupole, l=0.5, k1:=kq;\nqd: quadrupole, l=0.5, k1:=-kq;\n"
                   "d1: drift, l=2;\nr: line=(qf, d1, qd, d1);\nuse, period=r;\n"
                   "while (qx < 0.05) {\n  kq = kq + 0.01;\n  twiss;\n  qx = table(summ, q1);\n}\n"
                   "zz_end = 1;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(0.1, abs=1e-15)
    assert any("WHILE (qx < 0.05) not run: it runs TWISS" in w for w in meta["warnings"])
    assert len(meta["warnings"]) < 10


def test_a_while_loop_that_never_ends_is_undone(tmp_path, monkeypatch):
    import linac_gen.io.madx_parser as mp
    monkeypatch.setattr(mp, "_WHILE_LIMIT", 50)
    body = "n = 0;\nwhile (n < 1) {\n  qf->k1 = qf->k1 + 0.1;\n  ealign, dx=0.001;\n}"
    # (MAD-X itself never leaves this loop: it is not asked)
    (kqf, _kqd, _kqe), ws, _mad = _strengths(tmp_path, body, madx=False)
    assert kqf == pytest.approx(0.3, abs=1e-15)
    assert any("still true after 50 passes: the loop is undone" in w for w in ws)
    assert not any("'ealign'" in w for w in ws)          # (its passes' warnings went too)


def test_a_while_loop_reports_each_warning_once(tmp_path):
    body = "n = 0;\nwhile (n < 3) {\n  n = n + 1;\n  ealign, dx=0.001;\n}"
    _strengths_, ws, _mad = _strengths(tmp_path, body)
    assert sum("'ealign'" in w for w in ws) == 1


USE_CASES = {     # deck → MAD-X 5.09 SURVEY exits and length: positions and lengths at USE
    "at": ("qf: quadrupole, l=0.5, k1=0.3;\ns: sequence, l=4, refer=centre;\nqf, at:=pq;\n"
           "endsequence;\npq = 1.5;\nuse, sequence=s;\npq = 2.2;", [("qf", 1.75)], 4.0),
    "sequence_length": ("qf: quadrupole, l=0.5, k1=0.3;\ns: sequence, l:=circ, refer=centre;\n"
                        "qf, at=1;\nendsequence;\ncirc = 5;\nuse, sequence=s;\ncirc = 7;",
                        [("qf", 1.25)], 5.0),
    "last_use": ("qf: quadrupole, l=0.5, k1=0.3;\ns: sequence, l=4, refer=centre;\nqf, at:=pq;\n"
                 "endsequence;\npq = 1.5;\nuse, sequence=s;\npq = 2.2;\nuse, sequence=s;\npq = 3;",
                 [("qf", 2.45)], 4.0),
    "element_length": ("qf: quadrupole, l:=lq, k1=0.3;\nlq = 0.5;\ns: sequence, l=4, refer=entry;\n"
                       "qf, at=1;\nmk: marker, at=2, from=qf;\nendsequence;\nuse, sequence=s;\n"
                       "lq = 1.0;", [("qf", 1.5), ("mk", 3.25)], 4.0),
}


@pytest.mark.parametrize("case", sorted(USE_CASES))
def test_positions_and_lengths_are_those_of_the_use(tmp_path, case):
    body, expected, length = USE_CASES[case]
    deck = BEAM + body + "\nzz_end = 1;\n"
    lat, _meta = parse_madx(_write(tmp_path, deck))
    assert _exits(lat) == [(n, pytest.approx(x, abs=1e-12)) for n, x in expected]
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(length, abs=1e-12)
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        assert survey == [(n, pytest.approx(x, abs=1e-12)) for n, x in expected]


SEQUENCE_STATEMENT_CASES = {      # deck → MAD-X 5.09 SURVEY exits and length
    "assignment": ("qd: quadrupole, l=0.5, k1=-0.3;\ns: sequence, l=4, refer=centre;\nx = 2.5;\n"
                   "qd, at=x;\nendsequence;", [("qd", 2.75)], 4.0),
    "exec": ("qd: quadrupole, l=0.5, k1=-0.3;\npl(p): macro = {\n  qd, at=p;\n};\n"
             "s: sequence, l=4, refer=centre;\nexec, pl(2);\nendsequence;", [("qd", 2.25)], 4.0),
    "while": ("mk: marker;\ns: sequence, l=4, refer=centre;\ni = 0;\nwhile (i < 3) {\n  i = i + 1;\n"
              "  mk, at=i;\n}\nendsequence;", [("mk", 1.0), ("mk", 2.0), ("mk", 3.0)], 4.0),
    "if": ("qd: quadrupole, l=0.5, k1=-0.3;\nf = 1;\ns: sequence, l=4, refer=centre;\n"
           "if (f == 1) {\n  qd, at=2;\n}\nendsequence;", [("qd", 2.25)], 4.0),
    "attribute": ("qd: quadrupole, l=0.5, k1=-0.3;\ns: sequence, l=4, refer=centre;\n"
                  "qd->k1 = -0.7;\nqd, at=2;\nendsequence;", [("qd", 2.25)], 4.0),
    "from_occurrence": ("qf: quadrupole, l=0.5, k1=0.3;\nip: marker;\ns: sequence, l=12, refer=centre;\n"
                        "ip, at=2;\nip, at=6;\nqf, at=1, from=ip[2];\nendsequence;",
                        [("ip", 2.0), ("ip", 6.0), ("qf", 7.25)], 12.0),
    "from_end": ("qf: quadrupole, l=0.5, k1=0.3;\ns: sequence, l=10, refer=centre;\n"
                 "qf, at=-0.75, from=s$end;\nendsequence;", [("qf", 9.5)], 10.0),
    "from_start": ("qf: quadrupole, l=0.5, k1=0.3;\ns: sequence, l=10, refer=centre;\n"
                   "qf, at=1.5, from=s$start;\nendsequence;", [("qf", 1.75)], 10.0),
}


@pytest.mark.parametrize("case", sorted(SEQUENCE_STATEMENT_CASES))
def test_statements_inside_a_sequence_and_from_as_in_madx(tmp_path, case):
    # an assignment, EXEC, WHILE, IF or ``elem->attr =`` inside a SEQUENCE runs
    # (it used to become a member named ``x`` or ``exec``); FROM= may name the
    # k-th occurrence or the sequence's $start/$end
    body, expected, length = SEQUENCE_STATEMENT_CASES[case]
    deck = BEAM + body + "\nuse, sequence=s;\nzz_end = 1;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _exits(lat) == [(n, pytest.approx(x, abs=1e-12)) for n, x in expected]
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(length, abs=1e-12)
    if case == "attribute":
        assert _k1(lat, meta)["qd"] == pytest.approx(-0.7, abs=1e-15)
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        assert survey == [(n, pytest.approx(x, abs=1e-12)) for n, x in expected]


def test_from_counts_from_the_centre_of_an_element_not_modelled(tmp_path):
    # a collimator (here a marker) keeps its declared length for FROM=
    deck = BEAM + ("qf: quadrupole, l=0.5, k1=0.3; tc: collimator, l=1;\n"
                   "s: sequence, l=6, refer=entry;\ntc, at=2;\nqf, at=1, from=tc;\nendsequence;\n"
                   "use, sequence=s;\nzz_end = 1;\n")
    lat, _meta = parse_madx(_write(tmp_path, deck))
    assert dict(_exits(lat))["qf"] == pytest.approx(4.0, abs=1e-12)
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        assert dict(survey)["qf"] == pytest.approx(4.0, abs=1e-12)


def test_int_declarations_truncate_as_in_madx(tmp_path):
    deck = BEAM + ("int n = 3.7;\nint m = -3.7;\nint k := 2.9;\nint const c = 5.9;\nint r = 3;\n"
                   "r = 4.6;\nqa: quadrupole, l=0.5, k1:=n/10;\nqb: quadrupole, l=0.5, k1:=m/10;\n"
                   "qc: quadrupole, l=0.5, k1:=k/10;\nqd: quadrupole, l=0.5, k1:=c/10;\n"
                   "qe: quadrupole, l=0.5, k1:=r/10;\nln: line=(qa, qb, qc, qd, qe);\nuse, period=ln;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    expected = {"qa": 0.3, "qb": -0.3, "qc": 0.2, "qd": 0.5, "qe": 0.46}
    assert _k1(lat, meta) == pytest.approx(expected, abs=1e-15)
    mad = _madx_values(tmp_path, deck, "ln")
    if mad is not None:
        assert {n: mad[n]["k1"] for n in expected} == pytest.approx(expected, abs=1e-15)


def test_version_twiss_tol_and_an_ion_beam_as_in_madx(tmp_path):
    deck = ("beam, particle=ion;\nversion = 7;\ntwiss_tol = 3e-6;\n"
            "qa: quadrupole, l=0.5, k1:=version/1e6;\nqb: quadrupole, l=0.5, k1:=twiss_tol*1e5;\n"
            "qc: quadrupole, l=0.5, k1:=beam->mass;\nln: line=(qa, qb, qc);\nuse, period=ln;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    expected = {"qa": 0.050902, "qb": 0.3, "qc": 0.93956542052}   # (an ion: NMASS in MAD-X)
    assert _k1(lat, meta) == pytest.approx(expected, rel=1e-15)
    assert any("version = 7: attempt to redefine the built-in constant ignored" in w
               for w in meta["warnings"])
    mad = _madx_values(tmp_path, deck, "ln")
    if mad is not None:
        assert {n: mad[n]["k1"] for n in expected} == pytest.approx(expected, rel=1e-15)


def test_a_deck_testing_the_madx_version_takes_its_branch(tmp_path):
    # CERN sequences: ``if (version >= 50401) { option, -rbarc; }`` — the
    # RBEND length is then the arc (MAD-X: exit at 11, not 11.10493)
    deck = BEAM + ("if (version >= 50401) {\n  option, -rbarc;\n}\nmb: rbend, l=10, angle=0.5;\n"
                   "s: sequence, l=12, refer=entry;\nmb, at=1;\nendsequence;\nuse, sequence=s;\n"
                   "zz_end = 1;\n")
    lat, _meta = parse_madx(_write(tmp_path, deck))
    assert dict(_exits(lat))["mb"] == pytest.approx(11.0, abs=1e-12)
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        assert dict(survey)["mb"] == pytest.approx(11.0, abs=1e-12)


@pytest.mark.parametrize("tail, kqf", [
    ("a = 1;\nif (a == 1) {\n  qf->k1 = 0.5;\n", 0.3),
    ("n = 0;\nwhile (n < 2) {\n  n = n + 1;\n  qf->k1 = qf->k1 + 0.1;\n", 0.3),
    ("a = 0;\nif (a == 1) {\n  qf->k1 = 0.1;\n} else {\n  qf->k1 = 0.5;\n", 0.3),
    ("qf->k1 = 0.1;\n}\nqf->k1 = 0.5;\n", 0.1),
    ("qf->k1 = 0.1;\n{ qf->k1 = 0.2; }\nqf->k1 = 0.5;\n", 0.1),
], ids=["unclosed_if", "unclosed_while", "unclosed_else", "stray_close", "stray_open"])
def test_unclosed_blocks_and_stray_braces_as_in_madx(tmp_path, tail, kqf):
    # MAD-X waits for the closing brace to the end of the file (an unclosed
    # block never runs); a stray brace starts a statement it does not
    # recognise, up to the next ';'
    strengths, _ws, mad = _strengths(tmp_path, tail, tail="")
    assert strengths[0] == pytest.approx(kqf, abs=1e-15)
    if mad is not None:
        assert mad[0] == pytest.approx(kqf, abs=1e-15)


def test_a_cavity_with_no_frequency_is_reported_once(tmp_path):
    # (no MATRIX follows it: the "reference particle lost" note has nothing
    # to say)
    deck = BEAM + "cav: rfcavity, l=0.5, volt=1, lag=0.25;\nr: line=(cav);\nuse, period=r;\n"
    _lat, meta = parse_madx(_write(tmp_path, deck))
    ws = meta["warnings"]
    assert not any("reference particle lost" in w for w in ws)
    assert any("a voltage but neither FREQ nor HARMON" in w for w in ws)


def test_a_macro_calling_itself_inside_a_loop_does_not_hang(tmp_path):
    # two EXECs of itself under an IF that never holds: MAD-X 0.3 at once (a
    # scan of the macro bodies once took 2^50 steps here)
    body = ("x = 0;\nrec: macro = {\n  if (x > 100) {\n    exec, rec;\n    exec, rec;\n  }\n"
            "  x = x + 1;\n};\nn = 0;\nwhile (n < 3) {\n  n = n + 1;\n  exec, rec;\n}\n"
            "qf->k1 = x/10;")
    (kqf, _kqd, _kqe), _ws, mad = _strengths(tmp_path, body)
    assert kqf == pytest.approx(0.3, abs=1e-15)
    if mad is not None:
        assert mad[0] == pytest.approx(0.3, abs=1e-15)


def test_a_while_loop_may_run_exactly_its_pass_limit(tmp_path, monkeypatch):
    import linac_gen.io.madx_parser as mp
    monkeypatch.setattr(mp, "_WHILE_LIMIT", 50)
    body = "n = 0;\nwhile (n < 50) {\n  n = n + 1;\n}\nqf->k1 = n/100;"
    (kqf, _kqd, _kqe), ws, _mad = _strengths(tmp_path, body, madx=False)
    assert kqf == pytest.approx(0.5, abs=1e-15)
    assert not any("undone" in w for w in ws)
    body = "n = 0;\nwhile (n < 51) {\n  n = n + 1;\n}\nqf->k1 = n/100;"
    (kqf, _kqd, _kqe), ws, _mad = _strengths(tmp_path, body, madx=False)
    assert kqf == pytest.approx(0.0, abs=1e-15)
    assert any("still true after 50 passes: the loop is undone" in w for w in ws)


@pytest.mark.parametrize("body, kqf", [
    # a name that only starts like a command, and a TWISS under an IF that never holds
    ("ptc_n = 1;\nn = 0;\nwhile (n < 3) {\n  n = n + 1;\n  ptc_n = ptc_n * 2;\n}\nqf->k1 = ptc_n/100;",
     0.08),
    ("dotwiss = 0;\nn = 0;\nwhile (n < 4) {\n  n = n + 1;\n  qf->k1 = qf->k1 + 0.1;\n"
     "  if (dotwiss == 1) {\n    twiss;\n  }\n}", 0.7),
], ids=["variable_named_like_a_command", "twiss_never_run"])
def test_a_while_loop_is_undone_only_when_a_pass_needs_madx(tmp_path, body, kqf):
    (got, _kqd, _kqe), ws, mad = _strengths(tmp_path, body)
    assert got == pytest.approx(kqf, abs=1e-14)
    assert not any("not run" in w for w in ws)
    if mad is not None:
        assert mad[0] == pytest.approx(kqf, abs=1e-14)


def test_an_undone_loop_leaves_no_notes_behind(tmp_path, monkeypatch):
    import linac_gen.io.madx_parser as mp
    monkeypatch.setattr(mp, "_WHILE_LIMIT", 20)
    body = "n = 0;\nwhile (n < 1) {\n  x = 1/zero;\n  y = ranf();\n  w = nosuch;\n}"
    _s, ws, _mad = _strengths(tmp_path, body, madx=False)
    assert any("the loop is undone" in w for w in ws)
    assert not any("division" in w or "random" in w or "nosuch" in w for w in ws)


USE_ATTRIBUTE_CASES = {     # deck → MAD-X 5.09 SURVEY exits: what is frozen at USE
    "length_set_after": ("q: quadrupole, l=1, k1=0.1;\nm: marker;\ns: sequence, l=10, refer=entry;\n"
                         "q, at=1;\nm, at=5;\nendsequence;\nuse, sequence=s;\nq->l = 2;",
                         [("q", 2.0), ("m", 5.0)]),
    "at_reading_a_length": ("lq = 1;\nq: quadrupole, l:=lq, k1=0.1;\nm: marker;\n"
                            "s: sequence, l=10, refer=entry;\nq, at=0;\nm, at := 2 + q->l;\n"
                            "endsequence;\nuse, sequence=s;\nlq = 2;", [("q", 1.0), ("m", 3.0)]),
    "bend_angle_set_after": ("ang = 0.5;\nmb: rbend, l=10, angle := ang;\n"
                             "s: sequence, l=12, refer=entry;\nmb, at=1;\nendsequence;\n"
                             "use, sequence=s;\nang = 0.1;", [("mb", 11.104931253052678)]),
    "rbarc_set_after": ("mb: rbend, l=10, angle=0.5;\ns: sequence, l=12, refer=entry;\nmb, at=1;\n"
                        "endsequence;\nuse, sequence=s;\noption, -rbarc;",
                        [("mb", 11.104931253052678)]),
}


@pytest.mark.parametrize("case", sorted(USE_ATTRIBUTE_CASES))
def test_the_use_freezes_the_expanded_machine(tmp_path, case):
    body, expected = USE_ATTRIBUTE_CASES[case]
    deck = BEAM + body + "\nzz_end = 1;\n"
    lat, _meta = parse_madx(_write(tmp_path, deck))
    got = dict(_exits(lat))
    for n, x in expected:
        assert got[n] == pytest.approx(x, abs=1e-9), n
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        sv = dict(survey)
        for n, x in expected:
            assert sv[n] == pytest.approx(x, abs=1e-9), n


def test_the_bend_angle_after_use_is_the_late_one(tmp_path):
    # (only the length is frozen: the bend's angle, a strength, is the last one)
    deck = BEAM + USE_ATTRIBUTE_CASES["bend_angle_set_after"][0] + "\nzz_end = 1;\n"
    lat, _meta = parse_madx(_write(tmp_path, deck))
    (b,) = [e for e in lat.elements if isinstance(e, Dipole)]
    assert math.radians(b.angle) == pytest.approx(0.1, abs=1e-15)
    assert b.length / 1000 == pytest.approx(10.104931253052678, abs=1e-12)


def test_int_saturates_at_32_bits_as_in_madx(tmp_path):
    deck = BEAM + ("int k1 = 1e400;\nint k2 := 1e400;\nint k3 = -1e400;\nint k4 = 3e9;\n"
                   "qa: quadrupole, l=0.5, k1:=k1*1e-10;\nqb: quadrupole, l=0.5, k1:=k2*1e-10;\n"
                   "qc: quadrupole, l=0.5, k1:=k3*1e-10;\nqd: quadrupole, l=0.5, k1:=k4*1e-10;\n"
                   "ln: line=(qa, qb, qc, qd);\nuse, period=ln;\nzz_end = 1;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    expected = {"qa": 0.2147483647, "qb": 0.2147483647, "qc": -0.2147483648, "qd": 0.2147483647}
    assert _k1(lat, meta) == pytest.approx(expected, rel=1e-15)
    mad = _madx_values(tmp_path, deck, "ln")
    if mad is not None:
        assert {n: mad[n]["k1"] for n in expected} == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("body, kqf", [
    ("exec.k = 0.4;\nqf->k1 = exec.k;", 0.4),                      # a variable, not EXEC
    ("qf->k1 = 0.4;\nqf->k1 = table(twiss, r$end, betx);", 0.0),   # (a '$' in a table row)
    ("qf->k1 = 0.1;\nreturn, ;\nqf->k1 = 0.5;", 0.1),              # RETURN with a comma
], ids=["exec_dot_variable", "table_row_with_dollar", "return_with_comma"])
def test_names_rows_and_commands_read_as_madx_reads_them(tmp_path, body, kqf):
    (got, _kqd, _kqe), _ws, mad = _strengths(tmp_path, body)
    assert got == pytest.approx(kqf, abs=1e-15)
    if mad is not None:
        assert mad[0] == pytest.approx(kqf, abs=1e-15)


def test_a_last_statement_with_no_semicolon_is_not_run(tmp_path):
    deck = BEAM + QUADS + "qf->k1 = 0.1;\nqf->k1 = 0.5"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(0.1, abs=1e-15)
    assert any("'qf->k1 = 0.5' is not ended by ';' — not run" in w for w in meta["warnings"])
    mad = _madx_values(tmp_path, deck, "r")
    if mad is not None:
        assert mad["qf"]["k1"] == pytest.approx(0.1, abs=1e-15)


def test_a_stray_closing_brace_ends_reading(tmp_path):
    # MAD-X reads nothing after a '}' that closes no block
    tail = "qf->k1 = 0.1;\n}\nqf->k1 = 0.5;\nqd->k1 = -0.5;\n"
    strengths, ws, mad = _strengths(tmp_path, tail, tail="")
    assert strengths == pytest.approx((0.1, -0.3, 0.0), abs=1e-15)
    assert any("a '}' that closes no block ends reading in MAD-X: the 2 statement(s)" in w
               for w in ws)
    if mad is not None:
        assert mad == pytest.approx((0.1, -0.3, 0.0), abs=1e-15)


def test_a_use_before_the_sequence_is_defined_is_skipped(tmp_path):
    # MAD-X: "unknown sequence skipped" — positions are those of the next USE
    deck = BEAM + ("use, sequence=s;\npq = 1.5;\nqf: quadrupole, l=0.5, k1=0.3;\n"
                   "s: sequence, l=4, refer=centre;\nqf, at := pq;\nendsequence;\n"
                   "use, sequence=s;\npq = 3;\nzz_end = 1;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert dict(_exits(lat))["qf"] == pytest.approx(1.75, abs=1e-12)
    assert any("USE of 's' before it is defined — MAD-X skips it" in w for w in meta["warnings"])
    survey = _madx_survey(tmp_path, deck)
    if survey is not None:
        assert dict(survey)["qf"] == pytest.approx(1.75, abs=1e-12)


@pytest.mark.parametrize("body, kqf", [
    ('title, "a}b{c";\nqf->k1 = 0.5;', 0.5),       # the statement balances its braces
    ("qf->k1 = 0.4;\nqf->k1 = -;", 0.0),             # a lone sign is 0
    ("qf->k1 = 0.4;\nqf->k1 = 2*10 + -;", 0.4),      # an illegal expression is not taken
    ("qf->k1 = 0.4;\nqf -> k1 = 0.5;", 0.4),         # (not recognised: blanks in '->')
    ("n = 0;\nwhile (n < 3) {\n  n = n + 1;\n  qf->k1 = qf->k1 + 0.01 + table(summ, q1);\n}",
     0.33),                                          # no table yet: MAD-X reads 0 and loops
    ('qf->k1 = 0.4;\ntitle, "}{}";\nqf->k1 = 0.5;', 0.4),   # unbalanced: reading ends
    ("x = 1;\n}\nmm: multipole, knl={0, 1};\nqf->k1 = 0.5;", 0.3),
], ids=["balanced_braces_in_a_title", "lone_sign", "trailing_sign", "arrow_with_blanks",
        "table_before_any_table", "unbalanced_braces_in_a_title", "stray_brace_then_vector"])
def test_more_statements_read_as_madx_reads_them(tmp_path, body, kqf):
    (got, _kqd, _kqe), _ws, mad = _strengths(tmp_path, body)
    assert got == pytest.approx(kqf, abs=1e-14)
    if mad is not None:
        assert mad[0] == pytest.approx(kqf, abs=1e-14)


FODO_TWISS = ("kq = 0.3;\nqf: quadrupole, l=0.5, k1:=kq;\nqd: quadrupole, l=0.5, k1:=-kq;\n"
              "d1: drift, l=2;\nr: line=(qf, d1, qd, d1);\nuse, period=r;\n")


@pytest.mark.parametrize("body, kq", [
    # after TWISS, table() in a condition still reads 0 in MAD-X: the loop runs
    ("twiss;\nm = 0;\nwhile (table(summ, q1) < 0.05 && m < 3) {\n  m = m + 1;\n  kq = kq + 0.01;\n}",
     0.33),
    # a TWISS inside a loop (undone here) still made the table: a later loop
    # reading it in its body is undone too
    ("n = 0;\nwhile (n < 1) {\n  n = n + 1;\n  twiss;\n}\nm = 0;\nwhile (m < 2) {\n  m = m + 1;\n"
     "  kq = kq + table(summ, q1);\n}", 0.3),
    # only table() written in the condition reads 0 there: through a variable
    # it is the table's value (unknown here) — the loop is undone (MAD-X: 0
    # passes, 0.3)
    ("twiss;\nx := table(summ, q1);\nm = 0;\nwhile (x < 0.05 && m < 3) {\n  m = m + 1;\n"
     "  kq = kq + 0.01;\n}", 0.3),
], ids=["table_in_a_condition", "table_made_in_an_undone_loop", "table_through_a_variable"])
def test_table_reads_after_twiss(tmp_path, body, kq):
    deck = BEAM + FODO_TWISS + body + "\nzz_end = 1;\n"
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qf"] == pytest.approx(kq, abs=1e-14)


def test_exec_with_only_commas_replaces_nothing(tmp_path):
    deck = BEAM + ("x1 = 0.2; x2 = 0.5;\nmk(x1, x2): macro = {\n  qa: quadrupole, l=0.5, k1=x1 + x2;\n};\n"
                   "exec, mk(,);\nm: line=(qa);\nuse, period=m;\nzz_end = 1;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _k1(lat, meta)["qa"] == pytest.approx(0.7, abs=1e-15)
    mad = _madx_values(tmp_path, deck, "m")
    if mad is not None:
        assert mad["qa"]["k1"] == pytest.approx(0.7, abs=1e-15)


def test_a_use_before_any_sequence_still_names_the_line(tmp_path):
    # MAD-X skips it (no sequence in use at all); the importer takes the line
    # it names, as it always did, with no positions frozen there
    deck = BEAM + ("use, sequence=s1;\nqf: quadrupole, l=0.5, k1=0.3;\nqd: quadrupole, l=0.5, k1=-0.3;\n"
                   "s1: sequence, l=2; qf, at=1; endsequence;\n"
                   "s2: sequence, l=2; qd, at=1; endsequence;\nzz_end = 1;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert "qf" in _names(lat) and "qd" not in _names(lat)
    assert any("it still names the line imported" in w for w in meta["warnings"])


def test_a_table_value_reached_through_a_variable_is_not_claimed_exact(tmp_path):
    # after TWISS, ``x := table(summ, q1)`` read in an IF condition is the
    # table's value in MAD-X (0.0558 here); the importer cannot know it and
    # says so — it does not claim MAD-X reads 0 there
    deck = BEAM + FODO_TWISS + ("twiss;\nx := table(summ, q1);\nif (x > 0.05) {\n  kq = 0.1;\n}"
                                " else {\n  kq = 0.2;\n}\nzz_end = 1;\n")
    _lat, meta = parse_madx(_write(tmp_path, deck))
    ws = meta["warnings"]
    assert any("MAD-X's value is unknown here" in w for w in ws)
    assert not any("as MAD-X does there" in w for w in ws)
