"""MAD8 flat-file importer (`parse_mad8`).

The heavyweight anchor compares a native `.lat` import against the
independently generated and exhaustively verified TraceWin conversion of
the same file (examples/pipii/btl/btl_2025v0703.dat), element by element
at the transfer-matrix level — the two paths share no code beyond the
element classes, so a systematic conversion error cannot cancel.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.io.mad8_parser import parse_mad8

_REPO = Path(__file__).resolve().parents[2]
_BTL_LAT = _REPO / "BTL2025v0703.lat"
_BTL_DAT = _REPO / "examples" / "pipii" / "btl" / "btl_2025v0703.dat"

_BRHO = 4.881          # T·m, declared in the BTL file


# ---------------------------------------------------------------------------
# Mini-file helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, text, name="mini.lat"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


_MINI = """
! minimal H- line
BRHO := 4.881
LQ := 0.2
KF := 1.5
D1: DRIFT, L=0.5
QF: QUADRUPOLE, L=LQ, K1=KF
QD: QUADRUPOLE, L=LQ, K1=-KF
CELL: LINE=(D1, QF, D1, QD)
TOP: LINE=(CELL)
RETURN
"""


# ---------------------------------------------------------------------------
# Dialect front-end
# ---------------------------------------------------------------------------

def test_basic_parse_and_units(tmp_path):
    lat, meta = parse_mad8(_write(tmp_path, _MINI))
    kinds = [type(e).__name__ for e in lat.elements]
    assert kinds == ["Drift", "Quadrupole", "Drift", "Quadrupole"]
    assert lat.elements[0].length == pytest.approx(500.0)      # m -> mm
    assert lat.elements[1].length == pytest.approx(200.0)
    assert meta["title"] == "TOP"


def test_charge_sign_hminus_default(tmp_path):
    """H- (default): G = sign(q)·K1·Bρ = -K1·Bρ — the legacy mad2tw
    convention (btl.dat header: variable mad2tw -4.8829)."""
    lat, _ = parse_mad8(_write(tmp_path, _MINI))
    qf = lat.elements[1]
    assert qf.gradient == pytest.approx(-1.5 * _BRHO, rel=1e-9)


def test_charge_sign_proton_flips(tmp_path):
    lat, _ = parse_mad8(_write(tmp_path, _MINI), species="proton")
    assert lat.elements[1].gradient == pytest.approx(+1.5 * _BRHO, rel=1e-9)


def test_continuation_and_comments(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, &
    L=0.25   ! trailing comment on the continued line
TOP: LINE=(D1, D1)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert [e.length for e in lat.elements] == [250.0, 250.0]


def test_deferred_params_and_attr_refs(tmp_path):
    """`NAME[L]` element-attribute references and chained := params must
    resolve — NOT silently coerce to zero (the _gf-default trap)."""
    text = """BRHO := 4.881
A := 0.1
B := 2.0*A
D1: DRIFT, L=B
D2: DRIFT, L=0.899-D1[L]
TOP: LINE=(D1, D2)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert lat.elements[0].length == pytest.approx(200.0)
    assert lat.elements[1].length == pytest.approx(699.0)


def test_sci_notation_not_identifier(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, L=1E-03
TOP: LINE=(D1)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert lat.elements[0].length == pytest.approx(1.0)


def test_line_reversal_and_repetition(tmp_path):
    text = """BRHO := 4.881
DA: DRIFT, L=0.1
DB: DRIFT, L=0.2
SUB: LINE=(DA, DB)
TOP: LINE=(2*DA, -SUB)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert [e.length for e in lat.elements] == [100.0, 100.0, 200.0, 100.0]


def test_apostrophe_param_skipped_not_fatal(tmp_path):
    text = """BRHO := 4.881
QX' := -9.13
D1: DRIFT, L=0.5
TOP: LINE=(D1)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert lat.elements[0].length == pytest.approx(500.0)


def test_negative_drift_survives(tmp_path):
    """MAD overlap-bookkeeping drifts (BTL: DBV3NT = -204.288 mm)."""
    text = """BRHO := 4.881
D1: DRIFT, L=0.5
DN: DRIFT, L=-0.204288
TOP: LINE=(D1, DN, D1)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert lat.elements[1].length == pytest.approx(-204.288)
    tot = sum(e.length for e in lat.elements)
    assert tot == pytest.approx(795.712)


# ---------------------------------------------------------------------------
# Rigidity resolution
# ---------------------------------------------------------------------------

def test_no_rigidity_is_a_hard_error(tmp_path):
    text = """D1: DRIFT, L=0.5
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(D1, Q1)
"""
    with pytest.raises(ValueError, match="rigidity"):
        parse_mad8(_write(tmp_path, text))


def test_brho_argument_fallback(tmp_path):
    text = """D1: DRIFT, L=0.5
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(D1, Q1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text), brho=4.881)
    assert lat.elements[1].gradient == pytest.approx(-4.881, rel=1e-9)
    # Brho = 4.881 T·m inverted with the physical H⁻ ion mass (939.294 MeV)
    # gives 799.52 MeV kinetic.  The source file's nominal "800 MeV" label
    # pairs with 4.881 only under the proton-mass convention (m_p → 799.99) —
    # evidence that the BTL optics-file lineage treats H⁻ as a bare proton.
    assert meta["reference"].w_kin == pytest.approx(799.52, abs=0.2)


def test_beam_statement(tmp_path):
    text = """BEAM, PARTICLE=PROTON, ENERGY=1.938272
D1: DRIFT, L=0.5
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(D1, Q1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["reference"].species.name == "proton"
    assert meta["reference"].w_kin == pytest.approx(1000.0, abs=0.5)
    assert lat.elements[1].gradient > 0        # proton: +K1·Bρ


# ---------------------------------------------------------------------------
# Element mapping specials
# ---------------------------------------------------------------------------

def test_kicker_body_length_preserved(tmp_path):
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.marker import Marker
    text = """BRHO := 4.881
K1: HKICKER, L=0.06, KICK=0.0
TOP: LINE=(K1)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert isinstance(lat.elements[0], Marker)
    assert isinstance(lat.elements[1], Drift)
    assert lat.elements[1].length == pytest.approx(60.0)


def test_zero_angle_rbend_is_drift(tmp_path):
    from linac_gen.elements.drift import Drift
    text = """BRHO := 4.881
B0: RBEND, L=3.05, ANGLE=0.0
TOP: LINE=(B0)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert isinstance(lat.elements[0], Drift)
    assert lat.elements[0].length == pytest.approx(3050.0)


def test_vertical_bend_tilt(tmp_path):
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.edge import Edge
    text = """BRHO := 4.881
BV: RBEND, L=1.05, ANGLE=0.0416, TILT=-1.570796327
TOP: LINE=(BV)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    e1, b, e2 = lat.elements
    assert isinstance(b, Dipole) and b.hv == 1
    assert b.rho > 0                       # TraceWin convention: sign in angle
    assert isinstance(e1, Edge) and e1.hv == 1 and isinstance(e2, Edge)
    # TILT=-pi/2 bends towards -y: the direction goes into the angle's
    # sign (pinned against MAD-X's vertical dispersion in
    # tests/io/test_madx_conventions.py; before 2026-09-03 the sign was
    # dropped and every TILT=-pi/2 bend was imported bending upwards).
    assert b.angle == pytest.approx(-math.degrees(0.0416))


def test_skew_quad_tilt(tmp_path):
    text = """BRHO := 4.881
QS: QUADRUPOLE, L=0.2, K1=0.0, TILT=0.7853981634
TOP: LINE=(QS)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert lat.elements[0].skew_angle == pytest.approx(45.0)


def test_monitor_vs_hvmonitor(tmp_path):
    text = """BRHO := 4.881
M1: MONITOR
HP: HMONITOR
TOP: LINE=(M1, HP)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert not getattr(lat.elements[0], "is_bpm", False)
    assert getattr(lat.elements[1], "is_bpm", False)


def test_unknown_type_warns(tmp_path):
    text = """BRHO := 4.881
X1: ELSEPARATOR, L=0.5
TOP: LINE=(X1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert any("unsupported type" in w for w in meta["warnings"])


# ---------------------------------------------------------------------------
# Auto-declared periodicity
# ---------------------------------------------------------------------------

def test_auto_periods_synthetic(tmp_path):
    from linac_gen.analysis.period_detect import detect_periods
    text = """BRHO := 4.881
D1: DRIFT, L=0.5
QF: QUADRUPOLE, L=0.2, K1=1.5
QD: QUADRUPOLE, L=0.2, K1=-1.5
CELL1: LINE=(D1, QF, D1, QD)
CELL2: LINE=(D1, QF, D1, QD)
CELL3: LINE=(D1, QF, D1, QD)
FODO: LINE=(CELL1, CELL2, CELL3)
TOP: LINE=(FODO)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert len(meta["periods"]) == 1
    assert meta["periods"][0]["n_repeats"] == 3
    # the type-sequence heuristic may independently find the same period;
    # the declared bracket is the lattice_card entry
    ps = [p for p in detect_periods(lat) if p.source == "lattice_card"]
    assert len(ps) == 1 and ps[0].n_repeats == 3


def test_auto_periods_off(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, L=0.5
QF: QUADRUPOLE, L=0.2, K1=1.5
CELL1: LINE=(D1, QF)
CELL2: LINE=(D1, QF)
SEC: LINE=(CELL1, CELL2)
TOP: LINE=(SEC)
"""
    lat, meta = parse_mad8(_write(tmp_path, text), auto_periods=False)
    assert meta["periods"] == []
    from linac_gen.elements.marker import Marker
    assert not any(isinstance(e, Marker) for e in lat.elements)


# ---------------------------------------------------------------------------
# The BTL anchor: native import ↔ verified TraceWin conversion
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (_BTL_LAT.exists() and _BTL_DAT.exists()),
                    reason="BTL v0703 files not present")
def test_btl_anchor_lockstep():
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.edge import Edge
    from linac_gen.elements.quadrupole import Quadrupole

    lat8, meta = parse_mad8(str(_BTL_LAT))
    latd, _ = parse_tracewin(str(_BTL_DAT))

    t8 = sum(float(getattr(e, "length", 0) or 0) for e in lat8.elements)
    td = sum(float(getattr(e, "length", 0) or 0) for e in latd.elements)
    assert t8 == pytest.approx(307969.918, abs=1e-2)
    assert t8 == pytest.approx(td, abs=1e-3)

    # transport-element lockstep at the transfer-matrix level.  The MAD8
    # importer folds TILT=-pi/2 into the angle sign (MAD-X-verified: the
    # bend goes towards -y); the .dat keeps the .lat's angle sign in
    # TraceWin's convention, so the two are vertical mirrors of each
    # other — compare through S = diag(1, 1, -1, -1, 1, 1), which commutes
    # with every y-symmetric element and flips a vertical bend's
    # dispersion column.  (Until 2026-09-06 the deck also carried the
    # SIGNED theta/2 on the EDGE cards of BVDD and ORB1, i.e. edge-
    # focusing rectangular magnets; TraceWin reads the EDGE angle
    # literally, so the deck — not the Edge element — was wrong.)
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    kinds = (Drift, Quadrupole, Dipole, Edge)
    S = np.diag([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])

    def transport(els):
        return [e for e in els if isinstance(e, kinds)]
    a, b = transport(lat8.elements), transport(latd.elements)
    assert len(a) == len(b) == 949
    for x, y in zip(a, b):
        assert type(x) is type(y)
        if isinstance(x, Drift):
            assert x.length == pytest.approx(y.length, abs=1e-3)
        else:
            mx = S @ x.transfer_matrix(ref) @ S
            my = y.transfer_matrix(ref)
            assert np.abs(mx - my).max() < 1e-8, getattr(y, "name", "?")

    # the 10 hand-verified periodicity brackets, bit-equal sets
    p8 = sorted((p.n_repeats, p.label)
                for p in detect_periods(lat8) if p.source != "fallback")
    pd = sorted((p.n_repeats, p.label)
                for p in detect_periods(latd) if p.source != "fallback")
    assert len(p8) == 10 and p8 == pd

    assert meta["reference"].species.name == "H-"
    # Brho = 4.881 T·m inverted with the physical H⁻ ion mass (939.294 MeV)
    # gives 799.52 MeV kinetic.  The source file's nominal "800 MeV" label
    # pairs with 4.881 only under the proton-mass convention (m_p → 799.99) —
    # evidence that the BTL optics-file lineage treats H⁻ as a bare proton.
    assert meta["reference"].w_kin == pytest.approx(799.52, abs=0.2)


@pytest.mark.skipif(not _BTL_LAT.exists(), reason="BTL v0703 .lat not present")
def test_btl_writer_roundtrip(tmp_path):
    """Import .lat → write .dat → re-parse: geometry and declared
    periods must survive (LATTICE markers round-trip, R2)."""
    from linac_gen.io.tracewin_writer import write_tracewin
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.analysis.period_detect import detect_periods

    lat8, _ = parse_mad8(str(_BTL_LAT))
    out = tmp_path / "roundtrip.dat"
    write_tracewin(lat8, str(out))
    lat2, _ = parse_tracewin(str(out))
    t1 = sum(float(getattr(e, "length", 0) or 0) for e in lat8.elements)
    t2 = sum(float(getattr(e, "length", 0) or 0) for e in lat2.elements)
    assert t1 == pytest.approx(t2, abs=1e-2)
    p1 = sorted((p.n_repeats, p.label)
                for p in detect_periods(lat8) if p.source != "fallback")
    p2 = sorted((p.n_repeats, p.label)
                for p in detect_periods(lat2) if p.source != "fallback")
    assert p1 == p2


# ---------------------------------------------------------------------------
# Dispatch + isolation
# ---------------------------------------------------------------------------

def test_load_lattice_dispatch(tmp_path):
    from linac_gen.cli.common import load_lattice
    lat = load_lattice(_write(tmp_path, _MINI))
    assert len(lat.elements) == 4


def test_scan_pool_dispatch(tmp_path):
    from linac_gen.parallel.scan_pool import _parse_lattice_for_scan
    lat = _parse_lattice_for_scan(_write(tmp_path, _MINI))
    assert len(lat.elements) == 4


@pytest.mark.parametrize("dat, n", [
    ("examples/pipii/btl/btl.dat", 960),
])
def test_tracewin_parser_untouched(dat, n):
    from linac_gen.io.tracewin_parser import parse_tracewin
    p = _REPO / dat
    if not p.exists():
        pytest.skip("example not present")
    lat, _ = parse_tracewin(str(p))
    assert len(lat.elements) == n


# ---------------------------------------------------------------------------
# The shared charge-aware helper also fixes MAD-X H- decks
# ---------------------------------------------------------------------------

def test_madx_hminus_sign(tmp_path):
    """A MAD-X deck with BEAM PARTICLE=H- must import with flipped
    gradients (dual-regime counterpart of the proton tests in
    test_madx_parser.py)."""
    from linac_gen.io.madx_parser import parse_madx
    text = """
BEAM, PARTICLE=HMINUS, ENERGY=1.738272;
q1: QUADRUPOLE, l=0.2, k1=2.0;
seq: SEQUENCE, l=0.2;
  q1, at=0.1;
ENDSEQUENCE;
USE, SEQUENCE=seq;
"""
    p = tmp_path / "h.madx"
    p.write_text(text)
    lat, meta = parse_madx(str(p))
    quads = [e for e in lat.elements
             if type(e).__name__ == "Quadrupole"]
    assert quads and quads[0].gradient < 0        # sign(q) flip for H-


# ===========================================================================
# 2026-09-24 language overhaul: functions, constants, `;`, `&` tails,
# CONSTANT/SET, abbreviations, inheritance, nested groups, line arguments,
# SEQUENCE, CALL, USE, BEAM species/energy, fallback rigidity, element
# coverage.  External anchors (MAD-X through cpymad) at the end.
# ===========================================================================

def _elems(lat, cls=None):
    return [e for e in lat.elements
            if cls is None or type(e).__name__ == cls]


def test_functions_constants_and_powers(tmp_path):
    """SQRT & co. used to be looked up as parameters ('unknown identifier
    SQRT') — the BAL exports failed on it.  A negative parameter raised to
    a power used to be substituted as text (-1.5**2 = -2.25)."""
    text = """BRHO := 4.881
A := -1.5
P := A^2
S := SQRT(16.0) + ABS(-1) + MAX(2, 3) + MIN(2, 3) + LOG(EXP(1.0))
T := SIN(PI/2) + COS(0) + TAN(0) + ASIN(1)*2/PI + ATAN(0)
U := 1.0D-3*1000 + 90*RADDEG*DEGRAD/90
M := PMASS*1000 + EMASS*0 + CLIGHT*0 + TWOPI*0
D1: DRIFT, L=P
D2: DRIFT, L=S
D3: DRIFT, L=T
D4: DRIFT, L=U
D5: DRIFT, L=M/1000
TOP: LINE=(D1, D2, D3, D4, D5)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    L = [e.length for e in lat.elements]
    assert L[0] == pytest.approx(2250.0)                 # (-1.5)^2, not -2.25
    assert L[1] == pytest.approx(4000.0 + 1000 + 3000 + 2000 + 1000)
    assert L[2] == pytest.approx(3000.0)
    assert L[3] == pytest.approx(2000.0)
    assert L[4] == pytest.approx(938.27231)              # MAD8's PMASS
    assert not [w for w in meta["warnings"] if "could not be evaluated" in w]


def test_random_functions_evaluate_at_mean_with_warning(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, L=1 + 0.1*GAUSS() + 0.2*(RANF() - 0.5)
TOP: LINE=(D1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert lat.elements[0].length == pytest.approx(1000.0)
    assert sum("random function" in w for w in meta["warnings"]) == 2


def test_unknown_function_is_reported(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, L=USER1(2)
TOP: LINE=(D1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert any("unsupported function" in w for w in meta["warnings"])
    with pytest.raises(ValueError, match="unsupported function"):
        parse_mad8(_write(tmp_path, text), strict=True)


def test_semicolons_and_ampersand_tail(tmp_path):
    """`;` separates statements; text after `&` on a line is ignored (MAD8
    rule — the Main Injector deck writes `K400_SPOOL,&)`)."""
    text = """BRHO := 4.881; LA := 0.5
A: DRIFT, L=LA; B: DRIFT, L=2*LA
TOP: LINE=(A, B,&) this text is ignored
   A, B)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert [e.length for e in lat.elements] == [500.0, 1000.0, 500.0, 1000.0]


def test_comment_block_constant_set(tmp_path):
    text = """BRHO := 4.881
COMMENT
  X: DRIFT, L=99
ENDCOMMENT
LC: CONSTANT = 0.25
SET, LS, 0.75
A: DRIFT, L=LC
B: DRIFT, L=LS
TOP: LINE=(A, B)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert [e.length for e in lat.elements] == [250.0, 750.0]


def test_abbreviations_inheritance_and_attribute_change(tmp_path):
    text = """BRHO := 4.881
QF: QUAD, L=0.2, K1=1.0
QF2: QF, K1=2.0
QF3: QF2
QF3, K1=3.0
SX: SEXT, L=0.2, K2=1.5
TOP: LINE=(QF, QF2, QF3, SX)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    q = _elems(lat, "Quadrupole")
    assert [e.length for e in q] == [200.0] * 3          # L inherited
    assert [e.gradient for e in q] == pytest.approx([-4.881, -9.762, -14.643])
    assert _elems(lat, "Multipole")[0].knl[2] == pytest.approx(0.3)
    assert not [w for w in meta["warnings"] if "unsupported" in w]


def test_type_attribute_is_not_arithmetic(tmp_path):
    text = """BRHO := 4.881
Q1: QUADRUPOLE, TYPE=IQB, L=0.2, K1=1.0
M1: MARKER, TYPE="CELL BOUNDARY"
TOP: LINE=(Q1, M1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert not [w for w in meta["warnings"] if ".type" in w.lower()]


def test_nested_groups_reflection_and_line_arguments(tmp_path):
    text = """BRHO := 4.881
A: DRIFT, L=0.1
B: DRIFT, L=0.2
C: DRIFT, L=0.3
D: DRIFT, L=0.4
CELL(X, Y): LINE=(X, D, Y)
TOP: LINE=(2*(A, -(B, C)), CELL(C, A), -CELL(B, 2*A))
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    names = [e.name for e in lat.elements]
    assert names == ["A", "C", "B", "A", "C", "B",
                     "C", "D", "A",
                     "A", "A", "D", "B"]


def test_use_selects_the_line(tmp_path):
    text = """BRHO := 4.881
A: DRIFT, L=0.1
B: DRIFT, L=0.2
SHORT: LINE=(A)
LONG: LINE=(A, B, A, B)
USE, SHORT
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["title"] == "SHORT" and len(lat.elements) == 1
    assert not [w for w in meta["warnings"] if "top-level" in w]
    lat, meta = parse_mad8(_write(tmp_path, text), line="LONG")
    assert meta["title"] == "LONG" and len(lat.elements) == 4


def test_sequence_placement(tmp_path):
    text = """BRHO := 4.881
Q1: QUADRUPOLE, L=0.2, K1=1.0
M1: MARKER
S: SEQUENCE, REFER=CENTRE
Q1, AT=1.0
QA: QUADRUPOLE, L=0.4, K1=-1.0, AT=2.0
M1, AT=0.5, FROM=QA
ENDSEQUENCE, AT=3.0
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    got = [(type(e).__name__, round(e.length, 9)) for e in lat.elements]
    assert got == [("Drift", 900.0), ("Quadrupole", 200.0), ("Drift", 700.0),
                   ("Quadrupole", 400.0), ("Drift", 300.0), ("Marker", 0.0),
                   ("Drift", 500.0)]


def test_call_includes_a_file(tmp_path):
    (tmp_path / "defs.mad8").write_text(
        "BRHO := 4.881\nQF: QUADRUPOLE, L=0.2, K1=1.0\nRETURN\n"
        "NEVER: DRIFT, L=9\n")
    text = """CALL, FILENAME=defs.mad8
D1: DRIFT, L=0.5
TOP: LINE=(D1, QF)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert [type(e).__name__ for e in lat.elements] == ["Drift", "Quadrupole"]
    assert meta["called_files"] and meta["called_files"][0].endswith("defs.mad8")


def test_action_commands_summarised_not_per_statement(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, L=0.5
TOP: LINE=(D1)
USE, TOP
TWISS, COUPLE
SELECT, FLAG=ERROR, RANGE=D1
EALIGN, DX=0.001
MATCH, LINE=TOP
VARY, NAME=X
ENDMATCH
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert not [w for w in meta["warnings"] if "unrecognised" in w]
    assert any("EALIGN" in w and "nominal machine" in w for w in meta["warnings"])


# ---------------------------------------------------------------------------
# BEAM / rigidity
# ---------------------------------------------------------------------------

def test_beam_mass_and_charge_resolve_hminus(tmp_path):
    """BEAM with MASS/CHARGE but no PARTICLE (lattix's H⁻ export) used to be
    read as a proton — every gradient came out with the wrong sign."""
    text = """BEAM, MASS=0.93929408606, CHARGE=-1, ENERGY=1.73881631804175
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(Q1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["reference"].species.name == "H-"
    assert meta["rigidity_source"] == "beam"
    assert lat.elements[0].gradient == pytest.approx(-4.8810, abs=1e-4)


def test_beam_expression_semicolon_and_pmass(tmp_path):
    text = "BEAM, PARTICLE=proton, Energy=8.0+pmass;\nD1: DRIFT, L=1\nTOP: LINE=(D1)\n"
    _lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["reference"].species.name == "proton"
    assert meta["reference"].w_kin == pytest.approx(8000.0, abs=1e-3)


def test_labelled_beam_statement(tmp_path):
    text = "B0: BEAM, PARTICLE=PROTON, ENERGY=1.938272\nD1: DRIFT, L=1\nTOP: LINE=(D1)\n"
    _lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["reference"].w_kin == pytest.approx(1000.0, abs=1e-3)


def test_electron_beam_keeps_its_rigidity(tmp_path):
    """An electron lattice: magnets use the electron's own Bρ; the tracking
    reference is a proton at that rigidity (said in a warning)."""
    from linac_gen.io.madx_parser import _brho
    text = """BEAM, PARTICLE=ELECTRON, ENERGY=10.0
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(Q1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    m_e = 0.51099906e-3
    brho_e = math.sqrt(10.0 ** 2 - m_e ** 2) * 1e9 / 2.99792458e8
    assert _brho(meta["reference"]) == pytest.approx(brho_e, rel=1e-12)
    assert lat.elements[0].gradient == pytest.approx(brho_e, rel=1e-12)
    assert any("not a HELIX species" in w for w in meta["warnings"])


def test_fallback_beam_only_when_file_has_no_rigidity(tmp_path):
    text = "Q1: QUADRUPOLE, L=0.2, K1=1.0\nTOP: LINE=(Q1)\n"
    lat, meta = parse_mad8(_write(tmp_path, text), fallback_beam=("H-", 800.0))
    assert meta["rigidity_source"] == "fallback_beam"
    assert lat.elements[0].gradient == pytest.approx(-4.8829, abs=1e-4)
    assert any("declares no rigidity" in w for w in meta["warnings"])
    # a BRHO in the file wins over the fallback's energy; the fallback still
    # names the species (the beam that will be tracked): proton sign here
    lat, meta = parse_mad8(_write(tmp_path, "BRHO := 4.881\n" + text),
                           fallback_beam=("proton", 3.0))
    assert meta["rigidity_source"] == "brho_parameter"
    assert meta["reference"].species.name == "proton"
    assert lat.elements[0].gradient == pytest.approx(4.881)


def test_brho_unit_slip_is_refused(tmp_path):
    """The BAL exports' BRHO := P0/C*1.0E11 (P0 in MeV/c, C in cm/s) is
    4883 T·m — 1000x.  Used silently it would scale every magnet 1000x."""
    text = """E0 := 8.0E2
MASS_HMINUS := 9.39294E2
C := 2.997925E10
P0 := SQRT(E0*(2.0*MASS_HMINUS+E0))
BRHO := P0/C*1.0E11
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(Q1)
"""
    with pytest.raises(ValueError, match="impossible for H"):
        parse_mad8(_write(tmp_path, text))
    lat, meta = parse_mad8(_write(tmp_path, text), fallback_beam=("H-", 800.0))
    assert meta["rigidity_source"] == "fallback_beam"
    assert lat.elements[0].gradient == pytest.approx(-4.8829, abs=1e-4)
    # an explicit rigidity wins, without complaint
    lat, meta = parse_mad8(_write(tmp_path, text), brho=4.8829)
    assert meta["rigidity_source"] == "argument"


def test_dispatcher_passes_the_fallback_beam(tmp_path):
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.formats import parse_lattice_file
    p = _write(tmp_path, "Q1: QUADRUPOLE, L=0.2, K1=1.0\nTOP: LINE=(Q1)\n",
               name="x.mad8")
    with pytest.raises(ValueError, match="rigidity"):
        parse_lattice_file(p)
    lat, meta = parse_lattice_file(p, fallback_beam=BeamConfig(species="H-", energy=800.0))
    assert meta["rigidity_source"] == "fallback_beam"


# ---------------------------------------------------------------------------
# Element coverage
# ---------------------------------------------------------------------------

def test_unsupported_class_keeps_its_length(tmp_path):
    text = """BRHO := 4.881
EL: ELENS, L=0.5, CURRENT=1
EZ: ELENS, L=0
TOP: LINE=(EL, EZ)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert [(type(e).__name__, e.length) for e in lat.elements] == [
        ("Drift", 500.0), ("Marker", 0.0)]
    assert any("unsupported type 'elens'" in w for w in meta["warnings"])


def test_octupole_and_mad8_multipole_orders(tmp_path):
    text = """BRHO := 4.881
OC: OCTUPOLE, L=0.2, K3=2.5
MP: MULTIPOLE, K1L=0.1, T1, K2L=0.4, T2=0.1
TOP: LINE=(OC, MP)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    oc, mp = _elems(lat, "Multipole")
    assert [e.length for e in lat.elements[:3]] == [100.0, 0.0, 100.0]
    assert oc.knl == pytest.approx([0, 0, 0, 0.5])
    # bare T1 = pi/4: a normal quad turned into a NEGATIVE skew quad (MAD-X)
    assert mp.knl[1] == pytest.approx(0.0, abs=1e-15)
    assert mp.ksl[1] == pytest.approx(-0.1)
    assert mp.knl[2] == pytest.approx(0.4 * math.cos(0.3))
    assert mp.ksl[2] == pytest.approx(-0.4 * math.sin(0.3))


def test_zero_angle_bend_with_k1_is_a_quadrupole(tmp_path):
    text = "BRHO := 4.881\nQB: SBEND, L=0.6, ANGLE=0.0, K1=1.5\nTOP: LINE=(QB)\n"
    lat, meta = parse_mad8(_write(tmp_path, text))
    (q,) = lat.elements
    assert type(q).__name__ == "Quadrupole"
    assert q.gradient == pytest.approx(-1.5 * 4.881)
    assert not [w for w in meta["warnings"] if "focusing is lost" in w]


def test_bare_quad_tilt_is_45_degrees(tmp_path):
    text = "BRHO := 4.881\nQS: QUADRUPOLE, L=0.2, K1=1.0, TILT\nTOP: LINE=(QS)\n"
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert lat.elements[0].skew_angle == pytest.approx(45.0)


def test_kicker_with_a_kick_becomes_a_steerer(tmp_path):
    from linac_gen.elements.steerer import Steerer
    text = """BRHO := 4.881
KZ: HKICKER, L=0.06, KICK=0.0
KS: KICKER, L=0.1, HKICK=0.001, VKICK=-0.002, TILT=0.3
TOP: LINE=(KZ, KS)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert [type(e).__name__ for e in lat.elements] == [
        "Marker", "Drift", "Drift", "Steerer", "Drift"]
    st = _elems(lat, "Steerer")[0]
    dxp, dyp = st._kick_mrad(meta["reference"])
    c, s = math.cos(0.3), math.sin(0.3)
    assert dxp == pytest.approx((0.001 * c + 0.002 * s) * 1e3, rel=1e-12)
    assert dyp == pytest.approx((0.001 * s - 0.002 * c) * 1e3, rel=1e-12)


def test_rf_cavity_frequency_from_harmon_or_drift(tmp_path):
    from linac_gen.core.constants import C_LIGHT
    text = """BEAM, PARTICLE=PROTON, ENERGY=1.938272
CAV: RFCAVITY, L=0.4, VOLT=0.1, LAG=0.25, HARMON=4
DEAD: RFCAVITY, L=0.4, VOLT=0.0
D1: DRIFT, L=9.2
TOP: LINE=(CAV, DEAD, D1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    gap = _elems(lat, "RFGap")[0]
    ref = meta["reference"]
    beta = ref.bg / math.sqrt(1 + ref.bg ** 2)
    assert gap.frequency == pytest.approx(4 * beta * C_LIGHT / 10.0 / 1e6, rel=1e-12)
    dead = [e for e in lat.elements if e.name == "DEAD"]
    assert len(dead) == 1 and type(dead[0]).__name__ == "Drift"
    assert dead[0].length == pytest.approx(400.0)


def test_lcavity_uses_local_rigidity_downstream(tmp_path):
    """MAD8 LCAVITY changes the reference energy; strengths after it are
    normalised to the local momentum."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.io.madx_parser import _brho
    text = """BEAM, PARTICLE=PROTON, ENERGY=1.938272
Q0: QUADRUPOLE, L=0.2, K1=1.0
LC: LCAVITY, L=1.0, DELTAE=200.0, PHI0=0.0, FREQ=650
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(Q0, LC, Q1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    q0, q1 = _elems(lat, "Quadrupole")
    gap = _elems(lat, "RFGap")[0]
    assert gap.voltage == pytest.approx(200.0) and gap.phase == pytest.approx(0.0)
    r_out = ReferenceParticle(species=PROTON, w_kin=meta["reference"].w_kin + 200.0,
                              frequency=650.0)
    assert q0.gradient == pytest.approx(_brho(meta["reference"]), rel=1e-12)
    assert q1.gradient == pytest.approx(_brho(r_out), rel=1e-12)


# ---------------------------------------------------------------------------
# External anchors: MAD-X (cpymad) reading the same optics
# ---------------------------------------------------------------------------

_ORACLE_MAD8 = """BEAM, PARTICLE=PROTON, ENERGY=1.738272
LQ := 0.25
KF := SQRT(2.25)*0.4
KD := -(KF^2)^0.5
D1: DRIFT, L=0.5
D2: DRIFT, L=D1[L]*0.6
QF: QUAD, L=LQ, K1=KF
QD: QF, K1=KD
SX: SEXT, L=0.2, K2=1.5
OC: OCT, L=0.2, K3=2.5
B1: SBEND, L=1.0, ANGLE=0.1, E1=0.03, E2=0.07, HGAP=0.02, FINT=0.5
BV: RBEND, L=1.05, ANGLE=0.0416, TILT=-1.5707963267949
SOL: SOLENOID, L=0.4, KS=0.3
QB: SBEND, L=0.3, ANGLE=0, K1=0.8
MP: MULTIPOLE, K1L=0.1, T1=0.3
KZ: HKICKER, L=0.06, KICK=0
MON: MONITOR, L=0.1
CELL(X, Y): LINE=(X, D1, Y, D2)
TOP: LINE=(2*(QF, D1, -(QD, D2, SX)), CELL(QD, OC), B1, D1, BV, SOL, &
     D2, QB, MP, KZ, MON, D1)
USE, TOP
"""
# The same optics written by hand in MAD-X (expanded line, evaluated
# numbers, MAD-X's tilted-multipole form): nothing shared with the MAD8
# front end.  OPTION, RBARC=FALSE = MAD8's arc-length RBEND.
_ORACLE_MADX = """option, rbarc=false;
beam, particle=proton, energy=1.738272;
D1: DRIFT, L=0.5; D2: DRIFT, L=0.3;
QF: QUADRUPOLE, L=0.25, K1=0.6; QD: QUADRUPOLE, L=0.25, K1=-0.6;
SX: SEXTUPOLE, L=0.2, K2=1.5; OC: OCTUPOLE, L=0.2, K3=2.5;
B1: SBEND, L=1.0, ANGLE=0.1, E1=0.03, E2=0.07, HGAP=0.02, FINT=0.5;
BV: RBEND, L=1.05, ANGLE=0.0416, TILT=-1.5707963267949;
SOL: SOLENOID, L=0.4, KS=0.3;
QB: SBEND, L=0.3, ANGLE=0, K1=0.8;
MP: MULTIPOLE, KNL={0, 0.1}, TILT=0.3;
KZ: HKICKER, L=0.06, KICK=0; MON: MONITOR, L=0.1;
TOP: LINE=(QF,D1,SX,D2,QD, QF,D1,SX,D2,QD, QD,D1,OC,D2,
           B1,D1,BV,SOL,D2,QB,MP,KZ,MON,D1);
use, period=TOP;
"""


def test_madx_oracle_transfer_matrix(tmp_path):
    """Language features + element conventions end to end: the 4x4
    transverse map of the MAD8 import equals MAD-X's own map of the same
    optics (coupled: solenoid, vertical bend, skew component)."""
    madx = pytest.importorskip("cpymad.madx")
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    lat, meta = parse_mad8(_write(tmp_path, _ORACLE_MAD8))
    Rh = np.asarray(compute_transfer_matrix(lat, meta["reference"].copy()))[:4, :4]
    m = madx.Madx(stdout=False)
    try:
        m.input(_ORACLE_MADX)
        tw = m.twiss(betx=10, bety=10, rmatrix=True)
        Rm = np.array([[tw[f"re{i}{j}"][-1] for j in range(1, 5)]
                       for i in range(1, 5)])
        L_madx = tw.s[-1]
    finally:
        m.quit()
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(L_madx, abs=1e-12)
    assert np.abs(Rh - Rm).max() < 1e-10
    assert abs(Rh[0, 2]) + abs(Rh[2, 0]) > 1e-3           # genuinely coupled


@pytest.mark.parametrize("n, k, t", [(1, 0.7, 0.2), (1, 0.7, None),
                                      (2, 3.0, 0.15), (2, 3.0, None),
                                      (3, 40.0, 0.1)])
def test_madx_oracle_multipole_tilt(tmp_path, n, k, t):
    """MAD8 KnL/Tn → HELIX knl/ksl gives MAD-X's kick of the same
    multipole rotated by Tn (bare Tn = pi/(2(n+1)))."""
    madx = pytest.importorskip("cpymad.madx")
    from linac_gen.core.beam import Beam
    x0, y0 = 0.003, -0.002
    tt = "" if t is None else f"={t}"
    lat, meta = parse_mad8(_write(tmp_path, (
        f"BEAM, PARTICLE=PROTON, ENERGY=1.938272\nMP: MULTIPOLE, K{n}L={k}, T{n}{tt}\n"
        "TOP: LINE=(MP)\n")))
    mp = _elems(lat, "Multipole")[0]
    b = Beam(meta["reference"].copy(), 1, 0.0)
    b.particles[:] = 0.0
    b.particles[0, 0], b.particles[0, 2] = x0 * 1e3, y0 * 1e3
    _alive, dxp, dyp = mp._kick_mrad(b)
    tilt = math.pi / (2 * (n + 1)) if t is None else t
    knl = ",".join(["0"] * n + [repr(k)])
    m = madx.Madx(stdout=False)
    try:
        m.input(f"beam, particle=proton, energy=1.938272; MP: MULTIPOLE, "
                f"KNL={{{knl}}}, TILT={tilt!r}; D: DRIFT, L=0; "
                "TOP: LINE=(MP, D); use, period=TOP;")
        tw = m.twiss(betx=1, bety=1, x=x0, y=y0)
        px, py = tw.px[-1], tw.py[-1]
    finally:
        m.quit()
    assert float(np.ravel(dxp)[0]) * 1e-3 == pytest.approx(px, rel=1e-12, abs=1e-18)
    assert float(np.ravel(dyp)[0]) * 1e-3 == pytest.approx(py, rel=1e-12, abs=1e-18)


def test_madx_oracle_kicker_tilt(tmp_path):
    madx = pytest.importorskip("cpymad.madx")
    from linac_gen.elements.steerer import Steerer
    attrs = "HKICK=0.001, VKICK=-0.002, TILT=0.3"
    lat, meta = parse_mad8(_write(tmp_path, (
        f"BEAM, PARTICLE=PROTON, ENERGY=1.738\nK: KICKER, L=0, {attrs}\n"
        "TOP: LINE=(K)\n")))
    st = _elems(lat, "Steerer")[0]
    dxp, dyp = st._kick_mrad(meta["reference"])
    m = madx.Madx(stdout=False)
    try:
        m.input(f"beam, particle=proton, energy=1.738; K: KICKER, L=0, {attrs}; "
                "D: DRIFT, L=0; TOP: LINE=(K, D); use, period=TOP;")
        tw = m.twiss(betx=1, bety=1)
        px, py = tw.px[-1], tw.py[-1]
    finally:
        m.quit()
    assert dxp * 1e-3 == pytest.approx(px, rel=1e-12)
    assert dyp * 1e-3 == pytest.approx(py, rel=1e-12)


_BAL_2026 = _REPO / "BAL2026V0916.FLAT"


@pytest.mark.skipif(not _BAL_2026.exists(), reason="BAL2026V0916.FLAT not present")
def test_bal_2026_imports_and_matches_madx():
    """The file the SQRT fix was for: with the session beam as fallback it
    imports; the whole line's transverse map equals MAD-X's reading of the
    same file (statements terminated, attribute refs spelled NAME->ATTR,
    definitions made deferred — MAD8 evaluates lazily)."""
    import re
    lat, meta = parse_mad8(str(_BAL_2026), fallback_beam=("H-", 800.0))
    assert meta["rigidity_source"] == "fallback_beam"
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(44.91200377975, abs=1e-9)
    madx = pytest.importorskip("cpymad.madx")
    from linac_gen.io.mad8_parser import _statements
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    params, rest = [], []
    for st in _statements(_BAL_2026.read_text(encoding="latin-1")):
        if st.upper().startswith("RETURN"):
            break
        if re.match(r"^[A-Za-z_][\w.]*'", st):
            continue                               # QX' := … is not MAD-X
        st = re.sub(r"([A-Za-z_][\w.]*)\[\s*(\w+)\s*\]", r"\1->\2", st)
        if re.match(r"^[A-Za-z_][\w.]*\s*:?=", st):
            params.append(re.sub(r"^([A-Za-z_][\w.]*)\s*:?=", r"\1 :=", st) + ";")
        else:
            rest.append(re.sub(r"(?<=[,:])\s*(\w+)\s*=(?!=)", r" \1 :=", st)
                        if "LINE" not in st.upper() else st)
    src = "\n".join(params + [r + ";" for r in rest])
    root = meta["title"]
    Rh = np.asarray(compute_transfer_matrix(lat, meta["reference"].copy()))[:4, :4]
    m = madx.Madx(stdout=False)
    try:
        m.input("option, rbarc=false; beam, particle=proton, energy=1.738;\n" + src)
        m.input(f"use, period={root};")
        tw = m.twiss(betx=10, bety=10, rmatrix=True)
        Rm = np.array([[tw[f"re{i}{j}"][-1] for j in range(1, 5)]
                       for i in range(1, 5)])
    finally:
        m.quit()
    assert np.abs(Rh - Rm).max() < 1e-8


def test_sequence_refer_entry_and_from(tmp_path):
    """FROM is relative to the earlier member's CENTRE, whatever REFER is
    (MAD-X's rule — test_madx_oracle_sequence_from pins it)."""
    text = """BRHO := 4.881
QA: QUADRUPOLE, L=0.4, K1=1.0
QB: QUADRUPOLE, L=0.2, K1=-1.0
S: SEQUENCE, REFER=ENTRY, L=3.0
QA, AT=1.0
QB, AT=1.0, FROM=QA
ENDSEQUENCE
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    got = [(type(e).__name__, round(e.length, 9)) for e in lat.elements]
    # QA occupies 1.0–1.4 (centre 1.2); QB's entry at 1.2 + 1.0 → 2.2–2.4
    assert got == [("Drift", 1000.0), ("Quadrupole", 400.0), ("Drift", 800.0),
                   ("Quadrupole", 200.0), ("Drift", 600.0)]


def test_bare_call_keeps_file_name_case_and_inst_abbreviation(tmp_path):
    (tmp_path / "Defs_MixedCase.mad8").write_text(
        "BRHO := 4.881\nIN1: INST, L=0.2\n")
    text = "CALL, Defs_MixedCase.mad8\nTOP: LINE=(IN1)\n"
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert sum(e.length for e in lat.elements) == pytest.approx(200.0)
    assert not [w for w in meta["warnings"] if "unsupported" in w or "not found" in w]


# ---------------------------------------------------------------------------
# Adversarial-review round (2026-09-24)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("refer", ["ENTRY", "CENTRE", "EXIT"])
def test_madx_oracle_sequence_from(tmp_path, refer):
    """SEQUENCE placement with REFER and FROM against MAD-X's own survey."""
    madx = pytest.importorskip("cpymad.madx")
    body = f"""Q1: QUADRUPOLE, L=0.2, K1=1.0
Q2: QUADRUPOLE, L=0.4, K1=-1.0
M1: MARKER
S: SEQUENCE, REFER={refer}, L=6.0
Q1, AT=1.0
Q2, AT=3.0
M1, AT=0.5, FROM=Q2
ENDSEQUENCE
"""
    lat, _ = parse_mad8(_write(tmp_path, "BRHO := 4.881\n" + body))
    s, pos = 0.0, {}
    for e in lat.elements:
        s += e.length
        pos[e.name] = s / 1000.0
    madx_src = body.replace("\n", ";\n").replace("ENDSEQUENCE;", "ENDSEQUENCE;")
    m = madx.Madx(stdout=False)
    try:
        m.input("beam, particle=proton, energy=1.738;\n" + madx_src + "use, sequence=S;")
        tw = m.twiss(betx=1, bety=1)
        s_m = {n.split(":")[0].upper(): v for n, v in zip(tw.name, tw.s)}
    finally:
        m.quit()
    for name in ("Q1", "Q2", "M1"):
        assert pos[name] == pytest.approx(s_m[name], abs=1e-12), name


@pytest.mark.parametrize("tilt", ["TILT=1.5707963267949", "TILT=0.785398163397448",
                                  "TILT=0.3"])
def test_madx_oracle_zero_angle_bend_keeps_tilt(tmp_path, tilt):
    madx = pytest.importorskip("cpymad.madx")
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    el = f"B0: SBEND, L=1.0, ANGLE=0, K1=0.5, {tilt}"
    lat, meta = parse_mad8(_write(tmp_path, (
        f"BEAM, PARTICLE=PROTON, ENERGY=1.738\n{el}\nTOP: LINE=(B0)\n")))
    Rh = np.asarray(compute_transfer_matrix(lat, meta["reference"].copy()))[:4, :4]
    m = madx.Madx(stdout=False)
    try:
        m.input(f"beam, particle=proton, energy=1.738; {el}; TOP: LINE=(B0); "
                "use, period=TOP;")
        tw = m.twiss(betx=1, bety=1, rmatrix=True)
        Rm = np.array([[tw[f"re{i}{j}"][-1] for j in range(1, 5)]
                       for i in range(1, 5)])
    finally:
        m.quit()
    assert np.abs(Rh - Rm).max() < 1e-12


def test_zero_angle_bend_bare_tilt_is_half_pi(tmp_path):
    """MAD8: a bare TILT on a bend means π/2 (MAD-X reads it as 0, so this
    case is pinned against the explicit value instead)."""
    lat_b, _ = parse_mad8(_write(tmp_path, (
        "BRHO := 4.881\nB0: SBEND, L=1, ANGLE=0, K1=0.5, TILT\nTOP: LINE=(B0)\n")))
    lat_e, _ = parse_mad8(_write(tmp_path, (
        "BRHO := 4.881\nB0: SBEND, L=1, ANGLE=0, K1=0.5, TILT=1.5707963267948966\n"
        "TOP: LINE=(B0)\n")))
    assert lat_b.elements[0].skew_angle == pytest.approx(lat_e.elements[0].skew_angle)
    assert lat_b.elements[0].skew_angle == pytest.approx(90.0)


def test_zero_angle_bend_k2_still_warns(tmp_path):
    text = "BRHO := 4.881\nB0: SBEND, L=1, ANGLE=0, K1=0.5, K2=3.0\nTOP: LINE=(B0)\n"
    _lat, meta = parse_mad8(_write(tmp_path, text))
    assert any("B0.K2" in w for w in meta["warnings"])


def test_lcavity_electron_follows_the_electron_momentum(tmp_path):
    from linac_gen.io.madx_parser import _brho
    text = """BEAM, PARTICLE=ELECTRON, ENERGY=1.0
LC: LCAVITY, L=1.0, DELTAE=100.0, PHI0=0.0, FREQ=1300
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(LC, Q1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    q1 = _elems(lat, "Quadrupole")[0]
    m_e = 0.51099906                                   # MeV
    pc = math.sqrt(1100.0 ** 2 - m_e ** 2)
    assert q1.gradient == pytest.approx(pc * 1e6 / 2.99792458e8, rel=1e-12)
    assert any("substituted BEAM species" in w for w in meta["warnings"])


def test_beam_derived_attribute_reference(tmp_path):
    text = """BEAM, PARTICLE=PROTON, ENERGY=2.0
D1: DRIFT, L=BEAM[PC]
D2: DRIFT, L=BEAM[GAMMA]
TOP: LINE=(D1, D2)
"""
    lat, _ = parse_mad8(_write(tmp_path, text))
    m = 0.93827231
    assert lat.elements[0].length == pytest.approx(math.sqrt(4.0 - m * m) * 1000, rel=1e-12)
    assert lat.elements[1].length == pytest.approx(2.0 / m * 1000, rel=1e-12)


def test_brho_species_follows_the_fallback_beam(tmp_path):
    """A 400 T·m BRHO is impossible for H⁻ but fine for a 120 GeV proton:
    with a proton project beam it must be used, with the proton sign."""
    text = "BRHO := 400\nQ1: QUADRUPOLE, L=0.2, K1=0.01\nTOP: LINE=(Q1)\n"
    lat, meta = parse_mad8(_write(tmp_path, text), fallback_beam=("proton", 8000.0))
    assert meta["rigidity_source"] == "brho_parameter"
    assert meta["reference"].species.name == "proton"
    assert lat.elements[0].gradient == pytest.approx(4.0)
    with pytest.raises(ValueError, match="impossible for H"):
        parse_mad8(_write(tmp_path, text))           # no beam: H⁻ lineage


def test_beam_without_particle_or_energy_is_not_a_rigidity(tmp_path):
    text = """BRHO := 4.881
BEAM, EX=1E-6, EY=1E-6
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(Q1)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["rigidity_source"] == "brho_parameter"
    assert lat.elements[0].gradient == pytest.approx(-4.881)


def test_leading_zeros_quoted_commas_set_attribute_and_bad_unused_line(tmp_path):
    text = """BRHO := 4.881
D1: DRIFT, L=010*0.1, TYPE="A, B"
QF: QUADRUPOLE, L=0.2, K1=1.0
SET, QF[K1], 2.0
CELL(X, Y): LINE=(X, Y)
JUNK: LINE=(CELL(QF))
TOP: LINE=(D1, QF, D1, QF, D1, QF)
"""
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["title"] == "TOP"
    assert lat.elements[0].length == pytest.approx(1000.0)
    assert _elems(lat, "Quadrupole")[0].gradient == pytest.approx(-9.762)
    assert not [w for w in meta["warnings"] if "could not be evaluated" in w
                or "unrecognised" in w]
    assert any("JUNK" in w and "cannot be expanded" in w for w in meta["warnings"])


def test_cli_scan_points_carry_the_project_beam(tmp_path):
    """Scan workers convert a rigidity-less MAD8 deck with the project's
    beam as saved, never the per-point overridden beam; a bare lattice has
    none (it refuses, as `run` does)."""
    import json
    from linac_gen.cli.common import build_scan_point
    from linac_gen.parallel.scan_pool import _parse_lattice_for_scan
    deck = tmp_path / "x.mad8"
    deck.write_text("Q1: QUADRUPOLE, L=0.2, K1=1.0\nTOP: LINE=(Q1)\n")
    proj = tmp_path / "x.lgproj"
    proj.write_text(json.dumps({"__kind__": "linac_gen_project", "__version__": 1,
                                "lattice_path": "x.mad8",
                                "beam": {"species": "H-", "energy": 800.0}}))
    pt = build_scan_point(str(proj), beam_overrides={"energy": "100"})
    assert pt.beam_config["energy"] == pytest.approx(100.0)
    assert pt.lattice_beam["energy"] == pytest.approx(800.0)
    lat = _parse_lattice_for_scan(pt.lattice_path, pt.lattice_beam)
    assert lat.elements[0].gradient == pytest.approx(-4.8829, abs=1e-4)
    bare = build_scan_point(str(deck))
    assert bare.lattice_beam == {}
    with pytest.raises(ValueError, match="rigidity"):
        _parse_lattice_for_scan(bare.lattice_path, bare.lattice_beam or None)


def test_beam_self_reference_and_circular_beam(tmp_path):
    ok = """BEAM, PARTICLE=PROTON, ENERGY=BEAM[MASS]+0.8
Q1: QUADRUPOLE, L=0.2, K1=1.0
TOP: LINE=(Q1)
"""
    _lat, meta = parse_mad8(_write(tmp_path, ok))
    assert meta["reference"].w_kin == pytest.approx(800.0, abs=1e-3)
    bad = ok.replace("ENERGY=BEAM[MASS]+0.8", "PC=BEAM[ENERGY]")
    with pytest.raises(ValueError, match="BEAM"):
        parse_mad8(_write(tmp_path, bad))


def test_apostrophe_identifier_with_zero_digits(tmp_path):
    text = "BRHO := 4.881\nQ'01 := 0.3\nQF: QUADRUPOLE, L=0.5, K1=Q'01\nTOP: LINE=(QF)\n"
    lat, _ = parse_mad8(_write(tmp_path, text))
    assert lat.elements[0].gradient == pytest.approx(-0.3 * 4.881)


def test_charge_only_beam_does_not_override_brho(tmp_path):
    text = "BRHO := 4.881\nBEAM, CHARGE=-1\nQ1: QUADRUPOLE, L=0.2, K1=1.0\nTOP: LINE=(Q1)\n"
    lat, meta = parse_mad8(_write(tmp_path, text))
    assert meta["rigidity_source"] == "brho_parameter"
    assert lat.elements[0].gradient == pytest.approx(-4.881)


def test_lattice_remembers_its_import_beam(tmp_path):
    lat, meta = parse_mad8(_write(tmp_path, "Q1: QUADRUPOLE, L=0.2, K1=1.0\nTOP: LINE=(Q1)\n"),
                           fallback_beam=("H-", 800.0))
    assert lat.mad8_import_beam == {"species": "H-", "energy": pytest.approx(800.0)}
