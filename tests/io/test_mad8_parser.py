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
    assert b.angle == pytest.approx(math.degrees(0.0416))


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

    # transport-element lockstep at the transfer-matrix level
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    kinds = (Drift, Quadrupole, Dipole, Edge)

    def transport(els):
        return [e for e in els if isinstance(e, kinds)]
    a, b = transport(lat8.elements), transport(latd.elements)
    assert len(a) == len(b) == 949
    for x, y in zip(a, b):
        assert type(x) is type(y)
        if isinstance(x, Drift):
            assert x.length == pytest.approx(y.length, abs=1e-3)
        else:
            mx = x.transfer_matrix(ref)
            my = y.transfer_matrix(ref)
            assert np.abs(mx - my).max() < 1e-8

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
