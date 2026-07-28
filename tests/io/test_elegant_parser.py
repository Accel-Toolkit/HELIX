"""Elegant .lte importer — hermetic mapping tests + a cross-check against
Cheetah's own from_elegant (the reference oracle) on its test lattices."""
from __future__ import annotations

from tests.dataguard import needs, require  # noqa: E402

import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.io.elegant_parser import parse_elegant
from linac_gen.io.madx_parser import _signed_brho, _brho

# Cheetah lives in the sibling clone; used only as the cross-check oracle.
_CHEETAH = Path("/Users/abhishekpathak/Desktop/Projects/"
                "particle_tracking_codes/tier2_modern/cheetah")
_FODO = _CHEETAH / "tests/resources/fodo.lte"
_CAVITY = _CHEETAH / "tests/resources/cavity.lte"


def _cheetah_seg(path, name):
    import sys
    if str(_CHEETAH) not in sys.path:
        sys.path.insert(0, str(_CHEETAH))
    from cheetah import Segment
    return Segment.from_elegant(str(path), name)


def _has_cheetah():
    try:
        import sys
        if str(_CHEETAH) not in sys.path:
            sys.path.insert(0, str(_CHEETAH))
        import cheetah  # noqa: F401
        return _FODO.is_file()
    except Exception:
        return False


_INLINE = """
! a small self-contained Elegant deck
d1: drift, l=0.5
q1: quad, l=0.1, k1=1.5
q2: kquad, l=0.2, k1=-3
s1: sext, l=0.2, k2=-87.1
b1: sben, l=0.3, angle=0.25, e1=0.1, e2=0.1, hgap=0.02
c1: rfca, l=0.7, phase=90, volt=16175000, freq=1300000000
mk: mark
cell: line=(q1, d1, b1, d1, q2, s1, c1, mk)
main: line=(2*cell)
"""


# ---------------------------------------------------------------------------
# hermetic tests (no external dependency)
# ---------------------------------------------------------------------------
def _write(tmp_path, text) -> str:
    p = tmp_path / "deck.lte"
    p.write_text(text)
    return str(p)


def test_inline_maps_all_types(tmp_path):
    lat, meta = parse_elegant(_write(tmp_path, _INLINE), name="main")
    names = [e.name for e in lat.elements]
    # 2*cell repetition expanded
    assert names.count("q1") == 2 and names.count("b1") == 2
    types = {type(e).__name__ for e in lat.elements}
    assert {"Drift", "Quadrupole", "Multipole", "Dipole", "Edge", "RFGap",
            "Marker"} <= types
    assert meta["warnings"] == []


def test_unit_conversions(tmp_path):
    lat, meta = parse_elegant(_write(tmp_path, _INLINE), name="main",
                              species="proton", w_kin=100.0)
    ref = meta["reference"]
    sbrho = _signed_brho(ref)
    byname = {e.name: e for e in lat.elements}
    # 0.1 m -> 100 mm
    assert byname["q1"].length == pytest.approx(100.0)
    # k1 preserved through gradient = sign(q)*Bρ*k1
    assert byname["q1"].gradient / sbrho == pytest.approx(1.5, rel=1e-9)
    assert byname["q2"].gradient / sbrho == pytest.approx(-3.0, rel=1e-9)
    # sben: angle rad->deg, rho = arc/angle
    assert byname["b1"].angle == pytest.approx(0.25 * 180 / math.pi)
    assert byname["b1"].rho == pytest.approx(300.0 / 0.25)   # mm
    # rfca: V/1e6, phase-90, f/1e6
    assert byname["c1"].voltage == pytest.approx(16.175)
    assert byname["c1"].phase == pytest.approx(0.0)
    assert byname["c1"].frequency == pytest.approx(1300.0)


def test_species_sign_flip(tmp_path):
    """H- flips the gradient sign vs proton for the same k1 (the load-
    bearing convention)."""
    deck = _write(tmp_path, "q: quad, l=0.1, k1=2\nl: line=(q)\n")
    gp = {e.name: e for e in parse_elegant(deck, name="l",
          species="proton")[0].elements}["q"].gradient
    gh = {e.name: e for e in parse_elegant(deck, name="l",
          species="H-")[0].elements}["q"].gradient
    assert gp * gh < 0


def test_rpn_sto_variable(tmp_path):
    """Elegant '% <rpn> sto <var>' RPN definitions resolve into element
    expressions (real XFEL decks use these; Cheetah hard-crashes on them)."""
    deck = ("% 0.5 sto LEN\n"
            "d: drift, l=LEN\n"
            "l: line=(d)\n")
    lat, meta = parse_elegant(_write(tmp_path, deck), name="l")
    assert lat.elements[0].length == pytest.approx(500.0)   # 0.5 m -> mm
    assert meta["warnings"] == []


def test_unresolved_variable_defaults_to_zero_not_nan(tmp_path):
    """An undefined variable in a length must default to 0 with a warning,
    never poison total_length with NaN."""
    deck = "d: drift, l=UNDEFINED_VAR\nl: line=(d)\n"
    lat, meta = parse_elegant(_write(tmp_path, deck), name="l")
    assert math.isfinite(lat.total_length)
    assert lat.elements[0].length == 0.0
    assert any("defaulted to 0" in w for w in meta["warnings"])


def test_reversed_line(tmp_path):
    deck = _write(tmp_path, "a: quad,l=0.1,k1=1\nb: drift,l=0.5\n"
                            "f: line=(a,b)\nr: line=(-f)\n")
    fwd = [e.name for e in parse_elegant(deck, name="f")[0].elements]
    rev = [e.name for e in parse_elegant(deck, name="r")[0].elements]
    assert rev == fwd[::-1]


def test_matrix_element_roundtrips_geometry(tmp_path):
    lat, _ = parse_elegant(_write(tmp_path, _INLINE), name="cell")
    total = lat.total_length
    assert total > 0
    # geometry survives a TraceWin round-trip for the non-matrix deck
    from linac_gen.io.tracewin_writer import write_tracewin
    from linac_gen.io.tracewin_parser import parse_tracewin
    out = tmp_path / "rt.dat"
    write_tracewin(lat, str(out))
    lat2, _ = parse_tracewin(str(out))
    assert lat2.total_length == pytest.approx(total, rel=1e-6)


# ---------------------------------------------------------------------------
# cross-check vs Cheetah (the external anchor) — skipped if unavailable
# ---------------------------------------------------------------------------
skip_no_cheetah = pytest.mark.skipif(not _has_cheetah(),
                                     reason="Cheetah reference not available")


@skip_no_cheetah
def test_fodo_crosscheck_vs_cheetah():
    lat, meta = parse_elegant(str(_FODO), name="fodo")
    sbrho = _signed_brho(meta["reference"])
    hz = {e.name: e for e in lat.elements}
    seg = _cheetah_seg(_FODO, "fodo")
    checked = 0
    for ce in seg.elements:
        cls = type(ce).__name__
        assert ce.name in hz, f"{ce.name} missing from HELIX import"
        he = hz[ce.name]
        # Length check only where HELIX keeps the element thick on the same
        # name.  Thin elements HELIX drift-pads (Sextupole -> Multipole +
        # half-drifts) carry length on the *_din/_dout pads, so the named
        # element is zero-length by construction — its strength check
        # (knl[2] = k2*l) validates the length instead.
        if hasattr(ce, "length") and cls not in ("Sextupole",):
            assert he.length == pytest.approx(float(ce.length) * 1000.0,
                                              rel=1e-5, abs=1e-6)
        if cls == "Quadrupole":
            assert he.gradient / sbrho == pytest.approx(float(ce.k1),
                                                        rel=1e-5)
            checked += 1
        elif cls == "Dipole":
            assert he.angle == pytest.approx(
                float(ce.angle) * 180 / math.pi, rel=1e-5)
            k1 = -he.field_index / (he.rho * 1e-3) ** 2
            assert k1 == pytest.approx(float(ce.k1), abs=1e-6)
            checked += 1
        elif cls == "Sextupole":
            assert he.knl[2] == pytest.approx(
                float(ce.k2) * float(ce.length), rel=1e-5)
            checked += 1
    assert checked >= 5      # exercised quads + dipoles + sextupole


@skip_no_cheetah
def test_cavity_crosscheck_matrix_and_rfgap():
    lat, _ = parse_elegant(str(_CAVITY), name="cavity")
    hz = {e.name: e for e in lat.elements}
    seg = _cheetah_seg(_CAVITY, "cavity")
    # EMATRIX transverse block matches Cheetah CustomTransferMap exactly
    ce = [e for e in seg.elements
          if type(e).__name__ == "CustomTransferMap"][0]
    R = ce.predefined_transfer_map.detach().numpy()
    np.testing.assert_allclose(hz["c1e"].matrix[:4, :4], R[:4, :4],
                               atol=1e-5)
    # RFCA -> RFGap params
    cav = [e for e in seg.elements if type(e).__name__ == "Cavity"][0]
    assert hz["c1"].voltage == pytest.approx(float(cav.voltage) / 1e6,
                                             rel=1e-5)
    assert hz["c1"].frequency == pytest.approx(float(cav.frequency) / 1e6,
                                               rel=1e-5)


# ---------------------------------------------------------------------------
# dispatch + isolation
# ---------------------------------------------------------------------------
def test_dispatch_routes_lte(tmp_path):
    deck = _write(tmp_path, "q: quad,l=0.1,k1=1\nl: line=(q)\n")
    lte = tmp_path / "d.lte"
    Path(deck).rename(lte)
    from linac_gen.cli.common import load_lattice
    from linac_gen.parallel.scan_pool import _parse_lattice_for_scan
    assert len(load_lattice(lte).elements) == 1
    assert len(_parse_lattice_for_scan(str(lte)).elements) == 1


@needs("examples/pipii/mebt/mebt.dat")
def test_existing_parsers_unaffected():
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin("examples/pipii/mebt/mebt.dat")
    assert len(lat.elements) > 0     # TraceWin path still works
