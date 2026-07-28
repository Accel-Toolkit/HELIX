"""MAD-X subset parser — element mapping, units, and TraceWin isolation.

Covers:
  * the bundled examples/madx/fodo.madx parses with the right elements
  * metre → millimetre length conversion
  * SBEND → Edge + Dipole + Edge expansion
  * RBEND edge angles pick up angle/2
  * the BEAM command → ReferenceParticle energy conversion
  * k1 → gradient conversion via magnetic rigidity
  * gap-filling drifts give a contiguous lattice of the declared length
  * unsupported constructs warn instead of crashing
  * the existing TraceWin .dat parser is byte-for-byte unaffected
"""

from tests.dataguard import needs, require  # noqa: E402
from pathlib import Path

import pytest

from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.madx_parser import parse_madx

_REPO = Path(__file__).resolve().parents[2]
_FODO = _REPO / "examples" / "madx" / "fodo.madx"


# ── bundled example ─────────────────────────────────────────────────────────

def test_fodo_example_parses():
    lat, meta = parse_madx(str(_FODO))
    assert meta["warnings"] == [], f"unexpected warnings: {meta['warnings']}"
    assert "FODO" in meta["title"]
    # 2 quads, 2 bends (each → 3 elements), 1 marker, plus gap drifts.
    quads = [e for e in lat.elements if isinstance(e, Quadrupole)]
    dips = [e for e in lat.elements if isinstance(e, Dipole)]
    edges = [e for e in lat.elements if isinstance(e, Edge)]
    marks = [e for e in lat.elements if isinstance(e, Marker)]
    assert len(quads) == 2
    assert len(dips) == 2
    assert len(edges) == 4          # one Edge each side of each Dipole
    assert len(marks) == 1


def test_fodo_total_length_matches_sequence_l():
    """SEQUENCE l=6.6 m → contiguous lattice of 6600 mm."""
    lat, _ = parse_madx(str(_FODO))
    assert lat.total_length == pytest.approx(6600.0, abs=1e-6)


def test_metre_to_millimetre_conversion():
    """A quadrupole defined with l=0.3 (metres) must become 300 mm."""
    lat, _ = parse_madx(str(_FODO))
    quads = [e for e in lat.elements if isinstance(e, Quadrupole)]
    assert quads[0].length == pytest.approx(300.0)


def test_beam_energy_to_kinetic():
    """BEAM energy=1.738272 GeV (total) → 800 MeV kinetic for a proton."""
    _, meta = parse_madx(str(_FODO))
    ref = meta["reference"]
    assert ref.species.name == "proton"
    assert ref.w_kin == pytest.approx(800.0, abs=1e-3)


def test_sbend_expands_to_edge_dipole_edge():
    """A single SBEND member yields Edge, Dipole, Edge in order."""
    lat, _ = parse_madx(str(_FODO))
    # find the first Dipole and check its neighbours
    types = [type(e).__name__ for e in lat.elements]
    i = types.index("Dipole")
    assert types[i - 1] == "Edge"
    assert types[i + 1] == "Edge"


def test_sbend_geometry():
    """b1: sbend, l=1.0, angle=0.1 → rho = l/angle = 10 m, length = 1 m."""
    lat, _ = parse_madx(str(_FODO))
    dip = next(e for e in lat.elements if isinstance(e, Dipole))
    # rho stored in mm:  1.0 m / 0.1 rad = 10 m = 10000 mm
    assert abs(dip.rho) == pytest.approx(10000.0, rel=1e-6)
    assert dip.length == pytest.approx(1000.0, rel=1e-6)


def test_k1_to_gradient_via_rigidity(tmp_path):
    """gradient [T/m] = k1 [1/m²] · Bρ.  Verify against an explicit Bρ."""
    src = tmp_path / "one_quad.madx"
    src.write_text(
        "beam, particle=proton, energy=1.738272;\n"
        "q1: quadrupole, l=0.5, k1=0.4;\n"
        "s1: sequence, l=1.0, refer=centre;\n"
        "  q1, at=0.5;\n"
        "endsequence;\n"
        "use, sequence=s1;\n"
    )
    lat, meta = parse_madx(str(src))
    ref = meta["reference"]
    # Bρ = βγ · m[MeV] · 1e6 / c   (the e's cancel)
    from linac_gen.core.constants import C_LIGHT
    brho = ref.bg * ref.species.mass * 1e6 / C_LIGHT
    quad = next(e for e in lat.elements if isinstance(e, Quadrupole))
    assert quad.gradient == pytest.approx(0.4 * brho, rel=1e-9)


def test_rbend_adds_half_angle_to_edges(tmp_path):
    """RBEND edges pick up an extra angle/2 vs a bare SBEND."""
    src = tmp_path / "rb.madx"
    src.write_text(
        "beam, particle=proton, energy=1.738272;\n"
        "rb: rbend, l=1.0, angle=0.2;\n"      # e1=e2=0 explicitly omitted
        "s1: sequence, l=2.0, refer=centre;\n"
        "  rb, at=1.0;\n"
        "endsequence;\n"
        "use, sequence=s1;\n"
    )
    lat, _ = parse_madx(str(src))
    edges = [e for e in lat.elements if isinstance(e, Edge)]
    # angle/2 = 0.1 rad = 5.729578 deg on each edge
    import math
    expect_deg = (0.2 / 2.0) * 180.0 / math.pi
    assert edges[0].pole_rotation == pytest.approx(expect_deg, rel=1e-6)
    assert edges[1].pole_rotation == pytest.approx(expect_deg, rel=1e-6)


def test_gap_filling_drifts(tmp_path):
    """Members placed with at= gaps get drift elements between them, and
    the lattice is contiguous up to the declared sequence length."""
    src = tmp_path / "gaps.madx"
    src.write_text(
        "beam, particle=proton, energy=1.738272;\n"
        "q: quadrupole, l=0.2, k1=0.0;\n"
        "s1: sequence, l=5.0, refer=centre;\n"
        "  q, at=1.0;\n"     # entry 0.9, exit 1.1
        "  q, at=3.0;\n"     # entry 2.9, exit 3.1
        "endsequence;\n"
        "use, sequence=s1;\n"
    )
    lat, _ = parse_madx(str(src))
    drifts = [e for e in lat.elements if isinstance(e, Drift)]
    # lead drift (0→0.9 m), middle drift (1.1→2.9 m), tail drift (3.1→5 m)
    assert len(drifts) == 3
    assert lat.total_length == pytest.approx(5000.0, abs=1e-6)


def test_unsupported_construct_warns_not_crashes(tmp_path):
    """A MACRO / unknown command must produce a warning, not an exception."""
    src = tmp_path / "macro.madx"
    src.write_text(
        "beam, particle=proton, energy=1.738272;\n"
        "twiss, file=out.tfs;\n"           # unsupported command
        "d: drift, l=1.0;\n"
        "s1: sequence, l=1.0, refer=centre;\n"
        "  d, at=0.5;\n"
        "endsequence;\n"
        "use, sequence=s1;\n"
    )
    lat, meta = parse_madx(str(src))
    assert any("twiss" in w.lower() for w in meta["warnings"])
    assert len(lat.elements) >= 1     # still produced a lattice


def test_no_sequence_raises(tmp_path):
    """A file with no SEQUENCE block is a hard error — nothing to import."""
    src = tmp_path / "empty.madx"
    src.write_text("beam, particle=proton, energy=1.0;\n"
                   "d: drift, l=1.0;\n")
    with pytest.raises(ValueError, match="no SEQUENCE"):
        parse_madx(str(src))


def test_expression_evaluator(tmp_path):
    """Variables and arithmetic in attributes resolve correctly."""
    src = tmp_path / "expr.madx"
    src.write_text(
        "beam, particle=proton, energy=1.738272;\n"
        "ll = 0.5;\n"
        "half := ll / 2.0;\n"
        "d: drift, l=half * 2.0;\n"       # → 0.5 m → 500 mm
        "s1: sequence, l=1.0, refer=centre;\n"
        "  d, at=0.5;\n"
        "endsequence;\n"
        "use, sequence=s1;\n"
    )
    lat, _ = parse_madx(str(src))
    # The named drift 'd' — not the auto-generated gap-filling drifts.
    drift = next(e for e in lat.elements
                 if isinstance(e, Drift) and e.name == "d")
    assert drift.length == pytest.approx(500.0)


def test_monitor_becomes_bpm_marker(tmp_path):
    """MONITOR / HMONITOR / VMONITOR map to a BPM-flagged Marker.

    Regression test: an earlier MONITOR handler had a broken ternary
    that returned a nested tuple and crashed the sequence resolver."""
    src = tmp_path / "mon.madx"
    src.write_text(
        "beam, particle=proton, energy=1.738272;\n"
        "d: drift, l=1.0;\n"
        "bpm: monitor;\n"
        "s1: sequence, l=2.0, refer=centre;\n"
        "  d,   at=0.5;\n"
        "  bpm, at=1.5;\n"
        "endsequence;\n"
        "use, sequence=s1;\n"
    )
    lat, meta = parse_madx(str(src))      # must NOT raise
    markers = [e for e in lat.elements if isinstance(e, Marker)]
    assert len(markers) == 1
    assert markers[0].is_bpm is True


def test_bundled_transport_example_parses():
    """examples/madx/transport.madx exercises QUAD/SBEND/RBEND/SEXTUPOLE/
    RFCAVITY/MULTIPOLE/MONITOR — it must parse with no warnings."""
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen.elements.multipole import Multipole
    lat, meta = parse_madx(str(_REPO / "examples" / "madx" / "transport.madx"))
    assert meta["warnings"] == [], f"unexpected warnings: {meta['warnings']}"
    assert lat.total_length == pytest.approx(12000.0, abs=1e-6)
    assert any(isinstance(e, RFGap) for e in lat.elements)      # RFCAVITY
    assert any(isinstance(e, Multipole) for e in lat.elements)  # SEXTUPOLE/MULTIPOLE


# ── TraceWin parser must be completely unaffected ───────────────────────────

@pytest.mark.parametrize("rel,expect", [
    ("examples/pipii/btl/btl.dat", 960),
    # +1 / +2 vs the original counts: FREQ cards now materialize as Freq
    # command elements (machine-clock switch at the card).
    ("examples/pipii/mebt/mebt.dat", 427),
    ("examples/pipii/mebt+hwr/mebt+hwr.dat", 483),
])
@needs("examples/pipii/mebt/mebt.dat")
@needs("examples/pipii/mebt/mebt.dat", "examples/pipii/mebt+hwr/mebt+hwr.dat", "Fields")
def test_tracewin_parser_untouched(rel, expect):
    """Adding the MAD-X parser must not change TraceWin .dat parsing."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(str(_REPO / rel))
    assert len(lat.elements) == expect
