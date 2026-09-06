"""``write_madx`` — the TraceWin(.dat) → MAD-X exporter.

Four tiers:

T1  exact inverse of the importer's formulas, element by element, with
    random parameters, for BOTH charge signs (proton / H⁻) and BOTH bend
    orientations (horizontal / vertical), plus an RBEND-imported line;
T2  round trips on every public example deck: element sequence, names,
    attributes at 1e-12, the composed 6×6 transfer matrix of the original
    vs the re-imported lattice (bit-for-bit or ≤1e-12 relative), and
    byte-idempotence  export(import(export(L))) == export(L);
T3  the unsupported-element paths (marker + body drift + warning, or
    ``on_unsupported="error"``) and every other warning;
T4  the output is MAD-X-legal: unique legal names, finite numbers, the
    importer's own statement splitter accepts every line, and the file
    ends with USE.
The real-MAD-X (cpymad) cross-checks live in tests/io/test_madx_oracle.py.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from linac_gen.cli.common import build_ref
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.lattice_commands import LatticeCommand
from linac_gen.elements.marker import Marker
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.thin_lens import ThinLens
from linac_gen.io.madx_parser import _split_statements, _strip_comments, parse_madx
from linac_gen.io.madx_writer import write_madx
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

REPO = Path(__file__).resolve().parents[2]

# Public decks (all survive scripts/cut_public_release.sh) and the species
# they can be tracked with (a proton DTL decelerates H⁻ below zero energy).
PUBLIC_DECKS = [
    ("examples/fodo_cell.dat", ("proton", "H-")),
    ("examples/impactx_features/coupled_fodo.dat", ("proton", "H-")),
    ("examples/solenoid_channel.dat", ("proton", "H-")),
    ("examples/csr_chicane.dat", ("proton", "H-")),
    ("examples/correction_demo/correction_demo.dat", ("proton", "H-")),
    ("examples/lebt_scc_demo.dat", ("proton",)),
    ("examples/matching_demo.dat", ("proton",)),
    ("examples/dtl_section.dat", ("proton",)),
    ("examples/hofmann_stability/hofmann_demo_linac.dat", ("proton",)),
]


def _ref(species="proton", energy=3.0):
    return build_ref(BeamConfig(species=species, energy=energy, frequency=352.21))


def _physical(lat):
    """Elements that carry physics (commands / SC comp become comments)."""
    return [e for e in lat.elements
            if not isinstance(e, (LatticeCommand, SpaceChargeComp))]


def _matrix(lat, ref):
    return np.asarray(compute_transfer_matrix(lat, copy.deepcopy(ref)))


# ---------------------------------------------------------------------------
# T1 — exact inverse of the importer, per element type
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("species", ["proton", "H-"])
@pytest.mark.parametrize("hv", [0, 1])
def test_every_element_type_round_trips_exactly(tmp_path, species, hv):
    rng = np.random.default_rng(20260903 + hv)
    ref = _ref(species)
    lat = Lattice()
    lat.add(Drift(name="D1", length=float(rng.uniform(50, 500))))
    lat.add(Quadrupole(name="QF", length=float(rng.uniform(50, 300)),
                       gradient=float(rng.uniform(-20, 20)),
                       skew_angle=float(rng.uniform(-40, 40))))
    rho = float(rng.uniform(800, 5000))
    ang = float(rng.uniform(5, 30))
    lat.add(Edge(name="B1_e1", pole_rotation=float(rng.uniform(-10, 10)), rho=rho,
                 gap=40.0, k1=0.5, hv=hv))
    lat.add(Dipole(name="B1", angle=ang, rho=rho,
                   field_index=float(rng.uniform(-0.6, 0.6)), hv=hv))
    lat.add(Edge(name="B1_e2", pole_rotation=float(rng.uniform(-10, 10)), rho=rho,
                 gap=40.0, k1=0.5, hv=hv))
    lat.add(Solenoid(name="S1", length=float(rng.uniform(100, 400)),
                     field=float(rng.uniform(-3, 3))))
    lat.add(RFGap(name="G1", voltage=float(rng.uniform(0.1, 2.0)),
                  phase=float(rng.uniform(-90, 60)), frequency=352.21, ttf=1.0))
    lat.add(Steerer(name="ST1", bx_l=float(rng.uniform(-3e-3, 3e-3)),
                    by_l=float(rng.uniform(-3e-3, 3e-3))))
    lat.add(Multipole(name="M1", knl=[0.0, 0.0, float(rng.uniform(-2, 2))],
                      ksl=[0.0, float(rng.uniform(-1, 1))]))
    lat.add(Marker(name="BPM1", is_bpm=True))
    lat.add(Marker(name="MK1"))
    lat.add(Aperture(name="AP1", dx=12.0, dy=8.0, aperture_type=0))
    lat.add(Aperture(name="AP2", dx=15.0, dy=15.0, aperture_type=1))
    lat.add(Drift(name="D2", length=123.456))

    out = tmp_path / "t1.madx"
    warnings = write_madx(lat, out, ref)
    # the only admissible note: the random gap accelerates the line
    assert [w for w in warnings if "line accelerates" not in w] == []
    lat2, meta = parse_madx(str(out))
    assert meta["warnings"] == []
    assert meta["reference"].species.name == ref.species.name
    assert meta["reference"].w_kin == pytest.approx(ref.w_kin, rel=1e-12)

    a, b = _physical(lat), _physical(lat2)
    assert [type(e).__name__ for e in a] == [type(e).__name__ for e in b]
    assert [e.name.lower() for e in a] == [e.name.lower() for e in b]
    for e1, e2 in zip(a, b):
        for attr in ("length", "gradient", "skew_angle", "angle", "rho",
                     "field_index", "hv", "pole_rotation", "gap", "k1",
                     "field", "voltage", "phase", "frequency", "ttf",
                     "bx_l", "by_l", "dx", "dy", "aperture_type", "is_bpm"):
            if hasattr(e1, attr):
                v1, v2 = getattr(e1, attr), getattr(e2, attr)
                if isinstance(v1, (bool, int)):
                    assert v1 == v2, (e1.name, attr)
                else:
                    assert float(v1) == pytest.approx(float(v2), rel=1e-12, abs=1e-15), \
                        (e1.name, attr, v1, v2)
        if isinstance(e1, Multipole):
            assert e1.knl == pytest.approx(e2.knl, rel=1e-12)
            assert e1.ksl == pytest.approx(e2.ksl, rel=1e-12)
    # and the physics: composed transfer matrix identical
    np.testing.assert_allclose(_matrix(lat, ref), _matrix(lat2, ref), rtol=1e-12, atol=1e-12)


def test_rbend_imported_line_reexports_as_equivalent_sbend(tmp_path):
    """An RBEND (edges get ±angle/2 on import) exports as an SBEND with the
    folded edge angles and re-imports to the same elements."""
    src = tmp_path / "rb.madx"
    src.write_text("""
    beam, particle=proton, energy=0.938272+0.003;
    b: rbend, l=0.8, angle=0.2, e1=0.02, e2=-0.01, k1=0.3;
    seq: sequence, refer=entry, l=0.8; b, at=0; endsequence;
    use, sequence=seq;
    """)
    lat, meta = parse_madx(str(src))
    out = tmp_path / "rb_out.madx"
    assert write_madx(lat, out, meta["reference"]) == []
    txt = out.read_text()
    assert "SBEND" in txt and "RBEND" not in txt
    lat2, _ = parse_madx(str(out))
    for e1, e2 in zip(_physical(lat), _physical(lat2)):
        assert type(e1) is type(e2)
        for attr in ("pole_rotation", "angle", "rho", "field_index", "length"):
            if hasattr(e1, attr):
                assert float(getattr(e1, attr)) == pytest.approx(float(getattr(e2, attr)), rel=1e-12)
    np.testing.assert_allclose(_matrix(lat, meta["reference"]),
                               _matrix(lat2, meta["reference"]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("species", ["proton", "H-", "deuteron"])
def test_beam_line_round_trips_species_and_energy(tmp_path, species):
    ref = _ref(species, energy=7.5)
    lat = Lattice(); lat.add(Drift(name="D", length=100.0))
    out = tmp_path / "beam.madx"
    write_madx(lat, out, ref)
    _, meta = parse_madx(str(out))
    r2 = meta["reference"]
    assert r2.species.name == ref.species.name
    assert r2.species.charge == ref.species.charge
    assert r2.w_kin == pytest.approx(7.5, rel=1e-12)
    assert not meta["warnings"]


# ---------------------------------------------------------------------------
# T2 — public decks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deck,species_list", PUBLIC_DECKS)
def test_public_deck_round_trip_matrix_parity(tmp_path, deck, species_list):
    lat, meta = parse_tracewin(str(REPO / deck))[:2]
    for species in species_list:
        ref = _ref(species)
        out = tmp_path / f"{Path(deck).stem}_{species}.madx"
        warnings = write_madx(lat, out, ref)
        lat2, meta2 = parse_madx(str(out))
        assert meta2["warnings"] == [], meta2["warnings"]
        a, b = _physical(lat), _physical(lat2)
        assert [type(e).__name__ for e in a] == [type(e).__name__ for e in b]
        # edge names cannot survive (MAD-X folds them into the SBEND; the
        # importer re-creates them as <bend>_e1/_e2) — everything else must
        assert [e.name.lower() for e in a if not isinstance(e, Edge)] == \
               [e.name.lower() for e in b if not isinstance(e, Edge)]
        La = sum(float(getattr(e, "length", 0.0) or 0.0) for e in a)
        Lb = sum(float(getattr(e, "length", 0.0) or 0.0) for e in b)
        assert La == pytest.approx(Lb, rel=1e-13)
        M1, M2 = _matrix(lat, ref), _matrix(lat2, ref)
        scale = max(np.abs(M1).max(), 1.0)
        np.testing.assert_allclose(M1, M2, rtol=1e-12, atol=1e-12 * scale)
        # nothing is silently dropped: every command card became a comment
        n_cmd = sum(1 for e in lat.elements if isinstance(e, (LatticeCommand, SpaceChargeComp)))
        txt = out.read_text()
        assert txt.count("! HELIX:") >= n_cmd
        # warnings, if any, must be about things MAD-X cannot hold
        for w in warnings:
            assert any(k in w for k in ("misalignment", "quadrupole g", "K2",
                                        "line accelerates",
                                        "transit-time", "absolute-phase")), w


@pytest.mark.parametrize("deck,_", PUBLIC_DECKS)
def test_public_deck_export_is_idempotent(tmp_path, deck, _):
    lat = parse_tracewin(str(REPO / deck))[0]
    ref = _ref("proton")
    out1 = tmp_path / "a.madx"
    write_madx(lat, out1, ref, sequence_name="s", title="t")
    lat2, meta2 = parse_madx(str(out1))
    out2 = tmp_path / "b.madx"
    # same reference both times: the importer's reference takes the first
    # cavity's frequency, which only changes the informational header
    write_madx(lat2, out2, ref, sequence_name="s", title="t")
    l1 = out1.read_text().splitlines()[1:]      # drop the dated header line
    l2 = out2.read_text().splitlines()[1:]
    # comment lines carrying TraceWin commands do not survive an import
    l1 = [x.lower() for x in l1 if not x.strip().startswith("! HELIX:")]
    l2 = [x.lower() for x in l2 if not x.strip().startswith("! HELIX:")]
    # MAD-X is case-insensitive and the importer lower-cases names, so the
    # comparison is case-insensitive too; everything else must be byte-equal
    assert l1 == l2


def test_matching_cards_become_comments(tmp_path):
    lat = parse_tracewin(str(REPO / "examples/matching_demo.dat"))[0]
    out = tmp_path / "m.madx"
    write_madx(lat, out, _ref())
    txt = out.read_text()
    assert "! HELIX: ADJUST" in txt and "! HELIX: SET_SIZE" in txt


# ---------------------------------------------------------------------------
# T3 — unsupported elements and warnings
# ---------------------------------------------------------------------------
def _unsupported_lattice():
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0))
    lat.add(MatrixElement(name="MX", matrix=np.eye(6), length=250.0))
    lat.add(ThinLens(name="TL", fx=500.0, fy=-700.0))
    lat.add(Steerer(name="ES", bx_l=1.0, by_l=0.0, elec=True))
    lat.add(Drift(name="D2", length=100.0))
    return lat


def test_unsupported_elements_keep_geometry_and_warn(tmp_path):
    lat = _unsupported_lattice()
    out = tmp_path / "u.madx"
    warnings = write_madx(lat, out, _ref())
    assert len(warnings) == 3
    assert any("'MX'" in w and "MARKER + body DRIFT" in w for w in warnings)
    assert any("'TL'" in w for w in warnings)
    assert any("'ES'" in w and "electrostatic" in w for w in warnings)
    lat2, meta = parse_madx(str(out))
    assert not meta["warnings"]
    assert sum(e.length for e in lat2.elements) == pytest.approx(450.0)
    names = [e.name.lower() for e in lat2.elements]
    assert "mx" in names and "mx_body" in names and "tl" in names and "es" in names


def test_unsupported_elements_refused_in_strict_mode(tmp_path):
    with pytest.raises(ValueError, match="MAD-X export refused: 'MX'"):
        write_madx(_unsupported_lattice(), tmp_path / "u.madx", _ref(),
                   on_unsupported="error")


def test_quad_multipole_content_ttf_and_misalignment_warn(tmp_path):
    lat = Lattice()
    q = Quadrupole(name="Q", length=100.0, gradient=5.0, g3=0.2)
    q.dx = 0.3
    lat.add(q)
    lat.add(RFGap(name="G", voltage=1.0, phase=-30.0, frequency=352.21, ttf=0.8))
    lat.add(Aperture(name="PP", dx=5.0, dy=5.0, aperture_type=2))
    out = tmp_path / "w.madx"
    warnings = write_madx(lat, out, _ref())
    joined = "\n".join(warnings)
    assert "quadrupole g3" in joined
    assert "misalignment dx" in joined
    assert "transit-time factor T=0.8 folded" in joined
    assert "'PP'" in joined and "aperture type 2" in joined
    # the TTF is folded so the MAD-X reference gain equals HELIX's
    lat2, meta = parse_madx(str(out))
    g2 = [e for e in lat2.elements if isinstance(e, RFGap)][0]
    assert g2.voltage * g2.ttf == pytest.approx(0.8, rel=1e-12)
    assert g2.phase == pytest.approx(-30.0, rel=1e-12)


def test_bad_on_unsupported_value(tmp_path):
    lat = Lattice(); lat.add(Drift(name="D", length=10.0))
    with pytest.raises(ValueError):
        write_madx(lat, tmp_path / "x.madx", _ref(), on_unsupported="ignore")


# ---------------------------------------------------------------------------
# T4 — MAD-X legality of the text
# ---------------------------------------------------------------------------
def test_output_is_madx_legal_text(tmp_path):
    lat = Lattice()
    lat.add(Drift(name="drift", length=100.0))          # reserved word
    lat.add(Quadrupole(name="1Q", length=100.0, gradient=1.0))   # leading digit
    lat.add(Quadrupole(name="Q#2", length=100.0, gradient=1.0))  # illegal char
    lat.add(Quadrupole(name="q_2", length=100.0, gradient=1.0))  # collides after sanitising
    lat.add(Marker(name=""))                                       # empty name
    out = tmp_path / "legal.madx"
    write_madx(lat, out, _ref(), sequence_name="1bad name")
    txt = out.read_text()
    assert "nan" not in txt.lower() and "inf" not in txt.lower()
    stmts = _split_statements(_strip_comments(txt))
    assert stmts[-1].strip().lower().startswith("use, sequence=")
    names = [s.split(":")[0].strip().lower() for s in stmts
             if ":" in s and not s.lower().startswith(("title", "beam", "use"))
             and "sequence" not in s.lower()]
    assert len(names) == len(set(names)), names
    for nm in names:
        assert nm[0].isalpha() and all(c.isalnum() or c in "_." for c in nm), nm
        assert nm not in ("drift", "quadrupole", "marker"), nm
    lat2, meta = parse_madx(str(out))
    assert not meta["warnings"]
    assert len(_physical(lat2)) == 5


def test_element_named_like_the_sequence_is_renamed(tmp_path):
    """MAD-X crashes (no error message) when an element and the sequence
    share a name — the file stem names the sequence, so a deck whose stem
    matches an element name must rename the element."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    lat.add(Drift("d", length=100.0))
    lat.add(Quadrupole("cell", length=100.0, gradient=1.0))
    out = tmp_path / "cell.madx"
    write_madx(lat, out, ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21))
    txt = out.read_text()
    assert "cell: SEQUENCE" in txt
    assert "cell_1: QUADRUPOLE" in txt and "\ncell: QUADRUPOLE" not in txt


def test_accelerating_line_warns_once_but_zero_gain_does_not(tmp_path):
    """MAD-X holds the reference energy: an accelerating deck gets one
    line-level note (dtl_section); a cavity at zero design gain does not."""
    lat = parse_tracewin("examples/dtl_section.dat")[0]
    warnings = write_madx(lat, tmp_path / "dtl.madx", _ref())
    notes = [w for w in warnings if "line accelerates" in w]
    assert len(notes) == 1 and "RFCAVITY" in notes[0] and "entrance rigidity" in notes[0]
    lat2 = Lattice()
    lat2.add(Drift(name="D", length=100.0))
    lat2.add(RFGap(name="G", voltage=1.0, phase=-90.0, frequency=352.21, ttf=1.0))
    assert write_madx(lat2, tmp_path / "zero.madx", _ref()) == []


# ---------------------------------------------------------------------------
# Adversarial-review round (2026-09-03): legality guards and bend folding
# ---------------------------------------------------------------------------
def _cpymad_loads(path) -> bool:
    cpymad = pytest.importorskip("cpymad.madx")
    m = cpymad.Madx(stdout=False)
    try:
        m.call(str(path))
        m.input("twiss, betx=1, bety=1;")
        return len(m.table.twiss.s) > 1
    finally:
        m.quit()


def test_reserved_command_and_class_names_are_renamed_and_load(tmp_path):
    """Element names that are MAD-X commands (fatal) or element classes
    (silently re-typed) get the ``_el`` suffix — and MAD-X loads the file."""
    lat = Lattice()
    for nm in ("start", "wire", "run", "proton", "srotation", "d"):
        lat.add(Drift(name=nm, length=50.0))
    lat.add(Quadrupole(name="exit", length=100.0, gradient=1.0))
    out = tmp_path / "res.madx"
    write_madx(lat, out, _ref())
    txt = out.read_text()
    for nm in ("start", "wire", "run", "proton", "srotation", "exit"):
        assert f"{nm}_el:" in txt and f"\n{nm}:" not in txt
    assert "\nd: DRIFT" in txt
    assert _cpymad_loads(out)
    back, _ = parse_madx(str(out))
    assert sum(isinstance(e, Quadrupole) for e in back.elements) == 1


def test_long_names_are_capped_and_load(tmp_path):
    lat = Lattice()
    lat.add(Drift(name="a" * 60, length=50.0))
    lat.add(Drift(name="a" * 60 + "b", length=50.0))     # same first 40 chars
    out = tmp_path / "long.madx"
    write_madx(lat, out, _ref())
    names = [ln.split(":")[0] for ln in out.read_text().splitlines()
             if ": DRIFT" in ln]
    assert names == ["a" * 40, "a" * 38 + "_1"]          # suffix inside the cap
    assert all(len(n) <= 40 for n in names)
    assert _cpymad_loads(out)


def test_zero_length_lattice_is_refused(tmp_path):
    lat = Lattice()
    lat.add(Marker(name="m"))
    with pytest.raises(ValueError, match="zero total length"):
        write_madx(lat, tmp_path / "z.madx", _ref())


def test_negative_drift_is_refused(tmp_path):
    lat = Lattice()
    lat.add(Drift(name="d1", length=100.0))
    lat.add(Drift(name="back", length=-30.0))
    lat.add(Quadrupole(name="q", length=50.0, gradient=1.0))
    with pytest.raises(ValueError, match="negative length"):
        write_madx(lat, tmp_path / "n.madx", _ref())


def test_title_with_comment_characters_survives_reimport(tmp_path):
    lat = Lattice()
    lat.add(Drift(name="d", length=100.0))
    out = tmp_path / "t.madx"
    write_madx(lat, out, _ref("H-", 3.0), title='cell "A"; see note! ok')
    txt = out.read_text()
    assert "!" not in txt.split("TITLE")[1].split("\n")[0]
    back, meta = parse_madx(str(out))
    assert meta["reference"].species.charge == -1          # BEAM line intact
    assert meta["reference"].w_kin == pytest.approx(3.0, rel=1e-9)


def test_collimators_carry_madx_aperture_model_and_rect_family(tmp_path):
    lat = Lattice()
    lat.add(Drift(name="d", length=100.0))
    lat.add(Aperture(name="r", dx=12.0, dy=8.0, aperture_type=0))
    lat.add(Aperture(name="c", dx=15.0, dy=15.0, aperture_type=1))
    lat.add(Aperture(name="f", dx=10.0, dy=9.0, aperture_type=Aperture.FINGER_H))
    out = tmp_path / "ap.madx"
    warnings = write_madx(lat, out, _ref())
    txt = out.read_text()
    assert "r: RCOLLIMATOR, xsize=0.012, ysize=0.008, apertype=rectangle, aperture={0.012, 0.008};" in txt
    assert "c: ECOLLIMATOR, xsize=0.015, ysize=0.015, apertype=circle, aperture={0.015};" in txt
    assert "f: RCOLLIMATOR, xsize=0.01, ysize=0.009" in txt
    assert any("'f'" in w and "rectangular cut" in w for w in warnings)
    assert _cpymad_loads(out)


def _bend_parts(angle_deg=8.0, rho=1500.0, beta=2.0, hv=0, **edge_kw):
    e1 = Edge(name="B_e1", pole_rotation=beta, rho=rho, hv=hv, **edge_kw)
    b = Dipole(name="B", angle=angle_deg, rho=rho, hv=hv)
    e2 = Edge(name="B_e2", pole_rotation=beta, rho=rho, hv=hv, **edge_kw)
    return e1, b, e2


def _line_matrix(lat, ref):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    return compute_transfer_matrix(lat, ref.copy())


def test_commands_between_edge_and_bend_do_not_break_folding(tmp_path):
    """Edge, SET card, Dipole, Edge: still one SBEND (the card becomes a
    comment in place); no edge is degraded to a marker."""
    from linac_gen.elements.lattice_commands import Freq
    e1, b, e2 = _bend_parts()
    lat = Lattice()
    lat.add(Drift(name="d", length=100.0))
    lat.add(e1)
    lat.add(Freq(name="F", frequency_mhz=352.21))
    lat.add(b)
    lat.add(e2)
    out = tmp_path / "fold.madx"
    warnings = write_madx(lat, out, _ref())
    txt = out.read_text()
    assert txt.count("SBEND") == 1 and "MARKER" not in txt
    assert "! HELIX: FREQ" in txt
    assert [w for w in warnings if "MARKER" in w] == []
    back, meta = parse_madx(str(out))
    np.testing.assert_allclose(_line_matrix(back, meta["reference"]),
                               _line_matrix(lat, _ref()), rtol=1e-12, atol=1e-13)


def test_split_bend_shares_one_edge_pair(tmp_path):
    """Edge, Dipole, Dipole, Edge (a magnet split in two BEND cards):
    two SBENDs, entrance edge on the first, exit edge on the last."""
    e1, b1, e2 = _bend_parts()
    b2 = Dipole(name="B2", angle=8.0, rho=1500.0)
    lat = Lattice()
    lat.add(Drift(name="d", length=100.0))
    for e in (e1, b1, b2, e2):
        lat.add(e)
    out = tmp_path / "split.madx"
    warnings = write_madx(lat, out, _ref())
    txt = out.read_text()
    assert txt.count("SBEND") == 2 and "MARKER" not in txt
    assert any("share one EDGE pair" in w for w in warnings)
    b_line = [ln for ln in txt.splitlines() if ": SBEND" in ln]
    assert "e1=" in b_line[0] and "e2=" not in b_line[0]
    assert "e2=" in b_line[1] and "e1=" not in b_line[1]
    back, meta = parse_madx(str(out))
    np.testing.assert_allclose(_line_matrix(back, meta["reference"]),
                               _line_matrix(lat, _ref()), rtol=1e-12, atol=1e-13)


def test_inconsistent_edges_warn_and_body_e1_is_combined(tmp_path):
    e1, b, e2 = _bend_parts()
    e1.rho = 1000.0                          # edge rho != body rho
    b.e1 = 1.0                               # body carries its own e1 too
    lat = Lattice()
    lat.add(Drift(name="d", length=100.0))
    for e in (e1, b, e2):
        lat.add(e)
    out = tmp_path / "inc.madx"
    warnings = write_madx(lat, out, _ref())
    assert any("re-expressed through the body's curvature" in w for w in warnings)
    assert any("both the body's e1 and an EDGE" in w for w in warnings)
    back, meta = parse_madx(str(out))
    # thin-edge term is exact under both adjustments (no gap here)
    np.testing.assert_allclose(_line_matrix(back, meta["reference"])[:4, :4],
                               _line_matrix(lat, _ref())[:4, :4], rtol=1e-12, atol=1e-13)


def test_negative_angle_edges_export_with_madx_sign_convention(tmp_path):
    """A TraceWin-style negative bend (rho>0) with +beta edges (horizontally
    DEfocusing whichever way it bends): MAD-X needs e = -beta.  Pinned on
    a VERTICAL bend (the horizontal negative-angle body is a known HELIX
    defect, see test_madx_conventions)."""
    e1, b, e2 = _bend_parts(angle_deg=-8.0, beta=2.0, hv=1)
    lat = Lattice()
    lat.add(Drift(name="d", length=100.0))
    for e in (e1, b, e2):
        lat.add(e)
    out = tmp_path / "neg.madx"
    write_madx(lat, out, _ref())
    line = [ln for ln in out.read_text().splitlines() if ": SBEND" in ln][0]
    assert "e1=-0.0349" in line and "e2=-0.0349" in line and "tilt=pi/2" in line
    back, meta = parse_madx(str(out))
    edges = [e for e in back.elements if isinstance(e, Edge)]
    assert all(e.pole_rotation == pytest.approx(2.0, rel=1e-12) for e in edges)
    np.testing.assert_allclose(_line_matrix(back, meta["reference"]),
                               _line_matrix(lat, _ref()), rtol=1e-12, atol=1e-13)
    cpymad = pytest.importorskip("cpymad.madx")
    from linac_gen.tracking.longitudinal_coords import matrix_to_madx
    m = cpymad.Madx(stdout=False)
    m.call(str(out))
    m.input("twiss, betx=1, bety=1, rmatrix;")
    t = m.table.twiss
    R = np.array([[t[f"re{i}{j}"][-1] for j in range(1, 7)] for i in range(1, 7)])
    m.quit()
    ref = _ref()
    np.testing.assert_allclose(matrix_to_madx(_line_matrix(lat, ref), ref)[:4, :4],
                               R[:4, :4], rtol=1e-9, atol=1e-12)


def test_species_that_dies_in_the_deck_rf_still_exports(tmp_path):
    """A proton DTL exported with an H⁻ reference: the cavities decelerate
    H⁻ below zero energy.  The static conversions (k1/ks via the entrance
    rigidity, geometry) are still written, with one warning naming the
    element where the reference was lost — no bare math-domain error
    (regression: found while packaging the exporter, 2026-09-03)."""
    lat = parse_tracewin("examples/dtl_section.dat")[0]
    out = tmp_path / "dtl_hminus.madx"
    warnings = write_madx(lat, out, _ref("H-", 3.0))
    assert out.exists()
    lost = [w for w in warnings if "reference particle lost" in w]
    assert len(lost) == 1 and "decelerates" in lost[0]
    # the gain accumulated up to the loss point is still reported (negative)
    acc = [w for w in warnings if "line accelerates" in w]
    assert len(acc) == 1 and "ΔW=-" in acc[0]
    back, meta = parse_madx(str(out))
    assert meta["reference"].species.charge == -1
    assert sum(isinstance(e, RFGap) for e in back.elements) == \
        sum(isinstance(e, RFGap) for e in lat.elements)
    # linearize=True on the same deck degrades to MARKER + DRIFT, no crash
    warnings2 = write_madx(lat, tmp_path / "dtl_lin.madx", _ref("H-", 3.0),
                           linearize=True)
    assert any("reference particle lost" in w for w in warnings2)


def test_error_study_definitions_are_reported_not_dropped_silently(tmp_path):
    """ERROR_* cards attach an error study to the lattice; like
    write_tracewin, the MAD-X writer does not serialise them but says so
    (warning + a comment in the sequence)."""
    deck = tmp_path / "err.dat"
    # TraceWin's card is stateful: N counts the quads that FOLLOW it
    # (form taken from examples/pip2_misalignment_study).
    deck.write_text("DRIFT 100 30 0\n"
                    "ERROR_QUAD_NCPL_STAT 1 2 0.1 0.1 0 0 1.0 0.5 0 0 0 0\n"
                    "QUAD 100 5 30\nDRIFT 100 30 0\nEND\n")
    lat = parse_tracewin(str(deck))[0]
    assert getattr(lat, "errors", None), "the deck's ERROR_QUAD card should attach an error study"
    out = tmp_path / "err.madx"
    warnings = write_madx(lat, out, _ref())
    assert any("ERROR_*" in w and "EALIGN" in w for w in warnings)
    n = len(lat.errors)                       # one ErrorDef per non-zero slot
    assert f"! HELIX: {n} ERROR_* element + 0 beam error definitions" in out.read_text()
