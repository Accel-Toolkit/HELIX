"""MAD-X importer conventions that the (planned) exporter must mirror exactly.

Every test here pins a *physics* relation, not a parsing detail:

* KICKER/HKICKER/VKICKER kicks are normalised to the reference charge in
  MAD-X, so the imported Steerer must reproduce ``hkick``/``vkick`` as the
  deflection angle for BOTH a proton and an H⁻ beam.
* QUADRUPOLE ``tilt`` → ``skew_angle`` (degrees).
* SBEND ``k1`` → combined-function field index ``n = -k1·ρ²`` (from
  HELIX ``k_x² = (1-n)/ρ²``, ``k_y² = n/ρ²`` vs MAD-X ``k_x² = h² + k1``,
  ``k_y² = -k1``), pinned against MAD-X's own sector map through cpymad.
* ``tilt = π/2`` bends are vertical (``hv = 1``); ``hgap``/``fint`` map onto
  the Edge fringe model (``gap = 2·hgap``, ``k1 = fint``).
* RCOLLIMATOR / ECOLLIMATOR → Aperture (rectangular / circular).
* ``refer = entry | centre | exit`` place elements identically.
* RFCAVITY ``lag`` follows the MAD-X definition — reference energy gain
  ``q·V·sin(2π·lag)`` — so the HELIX synchronous phase is ``360·lag − 90°``
  (HELIX gain is ``q·V·T·cos φs``).  Confirmed against MAD-X ``TRACK``.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from linac_gen.elements.aperture import Aperture
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.edge import Edge
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.steerer import Steerer
from linac_gen.io.madx_parser import parse_madx


def _write(tmp_path, text: str):
    p = tmp_path / "deck.madx"
    p.write_text(text)
    return str(p)


def _only(lat, cls):
    found = [e for e in lat.elements if isinstance(e, cls)]
    assert len(found) == 1, f"expected exactly one {cls.__name__}, got {found}"
    return found[0]


# ---------------------------------------------------------------------------
# KICKER family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("particle", ["proton", "h-"])
def test_kicker_reproduces_madx_kick_angle_for_both_charge_signs(tmp_path, particle):
    """MAD-X hkick/vkick are deflection angles for the reference charge.

    HELIX stores ∫B·dl and applies ``Δx' = sign(q)·by_l/Bρ`` — so the
    importer must fold sign(q) in, or an H⁻ deck would kick the wrong way.
    """
    deck = f"""
    beam, particle={particle}, energy=0.938272+0.0021;
    k1: kicker, hkick=1.5e-3, vkick=-2.0e-3;
    seq: sequence, refer=entry, l=0.5;
    k1, at=0.25;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    st = _only(lat, Steerer)
    dxp, dyp = st._kick_mrad(meta["reference"])
    assert dxp == pytest.approx(1.5, rel=1e-12)       # mrad == 1e3 · hkick
    assert dyp == pytest.approx(-2.0, rel=1e-12)
    assert not st.elec


def test_hkicker_vkicker_single_plane(tmp_path):
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    h1: hkicker, kick=1e-3, l=0.1;
    v1: vkicker, kick=-3e-3;
    seq: sequence, refer=entry, l=1.0;
    h1, at=0.0; v1, at=0.5;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    steerers = [e for e in lat.elements if isinstance(e, Steerer)]
    assert [s.name for s in steerers] == ["h1", "v1"]
    ref = meta["reference"]
    assert steerers[0]._kick_mrad(ref) == pytest.approx((1.0, 0.0), rel=1e-12)
    assert steerers[1]._kick_mrad(ref) == pytest.approx((0.0, -3.0), rel=1e-12)
    # l=0.1 m hkicker is drift-padded; total length is still 1.0 m.
    assert sum(e.length for e in lat.elements) == pytest.approx(1000.0)
    assert not meta["warnings"]


# ---------------------------------------------------------------------------
# QUADRUPOLE tilt, SBEND k1 / tilt / hgap / fint
# ---------------------------------------------------------------------------
def test_quadrupole_tilt_becomes_skew_angle_degrees(tmp_path):
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    qs: quadrupole, l=0.2, k1=3.0, tilt=pi/8;
    seq: sequence, refer=entry, l=0.2;
    qs, at=0;
    endsequence;
    use, sequence=seq;
    """
    lat, _ = parse_madx(_write(tmp_path, deck))
    q = _only(lat, Quadrupole)
    assert q.skew_angle == pytest.approx(22.5, rel=1e-12)


def test_sbend_k1_becomes_field_index(tmp_path):
    """n = -k1·ρ² with ρ in metres (angle=0.1 rad over l=1 m ⇒ ρ = 10 m)."""
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    b: sbend, l=1.0, angle=0.1, k1=0.5;
    seq: sequence, refer=entry, l=1.0;
    b, at=0;
    endsequence;
    use, sequence=seq;
    """
    lat, _ = parse_madx(_write(tmp_path, deck))
    d = _only(lat, Dipole)
    assert d.rho == pytest.approx(10_000.0)                    # mm
    assert d.field_index == pytest.approx(-0.5 * 10.0 ** 2, rel=1e-12)
    assert d.hv == 0


def test_sbend_k1_matches_madx_sector_map():
    """The field-index mapping is only right if HELIX's combined-function
    bend reproduces MAD-X's own sector map: compare the 4x4 transverse block
    of a single SBEND(k1≠0) from cpymad with HELIX's element matrix."""
    cpymad = pytest.importorskip("cpymad.madx")
    from linac_gen.tracking.matrix_tracking import get_element_matrix

    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    b: sbend, l=1.0, angle=0.1, k1=0.5;
    seq: sequence, refer=entry, l=1.0;
    b, at=0;
    endsequence;
    use, sequence=seq;
    """
    import tempfile, os
    td = tempfile.mkdtemp()
    path = os.path.join(td, "b.madx")
    with open(path, "w") as fh:
        fh.write(deck)
    lat, meta = parse_madx(path)
    ref = meta["reference"]
    # HELIX: Edge(0) + Dipole + Edge(0) — zero pole rotation, zero gap ⇒
    # the edges are identity; the body carries everything.
    M = np.eye(6)
    for e in lat.elements:
        M = get_element_matrix(e, copy.deepcopy(ref)) @ M
    # HELIX basis is (mm, mrad); MAD-X is (m, rad) — the 4x4 transverse
    # block is invariant under the common rescale except off-diagonal
    # mixing of length and angle: R12 [mm/mrad] == R12 [m/rad], so the
    # transverse block compares directly.
    m = cpymad.Madx(stdout=False)
    m.input(deck)
    # sectorfile= keeps MAD-X from dropping a "sectormap" file in the cwd
    # (the repo root under pytest).
    sector_file = os.path.join(td, "sectormap.tfs")
    m.input(f'twiss, betx=1, bety=1, sectormap, sectorfile="{sector_file}";')
    st = m.table.sectortable
    # sectortable rows: one per element boundary; take the row for 'b'.
    idx = [i for i, n in enumerate(st.name) if n.lower().startswith("b")][0]
    R = np.array([[st[f"r{i}{j}"][idx] for j in range(1, 7)] for i in range(1, 7)])
    m.quit()
    np.testing.assert_allclose(M[:4, :4], R[:4, :4], rtol=1e-8, atol=1e-10)


def test_vertical_bend_via_tilt_and_fringe_via_hgap_fint(tmp_path):
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    b: sbend, l=1.0, angle=0.1, tilt=pi/2, hgap=0.02, fint=0.5;
    seq: sequence, refer=entry, l=1.0;
    b, at=0;
    endsequence;
    use, sequence=seq;
    """
    lat, _ = parse_madx(_write(tmp_path, deck))
    d = _only(lat, Dipole)
    assert d.hv == 1
    edges = [e for e in lat.elements if isinstance(e, Edge)]
    assert len(edges) == 2
    for e in edges:
        assert e.gap == pytest.approx(40.0)     # 2·hgap in mm
        assert e.k1 == pytest.approx(0.5)
        assert e.hv == 1


# ---------------------------------------------------------------------------
# Collimators → Aperture
# ---------------------------------------------------------------------------
def test_collimators_become_apertures(tmp_path):
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    rc: rcollimator, xsize=0.01, ysize=0.005;
    ec: ecollimator, xsize=0.02, ysize=0.02, l=0.1;
    seq: sequence, refer=entry, l=1.0;
    rc, at=0.2; ec, at=0.5;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    aps = [e for e in lat.elements if isinstance(e, Aperture)]
    assert [a.name for a in aps] == ["rc", "ec"]
    assert aps[0].aperture_type == 0 and aps[0].dx == pytest.approx(10.0) \
        and aps[0].dy == pytest.approx(5.0)
    assert aps[1].aperture_type == 1 and aps[1].dx == pytest.approx(20.0)
    assert sum(e.length for e in lat.elements) == pytest.approx(1000.0)
    assert not meta["warnings"]


# ---------------------------------------------------------------------------
# refer = entry | centre | exit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("refer, at_q", [("entry", 0.3), ("centre", 0.4), ("exit", 0.5)])
def test_refer_modes_place_identically(tmp_path, refer, at_q):
    deck = f"""
    beam, particle=proton, energy=0.938272+0.0021;
    q: quadrupole, l=0.2, k1=1.0;
    seq: sequence, refer={refer}, l=1.0;
    q, at={at_q};
    endsequence;
    use, sequence=seq;
    """
    lat, _ = parse_madx(_write(tmp_path, deck))
    s = 0.0
    entry = None
    for e in lat.elements:
        if isinstance(e, Quadrupole):
            entry = s
        s += e.length
    assert entry == pytest.approx(300.0)     # quad entry always at 0.3 m
    assert s == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# RFCAVITY lag convention
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lag", [0.0, 0.125, 0.25, 0.5])
def test_rfcavity_lag_gives_madx_reference_energy_gain(tmp_path, lag):
    """HELIX gain q·V·T·cos(φs) must equal MAD-X's q·V·sin(2π·lag)."""
    deck = f"""
    beam, particle=proton, energy=0.938272+0.0021;
    c: rfcavity, volt=1.0, lag={lag}, freq=352.21;
    seq: sequence, refer=entry, l=0.1;
    c, at=0.05;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    gap = _only(lat, RFGap)
    assert gap.ttf == pytest.approx(1.0)
    ref = copy.deepcopy(meta["reference"])
    w0 = ref.w_kin
    gap.advance_ref(ref)
    gain_MeV = ref.w_kin - w0
    assert gain_MeV == pytest.approx(1.0 * math.sin(2 * math.pi * lag), abs=1e-12)


@pytest.mark.parametrize("lag", [0.0, 0.125, 0.25, 0.5])
def test_rfcavity_lag_matches_madx_track(lag):
    """The MAD-X side of the convention, measured rather than assumed:
    TRACK one reference particle (pt=0) through the cavity in MAD-X and
    read the energy gain from pt (pt = ΔE / (p0·c))."""
    cpymad = pytest.importorskip("cpymad.madx")
    m = cpymad.Madx(stdout=False)
    m.input(f"""
    beam, particle=proton, energy=0.938272+0.0021;
    c: rfcavity, volt=1.0, lag={lag}, freq=352.21;
    seq: sequence, refer=entry, l=0.1;
    c, at=0.05;
    endsequence;
    use, sequence=seq;
    track, onepass, onetable;
    start, x=0, px=0, y=0, py=0, t=0, pt=0;
    run, turns=1;
    endtrack;
    """)
    tab = m.table.trackone
    pt_end = float(tab.pt[-1])
    pc0_GeV = float(m.sequence.seq.beam.pc)
    m.quit()
    gain_MeV = pt_end * pc0_GeV * 1000.0
    assert gain_MeV == pytest.approx(1.0 * math.sin(2 * math.pi * lag), abs=1e-6)


def test_bare_tilt_flag_means_natural_skew(tmp_path):
    """MAD-X ``tilt`` with no value = the natural skew: π/4 for a
    quadrupole (45°), π/2 for a bend (vertical)."""
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    qs: quadrupole, l=0.2, k1=3.0, tilt;
    b: sbend, l=1.0, angle=0.1, tilt;
    seq: sequence, refer=entry, l=2.0;
    qs, at=0; b, at=0.5;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _only(lat, Quadrupole).skew_angle == pytest.approx(45.0)
    assert _only(lat, Dipole).hv == 1
    assert not [w for w in meta["warnings"] if "tilt" in w]


def test_bend_arbitrary_tilt_warns_and_stays_horizontal(tmp_path):
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    b: sbend, l=1.0, angle=0.1, tilt=0.3;
    seq: sequence, refer=entry, l=1.0;
    b, at=0;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert _only(lat, Dipole).hv == 0
    assert any("tilt=0.3" in w and "horizontal" in w for w in meta["warnings"])


def test_zero_angle_bend_with_k1_warns(tmp_path):
    deck = """
    beam, particle=proton, energy=0.938272+0.0021;
    b: sbend, l=1.0, angle=0.0, k1=0.5;
    seq: sequence, refer=entry, l=1.0;
    b, at=0;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert not [e for e in lat.elements if isinstance(e, Dipole)]
    assert any("k1 focusing is lost" in w for w in meta["warnings"])


def test_rbend_length_is_the_chord_arc_matches_madx_s():
    """MAD-X (RBARC=TRUE default): an RBEND's ``l`` is the straight chord;
    the machine s-extent is the arc l·(θ/2)/sin(θ/2).  Pin it against
    MAD-X's own twiss table so HELIX's imported RBEND has the same length."""
    cpymad = pytest.importorskip("cpymad.madx")
    deck = """
    beam, particle=proton, energy=0.938272+0.003;
    b: rbend, l=0.8, angle=0.4;
    seq: sequence, refer=entry, l=1.0; b, at=0.1; endsequence;
    use, sequence=seq;
    """
    m = cpymad.Madx(stdout=False)
    m.input(deck + "twiss, betx=1, bety=1;")
    t = m.table.twiss
    i = [k for k, n in enumerate(t.name) if n.startswith("b")][0]
    s_exit_madx = float(t.s[i])
    m.quit()
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rb.madx")
        with open(path, "w") as fh:
            fh.write(deck)
        lat, _ = parse_madx(path)
    d = _only(lat, Dipole)
    arc_expected = 0.8 * 0.2 / math.sin(0.2)
    assert d.length == pytest.approx(arc_expected * 1000.0, rel=1e-12)
    assert (0.1 + d.length / 1000.0) == pytest.approx(s_exit_madx, rel=1e-12)


@pytest.mark.parametrize("stmt, chord", [
    ("option, rbarc=false;", False),
    ("option, -rbarc;", False),
    ("option, rbarc;", True),
    ("option, rbarc=true;", True),
    ("", True),
])
def test_option_rbarc_controls_rbend_length_interpretation(tmp_path, stmt, chord):
    """``OPTION, RBARC=false`` (LHC-style) makes ``l`` the arc length;
    the default and the explicit true form make it the chord."""
    path = tmp_path / "rb.madx"
    path.write_text(
        "beam, particle=proton, energy=0.938272+0.003;\n" + stmt + "\n"
        "b: rbend, l=0.8, angle=0.4;\n"
        "seq: sequence, refer=entry, l=1.0; b, at=0.1; endsequence;\n"
        "use, sequence=seq;\n")
    lat, _ = parse_madx(str(path))
    d = _only(lat, Dipole)
    expected_m = 0.8 * 0.2 / math.sin(0.2) if chord else 0.8
    assert d.length == pytest.approx(expected_m * 1000.0, rel=1e-12)
    assert d.rho == pytest.approx(expected_m / 0.4 * 1000.0, rel=1e-12)


def _madx_R(deck: str):
    cpymad = pytest.importorskip("cpymad.madx")
    m = cpymad.Madx(stdout=False)
    m.input(deck + "twiss, betx=1, bety=1, rmatrix;")
    t = m.table.twiss
    R = np.array([[t[f"re{i}{j}"][-1] for j in range(1, 7)] for i in range(1, 7)])
    m.quit()
    return R


def _helix_R_from_deck(deck: str, tmp_path):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    from linac_gen.tracking.longitudinal_coords import matrix_to_madx
    path = tmp_path / "b.madx"
    path.write_text(deck)
    lat, meta = parse_madx(str(path))
    ref = meta["reference"]
    return matrix_to_madx(compute_transfer_matrix(lat, ref.copy()), ref), lat


_BEND_DECK = """beam, particle=proton, energy=0.94327208816;
b: sbend, l=1.0, angle={angle}, e1=0.05, e2=0.03, tilt={tilt};
s: sequence, refer=entry, l=1.0; b, at=0; endsequence;
use, sequence=s;
"""


@pytest.mark.parametrize("tilt, angle", [("pi/2", 0.2), ("pi/2", -0.2),
                                         ("-pi/2", 0.2), ("-pi/2", -0.2),
                                         ("0", 0.2)])
def test_bend_direction_and_pole_faces_match_madx(tmp_path, tilt, angle):
    """Vertical bends of either tilt sign and either angle sign, and the
    horizontal positive bend, with pole-face angles: transverse 4x6 block
    (incl. dispersion) equals MAD-X's.  ``tilt=-pi/2`` bends towards -y
    (negative HELIX angle); e1/e2 map to β = sign(angle)·e."""
    deck = _BEND_DECK.format(angle=angle, tilt=tilt)
    Rx = _madx_R(deck)
    Rh, lat = _helix_R_from_deck(deck, tmp_path)
    d = _only(lat, Dipole)
    assert d.rho > 0
    assert (d.angle < 0) == ((angle < 0) != (tilt == "-pi/2"))
    np.testing.assert_allclose(Rh[:4, :6], Rx[:4, :6], rtol=1e-10, atol=1e-12)


@pytest.mark.xfail(strict=True, reason=(
    "HELIX Dipole defect (found 2026-09-03 via the MAD-X oracle): the "
    "horizontal branch uses the SIGNED angle in its focusing trig, so a "
    "negative-angle BEND with rho>0 (TraceWin convention; fnalscl.dat:506/"
    "561/567) gets the inverse sector map and an unflipped dispersion sign. "
    "The vertical branch is correct.  Flip this to a plain test when "
    "dipole.py is fixed."))
def test_horizontal_negative_bend_matches_madx(tmp_path):
    deck = _BEND_DECK.format(angle=-0.2, tilt="0")
    Rx = _madx_R(deck)
    Rh, _ = _helix_R_from_deck(deck, tmp_path)
    np.testing.assert_allclose(Rh[:4, :6], Rx[:4, :6], rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Adversarial-review round (2026-09-03): items that round-trip symmetrically
# but were wrong against real MAD-X
# ---------------------------------------------------------------------------
def _track_gain_and_slope(deck: str, pc0_GeV: float):
    """MAD-X TRACK: reference gain [MeV] and dW/dt slope sign through one cavity."""
    cpymad = pytest.importorskip("cpymad.madx")
    m = cpymad.Madx(stdout=False)
    m.input(deck + "track, onepass, onetable; start, t=0; start, t=0.01; "
                   "run, turns=1; endtrack;")
    t = m.table.trackone
    pts = [float(p) for p, s in zip(t.pt, t.s) if s > 0]
    m.quit()
    gain = pts[0] * pc0_GeV * 1000.0
    late_minus_early = (pts[1] - pts[0]) * pc0_GeV * 1000.0   # t>0 is EARLY in MAD-X
    return gain, late_minus_early


@pytest.mark.parametrize("species", ["proton", "H-"])
def test_rfcavity_gain_and_slope_match_madx_for_both_charges(tmp_path, species):
    """MAD-X does not multiply the RF kick by the charge; HELIX gains
    q·V·T·cos φs.  The exporter writes volt = sign(q)·V so BOTH the
    reference gain and the phase slope agree with MAD-X TRACK for H⁻."""
    from linac_gen.cli.common import build_ref
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.io.madx_writer import write_madx
    ref = build_ref(BeamConfig(species=species, energy=3.0, frequency=352.21))
    gap = RFGap("g", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
    lat = Lattice()
    lat.add(Drift("d", length=10.0))
    lat.add(gap)
    out = tmp_path / "cav.madx"
    write_madx(lat, out, ref)
    r = ref.copy()
    r.s += 10.0
    r.phi_s += 360.0 * 10.0 / (r.beta * r.wavelength)
    gap.advance_ref(r)
    dW_helix = r.w_kin - ref.w_kin
    assert dW_helix == pytest.approx(ref.species.charge * math.cos(math.radians(-30.0)), rel=1e-12)
    pc0 = ref.bg * ref.species.mass / 1000.0
    gain, slope = _track_gain_and_slope(out.read_text(encoding="utf-8"), pc0)
    assert gain == pytest.approx(dW_helix, rel=1e-9)
    # φs = −30° (before crest): a LATE particle (t<0 in MAD-X) gains more
    # for a proton; the whole curve flips sign with the charge.
    assert math.copysign(1.0, slope) == -math.copysign(1.0, ref.species.charge)
    # exact inverse
    back, meta = parse_madx(str(out))
    g2 = _only(back, RFGap)
    assert g2.voltage == pytest.approx(1.0, rel=1e-12)
    assert g2.phase == pytest.approx(-30.0, rel=1e-12)
    assert meta["reference"].species.charge == ref.species.charge


def test_negative_volt_imports_as_180_degree_shift(tmp_path):
    path = tmp_path / "neg.madx"
    path.write_text("beam, particle=proton, energy=0.94327208816;\n"
                    "c: rfcavity, l=0, volt=-1.5, lag=0.1, freq=352.21;\n"
                    "s: sequence, refer=entry, l=0.1; c, at=0; endsequence;\n"
                    "use, sequence=s;\n")
    lat, meta = parse_madx(str(path))
    g = _only(lat, RFGap)
    assert g.voltage == pytest.approx(1.5)
    assert g.phase == pytest.approx(0.1 * 360.0 - 90.0 + 180.0)
    r = meta["reference"].copy()
    g.advance_ref(r)
    assert r.w_kin - meta["reference"].w_kin == pytest.approx(-1.5 * math.sin(2 * math.pi * 0.1), rel=1e-12)


def test_hgap_without_fint_uses_madx_default_zero_and_fintx(tmp_path):
    """MAD-X: fint defaults to 0 (no fringe effect); fintx overrides the
    exit face.  Pinned against MAD-X's R43."""
    for extra, k1_in, k1_out in (("", 0.0, 0.0), (", fint=0.5", 0.5, 0.5),
                                 (", fint=0.5, fintx=0.2", 0.5, 0.2)):
        deck = ("beam, particle=proton, energy=0.94327208816;\n"
                f"b: sbend, l=1.2, angle=0.1726, e1=0.05, e2=0.05, hgap=0.03{extra};\n"
                "s: sequence, refer=entry, l=1.2; b, at=0; endsequence;\n"
                "use, sequence=s;\n")
        Rx = _madx_R(deck)
        Rh, lat = _helix_R_from_deck(deck, tmp_path)
        edges = [e for e in lat.elements if isinstance(e, Edge)]
        assert edges[0].k1 == k1_in and edges[1].k1 == k1_out
        assert edges[0].gap == pytest.approx(60.0)
        np.testing.assert_allclose(Rh[:4, :4], Rx[:4, :4], rtol=1e-10, atol=1e-12)


def test_multipole_tilt_round_trips_and_matches_madx(tmp_path):
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.multipole import Multipole
    from linac_gen.io.madx_writer import write_madx
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    from linac_gen.tracking.longitudinal_coords import matrix_to_madx
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    lat.add(Multipole("m", knl=[0.0, 0.5], ksl=[0.0], tilt_deg=45.0))
    lat.add(Drift("d2", length=100.0))
    out = tmp_path / "tilt.madx"
    assert write_madx(lat, out, ref) == []
    assert "tilt=" in out.read_text(encoding="utf-8")
    back, _ = parse_madx(str(out))
    assert _only(back, Multipole).tilt_deg == pytest.approx(45.0, rel=1e-12)
    # The physics is what is pinned (not the sign of the tilt attribute):
    # Multipole.tilt_deg rotates opposite to MAD-X's tilt and to
    # Quadrupole.skew_angle — the converter compensates.  If the core
    # sense is ever unified, this assertion flags the converter to follow.
    Rx = _madx_R(out.read_text(encoding="utf-8"))
    Rh = matrix_to_madx(compute_transfer_matrix(lat, ref.copy()), ref)
    np.testing.assert_allclose(Rh[:4, :4], Rx[:4, :4], rtol=1e-10, atol=1e-12)
    assert abs(Rx[1, 2]) > 0.1          # skew: x' couples to y
    q = Quadrupole("q", length=10.0, gradient=1.0, skew_angle=45.0)
    m_equiv = Multipole("m", knl=[0.0, 0.01 / ref.brho], ksl=[0.0], tilt_deg=45.0)
    assert np.sign(q.transfer_matrix(ref.copy())[1, 2]) == -np.sign(m_equiv.kick_matrix(ref.copy())[1, 2])


def test_ion_without_charge_defaults_to_plus_one_like_madx(tmp_path):
    cpymad = pytest.importorskip("cpymad.madx")
    m = cpymad.Madx(stdout=False)
    m.input("beam, particle=ion, mass=0.93827208816, energy=1.5;")
    assert m.beam.charge == 1.0
    m.quit()
    path = tmp_path / "ion.madx"
    path.write_text("beam, particle=ion, mass=0.93827208816, energy=1.5;\n"
                    "q: quadrupole, l=0.2, k1=2.0;\n"
                    "s: sequence, refer=entry, l=0.2; q, at=0; endsequence;\n"
                    "use, sequence=s;\n")
    lat, meta = parse_madx(str(path))
    assert meta["reference"].species.charge == 1
    assert meta["warnings"] == []
    assert _only(lat, Quadrupole).gradient > 0
