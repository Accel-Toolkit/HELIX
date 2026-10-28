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


#: MAD-X 5.09 (cpymad 1.19): the kick its TWISS gives [mrad] (px, py at the
#: end of a line with the kicker alone), to all its digits — an HKICKER
#: (VKICKER) kicks by KICK, or by HKICK (VKICK) when KICK is zero; KICKER and
#: TKICKER by HKICK and VKICK; CHKICK (CVKICK) adds to the horizontal
#: (vertical) kick; TILT rolls the kick about s; a bare flag is 0.
_MADX_KICKS = [
    ("hkicker, kick=1e-3", (1.0, 0.0)),
    ("hkicker, hkick=1e-3", (1.0, 0.0)),
    ("hkicker, kick=1e-3, hkick=2e-3", (1.0, 0.0)),
    ("hkicker, kick=0, hkick=2e-3", (2.0, 0.0)),
    ("vkicker, vkick=1e-3", (0.0, 1.0)),
    ("vkicker, kick=1e-3, vkick=2e-3", (0.0, 1.0)),
    ("vkicker, kick=0, vkick=2e-3", (0.0, 2.0)),
    ("hkicker, kick=1e-3, tilt=0.5", (0.8775825618903728, 0.479425538604203)),
    ("hkicker, hkick=1e-3, tilt=-0.5", (0.8775825618903728, -0.479425538604203)),
    ("vkicker, vkick=1e-3, tilt=0.5", (-0.479425538604203, 0.8775825618903728)),
    ("kicker, hkick=1e-3, vkick=2e-3, tilt=0.5", (-0.08126851531803328, 2.2345906623849485)),
    ("tkicker, hkick=1e-3, vkick=2e-3, tilt=0.5", (-0.08126851531803328, 2.2345906623849485)),
    ("hkicker, kick=1e-3, tilt", (1.0, 0.0)),
    ("kicker, hkick=1e-3, vkick=2e-3, tilt", (1.0, 2.0)),
    ("hkicker, kick", (0.0, 0.0)),
    ("hkicker, hkick", (0.0, 0.0)),
    ("vkicker, vkick", (0.0, 0.0)),
    ("kicker, hkick", (0.0, 0.0)),
    ("hkicker, kick=1e-3, chkick=5e-4", (1.5, 0.0)),
    ("hkicker, kick=0, hkick=1e-3, chkick=5e-4", (1.5, 0.0)),
    ("hkicker, chkick=5e-4", (0.5, 0.0)),
    ("vkicker, kick=1e-3, cvkick=5e-4", (0.0, 1.5)),
    ("kicker, hkick=1e-3, vkick=2e-3, chkick=5e-4, cvkick=3e-4", (1.5, 2.3)),
    ("kicker, hkick=1e-3, vkick=2e-3, chkick=5e-4, cvkick=3e-4, tilt=0.5",
     (0.21369510404589218, 2.7375782002541618)),
    ("tkicker, hkick=1e-3, chkick=5e-4, cvkick=3e-4", (1.5, 0.3)),
    ("hkicker, kick=1e-3, chkick=5e-4, tilt=0.5", (1.3163738428355591, 0.7191383079063045)),
    ("hkicker, kick=2e-4, tilt=pi/2", (1.2246467991473533e-17, 0.2)),
    ("hkicker, kick=1e-3, chkick", (1.0, 0.0)),
]


def _kicker(tmp_path, particle, definition, before=""):
    deck = f"""
    beam, particle={particle}, energy=0.938272+0.0021;
    {before}
    k1: {definition}, l=0.5;
    seq: sequence, refer=centre, l=3.0;
    k1, at=1.0;
    endsequence;
    use, sequence=seq;
    """
    lat, meta = parse_madx(_write(tmp_path, deck))
    return _only(lat, Steerer), meta


@pytest.mark.parametrize("particle", ["proton", "h-"])
@pytest.mark.parametrize("definition, madx_mrad", _MADX_KICKS)
def test_kicker_attributes_and_tilt_as_madx(tmp_path, particle, definition, madx_mrad):
    st, meta = _kicker(tmp_path, particle, definition)
    assert st._kick_mrad(meta["reference"]) == pytest.approx(madx_mrad, rel=1e-12, abs=1e-15)
    bare = any(kv.strip() in ("kick", "hkick", "vkick", "chkick", "cvkick", "tilt")
               for kv in definition.split(",")[1:])
    assert bool(meta["warnings"]) == bare, meta["warnings"]
    if bare:
        assert "without a value is 0" in meta["warnings"][0]


def test_a_quarter_turn_tilt_leaves_no_kick_in_the_other_plane(tmp_path):
    # (MAD-X keeps cos(pi/2) x kick = 1e-20 rad; HELIX none: a bump magnet in
    # ring mode must kick in one plane)
    st, _ = _kicker(tmp_path, "proton", "hkicker, kick=2e-4, tilt=pi/2")
    assert st.by_l == 0.0 and st.bx_l != 0.0


def test_a_kicker_class_and_attributes_madx_refuses(tmp_path):
    # an element of a class derived from HKICKER kicks by its HKICK (MAD-X: 1 mrad)
    st, meta = _kicker(tmp_path, "proton", "hk1, hkick=1e-3", before="hk1: hkicker;")
    assert st._kick_mrad(meta["reference"]) == pytest.approx((1.0, 0.0), rel=1e-12)
    # an attribute MAD-X refuses on the type (its run stops) is not read, with a warning
    st, meta = _kicker(tmp_path, "proton", "hkicker, kick=1e-3, vkick=2e-3")
    assert st._kick_mrad(meta["reference"]) == pytest.approx((1.0, 0.0), rel=1e-12)
    assert any("MAD-X refuses VKICK on HKICKER" in w for w in meta["warnings"])


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


#: MAD-X 5.09 (cpymad 1.19) TWISS RE matrices of a sector bend (e1 = e2 = 0)
#: with a fringe gap: the pole faces at zero angle still focus through the
#: fringe field, R43 = h tan(psi) at each face (horizontal bend; with
#: tilt = pi/2 the same in R21)
_SECTOR_FRINGE = {
    "horizontal": ("beam, particle=proton, energy=0.8+pmass;\n"
                   "b: sbend, l=1.0, angle=0.1, e1=0, e2=0, hgap=0.03, fint=0.5;\n",
                   {(0, 0): 0.9950041652780257, (0, 1): 0.9983341664682817,
                    (1, 0): -0.009983341664682817, (1, 1): 0.9950041652780258,
                    (2, 2): 1.0003000009000032, (2, 3): 1.0000000000000002,
                    (3, 2): 0.0006000918005464829, (3, 3): 1.0003000009000034}),
    "vertical": ("beam, particle=proton, energy=0.938272+0.0021;\n"
                 "b: sbend, l=1.0, angle=0.1, tilt=pi/2, hgap=0.02, fint=0.5;\n",
                 {(0, 0): 1.000200000266667, (0, 1): 1.0,
                  (1, 0): 0.0004000405334408536, (1, 1): 1.000200000266667,
                  (2, 2): 0.9950041652780257, (2, 3): 0.9983341664682817,
                  (3, 2): -0.009983341664682817, (3, 3): 0.9950041652780258}),
}


@pytest.mark.parametrize("case", sorted(_SECTOR_FRINGE))
def test_sector_bend_fringe_matches_madx(tmp_path, case):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    head, want = _SECTOR_FRINGE[case]
    deck = head + ("seq: sequence, refer=entry, l=1.0; b, at=0; endsequence;\n"
                   "use, sequence=seq;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    M = compute_transfer_matrix(lat, meta["reference"])
    for (i, j), v in want.items():
        assert M[i, j] == pytest.approx(v, abs=1e-12), (case, i, j)


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
b: sbend, l=1.0, angle={angle}, e1={e1}, e2={e2}, tilt={tilt}, k1={k1};
s: sequence, refer=entry, l=1.0; b, at=0; endsequence;
use, sequence=s;
"""

# k1 -> field index N = -k1 rho^2 with rho = l/angle = 5 m:
#   0 -> N = 0 (pure sector), 0.5 -> N = -12.5 (elliptic, N < 0),
#   -0.5 -> N = +12.5 (hyperbolic), -3 -> N = +75 (hyperbolic),
#   2 -> N = -50 (elliptic), -0.04 -> N = 1 (parabolic, kx2 = 0 to rounding)
# k1 -> field index n = -k1 rho^2 with rho = l/angle:
#   0 -> n = 0 (pure sector); 0.5, 2 -> n < 0; -0.5, -3 -> n > 1 (hyperbolic);
#   -0.04 -> n = 1 exactly (kx2 == 0, the parabolic limit).  The last two
#   straddle the series threshold |kx2 L^2| = 1e-3 used by the path-length
#   integral, which n = 1 exactly would otherwise never exercise.
_BEND_K1 = (0.0, 0.5, -0.5, -3.0, 2.0, -0.04, -0.039999999, -0.0400008)
_BEND_EDGES = ((0.05, 0.03), (0.0, 0.0))


@pytest.mark.parametrize("edges", _BEND_EDGES)
@pytest.mark.parametrize("k1", _BEND_K1)
@pytest.mark.parametrize("angle", [0.2, -0.2, 0.9, -0.9])
@pytest.mark.parametrize("tilt", ["0", "pi/2", "-pi/2"])
def test_bend_body_matches_madx_grid(tmp_path, tilt, angle, k1, edges):
    """Every body branch (N = 0, 0 < N, N < 0, N = 1, N > 1), both bend
    directions and all three planes (horizontal, vertical of either tilt
    sign), with pole-face angles: the transverse 4x6 block including the
    dispersion column equals MAD-X's.  ``tilt=-pi/2`` bends towards -y
    (negative HELIX angle); e1/e2 map to β = sign(angle)·e.

    Until 2026-09-06 the horizontal negative-angle bend got the inverse
    sector map (signed angle in the trig), the hyperbolic branch lost the
    x1000 unit factor on the dispersion and N = 1 had no dispersion at all
    (found 2026-09-03 by this oracle; the ("0", -0.2) case was xfail).
    Until 2026-09-07 the path-length row was identically zero and M[4,5]
    carried only a straight drift's velocity term.

    Not extended to theta = pi: MAD-X itself returns R52 off by a factor
    two and a nonsensical R56 there, while HELIX is smooth through."""
    deck = _BEND_DECK.format(angle=angle, tilt=tilt, k1=k1, e1=edges[0], e2=edges[1])
    Rx = _madx_R(deck)
    Rh, lat = _helix_R_from_deck(deck, tmp_path)
    d = _only(lat, Dipole)
    assert d.rho > 0
    assert (d.angle < 0) == ((angle < 0) != (tilt == "-pi/2"))
    # rows 0-4: the transverse block, the dispersion column AND the
    # path-length row (R51, R52, R56), added 2026-09-07.  Row 5 is
    # (0,0,0,0,0,1) in both codes.
    np.testing.assert_allclose(Rh[:5, :6], Rx[:5, :6], rtol=1e-10, atol=1e-12)


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
    # One rotation sense everywhere (since 2026-09-06): Multipole.tilt_deg,
    # Quadrupole.skew_angle and MAD-X's tilt agree, the converter copies the
    # angle through, and kick_matrix is the Jacobian of apply_kick.
    Rx = _madx_R(out.read_text(encoding="utf-8"))
    Rh = matrix_to_madx(compute_transfer_matrix(lat, ref.copy()), ref)
    np.testing.assert_allclose(Rh[:4, :4], Rx[:4, :4], rtol=1e-10, atol=1e-12)
    assert abs(Rx[1, 2]) > 0.1          # skew: x' couples to y
    q = Quadrupole("q", length=10.0, gradient=1.0, skew_angle=45.0)
    m_equiv = Multipole("m", knl=[0.0, 0.01 / ref.brho], ksl=[0.0], tilt_deg=45.0)
    assert np.sign(q.transfer_matrix(ref.copy())[1, 2]) == np.sign(m_equiv.kick_matrix(ref.copy())[1, 2])


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


# ---------------------------------------------------------------------------
# Tilted MULTIPOLE: the PARTICLE path against MAD-X TRACK (2026-09-06 — the
# converter used to negate the tilt to hide a sign error in kick_matrix, so
# multi-particle tracking of an imported tilted multipole rotated the wrong way)
# ---------------------------------------------------------------------------

_TILTED_MP_DECK = """beam, particle=proton, energy=0.94327208816;
m: multipole, knl={{0, 0.5}}, ksl={{0, {ksl1}}}, tilt={tilt};
s: sequence, refer=centre, l=0.2; m, at=0.1; endsequence;
use, sequence=s;
"""


@pytest.mark.parametrize("tilt, ksl1", [(0.4, 0.0), (-0.7, 0.0), (0.3, 0.2)])
def test_tilted_multipole_particle_kick_matches_madx_track(tmp_path, tilt, ksl1):
    cpymad = pytest.importorskip("cpymad.madx")
    from linac_gen.core.beam import Beam
    from linac_gen.elements.multipole import Multipole
    deck = _TILTED_MP_DECK.format(tilt=tilt, ksl1=ksl1)
    x0, y0 = 1.0e-3, 0.5e-3                                   # m
    m = cpymad.Madx(stdout=False)
    m.input(deck + f"track, onepass, onetable; start, x={x0}, px=0, y={y0}, py=0; "
                   "run, turns=1; endtrack;")
    t = m.table.trackone
    end = [(float(px), float(py)) for px, py, s in zip(t.px, t.py, t.s) if s > 0]
    m.quit()
    px_madx, py_madx = end[-1]
    path = tmp_path / "mp.madx"
    path.write_text(deck, encoding="utf-8")
    lat, meta = parse_madx(str(path))
    mp = _only(lat, Multipole)
    assert mp.tilt_deg == pytest.approx(tilt * 180.0 / np.pi, rel=1e-12)
    beam = Beam(ref=meta["reference"].copy(), n_particles=1, current=0.0)
    beam.particles[0, :] = [x0 * 1e3, 0.0, y0 * 1e3, 0.0, 0.0, 0.0]   # mm, mrad
    mp.apply_kick(beam)
    assert beam.particles[0, 1] * 1e-3 == pytest.approx(px_madx, rel=1e-9, abs=1e-12)
    assert beam.particles[0, 3] * 1e-3 == pytest.approx(py_madx, rel=1e-9, abs=1e-12)
    assert abs(px_madx) > 1e-5 and abs(py_madx) > 1e-5      # a real, coupled kick


def test_sextupole_tilt_is_imported_and_matches_a_tilted_multipole(tmp_path):
    """MAD-X ``SEXTUPOLE, tilt=`` used to be dropped silently; it is the same
    rotation as a MULTIPOLE knl={0,0,k2·l} with that tilt (both pinned
    against MAD-X TRACK through the multipole test above)."""
    from linac_gen.core.beam import Beam
    from linac_gen.elements.multipole import Multipole
    deck = """beam, particle=proton, energy=0.94327208816;
s6: sextupole, l=0.2, k2=5.0, tilt=0.4;
m6: multipole, knl={0, 0, 1.0}, tilt=0.4;
s: sequence, refer=centre, l=1.0; s6, at=0.2; m6, at=0.7; endsequence;
use, sequence=s;
"""
    path = tmp_path / "s6.madx"
    path.write_text(deck, encoding="utf-8")
    lat, meta = parse_madx(str(path))
    mps = [e for e in lat.elements if isinstance(e, Multipole)]
    assert len(mps) == 2
    s6, m6 = mps
    assert s6.tilt_deg == pytest.approx(0.4 * 180.0 / np.pi, rel=1e-12)
    assert s6.knl[2] == pytest.approx(1.0, rel=1e-12)
    for x0, y0 in ((1.0, 0.5), (-0.7, 1.2)):
        kicks = []
        for el in (s6, m6):
            beam = Beam(ref=meta["reference"].copy(), n_particles=1, current=0.0)
            beam.particles[0, :] = [x0, 0.0, y0, 0.0, 0.0, 0.0]
            el.apply_kick(beam)
            kicks.append(beam.particles[0, [1, 3]].copy())
        np.testing.assert_allclose(kicks[0], kicks[1], rtol=1e-12, atol=1e-15)
        assert abs(kicks[0][0]) > 1e-6
    assert not any("tilt" in w for w in meta.get("warnings", []))


def test_bare_tilt_flag_on_multipole_warns(tmp_path):
    deck = """beam, particle=proton, energy=0.94327208816;
m: multipole, knl={0, 0.5}, tilt;
s: sequence, refer=centre, l=0.2; m, at=0.1; endsequence;
use, sequence=s;
"""
    path = tmp_path / "bare.madx"
    path.write_text(deck, encoding="utf-8")
    lat, meta = parse_madx(str(path))
    assert any("bare 'tilt'" in w for w in meta.get("warnings", []))


# ---------------------------------------------------------------------------
# FINTX < 0, K1S / K2S, thin bends written as MULTIPOLEs, the declared beam
# (review findings 2026-10-27; every number below is MAD-X 5.09's, cpymad 1.19)
# ---------------------------------------------------------------------------

_FINTX_DECK = ("beam, particle=proton, energy=0.938272+0.0021;\n"
               "b: sbend, l=1.0, angle=0.1, hgap=0.02, fint=0.5, {fintx};\n"
               "seq: sequence, refer=entry, l=1.0; b, at=0; endsequence;\n"
               "use, sequence=seq;\n")
#: MAD-X's R43 and the exit edge's fringe integral: a negative FINTX (MAD-X's
#: default is -1) is the same as FINT; FINTX >= 0 is the exit face's own
_FINTX_R43 = {"fintx=-1": (0.0004000405334408536, 0.5),
              "fintx=-0.3": (0.0004000405334408536, 0.5),
              "fintx=0": (0.00020000026666709334, 0.0),
              "fintx=0.2": (0.00028001628375851095, 0.2)}


@pytest.mark.parametrize("fintx", sorted(_FINTX_R43))
def test_negative_fintx_is_fint_as_in_madx(tmp_path, fintx):
    """``fintx=-1`` used to become an exit-face fringe integral of -1
    (R43 = -2.0008e-4 against MAD-X's +4.0004e-4)."""
    deck = _FINTX_DECK.format(fintx=fintx)
    Rh, lat = _helix_R_from_deck(deck, tmp_path)
    r43, k1_out = _FINTX_R43[fintx]
    assert Rh[3, 2] == pytest.approx(r43, rel=1e-12)
    edges = [e for e in lat.elements if isinstance(e, Edge)]
    assert (edges[0].k1, edges[1].k1) == (0.5, k1_out)


@pytest.mark.parametrize("fintx", sorted(_FINTX_R43))
def test_negative_fintx_matches_madx_live(tmp_path, fintx):
    deck = _FINTX_DECK.format(fintx=fintx)
    Rx = _madx_R(deck)
    Rh, _ = _helix_R_from_deck(deck, tmp_path)
    np.testing.assert_allclose(Rh[:4, :4], Rx[:4, :4], rtol=1e-12, atol=1e-15)


def test_export_of_a_negative_exit_fringe_integral_warns(tmp_path):
    """The exporter's side of the FINTX rule: a negative exit K1 written as
    fintx comes back (in MAD-X and in HELIX) as fint."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.io.madx_writer import write_madx
    lat = Lattice()
    lat.add(Edge("b_e1", pole_rotation=0.0, rho=10000.0, gap=40.0, k1=0.5))
    lat.add(Dipole("b", angle=5.729577951308232, rho=10000.0))
    lat.add(Edge("b_e2", pole_rotation=0.0, rho=10000.0, gap=40.0, k1=-0.2))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=352.21)
    out = tmp_path / "b.madx"
    notes = [w for w in write_madx(lat, out, ref) if "fringe integral is negative" in w]
    assert len(notes) == 1 and "fint=0.5" in notes[0]
    back = [e for e in parse_madx(str(out))[0].elements if isinstance(e, Edge)]
    assert (back[0].k1, back[1].k1) == (0.5, 0.5)
    lat.elements[2].k1 = 0.2                    # a positive one is the exit's own
    assert not [w for w in write_madx(lat, out, ref) if "fringe integral" in w]
    back = [e for e in parse_madx(str(out))[0].elements if isinstance(e, Edge)]
    assert (back[0].k1, back[1].k1) == (0.5, 0.2)


_K1S_DECK = ("beam, {beam};\nq: quadrupole, l=0.4, k1={k1}, k1s={k1s}, tilt={tilt};\n"
             "s: sequence, refer=entry, l=0.4; q, at=0; endsequence;\nuse, sequence=s;\n")
_PROTON = "particle=proton, energy=0.94327208816"
_HMINUS = "particle=ion, mass=0.9392943, charge=-1, energy=1.7392943"
#: (beam, k1, k1s, tilt) -> MAD-X's transverse R (first two rows)
_K1S_R = {
    (_PROTON, -0.5, 0.3, 0.1): [
        [1.0347981002737194, 0.4046203408812423, 0.03146913193658969, 0.004195826289682009],
        [0.17581218843596927, 1.0347981002737194, 0.1573532680620623, 0.03146913193658969]],
    (_PROTON, 0.0, 0.4, 0.0): [
        [1.0001706670827684, 0.4000136533518268, 0.0320003640891848, 0.004266687471756865],
        [0.0017066749887027213, 1.0001706670827684, 0.16000546134073074, 0.0320003640891848]],
    (_HMINUS, 0.5, -0.3, 0.0): [
        [0.9603617014328453, 0.3946956248199565, -0.024000580267668783, -0.0032000331581316928],
        [-0.19638780246253873, 0.9603617014328453, -0.12000870402505279,
         -0.024000580267668783]],
}


@pytest.mark.parametrize("case", list(_K1S_R), ids=["k1<0", "k1=0", "h-"])
def test_quadrupole_k1s_turns_the_quadrupole_as_madx(tmp_path, case):
    """K1S used to be dropped without a word: sqrt(k1² + k1s²) at tilt −
    atan2(k1s, k1)/2, written with the sign of k1 kept."""
    beam, k1, k1s, tilt = case
    Rh, lat = _helix_R_from_deck(_K1S_DECK.format(beam=beam, k1=k1, k1s=k1s, tilt=tilt),
                                 tmp_path)
    np.testing.assert_allclose(Rh[:2, :4], _K1S_R[case], rtol=1e-12, atol=1e-15)
    q = _only(lat, Quadrupole)
    if k1 < 0:
        assert q.gradient < 0                 # still a defocusing quadrupole, turned a little
        assert q.skew_angle == pytest.approx(math.degrees(tilt + 0.5 * math.atan2(k1s, -k1)))


@pytest.mark.parametrize("case", list(_K1S_R), ids=["k1<0", "k1=0", "h-"])
def test_quadrupole_k1s_matches_madx_live(tmp_path, case):
    deck = _K1S_DECK.format(beam=case[0], k1=case[1], k1s=case[2], tilt=case[3])
    Rx = _madx_R(deck)
    Rh, _ = _helix_R_from_deck(deck, tmp_path)
    np.testing.assert_allclose(Rh[:4, :4], Rx[:4, :4], rtol=1e-12, atol=1e-15)


def test_quadrupole_without_k1s_is_unchanged(tmp_path):
    deck = _K1S_DECK.format(beam=_PROTON, k1=-0.5, k1s=0, tilt=0.1)
    _, lat = _helix_R_from_deck(deck, tmp_path)
    q = _only(lat, Quadrupole)
    assert q.skew_angle == math.degrees(0.1) and q.gradient < 0


_K2S_DECK = ("beam, particle=proton, energy=0.94327208816;\n"
             "s6: sextupole, l=0.2, k2={k2}, k2s={k2s}, tilt={tilt};\n"
             "s: sequence, refer=centre, l=0.2; s6, at=0.1; endsequence;\nuse, sequence=s;\n")
#: (k2, k2s, tilt, x0, y0) -> (px, py) after MAD-X's MAKETHIN (one slice: half
#: drift, kick, half drift — the importer's model) and TRACK
_K2S_KICKS = {
    (-5.0, 3.0, 0.25, 1e-3, 0.5e-3): (6.81340645496246e-07, 2.5889172407544985e-07),
    (-5.0, 3.0, 0.25, -0.7e-3, 1.2e-3): (-1.0946329144404207e-06, -2.612370238378735e-07),
    (0.0, 4.0, 0.0, 1e-3, 0.5e-3): (4.0000000000000003e-07, 3.0000000000000004e-07),
    (0.0, 4.0, 0.0, -0.7e-3, 1.2e-3): (-6.72e-07, -3.799999999999999e-07),
}


def _sextupole_kick(tmp_path, k2, k2s, tilt, x0, y0):
    from linac_gen.core.beam import Beam
    from linac_gen.elements.multipole import Multipole
    path = tmp_path / "s6.madx"
    path.write_text(_K2S_DECK.format(k2=k2, k2s=k2s, tilt=tilt))
    lat, meta = parse_madx(str(path))
    mp = _only(lat, Multipole)
    beam = Beam(ref=meta["reference"].copy(), n_particles=1, current=0.0)
    beam.particles[0, :] = [x0 * 1e3, 0.0, y0 * 1e3, 0.0, 0.0, 0.0]   # mm, mrad
    mp.apply_kick(beam)
    return beam.particles[0, 1] * 1e-3, beam.particles[0, 3] * 1e-3, mp, meta


@pytest.mark.parametrize("case", list(_K2S_KICKS))
def test_sextupole_k2s_turns_the_sextupole_as_madx(tmp_path, case):
    """K2S used to be dropped without a word: sqrt(k2² + k2s²) at tilt −
    atan2(k2s, k2)/3."""
    px, py, mp, meta = _sextupole_kick(tmp_path, *case)
    assert (px, py) == pytest.approx(_K2S_KICKS[case], rel=1e-12, abs=1e-20)
    assert math.copysign(1.0, mp.knl[2]) == math.copysign(1.0, case[0])
    assert meta["warnings"] == []


@pytest.mark.parametrize("case", list(_K2S_KICKS)[::2])
def test_sextupole_k2s_matches_madx_track_live(tmp_path, case):
    cpymad = pytest.importorskip("cpymad.madx")
    k2, k2s, tilt, x0, y0 = case
    m = cpymad.Madx(stdout=False)
    m.input(_K2S_DECK.format(k2=k2, k2s=k2s, tilt=tilt)
            + "select, flag=makethin, class=sextupole, slice=1;"
              " makethin, sequence=s, style=teapot; use, sequence=s;"
            + f"track, onepass, onetable; start, x={x0}, px=0, y={y0}, py=0;"
              " run, turns=1; endtrack;")
    t = m.table.trackone
    want = (float(t.px[-1]), float(t.py[-1]))
    m.quit()
    px, py, _mp, _meta = _sextupole_kick(tmp_path, *case)
    assert (px, py) == pytest.approx(want, rel=1e-12, abs=1e-20)


def test_bend_k1s_and_k2s_are_reported_not_modelled(tmp_path):
    deck = ("beam, particle=proton, energy=0.94327208816;\n"
            "b: sbend, l=1.0, angle=0.1, k1s=0.3, k2s=0.1;\n"
            "b0: sbend, l=1.0, angle=0.1;\n"
            "s: sequence, refer=entry, l=2.0; b, at=0; b0, at=1; endsequence;\n"
            "use, sequence=s;\n")
    lat, meta = parse_madx(_write(tmp_path, deck))
    assert [w for w in meta["warnings"] if "skew" in w] == [
        "bend 'b': K1S=0.3, K2S=0.1 (skew components) not modelled — dropped (MAD-X's "
        "TWISS leaves them out too)"]
    b, b0 = [e for e in lat.elements if isinstance(e, Dipole)]
    assert vars(b).keys() - {"name"} == vars(b0).keys() - {"name"}


def test_bend_k1s_is_left_out_by_madx_twiss_too_live(tmp_path):
    """What the warning says: MAD-X's TWISS map of a bend is the same with
    and without K1S/K2S."""
    head = ("beam, particle=proton, energy=0.94327208816;\n"
            "b: sbend, l=1.0, angle=0.1{};\n"
            "s: sequence, refer=entry, l=1.0; b, at=0; endsequence;\nuse, sequence=s;\n")
    np.testing.assert_array_equal(_madx_R(head.format(", k1s=0.3, k2s=0.1")),
                                  _madx_R(head.format("")))


_THIN_RING = """beam, particle=proton, energy=1.7;
qf: multipole, knl={0, 0.25};
qd: multipole, knl={0, -0.25};
bthin: multipole, knl={0.2617993877991494}, angle=0.2617993877991494;
d: drift, l=1.0;
cell: line=(qf, d, bthin, d, qd, d, bthin, d);
ring: line=(12*cell);
use, sequence=ring;
"""


def test_thin_bend_multipole_is_reported(tmp_path):
    """A MULTIPOLE with a dipole component or an ANGLE bends MAD-X's
    reference orbit (a thin bend); imported as a kick about a straight
    reference, it used to come in without a word (review experiment 10: a
    ring of thin bends imported as a straight line)."""
    from linac_gen.elements.multipole import Multipole
    lat, meta = parse_madx(_write(tmp_path, _THIN_RING))
    notes = [w for w in meta["warnings"] if "MULTIPOLE" in w]
    assert len(notes) == 1                         # one note for the 24 placements
    assert notes[0].startswith("MULTIPOLE 'bthin' has a dipole component or an ANGLE: "
                               "MAD-X bends the reference orbit there")
    assert "the reference path does not bend there" in notes[0]
    bends = [e for e in lat.elements if isinstance(e, Multipole) and e.name == "bthin"]
    assert len(bends) == 24 and bends[0].knl == [0.2617993877991494]   # (unchanged)


@pytest.mark.parametrize("definition, reported", [
    ("knl={0.01}", True), ("knl={0, 0.01}, angle=0.01", True), ("angle=0.01", True),
    ("ksl={0.01}", True), ("knl={0, 0.5}, ksl={0, 0.1}", False), ("knl={0}", False)])
def test_which_multipoles_are_thin_bends(tmp_path, definition, reported):
    deck = ("beam, particle=proton, energy=1.7;\n"
            f"mp: multipole, {definition};\nmq: multipole, knl={{0, 0.1}};\n"
            "s: sequence, l=2, refer=centre; mp, at=1; mq, at=1.5; endsequence;\n"
            "use, sequence=s;\n")
    _lat, meta = parse_madx(_write(tmp_path, deck))
    notes = [w for w in meta["warnings"] if "MULTIPOLE" in w]
    assert len(notes) == (1 if reported else 0)
    if reported:
        assert notes[0].startswith("MULTIPOLE 'mp' has a dipole component or an ANGLE")


def test_multipole_dipole_component_bends_madx_reference_live():
    """The premise of the note above: in MAD-X a MULTIPOLE's KNL[0] (no
    ANGLE) bends the reference orbit — TWISS keeps the orbit at 0 and SURVEY
    turns by -KNL[0] — while an HKICKER of the same strength kicks."""
    cpymad = pytest.importorskip("cpymad.madx")
    out = {}
    for kind in ("multipole, knl={0.01}", "hkicker, kick=0.01"):
        m = cpymad.Madx(stdout=False)
        m.input(f"beam, particle=proton, energy=1.7;\nmp: {kind};\n"
                "s: sequence, l=2, refer=centre; mp, at=1; endsequence; use, sequence=s;")
        tw = m.twiss(betx=1, bety=1)
        m.survey()
        out[kind] = (float(tw.x[-1]), float(m.table.survey.theta[-1]))
        m.quit()
    assert out["multipole, knl={0.01}"] == pytest.approx((0.0, -0.01), abs=1e-15)
    assert out["hkicker, kick=0.01"] == pytest.approx((0.01, 0.0), abs=1e-15)


@pytest.mark.parametrize("knl, ksl, reported", [
    ([0.01], [], True), ([0.0, 0.1], [0.002], True), ([0.0, 0.1], [], False),
    ([0.0, 0.1], [0.0, 0.05], False)])
def test_export_of_a_multipole_with_a_dipole_component_warns(tmp_path, knl, ksl, reported):
    """The export side of the thin-bend note: MAD-X takes a MULTIPOLE's
    KNL[0] / KSL[0] as a thin bend; HELIX's own reading of the file keeps
    the kick (round trip unchanged)."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.multipole import Multipole
    from linac_gen.io.madx_writer import write_madx
    lat = Lattice()
    lat.add(Drift("d1", length=500.0))
    lat.add(Multipole("mp", knl=list(knl), ksl=list(ksl)))
    lat.add(Multipole("mp2", knl=list(knl), ksl=list(ksl)))
    lat.add(Drift("d2", length=500.0))
    ref = ReferenceParticle(species=PROTON, w_kin=1700.0, frequency=352.21)
    out = tmp_path / "mp.madx"
    notes = [w for w in write_madx(lat, out, ref) if "dipole component" in w]
    assert len(notes) == (1 if reported else 0)    # one note for both
    if reported:
        assert notes[0].startswith("2 Multipoles (e.g. 'mp', 'mp2') have a dipole component")
    lat2, _meta = parse_madx(str(out))
    back = [e for e in lat2.elements if isinstance(e, Multipole)]
    assert [list(e.knl) for e in back] == [list(knl)] * 2


@pytest.mark.parametrize("beam, declared", [
    ("beam, particle=proton, energy=1.7;", True),
    ("beam, particle=proton, pc=1.2;", True),
    ("beam, particle=proton, gamma=1.9;", True),
    ("beam, particle=proton, beta=0.8;", True),
    ("beam, particle=proton, brho=4.0;", True),
    ("beam, particle=ion, mass=0.9392943, charge=-1;", False),
    ("beam, particle=proton;", False),
    ("", False),
    ("beam, particle=proton, energy=1.7;\nbeam, particle=proton;", True),
])
@pytest.mark.parametrize("form", ["sequence", "line"])
def test_reference_declared_only_when_the_beam_gives_the_energy(tmp_path, beam, declared, form):
    """``reference`` stays the beam the magnets are scaled with (a default
    without an energy); ``reference_declared`` says whether the deck gave it."""
    body = ("q: quadrupole, l=0.2, k1=1.0;\n"
            + ("s: sequence, l=0.2, refer=entry; q, at=0; endsequence;\nuse, sequence=s;\n"
               if form == "sequence" else "s: line=(q);\nuse, sequence=s;\n"))
    _lat, meta = parse_madx(_write(tmp_path, beam + "\n" + body))
    assert meta["reference_declared"] is declared
    assert meta["reference"] is not None


def test_harmon_is_taken_as_written_like_madx(tmp_path):
    """MAD-X computes a HARMON cavity's frequency with HARMON as written
    (84.7 gives 1058.729276223631 MHz on a 20 m line at 1.7 GeV, not the
    1049.979 MHz of h = 84); the gap keeps it as ``harmonic``."""
    deck = ("beam, particle=proton, energy=1.7;\n"
            "cav: rfcavity, l=0, volt=0.1, harmon=84.7, lag=0;\n"
            "s: sequence, l=20, refer=centre; cav, at=5; endsequence;\nuse, sequence=s;\n")
    lat, _meta = parse_madx(_write(tmp_path, deck))
    gap = _only(lat, RFGap)
    assert gap.frequency == pytest.approx(1058.729276223631, rel=1e-12)
    assert gap.harmonic == 84.7


def test_logical_attributes_read_true_without_a_warning(tmp_path):
    """``true`` stays 1 in the importer's expressions: logical attributes
    (``kill_ent_fringe=true``, ``thick=true`` — 1 488 of each in the reviewed
    CERN decks) go through the same evaluator, and MAD-X reads them as true.
    (MAD-X 5.09 takes ``true`` in a real-valued attribute as an undefined
    name, 0: ``k1=true`` is 0 there and 1 here — no deck writes that.)"""
    deck = ("beam, particle=proton, energy=1.7;\n"
            "b: sbend, l=1.0, angle=0.1, kill_ent_fringe=true, kill_exi_fringe=false, thick=true;\n"
            "s: sequence, l=1.0, refer=entry; b, at=0; endsequence;\nuse, sequence=s;\n")
    _lat, meta = parse_madx(_write(tmp_path, deck))
    assert meta["warnings"] == []
