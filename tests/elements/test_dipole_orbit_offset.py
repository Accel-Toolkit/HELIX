"""Dipole field error and roll steer the beam (orbit, not only optics).

Until 2026-10-07 ``Dipole.field_rel`` and a dipole ``tilt_deg`` changed the
linear matrix only: a beam entering on axis left the magnet on axis, so a
0.1 % BdL error (0.115 mrad on a BTL arc dipole) produced no orbit at all.
These tests pin the constant (inhomogeneous) term of the dipole map:

* field error — against MAD-X (cpymad): SBEND with ``K0 = h(1+δ)``;
* roll — against MAD-X ``EALIGN, DPSI`` (a rigid roll of the magnet; note
  MAD-X ``TILT`` is geometry, the reference follows it, and gives no
  orbit by design);
* both together, and a combined-function body;
* envelope centroid vs multi-particle centroid through a lattice,
  including vertical bends (hv=1) and both bend signs;
* slices compose to the full-element offset (the MP tracker slices);
* backtracking undoes it exactly;
* an ideal magnet adds nothing (None — the old paths stay bit-identical).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.mixins import Misalignment

W_MEV, F_MHZ = 800.0, 650.0


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=W_MEV, frequency=F_MHZ)


def _dipole(L_m, ang_rad, delta=0.0, psi_rad=0.0, n=0.0, hv=0):
    return Dipole("b", angle=math.degrees(ang_rad), rho=L_m / abs(ang_rad) * 1e3,
                  field_index=n, hv=hv, field_rel=delta,
                  tilt_deg=math.degrees(psi_rad))


def _helix_exit_offset(d, ref):
    """What the tracker wrapper produces for an on-axis particle:
    R(−ψ)·(magnet-frame field-error term) + roll exit offset (lab)."""
    o = d.orbit_offset(ref)
    o = np.zeros(6) if o is None else o
    o = Misalignment.tilt_rotation_matrix(-d.tilt_deg) @ o
    r = d.roll_exit_offset(ref)
    return o if r is None else o + r


def _madx_orbit(L, ang, delta, dpsi, k1=0.0):
    cpymad = pytest.importorskip("cpymad.madx")
    m = cpymad.Madx(stdout=False)
    m.input(f"""
    beam, particle=proton, energy={PROTON.mass / 1e3}+{W_MEV / 1e3};
    b: sbend, l={L}, angle={ang}, k0={ang / L * (1 + delta)}, k1={k1};
    seq: sequence, refer=entry, l={L}; b, at=0; endsequence;
    use, sequence=seq;
    select, flag=error, pattern="^b$"; ealign, dpsi={dpsi};
    twiss, betx=1, bety=1, x=0, px=0, y=0, py=0, t=0, pt=0;
    """)
    t = m.table.twiss
    out = np.array([t.x[-1], t.px[-1], t.y[-1], t.py[-1], t.t[-1]])
    m.quit()
    return out


def _to_madx_units(o, ref):
    """HELIX (mm, mrad, mm, mrad, deg) → MAD-X (m, rad, m, rad, T = −cΔt)."""
    lam_m = 299792458.0 / (F_MHZ * 1e6)
    return np.array([o[0] * 1e-3, o[1] * 1e-3, o[2] * 1e-3, o[3] * 1e-3,
                     -o[4] * lam_m / 360.0])


# ---------------------------------------------------------------------------
# MAD-X oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,ang,delta", [
    (2.34, 0.11455892, 1e-3),      # BTL arc dipole, +0.1 %
    (2.34, 0.11455892, -1e-3),
    (0.85, -0.0716337154, 2e-3),   # injection-chicane-like, negative angle
    (1.0, 0.5, 5e-3),              # large angle
])
def test_field_error_orbit_matches_madx(L, ang, delta):
    ref = _ref()
    got = _to_madx_units(_helix_exit_offset(_dipole(L, ang, delta), ref), ref)
    want = _madx_orbit(L, ang, delta, 0.0)
    # transverse: first order exact; the models differ at O(δ²)
    np.testing.assert_allclose(got[:2], want[:2], rtol=3 * abs(delta), atol=1e-15)
    assert got[2] == 0.0 and got[3] == 0.0
    # path length (time) term: right sign and size, O(δ) relative
    np.testing.assert_allclose(got[4], want[4], rtol=10 * abs(delta))


@pytest.mark.parametrize("L,ang,psi", [
    (2.34, 0.11455892, 2e-3),
    (2.34, 0.11455892, -2e-3),
    (2.34, -0.11455892, 2e-3),
    (1.0, 0.5, 0.01),
])
def test_roll_orbit_matches_madx_ealign_dpsi(L, ang, psi):
    ref = _ref()
    got = _to_madx_units(_helix_exit_offset(_dipole(L, ang, 0.0, psi), ref), ref)
    want = _madx_orbit(L, ang, 0.0, psi)
    # cross plane (the physics): first order in ψ exact in θ; the two
    # frame conventions differ at O(θ²ψ²) relative (2.6e-8 for a BTL arc
    # dipole rolled 2 mrad, 1.1e-5 at θ = 0.5 rad, ψ = 10 mrad)
    np.testing.assert_allclose(got[2:4], want[2:4],
                               rtol=max(1e-9, (ang * psi) ** 2))
    # bend plane is second order in ψ (∝ 1 − cos ψ): agree to that order
    np.testing.assert_allclose(got[:2], want[:2], rtol=2e-3, atol=1e-15)
    # path (time) term, also second order in ψ
    np.testing.assert_allclose(got[4], want[4], rtol=5e-3, atol=1e-16)


def test_field_error_plus_roll_matches_madx():
    L, ang, delta, psi = 1.0, 0.5, 1e-3, 0.01
    ref = _ref()
    got = _to_madx_units(_helix_exit_offset(_dipole(L, ang, delta, psi), ref), ref)
    want = _madx_orbit(L, ang, delta, psi)
    np.testing.assert_allclose(got[:4], want[:4], rtol=5e-3)


def test_combined_function_field_error_matches_madx():
    """N ≠ 0: the dispersion column already carries the body focusing."""
    L, ang, delta, k1 = 1.0, 0.1, 2e-3, 0.5
    n = -k1 * (L / ang) ** 2          # HELIX field index (madx_parser)
    ref = _ref()
    got = _to_madx_units(_helix_exit_offset(_dipole(L, ang, delta, n=n), ref), ref)
    want = _madx_orbit(L, ang, delta, 0.0, k1=k1)
    np.testing.assert_allclose(got[:2], want[:2], rtol=5 * abs(delta))



@pytest.mark.parametrize("L,ang,psi,hv", [
    (2.34, 0.11455892, 2e-3, 0),    # BTL arc dipole
    (1.0, 0.3, 5e-3, 0),
    (1.0, 0.5, -5e-3, 0),
    (0.85, 0.0716337154, 3e-3, 1),  # chicane-like vertical bend
])
def test_rolled_rectangular_magnet_with_separate_edges_matches_madx(L, ang, psi, hv):
    """EDGE + rolled BEND + EDGE (the layout every importer writes): the
    separate, unrolled exit Edge must not kick the roll offset — the beam
    of a rigidly rolled magnet leaves on that magnet's own exit.  MAD-X:
    SBEND with e1 = e2 = θ/2 and EALIGN DPSI (adversarial-review finding:
    before the fix the cross-plane kick was off by tan(e2)·tan(θ/2))."""
    cpymad = pytest.importorskip("cpymad.madx")
    from linac_gen.cli import common
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.edge import Edge
    rho_mm = L / ang * 1e3
    e_deg = math.degrees(ang) / 2
    lat = Lattice()
    lat.add(Edge("e1", pole_rotation=e_deg, rho=rho_mm, gap=0.0, k1=0.0, hv=hv))
    lat.add(Dipole("b", angle=math.degrees(ang), rho=rho_mm, hv=hv,
                   tilt_deg=math.degrees(psi)))
    lat.add(Edge("e2", pole_rotation=e_deg, rho=rho_mm, gap=0.0, k1=0.0, hv=hv))
    cfg = _cfg()
    c = np.asarray(common.run_envelope_sim(lat, cfg).centroid[-1])
    m = cpymad.Madx(stdout=False)
    tilt = f", tilt={math.pi / 2}" if hv else ""
    m.input(f"""
    beam, particle=proton, energy={PROTON.mass / 1e3}+{W_MEV / 1e3};
    b: sbend, l={L}, angle={ang}, e1={ang / 2}, e2={ang / 2}{tilt};
    seq: sequence, refer=entry, l={L}; b, at=0; endsequence;
    use, sequence=seq;
    select, flag=error, pattern="^b$"; ealign, dpsi={psi};
    twiss, betx=1, bety=1, x=0, px=0, y=0, py=0, t=0, pt=0;
    """)
    t = m.table.twiss
    want = np.array([t.x[-1], t.px[-1], t.y[-1], t.py[-1]]) * 1e3
    m.quit()
    # cross plane (x for a vertical bend, y for a horizontal one)
    cp = (0, 1) if hv else (2, 3)
    np.testing.assert_allclose(c[list(cp)], want[list(cp)],
                               rtol=max(1e-6, 2 * (ang * psi) ** 2))

# ---------------------------------------------------------------------------
# first-order identities (no oracle needed)
# ---------------------------------------------------------------------------
def test_small_error_limit_is_minus_delta_times_dispersion():
    ref = _ref()
    d = _dipole(2.34, 0.11455892, 1e-7)
    o = d.orbit_offset(ref)
    rho = 2.34 / 0.11455892
    theta = 0.11455892
    assert o[0] == pytest.approx(-1e-7 * rho * (1 - math.cos(theta)) * 1e3, rel=1e-6)
    assert o[1] == pytest.approx(-1e-7 * math.sin(theta) * 1e3, rel=1e-6)
    assert o[5] == 0.0


def test_vertical_bend_roll_kicks_horizontally():
    ref = _ref()
    psi = 1e-3
    for ang in (0.05, -0.05):
        d = _dipole(1.0, ang, 0.0, psi, hv=1)
        o = d.roll_exit_offset(ref)
        assert o[1] == pytest.approx(math.sin(ang) * math.sin(psi) * 1e3, rel=1e-9)
        # bend plane only at second order
        assert abs(o[3]) < 1e-3 * abs(o[1])
    # field error of a vertical bend acts in y
    o = _dipole(1.0, 0.05, 1e-3, hv=1).orbit_offset(ref)
    assert o[0] == 0.0 and o[1] == 0.0 and o[3] < 0.0


def test_ideal_magnet_adds_nothing():
    ref = _ref()
    d = _dipole(2.34, 0.11455892)
    assert d.orbit_offset(ref) is None and d.roll_exit_offset(ref) is None


def test_field_off_is_the_composable_limit():
    """δ = −1 (no field): the general formula is 0/0; its limit is the
    pipe's response to the whole missing deflection, x = θL/2, x' = θ,
    and the sliced (MP) and full (envelope) paths agree — the special case
    must compose like the general one (adversarial-review finding)."""
    ref = _ref()
    for hv, ang in ((0, 0.2), (1, -0.15)):
        d = _dipole(1.0, ang, -1.0, hv=hv)
        o = d.orbit_offset(ref)
        ip, ia = (2, 3) if hv else (0, 1)
        assert o[ip] == pytest.approx(0.5 * ang * 1000.0, rel=1e-12)
        assert o[ia] == pytest.approx(ang * 1e3, rel=1e-12)
        assert o[4] == 0.0
        q = d.length / 4
        c = np.zeros(6)
        for _ in range(4):
            c = d.transfer_matrix(ref, ds=q) @ c + d.orbit_offset(ref, ds=q)
        np.testing.assert_allclose(c, o, rtol=1e-12, atol=1e-15)
    # and it is the δ → −1 limit of the general formula
    near = _dipole(1.0, 0.2, -1.0 + 1e-9).orbit_offset(ref)
    np.testing.assert_allclose(near[:2], _dipole(1.0, 0.2, -1.0).orbit_offset(ref)[:2],
                               rtol=1e-6)


def test_slices_compose_to_the_full_offset():
    """Σ M_later·o_slice over 4 slices equals the full-element offset —
    the MP tracker slices, the envelope's I=0 path does not."""
    ref = _ref()
    for hv, n in ((0, 0.0), (1, 0.0), (0, 0.3)):
        d = _dipole(2.34, 0.11455892, 3e-3, n=n, hv=hv)
        full = d.orbit_offset(ref)
        q = d.length / 4
        c = np.zeros(6)
        for _ in range(4):
            c = d.transfer_matrix(ref, ds=q) @ c + d.orbit_offset(ref, ds=q)
        np.testing.assert_allclose(c, full, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# end to end through the real tracking paths
# ---------------------------------------------------------------------------
def _lattice():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    els = [
        Drift("d0", length=500.0),
        Dipole("b1", angle=6.5637, rho=20430.0, field_rel=1e-3, tilt_deg=0.1),
        Drift("d1", length=1500.0),
        Quadrupole("q1", length=200.0, gradient=6.0),
        Drift("d2", length=1000.0),
        Dipole("b2", angle=-4.1043, rho=11866.0, hv=1, field_rel=-2e-3,
               tilt_deg=-0.2),
        Drift("d3", length=800.0),
        Quadrupole("q2", length=200.0, gradient=-6.0),
        Dipole("b3", angle=2.3835, rho=13460.0, hv=1, field_rel=5e-4),
        Drift("d4", length=2000.0),
    ]
    for e in els:
        lat.add(e) if hasattr(lat, "add") else lat.elements.append(e)
    return lat


def _cfg(n=1):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=W_MEV, frequency=F_MHZ,
                      current=0.0, n_particles=n)


def test_envelope_centroid_equals_multiparticle_centroid():
    from linac_gen.cli import common
    from linac_gen.tracking.tracker import Tracker
    from linac_gen.core.beam import Beam
    lat = _lattice()
    env = common.run_envelope_sim(lat, _cfg())
    c_env = np.asarray(env.centroid[-1])
    ref = common.build_ref(_cfg())
    beam = Beam(ref=ref, n_particles=1, current=0.0)
    beam.particles[:] = 0.0
    rec = Tracker(copy.deepcopy(lat), beam, pic_solver="off").run()
    c_mp = np.asarray(rec.centroid[-1])
    assert np.max(np.abs(c_env[:4])) > 0.05           # the errors steer
    np.testing.assert_allclose(c_mp[:4], c_env[:4], rtol=1e-9, atol=1e-12)


def test_backtrack_undoes_errors_and_roll_exactly():
    from linac_gen.cli import common
    from linac_gen.core.beam import Beam
    from linac_gen.tracking.tracker import Tracker
    from linac_gen.tracking.backtrack import backtrack_distribution
    lat = _lattice()
    ref = common.build_ref(_cfg())
    beam = Beam(ref=ref.copy(), n_particles=8, current=0.0)
    rng = np.random.default_rng(3)
    beam.particles[:, :4] = rng.normal(scale=0.5, size=(8, 4))
    beam.particles[:, 4:] = rng.normal(scale=[1.0, 0.01], size=(8, 2))
    p0 = beam.particles[:, :6].copy()
    tr = Tracker(lat, beam, pic_solver="off")
    tr.run()
    assert np.max(np.abs(tr.beam.particles[:, :6] - p0)) > 0.1
    backtrack_distribution(lat, tr.beam, ref.copy(), pic_solver="off")
    np.testing.assert_allclose(tr.beam.particles[:, :6], p0, rtol=0, atol=1e-10)


def test_exit_edge_ownership_patterns():
    """Only a bend's OWN exit face is used (adversarial-review round 3):
    EDGE+BEND+EDGE yes; an edge-less bend before another magnet's
    EDGE+BEND+EDGE no; split magnet EDGE+B1+B2+EDGE: B2 yes, B1 no."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.edge import Edge
    from linac_gen.tracking.tracker import _exit_edge_after

    def E(n, rho=20000.0, hv=0):
        return Edge(n, pole_rotation=1.0, rho=rho, gap=0.0, k1=0.0, hv=hv)

    def B(n, rho=20000.0, hv=0):
        return Dipole(n, angle=2.0, rho=rho, hv=hv)

    lat = Lattice()
    els = [Drift("d0", length=100.0), E("e1"), B("b1"), E("e2"),          # owned
           Drift("d1", length=100.0), B("a"), E("e3"), B("c"), E("e4"),  # a: edge-less
           Drift("d2", length=100.0), E("e5"), B("s1"), B("s2"), E("e6"),  # split
           Drift("d3", length=100.0), E("e7", hv=1), B("v", hv=1), E("e8", hv=1),
           E("e9", rho=10000.0), B("w", rho=20000.0), E("e10", rho=10000.0)]  # rho mismatch
    for e in els:
        lat.add(e)
    get = {e.name: e for e in els}
    assert _exit_edge_after(lat, get["b1"]) is get["e2"]
    assert _exit_edge_after(lat, get["a"]) is None
    assert _exit_edge_after(lat, get["c"]) is get["e4"]
    assert _exit_edge_after(lat, get["s1"]) is None
    assert _exit_edge_after(lat, get["s2"]) is get["e6"]
    assert _exit_edge_after(lat, get["v"]) is get["e8"]
    assert _exit_edge_after(lat, get["w"]) is None


def test_rolled_edgeless_bend_before_a_rectangular_magnet_matches_madx():
    """The failing pattern of review round 3: a rolled sector bend with no
    edges directly followed by another magnet's EDGE+BEND+EDGE."""
    cpymad = pytest.importorskip("cpymad.madx")
    from linac_gen.cli import common
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.edge import Edge
    th, L, psi = 0.3, 1.0, 0.01
    rho_mm = L / th * 1e3
    lat = Lattice()
    lat.add(Dipole("a", angle=math.degrees(th), rho=rho_mm, tilt_deg=math.degrees(psi)))
    lat.add(Edge("e1", pole_rotation=math.degrees(th) / 2, rho=rho_mm, gap=0.0, k1=0.0))
    lat.add(Dipole("b", angle=math.degrees(th), rho=rho_mm))
    lat.add(Edge("e2", pole_rotation=math.degrees(th) / 2, rho=rho_mm, gap=0.0, k1=0.0))
    c = np.asarray(common.run_envelope_sim(lat, _cfg()).centroid[-1])
    m = cpymad.Madx(stdout=False)
    m.input(f"""
    beam, particle=proton, energy={PROTON.mass / 1e3}+{W_MEV / 1e3};
    a: sbend, l={L}, angle={th};
    b: sbend, l={L}, angle={th}, e1={th / 2}, e2={th / 2};
    seq: sequence, refer=entry, l={2 * L}; a, at=0; b, at={L}; endsequence;
    use, sequence=seq;
    select, flag=error, pattern="^a$"; ealign, dpsi={psi};
    twiss, betx=1, bety=1, x=0, px=0, y=0, py=0, t=0, pt=0;
    """)
    t = m.table.twiss
    want = np.array([t.y[-1], t.py[-1]]) * 1e3
    m.quit()
    np.testing.assert_allclose(c[2:4], want, rtol=1e-4)
