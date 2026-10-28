"""Curved field-map clusters (TraceWin ``SUPERPOSE_MAP_OUT``).

Pinned against exact geometry, not against the code itself:

1. A uniform B_y over the whole arc bends the reference on a circle: the
   traced path is ρθ and the exit lands on the card point for 30°, 90°,
   135°, 180° and a bend the other way (−45°); the 6×6 is the pure sector
   bend of :class:`Dipole` (x plane, dispersion, path-length row) and a
   drift in y.
2. The card frame's sign conventions: a card moved by δ gives the offset
   of the circle seen from the moved plane (closed form); a child's θY > 0
   turns its z axis toward +X and the exit card's toward −X (a 30°
   switching-dipole deck and the manual's dipole + quadrupole example).
3. The compiled push equals the numpy twin bit for bit, at any thread
   count, and the reference particle stays exactly on zero.
4. Envelope and multiparticle agree (centroid = M·0 + o), slices compose,
   the deck writer round-trips the cards, and the run modes HELIX does not
   offer for these maps refuse with a message.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements import curved_field_map as cfm
from linac_gen.elements.curved_field_map import (
    CurvedFieldMapCluster,
    child_rotation,
    cluster_field,
    exit_rotation,
)
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking.matrix_tracking import get_element_matrix

W = 2.5                                     # MeV, proton
FREQ = 352.21
RHO = 200.0                                 # mm
P_MEV = math.sqrt(W * (W + 2.0 * PROTON.mass))
B_T = P_MEV / (299.792458 * RHO * 1e-3)     # T for ρ = 200 mm
MARGIN = 30.0                               # mm of field before/after the arc


def _ref():
    return ReferenceParticle(PROTON, W, FREQ)


def _uniform_map(name="U", By=B_T, Bx=0.0, Bz=0.0, h=10.0):
    """Uniform field over a box that covers any arc of radius ρ starting at
    the origin along +z (the map is placed at Z = −MARGIN)."""
    x = np.arange(-(2 * RHO + MARGIN), 2 * RHO + MARGIN + 1e-9, h)
    y = np.arange(-20.0, 20.0 + 1e-9, h)
    z = np.arange(0.0, RHO + 2 * MARGIN + 1e-9, h)
    fd = FieldMapData(z=z, frequency=0.0, symmetry="3d")
    sh = (len(x), len(y), len(z))
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=7, x=x, y=y, z=z, Fx=np.full(sh, float(Bx)),
        Fy=np.full(sh, float(By)), Fz=np.full(sh, float(Bz)), norm_factor=1.0)
    return FieldMap3D(name=name, length=float(z[-1]), field_data=fd,
                      kb=1.0, ke=0.0)


def _circle_card(theta_deg):
    """SUPERPOSE_MAP_OUT operands of the exit of a circle of radius ρ that
    starts at the origin along +z (θ > 0 bends toward −x)."""
    a = math.radians(abs(theta_deg))
    sgn = 1.0 if theta_deg > 0 else -1.0
    return (RHO * math.sin(a), -sgn * RHO * (1.0 - math.cos(a)), 0.0,
            0.0, 0.0, float(theta_deg))


def _sector(theta_deg, card=None):
    sgn = 1.0 if theta_deg > 0 else -1.0
    fm = _uniform_map(By=sgn * B_T)
    return CurvedFieldMapCluster(
        f"C{theta_deg:g}", [(-MARGIN, fm)],
        placements=[(-MARGIN, 0.0, 0.0, 0.0, 0.0, 0.0)],
        exit_frame=card if card is not None else _circle_card(theta_deg))


# --------------------------------------------------------------------------- #
#  1. Exact circle geometry and the sector-bend map
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [30.0, 90.0, 135.0, 180.0, -45.0])
def test_uniform_field_is_an_exact_sector_bend(theta):
    cl = _sector(theta)
    traj = cl.prepare_reference(_ref())
    arc = RHO * math.radians(abs(theta))
    assert cl.length == pytest.approx(arc, rel=1e-12)
    assert not cl.length_provisional
    assert np.allclose(traj.residual, 0.0, atol=1e-9)
    M = cl.fitted_matrix(_ref())
    D = get_element_matrix(Dipole("D", angle=abs(theta), rho=RHO), _ref())
    if theta < 0:                   # bending toward +x: x mirrored
        S = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        D = S @ D @ S
    # x plane, dispersion, path-length row: the sector bend
    assert np.allclose(M, D, rtol=0.0, atol=1e-4), (M, D)
    # y: a drift of the arc length (B_y only: no vertical focusing)
    assert np.allclose(M[2:4, 2:4], [[1.0, arc * 1e-3], [0.0, 1.0]],
                       atol=1e-6)


def test_card_moved_by_delta_gives_the_circle_offset():
    """Card point moved by δ in X: the reference crosses the moved plane at
    φ with ρ sin(φ − θ) = −δ sin θ; seen from the card the beam sits at
    x = ρ(cos(φ − θ) − 1) − δ cos θ with x' = −tan(φ − θ) (closed form)."""
    th, delta = 30.0, 1.0
    Z, X, Y, tz, tx, ty = _circle_card(th)
    cl = _sector(th, card=(Z, X + delta, Y, tz, tx, ty))
    traj = cl.prepare_reference(_ref())
    a = math.radians(th)
    dphi = math.asin(-delta * math.sin(a) / RHO)
    x_exp = RHO * (math.cos(dphi) - 1.0) - delta * math.cos(a)
    xp_exp = -1e3 * math.tan(dphi)
    assert traj.residual[0] == pytest.approx(x_exp, abs=1e-9)
    assert traj.residual[1] == pytest.approx(xp_exp, abs=1e-8)
    assert abs(traj.residual[2]) < 1e-12 and abs(traj.residual[3]) < 1e-12
    # the run carries it as the beam offset: zero deviation in, offset out
    M, o = cl.exit_map(_ref())
    assert cl.orbit_offset(_ref()) is not None
    full = cl.fitted_matrix(_ref())
    assert np.isfinite(full).all()
    assert o[0] == pytest.approx(x_exp, abs=1e-9)
    assert o[1] == pytest.approx(xp_exp, abs=1e-8)


def test_reference_that_never_reaches_the_exit_plane_is_refused():
    cl = _sector(30.0, card=(-500.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="leaves the field maps without "
                                         "reaching the exit plane"):
        cl.prepare_reference(_ref())


# --------------------------------------------------------------------------- #
#  2. Sign conventions
# --------------------------------------------------------------------------- #
def test_child_and_exit_rotation_signs():
    # a child's θY > 0 turns its z axis toward +X (active rotation)
    assert child_rotation(0.0, 0.0, 10.0)[0, 2] == pytest.approx(math.sin(math.radians(10)))
    # the exit card's θY > 0 turns toward −X (a 30° switching dipole: +30°
    # with the exit at X < 0)
    assert exit_rotation(0.0, 0.0, 30.0)[0, 2] == pytest.approx(-0.5)
    # the manual's example: SUPERPOSE_MAP_OUT … -9.59023 and the
    # quadrupole's SUPERPOSE_MAP … 9.59023 both turn toward +X
    assert exit_rotation(0.0, 0.0, -9.59023)[0, 2] > 0.0
    assert child_rotation(0.0, 0.0, 9.59023)[0, 2] > 0.0
    # θX > 0 turns z toward +Y; θZ is right-handed about z
    assert child_rotation(0.0, 5.0, 0.0)[1, 2] == pytest.approx(math.sin(math.radians(5)))
    assert child_rotation(90.0, 0.0, 0.0) @ np.array([1.0, 0, 0]) == pytest.approx([0, 1.0, 0], abs=1e-15)
    for R in (child_rotation(12.0, -7.0, 33.0), exit_rotation(12.0, -7.0, 33.0)):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_placed_child_field_is_rotated_into_the_cluster_frame():
    """A uniform B_z map placed with θY = +10°: in the cluster frame the
    field points along the map's turned z axis, (sin 10°, 0, cos 10°)."""
    fm = _uniform_map(By=0.0, Bz=0.5)
    cl = CurvedFieldMapCluster("P", [(0.0, fm)],
                               placements=[(0.0, 3.0, -2.0, 0.0, 0.0, 10.0)],
                               exit_frame=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    pt = np.array([5.0]), np.array([1.0]), np.array([50.0])
    bx, by, bz = cluster_field(cl.packs(), *pt)
    s10 = math.sin(math.radians(10.0))
    assert bx[0] == pytest.approx(0.5 * s10, rel=1e-12)
    assert by[0] == pytest.approx(0.0, abs=1e-15)
    assert bz[0] == pytest.approx(0.5 * math.cos(math.radians(10.0)), rel=1e-12)


# --------------------------------------------------------------------------- #
#  3. Compiled push, threads, the zero orbit
# --------------------------------------------------------------------------- #
def _cloud(n=64, seed=7):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, 6))
    P[:, 0] = rng.normal(0, 2.0, n)
    P[:, 1] = rng.normal(0, 5.0, n)
    P[:, 2] = rng.normal(0, 2.0, n)
    P[:, 3] = rng.normal(0, 5.0, n)
    P[:, 4] = rng.uniform(-30, 30, n)
    P[:, 5] = rng.normal(0, 0.01, n)
    return P


def _push_through(cl, P, kernel):
    ref = _ref()
    cl.prepare_reference(ref)
    Q = P.copy()
    kq, inv_beta = cl._particle_constants(ref, Q[:, 5])
    kq_ref, ib_ref = cl._particle_constants(ref, np.zeros(1))
    n = cl.n_steps
    ds = cl.length / n
    frames = cl._stage_frames(0.5 * ds * np.arange(2 * n + 1), float(kq_ref[0]))
    c_phi = 360.0 / ref.wavelength
    if kernel:
        cfm._curved_kernel(Q, frames, ds, n, [p.as_tuple() for p in cl.packs()],
                           kq, inv_beta, float(ib_ref[0]), c_phi)
    else:
        cfm.curved_rk4_numpy(Q, frames, ds, n, cl.packs(), kq, inv_beta,
                             float(ib_ref[0]), c_phi)
    return Q


@pytest.mark.skipif(not cfm.curved_kernel_available(),
                    reason="compiled curved push not built")
def test_compiled_push_equals_numpy_twin_bitwise():
    cl = _sector(90.0)
    P = _cloud()
    a = _push_through(cl, P, kernel=True)
    b = _push_through(cl, P, kernel=False)
    assert np.array_equal(a, b)
    assert np.abs(a[:, :4] - P[:, :4]).max() > 1.0      # the particles moved


@pytest.mark.skipif(not cfm.curved_kernel_available(),
                    reason="compiled curved push not built")
def test_compiled_push_identical_across_thread_counts():
    from tests.helpers import kept_thread_setting
    cl = _sector(135.0)
    P = _cloud(n=257)
    out = []
    with kept_thread_setting() as threads:
        for n in (1, 2, 4):
            threads.set_threads(n)
            out.append(_push_through(cl, P, kernel=True))
    assert np.isfinite(out[0]).all()
    for o in out[1:]:
        assert np.array_equal(o, out[0])


@pytest.mark.parametrize("kernel", [True, False])
def test_reference_particle_stays_exactly_on_zero(kernel):
    if kernel and not cfm.curved_kernel_available():
        pytest.skip("compiled curved push not built")
    cl = _sector(30.0)
    ref = _ref()
    cl.prepare_reference(ref)
    beam = Beam(ref=ref, n_particles=1, current=0.0)
    beam.particles[:] = 0.0
    saved = cfm._USE_KERNEL
    cfm.use_curved_kernel(kernel)
    try:
        cl.reset_run_state()
        n = 2 * cl.n_steps
        for _ in range(n - 1):
            cl.track_rk4(beam, cl.length / n)
        assert np.array_equal(beam.particles[0, :4], np.zeros(4))
        cl.track_rk4(beam, cl.length / n)            # the step into the card
    finally:
        cfm._USE_KERNEL = saved
    assert np.allclose(beam.particles[0, :4], 0.0, atol=1e-9)
    assert beam.ref.s == pytest.approx(cl.length, rel=1e-12)


# --------------------------------------------------------------------------- #
#  4. Run modes
# --------------------------------------------------------------------------- #
def _offset_card_cluster():
    Z, X, Y, tz, tx, ty = _circle_card(30.0)
    return _sector(30.0, card=(Z, X + 0.5, Y + 0.2, tz, tx, ty + 0.05))


def test_multiparticle_matches_the_linear_map_and_offset():
    """Small deviations through the tracker: x_out = M x_in + o."""
    from linac_gen.tracking.tracker import Tracker
    cl = _offset_card_cluster()
    lat = Lattice()
    lat.elements = [cl]
    ref = _ref()
    beam = Beam(ref=ref, n_particles=7, current=0.0)
    d = np.zeros((7, 6))
    for k in range(6):
        d[k + 1, k] = (1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-5)[k]
    beam.particles[:] = d
    Tracker(lat, beam, pic_solver="off").run()
    M = cl.fitted_matrix(_ref())
    o = cl.orbit_offset(_ref())
    pred = d @ M.T + o
    assert np.allclose(beam.particles, pred, rtol=0.0, atol=1e-7)
    assert abs(o[2]) > 0.1 and abs(o[0]) > 0.1          # a real offset


def test_envelope_centroid_follows_the_exit_offset_and_slices_compose():
    from linac_gen.tracking.envelope import EnvelopeSolver
    cl = _offset_card_cluster()
    lat = Lattice()
    lat.elements = [Drift("D0", 50.0), cl, Drift("D1", 50.0)]
    init = dict(alpha_x=0.0, beta_x=1.0, emit_x=1.0, alpha_y=0.0, beta_y=1.0,
                emit_y=1.0, alpha_z=0.0, beta_z=1.0, emit_z=1e-3,
                centroid=[0.0] * 6)
    res = EnvelopeSolver(lat, _ref(), init, current=0.0).run()
    o = cl.orbit_offset(_ref())
    Dm = get_element_matrix(Drift("D1", 50.0), _ref())
    c_end = np.asarray(res.centroid[-1])
    assert np.allclose(c_end, Dm @ o, atol=1e-9)
    assert float(res.s[-1]) == pytest.approx(100.0 + cl.length, rel=1e-12)
    # slices compose to the whole-element map (the last one carries the exit)
    ref = _ref()
    cl.reset_run_state()
    n = 5
    Mc = np.eye(6)
    for _ in range(n):
        Mc = cl.fitted_matrix_slice(ref, cl.length / n) @ Mc
    assert np.allclose(Mc, cl.fitted_matrix(_ref()), rtol=0.0, atol=1e-5)


def test_refused_modes():
    from linac_gen.tracking.backtrack import backtrack_envelope
    from linac_gen.tracking.sacherer import SachererSolver
    from linac_gen.tracking.torch_tracking import element_matrix_torch
    cl = _sector(30.0)
    lat = Lattice()
    lat.elements = [cl]
    with pytest.raises(NotImplementedError, match="TraceWin does not offer it"):
        backtrack_envelope(lat, _ref(), {})
    with pytest.raises(ValueError, match="env_solver='matrix'"):
        SachererSolver(lat, _ref(), {}, current=0.0)
    with pytest.raises(TypeError, match="curved reference trajectory"):
        element_matrix_torch(cl, _ref())
    with pytest.raises(NotImplementedError):
        cl.untrack_rk4(None, 1.0)
    with pytest.raises(RuntimeError, match="straight-axis"):
        cl._ref_walk(_ref(), 0.0, 1.0, 1.0)


def test_electric_or_rf_child_is_refused():
    x = np.linspace(-10, 10, 3)
    z = np.linspace(0, 100, 11)
    fd = FieldMapData(z=z, frequency=0.0, symmetry="3d")
    sh = (3, 3, 11)
    fd.channels[Channel.STAT_E] = FieldChannel(
        geometry=7, x=x, y=x, z=z, Fx=np.zeros(sh), Fy=np.zeros(sh),
        Fz=np.ones(sh), norm_factor=1.0)
    fm = FieldMap3D(name="E", length=100.0, field_data=fd)
    with pytest.raises(ValueError, match="electric or RF channel"):
        CurvedFieldMapCluster("X", [(0.0, fm)],
                              exit_frame=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_prepare_sets_lengths_and_survey_draws_the_bend():
    from linac_gen.ring.detect import survey_2d
    cl = _sector(30.0)
    assert cl.length_provisional
    lat = Lattice()
    lat.elements = [Drift("D0", 100.0), cl, Drift("D1", 100.0)]
    assert cfm.prepare_curved_elements(lat, _ref()) == 1
    assert cl.length == pytest.approx(RHO * math.radians(30.0), rel=1e-12)
    x_end, z_end, th, vert = survey_2d(lat)
    # drawn like a +30° Dipole (beam physically toward −x, R16 > 0)
    assert math.degrees(th) == pytest.approx(30.0, abs=1e-12)
    a = math.radians(30.0)
    assert x_end == pytest.approx(RHO * (1 - math.cos(a)) + 100.0 * math.sin(a), abs=1e-9)
    assert z_end == pytest.approx(100.0 + RHO * math.sin(a) + 100.0 * math.cos(a), abs=1e-9)
    assert vert == 0.0
    from linac_gen.lattice_semantics import classify_element
    assert classify_element(cl)["kind"] == "dipole"


# --------------------------------------------------------------------------- #
#  Deck cards
# --------------------------------------------------------------------------- #
def _write_uniform_map_files(dirpath, name, By):
    x = np.arange(-(2 * RHO + MARGIN), 2 * RHO + MARGIN + 1e-9, 20.0) * 1e-3
    y = np.arange(-20.0, 20.0 + 1e-9, 20.0) * 1e-3
    z = np.arange(0.0, RHO + 2 * MARGIN + 1e-9, 20.0) * 1e-3
    for comp, val in (("bsx", 0.0), ("bsy", By), ("bsz", 0.0)):
        with open(dirpath / f"{name}.{comp}", "w") as f:
            f.write(f"{len(z) - 1} {z[-1]:.9e}\n{len(x) - 1} {x[0]:.9e} {x[-1]:.9e}\n"
                    f"{len(y) - 1} {y[0]:.9e} {y[-1]:.9e}\n1.0\n")
            f.write(f"{val!r}\n" * (len(x) * len(y) * len(z)))
    return float(z[-1] * 1e3)


def test_deck_cards_parse_and_round_trip(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin
    L = _write_uniform_map_files(tmp_path, "uni", 1.0)
    Z, X, Y, tz, tx, ty = _circle_card(30.0)
    deck = tmp_path / "bend.dat"
    deck.write_text(
        f"FIELD_MAP_PATH {tmp_path}\nDRIFT 100 20 0\n"
        f"SUPERPOSE_MAP_OUT {Z!r} {X!r} 0 0 0 30\n"
        f"SUPERPOSE_MAP {-MARGIN!r} 0 0 0 0 0\n"
        f"BEND1 : FIELD_MAP 70 {L!r} 0 1000 {B_T!r} 0 0 0 uni\n"
        "DRIFT 100 20 0\nEND\n")
    lat, meta = parse_tracewin(str(deck), strict=True)
    cl = next(e for e in lat.elements if getattr(e, "is_curved", False))
    assert cl.exit_frame == pytest.approx((Z, X, 0.0, 0.0, 0.0, 30.0))
    assert cl.placements == [(-MARGIN, 0.0, 0.0, 0.0, 0.0, 0.0)]
    assert cl.label == "BEND1"
    assert not any("SUPERPOSE" in w for w in meta["warnings"]), meta["warnings"]
    out = tmp_path / "out.dat"
    write_tracewin(lat, str(out))
    text = out.read_text()
    assert "SUPERPOSE_MAP_OUT" in text
    lat2, meta2 = parse_tracewin(str(out), strict=True)
    cl2 = next(e for e in lat2.elements if getattr(e, "is_curved", False))
    assert cl2.exit_frame == pytest.approx(cl.exit_frame)
    for p2, p in zip(cl2.placements, cl.placements, strict=True):
        assert p2 == pytest.approx(p)
    assert cl2.label == "BEND1"
    assert len(lat2.elements) == len(lat.elements)
    # the physics survives the round trip
    # (the deck prints the exit frame and the field factor with 10 digits)
    assert cl2.prepare_reference(_ref()).length == pytest.approx(
        cl.prepare_reference(_ref()).length, rel=1e-9)


def test_dangling_exit_card_warns_and_strict_raises(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "d.dat"
    deck.write_text("DRIFT 100 15 0\nSUPERPOSE_MAP_OUT 100 0 0 0 0 0\n"
                    "DRIFT 100 15 0\nEND\n")
    lat, meta = parse_tracewin(str(deck))
    assert any("SUPERPOSE_MAP_OUT not followed" in w for w in meta["warnings"])
    with pytest.raises(ValueError, match="SUPERPOSE_MAP_OUT"):
        parse_tracewin(str(deck), strict=True)


# --------------------------------------------------------------------------- #
#  Review round (2026-10-26): exact helix, H⁻ at 30 keV, guards, copies
# --------------------------------------------------------------------------- #
def _uniform_box(name, B, half=350.0, h=25.0):
    """Uniform field vector B (T) over a cube of ±half mm (the map placed at
    Z = −half, so it covers the entrance region too)."""
    a = np.arange(-half, half + 1e-9, h)
    z = np.arange(0.0, 2.0 * half + 1e-9, h)
    fd = FieldMapData(z=z, frequency=0.0, symmetry="3d")
    sh = (len(a), len(a), len(z))
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=7, x=a, y=a, z=z, Fx=np.full(sh, float(B[0])),
        Fy=np.full(sh, float(B[1])), Fz=np.full(sh, float(B[2])),
        norm_factor=1.0)
    return FieldMap3D(name=name, length=float(z[-1]), field_data=fd,
                      kb=1.0, ke=0.0)


def test_non_planar_helix_of_an_h_minus_beam():
    """A uniform field with all three components bends a 30 keV H⁻ reference
    on a helix (both curvatures of the frame non-zero).  The card set from
    the exact helix at arc length S: the traced path is S and the reference
    lands on the card's point and direction."""
    from linac_gen.core.particle import H_MINUS
    ref = ReferenceParticle(H_MINUS, 0.030, 162.5)
    B = np.array([0.03, 0.06, 0.02])                      # T
    p = math.sqrt(0.030 * (0.030 + 2.0 * H_MINUS.mass))
    k = -1.0 * 299.792458e-3 / p                          # q/p, 1/(T·mm)
    Bn = float(np.linalg.norm(B))
    w, b = k * Bn, B / Bn                                 # du/ds = w u × b
    u0 = np.array([0.0, 0.0, 1.0])
    upar = (u0 @ b) * b
    uperp = u0 - upar
    S = 300.0
    ws = w * S
    r = upar * S + uperp * math.sin(ws) / w + np.cross(uperp, b) * (1 - math.cos(ws)) / w
    t = upar + uperp * math.cos(ws) + np.cross(uperp, b) * math.sin(ws)
    # exit_rotation(0, θX, θY) has z axis (−sin θY, −sin θX cos θY, cos θX cos θY)
    ty_ = -math.degrees(math.asin(t[0]))
    tx_ = math.degrees(math.atan2(-t[1], t[2]))
    fm = _uniform_box("H", B)
    cl = CurvedFieldMapCluster("HX", [(-350.0, fm)],
                               placements=[(-350.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                               exit_frame=(r[2], r[0], r[1], 0.0, tx_, ty_))
    traj = cl.prepare_reference(ref)
    assert cl.length == pytest.approx(S, rel=1e-11)
    assert np.allclose(traj.residual, 0.0, atol=1e-8)
    assert np.abs(traj.k2).max() > 1e-5 and np.abs(traj.k1).max() > 1e-5


def test_dispersion_terms_at_30_kev():
    """H⁻ at 30 keV through a uniform-field sector: R16, R26 and R56 follow
    the Dipole's (the energy step of the fitted matrix scales with W; a
    fixed 1 keV step biased them by 7e-4)."""
    from linac_gen.core.particle import H_MINUS
    ref = ReferenceParticle(H_MINUS, 0.030, 162.5)
    p = math.sqrt(0.030 * (0.030 + 2.0 * H_MINUS.mass))
    B = p / (299.792458 * RHO * 1e-3)
    # H⁻ in B_y > 0 bends toward +x: the card of a −30° bend
    fm = _uniform_map(By=B)
    cl = CurvedFieldMapCluster("L", [(-MARGIN, fm)],
                               placements=[(-MARGIN, 0.0, 0.0, 0.0, 0.0, 0.0)],
                               exit_frame=_circle_card(-30.0))
    M = cl.fitted_matrix(ref)
    S = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    D = S @ get_element_matrix(Dipole("D", angle=30.0, rho=RHO), ref.copy()) @ S
    for i, j in ((0, 5), (1, 5), (4, 5), (4, 0), (4, 1)):
        assert M[i, j] == pytest.approx(D[i, j], rel=1e-5), (i, j, M[i, j], D[i, j])


def test_the_exit_into_the_card_frame_applies_once():
    cl = _sector(30.0)
    ref = _ref()
    cl.prepare_reference(ref)
    beam = Beam(ref=ref, n_particles=2, current=0.0)
    beam.particles[:] = 0.0
    cl.reset_run_state()
    for _ in range(4):
        cl.track_rk4(beam, cl.length / 4)
    with pytest.raises(RuntimeError, match="past the end"):
        cl.track_rk4(beam, 1e-6)
    cl.reset_run_state()                                   # a new run starts clean
    cl.track_rk4(beam, cl.length / 4)
    # the envelope's slices: none past the end either
    cl.reset_run_state()
    for _ in range(2):
        cl.fitted_matrix_slice(_ref(), cl.length / 2)
    with pytest.raises(RuntimeError, match="past the end"):
        cl.fitted_matrix_slice(_ref(), 1.0)


def test_a_copy_of_the_lattice_keeps_the_trace(monkeypatch):
    """deepcopy (error studies) and pickling (worker processes) reuse the
    trace: its cache keys on the maps' content, not on object identity."""
    import copy
    import pickle
    cl = _sector(90.0)
    cl.prepare_reference(_ref())

    def no_trace(*a, **k):
        raise AssertionError("traced again")
    for clone in (copy.deepcopy(cl), pickle.loads(pickle.dumps(cl))):
        monkeypatch.setattr(cfm, "integrate_reference", no_trace)
        assert clone.prepare_reference(_ref()).length == pytest.approx(cl.length)
        monkeypatch.undo()
    # …and a real change is traced again
    clone = copy.deepcopy(cl)
    clone.children[0][1].kb = 1.01
    assert clone.prepare_reference(_ref()).length != pytest.approx(cl.length, rel=1e-9)


def test_interior_diagnostic_beyond_the_path_is_reported():
    from linac_gen.elements.marker import Marker
    fm = _uniform_map(By=B_T)
    cl = CurvedFieldMapCluster(
        "C", [(-MARGIN, fm)], placements=[(-MARGIN, 0.0, 0.0, 0.0, 0.0, 0.0)],
        exit_frame=_circle_card(30.0),
        interior_markers=[(50.0, Marker("IN")), (150.0, Marker("OUT"))])
    with pytest.warns(UserWarning, match="'OUT' is not recorded"):
        cl.prepare_reference(_ref())


def test_bends_up_to_a_full_turn_are_traced():
    """A 350° circle (the field covers the whole turn): the exit plane lies
    just behind the entrance and the start is on its downstream side."""
    half = 2.0 * RHO + MARGIN
    fm = _uniform_box("F", (0.0, B_T, 0.0), half=half, h=20.0)
    cl = CurvedFieldMapCluster("F", [(-half, fm)],
                               placements=[(-half, 0.0, 0.0, 0.0, 0.0, 0.0)],
                               exit_frame=_circle_card(350.0))
    traj = cl.prepare_reference(_ref())
    assert cl.length == pytest.approx(RHO * math.radians(350.0), rel=1e-11)
    assert np.allclose(traj.residual, 0.0, atol=1e-8)


def test_magnetic_stripping_sees_the_bend_field():
    from linac_gen.analysis.magnetic_stripping import _design_b_for_element
    cl = _sector(30.0)
    cl.prepare_reference(_ref())
    b, kind = _design_b_for_element(cl, 1.0, 1.0, 1.0, "2sigma", "max")
    assert kind == "dipole"
    assert b == pytest.approx(B_T, rel=1e-12)


@pytest.mark.parametrize("mode", ["advance_ref", "tracker"])
def test_backtrack_replay_traces_the_cluster_with_its_entrance_beam(mode):
    """The backtracking replay (over the elements downstream of a curved
    cluster) traces it for the reference that reaches it, whatever another
    beam traced it for before."""
    from linac_gen.tracking.backtrack import build_replay_table
    cl = _sector(30.0)
    stale = ReferenceParticle(PROTON, 3.0, FREQ)           # traced for another beam
    cl.prepare_reference(stale)
    assert cl.length != pytest.approx(RHO * math.radians(30.0), rel=1e-6)
    lat = Lattice()
    lat.elements = [Drift("D0", 50.0), cl, Drift("D1", 50.0), Drift("D2", 50.0)]
    table = build_replay_table(lat, _ref(), end=3, field_map_ref_mode=mode)
    assert table[2].s == pytest.approx(50.0 + RHO * math.radians(30.0), rel=1e-12)
    assert table[-1].s == pytest.approx(150.0 + RHO * math.radians(30.0), rel=1e-12)


def test_a_bare_deck_runs_with_the_command_line_beam(tmp_path):
    """No trace with the default beam at load: an H⁻ deck runs with
    --species/--energy on the command line."""
    import h5py

    from linac_gen.__main__ import main
    from linac_gen.core.particle import H_MINUS
    p = math.sqrt(2.5 * (2.5 + 2.0 * H_MINUS.mass))
    B = p / (299.792458 * RHO * 1e-3)
    L = _write_uniform_map_files(tmp_path, "uni", 1.0)
    Z, X, Y, tz, tx, ty = _circle_card(30.0)
    deck = tmp_path / "hminus.dat"
    deck.write_text(
        f"FIELD_MAP_PATH {tmp_path}\nDRIFT 100 20 0\n"
        f"SUPERPOSE_MAP_OUT {Z!r} {X!r} 0 0 0 30\n"
        f"SUPERPOSE_MAP {-MARGIN!r} 0 0 0 0 0\n"
        f"FIELD_MAP 70 {L!r} 0 1000 {-B!r} 0 0 0 uni\n"   # H⁻: B_y < 0 bends to −x
        "DRIFT 100 20 0\nEND\n")
    rc = main(["run", str(deck), "--species", "H-", "--energy", "2.5",
               "--freq", "352.21", "--mode", "envelope",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    (h5,) = tmp_path.glob("*_results.h5")
    with h5py.File(h5, "r") as f:
        s = f["envelope/s"][:]
    assert s[-1] == pytest.approx(200.0 + RHO * math.radians(30.0), rel=1e-9)


def test_untraced_until_nothing_before_can_change_the_energy():
    """The GUI's preparation touches no other element: a cluster behind an
    RF gap waits for a run; one behind drifts and magnets is traced."""
    from linac_gen.elements.rf_gap import RFGap
    cl = _sector(30.0)
    lat = Lattice()
    lat.elements = [Drift("D0", 50.0), RFGap("G", 0.1, -30.0, FREQ), cl]
    assert cfm.prepare_curved_elements(lat, _ref(), replay=False) == 0
    assert cl.length_provisional
    lat.elements = [Drift("D0", 50.0), cl]
    assert cfm.prepare_curved_elements(lat, _ref(), replay=False) == 1
    assert cl.length == pytest.approx(RHO * math.radians(30.0), rel=1e-12)


def test_a_reference_that_leaves_the_maps_is_refused_at_once():
    """Stiff beam, card far to the side: the reference leaves the maps away
    from the exit plane — refused without stepping to the path bound, and
    the failure is remembered."""
    import time

    from linac_gen.core.particle import H_MINUS
    cl = _sector(90.0)
    ref = ReferenceParticle(H_MINUS, 200.0, FREQ)          # bends the other way
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="leaves the field maps|does not reach"):
        cl.prepare_reference(ref)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    with pytest.raises(ValueError):
        cl.prepare_reference(ref)
    assert time.perf_counter() - t0 < 0.05 * max(first, 1e-3) + 0.01
