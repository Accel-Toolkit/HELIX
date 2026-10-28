"""The one-turn contract: ring mode and HELIX's linac matrices describe the
same lattice — every element's ring nodes, linearized at the reference, equal
HELIX's own element matrix to 1e-10 (after the change of coordinates), for a
proton and for an H- lattice, and so does the whole turn up to the float64
finite-difference floor.

Where the two differ by design, the test says so and checks what both
describe: an off-centre magnet (the ring map also carries the reference's
chromatic oscillation about the magnet axis — its dispersion is checked
against the closed form) and an RF gap (a thin ring cavity has no transverse
map, as in MAD-X; the linac gap adds its transverse RF
defocusing — the longitudinal rows are compared).  A pole face at zero angle
with a gap acts through its fringe field in both (weakly defocusing across
the bend plane, MAD-X's -h tan(-psi); tests/ring/test_optics_booster.py pins
the ring to MAD-X).

The element Jacobians are taken in 50-digit arithmetic (the kernel sources on
mpmath), so 1e-10 measures the maps, not rounding; bends use enough
integration steps that the integrator error is below it too.
"""
from __future__ import annotations

import copy
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.thin_lens import ThinLens
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.reference import RingReference
from linac_gen.tracking.matrix_tracking import _with_tilt, get_element_matrix
from tests.ring import mp_kernels

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"
C = 299792458.0
CART, _CURV, ND = mp_kernels.load(50)


def _to_helix(rl, f_mhz):
    """T with M_helix = T M_ring T^-1 (x mm, x' mrad, y, y', phi deg at f, W MeV)."""
    r = rl.reference
    return np.diag([1e3, 1e3, 1e3, 1e3, -360.0 * f_mhz * 1e6 / (r.beta0 * C), r.beta0 * r.p0c])


def _element_jacobian(rl, idx):
    """Jacobian (6x6, ring coordinates) of the nodes of element ``idx`` at the
    reference, in 50-digit arithmetic."""
    js = [j for j in range(rl.n_nodes) if rl.node_element[j] == idx]
    ends = list(rl.offsets[1:]) + [rl.params.size]
    blocks = [(int(rl.kinds[j]), [mp.mpf(float(v)) for v in rl.params[rl.offsets[j]:ends[j]]])
              for j in js]
    b0 = mp.mpf(float(rl.reference.beta0))
    p0c = mp.mpf(float(rl.reference.p0c))
    bg2 = (p0c / mp.mpf(float(rl.reference.mass))) ** 2

    def f(z):
        x, px, y, py, zeta, delta = z
        ptau, rpp, rvv = CART.energy_from_delta(delta, b0)
        for kind, p in blocks:
            (x, px, y, py, zeta, delta, ptau, rpp, rvv, _l) = ND.track_node(
                kind, p, 0, b0, p0c, bg2, x, px, y, py, zeta, delta, ptau, rpp, rvv)
        return [x, px, y, py, zeta, delta]
    h = mp.mpf("1e-20")
    J = np.zeros((6, 6))
    for k in range(6):
        zp = [mp.mpf(0)] * 6
        zm = [mp.mpf(0)] * 6
        zp[k] += h
        zm[k] -= h
        fp, fm = f(zp), f(zm)
        for i in range(6):
            J[i, k] = float((fp[i] - fm[i]) / (2 * h))
    return J


def _check_elements(lat, href, *, skip=(), species="proton"):
    w = href.w_kin
    rr = RingReference.from_kinetic(species, w)
    rl = compile_ring(lat, rr, lattice_species=species, lattice_energy=w, bend_steps=64)
    T = _to_helix(rl, href.frequency)
    Ti = np.linalg.inv(T)
    ref = href.copy()
    checked = 0
    for idx, e in enumerate(lat.elements):
        Mh = _with_tilt(e, get_element_matrix(e, ref))
        ref.s += e.length
        if type(e).__name__ in skip:
            continue
        Mr = T @ _element_jacobian(rl, idx) @ Ti
        scale = np.maximum(np.abs(Mh), 1.0)
        if getattr(e, "dx", 0.0) or getattr(e, "dy", 0.0):
            # an off-centre magnet also kicks the reference, by an amount
            # that scales with 1/(1 + delta) in a ring (dispersion, path
            # length); HELIX's linac matrix leaves that to the orbit — the
            # transverse block is what both describe
            Mr, Mh, scale = Mr[:4, :4], Mh[:4, :4], scale[:4, :4]
        if isinstance(e, RFGap):
            # a thin cavity has no transverse map in a ring (MAD-X's RF
            # cavity); HELIX's linac gap adds the transverse RF
            # defocusing — the longitudinal rows are what both describe
            assert np.abs(Mr[:4] - np.eye(6)[:4]).max() <= 1e-15
            assert Mh[1, 0] != 0.0 and Mh[3, 2] != 0.0
            Mr, Mh, scale = Mr[4:], Mh[4:], scale[4:]
            assert abs(Mh[1, 4]) > 1e-6            # (the cavity acts: dW/dphi)
        err = (np.abs(Mr - Mh) / scale).max()
        assert err <= 1e-10, (f"{e.name} ({type(e).__name__}): |M_ring - M_helix| = {err:.2e}\n"
                              f"ring:\n{Mr}\nHELIX:\n{Mh}")
        checked += 1
    return checked


def test_every_booster_element_type_matches_helix():
    lat, meta = parse_madx(str(FIXTURE))
    href = meta["reference"]
    # one cell: drifts, quadrupoles, sextupoles (half drift, thin, half drift),
    # combined-function bends with their pole faces, an RF cavity
    cell = Lattice()
    for e in lat.elements[:37]:
        cell.add(copy.deepcopy(e))
    # (at the deck's own voltage per cavity, 200 kV / 22)
    gap = RFGap(name="rf", voltage=0.2 / 22, phase=-90.0, frequency=href.frequency)
    cell.add(gap)
    assert _check_elements(cell, href) == 38


def test_an_h_minus_lattice_matches_helix_with_its_charge():
    # the magnets of an H- lattice are normalized with its (negative)
    # rigidity and its cavities give it energy with its charge, as HELIX's
    # linac does: the same matrices, RF included
    href = ReferenceParticle(H_MINUS, 800.0, 44.7044096)
    lat = Lattice()
    rho = 40847.086
    lat.add(Edge(name="e1", pole_rotation=2.0, rho=rho, gap=50.0, k1=0.45))
    lat.add(Dipole(name="b", angle=4.05, rho=rho, field_index=-80.0))
    lat.add(Edge(name="e2", pole_rotation=2.0, rho=rho, gap=50.0, k1=0.45))
    lat.add(Quadrupole(name="q", length=300.0, gradient=2.5))
    lat.add(Steerer(name="st", by_l=0.001, bx_l=-0.002))
    lat.add(RFGap(name="rf", voltage=0.2 / 22, phase=-90.0, frequency=44.7044096))
    assert _check_elements(lat, href, species="H-") == len(lat.elements)


def test_vertical_negative_tilted_and_shifted_elements_match_helix():
    href = ReferenceParticle(PROTON, 800.0, 44.7044096)
    lat = Lattice()
    rho = 40847.086
    for hv, ang in ((1, 4.05), (0, -4.05), (1, -3.0)):
        beta_e = 2.0 if ang > 0 else -2.0
        lat.add(Edge(name=f"e1_{hv}_{ang}", pole_rotation=beta_e, rho=rho, gap=50.0, k1=0.45, hv=hv))
        lat.add(Dipole(name=f"b_{hv}_{ang}", angle=ang, rho=rho, field_index=-80.0, hv=hv))
        lat.add(Edge(name=f"e2_{hv}_{ang}", pole_rotation=beta_e, rho=rho, gap=50.0, k1=0.45, hv=hv))
        lat.add(Drift(name=f"d_{hv}_{ang}", length=500.0))
    lat.add(Dipole(name="own_edges", angle=3.0, rho=rho, e1=1.5, e2=-0.5, field_index=20.0))
    lat.add(Quadrupole(name="q_tilted", length=300.0, gradient=2.5, tilt_deg=7.0))
    lat.add(Quadrupole(name="q_skew", length=300.0, gradient=-1.5, skew_angle=45.0))
    lat.add(Quadrupole(name="q_shifted", length=300.0, gradient=2.0, dx=1.5, dy=-0.7))
    lat.add(Multipole(name="m_tilted", knl=[0.0, 0.02, 0.5], tilt_deg=12.0))
    lat.add(ThinLens(name="lens", fx=12.0, fy=-30.0))
    lat.add(Steerer(name="st", by_l=0.001, bx_l=-0.002))
    assert _check_elements(lat, href) == len(lat.elements)


def test_square_pole_faces_with_a_fringe_gap_match_helix():
    # (sector bends with a fringe gap, both bend planes: the faces act at
    # zero angle — weakly defocusing — in the ring and in HELIX's Edge alike)
    href = ReferenceParticle(PROTON, 800.0, 44.7044096)
    lat = Lattice()
    rho = 40847.086
    for hv in (0, 1):
        lat.add(Edge(name=f"s1_{hv}", pole_rotation=0.0, rho=rho, gap=50.0, k1=0.45, hv=hv))
        lat.add(Dipole(name=f"sb_{hv}", angle=4.05, rho=rho, field_index=0.0, hv=hv))
        lat.add(Edge(name=f"s2_{hv}", pole_rotation=0.0, rho=rho, gap=50.0, k1=0.45, hv=hv))
        lat.add(Drift(name=f"sd_{hv}", length=500.0))
    m = Edge("x", pole_rotation=0.0, rho=rho, gap=50.0, k1=0.45).transfer_matrix(href)
    assert m[3, 2] > 0.0                          # (the fringe acts at zero angle)
    assert _check_elements(lat, href) == len(lat.elements)


def test_a_shifted_quadrupole_makes_dispersion_by_its_chromatic_focusing():
    # the part of an off-centre magnet's ring map that HELIX's linac matrix
    # leaves out: the reference itself oscillates about the magnet's axis
    # with K = k1/(1 + delta), so x_out = dx (1 - cos(sqrt(K) L)) and
    # px_out = (1 + delta) dx sqrt(K) sin(sqrt(K) L) depend on delta:
    #   R16 = -dx L sqrt(k1) sin(phi) / 2,
    #   R26 = dx sqrt(k1) (sin(phi) - phi cos(phi)) / 2,  phi = sqrt(k1) L
    href = ReferenceParticle(PROTON, 800.0, 44.7044096)
    lat = Lattice()
    lat.add(Quadrupole(name="q", length=300.0, gradient=2.0, dx=1.5))
    w = href.w_kin
    rl = compile_ring(lat, RingReference.from_kinetic("proton", w), lattice_energy=w)
    J = _element_jacobian(rl, 0)
    k1 = 2.0 / rl.brho_norm
    L, dx = 0.3, 1.5e-3
    phi = np.sqrt(k1) * L
    assert J[0, 5] == pytest.approx(-dx * L * np.sqrt(k1) * np.sin(phi) / 2, rel=1e-12)
    assert J[1, 5] == pytest.approx(dx * np.sqrt(k1) * (np.sin(phi) - phi * np.cos(phi)) / 2,
                                    rel=1e-12)


def test_whole_turn_matches_helix_to_the_finite_difference_floor():
    from linac_gen.ring.optics import R_STEPS, one_turn_matrix
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    lat, meta = parse_madx(str(FIXTURE))
    href = meta["reference"]
    for e in lat.elements:
        if isinstance(e, RFGap):
            e.voltage = 0.0
    w = href.w_kin
    rl = compile_ring(lat, RingReference.from_kinetic("proton", w), lattice_energy=w,
                      bend_steps=28).with_rf(False)
    T = _to_helix(rl, href.frequency)
    Mr = T @ one_turn_matrix(rl, np.zeros(6), steps=10 * R_STEPS) @ np.linalg.inv(T)
    Mh = np.asarray(compute_transfer_matrix(copy.deepcopy(lat), href.copy()))
    err = (np.abs(Mr - Mh) / np.maximum(np.abs(Mh), 1e-3)).max()
    assert err < 2e-8, err
