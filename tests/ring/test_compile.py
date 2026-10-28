"""The ring compiler: what it emits for each HELIX element, how it normalizes
strengths and locks cavities to the revolution frequency, and what it refuses
or warns about."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.foil import Foil
from linac_gen.elements.lattice_commands import SetBeamEnergy
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.steerer import Steerer
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring.compile import (
    RingCompileError,
    compile_from_config,
    compile_ring,
    lattice_beam,
    lattice_rigidity,
)
from linac_gen.ring.config import RingConfig
from linac_gen.ring.kernels import cartesian as cart
from linac_gen.ring.kernels import nodes as N
from linac_gen.ring.reference import RingReference

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"
C = 299792458.0


@pytest.fixture(scope="module")
def booster():
    lat, meta = parse_madx(str(FIXTURE))
    return lat, meta["reference"].w_kin


def _ref(w=800.0, constants="helix"):
    return RingReference.from_kinetic("proton", w, constants=constants)


def _lat(*elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    return lat


def test_booster_compiles_with_the_deck_length_and_one_node_per_element_end(booster):
    lat, w = booster
    rl = compile_ring(lat, _ref(w), lattice_energy=w)
    assert rl.length == pytest.approx(474.202752, abs=1e-9)
    assert rl.node_length.sum() == pytest.approx(rl.length, rel=1e-15)
    assert len(rl.element_names) == len(lat.elements)
    assert np.all(rl.element_last_node >= 0)
    assert np.all(np.diff(rl.element_last_node) >= 0)
    assert rl.harmonic == 84 and len(rl.cavities) == 22 and not rl.warnings


def test_strengths_are_the_decks_normalized_ones_for_any_constants(booster):
    # a quadrupole's gradient is normalized with the rigidity the importer used
    # (HELIX constants, the deck's beam): the deck's k1 comes back exactly, also
    # when the ring reference uses other constants
    lat, w = booster
    for prof in ("helix", "codata2010", "codata2022"):
        rl = compile_ring(lat, _ref(w, prof), lattice_energy=w)
        j = rl.element_last_node[[i for i, n in enumerate(rl.element_names) if n == "qsxx"][0]]
        assert rl.kinds[j] == N.N_MAGNET
        k1 = rl.params[rl.offsets[j] + cart.MG_K1_DRIFT]
        assert k1 == pytest.approx(0.02025411608767982777, rel=2e-16), prof


def test_cavities_lock_to_the_harmonic_of_the_revolution_frequency(booster):
    lat, w = booster
    for prof in ("helix", "codata2022"):
        ref = _ref(w, prof)
        rl = compile_ring(lat, ref, lattice_energy=w)
        f = 84 * ref.beta0 * C / rl.length
        assert all(c["frequency_hz"] == pytest.approx(f, rel=1e-15) for c in rl.cavities)
        assert rl.params[rl.offsets[rl.cavities[0]["node"]] + 1] == pytest.approx(f, rel=1e-15)
    # without a harmonic number a cavity keeps its own frequency
    g = RFGap(name="rf", voltage=0.01, phase=-90.0, frequency=44.7)
    rl = compile_ring(_lat(Drift("d", 1000.0), g), _ref())
    assert rl.cavities[0]["frequency_hz"] == pytest.approx(44.7e6, rel=1e-15)
    assert rl.cavities[0]["lag_deg"] == pytest.approx(0.0, abs=1e-15)   # HELIX -90 = lag 0
    # ... unless the caller locks it
    rl = compile_ring(_lat(Drift("d", 1000.0), g), _ref(), harmonic=3)
    assert rl.cavities[0]["frequency_hz"] == pytest.approx(3 * _ref().beta0 * C / 1.0, rel=1e-15)


def test_curvilinear_model_rebuilds_the_thick_sextupoles(booster):
    lat, w = booster
    rl = compile_ring(lat, _ref(w), lattice_energy=w, model="curvilinear")
    names = rl.element_names
    i = names.index("sxsxx_din")
    assert names[i + 1] == "sxsxx" and names[i + 2] == "sxsxx_dout"
    nodes = [j for j in range(rl.n_nodes) if rl.node_element[j] in (i, i + 1, i + 2)]
    assert len(nodes) == 1 and rl.kinds[nodes[0]] == N.N_CURVILINEAR
    assert rl.node_length[nodes[0]] == pytest.approx(0.024, abs=1e-15)
    assert rl.element_last_node[i] == rl.element_last_node[i + 2] == nodes[0]
    assert rl.length == pytest.approx(474.202752, abs=1e-9)
    # (the cartesian model keeps the thin kick between its half drifts: the same map)
    rl2 = compile_ring(lat, _ref(w), lattice_energy=w)
    assert [rl2.kinds[j] for j in range(rl2.n_nodes) if rl2.node_element[j] in (i, i + 1, i + 2)] \
        == [N.N_DRIFT, N.N_MULTIPOLE, N.N_DRIFT]


def test_pole_faces_follow_the_edge_model(booster):
    lat, w = booster
    count = lambda rl, k: int(np.sum(rl.kinds == k))      # noqa: E731
    exact = compile_ring(lat, _ref(w), lattice_energy=w)
    assert count(exact, N.N_EDGE_EXACT) == 192 and count(exact, N.N_EDGE_LINEAR) == 0
    lin = compile_ring(lat, _ref(w), lattice_energy=w, edge_model="linear")
    assert count(lin, N.N_EDGE_LINEAR) == 192 and count(lin, N.N_EDGE_EXACT) == 0
    full = compile_ring(lat, _ref(w), lattice_energy=w, edge_model="full")
    j = int(np.flatnonzero(full.kinds == N.N_EDGE_EXACT)[0])
    assert full.params[full.offsets[j] + N.EE_MULT_FRINGE] == 1.0
    assert full.params[full.offsets[j] + N.EE_QUAD_WEDGE] == 1.0       # (full model)
    j = int(np.flatnonzero(exact.kinds == N.N_EDGE_EXACT)[0])
    assert exact.params[exact.offsets[j] + N.EE_QUAD_WEDGE] == 2.0      # (exact model)


def test_a_bend_without_pole_faces_still_has_its_fringe():
    # (the hard-edge dipole fringe is there at a face angle of zero; a bend
    # with no EDGE next to it gets zero-angle faces of its own)
    b = Dipole(name="b", angle=10.0, rho=5000.0, field_index=0.3)
    rl = compile_ring(_lat(Drift("d", 1000.0), b, Drift("d2", 1000.0)), _ref())
    js = [j for j in range(rl.n_nodes) if rl.element_names[rl.node_element[j]] == "b"]
    kinds = [int(rl.kinds[j]) for j in js]
    assert kinds == [N.N_EDGE_EXACT, N.N_MAGNET, N.N_EDGE_EXACT]
    assert rl.params[rl.offsets[js[0]] + N.EE_ANGLE] == 0.0
    lin = compile_ring(_lat(Drift("d", 1000.0), b, Drift("d2", 1000.0)), _ref(), edge_model="linear")
    assert [int(lin.kinds[j]) for j in range(lin.n_nodes)
            if lin.element_names[lin.node_element[j]] == "b"] == [N.N_MAGNET]


def test_vertical_bend_is_a_rotated_horizontal_one():
    b = Dipole(name="b", angle=10.0, rho=5000.0, hv=1)
    rl = compile_ring(_lat(b), _ref(), edge_model="linear")
    assert [int(k) for k in rl.kinds] == [N.N_SROTATION, N.N_MAGNET, N.N_SROTATION]
    assert list(rl.params[rl.offsets[0]:rl.offsets[0] + 2]) == [1.0, 0.0]
    assert list(rl.params[rl.offsets[2]:rl.offsets[2] + 2]) == [-1.0, 0.0]


def test_misalignments_wrap_the_element():
    q = Quadrupole(name="q", length=200.0, gradient=3.0, dx=1.0, dy=-2.0, tilt_deg=30.0)
    rl = compile_ring(_lat(q), _ref())
    assert [int(k) for k in rl.kinds] == [N.N_XYSHIFT, N.N_SROTATION, N.N_MAGNET,
                                          N.N_SROTATION, N.N_XYSHIFT]
    assert list(rl.params[rl.offsets[0]:rl.offsets[0] + 2]) == [1e-3, -2e-3]
    s, c = rl.params[rl.offsets[1]:rl.offsets[1] + 2]
    assert s == pytest.approx(0.5, abs=1e-15) and c == pytest.approx(math.sqrt(3) / 2, abs=1e-15)


def test_unmodelled_degrees_of_freedom_are_reported():
    q = Quadrupole(name="q", length=200.0, gradient=3.0, pitch_deg=0.1)
    rl = compile_ring(_lat(q), _ref())
    assert any("dz/pitch/yaw" in w and "q" in w for w in rl.warnings)


@pytest.mark.parametrize("element, match", [
    (Solenoid(name="sol", length=100.0, field=0.1), "solenoids"),
    (Steerer(name="es", bx_l=1.0, elec=True), "electrostatic steerers"),
    (Dipole(name="rolled", angle=5.0, rho=5000.0, tilt_deg=0.5), "rolled bend"),
    (MatrixElement(name="m", matrix=np.eye(6) + np.eye(6, k=5) * 0.1), "longitudinal plane"),
    (MatrixElement(name="m2", matrix=np.diag([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])), "not symplectic"),
])
def test_elements_ring_mode_cannot_track_are_refused(element, match):
    with pytest.raises(RingCompileError, match=match):
        compile_ring(_lat(Drift("d", 1000.0), element), _ref())


def test_a_foil_is_a_marker_where_the_foil_acts():
    # (the foil itself acts on the host: linac_gen.ring.injection)
    rl = compile_ring(_lat(Drift("d", 1000.0), Foil(name="foil")), _ref())
    j = rl.element_last_node[1]
    assert rl.element_types[1] == "Foil" and rl.kinds[j] == N.N_MARKER
    assert rl.length == pytest.approx(1.0)


def test_passive_cards_are_skipped_with_a_note():
    rl = compile_ring(_lat(Drift("d", 1000.0), SetBeamEnergy(name="e", energy_MeV=900.0),
                           SpaceChargeComp(name="scc", factor=0.5)), _ref())
    assert any("SetBeamEnergy" in w and "SpaceChargeComp" in w for w in rl.warnings)
    assert rl.length == pytest.approx(1.0)


def test_lattice_beam_and_the_species_note():
    lat = _lat(Drift("d", 1000.0), Quadrupole(name="q", length=100.0, gradient=2.0))
    with pytest.raises(ValueError, match="no reference beam"):
        lattice_beam(lat)
    lat.ring_reference_beam = {"species": "proton", "energy": 800.0, "frequency": 44.7}
    assert lattice_beam(lat) == ("proton", 800.0)
    rl = compile_from_config(lat, RingConfig(design_energy_mev=1000.0))
    assert rl.reference.kinetic0 == pytest.approx(1000.0, rel=1e-14)
    # the magnets keep the normalized strengths of the lattice's beam
    j = rl.element_last_node[1]
    assert rl.params[rl.offsets[j] + cart.MG_K1_DRIFT] == pytest.approx(
        2.0 / lattice_rigidity("proton", 800.0), rel=1e-15)
    rl_h = compile_ring(lat, _ref(), lattice_species="H-", lattice_energy=800.0)
    assert any("scaled for H-" in w for w in rl_h.warnings)
    assert lattice_rigidity("H-", 800.0) < 0.0


def test_model_options_are_validated():
    lat = _lat(Drift("d", 1000.0))
    with pytest.raises(ValueError, match="curvilinear model has linear pole faces only"):
        compile_ring(lat, _ref(), model="curvilinear", edge_model="exact")
    with pytest.raises(ValueError, match="symp2, symp4 or symp6"):
        compile_ring(lat, _ref(), model="curvilinear", integrator="yoshida4")
    with pytest.raises(ValueError, match="edge model"):
        compile_ring(lat, _ref(), edge_model="soft")
    with pytest.raises(ValueError, match="model"):
        compile_ring(lat, _ref(), model="spherical")
    with pytest.raises(ValueError, match="model"):
        RingConfig(model="spherical")
    with pytest.raises(ValueError, match="curvilinear model integrates"):
        RingConfig(model="curvilinear", integrator="teapot")


def test_model_names_of_earlier_versions_are_read():
    # (projects saved before the models were named by their physics)
    from linac_gen.ring.config import MODEL_ALIASES
    lat = _lat(Drift("d", 1000.0))
    assert sorted(MODEL_ALIASES.values()) == ["cartesian", "curvilinear"]
    for old, new in MODEL_ALIASES.items():
        assert RingConfig(model=old).model == new
        assert compile_ring(lat, _ref(), model=old).options["model"] == new
    # (a malformed value is refused as such, not with a TypeError)
    with pytest.raises(ValueError, match="model"):
        RingConfig(model=["cartesian"])
    with pytest.raises(ValueError, match="model"):
        compile_ring(lat, _ref(), model=["cartesian"])


def test_multipole_offset_and_tilt_are_its_own_frame():
    # HELIX's Multipole evaluates its kick at (x - dx, y - dy) in a frame
    # rolled by tilt: the same as shift + roll around the thin kick
    m = Multipole(name="m", knl=[0.0, 0.0, 2.0], dx=1.0, tilt_deg=20.0)
    rl = compile_ring(_lat(m, Drift("d", 1000.0)), _ref())
    assert [int(k) for k in rl.kinds] == [N.N_XYSHIFT, N.N_SROTATION, N.N_MULTIPOLE,
                                          N.N_SROTATION, N.N_XYSHIFT, N.N_DRIFT]
    # (a lattice with no length is refused)
    with pytest.raises(RingCompileError, match="no length"):
        compile_ring(_lat(m), _ref())


@pytest.mark.parametrize("edge_model", ["exact", "linear"])
def test_a_bend_with_no_field_has_no_pole_face_kicks(edge_model):
    # (a bend switched off, field error -100 %: no field at its faces either,
    # as Dipole.transfer_matrix has it)
    lat = Lattice()
    lat.add(Drift(name="d", length=1000.0))
    lat.add(Dipole(name="b", angle=4.05, rho=40847.086, e1=2.0, e2=2.0, field_rel=-1.0))
    rl = compile_ring(lat, RingReference.from_kinetic("proton", 800.0), lattice_energy=800.0,
                      edge_model=edge_model)
    kinds = [int(rl.kinds[j]) for j in range(rl.n_nodes) if rl.node_element[j] == 1]
    assert N.N_MAGNET in kinds
    assert N.N_EDGE_LINEAR not in kinds and N.N_EDGE_EXACT not in kinds, kinds
    # (with its field, the same bend has its two faces)
    lat.elements[1].field_rel = 0.0
    rl = compile_ring(lat, RingReference.from_kinetic("proton", 800.0), lattice_energy=800.0,
                      edge_model=edge_model)
    kinds = [int(rl.kinds[j]) for j in range(rl.n_nodes) if rl.node_element[j] == 1]
    assert sum(k in (N.N_EDGE_LINEAR, N.N_EDGE_EXACT) for k in kinds) == 2


# ---------------------------------------------------------------------------
# the deck's own beam, apertures of moved elements, a cavity's transit-time factor
def _track_once(rl, X):
    """Particle state after one pass through ``rl`` (1 alive, 0 lost)."""
    from linac_gen.ring.kernels.cpu import backend
    X = np.array(X, dtype=float, copy=True)
    n = X.shape[1]
    ptau, rpp, rvv = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        ptau[i], rpp[i], rvv[i] = cart.energy_from_delta(X[5, i], rl.reference.beta0)
    state = np.ones(n, dtype=np.int8)
    lt = np.full(n, -1, dtype=np.int32)
    ln = np.full(n, -1, dtype=np.int32)
    lc = np.zeros(n, dtype=np.int8)
    backend().track_turns(X, ptau, rpp, rvv, state, lt, ln, lc, *rl.node_table(), 0,
                          rl.n_nodes, 1, 0, 0.0)
    return state


def _two_particles(x1_mm, x2_mm):
    X = np.zeros((6, 2))
    X[0] = [x1_mm * 1e-3, x2_mm * 1e-3]
    return X


def test_a_deck_without_its_beam_runs_at_the_beam_given():
    from linac_gen.core.config import BeamConfig
    from linac_gen.ring.compile import declared_beam
    lat = _lat(Drift("d", 1000.0), Quadrupole(name="q", length=100.0, gradient=2.0))
    # (an importer's default: the magnets were scaled with it, nothing more)
    lat.ring_reference_beam = {"species": "H-", "energy": 1000.0, "frequency": 352.21,
                               "declared": False}
    assert declared_beam(lat) is None and lattice_beam(lat) == ("H-", 1000.0)
    rl = compile_from_config(lat, RingConfig(), beam=BeamConfig(species="proton", energy=800.0))
    assert rl.reference.kinetic0 == pytest.approx(800.0, rel=1e-14)
    # the magnets keep the deck's normalized strengths (field / the importer's rigidity)
    j = rl.element_last_node[1]
    assert rl.params[rl.offsets[j] + cart.MG_K1_DRIFT] == pytest.approx(
        2.0 / lattice_rigidity("H-", 1000.0), rel=1e-15)
    assert not any("scaled for" in w for w in rl.warnings)
    with pytest.raises(ValueError, match="states no beam"):
        compile_from_config(lat, RingConfig())
    # a declared beam (a missing flag reads as declared) sets the ring's energy
    lat.ring_reference_beam = dict(lat.ring_reference_beam, declared=True)
    assert declared_beam(lat) is lat.ring_reference_beam
    rl = compile_from_config(lat, RingConfig(), beam=BeamConfig(species="proton", energy=800.0))
    assert rl.reference.kinetic0 == pytest.approx(1000.0, rel=1e-14)
    assert any("scaled for H-" in w for w in rl.warnings)
    del lat.ring_reference_beam["declared"]
    assert declared_beam(lat) is lat.ring_reference_beam


def test_a_moved_quadrupole_and_dipole_check_their_aperture_in_their_frame():
    pytest.importorskip("numba")
    # a 10 mm pipe moved by +5 mm with its magnet: +12 mm is 7 mm from its
    # axis (inside), -6 mm is 11 mm (outside) — as the linac checks it
    q = Quadrupole(name="q", length=100.0, gradient=0.01, dx=5.0, aperture=10.0)
    state = _track_once(compile_ring(_lat(q, Drift("d", 100.0)), _ref()),
                        _two_particles(12.0, -6.0))
    assert list(state) == [1, 0]
    b = Dipole(name="b", angle=0.001, rho=1.0e7, dx=5.0, aperture=10.0)
    state = _track_once(compile_ring(_lat(b, Drift("d", 100.0)), _ref(), lattice_energy=800.0),
                        _two_particles(12.0, -6.0))
    assert list(state) == [1, 0]
    # (in place, the same pipe keeps -6 mm and stops +12 mm)
    q0 = Quadrupole(name="q", length=100.0, gradient=0.01, aperture=10.0)
    state = _track_once(compile_ring(_lat(q0, Drift("d", 100.0)), _ref()),
                        _two_particles(12.0, -6.0))
    assert list(state) == [0, 1]


def test_a_moved_drift_moves_its_pipe_only():
    pytest.importorskip("numba")
    d = Drift(name="d", length=100.0, aperture=10.0)
    d.dx = 5.0
    rl = compile_ring(_lat(d), _ref())
    # (the pipe's centre in its aperture block; no shift nodes: the drift
    # itself does not see the offset)
    assert [int(k) for k in rl.kinds] == [N.N_DRIFT, N.N_APERTURE]
    assert list(_track_once(rl, _two_particles(12.0, -6.0))) == [1, 0]
    d.dx = 0.0
    assert list(_track_once(compile_ring(_lat(d), _ref()), _two_particles(12.0, -6.0))) == [0, 1]


def test_a_cavity_with_transit_time_factor_zero_has_no_voltage():
    g = RFGap(name="rf", voltage=0.1, phase=-90.0, frequency=44.7)
    g.ttf = 0.0
    rl = compile_ring(_lat(Drift("d", 1000.0), g), _ref())
    assert rl.cavities[0]["voltage_mv"] == 0.0
    g.ttf = 0.5
    rl = compile_ring(_lat(Drift("d", 1000.0), g), _ref())
    assert rl.cavities[0]["voltage_mv"] == pytest.approx(0.05, rel=1e-15)
