"""MAD-X keeps one reference momentum p0; HELIX's reference follows the energy.  A linearised MATRIX is written about
HELIX's reference orbit in MAD-X's coordinates (pt = (W - W0)/(p0 c); t restarting at each MATRIX exit, growing by
L (1/beta0 - 1/beta) over the length between), and the importer reads it about the same orbit: the pt at the exit is
the reference's gain (MatrixElement.ref_gain), the rest an offset of every particle.  So a line comes back with its
energies and its maps, and MAD-X sees HELIX's particles: its TRACK of the reference ends on the orbit."""
import copy
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.elements.drift import Drift
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.io.madx_parser import parse_madx
from linac_gen.io.madx_writer import write_madx
from linac_gen.tracking.longitudinal_coords import madx_transform
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, reference_at
from linac_gen.tracking.tracker import Tracker
from tests.dataguard import needs
from tests.io.test_madx_linearize import _ref, _rf_map
from tests.tracking.test_envelope_reference_matches_tracker import _ncells

W0 = 20.0


def _slice():
    """Two multi-cell cavities (on a clock of their own) and a box RF map between drifts."""
    lat = Lattice()
    for e in (Drift("d0", 100.0), _ncells(), Drift("d1", 150.0), _rf_map("CAV1"), Drift("d2", 120.0), _ncells(),
              Drift("d3", 80.0)):
        lat.add(e)
    return lat


def _round_trip(tmp_path, lat, ref):
    out = tmp_path / "line.madx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_madx(lat, out, ref, linearize=True)
        back, meta = parse_madx(str(out))
    return out, back, copy.deepcopy(meta["reference"])


def _energies(lat, ref):
    """The reference energy at every boundary, one value per plateau (the cards that MAD-X carries as comments
    come back as none, so the boundaries of the two lines are not one for one)."""
    w = [reference_at(lat, ref, k).w_kin for k in range(len(lat.elements) + 1)]
    return [x for j, x in enumerate(w) if j == 0 or abs(x - w[j - 1]) > 1e-12]


def test_a_linac_slice_comes_back_with_its_energies_and_its_maps(tmp_path):
    """Energies at every boundary, and the walk on the importer's one clock is the original's, D_out M D_in^-1
    (the gain was a ΔW offset on every particle, which left the reference, and so the line after it, behind)."""
    ref = _ref(w=W0, f=352.21)
    lat = _slice()
    _out, back, ref_back = _round_trip(tmp_path, lat, ref)
    assert [type(e).__name__ for e in back.elements].count("MatrixElement") == 3
    np.testing.assert_allclose(_energies(back, ref_back), _energies(lat, ref), rtol=0.0, atol=1e-9)
    gains = [e.ref_gain for e in back.elements if isinstance(e, MatrixElement)]
    assert all(g != 0.0 for g in gains) and all(e.offset is None for e in back.elements
                                                 if isinstance(e, MatrixElement))
    exit_orig, exit_back = reference_at(lat, ref, len(lat.elements)), reference_at(back, ref_back, len(back.elements))
    D_in, D_out = np.eye(6), np.eye(6)
    D_in[4, 4] = ref_back.frequency / ref.frequency
    D_out[4, 4] = exit_back.frequency / exit_orig.frequency
    want = D_out @ compute_transfer_matrix(lat, ref.copy()) @ np.linalg.inv(D_in)
    np.testing.assert_allclose(compute_transfer_matrix(back, ref_back.copy()), want, rtol=1e-12, atol=1e-12)


def test_an_rf_gap_before_the_maps_moves_the_orbit(tmp_path):
    """An RF gap written as an RFCAVITY gains energy for MAD-X too: the orbit the maps after it are written about
    starts from that energy, and the line comes back with its energies and maps."""
    from linac_gen.elements.rf_gap import RFGap
    ref = _ref(w=W0, f=352.21)
    lat = Lattice()
    for e in (Drift("d0", 60.0), RFGap("G1", voltage=0.8, phase=-30.0, frequency=352.21), Drift("d1", 140.0),
              _rf_map("CAV1"), Drift("d2", 90.0)):
        lat.add(e)
    _out, back, ref_back = _round_trip(tmp_path, lat, ref)
    np.testing.assert_allclose(_energies(back, ref_back), _energies(lat, ref), rtol=0.0, atol=1e-9)
    assert all(e.offset is None for e in back.elements if isinstance(e, MatrixElement))
    np.testing.assert_allclose(compute_transfer_matrix(back, ref_back.copy()), compute_transfer_matrix(lat, ref.copy()),
                               rtol=1e-12, atol=1e-12)


def test_a_map_that_gains_nothing_comes_back_with_no_gain(tmp_path):
    """A map with a phase-to-energy term but no gain, after a gain and drifts: its orbit cancels (to round-off at
    most), so it comes back with no gain and no offset."""
    ref = _ref(w=W0, f=352.21)
    M = np.eye(6)
    M[5, 4] = 0.01
    lat = Lattice()
    # (several drifts: the writer counts the orbit's t over their positions, the importer over their lengths)
    for e in (MatrixElement("g", np.eye(6), length=80.0, ref_gain=0.4), Drift("d", 33.3), Drift("e", 41.7),
              Drift("f", 25.1), Drift("h", 17.9), MatrixElement("n", M, length=50.0), Drift("d2", 40.0)):
        lat.add(e)
    _out, back, _ref_back = _round_trip(tmp_path, lat, ref)
    g, n = [e for e in back.elements if isinstance(e, MatrixElement)]
    assert g.ref_gain == pytest.approx(0.4, rel=1e-12)
    assert n.ref_gain == pytest.approx(0.0, abs=1e-12) and n.offset is None


def test_the_read_back_line_is_the_same_in_every_mode(tmp_path):
    """The multiparticle Jacobian and the envelope's sigma of the line read back are its walk's."""
    from linac_gen.tracking.envelope import EnvelopeSolver
    ref = _ref(w=W0, f=352.21)
    _out, back, ref_back = _round_trip(tmp_path, _slice(), ref)
    M = compute_transfer_matrix(back, ref_back.copy())
    h = np.array([1e-4, 1e-5, 1e-4, 1e-5, 1e-3, 1e-6])
    beam = Beam(ref_back.copy(), n_particles=12, current=0.0)
    beam.particles[:] = 0.0
    for k in range(6):
        beam.particles[2 * k, k], beam.particles[2 * k + 1, k] = h[k], -h[k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(copy.deepcopy(back), beam).run()
    J = np.column_stack([(beam.particles[2 * k] - beam.particles[2 * k + 1]) / (2 * h[k]) for k in range(6)])
    np.testing.assert_allclose(J, M, rtol=1e-9, atol=1e-9)
    init = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
                emit_z=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = EnvelopeSolver(copy.deepcopy(back), ref_back.copy(), init, current=0.0).run()
    s0 = np.asarray(res.sigma_matrix[0])
    np.testing.assert_allclose(np.asarray(res.sigma_matrix[-1]), M @ s0 @ M.T, rtol=1e-11, atol=1e-13)


def test_a_reference_gain_moves_the_reference_in_every_walk():
    from linac_gen.tracking.backtrack import backtrack_distribution, build_replay_table
    from linac_gen.tracking.envelope import EnvelopeSolver
    ref = _ref(w=W0, f=352.21)
    M = np.eye(6)
    M[1, 1] = M[3, 3] = 0.98                       # (an energy gain damps the angles)
    lat = Lattice()
    for e in (Drift("d0", 50.0), MatrixElement("g", M, length=80.0, ref_gain=0.4), Drift("d1", 50.0)):
        lat.add(e)
    assert reference_at(lat, ref, 3).w_kin == pytest.approx(W0 + 0.4, abs=1e-12)
    beam = Beam(ref.copy(), n_particles=8, current=0.0)
    beam.particles[:] = np.random.default_rng(1).normal(size=(8, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
    start = beam.particles.copy()
    init = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
                emit_z=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(copy.deepcopy(lat), beam).run()
        assert beam.ref.w_kin == pytest.approx(W0 + 0.4, abs=1e-12)
        env = EnvelopeSolver(copy.deepcopy(lat), ref.copy(), init, current=0.0).run()
        assert env.ref_w_kin[-1] == pytest.approx(W0 + 0.4, abs=1e-12)
        table = build_replay_table(copy.deepcopy(lat), ref.copy(), end=2)
        assert table[-1].w_kin == pytest.approx(W0 + 0.4, abs=1e-12)
        backtrack_distribution(copy.deepcopy(lat), beam, ref.copy())
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-10)


# --------------------------------------------------------------------- MAD-X itself
def _cpymad():
    return pytest.importorskip("cpymad.madx")


def _track(path, starts):
    """MAD-X TRACK of particles (rows: x, px, y, py, t, pt) through the file; their coordinates at the end."""
    cpymad = _cpymad()
    m = cpymad.Madx(stdout=False, cwd=str(path.parent))
    m.call(str(path))
    m.input("track, onepass, onetable;"
            + "".join(f"start, x={p[0]!r}, px={p[1]!r}, y={p[2]!r}, py={p[3]!r}, t={p[4]!r}, pt={p[5]!r};"
                      for p in starts)
            + "run, turns=1; endtrack;")
    t = m.table.trackone
    cols = {c: np.array(t[c]) for c in ("number", "s", "x", "px", "y", "py", "t", "pt")}
    m.quit()
    end_s = cols["s"].max()
    out = np.zeros((len(starts), 6))
    for j in np.where(np.isclose(cols["s"], end_s))[0]:
        out[int(cols["number"][j]) - 1] = [cols[c][j] for c in ("x", "px", "y", "py", "t", "pt")]
    return out


def _madx_jacobian(path, origin, h):
    starts = [origin]
    for k in range(6):
        for sign in (1.0, -1.0):
            p = np.array(origin, dtype=float)
            p[k] += sign * h[k]
            starts.append(p)
    end = _track(path, starts)
    return end[0], np.column_stack([(end[1 + 2 * k] - end[2 + 2 * k]) / (2 * h[k]) for k in range(6)])


def test_madx_tracks_the_reference_on_the_orbit_and_the_particles_by_the_maps(tmp_path):
    """MAD-X TRACK of HELIX's export: the reference ends on the orbit (pt the gain, t that of the last drift), and the
    Jacobian about it is HELIX's walk in MAD-X's coordinates."""
    ref = _ref(w=W0, f=352.21)
    lat = _slice()
    out, back, ref_back = _round_trip(tmp_path, lat, ref)
    p0c = ref.bg * PROTON.mass
    end_ref = reference_at(back, ref_back, len(back.elements))
    pt_end = (end_ref.w_kin - W0) / p0c
    t_end = 0.080 * (1.0 / ref.beta - 1.0 / end_ref.beta)            # d3, after the last MATRIX
    h = np.array([1e-5, 1e-6, 1e-5, 1e-6, 1e-5, 1e-6])
    orbit_end, J = _madx_jacobian(out, np.zeros(6), h)
    np.testing.assert_allclose(orbit_end[4:], [t_end, pt_end], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(orbit_end[:4], 0.0, atol=1e-12)
    T_in = madx_transform(ref_back)
    T_out = madx_transform(ref_back, p_local_over_p0=end_ref.bg / ref_back.bg,
                           wavelength_local_mm=end_ref.wavelength)
    want = np.linalg.inv(T_out) @ compute_transfer_matrix(back, ref_back.copy()) @ T_in
    scale = np.maximum(np.abs(want), 1e-3)
    assert np.max(np.abs(J - want) / scale) < 1e-6


def test_a_madx_deck_of_two_kicked_maps_reads_as_madx_tracks_it(tmp_path):
    """A deck HELIX did not write: two MATRIX elements with energy kicks and a t-kick.  The line HELIX reads has the
    energies MAD-X's reference orbit has, and its walk is MAD-X TRACK's Jacobian about that orbit."""
    energy_gev = (W0 + PROTON.mass) / 1000.0
    deck = tmp_path / "two.madx"
    deck.write_text(
        f"beam, particle=proton, energy={energy_gev!r};\n"
        "a: matrix, l=0.3, kick6=0.004, rm12=0.3, rm34=0.3, rm56=0.2, rm65=-0.05;\n"
        "d: drift, l=0.4;\n"
        "b: matrix, l=0.3, kick5=0.0002, kick6=0.003, rm12=0.3, rm34=0.3, rm56=0.2, rm65=-0.05;\n"
        "line_t: line=(a, d, b);\nuse, period=line_t;\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_madx(str(deck))
    ref0 = copy.deepcopy(meta["reference"])
    p0c = ref0.bg * PROTON.mass
    h = np.array([1e-5, 1e-6, 1e-5, 1e-6, 1e-5, 1e-6])
    orbit_end, J = _madx_jacobian(deck, np.zeros(6), h)
    end_ref = reference_at(lat, ref0, len(lat.elements))
    assert end_ref.w_kin == pytest.approx(W0 + orbit_end[5] * p0c, abs=1e-9)
    T_in = madx_transform(ref0)
    T_out = madx_transform(ref0, p_local_over_p0=end_ref.bg / ref0.bg, wavelength_local_mm=end_ref.wavelength)
    want = np.linalg.inv(T_out) @ compute_transfer_matrix(lat, ref0.copy()) @ T_in
    scale = np.maximum(np.abs(want), 1e-3)
    assert np.max(np.abs(J - want) / scale) < 1e-6


def test_madx_twiss_follows_the_orbit_of_the_export(tmp_path):
    """MAD-X TWISS of HELIX's export follows the orbit: its pt at the end is the energy the line gains, over p0 c,
    to 1e-4 (TWISS expands the drifts about its orbit to second order, which the maps' phase-to-energy terms carry
    into pt; TRACK, above, is exact)."""
    cpymad = _cpymad()
    ref = _ref(w=W0, f=352.21)
    out, back, ref_back = _round_trip(tmp_path, _slice(), ref)
    m = cpymad.Madx(stdout=False, cwd=str(tmp_path))
    m.call(str(out))
    tw = m.twiss(betx=1.0, bety=1.0, alfx=0.0, alfy=0.0)
    pt_end = float(tw.pt[-1])
    m.quit()
    p0c = ref.bg * PROTON.mass
    assert pt_end == pytest.approx((reference_at(back, ref_back, len(back.elements)).w_kin - W0) / p0c, rel=1e-3)


def test_what_cannot_carry_a_reference_gain_says_so(tmp_path):
    """Ring mode refuses a matrix element that moves the reference (it had refused one only through its energy
    offset); the TraceWin writer, which has no card for it, warns that the energies after it change; a curved
    cluster does not take it as energy-neutral."""
    from linac_gen.elements.curved_field_map import _energy_neutral
    from linac_gen.io.tracewin_writer import write_tracewin
    from linac_gen.ring.compile import RingCompileError
    lat = Lattice()
    lat.add(Drift("d0", 50.0))
    lat.add(MatrixElement("g", np.eye(6), length=80.0, ref_gain=0.4))
    lat.add(Drift("d1", 50.0))
    with pytest.warns(UserWarning, match="moves the reference energy by 0.4 MeV"):
        write_tracewin(lat, str(tmp_path / "g.dat"))
    assert not _energy_neutral(lat.elements[1]) and _energy_neutral(MatrixElement("n", np.eye(6)))
    from linac_gen.ring.compile import RingReference, compile_ring
    with pytest.raises(RingCompileError, match="changes the reference energy"):
        compile_ring(lat, RingReference.from_kinetic("proton", 800.0, constants="helix"))


@needs("examples/mebt_plus_hwr_plus_ssr1.dat", "Fields")
@pytest.mark.parametrize("n", [30, 60])
def test_a_spoke_cavity_section_of_a_deck_comes_back(tmp_path, n):
    """A slice of a deck's spoke-cavity section (field maps on a 325 MHz clock, through several MeV): energies at
    every boundary and the walk, D_out M D_in^-1 (the gain used to stay out of the reference: the 60-element slice
    read back ended at its entrance energy, 11.4 for 20.5 MeV, R55 0.43 for 0.63)."""
    from linac_gen.cli.common import build_ref, load_input
    from linac_gen.elements.base import FieldMapElement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full, cfg, _ = load_input("examples/mebt_plus_hwr_plus_ssr1.dat")
        ref0 = build_ref(cfg)
        k = next(i for i, e in enumerate(full.elements)
                 if isinstance(e, FieldMapElement) and getattr(e, "frequency", 0.0) == 325.0) - 2
        sl = Lattice()
        for e in copy.deepcopy(full.elements[k:k + n]):
            sl.add(e)
        sl.step_config = getattr(full, "step_config", None)
        r_in = reference_at(full, ref0, k)
        _out, back, ref_back = _round_trip(tmp_path, sl, r_in)
        np.testing.assert_allclose(_energies(back, ref_back), _energies(sl, r_in), rtol=0.0, atol=1e-9)
        exit_sl, exit_back = reference_at(sl, r_in, len(sl.elements)), reference_at(back, ref_back, len(back.elements))
        assert exit_sl.w_kin - r_in.w_kin > 2.0
        D_in, D_out = np.eye(6), np.eye(6)
        D_in[4, 4] = ref_back.frequency / r_in.frequency
        D_out[4, 4] = exit_back.frequency / exit_sl.frequency
        want = D_out @ compute_transfer_matrix(sl, r_in.copy()) @ np.linalg.inv(D_in)
        np.testing.assert_allclose(compute_transfer_matrix(back, ref_back.copy()), want, rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------- review fixes
def test_a_mad8_deck_keeps_its_maps_about_its_own_reference(tmp_path):
    """MAD-8's LCAVITY moves MAD-8's own reference, so its maps are about the local energy: no orbit, no false
    offsets (the MAD-X orbit read this map's place after the cavity as an excursion: 59 mm, -42 deg)."""
    from linac_gen.io.mad8_parser import parse_mad8
    deck = tmp_path / "lcav.mad8"
    deck.write_text("BEAM, PARTICLE=PROTON, ENERGY=1.938272\n"
                    "D1: DRIFT, L=0.5\n"
                    "LC: LCAVITY, L=1.0, DELTAE=200.0, PHI0=0.0, FREQ=650\n"
                    "M1: MATRIX, RM(1,6)=0.5, RM(5,6)=0.2\n"
                    "TOP: LINE=(D1, LC, D1, M1, D1)\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _meta = parse_mad8(str(deck))
    (m,) = [e for e in lat.elements if isinstance(e, MatrixElement)]
    assert m.offset is None and m.ref_gain == 0.0
    # and a kick6 stays an energy offset of every particle, as MAD-8's maps are about its reference
    deck.write_text(deck.read_text(encoding="utf-8").replace("RM(5,6)=0.2\n", "RM(5,6)=0.2, KICK(6)=0.001\n"),
                    encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_mad8(str(deck))
    (m,) = [e for e in lat.elements if isinstance(e, MatrixElement)]
    p0c = meta["reference"].bg * meta["reference"].species.mass
    assert m.ref_gain == 0.0 and m.offset[5] == pytest.approx(0.001 * p0c, rel=1e-12)


def test_a_gap_with_its_own_errors_before_the_maps_comes_back(tmp_path):
    """The RFCAVITY is written as the gap fires (its voltage and phase errors in), which is what the importer replays
    and what the orbit gains: the maps after it come back with no false offset."""
    from linac_gen.elements.rf_gap import RFGap
    ref = _ref(w=W0, f=352.21)
    gap = RFGap("G1", voltage=0.8, phase=-30.0, frequency=352.21)
    gap.phase_offset, gap.voltage_rel = 5.0, 0.02
    lat = Lattice()
    for e in (Drift("d0", 60.0), gap, Drift("d1", 140.0), _rf_map("CAV1"), Drift("d2", 90.0)):
        lat.add(e)
    _out, back, ref_back = _round_trip(tmp_path, lat, ref)
    np.testing.assert_allclose(_energies(back, ref_back), _energies(lat, ref), rtol=0.0, atol=1e-9)
    assert all(e.offset is None for e in back.elements if isinstance(e, MatrixElement))
    np.testing.assert_allclose(compute_transfer_matrix(back, ref_back.copy()), compute_transfer_matrix(lat, ref.copy()),
                               rtol=1e-12, atol=1e-12)


def test_the_envelope_kicks_a_gaining_map_at_its_exit_energy():
    """The tracker's track() moves the reference through the gain before its space-charge kick: so does the
    envelope."""
    from linac_gen.tracking.envelope import EnvelopeSolver
    ref = _ref(w=W0, f=352.21)
    M = np.eye(6) + 0.04 * np.random.default_rng(7).normal(size=(6, 6))
    init = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
                emit_z=0.2)
    gaining, thin = Lattice(), Lattice()
    gaining.add(MatrixElement("g", M, length=120.0, ref_gain=2.0))
    thin.add(MatrixElement("t", M, length=0.0, ref_gain=2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = EnvelopeSolver(thin, ref.copy(), dict(init), current=5.0)
        after_map = np.asarray(solver.run().sigma_matrix[-1])
        assert solver._ref.w_kin == pytest.approx(W0 + 2.0)
        kick = solver._sc_matrix(after_map, ds_mm=120.0)                 # at the exit energy
        got = np.asarray(EnvelopeSolver(gaining, ref.copy(), dict(init), current=5.0).run().sigma_matrix[-1])
    np.testing.assert_allclose(got, kick @ after_map @ kick.T, rtol=1e-12, atol=1e-15)


def test_a_map_that_gains_nothing_after_a_gain_at_low_energy_has_no_gain(tmp_path):
    """At 2.1 MeV the cancellation of the orbit's pt leaves round-off (1e-15 MeV), which every consumer would read as
    a gain (ring mode refusing, a TraceWin warning, not energy-neutral): it is no gain."""
    from linac_gen.elements.curved_field_map import _energy_neutral
    ref = _ref(w=2.1, f=352.21)
    M = np.eye(6)
    M[5, 4] = 0.01
    lat = Lattice()
    for e in (MatrixElement("g", np.eye(6), length=80.0, ref_gain=0.4), Drift("d", 33.3), Drift("e", 41.7),
              MatrixElement("n", M, length=50.0), Drift("d2", 40.0)):
        lat.add(e)
    out = tmp_path / "low.madx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_madx(lat, out, ref, linearize=True)
        back, meta = parse_madx(str(out))
    g, n = [e for e in back.elements if isinstance(e, MatrixElement)]
    assert n.ref_gain == 0.0 and n.offset is None and _energy_neutral(n)
    assert any(w.startswith("1 MATRIX element(s) change the energy") for w in meta["warnings"])


def test_a_plain_export_says_a_gain_is_lost(tmp_path):
    lat = Lattice()
    lat.add(MatrixElement("g", np.eye(6), length=80.0, ref_gain=1.5))
    warns = write_madx(lat, tmp_path / "plain.madx", _ref(w=W0), linearize=False)
    assert any("reference gain of 1.5 MeV is lost" in w for w in warns)


def test_a_kick_that_loses_the_reference_is_named(tmp_path):
    energy_gev = (W0 + PROTON.mass) / 1000.0
    deck = tmp_path / "lost.madx"
    deck.write_text(f"beam, particle=proton, energy={energy_gev!r};\n"
                    "a: matrix, l=0.1, kick6=0.01;\nd1: drift, l=0.2;\nb: matrix, l=0.1, kick6=-0.5;\n"
                    "d2: drift, l=0.2;\nc: matrix, l=0.1, rm12=0.1;\nline_t: line=(a, d1, b, d2, c);\n"
                    "use, period=line_t;\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _lat, meta = parse_madx(str(deck))
    assert any("lost at 'b': its kick6 takes the reference below zero energy" in w for w in meta["warnings"])
