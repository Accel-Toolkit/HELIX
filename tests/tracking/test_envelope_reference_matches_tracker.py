# tests/tracking/test_envelope_reference_matches_tracker.py
"""The envelope, matrix and backtracking modes advance the reference particle
through FieldMap / FieldMap3D / SuperposedFieldMap exactly as a multiparticle
run does.

The multiparticle tracker walks a map in half steps (``2 * n_int`` calls of
``track_rk4(ds / 2)``).  The envelope used to advance its reference with the
map's own full steps (``advance_ref`` / ``advance_ref_over``), the same
midpoint-phase formula at twice the step: through a 650 MHz β = 0.9 cavity
that is ~4e-4 of the energy gain, 290 keV over a linac to 800 MeV.
Now every mode loads the tracker's own reference states
(``tracker.field_map_reference_states``), so energy, phase, ``s`` and
frequency agree bit for bit through a map.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.lattice_commands import Freq, SetBeamE0P0, SetReferenceEnergy
from linac_gen.elements.marker import Marker
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.tracking import tracker as T
from linac_gen.tracking.backtrack import build_replay_table
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import reference_at

INITIAL = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
           "alpha_z": 0.0, "beta_z": 1.0, "emit_x": 0.2, "emit_y": 0.2, "emit_z": 0.2}


# ---------------------------------------------------------------- elements
def _fd3d(L, freq, ez=3.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-20.0, 20.0, 9)
    z = np.linspace(0.0, L, 61)
    shape = (x.size, x.size, z.size)
    fz = ez * np.sin(np.pi * z / L)[None, None, :] + 0.3 * rng.normal(size=shape)
    fd = FieldMapData(z=z, frequency=freq)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=x, y=x, z=z, Fx=0.3 * rng.normal(size=shape),
        Fy=0.3 * rng.normal(size=shape), Fz=fz)
    return fd


def _fm3d(name="FM3", L=240.0, freq=162.5, p_flag=0, kind="kd", **kw):
    el = FieldMap3D(name=name, length=L, field_data=_fd3d(L, freq), scale=1.0,
                    phase=-30.0, frequency=freq, n_steps=200, p_flag=p_flag, **kw)
    el.integrator_kind = kind
    return el


def _rf_1d(L=120.0, ez=2.0):
    z = np.linspace(0.0, L, 121)
    return FieldMapData(z=z, Ez=ez * np.sin(np.pi * z / L), symmetry="1d")


def _sol_1d(L=300.0, bz=0.35):
    z = np.linspace(0.0, L, 151)
    return FieldMapData(z=z, Bz=bz * (0.5 - 0.5 * np.cos(2 * np.pi * z / L)),
                        symmetry="1d")


def _fm1d(name="FM1", p_flag=0, **kw):
    return FieldMap(name, length=120.0, field_data=_rf_1d(), phase=-25.0,
                    frequency=352.21, aperture=30.0, n_steps=125, p_flag=p_flag, **kw)


def _sfm(p_flag=1, markers=False):
    sol = FieldMap("SOL", length=300.0, field_data=_sol_1d(), aperture=30.0, n_steps=150)
    cav = FieldMap("CAV", length=120.0, field_data=_rf_1d(), phase=-25.0,
                   frequency=352.21, aperture=30.0, n_steps=120, p_flag=p_flag)
    mk = ([(120.0, Marker("DIA1", snapshot=False)), (250.0, Marker("DIA2", snapshot=False))]
          if markers else [])
    return SuperposedFieldMap("SUP", [(0.0, sol), (90.0, cav)], interior_markers=mk)


def _ref(element):
    f = 162.5 if getattr(element, "frequency", 352.21) == 162.5 else 352.21
    w = 2.1 if f == 162.5 else 3.0
    return ReferenceParticle(species=PROTON, w_kin=w, frequency=f)


def _lattice(*elements, step=(100.0, 50.0)):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    lat.step_config = StepConfig(*step)
    return lat


def _beam(ref, n=200, current=0.0, seed=2):
    b = Beam(ref=ref.copy(), n_particles=n, current=current)
    rng = np.random.default_rng(seed)
    b.particles[:] = rng.normal(size=(n, 6)) * [0.5, 0.5, 0.5, 0.5, 2.0, 1e-4]
    return b


def _mp(lat, ref, current=0.0):
    beam = _beam(ref, current=current)
    pic = (PicSolver(SpaceChargeConfig(nx=8, ny=8, nz=8, grid_mode="adaptive",
                                       use_gpu="cpu")) if current else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = T.Tracker(lat, beam, pic_solver=pic).run()
    return beam.ref, rec


def _env(lat, ref, current=0.0, record_substeps=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = EnvelopeSolver(lat, ref.copy(), dict(INITIAL), current=current,
                                record_substeps=record_substeps)
        res = solver.run()
    return solver._ref, res


def _state(r):
    return (r.w_kin, r.phi_s, r.s, r.frequency)


ELEMENTS = {
    "FieldMap3D kd": lambda: _fm3d(kind="kd"),
    "FieldMap3D dkd": lambda: _fm3d(kind="dkd"),
    "FieldMap3D SET_SYNC_PHASE": lambda: _fm3d(p_flag=1),
    "FieldMap 1-D": lambda: _fm1d(),
    "FieldMap 1-D SET_SYNC_PHASE": lambda: _fm1d(p_flag=1),
    "SuperposedFieldMap": lambda: _sfm(),
}


# ------------------------------------------------------- 1. the helper = MP
@pytest.mark.parametrize("make", ELEMENTS.values(), ids=ELEMENTS.keys())
def test_helper_reproduces_every_tracker_call(monkeypatch, make):
    el = make()
    lat = _lattice(el)
    n_int, _n_sc, _ds, sc_every, nb = T.field_map_step_counts(lat, el)
    seen = []
    orig = type(el).track_rk4

    def spy(self, beam, ds):
        seen.append(_state(beam.ref))
        return orig(self, beam, ds)
    monkeypatch.setattr(type(el), "track_rk4", spy)
    _mp(lat, _ref(el))
    monkeypatch.undo()
    states = T.field_map_reference_states(lat, make(), _ref(el))
    assert len(states) == len(seen) == 2 * n_int
    assert [_state(st.ref_before) for st in states] == seen
    if isinstance(el, FieldMap3D):
        assert n_int > nb * sc_every               # a short last bundle is in the walk


# ------------------------------------------- 2. every envelope regime = MP
@pytest.mark.parametrize("make", ELEMENTS.values(), ids=ELEMENTS.keys())
@pytest.mark.parametrize("current, substeps", [(0.0, False), (0.0, True), (5.0, False),
                                               (5.0, True)])
def test_envelope_exit_reference_equals_multiparticle(make, current, substeps):
    el = make()
    mp_exit, _rec = _mp(_lattice(make()), _ref(el))
    env_exit, res = _env(_lattice(make()), _ref(el), current=current,
                         record_substeps=substeps)
    assert _state(env_exit) == _state(mp_exit)
    assert res.ref_w_kin[-1] == mp_exit.w_kin


@pytest.mark.parametrize("make", ELEMENTS.values(), ids=ELEMENTS.keys())
@pytest.mark.parametrize("current, substeps", [(0.0, True), (1e-9, False)])
def test_sliced_walk_matches_the_whole_map_matrix(make, current, substeps):
    """The sliced walks (sub-step rows without space charge; the space-charge
    walk, here at a negligible current) must evaluate every slice matrix at its
    own place in the map after the tracker walk ran: their product matches the
    whole-map matrix of the plain I=0 step (7e-4 for these 3-D maps, whose
    slices integrate with other steps; 2e-6 in 1-D).  With the map's cursor
    left at the exit the slices see no field and the 3-D map is off by 1.5e-2,
    the 1-D map by 1e-3."""
    tol = 3e-3 if isinstance(make(), FieldMap3D) else 1e-4
    _e, whole = _env(_lattice(make()), _ref(make()), current=0.0)
    _e, sliced = _env(_lattice(make()), _ref(make()), current=current,
                      record_substeps=substeps)
    a, b = np.asarray(whole.sigma_matrix[-1]), np.asarray(sliced.sigma_matrix[-1])
    assert np.max(np.abs(b - a)) / np.max(np.abs(a)) < tol


def test_envelope_interior_markers_keep_the_exit_reference():
    mp_exit, rec = _mp(_lattice(_sfm(markers=True)), _ref(_sfm()))
    for current in (0.0, 5.0):
        env_exit, res = _env(_lattice(_sfm(markers=True)), _ref(_sfm()), current=current)
        assert _state(env_exit) == _state(mp_exit)
        names = list(res.element_names)
        assert "DIA1" in names and "DIA2" in names
        assert res.s[names.index("DIA1")] == pytest.approx(120.0, abs=1e-9)
        assert res.s[names.index("DIA2")] == pytest.approx(250.0, abs=1e-9)


def test_full_step_advance_differs_from_the_tracker_walk():
    """Guard: the map's own full-step ``advance_ref`` (what the envelope used)
    is not the tracker's half-step walk, so the tests above can tell them
    apart."""
    el = _fm3d(L=900.0, freq=650.0)
    ref = ReferenceParticle(species=PROTON, w_kin=200.0, frequency=650.0)
    old = ref.copy()
    el.reset_run_state()
    el.advance_ref(old)
    new = ref.copy()
    T.field_map_reference_states(_lattice(el), el, new)
    assert new.w_kin != old.w_kin


# ------------------------------------ 3. rows and the β the SC kick sees
def test_space_charge_kicks_see_the_tracker_beta(monkeypatch):
    el = _fm3d()
    betas_mp, betas_env = [], []
    orig_k = T.Tracker._apply_sc_kick

    def spy_k(self, ds_mm, *a, **k):
        betas_mp.append(self.beam.ref.beta)
        return orig_k(self, ds_mm, *a, **k)
    monkeypatch.setattr(T.Tracker, "_apply_sc_kick", spy_k)
    _mp(_lattice(_fm3d()), _ref(el), current=5.0)
    orig_e = EnvelopeSolver._sc_matrix

    def spy_e(self, sigma, *a, **k):
        betas_env.append(self._ref.beta)
        return orig_e(self, sigma, *a, **k)
    monkeypatch.setattr(EnvelopeSolver, "_sc_matrix", spy_e)
    _env(_lattice(_fm3d()), _ref(el), current=5.0)
    assert betas_env == betas_mp and len(betas_env) > 0


def test_substep_rows_carry_the_tracker_reference():
    el = _fm3d()
    beam = _beam(_ref(el))
    rec_sub = T.Tracker(_lattice(_fm3d()), beam, record_substeps=True).run()
    mp_rows = dict(zip(np.round(rec_sub.s, 9), rec_sub.ref_w_kin, strict=True))
    for current in (0.0, 5.0):
        _e, res = _env(_lattice(_fm3d()), _ref(el), current=current, record_substeps=True)
        inside = [(s, w) for s, w in zip(res.s, res.ref_w_kin, strict=True) if s > 0.0]
        assert len(inside) > 2
        for s, w in inside:
            assert mp_rows[round(s, 9)] == w


# ------------------------------------- 4./5. frequency jumps and offsets
@pytest.mark.parametrize("current", [0.0, 5.0])
def test_freq_card_between_maps(current):
    def lattice():
        return _lattice(_fm3d("A"), Freq("F", frequency_mhz=325.0),
                        _fm3d("B", freq=325.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    mp_exit, _ = _mp(lattice(), ref)
    env_exit, _ = _env(lattice(), ref, current=current)
    assert env_exit.frequency == 325.0
    assert env_exit.w_kin == pytest.approx(mp_exit.w_kin, rel=1e-13)
    assert env_exit.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-12)


@pytest.mark.parametrize("current", [0.0, 5.0])
def test_cavity_frequency_jump_and_offset(current):
    """A 325 MHz map entered at 162.5 MHz, with a frequency error: both modes
    carry the effective frequency, and the envelope no longer warns that it
    drops the offset."""
    def lattice():
        return _lattice(_fm3d("A"), _fm3d("B", freq=325.0, frequency_offset=0.004))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    mp_exit, _ = _mp(lattice(), ref)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        solver = EnvelopeSolver(lattice(), ref.copy(), dict(INITIAL), current=current)
        solver.run()
    assert not [w for w in rec if "frequency_offset" in str(w.message)]
    assert solver._ref.frequency == mp_exit.frequency == pytest.approx(325.004)
    assert solver._ref.w_kin == mp_exit.w_kin


# ---------------------------------------- 6. superposed child phases
def test_superposed_child_phases_match_multiparticle():
    lat_mp = _lattice(_sfm(p_flag=1))
    _mp(lat_mp, _ref(_sfm()))
    lat_env = _lattice(_sfm(p_flag=1))
    _env(lat_env, _ref(_sfm()), current=5.0)
    mp_children = [(c._phi_s_at_entrance, c._sync_offset_deg)
                   for _z, c in lat_mp.elements[0].children if c._has_electric_channel()]
    env_children = [(c._phi_s_at_entrance, c._sync_offset_deg)
                    for _z, c in lat_env.elements[0].children if c._has_electric_channel()]
    assert env_children == mp_children


def test_curved_cluster_keeps_its_own_advance():
    sfm = _sfm()
    assert T.tracks_reference_on_tracker_schedule(sfm)
    sfm.is_curved = True
    assert not T.tracks_reference_on_tracker_schedule(sfm)


# ------------------------------------------------------ 8. lattice reuse
def test_lattice_reuse_gives_identical_results():
    lat = _lattice(Drift("D0", 50.0), _fm3d(p_flag=1), Drift("D1", 80.0), _fm1d(p_flag=1))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    first, r1 = _env(lat, ref, current=5.0)
    m1, _ = _mp(lat, ref)
    second, r2 = _env(lat, ref, current=5.0)
    m2, _ = _mp(lat, ref)
    assert _state(first) == _state(second)
    assert _state(m1) == _state(m2)
    np.testing.assert_array_equal(np.asarray(r1.sigma_x), np.asarray(r2.sigma_x))


# ------------------------------- 9./10. matrix mode and backtracking tables
def test_matrix_mode_and_replay_table_match_multiparticle():
    lat = _lattice(Drift("D0", 50.0), _fm3d(p_flag=1), _fm1d(), _sfm())
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    mp_exit, _ = _mp(lat, ref)
    via_matrix = reference_at(lat, ref, len(lat.elements))
    table = build_replay_table(lat, ref, end=len(lat.elements) - 1)
    for got in (via_matrix, table[-1]):
        assert got.w_kin == pytest.approx(mp_exit.w_kin, rel=1e-13)
        assert got.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-12)
        assert got.frequency == mp_exit.frequency


def test_whole_line_envelope_matches_multiparticle_at_every_exit():
    """Drifts between the maps: MP moves φ_s through a drift in pieces, the
    envelope in one, so the whole line agrees to round-off, not to the bit."""
    def lattice():
        return _lattice(Drift("D0", 50.0), _fm3d("A", p_flag=1), Drift("D1", 80.0),
                        _fm3d("B", freq=325.0), Drift("D2", 40.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    _m, rec = _mp(lattice(), ref)
    for current in (0.0, 5.0):
        _e, res = _env(lattice(), ref, current=current)
        np.testing.assert_allclose(res.ref_w_kin, rec.ref_w_kin, rtol=1e-12, atol=0)
        np.testing.assert_allclose(res.ref_phi_s, rec.ref_phi_s, rtol=1e-12, atol=1e-9)


# --------------------------- analysis walks on a lattice reused at another energy
def _bend_line():
    from linac_gen.elements.dipole import Dipole
    lat = _lattice(Drift("D0", 50.0), _fm3d("A", p_flag=1), Drift("D1", 100.0))
    lat.add(Dipole("B1", angle=15.0, rho=1500.0, aperture=40.0))     # deg, mm
    lat.add(Drift("D2", 200.0))
    lat.add(_fm3d("B", p_flag=1))
    lat.add(Drift("D3", 100.0))
    return lat


def test_dispersion_and_matrix_mode_ignore_a_previous_run_at_another_energy():
    """The dispersion walk built each map's matrix on whatever SET_SYNC_PHASE
    calibration the last run left on the lattice (a multiparticle run at another
    energy, say); it now starts each map from a clean state, as the matrix mode
    does, so a reused lattice gives the fresh lattice's result."""
    from linac_gen.analysis.dispersion import dispersion_along_s
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    fresh_d = dispersion_along_s(_bend_line(), ref)
    fresh_m = compute_transfer_matrix(_bend_line(), ref)
    reused = _bend_line()
    _mp(reused, ReferenceParticle(species=PROTON, w_kin=2.6, frequency=162.5))
    reused_d = dispersion_along_s(reused, ref)
    reused_m = compute_transfer_matrix(reused, ref)
    np.testing.assert_array_equal(reused_d["eta_x"], fresh_d["eta_x"])
    np.testing.assert_array_equal(reused_m, fresh_m)


# ------------------------------------------------------------ negative drift
@pytest.mark.parametrize("current", [0.0, 5.0])
def test_a_negative_drift_steps_the_clock_back_in_every_mode(current):
    """A deck can overlap two elements with a negative drift.  A multiparticle run
    steps ``s`` and φ_s back through it; the other modes skipped φ_s, and the
    backtracking table skipped ``s`` as well."""
    def lat():
        return _lattice(Drift("D0", 100.0), Drift("D1", -204.288), Drift("D2", 300.0))
    ref = ReferenceParticle(species=PROTON, w_kin=800.0, frequency=162.5)
    s = 100.0 - 204.288 + 300.0
    phi = 360.0 * s / (ref.beta * ref.wavelength)
    mp_exit, _ = _mp(lat(), ref, current)
    env_exit, _ = _env(lat(), ref, current)
    rows = [mp_exit, env_exit, reference_at(lat(), ref, 3),
            build_replay_table(lat(), ref, end=2)[-1]]
    for got in rows:
        assert got.s == pytest.approx(s, rel=1e-12)
        assert got.phi_s == pytest.approx(phi, rel=1e-12)


@pytest.mark.parametrize("current", [0.0, 5.0])
def test_rf_after_a_negative_drift_reads_the_same_clock(current):
    """No deck has this today.  An absolute-phase map after the drift is what the
    skipped φ_s would have got wrong."""
    def lat():
        return _lattice(Drift("D0", 100.0), Drift("D1", -40.0), _fm3d(p_flag=0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    mp_exit, _ = _mp(lat(), ref, current)
    env_exit, _ = _env(lat(), ref, current)
    assert env_exit.w_kin == pytest.approx(mp_exit.w_kin, rel=1e-12)
    assert env_exit.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-9)


def test_a_negative_drift_in_the_matrix_and_phase_advance_walks():
    """Overlapping with a negative drift is the same line as the shorter drift: the
    map after it sees the same clock, so the whole matrix and the walk agree."""
    from linac_gen.analysis.phase_advance import _advance_ref_like_matrix_walk
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    def overlap():
        return _lattice(Drift("D0", 100.0), Drift("D1", -40.0), _fm3d(p_flag=0))

    def shorter():
        return _lattice(Drift("D0", 60.0), _fm3d(p_flag=0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)
    np.testing.assert_allclose(compute_transfer_matrix(overlap(), ref),
                               compute_transfer_matrix(shorter(), ref),
                               rtol=1e-9, atol=1e-12)
    lat = overlap()
    walked = ref.copy()
    for el in lat.elements:
        _advance_ref_like_matrix_walk(el, walked, lat)
    mp_exit, _ = _mp(overlap(), ref)
    assert walked.w_kin == pytest.approx(mp_exit.w_kin, rel=1e-12)
    assert walked.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-9)


# ------------------------------------------------- FREQ cards in the matrix walks
def _tracker_matrix(lat, ref):
    """The multiparticle tracker's whole-line map, by central differences."""
    eps = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-6])
    beam = Beam(ref=ref.copy(), n_particles=13, current=0.0)
    beam.particles[:] = 0.0
    for i in range(6):
        beam.particles[2 * i + 1, i] = eps[i]
        beam.particles[2 * i + 2, i] = -eps[i]
    T.Tracker(lat, beam).run()
    M = np.zeros((6, 6))
    for i in range(6):
        M[:, i] = (beam.particles[2 * i + 1] - beam.particles[2 * i + 2]) / (2 * eps[i])
    return M, beam.ref


def _ncells(frequency=325.0):
    from linac_gen.elements.ncells import NCells
    return NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6,
                  theta_s_deg=-30.0, aperture_mm=15.0, p_flag=1,
                  frequency_mhz=frequency)


def _f_ref(w=2.1, f=162.5):
    return ReferenceParticle(species=PROTON, w_kin=w, frequency=f)


@pytest.mark.parametrize("current", [0.0, 5.0])
@pytest.mark.parametrize("rf_after", [False, True], ids=["no_rf", "map_after"])
def test_a_freq_card_moves_the_clock_in_every_walk(current, rf_after):
    """The matrix walks treated FREQ as passive, so the clock stayed at the old
    frequency up to the next RF element, and to the end of a line without one."""
    from linac_gen.analysis.phase_advance import _advance_ref_like_matrix_walk

    def lat():
        tail = [_fm3d(freq=325.0, p_flag=0)] if rf_after else []
        return _lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0), *tail)
    ref = _f_ref()
    mp_exit, _ = _mp(lat(), ref, current)
    env_exit, _ = _env(lat(), ref, current)
    line = lat()
    walked = ref.copy()
    for el in line.elements:
        _advance_ref_like_matrix_walk(el, walked, line)
    rows = [env_exit, reference_at(lat(), ref, len(line.elements)), walked,
            build_replay_table(lat(), ref, end=len(line.elements) - 1)[-1]]
    assert mp_exit.frequency == 325.0
    for got in rows:
        assert got.frequency == mp_exit.frequency
        assert got.w_kin == pytest.approx(mp_exit.w_kin, rel=1e-12)
        assert got.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-9)


def test_the_card_is_a_change_of_the_phase_coordinate_in_the_matrix():
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    def lat():
        return _lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0))
    ref = _f_ref()
    M = compute_transfer_matrix(lat(), ref)
    assert M[4, 4] == 2.0
    Mt, _ = _tracker_matrix(lat(), ref)
    np.testing.assert_allclose(M, Mt, rtol=1e-6, atol=1e-9)
    # the card alone, read at the clock it meets
    line = lat()
    D = np.eye(6)
    D[4, 4] = 2.0
    np.testing.assert_array_equal(compute_transfer_matrix(line, ref, start=1, end=1), D)
    # a window after the card starts on the new clock, as a line built at 325 MHz does
    np.testing.assert_allclose(
        compute_transfer_matrix(lat(), ref, start=2),
        compute_transfer_matrix(_lattice(Drift("D1", 150.0)), _f_ref(f=325.0)),
        rtol=1e-13, atol=0)


@pytest.mark.parametrize("card_mhz", [162.5, 0.0], ids=["same", "card_zero"])
def test_a_card_that_rescales_nothing_leaves_the_matrix_alone(card_mhz):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    card = Freq("F", card_mhz)
    ref = _f_ref()
    np.testing.assert_array_equal(card.transfer_matrix(ref), np.eye(6))
    line = _lattice(Marker("M0", snapshot=False), card)
    np.testing.assert_array_equal(compute_transfer_matrix(line, ref), np.eye(6))
    after = reference_at(line, ref, 2)
    assert after.frequency == 162.5
    assert after.phi_s == ref.phi_s


def test_a_card_meeting_a_clock_without_a_frequency_has_nothing_to_scale():
    """``apply_command`` guards this; a ``ReferenceParticle`` cannot be built at 0 MHz."""
    from types import SimpleNamespace
    np.testing.assert_array_equal(
        Freq("F", 325.0).transfer_matrix(SimpleNamespace(frequency=0.0)), np.eye(6))


def test_an_equal_card_is_bit_identical_to_no_card():
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    ref = _f_ref()

    def lat(*card):
        return _lattice(Drift("D0", 100.0), *card, _fm3d(p_flag=0), Drift("D1", 50.0))
    np.testing.assert_array_equal(compute_transfer_matrix(lat(Freq("F", 162.5)), ref),
                                  compute_transfer_matrix(lat(), ref))
    a, b = reference_at(lat(Freq("F", 162.5)), ref, 4), reference_at(lat(), ref, 3)
    assert (a.s, a.w_kin, a.phi_s, a.frequency) == (b.s, b.w_kin, b.phi_s, b.frequency)


def test_a_card_before_a_calibrated_map_gives_what_the_map_alone_gives():
    """A map with no card makes the jump itself on entry.  A drift's matrix obeys
    Drift(f₂)·D = D·Drift(f₁), so both routes are the same whole-line map when the
    map calibrates its own phase.  (Not φ_s, and not an absolute-phase map: that
    route leaves φ_s in degrees of the old clock.)"""
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    ref = _f_ref()
    with_card = _lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0),
                         _fm3d(freq=325.0, p_flag=1))
    no_card = _lattice(Drift("D0", 100.0), Drift("D1", 150.0), _fm3d(freq=325.0, p_flag=1))
    np.testing.assert_allclose(compute_transfer_matrix(with_card, ref),
                               compute_transfer_matrix(no_card, ref),
                               rtol=1e-6, atol=1e-8)
    a, b = reference_at(with_card, ref, 4), reference_at(no_card, ref, 3)
    assert a.frequency == b.frequency == 325.0
    assert a.w_kin == pytest.approx(b.w_kin, rel=1e-9)


@pytest.mark.parametrize("make", [
    lambda: _fm3d(freq=325.0, p_flag=0),
    lambda: _fm3d(freq=325.0, p_flag=1),
    _ncells,
], ids=["map_absolute", "map_sync", "ncells"])
@pytest.mark.parametrize("jump", [True, False], ids=["jump", "same_frequency"])
def test_matrix_mode_reproduces_the_envelope_at_zero_current(make, jump):
    """M Σ₀ Mᵀ is the envelope's exit Σ on every block, with a card and without."""
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    def lat():
        return _lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0), make())
    ref = _f_ref(f=162.5 if jump else 325.0)
    _e, res = _env(lat(), ref, 0.0)
    S0, S1 = np.asarray(res.sigma_matrix[0]), np.asarray(res.sigma_matrix[-1])
    M = compute_transfer_matrix(lat(), ref)
    got = M @ S0 @ M.T
    scale = np.sqrt(np.outer(np.diag(S1), np.diag(S1)))
    assert np.abs((got - S1) / scale).max() < 1e-10


@pytest.mark.parametrize("jump", [True, False], ids=["jump", "same_frequency"])
def test_ncells_after_a_card_agrees_with_the_tracker_on_x_y_energy_and_phase(jump):
    """Not z: the tracker adds NCELLS's synchronism factor to the phase it hands the
    gaps (the 2-engine split in ``NCells._sync_delta``), and matrix mode and the
    envelope do not, by design."""
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    def lat():
        return _lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0), _ncells())
    ref = _f_ref(f=162.5 if jump else 325.0)
    M = compute_transfer_matrix(lat(), ref)
    Mt, rt = _tracker_matrix(lat(), ref)
    np.testing.assert_allclose(M[:4, :4], Mt[:4, :4], rtol=1e-5, atol=1e-6)
    line = lat()
    r = reference_at(line, ref, len(line.elements))
    assert r.w_kin == pytest.approx(rt.w_kin, rel=1e-11)
    assert r.phi_s == pytest.approx(rt.phi_s, rel=1e-9)
    assert r.frequency == rt.frequency == 325.0


def test_longitudinal_twiss_and_phase_advance_across_a_card():
    """β_z is in deg/MeV at the running clock, so it takes the factor at the card
    (the recorder's documented rule), the card adds no phase advance, and the drift
    after it advances on the new clock, as the same drift in a line at 325 MHz does."""
    from linac_gen.analysis.period_detect import PeriodicStructure
    from linac_gen.analysis.phase_advance import structure_phase_advance_along_s

    def walk(lat, ref, seed):
        n = len(lat.elements)
        period = PeriodicStructure(start=0, end=n, inner_period_length=n, inner_slice_end=n,
                                   n_repeats=1, label="p", source="test")
        return structure_phase_advance_along_s(lat, ref, period, seed=seed)
    seed = {"alpha_x": 0.0, "beta_x": 1.0, "alpha_y": 0.0, "beta_y": 1.0,
            "alpha_z": 0.0, "beta_z": 2.0}
    out = walk(_lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0)), _f_ref(), seed)
    assert out["beta_z"][2] == pytest.approx(2.0 * out["beta_z"][1], rel=1e-13)
    assert out["mu_z_deg"][2] == pytest.approx(out["mu_z_deg"][1], rel=1e-13)
    assert out["beta_x"][2] == pytest.approx(out["beta_x"][1], rel=1e-13)
    after = walk(_lattice(Drift("D1", 150.0)), _f_ref(f=325.0),
                 {**seed, "alpha_z": out["alpha_z"][2], "beta_z": out["beta_z"][2]})
    assert out["beta_z"][3] == pytest.approx(after["beta_z"][1], rel=1e-13)
    assert out["mu_z_deg"][3] - out["mu_z_deg"][2] == pytest.approx(after["mu_z_deg"][1], rel=1e-11)


def test_a_window_after_a_card_starts_on_the_new_clock():
    """The first cavity after a card used to make the jump itself inside the window,
    so a period there had a longitudinal determinant of f₂/f₁, and a wrong β_z and μ_z."""
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    lat = _lattice(Drift("D0", 100.0), Freq("F", 325.0), _fm3d(freq=325.0, p_flag=1))
    M = compute_transfer_matrix(lat, _f_ref(), start=2)
    assert np.linalg.det(M[4:, 4:]) == pytest.approx(1.0, abs=2e-2)


def test_dispersion_walk_applies_the_card_mid_line():
    """Downstream of a bend the dispersion vector has a Δφ component, so the card's
    factor reaches η_x through the cavity's phase dependence.  Against the tracker,
    which NCELLS's x plane matches."""
    from linac_gen.analysis.dispersion import dispersion_along_s
    from linac_gen.elements.dipole import Dipole

    def lat():
        return _lattice(Drift("D0", 100.0), Dipole("B1", angle=15.0, rho=1500.0, aperture=40.0),
                        Drift("D1", 200.0), Freq("F", 325.0), Drift("D2", 150.0),
                        _ncells(), Drift("D3", 100.0))
    ref = _f_ref()
    got = dispersion_along_s(lat(), ref)
    Mt, _rt = _tracker_matrix(lat(), ref)
    assert got["eta_x"][-1] == pytest.approx(Mt[0, 5] / Mt[5, 5], rel=1e-3)


def test_the_torch_mirror_applies_the_card():
    """``compute_transfer_matrix_torch`` is the mirror of the numpy walk."""
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch

    def lat():
        return _lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0),
                        Quadrupole("Q", length=100.0, gradient=2.0, aperture=30.0))
    ref = _f_ref()
    for start, end in ((0, None), (2, None), (1, 1)):
        np.testing.assert_allclose(
            compute_transfer_matrix_torch(lat(), ref, start=start, end=end).detach().numpy(),
            compute_transfer_matrix(lat(), ref, start=start, end=end), rtol=1e-13, atol=1e-13)


def test_envelope_backtracking_closes_across_a_card():
    """The backward walk inverts each element's matrix, and the card's was the
    identity, so the entrance σ_φ came back wrong."""
    from linac_gen.tracking.backtrack import backtrack_envelope

    def lat():
        return _lattice(Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0))
    ref = _f_ref()
    _e, fw = _env(lat(), ref, 0.0)
    exit_twiss = {"alpha_x": fw.alpha_x[-1], "beta_x": fw.beta_x[-1], "emit_x": fw.emit_x[-1],
                  "alpha_y": fw.alpha_y[-1], "beta_y": fw.beta_y[-1], "emit_y": fw.emit_y[-1],
                  "alpha_z": fw.alpha_z[-1], "beta_z": fw.beta_z[-1], "emit_z": fw.emit_z[-1]}
    back = backtrack_envelope(lat(), ref, exit_twiss)
    np.testing.assert_allclose(np.asarray(back.sigma_matrix[0]), np.asarray(fw.sigma_matrix[0]),
                               rtol=1e-9, atol=1e-11)


# ------------------------------------------------- cards that set the energy
# (the HELIX card; SET_BEAM_ENERGY and SET_BEAM_E0_P0 set and shifted the energy until
# 2026-11, and are TraceWin matching criteria, tests/elements/test_energy_cards.py)
ENERGY_CARDS = {
    "HELIX_SET_ENERGY": lambda: SetReferenceEnergy("E", energy_MeV=120.0),
    "HELIX_SHIFT_ENERGY": lambda: SetReferenceEnergy("E", energy_MeV=20.0, shift=True),
}


def _quad(name, g):
    from linac_gen.elements.quadrupole import Quadrupole
    return Quadrupole(name, length=100.0, gradient=g, aperture=30.0)


def _energy_line(make):
    return _lattice(Drift("D0", 300.0), _quad("Q0", 4.0), make(),
                    Drift("D1", 300.0), _quad("Q1", -4.0), Drift("D2", 100.0))


@pytest.mark.parametrize("current", [0.0, 5.0])
@pytest.mark.parametrize("make", ENERGY_CARDS.values(), ids=ENERGY_CARDS.keys())
def test_an_energy_card_moves_the_reference_in_every_walk(make, current):
    """The walks stepped over it, so the energy, the phase and every map after it
    were at the energy before the card."""
    from linac_gen.analysis.phase_advance import _advance_ref_like_matrix_walk

    ref = _f_ref(w=100.0, f=352.21)
    mp_exit, _ = _mp(_energy_line(make), ref, current)
    env_exit, _ = _env(_energy_line(make), ref, current)
    line = _energy_line(make)
    walked = ref.copy()
    for el in line.elements:
        _advance_ref_like_matrix_walk(el, walked, line)
    rows = [env_exit, reference_at(_energy_line(make), ref, len(line.elements)), walked,
            build_replay_table(_energy_line(make), ref, end=len(line.elements) - 1)[-1]]
    assert mp_exit.w_kin == 120.0
    for got in rows:
        assert got.w_kin == mp_exit.w_kin
        assert got.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-12)
        assert got.s == mp_exit.s


@pytest.mark.parametrize("make", ENERGY_CARDS.values(), ids=ENERGY_CARDS.keys())
def test_matrix_mode_reproduces_the_envelope_across_an_energy_card(make):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    ref = _f_ref(w=100.0, f=352.21)
    _e, res = _env(_energy_line(make), ref, 0.0)
    S0, S1 = np.asarray(res.sigma_matrix[0]), np.asarray(res.sigma_matrix[-1])
    M = compute_transfer_matrix(_energy_line(make), ref)
    got = M @ S0 @ M.T
    scale = np.sqrt(np.outer(np.diag(S1), np.diag(S1)))
    assert np.abs((got - S1) / scale).max() < 1e-12
    # the card's own map changes no coordinate
    line = _energy_line(make)
    np.testing.assert_array_equal(compute_transfer_matrix(line, ref, start=2, end=2), np.eye(6))


@pytest.mark.parametrize("make", ENERGY_CARDS.values(), ids=ENERGY_CARDS.keys())
def test_matrix_mode_matches_the_tracker_across_an_energy_card(make):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    ref = _f_ref(w=100.0, f=352.21)
    M = compute_transfer_matrix(_energy_line(make), ref)
    Mt, _r = _tracker_matrix(_energy_line(make), ref)
    np.testing.assert_allclose(M, Mt, rtol=1e-6, atol=1e-9)


def test_a_window_after_an_energy_card_starts_at_its_energy():
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    ref = _f_ref(w=100.0, f=352.21)
    line = _energy_line(ENERGY_CARDS["HELIX_SET_ENERGY"])
    alone = _lattice(Drift("D1", 300.0), _quad("Q1", -4.0), Drift("D2", 100.0))
    np.testing.assert_allclose(compute_transfer_matrix(line, ref, start=3),
                               compute_transfer_matrix(alone, _f_ref(w=120.0, f=352.21)),
                               rtol=1e-13, atol=1e-13)


def test_an_energy_card_before_a_map_evaluates_it_at_the_new_energy():
    """Through a cavity the gain starts from the card's energy, in the walk as in a run."""
    def lat():
        return _lattice(Drift("D0", 50.0), SetReferenceEnergy("E", energy_MeV=3.0),
                        _fm3d(freq=162.5, p_flag=0), Drift("D1", 50.0))
    ref = _f_ref(w=2.1, f=162.5)
    mp_exit, _ = _mp(lat(), ref, 0.0)
    line = lat()
    got = reference_at(line, ref, len(line.elements))
    assert got.w_kin == pytest.approx(mp_exit.w_kin, rel=1e-12)
    assert got.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-9)


@pytest.mark.parametrize("make, where", [
    (ENERGY_CARDS["HELIX_SET_ENERGY"], "first"), (ENERGY_CARDS["HELIX_SET_ENERGY"], "last"),
    (ENERGY_CARDS["HELIX_SHIFT_ENERGY"], "first"), (ENERGY_CARDS["HELIX_SHIFT_ENERGY"], "last")])
def test_an_energy_card_at_either_end_of_the_line(make, where):
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    def lat():
        body = [Drift("D0", 300.0), _quad("Q0", 4.0)]
        return _lattice(*([make()] + body if where == "first" else body + [make()]))
    ref = _f_ref(w=100.0, f=352.21)
    mp_exit, _ = _mp(lat(), ref, 0.0)
    line = lat()
    got = reference_at(line, ref, len(line.elements))
    assert got.w_kin == mp_exit.w_kin == 120.0
    assert got.phi_s == pytest.approx(mp_exit.phi_s, rel=1e-12)
    M = compute_transfer_matrix(lat(), ref)
    Mt, _r = _tracker_matrix(lat(), ref)
    np.testing.assert_allclose(M, Mt, rtol=1e-6, atol=1e-9)


def test_e0_p0_without_its_energy_flag_changes_nothing():
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    def lat(card):
        return _lattice(Drift("D0", 300.0), *card, _quad("Q0", 4.0))
    ref = _f_ref(w=100.0, f=352.21)
    a = compute_transfer_matrix(lat([]), ref)
    b = compute_transfer_matrix(lat([SetBeamE0P0("E", dE_MeV=20.0, dphi_deg=5.0, ke=0, kp=1)]), ref)
    assert a.tobytes() == b.tobytes()


def test_the_phase_advance_and_dispersion_walks_follow_an_energy_card():
    """Each evaluates its maps at the walk's energy, against a line with no card that
    starts at that energy."""
    from linac_gen.analysis.dispersion import dispersion_along_s
    from linac_gen.elements.dipole import Dipole

    def bend():
        return Dipole("B1", angle=15.0, rho=1500.0, aperture=40.0)
    line = _lattice(Drift("D0", 100.0), SetReferenceEnergy("E", energy_MeV=120.0), bend(), Drift("D1", 200.0))
    alone = _lattice(Drift("D0", 100.0), bend(), Drift("D1", 200.0))
    got = dispersion_along_s(line, _f_ref(w=100.0, f=352.21))
    want = dispersion_along_s(alone, _f_ref(w=120.0, f=352.21))
    assert got["eta_x"][-1] == pytest.approx(want["eta_x"][-1], rel=1e-12)
    assert abs(want["eta_x"][-1]) > 1e-3


def test_the_torch_mirror_follows_the_energy_cards():
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch

    ref = _f_ref(w=100.0, f=352.21)
    for make in ENERGY_CARDS.values():
        for start, end in ((0, None), (3, None), (2, 2)):
            np.testing.assert_allclose(
                compute_transfer_matrix_torch(_energy_line(make), ref, start=start, end=end).detach().numpy(),
                compute_transfer_matrix(_energy_line(make), ref, start=start, end=end),
                rtol=1e-13, atol=1e-13)


def test_a_curved_cluster_after_an_energy_card_is_not_traced_at_the_old_energy():
    from linac_gen.elements.curved_field_map import _energy_neutral

    assert not _energy_neutral(SetReferenceEnergy("E", energy_MeV=20.0, shift=True))
    assert not _energy_neutral(SetReferenceEnergy("E", energy_MeV=120.0))
    assert _energy_neutral(Drift("D", 100.0))


# ---------------------------------------------- NCELLS with no card before it
def _cardless(frequency=325.0):
    return _lattice(Drift("D0", 200.0), _ncells(frequency), Drift("D1", 100.0))


def test_ncells_without_a_card_makes_its_jump_in_the_matrix():
    """On Σ, not the determinant: scaling the Δφ column without evaluating the cavity
    on its own clock has the right determinant and gives σ_φ 7.08 for 15.56."""
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    ref = _f_ref(w=20.0)
    _e, res = _env(_cardless(), ref, 0.0)
    S0, S1 = np.asarray(res.sigma_matrix[0]), np.asarray(res.sigma_matrix[-1])
    M = compute_transfer_matrix(_cardless(), ref)
    got = M @ S0 @ M.T
    scale = np.sqrt(np.outer(np.diag(S1), np.diag(S1)))
    assert np.abs((got - S1) / scale).max() < 1e-10
    assert np.linalg.det(M[4:6, 4:6]) == pytest.approx(2.0, rel=1e-12)


def test_ncells_without_a_card_agrees_with_the_tracker_on_x_y_energy_and_phase():
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    ref = _f_ref(w=20.0)
    M = compute_transfer_matrix(_cardless(), ref)
    Mt, rt = _tracker_matrix(_cardless(), ref)
    np.testing.assert_allclose(M[:4, :4], Mt[:4, :4], rtol=1e-5, atol=1e-6)
    line = _cardless()
    r = reference_at(line, ref, len(line.elements))
    assert r.w_kin == pytest.approx(rt.w_kin, rel=1e-11)
    assert r.phi_s == pytest.approx(rt.phi_s, rel=1e-9)
    assert r.frequency == rt.frequency == 325.0


def test_ncells_on_its_own_clock_is_bit_identical():
    """The branch every deck with a card takes: nothing to do."""
    el = _ncells()
    M = el.fitted_matrix(_f_ref(w=20.0, f=325.0))
    el2 = _ncells()
    el2.reset_run_state()
    N = el2.fitted_matrix_slice(_f_ref(w=20.0, f=325.0), el2.length)
    assert M.tobytes() == N.tobytes()


def test_ncells_fitted_matrix_leaves_the_callers_reference_alone():
    ref = _f_ref(w=20.0)
    before = (ref.frequency, ref.w_kin, ref.phi_s, ref.s)
    _ncells().fitted_matrix(ref)
    assert (ref.frequency, ref.w_kin, ref.phi_s, ref.s) == before


@pytest.mark.parametrize("f_cavity, clock", [(0.0, 162.5), (325.0, 325.0)], ids=["no_frequency", "same"])
def test_ncells_with_nothing_to_scale_makes_no_jump(f_cavity, clock):
    el = _ncells(f_cavity)
    M = el.fitted_matrix(_f_ref(w=20.0, f=clock))
    assert np.linalg.det(M[4:6, 4:6]) == pytest.approx(1.0, rel=1e-9)


def test_the_envelope_is_not_given_the_jump_twice():
    """It makes the jump, then asks for the matrix on the new clock."""
    ref = _f_ref(w=20.0)
    _e, res = _env(_cardless(), ref, 0.0)
    S0, S1 = np.asarray(res.sigma_matrix[0]), np.asarray(res.sigma_matrix[-1])
    # phase space area scales with the clock once
    assert np.sqrt(np.linalg.det(S1[4:6, 4:6]) / np.linalg.det(S0[4:6, 4:6])) == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize("start", [0, 2], ids=["in the span", "before the window"])
def test_a_card_that_leaves_no_energy_stops_the_torch_walk(start):
    from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(SetReferenceEnergy("E", energy_MeV=0.0))
    lat.add(Drift("D2", length=100.0))
    with pytest.raises(ValueError, match="no energy"):
        compute_transfer_matrix_torch(lat, ReferenceParticle(species=PROTON, w_kin=5.0, frequency=162.5),
                                      start=start)
