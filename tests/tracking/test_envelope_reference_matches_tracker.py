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
from linac_gen.elements.lattice_commands import Freq
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
