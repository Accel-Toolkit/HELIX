import os

import pytest

if not os.path.isdir("Fields"):
    pytest.skip("Fields/ field-map data is not distributed with the "
                "repository (third-party ANL/CEA data) — see "
                "examples/FIELD_MAPS.md", allow_module_level=True)

# tests/tracking/test_superpose_tracking.py
"""SuperposedFieldMap through the REAL tracking stack (phase C).

The Phase A anchors drove track_rk4 by hand; here the container rides
the actual entry paths: Simulation/Tracker (MP + Strang SC bundles),
the envelope solver (SC bundle loop with the _z_from_mm plumbing), and
matrix mode — plus the predicate/step-count integration edits.
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.field_map_reader import FieldMapData


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _beam(n=64, seed=11):
    b = Beam(ref=_ref(), n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    b.particles[:, 0] = rng.normal(0, 1.5, n)
    b.particles[:, 1] = rng.normal(0, 0.8, n)
    b.particles[:, 2] = rng.normal(0, 1.5, n)
    b.particles[:, 3] = rng.normal(0, 0.8, n)
    b.particles[:, 4] = rng.normal(0, 4.0, n)
    b.particles[:, 5] = rng.normal(0, 0.008, n)
    return b


def _rf_fd(length=120.0, nz=121, ez=2.0):
    z = np.linspace(0.0, length, nz)
    return FieldMapData(z=z, Ez=ez * np.sin(np.pi * z / length),
                        symmetry="1d")


def _sol_fd(length=300.0, nz=151, bz=0.35):
    z = np.linspace(0.0, length, nz)
    return FieldMapData(z=z, Bz=bz * (0.5 - 0.5 * np.cos(2 * np.pi * z
                                                         / length)),
                        symmetry="1d")


def _lat(elem):
    lat = Lattice()
    lat.add(Drift("D1", length=50.0, aperture=30.0))
    lat.add(elem)
    lat.add(Drift("D2", length=50.0, aperture=30.0))
    return lat


def _plain_cav(**kw):
    return FieldMap("CAV", length=120.0, field_data=_rf_fd(),
                    phase=-25.0, frequency=352.21, aperture=30.0,
                    n_steps=120, **kw)


def _cluster_cav():
    child = FieldMap("CAVc", length=120.0, field_data=_rf_fd(),
                     phase=-25.0, frequency=352.21, aperture=30.0,
                     n_steps=120)
    return SuperposedFieldMap("SUP", [(0.0, child)])


# ── (a) cross-mode identity: 1-child container ≡ plain element ───────────
def test_mp_tracker_identity():
    r1 = Simulation(_lat(_plain_cav()), _beam()).run()
    r2 = Simulation(_lat(_cluster_cav()), _beam()).run()
    assert np.allclose(r1.sigma_x, r2.sigma_x, rtol=1e-12)
    assert np.allclose(r1.emit_x, r2.emit_x, rtol=1e-12)
    assert r1.ref_w_kin[-1] == pytest.approx(r2.ref_w_kin[-1], rel=1e-13)
    assert len(r1.s) == len(r2.s)                # same recorder cadence


def test_envelope_identity():
    from linac_gen.tracking.envelope import EnvelopeSolver
    init = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
                alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                alpha_z=0.0, beta_z=10.0, emit_z=0.3)
    r1 = EnvelopeSolver(_lat(_plain_cav()), _ref(), dict(init),
                        current=0.0).run()
    r2 = EnvelopeSolver(_lat(_cluster_cav()), _ref(), dict(init),
                        current=0.0).run()
    assert np.allclose(r1.sigma_x, r2.sigma_x, rtol=1e-10)
    assert np.allclose(r1.sigma_phi, r2.sigma_phi, rtol=1e-10)
    assert r1.ref_w_kin[-1] == pytest.approx(r2.ref_w_kin[-1], rel=1e-12)


def test_envelope_identity_with_sc():
    """The SC bundle branch (advance_ref_over + fitted_matrix_slice with
    the explicit z-cursor) must treat the container like the plain map."""
    from linac_gen.tracking.envelope import EnvelopeSolver
    init = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
                alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                alpha_z=0.0, beta_z=10.0, emit_z=0.3)
    r1 = EnvelopeSolver(_lat(_plain_cav()), _ref(), dict(init),
                        current=20.0).run()
    r2 = EnvelopeSolver(_lat(_cluster_cav()), _ref(), dict(init),
                        current=20.0).run()
    assert np.allclose(r1.sigma_x, r2.sigma_x, rtol=1e-9)
    assert np.allclose(r1.sigma_w, r2.sigma_w, rtol=1e-9)


def test_matrix_mode_identity():
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
    M1 = compute_transfer_matrix(_lat(_plain_cav()), _ref())
    M2 = compute_transfer_matrix(_lat(_cluster_cav()), _ref())
    assert np.allclose(M1, M2, atol=1e-10)


# ── genuine overlap through the real stack ───────────────────────────────
def _overlap_cluster():
    sol = FieldMap("SOL", length=300.0, field_data=_sol_fd(),
                   aperture=30.0, n_steps=150)
    cav = FieldMap("CAV", length=120.0, field_data=_rf_fd(),
                   phase=-25.0, frequency=352.21, aperture=30.0,
                   n_steps=120)
    return SuperposedFieldMap("SUP", [(0.0, sol), (90.0, cav)])


def test_predicates_on_clusters():
    from linac_gen.tracking.tracker import (_is_magnetic_only_fieldmap,
                                            _is_rf_bunching_element,
                                            field_map_step_counts)
    sol_only = SuperposedFieldMap("S1", [
        (0.0, FieldMap("A", length=300.0, field_data=_sol_fd(),
                       n_steps=150)),
        (0.0, FieldMap("B", length=300.0, field_data=_sol_fd(bz=0.1),
                       n_steps=150))])
    mixed = _overlap_cluster()
    assert _is_magnetic_only_fieldmap(sol_only) is True
    assert _is_magnetic_only_fieldmap(mixed) is False
    assert _is_rf_bunching_element(sol_only) is False
    assert _is_rf_bunching_element(mixed) is True
    # Container n_steps floors the canonical slicing.
    lat = _lat(mixed)
    n_int, _n_sc, ds, _sc_every, _nb = field_map_step_counts(lat, mixed)
    assert n_int >= mixed.n_steps
    assert ds == pytest.approx(mixed.length / n_int)


def test_mp_and_envelope_run_overlap_cluster_with_sc():
    """(e) env-vs-MP parity on a genuine solenoid+cavity overlap with
    space charge — the two modes must agree at the usual few-percent
    level (loose parity bound; the tight anchors live in phase A)."""
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat_mp = _lat(_overlap_cluster())
    beam = _beam(n=3000, seed=3)
    beam.current = 15.0
    from linac_gen.core.config import SpaceChargeConfig
    sc = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0,
                           use_gpu="cpu", grid_mode="adaptive")
    rec = Simulation(lat_mp, beam, space_charge=sc).run()
    assert rec.transmission[-1] == pytest.approx(100.0)

    init = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.284,
                alpha_y=0.0, beta_y=2.0, emit_y=1.284,
                alpha_z=0.0, beta_z=10.0, emit_z=0.3)
    env = EnvelopeSolver(_lat(_overlap_cluster()), _ref(), init,
                         current=15.0).run()
    # Loose parity: same physics regime, different SC models.
    assert env.sigma_x[-1] == pytest.approx(rec.sigma_x[-1], rel=0.15)
    assert env.sigma_y[-1] == pytest.approx(rec.sigma_y[-1], rel=0.15)
    # Envelope and MP discretize the container walk with slightly
    # different bundle cadences (SC Strang bundling vs native grid),
    # so the reference energies agree to ~1e-5 relative, not exactly.
    assert env.ref_w_kin[-1] == pytest.approx(rec.ref_w_kin[-1], rel=5e-5)


def test_adjust_on_container_warn_skips():
    from linac_gen.elements.lattice_commands import Adjust
    from linac_gen.matching.variables import collect_variables
    lat = Lattice()
    lat.add(Adjust("A1", target="SUP", param_idx=2, link_group=1,
                   vmin=0.0, vmax=10.0, start_step=0.5))
    lat.add(_overlap_cluster())
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21)
    with pytest.warns(UserWarning, match="SUPERPOSE cluster"):
        variables = collect_variables(lat, cfg)
    assert variables == []


def test_registry_refuses_cluster_wrapping():
    from linac_gen.surrogates import registry
    class _FakeSurr:
        _wrapped = _overlap_cluster()
        metadata = None
    with pytest.raises(ValueError, match="SUPERPOSE cluster"):
        registry.register(_FakeSurr(), "hash", element_key="SUP")


# ── exact backward closure ───────────────────────────────────────────────
def test_backtrack_exact_closure_through_cluster():
    """Forward-track a solenoid+cavity overlap cluster, then backtrack
    with the exact inverse: closure at the 1e-12 level (the same
    ceiling class as the plain-FieldMap exact tests)."""
    import warnings as _w
    from linac_gen.tracking.tracker import Tracker
    from linac_gen.tracking.backtrack import backtrack_distribution

    lat = _lat(_overlap_cluster())
    beam = _beam(n=200, seed=13)
    snap = beam.particles.copy()
    Tracker(lat, beam).run()
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        backtrack_distribution(lat, beam, _ref(), start=0,
                               end=len(lat.elements) - 1,
                               field_map_mode="rk4")
    alive = beam.alive_mask
    err = np.abs(beam.particles[alive] - snap[alive]).max(axis=0)
    ceiling = np.array([1e-11, 1e-10, 1e-11, 1e-10, 1e-10, 1e-12])
    assert np.all(err < ceiling), err


# ── (d) HWR physics case on real field files ─────────────────────────────
@pytest.mark.slow
def test_hwr_gapped_cluster_matches_sequential_workaround():
    """The shipped decks lay HWR solenoid + cavity END-TO-END with a
    67.9 mm drift.  The same geometry expressed as one gapped cluster
    (solenoid@0, cavity@367.9, span 617.9) must reproduce the
    sequential layout closely — the only differences are the shared
    entrance bookkeeping and interpolated child-entry clocks."""
    import os
    if not os.path.isfile("Fields/HWR-SOL-ANLMAP.bsz"):
        pytest.skip("Fields/ not present")
    from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
    from linac_gen.io.tracewin_geom import decode_geom
    from linac_gen.elements.field_map_factory import make_field_map_element

    def _mk(geom, length, phase, ke_kb, name, prefix, p_flag=0):
        # ka=0 like the shipped decks (ka=1 would demand a .ouv file).
        fd = read_tracewin_fieldmap(geom=geom, prefix=prefix,
                                    base_dir=None, frequency=162.5,
                                    Ki=0.0, Ka=0)
        return make_field_map_element(
            name=name, code=decode_geom(geom), length_mm=length,
            field_data=fd, kb=ke_kb[0], ke=ke_kb[1], ki=0.0, ka=0,
            phase=phase, frequency=162.5, aperture=15.0, p_flag=p_flag,
            n_steps=100, geom=geom, field_file=prefix)

    ref = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5)

    def _beam2():
        b = Beam(ref=ref.copy(), n_particles=200, current=0.0)
        rng = np.random.default_rng(5)
        b.particles[:, :4] = rng.normal(0, 1.0, (200, 4))
        b.particles[:, 4] = rng.normal(0, 3.0, 200)
        b.particles[:, 5] = rng.normal(0, 0.005, 200)
        return b

    sol = _mk(10, 300.0, 0.0, (2.3192, 1.0), "SOL",
              "Fields/HWR-SOL-ANLMAP")
    cav = _mk(7700, 250.0, -35.0, (1.0, 1.0), "CAV", "Fields/HWRDonut")

    lat_seq = Lattice()
    lat_seq.add(sol)
    lat_seq.add(Drift("G", length=67.9, aperture=15.0))
    lat_seq.add(cav)

    sol2 = _mk(10, 300.0, 0.0, (2.3192, 1.0), "SOL2",
               "Fields/HWR-SOL-ANLMAP")
    cav2 = _mk(7700, 250.0, -35.0, (1.0, 1.0), "CAV2", "Fields/HWRDonut")
    lat_cl = Lattice()
    lat_cl.add(SuperposedFieldMap("PKG", [(0.0, sol2), (367.9, cav2)]))

    r1 = Simulation(lat_seq, _beam2()).run()
    r2 = Simulation(lat_cl, _beam2()).run()
    assert r1.ref_w_kin[-1] == pytest.approx(r2.ref_w_kin[-1], rel=2e-4)
    assert r1.sigma_x[-1] == pytest.approx(r2.sigma_x[-1], rel=2e-2)
    assert r1.sigma_y[-1] == pytest.approx(r2.sigma_y[-1], rel=2e-2)
    assert r1.sigma_phi[-1] == pytest.approx(r2.sigma_phi[-1], rel=5e-2)


# ── SHIFT_IN_FIELD_MAP interior diagnostics (phase D) ────────────────────
def _marked_cluster():
    from linac_gen.elements.marker import Marker
    sol = FieldMap("SOLm", length=300.0, field_data=_sol_fd(),
                   aperture=30.0, n_steps=150)
    cav = FieldMap("CAVm", length=120.0, field_data=_rf_fd(),
                   phase=-25.0, frequency=352.21, aperture=30.0,
                   n_steps=120)
    mk1 = Marker("DIA1", snapshot=False)
    mk2 = Marker("DIA2", snapshot=False)
    return SuperposedFieldMap(
        "SUPm", [(0.0, sol), (90.0, cav)],
        interior_markers=[(120.0, mk1), (250.0, mk2)])


def test_mp_records_interior_marker_rows():
    lat = _lat(_marked_cluster())
    rec = Simulation(lat, _beam()).run()
    names = list(rec.element_names)
    assert "DIA1" in names and "DIA2" in names
    i1 = names.index("DIA1")
    i2 = names.index("DIA2")
    # Rows attributed at s_entry(=50) + dz.
    assert rec.s[i1] == pytest.approx(50.0 + 120.0)
    assert rec.s[i2] == pytest.approx(50.0 + 250.0)
    # element_exit_idx alignment survives: one entry per lattice
    # element, and the container's exit row sits at its true exit s.
    assert len(rec.element_exit_idx) == len(lat.elements)
    exit_row = rec.element_exit_idx[1]
    assert rec.s[exit_row] == pytest.approx(50.0 + 300.0)
    # The marker rows carry real interior beam data (σ between the
    # entrance and exit values, not zeros).
    assert rec.sigma_x[i1] > 0


def test_mp_interior_rows_with_substeps_on():
    lat = _lat(_marked_cluster())
    rec = Simulation(lat, _beam(), record_substeps=True).run()
    names = list(rec.element_names)
    assert "DIA1" in names and "DIA2" in names
    assert len(rec.element_exit_idx) == len(lat.elements)
    assert rec.s[rec.element_exit_idx[1]] == pytest.approx(350.0)


def test_envelope_records_interior_marker_rows_no_sc():
    """current=0 without record_substeps: the markers-only route must
    engage the sub-stepped walk and emit the rows."""
    from linac_gen.tracking.envelope import EnvelopeSolver
    init = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
                alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                alpha_z=0.0, beta_z=10.0, emit_z=0.3)
    res = EnvelopeSolver(_lat(_marked_cluster()), _ref(), init,
                         current=0.0).run()
    names = list(res.element_names)
    assert "DIA1" in names and "DIA2" in names
    # The marker advance is split at z=dz: rows carry the EXACT
    # s = s_entry + dz even off the SC-bundle grid (dz=250 is
    # mid-bundle at the default 20 mm cadence).
    i1 = names.index("DIA1")
    i2 = names.index("DIA2")
    assert res.s[i1] == pytest.approx(50.0 + 120.0, abs=1e-6)
    assert res.s[i2] == pytest.approx(50.0 + 250.0, abs=1e-6)
    assert len(res.element_exit_idx) == 3
    assert res.s[res.element_exit_idx[1]] == pytest.approx(350.0)
    # len(centroid) == len(s) invariant holds with the extra rows.
    assert len(res.centroid) == len(res.s)


def test_envelope_records_interior_marker_rows_with_sc():
    from linac_gen.tracking.envelope import EnvelopeSolver
    init = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
                alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                alpha_z=0.0, beta_z=10.0, emit_z=0.3)
    res = EnvelopeSolver(_lat(_marked_cluster()), _ref(), init,
                         current=20.0).run()
    names = list(res.element_names)
    assert "DIA1" in names and "DIA2" in names
    # Exact s even with the Strang SC bundles (split transport).
    assert res.s[names.index("DIA1")] == pytest.approx(170.0, abs=1e-6)
    assert res.s[names.index("DIA2")] == pytest.approx(300.0, abs=1e-6)
    assert len(res.element_exit_idx) == 3
    assert len(res.centroid) == len(res.s)
