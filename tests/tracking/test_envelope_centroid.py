# tests/tracking/test_envelope_centroid.py
"""Envelope-mode first-moment (centroid) propagation.

The centroid rides the SAME matrices Σ uses (c → Mc via the _probe_push
choke point) plus three sources Σ cannot see: steerer kicks, dx/dy
misalignment feed-down, and the tilt conjugation.  Every analytic pin
here is exact (envelope is noiseless); the env-vs-MP case is limited
only by the MP sample mean.
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import get_element_matrix
from linac_gen.tracking.tracker import Tracker

INIT = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0,
            alpha_y=0.0, beta_y=2.0, emit_y=1.0,
            alpha_z=0.0, beta_z=10.0, emit_z=0.3)


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _run(lat, centroid=None, **kw):
    initial = dict(INIT)
    if centroid is not None:
        initial["centroid"] = centroid
    return EnvelopeSolver(lat, _ref(), initial, current=0.0, **kw).run()


def test_row_count_invariant_both_cadences():
    lat = Lattice()
    lat.add(Drift("D1", 500.0))
    lat.add(Quadrupole("Q", 100.0, 8.0, n_steps=4))
    lat.add(Drift("D2", 500.0))
    for substeps in (False, True):
        res = _run(lat, record_substeps=substeps)
        assert len(res.centroid) == len(res.s)
        assert all(np.shape(c) == (6,) for c in res.centroid)


def test_on_axis_stays_on_axis():
    lat = Lattice()
    lat.add(Drift("D", 500.0))
    lat.add(Quadrupole("Q", 100.0, 8.0))
    res = _run(lat)
    assert np.allclose(res.centroid[-1][:4], 0.0, atol=1e-15)


def test_steerer_kick_then_drift_analytic():
    lat = Lattice()
    lat.add(Steerer("S", by_l=1e-3))
    lat.add(Drift("D", 500.0))
    res = _run(lat)
    ref = _ref()
    kick = 1e-3 / ref.brho * 1e3          # mrad (positive charge)
    c = res.centroid[-1]
    assert c[1] == pytest.approx(kick, rel=1e-12)
    assert c[0] == pytest.approx(kick * 0.5, rel=1e-12)   # 0.5 m drift


def test_misaligned_quad_feed_down_is_I_minus_M_delta():
    lat = Lattice()
    q = Quadrupole("Q", 100.0, 8.0)
    q.dx, q.dy = 2.0, -1.0
    lat.add(q)
    res = _run(lat)
    M = get_element_matrix(Quadrupole("Qc", 100.0, 8.0), _ref())
    delta = np.array([2.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    expect = (np.eye(6) - M) @ delta
    assert np.allclose(res.centroid[-1][:4], expect[:4], atol=1e-12)


def test_tilt_sign_matches_mp_tracker():
    """Regression for the latent envelope tilt-sign bug: the entry
    rotation must be tilt_rotation_matrix(+tilt) exactly as the MP
    tracker rotates particles — the old R(−tilt) modelled −tilt,
    invisible to symmetric Σ but sign-flipping the centroid's coupled
    plane."""
    lat = Lattice()
    q = Quadrupole("Q", 100.0, 8.0)
    q.tilt_deg = 5.0
    lat.add(q)
    res = _run(lat, centroid=[0.8, 0, 0, 0, 0, 0])

    ref = _ref()
    n = 200000
    beam = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(3)
    beam.particles[:, 0] = rng.normal(0.8, 0.3, n)
    rec = Tracker(lat, beam).run()
    e, m = res.centroid[-1][:4], np.asarray(rec.centroid[-1][:4])
    assert np.allclose(e, m, atol=0.01)
    assert abs(e[3]) > 0.1                # coupled plane genuinely moves
    assert np.sign(e[3]) == np.sign(m[3])  # ... with the MP sign


def test_env_vs_mp_combined_stress():
    lat = Lattice()
    lat.add(Drift("D1", 200.0))
    q = Quadrupole("Q", 100.0, 8.0)
    q.dx, q.tilt_deg = 1.5, 5.0
    lat.add(q)
    lat.add(Steerer("S", bx_l=5e-4, by_l=-5e-4))
    lat.add(Drift("D2", 300.0))
    c0 = [0.8, -0.2, -0.4, 0.1, 0.0, 0.0]
    res = _run(lat, centroid=c0)

    ref = _ref()
    n = 200000
    beam = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(3)
    for k in range(4):
        beam.particles[:, k] = rng.normal(c0[k], 0.3, n)
    rec = Tracker(lat, beam).run()
    assert np.allclose(res.centroid[-1][:4],
                       np.asarray(rec.centroid[-1][:4]), atol=0.01)


def test_dispersion_moves_energy_offset_centroid():
    # A ΔW-offset centroid through a drift picks up Δφ via the drift's
    # longitudinal M[4,5]; transverse stays zero without bends.
    lat = Lattice()
    lat.add(Drift("D", 1000.0))
    res = _run(lat, centroid=[0, 0, 0, 0, 0.0, 0.05])
    c = res.centroid[-1]
    assert c[5] == pytest.approx(0.05, rel=1e-12)
    assert abs(c[4]) > 1.0                # TOF dispersion at 3 MeV is large
    assert np.allclose(c[:4], 0.0, atol=1e-15)


def test_space_charge_does_not_move_centroid():
    lat = Lattice()
    lat.add(Drift("D", 1000.0))
    initial = dict(INIT, centroid=[1.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    r0 = EnvelopeSolver(lat, _ref(), initial, current=0.0).run()
    r1 = EnvelopeSolver(lat, _ref(), initial, current=50.0).run()
    # Linear SC is centred on the beam: identical centroid trajectories.
    assert np.allclose(r0.centroid[-1], r1.centroid[-1], atol=1e-12)


def test_expected_field_present():
    res = _run(Lattice())
    assert hasattr(res, "centroid")


def test_multipole_dipole_component_moves_centroid():
    """Review F3: a MULTIPOLE's constant k0L kick is dropped by its
    kick_matrix — the envelope centroid must recover it via the
    nonlinear-remainder path (single application, matching the
    element's own apply_kick physics)."""
    from linac_gen.core.beam import Beam
    from linac_gen.elements.multipole import Multipole
    mp_el = Multipole("M", knl=[2e-3])
    lat = Lattice()
    lat.add(mp_el)
    lat.add(Drift("D", 500.0))
    res = _run(lat)
    # Analytic single-application prediction from the element itself:
    ref = _ref()
    probe = Beam(ref=ref, n_particles=1, current=0.0)
    Multipole("M2", knl=[2e-3]).apply_kick(probe)
    kick_xp = float(probe.particles[0, 1])            # mrad
    c = res.centroid[-1]
    assert abs(kick_xp) > 0.5                          # genuinely kicks
    assert c[1] == pytest.approx(kick_xp, rel=1e-9)
    assert c[0] == pytest.approx(kick_xp * 0.5, rel=1e-9)


def test_freq_jump_rescales_centroid_phase():
    """Dual-regime pin (review): a FREQ card must rescale c[4] by
    f_new/f_old — same instant on the machine clock, new degrees —
    exactly as MP does particles[:, 4] *= ratio and Σ does D Σ Dᵀ.
    The energy offset c[5] is frequency-independent."""
    from linac_gen.elements.lattice_commands import Freq
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Freq("F", frequency_mhz=352.21 * 2))
    lat.add(Drift("D2", 100.0))
    # Zero ΔW so drifts add no TOF phase — isolates the jump itself.
    res = _run(lat, centroid=[0, 0, 0, 0, 5.0, 0.0])
    assert res.centroid[-1][4] == pytest.approx(10.0, rel=1e-12)
    assert res.centroid[-1][5] == pytest.approx(0.0, abs=1e-15)
    # No jump → no rescale (the other regime).
    lat2 = Lattice()
    lat2.add(Drift("D1", 100.0))
    lat2.add(Drift("D2", 100.0))
    res2 = _run(lat2, centroid=[0, 0, 0, 0, 5.0, 0.0])
    assert res2.centroid[-1][4] == pytest.approx(5.0, rel=1e-12)


def test_h_minus_steerer_kick_sign():
    """Species-sign pin (review): the steerer kick reaches the envelope
    centroid through the element's own apply_kick, so H⁻ (negative
    charge) must deflect exactly opposite to a proton — pinned against
    a 1-particle Beam through the same element (noiseless)."""
    from linac_gen.core.particle import H_MINUS
    ref_h = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    lat = Lattice()
    lat.add(Steerer("S", by_l=1e-3))
    lat.add(Drift("D", 500.0))
    initial = dict(INIT)
    res = EnvelopeSolver(lat, ref_h.copy(), initial, current=0.0).run()

    probe = Beam(ref=ref_h.copy(), n_particles=1, current=0.0)
    Steerer("S2", by_l=1e-3).apply_kick(probe)
    kick = float(probe.particles[0, 1])                # mrad, H⁻ sign
    assert abs(kick) > 1e-3                            # genuinely kicks
    assert res.centroid[-1][1] == pytest.approx(kick, rel=1e-12)
    # Opposite sign to the proton case of the same magnet:
    res_p = _run(lat)
    assert np.sign(res.centroid[-1][1]) == -np.sign(res_p.centroid[-1][1])


def test_row_count_invariant_with_sc_and_substeps():
    """The len(centroid) == len(s) invariant must hold on the SC
    propagation branch too (Strang splitting records differently) —
    review coverage gap: the cadence test above runs current=0 only."""
    lat = Lattice()
    lat.add(Drift("D1", 300.0))
    lat.add(Quadrupole("Q", 100.0, 8.0, n_steps=4))
    lat.add(Drift("D2", 300.0))
    initial = dict(INIT, centroid=[0.5, 0.0, -0.5, 0.0, 0.0, 0.0])
    for substeps in (False, True):
        res = EnvelopeSolver(lat, _ref(), initial, current=30.0,
                             record_substeps=substeps).run()
        assert len(res.centroid) == len(res.s)


def test_shell_record_without_run_leaves_centroid_empty():
    """Backtrack-style shell solvers call _record without run(): the
    centroid list must stay EMPTY (honest "no first moment"), never
    fill with zeros a consumer would read as a real on-axis orbit."""
    lat = Lattice()
    lat.add(Drift("D", 100.0))
    solver = EnvelopeSolver(lat, _ref(), dict(INIT), current=0.0)
    from linac_gen.tracking.envelope import EnvelopeResults
    results = EnvelopeResults()
    sigma = np.eye(6)                      # any SPD Σ — content unused here
    solver._record(results, sigma, solver._ref)
    solver._record(results, sigma, solver._ref, element_name="D")
    assert len(results.s) == 2
    assert results.centroid == []          # no fake on-axis orbit


def test_torch_tilt_sign_matches_envelope():
    """Review F1: the torch differentiable map used the old R(−tilt)
    convention — its composed tilted-quad map must now match the
    envelope/MP sense (coupling term sign)."""
    torch = pytest.importorskip("torch")
    from linac_gen.elements.mixins import Misalignment
    from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch
    lat = Lattice()
    q = Quadrupole("Q", 100.0, 8.0)
    q.tilt_deg = 5.0
    lat.add(q)
    ref = _ref()
    M_t = compute_transfer_matrix_torch(lat, ref).detach().cpu().numpy()
    R = Misalignment.tilt_rotation_matrix(5.0)
    M_ref = R.T @ get_element_matrix(Quadrupole("Qc", 100.0, 8.0),
                                     _ref()) @ R
    assert np.allclose(M_t, M_ref, atol=1e-9)
