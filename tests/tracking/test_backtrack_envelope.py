# tests/tracking/test_backtrack_envelope.py
"""Backward envelope transport: Σ_entry = M⁻¹ Σ M⁻ᵀ off the replay table."""
import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.envelope import EnvelopeSolver, _sigma_to_twiss
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
from linac_gen.tracking.backtrack import backtrack_envelope

INITIAL = dict(alpha_x=0.5, beta_x=2.0, emit_x=0.25,
               alpha_y=-0.3, beta_y=1.5, emit_y=0.22,
               alpha_z=0.1, beta_z=3.0, emit_z=0.4)


def _make_ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _fodo_rfgap():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, n_steps=5))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, n_steps=5))
    lat.add(Drift("D2", 200.0))
    lat.add(RFGap("GAP", voltage=0.4, phase=-30.0, frequency=352.21,
                  ttf=0.85))
    lat.add(Drift("D3", 150.0))
    return lat


def _exit_twiss_of(fwd):
    tz = _sigma_to_twiss(fwd.sigma_matrix[-1], "z")
    return dict(alpha_x=fwd.alpha_x[-1], beta_x=fwd.beta_x[-1],
                emit_x=fwd.emit_x[-1],
                alpha_y=fwd.alpha_y[-1], beta_y=fwd.beta_y[-1],
                emit_y=fwd.emit_y[-1],
                alpha_z=tz["alpha"], beta_z=tz["beta"], emit_z=tz["emit"])


def test_envelope_roundtrip_sigma_exact():
    lat = _fodo_rfgap()
    fwd = EnvelopeSolver(lat, _make_ref(), dict(INITIAL), current=0.0).run()
    bwd = backtrack_envelope(lat, _make_ref(), _exit_twiss_of(fwd))
    np.testing.assert_allclose(bwd.sigma_matrix[0], fwd.sigma_matrix[0],
                               rtol=1e-9, atol=1e-12)
    for k in ("alpha_x", "beta_x", "emit_x", "alpha_y", "beta_y", "emit_y"):
        assert getattr(bwd, k)[0] == pytest.approx(INITIAL[k], rel=1e-8)


def test_envelope_consistent_with_inverse_transfer_matrix():
    """Σ_entry from the walk must equal inv(M_total)·Σ_exit·inv(M_total)ᵀ."""
    lat = _fodo_rfgap()
    fwd = EnvelopeSolver(lat, _make_ref(), dict(INITIAL), current=0.0).run()
    bwd = backtrack_envelope(lat, _make_ref(), _exit_twiss_of(fwd))
    M = compute_transfer_matrix(lat, _make_ref())
    Minv = np.linalg.inv(M)
    sigma_expected = Minv @ bwd.sigma_matrix[-1] @ Minv.T
    np.testing.assert_allclose(bwd.sigma_matrix[0], sigma_expected,
                               rtol=1e-9, atol=1e-12)


def test_envelope_results_shape_and_tags():
    lat = _fodo_rfgap()
    fwd = EnvelopeSolver(lat, _make_ref(), dict(INITIAL), current=0.0).run()
    bwd = backtrack_envelope(lat, _make_ref(), _exit_twiss_of(fwd))
    assert bwd.direction == "backward"
    assert bwd.backtrack_range == (0, len(lat.elements) - 1)
    assert len(bwd.s) == len(lat.elements) + 1
    assert all(b >= a for a, b in zip(bwd.s, bwd.s[1:]))
    assert bwd.s[0] == pytest.approx(0.0, abs=1e-12)
    assert bwd.element_names[-1] == "INPUT"
    # RFGap energy gain visible in the reversed ref series:
    assert bwd.ref_w_kin[-1] > bwd.ref_w_kin[0]


def test_envelope_subrange():
    lat = _fodo_rfgap()
    fwd = EnvelopeSolver(lat, _make_ref(), dict(INITIAL), current=0.0).run()
    # Twiss at the exit of element 3 (recorder row 4 = INPUT + elems 0-3):
    row = 4
    tz = _sigma_to_twiss(fwd.sigma_matrix[row], "z")
    mid_twiss = dict(alpha_x=fwd.alpha_x[row], beta_x=fwd.beta_x[row],
                     emit_x=fwd.emit_x[row],
                     alpha_y=fwd.alpha_y[row], beta_y=fwd.beta_y[row],
                     emit_y=fwd.emit_y[row],
                     alpha_z=tz["alpha"], beta_z=tz["beta"],
                     emit_z=tz["emit"])
    bwd = backtrack_envelope(lat, _make_ref(), mid_twiss, start=0, end=3)
    np.testing.assert_allclose(bwd.sigma_matrix[0], fwd.sigma_matrix[0],
                               rtol=1e-9, atol=1e-12)


def test_envelope_sc_raises():
    lat = _fodo_rfgap()
    with pytest.raises(ValueError, match="find_sc_matched_input_twiss"):
        backtrack_envelope(lat, _make_ref(), dict(INITIAL), current=5.0)


def test_envelope_invalid_range_raises():
    lat = _fodo_rfgap()
    with pytest.raises(ValueError, match="invalid backtrack range"):
        backtrack_envelope(lat, _make_ref(), dict(INITIAL), start=4, end=2)
