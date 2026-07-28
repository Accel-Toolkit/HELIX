"""Transfer-matrix dispersion along s (`analysis.dispersion`).

External anchor: on a STATIC line (bends/quads/drifts, I = 0) fed with a
dispersion-free beam, the statistical dispersion the results popup
computes from the Σ-matrix (``Σ[0,5]/Σ[5,5] · β²γ·mc²``) must equal the
transfer-matrix dispersion exactly — the Σ-matrix is propagated by the
very same element matrices, and ΔW is invariant, so
``Σ[0,5](s) = M[0,5]·Σ[5,5]``.  Both regimes of the seeding are tested
(η₀ = 0 and η₀ ≠ 0), per the dual-regime house rule.
"""
import numpy as np

from linac_gen.analysis.dispersion import dispersion_along_s
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.envelope import EnvelopeSolver

INITIAL = dict(alpha_x=0.5, beta_x=2.0, emit_x=0.25,
               alpha_y=-0.3, beta_y=1.5, emit_y=0.22,
               alpha_z=0.1, beta_z=3.0, emit_z=0.4)


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _bend_line(n_cells=2):
    lat = Lattice()
    for c in range(n_cells):
        lat.add(Quadrupole(f"QF_{c}", 50.0, gradient=5.0, aperture=20.0,
                           n_steps=5))
        lat.add(Drift(f"D1_{c}", 200.0, aperture=50.0))
        lat.add(Dipole(f"B_{c}", angle=10.0, rho=2000.0, aperture=50.0))
        lat.add(Quadrupole(f"QD_{c}", 50.0, gradient=-5.0, aperture=20.0,
                           n_steps=5))
        lat.add(Drift(f"D2_{c}", 200.0, aperture=50.0))
    return lat


def _statistical_disp(res):
    """Exactly the popup formula: D_x [m] = Σ05/Σ55 · β²γ·mc² · 1e-3."""
    S = np.asarray(res.sigma_matrix, dtype=float)
    beta = np.asarray(res.ref_beta, dtype=float)
    gamma = np.asarray(res.ref_gamma, dtype=float)
    f = beta**2 * gamma * res.mass_mev * 1e-3
    return (np.asarray(res.s, dtype=float),
            S[:, 0, 5] / S[:, 5, 5] * f,
            S[:, 2, 5] / S[:, 5, 5] * f)


def _compare(lat, initial, eta0):
    res = EnvelopeSolver(lat, _ref(), dict(initial), current=0.0).run()
    s_env, dx_env, dy_env = _statistical_disp(res)
    out = dispersion_along_s(lat, _ref(), eta0=eta0)
    assert out["complete"]
    dx_i = np.interp(out["s"], s_env, dx_env)
    dy_i = np.interp(out["s"], s_env, dy_env)
    np.testing.assert_allclose(out["disp_x_m"], dx_i, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(out["disp_y_m"], dy_i, rtol=1e-7, atol=1e-9)
    return out


def test_static_line_matches_statistical_dispersion():
    out = _compare(_bend_line(), INITIAL, eta0=None)
    # the bends must actually produce dispersion, or the test is vacuous
    assert np.nanmax(np.abs(out["disp_x_m"])) > 1e-3


def test_seeded_dispersion_matches():
    eta0 = (120.0, 15.0, -40.0, 5.0)          # mm/MeV, mrad/MeV
    seeded = dict(INITIAL, disp_x=eta0[0], disp_xp=eta0[1],
                  disp_y=eta0[2], disp_yp=eta0[3])
    out = _compare(_bend_line(), seeded, eta0=eta0)
    assert abs(out["disp_x_m"][0]) > 1e-4      # entrance value survives


def test_dispersion_free_lattice_exact_zero():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, aperture=20.0, n_steps=5))
    lat.add(Drift("D1", 200.0, aperture=50.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, aperture=20.0, n_steps=5))
    out = dispersion_along_s(lat, _ref())
    assert out["complete"]
    assert np.all(out["disp_x_m"] == 0.0)
    assert np.all(out["disp_y_m"] == 0.0)


def test_rf_cavity_stays_finite_and_complete():
    lat = _bend_line(1)
    lat.add(RFGap("GAP", voltage=0.4, phase=-30.0, frequency=352.21,
                  ttf=0.85))
    lat.add(Drift("D3", 150.0, aperture=50.0))
    out = dispersion_along_s(lat, _ref())
    assert out["complete"]
    assert np.all(np.isfinite(out["disp_x_m"]))
    assert np.all(np.isfinite(out["disp_y_m"]))


def test_unsupported_element_breaks_chain_not_run():
    class _Alien:
        name = "ALIEN"
        length = 100.0

    lat = _bend_line(1)
    n_good = len(lat.elements)
    lat.elements.append(_Alien())
    lat.add(Drift("TAIL", 300.0, aperture=50.0))
    out = dispersion_along_s(lat, _ref())
    assert not out["complete"]
    assert np.isfinite(out["disp_x_m"][n_good])          # last good exit
    assert np.isnan(out["disp_x_m"][n_good + 1])         # alien exit: NaN
    assert out["s"][-1] > out["s"][n_good]               # s grid still spans
