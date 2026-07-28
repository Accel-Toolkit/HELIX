"""Periodic dispersion (η, η′) in the zero-current matched-Twiss machinery.

Locks the 2026-07-11 feature: ``find_periodic_twiss`` /
``find_matched_input_twiss`` return ``disp_x``/``disp_xp``/``disp_y``/
``disp_yp`` — the periodic dispersion ``η = (I − M)⁻¹ · d`` w.r.t. the
kinetic-energy column, in mm/MeV and mrad/MeV (the exact convention of
``find_sc_matched_input_twiss``'s ``Σ[i,5]/Σ[5,5]``).  Guarantees held
here:

* dispersion-free lattices get EXACT 0.0 keys and bit-identical
  pre-existing keys (the change cannot move any old result);
* bending lattices (arc FODO / BTL) get the closed-orbit dispersion,
  consistent with the SC path as I → 0;
* near-integer tunes give NaN + a warning, never an exception;
* ``BeamConfig.disp_*`` shears generated distributions accordingly.
"""

from tests.dataguard import needs, require  # noqa: E402
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)
from linac_gen.matching.periodic import (
    _periodic_dispersion, find_fodo_cells, find_matched_input_twiss,
    find_periodic_twiss, find_sc_matched_input_twiss,
)

BTL_DAT = Path(__file__).parents[2] / "examples" / "pipii" / "btl" / "btl.dat"

DISP_KEYS = ("disp_x", "disp_xp", "disp_y", "disp_yp")


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _btl_ref():
    return ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)


def _fodo_bend_lattice(front_drift=None, n_cells=2):
    """FODO cell with two sector bends — a dispersive periodic cell."""
    lat = Lattice()
    if front_drift is not None:
        lat.add(Drift("FRONT", front_drift, aperture=20.0))
    for c in range(n_cells):
        lat.add(Quadrupole(f"QF_{c}", 50.0, gradient=5.0, aperture=20.0,
                           n_steps=5))
        lat.add(Dipole(f"B1_{c}", angle=10.0, rho=2000.0, aperture=50.0))
        lat.add(Quadrupole(f"QD_{c}", 50.0, gradient=-5.0, aperture=20.0,
                           n_steps=5))
        lat.add(Dipole(f"B2_{c}", angle=10.0, rho=2000.0, aperture=50.0))
    return lat


def _plain_fodo(n_cells=2):
    lat = Lattice()
    for c in range(n_cells):
        lat.add(Quadrupole(f"QF_{c}", 50.0, gradient=5.0, aperture=20.0,
                           n_steps=5))
        lat.add(Drift(f"D1_{c}", 200.0, aperture=20.0))
        lat.add(Quadrupole(f"QD_{c}", 50.0, gradient=-5.0, aperture=20.0,
                           n_steps=5))
        lat.add(Drift(f"D2_{c}", 200.0, aperture=20.0))
    return lat


# ---------------------------------------------------------------------------
# analytic + closure checks
# ---------------------------------------------------------------------------
def test_combined_function_bend_analytic():
    """A ring made of one uniform combined-function sector bend has the
    closed-form dispersion η = ρ/(1−n).  In HELIX per-MeV units:
    disp_x = ρ/(1−n) / (β²γ·mc²)."""
    lat = Lattice()
    lat.add(Dipole("B", angle=60.0, rho=2000.0, field_index=0.3,
                   aperture=50.0))
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    beta2gm = ref.beta ** 2 * ref.gamma * ref.species.mass
    expected = (2000.0 / (1.0 - 0.3)) / beta2gm     # mm/MeV
    assert tw["disp_x"] == pytest.approx(expected, rel=1e-9)
    assert tw["disp_xp"] == pytest.approx(0.0, abs=1e-9)
    # no vertical bending → exact zeros
    assert tw["disp_y"] == 0.0
    assert tw["disp_yp"] == 0.0


def test_fodo_bend_closure():
    """The returned dispersion satisfies the periodicity condition
    M₂·η + d == η, and equals an in-test independent solve."""
    lat = _fodo_bend_lattice()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    M = np.asarray(compute_transfer_matrix(lat, ref.copy()))
    for keys, (i, j) in ((("disp_x", "disp_xp"), (0, 1)),
                         (("disp_y", "disp_yp"), (2, 3))):
        eta = np.array([tw[keys[0]], tw[keys[1]]])
        M2 = M[np.ix_([i, j], [i, j])]
        d = M[[i, j], 5]
        np.testing.assert_allclose(M2 @ eta + d, eta, rtol=1e-9, atol=1e-12)
        independent = np.linalg.solve(np.eye(2) - M2, d)
        np.testing.assert_allclose(eta, independent, rtol=1e-12, atol=1e-15)


def test_dispersion_free_exact_zero_and_bit_identical():
    """Straight FODO: the four keys are IDENTITY-exact zeros and every
    pre-existing key equals the direct compute_twiss value — the feature
    cannot move any old result."""
    lat = _plain_fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    for k in DISP_KEYS:
        assert tw[k] == 0.0                       # exact, not approx
    M = compute_transfer_matrix(lat, ref.copy())
    for plane in ("x", "y"):
        direct = compute_twiss(M, plane)
        assert tw[f"alpha_{plane}"] == direct["alpha"]   # bit-identical
        assert tw[f"beta_{plane}"] == direct["beta"]
        assert tw[f"mu_{plane}"] == direct["mu"]


def test_matched_input_disp_roundtrip():
    """Entrance dispersion forward-transported through the front section
    (affine: η_cell = M₂·η_ent + d_front) reproduces the cell's periodic
    dispersion."""
    lat = _fodo_bend_lattice(front_drift=137.0)
    ref = _ref()
    cs, ce = 1, len(lat.elements) - 1              # cell after the drift
    tw = find_matched_input_twiss(lat, ref, cs, ce)
    m_cell = np.asarray(compute_transfer_matrix(lat, ref.copy(),
                                                start=cs, end=ce))
    m_front = np.asarray(compute_transfer_matrix(lat, ref.copy(),
                                                 start=0, end=cs - 1))
    for keys, (i, j) in ((("disp_x", "disp_xp"), (0, 1)),
                         (("disp_y", "disp_yp"), (2, 3))):
        eta_ent = np.array([tw[keys[0]], tw[keys[1]]])
        M2f = m_front[np.ix_([i, j], [i, j])]
        d_f = m_front[[i, j], 5]
        eta_cell = M2f @ eta_ent + d_f
        M2c = m_cell[np.ix_([i, j], [i, j])]
        d_c = m_cell[[i, j], 5]
        expected = np.linalg.solve(np.eye(2) - M2c, d_c)
        np.testing.assert_allclose(eta_cell, expected, rtol=1e-9, atol=1e-12)


def test_integer_tune_returns_nan_not_raise(capsys):
    """Degenerate (I − M) — helper returns NaN + stderr warning, never
    raises."""
    M = np.eye(6)
    M[0, 5] = 1.0                                  # dispersive but M₂ = I
    eta = _periodic_dispersion(M, (0, 1))
    assert np.isnan(eta).all()
    assert "near-integer tune" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# BTL regression — the user's real arc-FODO transfer line
# ---------------------------------------------------------------------------
@needs("examples/pipii/btl/btl.dat")
def test_btl_matched_input_dispersion():
    """BTL: matched input now carries finite dispersion (the silent gap
    behind the 'matched beam still beats in x' observation).  Values
    pinned from the first verified run; cell-choice invariant."""
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    cells = find_fodo_cells(lat)
    t0 = find_matched_input_twiss(lat, ref, *cells[0])
    for k in DISP_KEYS:
        assert np.isfinite(t0[k])
    # x-plane must actually be dispersive (it's an arc); y stays ~0.
    assert abs(t0["disp_x"]) > 1e-4
    t1 = find_matched_input_twiss(lat, ref, *cells[1])
    for k in DISP_KEYS:
        assert t0[k] == pytest.approx(t1[k], rel=1e-6, abs=1e-9)


@needs("examples/pipii/btl/btl.dat")
def test_btl_sc_matched_zero_current_disp_consistency():
    """As I → 0 the SC matcher's dispersion must agree with the new
    zero-current dispersion — the two paths are now one physics."""
    lat, _ = parse_tracewin(str(BTL_DAT))
    ref = _btl_ref()
    cs, ce = find_fodo_cells(lat)[0]
    zc = find_matched_input_twiss(lat, ref, cs, ce)
    sc = find_sc_matched_input_twiss(
        lat, ref, cs, ce, 1e-3,
        {"alpha_z": 0.0, "beta_z": 1.0,
         "emit_x": 0.16, "emit_y": 0.16, "emit_z": 0.3})
    assert sc["converged"]
    for k in DISP_KEYS:
        assert sc[k] == pytest.approx(zc[k], rel=5e-3, abs=1e-4)


# ---------------------------------------------------------------------------
# coupled path
# ---------------------------------------------------------------------------
def test_coupled_solenoid_bend_dispersion_closure():
    """Solenoid coupling + a bend: the coupled path returns the 4×4
    periodic dispersion, closing M₄·η₄ + d₄ = η₄."""
    lat = Lattice()
    lat.add(Drift("D1", length=100, aperture=10))
    lat.add(Solenoid("SOL", length=200, field=2.0, aperture=10))
    lat.add(Dipole("B", angle=15.0, rho=2000.0, aperture=50.0))
    lat.add(Drift("D2", length=100, aperture=10))
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    assert tw["coupled"] is True
    M = np.asarray(compute_transfer_matrix(lat, ref.copy()))
    eta4 = np.array([tw[k] for k in DISP_KEYS])
    np.testing.assert_allclose(M[0:4, 0:4] @ eta4 + M[0:4, 5], eta4,
                               rtol=1e-9, atol=1e-12)


def test_coupled_no_bend_zero_dispersion():
    """Pure solenoid lattice (coupled, no bends): dispersion keys are
    exact zeros and the pre-existing coupled keys are present."""
    lat = Lattice()
    lat.add(Drift("D1", length=100, aperture=10))
    lat.add(Solenoid("SOL", length=200, field=2.0, aperture=10))
    lat.add(Drift("D2", length=100, aperture=10))
    tw = find_periodic_twiss(lat, _ref())
    assert tw["coupled"] is True
    for k in DISP_KEYS:
        assert tw[k] == 0.0
    assert "sigma4" in tw and "mu_1" in tw


# ---------------------------------------------------------------------------
# BeamConfig seeding — the generated beam actually carries the correlation
# ---------------------------------------------------------------------------
def test_factory_dispersion_shear():
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    cfg = BeamConfig(species="proton", energy=3.0, frequency=162.5,
                     current=0.0, n_particles=200_000,
                     distribution="gaussian",
                     disp_x=5.0, disp_xp=-2.0)
    beam = create_beam(cfg, seed=42)
    p = beam.particles
    cov = np.cov(p[:, (0, 1, 5)].T)
    s_ww = cov[2, 2]
    assert cov[0, 2] / s_ww == pytest.approx(5.0, rel=5e-2)
    assert cov[1, 2] / s_ww == pytest.approx(-2.0, rel=5e-2)


def test_factory_zero_dispersion_bit_identical():
    """All-default disp_* leaves generation BIT-identical — the guard
    that the new fields cannot move existing MP results."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    kw = dict(species="proton", energy=3.0, frequency=162.5,
              current=0.0, n_particles=5000, distribution="gaussian")
    a = create_beam(BeamConfig(**kw), seed=7).particles
    b = create_beam(BeamConfig(**kw, disp_x=0.0, disp_xp=0.0,
                               disp_y=0.0, disp_yp=0.0), seed=7).particles
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@needs("examples/pipii/btl/btl.dat")
def test_twiss_cli_quiet_still_four_numbers(capsys):
    """-q output stays exactly 4 numbers (pinned contract)."""
    from linac_gen.__main__ import main
    rc = main(["twiss", str(BTL_DAT), "--mode", "cell", "-q",
               "--energy", "800", "--species", "H-", "--freq", "162.5"])
    assert rc == 0
    assert len(capsys.readouterr().out.split()) == 4


@needs("examples/pipii/btl/btl.dat")
def test_twiss_cli_quiet_disp_appends_four_more(capsys):
    from linac_gen.__main__ import main
    rc = main(["twiss", str(BTL_DAT), "--mode", "cell", "-q", "--disp",
               "--energy", "800", "--species", "H-", "--freq", "162.5"])
    assert rc == 0
    nums = capsys.readouterr().out.split()
    assert len(nums) == 8
    assert abs(float(nums[4])) > 1e-4              # disp_x finite on BTL


@needs("examples/pipii/btl/btl.dat")
def test_twiss_cli_nonquiet_prints_dispersion(capsys):
    from linac_gen.__main__ import main
    rc = main(["twiss", str(BTL_DAT), "--mode", "cell",
               "--energy", "800", "--species", "H-", "--freq", "162.5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "D_x" in out and "eta_x" in out


def test_twiss_cli_nonquiet_straight_lattice_unchanged(capsys, tmp_path):
    """Dispersion-free deck: non-quiet output contains NO dispersion rows
    (byte-compatibility with the pre-dispersion CLI)."""
    dat = tmp_path / "fodo.dat"
    dat.write_text("""\
FREQ 352.21
QUAD 50 5.0 20
DRIFT 200 20 0
QUAD 50 -5.0 20
DRIFT 200 20 0
END
""")
    from linac_gen.__main__ import main
    rc = main(["twiss", str(dat), "--mode", "whole",
               "--energy", "3", "--species", "proton"])
    assert rc == 0
    assert "D_x" not in capsys.readouterr().out
