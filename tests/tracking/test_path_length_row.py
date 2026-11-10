"""Line-level physics of the bend's path-length row (added 2026-09-07).

Per-element agreement with MAD-X and TraceWin is pinned in
``tests/io/test_madx_conventions.py`` and
``tests/elements/test_dipole_tracewin_anchor.py``.  What those cannot catch is
a composition or edge error, so the checks here are cumulative: an exact
symplectic relation over a whole line, and the conservation of the six
dimensional eigen-emittances through a dispersive one.
"""
import numpy as np
import pytest

from linac_gen.cli.common import build_ref, load_lattice
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
from tests.dataguard import needs


BEND_LINE_MHZ = 352.21      # examples/bend_line.dat opens with this FREQ card


def _ref(w_kin=100.0):
    return ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=352.21)


def _mixed_line():
    """Both bend planes, a skew quad, pole faces, and every field-index branch."""
    lat = Lattice()
    lat.add(Drift("d1", length=500.0))
    lat.add(Dipole("b1", angle=12.0, rho=2000.0, e1=6.0, e2=6.0))
    lat.add(Quadrupole("q1", length=200.0, gradient=4.0, skew_angle=15.0))
    lat.add(Dipole("b2", angle=-9.0, rho=1500.0, field_index=0.4, hv=1))
    lat.add(Drift("d2", length=300.0))
    lat.add(Dipole("b3", angle=7.0, rho=1800.0, field_index=1.5))
    return lat


def _symplectic_r5(M, ref):
    """R51, R52 implied by the rest of the matrix.

    For a symplectic map the path-length row is not free: it is fixed by the
    transverse block and the dispersion column.  In HELIX's units the factor
    is k = 0.36 beta gamma m / lambda.
    """
    k = 0.36 * ref.beta * ref.gamma * ref.species.mass / ref.wavelength
    r51 = k * (M[0, 0] * M[1, 5] - M[1, 0] * M[0, 5]
               + M[2, 0] * M[3, 5] - M[3, 0] * M[2, 5])
    r52 = k * (M[0, 1] * M[1, 5] - M[1, 1] * M[0, 5]
               + M[2, 1] * M[3, 5] - M[3, 1] * M[2, 5])
    return r51, r52


def test_composed_line_obeys_the_symplectic_path_length_relation():
    ref = _ref()
    M = compute_transfer_matrix(_mixed_line(), ref.copy())
    r51, r52 = _symplectic_r5(M, ref)
    assert abs(M[4, 0]) > 1e-3 and abs(M[4, 1]) > 1e-3      # a real coupling
    assert M[4, 0] == pytest.approx(r51, rel=1e-12, abs=1e-14)
    assert M[4, 1] == pytest.approx(r52, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("deck, species, energy, frequency", [
    ("examples/bend_line.dat", "H-", 2.1226695, BEND_LINE_MHZ),
    ("examples/csr_chicane.dat", "H-", 2.1226695, 162.5),
])
def test_shipped_bend_decks_obey_the_relation(deck, species, energy, frequency):
    """The same relation on real decks, where edges and drifts compose, on the
    clock each deck runs at."""
    lat = load_lattice(deck)
    ref = build_ref(BeamConfig(species=species, energy=energy, frequency=frequency))
    M = compute_transfer_matrix(lat, ref.copy())
    r51, r52 = _symplectic_r5(M, ref)
    scale = max(abs(M[4, 0]), abs(M[4, 1]), 1.0)
    assert abs(M[4, 0] - r51) < 1e-11 * scale
    assert abs(M[4, 1] - r52) < 1e-11 * scale


@needs("examples/pipii/btl/btl.dat")
def test_the_btl_obeys_the_relation_over_958_elements():
    lat = load_lattice("examples/pipii/btl/btl.dat")
    ref = build_ref(BeamConfig(species="H-", energy=800.0, frequency=162.5))
    M = compute_transfer_matrix(lat, ref.copy())
    r51, r52 = _symplectic_r5(M, ref)
    assert abs(M[4, 0] - r51) < 1e-12 and abs(M[4, 1] - r52) < 1e-12


def _canonical(ref):
    """HELIX (mm, mrad, mm, mrad, deg, MeV) -> trace space (m, rad, m, rad, m, delta).

    HELIX's own units are not canonically conjugate, so symplecticity has to
    be judged after this diagonal rescaling.  (``eigenemittances`` has the
    same requirement, and at SI scale its degeneracy guard returns zeros,
    which is why symplecticity is checked directly here instead.)
    """
    b, g, m, wl_m = ref.beta, ref.gamma, ref.species.mass, ref.wavelength * 1e-3
    return np.diag([1e-3, 1e-3, 1e-3, 1e-3, -b * wl_m / 360.0, 1.0 / (b * b * g * m)])


def _symplectic_residual(M, ref):
    D = _canonical(ref)
    Msi = D @ M @ np.linalg.inv(D)
    S = np.zeros((6, 6))
    for i in (0, 2, 4):
        S[i, i + 1] = 1.0
        S[i + 1, i] = -1.0
    return np.abs(Msi.T @ S @ Msi - S).max()


def test_a_bend_line_map_is_symplectic_and_would_not_be_without_the_row():
    """The strongest statement available: in canonical coordinates the whole
    map obeys M^T S M = S.  Zeroing the path-length row breaks it by ten
    orders of magnitude, so the check has teeth."""
    ref = _ref()
    M = compute_transfer_matrix(_mixed_line(), ref.copy())
    assert _symplectic_residual(M, ref) < 1e-12

    M_without = M.copy()
    M_without[4, 0] = M_without[4, 1] = M_without[4, 2] = M_without[4, 3] = 0.0
    assert _symplectic_residual(M_without, ref) > 1e-3


def test_shipped_deck_map_is_symplectic():
    lat = load_lattice("examples/bend_line.dat")
    ref = build_ref(BeamConfig(species="H-", energy=2.1226695, frequency=BEND_LINE_MHZ))
    assert _symplectic_residual(compute_transfer_matrix(lat, ref.copy()), ref) < 1e-12


def test_a_beam_on_another_clock_takes_the_cards_factor_on_entry():
    """A beam arriving at 162.5 MHz changes clock at the deck's first card.  Every
    element sits after that card, so the map is the deck's own map times the card's
    coordinate change: the Δφ column scales by f₂/f₁, and so does the determinant.
    This is not a failure of symplecticity: Δφ is in degrees of a different clock."""
    lat = load_lattice("examples/bend_line.dat")
    own = compute_transfer_matrix(
        lat, build_ref(BeamConfig(species="H-", energy=2.1226695, frequency=BEND_LINE_MHZ)))
    jumped = compute_transfer_matrix(
        lat, build_ref(BeamConfig(species="H-", energy=2.1226695, frequency=162.5)))
    D = np.eye(6)
    D[4, 4] = BEND_LINE_MHZ / 162.5
    np.testing.assert_allclose(jumped, own @ D, rtol=1e-13, atol=1e-13)
    assert np.linalg.det(jumped) == pytest.approx(BEND_LINE_MHZ / 162.5, rel=1e-9)
    assert np.linalg.det(own) == pytest.approx(1.0, rel=1e-9)


def test_the_projected_longitudinal_emittance_grows_from_a_real_correlation():
    """A bend correlates position with energy, so the PROJECTED longitudinal
    emittance grows through a dispersive line - x4.9 for this sigma0 on
    ``bend_line.dat``.  That is a projection, not growth: the map is
    symplectic (above) and the phase-space volume det(Sigma) is conserved.
    Read the two together before reporting an emittance blow-up.
    """
    lat = load_lattice("examples/bend_line.dat")
    ref = build_ref(BeamConfig(species="H-", energy=2.1226695, frequency=BEND_LINE_MHZ))
    M = compute_transfer_matrix(lat, ref.copy())

    sigma0 = np.diag([1.0, 0.25, 1.0, 0.25, 4.0, 0.01])   # uncoupled, HELIX units
    sigma1 = M @ sigma0 @ M.T

    proj = lambda s, i: float(np.sqrt(np.linalg.det(s[i:i + 2, i:i + 2])))
    assert proj(sigma1, 4) > 2.0 * proj(sigma0, 4)        # longitudinal projection
    assert proj(sigma1, 0) > 2.0 * proj(sigma0, 0)        # so does the transverse, from dispersion
    assert np.linalg.det(sigma1) == pytest.approx(np.linalg.det(sigma0), rel=1e-9)
    assert abs(M[4, 0]) > 0.0                             # the row is what couples them


def test_the_manuals_figures_for_that_growth():
    """Two manual pages quote these, for a beam the deck only half names: it gives
    the species and the energy, and the factor depends on the input beam.  Two modes
    that share no code for this agree, and nothing is lost.  (At 100 000 particles the
    spread from seed to seed is 0.03; at 2000 it is 0.17, and seed 42 reads 10.97.)"""
    from pathlib import Path

    from linac_gen.cli.common import run_envelope_sim, run_mp_sim
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.step_config import StepConfig

    cfg = BeamConfig(species="proton", energy=100.0, frequency=BEND_LINE_MHZ,
                     current=0.0, n_particles=100_000)
    ez = np.asarray(run_envelope_sim(load_lattice("examples/bend_line.dat"), cfg).emit_z)
    assert ez[-1] / ez[0] == pytest.approx(10.71, rel=1e-3)
    rec, beam = run_mp_sim(load_lattice("examples/bend_line.dat"), cfg,
                           SpaceChargeConfig(nx=24, ny=24, nz=24), StepConfig(), seed=42)
    assert beam.n_alive == cfg.n_particles
    ez = np.asarray(rec.emit_z)
    assert ez[-1] / ez[0] == pytest.approx(10.71, rel=0.01)

    root = Path(__file__).resolve().parents[2] / "docs" / "manual"
    dipole = (root / "03_elements" / "04_dipole.md").read_text(encoding="utf-8")
    limits = (root / "12_validation" / "03_known_limitations.md").read_text(encoding="utf-8")
    assert "×10.7" in dipole and "×10.7" in limits
    assert "×93" not in dipole + limits and "×53" not in dipole and "×11" not in dipole + limits


# ---------------------------------------------------------------------------
# Downstream behaviour change: compute_twiss(M, "z") on a dispersive lattice
# ---------------------------------------------------------------------------

def test_compute_twiss_z_refuses_a_bend_lattice_and_offers_the_projection():
    """A bend couples x to (dphi, dW), so the 2x2 (4,5) trace is no longer the
    longitudinal mode and ``compute_twiss`` refuses it at the default tol.

    That refusal is the correct behaviour, not a regression: the longitudinal
    normal mode of a dispersive lattice is only defined once the dispersion is
    transformed away, which HELIX does not do -- the raw 2x2 would be a
    projection reported as a tune.  It is also, measurably, not a loss: across
    the 20 shipped decks, every period ``detect_periods`` finds (29 of them
    with a bend) and every whole-lattice matrix, mu_x / mu_y / mu_z / beta_z
    are identical before and after this change -- 157 values equal, 0 gained,
    0 lost.  A bend-containing period in a transport line has a shear (4,5)
    block, so it was already refused, as unstable; only the message moved.

    Callers who knowingly want the projection ask for it by name, which is
    what ``phase_advance.py`` does at its coupled-lattice call sites.
    """
    from linac_gen.tracking.matrix_tracking import compute_twiss
    ref = _ref()
    M = compute_transfer_matrix(_mixed_line(), ref.copy())

    with pytest.raises(ValueError, match="coupled to plane z"):
        compute_twiss(M, "z")

    # Lift the tolerance and the answer is exactly the pre-change one: an
    # RF-free line's (4,5) block is a shear (M[4,4] = M[5,5] = 1, M[5,4] = 0),
    # so cos(mu) = 1 and it was refused as unstable before this change too.
    # That is the mechanism behind "0 lost" above -- there was no number here
    # to lose.  A period needs RF *and* a bend for the tolerance to matter.
    assert M[4, 4] == 1.0 and M[5, 5] == 1.0 and M[5, 4] == 0.0
    with pytest.raises(ValueError, match="Unstable"):
        compute_twiss(M, "z", coupling_tol=1e3)

    # The x and y guards are untouched: a bend couples x to (dphi, dW), never
    # x to y, so the transverse off-plane blocks a bend line feeds them are
    # still exactly zero.  (_mixed_line carries a skew quad, which does couple
    # x to y, so this uses a line without one.)
    plain = Lattice()
    plain.add(Drift("d1", length=400.0))
    plain.add(Dipole("b1", angle=12.0, rho=2000.0, e1=6.0, e2=6.0))
    plain.add(Quadrupole("q1", length=200.0, gradient=4.0))
    plain.add(Dipole("b2", angle=-8.0, rho=1600.0, field_index=0.4))
    Mp = compute_transfer_matrix(plain, ref.copy())
    assert abs(Mp[4, 0]) > 1e-3                              # the row is live
    for i, j in ((0, 2), (0, 3), (1, 2), (1, 3),
                 (2, 0), (2, 1), (3, 0), (3, 1)):
        assert Mp[i, j] == 0.0, f"a bend leaked x-y coupling into M[{i},{j}]"
    with pytest.raises(ValueError, match="coupled to plane z"):
        compute_twiss(Mp, "z")


def test_a_bend_free_line_is_untouched_by_the_coupling_check():
    """The guard must not fire on lattices that have no bend: those matrices
    are bit-identical across this change, so their z Twiss must be too."""
    from linac_gen.tracking.matrix_tracking import compute_twiss
    lat = Lattice()
    lat.add(Drift("d1", length=500.0))
    lat.add(Quadrupole("q1", length=200.0, gradient=4.0))
    lat.add(Drift("d2", length=500.0))
    lat.add(Quadrupole("q2", length=200.0, gradient=-4.0))
    ref = _ref()
    M = compute_transfer_matrix(lat, ref.copy())
    assert M[4, 0] == 0.0 and M[4, 1] == 0.0
    for plane in ("x", "y"):
        compute_twiss(M, plane)
    # Longitudinally a drift-like line is a shear: unstable, as it always was.
    with pytest.raises(ValueError, match="Unstable"):
        compute_twiss(M, "z")
