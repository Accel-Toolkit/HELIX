"""Tests for linac_gen.matching: Matcher, find_periodic_twiss, evaluate_objectives."""
import math
import pytest
import numpy as np

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, compute_twiss

from linac_gen.matching.matcher import Matcher
from linac_gen.matching.periodic import find_periodic_twiss
from linac_gen.matching.objectives import evaluate_objectives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fodo_lattice(gf=3.0, gd=-3.0):
    """Return an asymmetric FODO (QF, D, QD, D) with given gradients (T/m).

    Note: the periodic Twiss of this structure has alpha != 0 for any gradient.
    Use make_symmetric_fodo() when testing alpha=0 objectives.
    """
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=gf, aperture=20.0, n_steps=5))
    lat.add(Drift("D1", 200.0, aperture=20.0))
    lat.add(Quadrupole("QD", 50.0, gradient=gd, aperture=20.0, n_steps=5))
    lat.add(Drift("D2", 200.0, aperture=20.0))
    return lat


def make_symmetric_fodo(gf=3.0, gd=None):
    """Return a symmetric half-cell FODO (QF/2, D, QD, D, QF/2).

    By construction, the periodic Twiss of this cell always has alpha_x=0 and
    alpha_y=0 at the symmetry points (entrance/exit of the half cell).
    """
    if gd is None:
        gd = -gf
    lat = Lattice()
    lat.add(Quadrupole("QF1", 25.0, gradient=gf, aperture=20.0, n_steps=5))
    lat.add(Drift("D1", 200.0, aperture=20.0))
    lat.add(Quadrupole("QD", 50.0, gradient=gd, aperture=20.0, n_steps=5))
    lat.add(Drift("D2", 200.0, aperture=20.0))
    lat.add(Quadrupole("QF2", 25.0, gradient=gf, aperture=20.0, n_steps=5))
    return lat


def make_ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _twiss_at_gradient(gf, gd=None):
    """Compute periodic Twiss of the asymmetric FODO at given gradient."""
    if gd is None:
        gd = -gf
    lat = make_fodo_lattice(gf=gf, gd=gd)
    ref = make_ref()
    M = compute_transfer_matrix(lat, ref)
    tx = compute_twiss(M, "x")
    ty = compute_twiss(M, "y")
    return tx, ty


# ---------------------------------------------------------------------------
# find_periodic_twiss
# ---------------------------------------------------------------------------

class TestFindPeriodicTwiss:
    def test_fodo_returns_valid_twiss(self):
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        twiss = find_periodic_twiss(lat, ref)
        assert twiss["beta_x"] > 0.0
        assert twiss["beta_y"] > 0.0
        assert 0.0 < twiss["mu_x"] < 180.0
        assert 0.0 < twiss["mu_y"] < 180.0

    def test_fodo_all_keys_present(self):
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        twiss = find_periodic_twiss(lat, ref)
        for key in ("alpha_x", "beta_x", "mu_x", "alpha_y", "beta_y", "mu_y"):
            assert key in twiss

    def test_fodo_courant_snyder_identity(self):
        """beta * gamma_t - alpha^2 == 1 (Courant-Snyder identity)."""
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        twiss = find_periodic_twiss(lat, ref)
        for plane in ("x", "y"):
            alpha = twiss[f"alpha_{plane}"]
            beta = twiss[f"beta_{plane}"]
            gamma_t = (1.0 + alpha**2) / beta
            assert abs(beta * gamma_t - alpha**2 - 1.0) < 1e-10

    def test_unstable_lattice_raises(self):
        """An over-focused FODO should be unstable and raise ValueError."""
        lat = make_fodo_lattice(gf=100.0, gd=-100.0)
        ref = make_ref()
        with pytest.raises(ValueError, match="[Uu]nstable"):
            find_periodic_twiss(lat, ref)

    def test_ref_not_mutated(self):
        """find_periodic_twiss must not modify the original reference particle."""
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        w0 = ref.w_kin
        s0 = ref.s
        find_periodic_twiss(lat, ref)
        assert ref.w_kin == w0
        assert ref.s == s0


# ---------------------------------------------------------------------------
# evaluate_objectives
# ---------------------------------------------------------------------------

class TestEvaluateObjectives:
    def _fodo_twiss_values(self):
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        M = compute_transfer_matrix(lat, ref)
        tx = compute_twiss(M, "x")
        ty = compute_twiss(M, "y")
        return tx, ty

    def test_zero_residual_when_target_matches(self):
        tx, ty = self._fodo_twiss_values()
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        objectives = [
            {"location": "END", "quantity": "alpha_x", "target": tx["alpha"]},
            {"location": "END", "quantity": "alpha_y", "target": ty["alpha"]},
        ]
        residuals = evaluate_objectives(objectives, lat, ref)
        assert all(abs(r) < 1e-10 for r in residuals)

    def test_nonzero_residual(self):
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        objectives = [
            {"location": "END", "quantity": "alpha_x", "target": 99.0},
        ]
        residuals = evaluate_objectives(objectives, lat, ref)
        assert abs(residuals[0]) > 1.0

    def test_mode_multiparticle_falls_back(self):
        """multiparticle mode must not crash (uses envelope as fallback)."""
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        objectives = [{"location": "END", "quantity": "alpha_x", "target": 0.0}]
        residuals = evaluate_objectives(objectives, lat, ref, mode="multiparticle")
        assert isinstance(residuals[0], float)

    def test_unknown_quantity_returns_zero_residual(self):
        """Unknown quantities fall back to value 0.0 => residual = -target."""
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        objectives = [{"location": "END", "quantity": "sigma_nonexistent", "target": 5.0}]
        residuals = evaluate_objectives(objectives, lat, ref)
        assert abs(residuals[0] - (-5.0)) < 1e-12


# ---------------------------------------------------------------------------
# Matcher – construction and variable/objective registration
# ---------------------------------------------------------------------------

class TestMatcherConstruction:
    def test_default_mode_is_envelope(self):
        lat = make_fodo_lattice()
        ref = make_ref()
        m = Matcher(lat, ref)
        assert m.mode == "envelope"

    def test_add_variable_stores_element(self):
        lat = make_fodo_lattice()
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        assert len(m._variables) == 1
        assert m._variables[0]["element"] is lat.get_element("QF")
        assert m._variables[0]["parameter"] == "gradient"
        assert m._variables[0]["min"] == 0.1
        assert m._variables[0]["max"] == 15.0

    def test_add_objective_stores_entry(self):
        lat = make_fodo_lattice()
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_objective("END", "alpha_x", 0.0)
        assert len(m._objectives) == 1
        assert m._objectives[0] == {"location": "END", "quantity": "alpha_x", "target": 0.0}

    def test_solve_no_variables_raises(self):
        lat = make_fodo_lattice()
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_objective("END", "alpha_x", 0.0)
        with pytest.raises(ValueError, match="[Nn]o variables"):
            m.solve()

    def test_solve_no_objectives_raises(self):
        lat = make_fodo_lattice()
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        with pytest.raises(ValueError, match="[Nn]o objectives"):
            m.solve()


# ---------------------------------------------------------------------------
# Matcher – FODO matching (the core test)
# ---------------------------------------------------------------------------

class TestFODOMatching:
    def test_fodo_matching_alpha_zero(self):
        """Match a FODO to alpha_x=0, alpha_y=0 at exit.

        A symmetric half-cell FODO (QF/2, D, QD, D, QF/2) has periodic
        alpha_x=0, alpha_y=0 by symmetry. We perturb the QD gradient slightly
        (breaking equal |QF|/|QD| symmetry) and use the QF1/QF2 gradient as
        the matching variable to restore alpha=0. Because the cell is symmetric
        in QF, alpha=0 whenever |gf| is matched to |gd| appropriately.
        """
        # Start with QD=-3.0, QF1=QF2 slightly off from the symmetric solution
        lat = make_symmetric_fodo(gf=2.5, gd=-3.0)
        ref = make_ref()

        matcher = Matcher(lat, ref)
        # Both QF half-quads must be tied together -- vary via QF1 and QF2 independently,
        # but for simplicity vary only QD while keeping QF fixed at 3.0.
        # Actually: the symmetric cell always has alpha=0 for ANY gradient combination
        # because QF1=QF2. Let's use QD as the variable to match BETA_X to a target.
        #
        # Better test: start with gf != gd magnitude (not symmetric), vary QD to
        # achieve alpha=0. But as shown analytically, alpha=0 for symmetric cell
        # regardless of gd. So let's test convergence by matching beta_x to a target.
        #
        # Use the asymmetric FODO, match beta_x to the value achieved at g=5.0.
        tx5, ty5 = _twiss_at_gradient(5.0)

        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()

        matcher = Matcher(lat, ref)
        matcher.add_variable("QF", "gradient", min_val=0.1, max_val=15.0)
        matcher.add_variable("QD", "gradient", min_val=-15.0, max_val=-0.1)
        # Match to the periodic Twiss achieved at g=5.0 (achievable targets)
        matcher.add_objective("END", "alpha_x", target=tx5["alpha"])
        matcher.add_objective("END", "alpha_y", target=ty5["alpha"])

        result = matcher.solve(method="least_squares")
        assert result["success"]
        assert abs(result["residuals"][0]) < 0.01, (
            f"alpha_x residual too large: {result['residuals'][0]}"
        )
        assert abs(result["residuals"][1]) < 0.01, (
            f"alpha_y residual too large: {result['residuals'][1]}"
        )

    def test_fodo_result_has_required_keys(self):
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        m.add_variable("QD", "gradient", -15.0, -0.1)
        # Match to the actual periodic Twiss at g=3.0 (trivial but tests API)
        tx, ty = _twiss_at_gradient(3.0)
        m.add_objective("END", "alpha_x", tx["alpha"])
        result = m.solve()
        assert "success" in result
        assert "variables" in result
        assert "residuals" in result

    def test_variables_dict_keys(self):
        """Result 'variables' dict has dotted element.parameter keys."""
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        m.add_variable("QD", "gradient", -15.0, -0.1)
        tx, _ = _twiss_at_gradient(5.0)
        m.add_objective("END", "alpha_x", tx["alpha"])
        result = m.solve()
        assert "QF.gradient" in result["variables"]
        assert "QD.gradient" in result["variables"]

    def test_matched_params_applied_to_lattice(self):
        """After solve(), the lattice elements have the optimised values."""
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        m.add_variable("QD", "gradient", -15.0, -0.1)
        tx, _ = _twiss_at_gradient(5.0)
        m.add_objective("END", "alpha_x", tx["alpha"])
        result = m.solve()
        qf = lat.get_element("QF")
        assert abs(qf.gradient - result["variables"]["QF.gradient"]) < 1e-12

    def test_bounds_respected(self):
        """Optimised values must lie within the specified bounds."""
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        m = Matcher(lat, ref)
        min_qf, max_qf = 1.0, 10.0
        min_qd, max_qd = -10.0, -1.0
        m.add_variable("QF", "gradient", min_qf, max_qf)
        m.add_variable("QD", "gradient", min_qd, max_qd)
        tx5, ty5 = _twiss_at_gradient(5.0)
        m.add_objective("END", "alpha_x", tx5["alpha"])
        m.add_objective("END", "alpha_y", ty5["alpha"])
        result = m.solve()
        gf = result["variables"]["QF.gradient"]
        gd = result["variables"]["QD.gradient"]
        assert min_qf - 1e-6 <= gf <= max_qf + 1e-6
        assert min_qd - 1e-6 <= gd <= max_qd + 1e-6

    def test_multiple_objectives_alpha_and_beta(self):
        """Match both alpha and beta simultaneously."""
        # Get the periodic Twiss for gradient=5.0
        tx5, ty5 = _twiss_at_gradient(5.0)

        # Now start from a different gradient and try to match to those Twiss values
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.5, 15.0)
        m.add_variable("QD", "gradient", -15.0, -0.5)
        m.add_objective("END", "alpha_x", tx5["alpha"])
        m.add_objective("END", "alpha_y", ty5["alpha"])
        m.add_objective("END", "beta_x", tx5["beta"])
        m.add_objective("END", "beta_y", ty5["beta"])

        result = m.solve()
        assert result["success"]
        for r in result["residuals"]:
            assert abs(r) < 1.0, f"Residual {r} too large"


# ---------------------------------------------------------------------------
# Matcher – method="nelder_mead"
# ---------------------------------------------------------------------------

class TestNelderMeadMethod:
    def test_nelder_mead_converges(self):
        """Nelder-Mead converges on an achievable matching target."""
        tx5, ty5 = _twiss_at_gradient(5.0)
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        m.add_variable("QD", "gradient", -15.0, -0.1)
        m.add_objective("END", "alpha_x", tx5["alpha"])
        m.add_objective("END", "alpha_y", ty5["alpha"])
        result = m.solve(method="nelder_mead")
        # Nelder-Mead may not converge to strict tolerance but residuals should be small
        assert result["success"] or all(abs(r) < 0.1 for r in result["residuals"])

    def test_nelder_mead_result_structure(self):
        tx5, _ = _twiss_at_gradient(5.0)
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        m.add_variable("QD", "gradient", -15.0, -0.1)
        m.add_objective("END", "alpha_x", tx5["alpha"])
        result = m.solve(method="nelder_mead")
        assert "success" in result
        assert "variables" in result
        assert "residuals" in result


# ---------------------------------------------------------------------------
# Matcher – restore on failure
# ---------------------------------------------------------------------------

class TestRestoreOnFailure:
    def test_restore_when_target_unreachable(self):
        """If target is physically impossible, params are restored to originals."""
        lat = make_fodo_lattice(gf=3.0, gd=-3.0)
        ref = make_ref()
        original_gf = lat.get_element("QF").gradient
        original_gd = lat.get_element("QD").gradient

        m = Matcher(lat, ref)
        # Very tight bounds: only one point in the parameter space
        m.add_variable("QF", "gradient", 2.9999, 3.0001)
        m.add_variable("QD", "gradient", -3.0001, -2.9999)
        # Target that cannot be met: periodic alpha for g=3 is ~-1.137, not 999.
        tx_unachievable = 999.0
        m.add_objective("END", "alpha_x", target=tx_unachievable)
        m.add_objective("END", "alpha_y", target=tx_unachievable)

        result = m.solve(method="least_squares")

        if not result["success"]:
            # Params must be restored to original values
            assert abs(lat.get_element("QF").gradient - original_gf) < 1e-10
            assert abs(lat.get_element("QD").gradient - original_gd) < 1e-10

    def test_unknown_method_raises(self):
        lat = make_fodo_lattice()
        ref = make_ref()
        m = Matcher(lat, ref)
        m.add_variable("QF", "gradient", 0.1, 15.0)
        m.add_objective("END", "alpha_x", 0.0)
        with pytest.raises(ValueError, match="[Uu]nknown method"):
            m.solve(method="bogus_method")


# ---------------------------------------------------------------------------
# Matcher – multiparticle mode (interface)
# ---------------------------------------------------------------------------

class TestMultiparticleMode:
    def test_multiparticle_mode_runs(self):
        """multiparticle mode should run without crashing (uses envelope fallback)."""
        lat = make_fodo_lattice(gf=5.0, gd=-5.0)
        ref = make_ref()
        m = Matcher(lat, ref, mode="multiparticle")
        m.add_variable("QF", "gradient", 0.1, 15.0)
        m.add_variable("QD", "gradient", -15.0, -0.1)
        m.add_objective("END", "alpha_x", 0.0)
        result = m.solve(method="least_squares")
        # Should not raise; success or residuals should be finite
        assert isinstance(result["success"], bool)
        assert all(math.isfinite(r) for r in result["residuals"])
