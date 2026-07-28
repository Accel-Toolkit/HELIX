"""Matcher: scipy-based optimizer for beam optics matching."""
import numpy as np


class Matcher:
    """Optimize lattice element parameters to satisfy beam optics objectives.

    Parameters
    ----------
    lattice : Lattice
        Lattice object whose elements will be modified in-place during matching.
    ref : ReferenceParticle
        Initial reference particle state. A copy is used internally during
        each function evaluation so the original is never mutated.
    mode : str
        'envelope' (default) -- uses transfer-matrix Twiss, very fast.
        'multiparticle' -- interface reserved; currently falls back to envelope.
    """

    def __init__(self, lattice, ref, mode="envelope"):
        self.lattice = lattice
        self.ref = ref
        self.mode = mode
        self._variables = []
        self._objectives = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_variable(self, element_name, parameter, min_val, max_val):
        """Register an adjustable lattice parameter.

        Parameters
        ----------
        element_name : str
            Name of the element in the lattice (e.g. 'QUAD_001').
        parameter : str
            Attribute name on the element (e.g. 'gradient').
        min_val, max_val : float
            Optimiser bounds for the parameter.
        """
        element = self.lattice.get_element(element_name)
        self._variables.append({
            "element": element,
            "element_name": element_name,
            "parameter": parameter,
            "min": min_val,
            "max": max_val,
        })

    def add_objective(self, location, quantity, target):
        """Register a matching objective.

        Parameters
        ----------
        location : str
            'END' or the name of a specific element (currently only 'END' is
            supported in envelope mode).
        quantity : str
            One of 'alpha_x', 'beta_x', 'alpha_y', 'beta_y', 'mu_x', 'mu_y'.
        target : float
            Desired value for the quantity.
        """
        self._objectives.append({
            "location": location,
            "quantity": quantity,
            "target": target,
        })

    def solve(self, method="least_squares"):
        """Run the optimiser to satisfy all objectives.

        Parameters
        ----------
        method : str
            'least_squares' -- Levenberg-Marquardt (scipy.optimize.least_squares).
            'nelder_mead'   -- Nelder-Mead simplex (scipy.optimize.minimize).
            'differential_evolution' -- global, gradient-free.

        Returns
        -------
        dict with keys:
            'success'   : bool
            'variables' : dict of {element_name.parameter: optimised_value}
            'residuals' : list of residuals (actual - target) for each objective
        """
        if not self._variables:
            raise ValueError("No variables have been registered. Call add_variable() first.")
        if not self._objectives:
            raise ValueError("No objectives have been registered. Call add_objective() first.")

        # Save originals so we can restore on failure
        originals = self._save_originals()

        # Build initial guess and bounds
        x0 = np.array([getattr(v["element"], v["parameter"]) for v in self._variables],
                       dtype=float)
        bounds_lower = np.array([v["min"] for v in self._variables], dtype=float)
        bounds_upper = np.array([v["max"] for v in self._variables], dtype=float)

        # Clip initial guess to be inside bounds
        x0 = np.clip(x0, bounds_lower, bounds_upper)

        result_success = False
        best_x = x0.copy()
        best_residuals = None

        try:
            if method == "least_squares":
                result_success, best_x, best_residuals = self._solve_least_squares(
                    x0, bounds_lower, bounds_upper
                )
            elif method == "nelder_mead":
                result_success, best_x, best_residuals = self._solve_nelder_mead(
                    x0, bounds_lower, bounds_upper
                )
            elif method == "differential_evolution":
                result_success, best_x, best_residuals = self._solve_differential_evolution(
                    bounds_lower, bounds_upper
                )
            else:
                raise ValueError(f"Unknown method '{method}'. "
                                 "Use 'least_squares', 'nelder_mead', or 'differential_evolution'.")

            if result_success:
                # Apply best parameters
                self._apply_params(best_x)
            else:
                # Restore originals on failure
                self._restore_originals(originals)
                # Still compute residuals at original values for reporting
                if best_residuals is None:
                    best_residuals = self._compute_residuals(x0)

        except Exception:
            self._restore_originals(originals)
            raise

        # Build the output dict
        variables_out = {
            f"{v['element_name']}.{v['parameter']}": getattr(v["element"], v["parameter"])
            for v in self._variables
        }

        return {
            "success": result_success,
            "variables": variables_out,
            "residuals": best_residuals,
        }

    # ------------------------------------------------------------------
    # Solver backends
    # ------------------------------------------------------------------

    def _solve_least_squares(self, x0, lower, upper):
        """Use scipy.optimize.least_squares (Levenberg-Marquardt via 'trf' with bounds)."""
        from scipy.optimize import least_squares

        bounds = (lower, upper)
        res = least_squares(
            self._compute_residuals,
            x0,
            bounds=bounds,
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=10000,
        )
        success = res.success or res.cost < 1e-10
        return success, res.x, list(res.fun)

    def _solve_nelder_mead(self, x0, lower, upper):
        """Use scipy.optimize.minimize with Nelder-Mead (no bounds, but we clip internally)."""
        from scipy.optimize import minimize

        def scalar_cost(x):
            # Project inside bounds before evaluating
            x_clipped = np.clip(x, lower, upper)
            residuals = self._compute_residuals(x_clipped)
            return float(np.dot(residuals, residuals))

        res = minimize(
            scalar_cost,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 50000, "maxfev": 50000},
        )
        best_x = np.clip(res.x, lower, upper)
        best_residuals = self._compute_residuals(best_x)
        success = res.success or res.fun < 1e-10
        return success, best_x, best_residuals

    def _solve_differential_evolution(self, lower, upper):
        """Use scipy.optimize.differential_evolution (global, gradient-free)."""
        from scipy.optimize import differential_evolution

        bounds_list = list(zip(lower, upper))

        def scalar_cost(x):
            residuals = self._compute_residuals(x)
            return float(np.dot(residuals, residuals))

        res = differential_evolution(
            scalar_cost,
            bounds_list,
            tol=1e-8,
            maxiter=10000,
            seed=42,
        )
        best_residuals = self._compute_residuals(res.x)
        success = res.success or res.fun < 1e-10
        return success, res.x, best_residuals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_residuals(self, x):
        """Set parameters x, evaluate objectives, return residual array."""
        self._apply_params(x)
        from linac_gen.matching.objectives import evaluate_objectives
        try:
            residuals = evaluate_objectives(self._objectives, self.lattice,
                                            self.ref, mode=self.mode)
        except ValueError:
            # Unstable lattice: return large residuals to push optimizer away
            residuals = [1e6] * len(self._objectives)
        return np.array(residuals, dtype=float)

    def _apply_params(self, x):
        """Write parameter values x to lattice elements."""
        for val, var in zip(x, self._variables):
            setattr(var["element"], var["parameter"], float(val))

    def _save_originals(self):
        """Return a list of original parameter values."""
        return [getattr(v["element"], v["parameter"]) for v in self._variables]

    def _restore_originals(self, originals):
        """Restore saved parameter values to lattice elements."""
        for orig, var in zip(originals, self._variables):
            setattr(var["element"], var["parameter"], orig)
