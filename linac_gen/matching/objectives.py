"""Objective evaluation for the matching module."""


def evaluate_objectives(objectives, lattice, ref, mode="envelope"):
    """Evaluate all objectives and return a list of residuals (actual - target).

    Parameters
    ----------
    objectives : list of dict
        Each dict has keys 'location', 'quantity', and 'target'.
    lattice : Lattice
        The lattice to evaluate.
    ref : ReferenceParticle
        Initial reference particle state. A copy is used internally.
    mode : str
        'envelope' uses transfer-matrix Twiss (fast).
        'multiparticle' falls back to envelope for now.

    Returns
    -------
    list of float
        Residuals: actual_value - target for each objective.
    """
    if mode in ("envelope", "multiparticle"):
        from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, compute_twiss
        M = compute_transfer_matrix(lattice, ref.copy())
        twiss_x = compute_twiss(M, "x")
        twiss_y = compute_twiss(M, "y")
        values = {
            "alpha_x": twiss_x["alpha"],
            "beta_x": twiss_x["beta"],
            "alpha_y": twiss_y["alpha"],
            "beta_y": twiss_y["beta"],
            "mu_x": twiss_x["mu"],
            "mu_y": twiss_y["mu"],
        }
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'envelope' or 'multiparticle'.")

    residuals = []
    for obj in objectives:
        actual = values.get(obj["quantity"], 0.0)
        residuals.append(actual - obj["target"])
    return residuals
