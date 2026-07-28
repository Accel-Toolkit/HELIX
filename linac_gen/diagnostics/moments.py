"""Statistical moment computation from particle arrays."""
import numpy as np


def compute_moments(particles: np.ndarray) -> dict:
    if len(particles) == 0:
        return {
            "mean": np.zeros(6), "sigma_matrix": np.zeros((6, 6)),
            "sigma_x": 0.0, "sigma_xp": 0.0, "sigma_y": 0.0,
            "sigma_yp": 0.0, "sigma_phi": 0.0, "sigma_w": 0.0,
        }
    mean = np.mean(particles, axis=0)
    centered = particles - mean
    sigma = (centered.T @ centered) / len(particles)
    return {
        "mean": mean, "sigma_matrix": sigma,
        "sigma_x": np.sqrt(sigma[0, 0]), "sigma_xp": np.sqrt(sigma[1, 1]),
        "sigma_y": np.sqrt(sigma[2, 2]), "sigma_yp": np.sqrt(sigma[3, 3]),
        "sigma_phi": np.sqrt(sigma[4, 4]), "sigma_w": np.sqrt(sigma[5, 5]),
    }


def compute_emittance(particles: np.ndarray, plane: str = "x") -> float:
    if len(particles) == 0:
        return 0.0
    col_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    i, j = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    up = particles[:, j] - np.mean(particles[:, j])
    uu = np.mean(u * u)
    upup = np.mean(up * up)
    uup = np.mean(u * up)
    emit_sq = uu * upup - uup * uup
    # emit^2 is mathematically >= 0. Round-off can make it slightly negative
    # for very small samples or near-singular distributions; clamp to 0 rather
    # than taking abs() so a genuinely large negative would surface as zero
    # (loud via downstream "emit == 0" guards) instead of being silently flipped.
    return float(np.sqrt(max(emit_sq, 0.0)))


def compute_twiss_from_particles(particles: np.ndarray, plane: str = "x") -> dict:
    if len(particles) == 0:
        return {"alpha": 0.0, "beta": 0.0, "gamma_t": 0.0, "emittance": 0.0}
    col_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    i, j = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    up = particles[:, j] - np.mean(particles[:, j])
    emit = compute_emittance(particles, plane)
    if emit < 1e-30:
        return {"alpha": 0.0, "beta": 0.0, "gamma_t": 0.0, "emittance": 0.0}
    beta = np.mean(u * u) / emit
    alpha = -np.mean(u * up) / emit
    gamma_t = np.mean(up * up) / emit
    return {"alpha": float(alpha), "beta": float(beta), "gamma_t": float(gamma_t), "emittance": float(emit)}


def compute_halo(particles: np.ndarray, plane: str = "x") -> float:
    if len(particles) == 0:
        return 0.0
    col_map = {"x": 0, "y": 2, "z": 4}
    i = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    u2 = np.mean(u**2)
    u4 = np.mean(u**4)
    if u2 < 1e-30:
        return 0.0
    return float(u4 / u2**2 - 1.0)
