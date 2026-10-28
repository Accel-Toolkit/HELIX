"""Linear normal form of a one-turn matrix.

For a stable symplectic matrix M (4×4 or 6×6, pairs of canonical coordinates)
find W and the mode phases μ_k so that M = W R W⁻¹, R = blockdiag of
[[cos μ, sin μ], [−sin μ, cos μ]], with W symplectic and in the
Courant-Snyder form: W's position-of-mode-k entry of its second column is
zero, so for uncoupled motion W's 2×2 blocks are
[[√β, 0], [−α/√β, 1/√β]].

Each eigenvalue pair e^{±iμ} has two eigenvectors v = a ± ib; the one with
aᵀ S b > 0 belongs to the positive amplitude function, and its eigenvalue
gives μ in (0, 2π) — so a fractional tune above 0.5 comes out as such.
Modes are assigned to planes by where their eigenvectors live (largest
weight first).
"""
from __future__ import annotations

import itertools

import numpy as np


def symplectic_form(n: int) -> np.ndarray:
    S = np.zeros((n, n))
    for k in range(0, n, 2):
        S[k, k + 1] = 1.0
        S[k + 1, k] = -1.0
    return S


class UnstableError(ValueError):
    """The one-turn matrix has an eigenvalue off the unit circle."""


def normal_form(M: np.ndarray, *, stability_tol: float = 1e-6):
    """(W, mu) for the stable symplectic matrix ``M`` (n×n, n = 4 or 6):
    ``mu[k]`` is the phase advance of mode k in [0, 2π) (mode k lives mostly
    in the coordinate pair k), ``W`` the normalizing matrix."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    S = symplectic_form(n)
    w, v = np.linalg.eig(M)
    if np.any(np.abs(np.abs(w) - 1.0) > stability_tol):
        raise UnstableError(f"one-turn matrix is unstable: |eigenvalues| = "
                            f"{np.sort(np.abs(w))[::-1].round(9).tolist()}")
    # one eigenvector per pair: the one with a^T S b > 0
    cands = []
    for k in range(n):
        a, b = v[:, k].real, v[:, k].imag
        if a @ S @ b > 0.0:
            cands.append(k)
    if len(cands) != n // 2:
        raise UnstableError("one-turn matrix has no normal form (degenerate or real eigenvalues)")
    # assign modes to planes: maximise the summed plane weights
    weight = np.array([[abs(v[2 * p, k]) ** 2 + abs(v[2 * p + 1, k]) ** 2
                        for p in range(n // 2)] for k in cands])
    best, order = -1.0, None
    for perm in itertools.permutations(range(n // 2)):
        tot = sum(weight[i, perm[i]] for i in range(n // 2))
        if tot > best:
            best, order = tot, perm
    plane_of = {cands[i]: order[i] for i in range(n // 2)}
    W = np.zeros((n, n))
    mu = np.zeros(n // 2)
    for k, p in plane_of.items():
        vk = v[:, k] * np.exp(-1j * np.angle(v[2 * p, k]))     # position of plane p real
        a, b = vk.real, vk.imag
        norm = a @ S @ b
        a = a / np.sqrt(norm)
        b = b / np.sqrt(norm)
        W[:, 2 * p] = a
        W[:, 2 * p + 1] = b
        mu[p] = np.angle(w[k]) % (2.0 * np.pi)
    return W, mu


def twiss_from_W(W: np.ndarray) -> dict:
    """Mode amplitude functions of a normalizing matrix, or of a stack of
    them (..., n, n) (Mais-Ripken β of mode I in x and mode II in y; equal to
    Courant-Snyder for uncoupled motion): beta_x, alpha_x, beta_y, alpha_y,
    and the phases of the two modes in their own planes (atan2 of W's
    position entries)."""
    W = np.asarray(W)
    return {
        "beta_x": W[..., 0, 0] ** 2 + W[..., 0, 1] ** 2,
        "alpha_x": -(W[..., 0, 0] * W[..., 1, 0] + W[..., 0, 1] * W[..., 1, 1]),
        "beta_y": W[..., 2, 2] ** 2 + W[..., 2, 3] ** 2,
        "alpha_y": -(W[..., 2, 2] * W[..., 3, 2] + W[..., 2, 3] * W[..., 3, 3]),
        "phase_x": np.arctan2(W[..., 0, 1], W[..., 0, 0]),
        "phase_y": np.arctan2(W[..., 2, 3], W[..., 2, 2]),
    }
