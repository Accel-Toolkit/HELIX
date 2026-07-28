"""Per-kick conditioning features for the HALO-PIC corrector.

Everything here is O(N) on data the kick already touches (rest-frame
coordinates), plus static run metadata.  Features are individually
normalized to O(1) with fixed, documented scales so a net trained on
one run transfers across grid geometries.
"""
from __future__ import annotations

import numpy as np

FEATURE_DIM = 20


def kick_features(coords_rest: np.ndarray, s_mm: float,
                  n_alive: int, n_grid, step2_per_m: float,
                  current_mA: float, period_len_mm: float) -> np.ndarray:
    """(FEATURE_DIM,) float64 feature vector for one SC kick.

    coords_rest : (N, 3) rest-frame particle positions (mm) — the same
    array the deposit uses, so shape statistics match the deposited rho.
    """
    x, y, z = coords_rest[:, 0], coords_rest[:, 1], coords_rest[:, 2]
    xm, ym, zm = x.mean(), y.mean(), z.mean()
    x, y, z = x - xm, y - ym, z - zm
    sx = max(float(np.sqrt((x * x).mean())), 1e-12)
    sy = max(float(np.sqrt((y * y).mean())), 1e-12)
    sz = max(float(np.sqrt((z * z).mean())), 1e-12)
    u, v, w = x / sx, y / sy, z / sz

    def kurt(a):
        return float((a ** 4).mean()) - 3.0

    def skew(a):
        return float((a ** 3).mean())

    r2 = u * u + v * v
    r999 = float(np.quantile(np.sqrt(r2), 0.999))

    phase = 2.0 * np.pi * (s_mm / period_len_mm)
    feats = np.array([
        np.log(sx), np.log(sy), np.log(sz),          # absolute scales
        np.log(sy / sx), np.log(sz / sx),            # aspect
        float((u * v).mean()),                       # xy tilt
        float((u * w).mean()), float((v * w).mean()),
        skew(u), skew(v), skew(w),
        kurt(u) / 3.0, kurt(v) / 3.0, kurt(w) / 3.0,
        (r999 - 3.3) / 3.3,                          # tail extent vs Gaussian
        np.sin(phase), np.cos(phase),                # position in the cell
        np.log(max(n_alive, 1) / 2e4),               # sampling density
        np.log(int(n_grid[0]) / 32.0),               # grid resolution
        np.log(max(current_mA, 1e-3) / 5.0),         # SC strength
    ], dtype=np.float64)
    assert feats.shape == (FEATURE_DIM,)
    return feats
