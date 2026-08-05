"""Offline training of the HALO-PIC corrector from logged anchor pairs.

Input: one or more .npz logs written by ``HaloPicSolver.save_log``
(arrays ``features`` (M, FEATURE_DIM) and ``coeffs`` (M, n_basis)).
Output: ``weights.pt`` + ``metadata.json`` loadable by
``HaloPicSolver.load``.

Plain supervised regression (Adam, MSE on standardized targets, early
stop on a validation split) — deliberately boring; all the physics
lives in the basis and the anchor protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def train_corrector(log_paths, out_dir, hidden=(64, 64), epochs=2000,
                    lr=1e-3, val_frac=0.2, seed=0, basis_degree=4,
                    basis_width=1.0, scale_normalize=False,
                    verbose=True) -> dict:
    import torch
    from linac_gen.surrogates.base import MlpHead

    X, Y, S = [], [], []
    for p in log_paths:
        d = np.load(p)
        if d["features"].size:
            X.append(np.atleast_2d(d["features"]))
            Y.append(np.atleast_2d(d["coeffs"]))
            if "scale" in d and d["scale"].size:
                S.append(np.asarray(d["scale"], float))
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    # OPT-IN per-sample scale normalization (targets per unit |E|_w).
    # Measured on the FODO testbed: raw targets train ~75x better
    # (val 0.007 vs 0.53) — dividing by the breathing field norm
    # destroys the target SNR.  Small-defect anchors where the raw-target
    # net over-corrects are handled by the solver's per-anchor alpha gate
    # instead.  Kept as an option for beams with strong current/size
    # variation across a dataset (multi-lattice training).
    scale_normalized = (scale_normalize
                        and len(S) == len(log_paths) and len(S) > 0)
    if scale_normalized:
        S = np.concatenate(S)
        Y = Y / np.maximum(S[:, None], 1e-30)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    vi, ti = idx[:n_val], idx[n_val:]

    # standardize features and targets (stored in metadata)
    xm, xs = X[ti].mean(0), X[ti].std(0) + 1e-12
    ym, ys = Y[ti].mean(0), Y[ti].std(0) + 1e-12
    Xn = (X - xm) / xs
    Yn = (Y - ym) / ys

    torch.manual_seed(seed)
    net = MlpHead(X.shape[1], Y.shape[1], tuple(hidden))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = torch.from_numpy(Xn[ti]); yt = torch.from_numpy(Yn[ti])
    xv = torch.from_numpy(Xn[vi]); yv = torch.from_numpy(Yn[vi])
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl = float(torch.mean((net(xv) - yv) ** 2))
        if vl < best - 1e-6:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience > 200:
                break
        if verbose and ep % 200 == 0:
            print(f"  ep {ep:5d}  train {float(loss):.4f}  val {vl:.4f}")
    net.load_state_dict(best_state)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # bake the standardization into a wrapper-free form: the solver calls
    # net(features) raw, so store an affine-absorbing first/last layer is
    # overkill — instead store scalers in metadata and wrap at load time.
    # Simpler: fold scalers into the net weights (exact):
    with torch.no_grad():
        first = net.net[0]
        first.bias.copy_(first.bias - first.weight @ torch.from_numpy(xm / xs))
        first.weight.copy_(first.weight / torch.from_numpy(xs))
        last = net.net[-1]
        last.weight.copy_(last.weight * torch.from_numpy(ys).unsqueeze(1))
        last.bias.copy_(last.bias * torch.from_numpy(ys)
                        + torch.from_numpy(ym))
    torch.save(net.state_dict(), out / "weights.pt")
    (out / "metadata.json").write_text(json.dumps({
        "input_dim": int(X.shape[1]), "output_dim": int(Y.shape[1]),
        "hidden_dims": list(hidden), "basis_degree": int(basis_degree),
        "basis_width": float(basis_width), "n_train": int(len(ti)),
        "scale_normalized": bool(scale_normalized),
        "val_mse_normalized": float(best), "seed": int(seed),
    }, indent=1), encoding="utf-8")
    return {"val_mse": best, "n_train": len(ti), "out": str(out)}
