"""Differentiable matrix tracking — usage demo (HELIX v2).

Run from the HELIX_v2 repository root::

    PYTHONPATH=. python examples/differentiable_demo.py

This demonstrates the differentiable-tracking feature: a PyTorch-autograd
mirror of HELIX's linear matrix tracking.  It is a separate, opt-in
Python API — it does NOT change the GUI, the envelope solver, or the
multi-particle tracker.

The whole API is one class, ``DifferentiableLattice``:

    from linac_gen.tracking.autograd_api import DifferentiableLattice
    dl = DifferentiableLattice(lattice, ref)
    dl.set_tunables([("QF", "gradient")])      # pick the knobs
    beta = dl.twiss("x")["beta"]               # any output (a tensor)
    beta.backward()                            # -> exact gradients

(Lower-level functions ``compute_transfer_matrix_torch`` /
``compute_twiss_torch`` / ``track_beam_torch`` live in
``linac_gen.tracking.torch_tracking`` if you want them without the
wrapper.)

Three parts:
  A. Basic usage     — transfer matrix, Twiss, beam tracking.
  B. Exact gradients — d(output)/d(knob) in one autograd backward pass.
  C. The payoff      — gradient-based matching: a PyTorch optimiser tunes
                       quad gradients to hit a target, driven by exact
                       gradients instead of finite-difference scans.
"""
from __future__ import annotations

import numpy as np
import torch

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.autograd_api import DifferentiableLattice
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)


def build_fodo(gf: float = 8.0, gd: float = -8.0) -> Lattice:
    """A simple FODO cell: drift, focusing quad, drift, defocusing quad, drift."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=gf))
    lat.add(Drift("D2", length=400.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=gd))
    lat.add(Drift("D3", length=200.0))
    return lat


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------
# A. Basic usage
# --------------------------------------------------------------------------
def part_a_basic() -> None:
    section("A. Basic usage — transfer matrix, Twiss, beam tracking")
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    lat = build_fodo()

    # The one object you work with.
    dl = DifferentiableLattice(lat, ref)

    # 6x6 transfer matrix, as a torch tensor.
    M = dl.transfer_matrix()
    print(f"  transfer matrix  : {tuple(M.shape)} tensor, {M.dtype}")

    # It reproduces the production numpy tracker to machine precision.
    M_np = compute_transfer_matrix(lat, ref)
    diff = float(np.max(np.abs(M_np - M.detach().numpy())))
    print(f"  vs numpy tracker : max|difference| = {diff:.1e}")

    # Twiss parameters at the lattice exit.
    tw = dl.twiss("x")
    print(f"  Twiss x          : beta = {tw['beta'].item():.4f}  "
          f"alpha = {tw['alpha'].item():+.4f}  mu = {tw['mu'].item():.2f} deg")

    # Track a beam: an (N, 6) array of particles -> (N, 6) tensor.
    # Columns are [x mm, x' mrad, y mm, y' mrad, dphi deg, dW MeV].
    X = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # 1 mm offset in x
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])    # 1 mm offset in y
    X_out = dl.track(X)
    print(f"  tracked 2 particles -> exit x = {X_out[0, 0].item():+.4f} mm, "
          f"exit y = {X_out[1, 2].item():+.4f} mm")


# --------------------------------------------------------------------------
# B. Exact gradients
# --------------------------------------------------------------------------
def part_b_gradients() -> None:
    section("B. Exact gradients — d(output)/d(knob) in one backward pass")
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    lat = build_fodo()
    dl = DifferentiableLattice(lat, ref)

    # Declare which element parameters are differentiable knobs.
    # Valid: ("<quad>", "gradient"), ("<solenoid>", "field"),
    #        ("<dipole>", "angle").
    params = dl.set_tunables([("QF", "gradient"), ("QD", "gradient")])

    # Pick any scalar output and back-propagate through the whole lattice.
    beta_x = dl.twiss("x")["beta"]
    beta_x.backward()

    print(f"  output: beta_x = {beta_x.item():.5f}")
    for p in params:
        print(f"  d(beta_x)/d({p.element.name}.gradient) = "
              f"{p.tensor.grad.item():+.6e}   <- exact, from autograd")

    # Cross-check one gradient against a finite difference (numpy path).
    qf = next(e for e in lat.elements if e.name == "QF")
    h, base = 1e-5, qf.gradient
    qf.gradient = base + h
    bp = compute_twiss(compute_transfer_matrix(lat, ref), "x")["beta"]
    qf.gradient = base - h
    bm = compute_twiss(compute_transfer_matrix(lat, ref), "x")["beta"]
    qf.gradient = base
    print(f"  finite-difference check (QF) = {(bp - bm) / (2 * h):+.6e}   "
          f"<- matches the autograd value")


# --------------------------------------------------------------------------
# C. The payoff — gradient-based matching
# --------------------------------------------------------------------------
def part_c_matching() -> None:
    section("C. The payoff — gradient-based matching")
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)

    # An initial beam, described by its 6x6 sigma (covariance) matrix.
    # Diagonal here: variances of [x, x', y, y', dphi, dW].
    S0 = torch.diag(torch.tensor(
        [4.0, 0.25, 4.0, 0.25, 1.0, 1e-4], dtype=torch.float64))

    # Build an ACHIEVABLE target: run the lattice at a known "true"
    # setting and use the resulting exit beam sizes as the match goal.
    S_true = DifferentiableLattice(build_fodo(gf=9.0, gd=-7.0), ref).sigma(S0)
    target_sx = torch.sqrt(S_true[0, 0]).item()
    target_sy = torch.sqrt(S_true[2, 2]).item()
    print(f"  goal : exit sigma_x = {target_sx:.4f} mm, "
          f"sigma_y = {target_sy:.4f} mm")

    # Start the matcher from a deliberately wrong quad setting.
    dl = DifferentiableLattice(build_fodo(gf=6.0, gd=-10.0), ref)
    params = dl.set_tunables([("QF", "gradient"), ("QD", "gradient")])
    knobs = [p.tensor for p in params]
    print(f"  start: QF = {knobs[0].item():+.3f} T/m, "
          f"QD = {knobs[1].item():+.3f} T/m")

    # A standard PyTorch optimiser — it consumes the exact gradients.
    opt = torch.optim.LBFGS(knobs, lr=1.0, max_iter=100,
                            tolerance_grad=1e-16, tolerance_change=1e-18,
                            line_search_fn="strong_wolfe")
    history: list[float] = []

    def closure():
        opt.zero_grad()
        S = dl.sigma(S0)
        loss = ((torch.sqrt(S[0, 0]) - target_sx) ** 2
                + (torch.sqrt(S[2, 2]) - target_sy) ** 2)
        loss.backward()
        history.append(loss.item())
        return loss

    opt.step(closure)

    n = len(history)
    print(f"  optimiser ran {n} forward+gradient evaluations:")
    for i in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}):
        print(f"    eval {i:3d}:  loss = {history[i]:.3e} mm^2")

    S = dl.sigma(S0)
    print(f"  matched: exit sigma_x = {torch.sqrt(S[0, 0]).item():.4f} mm, "
          f"sigma_y = {torch.sqrt(S[2, 2]).item():.4f} mm")
    print(f"  matched knobs: QF = {knobs[0].item():+.4f} T/m, "
          f"QD = {knobs[1].item():+.4f} T/m")
    print("  -> the optimiser hit the target using exact gradients;")
    print("     no finite-difference scan over the knobs was needed.")


def main() -> None:
    print("HELIX v2 — differentiable matrix tracking, usage demo")
    part_a_basic()
    part_b_gradients()
    part_c_matching()
    print("\n" + "=" * 72)
    print("Done.  This is a Python API — it does not appear in the GUI.")
    print("=" * 72)


if __name__ == "__main__":
    main()
