"""M2 — train an envelope-mode MLP surrogate for one FieldMap3D.

End-to-end driver demonstrating the surrogate-elements pipeline:

  1. Load a TraceWin lattice (default: PIP-II MEBT).
  2. Pick the first FieldMap3D instance (the MEBT buncher #1, "FMAP_001").
  3. Latin-Hypercube sample over (incoming kinetic energy + element
     ``ke`` + element ``phase``); for each sample, call the element's
     own ``fitted_matrix(ref)`` to get the 6x6 reference matrix.
  4. Train an MLP residual (FP64 CPU, SiLU) to predict the 6x6 matrix.
  5. Save weights + metadata.json to
     ``linac_gen/surrogates/weights/<hash>/<element>/`` and validate
     against held-out samples.

Usage
-----
Short cycle (smoke test, ~1 minute on CPU)::

    python examples/surrogates/train_fieldmap_mlp.py --samples 200 --epochs 40

Full M2 acceptance run (~hours)::

    python examples/surrogates/train_fieldmap_mlp.py --samples 50000 \\
        --epochs 300 --hidden 128,128,128

Output
------
* ``linac_gen/surrogates/weights/<lattice-hash>/<element-name>/weights.pt``
* ``linac_gen/surrogates/weights/<lattice-hash>/<element-name>/metadata.json``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import SurrogateFieldMap
from linac_gen.surrogates.training import train_surrogate_for_element


def _find_first_field_map(lattice) -> FieldMap3D:
    for elem in lattice.elements:
        if isinstance(elem, FieldMap3D):
            return elem
    raise SystemExit("no FieldMap3D element found in lattice")


def _validate(surr: SurrogateFieldMap, element: FieldMap3D, ref_template,
              w_kin_range: tuple[float, float],
              param_ranges: dict[str, tuple[float, float]],
              n_probe: int = 8, seed: int = 0) -> dict:
    """Sample n_probe in-scope (w_kin, params), compare surrogate's
    fitted_matrix to the element's RK4-based fitted_matrix.

    Returns metrics: per-probe Frobenius rel.diff + symplecticity
    defect.
    """
    rng = np.random.default_rng(seed)
    w_lo, w_hi = w_kin_range
    S = np.zeros((6, 6))
    S[0, 1] = 1; S[1, 0] = -1
    S[2, 3] = 1; S[3, 2] = -1
    S[4, 5] = 1; S[5, 4] = -1

    frob_diffs: list[float] = []
    sympl_defects: list[float] = []
    for _ in range(n_probe):
        w_kin = float(w_lo + (w_hi - w_lo) * rng.random())
        for name, (lo, hi) in param_ranges.items():
            setattr(element, name, float(lo + (hi - lo) * rng.random()))
        element.reset_run_state()
        ref = ref_template.copy()
        ref.w_kin = w_kin
        M_truth = element.fitted_matrix(ref)
        M_surr = surr.fitted_matrix(ref)
        rel = float(np.linalg.norm(M_truth - M_surr, "fro")
                    / max(np.linalg.norm(M_truth, "fro"), 1e-12))
        frob_diffs.append(rel)
        # Symplecticity defect of the surrogate's prediction.
        defect = float(np.linalg.norm(M_surr.T @ S @ M_surr - S, "fro") / 6.0)
        sympl_defects.append(defect)
    return {
        "frob_rel_diff": np.asarray(frob_diffs),
        "sympl_defect": np.asarray(sympl_defects),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an envelope-mode surrogate for a FieldMap3D.")
    parser.add_argument("--lattice", default="examples/pipii/mebt/mebt.dat",
                        help="TraceWin .dat (default: PIP-II MEBT)")
    parser.add_argument("--samples", type=int, default=200,
                        help="LHS samples (smoke: 200; full: 50000)")
    parser.add_argument("--epochs", type=int, default=40,
                        help="training epochs (smoke: 40; full: 300)")
    parser.add_argument("--hidden", type=str, default="64,64",
                        help="hidden dims, comma-sep (default: 64,64)")
    parser.add_argument("--w-kin-range", type=str, default="2.0,2.5",
                        help="ref kinetic-energy range MeV (lo,hi)")
    parser.add_argument("--ke-rel", type=float, default=0.20,
                        help="ke fractional sweep range (default: +/-20%%)")
    parser.add_argument("--phase-rel-deg", type=float, default=20.0,
                        help="phase sweep range in deg (default: +/-20 deg)")
    parser.add_argument("--out", default=None,
                        help="output dir (default: linac_gen/surrogates/"
                             "weights/<hash>/<name>)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    hidden_dims = tuple(int(h) for h in args.hidden.split(","))
    w_lo, w_hi = (float(v) for v in args.w_kin_range.split(","))

    print(f"Loading lattice {args.lattice}...")
    lattice, _ = parse_tracewin(args.lattice)
    element = _find_first_field_map(lattice)
    print(f"Picked FieldMap3D: name={element.name!r}  length={element.length} mm")
    print(f"  ke={element.ke}  kb={element.kb}  ki={element.ki}")
    print(f"  phase={element.phase} deg  frequency={element.frequency} MHz")

    ke0 = float(element.ke)
    phase0 = float(element.phase)
    param_ranges = {
        "ke": (ke0 * (1.0 - args.ke_rel), ke0 * (1.0 + args.ke_rel)),
        "phase": (phase0 - args.phase_rel_deg, phase0 + args.phase_rel_deg),
    }

    # Build a ref template (PIP-II H- at 2.12 MeV, 162.5 MHz, the MEBT
    # entry conditions).
    ref_template = ReferenceParticle(species=H_MINUS, w_kin=2.12,
                                     frequency=162.5)

    lattice_hash = registry.hash_lattice_file(args.lattice)
    out_dir = (Path(args.out) if args.out
               else Path("linac_gen/surrogates/weights")
                    / lattice_hash[:16] / element.name)

    print(f"\nTraining surrogate")
    print(f"  samples:   {args.samples}")
    print(f"  epochs:    {args.epochs}")
    print(f"  hidden:    {hidden_dims}")
    print(f"  w_kin:     [{w_lo}, {w_hi}] MeV")
    print(f"  ke:        [{param_ranges['ke'][0]:.4f}, "
          f"{param_ranges['ke'][1]:.4f}]")
    print(f"  phase:     [{param_ranges['phase'][0]:.1f}, "
          f"{param_ranges['phase'][1]:.1f}] deg")
    print(f"  out_dir:   {out_dir}")
    print()

    t0 = time.time()
    mlp, meta = train_surrogate_for_element(
        element=element,
        ref_template=ref_template,
        n_samples=args.samples,
        ref_w_kin_range=(w_lo, w_hi),
        param_ranges=param_ranges,
        hidden_dims=hidden_dims,
        activation="silu",
        epochs=args.epochs,
        lr=3e-3,
        batch_size=min(64, max(8, args.samples // 8)),
        val_frac=0.2,
        seed=args.seed,
        out_dir=out_dir,
        lattice_hash=lattice_hash,
        element_key=element.name,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n--- training done in {elapsed:.1f}s ---")
    print(f"val MAPE: {meta.val_mape:.4e}")
    print(f"saved:    {out_dir}")

    # Quick in-scope accuracy probe.
    surr = SurrogateFieldMap(element, mlp, meta)
    metrics = _validate(surr, element, ref_template,
                         (w_lo, w_hi), param_ranges,
                         n_probe=8, seed=args.seed + 1)
    fdiffs = metrics["frob_rel_diff"]
    sdefs = metrics["sympl_defect"]
    print(f"\n--- in-scope probe (n={len(fdiffs)}) ---")
    print(f"Frobenius rel.diff  mean={fdiffs.mean():.3e}  "
          f"max={fdiffs.max():.3e}")
    print(f"Sympl. defect       mean={sdefs.mean():.3e}  "
          f"max={sdefs.max():.3e}")

    # M2 acceptance gates: matrix Frobenius rel.diff <0.5%, sympl
    # defect <1e-3.  These are the FULL-RUN gates; a smoke cycle will
    # typically miss them and that's expected.
    GATE_FROB = 0.005
    GATE_SYMPL = 1e-3
    smoke = args.samples < 5000
    if smoke:
        print(f"\n(smoke cycle: full-run gates are <{GATE_FROB:.1%} Frob "
              f"and <{GATE_SYMPL:.1e} sympl defect; expect to miss them "
              f"with this few samples)")
    else:
        ok_frob = fdiffs.max() < GATE_FROB
        ok_sympl = sdefs.max() < GATE_SYMPL
        print(f"\nM2 acceptance: "
              f"frob {'OK' if ok_frob else 'FAIL'}, "
              f"sympl {'OK' if ok_sympl else 'FAIL'}")
        if not (ok_frob and ok_sympl):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
