"""HELIX surrogate-element CLI (M5).

Three subcommands wrap :mod:`linac_gen.surrogates.training` and
:mod:`linac_gen.surrogates.compare`:

* ``train``        — train a surrogate for one element of a lattice.
* ``compare``      — baseline vs. surrogate-enabled envelope diff.
* ``run-envelope`` — run the envelope once with surrogates engaged.

Invoke with::

    python -m linac_gen.surrogates.cli <subcommand> [options]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Default initial Twiss for the PIP-II MEBT entry (H-, 2.12 MeV).  Used by
# the compare and run-envelope subcommands when --beam-config is not
# supplied; override via the per-arg flags if your lattice is different.
_MEBT_INIT_TWISS = {
    "alpha_x": 1.228,    "beta_x": 0.316,    "emit_x_n": 0.21,
    "alpha_y": -0.095394, "beta_y": 0.113,   "emit_y_n": 0.21,
    "alpha_z": 0.0,      "beta_z": 819.05492, "emit_z": 0.06231832,
}


def _format_scope(scope) -> str:
    """One-line human summary of a Scope (input_names / input_lo /
    input_hi — the actual dataclass fields)."""
    try:
        return ", ".join(
            f"{name}∈[{float(lo):.4g}, {float(hi):.4g}]"
            for name, lo, hi in zip(scope.input_names,
                                    scope.input_lo, scope.input_hi))
    except Exception:                                       # noqa: BLE001
        return "(unreadable scope)"


def _resolve_element(lattice, key: str):
    """Find an element by name or 0-based index."""
    try:
        return lattice.elements[int(key)]
    except (ValueError, IndexError):
        pass
    for elem in lattice.elements:
        if elem.name == key:
            return elem
    raise SystemExit(f"element '{key}' not found in lattice "
                     f"(by name or index)")


def _build_ref_h_minus(w_kin_mev: float, frequency_mhz: float):
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    return ReferenceParticle(species=H_MINUS, w_kin=w_kin_mev,
                              frequency=frequency_mhz)


def _build_default_init_twiss(ref, alpha_x: float, beta_x: float,
                               emit_x_n: float, alpha_y: float,
                               beta_y: float, emit_y_n: float,
                               alpha_z: float, beta_z: float,
                               emit_z: float) -> dict:
    bg = max(float(ref.bg), 1e-9)
    return dict(
        alpha_x=alpha_x, beta_x=beta_x, emit_x=emit_x_n / bg,
        alpha_y=alpha_y, beta_y=beta_y, emit_y=emit_y_n / bg,
        alpha_z=alpha_z, beta_z=beta_z, emit_z=emit_z,
    )


# ---------------------------------------------------------------------------
def _cmd_train(args) -> int:
    """Train a FieldMap3D surrogate (envelope-mode 6x6 matrix)."""
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.surrogates import registry
    from linac_gen.surrogates.training import train_surrogate_for_element

    lat, _ = parse_tracewin(args.lattice)
    element = _resolve_element(lat, args.element)
    if not isinstance(element, FieldMap3D):
        raise SystemExit(
            f"element '{args.element}' is {type(element).__name__}, "
            f"not FieldMap3D (only FieldMap3D supported in M2)")

    hidden_dims = tuple(int(h) for h in args.hidden.split(","))
    w_lo, w_hi = (float(v) for v in args.w_kin_range.split(","))
    ke0 = float(element.ke)
    phase0 = float(element.phase)
    param_ranges = {
        "ke": (ke0 * (1.0 - args.ke_rel), ke0 * (1.0 + args.ke_rel)),
        "phase": (phase0 - args.phase_rel_deg, phase0 + args.phase_rel_deg),
    }

    ref_template = _build_ref_h_minus(args.ref_w_kin, args.frequency)
    lattice_hash = registry.hash_lattice_file(args.lattice)
    out_dir = (Path(args.out) if args.out
               else Path("linac_gen/surrogates/weights")
                    / lattice_hash[:16] / element.name)

    print(f"Training surrogate for element '{element.name}'  "
          f"({args.samples} samples, {args.epochs} epochs)")
    print(f"  ke      range: [{param_ranges['ke'][0]:.4f}, "
          f"{param_ranges['ke'][1]:.4f}]")
    print(f"  phase   range: [{param_ranges['phase'][0]:.1f}, "
          f"{param_ranges['phase'][1]:.1f}] deg")
    print(f"  w_kin   range: [{w_lo}, {w_hi}] MeV")
    print(f"  out_dir:       {out_dir}")

    _, meta = train_surrogate_for_element(
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
        n_workers=args.workers,
    )
    print(f"\nval MAPE: {meta.val_mape:.4e}")
    print(f"saved:    {out_dir}")
    return 0


def _cmd_compare(args) -> int:
    """Baseline vs surrogate-enabled envelope diff."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.surrogates.compare import (
        compare_envelope, plot_compare_report,
    )
    from linac_gen.surrogates.training import load_surrogate

    lat, _ = parse_tracewin(args.lattice)

    surrogates = []
    for w in args.weights:
        weights_dir = Path(w)
        mlp, meta = load_surrogate(weights_dir)
        element = next((e for e in lat.elements
                        if e.name == meta.element_key), None)
        if element is None:
            print(f"warning: surrogate '{meta.element_key}' (from {w}) "
                  f"not in lattice; skipping", file=sys.stderr)
            continue
        surr = SurrogateFieldMap(element, mlp, meta)
        surr.weights_dir = str(weights_dir)     # provenance: weights hash
        surrogates.append(surr)
        print(f"loaded surrogate '{meta.element_key}' "
              f"(val MAPE {meta.val_mape:.2e})")

    ref = _build_ref_h_minus(args.ref_w_kin, args.frequency)
    twiss = _build_default_init_twiss(
        ref,
        alpha_x=args.alpha_x, beta_x=args.beta_x, emit_x_n=args.emit_x_n,
        alpha_y=args.alpha_y, beta_y=args.beta_y, emit_y_n=args.emit_y_n,
        alpha_z=args.alpha_z, beta_z=args.beta_z, emit_z=args.emit_z,
    )

    print(f"\nRunning baseline vs surrogate-enabled envelope...")
    report = compare_envelope(lat, ref, twiss, current=args.current,
                              surrogates=surrogates)
    print()
    print(report.summary_text())

    if args.out:
        path = plot_compare_report(report, args.out)
        print(f"\nplot saved to: {path}")
    return 0


def _cmd_register_multi(args) -> int:
    """Register a single trained surrogate under N element names.

    Use case: HWR cryomodule has 8 cavities all backed by the same
    ``HWRDonut.map`` file, with different per-element ``ke`` / ``phase``
    settings.  If the trained sweep range is wide enough to cover all
    of them, you can share one surrogate across the whole family
    instead of training 8 separate ones -- TRACKING looks surrogates
    up by element name (last registration wins) GUARDED by the element
    in hand (identity / structural fingerprint; a same-named element
    from a different lattice is skipped with a warning), so each
    element name needs an explicit registration; the lattice hash only
    scopes weight paths and GUI bookkeeping.

    Quick check before registering: the trained metadata's scope is
    printed alongside each element's current ``(ke, phase)`` so the
    user can spot OOD risks before they bite at match time.
    """
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.surrogates import registry
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.surrogates.registry import hash_lattice_file
    from linac_gen.surrogates.training import load_surrogate

    src = Path(args.source)
    if not (src / "weights.pt").is_file():
        print(f"error: source '{src}' is not a saved surrogate "
              f"(no weights.pt).", file=sys.stderr)
        return 1
    mlp, meta = load_surrogate(src)

    lat, _ = parse_tracewin(args.lattice)
    lattice_hash = hash_lattice_file(args.lattice)
    print(f"lattice hash: {lattice_hash[:16]}", file=sys.stderr)
    print(f"source surrogate: {src.resolve()}", file=sys.stderr)
    # (The old f-string read meta.scope.w_kin_lo — an attribute Scope
    # never had — and crashed with AttributeError before registering
    # anything.)
    print(f"  scope: {_format_scope(meta.scope)}", file=sys.stderr)

    n_ok = 0
    n_skip = 0
    for name in args.names:
        element = next((e for e in lat.elements if e.name == name), None)
        if element is None:
            print(f"  SKIP {name}: not in lattice", file=sys.stderr)
            n_skip += 1
            continue
        try:
            surr = SurrogateFieldMap(element, mlp, meta)
            surr.weights_dir = str(src)         # provenance: weights hash
        except Exception as exc:    # noqa: BLE001
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            n_skip += 1
            continue
        registry.register(surr, lattice_hash, element_key=name)
        # Cosmetic: log this element's current params alongside the
        # surrogate's trained range, so OOD risk is visible at a glance.
        info_bits = []
        for attr in ("ke", "kb", "phase"):
            if hasattr(element, attr):
                info_bits.append(f"{attr}={float(getattr(element, attr)):+.4g}")
        info = "  " + " ".join(info_bits) if info_bits else ""
        print(f"  OK   {name}{info}", file=sys.stderr)
        n_ok += 1

    print(f"registered {n_ok} of {len(args.names)} requested "
          f"(skipped {n_skip})", file=sys.stderr)
    # Honesty: the registry is an in-process singleton and this CLI
    # process exits right here — nothing is persisted.
    print("note: registration is IN-PROCESS only (no persistence); "
          "as a standalone command this only VALIDATES that the "
          "surrogate loads and matches the named elements.  For runs, "
          "register from the GUI Surrogates tab or call "
          "linac_gen.surrogates.registry.register() in the same "
          "process as the tracking.", file=sys.stderr)
    return 0 if n_ok > 0 else 1


def _cmd_run_envelope(args) -> int:
    """Run the envelope once with the given surrogates engaged."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.surrogates import registry
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.surrogates.training import load_surrogate
    from linac_gen.tracking.envelope import EnvelopeSolver

    lat, _ = parse_tracewin(args.lattice)

    n_registered = 0
    for w in args.use_surrogates:
        weights_dir = Path(w)
        mlp, meta = load_surrogate(weights_dir)
        element = next((e for e in lat.elements
                        if e.name == meta.element_key), None)
        if element is None:
            print(f"warning: surrogate '{meta.element_key}' (from {w}) "
                  f"not in lattice; skipping", file=sys.stderr)
            continue
        surr = SurrogateFieldMap(element, mlp, meta)
        surr.weights_dir = str(weights_dir)     # provenance: weights hash
        registry.register(surr)
        n_registered += 1
        print(f"registered surrogate '{meta.element_key}' "
              f"(val MAPE {meta.val_mape:.2e})")
    print(f"\nregistered {n_registered} surrogates total.")

    ref = _build_ref_h_minus(args.ref_w_kin, args.frequency)
    twiss = _build_default_init_twiss(
        ref,
        alpha_x=args.alpha_x, beta_x=args.beta_x, emit_x_n=args.emit_x_n,
        alpha_y=args.alpha_y, beta_y=args.beta_y, emit_y_n=args.emit_y_n,
        alpha_z=args.alpha_z, beta_z=args.beta_z, emit_z=args.emit_z,
    )
    res = EnvelopeSolver(lat, ref, twiss, current=args.current).run()
    print(f"\nEnd-of-line:  "
          f"sigma_x={res.sigma_x[-1]:.4f} mm  "
          f"sigma_y={res.sigma_y[-1]:.4f} mm  "
          f"W={res.ref_w_kin[-1]:.4f} MeV  "
          f"sigma_phi={res.sigma_phi[-1]:.3f} deg")
    return 0


# ---------------------------------------------------------------------------
def _add_common_beam_args(p, defaults):
    p.add_argument("--ref-w-kin", type=float, default=2.12,
                   help="reference particle kinetic energy (MeV)")
    p.add_argument("--frequency", type=float, default=162.5,
                   help="reference RF frequency (MHz)")
    p.add_argument("--alpha-x", type=float, default=defaults["alpha_x"])
    p.add_argument("--beta-x", type=float, default=defaults["beta_x"])
    p.add_argument("--emit-x-n", type=float, default=defaults["emit_x_n"])
    p.add_argument("--alpha-y", type=float, default=defaults["alpha_y"])
    p.add_argument("--beta-y", type=float, default=defaults["beta_y"])
    p.add_argument("--emit-y-n", type=float, default=defaults["emit_y_n"])
    p.add_argument("--alpha-z", type=float, default=defaults["alpha_z"])
    p.add_argument("--beta-z", type=float, default=defaults["beta_z"])
    p.add_argument("--emit-z", type=float, default=defaults["emit_z"])
    p.add_argument("--current", type=float, default=0.0,
                   help="beam current in mA (for envelope SC; 0 = no SC)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="linac_gen.surrogates",
        description="HELIX surrogate-element CLI (M5).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- train ----------------------------------------------------------
    p_train = sub.add_parser(
        "train", help="train a FieldMap3D surrogate for one element")
    p_train.add_argument("--lattice", required=True,
                          help="TraceWin .dat file")
    p_train.add_argument("--element", required=True,
                          help="element name or 0-based index")
    p_train.add_argument("--samples", type=int, default=200,
                          help="LHS samples (smoke: 200, full: 50000)")
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--hidden", default="64,64",
                          help="hidden dims, comma-separated")
    p_train.add_argument("--w-kin-range", default="2.0,2.5",
                          help="ref kinetic energy MeV range (lo,hi)")
    p_train.add_argument("--ke-rel", type=float, default=0.20,
                          help="ke fractional sweep range")
    p_train.add_argument("--phase-rel-deg", type=float, default=20.0,
                          help="phase sweep range deg")
    p_train.add_argument("--ref-w-kin", type=float, default=2.12)
    p_train.add_argument("--frequency", type=float, default=162.5)
    p_train.add_argument("--out", default=None)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument(
        "--workers", type=int, default=1,
        help="CPU worker processes for data generation "
             "(default 1 = serial; >=2 uses multiprocessing.spawn).")
    p_train.set_defaults(fn=_cmd_train)

    # -- compare --------------------------------------------------------
    p_compare = sub.add_parser(
        "compare", help="baseline vs surrogate-enabled envelope diff")
    p_compare.add_argument("--lattice", required=True)
    p_compare.add_argument("--weights", nargs="+", required=True,
                            help="surrogate weights directories")
    p_compare.add_argument("--out", default=None,
                            help="optional PNG output path")
    _add_common_beam_args(p_compare, _MEBT_INIT_TWISS)
    p_compare.set_defaults(fn=_cmd_compare)

    # -- run-envelope ---------------------------------------------------
    p_run = sub.add_parser(
        "run-envelope", help="run envelope with surrogates engaged")
    p_run.add_argument("--lattice", required=True)
    p_run.add_argument("--use-surrogates", nargs="*", default=[],
                       help="weights directories to register")
    _add_common_beam_args(p_run, _MEBT_INIT_TWISS)
    p_run.set_defaults(fn=_cmd_run_envelope)

    # -- register-multi -------------------------------------------------
    p_multi = sub.add_parser(
        "register-multi",
        help="register one trained surrogate under multiple element "
             "names (e.g. share one HWR-cavity surrogate across all 8 "
             "cavities in the cryomodule).  Useful when the underlying "
             ".map file is the same and the per-element operating "
             "ranges overlap.")
    p_multi.add_argument("--source", required=True,
                          help="weights directory of an already-trained "
                               "surrogate (the one to share).")
    p_multi.add_argument("--lattice", required=True,
                          help="TraceWin .dat -- used to compute the "
                               "lattice hash for weight-path/GUI "
                               "bookkeeping (tracking resolves "
                               "surrogates by element name, last "
                               "registration wins, guarded against "
                               "cross-lattice name collisions by an "
                               "element identity/fingerprint check).")
    p_multi.add_argument("--names", nargs="+", required=True,
                          help="element names under which to register "
                               "this surrogate.")
    p_multi.set_defaults(fn=_cmd_register_multi)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
