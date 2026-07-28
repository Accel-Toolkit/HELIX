"""``python -m linac_gen backtrack`` — backward tracking of a distribution.

Walks a distribution from the exit of an element range to its entrance
by applying the exact algebraic inverse of every forward operation
(see :mod:`linac_gen.tracking.backtrack`).  Two source modes:

* ``--dst EXIT.dst`` — reconstruction: a measured / exported exit
  distribution is walked upstream (the file is authoritative for its
  reference energy and frequency);
* no ``--dst`` — design targeting: the project's Twiss/emittance are
  taken as the DESIRED EXIT beam, generated at the design exit energy,
  and backtracked to find the input that produces it.

``--validate`` re-tracks the reconstruction FORWARD over the same range
and reports exit residuals (Δσ, Δε, ΔW) against the original exit
distribution — the TWICE-style closure check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from linac_gen.cli import common

_EXT = {"hdf5": ".h5", "openpmd": ".opmd.h5", "partran": ".txt"}


def add_arguments(p) -> None:
    """Populate the ``backtrack`` sub-parser."""
    p.add_argument("input", help="a .lgproj project or a .dat/.madx lattice")
    p.add_argument("--dst", default=None, metavar="EXIT.dst",
                   help="exit-plane distribution to reconstruct from; "
                        "omitted → design mode (exit beam generated from "
                        "the project Twiss at the design exit energy)")
    p.add_argument("--mode", choices=("mp", "envelope"), default="mp",
                   help="multi-particle (default) or RMS-envelope walk")
    p.add_argument("--from-element", type=int, default=0, dest="start",
                   metavar="N", help="range start (entrance target, "
                                     "default 0 = lattice entrance)")
    p.add_argument("--to-element", type=int, default=None, dest="end",
                   metavar="N", help="range end (exit source, default "
                                     "last element)")
    p.add_argument("--out", default=".", help="output directory (default .)")
    p.add_argument("--format", choices=("hdf5", "openpmd", "partran"),
                   default="hdf5", help="results format (default hdf5)")
    p.add_argument("--write-dst", default=None, metavar="OUT.dst",
                   dest="write_dst",
                   help="write the reconstructed entrance distribution "
                        "to this .dst (mp only)")
    p.add_argument("--validate", action="store_true",
                   help="forward re-track the reconstruction over the "
                        "same range and report exit residuals (mp only)")
    p.add_argument("--allow-dc-crossing", action="store_true",
                   dest="allow_dc_crossing",
                   help="assert the beam was DC (continuous) upstream of "
                        "the first RF bunching element")
    p.add_argument("--approximate-backtracking", action="store_true",
                   dest="approximate_backtracking",
                   help="accept an APPROXIMATE reconstruction when the "
                        "forward run applied kicks with no backward "
                        "model (currently: CSR — skipped on the "
                        "backward walk with a loud warning); without "
                        "this flag such runs refuse to backtrack")
    p.add_argument("--fieldmap-inverse", choices=("rk4", "linear"),
                   default="rk4", dest="field_map_mode",
                   help="field-map inverse: 'rk4' (default) undoes each "
                        "forward integration step in closed form — round "
                        "trips close at float round-off (~1e-13); "
                        "'linear' is the v1 inverted fitted-matrix path "
                        "(~2%% closure through strong bunchers), kept "
                        "for comparison and for elements without an "
                        "exact inverse (RFQ cells, surrogates fall back "
                        "to it automatically)")
    p.add_argument("--energy-tolerance", type=float, default=1e-3,
                   dest="energy_tolerance", metavar="REL",
                   help="relative |ΔW|/W above which the exit-energy "
                        "mismatch warns (default 1e-3)")
    p.add_argument("--energy-hard-limit", type=float, default=5e-2,
                   dest="energy_hard_limit", metavar="REL",
                   help="relative |ΔW|/W above which the walk refuses "
                        "(default 5e-2)")
    # beam / SC / stepping overrides (same contract as `run`)
    p.add_argument("--energy", type=float, help="entrance kinetic energy (MeV)")
    p.add_argument("--current", type=float, help="beam current (mA)")
    p.add_argument("--freq", type=float, help="beam frequency (MHz)")
    p.add_argument("--n-particles", type=int, dest="n_particles",
                   help="number of macroparticles (design mode)")
    p.add_argument("--species", help="proton / deuteron / H-")
    p.add_argument("--beam", action="append", default=[], metavar="NAME=VALUE",
                   help="generic beam-config override (repeatable)")
    p.add_argument("--sc-on", action="store_true", dest="sc_on",
                   help="enable space charge on the backward walk "
                        "(mp only; exact only for adaptive-grid PIC)")
    p.add_argument("--nx", type=int, help="PIC grid size")
    p.add_argument("--grid-extent", type=float, dest="grid_extent",
                   help="PIC grid extent in sigma")
    p.add_argument("--step1", type=float, help="integration steps / metre")
    p.add_argument("--step2", type=float, help="space-charge kicks / metre")
    p.add_argument("--kernel", help="PIC deposit kernel (cic / tsc)")
    p.add_argument("--backend", help="compute backend (auto / cpu / gpu)")
    p.add_argument("--sc", action="append", default=[], metavar="NAME=VALUE",
                   help="generic SpaceChargeConfig override (repeatable)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress the run report")


def _beam_overrides(args) -> dict:
    d: dict = {}
    for name, val in (("energy", args.energy), ("current", args.current),
                      ("frequency", args.freq),
                      ("n_particles", args.n_particles),
                      ("species", args.species)):
        if val is not None:
            d[name] = val
    d.update(common.parse_assignments(args.beam))
    return d


def _cli_overrides(args) -> dict:
    return {
        "nx": args.nx, "grid_extent": args.grid_extent,
        "step1": args.step1, "step2": args.step2,
        "kernel": args.kernel, "backend": args.backend,
        "sc": common.parse_assignments(args.sc),
    }


def run(args) -> int:
    """Execute the ``backtrack`` subcommand; return an exit code."""
    if not Path(args.input).is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.dst is not None and not Path(args.dst).is_file():
        print(f"error: --dst not found: {args.dst}", file=sys.stderr)
        return 2
    try:
        lattice, beam_cfg, conv = common.load_input(args.input)
        common.apply_beam_overrides(beam_cfg, _beam_overrides(args))
        cli = _cli_overrides(args)
        common.apply_fieldmap_settings(conv, cli)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    end = args.end if args.end is not None else len(lattice.elements) - 1
    if not (0 <= args.start <= end < len(lattice.elements)):
        print(f"error: invalid range [--from-element {args.start}, "
              f"--to-element {end}] for {len(lattice.elements)} elements",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    stem = Path(args.input).stem
    try:
        if args.mode == "envelope":
            for flag, name in ((args.dst, "--dst"),
                               (args.write_dst, "--write-dst"),
                               (args.validate, "--validate"),
                               (args.sc_on, "--sc-on")):
                if flag:
                    print(f"warning: {name} is ignored in envelope mode "
                          "(RMS walk has no particles)", file=sys.stderr)
            return _run_envelope(args, lattice, beam_cfg, out_dir, stem, end)
        step_cfg = common.make_step_config(conv, cli)
        sc = (common.make_sc_config(beam_cfg, conv, cli)
              if args.sc_on else None)
        recorder, beam = common.run_backtrack_sim(
            lattice, beam_cfg, sc, step_cfg, seed=args.seed,
            dst_path=args.dst, start=args.start, end=end,
            approximate_backtracking=args.approximate_backtracking,
            allow_dc_crossing=args.allow_dc_crossing,
            energy_tolerance=args.energy_tolerance,
            energy_hard_limit=args.energy_hard_limit,
            field_map_mode=args.field_map_mode)
    except Exception as exc:                                # noqa: BLE001
        print(f"error: backtrack failed: {exc}", file=sys.stderr)
        return 1

    out_path = out_dir / f"{stem}_backtrack{_EXT[args.format]}"
    written = [common.write_results(recorder, out_path, args.format,
                                    beam_cfg, lattice,
                                    lattice_path=args.input,
                                    seed=args.seed,
                                    sc_config=sc)]
    if args.write_dst:
        written.append(common.write_final_dst(beam, Path(args.write_dst)))

    if not args.quiet:
        _report(args, recorder, beam, written)

    if args.validate:
        return _validate(args, lattice, beam_cfg, recorder, beam, end)
    return 0


def _run_envelope(args, lattice, beam_cfg, out_dir, stem, end) -> int:
    """Envelope mode: the project Twiss is the DESIRED EXIT state."""
    from linac_gen.tracking.backtrack import backtrack_envelope
    ref = common.build_ref(beam_cfg)
    exit_twiss = common._envelope_initial(beam_cfg, ref)
    results = backtrack_envelope(lattice, ref, exit_twiss,
                                 start=args.start, end=end,
                                 current=beam_cfg.current)
    out_path = out_dir / f"{stem}_backtrack{_EXT[args.format]}"
    written = common.write_results(results, out_path, args.format,
                                   beam_cfg, lattice,
                                   lattice_path=args.input)
    if not args.quiet:
        print(f"[backtrack] mode=envelope  input={args.input}  "
              f"range=[{args.start}, {end}]")
        print(f"  entrance Twiss: alpha_x={results.alpha_x[0]:+.4f}  "
              f"beta_x={results.beta_x[0]:.4f}  "
              f"alpha_y={results.alpha_y[0]:+.4f}  "
              f"beta_y={results.beta_y[0]:.4f}")
        print(f"  entrance sigma: x={results.sigma_x[0]:.4f} mm  "
              f"y={results.sigma_y[0]:.4f} mm")
        print(f"[backtrack] wrote {written}")
    return 0


def _validate(args, lattice, beam_cfg, recorder, beam, end) -> int:
    """Forward closure check: re-track the reconstruction over the same
    range and compare the recovered exit against the original input."""
    from linac_gen.core.beam import Beam
    from linac_gen.pic.pic_solver import PicSolver
    from linac_gen.tracking.tracker import Tracker
    from linac_gen.tracking.backtrack import build_replay_table

    entrance_ref = common.build_ref(beam_cfg)
    table = build_replay_table(lattice, entrance_ref, end=end)

    # The original exit distribution is the LAST recorder row's source —
    # recorder holds moments only, so re-derive the reference exit stats
    # from the recorder tail (index -1 = supplied exit, tagged INPUT).
    sig_exit_ref = (recorder.sigma_x[-1], recorder.sigma_y[-1],
                    recorder.sigma_phi[-1], recorder.sigma_w[-1])
    emit_exit_ref = (recorder.emit_x[-1], recorder.emit_y[-1],
                     recorder.emit_z[-1])
    w_exit_ref = recorder.ref_w_kin[-1]

    vbeam = Beam(ref=table[args.start].make_ref(entrance_ref.species),
                 n_particles=beam.n_particles, current=beam.current)
    vbeam.particles[:] = beam.particles
    vbeam.lost[:] = beam.lost
    vbeam.continuous = beam.continuous
    sc = None
    if args.sc_on:
        conv: dict = {}
        sc_cfg = common.make_sc_config(beam_cfg, conv, _cli_overrides(args))
        sc = PicSolver(sc_cfg) if sc_cfg else None
    tracker = Tracker(lattice, vbeam, pic_solver=sc)
    # Forward sub-range replay: element loop with the table's entrance
    # state (mirrors the backward walk's use of the replay table).
    for i in range(args.start, end + 1):
        tracker._track_element(lattice.elements[i])
        tracker._check_aperture(lattice.elements[i])

    alive = vbeam.alive_mask
    p = vbeam.particles[alive]
    sig = (p[:, 0].std(), p[:, 2].std(), p[:, 4].std(), p[:, 5].std())
    names = ("sigma_x/mm", "sigma_y/mm", "sigma_phi/deg", "sigma_w/MeV")
    print(f"[validate] forward closure over [{args.start}, {end}] "
          f"({int(alive.sum())} particles):")
    worst = 0.0
    for name, got, want in zip(names, sig, sig_exit_ref):
        rel = abs(got - want) / max(abs(want), 1e-12)
        worst = max(worst, rel)
        print(f"  {name:<14} recovered={got:.6g}  original={want:.6g}  "
              f"rel={rel:.3e}")
    dw = abs(vbeam.ref.w_kin - w_exit_ref)
    print(f"  ref W_kin      recovered={vbeam.ref.w_kin:.6f} MeV  "
          f"original={w_exit_ref:.6f} MeV  |dW|={dw:.3e}")
    _ = emit_exit_ref   # emittances are in the written results file
    if worst > 0.05:
        print("[validate] FAIL: worst relative sigma residual "
              f"{worst:.3e} > 5e-2 (field-map linear-inverse fidelity "
              "exceeded — inspect the range for cavities)",
              file=sys.stderr)
        return 1
    print(f"[validate] OK (worst relative sigma residual {worst:.3e})")
    return 0


def _report(args, recorder, beam, written) -> None:
    print(f"[backtrack] mode=mp  input={args.input}  "
          f"source={'design-mode Twiss' if args.dst is None else args.dst}")
    rng = getattr(recorder, "backtrack_range", None)
    if rng:
        print(f"  range          = [{rng[0]}, {rng[1]}] (exit → entrance)")
    print(f"  entrance s     = {recorder.s[0]:.3f} mm")
    print(f"  entrance W_kin = {recorder.ref_w_kin[0]:.6f} MeV")
    alive = beam.alive_mask
    p = beam.particles[alive]
    print(f"  reconstructed sigma: x={p[:, 0].std():.4f} mm  "
          f"y={p[:, 2].std():.4f} mm  dphi={p[:, 4].std():.4f} deg  "
          f"dW={p[:, 5].std():.6f} MeV   ({int(alive.sum())} particles)")
    for w in written:
        print(f"[backtrack] wrote {w}")
