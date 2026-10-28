"""``python -m linac_gen ring …`` — ring mode from the command line.

``ring run INPUT`` — turns of a bunch through the ring (the ring kernels on
every CPU core, or a CUDA GPU), at its fixed reference energy or, with
``--ramp FILE``, through a ramp (the reference energy, the RF voltage and
phase programs; ``--to-end`` runs to the end of the energy program), with
``--space-charge FILE`` (or ``--intensity N``) with space charge, with
``--injection FILE`` filled turn by turn through a stripping foil (the bump
and painting magnets following their waveforms): the statistics turn by
turn, snapshots, losses, the tune probes' NAFF tunes and the foil's record,
written to ``<stem>_ring.h5`` (``--out``); checkpoints and ``--restart``.

``ring ramp INPUT --ramp FILE`` — the reference particle alone through a
ramp: the turns to the end of the program, the transition turn, the RF
frequency range, the largest gain asked of the voltage, the turns capped;
``--csv`` writes the table turn by turn.

``ring optics INPUT`` — the periodic optics of a ring from the ring kernels:
tunes with their integer part, chromaticities, momentum compaction, slip
factor and transition gamma, the closed orbit and Twiss at the start, and
with ``--twiss-out`` the Twiss functions, dispersion and closed orbit at
every element (``.csv`` or ``.json``); ``--six-d`` adds the synchrotron tune
and the stationary RF bucket.

The ring runs at the beam the lattice's magnets were set for — a MAD-X deck's
BEAM, a project's beam — or, for a bare TraceWin deck, at ``--energy`` /
``--species``.  ``--energy`` with a deck that has its own beam runs the ring
at that energy with the magnets following (the same normalized strengths).
A project's run starts from its beam (the GUI's Beam tab: distribution,
particles — ``--particles`` changes their number — emittances and Twiss
functions) unless ``--source`` says otherwise; a bare lattice file's from a
Gaussian matched to the ring with ``--particles`` and the ``--beam``
emittances.  A project's element model, pole faces and constants (the GUI's
Ring tab) are used unless the command line gives them, its ramp (the Ring
tab's Programs page) unless ``--ramp`` gives another or ``--fixed-energy``
none, its space charge (the Space charge page) unless ``--space-charge``
gives other settings or ``--no-space-charge`` none, and its injection (the
Injection page) unless ``--injection`` gives other settings or
``--no-injection`` none.

Nothing here loads ring mode until the command runs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from linac_gen.cli import common


def add_arguments(p) -> None:
    for action in p._actions:
        # (`ring --threads N run …`: the shared option, in ring mode's words)
        if "--threads" in action.option_strings:
            action.help = ("CPU threads for the ring kernels (default: $HELIX_THREADS, else "
                           "$OMP_NUM_THREADS, else every CPU); the results are the same for "
                           "any number")
    sub = p.add_subparsers(dest="ring_command", required=True)
    o = sub.add_parser("optics", help="periodic optics of a ring (tunes, chromaticity, "
                                      "momentum compaction, Twiss)")
    _model_arguments(o)
    o.add_argument("--six-d", action="store_true", dest="six_d",
                   help="also the 6-D closed orbit, synchrotron tune and RF bucket (RF on)")
    o.add_argument("--twiss-out", default=None, dest="twiss_out", metavar="FILE",
                   help="write the Twiss table per element (.csv or .json)")
    o.add_argument("--json", action="store_true", help="print the summary as JSON")

    r = sub.add_parser("run", help="track a bunch turn after turn (fixed energy, or a ramp): "
                                   "statistics, snapshots, losses, NAFF tunes → <stem>_ring.h5")
    _model_arguments(r)
    r.add_argument("--turns", type=int, default=None,
                   help="number of turns (default 100)")
    _ramp_arguments(r)
    r.add_argument("--to-end", action="store_true", dest="to_end",
                   help="run the turns to the end of the ramp's energy program (instead of "
                        "--turns)")
    r.add_argument("--corrector-mode", choices=("follow", "fixed_field"), default=None,
                   dest="corrector_mode",
                   help="in a ramp, steerers follow the reference momentum (default) or keep "
                        "their field")
    r.add_argument("--space-charge", default=None, dest="space_charge", metavar="FILE",
                   help="space charge: a JSON (or YAML) file of the settings (model, "
                        "intensity, kicks_per_turn, grid, pipe, longitudinal, ...; the "
                        "manual's Space charge page)")
    r.add_argument("--intensity", type=float, default=None,
                   help="real particles in the bunch, with space charge (alone: space charge "
                        "with the default settings)")
    r.add_argument("--no-space-charge", action="store_true", dest="no_space_charge",
                   help="no space charge, although the project holds its settings")
    r.add_argument("--injection", default=None, metavar="FILE",
                   help="injection through a stripping foil: a JSON (or YAML) file of the "
                        "settings (foil, source, turns, particles_per_turn, current_ma, "
                        "chopper, magnets; the manual's Injection page)")
    r.add_argument("--no-injection", action="store_true", dest="no_injection",
                   help="no injection, although the project holds its settings")
    r.add_argument("--particles", type=int, default=None,
                   help="macroparticles at the start (default: a project's beam's; 10000 "
                        "for the matched source)")
    r.add_argument("--source", choices=("matched", "beam", "empty"), default=None,
                   help="beam: the project's beam (its distribution, particles, emittances "
                        "and Twiss functions, with --beam's changes), generated at the "
                        "ring's species and energy (default for a project); matched: a "
                        "Gaussian matched to the ring's optics with the beam's emittances "
                        "(default for a bare lattice file); empty: no beam at the start "
                        "(default with injection)")
    r.add_argument("--beam", action="append", default=[], metavar="NAME=VALUE",
                   help="beam setting, e.g. emit_nx=1.5 (pi mm mrad, normalized rms), "
                        "emit_z, beta_z, frequency, cutoff (repeatable)")
    r.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    r.add_argument("--snapshot-turns", default="", dest="snapshot_turns", metavar="T1,T2,...",
                   help="turns at which the whole distribution is kept (0 = the start, "
                        "last = the last turn)")
    r.add_argument("--stats-every", type=int, default=None, dest="stats_every",
                   help="turns between statistics rows (default: every turn, or every k turns "
                        "to keep at most 100 000 rows; with --restart, the checkpoint's)")
    r.add_argument("--probes", type=int, default=None,
                   help="tune probes: 0 none, 1 the tunes (default), n: an n x n footprint "
                        "(with --restart, the checkpoint's)")
    r.add_argument("--probe-max-sigma", type=float, default=None, dest="probe_max_sigma",
                   help="largest footprint amplitude in rms beam sizes (default 3; with "
                        "--restart, the checkpoint's)")
    r.add_argument("--checkpoint-every", type=int, default=0, dest="checkpoint_every",
                   help="turns between checkpoints (<stem>_ring_checkpoint.npz; 0 none)")
    r.add_argument("--restart", default=None, metavar="CHECKPOINT",
                   help="continue the run of this checkpoint (bit for bit; its probes and "
                        "statistics interval unless given)")
    r.add_argument("--seed", type=int, default=None,
                   help="random seed of the beam, the probes and an injection's draws "
                        "(default 1; with --restart, the checkpoint's)")
    r.add_argument("--out", default=".", help="output directory (default .)")
    r.add_argument("--json", action="store_true", help="print the summary as JSON")

    m = sub.add_parser("ramp", help="the reference particle alone through a ramp: turns to the "
                                    "end, transition, RF frequency, gain against the voltage")
    _model_arguments(m)
    _ramp_arguments(m)
    m.add_argument("--turns", type=int, default=None,
                   help="number of turns (default: to the end of the energy program)")
    m.add_argument("--csv", default=None, metavar="FILE",
                   help="write the reference turn by turn (time, energy, gain, voltage, phase, "
                        "frequency)")
    m.add_argument("--json", action="store_true", help="print the summary as JSON")


def _ramp_arguments(o) -> None:
    o.add_argument("--ramp", default=None, metavar="FILE",
                   help="the ramp: a JSON (or YAML) file of the energy, voltage and phase "
                        "programs, tables inline or in CSV files (the manual's Programs page)")
    o.add_argument("--fixed-energy", action="store_true", dest="fixed_energy",
                   help="no ramp, although the project holds one")
    o.add_argument("--gamma-t", type=float, default=None, dest="gamma_t",
                   help="transition gamma of the ramp (default: the ramp's own, else from the "
                        "ring's optics)")


def _model_name(value):
    """A ring model's name; the names of earlier versions map to today's."""
    if not isinstance(value, str):
        return value
    from linac_gen.ring.config import MODEL_ALIASES
    return MODEL_ALIASES.get(value, value)


def _model_arguments(o) -> None:
    o.add_argument("input", help="a .lgproj project or a lattice file (.madx, .dat, ...)")
    # (also after the subcommand: `ring run deck --threads 4`; SUPPRESS keeps
    # a value given before it, `ring --threads 4 run deck`)
    o.add_argument("--threads", metavar="N|all", type=common._threads_value,
                   default=argparse.SUPPRESS,
                   help="CPU threads for the ring kernels (default: $HELIX_THREADS, else "
                        "$OMP_NUM_THREADS, else every CPU); the results are the same for any "
                        "number")
    o.add_argument("--energy", type=float, help="kinetic energy of the ring (MeV)")
    o.add_argument("--species", default=None,
                   help="the beam the lattice's magnets were set for, when the lattice "
                        "does not say (proton / H-); the ring tracks protons")
    o.add_argument("--model", choices=("cartesian", "curvilinear"), default=None,
                   type=_model_name,
                   help="element maps: cartesian (bend multipoles expanded in Cartesian "
                        "coordinates, as MAD-X decks define them; default) or curvilinear (exact "
                        "elements, bend multipoles in the curved frame); a project's own "
                        "choice by default")
    o.add_argument("--edge-model", choices=("exact", "full", "linear"), default=None,
                   dest="edge_model",
                   help="pole faces: exact (frame rotation and hard-edge fringe; default "
                        "for cartesian), full (also the quadrupole fringe, and the wedge's "
                        "quadrupole field to first order), linear (thin linear kicks with the "
                        "fringe correction; the curvilinear model's); a project's own choice "
                        "by default")
    o.add_argument("--constants", choices=("helix", "codata2010", "codata2022"),
                   default=None, help="physical constants of the reference (default helix, "
                                      "or a project's own choice)")
    o.add_argument("--harmonic", type=int, default=None,
                   help="lock every cavity to this harmonic of the revolution frequency")
    o.add_argument("--bend-steps", type=int, default=0, dest="bend_steps",
                   help="integration steps per bend (default 0: automatic)")


def run(args) -> int:
    if args.ring_command == "optics":
        return _optics(args)
    if args.ring_command == "run":
        return _run_ring(args)
    if args.ring_command == "ramp":
        return _ramp_table(args)
    return 2


def _inputs(args):
    """(source path, lattice, beam, element-model settings) of the command,
    or an exit code."""
    src = Path(args.input)
    if not src.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        from linac_gen.core.config import BeamConfig
    except ImportError as exc:                        # pragma: no cover
        print(f"error: ring mode is not available: {exc}", file=sys.stderr)
        return 2
    try:
        lattice, beam, _conv = common.load_input(src)
    except (OSError, ValueError) as exc:
        # (e.g. a project whose lattice file is missing)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from linac_gen.ring.compile import declared_beam
    is_project = src.suffix.lower() == ".lgproj"
    own = declared_beam(lattice) is not None
    if not own and not is_project:
        if args.energy is None:
            print("error: this lattice does not say which beam the ring runs at: "
                  "give --energy (and --species)", file=sys.stderr)
            return 2
        beam = BeamConfig(species=args.species or "proton", energy=args.energy)
    elif args.species and not own:
        beam.species = args.species
    elif args.species:
        print("note: the lattice's own beam sets the species its magnets were set for "
              "(--species is not used)", file=sys.stderr)
    saved = _project_ring_settings(src) if is_project else {}
    saved_model = _model_name(saved.get("model")) or "cartesian"
    model = args.model if args.model is not None else saved_model
    edge = args.edge_model if args.edge_model is not None else (
        saved.get("edge_model") if model == saved_model else None)
    constants = args.constants if args.constants is not None else saved.get("constants", "helix")
    settings = dict(design_energy_mev=args.energy, constants=constants, model=model,
                    edge_model=edge, harmonic=args.harmonic, bend_steps=args.bend_steps)
    return src, lattice, beam, settings


def _optics(args) -> int:
    got = _inputs(args)
    if isinstance(got, int):
        return got
    src, lattice, beam, settings = got
    from linac_gen.ring.compile import RingCompileError, compile_from_config
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.optics import rf_bucket, twiss
    from linac_gen.ring.optics.normal_form import UnstableError
    try:
        cfg = RingConfig(**settings)
        rl = compile_from_config(lattice, cfg, beam=beam)
        opt = twiss(rl, six_d=args.six_d)
        bucket = rf_bucket(rl, opt) if args.six_d else None
    except (RingCompileError, UnstableError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = opt.summary()
    summary.update({"input": src.name, "n_elements": len(rl.element_names),
                    "n_nodes": rl.n_nodes, "kinetic_energy_mev": rl.reference.kinetic0,
                    "constants": cfg.constants, "options": rl.options,
                    "harmonic": rl.harmonic, "warnings": rl.warnings})
    if bucket is not None:
        summary["bucket"] = {k: v for k, v in bucket.items() if k != "separatrix"}
    if args.twiss_out:
        _write_table(opt.table, Path(args.twiss_out))
    if args.json:
        # (strict JSON: a number that is not finite — the synchrotron tune of
        # the 4-D optics, say — is null)
        print(json.dumps(_finite_json(summary), indent=1, allow_nan=False, default=float))
        return 0
    _print_summary(summary, opt, rl)
    if args.twiss_out:
        print(f"Twiss table: {args.twiss_out} ({len(opt.table['name'])} elements)")
    return 0


def _ramp_setting(args, src: Path):
    """(the command's ramp as a dict — ``RampConfig.to_dict`` — or None, where
    it comes from): ``--ramp``'s file, else the project's own unless
    ``--fixed-energy``; ``--gamma-t`` replaces its transition gamma."""
    import dataclasses

    from linac_gen.ring.programs import RampConfig, ramp_from_file
    if args.fixed_energy and args.ramp:
        raise ValueError("--ramp and --fixed-energy: give one of them")
    rc, where = None, None
    if args.ramp:
        rc, where = ramp_from_file(args.ramp), "file"
    elif not args.fixed_energy and src.suffix.lower() == ".lgproj":
        saved = _project_ring_settings(src).get("ramp")
        if saved is not None:
            rc, where = RampConfig.from_dict(saved), "project"
    if args.gamma_t is not None:
        if rc is None:
            raise ValueError("--gamma-t belongs to a ramp: give --ramp FILE (or a project "
                             "with a ramp)")
        rc = dataclasses.replace(rc, gamma_t=args.gamma_t)
    return (None, None) if rc is None else (rc.to_dict(), where)


def _space_charge_setting(args, src: Path):
    """(the command's space-charge settings as a dict, or None; where they come
    from): ``--space-charge``'s file, else the project's own unless
    ``--no-space-charge``; ``--intensity`` replaces the intensity (alone: the
    default settings with it)."""
    from linac_gen.ring.sc.config import SpaceChargeConfig, space_charge_from_file
    if args.no_space_charge and (args.space_charge or args.intensity is not None):
        raise ValueError("--no-space-charge with --space-charge / --intensity: give one of "
                         "them")
    sc, where = None, None
    if args.space_charge:
        sc, where = space_charge_from_file(args.space_charge).to_dict(), "file"
    elif not args.no_space_charge and src.suffix.lower() == ".lgproj":
        saved = _project_ring_settings(src).get("space_charge")
        if saved is not None:
            sc, where = SpaceChargeConfig.from_dict(saved).to_dict(), "project"
    if args.intensity is not None:
        sc = SpaceChargeConfig.from_dict(dict(sc or {}, intensity=args.intensity)).to_dict()
        where = where or "command line"
    return sc, where


def _injection_setting(args, src: Path):
    """(the command's injection settings as a dict, or None; where they come
    from): ``--injection``'s file, else the project's own unless
    ``--no-injection``."""
    from linac_gen.ring.injection.config import InjectionConfig, injection_from_file
    if args.no_injection and args.injection:
        raise ValueError("--injection and --no-injection: give one of them")
    if args.injection:
        return injection_from_file(args.injection).to_dict(), "file"
    if not args.no_injection and src.suffix.lower() == ".lgproj":
        saved = _project_ring_settings(src).get("injection")
        if saved is not None:
            return InjectionConfig.from_dict(saved).to_dict(), "project"
    return None, None


def _run_ring(args) -> int:
    got = _inputs(args)
    if isinstance(got, int):
        return got
    src, lattice, beam, settings = got
    import dataclasses
    import time

    from linac_gen.ring.compile import RingCompileError
    from linac_gen.ring.config import RingConfig, auto_stats_every, parse_snapshot_turns
    from linac_gen.ring.engine import RingSimulation, checkpoint_config
    from linac_gen.ring.results import save_ring_hdf5
    try:
        ramp, ramp_from = _ramp_setting(args, src)
        sc, sc_from = _space_charge_setting(args, src)
        inj, inj_from = _injection_setting(args, src)
        if args.to_end and ramp is None:
            raise ValueError("--to-end runs to the end of a ramp: give --ramp FILE")
        if args.to_end and args.turns is not None:
            raise ValueError("give --turns or --to-end, not both")
        if args.corrector_mode is not None and ramp is None:
            raise ValueError("--corrector-mode applies to a ramp: give --ramp FILE")
        # (in a ramp: the command line's, else a project's own; at fixed
        # energy steerers keep their kicks)
        corrector_mode = "follow" if ramp is None else (
            args.corrector_mode or _project_ring_settings(src).get("corrector_mode", "follow"))
        overrides = {}
        for item in args.beam:
            if "=" not in item:
                raise ValueError(f"--beam expects NAME=VALUE, got {item!r}")
            k, v = item.split("=", 1)
            overrides[k.strip()] = v.strip()
        common.apply_beam_overrides(beam, overrides)
        snaps = parse_snapshot_turns(args.snapshot_turns)
        saved = {}
        if args.restart:
            if not Path(args.restart).is_file():
                raise ValueError(f"--restart: no checkpoint file {args.restart}")
            import zipfile
            try:
                saved = checkpoint_config(args.restart)
            except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
                raise ValueError(f"--restart: {args.restart} is not a ring checkpoint "
                                 f"({exc})") from None
        out = Path(args.out)
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"--out: {args.out} is not a directory ({exc})") from None
        ck = out / f"{src.stem}_ring_checkpoint.npz"
        n_turns = 100 if args.turns is None else args.turns
        # (a continued run keeps the checkpoint's probes, statistics interval and
        # seed unless they are given; a project's run its saved probes, as the GUI)
        project_probes = (_project_run_setting(src, "tune_probes")
                          if src.suffix.lower() == ".lgproj" else None)
        probes = (args.probes if args.probes is not None
                  else saved.get("tune_probes", project_probes if project_probes is not None
                                 else 1))
        seed = args.seed if args.seed is not None else saved.get("seed", 1)
        pmax = (args.probe_max_sigma if args.probe_max_sigma is not None
                else saved.get("probe_max_sigma", 3.0))

        def every_for(n):
            return (args.stats_every if args.stats_every is not None
                    else saved.get("stats_every") or auto_stats_every(n))

        # (a project's run starts from its beam, as the GUI's — or, as its Run
        # page saved it, from an empty ring filled by its injection; a bare
        # lattice file has default beam settings: a beam matched to the ring)
        from_project = src.suffix.lower() == ".lgproj" and _project_has_beam(src)
        default = "empty" if inj is not None else ("beam" if from_project else "matched")
        if from_project and inj_from != "file":
            # (the start the Run page saved, with the project's own injection:
            # its beam — with the injection too — or an empty ring)
            saved_start = _project_source(src)
            if saved_start == "beam" or (saved_start == "empty" and inj is not None):
                default = saved_start
        source = args.source or default
        if source == "empty" and args.particles is not None:
            raise ValueError("--particles: the run starts from an empty ring (its injection "
                             "fills it) — there are no particles at the start to set")
        if source == "beam" and args.particles is not None:
            if getattr(beam, "source", "generate") == "file":
                raise ValueError("--particles: the beam is read from a distribution file, "
                                 "whose particles are all used")
            beam.n_particles = int(args.particles)
        cfg = RingConfig(n_turns=n_turns, source=source,
                         n_particles=args.particles if args.particles is not None else 10000,
                         device=args.device, snapshot_turns=snaps,
                         stats_every=every_for(n_turns), tune_probes=probes,
                         probe_max_sigma=pmax, checkpoint_every=args.checkpoint_every,
                         seed=seed, ramp=ramp, corrector_mode=corrector_mode,
                         space_charge=sc, injection=inj, **settings)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    last = [0.0]

    def progress(turn, n):
        now = time.monotonic()
        if not args.json and (now - last[0] > 2.0 or turn == n):
            last[0] = now
            print(f"  turn {turn}/{n}", file=sys.stderr, flush=True)

    try:
        sim = RingSimulation(lattice, beam, cfg)
        if args.to_end:
            r = sim.ramp()
            n = r.turns_to_end()
            if n is None:
                raise ValueError("the ramp's energy program is flat, it has no end: give "
                                 "--turns")
            if n < 1:
                r.orbit(0)                  # (refused, saying why)
            cfg = dataclasses.replace(cfg, n_turns=n, stats_every=every_for(n))
            sim.config = cfg
        if (src.suffix.lower() != ".lgproj" and not args.beam and not args.restart
                and cfg.source in ("matched", "beam") and not args.json):
            # (a bare lattice file: the beam settings are the defaults — the
            # matched beam's emittances and bunch length come from them)
            print(f"note: the beam settings are the defaults (normalized rms emittances "
                  f"{beam.emit_nx:g} and {beam.emit_ny:g} pi mm mrad, emit_z {beam.emit_z:g} "
                  f"deg MeV at {beam.frequency:g} MHz): give the beam's with --beam "
                  "(emit_nx=, emit_ny=, emit_z=, beta_z=, frequency=)", file=sys.stderr)
        if from_project and cfg.source == "beam" and not args.restart and not args.json:
            what = (f"its distribution file {Path(str(beam.distribution_file)).name}"
                    if getattr(beam, "source", "generate") == "file"
                    else f"{beam.n_particles} particles"
                    + (" (--particles)" if args.particles is not None else ""))
            print(f"beam: the project's, {what} (--source matched starts from a Gaussian "
                  "matched to the ring)", file=sys.stderr)
        if ramp_from == "project" and not args.json:
            print("ramp: the project's (--fixed-energy runs without it)", file=sys.stderr)
        if sc_from == "project" and not args.json:
            print("space charge: the project's" + (" with --intensity's" if args.intensity
                                                    is not None else "")
                  + " (--no-space-charge runs without it)", file=sys.stderr)
        if inj_from == "project" and not args.json:
            print("injection: the project's (--no-injection runs without it)", file=sys.stderr)
        res = sim.run(progress_callback=progress,
                      checkpoint_path=ck if args.checkpoint_every else None,
                      restart_from=args.restart)
    except (RingCompileError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    path = save_ring_hdf5(res, out / f"{src.stem}_ring.h5", lattice_path=str(src),
                          beam_config=beam)
    summary = _run_summary(res, path)
    if args.json:
        # (strict JSON: a number that is not finite — the emittances at the
        # start of a ring that starts empty, say — is null)
        print(json.dumps(_finite_json(summary), indent=1, allow_nan=False))
    else:
        _print_run(summary)
    # (a ramp's turn refused — the gain beyond the voltage — stopped the run:
    # the turns done are written, the status says it did not end)
    return 1 if res.meta.get("ramp_refused") else 0


def _ramp_table(args) -> int:
    got = _inputs(args)
    if isinstance(got, int):
        return got
    src, lattice, beam, settings = got
    from linac_gen.ring.compile import RingCompileError
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.engine import RingSimulation
    try:
        ramp_d, _where = _ramp_setting(args, src)
        if ramp_d is None:
            raise ValueError("no ramp: --fixed-energy asks for none" if args.fixed_energy else
                             "no ramp: give --ramp FILE (or a project with a ramp)")
        if args.turns is not None and args.turns < 1:
            raise ValueError("--turns must be at least 1")
        cfg = RingConfig(ramp=ramp_d, **settings)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        ramp = RingSimulation(lattice, beam, cfg).ramp()
        if args.turns is None:
            n_end = ramp.turns_to_end()
        else:
            # (the program's own length for the summary, unless it is very long)
            from linac_gen.ring.engine import TURNS_TO_END_CAP
            from linac_gen.ring.programs import RampError
            try:
                n_end = ramp.turns_to_end(max_turns=TURNS_TO_END_CAP)
            except RampError:
                n_end = None
        n = args.turns if args.turns is not None else n_end
        if n is None:
            raise ValueError("the energy program is flat, it has no end: give --turns")
        orbit = ramp.orbit(n)
    except (RingCompileError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    gt_from = ("from --gamma-t" if args.gamma_t is not None else
               "given by the ramp" if ramp_d.get("gamma_t") is not None
               else "from the ring's optics")
    summary = _ramp_summary(src, cfg, ramp, orbit, n, n_end, gt_from)
    if args.csv:
        _write_ramp_csv(orbit, ramp.mass, math.degrees(ramp.arg), Path(args.csv))
    if args.json:
        # (strict JSON: a number that is not finite — the synchrotron tune of
        # the 4-D optics, say — is null)
        print(json.dumps(_finite_json(summary), indent=1, allow_nan=False, default=float))
        return 0
    _print_ramp(summary)
    if args.csv:
        print(f"turn table: {args.csv} ({n} turns)")
    return 0


def _ramp_summary(src, cfg, ramp, o, n, n_end, gt_from) -> dict:
    m = ramp.mass
    gain, volts = o["gain_mev"], o["voltage_mv"] * ramp.r_unit
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(volts > 0.0, np.abs(gain) / volts, np.nan)
    k_tr = o["transition_turn"]
    capped = np.flatnonzero(o["capped"])
    rc = ramp.cfg
    return {
        "input": src.name, "length_m": ramp.length, "harmonic": int(ramp.h),
        "mass_mev": m, "constants": cfg.constants, "turns": int(n), "turns_to_end": n_end,
        "program": rc.to_dict(), "t_start_s": rc.t_start_s, "t_end_program_s": rc.energy.t_end(),
        "kinetic_start_mev": float(o["energy_mev"][0] - m),
        "kinetic_end_mev": float(o["energy_end_mev"] - m), "gamma_end": float(o["gamma_end"]),
        "t_end_s": float(o["t_end_s"]), "gamma_t": ramp.gamma_t, "gamma_t_from": gt_from,
        "transition_turn": k_tr,
        "transition_t_s": None if k_tr is None else float(o["t_s"][k_tr]),
        "above_transition_at_start": bool(o["above_transition_at_start"]),
        "frequency_hz": [float(o["frequency_hz"].min()), float(o["frequency_hz"].max())],
        "max_gain_mev": float(np.abs(gain).max()), "max_gain_turn": int(np.argmax(np.abs(gain))),
        "max_gain_over_voltage": (float(np.nanmax(x)) if np.isfinite(x).any() else None),
        "max_gain_over_voltage_turn": (int(np.nanargmax(x)) if np.isfinite(x).any() else None),
        "capped_turns": int(capped.size),
        "first_capped_turn": int(capped[0]) if capped.size else None,
        "voltage_eff_factor": float(ramp.r_unit),
        "phase_offset_deg": float(np.degrees(ramp.arg)),
    }


def _print_ramp(s) -> None:
    rc = s["program"]
    e, v, ph = rc["energy"], rc["voltage"], rc["phase"]
    print(f"ramp: {s['input']} — {s['length_m']:.9g} m, harmonic {s['harmonic']}, proton "
          f"(constants {s['constants']})")
    if e["kind"] == "sine":
        start = ("the ring's" if e["start_kinetic_mev"] is None
                 else f"{e['start_kinetic_mev']:.6g}")
        prog = (f"sine {start} -> {e['end_kinetic_mev']:.6g} MeV kinetic at "
                f"{e['frequency_hz']:.6g} Hz")
    elif e["kind"] == "table":
        prog = f"table of {len(e['times_s'])} points ({e['quantity']})"
    else:
        prog = "flat"
    volt = ("the cavities' own" if v["kind"] == "constant"
            else f"table of {len(v['times_s'])} points")
    phase = ("from the gain, " + ("capped at 90 deg" if ph["on_excess"] == "cap" else
                                  "refused") + " beyond the voltage"
             if ph["kind"] == "auto" else f"table of {len(ph['times_s'])} points")
    print(f"programs: energy {prog}; voltage {volt}; phase {phase}")
    if rc["paraphasing"]:
        print(f"paraphasing: {len(rc['paraphasing'])} group(s), net voltage "
              f"{s['voltage_eff_factor']:.6g} of the sum, phase offset "
              f"{s['phase_offset_deg']:.6g} deg")
    if s["turns_to_end"] is None:
        end = ""
    elif s["turns_to_end"] == s["turns"]:
        end = (f" to the end of the energy program (t = {s['t_end_program_s'] * 1e3:.9g} ms "
               "on its clock)")
    else:
        end = (f" ({s['turns_to_end']} to the end of the energy program, t = "
               f"{s['t_end_program_s'] * 1e3:.9g} ms on its clock)")
    print(f"turns: {s['turns']} from t = {s['t_start_s'] * 1e3:.9g} ms{end}")
    print(f"reference: {s['kinetic_start_mev']:.9g} -> {s['kinetic_end_mev']:.9g} MeV kinetic "
          f"(gamma {s['gamma_end']:.9f} at the end), t = {s['t_end_s'] * 1e3:.9g} ms at the end")
    gt = "none" if s["gamma_t"] is None else f"{s['gamma_t']:.10g}"
    if s["transition_turn"] is not None:
        print(f"transition: gamma_t {gt} ({s['gamma_t_from']}), crossed at turn "
              f"{s['transition_turn']} (t = {s['transition_t_s'] * 1e3:.9g} ms)")
    else:
        side = "above it from the start" if s["above_transition_at_start"] else "not crossed"
        print(f"transition: gamma_t {gt} ({s['gamma_t_from']}), {side}")
    f0, f1 = s["frequency_hz"]
    print(f"RF frequency: {f0 / 1e6:.9g} to {f1 / 1e6:.9g} MHz")
    ratio = ("" if s["max_gain_over_voltage"] is None else
             f"; largest gain / voltage {s['max_gain_over_voltage']:.6g} "
             f"(turn {s['max_gain_over_voltage_turn']})")
    print(f"energy gain per turn: largest {s['max_gain_mev'] * 1e3:.6g} keV (turn "
          f"{s['max_gain_turn']}){ratio}")
    if s["capped_turns"]:
        print(f"turns capped at 90 deg (the gain beyond the voltage): {s['capped_turns']}, the "
              f"first at turn {s['first_capped_turn']}")
    else:
        print("turns capped at 90 deg: none")


def _write_ramp_csv(o, mass, offset_deg, path: Path) -> None:
    """The reference turn by turn; phi_s_deg is the synchronous phase of the
    net voltage (the phase rule's plus the offset of the paraphasing groups
    and of negative voltages)."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn", "t_s", "kinetic_mev", "gamma", "gain_mev", "voltage_mv",
                    "phi_s_deg", "frequency_hz", "above_transition", "capped"])
        for k in range(len(o["t_s"])):
            e = float(o["energy_mev"][k])
            w.writerow([k, repr(float(o["t_s"][k])), repr(e - mass), repr(e / mass),
                        repr(float(o["gain_mev"][k])), repr(float(o["voltage_mv"][k])),
                        repr(float(o["phi_deg"][k]) + offset_deg),
                        repr(float(o["frequency_hz"][k])),
                        int(o["above_transition"][k]), int(o["capped"][k])])


def _finite_json(v):
    """``v`` with every number that is not finite as None (numpy numbers and
    arrays as plain ones), for strict JSON."""
    if isinstance(v, dict):
        return {k: _finite_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_finite_json(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return v


def _run_summary(res, path) -> dict:
    en = res.emit_normalized
    el = res.emit_long_evs
    names = res.meta["element_names"]
    lost_at = {}
    for i in np.asarray(res.losses.get("element", []), dtype=int):
        nm = names[i] if 0 <= i < len(names) else "?"
        lost_at[nm] = lost_at.get(nm, 0) + 1
    o = res.optics
    n0, surv = int(res.n_alive[0]), float(res.survival[-1]) if res.n_alive[0] else math.nan
    if (res.injection or {}).get("settings"):
        # (with injection: the particles the ring received — the stripped
        # ones — and those still circulating)
        w = res.injection.get("waste") or {}
        n0 = int(res.ledger.get("injected", 0)) - int(w.get("missed", 0)) \
            - int(w.get("unstripped", 0))
        surv = float(res.n_alive[-1]) / n0 if n0 else math.nan
    return {
        "file": str(path), "turns_done": res.turns_done, "turns_requested": res.n_turns_requested,
        "truncated": res.truncated, "particles": n0,
        "survival": surv, "lost": int(res.ledger.get("lost", 0)),
        "lost_at": dict(sorted(lost_at.items(), key=lambda kv: -kv[1])[:8]),
        "emit_nx_mm_mrad": [float(en[0, 0] * 1e6), float(en[-1, 0] * 1e6)],
        "emit_ny_mm_mrad": [float(en[0, 1] * 1e6), float(en[-1, 1] * 1e6)],
        "emit_long_evs": [float(el[0]), float(el[-1])],
        "tunes_naff": res.tunes, "tunes_optics": {"qx": o.get("qx"), "qy": o.get("qy"),
                                                  "qs": o.get("qs")},
        "eigen_tunes_6d": o.get("eigen_tunes_6d"), "wall_s": res.wall_s,
        "device": res.meta["device"], "threads": res.meta["threads"],
        "species": res.meta["species"], "kinetic_mev": res.meta["kinetic_mev"],
        "length_m": res.meta["length_m"], "stats_every": res.meta.get("stats_every", 1),
        "notes": list(res.meta.get("notes", [])),
        "start_mismatch": res.meta.get("start_mismatch"),
        "ramp": _ramp_run_summary(res),
        "space_charge": _sc_run_summary(res),
        "injection": _injection_run_summary(res),
    }


def _injection_run_summary(res):
    """The foil's and the injection's totals (None without a foil): the real
    particles and energies the ring's (every filled bucket's), the
    macroparticles the tracked bucket's."""
    i = res.injection
    if not i:
        return None
    f = i.get("foil", {})
    led = res.ledger
    return {"foil": {k: f.get(k) for k in ("name", "material", "thickness_ug_cm2",
                                             "strip_model", "theta0_rad", "mean_loss_mev",
                                             "straggling_used", "stripped_fraction",
                                             "x_min_m", "x_max_m", "y_min_m", "y_max_m")},
            "injected": int(np.sum(i["injected"])),
            "injected_particles": float(np.sum(i["injected_charge"])),
            "buckets": i.get("buckets"),
            "circulating": led.get("circulating"), "waste": i.get("waste"),
            "foil_hits": int(np.sum(i["foil_hits"])),
            "mean_hits_circulating": i.get("mean_hits_circulating"),
            "deposit_mev": float(np.sum(i["deposit_mev"])),
            "electron_mev": float(np.sum(i.get("electron_mev", 0.0))),
            "settings": i.get("settings")}


def _sc_run_summary(res):
    """A space-charge run's settings and what it measured (None without)."""
    m = res.meta.get("space_charge")
    if not m:
        return None
    keys = ("model", "intensity", "intensity_used", "kicks", "kicks_per_turn",
            "largest_spacing_m", "grid", "deposit", "box_sigma", "z_bins", "z_smoothing_bins",
            "boundary_order", "g_factor", "kicks_charge_beyond_wall", "pipe_fit_residual",
            "pipe",
            "pipe_a_m", "pipe_b_m", "reference_radius_m", "longitudinal", "longitudinal_kick",
            "bunched", "refresh_turns", "outside_grid_max", "laslett_gaussian",
            "line_density_resolution", "z_bins_used", "z_bins_changes")
    return {k: m.get(k) for k in keys}


def _ramp_run_summary(res):
    """A ramp run's reference: start and end, transition, capped turns (None
    at fixed energy)."""
    r = res.ramp
    if not r:
        return None
    n = len(r["t_s"])
    return {"kinetic_start_mev": float(r["kinetic_mev"][0]) if n else None,
            "kinetic_end_mev": r["kinetic_end_mev"], "gamma_end": r["gamma_end"],
            "t_start_s": float(r["t_s"][0]) if n else None, "t_end_s": r["t_end_s"],
            "gamma_t": r.get("gamma_t"), "transition_turn": r["transition_turn"],
            "above_transition_at_start": r.get("above_transition_at_start"),
            "n_capped": r["n_capped"], "turns_to_end": r["turns_to_end"],
            "frequency_hz": ([float(np.min(r["frequency_hz"])), float(np.max(r["frequency_hz"]))]
                             if n else None)}


def _print_run(s) -> None:
    from linac_gen.ring.results import survival_text
    st = "stopped" if s["truncated"] else "done"
    print(f"ring run {st}: {s['turns_done']} of {s['turns_requested']} turns, "
          f"{s['particles']} particles, {s['species']} {s['kinetic_mev']:.6g} MeV, "
          f"{s['length_m']:.6g} m ({s['device']}, {s['threads']} threads, {s['wall_s']:.1f} s)")
    for note in s.get("notes", []):
        print(f"note: {note}")
    bm = s.get("start_mismatch") or []
    if any(v is not None for v in bm):
        print("start: BMAG against the ring's matched beam (1 when matched) "
              + ", ".join(f"{p} {v:.4f}" for p, v in zip("xyz", bm, strict=False)
                          if v is not None))
    r = s.get("ramp")
    if r:
        gt = "none" if r["gamma_t"] is None else f"{r['gamma_t']:.10g}"
        tr = (f"crossed at turn {r['transition_turn']}" if r["transition_turn"] is not None
              else ("above it from the start" if r.get("above_transition_at_start")
                    else "not crossed"))
        print(f"ramp: {r['kinetic_start_mev']:.9g} -> {r['kinetic_end_mev']:.9g} MeV kinetic "
              f"(gamma {r['gamma_end']:.9f}), t {r['t_start_s'] * 1e3:.9g} -> "
              f"{r['t_end_s'] * 1e3:.9g} ms; transition (gamma_t {gt}) {tr}"
              + (f"; {r['n_capped']} turn(s) capped at 90 deg" if r["n_capped"] else ""))
    c = s.get("space_charge")
    if c:
        _print_space_charge(c)
    j = s.get("injection")
    if j:
        _print_injection(j)
    if s.get("stats_every", 1) > 1:
        print(f"statistics every {s['stats_every']} turns")
    print(f"survival {survival_text(s['particles'], s['particles'] - s['lost'])}"
          + ("" if not s["lost_at"] else "; lost at " + ", ".join(
              f"{k} {v}" for k, v in s["lost_at"].items())))
    ex, ey, el = s["emit_nx_mm_mrad"], s["emit_ny_mm_mrad"], s["emit_long_evs"]
    if math.isfinite(ex[0]):
        print(f"normalized rms emittance x {ex[0]:.6g} -> {ex[1]:.6g}  y {ey[0]:.6g} -> "
              f"{ey[1]:.6g} pi mm mrad; longitudinal {el[0]:.6g} -> {el[1]:.6g} eV s (rms)")
    else:
        # (a ring that started empty)
        print(f"normalized rms emittance at the end x {ex[1]:.6g}  y {ey[1]:.6g} pi mm mrad; "
              f"longitudinal {el[1]:.6g} eV s (rms)")
    t, o = s["tunes_naff"], s["tunes_optics"]
    if t:
        line = "NAFF tunes (probe) " + "  ".join(
            f"{k.capitalize()} {t[k]:.8f}" for k in ("qx", "qy", "qs") if k in t)
        if o.get("qs") and "qs" not in t:
            line += "  (Qs: fewer than 3 synchrotron periods in the run)"
        print(line)
    q6 = s.get("eigen_tunes_6d")
    ref = (f"6-D one-turn map {q6[0]:.8f} / {q6[1]:.8f} / {q6[2]:.8f}" if q6 else
           f"optics {o['qx']:.8f} / {o['qy']:.8f}")
    when = " at the start of the ramp" if s.get("ramp") else ""
    print(f"tunes of the {ref}{when}")
    print(f"results: {s['file']}")


def _print_injection(j) -> None:
    f = j["foil"]
    print(f"foil {f['name']}: theta0 {f['theta0_rad'] * 1e6:.4g} urad for one crossing (Highland; "
          "a particle's crossings add up to Highland at their total thickness), mean loss "
          f"{f['mean_loss_mev'] * 1e3:.4g} keV ({f['straggling_used']} straggling)"
          + ("" if f.get("stripped_fraction") is None else
             f", stripped {f['stripped_fraction'] * 100:.4f} %"))
    w = j.get("waste") or {}
    if j.get("settings"):
        b = j.get("buckets") or 1
        print(f"injected {j['injected']} macroparticles: {j['injected_particles']:.6g} particles "
              f"into the ring" + (f" ({b} filled buckets, each as the one tracked)" if b > 1
                                  else "")
              + f"; circulating {j['circulating']}; waste: missed the foil "
              f"{w.get('missed', 0)}, not stripped {w.get('unstripped', 0)}")
    mh = j.get("mean_hits_circulating")
    mev_j = 1.602176634e-13
    print(f"foil crossings {j['foil_hits']}"
          + ("" if mh is None or not math.isfinite(mh) else
             f"; {mh:.4g} per circulating particle")
          + f"; ionization energy left in the foil {j['deposit_mev'] * mev_j:.4g} J"
          + ("" if not j.get("electron_mev") else
             f"; the stripped electrons carry {j['electron_mev'] * mev_j:.4g} J (not tracked)"))


def _bins_text(z_bins, used, changes=None) -> str:
    """The line density's bins over the RF period: a fixed number, or "auto"'s
    bins with the turn from which each was used (``changes``: how many the
    record held before it kept only its first and last hundred)."""
    if z_bins != "auto":
        return f"{z_bins} bins over the RF period"
    if not used:
        return "bins following the bunch (auto)"
    last = {}
    for t, b in used:                    # (at a turn, the last bins set are the ones used)
        last[int(t)] = int(b)
    steps = [f"{b} from turn {t}" for t, b in sorted(last.items())]
    dropped = max(int(changes or 0) - len(used), 0)
    if len(steps) > 8 or dropped:
        steps = (steps[:3] + [f"... ({len(steps) - 6 + dropped} more changes) ..."]
                 + steps[-3:])
    return "bins following the bunch (auto) — " + ", ".join(steps)


def _print_space_charge(c) -> None:
    model = ("2.5-D particle-in-cell" if c["model"] == "pic" else
             f"frozen Gaussian (refreshed every {c['refresh_turns']} turns)"
             if c["refresh_turns"] else "frozen Gaussian (from the start)")
    if c["pipe"] == "open":
        pipe = f"open (potential zero at {c['reference_radius_m']:g} m)"
    elif c["pipe"] == "round":
        pipe = f"round pipe, radius {c['pipe_a_m']:g} m"
    else:
        pipe = f"{c['pipe']} pipe, half widths {c['pipe_a_m']:g} x {c['pipe_b_m']:g} m"
    lon = ("longitudinal kick: " + (f"g-factor g = {c['g_factor']:g}"
                                    if c["longitudinal"] == "g_factor" else c["longitudinal"])
           if c["longitudinal_kick"] else
           "no longitudinal kick" + ("" if c["bunched"] else " (a coasting beam)"))
    n_b = c.get("intensity_used")
    n_b = c.get("intensity") if n_b is None else n_b
    src = ("" if c.get("intensity") is not None else
           " (circulating at the end, from the injected current)"
           if c.get("intensity_from") == "injection" else
           " (the beam's current over its frequency)" if c.get("bunched") else
           " (the beam's current times the revolution time)")
    where_n = "the bunch" if c.get("bunched", True) else "the ring"
    print(f"space charge: {model}, {float(n_b or 0.0):.6g} particles in {where_n}{src}, "
          f"{c['kicks']} kicks per turn (largest spacing {c['largest_spacing_m']:.4g} m), grid "
          f"{c['grid'][0]} x {c['grid'][1]} ({c['deposit'].upper()}), {pipe}; {lon}")
    if c.get("bunched"):
        print("line density: " + _bins_text(c.get("z_bins"), c.get("z_bins_used"),
                                            c.get("z_bins_changes")))
    q = c.get("laslett_gaussian")
    if q:
        where = ("at the centre of a Gaussian bunch" if c.get("bunched")
                 else "of a coasting Gaussian beam")
        if c.get("intensity_from") == "injection":
            where += " (the beam at the end)"
        print(f"analytic tune shift {where}: dQx {q[0]:.4g}, dQy {q[1]:.4g}")
    if c.get("outside_grid_max"):
        print(f"at most {c['outside_grid_max']} particle(s) off the grid at a kick (they felt "
              "the grid charge's field: its multipole series far out, the sum over its cells "
              "nearer)")
    if c.get("kicks_charge_beyond_wall"):
        print(f"{c['kicks_charge_beyond_wall']} kick(s) with charge at or beyond the round "
              "pipe's wall (left out of its images)")
    if c.get("pipe_fit_residual") is not None:
        print(f"the pipe's fit leaves a potential of {c['pipe_fit_residual']:.2g} on its wall "
              "for a unit line charge")


def _project_has_beam(src: Path) -> bool:
    """Whether a project holds beam settings (the GUI saves them with every
    project; a hand-written project may have none)."""
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and isinstance(data.get("beam"), dict)


def _project_source(src: Path) -> str | None:
    """The start a project's Run page saved ("beam" or "empty"; an earlier
    version's "matched" is now the project's beam), None when it saved none."""
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ring = data.get("ring") if isinstance(data, dict) else None
    cfg = ring.get("config") if isinstance(ring, dict) else None
    if not isinstance(cfg, dict):
        return None
    # (an earlier version's settings read as the GUI reads them)
    from linac_gen.ring.config import migrate_source
    cfg = migrate_source(cfg)[0]
    src_ = cfg.get("source")
    return src_ if src_ in ("beam", "empty") else None


def _project_run_setting(src: Path, key: str):
    """One run setting a project's Run page saved (e.g. "tune_probes"), None
    when it saved none."""
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ring = data.get("ring") if isinstance(data, dict) else None
    cfg = ring.get("config") if isinstance(ring, dict) else None
    value = cfg.get(key) if isinstance(cfg, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _project_ring_settings(src: Path) -> dict:
    """The element model, pole faces, constants, ramp, corrector mode and space
    charge a project's ring section holds (saved by the GUI's Ring tab); empty when it
    holds none."""
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    ring = data.get("ring") if isinstance(data, dict) else None
    cfg = ring.get("config") if isinstance(ring, dict) else None
    if not isinstance(cfg, dict):
        return {}
    return {k: cfg[k] for k in ("model", "edge_model", "constants", "ramp", "corrector_mode",
                                "space_charge", "injection") if cfg.get(k) is not None}


def _print_summary(s, opt, rl) -> None:
    o = s["options"]
    print(f"ring: {s['input']} — {s['length_m']:.9g} m, {s['n_elements']} elements, "
          f"{s['n_nodes']} nodes (model {o['model']}, pole faces {o['edge_model']})")
    print(f"reference: proton {s['kinetic_energy_mev']:.9g} MeV  (beta0 {opt.beta0:.9f}, "
          f"gamma0 {opt.gamma0:.9f}; constants {s['constants']})")
    print(f"tunes:              Qx {opt.qx:.9f}   Qy {opt.qy:.9f}")
    if opt.dqx is not None:
        print(f"chromaticity:       Q'x {opt.dqx:.5f}   Q'y {opt.dqy:.5f}   (dQ/d delta)")
    gt = f"{opt.gamma_t:.6f}" if opt.gamma_t else "none (alpha_c <= 0)"
    print(f"momentum compaction {opt.alpha_c:.9g}   slip factor {opt.eta:.9g}   gamma_t {gt}")
    st = opt.start
    print(f"Twiss at s = 0:     beta_x {st['betx']:.6f} m  alpha_x {st['alfx']:.6f}  "
          f"beta_y {st['bety']:.6f} m  alpha_y {st['alfy']:.6f}")
    print(f"dispersion at s = 0: D_x {st['dx']:.6f} m  D'_x {st['dpx']:.6g}  "
          f"D_y {st['dy']:.6g} m  D'_y {st['dpy']:.6g}")
    co = opt.closed_orbit
    print(f"closed orbit at s = 0 ({opt.mode}): x {co[0] * 1e3:.6g} mm  px {co[1] * 1e3:.6g} mrad  "
          f"y {co[2] * 1e3:.6g} mm  py {co[3] * 1e3:.6g} mrad")
    if opt.qs is not None:
        b = s["bucket"]
        print(f"synchrotron tune:   Qs {opt.qs:.7f}  (small-amplitude formula {b['qs']:.7f})")
        print(f"RF bucket:          harmonic {b['harmonic']:g}, {b['voltage_mv'] * 1e3:.6g} kV, "
              f"lag {b['lag_deg']:.6g} deg, half-height delta {b['delta_max']:.6g}, "
              f"length {b['bucket_length_m']:.6g} m")
    for w in s["warnings"]:
        print(f"note: {w}")


_COLUMNS = ("name", "type", "s", "betx", "alfx", "bety", "alfy", "mux", "muy", "dx", "dpx", "dy",
            "dpy", "x", "px", "y", "py")


def _write_table(table: dict, path: Path) -> None:
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({k: (list(map(float, table[k])) if k not in ("name", "type")
                                        else list(table[k])) for k in _COLUMNS}, indent=0))
        return
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "type", "s_m", "beta_x_m", "alpha_x", "beta_y_m", "alpha_y",
                    "mu_x", "mu_y", "d_x_m", "dp_x", "d_y_m", "dp_y", "x_m", "px", "y_m",
                    "py"])
        n = len(table["name"])
        for i in range(n):
            row = [table["name"][i], table["type"][i]]
            row += [repr(float(table[k][i])) for k in _COLUMNS[2:]]
            w.writerow(row)

