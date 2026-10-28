"""Ring mode for the assistant: the multi-turn synchrotron simulation of
:mod:`linac_gen.ring` — the engine behind ``python -m linac_gen ring`` and the
GUI's Ring tab — on the session's lattice and beam, with the session's ring
settings (in the GUI: the Ring tab's).

``ring_status``       the lattice as a ring, ring mode, the ring settings, the last run
``ring_optics``       the ring's periodic optics (tunes, chromaticity, transition)
``ring_programs``     the ramp of the ring settings, previewed
``set_ring_program``  set (or clear) that ramp
``set_ring_mode``     ring mode on or off (the desktop GUI)
``run_ring``          a bunch turn after turn (long-running)

The ring package is imported inside the tools: a linac session never loads it.
"""
from __future__ import annotations

import copy
import dataclasses
import threading
import time
import warnings
from pathlib import Path

from linac_gen.assist.tools import (
    _capture,
    _ctx_provenance,
    _err,
    _need,
    _ok,
    _refused,
    _tool,
)

#: the settings that shape the compiled ring (its optics and its ramp's reference)
_OPTICS_KEYS = ("harmonic", "constants", "species", "design_energy_mev", "model",
                "edge_model", "drift_model", "bend_model", "integrator",
                "magnet_integrator", "bend_steps", "magnet_steps")


#: a ramp preview counts at most this many turns to the end of its program
#: (a few seconds: the tool answers at once)
PREVIEW_TURNS_CAP = 300_000
#: run_ring's macroparticles for the matched source when neither the call nor
#: the session gives them (the command line's default; the beam source has the
#: beam's)
DEFAULT_PARTICLES = 10_000
#: results files named by run_ring in this process (two runs finishing in the
#: same second get their own names)
_NAMES: set = set()
_NAMES_LOCK = threading.Lock()


def _settings(ctx) -> dict:
    """The session's ring settings, as RingConfig keys (others dropped; the
    element model's names of earlier versions mapped to today's)."""
    from linac_gen.ring.config import MODEL_ALIASES, RingConfig
    known = {f.name for f in dataclasses.fields(RingConfig)}
    out = {k: copy.deepcopy(v) for k, v in dict(ctx.ring_config or {}).items()
           if k in known}
    if isinstance(out.get("model"), str):
        out["model"] = MODEL_ALIASES.get(out["model"], out["model"])
    return out


def _results_file(ctx, res):
    """The file a ring run's results were written to (None when unknown)."""
    path = (getattr(res, "meta", None) or {}).get("dump_path") if res is not None else None
    return str(path) if path else (ctx.ring_results_path or None)


def _optics_settings(s: dict) -> dict:
    return {k: s[k] for k in _OPTICS_KEYS if s.get(k) is not None}


def _plain(v):
    from linac_gen.cli.ring import _finite_json
    return _finite_json(v)


_SESSION = object()


def _beam(ctx, cfg=_SESSION):
    """(the beam settings for the ring, a note or None): the session's — in
    the GUI the Beam tab's, which the Ring tab's Run takes (``cfg``: already
    read) —; else, for a lattice whose deck declares its beam (a MAD-X or MAD8
    BEAM with its energy), the default settings, as the command line takes for
    a bare lattice file (the lattice gives the species and energy)."""
    if cfg is _SESSION:
        cfg = ctx.ring_beam_config()
    if cfg is not None:
        return cfg, None
    from linac_gen.ring.compile import declared_beam
    if declared_beam(ctx.lattice) is None:
        raise ValueError("this lattice does not say which beam the ring runs at: "
                         "set the beam first (set_beam_config: species, energy)")
    from linac_gen.core.config import BeamConfig
    return BeamConfig(), ("no beam settings in the session: the default emittances (the "
                          "lattice's own beam gives the species and energy)")


def _provenance(ctx) -> dict:
    p = _ctx_provenance(ctx)
    f = _results_file(ctx, ctx.ring_results)
    if f:
        p["ring_results_path"] = f
    return p


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
def _status(ctx) -> dict:
    from linac_gen.ring.detect import detect_ring
    v = dataclasses.asdict(detect_ring(ctx.lattice))
    s = dict(ctx.ring_config or {})
    sc = s.get("space_charge") or {}
    try:
        beam_cfg = ctx.ring_beam_config()
    except RuntimeError:                 # (GUI: a Beam form that does not read)
        beam_cfg = ctx.beam_config
    # (the start a run would take, as _run chooses it, and its particles)
    start = s.get("source") or ("empty" if s.get("injection")
                                else "beam" if beam_cfg is not None else "matched")
    particles = (0 if start == "empty"
                 else int(s.get("n_particles") or DEFAULT_PARTICLES) if start == "matched"
                 else getattr(beam_cfg, "n_particles", None))
    out = {"verdict": v, "ring_mode": bool(ctx.ring_mode),
           "run_in_progress": bool(ctx.ring_running()),
           "settings": {"model": s.get("model", "cartesian"),
                        "constants": s.get("constants", "helix"),
                        "ramp": "set" if s.get("ramp") else "none (fixed energy)",
                        "space_charge": sc.get("model", "set") if sc else "off",
                        "injection": "set" if s.get("injection") else "none",
                        "turns": s.get("n_turns"), "particles": particles,
                        "source": start, "device": s.get("device")}}
    res = ctx.ring_results
    if res is not None:
        n0 = int(res.n_alive[0]) if len(res.n_alive) else 0
        out["last_run"] = {"turns_done": int(res.turns_done),
                           "turns_requested": int(res.n_turns_requested),
                           "particles_at_start": n0,
                           "circulating_at_end": (int(res.n_alive[-1]) if len(res.n_alive)
                                                  else 0),
                           "tunes_naff": dict(res.tunes or {}),
                           "file": _results_file(ctx, res)}
    return _plain(out)


@_tool("ring_status",
       "Ring mode's view of the session: whether the lattice closes on itself "
       "(its bends add up to 360 degrees and its survey returns — the verdict "
       "with its evidence), whether ring mode is on in the GUI and a ring run "
       "is in progress there, the session's ring settings (element model, "
       "ramp, space charge, injection, turns, particles) and the last ring "
       "run's headline and file.  Read-only.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _ring_status(ctx):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    out, refusal, warns = _capture(_status, ctx)
    if refusal is not None:
        return refusal
    return _ok(out, _provenance(ctx), warns)


def _optics(ctx, six_d: bool) -> dict:
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.optics import rf_bucket, twiss
    beam, note = _beam(ctx)
    if note:
        warnings.warn(note, stacklevel=2)
    cfg = RingConfig(**_optics_settings(_settings(ctx)))
    rl = compile_from_config(ctx.lattice_snapshot(), cfg, beam=beam)
    opt = twiss(rl, six_d=bool(six_d))
    out = opt.summary()
    out.update({"n_elements": len(rl.element_names), "n_nodes": rl.n_nodes,
                "kinetic_energy_mev": rl.reference.kinetic0, "constants": cfg.constants,
                "options": rl.options, "harmonic": rl.harmonic, "warnings": rl.warnings})
    if six_d:
        out["bucket"] = {k: v for k, v in rf_bucket(rl, opt).items() if k != "separatrix"}
    return _plain(out)


@_tool("ring_optics",
       "The periodic optics of the session's lattice as a ring, from ring mode's "
       "element maps (the session's ring settings: element model, pole faces, "
       "constants): tunes with their integer part, chromaticities, momentum "
       "compaction, slip factor and transition gamma, the Twiss functions and "
       "dispersion at the start.  six_d=true adds the 6-D closed orbit, the "
       "synchrotron tune and the RF bucket.  The same numbers as "
       "`linac_gen ring optics` and the Ring tab's Compute optics.",
       {"type": "object",
        "properties": {"six_d": {"type": "boolean", "default": False,
                                 "description": "also the 6-D optics (the RF on)"}},
        "required": []},
       "compute")
def _ring_optics(ctx, six_d: bool = False):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    out, refusal, warns = _capture(_optics, ctx, six_d)
    if refusal is not None:
        return refusal
    return _ok(out, _ctx_provenance(ctx), warns)


def _preview(ctx, s: dict) -> dict:
    from linac_gen.cli.ring import _ramp_summary
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.engine import TURNS_TO_END_CAP, RingSimulation
    beam, note = _beam(ctx)
    if note:
        warnings.warn(note, stacklevel=2)
    cfg = RingConfig(ramp=s["ramp"], corrector_mode=s.get("corrector_mode", "follow"),
                     **_optics_settings(s))
    from linac_gen.ring.programs import RampError
    ramp = RingSimulation(ctx.lattice_snapshot(), beam, cfg).ramp()
    try:
        n_end = ramp.turns_to_end(max_turns=min(TURNS_TO_END_CAP, PREVIEW_TURNS_CAP))
    except RampError:
        raise ValueError(f"the energy program takes more than {PREVIEW_TURNS_CAP} turns to "
                         "its end: preview it with `python -m linac_gen ring ramp`") from None
    # (a flat energy program has no end: the session's turns, as the GUI's
    # preview takes the Run page's)
    n = n_end if n_end is not None else int(s.get("n_turns") or 100)
    orbit = ramp.orbit(n)
    gt_from = ("given by the ramp" if s["ramp"].get("gamma_t") is not None
               else "from the ring's optics")
    out = _ramp_summary(Path(ctx.lattice_path or "lattice"), cfg, ramp, orbit, n, n_end,
                        gt_from)
    out.pop("program", None)
    out.pop("input", None)
    return _plain(out)


@_tool("ring_programs",
       "The ramp of the session's ring settings — the energy, RF voltage and "
       "phase programs, paraphasing and transition gamma — with its preview: "
       "the reference particle alone through it (the turns to the end of the "
       "energy program, the transition turn, the RF frequency range, the "
       "largest gain against the voltage, capped turns), as the Ring tab's "
       "Programs → Preview and `linac_gen ring ramp`.  'none' when the ring "
       "runs at fixed energy.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _ring_programs(ctx):
    s = _settings(ctx)
    if not s.get("ramp"):
        return _ok({"ramp": "none", "note": "the ring runs at its fixed energy (set_ring_program "
                                            "sets a ramp)"}, _ctx_provenance(ctx))
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    preview, refusal, warns = _capture(_preview, ctx, s)
    if refusal is not None:
        return refusal
    return _ok({"ramp": _plain(s["ramp"]), "preview": preview}, _ctx_provenance(ctx), warns)


# ---------------------------------------------------------------------------
# mutate
# ---------------------------------------------------------------------------
def _check_ramp(ramp: dict) -> dict:
    from linac_gen.ring.programs import RampConfig
    try:
        return RampConfig.from_dict(ramp).to_dict()
    except TypeError as exc:                    # (an unknown or missing key)
        raise ValueError(str(exc)) from None


@_tool("set_ring_program",
       "Set the ramp of the session's ring settings (in the GUI: the Ring "
       "tab's Programs page) — RampConfig's dict, as a ring-mode program file: "
       "energy {kind: flat | sine (a resonant cycle: start_kinetic_mev, "
       "end_kinetic_mev, frequency_hz, t_min_s, t_start_s) | table (times_s, "
       "values, quantity)}, voltage {kind: constant | table (times_s, "
       "values_mv)}, phase {kind: auto | table (times_s, values_deg), "
       "on_excess: cap | refuse}, paraphasing [{cavities, angle_deg}], gamma_t, "
       "t_start_s — or clear=true for fixed energy.  Validated before it is "
       "kept; a refusal names what is wrong.",
       {"type": "object",
        "properties": {"ramp": {"type": "object",
                                "description": "the ramp program (energy, voltage, phase, "
                                               "paraphasing, gamma_t, t_start_s)"},
                       "clear": {"type": "boolean", "default": False,
                                 "description": "no ramp: the ring at fixed energy"}},
        "required": []},
       "mutate")
def _set_ring_program(ctx, ramp=None, clear: bool = False):
    warns = []
    if clear:
        new = None
    else:
        if not isinstance(ramp, dict) or not ramp:
            return _refused("give the ramp as an object (energy, voltage, phase, ...) or "
                            "clear=true")
        new, refusal, warns = _capture(_check_ramp, ramp)
        if refusal is not None:
            return refusal
    cfg = dict(ctx.ring_config or {})
    if new is None:
        cfg.pop("ramp", None)
    else:
        cfg["ramp"] = new
    try:
        ctx.set_ring_config(cfg)
    except RuntimeError as exc:
        return _err(str(exc))
    return _ok({"ramp": _plain(new) if new is not None else "none (fixed energy)"},
               _ctx_provenance(ctx), warns)


@_tool("set_ring_mode",
       "Switch ring mode on or off in the desktop GUI (Simulate → Ring Mode: "
       "the Ring tab shown or hidden; switching it off stops the Ring tab's "
       "run in progress, not a run_ring job).  Desktop GUI only.  The other "
       "ring tools work on the session's lattice whether it is on or not.",
       {"type": "object",
        "properties": {"on": {"type": "boolean"}},
        "required": ["on"]},
       "mutate")
def _set_ring_mode(ctx, on: bool):
    try:
        now = ctx.set_ring_mode(bool(on))
    except RuntimeError as exc:
        return _err(str(exc))
    if now is None:
        return _refused("ring mode is a switch of the desktop GUI; outside it the ring tools "
                        "(ring_optics, run_ring, ...) work on the session's lattice as it is")
    if bool(now) != bool(on):
        return _err("ring mode could not be switched on (the GUI's status bar says why)")
    return _ok({"ring_mode": bool(now)})


# ---------------------------------------------------------------------------
# compute (long-running)
# ---------------------------------------------------------------------------
def _reserve_name(out_dir: Path, stem: str) -> Path:
    """<calc dir>/<stem>_<time>_ring.h5, a name no file and no other run of
    this process has (_2, _3, ...)."""
    base = f"{stem}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_ring"
    with _NAMES_LOCK:
        path, k = out_dir / f"{base}.h5", 1
        while str(path) in _NAMES or path.exists():
            k += 1
            path = out_dir / f"{base}_{k}.h5"
        _NAMES.add(str(path))
    return path


def _run(ctx, overrides: dict, to_end: bool, progress_callback, should_abort):
    from linac_gen.cli.ring import _run_summary
    from linac_gen.ring.config import RingConfig, auto_stats_every
    from linac_gen.ring.engine import RingSimulation
    from linac_gen.ring.memory import MEMORY_SHARE, physical_memory, run_need
    from linac_gen.ring.results import save_ring_hdf5
    lattice = ctx.lattice
    d = _settings(ctx)
    particles = overrides.pop("n_particles", None)
    d.update({k: v for k, v in overrides.items() if v is not None})
    beam_cfg = ctx.ring_beam_config()
    if d.get("source") is None:
        # (the session's beam, as the GUI's Ring tab runs it; without beam
        # settings, a beam matched to the ring, as the command line's for a
        # bare lattice file)
        d["source"] = ("empty" if d.get("injection")
                       else "beam" if beam_cfg is not None else "matched")
    beam, note = _beam(ctx, beam_cfg)
    if note:
        warnings.warn(note, stacklevel=2)
    if d["source"] == "beam":
        d.pop("n_particles", None)
        if particles is not None:
            if getattr(beam, "source", "generate") == "file":
                raise ValueError("particles: the session's beam is read from a distribution "
                                 "file, whose particles are all used")
            beam = dataclasses.replace(beam, n_particles=int(particles))
    elif particles is not None:
        d["n_particles"] = int(particles)
    else:
        d.setdefault("n_particles", DEFAULT_PARTICLES)
    if not d.get("ramp"):
        d["corrector_mode"] = "follow"          # (at fixed energy steerers keep their kicks)
    fixed_every = d.get("stats_every")
    d["stats_every"] = fixed_every or auto_stats_every(int(d["n_turns"]))
    cfg = RingConfig.from_dict(d)
    sim = RingSimulation(ctx.lattice_snapshot(lattice, should_abort), beam, cfg)
    if to_end:
        r = sim.ramp()
        n = r.turns_to_end()
        if n is None:
            raise ValueError("the ramp's energy program is flat, it has no end: give turns")
        if n < 1:
            r.orbit(0)                          # (refused, saying why)
        sim.config = cfg = dataclasses.replace(cfg, n_turns=n,
                                               stats_every=fixed_every or auto_stats_every(n))
    # (a run that cannot fit in memory is refused before it starts, as the
    # Ring tab refuses it)
    need = run_need(cfg, int(getattr(beam, "n_particles", 0) or 0), len(lattice.elements))
    have = physical_memory()
    if have and need > MEMORY_SHARE * have:
        raise ValueError(f"this run would need about {need / 2 ** 30:.1f} GB of the machine's "
                         f"{have / 2 ** 30:.0f} GB: fewer particles, turns or snapshot turns")

    def progress(turn, n_turns):
        if progress_callback is not None:
            progress_callback(float(turn), int(turn), int(n_turns))

    res = sim.run(progress_callback=progress, should_abort=should_abort)
    if res.turns_done == 0 and res.truncated:
        raise ValueError("the run was stopped before its first turn: nothing was kept (the "
                         "session keeps the results it had)")
    path = _reserve_name(Path(ctx.calc_dir or "."), Path(ctx.lattice_path).stem
                         if ctx.lattice_path else "ring")
    path.parent.mkdir(parents=True, exist_ok=True)
    path = save_ring_hdf5(res, path, lattice_path=str(ctx.lattice_path or "") or None,
                          beam_config=beam)
    res.meta["dump_path"] = str(path)
    summary = _plain(_run_summary(res, path))
    # (the session takes the results unless its lattice changed meanwhile or the
    # GUI's own ring run is going: their file is written either way)
    if ctx.lattice is not lattice:
        summary.setdefault("notes", []).append(
            f"the session's lattice changed during the run: its results are not the "
            f"session's; they are in {path}")
        return summary
    if ctx.ring_running():
        summary.setdefault("notes", []).append(
            f"a ring run of the Ring tab is in progress: its results stay the session's; "
            f"this run's are in {path}")
        return summary
    try:
        ctx.set_ring_results(res, str(path))
    except RuntimeError as exc:
        summary.setdefault("notes", []).append(f"the GUI did not take the results ({exc}); "
                                               f"they are in {path}")
    return summary


@_tool("run_ring",
       "Track a bunch turn after turn in ring mode on the session's lattice — "
       "at fixed energy or through the session's ramp, with its space-charge "
       "and injection settings — and write the results (HDF5) into the calc "
       "directory: turns done, survival and losses, betatron and longitudinal "
       "emittances, NAFF tunes, and the ramp's, space charge's and "
       "injection's summaries.  The run starts from the session's beam (the "
       "Beam tab's: distribution, particles, emittances, Twiss functions) "
       "unless the settings say otherwise.  Overrides of the session's "
       "settings: turns (default the session's, else 100), particles (default "
       "the beam's; 10000 for the matched source), to_end (the turns to the end "
       "of the ramp's energy program), source, device, snapshot_turns, seed.  "
       "Refused while the Ring tab's "
       "own run is going, or when it would not fit in memory.  In the desktop "
       "GUI the results go to the Ring tab and the Results tab (Run shown: "
       "Ring run); run_in_gui(kind='ring') instead "
       "presses the Ring tab's Run button.  Long-running background job.",
       {"type": "object",
        "properties": {
            "turns": {"type": "integer", "minimum": 1,
                      "description": "turns to run (default: the session's, else 100)"},
            "particles": {"type": "integer", "minimum": 1, "maximum": 100_000_000,
                          "description": "macroparticles at the start (default: the "
                                         "session's beam's; 10000 for the matched "
                                         "source)"},
            "to_end": {"type": "boolean", "default": False,
                       "description": "run to the end of the ramp's energy program "
                                      "(instead of turns)"},
            "source": {"type": "string", "enum": ["matched", "beam", "empty"],
                       "description": "beam: the session's beam (the Beam tab's "
                                      "distribution, the default); matched: a Gaussian "
                                      "matched to the ring with the beam's emittances; "
                                      "empty: the injection fills the ring (the default "
                                      "with injection)"},
            "device": {"type": "string", "enum": ["cpu", "cuda"]},
            "snapshot_turns": {"type": "array",
                               "items": {"type": ["integer", "string"]},
                               "description": "turns whose whole distribution is kept "
                                              "(0 the start, \"last\" the last turn)"},
            "seed": {"type": "integer"}},
        "required": []},
       "compute")
def _run_ring(ctx, turns=None, particles=None, to_end: bool = False, source=None,
              device=None, snapshot_turns=None, seed=None, progress_callback=None,
              should_abort=None):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    if ctx.ring_running():
        return _refused("a ring run of the Ring tab is in progress — wait for it or stop it "
                        "first")
    s = _settings(ctx)
    if to_end and turns is not None:
        return _refused("give turns or to_end, not both")
    if to_end and not s.get("ramp"):
        return _refused("to_end runs to the end of a ramp: the session has none "
                        "(set_ring_program)")
    try:
        n_turns = int(turns) if turns is not None else int(s.get("n_turns") or 100)
        overrides = {"n_turns": n_turns,
                     "n_particles": None if particles is None else int(particles),
                     "source": source, "device": device, "snapshot_turns": snapshot_turns,
                     "seed": None if seed is None else int(seed)}
    except (TypeError, ValueError) as exc:
        return _refused(f"turns, particles and seed are whole numbers ({exc})")
    if n_turns < 1:
        return _refused("turns must be at least 1")
    if overrides["n_particles"] is not None and overrides["n_particles"] < 1:
        return _refused("particles must be at least 1")
    summary, refusal, warns = _capture(_run, ctx, overrides, bool(to_end), progress_callback,
                                       should_abort)
    if refusal is not None:
        return refusal
    return _ok(summary, _provenance(ctx), warns)
