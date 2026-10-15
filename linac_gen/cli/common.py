"""Shared, GUI-free building blocks for the HELIX batch-mode CLI.

This module is the config-resolution layer behind ``run`` / ``scan`` /
``batch``: load a lattice or ``.lgproj`` project, apply scalar and
element-parameter overrides, build the solver configs, run an
envelope / multi-particle / matrix simulation, and write results.

It mirrors the GUI's run wiring (``app.py:_run_mp`` / ``_run_envelope``)
without importing any GUI code.  All heavy / optional imports are
deferred to call time so importing this module stays cheap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

__all__ = [
    "load_lattice", "load_input", "coerce_like", "parse_assignments",
    "apply_beam_overrides", "apply_element_override",
    "build_ref", "run_envelope_sim", "run_mp_sim", "run_matrix",
    "make_step_config", "make_sc_config", "apply_fieldmap_settings",
    "write_results", "write_final_dst", "result_summary",
    "build_scan_point",
]


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _first(*vals):
    """First argument that is not None (else None)."""
    for v in vals:
        if v is not None:
            return v
    return None


def _as_bool(value) -> bool:
    """Coerce an on/off setting from the CLI, a project file or a study
    spec: bool, int, or one of on/off, true/false, yes/no, 1/0."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("on", "true", "yes", "1"):
        return True
    if text in ("off", "false", "no", "0"):
        return False
    raise ValueError(f"expected on/off (or true/false), got {value!r}")


def coerce_like(old, raw: str):
    """Coerce a string ``raw`` to the type of an existing value ``old``."""
    if isinstance(old, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(old, int):
        return int(float(raw))
    if isinstance(old, float):
        return float(raw)
    return raw


def parse_assignments(items) -> dict:
    """Parse a list of ``name=value`` strings into a dict (values as str)."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected name=value, got '{item}'")
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------
def load_lattice(path, *, fallback_beam=None):
    """Parse a lattice file → ``Lattice``: TraceWin ``.dat``, MAD-X
    ``.madx``/``.seq``, MAD8 ``.lat``/``.flat``, Elegant ``.lte`` natively;
    Bmad ``.bmad``, SciBmad ``.jl``/``.scibmad``, PALS ``.pals.yaml``/
    ``.pals.json`` and ``.lattix.json`` through the optional ``lattix``
    translator (see :mod:`linac_gen.io.formats`).
    Parse warnings (downgraded cards, dropped elements,
    approximations, an unrecognised suffix parsed as TraceWin) are echoed
    to stderr — they used to be silently discarded on every CLI path.
    ``fallback_beam`` supplies a MAD8 file's rigidity when the file declares
    none (see :func:`linac_gen.io.formats.parse_lattice_file`)."""
    import sys
    from linac_gen.io.formats import parse_lattice_file
    lat, meta = parse_lattice_file(str(path), fallback_beam=fallback_beam)
    warns = meta.get("warnings", []) if isinstance(meta, dict) else []
    for w in warns:
        print(f"parse warning: {w}", file=sys.stderr)
    # Carry the downgrade ledger with the lattice so the HDF5
    # provenance/ group can record WHICH physics downgrades the parse
    # accepted (previously echoed to stderr and lost).
    lat.parse_warnings = list(warns)
    return lat


def load_input(path, *, tracewin_ini=None):
    """Load a CLI input — a ``.lgproj`` project *or* a bare lattice file.

    Returns ``(lattice, beam_config, convergence_dict)``.  For a bare
    lattice the beam is a default ``BeamConfig`` — or, with
    ``tracewin_ini`` (``"auto"``/``True`` for the sibling ``<deck>.ini``,
    or an explicit path), beam 1 of the deck's TraceWin options file —
    and convergence is empty.  A project refuses ``tracewin_ini``: its
    saved beam always wins.
    """
    p = Path(path)
    if p.suffix.lower() == ".lgproj":
        if tracewin_ini:
            raise ValueError(_TW_INI_PROJECT_MSG.format(name=p.name))
        from linac_gen.io.project import load_project
        proj = load_project(p)
        return (load_lattice(proj.lattice_path, fallback_beam=proj.beam),
                proj.beam, proj.convergence)
    from linac_gen.core.config import BeamConfig
    lat = load_lattice(str(p))
    if tracewin_ini:
        return lat, beam_from_tracewin_ini(p, tracewin_ini, lattice=lat), {}
    return lat, BeamConfig(), {}


# ---------------------------------------------------------------------------
# TraceWin .ini beam for a bare lattice (shared by every input-taking command)
# ---------------------------------------------------------------------------
_TW_INI_PROJECT_MSG = (
    "--tracewin-ini cannot be combined with a project file ({name}): the "
    "project's saved beam always wins.  Convert the .ini with "
    "`python -m linac_gen twini` and save a new project instead.")
# Paths whose "beam from X.ini" note has been printed in this process — a
# scan builds one point per swept value from the same deck and must not
# repeat the note (and its warnings) per point.
_TW_INI_ANNOUNCED: set = set()


def add_tracewin_ini_argument(p) -> None:
    """The ``--tracewin-ini [INI]`` option shared by the input-taking
    subcommands (``nargs="?"``: bare flag → the sibling ``<deck>.ini``)."""
    p.add_argument("--tracewin-ini", nargs="?", const="auto", default=None,
                   dest="tracewin_ini", metavar="INI",
                   help="bare lattice file only: take the beam from the "
                        "deck's TraceWin options file (<deck>.ini next to "
                        "it, or INI); scalar overrides such as --energy "
                        "still apply on top.  A .lgproj input keeps its "
                        "saved beam and refuses this flag")


_TW_INI_OVERRIDE_KEYS = ("energy", "frequency", "freq", "species")


def note_tracewin_ini_overrides(tracewin_ini, overrides) -> None:
    """Warn (once per process) when a ``--tracewin-ini`` beam is combined
    with an energy / frequency / species override: ``emit_z`` and
    ``beta_z`` were converted at the .ini values and are NOT re-derived
    for the overridden ones (beta_z scales as 1/(beta*gamma)^3 and both
    scale with the frequency)."""
    import sys
    if not tracewin_ini or not overrides:
        return
    hit = [k for k in _TW_INI_OVERRIDE_KEYS
           if k in overrides and overrides[k] is not None]
    if hit and "overrides" not in _TW_INI_ANNOUNCED:
        _TW_INI_ANNOUNCED.add("overrides")
        print(f"warning: {', '.join(hit)} overridden on top of the .ini beam: "
              f"emit_z and beta_z were converted at the .ini energy, frequency "
              f"and species and are not re-derived (beta_z ~ 1/(beta*gamma)^3, "
              f"emit_z and beta_z ~ f)", file=sys.stderr)


def beam_from_tracewin_ini(lattice_path, spec, *, lattice=None, base=None):
    """Beam 1 of the TraceWin ``.ini`` selected by ``spec`` for a bare
    lattice (see :func:`linac_gen.io.tracewin_ini.resolve_tracewin_ini`),
    announced once per file on stderr together with its warnings."""
    import sys
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.tracewin_ini import (load_tracewin_ini,
                                           resolve_tracewin_ini,
                                           to_beam_config)
    ini_path = resolve_tracewin_ini(lattice_path, spec)
    if ini_path is None:
        return base if base is not None else BeamConfig()
    if ini_path.suffix.lower() != ".ini":
        raise ValueError(
            f"--tracewin-ini expects a TraceWin .ini options file, got "
            f"{ini_path.name} (put the flag after the input, or pass the "
            f".ini path explicitly)")
    ini = load_tracewin_ini(ini_path)
    cfg, warns = to_beam_config(ini, lattice=lattice, base=base)
    key = str(ini_path.resolve())
    if key not in _TW_INI_ANNOUNCED:
        _TW_INI_ANNOUNCED.add(key)
        print(f"beam from {ini_path.name}: {cfg.species}, {cfg.energy:g} MeV, "
              f"{cfg.frequency:g} MHz, {cfg.current:g} mA, {cfg.n_particles} "
              f"particles (replaces the BeamConfig default)", file=sys.stderr)
        for w in list(ini.warnings) + list(warns):
            print(f"warning: {w}", file=sys.stderr)
    return cfg


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------
def apply_beam_overrides(cfg, overrides: dict) -> None:
    """Apply ``name=value`` beam overrides onto a ``BeamConfig`` in place."""
    for key, raw in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise ValueError(f"unknown beam parameter '{key}'")
        setattr(cfg, key, coerce_like(getattr(cfg, key), str(raw)))


def apply_element_override(lattice, selector: str, value) -> None:
    """Set one element parameter from a ``--set`` selector.

    Selector forms:
      * ``NAME.attr`` — the element whose ``.name`` equals ``NAME``;
      * ``@N.attr``   — the N-th element (1-based, over ``lattice.elements``).

    The new value is coerced to the type of the CURRENT value.  A
    None-valued attribute (an unarmed optional slot, e.g. the multibunch
    ``sync_phase_pin``) carries no type to coerce to: it is parsed as a
    float when the string is numeric, else kept as a string — the M6
    hybrid replay transports float pins through exactly this path.
    """
    if "." not in selector:
        raise ValueError(
            f"--set selector '{selector}' must be NAME.attr or @N.attr")
    target, attr = selector.rsplit(".", 1)
    elements = list(lattice.elements)
    if target.startswith("@"):
        try:
            idx = int(target[1:])
        except ValueError:
            raise ValueError(f"--set: bad element index in '{selector}'")
        if not (1 <= idx <= len(elements)):
            raise ValueError(
                f"--set: element index {idx} out of range (1..{len(elements)})")
        elem = elements[idx - 1]
    else:
        matches = [e for e in elements
                   if getattr(e, "name", None) == target]
        if not matches:
            raise ValueError(f"--set: no element named '{target}'")
        if len(matches) > 1:
            raise ValueError(
                f"--set: element name '{target}' is ambiguous "
                f"({len(matches)} matches) — use @index")
        elem = matches[0]
    if not hasattr(elem, attr):
        raise ValueError(
            f"--set: element '{target}' has no attribute '{attr}'")
    current = getattr(elem, attr)
    if current is None:
        try:
            new = float(str(value))
        except ValueError:
            new = str(value)
        setattr(elem, attr, new)
        return
    setattr(elem, attr, coerce_like(current, str(value)))


# ---------------------------------------------------------------------------
# config builders
# ---------------------------------------------------------------------------
def rfq_settings(conv: dict, cli: dict) -> dict:
    """The RFQ solver fields of ``StepConfig`` — CLI override > project
    value (``convergence`` keys of the same names) > default.  Checked and
    normalised by ``StepConfig``: a bad value raises ``ValueError`` here,
    before anything runs."""
    from linac_gen.core.step_config import StepConfig
    d = StepConfig()
    conv = conv or {}
    s = StepConfig(
        rfq_solver=str(_first(cli.get("rfq_solver"), conv.get("rfq_solver"),
                              d.rfq_solver)),
        rfq_td_steps_per_period=int(_first(
            cli.get("rfq_td_steps"), conv.get("rfq_td_steps_per_period"),
            d.rfq_td_steps_per_period)),
        rfq_td_sc_every=int(_first(cli.get("rfq_td_sc_every"),
                                   conv.get("rfq_td_sc_every"),
                                   d.rfq_td_sc_every)))
    return dict(rfq_solver=s.rfq_solver,
                rfq_td_steps_per_period=s.rfq_td_steps_per_period,
                rfq_td_sc_every=s.rfq_td_sc_every)


def make_step_config(conv: dict, cli: dict):
    """Build a ``StepConfig`` — CLI override > project value > default."""
    from linac_gen.core.step_config import StepConfig
    d = StepConfig()
    step1 = _first(cli.get("step1"), (conv or {}).get("step1_per_m"),
                   d.integration_steps_per_metre)
    step2 = _first(cli.get("step2"), (conv or {}).get("step2_per_m"),
                   d.sc_steps_per_metre)
    single = _first(cli.get("drift_single_push"),
                    (conv or {}).get("drift_single_push"),
                    d.drift_single_push)
    return StepConfig(integration_steps_per_metre=float(step1),
                      sc_steps_per_metre=float(step2),
                      drift_single_push=_as_bool(single),
                      **rfq_settings(conv, cli))


def check_sc_choice(cli: dict, conv: dict | None = None) -> None:
    """Raise ``ValueError`` for an unknown method name on the command line
    (``--sc-method`` or ``--sc method=``), two different names there, or an
    unknown project ``sc_method``, so the run fails before any tracking."""
    from linac_gen.pic import methods
    methods.cli_method(cli or {})
    methods.check_name((conv or {}).get("sc_method"))


def print_sc_methods() -> None:
    """Print the named space-charge methods (``--sc-method list``)."""
    from linac_gen.pic import methods
    print(methods.table())


class ScMethodAction(argparse.Action):
    """``--sc-method NAME``; ``--sc-method list`` prints the methods and
    exits at once, so no input file or other required option is needed."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values == "list":
            print_sc_methods()
            parser.exit(0)
        setattr(namespace, self.dest, values)


def make_sc_config(cfg, conv: dict, cli: dict):
    """Build a ``SpaceChargeConfig`` (or ``None`` when ``cfg.current<=0``).

    Resolved by :func:`linac_gen.pic.methods.resolve_sc`: the method named
    by ``cli["sc_method"]`` (or ``cli["sc"]["method"]``) or the project's
    ``sc_method``; a project without ``sc_method`` resolves exactly as it
    always did (CLI override > project ``convergence`` value > classic
    value).  ``cli`` may carry ``nx`` / ``grid_extent`` / ``kernel`` /
    ``backend`` and a nested ``sc`` dict of generic ``name=value``
    overrides.
    """
    if cfg.current <= 0:
        return None
    import sys

    from linac_gen.pic import methods
    notes: list = []
    sc = methods.resolve_sc(conv or {}, cli or {}, notes=notes)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return sc


def apply_fieldmap_settings(conv: dict, cli: dict) -> None:
    """Apply 3-D field-map integrator / interpolation kinds (class attrs).

    Only sets them when explicitly given (CLI or project) — otherwise the
    FieldMap3D class defaults stand."""
    integ = _first((cli or {}).get("integrator"),
                   (conv or {}).get("integrator_kind"))
    interp = _first((cli or {}).get("interp"),
                    (conv or {}).get("interp_kind"))
    sampling = _first((cli or {}).get("fieldmap_sampling"),
                      (conv or {}).get("fieldmap_sampling"),
                      (conv or {}).get("fieldmap_kernel"))
    if integ is None and interp is None and sampling is None:
        return
    from linac_gen.elements import field_map_3d
    # helper mirrors into env vars so spawned scan/batch workers inherit
    field_map_3d.set_fieldmap_numerics(integrator=integ, interp=interp)
    if sampling is not None:
        # bool (project "fieldmap_kernel") or "kernel"/"scipy" (CLI flag).
        # Both sampling paths are bitwise identical; "scipy" exists for
        # verification/debugging.
        if isinstance(sampling, bool):
            enabled = sampling
        else:
            val = str(sampling).strip().lower()
            _false = ("0", "false", "off", "scipy", "legacy")
            _true = ("1", "true", "on", "kernel")
            if val not in _false + _true:
                import warnings
                warnings.warn(
                    f"unrecognized fieldmap_sampling value {sampling!r} — "
                    f"defaulting to the kernel (results are identical "
                    f"either way)", stacklevel=2)
            enabled = val not in _false
        field_map_3d.use_fused_kernel(enabled)
        # Propagate to spawned scan/batch workers (fresh re-imports).
        import os
        os.environ["LINAC_GEN_FIELDMAP_KERNEL"] = "1" if enabled else "0"


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def build_ref(cfg):
    """Build a ``ReferenceParticle`` from a ``BeamConfig``."""
    from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    sp = {"proton": PROTON, "deuteron": DEUTERON,
          "H-": H_MINUS}.get(cfg.species, PROTON)
    return ReferenceParticle(species=sp, w_kin=cfg.energy,
                             frequency=cfg.frequency)


def _envelope_initial(cfg, ref) -> dict:
    """The envelope-solver ``initial`` dict — mirrors app.py:_run_envelope
    (normalised emittance → geometric via βγ, mismatch applied)."""
    from linac_gen.distributions.factory import geometric_emittances
    emit_x, emit_y, emit_z = geometric_emittances(cfg, ref.bg)
    return dict(
        alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=emit_x,
        alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=emit_y,
        alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=emit_z,
        continuous=bool(getattr(cfg, "continuous", False)),
        dc_energy_spread_keV=float(getattr(cfg, "dc_energy_spread_keV", 0.0)),
        # First-moment seed — mirrors create_beam's centroid offsets so
        # envelope and MP runs of the same config launch the same orbit.
        centroid=[float(getattr(cfg, "centroid_x", 0.0) or 0.0),
                  float(getattr(cfg, "centroid_xp", 0.0) or 0.0),
                  float(getattr(cfg, "centroid_y", 0.0) or 0.0),
                  float(getattr(cfg, "centroid_yp", 0.0) or 0.0),
                  float(getattr(cfg, "centroid_dphi", 0.0) or 0.0),
                  float(getattr(cfg, "centroid_dw", 0.0) or 0.0)],
    )


def run_envelope_sim(lattice, cfg, env_solver: str = "matrix"):
    """Run an envelope simulation; returns an ``EnvelopeResults``."""
    ref = build_ref(cfg)
    initial = _envelope_initial(cfg, ref)
    if env_solver == "sacherer":
        from linac_gen.tracking.sacherer import SachererSolver
        return SachererSolver(lattice, ref, initial,
                              current=cfg.current).run()
    from linac_gen.tracking.envelope import EnvelopeSolver
    f_bunch = float(getattr(cfg, "bunch_frequency_MHz", 0.0) or 0.0) or None
    return EnvelopeSolver(lattice, ref, initial, current=cfg.current,
                          bunch_frequency=f_bunch).run()


def run_mp_sim(lattice, cfg, sc_config, step_config, *, seed: int = 42,
               progress_callback=None, record_action_scan: bool = False,
               pic_setup_hook=None):
    """Run a multi-particle simulation.

    ``record_action_scan`` opts in to the per-step halo action scan
    (``Simulation(record_action_scan=True)``).  ``pic_setup_hook(solver)``
    is called with the space-charge solver once it is built (the
    ``Simulation`` hook), e.g. to instrument its kicks.

    Returns ``(recorder, beam)`` — after the run ``beam.particles`` holds
    the final distribution (used for ``.dst`` export)."""
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    lattice.step_config = step_config
    beam = create_beam(cfg, seed=seed)
    sim = Simulation(lattice, beam, space_charge=sc_config,
                     progress_callback=progress_callback,
                     record_action_scan=record_action_scan,
                     pic_setup_hook=pic_setup_hook)
    recorder = sim.run()
    return recorder, beam


def run_backtrack_sim(lattice, cfg, sc_config, step_config, *,
                      seed: int = 42, dst_path=None, start: int = 0,
                      end=None, allow_dc_crossing: bool = False,
                      approximate_backtracking: bool = False,
                      energy_tolerance: float = 1e-3,
                      energy_hard_limit: float = 5e-2,
                      field_map_mode: str = "rk4",
                      progress_callback=None):
    """Backtrack a distribution from the exit of ``elements[end]`` to the
    entrance of ``elements[start]``.

    ``dst_path`` given → the measured/exported exit distribution is
    loaded from it (the file is authoritative for energy + frequency,
    matching forward ``.dst`` semantics).  ``dst_path`` omitted → design
    mode: the project's Twiss/emittance describe the DESIRED EXIT beam,
    generated at the design exit energy/frequency from the forward
    replay of the entrance reference.

    Returns ``(recorder, beam)`` — after the walk ``beam.particles``
    holds the reconstructed upstream distribution (usable for ``.dst``
    export) and ``recorder`` is the reversed-to-increasing-s record with
    ``direction == "backward"``.
    """
    import copy

    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.tracking.backtrack import build_replay_table

    lattice.step_config = step_config
    entrance_ref = build_ref(cfg)
    if end is None:
        end = len(lattice.elements) - 1
    if dst_path is not None:
        cfg_file = copy.copy(cfg)
        cfg_file.source = "file"
        cfg_file.distribution_file = str(dst_path)
        beam = create_beam(cfg_file, seed=seed)
    else:
        table = build_replay_table(lattice, entrance_ref, end=end)
        cfg_exit = copy.copy(cfg)
        cfg_exit.energy = table[-1].w_kin
        cfg_exit.frequency = table[-1].frequency
        beam = create_beam(cfg_exit, seed=seed)
    sim = Simulation(lattice, beam, space_charge=sc_config,
                     progress_callback=progress_callback)
    recorder = sim.run_backtrack(
        entrance_ref=entrance_ref, start=start, end=end,
        allow_dc_crossing=allow_dc_crossing,
        approximate_backtracking=approximate_backtracking,
        energy_tolerance=energy_tolerance,
        energy_hard_limit=energy_hard_limit,
        field_map_mode=field_map_mode)
    return recorder, beam


def run_matrix(lattice, cfg):
    """Linear matrix tracking; returns ``(M, twiss)`` where ``twiss`` maps
    plane → Twiss dict (or ``{'error': …}`` for a coupled/unstable plane)."""
    from linac_gen.tracking.matrix_tracking import (
        compute_transfer_matrix, compute_twiss,
    )
    ref = build_ref(cfg)
    M = compute_transfer_matrix(lattice, ref)
    twiss = {}
    for plane in ("x", "y"):
        try:
            twiss[plane] = compute_twiss(M, plane)
        except ValueError as exc:
            twiss[plane] = {"error": str(exc)}
    return M, twiss


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
def write_results(results, out_path, fmt: str, beam_cfg, lattice, *,
                  lattice_path=None, seed=None, sc_config=None) -> str:
    """Write one run's results in ``fmt`` (hdf5 / openpmd / partran).

    The optional keyword args feed the HDF5 ``provenance/`` group
    (lattice hash, beam seed, effective SC backend) — pass whatever the
    call site has in scope."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Imported-beam provenance: when the run consumed a .dst (or other
    # distribution file), its hash belongs in the results file — the
    # lattice hash alone cannot reproduce a file-sourced beam.
    beam_file = (getattr(beam_cfg, "distribution_file", None)
                 if getattr(beam_cfg, "source", "generate") == "file"
                 else None)
    if fmt == "hdf5":
        from linac_gen.io.hdf5_output import save_results_hdf5
        save_results_hdf5(results, str(out), beam_config=beam_cfg,
                          lattice=lattice, lattice_path=lattice_path,
                          seed=seed, sc_config=sc_config,
                          input_beam_path=beam_file)
    elif fmt == "openpmd":
        from linac_gen.io.openpmd_output import save_results_openpmd
        save_results_openpmd(results, str(out), beam_config=beam_cfg,
                             lattice=lattice)
    elif fmt == "partran":
        from linac_gen.io.tracewin_outputs import write_partran_out
        write_partran_out(results, lattice, beam_cfg, str(out))
    else:
        raise ValueError(f"unknown output format '{fmt}' "
                         f"(use hdf5 / openpmd / partran)")
    return str(out)


def write_final_dst(beam, out_path) -> str:
    """Write the final particle distribution to a TraceWin ``.dst``.

    ALIVE particles only: lost particles sit frozen at their
    aperture-strike coordinates in ``beam.particles``, and writing them
    would (a) corrupt any moment/shape analysis of the file and (b)
    resurrect them as live beam on re-import (2026-07-26 finding — a
    treaty-point capture carried 2.7 % dead strikes whose frozen x'
    values inflated the geometric emittance 7.5x)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    from linac_gen.io.tracewin_dst import (write_dst,
                                           warn_if_nonstandard_dst_header)
    parts = getattr(beam, "alive_particles", None)
    if parts is None:
        parts = beam.particles
    warn_if_nonstandard_dst_header(beam)
    write_dst(str(out), parts,
              current_mA=float(beam.current),
              # KNOWN MISMATCH, deliberately left as-is — see the
              # DEFERRED FIX block in io/tracewin_dst.py.  write_dst
              # documents this parameter as the BUNCH frequency and
              # TraceWin writes the bunch rate here, but changing it
              # alone would break the clock: write_dst does no frequency
              # conversion, so the phase column must be rescaled in the
              # same change.  Blocked on one TraceWin round-trip.
              frequency_MHz=float(beam.ref.frequency),
              mass_MeV=float(beam.ref.species.mass),
              w_kin_ref=float(beam.ref.w_kin))
    return str(out)


def result_summary(results) -> dict:
    """End-of-lattice scalar metrics for the run report and exit-code logic."""
    def last(name):
        arr = getattr(results, name, None)
        try:
            return float(arr[-1]) if arr is not None and len(arr) else None
        except (TypeError, IndexError, ValueError):
            return None
    return {
        "sigma_x": last("sigma_x"), "sigma_y": last("sigma_y"),
        "sigma_phi": last("sigma_phi"), "sigma_w": last("sigma_w"),
        "emit_x": last("emit_x"), "emit_y": last("emit_y"),
        "emit_z": last("emit_z"),
        "transmission": last("transmission"),
        "ref_w_kin": last("ref_w_kin"),
    }


# ---------------------------------------------------------------------------
# scan-point construction (shared by the `scan` and `batch` subcommands)
# ---------------------------------------------------------------------------
def build_scan_point(input_path, *, beam_overrides=None, element_overrides=(),
                     sc_overrides=None, mode="mp", env_solver="matrix",
                     seed: int = 42, cli: dict | None = None,
                     tracewin_ini=None, notes: list | None = None):
    """Resolve a CLI input + overrides into a :class:`ScanPoint`.

    Used by both ``scan`` (one point per swept value) and ``batch`` (one
    point per job).  ``cli`` may carry fixed ``nx`` / ``grid_extent`` /
    ``step1`` / ``step2`` / ``drift_single_push`` / ``backend`` /
    ``sc_method`` overrides; ``beam_overrides`` and
    ``sc_overrides`` are ``name=value`` dicts; ``element_overrides`` is an
    iterable of ``(selector, value)`` pairs.  ``tracewin_ini`` has the
    :func:`load_input` meaning (bare lattice only; the beam of the deck's
    TraceWin options file before ``beam_overrides`` are applied).  Notes of
    the space-charge resolution (a method replacing the project's settings)
    are appended to ``notes``, or issued as warnings when ``notes`` is None.
    """
    from dataclasses import asdict
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.step_config import StepConfig
    from linac_gen.parallel.scan_pool import ScanPoint

    cli = cli or {}
    p = Path(input_path)
    if p.suffix.lower() == ".lgproj":
        if tracewin_ini:
            raise ValueError(_TW_INI_PROJECT_MSG.format(name=p.name))
        from linac_gen.io.project import load_project
        proj = load_project(p)
        lattice_path, beam_cfg, conv = (
            proj.lattice_path, proj.beam, proj.convergence)
    else:
        lattice_path, conv = str(p.resolve()), {}
        beam_cfg = (beam_from_tracewin_ini(p, tracewin_ini) if tracewin_ini
                    else BeamConfig())

    # The beam a rigidity-less MAD8 deck is converted with: the project's,
    # captured BEFORE the --beam overrides (a scan over energy must not
    # rescale the magnets); none for a bare lattice (load_input refuses too).
    lattice_beam = asdict(beam_cfg) if p.suffix.lower() == ".lgproj" else {}
    apply_beam_overrides(beam_cfg, beam_overrides or {})
    note_tracewin_ini_overrides(tracewin_ini, beam_overrides)

    sd = StepConfig()
    step1 = float(_first(cli.get("step1"), conv.get("step1_per_m"),
                         sd.integration_steps_per_metre))
    step2 = float(_first(cli.get("step2"), conv.get("step2_per_m"),
                         sd.sc_steps_per_metre))
    single = _as_bool(_first(cli.get("drift_single_push"),
                             conv.get("drift_single_push"),
                             sd.drift_single_push))

    # The full space-charge configuration, resolved once here and carried
    # by the point: the worker rebuilds exactly this config, and builds and
    # checks it only when the point runs with space charge (mp, current >
    # 0), as before.  As before, only the grid size, extent and backend
    # (and now a method name) are read from ``cli``; a kernel choice
    # arrives through ``sc_overrides``.
    from linac_gen.pic import methods
    sc_cli = {k: cli.get(k)
              for k in ("nx", "grid_extent", "backend", "sc_method")}
    sc_cli["sc"] = dict(sc_overrides or {})
    sc_notes: list = [] if notes is None else notes
    sc_kw = methods.filled_kwargs(
        methods.resolve_sc_kwargs(conv, sc_cli, notes=sc_notes))
    if notes is None:
        import warnings
        for note in sc_notes:
            warnings.warn(f"space charge: {note}", stacklevel=2)

    return ScanPoint(
        lattice_path=lattice_path,
        beam_config=asdict(beam_cfg),
        nx=int(sc_kw["nx"]), grid_extent=float(sc_kw["grid_extent"]),
        step1=step1, step2=step2,
        seed=int(seed), use_gpu=str(sc_kw.get("use_gpu", "auto")),
        mode=mode, env_solver=env_solver,
        element_overrides=tuple(element_overrides),
        sc_overrides=tuple((k, v) for k, v in sc_kw.items()
                           if k not in ("nx", "grid_extent", "use_gpu")),
        drift_single_push=single,
        lattice_beam=lattice_beam,
        **rfq_settings(conv, cli),
    )


def _threads_value(text: str) -> str:
    from linac_gen.parallel.threads import parse_threads
    try:
        parse_threads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return text


def add_threads_argument(p: argparse.ArgumentParser) -> None:
    """``--threads N|all`` (see :mod:`linac_gen.parallel.threads`)."""
    p.add_argument(
        "--threads", metavar="N|all", type=_threads_value, default=None,
        help="CPU threads for the space-charge FFT, field maps and charge "
             "deposition (default: $HELIX_THREADS, else $OMP_NUM_THREADS, "
             "else every CPU).  Results are identical for any number.")


def threads_parent() -> argparse.ArgumentParser:
    """Parent parser that gives a subcommand ``--threads N|all``."""
    p = argparse.ArgumentParser(add_help=False)
    add_threads_argument(p)
    return p


def apply_threads_arg(args) -> None:
    """Apply ``--threads`` when it was given."""
    value = getattr(args, "threads", None)
    if value is not None:
        from linac_gen.parallel.threads import set_threads
        set_threads(value)


def make_console_encoding_safe() -> None:
    """Degrade non-encodable console characters instead of crashing.

    CLI reports contain arrows/box glyphs outside cp1252; on a stock
    Windows console (locale cp1252) printing them raised
    UnicodeEncodeError (external Windows report, issue 3).  On utf-8
    platforms the error handler is never consulted, so output is
    unchanged there.
    """
    import sys as _sys
    for stream in (_sys.stdout, _sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(errors="replace")
            except Exception:                                # noqa: BLE001
                pass


def cpp_kernels_built() -> bool:
    """Cheap probe: was the C++ PIC extension built for this install?

    Uses find_spec so neither torch nor the kernels are imported — safe
    at CLI/GUI startup.  A no-compiler install degrades to the pure-
    Python PIC path silently (~20x slower; external Windows report,
    issue 18): callers surface this ONCE so users learn why runs crawl.
    """
    import importlib.util
    try:
        return importlib.util.find_spec("linac_gen._pic_kernels") is not None
    except (ImportError, ValueError):                        # broken pkg
        return False


def warn_if_no_cpp_kernels() -> None:
    """One stderr line when the C++ PIC kernels are missing (CLI)."""
    if not cpp_kernels_built():
        import sys as _sys
        print("[linac_gen] C++ PIC kernels not built — using the pure-"
              "Python space-charge path (~20x slower). Rebuild with "
              "`pip install -e .` (needs a C++ compiler) to regain speed.",
              file=_sys.stderr)
