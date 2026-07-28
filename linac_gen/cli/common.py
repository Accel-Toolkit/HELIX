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
def load_lattice(path):
    """Parse a ``.dat`` / ``.madx`` / ``.seq`` / ``.lat`` lattice file →
    ``Lattice``.  Parse warnings (downgraded cards, dropped elements,
    approximations) are echoed to stderr — they used to be silently
    discarded on every CLI path."""
    import sys
    suf = Path(path).suffix.lower()
    if suf in (".madx", ".seq"):
        from linac_gen.io.madx_parser import parse_madx
        lat, meta = parse_madx(str(path))[:2]
    elif suf in (".lat", ".flat"):
        from linac_gen.io.mad8_parser import parse_mad8
        lat, meta = parse_mad8(str(path))[:2]
    elif suf == ".lte":
        from linac_gen.io.elegant_parser import parse_elegant
        lat, meta = parse_elegant(str(path))[:2]
    else:
        from linac_gen.io.tracewin_parser import parse_tracewin
        lat, meta = parse_tracewin(str(path))
    warns = meta.get("warnings", []) if isinstance(meta, dict) else []
    for w in warns:
        print(f"parse warning: {w}", file=sys.stderr)
    # Carry the downgrade ledger with the lattice so the HDF5
    # provenance/ group can record WHICH physics downgrades the parse
    # accepted (previously echoed to stderr and lost).
    lat.parse_warnings = list(warns)
    return lat


def load_input(path):
    """Load a CLI input — a ``.lgproj`` project *or* a bare lattice file.

    Returns ``(lattice, beam_config, convergence_dict)``.  For a bare
    lattice the beam is a default ``BeamConfig`` and convergence is empty.
    """
    p = Path(path)
    if p.suffix.lower() == ".lgproj":
        from linac_gen.io.project import load_project
        proj = load_project(p)
        return load_lattice(proj.lattice_path), proj.beam, proj.convergence
    from linac_gen.core.config import BeamConfig
    return load_lattice(str(p)), BeamConfig(), {}


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
    setattr(elem, attr, coerce_like(getattr(elem, attr), str(value)))


# ---------------------------------------------------------------------------
# config builders
# ---------------------------------------------------------------------------
def make_step_config(conv: dict, cli: dict):
    """Build a ``StepConfig`` — CLI override > project value > default."""
    from linac_gen.core.step_config import StepConfig
    d = StepConfig()
    step1 = _first(cli.get("step1"), (conv or {}).get("step1_per_m"),
                   d.integration_steps_per_metre)
    step2 = _first(cli.get("step2"), (conv or {}).get("step2_per_m"),
                   d.sc_steps_per_metre)
    return StepConfig(integration_steps_per_metre=float(step1),
                      sc_steps_per_metre=float(step2))


def make_sc_config(cfg, conv: dict, cli: dict):
    """Build a ``SpaceChargeConfig`` (or ``None`` when ``cfg.current<=0``).

    Resolution priority: CLI override > project ``convergence`` value >
    ``SpaceChargeConfig`` default.  ``cli`` may carry ``nx`` /
    ``grid_extent`` / ``kernel`` / ``backend`` and a nested ``sc`` dict
    of generic ``name=value`` overrides.
    """
    if cfg.current <= 0:
        return None
    from linac_gen.core.config import SpaceChargeConfig
    d = SpaceChargeConfig()
    conv = conv or {}
    nx = int(_first(cli.get("nx"), conv.get("grid_nx"), d.nx))
    kw = dict(
        nx=nx, ny=nx, nz=nx,
        grid_extent=float(_first(cli.get("grid_extent"),
                                 conv.get("grid_extent_sigma"),
                                 d.grid_extent)),
        kernel=str(_first(cli.get("kernel"), conv.get("kernel"), d.kernel)),
        green_kind=str(conv.get("green_kind", d.green_kind)),
        grid_mode=str(conv.get("grid_mode", d.grid_mode)),
        dc_kernel=str(conv.get("dc_kernel", d.dc_kernel)),
        use_gpu=str(_first(cli.get("backend"), conv.get("backend"),
                           d.use_gpu)),
        csr_enabled=bool(conv.get("csr_enabled", d.csr_enabled)),
    )
    for key, raw in (cli.get("sc") or {}).items():
        if not hasattr(d, key):
            raise ValueError(f"unknown space-charge parameter '{key}'")
        kw[key] = coerce_like(getattr(d, key), str(raw))
    return SpaceChargeConfig(**kw)


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
    return EnvelopeSolver(lattice, ref, initial, current=cfg.current).run()


def run_mp_sim(lattice, cfg, sc_config, step_config, *, seed: int = 42,
               progress_callback=None):
    """Run a multi-particle simulation.

    Returns ``(recorder, beam)`` — after the run ``beam.particles`` holds
    the final distribution (used for ``.dst`` export)."""
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    lattice.step_config = step_config
    beam = create_beam(cfg, seed=seed)
    sim = Simulation(lattice, beam, space_charge=sc_config,
                     progress_callback=progress_callback)
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
    from linac_gen.io.tracewin_dst import write_dst
    parts = getattr(beam, "alive_particles", None)
    if parts is None:
        parts = beam.particles
    write_dst(str(out), parts,
              current_mA=float(beam.current),
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
                     seed: int = 42, cli: dict | None = None):
    """Resolve a CLI input + overrides into a :class:`ScanPoint`.

    Used by both ``scan`` (one point per swept value) and ``batch`` (one
    point per job).  ``cli`` may carry fixed ``nx`` / ``grid_extent`` /
    ``step1`` / ``step2`` / ``backend`` overrides; ``beam_overrides`` and
    ``sc_overrides`` are ``name=value`` dicts; ``element_overrides`` is an
    iterable of ``(selector, value)`` pairs.
    """
    from dataclasses import asdict
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig
    from linac_gen.core.step_config import StepConfig
    from linac_gen.parallel.scan_pool import ScanPoint

    cli = cli or {}
    p = Path(input_path)
    if p.suffix.lower() == ".lgproj":
        from linac_gen.io.project import load_project
        proj = load_project(p)
        lattice_path, beam_cfg, conv = (
            proj.lattice_path, proj.beam, proj.convergence)
    else:
        lattice_path, beam_cfg, conv = str(p.resolve()), BeamConfig(), {}

    apply_beam_overrides(beam_cfg, beam_overrides or {})

    sd, scd = StepConfig(), SpaceChargeConfig()
    nx = int(_first(cli.get("nx"), conv.get("grid_nx"), scd.nx))
    grid_extent = float(_first(cli.get("grid_extent"),
                               conv.get("grid_extent_sigma"),
                               scd.grid_extent))
    step1 = float(_first(cli.get("step1"), conv.get("step1_per_m"),
                         sd.integration_steps_per_metre))
    step2 = float(_first(cli.get("step2"), conv.get("step2_per_m"),
                         sd.sc_steps_per_metre))
    backend = str(_first(cli.get("backend"), conv.get("backend"),
                         scd.use_gpu))

    # Extra SpaceChargeConfig kwargs (project values + CLI --sc overrides).
    sc_kw: dict = {}
    for key in ("kernel", "green_kind", "grid_mode", "dc_kernel",
                "csr_enabled"):
        if conv and key in conv:
            sc_kw[key] = conv[key]
    for key, raw in (sc_overrides or {}).items():
        if not hasattr(scd, key):
            raise ValueError(f"unknown space-charge parameter '{key}'")
        sc_kw[key] = coerce_like(getattr(scd, key), str(raw))

    return ScanPoint(
        lattice_path=lattice_path,
        beam_config=asdict(beam_cfg),
        nx=nx, grid_extent=grid_extent, step1=step1, step2=step2,
        seed=int(seed), use_gpu=backend,
        mode=mode, env_solver=env_solver,
        element_overrides=tuple(element_overrides),
        sc_overrides=tuple(sc_kw.items()),
    )
