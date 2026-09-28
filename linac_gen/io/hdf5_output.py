"""HDF5 results output: save/load DiagnosticRecorder data (Task 12.2)."""
import hashlib
import warnings

import numpy as np
import h5py


def _git_commit() -> str:
    """Best-effort short commit of the running source tree ("unknown"
    outside a git checkout)."""
    import os
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        commit = out.stdout.strip()
        return commit if out.returncode == 0 and commit else "unknown"
    except Exception:                                       # noqa: BLE001
        return "unknown"


def _hash_file(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _write_provenance(f, lattice_path=None, seed=None, sc_config=None,
                      lattice=None, input_beam_path=None):
    """``provenance/`` attr group: everything needed to say WHICH code,
    WHICH machine description and WHICH numerical configuration produced
    this file.  A results file without these cannot reproduce its run
    (PRAB review finding: only beam-config attrs were stored)."""
    import datetime

    from linac_gen import __version__

    prov = f.create_group("provenance")
    prov.attrs["linac_gen_version"] = __version__
    prov.attrs["git_commit"] = _git_commit()
    prov.attrs["numpy_version"] = np.__version__
    prov.attrs["h5py_version"] = h5py.__version__
    prov.attrs["written"] = datetime.datetime.now().astimezone().isoformat()
    if lattice_path:
        try:
            h = hashlib.sha256()
            with open(lattice_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            prov.attrs["lattice_sha256"] = h.hexdigest()
        except OSError:
            pass
        try:
            import os
            prov.attrs["lattice_path"] = os.path.abspath(str(lattice_path))
        except Exception:                                   # noqa: BLE001
            prov.attrs["lattice_path"] = str(lattice_path)
    if seed is not None:
        prov.attrs["beam_seed"] = int(seed)
    # OpenMP configuration: the C++ deposit kernel is bit-deterministic
    # only at a FIXED thread count, so record what this run used.
    try:
        import os
        prov.attrs["omp_num_threads"] = os.environ.get(
            "OMP_NUM_THREADS", f"unset (cpu_count={os.cpu_count()})")
        prov.attrs["omp_dynamic"] = os.environ.get("OMP_DYNAMIC", "unset")
    except Exception:                                       # noqa: BLE001
        pass
    # Which deposit/gather implementation actually ran (C++ OpenMP
    # kernels vs the pure-Python fallback) — the two are numerically
    # equivalent but not bit-identical, so a reproduction needs to know.
    try:
        from linac_gen.pic import pic_solver as _ps
        prov.attrs["sc_cpp_kernels"] = bool(getattr(_ps, "_USE_CPP",
                                                    False))
    except Exception:                                       # noqa: BLE001
        pass
    if sc_config is not None and not isinstance(sc_config, str):
        for key in ("sc_backend", "use_gpu", "grid_mode", "nx", "ny",
                    "nz", "grid_extent", "kernel", "green_kind",
                    "shape_order", "dc_kernel", "boundary",
                    "csr_enabled", "csr_bins", "csr_model"):
            val = getattr(sc_config, key, None)
            if val is not None:
                try:
                    prov.attrs[f"sc_{key}"] = val
                except TypeError:
                    pass                    # h5py-unencodable — skip
        try:
            from linac_gen.pic.gpu_backend import (effective_backend_info,
                                                   select_backend)
            info = effective_backend_info(
                getattr(sc_config, "use_gpu", "auto"))
            # Resolve "auto" to the backend the FFTs would actually use
            # (cpu vs gpu) — the exact cpu/gpu-FP32 ambiguity the
            # provenance group exists to remove.
            try:
                info["effective"] = select_backend(
                    getattr(sc_config, "use_gpu", "auto"))
            except Exception:                               # noqa: BLE001
                pass
            for key, val in info.items():
                prov.attrs[f"backend_{key}"] = val
        except Exception:                                   # noqa: BLE001
            pass

    # ---- referenced-input hashes & run configuration (2026-07) --------
    # lattice_sha256 covers the .dat text only, not the files it points
    # at — a reproduction needs the actual field-map data, the imported
    # beam, and the surrogate identities as well.
    prov.attrs["fp_dtype"] = "float64"      # tracker/envelope precision
    # OpenMP schedule: read from the kernel binary itself — a .so built
    # from pre-2026-07 source lacks the contractual schedule(static)
    # clause and must not be recorded as carrying it; the pure-Python
    # fallback has no OpenMP at all.
    try:
        from linac_gen import _pic_kernels as _pk
        if getattr(_pk, "_openmp_enabled", False):
            prov.attrs["omp_schedule"] = getattr(
                _pk, "_omp_schedule",
                "unknown (kernel predates the schedule(static) clause)")
        else:
            prov.attrs["omp_schedule"] = "n/a (kernel built without OpenMP)"
    except Exception:                                       # noqa: BLE001
        prov.attrs["omp_schedule"] = "n/a (python fallback deposit)"
    if input_beam_path:
        digest = _hash_file(input_beam_path)
        if digest:
            prov.attrs["input_beam_sha256"] = digest
            prov.attrs["input_beam_path"] = str(input_beam_path)
    if lattice is not None:
        cfg = getattr(lattice, "step_config", None)
        if cfg is not None:
            try:
                prov.attrs["integration_steps_per_metre"] = float(
                    cfg.integration_steps_per_metre)
                prov.attrs["sc_steps_per_metre"] = float(
                    cfg.sc_steps_per_metre)
                prov.attrs["drift_single_push"] = bool(
                    getattr(cfg, "drift_single_push", True))
            except Exception:                               # noqa: BLE001
                pass
        # Parser downgrade ledger (attached by cli.common.load_lattice).
        ledger = getattr(lattice, "parse_warnings", None)
        if ledger:
            try:
                prov.attrs["parse_downgrades"] = "\n".join(
                    str(w) for w in ledger)
            except Exception:                               # noqa: BLE001
                pass
        # Field-map data files: hash every existing file sharing each
        # element's resolved field-file prefix (.edz/.bsz/...).
        import glob
        import os
        fm_lines = []
        seen = set()

        def _hash_prefix(prefix):
            if not prefix or prefix in seen:
                return
            seen.add(prefix)
            for path in sorted(glob.glob(glob.escape(str(prefix)) + ".*")):
                digest = _hash_file(path)
                if digest:
                    fm_lines.append(
                        f"{os.path.basename(path)}:{digest}")

        for elem in getattr(lattice, "elements", []):
            _hash_prefix(getattr(elem, "field_file", None))
            # SUPERPOSE containers: the container itself carries no
            # field_file — the real files live on the children
            # ((z0, FieldMap) tuples), which never appear in
            # lattice.elements directly.
            for entry in getattr(elem, "children", None) or []:
                child = entry[1] if isinstance(entry, (tuple, list)) \
                    and len(entry) >= 2 else entry
                _hash_prefix(getattr(child, "field_file", None))
        if fm_lines:
            prov.attrs["field_map_sha256"] = "\n".join(fm_lines)
    # Surrogate manifest: identity of every registered surrogate that
    # tracking could have consulted (name-scoped lookup).
    try:
        from linac_gen.surrogates import registry as _reg
        lines = []
        for lhash, ekey in _reg.list_registered():
            s = _reg.get(lhash, ekey)
            meta = getattr(s, "metadata", None)
            if meta is None:
                continue
            line = (
                f"{ekey}:cls={getattr(meta, 'element_class', '?')}"
                f":seed={getattr(meta, 'training_seed', '?')}"
                f":val_mape={getattr(meta, 'val_mape', '?')}"
                f":lattice={lhash[:12]}")
            # Weights checksum (identity fields alone can't detect a
            # swapped or corrupted weights.pt).  Preferred source: the
            # metadata-level hash load_surrogate() computes from the
            # file bytes — present on EVERY load route (CLI, GUI,
            # Python API).  Fallback: re-hash via the retained
            # directory (older objects).
            wsha = getattr(meta, "weights_sha256", "") or ""
            if not wsha:
                wdir = getattr(s, "weights_dir", None)
                if wdir:
                    import os as _os
                    wsha = _hash_file(
                        _os.path.join(str(wdir), "weights.pt")) or ""
            if wsha:
                line += f":weights={wsha[:16]}"
            lines.append(line)
        if lines:
            prov.attrs["surrogates_registered"] = "\n".join(lines)
    except Exception:                                       # noqa: BLE001
        pass


def _utf8_s32(name) -> bytes:
    """``name`` as at most 32 UTF-8 bytes, cut on a character boundary."""
    b = str(name).encode("utf-8", "replace")
    return b if len(b) <= 32 else b[:32].decode("utf-8", "ignore").encode()


def _write_action_scan(f, recorder) -> None:
    """``action_scan/`` group — the opt-in halo action scan
    (``Simulation(record_action_scan=True)``).  Written only when the run
    recorded one, so every other file keeps exactly its former tree.

    Accepts a live recorder (per-step lists) and re-saved loaded results
    (the dict :func:`load_results_hdf5` returns, whose per-step flags and
    names ride inside the scan).  Everything is validated before the
    group is created: a scan not aligned with ``recorder.s`` is skipped
    with a warning and leaves no partial group behind."""
    scan = getattr(recorder, "action_scan", None)
    if not scan or "n" not in scan:
        return
    from linac_gen.diagnostics.action_scan import (
        DEFINITION, EPS_UNITS, PLANES, SCHEMA_VERSION)
    n_steps = len(recorder.s)
    grid = np.asarray(scan["n"], dtype=float).ravel()
    planes = [p for p in PLANES if f"count_{p}" in scan]
    data: dict = {}
    problem = None
    try:
        data["n_alive"] = np.asarray(scan["n_alive"], dtype=np.int64)
        for p in planes:
            data[f"count_{p}"] = np.asarray(scan[f"count_{p}"],
                                            dtype=np.int32)
            for key in ("eps", "eps_n", "n_max"):
                data[f"{key}_{p}"] = np.asarray(scan[f"{key}_{p}"],
                                                dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        problem = f"{type(exc).__name__}: {exc}"
    if problem is None:
        if n_steps == 0 or not planes:
            problem = f"{n_steps} recorded steps, planes {planes}"
        else:
            for key, arr in data.items():
                want = ((n_steps, grid.size) if key.startswith("count_")
                        else (n_steps,))
                if arr.shape != want:
                    problem = f"{key} has shape {arr.shape}, expected {want}"
                    break
    if problem is not None:
        warnings.warn(f"action scan not written ({problem})",
                      RuntimeWarning, stacklevel=3)
        return

    def _per_step(attr, key):
        vals = getattr(recorder, attr, None)
        if vals is None or len(vals) != n_steps:
            vals = scan.get(key)
        return vals if vals is not None and len(vals) == n_steps else None

    cont = _per_step("continuous_at", "continuous")
    names = _per_step("element_names", "element")
    g = f.create_group("action_scan")
    g.attrs["schema_version"] = SCHEMA_VERSION
    g.attrs["definition"] = DEFINITION
    g.attrs["eps_units"] = EPS_UNITS
    g.attrs["planes"] = ",".join(planes)
    periodic = getattr(recorder, "periodic_phase", None)
    if periodic is None:
        periodic = (scan.get("attrs") or {}).get("periodic_phase")
    if periodic is not None:                     # absent = unknown
        g.attrs["periodic_phase"] = bool(periodic)
    g.create_dataset("n", data=grid)
    g.create_dataset("n_alive", data=data.pop("n_alive"))
    if cont is not None:
        g.create_dataset("continuous", data=np.asarray(cont, dtype=bool))
    if names is not None:
        g.create_dataset("element", data=np.array(
            [_utf8_s32(x) for x in names], dtype="S32"))
    for key, arr in data.items():
        if key.startswith("count_"):
            g.create_dataset(key, data=arr, chunks=True, compression="gzip",
                             compression_opts=4, shuffle=True)
        else:
            g.create_dataset(key, data=arr)


def _read_action_scan(g, n_steps: int):
    """Inverse of :func:`_write_action_scan`.

    Returns ``(scan, None)`` — the known datasets by name plus the group
    attributes under ``"attrs"`` — or ``(None, reason)`` when the group is
    from a newer schema, incomplete or not aligned with the file's
    ``envelope/s``.  Never raises: the rest of the file still loads.
    Unknown members (datasets, sub-groups, attributes a later writer may
    add) are ignored."""
    from linac_gen.diagnostics.action_scan import PLANES, SCHEMA_VERSION
    per_plane = ("count", "eps", "eps_n", "n_max")
    try:
        attrs = {}
        for k, v in g.attrs.items():
            try:
                attrs[k] = v.item() if getattr(v, "size", 1) == 1 else v
            except Exception:                               # noqa: BLE001
                attrs[k] = v
        version = int(attrs.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(f"schema_version {version} is newer than this "
                             f"reader ({SCHEMA_VERSION})")
        planes = [p for p in PLANES if f"count_{p}" in g]
        if "n" not in g or "n_alive" not in g or not planes:
            raise ValueError("no grid, particle counts or count datasets")
        keys = ["n_alive", "continuous", "element"] + [
            f"{k}_{p}" for p in planes for k in per_plane]
        scan: dict = {"attrs": attrs, "n": np.asarray(g["n"][:], float)}
        for key in keys:
            if key not in g:
                if key in ("continuous", "element"):
                    continue                         # optional per step
                raise ValueError(f"{key} missing")
            ds = g[key]
            if not isinstance(ds, h5py.Dataset) or ds.ndim == 0:
                raise ValueError(f"{key} is not an array")
            if key == "element":
                scan[key] = [b.decode("utf-8", "replace") for b in ds[:]]
            else:
                scan[key] = ds[:]
            if len(scan[key]) != n_steps:
                raise ValueError(f"{key} has {len(scan[key])} rows for "
                                 f"{n_steps} recorded steps")
        for p in planes:
            if scan[f"count_{p}"].ndim != 2 or \
                    scan[f"count_{p}"].shape[1] != scan["n"].size:
                raise ValueError(f"count_{p} does not match the grid")
    except Exception as exc:                                # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"action_scan group ignored: {reason}", RuntimeWarning,
                      stacklevel=3)
        return None, reason
    return scan, None


# Per-step recorder series beyond the historical envelope set.  Numeric
# ones are stored with their natural shape — (n,), (n, 6) for centroid,
# (n, 6, 6) for sigma_matrix; ``element_names`` as UTF-8 strings.
_EXTRA_PER_STEP = (
    "emit_z_mmmrad", "emit_nz", "emit_4d", "emit_n1", "emit_n2",
    "emit_e1", "emit_e2", "emit_e3", "x_max", "y_max",
    "continuous_at", "centroid", "sigma_matrix", "element_names",
)
# Loaded back as lists (like a live recorder) rather than numpy arrays:
# the GUI tests them with plain truthiness / iterates them per step.
_LIST_PER_STEP = ("centroid", "sigma_matrix", "element_names")


def _per_step(vals, n: int):
    """``vals`` when it is a sequence with one entry per recorded step,
    else None (a scalar, a short or a missing series is not written)."""
    if vals is None or n == 0 or isinstance(vals, (str, bytes)):
        return None
    try:
        return vals if len(vals) == n else None
    except TypeError:
        return None


def _create(grp, name, arr):
    """Dataset, gzip-compressed (with byte shuffle) when it is large —
    the per-step 6×6 beam matrices and density histograms dominate the
    file otherwise."""
    arr = np.asarray(arr)
    if arr.size >= 4096 and arr.dtype.kind in "fiub":
        return grp.create_dataset(name, data=arr, compression="gzip",
                                  compression_opts=4, shuffle=True)
    return grp.create_dataset(name, data=arr)


def _write_per_step(grp, recorder, attr: str, n: int) -> None:
    vals = _per_step(getattr(recorder, attr, None), n)
    if vals is None:
        return
    if attr == "element_names":
        grp.create_dataset(attr, data=["" if v is None else str(v)
                                       for v in vals],
                           dtype=h5py.string_dtype("utf-8"))
        return
    try:
        arr = np.asarray(vals, dtype=bool if attr == "continuous_at"
                         else float)
    except (TypeError, ValueError):
        return                          # ragged / non-numeric: skip
    _create(grp, attr, arr)


def _write_run_scalars(grp, recorder) -> None:
    """Run scalars the analyses read off the results: the particle rest
    mass (IBS / magnetic-stripping popups, 6-D emittance and σ(Δp/p)
    conversions — without it they fell back to a mass re-derived from
    W/(γ−1)) and the periodic-phase flag (φ wrapping in the plots)."""
    _mass = getattr(recorder, "mass_mev", None)
    if _mass:
        grp.attrs["mass_mev"] = float(_mass)
    if hasattr(recorder, "periodic_phase"):
        grp.attrs["periodic_phase"] = bool(recorder.periodic_phase)


def _restore_per_step(results: dict) -> None:
    """Per-step series stored with their natural shape → the live
    recorder's list form (names decoded to str, flags to bool).  Shared by
    the native and the openPMD loaders."""
    for key in _LIST_PER_STEP:
        if key in results:
            vals = results[key]
            if key == "element_names":
                results[key] = [v.decode("utf-8") if isinstance(v, bytes)
                                else str(v) for v in vals]
            else:
                results[key] = [np.asarray(v) for v in vals]
    if "continuous_at" in results:
        results["continuous_at"] = [bool(v) for v in results["continuous_at"]]


def _write_run_records(grp, recorder) -> None:
    """Per-particle loss record, stripper-foil record and halo action scan
    under ``grp`` (the file root in the native format; the first iteration
    group in the openPMD file) — each written only when the run has it."""
    # ── per-particle loss record ─────────────────────────────────────────
    # Attached by Simulation._run_mp (Beam.record_loss sites: apertures,
    # RFQ boundary, tracker limits).  Powers the loss-power analysis on
    # reloaded runs; n_macro is the LAUNCHED macroparticle count (each
    # carries I_avg/n_macro of beam current).
    losses = getattr(recorder, "loss_table", None)
    # the LAUNCHED macroparticle count at the root as well: the losses/
    # group (and its n_macro) exists only when something was lost, and
    # a lossless run still needs it for power densities
    if getattr(recorder, "n_macro", None):
        grp.attrs["n_macro"] = int(recorder.n_macro)
    if losses is not None and np.asarray(losses).size:
        lt = np.asarray(losses)
        lg = grp.create_group("losses")
        for key in ("particle_id", "s", "x", "y", "energy"):
            lg.create_dataset(key, data=np.asarray(lt[key]))
        names = np.asarray(lt["element_name"]).astype("S32")
        lg.create_dataset("element_name", data=names)
        lg.attrs["n_macro"] = int(getattr(recorder, "n_macro", 0))

    # ── stripper-foil record (Foil strip_model / extent) ─────────────────
    # Written only when non-empty, so every file without such a foil is
    # byte-identical to before this group existed.
    unstripped = getattr(recorder, "unstripped_table", None)
    if unstripped is not None and np.asarray(unstripped).size:
        ut = np.asarray(unstripped)
        ug = grp.create_group("unstripped")
        for key in ("particle_id", "x", "y", "energy"):
            ug.create_dataset(key, data=np.asarray(ut[key]))
        ug.create_dataset("state", data=np.asarray(ut["state"]).astype("S8"))
        ug.create_dataset("element_name",
                          data=np.asarray(ut["element_name"]).astype("S32"))

    # ── halo action scan (opt-in; only when recorded) ────────────────────
    _write_action_scan(grp, recorder)


def _read_run_records(grp, results: dict, n_steps: int) -> None:
    """Inverse of :func:`_write_run_records`."""
    if "losses" in grp:
        from linac_gen.core.beam import LOSS_DTYPE
        lg = grp["losses"]
        n = lg["s"].shape[0]
        lt = np.zeros(n, dtype=LOSS_DTYPE)
        for key in ("particle_id", "s", "x", "y", "energy"):
            lt[key] = lg[key][:]
        lt["element_name"] = lg["element_name"][:].astype("U32")
        results["loss_table"] = lt
        results["n_macro"] = int(lg.attrs.get("n_macro", 0))
    if "n_macro" in grp.attrs:
        results["n_macro"] = int(grp.attrs["n_macro"])
    if "unstripped" in grp:
        from linac_gen.core.beam import UNSTRIPPED_DTYPE
        ug = grp["unstripped"]
        n = ug["x"].shape[0]
        ut = np.zeros(n, dtype=UNSTRIPPED_DTYPE)
        for key in ("particle_id", "x", "y", "energy"):
            ut[key] = ug[key][:]
        ut["state"] = ug["state"][:].astype("U8")
        ut["element_name"] = ug["element_name"][:].astype("U32")
        results["unstripped_table"] = ut
    if "action_scan" in grp:
        scan, reason = _read_action_scan(grp["action_scan"], n_steps)
        if scan is not None:
            results["action_scan"] = scan
        else:                     # say why, instead of "not recorded"
            results["action_scan_error"] = reason


def _write_density(f, recorder, n: int) -> None:
    """``density/<axis>``: (n_steps, n_bins) int32 counts + ``edges``."""
    dens = getattr(recorder, "density", None) or {}
    edges = getattr(recorder, "density_edges", None) or {}
    axes = [a for a, cols in dens.items() if cols and len(cols) == n
            and a in edges]
    if not axes:
        return
    g = f.create_group("density")
    for a in axes:
        sub = g.create_group(a)
        _create(sub, "counts", np.asarray(dens[a], dtype=np.int32))
        sub.create_dataset("edges", data=np.asarray(edges[a], dtype=float))


def _write_tail(f, recorder, n: int) -> None:
    """``tail/<key>``: fractional-emittance / radial-quantile series."""
    tail = getattr(recorder, "tail", None) or {}
    keys = [k for k, v in tail.items() if len(v) == n]
    if not keys:
        return
    g = f.create_group("tail")
    g.attrs["fractions"] = np.asarray(
        getattr(recorder, "tail_fractions", ()), dtype=float)
    for k in keys:
        g.create_dataset(k, data=np.asarray(tail[k], dtype=float))


def save_results_hdf5(recorder, filepath: str, beam_config=None,
                      lattice=None, *, lattice_path=None, seed=None,
                      sc_config=None, input_beam_path=None) -> None:
    """Save DiagnosticRecorder to an HDF5 file.

    File structure
    --------------
    results.h5
    ├── envelope/         sigma_x, sigma_y, emit_x, … vs s  (1-D arrays)
    ├── reference/        w_kin, phi_s, beta, gamma, bg vs s (1-D arrays)
    ├── particles/        full phase-space snapshots (present only when snapshots exist)
    │   ├── s_0000/
    │   │   ├── data      (N, 6) float64 particle array
    │   │   └── attrs     s, w_kin, phi_s, beta, gamma
    │   └── …
    ├── action_scan/      halo action scan: particles outside n·ε_rms ellipses per
    │                     step and plane (present only when the run recorded it)
    ├── beam_config/      scalar config values as HDF5 attrs (present only when provided)
    └── provenance/       code version + git commit + numpy/h5py versions,
                          write timestamp; plus lattice SHA-256/path, beam
                          seed, SC backend/grid configuration, and — from
                          the ``lattice`` object — per-metre step config
                          (``integration_steps_per_metre``, ``sc_steps_per_metre``,
                          ``drift_single_push``),
                          the parser downgrade ledger and field-map
                          data-file hashes, when the caller provides them
                          (all optional kwargs)

    Parameters
    ----------
    recorder : DiagnosticRecorder
        Populated recorder instance.
    filepath : str
        Destination HDF5 file path.
    beam_config : object or None
        Object whose ``__dict__`` is inspected for scalar config entries.
        ``None`` values are silently skipped.
    lattice : object or None
        Feeds ``provenance/``: per-metre step configuration
        (``lattice.step_config``), the parser downgrade ledger
        (``lattice.parse_warnings``, attached by
        ``cli.common.load_lattice``) and SHA-256 hashes of every
        element's field-map data files.
    lattice_path : str or None
        Source deck path — stored together with its SHA-256 so the file
        pins WHICH machine description produced it.
    seed : int or None
        Beam-sampling RNG seed of the run.
    sc_config : SpaceChargeConfig or None
        Stored as ``sc_*`` attrs plus the EFFECTIVE backend resolution
        (including any ``LINAC_GEN_USE_GPU`` environment override).
    """
    with h5py.File(filepath, "w") as f:
        _write_provenance(f, lattice_path=lattice_path, seed=seed,
                          sc_config=sc_config, lattice=lattice,
                          input_beam_path=input_beam_path)
        # ── envelope ─────────────────────────────────────────────────────────
        env = f.create_group("envelope")
        env.create_dataset("s", data=np.array(recorder.s))
        for attr in (
            "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
            "emit_x", "emit_y", "emit_z", "emit_nx", "emit_ny",
            "alpha_x", "beta_x", "alpha_y", "beta_y",
            "alpha_z", "beta_z",
            "halo_x", "halo_y", "transmission",
        ):
            if hasattr(recorder, attr):
                env.create_dataset(attr, data=np.array(getattr(recorder, attr)))
        # DC/continuous markers + run current + element→record map: needed by
        # analyses of RELOADED runs (the LEBT SCC popup gates on `continuous`
        # and scales by `current_mA`; adversarial finding F2 — loaded DC runs
        # were refused as "bunched" because none of this was persisted).
        env.attrs["continuous"] = bool(getattr(recorder, "continuous", False))
        # Run current: written ONLY when known (finite) — 0.0 means "ran
        # at 0 mA", absence means unknown.  The boolean marker lets the
        # loader tell a genuine 0.0 from the pre-fix MP sentinel (files
        # written before this change stored 0.0 for every MP run).
        from linac_gen.diagnostics.recorder import run_current_mA
        _cur = run_current_mA(recorder)
        env.attrs["run_current_known"] = bool(_cur is not None)
        if _cur is not None:
            env.attrs["current_mA"] = float(_cur)   # 0.0 = ran at 0 mA
        _exit_idx = getattr(recorder, "element_exit_idx", None)
        if _exit_idx is not None and len(_exit_idx):
            env.create_dataset("element_exit_idx",
                               data=np.asarray(_exit_idx, dtype=np.int64))
        # Everything else a live run carries per step, so an imported run
        # fills the same Results tiles (longitudinal normalised and 4-D /
        # eigen-emittances, the full 6×6 beam matrix behind the dispersion,
        # σ-matrix and 6-D-emittance plots, centroids, peak excursions,
        # element names).  Written only when present with one entry per
        # recorded step.
        _n = len(recorder.s)
        for attr in _EXTRA_PER_STEP:
            _write_per_step(env, recorder, attr, _n)
        _write_run_scalars(env, recorder)

        # ── reference history ─────────────────────────────────────────────────
        ref_grp = f.create_group("reference")
        for attr, key in (
            ("ref_w_kin", "w_kin"),
            ("ref_phi_s", "phi_s"),
            ("ref_beta",  "beta"),
            ("ref_gamma", "gamma"),
            ("ref_bg",    "bg"),
        ):
            if hasattr(recorder, attr):
                ref_grp.create_dataset(key, data=np.array(getattr(recorder, attr)))
        _freq = _per_step(getattr(recorder, "ref_frequency", None), _n)
        if _freq is not None:
            ref_grp.create_dataset("frequency",
                                   data=np.asarray(_freq, dtype=float))

        # ── density vs s and tail quantiles (opt-in recordings) ─────────────
        _write_density(f, recorder, _n)
        _write_tail(f, recorder, _n)

        # ── particle snapshots ────────────────────────────────────────────────
        # ``_snapshots`` is a DiagnosticRecorder attribute; EnvelopeResults
        # (from the envelope solver) lacks it.  Use getattr so envelope-only
        # results dump cleanly to HDF5 without an AttributeError.
        snaps = getattr(recorder, "_snapshots", None) or {}
        masks = getattr(recorder, "_snapshot_masks", None) or {}
        if snaps:
            parts = f.create_group("particles")
            for i, (s_pos, (particles, ref_state)) in enumerate(
                sorted(snaps.items())
            ):
                grp = parts.create_group(f"s_{i:04d}")
                grp.create_dataset("data", data=particles)
                # the per-snapshot lost mask (True = lost at that s), so a
                # reader can restrict a snapshot to the alive particles
                mask = masks.get(s_pos)
                if mask is not None:
                    grp.create_dataset("lost", data=np.asarray(mask, dtype=bool))
                grp.attrs["s"]     = s_pos
                grp.attrs["w_kin"] = ref_state.w_kin
                grp.attrs["phi_s"] = ref_state.phi_s
                grp.attrs["beta"]  = ref_state.beta
                grp.attrs["gamma"] = ref_state.gamma

        _write_run_records(f, recorder)

        # ── beam config ───────────────────────────────────────────────────────
        if beam_config is not None:
            cfg = f.create_group("beam_config")
            for key, val in beam_config.__dict__.items():
                if val is not None:
                    try:
                        cfg.attrs[key] = val
                    except TypeError:
                        # Skip values that HDF5 attrs cannot encode
                        pass


def load_results_hdf5(filepath: str) -> dict:
    """Load results from an HDF5 file produced by :func:`save_results_hdf5`.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.

    Returns
    -------
    results : dict
        Dictionary containing all envelope arrays (keyed by dataset name) and
        all reference arrays (keyed as ``ref_<name>``); plus, when the file
        carries one, ``action_scan`` — a dict of the ``action_scan/``
        datasets and attrs.
    """
    results = {}
    with h5py.File(filepath, "r") as f:
        if "envelope" in f:
            for key in f["envelope"]:
                results[key] = f["envelope"][key][:]
            for key, val in f["envelope"].attrs.items():
                results[key] = val.item() if hasattr(val, "item") else val
            # Resolve the run current.  ``run_current_known`` is a
            # transport-only marker (never a results field): absent =
            # legacy writer, whose 0.0 was an unconditional MP sentinel;
            # False = declared unknown.  Either way fall back to the
            # dump-time beam_config current when the file has one, else
            # drop the key so getattr(..., None) reports "unknown".
            known = results.pop("run_current_known", None)
            cur = results.get("current_mA")
            if cur is not None and (known is False
                                    or (known is None
                                        and float(cur) == 0.0)):
                cur = None
            if cur is None:
                bc = (f["beam_config"].attrs.get("current")
                      if "beam_config" in f else None)
                if bc is not None:
                    results["current_mA"] = float(bc)
                else:
                    results.pop("current_mA", None)
        if "reference" in f:
            for key in f["reference"]:
                results[f"ref_{key}"] = f["reference"][key][:]
        _n_steps = (f["envelope"]["s"].shape[0]
                    if "envelope" in f and "s" in f["envelope"] else 0)
        _read_run_records(f, results, _n_steps)
        _restore_per_step(results)
        if "density" in f:
            dens, edges = {}, {}
            for axis, sub in f["density"].items():
                dens[axis] = [row for row in sub["counts"][:]]
                edges[axis] = sub["edges"][:]
            results["density"] = dens
            results["density_edges"] = edges
            results["density_axes"] = tuple(dens)
        if "tail" in f:
            tg = f["tail"]
            results["tail"] = {k: tg[k][:].tolist() for k in tg}
            results["tail_fractions"] = tuple(
                float(x) for x in tg.attrs.get("fractions", ()))
    return results
