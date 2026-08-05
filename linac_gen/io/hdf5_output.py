"""HDF5 results output: save/load DiagnosticRecorder data (Task 12.2)."""
import hashlib
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
    ├── beam_config/      scalar config values as HDF5 attrs (present only when provided)
    └── provenance/       code version + git commit + numpy/h5py versions,
                          write timestamp; plus lattice SHA-256/path, beam
                          seed and SC backend/grid configuration when the
                          caller provides them (all optional kwargs)

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
        Reserved for future use; currently ignored.
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
        env.attrs["current_mA"] = float(
            getattr(recorder, "current_mA", 0.0) or 0.0)
        _exit_idx = getattr(recorder, "element_exit_idx", None)
        if _exit_idx is not None and len(_exit_idx):
            env.create_dataset("element_exit_idx",
                               data=np.asarray(_exit_idx, dtype=np.int64))

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

        # ── particle snapshots ────────────────────────────────────────────────
        # ``_snapshots`` is a DiagnosticRecorder attribute; EnvelopeResults
        # (from the envelope solver) lacks it.  Use getattr so envelope-only
        # results dump cleanly to HDF5 without an AttributeError.
        snaps = getattr(recorder, "_snapshots", None) or {}
        if snaps:
            parts = f.create_group("particles")
            for i, (s_pos, (particles, ref_state)) in enumerate(
                sorted(snaps.items())
            ):
                grp = parts.create_group(f"s_{i:04d}")
                grp.create_dataset("data", data=particles)
                grp.attrs["s"]     = s_pos
                grp.attrs["w_kin"] = ref_state.w_kin
                grp.attrs["phi_s"] = ref_state.phi_s
                grp.attrs["beta"]  = ref_state.beta
                grp.attrs["gamma"] = ref_state.gamma

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
        all reference arrays (keyed as ``ref_<name>``).
    """
    results = {}
    with h5py.File(filepath, "r") as f:
        if "envelope" in f:
            for key in f["envelope"]:
                results[key] = f["envelope"][key][:]
            for key, val in f["envelope"].attrs.items():
                results[key] = val.item() if hasattr(val, "item") else val
        if "reference" in f:
            for key in f["reference"]:
                results[f"ref_{key}"] = f["reference"][key][:]
    return results
