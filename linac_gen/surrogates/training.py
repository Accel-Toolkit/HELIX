"""Training pipeline for HELIX envelope-mode surrogates.

For a chosen :class:`FieldMapElement` instance, this module:

1. Generates ``(X, Y)`` samples via Latin Hypercube over ref
   kinematics + element parameter ranges.  Each sample's ``Y`` is
   the flattened 6x6 transfer matrix from the wrapped element's
   :meth:`fitted_matrix`.
2. Trains an :class:`MlpHead` on the samples (FP64 CPU, smooth
   activation, Adam, early-stopping on val MAPE).
3. Persists the trained weights + a :class:`SurrogateMetadata`
   JSON manifest for runtime use.

CLI entry point lives in ``linac_gen/surrogates/cli.py`` (M5);
this module exposes the building blocks.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates.base import (
    F64,
    MlpHead,
    Scope,
    SurrogateMetadata,
)


# ---------------------------------------------------------------------------
# Multiprocessing worker plumbing.  These functions live at module scope
# (not as closures) so they're picklable on macOS spawn-mode Pools.  Each
# worker keeps a deep-copy of the element + ref_template + param_names in
# a module-level dict, populated once by the Pool's initializer; per-task
# calls only ship a small (idx, w_kin, params) tuple.
# ---------------------------------------------------------------------------
_WORKER_STATE: dict = {}


def _worker_init(element, ref_template, param_names,
                 counters=None, id_counter=None) -> None:
    """Pool initializer: stash a deep-copied element per worker process.

    For progress reporting each worker claims a unique slot ``wid`` in the
    shared, LOCK-FREE ``counters`` array (the id is handed out once via the
    locked ``id_counter`` at startup) and writes its own running sample
    count into that slot.  Single-writer-per-slot means zero lock
    contention on the hot per-sample path; the parent sums the array for
    total progress.
    """
    _WORKER_STATE["element"] = copy.deepcopy(element)
    _WORKER_STATE["ref_template"] = ref_template
    _WORKER_STATE["param_names"] = list(param_names)
    _WORKER_STATE["counters"] = counters
    _WORKER_STATE["local"] = 0
    if counters is not None and id_counter is not None:
        with id_counter.get_lock():
            _WORKER_STATE["wid"] = int(id_counter.value)
            id_counter.value += 1


def _worker_compute_one(args):
    """Worker task: set params, reset state, compute the 6x6 matrix."""
    idx, w_kin, params = args
    el = _WORKER_STATE["element"]
    for j, name in enumerate(_WORKER_STATE["param_names"]):
        setattr(el, name, float(params[j]))
    if hasattr(el, "reset_run_state"):
        el.reset_run_state()
    ref = _WORKER_STATE["ref_template"].copy()
    ref.w_kin = float(w_kin)
    M = el.fitted_matrix(ref)
    # Record progress in this worker's OWN slot -- single writer, so no
    # lock is needed.  The parent sums the slots while results are still
    # collected in large, low-IPC chunks.
    counters = _WORKER_STATE.get("counters")
    wid = _WORKER_STATE.get("wid")
    # Guard wid in range: if the Pool ever replaces a dead worker, the
    # extra init would hand out an id past the array -- skip recording for
    # it (the final 100% tick still corrects the small undercount) rather
    # than raising IndexError on the hot path.
    if counters is not None and wid is not None and wid < len(counters):
        _WORKER_STATE["local"] += 1
        counters[wid] = _WORKER_STATE["local"]
    return (
        int(idx),
        float(ref.beta),
        float(ref.gamma),
        np.asarray(M, dtype=np.float64).reshape(36),
    )


# ---------------------------------------------------------------------------
def _git_head_sha(cwd: str | None = None) -> str:
    """Capture the HELIX HEAD sha for reproducibility, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or os.getcwd(),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
def generate_training_data(
    element: FieldMapElement,
    ref_template,
    n_samples: int,
    ref_w_kin_range: tuple[float, float],
    param_ranges: dict[str, tuple[float, float]],
    seed: int = 42,
    verbose: bool = False,
    n_workers: int | None = None,
    progress_callback=None,
    should_stop=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate envelope-mode training data via Latin Hypercube sampling.

    Parameters
    ----------
    element : FieldMapElement
        Source whose :meth:`fitted_matrix(ref)` provides the
        ground-truth 6x6 matrix.  Deep-copied so the lattice is
        unmutated.
    ref_template
        Reference particle to clone for each sample; only ``w_kin``
        is varied (species, frequency, etc. stay fixed).
    n_samples : int
    ref_w_kin_range : (float, float)
        ``(w_lo, w_hi)`` in MeV.
    param_ranges : dict[str, (float, float)]
        Element-attribute names → ``(lo, hi)`` bounds.
    seed : int
    verbose : bool
    n_workers : int or None
        ``None`` or ``<= 1`` runs the original serial loop (default,
        backward-compatible).  ``>= 2`` parallelises sample generation
        across a :class:`multiprocessing.Pool`.  Output is
        bit-identical to the serial path for a fixed seed — each
        worker runs the exact scalar ``fitted_matrix(ref)`` call on
        its own deep-copied element; only the dispatch is parallel.

    Returns
    -------
    X : (n_samples, 3 + len(param_ranges)) ndarray
        Columns: ``[w_kin, beta, gamma, *params]``.
    Y : (n_samples, 36) ndarray
        Flattened 6x6 matrices.
    info : dict
        ``param_names``, ``input_names``, ``input_lo``, ``input_hi``
        (the observed sampling scope).
    """
    from scipy.stats import qmc

    param_names = list(param_ranges.keys())
    n_param = len(param_names)
    input_dim = 3 + n_param

    sampler = qmc.LatinHypercube(d=1 + n_param, seed=seed)
    raw = sampler.random(n_samples)
    w_lo, w_hi = float(ref_w_kin_range[0]), float(ref_w_kin_range[1])
    p_lo = np.asarray([param_ranges[p][0] for p in param_names], dtype=np.float64)
    p_hi = np.asarray([param_ranges[p][1] for p in param_names], dtype=np.float64)
    scaled = np.empty((n_samples, 1 + n_param), dtype=np.float64)
    scaled[:, 0] = w_lo + raw[:, 0] * (w_hi - w_lo)
    if n_param:
        scaled[:, 1:] = p_lo + raw[:, 1:] * (p_hi - p_lo)

    X = np.empty((n_samples, input_dim), dtype=np.float64)
    Y = np.empty((n_samples, 36), dtype=np.float64)

    # Pre-fill the X columns that don't need fitted_matrix to know.
    X[:, 0] = scaled[:, 0]
    if n_param:
        X[:, 3:] = scaled[:, 1:]

    use_mp = (n_workers is not None) and (int(n_workers) >= 2)

    t0 = time.time()
    # Fire one progress event per ~1% of samples (capped at 1000 so
    # callbacks don't flood for very large runs).  The callback is
    # cheap (a dict emit) but the GUI redraws downstream are not.
    progress_every = max(1, min(n_samples // 100, n_samples // 1000 + 1))
    if progress_callback is None:
        progress_every = n_samples + 1   # silently no-op
    if not use_mp:
        # ---- Serial path (backward-compatible default) -------------
        work_elem = copy.deepcopy(element)
        for i in range(n_samples):
            if should_stop is not None and should_stop():
                from linac_gen.core.cancelled import OperationCancelled
                raise OperationCancelled(
                    f"data generation cancelled at sample {i}/{n_samples}")
            w_kin = float(scaled[i, 0])
            for j, name in enumerate(param_names):
                setattr(work_elem, name, float(scaled[i, 1 + j]))
            if hasattr(work_elem, "reset_run_state"):
                work_elem.reset_run_state()
            ref = ref_template.copy()
            ref.w_kin = w_kin
            M = work_elem.fitted_matrix(ref)
            X[i, 1] = float(ref.beta)
            X[i, 2] = float(ref.gamma)
            Y[i] = np.asarray(M, dtype=np.float64).reshape(36)
            if verbose and ((i + 1) % max(1, n_samples // 10) == 0):
                elapsed = time.time() - t0
                print(f"  data gen {i+1}/{n_samples}  ({elapsed:.1f}s)",
                      flush=True)
            if progress_callback and ((i + 1) % progress_every == 0):
                progress_callback({
                    "stage": "data_gen", "done": i + 1,
                    "total": n_samples,
                    "elapsed_s": time.time() - t0,
                })
    else:
        # ---- Multiprocessing path --------------------------------
        import multiprocessing as mp
        # Force "spawn" everywhere so behaviour matches between
        # macOS (default) and Linux (default 'fork') — fork can
        # leak torch/CUDA fd's and copy-on-write is fragile here.
        ctx = mp.get_context("spawn")
        n_workers = max(1, int(n_workers))
        tasks = [
            (i, float(scaled[i, 0]),
             np.asarray(scaled[i, 1:1 + n_param], dtype=np.float64))
            for i in range(n_samples)
        ]
        if verbose:
            print(f"  data gen: {n_samples} samples on "
                  f"{n_workers} workers (mp.spawn)", flush=True)
        # Decouple PROGRESS from RESULT DELIVERY so the run is both FAST
        # and SMOOTH:
        #   * Results are collected in large, low-IPC chunks
        #     (chunksize = n_samples / (n_workers*4)) -> good throughput
        #     and load balance, just like a normal Pool.map.
        #   * Progress comes from a SHARED COUNTER each worker bumps per
        #     sample, polled here ~10x/s.  This sidesteps both failure
        #     modes: a big chunksize alone makes the synchronised workers
        #     deliver results in n_workers*chunksize waves (bar freezes
        #     then jumps ~96), while chunksize=1 streams smoothly but pays
        #     a per-sample IPC round-trip (slow).  Here the bar advances
        #     continuously regardless of chunk size, at full throughput.
        # Lock-free per-worker progress slots (one slot per worker, single
        # writer each); ``id_counter`` only hands out the slot indices once
        # at worker startup.
        id_counter = ctx.Value("i", 0)
        counters = ctx.Array("L", n_workers, lock=False)
        chunksize = max(1, n_samples // (n_workers * 4))
        log_every = max(1, n_samples // 10)
        last_log = 0
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(element, ref_template, param_names,
                      counters, id_counter),
        ) as pool:
            # Initial 0/N tick so the bar shows the run has started while
            # the spawn Pool brings its workers online (the one-time
            # startup gap before the first sample completes).
            if progress_callback:
                progress_callback({
                    "stage": "data_gen", "done": 0,
                    "total": n_samples, "elapsed_s": time.time() - t0,
                })
            async_res = pool.map_async(
                _worker_compute_one, tasks, chunksize=chunksize)
            # Poll the summed per-worker slots ~20x/s while the workers
            # churn; wait() returns as soon as everything finishes.
            while not async_res.ready():
                if should_stop is not None and should_stop():
                    # Raising INSIDE the with-block matters: Pool.__exit__
                    # calls terminate(), killing the spawn workers
                    # promptly instead of letting them finish the batch.
                    from linac_gen.core.cancelled import OperationCancelled
                    raise OperationCancelled(
                        "data generation cancelled (mp pool terminated)")
                async_res.wait(0.05)
                done = int(sum(counters))
                if verbose and done - last_log >= log_every:
                    last_log = done
                    print(f"  data gen {done}/{n_samples}  "
                          f"({time.time() - t0:.1f}s)", flush=True)
                if progress_callback:
                    progress_callback({
                        "stage": "data_gen", "done": done,
                        "total": n_samples,
                        "elapsed_s": time.time() - t0,
                    })
            for (idx, beta, gamma, M_flat) in async_res.get():
                X[idx, 1] = beta
                X[idx, 2] = gamma
                Y[idx] = M_flat
    # Final tick so the UI always sees 100%.
    if progress_callback:
        progress_callback({
            "stage": "data_gen", "done": n_samples,
            "total": n_samples, "elapsed_s": time.time() - t0,
        })

    # Scope = user-intended bounds (NOT observed LHS min/max — LHS
    # samples never hit the exact boundaries, so observed bounds
    # would incorrectly reject in-range queries near the edges).
    # For derived dims (beta, gamma) use generous physical bounds;
    # they're functions of w_kin and stay in range automatically.
    info = {
        "param_names": param_names,
        "input_names": ["w_kin", "beta", "gamma"] + param_names,
        "input_lo": [w_lo, 0.0, 1.0] + [param_ranges[p][0] for p in param_names],
        "input_hi": [w_hi, 1.0, 1000.0] + [param_ranges[p][1] for p in param_names],
    }
    return X, Y, info


# ---------------------------------------------------------------------------
def train_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_dims: Sequence[int] = (128, 128, 128),
    activation: str = "silu",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_frac: float = 0.2,
    seed: int = 42,
    verbose: bool = False,
    progress_callback=None,
    should_stop=None,
) -> tuple[MlpHead, dict, float]:
    """Train an MLP on (X, Y) for envelope-mode 6x6 matrix prediction.

    Returns ``(mlp, normalisation, best_val_mape)``.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = X.shape[0]
    perm = rng.permutation(n)
    X = X[perm]
    Y = Y[perm]
    n_val = max(1, int(n * val_frac))
    X_train, X_val = X[n_val:], X[:n_val]
    Y_train, Y_val = Y[n_val:], Y[:n_val]

    in_mean = X_train.mean(0)
    in_std = X_train.std(0) + 1e-12
    out_mean = Y_train.mean(0)
    out_std = Y_train.std(0) + 1e-12

    X_train_n = (X_train - in_mean) / in_std
    Y_train_n = (Y_train - out_mean) / out_std
    X_val_n = (X_val - in_mean) / in_std

    mlp = MlpHead(input_dim=X.shape[1], output_dim=Y.shape[1],
                  hidden_dims=hidden_dims, activation=activation)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    X_t = torch.as_tensor(X_train_n, dtype=F64)
    Y_t = torch.as_tensor(Y_train_n, dtype=F64)
    X_v = torch.as_tensor(X_val_n, dtype=F64)

    best_val_mape = float("inf")
    best_state = None      # state_dict snapshot at the best-MAPE epoch
    n_train = X_t.shape[0]
    t_epoch_0 = time.time()
    for epoch in range(epochs):
        if should_stop is not None and should_stop():
            from linac_gen.core.cancelled import OperationCancelled
            raise OperationCancelled(
                f"training cancelled at epoch {epoch}/{epochs}")
        mlp.train()
        idx = torch.randperm(n_train)
        epoch_loss_sum = 0.0
        epoch_n = 0
        for i in range(0, n_train, batch_size):
            sl = idx[i:i + batch_size]
            pred = mlp(X_t[sl])
            loss = ((pred - Y_t[sl]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.detach()) * sl.shape[0]
            epoch_n += int(sl.shape[0])
        mlp.eval()
        with torch.no_grad():
            v_pred_n = mlp(X_v).numpy()
        v_pred = v_pred_n * out_std + out_mean
        denom = np.abs(Y_val) + 1e-12
        per_entry_abs = np.abs(v_pred - Y_val) / denom   # (n_val, 36)
        val_mape = float(per_entry_abs.mean())
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            # Snapshot the weights that achieved this MAPE so the model
            # returned (and persisted) is the one the reported accuracy
            # describes.  Previously the FINAL epoch's weights were
            # returned alongside the BEST epoch's MAPE (fixed 2026-07-10).
            best_state = {k: v.detach().clone()
                          for k, v in mlp.state_dict().items()}
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch+1:4d}/{epochs}  val MAPE = {val_mape:.4e}",
                  flush=True)
        if progress_callback:
            # Per-entry val MAPE reshaped (6, 6) -- useful as a
            # heatmap of which matrix entries are hardest.
            per_entry_mean = per_entry_abs.mean(axis=0).reshape(6, 6)
            progress_callback({
                "stage": "epoch",
                "epoch": epoch + 1, "total": epochs,
                "train_loss": float(epoch_loss_sum / max(epoch_n, 1)),
                "val_mape": val_mape,
                "best_val_mape": best_val_mape,
                "per_entry_val_mape": per_entry_mean,
                "elapsed_s": time.time() - t_epoch_0,
            })

    # Restore the best-epoch weights so the returned/persisted model is
    # the one best_val_mape describes (not the last epoch's).
    if best_state is not None:
        mlp.load_state_dict(best_state)
        mlp.eval()

    norm = {
        "input_mean": [float(v) for v in in_mean],
        "input_std":  [float(v) for v in in_std],
        "output_mean": [float(v) for v in out_mean],
        "output_std":  [float(v) for v in out_std],
    }
    return mlp, norm, best_val_mape


# ---------------------------------------------------------------------------
def save_surrogate(mlp: MlpHead, metadata: SurrogateMetadata,
                   dir_path: str | Path) -> Path:
    """Persist weights + metadata.json under ``dir_path`` (created if needed)."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(mlp.state_dict(), dir_path / "weights.pt")
    with open(dir_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.to_json(), f, indent=2)
    return dir_path


def load_surrogate(dir_path: str | Path) -> tuple[MlpHead, SurrogateMetadata]:
    """Load ``(mlp, metadata)`` from a directory previously written by
    :func:`save_surrogate`."""
    dir_path = Path(dir_path)
    with open(dir_path / "metadata.json", encoding="utf-8") as f:
        meta = SurrogateMetadata.from_json(json.load(f))
    # Kinematic admissibility: the weights encode transfer maps sampled at
    # a specific species mass.  The cache key (lattice hash + element name)
    # does not include it, so a species-table change (e.g. the 2026-07-18
    # H- mass fix, 938.272 -> 939.294 MeV) would otherwise reuse stale
    # weights silently.
    import warnings as _warnings
    if meta.species_mass_mev > 0 and meta.species_name:
        from linac_gen.core import particle as _pt
        table = {p.name: p.mass
                 for p in (_pt.PROTON, _pt.H_MINUS, _pt.DEUTERON)}
        cur = table.get(meta.species_name)
        if cur is None:
            # Outside the kinematics table the mass cannot be checked
            # at all — say so instead of loading silently (an absurd
            # tagged mass used to pass without a word).
            _warnings.warn(
                f"surrogate '{meta.element_key}' is tagged with species "
                f"{meta.species_name!r} (mass {meta.species_mass_mev:.4f}"
                " MeV), which is not in the kinematics table "
                f"({sorted(table)}) — staleness is UNVERIFIABLE; make "
                "sure the weights match the beam you are tracking.",
                stacklevel=2)
        elif abs(cur - meta.species_mass_mev) > 1e-3:
            _warnings.warn(
                f"surrogate '{meta.element_key}' was trained with "
                f"{meta.species_name} mass {meta.species_mass_mev:.4f} MeV "
                f"but the current species table has {cur:.4f} MeV — its "
                "transfer maps encode STALE kinematics; retrain before "
                "trusting results.", stacklevel=2)
    else:
        _warnings.warn(
            f"surrogate '{meta.element_key}' carries no species-mass "
            "metadata (legacy weights from before the 2026-07-18 H- mass "
            "fix, or a species-less training ref) — kinematic validity "
            "unknown; retraining is recommended.",
            stacklevel=2)
    arch = meta.architecture
    mlp = MlpHead(
        input_dim=int(arch["input_dim"]),
        output_dim=int(arch["output_dim"]),
        hidden_dims=arch["hidden_dims"],
        activation=str(arch["activation"]),
    )
    mlp.load_state_dict(torch.load(dir_path / "weights.pt",
                                    weights_only=True))
    mlp.eval()
    # Content identity of the ACTUAL weights bytes, recomputed at every
    # load (never trusted from metadata.json): flows into the HDF5
    # provenance manifest identically for CLI, GUI and Python-API
    # loads — a swapped/corrupted weights.pt with unchanged metadata
    # is detectable post-hoc.
    import hashlib as _hashlib
    meta.weights_sha256 = _hashlib.sha256(
        (dir_path / "weights.pt").read_bytes()).hexdigest()
    return mlp, meta


def find_cached_surrogate(
    dir_path: str | Path,
) -> tuple[MlpHead, SurrogateMetadata] | None:
    """Return loaded ``(mlp, metadata)`` if ``dir_path`` holds a complete
    surrogate (both ``weights.pt`` and ``metadata.json`` readable), else
    ``None``.  Used by GUI/CLI flows that want to short-circuit training
    when a previously-saved surrogate is already on disk.
    """
    dir_path = Path(dir_path)
    if not (dir_path / "weights.pt").is_file():
        return None
    if not (dir_path / "metadata.json").is_file():
        return None
    try:
        return load_surrogate(dir_path)
    except Exception:
        return None


def discover_cached_surrogates(
    weights_root: str | Path,
    element_names: Sequence[str] | None = None,
) -> dict[str, tuple[MlpHead, SurrogateMetadata, Path]]:
    """Scan ``weights_root`` for surrogate subdirectories.

    Returns ``{element_key: (mlp, metadata, dir_path)}`` for every
    subdirectory whose contents load successfully via
    :func:`find_cached_surrogate`.  If ``element_names`` is supplied,
    only subdirectories whose name is in that set are loaded (useful
    for restricting discovery to elements present in the current
    lattice).
    """
    root = Path(weights_root)
    if not root.is_dir():
        return {}
    out: dict[str, tuple[MlpHead, SurrogateMetadata, Path]] = {}
    allow: set[str] | None = (
        set(element_names) if element_names is not None else None)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if allow is not None and child.name not in allow:
            continue
        loaded = find_cached_surrogate(child)
        if loaded is None:
            continue
        mlp, meta = loaded
        out[meta.element_key] = (mlp, meta, child)
    return out


# ---------------------------------------------------------------------------
def train_surrogate_for_element(
    element: FieldMapElement,
    ref_template,
    *,
    n_samples: int,
    ref_w_kin_range: tuple[float, float],
    param_ranges: dict[str, tuple[float, float]],
    out_dir: str | Path,
    hidden_dims: Sequence[int] = (128, 128, 128),
    activation: str = "silu",
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_frac: float = 0.2,
    seed: int = 42,
    lattice_hash: str = "unknown",
    element_key: str | None = None,
    verbose: bool = False,
    n_workers: int | None = None,
    progress_callback=None,
    should_stop=None,
) -> tuple[MlpHead, SurrogateMetadata]:
    """End-to-end orchestrator: data gen + train + save.

    Returns the trained ``(mlp, metadata)`` and persists them under
    ``out_dir``.

    Passing ``n_workers >= 2`` parallelises the (expensive) RK4 data
    generation across a process Pool; output is bit-identical to the
    serial path for a fixed seed.  See
    :func:`generate_training_data` for details.

    The optional ``progress_callback`` receives dicts at two
    stages.  Use the ``"stage"`` key to dispatch:

    * ``{"stage": "data_gen", "done", "total", "elapsed_s"}``
    * ``{"stage": "epoch", "epoch", "total", "train_loss",
       "val_mape", "best_val_mape", "per_entry_val_mape" (6x6),
       "elapsed_s"}``

    ``should_stop`` (optional zero-arg callable) is polled per sample /
    per epoch; when it returns True an
    :class:`~linac_gen.core.cancelled.OperationCancelled` is raised.
    Nothing is persisted on cancellation — ``save_surrogate`` only runs
    after both stages complete, so no partial weights can be cached.
    """
    X, Y, info = generate_training_data(
        element=element,
        ref_template=ref_template,
        n_samples=n_samples,
        ref_w_kin_range=ref_w_kin_range,
        param_ranges=param_ranges,
        seed=seed,
        verbose=verbose,
        n_workers=n_workers,
        progress_callback=progress_callback,
        should_stop=should_stop,
    )
    mlp, norm, val_mape = train_surrogate(
        X, Y,
        hidden_dims=hidden_dims,
        activation=activation,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        val_frac=val_frac,
        seed=seed,
        verbose=verbose,
        progress_callback=progress_callback,
        should_stop=should_stop,
    )
    metadata = SurrogateMetadata(
        element_key=element_key or element.name,
        element_class=type(element).__name__,
        architecture={
            "input_dim": int(X.shape[1]),
            "output_dim": int(Y.shape[1]),
            "hidden_dims": [int(h) for h in hidden_dims],
            "activation": str(activation),
            "param_names": info["param_names"],
        },
        scope=Scope(
            input_names=info["input_names"],
            input_lo=np.asarray(info["input_lo"], dtype=np.float64),
            input_hi=np.asarray(info["input_hi"], dtype=np.float64),
        ),
        input_norm={"mean": norm["input_mean"], "std": norm["input_std"]},
        output_norm={"mean": norm["output_mean"], "std": norm["output_std"]},
        training_seed=int(seed),
        n_samples=int(n_samples),
        epochs=int(epochs),
        val_mape=float(val_mape),
        helix_commit_sha=_git_head_sha(),
        lattice_hash=str(lattice_hash),
        species_name=str(getattr(getattr(ref_template, "species", None),
                                 "name", "") or ""),
        species_mass_mev=float(getattr(getattr(ref_template, "species", None),
                                       "mass", 0.0) or 0.0),
        created_iso=datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
    )
    save_surrogate(mlp, metadata, out_dir)
    return mlp, metadata
