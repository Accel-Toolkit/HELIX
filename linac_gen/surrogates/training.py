"""Training pipeline for HELIX envelope-mode surrogates.

For a chosen :class:`FieldMapElement` instance, this module:

1. Generates ``(X, Y)`` samples via Latin Hypercube over ref
   kinematics + element parameter ranges.  Each sample's ``Y`` is
   the flattened 6x6 transfer matrix from the wrapped element's
   :meth:`fitted_matrix`.
2. Trains an :class:`MlpHead` on the samples (FP64 CPU, smooth
   activation, Adam with a learning rate that decays along a half cosine,
   keeping the epoch with the lowest validation error).
3. Persists the trained weights + a :class:`SurrogateMetadata`
   JSON manifest for runtime use.

CLI entry point lives in ``linac_gen/surrogates/cli.py`` (M5);
this module exposes the building blocks.
"""
from __future__ import annotations

import copy
import json
import math
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
    design_copy,
    design_reference,
    settings_of,
    structure_record,
)
from linac_gen.parallel.threads import sync_torch as _sync_torch

# a run that loads torch here keeps its CPU work on one thread
_sync_torch()


# ---------------------------------------------------------------------------
# Multiprocessing worker plumbing.  These functions live at module scope
# (not as closures) so they're picklable on macOS spawn-mode Pools.  Each
# worker keeps a deep-copy of the element + ref_template + param_names in
# a module-level dict, populated once by the Pool's initializer; per-task
# calls only ship a small (idx, w_kin, params) tuple.
# ---------------------------------------------------------------------------
_WORKER_STATE: dict = {}


def _worker_init(element, ref_template, param_names,
                 counters=None, id_counter=None, threads: int = 1) -> None:
    """Pool initializer: stash a deep-copied element per worker process.

    For progress reporting each worker claims a unique slot ``wid`` in the
    shared, LOCK-FREE ``counters`` array (the id is handed out once via the
    locked ``id_counter`` at startup) and writes its own running sample
    count into that slot.  Single-writer-per-slot means zero lock
    contention on the hot per-sample path; the parent sums the array for
    total progress.
    """
    # this worker's share of the HELIX thread setting
    from linac_gen.parallel.threads import set_threads
    set_threads(threads)
    _WORKER_STATE["element"] = copy.deepcopy(element)
    _WORKER_STATE["ref_template"] = ref_template
    _WORKER_STATE["param_names"] = list(param_names)
    _WORKER_STATE["counters"] = counters
    _WORKER_STATE["local"] = 0
    if counters is not None and id_counter is not None:
        with id_counter.get_lock():
            _WORKER_STATE["wid"] = int(id_counter.value)
            id_counter.value += 1


def _at_element_frequency(element, ref) -> None:
    """Put ``ref`` on the map's own clock before asking for its matrix.  At
    another frequency ``fitted_matrix`` makes the jump itself and the target
    would hold the factor, which belongs to the jump: the envelope applies it at
    its seam, ``SurrogateFieldMap.fitted_matrix`` when asked at another clock."""
    propagate = getattr(element, "_propagate_frequency_to_ref", None)
    if propagate is not None:
        propagate(ref)


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
    _at_element_frequency(el, ref)
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
    """The HEAD of the HELIX checkout this module runs from, for reproducibility, or 'unknown'
    (an installed copy or an export, with no git history).  Asked from the working directory,
    a training run started elsewhere recorded another repository's HEAD, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or str(Path(__file__).resolve().parent),
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
        unmutated, with its error knobs at zero where a query folds them
        into the inputs (``base.design_copy``); the reference template
        loses its SET_BEAM_PHASE_ERROR shifts likewise.
    ref_template
        Reference particle to clone for each sample; ``w_kin`` is varied,
        and a cavity at another frequency gets its own (species etc. stay
        fixed).
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

    # the targets are the map of the element as its weights hold it: its error knobs and
    # the reference's RF shifts are folded into the inputs at each query (base.design_copy,
    # as the accuracy check takes its native maps)
    element, ref_template = design_copy(element), design_reference(ref_template)
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
            _at_element_frequency(work_elem, ref)
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
        from linac_gen.parallel.threads import worker_threads
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(element, ref_template, param_names,
                      counters, id_counter, worker_threads(n_workers)),
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


#: an output whose values spread over the training data by less than this is normalised by it
#: instead of by its own spread.  The structurally zero entries of a map (the transverse to
#: longitudinal couplings of a symmetric cavity) hold only round-off, spread about 1e-15 to
#: 2e-12 in the example decks' weights: scaled by that spread they weighed as much as a real entry,
#: held 99.8 % of a cavity's validation loss and kept the real entries from converging.  Such an
#: output is also left out of the validation MAPE (``mape_outputs``), whose relative error
#: there is one of round-off.
OUTPUT_STD_FLOOR = 1e-9
#: the learning rate falls along a half cosine from its starting value to this part of it at
#: the last epoch (``epoch_learning_rate``)
LR_FINAL_FRACTION = 0.01


def epoch_learning_rate(lr: float, epoch: int, epochs: int) -> float:
    """The learning rate of ``epoch`` (from 0) of ``epochs``: ``lr`` at the first, falling along
    a half cosine to ``LR_FINAL_FRACTION`` of it at the last.  At a constant rate the fit
    stopped improving where its steps were too large for the last digits."""
    return lr * (LR_FINAL_FRACTION + (1.0 - LR_FINAL_FRACTION) * 0.5
                 * (1.0 + math.cos(math.pi * epoch / max(epochs - 1, 1))))


def mape_outputs(output_std, output_mean=None) -> np.ndarray:
    """Which outputs the validation MAPE is over: those whose normalising spread
    (``train_surrogate``) is above ``OUTPUT_STD_FLOOR``; the others are entries of the map that
    are zero or constant up to round-off, where a relative error measures the round-off.  If
    none varies, the constant ones that are not zero (``output_mean`` above the floor), and if
    there are none of those either, all of them."""
    on = np.asarray(output_std, dtype=np.float64) > OUTPUT_STD_FLOOR
    if not on.any() and output_mean is not None:
        on = np.abs(np.asarray(output_mean, dtype=np.float64)) > OUTPUT_STD_FLOOR
    return on if on.any() else np.ones_like(on)


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
    report: dict | None = None,
) -> tuple[MlpHead, dict, float]:
    """Train an MLP on (X, Y) for envelope-mode 6x6 matrix prediction.

    Each input and output is normalised by its mean and spread over the training split, an
    output's spread floored at ``OUTPUT_STD_FLOOR``.  Adam minimises the mean squared error of
    the normalised outputs, its learning rate falling from ``lr`` along a half cosine
    (``epoch_learning_rate``), and the epoch kept is the one with the lowest validation error
    in the same units.  The validation MAPE is over the outputs ``mape_outputs`` names.

    Returns ``(mlp, normalisation, val_mape)``, the validation MAPE of the kept epoch.
    ``report``, if given, receives how the training ran, as a weights file records it
    (``SurrogateMetadata.train_recipe``).
    """
    if int(epochs) < 1:
        raise ValueError(f"a training needs at least one epoch, not {epochs}")
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
    out_std = np.maximum(Y_train.std(0) + 1e-12, OUTPUT_STD_FLOOR)
    mape_on = mape_outputs(out_std, out_mean)

    X_train_n = (X_train - in_mean) / in_std
    Y_train_n = (Y_train - out_mean) / out_std
    X_val_n = (X_val - in_mean) / in_std

    mlp = MlpHead(input_dim=X.shape[1], output_dim=Y.shape[1],
                  hidden_dims=hidden_dims, activation=activation)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    X_t = torch.as_tensor(X_train_n, dtype=F64)
    Y_t = torch.as_tensor(Y_train_n, dtype=F64)
    X_v = torch.as_tensor(X_val_n, dtype=F64)
    Y_v = torch.as_tensor((Y_val - out_mean) / out_std, dtype=F64)

    best_val_mse = float("inf")
    best_val_mape = float("inf")
    best_epoch = 0         # from 1; 0 = none kept yet
    best_state = None      # state_dict snapshot at the kept epoch
    n_train = X_t.shape[0]
    t_epoch_0 = time.time()
    for epoch in range(epochs):
        if should_stop is not None and should_stop():
            from linac_gen.core.cancelled import OperationCancelled
            raise OperationCancelled(
                f"training cancelled at epoch {epoch}/{epochs}")
        rate = epoch_learning_rate(lr, epoch, epochs)
        for group in opt.param_groups:
            group["lr"] = rate
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
            v_pred_n = mlp(X_v)
            val_mse = float(((v_pred_n - Y_v) ** 2).mean())
        v_pred = v_pred_n.numpy() * out_std + out_mean
        denom = np.abs(Y_val) + 1e-12
        per_entry_abs = np.abs(v_pred - Y_val) / denom   # (n_val, n_out)
        val_mape = float(per_entry_abs[:, mape_on].mean())
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mape = val_mape
            best_epoch = epoch + 1
            # the weights of this epoch, which the returned (and saved) model is and the
            # reported accuracy describes
            best_state = {k: v.detach().clone()
                          for k, v in mlp.state_dict().items()}
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch+1:4d}/{epochs}  val MSE = {val_mse:.4e}  "
                  f"val MAPE = {val_mape:.4e}", flush=True)
        if progress_callback:
            # Per-entry val MAPE reshaped (6, 6) -- a heatmap of which matrix entries are
            # hardest; 0 at the entries the MAPE leaves out (``mape_outputs``)
            per_entry_mean = np.where(mape_on, per_entry_abs.mean(axis=0), 0.0)
            if per_entry_mean.size == 36:
                per_entry_mean = per_entry_mean.reshape(6, 6)
            progress_callback({
                "stage": "epoch",
                "epoch": epoch + 1, "total": epochs,
                "lr": rate,
                "train_loss": float(epoch_loss_sum / max(epoch_n, 1)),
                "val_mse": val_mse,
                "best_val_mse": best_val_mse,
                "val_mape": val_mape,
                # of the epoch kept so far (the lowest val_mse), so not monotone
                "best_val_mape": best_val_mape,
                "best_epoch": best_epoch,
                "per_entry_val_mape": per_entry_mean,
                "elapsed_s": time.time() - t_epoch_0,
            })

    # Restore the kept epoch's weights so the returned/persisted model is
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
    if report is not None:
        report.update({
            "optimizer": "adam",
            "loss": "mean squared error of the normalised outputs",
            "lr": float(lr),
            "lr_schedule": "cosine",
            "lr_final": float(epoch_learning_rate(lr, max(epochs - 1, 0), epochs)),
            "batch_size": int(batch_size),
            "val_frac": float(val_frac),
            "output_std_floor": OUTPUT_STD_FLOOR,
            "kept_epoch": int(best_epoch),
            "kept_by": "lowest validation mean squared error",
            "val_mse": float(best_val_mse),
            "val_mape_outputs": int(np.count_nonzero(mape_on)),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        })
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


#: how far from 1 the longitudinal determinant of a map may be before its weights count
#: as holding a frequency jump.  A stale file's factor is its cavity's frequency over its
#: template's (162.5 MHz in the Surrogates tab), so a 201.25 MHz cavity reads 1.24; the
#: files the development tree keeps that record no clock are within 1e-4 of 1.
STALE_DET_TOLERANCE = 0.05


def _stale_factor(mlp: MlpHead, meta: SurrogateMetadata) -> float | None:
    """The longitudinal determinant of a map, at the training's mean input, if it says
    the targets hold a frequency jump, else None.

    On its own clock a map's is 1, and every file the development tree keeps that records no
    clock is within 1e-4 of it; a target taken at another clock reads f₂/f₁.  HELIX's training
    normalises each input by the mean and spread of its training data, so the mean input is
    normalised zero; a file normalised otherwise is judged at another point.  The scope's
    midpoint would not do: the scope holds β in [0, 1] and γ in [1, 1000].
    """
    out_mean = torch.as_tensor(meta.output_norm["mean"], dtype=torch.float64)
    out_std = torch.as_tensor(meta.output_norm["std"], dtype=torch.float64)
    with torch.no_grad():
        y = mlp(torch.zeros(int(meta.architecture["input_dim"]), dtype=torch.float64))
    M = (y * out_std + out_mean).numpy().reshape(6, 6)
    det = float(np.linalg.det(M[4:6, 4:6]))
    return det if abs(det - 1.0) > STALE_DET_TOLERANCE else None


def _file_notice(message: str) -> None:
    """``warnings.warn(message, stacklevel=2)`` from :func:`load_surrogate`, on every load.

    Python shows a warning once per line it is attributed to, and every load of a weight
    file is attributed to the same line, so only a process's first load of a stale or
    species-less file said so: the GUI's status bar heard it once per session.  The
    caller's warning registry forgets this text at this line before each load, and
    nothing else, so the ``default`` action shows it again while ``ignore``, ``error``,
    ``once``, ``module`` and ``always`` keep their meaning.  The source line is found by
    file name, as for any warning: handing over the caller's globals made Python ask its
    loader, which fails for ``python -m`` and ``python -c``."""
    import sys
    import warnings
    caller = sys._getframe(2)           # who called load_surrogate
    registry = caller.f_globals.setdefault("__warningregistry__", {})
    registry.pop((message, UserWarning, caller.f_lineno), None)
    warnings.warn_explicit(message, UserWarning, caller.f_code.co_filename, caller.f_lineno,
                           module=caller.f_globals.get("__name__", "<unknown>"),
                           registry=registry)


def _no_notice(message: str) -> None:
    """For a caller that has already told the user about this file."""


def load_surrogate(dir_path: str | Path, *,
                   notices: bool = True) -> tuple[MlpHead, SurrogateMetadata]:
    """Load ``(mlp, metadata)`` from a directory previously written by
    :func:`save_surrogate`.  ``notices=False`` leaves out what the load says about a stale
    or species-less file, for a caller that has said it already."""
    say = _file_notice if notices else _no_notice
    dir_path = Path(dir_path)
    with open(dir_path / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    meta = SurrogateMetadata.from_json(data)
    # a file from before the training phase was recorded counts as trained at 0; its map
    # is checked against the native element at its first use (SurrogateFieldMap)
    meta.phase_recorded = "train_phi_s_deg" in data
    # a file's weights are checked against the native element once, at their first use
    # (SurrogateFieldMap._accuracy_refusal): they may predate today's physics
    meta.accuracy_due = True
    # Kinematic admissibility: the weights encode transfer maps sampled at
    # a specific species mass.  The cache key (lattice hash + element name)
    # does not include it, so a species-table change (e.g. the 2026-07-18
    # H- mass fix, 938.272 -> 939.294 MeV) would otherwise reuse stale
    # weights silently.
    if meta.species_mass_mev > 0 and meta.species_name:
        from linac_gen.core import particle as _pt
        table = {p.name: p.mass
                 for p in (_pt.PROTON, _pt.H_MINUS, _pt.DEUTERON)}
        cur = table.get(meta.species_name)
        if cur is None:
            # Outside the kinematics table the mass cannot be checked
            # at all — say so instead of loading silently (an absurd
            # tagged mass used to pass without a word).
            say(
                f"surrogate '{meta.element_key}' is tagged with species "
                f"{meta.species_name!r} (mass {meta.species_mass_mev:.4f}"
                " MeV), which is not in the kinematics table "
                f"({sorted(table)}) — staleness is UNVERIFIABLE; make "
                "sure the weights match the beam you are tracking.")
        elif abs(cur - meta.species_mass_mev) > 1e-3:
            say(
                f"surrogate '{meta.element_key}' was trained with "
                f"{meta.species_name} mass {meta.species_mass_mev:.4f} MeV "
                f"but the current species table has {cur:.4f} MeV — its "
                "transfer maps encode STALE kinematics; retrain before "
                "trusting results.")
    else:
        if not meta.species_name:
            # the Surrogates tab and the CLI trained H⁻ only until files recorded
            # their species, and the Python API has recorded it since
            meta.species_assumed = "H-"
        say(
            f"surrogate '{meta.element_key}' carries no species-mass "
            "metadata (legacy weights from before the 2026-07-18 H- mass "
            "fix, or a species-less training ref) — kinematic validity "
            "unknown; retraining is recommended."
            + (" It counts as trained for H⁻ and answers for no other species."
               if meta.species_assumed else ""))
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
    # A file that records no clock: its cavity's targets may hold the jump's factor.
    meta.legacy_weights = not meta.train_frequency_mhz > 0
    if meta.legacy_weights:
        meta.stale_factor = _stale_factor(mlp, meta)
        if meta.stale_factor is not None:
            say(
                f"surrogate '{meta.element_key}' is stale: the longitudinal determinant "
                f"of its map is {meta.stale_factor:.3g}, not 1, as when a cavity is trained at "
                "another clock than its own and its targets hold the ratio.  The native "
                "element tracks instead; retrain it.")
    # Content identity of the ACTUAL weights bytes, recomputed at every
    # load (never trusted from metadata.json): flows into the HDF5
    # provenance manifest identically for CLI, GUI and Python-API
    # loads — a swapped/corrupted weights.pt with unchanged metadata
    # is detectable post-hoc.
    import hashlib as _hashlib
    meta.weights_sha256 = _hashlib.sha256(
        (dir_path / "weights.pt").read_bytes()).hexdigest()
    return mlp, meta


def weights_folder(lattice_hash: str, species: str | None = "H-") -> Path:
    """Where the weights of a lattice's maps for one species live, relative to the working
    directory: ``linac_gen/surrogates/weights/<hash16>``, with ``_<species>`` for another
    species than H⁻.  A map holds its species and a lattice's hash does not, so another
    species on the lattice neither reads nor overwrites them; H⁻ keeps the plain folder,
    which is where training always wrote."""
    name = lattice_hash[:16] if lattice_hash != "unknown" else "unknown"
    if species and species != "H-":
        name += f"_{species}"
    return Path("linac_gen/surrogates/weights") / name


def find_cached_surrogate(
    dir_path: str | Path, *, notices: bool = True,
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
        return load_surrogate(dir_path, notices=notices)
    except Exception:
        return None


def discover_cached_surrogates(
    weights_root: str | Path,
    element_names: Sequence[str] | None = None,
    *, notices: bool = True,
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
        loaded = find_cached_surrogate(child, notices=notices)
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
    * ``{"stage": "epoch", "epoch", "total", "lr", "train_loss",
       "val_mse", "best_val_mse", "val_mape", "best_val_mape" (of the kept
       epoch), "best_epoch", "per_entry_val_mape" (6x6), "elapsed_s"}``

    ``should_stop`` (optional zero-arg callable) is polled per sample /
    per epoch; when it returns True an
    :class:`~linac_gen.core.cancelled.OperationCancelled` is raised.
    Nothing is persisted on cancellation — ``save_surrogate`` only runs
    after both stages complete, so no partial weights can be cached.
    """
    # what the targets are taken at, for whoever asks at another clock or phase: the
    # map's own clock, and the reference phase on it (a template on another clock has
    # it moved there as a FREQ card moves it)
    clock = ref_template.copy()
    _at_element_frequency(element, clock)
    # the settings the targets are taken at, which the weights hold fixed, and what the element
    # is: read before the data, as the GUI leaves the element open to edits during a training
    held = settings_of(element, set(param_ranges))
    structure = structure_record(element)
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
    recipe: dict = {}
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
        report=recipe,
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
        train_frequency_mhz=float(getattr(clock, "frequency", 0.0) or 0.0),
        train_phi_s_deg=float(getattr(clock, "phi_s", 0.0) or 0.0),
        created_iso=datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        # the settings the targets were taken at and the weights hold fixed: a query on an
        # element set otherwise goes native (SurrogateFieldMap._settings_reason)
        train_settings=held,
        # how the network was fitted, and what the element was (its phase reference p_flag
        # among it): a record, which nothing checks
        train_recipe=recipe,
        train_structure=structure,
    )
    save_surrogate(mlp, metadata, out_dir)
    return mlp, metadata
