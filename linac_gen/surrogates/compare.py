"""Envelope-mode comparison: baseline (RK4) vs. surrogate-enabled.

M4 wires :func:`compare_envelope` to actually engage / disengage the
registry between the two runs.  Returns a :class:`CompareReport`
dataclass with the σ curves, end-of-line table, wall-clock, scope-
compliance flag, and helpers for the CLI / GUI (M5, M6).

Reusable from CLI and GUI; until slice-aware surrogates exist
(SurrogateFieldMap currently delegates ``fitted_matrix_slice`` to
the wrapped element), the diff will be near-zero — the framework is
ready for the speedup that lands with the next training milestone.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from linac_gen.surrogates import registry
from linac_gen.surrogates.base import SurrogateFieldMap

# Both compare paths swap the PROCESS-GLOBAL surrogate registry for the
# duration of their two runs.  Any solve running concurrently (toolbar
# envelope/MP, matching, a scan) would silently compute with the wrong
# registry.  Callers must not run compares next to other solves (the
# GUI excludes them via its running flag); this lock is the in-library
# backstop so two compares can never interleave their swaps.
_COMPARE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
@dataclass
class CompareReport:
    """Structured envelope-vs-envelope diff report."""
    s: np.ndarray                          # arc length (mm)
    sigma_baseline: dict[str, np.ndarray]  # {"x", "y", "phi", "w"}
    sigma_surrogate: dict[str, np.ndarray]
    end_of_line: dict[str, float]
    wall_baseline_s: float
    wall_surrogate_s: float
    scope_ok: bool                         # True if every query was in scope
    surrogate_names: list[str]             # names of engaged surrogates
    notes: list[str] = field(default_factory=list)

    def speedup(self) -> float:
        return (self.wall_baseline_s / self.wall_surrogate_s
                if self.wall_surrogate_s > 0 else float("nan"))

    def worst_rel_diff(self) -> float:
        return max(
            self.end_of_line.get("rel_diff_x", 0.0),
            self.end_of_line.get("rel_diff_y", 0.0),
            self.end_of_line.get("rel_diff_phi", 0.0),
            self.end_of_line.get("rel_diff_w", 0.0),
        )

    def summary_text(self) -> str:
        """Multi-line, CLI-friendly summary."""
        lines = [
            f"Surrogates engaged: {len(self.surrogate_names)} "
            f"({', '.join(self.surrogate_names) or 'none'})",
            f"Wall-clock: baseline {self.wall_baseline_s:.3f} s  "
            f"surrogate {self.wall_surrogate_s:.3f} s  "
            f"speedup {self.speedup():.2f}x",
            "End-of-line sigma moments:",
        ]
        for k, lbl, unit in [("x", "sigma_x", "mm"),
                             ("y", "sigma_y", "mm"),
                             ("phi", "sigma_phi", "deg"),
                             ("w", "sigma_w", "MeV")]:
            base = self.end_of_line.get(f"sigma_{k}_base", float('nan'))
            srr = self.end_of_line.get(f"sigma_{k}_surr", float('nan'))
            rel = self.end_of_line.get(f"rel_diff_{k}", float('nan'))
            lines.append(f"  {lbl:>10s}  baseline={base:11.4e}  "
                         f"surrogate={srr:11.4e}  rel.diff={rel:.2e}  {unit}")
        lines.append(f"Worst rel.diff: {self.worst_rel_diff():.2e}")
        lines.append(f"Scope OK: {self.scope_ok}")
        if self.notes:
            lines.append("Notes: " + "; ".join(self.notes))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def _envelope_run(lattice, ref, init_twiss: dict, current: float,
                  should_abort=None):
    from linac_gen.tracking.envelope import EnvelopeSolver
    solver = EnvelopeSolver(lattice, ref, init_twiss, current=current,
                            should_abort=should_abort)
    res = solver.run()
    if getattr(solver, "aborted", False):
        # The solver's abort contract is "return partial results" — a
        # report diffing two differently-truncated runs would be
        # meaningless, so cancellation must raise, never return.
        from linac_gen.core.cancelled import OperationCancelled
        raise OperationCancelled("surrogate compare cancelled")
    return res


def _extract(res) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return np.asarray(res.s), {
        "x": np.asarray(res.sigma_x),
        "y": np.asarray(res.sigma_y),
        "phi": np.asarray(res.sigma_phi),
        "w": np.asarray(res.sigma_w),
    }


def compare_envelope(
    lattice,
    ref,
    init_twiss: dict,
    current: float = 0.0,
    surrogates: Iterable[SurrogateFieldMap] | None = None,
    should_abort=None,
) -> CompareReport:
    """Run the envelope solver twice and produce a diff report.

    Baseline: registry temporarily cleared, RK4 throughout.
    Surrogate: surrogates from ``surrogates`` arg (or whatever is
    currently registered if ``surrogates=None``) registered and the
    envelope hook (M3) engages.

    The whole-process registry state is preserved across the call.
    ``should_abort`` (optional zero-arg callable) is polled per element
    inside both solver runs; cancellation raises
    :class:`~linac_gen.core.cancelled.OperationCancelled`.
    """
    with _COMPARE_LOCK:
        saved_registry = dict(registry._REGISTRY)
        saved_by_name = dict(registry._BY_NAME)

        try:
            # ---- baseline (no surrogates) ------------------------------
            registry.clear()
            t0 = time.time()
            res_b = _envelope_run(lattice, ref, init_twiss, current,
                                  should_abort)
            wall_b = time.time() - t0

            # ---- surrogate-enabled --------------------------------------
            registry.clear()
            if surrogates is not None:
                for s in surrogates:
                    registry.register(s)
            else:
                registry._REGISTRY.update(saved_registry)
                registry._BY_NAME.update(saved_by_name)
            engaged_names = [k[1] for k in registry.list_registered()]

            t0 = time.time()
            res_s = _envelope_run(lattice, ref, init_twiss, current,
                                  should_abort)
            wall_s = time.time() - t0
        finally:
            # Restore the original registry state regardless of outcome.
            registry._REGISTRY.clear()
            registry._BY_NAME.clear()
            registry._REGISTRY.update(saved_registry)
            registry._BY_NAME.update(saved_by_name)

    s_b, sig_b = _extract(res_b)
    _, sig_s = _extract(res_s)
    end: dict[str, float] = {}
    for k in ("x", "y", "phi", "w"):
        b = float(sig_b[k][-1])
        srr = float(sig_s[k][-1])
        end[f"sigma_{k}_base"] = b
        end[f"sigma_{k}_surr"] = srr
        end[f"rel_diff_{k}"] = (abs(b - srr) / max(abs(b), 1e-12))

    notes: list[str] = []
    if not engaged_names:
        notes.append("no surrogate engaged (registry empty)")
    return CompareReport(
        s=s_b,
        sigma_baseline=sig_b,
        sigma_surrogate=sig_s,
        end_of_line=end,
        wall_baseline_s=wall_b,
        wall_surrogate_s=wall_s,
        scope_ok=True,    # admissibility is enforced inside the hook;
                          # any OOD silently falls back, so the run as
                          # a whole is always scope-correct.
        surrogate_names=engaged_names,
        notes=notes,
    )


# ---------------------------------------------------------------------------
def plot_compare_report(report: CompareReport, path: str | Path,
                         title: str | None = None) -> Path:
    """Write a 2x2 σ-curves PNG for visual inspection.

    Lazy-imports matplotlib so this module stays importable from
    environments without it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    s_m = report.s * 1e-3  # mm -> m for plot

    pairs = [
        ("x",   r"$\sigma_x$ [mm]"),
        ("y",   r"$\sigma_y$ [mm]"),
        ("phi", r"$\sigma_\phi$ [deg]"),
        ("w",   r"$\sigma_W$ [MeV]"),
    ]
    for ax, (k, label) in zip(axes.flat, pairs):
        ax.plot(s_m, report.sigma_baseline[k], "C0-", lw=1.4,
                label="baseline (RK4)")
        ax.plot(s_m, report.sigma_surrogate[k], "C3--", lw=1.1,
                label="surrogate")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("s [m]")
    axes[1, 1].set_xlabel("s [m]")

    base_title = (title
                  or "Envelope comparison: baseline vs surrogate-enabled")
    fig.suptitle(
        f"{base_title}\n"
        f"speedup={report.speedup():.2f}x   "
        f"worst rel.diff={report.worst_rel_diff():.2e}   "
        f"engaged: {len(report.surrogate_names)}",
        fontsize=11)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
@dataclass
class CompareMpReport:
    """Structured multi-particle baseline-vs-hybrid diff report (M7)."""
    s: np.ndarray                          # arc length (mm)
    sigma_baseline: dict[str, np.ndarray]  # {"x", "y", "z"}  mm / deg
    sigma_surrogate: dict[str, np.ndarray]
    emit_baseline: dict[str, np.ndarray]   # {"nx", "ny"} mm-mrad
    emit_surrogate: dict[str, np.ndarray]
    transmission_baseline: np.ndarray
    transmission_surrogate: np.ndarray
    end_of_line: dict[str, float]
    wall_baseline_s: float
    wall_surrogate_s: float
    surrogate_names: list[str]
    residual_n_steps: int
    notes: list[str] = field(default_factory=list)
    # Process-global registry flag captured at run time: the experimental
    # linear-matrix fast path changes what _mp_run measures, so a report
    # that omitted it was not reproducible from its own contents.
    fast_path_enabled: bool = False

    def speedup(self) -> float:
        return (self.wall_baseline_s / self.wall_surrogate_s
                if self.wall_surrogate_s > 0 else float("nan"))

    def worst_rel_diff(self) -> float:
        return max(
            self.end_of_line.get("rel_diff_sigma_x", 0.0),
            self.end_of_line.get("rel_diff_sigma_y", 0.0),
            self.end_of_line.get("rel_diff_sigma_z", 0.0),
            self.end_of_line.get("rel_diff_emit_nx", 0.0),
            self.end_of_line.get("rel_diff_emit_ny", 0.0),
        )

    def summary_text(self) -> str:
        lines = [
            f"Surrogates engaged: {len(self.surrogate_names)} "
            f"({', '.join(self.surrogate_names) or 'none'})",
            f"Residual RK4 substeps: {self.residual_n_steps}",
            f"Wall-clock: baseline {self.wall_baseline_s:.2f} s  "
            f"surrogate {self.wall_surrogate_s:.2f} s  "
            f"speedup {self.speedup():.2f}x",
            "End-of-line moments:",
        ]
        for k, lbl, unit in [
            ("sigma_x", "sigma_x", "mm"),
            ("sigma_y", "sigma_y", "mm"),
            ("sigma_z", "sigma_z", "mm"),
            ("emit_nx", "emit_nx", "pi.mm.mrad"),
            ("emit_ny", "emit_ny", "pi.mm.mrad"),
            ("trans",   "trans  ", "frac"),
        ]:
            base = self.end_of_line.get(f"{k}_base", float("nan"))
            srr = self.end_of_line.get(f"{k}_surr", float("nan"))
            rel = self.end_of_line.get(f"rel_diff_{k}", float("nan"))
            lines.append(f"  {lbl:>9s}  baseline={base:11.4e}  "
                         f"surrogate={srr:11.4e}  rel.diff={rel:.2e}  {unit}")
        lines.append(f"Worst rel.diff: {self.worst_rel_diff():.2e}")
        if self.notes:
            lines.append("Notes: " + "; ".join(self.notes))
        return "\n".join(lines)


def _mp_run(lattice, beam):
    """Run a single Simulation on a deep-copy of ``beam`` so the
    second run starts from the same initial state.
    """
    import copy as _copy
    from linac_gen.core.simulation import Simulation
    return Simulation(lattice, _copy.deepcopy(beam)).run()


def _extract_mp(res):
    """Pull (s, sigma_*, emit_*, transmission) from a DiagnosticRecorder."""
    return (
        np.asarray(res.s, dtype=np.float64),
        {
            "x": np.asarray(getattr(res, "sigma_x", []), dtype=np.float64),
            "y": np.asarray(getattr(res, "sigma_y", []), dtype=np.float64),
            "z": np.asarray(getattr(res, "sigma_phi", []), dtype=np.float64),
        },
        {
            "nx": np.asarray(getattr(res, "emit_nx", []), dtype=np.float64),
            "ny": np.asarray(getattr(res, "emit_ny", []), dtype=np.float64),
        },
        np.asarray(getattr(res, "transmission", []), dtype=np.float64),
    )


def compare_mp(
    lattice,
    beam,
    surrogates: Iterable[SurrogateFieldMap] | None = None,
    residual_n_steps: int = 15,
) -> CompareMpReport:
    """Run the multi-particle tracker twice and produce a diff report.

    Baseline: registry temporarily cleared AND MP-engagement off
    -> pure wrapped-RK4 throughout.
    Surrogate: surrogates registered, MP-engagement on, each
    surrogate's ``residual_n_steps`` set to the caller's value
    -> hybrid linear-anchor + RK4-residual path engaged in
    ``tracker._track_field_map``.

    Whole-process registry + MP-flag state is restored after the
    call regardless of outcome, so this is safe to invoke from a
    long-running GUI session.
    """
    with _COMPARE_LOCK:
        saved_registry = dict(registry._REGISTRY)
        saved_by_name = dict(registry._BY_NAME)
        saved_mp = registry.is_mp_enabled()
        # Captured (not toggled): the fast path applies to BOTH runs
        # identically, so the comparison stays internally consistent —
        # but any number in the report depends on it, so it must be
        # recorded rather than left as ambient process state.
        fast_path = registry.is_fast_path_enabled()
        saved_residuals: dict[int, int] = {}

        try:
            # ---- baseline (RK4) -----------------------------------------
            registry.clear()
            registry.set_mp_enabled(False)
            t0 = time.time()
            res_b = _mp_run(lattice, beam)
            wall_b = time.time() - t0

            # ---- surrogate-enabled hybrid -------------------------------
            registry.clear()
            if surrogates is not None:
                for s in surrogates:
                    # Remember the caller's prior residual setting so we
                    # can restore it after the run (this routine should
                    # not silently mutate the user's surrogate instances).
                    saved_residuals[id(s)] = int(s.residual_n_steps)
                    s.residual_n_steps = int(residual_n_steps)
                    registry.register(s)
            else:
                registry._REGISTRY.update(saved_registry)
                registry._BY_NAME.update(saved_by_name)
            registry.set_mp_enabled(True)
            engaged_names = [k[1] for k in registry.list_registered()]

            t0 = time.time()
            res_s = _mp_run(lattice, beam)
            wall_s = time.time() - t0
        finally:
            registry._REGISTRY.clear()
            registry._BY_NAME.clear()
            registry._REGISTRY.update(saved_registry)
            registry._BY_NAME.update(saved_by_name)
            registry.set_mp_enabled(saved_mp)
            if surrogates is not None:
                for s in surrogates:
                    if id(s) in saved_residuals:
                        s.residual_n_steps = saved_residuals[id(s)]

    s_b, sig_b, emit_b, trans_b = _extract_mp(res_b)
    _,   sig_s, emit_s, trans_s = _extract_mp(res_s)

    end: dict[str, float] = {}
    for k in ("x", "y", "z"):
        b = float(sig_b[k][-1]) if len(sig_b[k]) else 0.0
        srr = float(sig_s[k][-1]) if len(sig_s[k]) else 0.0
        end[f"sigma_{k}_base"] = b
        end[f"sigma_{k}_surr"] = srr
        end[f"rel_diff_sigma_{k}"] = (abs(b - srr) / max(abs(b), 1e-12))
    for k in ("nx", "ny"):
        b = float(emit_b[k][-1]) if len(emit_b[k]) else 0.0
        srr = float(emit_s[k][-1]) if len(emit_s[k]) else 0.0
        end[f"emit_{k}_base"] = b
        end[f"emit_{k}_surr"] = srr
        end[f"rel_diff_emit_{k}"] = (abs(b - srr) / max(abs(b), 1e-12))
    tb = float(trans_b[-1]) if len(trans_b) else 1.0
    ts = float(trans_s[-1]) if len(trans_s) else 1.0
    end["trans_base"] = tb
    end["trans_surr"] = ts
    end["rel_diff_trans"] = abs(tb - ts) / max(abs(tb), 1e-12)

    notes: list[str] = []
    if not engaged_names:
        notes.append("no surrogate engaged (registry empty)")
    if fast_path:
        notes.append("linear-matrix fast path was ENABLED during both runs")

    return CompareMpReport(
        s=s_b,
        sigma_baseline=sig_b, sigma_surrogate=sig_s,
        emit_baseline=emit_b, emit_surrogate=emit_s,
        transmission_baseline=trans_b, transmission_surrogate=trans_s,
        end_of_line=end,
        wall_baseline_s=wall_b, wall_surrogate_s=wall_s,
        surrogate_names=engaged_names,
        residual_n_steps=int(residual_n_steps),
        notes=notes,
        fast_path_enabled=fast_path,
    )


def plot_compare_mp_report(report: CompareMpReport, path: str | Path,
                            title: str | None = None) -> Path:
    """Write a 2x3 PNG (sigma_{x,y,z}, emit_{nx,ny}, transmission)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    s_m = report.s * 1e-3

    panels = [
        (axes[0, 0], "x", r"$\sigma_x$ [mm]", report.sigma_baseline, report.sigma_surrogate),
        (axes[0, 1], "y", r"$\sigma_y$ [mm]", report.sigma_baseline, report.sigma_surrogate),
        (axes[0, 2], "z", r"$\sigma_\phi$ [deg]", report.sigma_baseline, report.sigma_surrogate),
        (axes[1, 0], "nx", r"$\varepsilon_{nx}$ [mm.mrad]", report.emit_baseline, report.emit_surrogate),
        (axes[1, 1], "ny", r"$\varepsilon_{ny}$ [mm.mrad]", report.emit_baseline, report.emit_surrogate),
    ]
    for ax, key, lbl, base_d, surr_d in panels:
        ax.plot(s_m, base_d[key], "C0-",  lw=1.4, label="baseline (RK4)")
        ax.plot(s_m, surr_d[key], "C3--", lw=1.1, label="hybrid surrogate")
        ax.set_ylabel(lbl); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(s_m, report.transmission_baseline, "C0-",  lw=1.4, label="baseline")
    ax.plot(s_m, report.transmission_surrogate, "C3--", lw=1.1, label="hybrid")
    ax.set_ylabel("transmission"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("s [m]")
    axes[1, 1].set_xlabel("s [m]")
    axes[1, 2].set_xlabel("s [m]")

    base_title = (title
                  or "MP envelope comparison: baseline RK4 vs hybrid surrogate")
    fig.suptitle(
        f"{base_title}\n"
        f"residual substeps={report.residual_n_steps}  "
        f"speedup={report.speedup():.2f}x  "
        f"worst rel.diff={report.worst_rel_diff():.2e}  "
        f"engaged: {len(report.surrogate_names)}",
        fontsize=11)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
