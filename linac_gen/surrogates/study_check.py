"""Before a study uses surrogates: its design lattice, run natively and with them.

A study compares many runs with its own baseline, and weights that answer off their native
element would bias every one of them.  Weights loaded from a file pass their own accuracy
check before they answer (``SurrogateFieldMap._accuracy_refusal``; weights built in memory
are not checked there); this one checks the line as a whole, on the study's own beam: the
design lattice is run natively and with the registered surrogates, and the study uses them
only when every exit rms size and emittance of the two runs agrees within
``DESIGN_TOLERANCE``.  Otherwise, and in a multiparticle study (the surrogate path is linear
inside a cavity, a few per cent off, and no faster) or with the Sacherer envelope (which
never asks them), the study runs native in its thread (``registry.suspended``) and says so.
``runs_native`` is the same for a study the check does not cover yet.  The check counts what
the weights do in its own thread (``base.tally``), whatever another run does meanwhile.
"""
from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass, field

import numpy as np

from linac_gen.surrogates import registry

__all__ = ["DESIGN_TOLERANCE", "DesignCheck", "check_design", "exit_values", "runs_native",
           "study_surrogates", "refusal_text"]

#: the largest relative difference of an exit rms size or emittance allowed between the
#: design lattice run natively and with the registered surrogates
DESIGN_TOLERANCE = 5e-3

#: what the check compares at the exit: the rms sizes, and the rms emittances as the
#: square roots of the determinants of the 2x2 blocks of Σ
EXIT_KEYS = ("sigma_x", "sigma_y", "sigma_phi", "sigma_w", "eps_x", "eps_y", "eps_z")


@dataclass
class DesignCheck:
    """What a study does with the registered surrogates (``use``), and the line it says
    about them (``note``, empty when none is registered).  ``nn_calls`` and ``refused``: the
    network answers and the refusals by reason on the design lattice run with them; its
    exit values natively and with them (``native``, ``surrogate``) when the check ran."""
    use: bool
    note: str
    worst: float | None = None
    worst_key: str | None = None
    nn_calls: int = 0
    diffs: dict = field(default_factory=dict)
    refused: dict = field(default_factory=dict)
    native: dict = field(default_factory=dict)
    surrogate: dict = field(default_factory=dict)


def exit_values(results) -> dict:
    """``{key: value}`` of ``EXIT_KEYS`` at the exit of an envelope run."""
    sigma = np.asarray(results.sigma_matrix)[-1]
    out = {k: float(np.asarray(getattr(results, k))[-1])
           for k in ("sigma_x", "sigma_y", "sigma_phi", "sigma_w")}
    for key, i in (("eps_x", 0), ("eps_y", 2), ("eps_z", 4)):
        out[key] = float(np.sqrt(max(np.linalg.det(sigma[i:i + 2, i:i + 2]), 0.0)))
    return out


def refusal_text(refused: dict) -> str:
    """``{reason: count}`` as "reason count, ...", the commonest first."""
    return ", ".join(f"{k} {n}" for k, n in sorted(refused.items(), key=lambda kv: (-kv[1], kv[0])))


def check_design(lattice, beam_config, *, mode: str = "envelope", env_solver: str = "matrix",
                 tolerance: float = DESIGN_TOLERANCE, should_stop=None) -> DesignCheck:
    """Whether a study of ``lattice`` with ``beam_config`` in ``mode`` ("envelope" or "mp")
    uses the registered surrogates.  ``env_solver`` is the study's envelope solver;
    ``should_stop`` (a zero-argument callable) stops the check at the next element, and the
    study then runs native."""
    if not registry.list_registered():
        return DesignCheck(use=False, note="")
    if mode != "envelope":
        return DesignCheck(use=False, note=(
            "surrogates not used: a multiparticle study runs native (the surrogate path is "
            "linear inside a cavity)"))
    if str(env_solver).lower() == "sacherer":
        return DesignCheck(use=False, note=(
            "surrogates not used: the Sacherer envelope does not use them"))
    from linac_gen.cli.common import run_envelope_sim
    from linac_gen.surrogates.base import tally
    def stopped() -> bool:
        return should_stop is not None and bool(should_stop())

    stop_note = DesignCheck(use=False, note="surrogates not used: the study was stopped")
    if stopped():
        return stop_note
    with registry.suspended():
        native_run = run_envelope_sim(copy.deepcopy(lattice), copy.deepcopy(beam_config),
                                      env_solver, should_abort=should_stop)
    if stopped():
        return stop_note
    native = exit_values(native_run)
    with tally() as counted:
        surr_run = run_envelope_sim(copy.deepcopy(lattice), copy.deepcopy(beam_config),
                                    env_solver, should_abort=should_stop)
    if stopped():
        return stop_note
    surr = exit_values(surr_run)
    calls, refused = counted.calls, dict(counted.refused)
    if calls == 0:
        why = (f"left native: {refusal_text(refused)}" if refused else
               "none was asked: with space charge the envelope steps through each field map "
               "in slices")
        return DesignCheck(use=False, note=(
            f"surrogates not used: none answered on the design lattice ({why})"),
            refused=refused)
    diffs = {}
    for key in EXIT_KEYS:
        a, b = native[key], surr[key]
        d = abs(b - a) / abs(a) if a != 0.0 else (0.0 if b == a else math.inf)
        diffs[key] = d if math.isfinite(d) else math.inf      # a NaN is no agreement
    worst_key = max(EXIT_KEYS, key=lambda k: diffs[k])
    worst = diffs[worst_key]
    if worst <= tolerance:
        return DesignCheck(True, (
            f"surrogates used: with them the design lattice ({calls} network answers) has every "
            f"exit rms size and emittance within {worst:.2g} of the native run (largest: "
            f"{worst_key}; limit {tolerance:g})"), worst, worst_key, calls, diffs, refused,
            native, surr)
    return DesignCheck(False, (
        f"surrogates not used: on the design lattice the exit {worst_key} with them differs "
        f"from the native run by {worst:.2g} (limit {tolerance:g})"),
        worst, worst_key, calls, diffs, refused, native, surr)


def runs_native(why: str) -> DesignCheck:
    """A study that runs native, with the line it says when surrogates are registered."""
    if not registry.list_registered():
        return DesignCheck(use=False, note="")
    return DesignCheck(use=False, note=f"surrogates not used: {why}")


@contextlib.contextmanager
def study_surrogates(check: DesignCheck):
    """A study's runs: with the registered surrogates if ``check`` uses them, else with
    none of them in this thread."""
    if check.use:
        yield
    else:
        with registry.suspended():
            yield
