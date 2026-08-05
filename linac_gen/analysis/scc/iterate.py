"""Self-consistent SCC: iterate cards <-> envelope to a fixed point.

The one-shot :func:`~linac_gen.analysis.scc.driver.scc_analysis` computes
f_c from the sigma(z) profile of a run that was tracked at a DIFFERENT
compensation state — its suggested cards are a first iteration presented
as an answer.  This module closes the loop: run the envelope on a working
copy of the lattice, analyse, apply damped cards, repeat until the card
factors stop moving.  The physical fixed point is

    f_card == f_c( sigma(f_card) )

The coupling is weak (sigma enters the balance only through
ln(r_pipe/2*sigma) in the well depth), so plain iteration (omega=1)
converges in 2-3 passes; measured on the LEBT testbed, saturated
regimes (eta >= 1 everywhere) make the map exactly constant, where any
under-relaxation omega < 1 only adds geometric lag — max_delta then
decays as (1-omega)^k without a single physics change.  ``omega`` stays
available as insurance if a configuration ever oscillates; the fixed
point itself is omega-independent.

The user's lattice is NEVER mutated: everything happens on a deepcopy.
Only the (confirm-gated) GUI Apply writes cards into the live lattice.
"""
from __future__ import annotations

import copy
import math


def _card_key(card) -> float:
    # z-bin edges are derived from the (constant) line length each
    # iteration; round to nm so bitwise jitter can't split a slot.
    return round(float(card["z_m"]), 9)


def scc_self_consistent(lattice, beam_config, *, omega: float = 1.0,
                        tol: float = 0.01, max_iter: int = 8,
                        record_substeps: bool = True,
                        should_stop=None, progress=None,
                        **scc_kwargs) -> dict:
    """Iterate envelope <-> SCC analysis to self-consistent card factors.

    Returns the final analysis dict (same shape as ``scc_analysis``) with
    ``scc_cards`` factors replaced by the set the final envelope run was
    actually tracked with, ``fc_source = "self-consistent (k iterations)"``
    and an ``iterate`` metadata block (plain floats, JSON-safe).  A
    non-DC beam config refuses in-band via ``reason``; non-convergence
    sets ``iterate["converged"] = False`` plus a note — never raises.
    A stop request raises ``RuntimeError("aborted")``.
    """
    meta = {"converged": False, "n_iter": 0, "history": [],
            "omega": float(omega), "tol": float(tol)}
    if not getattr(beam_config, "continuous", False):
        return {"reason": (
            "self-consistent SCC needs a DC (continuous) beam — set "
            "continuous beam in the Beam tab; bunched beams are outside "
            "this steady-state gas-neutralisation model"),
            "notes": [], "fc_source": "n/a", "iterate": meta}
    if not (0.0 < float(omega) <= 1.0):
        raise ValueError(f"omega must be in (0, 1], got {omega}")
    if int(max_iter) < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if not (0.0 < float(tol) < 1.0):
        # factors live in [0, 1]: tol >= 1 would declare the all-zero
        # first pass "converged" — a silent footgun, so refuse loudly
        raise ValueError(f"tol must be in (0, 1), got {tol}")

    from linac_gen.analysis.scc.apply import apply_scc_cards
    from linac_gen.analysis.scc.driver import scc_analysis
    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.elements.space_charge_comp import SpaceChargeComp
    from linac_gen.tracking.envelope import EnvelopeSolver

    work = copy.deepcopy(lattice)
    # Fresh start: strip any pre-existing cards so iteration 1 measures
    # the uncompensated line and the damping baseline (0.0) is honest.
    had_cards = any(isinstance(e, SpaceChargeComp) for e in work.elements)
    if had_cards:
        work.elements[:] = [e for e in work.elements
                            if not isinstance(e, SpaceChargeComp)]

    # Cleared regions are element ranges against the CALLER's lattice —
    # which may contain cards that the strip above removed, and gains
    # new cards every iteration.  Two translations, both explicit:
    # caller -> stripped ("bare") indices ONCE here, then bare -> work
    # per iteration (a card applied at boundary b displaces every
    # element >= b).  Skipping the first one silently cleared the WRONG
    # element on card-bearing decks (adversarial finding).
    cleared = scc_kwargs.pop("cleared_regions", None)
    bare_regions: list[list[int]] = []
    if cleared:
        card_pos = {i for i, e in enumerate(lattice.elements)
                    if isinstance(e, SpaceChargeComp)}
        for r in cleared:
            e0, e1 = int(r[0]), int(r[1])
            if e0 in card_pos or e1 in card_pos:
                raise ValueError(
                    f"cleared region [{e0}, {e1}] bound is a "
                    "SPACE_CHARGE_COMP card — the iterator strips cards "
                    "first; bound the region with physical elements")
            bare_regions.append(
                [e0 - sum(1 for p in card_pos if p < e0),
                 e1 - sum(1 for p in card_pos if p < e1)])
    inserted_bounds: list[int] = []

    prev: dict[float, float] = {}         # card key -> factor now IN work
    analysis: dict = {}
    n_done = 0
    for k in range(1, int(max_iter) + 1):
        if should_stop is not None and should_stop():
            raise RuntimeError("aborted")
        ref = build_ref(beam_config)
        solver = EnvelopeSolver(
            work, ref, _envelope_initial(beam_config, ref),
            current=float(getattr(beam_config, "current", 0.0) or 0.0),
            record_substeps=record_substeps, should_abort=should_stop)
        res = solver.run()
        if getattr(solver, "aborted", False):
            # should_abort does NOT raise inside the solver — it returns
            # TRUNCATED results, which must never reach the analysis.
            raise RuntimeError("aborted")
        kw = scc_kwargs
        if bare_regions:
            kw = dict(scc_kwargs, cleared_regions=[
                [e0 + sum(1 for b in inserted_bounds if b <= e0),
                 e1 + sum(1 for b in inserted_bounds if b <= e1)]
                for e0, e1 in bare_regions])
        analysis = scc_analysis(res, work, should_stop=should_stop,
                                advise_iterate=False, **kw)
        n_done = k
        if analysis.get("reason"):
            analysis.setdefault("iterate", dict(meta, n_iter=k))
            return analysis
        computed = {_card_key(c): float(c["factor"])
                    for c in analysis["scc_cards"]}
        max_delta = max((abs(v - prev.get(key, 0.0))
                         for key, v in computed.items()), default=0.0)
        meta["history"].append({
            "max_delta": float(max_delta),
            "mean_fc": float(analysis["mean_fc"])})
        if progress is not None:
            try:
                progress(k, float(max_delta))
            except Exception:                                # noqa: BLE001
                pass
        if max_delta < float(tol):
            meta["converged"] = True
            break
        if k == int(max_iter):
            # Do NOT advance past the last analysis: the returned cards
            # must be the set the final run was actually tracked with.
            break
        prev = {key: prev.get(key, 0.0)
                + float(omega) * (v - prev.get(key, 0.0))
                for key, v in computed.items()}
        inserted_bounds = sorted(apply_scc_cards(
            work, [{"z_m": key, "factor": f}
                   for key, f in prev.items()]))

    meta["n_iter"] = n_done
    # Report the factors the final envelope run was tracked with (prev),
    # not the last raw computation — they agree within tol on convergence
    # and the mismatch IS the message when not converged.
    for c in analysis["scc_cards"]:
        c["factor"] = round(prev.get(_card_key(c), 0.0), 4)
        # the driver derived this against the card-bearing WORK lattice —
        # shifted vs the caller's element indices, so don't report it
        c["element_index"] = None
    if cleared and "cleared_regions" in analysis:
        # report the CALLER's element indices, not the work lattice's
        analysis["cleared_regions"] = [[int(r[0]), int(r[1])]
                                       for r in cleared]
    analysis["fc_source"] = f"self-consistent ({n_done} iterations)"
    analysis["iterate"] = {
        "converged": bool(meta["converged"]), "n_iter": int(n_done),
        "history": meta["history"], "omega": float(omega),
        "tol": float(tol)}
    if had_cards:
        analysis["notes"].append(
            "pre-existing SPACE_CHARGE_COMP cards were set aside — the "
            "iteration starts from the uncompensated line")
    if not meta["converged"]:
        last = meta["history"][-1]["max_delta"] if meta["history"] else (
            math.nan)
        analysis["notes"].append(
            f"NOT converged after {n_done} iterations (last max "
            f"Dfactor {last:.4f} vs tol {tol:g}) — factors are the last "
            "applied damped set; raise max_iter or lower omega")
    return analysis
