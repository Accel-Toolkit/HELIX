"""Criticality scoring: turn a scenario's beam metrics into one damage score.

The score is a weighted, monotone-increasing-with-damage combination of
fractional transmission loss, normalised-emittance growth (x/y/z), and
fractional exit-energy deviation, all measured against the nominal baseline.
It is None-safe: a metric the forward pass didn't record (e.g. envelope mode
records no ``transmission``) contributes 0.
"""
from __future__ import annotations

import math

# Loss and energy dominate (mirrors the SET_KE_OUT_MIN / MIN_TRANSMISSION
# weighting guidance in lattice_commands.py).
DEFAULT_WEIGHTS = {
    "transmission": 10.0,
    "emit_x": 1.0,
    "emit_y": 1.0,
    "emit_z": 1.0,
    "energy": 5.0,
}


def _growth(base, val) -> float:
    if base is None or val is None or base <= 0:
        return 0.0
    return max(0.0, float(val) / float(base) - 1.0)


def criticality_score(
    baseline: dict, scn: dict, *, weights=None, lost_threshold_pct: float = 1.0
) -> tuple[float, dict, bool]:
    """Return ``(score, terms, beam_lost)``.

    ``baseline`` / ``scn`` are ``_scan_metrics`` dicts (transmission, emit_x/y/z,
    ref_w_kin, …). ``beam_lost`` is True when MP transmission falls below
    ``lost_threshold_pct`` or the exit energy is missing / NaN.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    terms: dict[str, float] = {}
    beam_lost = False

    t0, t = baseline.get("transmission"), scn.get("transmission")
    if t0 is not None and t is not None:
        terms["transmission"] = w["transmission"] * max(0.0, (t0 - t) / max(t0, 1.0))
        if t < lost_threshold_pct:
            beam_lost = True
    else:
        terms["transmission"] = 0.0

    # Use NORMALISED emittances (εn = βγ·ε_geo): geometric ε grows under
    # de-acceleration purely from reduced adiabatic damping, which would inflate
    # cavity-failure scores and double-count the energy term — and it would
    # disagree with the εn shown in the results table. εn isolates real dilution.
    terms["emit_x"] = w["emit_x"] * _growth(baseline.get("emit_nx"), scn.get("emit_nx"))
    terms["emit_y"] = w["emit_y"] * _growth(baseline.get("emit_ny"), scn.get("emit_ny"))
    terms["emit_z"] = w["emit_z"] * _growth(baseline.get("emit_nz"), scn.get("emit_nz"))

    e0, e = baseline.get("ref_w_kin"), scn.get("ref_w_kin")
    if e0 is not None and e is not None and e0 > 0 and math.isfinite(e):
        terms["energy"] = w["energy"] * abs(e0 - e) / max(e0, 1.0)
    else:
        terms["energy"] = 0.0
    if e is None or (isinstance(e, float) and not math.isfinite(e)):
        beam_lost = True

    return float(sum(terms.values())), terms, beam_lost
