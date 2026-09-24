"""ScanPoint construction for the campaign legs.

Override composition order (later wins on the same selector):
pins → error draws → correction kicks → fault → compensation settings → foil.
"""
from __future__ import annotations

import dataclasses

__all__ = ["compose_overrides", "make_point"]


def compose_overrides(*groups) -> tuple:
    out: list = []
    for g in groups:
        if not g:
            continue
        for sel, val in g:
            out.append((str(sel), val))
    return tuple(out)


def make_point(spec, input_path: str, *, overrides=(), mode: str = "envelope",
               seed: int = 42, out_path: str | None = None, beam_overrides=None,
               snapshot_elements=(), loss_metrics: bool = True, tracewin_ini=None):
    """A ``ScanPoint`` for one campaign run through ``cli.common.build_scan_point``
    (the spec's beam / sc / numerics overrides, then the run's own)."""
    from linac_gen.cli.common import build_scan_point
    beam = dict(spec.beam or {})
    beam.update(beam_overrides or {})
    point = build_scan_point(
        str(input_path), beam_overrides=beam or None,
        element_overrides=tuple(overrides),
        sc_overrides=dict(spec.sc) or None, mode=mode,
        env_solver=(spec.forward or {}).get("env_solver", "matrix"),
        seed=int(seed), cli=dict(spec.numerics or {}),
        tracewin_ini=tracewin_ini)
    return dataclasses.replace(
        point, out_path=(str(out_path) if out_path else None), capture_errors=True,
        loss_metrics=bool(loss_metrics), snapshot_elements=tuple(snapshot_elements or ()))
