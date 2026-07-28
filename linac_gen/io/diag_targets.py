# linac_gen/io/diag_targets.py
"""External BPM-target files for diagnostic matching.

A target file supplies the *set-points* the optimizer should steer the
computed BPM readings onto — e.g. a measured machine orbit.  Readings
themselves are ALWAYS computed by tracking; this file never supplies a
reading.

Format: whitespace-delimited text, ``#`` comments, no header required::

    # x_mm      y_mm     [weight]
      0.0356   -0.0007
      0.0579   -0.0603    2.0
      nan       0.12          # nan = leave that plane unconstrained

One data row per ``is_bpm`` marker, in lattice order; the row count must
equal the number of BPMs in the lattice (hard error otherwise — silent
truncation would misassign every downstream target).

Loaded targets are applied as a *runtime override* (a single duck-typed
``diag_target_override = (x, y, weight)`` tuple attribute) whose
PRESENCE takes precedence over the deck's ``DIAG_POSITION`` operands —
so a ``nan nan`` row genuinely frees both planes rather than falling
back to the deck values.  Overrides are never serialized back into a
``.dat`` by the writer.
"""
from __future__ import annotations

import math


def load_diag_targets(path):
    """Parse a target file → list of ``(x_mm|None, y_mm|None, weight|None)``.

    ``nan`` in a coordinate column maps to ``None`` (plane
    unconstrained); the third column (weight) is optional per row.
    """
    rows = []
    with open(path) as fh:
        for ln, raw in enumerate(fh, start=1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in (2, 3):
                raise ValueError(
                    f"{path}:{ln}: expected 'x_mm y_mm [weight]', "
                    f"got {len(parts)} column(s): {line!r}")
            try:
                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2]) if len(parts) == 3 else None
            except ValueError as exc:
                raise ValueError(f"{path}:{ln}: {exc}") from None
            rows.append((None if math.isnan(x) else x,
                         None if math.isnan(y) else y,
                         w))
    return rows


def apply_diag_targets(lattice, rows) -> int:
    """Attach ``rows`` (from :func:`load_diag_targets`) to the lattice's
    BPM markers, in lattice order.  Returns the number of BPMs updated.

    Raises ``ValueError`` when the row count does not match the number
    of ``is_bpm`` markers — a mismatch would silently misassign every
    target downstream of the first discrepancy.
    """
    bpms = [e for e in lattice.elements if getattr(e, "is_bpm", False)]
    if len(rows) != len(bpms):
        raise ValueError(
            f"target file has {len(rows)} row(s) but the lattice has "
            f"{len(bpms)} BPM marker(s) — counts must match exactly "
            "(one row per is_bpm marker, in lattice order)")
    for (x, y, w), bpm in zip(rows, bpms):
        bpm.diag_target_override = (x, y, w)
    return len(bpms)


def clear_diag_targets(lattice) -> int:
    """Remove any runtime target overrides; returns how many BPMs had one.

    Deleting the attribute (not None-ing fields) is what restores the
    deck targets — override PRESENCE is what wins downstream.
    """
    n = 0
    for e in lattice.elements:
        if getattr(e, "is_bpm", False) and \
                getattr(e, "diag_target_override", None) is not None:
            del e.diag_target_override
            n += 1
    return n
