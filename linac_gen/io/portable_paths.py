"""Portable-path helpers for everything HELIX persists to disk.

HELIX historically wrote machine-absolute paths into ``.dat`` lattices
(field-map files) and ``.lgproj`` projects (lattice path, beam ``.dst``)
— every saved artifact was welded to the author's directory tree.  The
writers now relativize against the artifact's own directory via
:func:`best_relpath`, and the loaders resolve relative entries via
:func:`resolve_candidates`.  Only *persistence* changed: in-memory
paths (``element.field_file``, ``state.lattice_path``) stay absolute,
because runtime code consumes them verbatim.

No GUI imports here — this module is shared by the TraceWin writer, the
CLI project loader, and the GUI app.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple


def best_relpath(target: str, anchor_dir: str) -> Tuple[str, bool]:
    """Relativize ``target`` against ``anchor_dir`` when meaningful.

    Returns ``(path, ok)``:

    * ``ok=True``  — ``path`` is ``target`` relative to ``anchor_dir``,
      with POSIX separators (``os.path.join`` accepts ``/`` on Windows
      too, so written files stay cross-OS readable).  Escaping-parent
      results (``../../Fields``) are allowed — they match the tracked-
      example idiom and survive moving the whole tree together.
    * ``ok=False`` — no meaningful relative form exists; ``path`` is the
      absolute target.  Two cases: a Windows cross-drive pair
      (``os.path.relpath`` raises ``ValueError``), or the only common
      ancestor is the filesystem/drive root (a ``../../..``-to-root
      relpath, e.g. ``/var/…`` vs ``/Users/…``, is not portable — the
      two trees only move together if the whole disk does).
    """
    t = os.path.abspath(str(target))
    a = os.path.abspath(str(anchor_dir))
    try:
        rel = os.path.relpath(t, a)
    except ValueError:          # Windows: different drives
        return t, False
    try:
        common = os.path.commonpath([t, a])
    except ValueError:          # pragma: no cover — mixed abs forms
        return t, False
    if os.path.splitdrive(common)[1] in (os.sep, "/"):
        return t, False
    return rel.replace(os.sep, "/"), True


def resolve_candidates(raw: str,
                       bases: Iterable[str]) -> Optional[str]:
    """First existing absolute resolution of ``raw`` against ``bases``.

    ``raw`` may be absolute (returned as-is when it exists) or relative
    (joined against each base directory in order).  Returns ``None``
    when nothing exists — callers decide how to warn/fall back.
    """
    raw = str(raw)
    if os.path.isabs(raw):
        return raw if os.path.exists(raw) else None
    for base in bases:
        cand = os.path.abspath(os.path.join(str(base), raw))
        if os.path.exists(cand):
            return cand
    return None
