"""Shared cooperative-cancellation exception.

Long-running core routines accept an optional ``should_stop`` callable
(default ``None`` → never polled, zero behavior change).  Two idioms:

* **partial-return** — aggregate loops whose partial output is useful
  (``ErrorStudy`` seeds, scan points) poll ``should_stop`` and return
  what they have;
* **raise** — all-or-nothing computations (training, phase advance,
  orbit correction, surrogate compare) raise :class:`OperationCancelled`
  so a half-finished result can never be mistaken for a complete one.

Deliberately dependency-free (no Qt, no torch) so every layer can import
it.  Mirrors the existing per-module patterns (``EnvelopeSolver``'s
``should_abort``, the matching engine's ``_MatchCancelled``).
"""
from __future__ import annotations


class OperationCancelled(Exception):
    """The user cancelled the operation via a ``should_stop`` hook."""
