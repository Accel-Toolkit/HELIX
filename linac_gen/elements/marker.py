# linac_gen/elements/marker.py
"""Marker element: named position with no dynamics."""
from linac_gen.elements.base import PassiveElement


class Marker(PassiveElement):
    """Named marker with no dynamics. Can trigger snapshot recording.

    ``is_bpm=True`` flags the marker as a beam-position monitor — used by
    the orbit-correction driver to resolve TraceWin ``DIAG_POSITION N``
    references and to filter "BPM-only" markers when building a response
    matrix.  The flag is set automatically by the parser for ``BPM`` and
    ``DIAG_POSITION`` cards; user-constructed markers are not BPMs by
    default.

    A ``DIAG_POSITION N X Y dm`` card additionally carries diagnostic
    matching data (TraceWin "matched with diagnostics"):

    * ``diag_family`` — the diagnostic number ``N`` linking this monitor
      to the ``ADJUST N v`` variables that serve it.
    * ``x_target_mm`` / ``y_target_mm`` — wanted beam centroid at this
      monitor (mm).  ``None`` = that plane is unconstrained (TraceWin's
      ``1e50`` sentinel, or the operand was absent).
    * ``accuracy_mm`` — diagnostic accuracy ``dm`` (mm, default 1);
      consumers weight residuals by ``1/dm``.
    * ``origin_keyword`` — ``"BPM"`` when the marker came from a native
      ``BPM :`` card so the writer can round-trip it as ``BPM`` instead
      of ``DIAG_POSITION``.

    Runtime target overrides loaded from an external file live in a
    single duck-typed ``diag_target_override = (x, y, weight)`` tuple
    attribute whose PRESENCE wins over the card operands (so a
    ``nan nan`` file row genuinely frees both planes) — deliberately
    NOT a constructor field so it can never be serialized into a deck.
    """

    def __init__(self, name: str, snapshot: bool = False,
                 is_bpm: bool = False,
                 diag_family: int | None = None,
                 x_target_mm: float | None = None,
                 y_target_mm: float | None = None,
                 accuracy_mm: float = 1.0,
                 origin_keyword: str | None = None):
        super().__init__(name=name)
        self.snapshot = snapshot
        self.is_bpm = is_bpm
        self.diag_family = None if diag_family is None else int(diag_family)
        self.x_target_mm = None if x_target_mm is None else float(x_target_mm)
        self.y_target_mm = None if y_target_mm is None else float(y_target_mm)
        self.accuracy_mm = (float(accuracy_mm)
                            if accuracy_mm and accuracy_mm > 0 else 1.0)
        self.origin_keyword = origin_keyword

    def apply(self, beam) -> None:
        pass  # no dynamics
