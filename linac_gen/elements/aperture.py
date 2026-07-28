"""Aperture passive element — dedicated obstruction check with shape support.

TraceWin ``APERTURE dx dy n`` convention:

    n = 0  : Rectangular aperture, ``dx`` = half width x, ``dy`` = half width y.
    n = 1  : Circular aperture, ``dx`` = radius (``dy`` ignored).
    n = 2  : Pepperpot -- not simulated; pass-through with a log warning.
    n = 3  : Rectangular + beam-fraction adjustment -- treated as rectangular.
    n = 4  : Horizontal finger -- treated as rectangular for now.
    n = 5  : Vertical finger   -- treated as rectangular for now.
    n = 6  : Ring              -- not simulated.

The legacy kwargs ``a`` / ``b`` (pre-TraceWin-compat) are accepted and mapped
to ``dx`` / ``dy`` so older lattice scripts keep working.  Legacy class
constants are renumbered to match the TraceWin ``n`` flag, so code that
referenced ``Aperture.CIRCULAR`` / ``Aperture.RECTANGULAR`` now sees the
TraceWin values (``1`` and ``0`` respectively); the old ``ELLIPTICAL`` constant
is removed (TraceWin has no elliptical shape on the APERTURE card).
"""
import logging

import numpy as np

from linac_gen.elements.base import PassiveElement

_log = logging.getLogger(__name__)


class Aperture(PassiveElement):
    """Shape-aware aperture obstruction (marks particles lost, zero length)."""

    # TraceWin shape flags
    RECTANGULAR = 0
    CIRCULAR = 1
    PEPPERPOT = 2
    FRACTION = 3
    FINGER_H = 4
    FINGER_V = 5
    RING = 6

    def __init__(self, name: str,
                 dx: float = 0.0, dy: float = 0.0,
                 aperture_type: int = 0,
                 a: float | None = None, b: float | None = None):
        super().__init__(name=name)
        # Legacy kwargs: map ``a`` / ``b`` to ``dx`` / ``dy`` when supplied.
        if a is not None:
            dx = a
        if b is not None:
            dy = b
        self.dx = float(dx)
        # When dy is not given (or <= 0), default vertical half-size to dx.
        self.dy = float(dy) if dy > 0 else float(dx)
        self.aperture_type = int(aperture_type)

    # Legacy aliases -- older code (and some GUI widgets) still uses ``a``/``b``.
    @property
    def a(self) -> float:
        return self.dx

    @property
    def b(self) -> float:
        return self.dy

    def apply(self, beam) -> None:
        alive_idx = np.where(beam.alive_mask)[0]
        if len(alive_idx) == 0:
            return
        x = beam.particles[alive_idx, 0]
        y = beam.particles[alive_idx, 2]

        t = self.aperture_type
        if t == self.CIRCULAR:
            lost = x * x + y * y > self.dx * self.dx
        elif t in (self.RECTANGULAR, self.FRACTION,
                   self.FINGER_H, self.FINGER_V):
            lost = (np.abs(x) > self.dx) | (np.abs(y) > self.dy)
        elif t == self.PEPPERPOT:
            _log.warning(
                "%s: pepperpot aperture (type 2) is parsed but not simulated; "
                "treating as a pass-through", self.name,
            )
            return
        elif t == self.RING:
            _log.warning("%s: ring aperture (type 6) not yet simulated",
                         self.name)
            return
        else:
            _log.warning("%s: unknown aperture type %s; pass-through",
                         self.name, t)
            return

        for idx in alive_idx[lost]:
            beam.record_loss(int(idx), beam.ref.s, self.name)
