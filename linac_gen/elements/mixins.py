"""Element mixins for cross-cutting capabilities.

This module exists to keep cross-cutting concerns (misalignment,
field errors, ...) out of every concrete element class.  An element
opts in by inheriting one or more of these mixins **after** its main
base (TransferMapElement, ThinKickElement, ...) and calling the
``_init_*`` helpers from its ``__init__``.

The pattern matches IMPACT-X's ``Alignment`` mixin (see
``src/particles/elements/mixin/alignment.H``) — single source of
truth for the misalignment contract, applied uniformly across element
types.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
class Misalignment:
    """Adds element-frame misalignment parameters.

    Six degrees of freedom:

    ===========  ==========================================
    Parameter    Meaning
    ===========  ==========================================
    ``dx``       horizontal offset of magnetic centre [mm]
    ``dy``       vertical offset of magnetic centre [mm]
    ``dz``       longitudinal offset (path-length error) [mm]
    ``tilt_deg`` rotation about the longitudinal (z) axis [°]
    ``pitch_deg`` rotation about the horizontal (x) axis [°]
    ``yaw_deg``   rotation about the vertical (y) axis [°]
    ===========  ==========================================

    The tracker honours **dx, dy, tilt_deg** today (Tier 1).  ``dz``,
    ``pitch_deg``, ``yaw_deg`` are reserved slots that elements
    (and the TraceWin parser) can populate; they will be consumed by
    a future tracker upgrade — currently they are stored but do not
    influence tracking.

    The tracker reads ``dx``, ``dy``, and ``tilt_deg`` directly from
    the element via ``getattr(elem, 'dx', 0.0)`` etc., so adding the
    mixin to an element is enough to make misalignment work — no
    tracker changes per element.
    """

    def _init_misalignment(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        tilt_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> None:
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)
        self.tilt_deg = float(tilt_deg)
        self.pitch_deg = float(pitch_deg)
        self.yaw_deg = float(yaw_deg)

    @property
    def is_misaligned(self) -> bool:
        """True if any misalignment parameter is non-zero."""
        return bool(
            self.dx or self.dy or self.dz
            or self.tilt_deg or self.pitch_deg or self.yaw_deg
        )

    @staticmethod
    def tilt_rotation_matrix(tilt_deg: float) -> np.ndarray:
        """6x6 rotation around the longitudinal axis (M8).

        Linear-optics version of the per-particle rotation in
        ``tracker.py:208-219``.  Operates on the (x, x', y, y') block;
        (phi, dW) are left unchanged (longitudinal rotation doesn't
        couple to them).  Returns identity when ``tilt_deg == 0``.

        Used by the envelope solver and the gradient-mode matcher to
        get the same misalignment treatment as the MP tracker; mirrors
        :func:`linac_gen.tracking.torch_tracking.tilt_rotation_matrix_torch`.
        """
        if abs(float(tilt_deg)) < 1e-12:
            return np.eye(6)
        th = math.radians(float(tilt_deg))
        c, s = math.cos(th), math.sin(th)
        R = np.eye(6)
        R[0, 0] = c;  R[0, 2] = s
        R[1, 1] = c;  R[1, 3] = s
        R[2, 0] = -s; R[2, 2] = c
        R[3, 1] = -s; R[3, 3] = c
        return R

    def _misalignment_repr(self) -> str:
        """Return a compact "dx=..,dy=..,tilt=.." string for __repr__.

        Returns an empty string if the element is perfectly aligned
        (so concrete __repr__ implementations can ``f'..{self._misalignment_repr()}..'``
        without unconditional clutter).
        """
        if not self.is_misaligned:
            return ""
        bits = []
        for label, val in (
            ("dx", self.dx), ("dy", self.dy), ("dz", self.dz),
            ("tilt", self.tilt_deg),
            ("pitch", self.pitch_deg),
            ("yaw", self.yaw_deg),
        ):
            if val:
                bits.append(f"{label}={val:g}")
        return " " + " ".join(bits)


# ---------------------------------------------------------------------------
class FieldError:
    """Adds a single relative field-error knob to an element.

    The element stores a *design* field strength (``gradient``,
    ``field``, ``voltage``, ...) and a small ``*_rel`` perturbation
    that the apply / transfer-matrix code multiplies in:

        effective = design * (1 + design_rel)

    Concrete elements pick the relevant attribute name (e.g.
    ``gradient_rel`` for Quadrupole, ``field_rel`` for Solenoid) — the
    mixin just provides a uniform initialiser and the convention.
    """

    def _init_field_error(self, **kwargs: float) -> None:
        """Set arbitrary ``*_rel`` attributes from kwargs.

        Example::

            class Quadrupole(..., FieldError):
                def __init__(self, ..., gradient_rel=0.0):
                    ...
                    self._init_field_error(gradient_rel=gradient_rel)
        """
        for name, val in kwargs.items():
            setattr(self, name, float(val))
