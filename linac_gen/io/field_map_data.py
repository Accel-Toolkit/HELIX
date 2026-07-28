"""Dataclasses for multi-channel field-map data.

A ``FIELD_MAP`` with ``geom`` may enable up to four *channels* —
static electric, static magnetic, RF electric, RF magnetic — each
possibly on its own geometry (1-D, 2-D cyl, 2-D Cart, 3-D Cart).
:class:`FieldMapData` holds one :class:`FieldChannel` per enabled
channel.

Units (enforced by every reader):
  * Positions: **mm** (converted from the manual's metres).
  * Electric-field values: **MV/m**.
  * Magnetic-field values: **T**.
  * Frequency: **MHz**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from linac_gen.io.tracewin_geom import Channel


@dataclass
class FieldChannel:
    """One channel's grid + field components.

    The populated attributes depend on ``geometry``:

    * 1-D (``geometry=1``):            ``z``, ``Fz``.
    * 2-D cyl (``geometry=4`` or 5):   ``z``, ``r``, ``Fz``, ``Fr``
      (plus ``Fq`` for RF TM/TE modes).
    * 2-D Cart (``geometry=6``):       ``x``, ``y``, ``Fx``, ``Fy``.
    * 3-D Cart (``geometry=7``):       ``x``, ``y``, ``z``, ``Fx``,
      ``Fy``, ``Fz``.
    * 1-D G(z) (``geometry=9``):       ``z``, ``Fz`` (gradient in T/m).

    ``F`` stands for the channel's native field — E (electric) or B
    (magnetic); which one is determined by the :class:`Channel` the
    data is attached to inside a :class:`FieldMapData`.
    """

    geometry: int                             # digit 1, 4, 5, 6, 7, or 9
    z: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    Fx: Optional[np.ndarray] = None
    Fy: Optional[np.ndarray] = None
    Fz: Optional[np.ndarray] = None
    Fr: Optional[np.ndarray] = None
    Fq: Optional[np.ndarray] = None           # Bθ (TM mode) or Eθ (TE mode)

    norm_factor: float = 1.0


class FieldMapData:
    """Container of per-channel data for a single FIELD_MAP element.

    Supports both the new multi-channel API and the legacy flat-field
    API for backwards compatibility.

    New API (Task 3+):
        fd = FieldMapData(z=z)
        fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=Ez)

    Legacy API (back-compat, accepted in constructor):
        fd = FieldMapData(z=z, Ez=Ez, symmetry="1d")
        fd = FieldMapData(z=z, r=r, Ez=Ez, Er=Er, symmetry="cylindrical")
        fd = FieldMapData(x=x, y=y, z=z, Ex=Ex, Ey=Ey, Ez=Ez, symmetry="3d")

    Legacy kwargs are automatically converted to the appropriate channel
    in ``__init__``.
    """

    def __init__(
        self,
        z: np.ndarray,
        channels: Optional[Dict[Channel, FieldChannel]] = None,
        frequency: float = 0.0,
        aperture_file: Optional[str] = None,
        symmetry: str = "",
        # ------------------------------------------------------------------
        # Space-charge compensation (Task 2c)
        # ------------------------------------------------------------------
        scc_profile: Optional[np.ndarray] = None,
        scc_scale: float = 0.0,
        scc_mode: int = 0,
        # ------------------------------------------------------------------
        # Aperture-override flag and pipe-radius profile (Task 2d)
        # ------------------------------------------------------------------
        # ``pipe_radius_profile`` holds the element-local aperture profile
        # loaded from a ``.ouv`` file when ``Ka=1`` on the FIELD_MAP card.
        # Format: ``(z_mm, rx_mm, ry_mm)`` — z along element, separate x/y
        # half-widths.  For circular ouv files both are equal.  The tracker
        # linearly interpolates these against the particle's z offset from
        # the element entrance and scrapes particles with |x|>rx or
        # |y|>ry (rectangular) or (x/rx)²+(y/ry)²>1 (elliptical per the
        # FIELD_MAP flag, not yet distinguished in loading).
        pipe_radius_profile: "Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]" = None,
        ka: int = 0,
        # ------------------------------------------------------------------
        # Legacy flat-field kwargs — accepted for back-compat; converted to
        # channels automatically in __init__.  Do not set these after
        # construction; use channels[] directly.
        # ------------------------------------------------------------------
        r: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        Ez: Optional[np.ndarray] = None,
        Er: Optional[np.ndarray] = None,
        Ex: Optional[np.ndarray] = None,
        Ey: Optional[np.ndarray] = None,
        Bz: Optional[np.ndarray] = None,
        Br: Optional[np.ndarray] = None,
        Bx: Optional[np.ndarray] = None,
        By: Optional[np.ndarray] = None,
        norm_factor: float = 1.0,
    ):
        self.z = z
        self.channels: Dict[Channel, FieldChannel] = channels if channels is not None else {}
        self.frequency = frequency
        self.aperture_file = aperture_file
        # Space-charge compensation (Task 2c)
        self.scc_profile: Optional[np.ndarray] = scc_profile
        self.scc_scale: float = scc_scale
        self.scc_mode: int = scc_mode
        # Aperture-override (Task 2d)
        self.pipe_radius_profile: "Optional[tuple[np.ndarray, np.ndarray]]" = pipe_radius_profile
        self.ka: int = ka

        # ------------------------------------------------------------------
        # Convert legacy flat-field kwargs into channels[] if any were given
        # and no channels were explicitly supplied.
        # ------------------------------------------------------------------
        has_legacy = any(
            v is not None
            for v in (r, x, y, Ez, Er, Ex, Ey, Bz, Br, Bx, By)
        )
        if has_legacy and not self.channels:
            if symmetry == "3d" or (
                x is not None or y is not None
                or Ex is not None or Ey is not None
                or Bx is not None or By is not None
            ):
                # 3-D Cartesian: store as RF_E and/or STAT_B channels
                if Ez is not None or Ex is not None or Ey is not None:
                    self.channels[Channel.RF_E] = FieldChannel(
                        geometry=7,
                        x=x, y=y, z=z,
                        Fx=Ex, Fy=Ey, Fz=Ez,
                        norm_factor=norm_factor,
                    )
                if Bz is not None or Bx is not None or By is not None:
                    self.channels[Channel.STAT_B] = FieldChannel(
                        geometry=7,
                        x=x, y=y, z=z,
                        Fx=Bx, Fy=By, Fz=Bz,
                        norm_factor=norm_factor,
                    )
            elif symmetry == "cylindrical" or r is not None:
                # 2-D cylindrical
                if Ez is not None or Er is not None:
                    self.channels[Channel.RF_E] = FieldChannel(
                        geometry=4,
                        z=z, r=r,
                        Fz=Ez, Fr=Er,
                        norm_factor=norm_factor,
                    )
                if Bz is not None or Br is not None:
                    self.channels[Channel.RF_B] = FieldChannel(
                        geometry=5,
                        z=z, r=r,
                        Fz=Bz, Fr=Br,
                        norm_factor=norm_factor,
                    )
            else:
                # 1-D (or 1-D-ish: may have Er as a side-component)
                if Ez is not None or Er is not None:
                    self.channels[Channel.RF_E] = FieldChannel(
                        geometry=1,
                        z=z,
                        Fz=Ez,
                        Fr=Er,
                        norm_factor=norm_factor,
                    )
                if Bz is not None or Br is not None:
                    self.channels[Channel.STAT_B] = FieldChannel(
                        geometry=1,
                        z=z,
                        Fz=Bz,
                        Fr=Br,
                        norm_factor=norm_factor,
                    )

        # Symmetry hint: auto-detect if not supplied but legacy args present
        if symmetry:
            self._symmetry = symmetry
        elif self.channels:
            # Infer from the first channel's geometry
            first = next(iter(self.channels.values()))
            g = first.geometry
            if g == 1 or g == 9:
                self._symmetry = "1d"
            elif g in (4, 5):
                self._symmetry = "cylindrical"
            elif g in (6, 7):
                self._symmetry = "3d"
            else:
                self._symmetry = ""
        else:
            self._symmetry = "1d"   # default for empty containers

    # ------------------------------------------------------------------
    # Symmetry property — back-compat
    # ------------------------------------------------------------------
    @property
    def symmetry(self) -> str:
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value: str) -> None:
        self._symmetry = value

    # ------------------------------------------------------------------
    def axis_length_mm(self) -> float:
        return float(self.z[-1] - self.z[0])

    def has_static(self) -> bool:
        return any(ch.is_static for ch in self.channels.keys())

    def has_rf(self) -> bool:
        return any(ch.is_rf for ch in self.channels.keys())

    def has_electric(self) -> bool:
        return any(ch.is_electric for ch in self.channels.keys())

    def has_magnetic(self) -> bool:
        return any(ch.is_magnetic for ch in self.channels.keys())

    # ------------------------------------------------------------------
    # Legacy factory constructors — wrap a flat set of arrays into a
    # single-channel FieldMapData so existing .edz fixtures keep working
    # until every caller has migrated.
    # ------------------------------------------------------------------
    @classmethod
    def from_legacy_1d(cls, z: np.ndarray, Ez: np.ndarray,
                       norm_factor: float = 1.0) -> "FieldMapData":
        """Legacy 1-D (z, Ez) → single RF_E channel with geometry=1.

        Existing fixtures (``test_1d.edz`` etc.) don't declare a geom;
        historically they represent RF cavities.
        """
        fd = cls(z=z, frequency=0.0, symmetry="1d")
        fd.channels[Channel.RF_E] = FieldChannel(
            geometry=1, z=z, Fz=Ez, norm_factor=norm_factor,
        )
        return fd

    @classmethod
    def from_legacy_2d_cyl(cls, z: np.ndarray, r: np.ndarray,
                           Ez: np.ndarray, Er: np.ndarray,
                           Bz: Optional[np.ndarray] = None,
                           Br: Optional[np.ndarray] = None,
                           norm_factor: float = 1.0) -> "FieldMapData":
        """Legacy 2-D cyl (z, r, Ez, Er[, Bz, Br]) → RF_E channel with
        geometry=4; if Bz/Br are present also populate an RF_B geometry=5
        channel."""
        fd = cls(z=z, frequency=0.0, symmetry="cylindrical")
        fd.channels[Channel.RF_E] = FieldChannel(
            geometry=4, z=z, r=r, Fz=Ez, Fr=Er, norm_factor=norm_factor,
        )
        if Bz is not None or Br is not None:
            fd.channels[Channel.RF_B] = FieldChannel(
                geometry=5, z=z, r=r, Fz=Bz, Fr=Br, norm_factor=norm_factor,
            )
        return fd

    # ------------------------------------------------------------------
    # DEPRECATED flat-field properties
    # -----------------------------------------------------------------
    # The properties below (Ez, Er, Ex, Ey, Bz, Br, Bx, By, x, y, r)
    # were the original single-channel API.  They are now derived from
    # the per-channel store (``self.channels``), so they still work for
    # any code that was written against the old API.
    #
    # **These properties are deprecated as of Task 10 (2026-04-19).**
    # New code should read from ``fd.channels[Channel.*].F{z,r,x,y,q}``
    # directly (or use the ``has_electric()`` / ``has_magnetic()``
    # helpers to discover which channels are present).
    # They will be removed once every external caller in Linac_Gen core
    # and the test suite has been migrated to the multi-channel API.
    #
    # Known callers that still use the legacy API (follow-up migration):
    #   - ``linac_gen/io/field_map_reader.py`` — ``expand_1d_offaxis``
    #     accesses ``fmap.z`` and ``fmap.Ez`` for the spline helper.
    #   - ``linac_gen/elements/field_map.py`` — ``_interpolate_Ez``
    #     reads ``fd.Ez`` / ``fd.z`` for the back-compat interpolation path.
    #   - ``tests/io/test_field_map_reader.py`` — dozens of assertions on
    #     ``fmd.Ez``, ``fmd.Er``, ``fmd.Bz``, ``fmd.Br``, ``fmd.z``.
    #   - ``tests/io/test_field_map_tracewin_spec.py`` — asserts on
    #     ``data.Ez``, ``data.Er``, ``data.z``.
    #   - ``tests/io/test_field_map_3d_reader.py`` — asserts on
    #     ``data.Ez``, ``data.Ex``, ``data.Ey``, ``data.Bx``.
    #   - ``tests/io/test_field_map_data.py`` — asserts on ``fd.Ez``,
    #     ``fd.Bz``, ``fd.x``, ``fd.y``.
    #   - ``tests/elements/test_field_map_3d.py`` — uses ``data3.Ez``,
    #     ``data3.Ex``, ``data3.Ey`` to build axis slice.
    # ------------------------------------------------------------------
    def _pick_electric(self) -> Optional[FieldChannel]:
        return (self.channels.get(Channel.RF_E)
                or self.channels.get(Channel.STAT_E))

    def _pick_magnetic(self) -> Optional[FieldChannel]:
        return (self.channels.get(Channel.RF_B)
                or self.channels.get(Channel.STAT_B))

    @property
    def Ez(self):
        ch = self._pick_electric()
        return ch.Fz if ch else None

    @property
    def Er(self):
        ch = self._pick_electric()
        return ch.Fr if ch else None

    @property
    def Ex(self):
        ch = self._pick_electric()
        return ch.Fx if ch else None

    @property
    def Ey(self):
        ch = self._pick_electric()
        return ch.Fy if ch else None

    @property
    def Bz(self):
        ch = self._pick_magnetic()
        return ch.Fz if ch else None

    @property
    def Br(self):
        ch = self._pick_magnetic()
        return ch.Fr if ch else None

    @property
    def Bx(self):
        ch = self._pick_magnetic()
        return ch.Fx if ch else None

    @property
    def By(self):
        ch = self._pick_magnetic()
        return ch.Fy if ch else None

    @property
    def x(self):
        """First available x axis across channels (stat_B → rf_B → stat_E → rf_E)."""
        for ch_enum in (Channel.STAT_B, Channel.RF_B, Channel.STAT_E, Channel.RF_E):
            ch = self.channels.get(ch_enum)
            if ch is not None and ch.x is not None:
                return ch.x
        return None

    @property
    def y(self):
        for ch_enum in (Channel.STAT_B, Channel.RF_B, Channel.STAT_E, Channel.RF_E):
            ch = self.channels.get(ch_enum)
            if ch is not None and ch.y is not None:
                return ch.y
        return None

    @property
    def r(self):
        for ch_enum in (Channel.RF_E, Channel.STAT_E, Channel.RF_B, Channel.STAT_B):
            ch = self.channels.get(ch_enum)
            if ch is not None and ch.r is not None:
                return ch.r
        return None

    @property
    def norm_factor(self) -> float:
        """First non-1 norm across channels, or 1.0 if all unity."""
        for ch in self.channels.values():
            if ch.norm_factor != 1.0:
                return ch.norm_factor
        return 1.0
