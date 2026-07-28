"""TraceWin ``geom`` 5-digit decoder.

See the TraceWin user manual, section *FIELD_MAP*:

    geom = aper·10⁴ + rf_B·10³ + rf_E·10² + stat_B·10 + stat_E

where each digit 0..9 describes the *geometry* of the corresponding
field channel (0=absent, 1=1D, 4=2D cyl E-type, 5=2D cyl B-type,
6=2D Cart, 7=3D Cart, 8=3D cyl (N/A), 9=1D G(z) quad gradient).

A negative ``geom`` means "do the off-axis expansion at 2nd order
instead of 1st".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeomCode:
    """Decomposed TraceWin ``geom`` parameter.

    Each field is the single-digit geometry code of one channel.
    ``second_order`` captures the sign of the original geom.
    """
    stat_E: int
    stat_B: int
    rf_E:   int
    rf_B:   int
    aper:   int
    second_order: bool


def decode_geom(geom: int) -> GeomCode:
    """Decode the FIELD_MAP ``geom`` integer into a :class:`GeomCode`."""
    second_order = geom < 0
    g = abs(int(geom))
    return GeomCode(
        stat_E=g % 10,
        stat_B=(g // 10) % 10,
        rf_E=(g // 100) % 10,
        rf_B=(g // 1000) % 10,
        aper=(g // 10000) % 10,
        second_order=second_order,
    )


from enum import Enum
from typing import List, Tuple


class Channel(Enum):
    """One of the four field channels a FIELD_MAP can contain."""
    STAT_E = ("e", "s")
    STAT_B = ("b", "s")
    RF_E   = ("e", "d")
    RF_B   = ("b", "d")

    @property
    def field_letter(self) -> str:
        """``'e'`` for electric channels, ``'b'`` for magnetic."""
        return self.value[0]

    @property
    def type_letter(self) -> str:
        """``'s'`` for static channels, ``'d'`` (dynamic) for RF."""
        return self.value[1]

    @property
    def is_static(self) -> bool:
        return self.type_letter == "s"

    @property
    def is_rf(self) -> bool:
        return self.type_letter == "d"

    @property
    def is_electric(self) -> bool:
        return self.field_letter == "e"

    @property
    def is_magnetic(self) -> bool:
        return self.field_letter == "b"


def component_files(channel: Channel, digit: int) -> List[str]:
    """Return the file-extension list (including leading dot) for one
    (channel, geometry-digit) pair per the TraceWin manual.

    Raises
    ------
    ValueError
        If the digit is not physically meaningful in this channel
        (e.g. magnetic-type 2-D cyl in an electric channel, or
        1-D G(z) anywhere other than the static-magnetic slot).
    NotImplementedError
        For digit 8 (3-D cylindrical), flagged by the TraceWin manual
        itself as "not implemented yet".
    """
    if digit == 0:
        return []
    fl, tl = channel.field_letter, channel.type_letter

    if digit == 1:
        return [f".{fl}{tl}z"]
    if digit == 4:
        if not channel.is_electric:
            raise ValueError(
                f"digit 4 (2-D cyl E-type) not valid in {channel.name} "
                f"(only stat_E / rf_E)"
            )
        base = [f".{fl}{tl}r", f".{fl}{tl}z"]
        if channel.is_rf:
            base.append(".bdq")           # TM mode Bθ
        return base
    if digit == 5:
        if not channel.is_magnetic:
            raise ValueError(
                f"digit 5 (2-D cyl B-type) not valid in {channel.name} "
                f"(only stat_B / rf_B)"
            )
        base = [f".{fl}{tl}r", f".{fl}{tl}z"]
        if channel.is_rf:
            base.append(".edq")           # TE mode Eθ
        return base
    if digit == 6:
        return [f".{fl}{tl}x", f".{fl}{tl}y"]
    if digit == 7:
        return [f".{fl}{tl}x", f".{fl}{tl}y", f".{fl}{tl}z"]
    if digit == 8:
        raise NotImplementedError(
            "digit 8 (3-D cyl) is marked 'not implemented yet' in the "
            "TraceWin manual and is not supported."
        )
    if digit == 9:
        if channel is not Channel.STAT_B:
            raise ValueError(
                f"digit 9 (1-D quad G(z)) only valid in stat_B channel, "
                f"got {channel.name}"
            )
        return [f".{fl}{tl}z"]
    raise ValueError(f"unknown geometry digit: {digit}")


def enabled_channels(code: "GeomCode") -> List[Tuple[Channel, int]]:
    """Return the (channel, digit) pairs with a non-zero digit.

    Fixed iteration order: STAT_E, STAT_B, RF_E, RF_B.  Aperture
    digit is handled separately by the parser (opens ``.ouv``).
    """
    out: List[Tuple[Channel, int]] = []
    for ch, d in ((Channel.STAT_E, code.stat_E),
                  (Channel.STAT_B, code.stat_B),
                  (Channel.RF_E,   code.rf_E),
                  (Channel.RF_B,   code.rf_B)):
        if d != 0:
            out.append((ch, d))
    return out
