"""Factory: pick the right FieldMap subclass for a decoded ``geom``.

A TraceWin ``geom`` may select one or more channels, each with its own
geometry digit.  This factory selects the element class:

* 1-D (digit 1), 2-D cyl (digit 4 or 5), 1-D quad gradient (digit 9):
  -> :class:`linac_gen.elements.field_map.FieldMap`.
* 3-D Cartesian (digit 7) for any enabled channel:
  -> :class:`linac_gen.elements.field_map_3d.FieldMap3D`.

Mixing a 3-D Cart channel with a non-3-D channel in a single FIELD_MAP
card is currently rejected (no element class handles it).  2-D Cartesian
(digit 6) is not yet supported by either element class and raises
``NotImplementedError``.
"""
from __future__ import annotations

from typing import Union

from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldMapData
from linac_gen.io.tracewin_geom import GeomCode, enabled_channels


def make_field_map_element(
    name: str,
    code: GeomCode,
    length_mm: float,
    field_data: FieldMapData,
    *,
    kb: float = 1.0,
    ke: float = 1.0,
    ki: float = 0.0,
    ka: int = 1,
    phase: float = 0.0,
    frequency: float = 0.0,
    aperture: float = 0.0,
    n_steps: int = 100,
    p_flag: int = 0,
    geom: int | None = None,
    field_file: str | None = None,
) -> Union[FieldMap, FieldMap3D]:
    """Build the right element for a decoded FIELD_MAP ``geom``.

    Parameters
    ----------
    name : str
        Element name.
    code : GeomCode
        Decoded geom (from :func:`~linac_gen.io.tracewin_geom.decode_geom`).
    length_mm : float
        Physical length in mm (from the FIELD_MAP card's ``length`` field).
    field_data : FieldMapData
        Pre-loaded field data.
    kb : float
        Magnetic-channel amplitude scale.
    ke : float
        Electric-channel amplitude scale.
    ki : float
        Space-charge compensation intensity (passed to FieldMap only).
    ka : int
        Aperture flag (passed to FieldMap only).
    phase : float
        RF phase offset in degrees.
    frequency : float
        RF frequency in MHz.
    aperture : float
        Aperture radius in mm; 0 = no check.
    n_steps : int
        Number of integration sub-steps.
    p_flag : int
        Phase reference flag.

    Returns
    -------
    FieldMap or FieldMap3D

    Raises
    ------
    ValueError
        If ``code`` decodes to no enabled channels, or if a mix of 3-D
        Cartesian and non-3-D channels is detected.
    NotImplementedError
        If digit 6 (2-D Cartesian) or digit 8 (3-D cylindrical) is present.
    """
    active = enabled_channels(code)
    if not active:
        # geom=0 => nothing to do; raise so the parser can skip with a
        # warning rather than add a meaningless element.
        raise ValueError(
            f"geom decodes to no enabled channels ({code}); refusing "
            "to build a FieldMap element."
        )
    digits = {d for _, d in active}

    if 6 in digits:
        raise NotImplementedError(
            "digit 6 (2-D Cartesian field) not yet supported by FieldMap "
            "or FieldMap3D (follow-up task)."
        )
    if 8 in digits:
        raise NotImplementedError(
            "digit 8 (3-D cylindrical) marked 'not implemented yet' in "
            "the TraceWin manual; not supported."
        )

    has_3d = 7 in digits
    has_non3d = digits - {7}

    if has_3d and has_non3d:
        raise ValueError(
            f"FIELD_MAP geom mixes 3-D Cart (digit 7) with non-3-D channels "
            f"{has_non3d} -- not supported by a single element class."
        )

    if has_3d:
        # FieldMap3D accepts ki (the .scc-normalisation flag).  ka is
        # forwarded only so the FIELD_MAP card round-trips faithfully — the
        # 3-D tracker does not yet act on TraceWin's per-element aperture-
        # shape flag; for 1-D / 2-D-cyl maps the FieldMap class below honours
        # both ki AND ka.
        return FieldMap3D(
            name=name,
            length=length_mm,
            field_data=field_data,
            scale=1.0,
            kb=kb,
            ke=ke,
            ki=ki,
            ka=ka,
            phase=phase,
            frequency=frequency,
            aperture=aperture,
            n_steps=n_steps,
            p_flag=p_flag,
            field_file=field_file,
            geom=geom,
        )

    # 1-D / 2-D cyl / 1-D quad-gradient: route to FieldMap.
    return FieldMap(
        name=name,
        length=length_mm,
        field_data=field_data,
        scale=1.0,
        kb=kb,
        ke=ke,
        ki=ki,
        ka=ka,
        phase=phase,
        frequency=frequency,
        aperture=aperture,
        n_steps=n_steps,
        p_flag=p_flag,
        field_file=field_file,
        geom=geom,
    )
