"""Default-element factory for the Add Element flow.

When a user adds a new element to the lattice, we must provide
plausible placeholder values — this module returns a fresh element of
each supported type so the GUI can hand it to ``InsertCommand``.

``clone(element)`` returns a deep-copy used by the Duplicate action and
by the in-process clipboard (Cut/Copy/Paste).  Identity-preserving
undo/redo NEVER deep-copies; only the explicit clone path does.
"""
from __future__ import annotations

import copy
from typing import Callable

from linac_gen.elements.aperture import Aperture
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.foil import Foil
from linac_gen.elements.lattice_commands import (
    Adjust, AdjustBeamCentroid, AdjustBeamCurrent, AdjustBeamEmit,
    AdjustBeamTwiss, AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
    SetAchromat, SetAdv, SetBeamE0P0, SetBeamEnergy, SetBeamPhaseAdv,
    SetBeamPhaseError, SetGaussianCutOff, SetPosition, SetSeparation,
    SetSize, SetSizeMax, SetSizeMin, SetSyncPhase, SetTwiss,
)
from linac_gen.elements.marker import Marker
from linac_gen.elements.multipole import Multipole, Octupole, Sextupole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.thin_lens import ThinLens


# Each entry: (display name, zero-arg factory).
_FACTORIES: dict[str, Callable] = {
    "Drift":       lambda: Drift(name="DRIFT", length=100.0, aperture=10.0),
    "Quadrupole":  lambda: Quadrupole(name="QUAD",  length=200.0, gradient=10.0,
                                      aperture=10.0),
    "Solenoid":    lambda: Solenoid(name="SOL",   length=400.0, field=2.0,
                                    aperture=20.0),
    "Dipole":      lambda: Dipole(name="BEND",   angle=15.0, rho=1000.0,
                                  e1=0.0, e2=0.0, aperture=20.0),
    "RFGap":       lambda: RFGap(name="GAP",     voltage=0.5, phase=-30.0,
                                 frequency=162.5),
    "Steerer":     lambda: Steerer(name="STR",   bx_l=0.0, by_l=0.0),
    "Sextupole":   lambda: Sextupole(name="SEXT", k2L=0.0, aperture=10.0),
    "Octupole":    lambda: Octupole(name="OCT",  k3L=0.0, aperture=10.0),
    "Multipole":   lambda: Multipole(name="MULT", knl=[0.0, 0.0, 0.0],
                                     ksl=[0.0, 0.0, 0.0], aperture=10.0),
    "Aperture":    lambda: Aperture(name="APE",  dx=10.0, dy=10.0,
                                    aperture_type=Aperture.CIRCULAR),
    "Marker":      lambda: Marker(name="MARK"),
    "SpaceChargeComp": lambda: SpaceChargeComp(name="SCC", factor=0.0),
    "ThinLens":    lambda: ThinLens(name="LENS", fx=1000.0, fy=1000.0,
                                    aperture=10.0),
    "RfqCell":     lambda: RfqCell(name="RFQ", voltage_V=70_000.0, r0_mm=3.5,
                                   A10=0.05, modulation=1.5, length_mm=10.0,
                                   phi_s_deg=-30.0, cell_type=2),
    "FieldMap":    lambda: _placeholder_field_map(FieldMap),
    "FieldMap3D":  lambda: _placeholder_field_map(FieldMap3D),
    "Foil":        lambda: Foil(name="FOIL", material="C",
                                 thickness_ug_cm2=600.0),
    # ------------ TraceWin SET_* / ADJUST_* lattice commands ----------
    # Each command is a zero-cost passive element; the inspector
    # introspects the dataclass-style attributes for editing.
    "SET_SYNC_PHASE":     lambda: SetSyncPhase(name="SET_SYNC_PHASE"),
    "SET_BEAM_PHASE_ERROR": lambda: SetBeamPhaseError(
                                name="SET_BEAM_PHASE_ERROR", dphi_deg=0.0),
    "SET_BEAM_ENERGY":    lambda: SetBeamEnergy(name="SET_BEAM_ENERGY",
                                                energy_MeV=0.0),
    "SET_BEAM_E0P0":      lambda: SetBeamE0P0(name="SET_BEAM_E0P0"),
    "SET_GAUSS_CUTOFF":   lambda: SetGaussianCutOff(name="SET_GAUSS_CUTOFF"),
    "SET_TWISS":          lambda: SetTwiss(name="SET_TWISS"),
    "SET_POSITION":       lambda: SetPosition(name="SET_POSITION"),
    "SET_ACHROMAT":       lambda: SetAchromat(name="SET_ACHROMAT"),
    "SET_SIZE":           lambda: SetSize(name="SET_SIZE"),
    "SET_SIZE_MAX":       lambda: SetSizeMax(name="SET_SIZE_MAX"),
    "SET_SIZE_MIN":       lambda: SetSizeMin(name="SET_SIZE_MIN"),
    "SET_BEAM_PHASE_ADV": lambda: SetBeamPhaseAdv(name="SET_BEAM_PHASE_ADV"),
    "SET_SEPARATION":     lambda: SetSeparation(name="SET_SEPARATION"),
    "SET_ADV":            lambda: SetAdv(name="SET_ADV"),
    "ADJUST":             lambda: Adjust(name="ADJUST"),
    "ADJUST_STEERER":     lambda: AdjustSteerer(name="ADJUST_STEERER"),
    "ADJUST_STEERER_BX":  lambda: AdjustSteererBx(name="ADJUST_STEERER_BX"),
    "ADJUST_STEERER_BY":  lambda: AdjustSteererBy(name="ADJUST_STEERER_BY"),
    "ADJUST_BEAM_TWISS":  lambda: AdjustBeamTwiss(name="ADJUST_BEAM_TWISS"),
    "ADJUST_BEAM_CENTROID": lambda: AdjustBeamCentroid(
                                  name="ADJUST_BEAM_CENTROID"),
    "ADJUST_BEAM_EMIT":   lambda: AdjustBeamEmit(name="ADJUST_BEAM_EMIT"),
    "ADJUST_BEAM_CURRENT": lambda: AdjustBeamCurrent(
                                  name="ADJUST_BEAM_CURRENT"),
}


def _placeholder_field_map(cls):
    """FieldMap and FieldMap3D require external field-map files; the GUI
    can't conjure those.  Return ``None`` so the caller can pop a "load
    a .field file first" dialog instead of constructing a broken
    element."""
    return None  # callers handle this


def make_field_map_from_file(type_name: str, filepath: str,
                              base_name: str | None = None,
                              length_mm: float | None = None,
                              fm_type: int = 1):
    """Build a ``FieldMap`` (or ``FieldMap3D``) from a user-picked file.

    This is the GUI counterpart to :func:`make_default` for the two
    types that need an external file.  Reads the field map via
    :func:`linac_gen.io.field_map_reader.read_field_map` (1-D / 2-D
    cylindrical) or the 3-D Cartesian readers, then wraps it in the
    matching element class with placeholder amplitude / phase values
    (the user can edit these in the inspector).

    Parameters
    ----------
    type_name : {"FieldMap", "FieldMap3D"}
    filepath : str
        Path to the .edz / .csv (1-D/2-D) or one of the .bdx/.bdy/.bdz/
        .edx/.edy/.edz files in a 3-D triplet.  For 3-D maps the reader
        infers the prefix automatically.
    base_name : optional element name (defaults to filename stem in
        UPPERCASE).
    length_mm : optional explicit length; if ``None`` the field-map's
        physical extent is used.
    fm_type : passed through to the 1-D/2-D reader (1 = E, 2 = E
        cylindrical, 7 = E + B).
    """
    import os
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(filepath))[0].upper() or "FMAP"
    if type_name == "FieldMap":
        from linac_gen.io.field_map_reader import read_field_map
        from linac_gen.elements.field_map import FieldMap
        data = read_field_map(filepath, fm_type=fm_type)
        if length_mm is None:
            try:
                length_mm = float(data.z[-1] - data.z[0])
            except Exception:
                length_mm = 100.0
        return FieldMap(name=base_name, length=length_mm, field_data=data,
                        ke=1.0, kb=1.0, scale=1.0,
                        phase=0.0, frequency=0.0, n_steps=100)
    if type_name == "FieldMap3D":
        # The 3-D reader takes a *prefix* (".../map" → expects
        # ".../map.bdx", ".../map.bdy", etc.).  Strip a known suffix.
        from linac_gen.io.field_map_reader import read_3d_cart_EB
        from linac_gen.elements.field_map_3d import FieldMap3D
        prefix = filepath
        for ext in (".bdx", ".bdy", ".bdz", ".edx", ".edy", ".edz",
                    ".bsx", ".bsy", ".bsz"):
            if prefix.lower().endswith(ext):
                prefix = prefix[: -len(ext)]; break
        data = read_3d_cart_EB(prefix)
        if length_mm is None:
            try:
                length_mm = float(data.z[-1] - data.z[0])
            except Exception:
                length_mm = 100.0
        return FieldMap3D(name=base_name, length=length_mm, field_data=data,
                          ke=1.0, kb=1.0, scale=1.0,
                          phase=0.0, frequency=0.0, n_steps=100)
    raise ValueError(f"unsupported field-map type: {type_name!r}")


def supported_types() -> list[str]:
    """Return the ordered list of types the Add dialog should offer."""
    return list(_FACTORIES.keys())


def make_default(type_name: str):
    """Build a fresh element of ``type_name`` with placeholder defaults.

    Returns ``None`` for types that need external resources (FieldMap*).
    """
    factory = _FACTORIES.get(type_name)
    if factory is None:
        raise ValueError(f"unknown element type: {type_name!r}")
    return factory()


def clone(element):
    """Deep-copy an element for Duplicate / Cut / Copy / Paste.

    Names get a ``_copy`` suffix so the new element doesn't collide with
    its source under name-based lookup.
    """
    new = copy.deepcopy(element)
    if hasattr(new, "name") and isinstance(new.name, str):
        new.name = f"{new.name}_copy"
    return new
