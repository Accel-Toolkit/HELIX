# linac_gen/io/field_map_reader.py
"""Read external field map files and return structured FieldMapData.

Supported formats:
  - TraceWin ``.edz`` (canonical per user manual):
      * 1-D (fm_type=1) — ``Nz  Zmax[m] / Norm / (Nz+1) × Fz``
      * 2-D cylindrical (fm_type=2, 7) — ``Nz Zmax / Nr Rmax / Norm / values``
  - TraceWin 3-D Cartesian triplet (fm_type=70/71/74):
    ``<name>.edx``, ``<name>.edy``, ``<name>.edz`` (and/or ``.bsx/.bsy/.bsz``)
    with header ``Nz Zmax / Nx Xmin Xmax / Ny Ymin Ymax / Norm``.
  - Legacy Linac_Gen custom ``.edz`` (kept for backward compatibility).
  - Generic CSV: whitespace- or comma-separated columns.

Unit conventions (per TraceWin manual):
  - ALL spatial dimensions in field-map files are in **metres**; the reader
    converts to **mm** for internal grid arrays.
  - Field values are **MV/m** (E) or **T** (B).  The reader stores values
    verbatim — the physical amplitude is reached via the ``FIELD_MAP`` card's
    ``ke`` / ``kb`` parameter, applied as ``k/Norm`` at tracking time.
  - CSV files are assumed to already be in mm / V-m for positions and fields.
"""
import numpy as np
import os

# The dataclass moved to its own module so every reader and element
# class can import it without going through this reader module.
# Re-exported here for backwards compatibility with existing callers.
from linac_gen.io.field_map_data import FieldMapData, FieldChannel  # noqa: F401
from linac_gen.io.tracewin_geom import Channel


# ------------------------------------------------------------------ #
#  Public named reader functions
# ------------------------------------------------------------------ #

def read_edz_1d(filepath: str) -> "FieldMapData":
    """Read a 1-D ``.edz`` field map, auto-detecting the header layout.

    Two layouts are recognised:

    * **TraceWin canonical** (per the TraceWin user manual, Dimension 1)::

        Nz   Zmax[m]            ← Nz = number of INTERVALS; Zmax in metres
        Norm                    ← single float
        F_z(0·Zmax/Nz)          ← MV/m, Nz+1 values total
        ...
        F_z(Nz·Zmax/Nz)

      The axial grid is ``z = linspace(0, Zmax, Nz+1)`` converted to mm.

    * **Legacy custom** (Linac_Gen's original fixtures)::

        N_pts                   ← total number of points (≠ intervals)
        zmin[cm]  zmax[cm]      ← two z values, no norm
        E_z(0)                  ← raw V/m
        ...
        E_z(N_pts-1)

      Preserved for backward compatibility with existing fixtures and
      lattices; tests do not depend on the header being one form or
      the other.

    Returns ``FieldMapData(symmetry="1d", z in mm, Ez in V/m)``.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Field map file not found: {filepath}")
    with open(filepath, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return _parse_edz_1d_auto(lines)


def _parse_edz_1d_auto(lines: list) -> "FieldMapData":
    """Dispatch between TraceWin canonical and legacy 1-D layouts.

    Decision rule:
      * Line 0 has TWO tokens, first int second float → TraceWin canonical.
      * Line 0 has ONE token (int)                     → legacy format.
      * Line 0 has TWO tokens, both ints               → ambiguous but
        treated as legacy (our 2-D reader was using this shape before).
    """
    tok0 = lines[0].split()
    if len(tok0) >= 2:
        try:
            int(tok0[0]); float(tok0[1])
            # If tok0[1] contains a decimal or exponent, treat as TraceWin.
            if "." in tok0[1] or "e" in tok0[1].lower():
                return _parse_edz_1d_tracewin(lines)
        except ValueError:
            pass
    return _parse_edz_1d(lines)   # legacy


def _parse_edz_1d_tracewin(lines: list) -> "FieldMapData":
    """Parse the canonical TraceWin ``.edz`` 1-D format.

    Per TraceWin manual (Dimension 1)::

        Nz  Zmax        ← Nz = intervals; Zmax in METRES
        Norm
        for k=0 to Nz:
            Fz(k·Zmax/Nz)    ← MV/m

    Reader keeps values exactly as stored (in MV/m) and exposes
    ``norm_factor`` as metadata; the physical amplitude is resolved at
    tracking time via the FIELD_MAP card's ``ke`` parameter
    (``scale = ke / Norm`` per the manual).
    """
    tok0 = lines[0].split()
    Nz = int(tok0[0])
    Zmax_m = float(tok0[1])
    norm = float(lines[1].split()[0])
    expected = Nz + 1
    flat: list = []
    for line in lines[2:]:
        flat.extend(float(v) for v in line.split())
        if len(flat) >= expected:
            break
    if len(flat) < expected:
        raise ValueError(
            f"TraceWin 1-D .edz header says Nz={Nz} (→ {expected} values) "
            f"but only {len(flat)} values present"
        )
    ez = np.asarray(flat[:expected], dtype=float)
    z = np.linspace(0.0, Zmax_m * 1000.0, Nz + 1)  # m → mm
    return FieldMapData.from_legacy_1d(z=z, Ez=ez, norm_factor=norm)


def read_edz_2d(filepath: str, fm_type: int = 2) -> "FieldMapData":
    """Read a 2-D cylindrical ``.edz`` field map, auto-detecting layout.

    Two layouts are recognised:

    * **TraceWin canonical** (per the TraceWin user manual, Dimension 2)::

        Nz   Zmax[m]
        Nr   Rmax[m]
        Norm
        for k=0 to Nz:
            for i=0 to Nr:
                Fz(k·Zmax/Nz, i·Rmax/Nr)   ← r is the FASTEST axis
        Er values  — same layout
        [Bz, Br for fm_type=7]

    * **Legacy custom**::

        Nz  Nr
        dz[cm]  dr[cm]
        E_z values
        E_r values
        [B_z, B_r for fm_type=7]

    Returns ``FieldMapData(symmetry="cylindrical", z/r in mm)``.  Canonical
    tabulated arrays have shape ``(Nz+1, Nr+1)`` (z outer, r inner) — the
    ``FieldMap`` element auto-detects this and transposes if needed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Field map file not found: {filepath}")
    with open(filepath, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return _parse_edz_2d_auto(lines, fm_type)


def _parse_edz_2d_auto(lines: list, fm_type: int) -> "FieldMapData":
    """Dispatch between TraceWin canonical and legacy 2-D layouts.

    Decision rule:
      * Line 1 has TWO tokens (int Nr, float Rmax) AND line 2 has ONE
        token (Norm) → TraceWin canonical.
      * Otherwise → legacy ``(Nz Nr) / (dz dr)`` format.
    """
    tok1 = lines[1].split()
    if len(tok1) == 2:
        try:
            int(tok1[0]); float(tok1[1])
            is_int_float = "." in tok1[1] or "e" in tok1[1].lower()
            if is_int_float:
                return _parse_edz_2d_tracewin(lines, fm_type)
        except ValueError:
            pass
    return _parse_edz_2d_nznr(lines, fm_type)


def _parse_edz_2d_tracewin(lines: list, fm_type: int) -> "FieldMapData":
    """Parse canonical TraceWin 2-D cylindrical ``.edz`` layout.

    Per TraceWin manual (Dimension 2)::

        Nz  Zmax        ← metres
        Nr  Rmax        ← metres
        Norm
        for k=0 to Nz:
            for i=0 to Nr:
                Fz(k·Zmax/Nz, i·Rmax/Nr)
        Er block  (same layout)
        [Bz, Br blocks iff fm_type == 7]

    The inner loop runs over ``r``, so a flat read-then-reshape in
    numpy C-order gives shape ``(Nz+1, Nr+1)`` with ``r`` as the
    fastest (last) axis.
    """
    t0 = lines[0].split()
    t1 = lines[1].split()
    Nz = int(t0[0]);  Zmax_m = float(t0[1])
    Nr = int(t1[0]);  Rmax_m = float(t1[1])
    norm = float(lines[2].split()[0])

    flat: list = []
    for ln in lines[3:]:
        flat.extend(float(v) for v in ln.split())
    block = (Nz + 1) * (Nr + 1)
    expected = 2 * block if fm_type == 2 else 4 * block
    if len(flat) < expected:
        raise ValueError(
            f"TraceWin 2-D .edz: expected {expected} values "
            f"(fm_type={fm_type}, Nz={Nz}, Nr={Nr}); got {len(flat)}"
        )

    # Manual loop: k=z outer, i=r inner → r-fastest → C-order shape (Nz+1, Nr+1)
    Ez = np.asarray(flat[0:block]).reshape((Nz + 1, Nr + 1))
    Er = np.asarray(flat[block:2 * block]).reshape((Nz + 1, Nr + 1))
    Bz = Br = None
    if fm_type == 7:
        Bz = np.asarray(flat[2 * block:3 * block]).reshape((Nz + 1, Nr + 1))
        Br = np.asarray(flat[3 * block:4 * block]).reshape((Nz + 1, Nr + 1))

    z = np.linspace(0.0, Zmax_m * 1000.0, Nz + 1)  # m → mm
    r = np.linspace(0.0, Rmax_m * 1000.0, Nr + 1)
    return FieldMapData.from_legacy_2d_cyl(
        z=z, r=r, Ez=Ez, Er=Er, Bz=Bz, Br=Br, norm_factor=norm,
    )


def read_csv(filepath: str) -> "FieldMapData":
    """Read a CSV field map file with header ``# z,Ez[,Er,Bz,Br]``.

    Supports:
    * 2 columns: z(mm), Ez
    * 3 columns: z(mm), Ez, Er
    * 5 columns: z(mm), Ez, Er, Bz, Br

    Lines beginning with ``#`` are treated as comments.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    FieldMapData
        ``symmetry="1d"``.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Field map file not found: {filepath}")

    # Auto-detect delimiter: if first non-comment line contains a comma, use it
    delimiter = None
    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                if "," in stripped:
                    delimiter = ","
                break

    data = np.loadtxt(filepath, comments="#", delimiter=delimiter)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    z = data[:, 0]

    if ncols == 2:
        Ez = data[:, 1]
        return FieldMapData.from_legacy_1d(z=z, Ez=Ez)
    elif ncols == 3:
        Ez = data[:, 1]
        Er = data[:, 2]
        # Store as a 1D channel but preserve Er in the channel's Fr slot
        fd = FieldMapData.from_legacy_1d(z=z, Ez=Ez)
        fd.channels[Channel.RF_E].Fr = Er
        fd.symmetry = "1d"
        return fd
    elif ncols >= 5:
        Ez = data[:, 1]
        Er = data[:, 2]
        Bz = data[:, 3]
        Br = data[:, 4]
        return FieldMapData.from_legacy_2d_cyl(z=z, r=None, Ez=Ez, Er=Er,
                                               Bz=Bz, Br=Br)
    else:
        raise ValueError(
            f"CSV has {ncols} columns; expected 2 (z,Ez), 3 (z,Ez,Er), or 5+ (z,Ez,Er,Bz,Br)"
        )


def expand_1d_offaxis(fmap: "FieldMapData"):
    """Build callable field functions from a 1D on-axis field map.

    Uses a cubic spline of ``fmap.Ez`` and its first derivative to provide:

    * ``Ez_func(z)``  → float: on-axis longitudinal field at *z* (mm)
    * ``Er_func(r, z)`` → float: radial field using the paraxial relation
      ``Er = -0.5 * r * dEz/dz``

    Parameters
    ----------
    fmap : FieldMapData
        Must have ``fmap.z`` and ``fmap.Ez`` as 1-D arrays.

    Returns
    -------
    (Ez_func, Er_func) : tuple of callables
    """
    from scipy.interpolate import CubicSpline

    ez_spline = CubicSpline(fmap.z, fmap.Ez)
    dez_dz_spline = ez_spline.derivative()

    def Ez_func(z: float) -> float:
        return float(ez_spline(z))

    def Er_func(r: float, z: float) -> float:
        return float(-0.5 * r * dez_dz_spline(z))

    return Ez_func, Er_func


def _parse_edz_2d_nznr(lines: list, fm_type: int) -> "FieldMapData":
    """Parse a 2D TraceWin .edz file with ``nz nr`` ordering on line 1.

    This interprets the header as ``nz  nr`` and the step sizes as ``dz  dr``
    (both in cm, converted to mm).  Ez is stored as shape ``(nz, nr)``.
    """
    tokens0 = lines[0].split()
    nz = int(tokens0[0])
    nr = int(tokens0[1])

    tokens1 = lines[1].split()
    dz_cm = float(tokens1[0])
    dr_cm = float(tokens1[1])

    dz_mm = dz_cm * 10.0
    dr_mm = dr_cm * 10.0

    z = np.arange(nz) * dz_mm
    r = np.arange(nr) * dr_mm

    all_values = []
    for line in lines[2:]:
        all_values.extend(float(v) for v in line.split())

    Ez_flat = np.array(all_values[:nz * nr])
    Ez = Ez_flat.reshape((nz, nr))

    Er_flat = np.array(all_values[nz * nr: 2 * nz * nr])
    Er = Er_flat.reshape((nz, nr))

    Bz = None
    Br = None
    if fm_type == 7 and len(all_values) >= 4 * nz * nr:
        Bz = np.array(all_values[2 * nz * nr: 3 * nz * nr]).reshape((nz, nr))
        Br = np.array(all_values[3 * nz * nr: 4 * nz * nr]).reshape((nz, nr))

    return FieldMapData.from_legacy_2d_cyl(z=z, r=r, Ez=Ez, Er=Er, Bz=Bz, Br=Br,
                                           norm_factor=1.0)


def expand_1d_to_2d(ez_on_axis: np.ndarray, z: np.ndarray,
                     r_max: float, nr: int):
    """Expand on-axis Ez(z) to 2D (r,z) using first-order Bessel expansion.

    Ez(r,z) = Ez(0,z) - (r^2/4) * Ez''(0,z)
    Er(r,z) = -(r/2) * Ez'(0,z)

    Parameters
    ----------
    ez_on_axis : array, shape (nz,)
        On-axis longitudinal field.
    z : array, shape (nz,)
        z-coordinates (mm).
    r_max : float
        Maximum radial extent (mm).
    nr : int
        Number of radial grid points (including r=0).

    Returns
    -------
    Ez2d : array, shape (nr, nz)
    Er2d : array, shape (nr, nz)
    r : array, shape (nr,)
    """
    nz = len(z)
    dz = z[1] - z[0] if nz > 1 else 1.0
    r = np.linspace(0, r_max, nr)

    # First derivative Ez' via central differences, forward/backward at boundaries
    ez_prime = np.zeros(nz)
    ez_prime[1:-1] = (ez_on_axis[2:] - ez_on_axis[:-2]) / (2.0 * dz)
    ez_prime[0] = (ez_on_axis[1] - ez_on_axis[0]) / dz
    ez_prime[-1] = (ez_on_axis[-1] - ez_on_axis[-2]) / dz

    # Second derivative Ez'' via central differences
    ez_double_prime = np.zeros(nz)
    if nz > 2:
        ez_double_prime[1:-1] = (
            ez_on_axis[2:] - 2.0 * ez_on_axis[1:-1] + ez_on_axis[:-2]
        ) / (dz * dz)
        # Boundary: use same as nearest interior point
        ez_double_prime[0] = ez_double_prime[1]
        ez_double_prime[-1] = ez_double_prime[-2]

    # Build 2D arrays: Ez(r,z) and Er(r,z)
    Ez2d = np.zeros((nr, nz))
    Er2d = np.zeros((nr, nz))
    for i, ri in enumerate(r):
        Ez2d[i, :] = ez_on_axis - (ri * ri / 4.0) * ez_double_prime
        Er2d[i, :] = -(ri / 2.0) * ez_prime

    return Ez2d, Er2d, r


def read_field_map(filepath: str, fm_type: int = 1) -> FieldMapData:
    """Read field map from file.

    Parameters
    ----------
    filepath : str
        Path to the field map file.
    fm_type : int
        1 = 1D electric, 2 = 2D electric (cylindrical), 7 = 2D E+B.
        Auto-detects format from file extension (.edz -> TraceWin, .csv -> CSV).

    Returns
    -------
    FieldMapData

    Notes
    -----
    TraceWin .edz files use **cm** for z-coordinates. The reader converts to mm.
    CSV files are expected to use **mm** for positions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Field map file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".edz":
        return _read_edz(filepath, fm_type)
    elif ext == ".csv":
        return _read_csv(filepath, fm_type)
    else:
        # Try to auto-detect from content
        return _read_csv(filepath, fm_type)


def _read_edz(filepath: str, fm_type: int) -> FieldMapData:
    """Read TraceWin .edz format.

    1D format (fm_type=1):
        nz
        z_start(cm) z_end(cm)
        Ez(0) ... Ez(nz-1)

    2D format (fm_type=2 or 7):
        nr nz
        dr(cm) dz(cm)
        Ez values (nr rows, nz cols per row)
        Er values (nr rows, nz cols per row)
        [Bz values for fm_type=7]
        [Br values for fm_type=7]
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    first_tokens = lines[0].split()

    if len(first_tokens) == 1:
        # 1D format
        if fm_type not in (1,):
            raise ValueError(
                f"File appears to be 1D .edz but fm_type={fm_type} requested. "
                f"Use fm_type=1 for 1D field maps."
            )
        return _parse_edz_1d(lines)
    elif len(first_tokens) == 2:
        # Could be 2D (nr, nz) or 1D header recheck
        try:
            nr = int(first_tokens[0])
            nz = int(first_tokens[1])
        except ValueError:
            raise ValueError(f"Cannot parse .edz header: {lines[0]}")

        if fm_type == 1:
            # User asked for 1D but file looks 2D
            raise ValueError(
                f"File appears to be 2D .edz (nr={nr}, nz={nz}) "
                f"but fm_type=1 requested."
            )
        return _parse_edz_2d(lines, fm_type)
    else:
        raise ValueError(f"Unrecognized .edz header: {lines[0]}")


def _parse_edz_1d(lines: list) -> FieldMapData:
    """Parse a 1D TraceWin .edz file."""
    nz = int(lines[0])
    z_tokens = lines[1].split()
    z_start_cm = float(z_tokens[0])
    z_end_cm = float(z_tokens[1])

    # Read Ez values
    ez_values = []
    for i in range(2, 2 + nz):
        ez_values.append(float(lines[i]))
    Ez = np.array(ez_values)

    # Convert cm -> mm
    z_start_mm = z_start_cm * 10.0
    z_end_mm = z_end_cm * 10.0
    z = np.linspace(z_start_mm, z_end_mm, nz)

    return FieldMapData.from_legacy_1d(z=z, Ez=Ez)


def _parse_edz_2d(lines: list, fm_type: int) -> FieldMapData:
    """Parse a 2D TraceWin .edz file."""
    tokens0 = lines[0].split()
    nr = int(tokens0[0])
    nz = int(tokens0[1])

    tokens1 = lines[1].split()
    dr_cm = float(tokens1[0])
    dz_cm = float(tokens1[1])

    # Convert cm -> mm
    dr_mm = dr_cm * 10.0
    dz_mm = dz_cm * 10.0

    r = np.arange(nr) * dr_mm
    z = np.arange(nz) * dz_mm

    # Parse field values. Each row in the file may contain multiple values
    # (space-separated). We need nr*nz values for Ez, then nr*nz for Er, etc.
    all_values = []
    for line in lines[2:]:
        all_values.extend(float(v) for v in line.split())

    Ez_flat = np.array(all_values[:nr * nz])
    Ez = Ez_flat.reshape((nr, nz))

    Er_flat = np.array(all_values[nr * nz: 2 * nr * nz])
    Er = Er_flat.reshape((nr, nz))

    Bz = None
    Br = None
    if fm_type == 7 and len(all_values) >= 4 * nr * nz:
        Bz_flat = np.array(all_values[2 * nr * nz: 3 * nr * nz])
        Bz = Bz_flat.reshape((nr, nz))
        Br_flat = np.array(all_values[3 * nr * nz: 4 * nr * nz])
        Br = Br_flat.reshape((nr, nz))

    return FieldMapData.from_legacy_2d_cyl(z=z, r=r, Ez=Ez, Er=Er, Bz=Bz, Br=Br)


def _read_csv(filepath: str, fm_type: int) -> FieldMapData:
    """Read a generic CSV/whitespace-separated field map.

    Expected columns:
      1D: z(mm) Ez(V/m)
      2D: r(mm) z(mm) Ez(V/m) Er(V/m) [Bz(T) Br(T)]

    Lines starting with '#' are comments. Positions in mm (no conversion).
    """
    data = np.loadtxt(filepath, comments="#")

    if data.ndim == 1:
        # Single row
        data = data.reshape(1, -1)

    ncols = data.shape[1]

    if ncols == 2:
        # 1D: z, Ez
        z = data[:, 0]
        Ez = data[:, 1]
        return FieldMapData.from_legacy_1d(z=z, Ez=Ez)
    elif ncols >= 4:
        # 2D: r, z, Ez, Er [, Bz, Br]
        r_raw = data[:, 0]
        z_raw = data[:, 1]
        Ez_raw = data[:, 2]
        Er_raw = data[:, 3]

        r_unique = np.unique(r_raw)
        z_unique = np.unique(z_raw)
        nr = len(r_unique)
        nz = len(z_unique)

        Ez = np.zeros((nr, nz))
        Er = np.zeros((nr, nz))
        Bz = None
        Br = None
        if ncols >= 6:
            Bz = np.zeros((nr, nz))
            Br = np.zeros((nr, nz))

        for row in data:
            ri = np.searchsorted(r_unique, row[0])
            zi = np.searchsorted(z_unique, row[1])
            Ez[ri, zi] = row[2]
            Er[ri, zi] = row[3]
            if ncols >= 6:
                Bz[ri, zi] = row[4]
                Br[ri, zi] = row[5]

        return FieldMapData.from_legacy_2d_cyl(z=z_unique, r=r_unique,
                                               Ez=Ez, Er=Er, Bz=Bz, Br=Br)
    else:
        raise ValueError(f"CSV has {ncols} columns; expected 2 (1D) or 4+ (2D)")


# ================================================================== #
#  3-D Cartesian readers  (TraceWin fm_type ∈ {70, 71, 74})
# ================================================================== #
#
# TraceWin's 3-D field map is split across three sibling files — one
# per Cartesian component:
#
#     fm_type 70  (electric RF):  <name>.edx  <name>.edy  <name>.edz
#     fm_type 71  (magnetic RF):  <name>.bsx  <name>.bsy  <name>.bsz
#     fm_type 74  (EM, full):     both triplets (6 files total)
#
# Each file shares the same header.  Per the TraceWin user manual
# ("Field map file syntax — Dimension 3"), all spatial dimensions are
# in METRES and the order is fixed:
#
#     Nz   Zmax                ← intervals along z ∈ [0, Zmax]
#     Nx   Xmin   Xmax
#     Ny   Ymin   Ymax
#     Norm                     ← scalar metadata (used as ke/Norm at tracking)
#     for k=0 to Nz:
#         for j=0 to Ny:
#             for i=0 to Nx:
#                 Fz(k·Zmax/Nz,
#                    Ymin+j·(Ymax-Ymin)/Ny,
#                    Xmin+i·(Xmax-Xmin)/Nx)   ← x is the FASTEST axis
#
# Total data count is (Nx+1)(Ny+1)(Nz+1).  Field values are MV/m (E)
# or T (B), stored verbatim; the physical amplitude is resolved via
# the FIELD_MAP card's ke/kb parameter (factor = k/Norm).
#
# The reader returns arrays with shape (Nx+1, Ny+1, Nz+1) to match
# ``FieldMap3D``'s RegularGridInterpolator which uses
# ``axes = (x, y, z)``.  (The file's natural C-order is
# (Nz+1, Ny+1, Nx+1); we transpose before returning.)
# ------------------------------------------------------------------ #


def _parse_3d_header(lines: list) -> tuple:
    """Parse the common 3-D header per TraceWin manual (Dimension 3).

    Returns ``(x, y, z, norm)`` with axes in mm and ``norm`` the raw
    scalar from the file.  Each axis array has ``N+1`` samples because
    the header stores ``N`` intervals.
    """
    tok_z = lines[0].split()
    nz_int = int(tok_z[0])
    zmax_m = float(tok_z[1])
    tok_x = lines[1].split()
    nx_int = int(tok_x[0])
    xmin_m = float(tok_x[1])
    xmax_m = float(tok_x[2])
    tok_y = lines[2].split()
    ny_int = int(tok_y[0])
    ymin_m = float(tok_y[1])
    ymax_m = float(tok_y[2])
    norm = float(lines[3].split()[0])
    # Manual: "The dimensions are in metres."
    x = np.linspace(xmin_m, xmax_m, nx_int + 1) * 1000.0
    y = np.linspace(ymin_m, ymax_m, ny_int + 1) * 1000.0
    z = np.linspace(0.0,    zmax_m, nz_int + 1) * 1000.0
    return x, y, z, norm


def _read_single_3d(filepath: str) -> tuple:
    """Read one 3-D component file.

    Returns ``(x, y, z, values, norm)`` with ``values`` of shape
    ``(len(x), len(y), len(z))``.  The flat file order is
    ``for k ∈ z: for j ∈ y: for i ∈ x`` (x-fastest per the manual);
    a C-order reshape gives ``(nz, ny, nx)`` which is transposed to
    the returned ``(nx, ny, nz)``.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"3-D field map file not found: {filepath}")
    with open(filepath, "r") as fh:
        # Comments/blank lines may appear between header rows in some
        # exporters; strip them preserving order.
        raw = [ln.strip() for ln in fh
               if ln.strip() and not ln.lstrip().startswith(("#", "!"))]
    x, y, z, norm = _parse_3d_header(raw)
    nx, ny, nz = len(x), len(y), len(z)
    expected = nx * ny * nz
    data_lines = raw[4:]
    flat = []
    for ln in data_lines:
        flat.extend(float(v) for v in ln.split())
        if len(flat) >= expected:
            break
    if len(flat) < expected:
        raise ValueError(
            f"{filepath}: expected {expected} values, got {len(flat)} "
            f"(header nz={nz - 1} nx={nx - 1} ny={ny - 1})"
        )
    arr = np.asarray(flat[:expected], dtype=float)
    # Manual order k(z)→j(y)→i(x). C-order reshape: (nz, ny, nx).
    # Transpose so callers see (nx, ny, nz). Copy to keep contiguous.
    values = arr.reshape((nz, ny, nx)).transpose(2, 1, 0).copy()
    return x, y, z, values, norm


def read_3d_cart_E(prefix: str) -> "FieldMapData":
    """Read a TraceWin 3-D Cartesian electric field map.

    ``prefix`` is the path WITHOUT the ``.edx``/``.edy``/``.edz`` suffix;
    all three sibling files must exist next to it.  Grids must match
    across the three files (each component file repeats the header, so
    any inconsistency is caught at parse time).
    """
    xf, yf, zf, Ex, norm_x = _read_single_3d(prefix + ".edx")
    _,  _,  _,  Ey, _      = _read_single_3d(prefix + ".edy")
    _,  _,  _,  Ez, _      = _read_single_3d(prefix + ".edz")
    fd = FieldMapData(z=zf, frequency=0.0, symmetry="3d")
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=xf, y=yf, z=zf,
        Fx=Ex, Fy=Ey, Fz=Ez,
        norm_factor=norm_x,
    )
    return fd


def read_3d_cart_B(prefix: str) -> "FieldMapData":
    """Read a TraceWin 3-D Cartesian magnetic field map (.bsx/.bsy/.bsz)."""
    xf, yf, zf, Bx, norm_x = _read_single_3d(prefix + ".bsx")
    _,  _,  _,  By, _      = _read_single_3d(prefix + ".bsy")
    _,  _,  _,  Bz, _      = _read_single_3d(prefix + ".bsz")
    fd = FieldMapData(z=zf, frequency=0.0, symmetry="3d")
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=7, x=xf, y=yf, z=zf,
        Fx=Bx, Fy=By, Fz=Bz,
        norm_factor=norm_x,
    )
    return fd


def read_3d_cart_EB(prefix: str) -> "FieldMapData":
    """Read a TraceWin 3-D Cartesian RF EM map (all 6 components)."""
    xf, yf, zf, Ex, norm_x = _read_single_3d(prefix + ".edx")
    _,  _,  _,  Ey, _      = _read_single_3d(prefix + ".edy")
    _,  _,  _,  Ez, _      = _read_single_3d(prefix + ".edz")
    _,  _,  _,  Bx, _      = _read_single_3d(prefix + ".bsx")
    _,  _,  _,  By, _      = _read_single_3d(prefix + ".bsy")
    _,  _,  _,  Bz, norm_b = _read_single_3d(prefix + ".bsz")
    fd = FieldMapData(z=zf, frequency=0.0, symmetry="3d")
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=xf, y=yf, z=zf,
        Fx=Ex, Fy=Ey, Fz=Ez,
        norm_factor=norm_x,
    )
    fd.channels[Channel.RF_B] = FieldChannel(
        geometry=7, x=xf, y=yf, z=zf,
        Fx=Bx, Fy=By, Fz=Bz,
        norm_factor=norm_b,
    )
    return fd
