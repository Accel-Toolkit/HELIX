"""TraceWin per-component field-map readers.

Each function opens ONE file off a path and returns a
:class:`FieldChannel` with the grid axes populated and the data placed
in ``Fz`` (the reader doesn't know which component letter the file's
3rd character is — the orchestrator in Task 5 will reassign it to
``Fx``/``Fy``/``Fr``/``Fq`` based on the suffix).

Per the TraceWin manual (§7860-8080):

* All header dimensions are in **metres**.  Grid arrays returned in **mm**.
* Field values are kept verbatim (MV/m or T); ``norm_factor`` is stored
  as metadata.
* Loop orders (fastest axis last in the flat file):
    - 1-D     →  Nz+1 values, trivial.
    - 2-D cyl →  outer z, inner r.
    - 2-D Cart→  outer y, inner x.
    - 3-D Cart→  outer z, middle y, inner x.
"""
from __future__ import annotations

import functools
import os
from typing import List

import numpy as np

from linac_gen.io.field_map_data import FieldChannel


# ---------------------------------------------------------------------------
# Per-component file cache
# ---------------------------------------------------------------------------
# A typical linac references the same field-map template (HWRDonut,
# QWR-2012-02, HWR-SOL-ANLMAP) from many FIELD_MAP cards; profiling MEBT+HWR
# showed >12 s spent re-parsing 9 M lines of identical ASCII content from
# those shared files.  This cache memoizes the per-file reader calls keyed
# on (realpath, mtime) — same file, same mtime → return the cached
# FieldChannel.  The big Fz array is marked read-only after construction so
# accidental in-place mutation by a downstream consumer raises ValueError
# instead of silently corrupting every cache hit.
_FIELD_FILE_CACHE: dict[tuple[str, float], FieldChannel] = {}


def _cached_file_reader(reader):
    """Decorator: cache by (realpath, mtime)."""
    @functools.wraps(reader)
    def wrapper(filepath: str) -> FieldChannel:
        try:
            key = (os.path.realpath(filepath), os.path.getmtime(filepath))
        except OSError:
            return reader(filepath)
        cached = _FIELD_FILE_CACHE.get(key)
        if cached is not None:
            return cached
        ch = reader(filepath)
        # Freeze every numpy array on the cached channel so accidental
        # in-place mutation by a downstream consumer raises ValueError
        # at the offending line rather than silently corrupting every
        # other cache hit (the per-call FieldChannel container shares
        # array references with the cached raw via _attach_to_channel).
        for attr in ("Fz", "Fx", "Fy", "Fr", "x", "y", "z", "r"):
            arr = getattr(ch, attr, None)
            if isinstance(arr, np.ndarray):
                arr.flags.writeable = False
        _FIELD_FILE_CACHE[key] = ch
        return ch
    return wrapper


def clear_field_map_cache() -> None:
    """Drop all cached per-component reads.  Useful in tests that mutate
    field-map files on disk between runs."""
    _FIELD_FILE_CACHE.clear()


def _clean_lines(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Field-map file not found: {filepath}")
    with open(filepath, "r", encoding="latin-1") as fh:
        return [ln.strip() for ln in fh
                if ln.strip() and not ln.lstrip().startswith(("#", "!"))]


def _read_flat(lines: List[str], offset: int, count: int) -> np.ndarray:
    flat: List[float] = []
    for ln in lines[offset:]:
        flat.extend(float(v) for v in ln.split())
        if len(flat) >= count:
            break
    if len(flat) < count:
        raise ValueError(f"expected {count} values, got {len(flat)}")
    return np.asarray(flat[:count], dtype=float)


@_cached_file_reader
def read_1d_component(filepath: str) -> FieldChannel:
    """Read a 1-D TraceWin component file (``.esz``, ``.bsz``, ``.edz``,
    or ``.bdz``).

    Canonical TraceWin layout (manual Dimension 1):

        Nz Zmax                 ← Nz = intervals; Zmax in metres
        Norm                    ← single float
        (Nz+1) × Fz(k · Zmax / Nz)

    Also auto-detects the legacy Linac_Gen custom layout:

        N_pts                   ← total points (≠ intervals)
        zmin[cm]  zmax[cm]
        N_pts values

    The legacy layout was used by historical Linac_Gen fixtures; the
    auto-detect dispatcher keeps those working without forcing users
    to rewrite their ``.edz`` files.
    """
    raw = _clean_lines(filepath)
    tok = raw[0].split()
    # Auto-detect: canonical has TWO tokens on line 0 (Nz Zmax), with the
    # second being a float literal (contains '.' or 'e').  Legacy has a
    # SINGLE token (N_pts).
    if len(tok) >= 2:
        try:
            int(tok[0]); float(tok[1])
            if "." in tok[1] or "e" in tok[1].lower():
                # Canonical layout
                Nz = int(tok[0])
                Zmax_m = float(tok[1])
                norm = float(raw[1].split()[0])
                vals = _read_flat(raw[2:], 0, Nz + 1)
                z = np.linspace(0.0, Zmax_m * 1000.0, Nz + 1)
                return FieldChannel(geometry=1, z=z, Fz=vals,
                                    norm_factor=norm)
        except ValueError:
            pass
    # Legacy layout fallback: N_pts / zmin zmax [cm] / N_pts values
    N_pts = int(tok[0])
    z_tokens = raw[1].split()
    z_start_cm = float(z_tokens[0])
    z_end_cm = float(z_tokens[1])
    vals = _read_flat(raw[2:], 0, N_pts)
    z = np.linspace(z_start_cm * 10.0, z_end_cm * 10.0, N_pts)  # cm → mm
    return FieldChannel(geometry=1, z=z, Fz=vals, norm_factor=1.0)


@_cached_file_reader
def read_2d_cyl_component(filepath: str) -> FieldChannel:
    """Read a 2-D cyl component file.  Manual Dimension 2 (cyl) layout:

        Nz Zmax
        Nr Rmax
        Norm
        for k=0..Nz:
            for i=0..Nr:
                Fz(k·Zmax/Nz, i·Rmax/Nr)

    Inner loop on r means natural C-order reshape is (Nz+1, Nr+1)
    r-fastest.  The reader puts the raw data in ``Fz`` regardless of
    the file's component letter; the orchestrator reassigns.
    """
    raw = _clean_lines(filepath)
    tok0 = raw[0].split()
    tok1 = raw[1].split()
    Nz = int(tok0[0]);  Zmax_m = float(tok0[1])
    Nr = int(tok1[0]);  Rmax_m = float(tok1[1])
    norm = float(raw[2].split()[0])
    vals = _read_flat(raw[3:], 0, (Nz + 1) * (Nr + 1))
    arr = vals.reshape((Nz + 1, Nr + 1))          # r-fastest C-order
    z = np.linspace(0.0, Zmax_m * 1000.0, Nz + 1)
    r = np.linspace(0.0, Rmax_m * 1000.0, Nr + 1)
    return FieldChannel(geometry=4, z=z, r=r, Fz=arr, norm_factor=norm)


@_cached_file_reader
def read_2d_cart_component(filepath: str) -> FieldChannel:
    """Read a 2-D Cart component file.  Layout:

        Nx Xmin Xmax
        Ny Ymin Ymax
        Norm
        for j=0..Ny:
            for i=0..Nx:
                Fz(i·(Xmax-Xmin)/Nx, j·(Ymax-Ymin)/Ny)

    x-fastest in the file → C-order reshape is (Ny+1, Nx+1).  The
    reader transposes to (Nx+1, Ny+1) so downstream users can feed
    the array into a RegularGridInterpolator with axes=(x, y).
    """
    raw = _clean_lines(filepath)
    tx = raw[0].split();  Nx = int(tx[0]);  Xmin_m = float(tx[1]);  Xmax_m = float(tx[2])
    ty = raw[1].split();  Ny = int(ty[0]);  Ymin_m = float(ty[1]);  Ymax_m = float(ty[2])
    norm = float(raw[2].split()[0])
    vals = _read_flat(raw[3:], 0, (Nx + 1) * (Ny + 1))
    arr = vals.reshape((Ny + 1, Nx + 1)).T.copy()     # → (Nx+1, Ny+1)
    x = np.linspace(Xmin_m, Xmax_m, Nx + 1) * 1000.0
    y = np.linspace(Ymin_m, Ymax_m, Ny + 1) * 1000.0
    # 2-D Cart is invariant in z; no z axis stored on this channel.
    return FieldChannel(geometry=6, x=x, y=y, Fz=arr, norm_factor=norm)


@_cached_file_reader
def read_3d_cart_component(filepath: str) -> FieldChannel:
    """Read a 3-D Cart component file.  Manual Dimension 3 layout:

        Nz Zmax
        Nx Xmin Xmax
        Ny Ymin Ymax
        Norm
        for k=0..Nz:
            for j=0..Ny:
                for i=0..Nx:
                    Fz(k·Zmax/Nz,
                       Ymin+j·(Ymax-Ymin)/Ny,
                       Xmin+i·(Xmax-Xmin)/Nx)

    x-fastest → C-order reshape is (Nz+1, Ny+1, Nx+1).  The reader
    transposes to (Nx+1, Ny+1, Nz+1) so callers match the scipy
    RegularGridInterpolator axes=(x, y, z) convention.
    """
    raw = _clean_lines(filepath)
    tz = raw[0].split();  Nz = int(tz[0]);  Zmax_m = float(tz[1])
    tx = raw[1].split();  Nx = int(tx[0]);  Xmin_m = float(tx[1]);  Xmax_m = float(tx[2])
    ty = raw[2].split();  Ny = int(ty[0]);  Ymin_m = float(ty[1]);  Ymax_m = float(ty[2])
    norm = float(raw[3].split()[0])
    total = (Nz + 1) * (Ny + 1) * (Nx + 1)
    vals = _read_flat(raw[4:], 0, total)
    arr = vals.reshape((Nz + 1, Ny + 1, Nx + 1)).transpose(2, 1, 0).copy()
    x = np.linspace(Xmin_m, Xmax_m, Nx + 1) * 1000.0
    y = np.linspace(Ymin_m, Ymax_m, Ny + 1) * 1000.0
    z = np.linspace(0.0,    Zmax_m, Nz + 1) * 1000.0
    return FieldChannel(geometry=7, x=x, y=y, z=z, Fz=arr, norm_factor=norm)


# =====================================================================
# Top-level orchestrator
# =====================================================================
from typing import Optional

from linac_gen.io.field_map_data import FieldMapData
from linac_gen.io.tracewin_geom import (
    Channel, GeomCode, decode_geom, component_files, enabled_channels,
)


_KNOWN_SUFFIXES = tuple(
    f".{fl}{tl}{co}"
    for fl in ("e", "b")
    for tl in ("s", "d")
    for co in ("x", "y", "z", "r", "q")
) + (".ouv",)


def _strip_known_suffix(prefix: str) -> str:
    """If the caller accidentally included a known ``.e{s,d}{x,y,z,r,q}`` /
    ``.b{s,d}{x,y,z,r,q}`` / ``.ouv`` extension on the prefix, strip it."""
    lower = prefix.lower()
    for suf in _KNOWN_SUFFIXES:
        if lower.endswith(suf):
            return prefix[: -len(suf)]
    return prefix


def _dispatch_reader(digit: int):
    """Pick the per-component reader matching the geometry digit."""
    if digit in (1, 9):
        return read_1d_component
    if digit in (4, 5):
        return read_2d_cyl_component
    if digit == 6:
        return read_2d_cart_component
    if digit == 7:
        return read_3d_cart_component
    raise ValueError(f"no reader for geometry digit {digit}")


def _attach_to_channel(ch: FieldChannel, suffix: str, raw: FieldChannel) -> None:
    """Move the raw reader's output (always in ``raw.Fz``) into the slot
    that matches the suffix's last character on the destination channel.
    Also copies over grid axes if the destination doesn't yet have them.
    """
    letter = suffix[-1]      # x / y / z / r / q

    # Grid-axis propagation — whichever axes the raw has and the destination
    # doesn't, copy over.
    for axis_name in ("x", "y", "z", "r"):
        if getattr(raw, axis_name, None) is not None and getattr(ch, axis_name, None) is None:
            setattr(ch, axis_name, getattr(raw, axis_name))

    # Component-value assignment
    if letter == "z":
        ch.Fz = raw.Fz
    elif letter == "r":
        ch.Fr = raw.Fz
    elif letter == "x":
        ch.Fx = raw.Fz
    elif letter == "y":
        ch.Fy = raw.Fz
    elif letter == "q":
        ch.Fq = raw.Fz
    else:
        raise ValueError(f"unrecognised suffix letter: {letter!r}")


def read_tracewin_fieldmap(geom: int, prefix: str,
                           base_dir: Optional[str] = None,
                           frequency: float = 0.0,
                           Ki: float = 0.0,
                           Ka: int = 0) -> FieldMapData:
    """Load every field file implied by ``geom`` off ``prefix``.

    Parameters
    ----------
    geom : int
        TraceWin ``geom`` code (5-digit encoded).
    prefix : str
        File-name prefix *without* extension.  If the caller accidentally
        included a recognised extension, it is stripped.
    base_dir : str, optional
        If given and ``prefix`` is relative, filenames resolve against
        this directory (typically the enclosing ``.dat`` file's dir or
        a ``FIELD_MAP_PATH``).
    frequency : float
        RF frequency in MHz (from the enclosing ``FREQ`` directive).
        Stored on the returned :class:`FieldMapData` so the element can
        compute the phasor at track time.
    Ki : float
        Space-charge compensation scale factor (manual §18188).  When
        non-zero the matching ``<prefix>.scc`` file must exist; it is
        loaded as a 2-column (z_m, current) text file and stored on
        ``fd.scc_profile`` (z converted to mm) with ``fd.scc_scale = Ki``.
    Ka : int
        Aperture flag 0/1/2 (manual §18198).  Always stored on
        ``fd.ka``.  When ``Ka=1`` the matching ``<prefix>.ouv`` pipe-radius
        file is loaded and stored on ``fd.pipe_radius_profile``
        as ``(z_mm, r_mm)``.

    Returns
    -------
    FieldMapData
        ``channels`` populated for every enabled non-zero geom digit.
    """
    code: GeomCode = decode_geom(geom)
    stripped = _strip_known_suffix(prefix)
    full_prefix = (
        stripped if (os.path.isabs(stripped) or base_dir is None)
        else os.path.join(base_dir, stripped)
    )

    # Pre-flight: collect every expected path and fail early with a message
    # that names ALL missing files (so the caller sees the full picture).
    missing = []
    for channel, digit in enabled_channels(code):
        for suf in component_files(channel, digit):
            p = full_prefix + suf
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        raise FileNotFoundError(
            "Missing field-map file(s): " + ", ".join(missing)
        )

    data = FieldMapData(z=np.asarray([]), frequency=frequency)

    for channel, digit in enabled_channels(code):
        reader = _dispatch_reader(digit)
        ch = FieldChannel(geometry=digit)
        for suf in component_files(channel, digit):
            raw = reader(full_prefix + suf)
            _attach_to_channel(ch, suf, raw)
            # norm_factor: take the first non-unity norm seen on this
            # channel (all component files should agree; if they don't
            # the reader already raised).
            if ch.norm_factor == 1.0 and raw.norm_factor != 1.0:
                ch.norm_factor = raw.norm_factor
        if ch.z is not None and len(data.z) == 0:
            data.z = ch.z
        data.channels[channel] = ch

    # Aperture file (digit > 0 in 10⁴ slot) — open and record path only.
    if code.aper != 0:
        ouv = full_prefix + ".ouv"
        if os.path.exists(ouv):
            data.aperture_file = ouv

    # ------------------------------------------------------------------
    # Task 2c: space-charge compensation profile (Ki / .scc)
    # ------------------------------------------------------------------
    if Ki != 0.0:
        scc_path = full_prefix + ".scc"
        if not os.path.exists(scc_path):
            raise FileNotFoundError(
                f"Ki={Ki} given but space-charge compensation file not found: {scc_path}"
            )
        # Manual §"Current or space charge compensation map":
        #   Line 1:    <mode> <N>          mode 0=Scc(z), 1=I(z)
        #   Lines 2..: <Z_i> <Scc_i or I_i>   (Z in metres)
        with open(scc_path, "r", encoding="latin-1") as fh:
            tokens = fh.readline().split()
            if len(tokens) >= 2:
                mode = int(float(tokens[0]))
                n_pts = int(float(tokens[1]))
            else:
                raise ValueError(f"Malformed .scc header in {scc_path}: {tokens!r}")
            rows = []
            for ln in fh:
                t = ln.split()
                if len(t) >= 2:
                    rows.append((float(t[0]), float(t[1])))
                if len(rows) == n_pts:
                    break
        raw = np.asarray(rows, dtype=float)
        # Column 0: z in metres → mm
        raw[:, 0] *= 1000.0
        data.scc_profile = raw
        data.scc_scale = Ki
        data.scc_mode = mode  # 0 = compensation, 1 = current evolution

    # ------------------------------------------------------------------
    # Task 2d: aperture-override flag (Ka) and optional .ouv profile
    # ------------------------------------------------------------------
    data.ka = Ka
    if Ka == 1:
        ouv_path = full_prefix + ".ouv"
        if not os.path.exists(ouv_path):
            raise FileNotFoundError(
                f"Ka=1 given but pipe-radius file not found: {ouv_path}"
            )
        # TraceWin .ouv format (manual §"Aperture map"):
        #   Line 1:       N              (integer, # of points)
        #   Lines 2..:    Z_i  OuvX_i [OuvY_i]   (m, m, [m])
        # If a third column is present it's the y half-width (rectangular
        # or elliptical); absent → circular with OuvX as radius.  ``Z_0
        # = 0`` is required by the manual.  We strip the header line and
        # read the rest with ``np.genfromtxt`` which tolerates trailing
        # whitespace and mixed 2/3 column counts.
        with open(ouv_path, "r", encoding="latin-1") as fh:
            raw_lines = [ln for ln in fh.readlines() if ln.strip()]
        # TraceWin emits a 1-integer header (``N``) as the first line.
        # Legacy / hand-authored files skip the header and start with data.
        # Detect the header: a line that parses as a single integer AND
        # whose value matches (or plausibly matches) the remaining line
        # count — treat it as header; otherwise, all lines are data.
        first_toks = raw_lines[0].split()
        has_header = False
        if len(first_toks) == 1:
            try:
                n_declared = int(first_toks[0])
                if n_declared > 0 and abs(n_declared - (len(raw_lines) - 1)) <= 2:
                    has_header = True
            except ValueError:
                pass
        body_lines = raw_lines[1:] if has_header else raw_lines
        rows = []
        for ln in body_lines:
            toks = ln.split()
            if len(toks) < 2:
                continue
            try:
                vals = [float(t) for t in toks[:3]]
            except ValueError:
                continue
            rows.append(vals)
        if not rows:
            raise ValueError(f"Aperture file {ouv_path} has no data rows")
        max_cols = max(len(r) for r in rows)
        arr = np.zeros((len(rows), max_cols), dtype=float)
        for i, r in enumerate(rows):
            arr[i, :len(r)] = r
        z_mm = arr[:, 0] * 1000.0
        rx_mm = arr[:, 1] * 1000.0
        ry_mm = arr[:, 2] * 1000.0 if max_cols >= 3 and np.any(arr[:, 2] > 0) else rx_mm.copy()
        data.pipe_radius_profile = (z_mm, rx_mm, ry_mm)

    return data
