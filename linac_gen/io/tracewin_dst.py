"""TraceWin .dst (binary particle distribution) reader/writer.

Format reference: TraceWin user manual (file `c/users/abhishek/TraceWin/
doc/tracewin.htm`, "Files" section, .dst description).  Verified
empirically against real TraceWin .dst files (e.g. PIP-II SCL+BTL
input.dst from CEA/Saclay).

Binary layout
-------------

    Offset  Size  Type      Field
    ------  ----  --------  -------------------------------------------
    0       1     CHAR      header byte 0 (TraceWin writes 0x7d)
    1       1     CHAR      header byte 1 (TraceWin writes 0x64)
    2       4     INT       Np   — number of particles (little-endian)
    6       8     DOUBLE    Ib   — beam current (mA)
    14      8     DOUBLE    freq — bunch frequency (MHz)
    22      1     CHAR      separator byte (TraceWin writes 0x7d)
    23      48*Np 6×DOUBLE  per-particle:
                            x  (cm), x' (rad), y (cm), y' (rad),
                            phi (rad, absolute), W (MeV, absolute)
    23+48*Np 8   DOUBLE    mc²  — particle rest-mass energy (MeV)

Total file size = 31 + 48*Np bytes.

The 5th and 6th columns are *absolute* phase (rad, relative to RF clock)
and *absolute* kinetic energy (MeV, not a deviation).  The on-disk
record contains no separate "synchronous" particle; the bunch centroid
is conventionally used as the reference.

LG internal phase-space units are (mm, mrad, deg, MeV) with the
longitudinal pair ``(dphi, dW)`` expressed as deviations from the
synchronous reference.  This loader converts on the fly:

    x   [cm]  → mm     (×10)
    x'  [rad] → mrad   (×1000)
    y   [cm]  → mm     (×10)
    y'  [rad] → mrad   (×1000)
    phi [rad] → dphi [deg]   ((phi − mean(phi)) × 180/π)
    W   [MeV] → dW [MeV]     (W − mean(W))

Centroid subtraction is the default reference convention; the raw
centroid (phi_ref_rad, w_kin_ref_MeV) is exposed in the returned
header so the caller can override / restore the absolute values if
needed.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np

# Header layout: 2*CHAR + INT(Np) + DOUBLE(Ib) + DOUBLE(freq) + CHAR
#              = 2 + 4 + 8 + 8 + 1 = 23 bytes (TraceWin packs without alignment)
_HEADER_FMT = "<2sIddB"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)   # 23
assert _HEADER_SIZE == 23, "Header size must be 23 bytes"
_PARTICLE_BYTES = 6 * 8                        # 48
_TRAILER_SIZE = 8                              # one DOUBLE for mc²

# TraceWin observed magic byte values (for compatibility on write).
_MAGIC_HEAD = b"\x7d\x64"
_MAGIC_SEP = 0x7d


def load_dst(path: str) -> Tuple[np.ndarray, dict]:
    """Read a TraceWin .dst file.

    Parameters
    ----------
    path : str
        Path to the .dst file.

    Returns
    -------
    particles : np.ndarray
        (N, 6) array in LG conventions:
        ``[x_mm, xp_mrad, y_mm, yp_mrad, dphi_deg, dW_MeV]``
        where ``dphi`` and ``dW`` are deviations from the bunch centroid
        (mean phi and mean W of the file, respectively).
    header : dict
        ``{n_particles, current_mA, frequency_MHz, mass_MeV,
           w_kin_ref, phi_ref_rad}``.

        ``w_kin_ref`` is the centroid energy used as the synchronous
        reference; ``phi_ref_rad`` is the raw centroid phase.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is too small or its size is inconsistent with the
        announced particle count.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    raw = p.read_bytes()
    if len(raw) < _HEADER_SIZE + _TRAILER_SIZE:
        raise ValueError(
            f"File too small to contain a .dst header + trailer: "
            f"{len(raw)} bytes ({path})"
        )

    _, n, current_mA, freq_MHz, _ = struct.unpack(
        _HEADER_FMT, raw[:_HEADER_SIZE]
    )
    if n <= 0:
        raise ValueError(f"Particle count must be positive; got {n}")

    expected_size = _HEADER_SIZE + n * _PARTICLE_BYTES + _TRAILER_SIZE
    if len(raw) != expected_size:
        raise ValueError(
            f"File size {len(raw)} != expected {expected_size} for N={n} "
            f"(31 + 48*N + 8 byte layout).  File may be truncated, in a "
            f"non-standard layout, or N may be misread."
        )

    body_start = _HEADER_SIZE
    body_end = body_start + n * _PARTICLE_BYTES
    body = raw[body_start:body_end]
    trailer = raw[body_end:]

    arr = np.frombuffer(body, dtype="<f8").reshape(n, 6)
    mass_MeV = float(struct.unpack("<d", trailer)[0])

    # Compute centroid for phi (rad) and W (MeV) — used as synchronous ref.
    phi_ref_rad = float(np.mean(arr[:, 4]))
    w_kin_ref = float(np.mean(arr[:, 5]))

    particles = np.empty((n, 6), dtype=np.float64)
    particles[:, 0] = arr[:, 0] * 10.0                          # cm → mm
    particles[:, 1] = arr[:, 1] * 1000.0                        # rad → mrad
    particles[:, 2] = arr[:, 2] * 10.0
    particles[:, 3] = arr[:, 3] * 1000.0
    particles[:, 4] = np.rad2deg(arr[:, 4] - phi_ref_rad)        # → dphi deg
    particles[:, 5] = arr[:, 5] - w_kin_ref                      # → dW MeV

    # Compute Twiss parameters and emittances from the sample covariance.
    # Convention: ε_g = √det(Σ_2D), α = −Σ_12 / ε_g, β = Σ_11 / ε_g.
    # ε_n_transverse = β·γ · ε_g (using w_kin_ref as the reference energy).
    twiss = _twiss_from_particles(particles, mass_MeV, w_kin_ref)

    header = {
        "n_particles": int(n),
        "current_mA": float(current_mA),
        "frequency_MHz": float(freq_MHz),
        "mass_MeV": mass_MeV,
        "w_kin_ref": w_kin_ref,
        "phi_ref_rad": phi_ref_rad,
        # Sample-derived emittances and Twiss
        "emit_x": twiss["emit_x"],         # geometric, mm·mrad
        "emit_y": twiss["emit_y"],
        "emit_z": twiss["emit_z"],         # deg·MeV (LG convention)
        "emit_nx": twiss["emit_nx"],       # normalised, mm·mrad
        "emit_ny": twiss["emit_ny"],
        "alpha_x": twiss["alpha_x"],
        "beta_x":  twiss["beta_x"],        # mm/mrad
        "alpha_y": twiss["alpha_y"],
        "beta_y":  twiss["beta_y"],
        "alpha_z": twiss["alpha_z"],
        "beta_z":  twiss["beta_z"],        # deg/MeV
    }
    return particles, header


def _twiss_from_particles(particles: np.ndarray,
                          mass_MeV: float, w_kin_MeV: float) -> dict:
    """Compute geometric / normalised emittance and Twiss parameters from
    a particle distribution (sample-covariance method).

    Returns a dict with keys ``emit_{x,y,z}``, ``emit_n{x,y}``,
    ``alpha_{x,y,z}``, ``beta_{x,y,z}``.  Units: transverse positions in
    mm, angles in mrad → ε in mm·mrad and β in mm/mrad; longitudinal
    dphi in deg, dW in MeV → ε_z in deg·MeV and β_z in deg/MeV.
    """
    out: dict = {}
    pairs = [("x", 0, 1), ("y", 2, 3), ("z", 4, 5)]
    for plane, i_pos, i_ang in pairs:
        u = particles[:, i_pos]
        up = particles[:, i_ang]
        s11 = float(np.var(u, ddof=0))
        s22 = float(np.var(up, ddof=0))
        s12 = float(np.mean((u - u.mean()) * (up - up.mean())))
        det = s11 * s22 - s12 * s12
        emit = float(np.sqrt(max(det, 0.0)))
        out[f"emit_{plane}"] = emit
        if emit > 0:
            out[f"alpha_{plane}"] = -s12 / emit
            out[f"beta_{plane}"] = s11 / emit
        else:
            out[f"alpha_{plane}"] = 0.0
            out[f"beta_{plane}"] = 1.0

    # Normalised transverse emittances at the reference energy.
    if mass_MeV > 0:
        gamma = 1.0 + w_kin_MeV / mass_MeV
        beta_rel = float(np.sqrt(max(1.0 - 1.0 / (gamma * gamma), 0.0)))
        bg = beta_rel * gamma
    else:
        bg = 1.0
    out["emit_nx"] = out["emit_x"] * bg
    out["emit_ny"] = out["emit_y"] * bg
    return out


class DstHeaderWarning(UserWarning):
    """The ``.dst`` header cannot represent this beam faithfully.

    The format carries ONE frequency, and HELIX writes the LOCAL RF
    clock into it because that is the clock its Δφ degrees are measured
    in.  Real TraceWin files write the BUNCH REPETITION RATE instead —
    measured 2026-07-31 across six genuine PIP-II files, every one
    carries 162.5 MHz whether the particles sit at 30 keV, 166 MeV
    (a 325 MHz section) or 752 MeV (650 MHz).  Downstream of a
    frequency jump the two values differ and a HELIX-written file is
    therefore non-standard.
    """


def warn_if_nonstandard_dst_header(beam) -> bool:
    """Warn when this beam's ``.dst`` header will not match TraceWin's.

    Returns True when a warning was issued.  Only fires downstream of a
    frequency jump, where ``bunch_frequency`` (pinned at beam creation)
    and ``ref.frequency`` (the live RF clock) have diverged.

    NOT fixed by simply writing ``bunch_frequency``: the phase column
    would then be local-clock degrees under a bunch-rate label, which
    is a worse failure than the present one because it looks
    standard-compliant.  Whether TraceWin converts its phases on export
    is unresolved and needs one round-trip through real TraceWin to
    settle; until then HELIX stays self-consistent and says so loudly.

    ------------------------------------------------------------------
    DEFERRED FIX (parked 2026-07-31 by decision — do not half-do it)
    ------------------------------------------------------------------
    This is NOT merely a caller passing the wrong variable, even though
    it looks like one: ``write_dst``'s own docstring documents its
    ``frequency_MHz`` parameter as the BUNCH FREQUENCY, and
    ``cli/common.py::write_final_dst`` hands it ``beam.ref.frequency``
    with no comment.  ``git blame`` shows the line was last touched by
    an unrelated alive-particles fix, so the mismatch was never a
    decision.  Fixing the argument ALONE would break the clock, because
    ``write_dst`` does no frequency conversion and therefore assumes the
    caller's Δφ is already in bunch-clock degrees.

    The fix is BOTH halves together:
        frequency_MHz = beam.bunch_frequency
        dphi_written  = dphi_local * (f_bunch / f_local)

    BLOCKED ON an external anchor.  A round-trip test proves nothing
    here — scale on write, unscale on read, and HELIX agrees with itself
    whether or not the convention is right (the same trap as the ×βγ
    writer bug that round-tripped perfectly while being ~15× wrong).
    Settle it with one real TraceWin run: a known bunch through a
    162.5 → 325 MHz jump, exported past the jump, its σ_φ compared
    against the σ_z TraceWin reports at that plane.  The σ_z test on the
    files we already hold cannot separate the hypotheses — 1.9 mm and
    0.9 mm at the SSR2 exit are both physically plausible.

    ALSO DEFERRED: ``BeamConfig.bunch_frequency_MHz`` (default 0 =
    "same as the clock").  A loaded ``.dst`` is authoritative for the
    RF clock, correctly, but it cannot be authoritative for the bunch
    REPETITION RATE — the format carries no such field.  Today there is
    no way, GUI or config, to state it: ``factory.py`` overrides
    ``BeamConfig.frequency`` from the header and ``Beam.__init__``
    derives ``bunch_frequency`` from that, so a Beam-tab value is
    silently ignored.  Needed before any segmented workflow (exporting
    mid-linac and restarting), which is exactly when the bunch rate can
    no longer be inferred.

    EXPOSURE TODAY: none.  The failure needs a frequency jump AND a
    ``.dst`` round-trip.  ``examples/MEBT_To_Foil`` has the jump
    (162.5 → 325 → 650, so the error would be ×4) but no project in the
    repo imports a ``.dst`` written after a jump — 34 of 35 are
    ``source=generate``, and the one importer reads a TraceWin file at
    z = 0 on a line with zero FREQ cards.
    """
    f_local = float(getattr(getattr(beam, "ref", None), "frequency", 0.0) or 0.0)
    f_bunch = float(getattr(beam, "bunch_frequency", 0.0) or 0.0)
    if f_local <= 0.0 or f_bunch <= 0.0 or abs(f_local - f_bunch) <= 1e-9:
        return False
    warnings.warn(
        f"writing a .dst downstream of a frequency jump: the header will "
        f"carry the LOCAL RF clock {f_local:g} MHz, which is what this "
        f"file's phase column is measured in, but real TraceWin files "
        f"carry the BUNCH REPETITION RATE ({f_bunch:g} MHz here) at every "
        f"location.  This file round-trips correctly through HELIX and is "
        f"NOT interchangeable with TraceWin; on re-import the macrocharge "
        f"Q = I/f would use {f_local:g} MHz instead of {f_bunch:g} MHz.",
        DstHeaderWarning, stacklevel=3)
    return True


def write_dst(path: str, particles: np.ndarray,
              current_mA: float, frequency_MHz: float,
              mass_MeV: float, w_kin_ref: float,
              phi_ref_rad: float = 0.0) -> None:
    """Write a TraceWin .dst file.

    Particles must be in LG units ``[x_mm, xp_mrad, y_mm, yp_mrad,
    dphi_deg, dW_MeV]`` with the longitudinal pair as deviations.  They
    are converted back to TraceWin's absolute conventions (positions in
    cm, angles in rad, phi/W absolute) on write.

    Parameters
    ----------
    path : str
        Output filename.
    particles : np.ndarray
        (N, 6) array in LG units.
    current_mA, frequency_MHz : float
        Beam current (mA) and bunch frequency (MHz) — written into the
        file header.
    mass_MeV : float
        Particle rest-mass energy (mc²) — written into the file trailer.
    w_kin_ref : float
        Synchronous reference kinetic energy (MeV).  Each particle's
        ``W = dW + w_kin_ref`` is written as the absolute energy.
    phi_ref_rad : float, default 0.0
        Synchronous reference phase (rad).  Each particle's
        ``phi = dphi*π/180 + phi_ref_rad`` is written as the absolute
        phase.
    """
    if particles.ndim != 2 or particles.shape[1] != 6:
        raise ValueError(
            f"particles must have shape (N, 6); got {particles.shape}"
        )
    n = particles.shape[0]
    arr = np.empty((n, 6), dtype="<f8")
    arr[:, 0] = particles[:, 0] * 0.1                           # mm → cm
    arr[:, 1] = particles[:, 1] * 1e-3                          # mrad → rad
    arr[:, 2] = particles[:, 2] * 0.1
    arr[:, 3] = particles[:, 3] * 1e-3
    arr[:, 4] = np.deg2rad(particles[:, 4]) + phi_ref_rad        # absolute phi
    arr[:, 5] = particles[:, 5] + w_kin_ref                      # absolute W

    header = struct.pack(
        _HEADER_FMT, _MAGIC_HEAD, int(n),
        float(current_mA), float(frequency_MHz), _MAGIC_SEP,
    )
    trailer = struct.pack("<d", float(mass_MeV))
    Path(path).write_bytes(header + arr.tobytes() + trailer)
