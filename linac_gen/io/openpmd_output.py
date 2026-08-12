"""openPMD-beamphysics-compatible HDF5 writer / reader.

HELIX writes results in two parallel HDF5 formats:

* The native :mod:`linac_gen.io.hdf5_output` schema (HELIX-internal,
  full-state round-trip).
* openPMD-standard 1.1, written by this module — meant for interop with
  openPMD-viewer, Astra, Genesis, and the LUME wrappers.

Implementation choice: rather than depending on the (separate)
``openpmd-beamphysics`` package, we emit a minimal openPMD-compliant HDF5
schema directly with :mod:`h5py` (already a hard dependency).  The
schema is the subset required by openPMD-viewer for visualisation;
extensions (1-D envelope arrays) live under a HELIX-specific
``/data/<iter>/envelope/`` group that openPMD-viewer simply ignores.

Spec reference
--------------
openPMD standard 1.1 — https://github.com/openPMD/openPMD-standard
openPMD-beamphysics extension — particle-mesh data for accelerators.

Schema produced
---------------
::

    file.opmd.h5
    ├── (root attrs: openPMD="1.1.0", openPMDextension=0,
    │                basePath="/data/%T/", iterationEncoding="groupBased",
    │                iterationFormat="/data/%T/", meshesPath="meshes/",
    │                particlesPath="particles/")
    └── data/
        └── 0/                              ← one iteration per snapshot
            ├── (attrs: time, dt, timeUnitSI)
            ├── particles/
            │   └── <species_name>/         ← e.g. "H-", "proton"
            │       ├── (attrs: speciesType, particleShape, currentDeposition)
            │       ├── position/
            │       │   ├── x   (N float64 SI:m, unitDimension length)
            │       │   ├── y
            │       │   └── z
            │       ├── momentum/
            │       │   ├── x   (N float64 SI:kg·m/s)
            │       │   ├── y
            │       │   └── z
            │       ├── mass             (scalar SI:kg)
            │       ├── charge           (scalar SI:C)
            │       ├── weighting        (N float64 dimensionless)
            │       └── particleStatus   (N int8: 1=alive, 0=lost)
            └── envelope/                   ← HELIX extension, not openPMD core
                ├── s                   (1-D)
                ├── sigma_x, sigma_y, sigma_phi, sigma_w
                ├── emit_x, emit_y, emit_z, emit_nx, emit_ny
                ├── alpha_x, beta_x, alpha_y, beta_y
                ├── halo_x, halo_y, transmission
                └── (ref attrs: w_kin, beta, gamma vs s — under sub-group "reference")

Reference frame note
--------------------
HELIX's per-particle coords are ``(x_mm, xp_mrad, y_mm, yp_mrad,
dphi_deg, dw_MeV)`` measured in the reference-particle frame (TraceWin
convention).  The openPMD conversion maps these to lab-frame SI as
follows:

* x, y in metres                       =  HELIX x_mm, y_mm × 1e-3
* z in metres                          =  −dphi_deg × (π/180) × β·c / ω_rf
* px, py, pz in kg·m/s                 =  HELIX (xp, yp, scaled longitudinal)
  with p_ref = γβmc derived from the per-iteration reference particle.

These conventions match openPMD-beamphysics' ``ParticleGroup``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from linac_gen.core.constants import C_LIGHT, E_CHARGE


# openPMD standard 1.1 constants -------------------------------------------
_OPENPMD_VERSION = "1.1.0"
_BASE_PATH = "/data/%T/"
_PARTICLES_PATH = "particles/"
_MESHES_PATH = "meshes/"

# unitDimension is a 7-tuple of base-SI exponents:
# (length, mass, time, current, temperature, amount, luminous_intensity)
_UD_LENGTH   = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_UD_MOMENTUM = (1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0)
_UD_MASS     = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_UD_CHARGE   = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
_UD_DIMLESS  = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _set_unit_attrs(dset: h5py.Dataset, unit_SI: float,
                    unit_dimension: tuple) -> None:
    """Apply the required openPMD per-record unit attributes."""
    dset.attrs["unitSI"] = float(unit_SI)
    dset.attrs["unitDimension"] = np.asarray(unit_dimension, dtype=np.float64)


def _write_global_attrs(f: h5py.File) -> None:
    """Required root-level openPMD attributes."""
    import datetime

    f.attrs["openPMD"] = _OPENPMD_VERSION
    f.attrs["openPMDextension"] = np.uint32(0)
    f.attrs["basePath"] = _BASE_PATH
    f.attrs["iterationEncoding"] = "groupBased"
    f.attrs["iterationFormat"] = _BASE_PATH
    f.attrs["meshesPath"] = _MESHES_PATH
    f.attrs["particlesPath"] = _PARTICLES_PATH
    # Recommended-by-standard provenance attributes.
    from linac_gen import __version__
    f.attrs["software"] = "HELIX (linac_gen)"
    f.attrs["softwareVersion"] = __version__
    # openPMD-1.1 recommends "YYYY-MM-DD HH:mm:ss tzOffset" (the
    # official validator regex-checks it — ISO-8601 'T' fails).
    f.attrs["date"] = datetime.datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %z")


def _species_label(ref) -> str:
    """openPMD ``speciesType`` label for the reference-particle's species."""
    return ref.species.name or "particle"


# ── HELIX → SI per-snapshot coordinate conversion --------------------------

def _helix_particles_to_si(particles: np.ndarray, ref,
                            bunch_frequency_MHz: float) -> dict[str, np.ndarray]:
    """Convert one (N, 6) HELIX particle snapshot to SI lab-frame arrays.

    HELIX columns:  (x_mm, xp_mrad, y_mm, yp_mrad, dphi_deg, dw_MeV)
    Output:         dict with 'x','y','z' in m and 'px','py','pz' in kg·m/s.
    """
    # Reference momentum [MeV/c] and rest mass [kg].
    mass_MeV = ref.species.mass
    p_ref_MeVc = ref.bg * mass_MeV
    # kg·m/s per MeV/c:  1 MeV/c = (1e6 · e) / c   (SI base)
    mev_c_to_kg_m_s = (1.0e6 * E_CHARGE) / C_LIGHT
    p_ref_si = p_ref_MeVc * mev_c_to_kg_m_s

    x_m = particles[:, 0] * 1e-3
    y_m = particles[:, 2] * 1e-3

    # dphi (deg) → z (m) at the bunch frequency.
    # ω = 2π · f,  z = -dphi · β · c / ω
    omega_rf = 2.0 * math.pi * bunch_frequency_MHz * 1e6  # rad/s
    if omega_rf > 0.0:
        dphi_rad = particles[:, 4] * (math.pi / 180.0)
        z_m = -dphi_rad * ref.beta * C_LIGHT / omega_rf
    else:
        z_m = np.zeros_like(x_m)

    # Transverse momenta: xp (mrad) is p_x / p_z.  In the small-angle
    # limit p_x = p_ref · xp_rad.  For HELIX-grade tracking, xp/yp magnitudes
    # are mrad-scale so the small-angle map is exact to ~1e-6.
    px_si = p_ref_si * (particles[:, 1] * 1e-3)
    py_si = p_ref_si * (particles[:, 3] * 1e-3)

    # dw (MeV) → kinetic energy shift → longitudinal momentum.
    # Use exact relativistic conversion: p_total = √((W + m)² − m²) / c.
    # Then pz = √(p_total² − p_x² − p_y²) ≈ p_total at small angles.
    dw_MeV = particles[:, 5]
    W_par_MeV = ref.w_kin + dw_MeV
    # p_par^2 = (W + m)^2 − m^2 = W·(W + 2m) — numerically stable for W>0.
    p_par_MeVc = np.sqrt(np.maximum(W_par_MeV * (W_par_MeV + 2.0 * mass_MeV),
                                     0.0))
    p_par_si = p_par_MeVc * mev_c_to_kg_m_s
    # Subtract transverse momentum² to get pz; clamp to ≥ 0 for safety.
    pz_si = np.sqrt(np.maximum(p_par_si ** 2 - px_si ** 2 - py_si ** 2, 0.0))

    return dict(x=x_m, y=y_m, z=z_m, px=px_si, py=py_si, pz=pz_si)


def _write_particles(group: h5py.Group, particles: np.ndarray, ref,
                      bunch_frequency_MHz: float,
                      weighting: float, alive_mask: np.ndarray | None = None,
                      ) -> None:
    """Write one particles group under /data/<iter>/particles/<species>/."""
    si = _helix_particles_to_si(particles, ref, bunch_frequency_MHz)

    # group-level openPMD record attributes
    group.attrs["speciesType"] = _species_label(ref)
    group.attrs["particleShape"] = 1.0      # CIC-like
    group.attrs["currentDeposition"] = "Esirkepov"
    group.attrs["particlePush"] = "Vay"

    # position/{x,y,z}
    pos = group.create_group("position")
    _set_unit_attrs(pos.create_dataset("x", data=si["x"]), 1.0, _UD_LENGTH)
    _set_unit_attrs(pos.create_dataset("y", data=si["y"]), 1.0, _UD_LENGTH)
    _set_unit_attrs(pos.create_dataset("z", data=si["z"]), 1.0, _UD_LENGTH)

    # positionOffset/{x,y,z} — required by openPMD-standard; all zeros here.
    off = group.create_group("positionOffset")
    zeros = np.zeros_like(si["x"])
    _set_unit_attrs(off.create_dataset("x", data=zeros), 1.0, _UD_LENGTH)
    _set_unit_attrs(off.create_dataset("y", data=zeros), 1.0, _UD_LENGTH)
    _set_unit_attrs(off.create_dataset("z", data=zeros), 1.0, _UD_LENGTH)

    # momentum/{x,y,z}
    mom = group.create_group("momentum")
    _set_unit_attrs(mom.create_dataset("x", data=si["px"]), 1.0, _UD_MOMENTUM)
    _set_unit_attrs(mom.create_dataset("y", data=si["py"]), 1.0, _UD_MOMENTUM)
    _set_unit_attrs(mom.create_dataset("z", data=si["pz"]), 1.0, _UD_MOMENTUM)

    # mass — scalar record (length-N constant; spec says scalar records have
    # macroWeighted=0 and value field).
    mass_kg = ref.species.mass * 1e6 * E_CHARGE / C_LIGHT ** 2
    n = particles.shape[0]
    mass_arr = np.full(n, mass_kg, dtype=np.float64)
    _set_unit_attrs(group.create_dataset("mass", data=mass_arr), 1.0, _UD_MASS)

    # charge — signed per species.
    q_C = ref.species.charge * E_CHARGE
    q_arr = np.full(n, q_C, dtype=np.float64)
    _set_unit_attrs(group.create_dataset("charge", data=q_arr), 1.0, _UD_CHARGE)

    # weighting — physical particles per macroparticle.
    w_arr = np.full(n, weighting, dtype=np.float64)
    _set_unit_attrs(
        group.create_dataset("weighting", data=w_arr), 1.0, _UD_DIMLESS,
    )

    # particleStatus: 1=alive, 0=lost (openPMD-beamphysics convention).
    if alive_mask is None:
        status = np.ones(n, dtype=np.int8)
    else:
        status = alive_mask.astype(np.int8)
    _set_unit_attrs(
        group.create_dataset("particleStatus", data=status), 1.0, _UD_DIMLESS,
    )


# ── public API --------------------------------------------------------------

def save_results_openpmd(results, filepath: str | Path,
                          beam_config: Any = None,
                          lattice: Any = None) -> None:
    """Write ``results`` to ``filepath`` in openPMD-standard 1.1 HDF5.

    Accepts either a :class:`linac_gen.diagnostics.recorder.DiagnosticRecorder`
    (with optional ``_snapshots`` dict) or an
    :class:`linac_gen.tracking.envelope.EnvelopeResults` (envelope only,
    no snapshots).

    On envelope-only runs (no particle snapshots) the file still contains
    one iteration with the envelope datasets under ``/data/0/envelope/``;
    the particles group is omitted.  openPMD-viewer treats this as a
    file with no particle records — non-fatal.

    Parameters
    ----------
    results : DiagnosticRecorder | EnvelopeResults
        Object exposing envelope arrays (``s``, ``sigma_x``, …) and
        optionally a ``_snapshots`` dict.
    filepath : str | Path
        Destination ``.opmd.h5`` file.
    beam_config : object | None
        Optional; its ``__dict__`` is stamped as attrs on the root
        group's ``/beam_config`` subgroup (HELIX extension).
    lattice : object | None
        Reserved for future use.
    """
    filepath = str(filepath)

    snapshots: dict = getattr(results, "_snapshots", {}) or {}
    iterations: list[tuple[float, np.ndarray, Any]] = []
    if snapshots:
        for s_pos, (particles, ref_state) in sorted(snapshots.items()):
            iterations.append((float(s_pos), np.asarray(particles), ref_state))

    # Bunch frequency (MHz) — best-effort retrieval.
    bunch_freq_MHz = 0.0
    if beam_config is not None:
        bunch_freq_MHz = float(
            getattr(beam_config, "bunch_frequency_MHz", 0.0) or 0.0) \
            or float(getattr(beam_config, "frequency", 0.0) or 0.0)
    if bunch_freq_MHz <= 0.0 and iterations:
        # Use the first snapshot's reference frequency as a fallback.
        bunch_freq_MHz = float(getattr(iterations[0][2], "frequency", 0.0) or 0.0)

    # Weighting: physical particles per macroparticle (CW current / freq / N).
    weighting = 1.0
    n_total_macro: int = 0
    if iterations:
        n_total_macro = int(iterations[0][1].shape[0])
    elif beam_config is not None:
        n_total_macro = int(getattr(beam_config, "n_particles", 0) or 0)
    if beam_config is not None and bunch_freq_MHz > 0.0 and n_total_macro > 0:
        current_mA = float(getattr(beam_config, "current", 0.0) or 0.0)
        if current_mA > 0.0:
            charge_per_bunch_C = (current_mA * 1e-3) / (bunch_freq_MHz * 1e6)
            weighting = charge_per_bunch_C / (n_total_macro * E_CHARGE)

    with h5py.File(filepath, "w") as f:
        _write_global_attrs(f)
        data_grp = f.create_group("data")

        if iterations:
            # One openPMD iteration per snapshot.
            for i, (s_pos, particles, ref_state) in enumerate(iterations):
                it = data_grp.create_group(str(i))
                # openPMD time attributes — we use s (m) as a stand-in for time
                # because HELIX is s-based.  An s-based simulation has dt=0
                # per spec convention; record s explicitly under a custom
                # attribute and the spec-required time as 0.
                it.attrs["time"] = float(s_pos) * 1e-3  # s in m (HELIX) → seconds is awkward; reuse as path-length-as-time
                it.attrs["dt"] = 1.0
                it.attrs["timeUnitSI"] = 1.0
                it.attrs["helix_s_mm"] = float(s_pos)

                # particles group
                parts_root = it.create_group(_PARTICLES_PATH.rstrip("/"))
                species_name = _species_label(ref_state)
                spec_grp = parts_root.create_group(species_name)
                _write_particles(
                    spec_grp, particles, ref_state, bunch_freq_MHz,
                    weighting=weighting,
                )

                # envelope arrays under HELIX extension subgroup (only on iter 0
                # to avoid duplication — they're whole-run arrays).
                if i == 0:
                    _write_envelope_extension(it, results)
        else:
            # Envelope-only run — single iteration, no particles.
            it = data_grp.create_group("0")
            it.attrs["time"] = 0.0
            it.attrs["dt"] = 1.0
            it.attrs["timeUnitSI"] = 1.0
            _write_envelope_extension(it, results)

        # Optional beam-config attributes (HELIX extension).
        if beam_config is not None:
            cfg = f.create_group("beam_config")
            for key, val in beam_config.__dict__.items():
                if val is None:
                    continue
                try:
                    cfg.attrs[key] = val
                except TypeError:
                    pass


def _write_envelope_extension(iteration_group: h5py.Group, results) -> None:
    """Write 1-D envelope arrays under ``<iter>/envelope/`` (HELIX extension)."""
    env = iteration_group.create_group("envelope")
    for attr in (
        "s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
        "emit_x", "emit_y", "emit_z", "emit_nx", "emit_ny",
        "alpha_x", "beta_x", "alpha_y", "beta_y",
        "alpha_z", "beta_z",
        "halo_x", "halo_y", "transmission",
    ):
        if hasattr(results, attr):
            arr = np.asarray(getattr(results, attr), dtype=np.float64)
            if arr.size > 0:
                env.create_dataset(attr, data=arr)
    ref_grp = iteration_group.create_group("reference_history")
    for attr, key in (
        ("ref_w_kin", "w_kin"),
        ("ref_phi_s", "phi_s"),
        ("ref_beta",  "beta"),
        ("ref_gamma", "gamma"),
        ("ref_bg",    "bg"),
    ):
        if hasattr(results, attr):
            arr = np.asarray(getattr(results, attr), dtype=np.float64)
            if arr.size > 0:
                ref_grp.create_dataset(key, data=arr)


# ── loader -----------------------------------------------------------------

def load_results_openpmd(filepath: str | Path) -> dict:
    """Read back an openPMD file written by :func:`save_results_openpmd`.

    Returns the same flat-dict shape as
    :func:`linac_gen.io.hdf5_output.load_results_hdf5` so the GUI's
    ``_LoadedResults`` adapter can wrap either format identically.
    """
    filepath = str(filepath)
    results: dict = {}
    with h5py.File(filepath, "r") as f:
        # Required root attr — we use it as the "is this openPMD?" sentinel.
        if "openPMD" not in f.attrs:
            raise ValueError(
                f"{filepath} does not have an openPMD root attribute; "
                "not produced by save_results_openpmd."
            )
        if "data" not in f:
            return results
        # Use iteration 0's envelope group as the canonical envelope source.
        first_iter = "0"
        if first_iter not in f["data"]:
            return results
        it = f["data"][first_iter]
        if "envelope" in it:
            for key in it["envelope"]:
                results[key] = it["envelope"][key][:]
        if "reference_history" in it:
            for key in it["reference_history"]:
                results[f"ref_{key}"] = it["reference_history"][key][:]
    return results


def is_openpmd_file(filepath: str | Path) -> bool:
    """Cheap sniff: is this an openPMD HDF5 file?"""
    try:
        with h5py.File(str(filepath), "r") as f:
            return "openPMD" in f.attrs
    except (OSError, ValueError, KeyError):
        return False
