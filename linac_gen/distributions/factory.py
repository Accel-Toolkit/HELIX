"""BeamConfig factory: create a Beam from a BeamConfig.

The factory:
1. Looks up the particle species from the config string.
2. Creates a ReferenceParticle at the specified energy and RF frequency.
3. Converts normalised transverse emittances to geometric:
       emit_geo = emit_n / (beta * gamma)
   and applies per-plane mismatch factors (generate path only).
   The longitudinal emittance (emit_z) is already geometric in the config.
4. Dispatches to the appropriate distribution generator.
5. Constructs a Beam, assigns the generated particle array, applies the
   configured centroid offsets, and returns it.
"""
import logging

import numpy as np

from linac_gen.core.config import BeamConfig
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
from linac_gen.distributions.gaussian import generate_gaussian
from linac_gen.distributions.waterbag import generate_waterbag
from linac_gen.distributions.kv import generate_kv
from linac_gen.distributions.parabolic import generate_parabolic
from linac_gen.distributions.uniform import generate_uniform
from linac_gen.distributions.thermal import generate_thermal

_log = logging.getLogger(__name__)

SPECIES_MAP = {
    "proton": PROTON,
    "deuteron": DEUTERON,
    "H-": H_MINUS,
}

_DIST_MAP = {
    "gaussian": generate_gaussian,
    "waterbag": generate_waterbag,
    "kv": generate_kv,
    "parabolic": generate_parabolic,
    "uniform": generate_uniform,
    "thermal": generate_thermal,
}


def geometric_emittances(config, bg: float) -> tuple:
    """(εx, εy, εz) GEOMETRIC emittances for a BeamConfig — normalised →
    geometric via βγ, with the per-plane ``mismatch_{x,y,z}`` scaling
    (×(1 + m/100)) applied.

    This is THE beam-size semantics: the MP generator and every
    envelope seed (matching engine, CLI, torch objective, GUI) must
    call this so a mismatched beam is the same beam in every mode —
    the envelope paths used to drop the mismatch silently.
    """
    bg = float(bg) if bg and bg > 0 else 1.0
    sx = 1.0 + float(getattr(config, "mismatch_x", 0.0) or 0.0) / 100.0
    sy = 1.0 + float(getattr(config, "mismatch_y", 0.0) or 0.0) / 100.0
    sz = 1.0 + float(getattr(config, "mismatch_z", 0.0) or 0.0) / 100.0
    return (config.emit_nx / bg * sx,
            config.emit_ny / bg * sy,
            config.emit_z * sz)   # longitudinal is already geometric


def create_beam(config: BeamConfig, seed: int = None) -> Beam:
    """Create a Beam from a BeamConfig by generating the specified distribution.

    Parameters
    ----------
    config : BeamConfig
        Beam configuration dataclass.
    seed : int or None
        Random seed forwarded to the distribution generator.  Useful for
        reproducible tests; ``None`` draws from system entropy.

    Returns
    -------
    Beam
        Initialised beam with particles set to phase-space deviations.

    Raises
    ------
    ValueError
        If ``config.species`` or ``config.distribution`` is not recognised.
    """
    # 1. Species lookup
    species_key = config.species
    if species_key not in SPECIES_MAP:
        raise ValueError(
            f"Unknown species '{species_key}'. "
            f"Supported: {list(SPECIES_MAP.keys())}"
        )
    species = SPECIES_MAP[species_key]

    # 2. Reference particle
    ref = ReferenceParticle(
        species=species,
        w_kin=config.energy,
        frequency=config.frequency,
    )

    # 3. Geometric emittance from normalised, with per-plane mismatch scaling.
    #    emit_geo = emit_n / (beta * gamma)
    emit_x, emit_y, emit_z = geometric_emittances(config, ref.bg)

    # 4. Generate distribution
    if config.source == "file":
        if not config.distribution_file:
            raise ValueError("source='file' requires distribution_file path")
        if any(m != 0.0 for m in (config.mismatch_x, config.mismatch_y, config.mismatch_z)):
            _log.warning(
                "mismatch_{x,y,z} are ignored for source='file' -- the file's "
                "emittance is used as-is.  Apply mismatch upstream if needed."
            )

        path = str(config.distribution_file)
        if path.lower().endswith(".dst"):
            # TraceWin binary distribution.  Per the TraceWin manual:
            # "If the input dst file is specified, the input beam parameters
            # (number of particles, emittances, Twiss parameters, beam
            # centroid, beam current and energy) are automatically extracted
            # from the specified file and used for the calculation."
            #
            # We follow the same convention: the file is authoritative for
            # the beam state.  We override ref energy + frequency + ε + Twiss
            # in the BeamConfig using values derived from the file.
            from linac_gen.io.tracewin_dst import load_dst
            particles_array, header = load_dst(path)
            file_W = header["w_kin_ref"]
            file_freq = header["frequency_MHz"]
            file_mc2 = header["mass_MeV"]
            if abs(file_W - config.energy) > 1e-6:
                _log.info(
                    "loaded .dst: using file's reference energy %.4f MeV "
                    "(lgproj had %.4f MeV)", file_W, config.energy,
                )
            if abs(file_freq - config.frequency) > 1e-6:
                _log.info(
                    "loaded .dst: using file's frequency %.3f MHz "
                    "(lgproj had %.3f MHz)", file_freq, config.frequency,
                )
            if abs(file_mc2 - species.mass) > 1.0:
                _log.warning(
                    "loaded .dst: mc² in file (%.3f MeV) differs from "
                    "species '%s' rest mass (%.3f MeV) by >1 MeV",
                    file_mc2, species_key, species.mass,
                )
            ref = ReferenceParticle(
                species=species, w_kin=file_W, frequency=file_freq,
            )
            # Surface the file's emittance + Twiss on the config so
            # downstream consumers (envelope solver matching display, GUI
            # beam-summary, lgproj round-trip) see the file's values rather
            # than the (now-overridden) lgproj inputs.  The user is informed
            # of the override via _log.info.
            for fld in ("emit_nx", "emit_ny", "emit_z",
                        "alpha_x", "beta_x",
                        "alpha_y", "beta_y",
                        "alpha_z", "beta_z"):
                if fld in header:
                    new_val = header[fld]
                    old_val = getattr(config, fld, None)
                    if old_val is not None and abs(new_val - old_val) > 1e-6:
                        _log.info(
                            "loaded .dst: overriding config.%s "
                            "%.6g → %.6g (file value)",
                            fld, old_val, new_val,
                        )
                    try:
                        setattr(config, fld, float(new_val))
                    except (AttributeError, TypeError):
                        pass  # frozen dataclass / read-only — skip silently
        else:
            from linac_gen.distributions.from_file import load_distribution
            particles_array, header = load_distribution(path)
            if "w_kin_ref" not in header:
                particles_array, header = load_distribution(
                    path,
                    ref_w_kin=ref.w_kin,
                    ref_phi_s=0.0,
                )
        n_particles = len(particles_array)
    elif config.continuous:
        # DC / continuous beam.  Generate transverse phase space only
        # (x, x', y, y') via the requested distribution, then overlay a
        # uniform phase spread across one RF period and a Gaussian
        # energy spread (0 mean, σ = dc_energy_spread_keV).  The beam
        # has no bunching structure; the tracker will start in 4-D mode
        # and transition to 6-D at the first RF element.
        dist_key = config.distribution
        if dist_key not in _DIST_MAP:
            raise ValueError(
                f"Unknown distribution '{dist_key}'. "
                f"Supported: {list(_DIST_MAP.keys())}"
            )
        generator = _DIST_MAP[dist_key]
        # Use a tiny ε_z / unit β_z just to satisfy the generator's
        # contract; the longitudinal columns are overwritten below.
        common_kwargs = dict(
            n=config.n_particles,
            emit_x=emit_x, alpha_x=config.alpha_x, beta_x=config.beta_x,
            emit_y=emit_y, alpha_y=config.alpha_y, beta_y=config.beta_y,
            emit_z=1.0e-12, alpha_z=0.0, beta_z=1.0,
            seed=seed,
        )
        if dist_key == "gaussian":
            particles_array = generator(**common_kwargs, cutoff=config.cutoff)
        elif dist_key == "thermal":
            particles_array = generator(
                **common_kwargs,
                halo_fraction=config.halo_fraction,
                halo_ratio=config.halo_ratio,
                cutoff=config.cutoff,
            )
        else:
            particles_array = generator(**common_kwargs)
        # Uniform phase across one full RF period ∈ [-180°, +180°].
        rng = np.random.default_rng(seed)
        particles_array[:, 4] = rng.uniform(-180.0, 180.0, config.n_particles)
        # Gaussian ΔW in MeV (config uses keV for the user-facing number).
        sig_w_MeV = config.dc_energy_spread_keV * 1e-3
        if sig_w_MeV > 0:
            particles_array[:, 5] = rng.normal(0.0, sig_w_MeV,
                                               config.n_particles)
        else:
            particles_array[:, 5] = 0.0
        n_particles = config.n_particles
    else:
        dist_key = config.distribution
        if dist_key not in _DIST_MAP:
            raise ValueError(
                f"Unknown distribution '{dist_key}'. "
                f"Supported: {list(_DIST_MAP.keys())}"
            )
        generator = _DIST_MAP[dist_key]

        common_kwargs = dict(
            n=config.n_particles,
            emit_x=emit_x, alpha_x=config.alpha_x, beta_x=config.beta_x,
            emit_y=emit_y, alpha_y=config.alpha_y, beta_y=config.beta_y,
            emit_z=emit_z, alpha_z=config.alpha_z, beta_z=config.beta_z,
            seed=seed,
        )

        if dist_key == "gaussian":
            particles_array = generator(**common_kwargs, cutoff=config.cutoff)
        elif dist_key == "thermal":
            particles_array = generator(
                **common_kwargs,
                halo_fraction=config.halo_fraction,
                halo_ratio=config.halo_ratio,
                cutoff=config.cutoff,
            )
        else:
            particles_array = generator(**common_kwargs)
        n_particles = config.n_particles

    # 5. Build and return Beam (with duty cycle, centroid offsets applied).
    beam = Beam(
        ref=ref, n_particles=n_particles,
        current=config.current, duty_cycle=config.duty_cycle,
    )
    beam.particles[:] = particles_array
    beam.continuous = bool(getattr(config, "continuous", False))

    # Input dispersion shear (generate path only — a loaded file is
    # authoritative as-is): x += disp_x·ΔW etc., applied BEFORE the
    # centroid offsets so the correlation couples to the energy SPREAD
    # rather than the centroid shift.  All-zero defaults leave the
    # array untouched (bit-identical generation).
    if config.source != "file":
        disp = (config.disp_x, config.disp_xp, config.disp_y, config.disp_yp)
        if any(d != 0.0 for d in disp):
            dw = beam.particles[:, 5]
            for col, d in enumerate(disp):
                if d != 0.0:
                    beam.particles[:, col] += d * dw

    # Per-coordinate centroid offsets.  Skip the copy if all zeros.
    offsets = (
        config.centroid_x, config.centroid_xp,
        config.centroid_y, config.centroid_yp,
        config.centroid_dphi, config.centroid_dw,
    )
    if any(abs(o) > 0.0 for o in offsets):
        for col, off in enumerate(offsets):
            if off != 0.0:
                beam.particles[:, col] += off

    return beam
