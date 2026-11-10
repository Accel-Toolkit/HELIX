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

#: notes already given about a loaded file: a match builds its beam at every evaluation
_DST_NOTES: set = set()


def _dst_note(path, message: str, *args) -> None:
    """Warn about a loaded ``.dst``, once for the process."""
    key = (str(path), message % args)
    if key not in _DST_NOTES:
        _DST_NOTES.add(key)
        _log.warning(message, *args)

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


_DISPERSION_KEYS = ("disp_x", "disp_xp", "disp_y", "disp_yp")


def input_dispersion(config) -> dict:
    """The input dispersion a beam from ``config`` starts with, keyed as the envelope
    solver reads it (``disp_x`` and ``disp_y`` in mm/MeV, ``disp_xp`` and ``disp_yp`` in
    mrad/MeV): the config's values, by which :func:`create_beam` shears the particles it
    generates, or zeros for a beam read from a file, whose particles are taken as they are.
    Every envelope seed takes it from here, so the envelope of a bunched beam starts from the
    beam a multiparticle run generates.  A continuous beam gets zeros too, with a warning: its
    envelope holds no energy spread before the first RF element, so a dispersion cannot enter
    it, while create_beam shears such a beam by its DC energy spread."""
    if getattr(config, "source", None) == "file":
        return dict.fromkeys(_DISPERSION_KEYS, 0.0)
    disp = {k: float(getattr(config, k, 0.0) or 0.0) for k in _DISPERSION_KEYS}
    if bool(getattr(config, "continuous", False)):
        if any(disp.values()):
            import warnings
            warnings.warn("a continuous beam: the envelope holds no energy spread before the "
                          "first RF element, so the input dispersion does not enter it (a "
                          "multiparticle run shears the beam by its DC energy spread)",
                          UserWarning, stacklevel=2)
        return dict.fromkeys(_DISPERSION_KEYS, 0.0)
    return disp


def reshape_to_twiss(particles, alpha_x: float, beta_x: float,
                     alpha_y: float, beta_y: float):
    """Particles re-shaped onto the given transverse rms Twiss.

    Each plane ``(u, u')`` (mm, mrad) is mapped by
    ``A(α, β) · A(α₀, β₀)⁻¹`` about its centroid, with
    ``A(α, β) = [[√β, 0], [−α/√β, 1/√β]]`` and (α₀, β₀) the particles' own
    rms Twiss: a symplectic linear map that keeps the plane's rms
    emittance, centroid and normalised phase-space shape (halo, tails,
    x–y correlations transform along) and gives exactly (α, β), β in
    mm/mrad.  The longitudinal columns are untouched.  Returns a new
    array; ``particles`` is not modified.
    """
    out = np.array(particles, dtype=float, copy=True)
    for (i, j), (a_t, b_t) in (((0, 1), (alpha_x, beta_x)),
                               ((2, 3), (alpha_y, beta_y))):
        u, up = out[:, i], out[:, j]
        c = np.cov(u, up)
        eps = float(np.sqrt(np.linalg.det(c)))
        if not (eps > 0.0 and b_t > 0.0):
            raise ValueError(
                f"cannot re-shape plane {'xy'[i // 2]}: rms emittance "
                f"{eps:g} mm mrad, target beta {b_t:g} mm/mrad")
        a0, b0 = -c[0, 1] / eps, c[0, 0] / eps
        A0 = np.array([[np.sqrt(b0), 0.0], [-a0 / np.sqrt(b0), 1.0 / np.sqrt(b0)]])
        At = np.array([[np.sqrt(b_t), 0.0],
                       [-a_t / np.sqrt(b_t), 1.0 / np.sqrt(b_t)]])
        T = At @ np.linalg.inv(A0)
        mu = np.array([u.mean(), up.mean()])
        new = T @ (np.vstack([u, up]) - mu[:, None]) + mu[:, None]
        out[:, i], out[:, j] = new[0], new[1]
    return out


#: what a TraceWin .dst file sets on the beam besides its energy and frequency
_DST_BEAM_FIELDS = ("emit_nx", "emit_ny", "emit_z", "alpha_x", "beta_x",
                    "alpha_y", "beta_y", "alpha_z", "beta_z")
_DST_HEADERS: dict = {}


def _dst_header(path: str) -> dict:
    """The header of a .dst file (its particles dropped), kept while the file is unchanged:
    every mode reads it for the beam it starts from."""
    import os
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)
    header = _DST_HEADERS.get(key)
    if header is None:
        from linac_gen.io.tracewin_dst import load_dst
        particles, header = load_dst(path)
        # the particles' centroid, which a run's beam keeps (create_beam)
        header = dict(header, centroid=tuple(float(v) for v in np.asarray(particles).mean(axis=0))
                      if len(particles) else (0.0,) * 6)
        if len(_DST_HEADERS) > 8:
            _DST_HEADERS.clear()
        _DST_HEADERS[key] = header
    return header


def _dst_beam_fields(config, header) -> dict:
    """The emittances and Twiss parameters a .dst file sets on the beam: all those it
    holds, but the transverse alpha/beta of a config that re-shapes the file onto them
    (``reshape_to_twiss``), which are its target."""
    keep = (("alpha_x", "beta_x", "alpha_y", "beta_y")
            if getattr(config, "reshape_to_twiss", False) else ())
    return {f: float(header[f]) for f in _DST_BEAM_FIELDS if f in header and f not in keep}


def reference_config(config):
    """The beam settings a run starts from, as :func:`create_beam` takes them: ``config``
    itself, or for a TraceWin .dst source a copy with the file's energy, frequency,
    emittances and Twiss parameters (the file wins over the config), its particles'
    centroid plus the config's offsets, and no mismatch (a file's emittance is used as
    it is).  No particles are made and ``config`` is left as it is, so the envelope,
    matrix and every other mode start where a multiparticle run does."""
    if getattr(config, "source", None) != "file":
        return config
    path = str(getattr(config, "distribution_file", "") or "")
    if not path.lower().endswith(".dst") or config.species not in SPECIES_MAP:
        return config
    import copy as _copy
    header = _dst_header(path)
    out = _copy.copy(config)
    out.energy = float(header["w_kin_ref"])
    out.frequency = float(header["frequency_MHz"])
    for fld, value in _dst_beam_fields(config, header).items():
        setattr(out, fld, value)
    out.mismatch_x = out.mismatch_y = out.mismatch_z = 0.0
    for fld, c in zip(("centroid_x", "centroid_xp", "centroid_y", "centroid_yp",
                       "centroid_dphi", "centroid_dw"), header["centroid"], strict=True):
        setattr(out, fld, float(getattr(config, fld, 0.0) or 0.0) + c)
    return out


def file_reference(config) -> ReferenceParticle | None:
    """The reference a beam from ``config`` starts with when its source is a
    TraceWin .dst file, as :func:`create_beam` takes it: the file's energy
    and frequency (the file wins over the config).  None for any other
    source."""
    if getattr(config, "source", None) != "file":
        return None
    path = str(getattr(config, "distribution_file", "") or "")
    if not path.lower().endswith(".dst") or config.species not in SPECIES_MAP:
        return None
    from linac_gen.io.tracewin_dst import load_dst
    _particles, header = load_dst(path)
    return ReferenceParticle(species=SPECIES_MAP[config.species],
                             w_kin=header["w_kin_ref"], frequency=header["frequency_MHz"])


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
            #
            # CAVEAT (deferred 2026-07-31): authoritative for the RF
            # CLOCK, but the file cannot be authoritative for the BUNCH
            # REPETITION RATE — the .dst format carries no such field.
            # ``Beam.__init__`` derives ``bunch_frequency`` from this
            # ref, so importing a distribution written downstream of a
            # frequency jump silently adopts the local clock as the
            # bunch rate and the macrocharge Q = I/f comes out wrong by
            # the harmonic ratio (×4 for a 650 MHz file off a 162.5 MHz
            # train).  A Beam-tab / BeamConfig frequency does NOT
            # override it — the file wins here by design.  Planned:
            # BeamConfig.bunch_frequency_MHz, explicitly NOT
            # file-overridden.  See io/tracewin_dst.py DEFERRED FIX.
            from linac_gen.io.tracewin_dst import load_dst
            particles_array, header = load_dst(path)
            file_W = header["w_kin_ref"]
            file_freq = header["frequency_MHz"]
            file_mc2 = header["mass_MeV"]
            if (config.current > 0.0
                    and header["current_mA"] > 0.0
                    and (abs(header["current_mA"] - config.current)
                         > 1e-3 * header["current_mA"])):
                # the beam keeps the config's current, on the file's particles
                # (said of the config: a match with space charge off, or a
                # preview, has no run at this current)
                if header["n_lost"]:
                    _dst_note(
                        path, "loaded .dst: %d of %d particles are flagged lost, so "
                        "the beam left in the file carries %.4g mA, but the beam is "
                        "set to %.4g mA", header["n_lost"], header["n_in_file"],
                        header["current_mA"], config.current,
                    )
                else:
                    _dst_note(
                        path, "loaded .dst: the file's %d particles carry %.4g mA, "
                        "but the beam is set to %.4g mA", header["n_particles"],
                        header["current_mA"], config.current,
                    )
            if header["n_off_energy"]:
                _dst_note(
                    path, "loaded .dst: %d of %d particles are further than half the "
                    "median energy from it (not accelerated?): they stay in the "
                    "beam, and dominate its energy spread and longitudinal "
                    "emittance", header["n_off_energy"], header["n_particles"],
                )
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
            # of the override via _log.info.  With reshape_to_twiss the
            # config's transverse alpha/beta are the target (kept).
            for fld, new_val in _dst_beam_fields(config, header).items():
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
        if getattr(config, "reshape_to_twiss", False):
            particles_array = reshape_to_twiss(
                particles_array, config.alpha_x, config.beta_x,
                config.alpha_y, config.beta_y)
            _log.info("re-shaped %s onto alpha/beta x %.4g/%.4g, y %.4g/%.4g "
                      "(the file's emittances kept)", path, config.alpha_x,
                      config.beta_x, config.alpha_y, config.beta_y)
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
    beam.periodic_phase = bool(getattr(config, "periodic_phase", False))
    # Explicit bunch-repetition frequency (macrocharge denominator).
    # 0.0 keeps the historical "RF clock at creation" default set by
    # Beam.__init__; nonzero overrides it (sub-harmonic trains,
    # segmented workflows, multibunch studies).
    _f_bunch = float(getattr(config, "bunch_frequency_MHz", 0.0) or 0.0)
    if _f_bunch > 0.0:
        beam.bunch_frequency = _f_bunch

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

    # the run's seed, which an unseeded foil's kicks follow
    beam.seed = None if seed is None else int(seed)
    return beam
