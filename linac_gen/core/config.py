"""Configuration dataclasses for beam and space charge."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class BeamConfig:
    """Initial beam parameters for distribution generation.

    Notes
    -----
    - ``current`` is the **peak** bunch current (drives space charge).
      ``duty_cycle`` scales it to a time-averaged value via
      :attr:`linac_gen.core.beam.Beam.average_current`, but the tracking
      / PIC physics always uses peak current.
    - ``centroid_*`` offsets are added to every particle after the
      distribution is generated, so the beam is *deliberately* off-axis.
    - ``mismatch_{x,y,z}`` scale the geometric emittance per plane by
      ``(1 + mismatch / 100)``; at 0 % the beam is perfectly matched to
      the supplied Twiss.  File-loaded distributions ignore these factors
      (their emittance is whatever the file specifies).
    """
    species: str = "proton"
    energy: float = 3.0
    frequency: float = 352.21
    current: float = 0.0
    duty_cycle: float = 100.0          # %; <100 => pulsed operation
    n_particles: int = 10000
    distribution: str = "waterbag"
    cutoff: float = 3.0
    emit_nx: float = 0.25
    alpha_x: float = 0.0
    beta_x: float = 0.1
    emit_ny: float = 0.25
    alpha_y: float = 0.0
    beta_y: float = 0.1
    emit_z: float = 0.3
    alpha_z: float = 0.0
    beta_z: float = 1.0
    # Centroid offsets -- beam displaced from the reference trajectory.
    centroid_x: float = 0.0            # mm
    centroid_xp: float = 0.0           # mrad
    centroid_y: float = 0.0            # mm
    centroid_yp: float = 0.0           # mrad
    centroid_dphi: float = 0.0         # deg
    centroid_dw: float = 0.0           # MeV
    # Input dispersion -- linear correlation between each transverse
    # coordinate and the particle's energy deviation ΔW, applied at
    # generation as x += disp_x·ΔW (etc.), BEFORE centroid offsets so
    # the dispersion couples to the energy SPREAD, not the centroid
    # shift.  Units mm/MeV and mrad/MeV -- the same convention as
    # find_periodic_twiss / find_sc_matched_input_twiss's
    # disp_x/disp_xp/disp_y/disp_yp (Σ[i,5]/Σ[5,5]).  The matched beam
    # of a bending line (arc FODO / BTL) carries nonzero values here.
    # Generate-path only; a .dst file is authoritative as loaded.
    disp_x: float = 0.0                # mm/MeV
    disp_xp: float = 0.0               # mrad/MeV
    disp_y: float = 0.0                # mm/MeV
    disp_yp: float = 0.0               # mrad/MeV
    # Courant-Snyder mismatch, applied as a per-plane emittance scale.
    mismatch_x: float = 0.0            # % (generate path only)
    mismatch_y: float = 0.0            # %
    mismatch_z: float = 0.0            # %
    source: str = "generate"
    distribution_file: Optional[str] = None
    # ---- DC / continuous-beam (pre-RFQ / LEBT) -----------------------
    # When ``True`` the beam is generated with uniform phase over one RF
    # period (no longitudinal bunching) and propagated in 4-D through
    # purely-transverse elements (drifts, solenoids, quads, steerers).
    # ``alpha_z``, ``beta_z``, ``emit_z`` are ignored in this mode; set
    # the energy spread explicitly via ``dc_energy_spread_keV``.  The
    # tracker flips ``beam.continuous = False`` automatically when the
    # beam first encounters an RF bunching element (RFQ / FIELD_MAP
    # cavity / RFGap), after which the run proceeds as bunched.
    continuous: bool = False
    dc_energy_spread_keV: float = 0.0
    # ---- Periodic phase coordinates (bunch train) --------------------
    # OPT-IN.  An RFQ turns a DC beam into a bunch TRAIN — one bunch per
    # RF period — and HELIX seeds exactly one period, so particles that
    # slip across a bucket boundary are stored 360° away and the beam
    # becomes a finite multi-bucket clump.  With this flag the tracker
    # folds Δφ into ONE BUNCH SPACING after every element and before
    # every space-charge kick (the Toutatis convention), so the train
    # never forms: the RF forces are unchanged (they depend only on
    # cos/sin φ), but σ_φ / ε_z / z-Twiss become single-bunch values
    # with no post-processing, and the space-charge solver sees one
    # bunch instead of a clump whose non-periodic field generates a
    # spurious energy chirp.
    #
    # Only applies once the beam is genuinely a train — DC-injected and
    # then bunched by a TIME-VARYING RF element (see
    # ``Beam.bunch_train``; a static electrostatic column does not
    # count) — and only when the local RF frequency is an integer
    # multiple of the bunch repetition rate.  That rate is
    # ``Beam.bunch_train_frequency``, captured from the buncher's own
    # clock at the transition, NOT this config's ``frequency``: a beam
    # configured at 162.5 MHz and bunched by a 325 MHz gap must fold at
    # 360°, and using the config value gave 720° and left every
    # satellite in place with no warning.
    #
    # Refused where it would be wrong: backtracking a folded beam (the
    # fold is not invertible) and ``csr_enabled`` (the CSR wake is built
    # from the ensemble's absolute longitudinal extent and is not
    # periodic in the bunch spacing).  A non-harmonic frequency warns
    # and skips if nothing has been folded yet, and RAISES if folds are
    # already in flight — they get rescaled by every downstream RF
    # element and stop being whole RF periods.
    periodic_phase: bool = False
    # ---- Bi-Gaussian (thermal halo) parameters -----------------------
    # Used only when distribution == "thermal".  ``halo_fraction`` is the
    # mixture weight of the wide-Gaussian halo (0..1); ``halo_ratio`` is
    # how much wider the halo is than the core (≥1).  See
    # ``linac_gen.distributions.thermal`` for the exact mixture formula.
    halo_fraction: float = 0.05
    halo_ratio: float = 5.0

    def __post_init__(self) -> None:
        if self.energy <= 0:
            raise ValueError(f"energy must be > 0 MeV, got {self.energy}")
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0 MHz, got {self.frequency}")
        if self.current < 0:
            raise ValueError(f"current must be >= 0 mA, got {self.current}")
        if not 0.0 < self.duty_cycle <= 100.0:
            raise ValueError(
                f"duty_cycle must be in (0, 100] %, got {self.duty_cycle}"
            )
        if self.n_particles <= 0:
            raise ValueError(f"n_particles must be > 0, got {self.n_particles}")
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be > 0, got {self.cutoff}")
        # Skip the longitudinal checks when the user has explicitly asked
        # for a continuous beam — ε_z / β_z are unused in that mode.
        trans_emit = ("emit_nx", "emit_ny")
        all_emit   = trans_emit if self.continuous else trans_emit + ("emit_z",)
        for attr in all_emit:
            val = getattr(self, attr)
            if val < 0:
                raise ValueError(f"{attr} must be >= 0, got {val}")
        trans_beta = ("beta_x", "beta_y")
        all_beta   = trans_beta if self.continuous else trans_beta + ("beta_z",)
        for attr in all_beta:
            val = getattr(self, attr)
            if val <= 0:
                raise ValueError(f"{attr} must be > 0, got {val}")
        if self.dc_energy_spread_keV < 0:
            raise ValueError(
                f"dc_energy_spread_keV must be >= 0, got {self.dc_energy_spread_keV}"
            )
        # mismatch < -100% would flip emittance negative.
        for attr in ("mismatch_x", "mismatch_y", "mismatch_z"):
            val = getattr(self, attr)
            if val <= -100.0:
                raise ValueError(
                    f"{attr} must be > -100 % (cannot zero out emittance), got {val}"
                )
        valid_source = {"generate", "file"}
        if self.source not in valid_source:
            raise ValueError(f"source must be one of {valid_source}, got {self.source!r}")
        if not 0.0 <= self.halo_fraction <= 1.0:
            raise ValueError(
                f"halo_fraction must be in [0, 1], got {self.halo_fraction}"
            )
        if self.halo_ratio < 1.0:
            raise ValueError(
                f"halo_ratio must be >= 1, got {self.halo_ratio}"
            )


@dataclass
class SpaceChargeConfig:
    """PIC space charge solver configuration.

    Defaults are tuned for the H- injector reference beam (5 mA CW) on
    a FODO-style focusing channel, following the convergence study in
    ``scripts/convergence_study.py`` and ``docs/convergence/``:

    - ``nx=ny=nz=96``: 1 % converged for emit_x at 5 mA.  Cheaper 64 is
      visible in convergence plots; 128+ is publication-quality.
    - ``grid_extent=5.0``: 3 sigma was clipping beam tails badly; 5 sigma
      is converged for 5 mA and preserves ~1e-6 of the Gaussian tail.
      For high-current cases (I >> 10 mA with tight focusing), 3 sigma
      can be safe because the beam is pinched small -- but verify via
      the Tools -> Check SC Convergence dialog.
    """
    nx: int = 96
    ny: int = 96
    nz: int = 96
    boundary: str = "open"
    solver: str = "fft"
    grid_extent: float = 5.0
    shape_order: int = 1
    grid_mode: str = "fixed"
    # FFT backend for the Poisson solver.  "auto" picks GPU (cupy) when
    # importable and a CUDA device is visible, else CPU (scipy.fft).  The
    # env var ``LINAC_GEN_USE_GPU`` overrides this field.
    use_gpu: str = "auto"
    # Particle-mesh shape function for deposition AND gather.  "cic" is
    # the legacy 8-corner trilinear (1st-order); "tsc" is the 27-cell
    # quadratic that reduces grid noise at the cost of ~3.4× more work.
    kernel: str = "cic"
    # Hockney FFT Green's function flavour.  "igf" (default) uses the
    # cell-integrated kernel of Qiang 2006; "point" samples 1/(4πε₀ r)
    # at cell centres and is kept for legacy-baseline reproduction.
    green_kind: str = "igf"
    # DC (continuous beam) SC kick model:
    #   "uniform"  — analytic linear uniform-elliptical-cylinder field
    #                (legacy; matches the rigid-Σ envelope solver).
    #   "gaussian" — Bassetti-Erskine field of a 2-D Gaussian density
    #                (rigid σ_x, σ_y; per-particle non-linear kick).
    #   "pic2d"    — 2-D Hockney FFT PIC over the actual particle
    #                distribution (open-BC, full per-particle non-linearity,
    #                analogue of TraceWin's PICNIC_2D).
    dc_kernel: str = "uniform"
    # Bunch-train neighbour images in the 3-D PIC.  A beam injected DC
    # and bunched in flight (an RFQ front end) is one period of an
    # infinite train: through the buncher the charge still fills the
    # whole RF period and each slice feels its ±1 neighbours — physics
    # Toutatis gets for free from its periodic time-domain solve, and
    # an isolated-bunch solve misses (+2-3 pts transmission on the PXIE
    # RFQ at 5 mA, 2026-08-02).  None (default) = automatic: images on
    # exactly when ``Beam.bunch_train`` is set (DC-injected beams),
    # off for beams born bunched — mirroring TraceWin, whose Partran
    # treats downstream bunches as isolated.  True/False force it.
    train_images: bool | None = None
    # Coherent Synchrotron Radiation (CSR) inside bends.  When enabled,
    # the multi-particle tracker applies a 1-D steady-state CSR energy
    # kick (Saldin-Schnizer) inside every Dipole element.  CSR needs the
    # actual bunch profile, so it acts only in multi-particle runs — the
    # envelope solver ignores these fields.
    csr_enabled: bool = False
    csr_bins: int = 200                # longitudinal line-density bin count
    csr_model: str = "1d_steady"       # forward-compatible model selector
    # Space-charge compute backend.  "numpy" is the production numpy/C++
    # PIC; "torch" routes the bunched-beam kick through the differentiable
    # PyTorch PIC (``linac_gen.pic.torch``) for autograd / gradient-based
    # optimisation — numerically matches "numpy" to ~1e-9.
    sc_backend: str = "numpy"
    # HALO-PIC options (sc_backend="halo"): alpha, k_init/k_min/k_max,
    # tau_lo/tau_hi/tau_hard, fine_factor, basis_degree, basis_width,
    # warmup_kicks, anchors, collect, corrector_dir, period_len_mm.
    halo: dict = None

    def __post_init__(self) -> None:
        for attr in ("nx", "ny", "nz"):
            val = getattr(self, attr)
            if val <= 0:
                raise ValueError(f"{attr} must be > 0, got {val}")
        if self.grid_extent <= 0:
            raise ValueError(f"grid_extent must be > 0, got {self.grid_extent}")
        if self.shape_order < 1:
            raise ValueError(f"shape_order must be >= 1, got {self.shape_order}")
        if self.shape_order != 1:
            # Deprecated no-op: deposition order is selected by
            # ``kernel`` ("cic"/"tsc"); shape_order was never read by
            # any solver.
            import warnings
            warnings.warn(
                f"shape_order={self.shape_order} is a deprecated no-op "
                "— the deposition order is selected by kernel="
                "'cic'/'tsc'; shape_order is ignored.",
                DeprecationWarning, stacklevel=2)
        # Only open-BC (Hockney doubled-grid) is implemented.  "periodic"
        # used to validate and then be silently ignored — refuse instead.
        valid_boundary = {"open"}
        if self.boundary not in valid_boundary:
            raise ValueError(
                f"boundary must be one of {valid_boundary}, got "
                f"{self.boundary!r} (the Poisson solve is open-boundary "
                f"Hockney only; 'periodic' is not implemented)"
            )
        valid_solver = {"fft"}
        if self.solver not in valid_solver:
            raise ValueError(
                f"solver must be one of {valid_solver}, got {self.solver!r}"
            )
        valid_grid_mode = {"fixed", "adaptive"}
        if self.grid_mode not in valid_grid_mode:
            raise ValueError(
                f"grid_mode must be one of {valid_grid_mode}, got {self.grid_mode!r}"
            )
        # Keep this whitelist in lock-step with
        # ``linac_gen.pic.gpu_backend._VALID_MODES``.  ``auto`` picks the best
        # available (cupy → MPS → CPU); ``gpu`` requires any GPU; ``cuda``
        # forces cupy; ``mps`` forces torch.mps (Apple Silicon Metal); ``cpu``
        # disables GPU even if one is present.
        valid_use_gpu = {"auto", "cpu", "gpu", "cuda", "mps"}
        if self.use_gpu not in valid_use_gpu:
            raise ValueError(
                f"use_gpu must be one of {valid_use_gpu}, got {self.use_gpu!r}"
            )
        valid_kernel = {"cic", "tsc"}
        if self.kernel not in valid_kernel:
            raise ValueError(
                f"kernel must be one of {valid_kernel}, got {self.kernel!r}"
            )
        valid_green = {"igf", "point"}
        if self.green_kind not in valid_green:
            raise ValueError(
                f"green_kind must be one of {valid_green}, got {self.green_kind!r}"
            )
        valid_dc = {"uniform", "gaussian", "pic2d"}
        if self.dc_kernel not in valid_dc:
            raise ValueError(
                f"dc_kernel must be one of {valid_dc}, got {self.dc_kernel!r}"
            )
        if self.csr_bins <= 0:
            raise ValueError(f"csr_bins must be > 0, got {self.csr_bins}")
        valid_csr = {"1d_steady"}
        if self.csr_model not in valid_csr:
            raise ValueError(
                f"csr_model must be one of {valid_csr}, got {self.csr_model!r}"
            )
        valid_sc_backend = {"numpy", "torch", "halo"}
        if self.sc_backend not in valid_sc_backend:
            raise ValueError(
                f"sc_backend must be one of {valid_sc_backend}, got "
                f"{self.sc_backend!r}"
            )
