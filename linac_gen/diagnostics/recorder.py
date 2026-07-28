# linac_gen/diagnostics/recorder.py
"""DiagnosticRecorder: stores beam diagnostics at each tracking step."""
import numpy as np
from linac_gen.diagnostics.moments import (
    compute_moments, compute_emittance,
    compute_twiss_from_particles, compute_halo,
)
from linac_gen.diagnostics.eigenemittance import eigenemittances


# ---------------------------------------------------------------------------
def _normal_mode_emittances(sigma4: np.ndarray) -> tuple:
    """Return the two transverse normal-mode emittances (ε_1 ≥ ε_2).

    The 4-D Courant–Snyder invariants are obtained as |Im(λ_i)| where
    λ_i are eigenvalues of J·Σ_4D, with the symplectic form

        J = [[ 0, 1, 0, 0],
             [-1, 0, 0, 0],
             [ 0, 0, 0, 1],
             [ 0, 0,-1, 0]]

    Returns ``(0.0, 0.0)`` if the determinant is non-positive (e.g. an
    initial dummy record before any particles exist).
    """
    if sigma4 is None or sigma4.shape != (4, 4):
        return 0.0, 0.0
    if np.linalg.det(sigma4) <= 0:
        return 0.0, 0.0
    J = np.zeros((4, 4))
    J[0, 1] = 1.0; J[1, 0] = -1.0
    J[2, 3] = 1.0; J[3, 2] = -1.0
    eigs = np.linalg.eigvals(J @ sigma4)
    # Symplectic eigenvalues come in conjugate pairs ±i·ε; take the unique
    # positive imaginary parts.
    ims = np.sort(np.abs(eigs.imag))
    if len(ims) >= 4:
        # 4 values in {ε_1, ε_1, ε_2, ε_2} after sorting → take ims[1] and ims[3]
        e2 = float(ims[1])
        e1 = float(ims[3])
    else:
        e2 = 0.0
        e1 = float(ims[-1]) if len(ims) else 0.0
    if e2 > e1:
        e1, e2 = e2, e1
    return e1, e2


def _convert_emit_z_to_mmmrad(emit_z_deg_mev: float, ref) -> float:
    """Convert longitudinal emittance from (deg*MeV) to (mm*mrad).

    Uses the local reference-particle state to build the (z, z') <-> (Phi, w)
    Jacobian:

        k_Phi = 360 / (beta * lambda_RF)   [deg/mm]
        k_w   = beta^2 * gamma * m / 1000  [MeV/mrad]   (z' = dp/p * 1000)

    Emittance transforms as the product of the diagonal Jacobian entries:

        emit_{Phi,w} = k_Phi * k_w * emit_{z,z'}
      => emit_{z,z'} = emit_{Phi,w} / (k_Phi * k_w)

    Returns 0 if the conversion factor degenerates (e.g. beta == 0).
    """
    beta = ref.beta
    wavelength = ref.wavelength      # mm
    mass = ref.species.mass          # MeV/c^2
    gamma = ref.gamma
    if beta <= 0.0 or wavelength <= 0.0:
        return 0.0
    k_phi = 360.0 / (beta * wavelength)
    k_w = beta * beta * gamma * mass * 1e-3
    denom = k_phi * k_w
    if denom <= 0.0:
        return 0.0
    return emit_z_deg_mev / denom


class DiagnosticRecorder:
    """Records beam diagnostics at each tracking step."""

    def __init__(self):
        self.s = []
        self.sigma_x = []
        self.sigma_y = []
        self.sigma_phi = []
        self.sigma_w = []
        self.emit_x = []              # mm.mrad  (geometric transverse)
        self.emit_y = []              # mm.mrad
        self.emit_z = []              # deg.MeV  (native longitudinal units)
        self.emit_z_mmmrad = []       # mm.mrad  (emit_z re-expressed so all three
                                      # planes share units; see _convert_emit_z)
        self.emit_nx = []
        self.emit_ny = []
        self.emit_nz = []             # βγ · emit_z_mmmrad  (normalised longitudinal
                                      # emittance in mm·mrad for all-planes parity)
        self.sigma_matrix = []        # list of (6, 6) arrays -- full beam sigma
        self.element_names = []       # name of the element whose exit this entry
                                      # corresponds to ("INPUT" for the initial record)
        # Record index of the row holding element j's EXIT state.  With
        # record_substeps off this is j+1 (row 0 = INPUT); with substeps
        # on, elements contribute a variable number of interior rows and
        # this mapping is the only correct element-span → record-span
        # translation (see analysis.phase_advance.element_record_span).
        self.element_exit_idx = []
        self.alpha_x = []
        self.beta_x = []
        self.alpha_y = []
        self.beta_y = []
        # Longitudinal Twiss from the (Δφ deg, ΔW MeV) particle columns —
        # HELIX-internal convention (α_z = −⟨Δφ·ΔW⟩/ε_z; TW's α_z is the
        # negative — appendix E); β_z in deg/MeV at the local machine
        # clock (scales f_new/f_old at FREQ; α_z invariant).
        self.alpha_z = []
        self.beta_z = []
        self.halo_x = []
        self.halo_y = []
        self.transmission = []
        self.centroid = []
        self.ref_w_kin = []
        self.ref_phi_s = []
        self.ref_beta = []
        self.ref_gamma = []
        self.ref_bg = []
        # Per-step RF frequency [MHz].  Mutates at every FREQ-card boundary
        # (e.g. PIP-II SSR1/SSR2 hop from 162.5 → 325 MHz), so σ_φ →
        # σ_z conversions need this rather than the constant
        # ``BeamConfig.frequency``.  Used by EnvelopeTriple's σ_z
        # derivation and by the IBS analyzer.
        self.ref_frequency = []
        # Peak excursion of any surviving particle (used for aperture
        # studies / halo tracking — captures beyond what σ implies).
        self.x_max = []
        self.y_max = []
        # 4-D transverse phase-space invariant.  The 2-D projections ε_x /
        # ε_y oscillate with solenoid rotation because our tracker uses
        # kinetic (x, x'=p_x/p_z) coordinates; the 4-D area √det(σ_4D) is
        # coupling-invariant (same as TraceWin's canonical ε_xx' × ε_yy'),
        # so ``emit_4d`` stays smooth through solenoids and reveals real
        # 4-D growth separately from the projection wobble.
        self.emit_4d = []
        # Normal-mode (eigen-) emittances of the 4-D transverse Σ block.
        # ε_1, ε_2 are obtained from the imaginary parts of the symplectic
        # eigenvalues of (J · Σ_4D), where J is the 4×4 symplectic form.
        # In an uncoupled lattice ε_1 = ε_x and ε_2 = ε_y; under solenoid
        # coupling they remain *invariant* under the rotation while ε_x
        # and ε_y wobble.  Useful for distinguishing coupling from real
        # phase-space growth.
        self.emit_n1 = []
        self.emit_n2 = []
        # 6-D normal-mode (eigen-) emittances of the full Σ-matrix.
        # Constants of motion under any linear symplectic transport — stay
        # invariant under x-y, energy-time, and dispersive coupling that
        # makes ε_x, ε_y, ε_z visibly oscillate.  Reduce to (ε_x, ε_y, ε_z)
        # in an uncoupled lattice.  Computed via the Balandin trace
        # invariants (I₂, I₄, I₆) — see ``eigenemittance.py`` for details.
        # Units: mm·mrad (geometric, matched to ε_x/ε_y/ε_z_mmmrad).
        self.emit_e1 = []
        self.emit_e2 = []
        self.emit_e3 = []
        # Per-record DC / bunched flag (True while pre-RFQ / LEBT; flips
        # to False at the first bunching element).  Populated by record()
        # via ``beam.continuous``; clients use it to mark those points
        # as having non-physical σ_φ / σ_W / ε_z values.
        self.continuous_at = []
        # Species rest mass (MeV) — constant across the whole run, stashed
        # once so popups can compute dispersion / dp-p / Δz without
        # rehydrating the ReferenceParticle object.
        self.mass_mev: float = 0.0
        self._snapshots = {}
        # Per-snapshot alive mask (True = lost), kept in a PARALLEL dict so
        # the (particles, ref) tuple of `_snapshots` stays 2-wide — several
        # consumers unpack it (io/hdf5_output.py, io/openpmd_output.py).
        # Lets viewers filter a snapshot to alive particles like the exit.
        self._snapshot_masks = {}

        # ------------------------------------------------------------------
        # 2-D density grids — one column per s-step, one row per bin.
        #
        # Built progressively via :meth:`record_density`.  Off by default;
        # callers opt in by setting :attr:`density_axes` and
        # :attr:`density_extent` before tracking.  Storage is ``int32`` —
        # 1000 steps × 200 bins × 4 B = 0.8 MB per axis.
        #
        # ``density`` maps an axis label ('x','y','xp','yp','phi','w') to a
        # list of histograms (one per recorded step).  ``density_edges``
        # caches the bin edges per axis so the GUI plot can label rows.
        # ------------------------------------------------------------------
        self.density: dict[str, list] = {}
        self.density_edges: dict[str, "np.ndarray"] = {}
        self.density_axes: tuple[str, ...] = ()
        self.density_extent: dict[str, tuple[float, float]] = {}
        self.density_n_bins: int = 200

    def record(self, beam, s_position: float, element_name: str = "INPUT") -> None:
        """Record diagnostics from current beam state.

        ``element_name`` tags the entry with the element whose exit this
        record corresponds to (``"INPUT"`` for the pre-lattice record).
        """
        self.s.append(s_position)
        alive = beam.alive_particles
        # Per-step continuous-beam flag.  Downstream plots can use this
        # to mark / colour pre-RFQ sections and to warn that σ_φ / σ_W /
        # ε_z are the values of a uniform DC distribution rather than a
        # physical bunch (σ_φ ≈ 104° is the uniform-over-360° signature,
        # not a physical bunch length).
        if not hasattr(self, "continuous_at"):
            self.continuous_at = []
        self.continuous_at.append(bool(getattr(beam, "continuous", False)))

        self.ref_w_kin.append(beam.ref.w_kin)
        self.ref_phi_s.append(beam.ref.phi_s)
        self.ref_beta.append(beam.ref.beta)
        self.ref_gamma.append(beam.ref.gamma)
        self.ref_bg.append(beam.ref.bg)
        self.ref_frequency.append(float(getattr(beam.ref, "frequency", 0.0)))
        self.transmission.append(beam.n_alive / beam.n_particles * 100.0)
        self.element_names.append(element_name)
        # Cache the rest mass once — popups use it for D_x / σ_z / dp/p
        # conversions without re-constructing a ReferenceParticle.
        if not self.mass_mev and hasattr(beam, "ref"):
            self.mass_mev = float(beam.ref.species.mass)

        if len(alive) == 0:
            self.sigma_x.append(0.0)
            self.sigma_y.append(0.0)
            self.sigma_phi.append(0.0)
            self.sigma_w.append(0.0)
            self.emit_x.append(0.0)
            self.emit_y.append(0.0)
            self.emit_z.append(0.0)
            self.emit_z_mmmrad.append(0.0)
            self.emit_nx.append(0.0)
            self.emit_ny.append(0.0)
            self.emit_nz.append(0.0)
            self.alpha_x.append(0.0)
            self.beta_x.append(0.0)
            self.alpha_y.append(0.0)
            self.beta_y.append(0.0)
            self.alpha_z.append(0.0)
            self.beta_z.append(0.0)
            self.halo_x.append(0.0)
            self.halo_y.append(0.0)
            self.centroid.append(np.zeros(6))
            self.sigma_matrix.append(np.zeros((6, 6)))
            self.x_max.append(0.0)
            self.y_max.append(0.0)
            self.emit_4d.append(0.0)
            self.emit_n1.append(0.0)
            self.emit_n2.append(0.0)
            self.emit_e1.append(0.0)
            self.emit_e2.append(0.0)
            self.emit_e3.append(0.0)
            # Keep density columns aligned with the s-array even when the
            # beam has fully been lost.
            self.record_density(beam)
            self._record_tail(alive)
            return

        m = compute_moments(alive)
        self.sigma_x.append(m["sigma_x"])
        self.sigma_y.append(m["sigma_y"])
        self.sigma_phi.append(m["sigma_phi"])
        self.sigma_w.append(m["sigma_w"])
        self.centroid.append(m["mean"].copy())
        self.sigma_matrix.append(m["sigma_matrix"].copy())

        ex = compute_emittance(alive, "x")
        ey = compute_emittance(alive, "y")
        ez = compute_emittance(alive, "z")
        self.emit_x.append(ex)
        self.emit_y.append(ey)
        self.emit_z.append(ez)
        self.emit_z_mmmrad.append(_convert_emit_z_to_mmmrad(ez, beam.ref))
        self.emit_nx.append(ex * beam.ref.bg)
        self.emit_ny.append(ey * beam.ref.bg)
        # Longitudinal normalised emittance.  emit_z is deg·MeV; convert to
        # mm·mrad via the recorder's helper (handles the β·λ / mc² Jacobian)
        # and scale by βγ for the canonical "normalised" form, so ε_nz sits
        # alongside ε_nx and ε_ny on a common axis.
        self.emit_nz.append(
            _convert_emit_z_to_mmmrad(ez, beam.ref) * beam.ref.bg
        )

        twx = compute_twiss_from_particles(alive, "x")
        twy = compute_twiss_from_particles(alive, "y")
        twz = compute_twiss_from_particles(alive, "z")
        self.alpha_x.append(twx["alpha"])
        self.beta_x.append(twx["beta"])
        self.alpha_y.append(twy["alpha"])
        self.beta_y.append(twy["beta"])
        self.alpha_z.append(twz["alpha"])
        self.beta_z.append(twz["beta"])

        self.halo_x.append(compute_halo(alive, "x"))
        self.halo_y.append(compute_halo(alive, "y"))

        # Peak excursion — max |x| and max |y| of any surviving particle at
        # this step.  Useful for aperture-vs-halo studies.
        self.x_max.append(float(np.max(np.abs(alive[:, 0]))))
        self.y_max.append(float(np.max(np.abs(alive[:, 2]))))

        # 4-D invariant ε_4D = √det(σ_4D).  σ_4D = σ_matrix[0:4, 0:4].
        # For a solenoid-coupled beam this stays constant even though the
        # 2-D projections ε_x and ε_y oscillate with rotation.  Units:
        # (mm·mrad)² — take sqrt of the 4×4 determinant.
        s4 = m["sigma_matrix"][0:4, 0:4]
        det4 = float(np.linalg.det(s4))
        self.emit_4d.append(float(np.sqrt(max(det4, 0.0))))

        # Normal-mode (eigen-) emittances of the 4-D Σ block.
        # The two transverse normal-mode emittances are |Im(λ_i)| where λ
        # are eigenvalues of (J · Σ_4D), with J the 4×4 symplectic form.
        # In the absence of x–y coupling these reduce to ε_x and ε_y; under
        # solenoid coupling they remain constants of motion.  Sorted so
        # ε_1 ≥ ε_2.
        e1, e2 = _normal_mode_emittances(s4)
        self.emit_n1.append(e1)
        self.emit_n2.append(e2)

        # 6-D eigenemittances (Balandin invariants).  Units are mm·mrad
        # for ε₁/ε₂ (transverse) and "deg·MeV / longitudinal-Jacobian"
        # for ε₃ — but the Σ-matrix is in the recorder's mixed
        # mm/mrad/deg/MeV units so ε₃ comes out with units of
        # mm·mrad·deg·MeV / (mm·mrad)² = deg·MeV (matches ε_z above).
        ee1, ee2, ee3 = eigenemittances(m["sigma_matrix"])
        self.emit_e1.append(ee1)
        self.emit_e2.append(ee2)
        self.emit_e3.append(ee3)

        # 2-D density column for this s-step (no-op when not configured).
        self.record_density(beam)
        # Tail quantiles (no-op when not configured — HALO-PIC M1).
        self._record_tail(alive)

    # ------------------------------------------------------------------
    # 2-D density vs s
    # ------------------------------------------------------------------
    # particles[:, k] column index per axis label.  Internal beam-coord
    # units match the rest of the codebase: mm / mrad / deg / MeV (the
    # transverse positions are NOT in metres — sigma_x is computed direct
    # from particles[:, 0] and is labelled "mm" everywhere).
    _DENSITY_COL_MAP = {
        "x":   0,   # mm
        "xp":  1,   # mrad
        "y":   2,   # mm
        "yp":  3,   # mrad
        "phi": 4,   # deg (φ relative to ref)
        "w":   5,   # MeV (kinetic, relative to ref)
    }

    def configure_tail(self, fractions=(0.99, 0.999)) -> None:
        """Opt in to per-step tail diagnostics (HALO-PIC instrumentation).

        Records, at every :meth:`record` call, the fractional (quantile)
        emittances per transverse plane and the normalized radial
        quantiles for each requested fraction.  Keys:

            emit_x_q99, emit_y_q99, emit_x_q999, ..., r_q99, r_q999

        Zero overhead when not configured (the default).  When configured
        it adds O(N log N) quantile passes per record step — noticeable
        at N >~ 1e6 with dense recording.
        """
        from linac_gen.diagnostics.tail import frac_key
        self.tail_fractions = tuple(float(f) for f in fractions)
        self.tail: dict[str, list] = {}
        for f in self.tail_fractions:
            k = frac_key(f)
            self.tail[f"emit_x_{k}"] = []
            self.tail[f"emit_y_{k}"] = []
            self.tail[f"r_{k}"] = []

    def _record_tail(self, alive) -> None:
        if not getattr(self, "tail_fractions", None):
            return
        from linac_gen.diagnostics.tail import (
            compute_fractional_emittance, compute_radial_quantiles, frac_key)
        if len(alive) == 0:
            for lst in self.tail.values():
                lst.append(0.0)
            return
        ex = compute_fractional_emittance(alive, self.tail_fractions, "x")
        ey = compute_fractional_emittance(alive, self.tail_fractions, "y")
        rq = compute_radial_quantiles(alive, self.tail_fractions)
        for f in self.tail_fractions:
            k = frac_key(f)
            self.tail[f"emit_x_{k}"].append(ex[f])
            self.tail[f"emit_y_{k}"].append(ey[f])
            self.tail[f"r_{k}"].append(rq[f])

    def configure_density(self, axes=("x", "y"),
                          extent: dict | None = None,
                          n_bins: int = 200) -> None:
        """Opt in to 2-D density-vs-s recording.

        Parameters
        ----------
        axes
            Which beam coordinates to histogram.  Any of ``x, xp, y, yp,
            phi, w``.
        extent
            ``{axis: (lo, hi)}`` mapping in *internal* tracking units
            (m, rad, deg, MeV).  Missing axes are auto-fitted on the
            first call to :meth:`record_density` using the live beam
            min/max with 10 % padding.
        n_bins
            Bins per histogram column.  200 is a good interactive
            default — 1000 steps × 200 bins × 4 B ≈ 0.8 MB per axis.
        """
        self.density_axes = tuple(axes)
        self.density_extent = dict(extent or {})
        self.density_n_bins = int(n_bins)
        # Reset state so a re-run starts fresh.
        self.density = {a: [] for a in self.density_axes}
        self.density_edges = {}

    def record_density(self, beam) -> None:
        """Append one histogram column for each configured axis.

        Cheap: ``np.histogram`` of N alive particles into 200 bins is
        ~5–20 µs at N=10 K.  Skips silently when no axes were
        configured, when the beam is empty, or when an axis label is
        unknown.
        """
        if not self.density_axes:
            return
        alive = beam.alive_particles
        if len(alive) == 0:
            # Append zero columns so the result is rectangular vs the
            # main s-array — keeps shape consistent for downstream plots.
            for axis in self.density_axes:
                edges = self.density_edges.get(axis)
                n = (len(edges) - 1) if edges is not None else self.density_n_bins
                self.density.setdefault(axis, []).append(
                    np.zeros(n, dtype=np.int32)
                )
            return
        for axis in self.density_axes:
            col = self._DENSITY_COL_MAP.get(axis)
            if col is None:
                continue
            data = alive[:, col]
            edges = self.density_edges.get(axis)
            if edges is None:
                lo, hi = self.density_extent.get(axis, (None, None))
                if lo is None or hi is None:
                    # Auto-fit on first record.  Use 5·std × 1.5 margin
                    # so the core occupies most of the bins (good
                    # visual resolution) and the beam has headroom to
                    # grow downstream without saturating the edges.
                    # Falls back to ±max(|data|) when the centred std
                    # collapses (single particle, dead beam, etc.).
                    if data.size:
                        m = float(np.mean(data))
                        s = float(np.std(data))
                        span = 5.0 * s * 1.5
                        if span <= 0.0:
                            span = float(np.max(np.abs(data - m))) * 1.1
                        if span <= 0.0:
                            span = 1.0
                        lo, hi = m - span, m + span
                    else:
                        lo, hi = -1.0, 1.0
                edges = np.linspace(lo, hi, self.density_n_bins + 1)
                self.density_edges[axis] = edges
            counts, _ = np.histogram(data, bins=edges)
            self.density.setdefault(axis, []).append(counts.astype(np.int32))

    def density_array(self, axis: str) -> "np.ndarray | None":
        """Return the recorded density for ``axis`` as a 2-D ``int32``
        array of shape ``(n_steps, n_bins)`` — one row per s-step.
        Returns ``None`` if no density was recorded for that axis."""
        cols = self.density.get(axis)
        if not cols:
            return None
        return np.asarray(cols, dtype=np.int32)

    def save_snapshot(self, beam, s_position: float) -> None:
        """Save full particle snapshot + reference state (+ alive mask)."""
        self._snapshots[s_position] = (beam.particles.copy(), beam.ref.copy())
        # `lost` may be absent on minimal test beams; default to all-alive.
        lost = getattr(beam, "lost", None)
        self._snapshot_masks[s_position] = (
            None if lost is None else np.asarray(lost).copy())

    def beam_at(self, s_position: float):
        """Retrieve saved snapshot. Returns (particles, ref) or raises KeyError."""
        return self._snapshots[s_position]

    def alive_at(self, s_position: float):
        """Alive particles of the snapshot at ``s_position`` (lost filtered),
        or all particles if no mask was recorded.  Raises KeyError if there
        is no snapshot there."""
        particles, _ref = self._snapshots[s_position]
        lost = self._snapshot_masks.get(s_position)
        if lost is None or len(lost) != len(particles):
            return particles
        return particles[~lost]
