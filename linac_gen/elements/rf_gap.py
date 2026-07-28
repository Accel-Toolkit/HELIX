# linac_gen/elements/rf_gap.py
"""RF gap element: thin kick that accelerates particles and provides transverse RF defocusing."""
import numpy as np
from linac_gen.elements.base import ThinKickElement
from linac_gen.elements.mixins import Misalignment, FieldError


class RFGap(ThinKickElement, Misalignment, FieldError):
    """Thin RF gap with energy kick, adiabatic damping, and RF defocusing.

    Parameters
    ----------
    name : str
        Element name.
    voltage : float
        Peak voltage V0 (MV).
    phase : float
        Synchronous phase phi_s (deg).
    frequency : float
        RF frequency (MHz).
    ttf : float
        Transit time factor T (dimensionless), default 1.0.
    aperture : float
        Aperture radius (mm), default 0.0 (no aperture check).
    voltage_rel, phase_offset, frequency_offset : float
        Per-seed cavity errors:
          * ``voltage_rel`` — relative amplitude error (V → V·(1+δ))
          * ``phase_offset`` — additive phase error in degrees
          * ``frequency_offset`` — additive frequency error in MHz
    """

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  kick_matrix uses effective_voltage (= voltage × (1+voltage_rel)),
    # effective_phase (= phase + phase_offset), effective_frequency
    # (= frequency + frequency_offset), and ttf.  p_flag governs which
    # phase branch is in effect.
    _cache_keys: tuple[str, ...] = (
        "voltage", "voltage_rel", "phase", "phase_offset",
        "frequency", "frequency_offset", "ttf", "p_flag",
    )

    def __init__(self, name: str, voltage: float, phase: float, frequency: float,
                 ttf: float = 1.0, aperture: float = 0.0,
                 p_flag: int = 0,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 voltage_rel: float = 0.0,
                 phase_offset: float = 0.0,
                 frequency_offset: float = 0.0):
        super().__init__(name=name, aperture=aperture)
        self.voltage = voltage      # design peak voltage V0 (MV)
        self.phase = phase          # design synchronous phase phi_s (deg)
        self.frequency = frequency  # design RF frequency (MHz)
        self.ttf = ttf              # transit time factor T (dimensionless)
        self.p_flag = p_flag        # TraceWin phase-mode flag (0 rel / 1 abs / ...)
        self._init_misalignment(dx=dx, dy=dy, dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        self._init_field_error(voltage_rel=voltage_rel,
                               phase_offset=phase_offset,
                               frequency_offset=frequency_offset)

    @property
    def effective_voltage(self) -> float:
        return self.voltage * (1.0 + self.voltage_rel)

    @property
    def effective_phase(self) -> float:
        return self.phase + self.phase_offset

    @property
    def effective_frequency(self) -> float:
        return self.frequency + self.frequency_offset

    def apply_kick(self, beam) -> None:
        """Apply energy kick, adiabatic damping, and RF defocusing to beam."""
        ref = beam.ref
        charge = ref.species.charge    # integer charge state (e.g. +1 or -1)
        mass = ref.species.mass        # MeV/c^2
        # Effective values include per-seed cavity errors (voltage_rel,
        # phase_offset, frequency_offset).  All physics below uses them.
        eff_V = self.effective_voltage
        eff_phase = self.effective_phase
        eff_freq = self.effective_frequency

        phi_s_rad = eff_phase * np.pi / 180.0

        # Save old momentum for adiabatic damping
        bg_old = ref.bg

        # Synchronous energy gain
        dW_sync = charge * eff_V * self.ttf * np.cos(phi_s_rad)  # MeV

        # Update reference particle energy and frequency
        ref.w_kin += dW_sync
        # FREQ-jump rescale: beam.particles[:,4] is in deg-at-ref.frequency, so
        # multiply by f_new/f_old BEFORE updating ref.frequency to preserve the
        # physical phase deviation (Liouville).  Mirrors field_map_3d._track_kd.
        if eff_freq > 0 and eff_freq != ref.frequency:
            ratio = eff_freq / ref.frequency
            beam.particles[beam.alive_mask, 4] *= ratio
            ref.frequency = eff_freq
        bg_new = ref.bg

        # --- Particle kicks (alive particles only) ---
        alive = beam.alive_mask
        dphi = beam.particles[alive, 4]  # phase deviation (deg)
        phi_i_rad = phi_s_rad + dphi * np.pi / 180.0

        # Energy deviation kick: difference from synchronous gain
        dW_i = (charge * eff_V * self.ttf * np.cos(phi_i_rad)
                - charge * eff_V * self.ttf * np.cos(phi_s_rad))
        beam.particles[alive, 5] += dW_i

        # Adiabatic damping of transverse slopes
        damping = bg_old / bg_new
        beam.particles[alive, 1] *= damping  # x' (mrad)
        beam.particles[alive, 3] *= damping  # y' (mrad)

        # RF defocusing (applied after damping)
        #   k_rf = -pi * q * V0 * T * sin(phi_s) / (m * bg^3 * lambda)
        #
        # Convention note: we evaluate bg at the POST-acceleration reference
        # state (bg_new) and use the post-acceleration RF wavelength.  This
        # choice ("kick after acceleration") matches several linac codes but
        # differs from conventions that use pre-gap bg or the geometric mean
        # sqrt(bg_old * bg_new).  The difference is ~(Delta bg / bg)^3 per
        # gap; for typical proton linacs (bg changes <1% per gap) it is well
        # below tracking tolerances.  Do not change without updating
        # test_rf_defocusing_magnitude -- it locks in this formula.
        wl = ref.wavelength  # mm (now consistent with the effective frequency)
        k_rf = (-np.pi * charge * eff_V * self.ttf * np.sin(phi_s_rad)
                / (mass * bg_new**3 * wl))
        beam.particles[alive, 1] += k_rf * beam.particles[alive, 0] * 1e3  # mrad
        beam.particles[alive, 3] += k_rf * beam.particles[alive, 2] * 1e3  # mrad

    def inverse_kick(self, beam, ref_entry) -> None:
        """Exactly undo :meth:`apply_kick` (backward tracking).

        Called with ``beam.ref`` still holding the gap's EXIT reference
        state (post-acceleration bg, effective frequency) and
        ``ref_entry`` the recorded ENTRANCE state from the forward
        replay table.  Operations run in the exact reverse of the
        forward order:

            forward:  ① ref dW_sync  ② dphi ×= f_new/f_old, ref.freq=f_new
                      ③ per-particle dW at post-rescale dphi
                      ④ damping ×(bg_old/bg_new)  ⑤ defocus (bg_new, λ_new)
            inverse:  ⑤⁻¹ subtract defocus (same k_rf: bg_new, λ_new)
                      ④⁻¹ divide damping
                      ③⁻¹ subtract dW at CURRENT dphi (still post-rescale —
                          the forward kick read exactly these values, and
                          steps ③④⑤ never modify dphi)
                      ②⁻¹ dphi ×= f_entry/eff_freq
            (the caller then restores beam.ref from the replay table,
             which is ①⁻¹.)
        """
        ref = beam.ref                       # EXIT state
        charge = ref.species.charge
        mass = ref.species.mass
        eff_V = self.effective_voltage
        eff_phase = self.effective_phase
        eff_freq = self.effective_frequency

        phi_s_rad = eff_phase * np.pi / 180.0
        bg_new = ref.bg                      # post-acceleration
        bg_old = ref_entry.bg                # pre-acceleration (table)

        alive = beam.alive_mask

        # ⑤⁻¹ RF defocus — identical k_rf formula (bg_new + the exit
        # wavelength, which ref still carries).
        wl = ref.wavelength
        k_rf = (-np.pi * charge * eff_V * self.ttf * np.sin(phi_s_rad)
                / (mass * bg_new**3 * wl))
        beam.particles[alive, 1] -= k_rf * beam.particles[alive, 0] * 1e3
        beam.particles[alive, 3] -= k_rf * beam.particles[alive, 2] * 1e3

        # ④⁻¹ adiabatic damping
        damping = bg_old / bg_new
        beam.particles[alive, 1] /= damping
        beam.particles[alive, 3] /= damping

        # ③⁻¹ per-particle energy kick at the CURRENT (post-rescale) dphi
        dphi = beam.particles[alive, 4]
        phi_i_rad = phi_s_rad + dphi * np.pi / 180.0
        dW_i = (charge * eff_V * self.ttf * np.cos(phi_i_rad)
                - charge * eff_V * self.ttf * np.cos(phi_s_rad))
        beam.particles[alive, 5] -= dW_i

        # ②⁻¹ FREQ-jump rescale back to the entrance frequency
        f_entry = ref_entry.frequency
        if eff_freq > 0 and eff_freq != f_entry:
            beam.particles[alive, 4] *= f_entry / eff_freq

    def advance_ref(self, ref) -> None:
        """Update reference energy and frequency for envelope/matrix tracking."""
        charge = ref.species.charge
        phi_s_rad = self.effective_phase * np.pi / 180.0
        dW_sync = charge * self.effective_voltage * self.ttf * np.cos(phi_s_rad)
        ref.w_kin += dW_sync
        if self.effective_frequency != ref.frequency:
            ref.frequency = self.effective_frequency

    def kick_matrix(self, ref) -> np.ndarray:
        """Linearized 6x6 matrix for envelope mode.

        Does NOT modify the reference particle (uses a copy for computing
        post-acceleration quantities).
        """
        M = np.eye(6)
        charge = ref.species.charge
        mass = ref.species.mass
        eff_V = self.effective_voltage
        eff_phase = self.effective_phase
        eff_freq = self.effective_frequency
        phi_s_rad = eff_phase * np.pi / 180.0

        # Compute post-acceleration bg using a copy (don't modify ref)
        bg_old = ref.bg
        dW_sync = charge * eff_V * self.ttf * np.cos(phi_s_rad)
        ref_copy = ref.copy()
        ref_copy.w_kin += dW_sync
        if eff_freq != ref_copy.frequency:
            ref_copy.frequency = eff_freq
        bg_new = ref_copy.bg
        wl = ref_copy.wavelength  # mm

        damping = bg_old / bg_new

        # RF defocusing strength
        k_rf = (-np.pi * charge * eff_V * self.ttf * np.sin(phi_s_rad)
                / (mass * bg_new**3 * wl))

        # Transverse: damping + RF defocusing
        M[1, 1] = damping                          # x' adiabatic damping
        M[3, 3] = damping                          # y' adiabatic damping
        M[1, 0] = k_rf * 1e3                          # RF defocusing in x (mrad/mm)
        M[3, 2] = k_rf * 1e3                          # RF defocusing in y

        # Longitudinal: the full gap map is K·D, mirroring apply_kick's
        # operation order — the FREQ-jump phase rescale D (dphi is stored
        # in degrees-at-ref.frequency, so a jump re-expresses it by
        # r = f_new/f_old) happens BEFORE the energy shear K reads dphi:
        #     D = diag(…, r, 1),   K = [[1, 0], [k_phi_W, 1]]  (dphi, dW)
        #     K·D → M[4,4] = r,  M[5,4] = k_phi_W · r
        # r = 1 whenever the caller's ref is already at the gap frequency
        # (the envelope pre-rescales σ and updates ref.frequency before
        # requesting this matrix, so it composes its own D exactly once;
        # matrix tracking calls at the entry frequency and needs K·D).
        # dW = q*V0*T*cos(phi_s + dphi) - q*V0*T*cos(phi_s)
        #    ~ -q*V0*T*sin(phi_s) * dphi_rad
        #    = -q*V0*T*sin(phi_s) * dphi_deg * pi/180
        r = 1.0
        if eff_freq > 0 and ref.frequency > 0 and eff_freq != ref.frequency:
            r = eff_freq / ref.frequency
        M[4, 4] = r
        M[5, 4] = (-charge * eff_V * self.ttf * np.sin(phi_s_rad)
                   * np.pi / 180.0) * r

        return M
