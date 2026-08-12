# linac_gen/elements/field_map.py
"""FieldMap element — 1-D, 2-D cylindrical, and 1-D quad-gradient tracking.

Supports geometries:
  * 1  — 1-D on-axis tabulation; paraxial off-axis via Er = -(r/2)·dFz/dz
  * 4  — 2-D cyl E-type (rf_E) with bilinear (r, z) interpolation;
          optional Fq = Bθ (TM-mode cross-field component)
  * 5  — 2-D cyl B-type (rf_B) with bilinear (r, z) interpolation;
          optional Fq = Eθ (TE-mode cross-field component)
  * 9  — 1-D quad gradient G(z) in T/m; Bx = G·y, By = G·x

Phasor rules (TraceWin manual §18122):
  * static E or B : phasor = 1
  * RF electric   : phasor = cos(φ)
  * RF magnetic   : phasor = sin(φ)

Lorentz kick (engineering units):
  E in MV/m, B in T, positions in mm, angles in mrad, Δs in m.

    Δx' [rad] = q · Δs · E_x / (γβ² · mc²)
              + q · Δs · (y'·Bz − By) · 299.792458 / (γβ · mc²)
    Δy' [rad] = q · Δs · E_y / (γβ² · mc²)
              + q · Δs · (Bx − x'·Bz) · 299.792458 / (γβ · mc²)

where q is in units of e, mc² in MeV, and the factor 299.792458 = c/(10⁶)
converts B[T] to MV/m-equivalent units.
"""
from __future__ import annotations

import numpy as np

from linac_gen.elements.base import FieldMapElement as FieldMapBase
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking.rk4 import numerical_jacobian

# Speed-of-light conversion c [m/s] × 1e-6 [MeV/V].
# Ensures B[T] × _C_LIGHT is numerically equivalent to E[MV/m].
_C_LIGHT = 299.792458


def _bilinear_2d(Fz_grid: np.ndarray, r_axis: np.ndarray, z_axis: np.ndarray,
                 r: float, z: float) -> float:
    """Bilinear interpolation on a 2-D (r, z) grid.

    ``Fz_grid`` must be laid out as ``(n_r, n_z)`` — i.e. first index is r,
    second is z.  Out-of-range values are clamped to the boundary edge.
    Returns 0.0 if *Fz_grid* is None.
    """
    if Fz_grid is None:
        return 0.0
    r = float(np.clip(r, r_axis[0], r_axis[-1]))
    z = float(np.clip(z, z_axis[0], z_axis[-1]))
    ir = int(np.searchsorted(r_axis, r)) - 1
    iz = int(np.searchsorted(z_axis, z)) - 1
    ir = max(0, min(ir, len(r_axis) - 2))
    iz = max(0, min(iz, len(z_axis) - 2))
    r0, r1 = r_axis[ir], r_axis[ir + 1]
    z0, z1 = z_axis[iz], z_axis[iz + 1]
    tr = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)
    tz = 0.0 if z1 == z0 else (z - z0) / (z1 - z0)
    v00 = Fz_grid[ir,     iz]
    v10 = Fz_grid[ir + 1, iz]
    v01 = Fz_grid[ir,     iz + 1]
    v11 = Fz_grid[ir + 1, iz + 1]
    return float((1 - tr) * (1 - tz) * v00 + tr * (1 - tz) * v10
                 + (1 - tr) * tz * v01 + tr * tz * v11)


def _normalize_2d_grid(grid: np.ndarray, r_axis: np.ndarray,
                       z_axis: np.ndarray) -> np.ndarray:
    """Return *grid* in (n_r, n_z) layout, transposing if needed.

    The FieldChannel stores Fz in whichever layout the reader used.
    Detect which index aligns with r and which with z, and return (n_r, n_z).
    """
    if grid is None:
        return None
    nr, nz = len(r_axis), len(z_axis)
    if grid.shape == (nr, nz):
        return grid
    if grid.shape == (nz, nr):
        return grid.T
    # Last resort: trust shape[0] == nz → transpose
    if grid.shape[0] == nz:
        return grid.T
    return grid


from linac_gen.elements.mixins import Misalignment, FieldError


class FieldMap(FieldMapBase, Misalignment, FieldError):
    """Element tracked through an external field map.

    Handles 1-D (geometry=1), 2-D cylindrical (geometry=4 or 5), and
    1-D quad-gradient (geometry=9) channels.  For 3-D Cartesian maps
    use :class:`~linac_gen.elements.field_map_3d.FieldMap3D`.

    Parameters
    ----------
    name : str
        Element name.
    length : float
        Physical length (mm).
    field_data : FieldMapData
        Field data; each channel's geometry must be 1, 4, 5, or 9.
    scale : float
        Global field amplitude multiplier.
    kb : float
        Additional magnetic-channel amplitude factor (default 1.0).
    ke : float
        Additional electric-channel amplitude factor (default 1.0).
    ki : float
        Space-charge compensation intensity factor (default 0.0).
    ka : int
        Aperture flag (default 1).
    phase : float
        RF phase offset in degrees.
    frequency : float
        RF frequency in MHz.
    aperture : float
        Aperture radius (mm); 0 = no check.
    n_steps : int
        Number of integration sub-steps.
    p_flag : int
        Phase reference flag (0 = relative, not yet fully implemented).
    """

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  fitted_matrix runs an RK4 jacobian through the 1-D / 2-D
    # field map — it reads length, scale, ke, kb, ki, ka, phase,
    # frequency, voltage_rel, phase_offset, frequency_offset, n_steps,
    # p_flag.  field_data is immutable per-instance.
    _cache_keys: tuple[str, ...] = (
        "length", "scale", "phase", "phase_offset",
        "frequency", "frequency_offset",
        "ke", "kb", "ki", "ka", "voltage_rel", "n_steps", "p_flag",
    )

    # Multibunch hybrid-replay channel (linac_gen/train/replay.py):
    # static transverse HOM kick [mrad] consumed once at element ENTRY
    # by the Tracker, applied to every alive particle.  Class-level
    # default 0.0 = inert (normal runs bit-identical); set only by the
    # M6 replay override transport and cleared by its teardown.
    hom_kick_x: float = 0.0
    hom_kick_y: float = 0.0

    def __init__(self, name: str, length: float, field_data: FieldMapData,
                 scale: float = 1.0,
                 kb: float = 1.0, ke: float = 1.0,
                 ki: float = 0.0, ka: int = 1,
                 phase: float = 0.0, frequency: float = 0.0,
                 aperture: float = 0.0, n_steps: int = 100,
                 p_flag: int = 0,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 voltage_rel: float = 0.0,
                 phase_offset: float = 0.0,
                 frequency_offset: float = 0.0,
                 field_file: str | None = None, geom: int | None = None):
        super().__init__(name=name, length=length, aperture=aperture,
                         n_steps=n_steps)
        self.field_data = field_data
        self.scale = scale
        self.kb = kb
        self.ke = ke
        self.ki = ki
        self.ka = ka
        self.phase = phase
        self.frequency = frequency
        self.p_flag = p_flag
        # Provenance for round-tripping back to a FIELD_MAP .dat card: the
        # resolved field-file prefix and the raw geom code (set by the parser
        # via field_map_factory).  None for programmatically-built maps, in
        # which case the writer falls back to a comment instead of a card.
        self.field_file = field_file
        self.geom = geom
        self._z_map_start = float(field_data.z[0])
        self._z_map_end = float(field_data.z[-1])
        self._step_idx = 0
        # Scan-fit SET_SYNC_PHASE offset ψ (deg) — populated lazily at
        # first track_rk4/advance_ref call when p_flag=1.
        self._sync_offset_deg: float | None = None
        self._phi_s_at_entrance: float = 0.0
        self._init_misalignment(dx=dx, dy=dy, dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        # Cavity-style errors: voltage_rel multiplies ke/kb (E and B amplitudes),
        # phase_offset adds to phase, frequency_offset adds to frequency.
        # Multibunch train mode: design-pass SET_SYNC_PHASE pin (see _calibrate_sync_phase); None = normal lazy calibration.
        self.sync_phase_pin = None
        self._init_field_error(voltage_rel=voltage_rel,
                               phase_offset=phase_offset,
                               frequency_offset=frequency_offset)

    @property
    def effective_phase(self) -> float:
        return self.phase + self.phase_offset

    @property
    def effective_frequency(self) -> float:
        return self.frequency + self.frequency_offset

    @property
    def effective_ke(self) -> float:
        return self.ke * (1.0 + self.voltage_rel)

    @property
    def effective_kb(self) -> float:
        return self.kb * (1.0 + self.voltage_rel)
        # Scan-fit SET_SYNC_PHASE offset ψ (deg) — populated lazily at
        # first track_rk4/advance_ref call when p_flag=1.
        self._sync_offset_deg: float | None = None

    # ------------------------------------------------------------------ #
    #  Class method constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_file(cls, name: str, filepath: str, length: float = None,
                  scale: float = 1.0,
                  kb: float = 1.0, ke: float = 1.0,
                  ki: float = 0.0, ka: int = 1,
                  phase: float = 0.0,
                  frequency: float = 0.0, aperture: float = 0.0,
                  n_steps: int = 100, fm_type: int = 1,
                  p_flag: int = 0) -> "FieldMap":
        """Create a FieldMap element from a field map file.

        Parameters
        ----------
        name : str
            Element name.
        filepath : str
            Path to the field map file (.edz or .csv).
        length : float, optional
            Physical length (mm).  If *None*, inferred from ``z[-1] - z[0]``
            in the field data.
        scale : float
            Field amplitude scaling factor.
        phase : float
            RF phase offset (degrees).
        frequency : float
            RF frequency (MHz).
        aperture : float
            Aperture radius (mm).
        n_steps : int
            Number of integration sub-steps.
        fm_type : int
            Field map type: 1 = 1D, 2 = 2D electric, 7 = 2D E+B.

        Returns
        -------
        FieldMap
        """
        from linac_gen.io.field_map_reader import read_field_map
        fdata = read_field_map(filepath, fm_type=fm_type)
        if length is None:
            length = float(fdata.z[-1] - fdata.z[0])
        return cls(name=name, length=length, field_data=fdata, scale=scale,
                   kb=kb, ke=ke, ki=ki, ka=ka,
                   phase=phase, frequency=frequency, aperture=aperture,
                   n_steps=n_steps, p_flag=p_flag)

    # ------------------------------------------------------------------ #
    #  Back-compat interpolation helper (used by legacy tests)
    # ------------------------------------------------------------------ #

    def _interpolate_Ez(self, z_pos: float) -> float:
        """Linearly interpolate Ez on axis at position *z_pos* (mm).

        For 1-D maps Ez is a 1-D array.  For 2-D (cylindrical) maps the
        on-axis (r=0) slice is used.  Back-compat with legacy callers.
        """
        fd = self.field_data
        ez = fd.Ez
        if ez is None:
            return 0.0
        z_arr = fd.z
        ez1d = ez if ez.ndim == 1 else ez[0, :]
        return float(np.interp(z_pos, z_arr, ez1d, left=0.0, right=0.0))

    # ------------------------------------------------------------------ #
    #  Internal helpers — channel dispatch
    # ------------------------------------------------------------------ #

    def _phi_sync_rad(self, ref, phi_s_midpoint_deg: float) -> float:
        """Synchronous phase in radians at the slice midpoint.

        * p_flag == 0 (relative): φ_sync = element.phase + phi_s_midpoint.
        * p_flag == 1 (SET_SYNC_PHASE): use the scan-fit offset ψ
          pre-computed by :meth:`_calibrate_sync_phase`.
              phi_sync_rad = (self.phase − ψ) + phi_s_midpoint.
        """
        if self.p_flag == 1:
            offset = getattr(self, "_sync_offset_deg", None) or 0.0
            internal = phi_s_midpoint_deg - self._phi_s_at_entrance
            return (self.effective_phase - offset + internal) * (np.pi / 180.0)
        return (self.effective_phase + phi_s_midpoint_deg) * (np.pi / 180.0)

    # -- Iterative self-consistent SET_SYNC_PHASE calibration ------------
    def _probe_voltage_integral(self, ref_entry,
                                 phi_input_deg: float) -> tuple[float, float]:
        """Integrate V_c = Σ q·E_z·e^{i·φ_gen(z)}·ds with self-consistent
        β-evolution of the reference.  See the matching docstring on
        ``FieldMap3D._probe_voltage_integral``.
        """
        rf = ref_entry.copy()
        rf.phi_s = 0.0
        n = self.n_steps
        ds = self.length / n
        ds_m = ds * 1e-3
        charge = rf.species.charge
        phi_input_rad = phi_input_deg * np.pi / 180.0
        Re_V = 0.0
        Im_V = 0.0
        for i in range(n):
            z_pos = self._z_map_start + (i + 0.5) * ds
            if rf.wavelength > 0:
                dphi_to_mid = 180.0 * ds / (rf.beta * rf.wavelength)
            else:
                dphi_to_mid = 0.0
            phi_s_mid_rad = (rf.phi_s + dphi_to_mid) * np.pi / 180.0
            Ez_sum = 0.0
            for ch_enum, ch in self.field_data.channels.items():
                if not ch_enum.is_electric:
                    continue
                if ch.geometry == 9:
                    continue
                Fz_ax = self._sample_Fz_on_axis(ch, z_pos)
                scale = self._scale(ch_enum, ch)
                Ez_sum += Fz_ax * scale
            dW_slice = charge * Ez_sum * np.cos(phi_input_rad + phi_s_mid_rad) * ds_m
            Re_V += charge * Ez_sum * np.cos(phi_s_mid_rad) * ds_m
            Im_V += charge * Ez_sum * np.sin(phi_s_mid_rad) * ds_m
            rf.w_kin += dW_slice
            rf.s += ds
            if rf.wavelength > 0:
                rf.phi_s += 360.0 * ds / (rf.beta * rf.wavelength)
        return Re_V, Im_V

    def _calibrate_sync_phase(self, ref_entry,
                               tol_deg: float = 0.005,
                               max_iter: int = 30) -> None:
        """Iterative self-consistent SET_SYNC_PHASE calibration, probed at
        the actual operating point.  See the matching docstring on
        ``FieldMap3D._calibrate_sync_phase``."""
        if self.p_flag != 1 or getattr(self, "_sync_offset_deg", None) is not None:
            return
        pin = getattr(self, "sync_phase_pin", None)
        if pin is not None:
            # Multibunch/train mode: the LLRF fixes the operating point
            # from the DESIGN pass; later passes must not re-fit psi
            # against a loaded (perturbed) voltage.  reset_run_state()
            # clears _sync_offset_deg but never the pin.
            self._sync_offset_deg = float(pin)
            return
        theta_s = float(self.effective_phase)
        psi = 0.0
        relax = 0.7
        for _ in range(max_iter):
            re_v, im_v = self._probe_voltage_integral(ref_entry, theta_s - psi)
            psi_new = float(np.rad2deg(np.arctan2(im_v, re_v)))
            if abs(psi_new - psi) < tol_deg:
                psi = psi_new
                break
            psi = (1.0 - relax) * psi + relax * psi_new
        self._sync_offset_deg = float(psi)

    def _scale(self, ch_enum: Channel, ch: FieldChannel) -> float:
        """Combined amplitude: global scale × k_e/k_b × 1/norm_factor.

        Uses ``effective_ke`` / ``effective_kb`` so the per-seed
        ``voltage_rel`` cavity-amplitude error is honoured by every
        physics path (RK4, slice fits, advance_ref).
        """
        k = self.effective_ke if ch_enum.is_electric else self.effective_kb
        return (k / ch.norm_factor) * self.scale

    def _sample_Fz_on_axis(self, ch: FieldChannel, z_pos: float) -> float:
        """On-axis longitudinal field at *z_pos* [mm] for channel *ch*."""
        g = ch.geometry
        if g == 1 or g == 9:
            # 1-D: Fz is a 1-D array
            return float(np.interp(z_pos, ch.z, ch.Fz, left=0.0, right=0.0))
        if g in (4, 5):
            # 2-D cyl: on-axis = r=0 slice.  Determine layout.
            fz_nr = _normalize_2d_grid(ch.Fz, ch.r, ch.z)
            # fz_nr[0, :] is the r=0 (on-axis) slice
            return float(np.interp(z_pos, ch.z, fz_nr[0, :],
                                   left=0.0, right=0.0))
        if g == 7:
            raise ValueError(
                "FieldMap doesn't handle geometry=7 (3-D Cartesian); "
                "use FieldMap3D."
            )
        raise ValueError(f"FieldMap doesn't handle geometry {g} on-axis")

    def _sample_cart(self, ch: FieldChannel,
                     xs: np.ndarray, ys: np.ndarray,
                     z_pos: float) -> tuple:
        """Return (Fx, Fy, Fz) arrays at each particle position.

        All arrays have shape (n_particles,).

        Geometry dispatch
        -----------------
        * 1 — on-axis Fz via np.interp; paraxial Fr = -(r/2)·dFz/dz,
              decomposed as Fx = Fr·x/r, Fy = Fr·y/r.
        * 4 or 5 — bilinear Fz(r, z) and Fr(r, z) from the (n_r, n_z)
              grid, decomposed to Cartesian.
        * 9 — 1-D quad gradient: Bx = G·y, By = G·x, Bz = 0.
        * 7 — 3-D Cartesian: raise; belongs to FieldMap3D.
        """
        g = ch.geometry
        n = len(xs)

        if g == 1:
            return self._sample_cart_1d(ch, xs, ys, z_pos)

        if g in (4, 5):
            return self._sample_cart_2d_cyl(ch, xs, ys, z_pos)

        if g == 9:
            return self._sample_cart_quad_grad(ch, xs, ys, z_pos)

        if g == 7:
            raise ValueError(
                "FieldMap._sample_cart: geometry=7 (3-D Cartesian) belongs "
                "to FieldMap3D, not FieldMap."
            )
        raise ValueError(f"FieldMap._sample_cart: unsupported geometry {g}")

    def _sample_cart_1d(self, ch: FieldChannel,
                        xs: np.ndarray, ys: np.ndarray,
                        z_pos: float) -> tuple:
        """Geometry=1: on-axis + paraxial off-axis (no *1e-3 bug)."""
        # On-axis longitudinal
        Fz_on = float(np.interp(z_pos, ch.z, ch.Fz, left=0.0, right=0.0))
        n = len(xs)
        Fz_arr = np.full(n, Fz_on)

        # Radial component from paraxial expansion: Fr = -(r/2)·dFz/dz
        # Units: r in mm, Fz in MV/m (or T), z in mm.
        # dFz/dz has units of (MV/m)/mm (or T/mm).
        # Fr = -(r_mm / 2) × dFz/dz_(MV/m/mm) has units MV/m (or T). Correct.
        dFz_dz = float(np.interp(
            z_pos, ch.z,
            np.gradient(ch.Fz, ch.z),
            left=0.0, right=0.0,
        ))
        rs = np.hypot(xs, ys)
        Fr_arr = -0.5 * rs * dFz_dz   # MV/m or T — no 1e-3 factor

        # Decompose Fr into Cartesian (avoid division by zero at r=0)
        safe_r = np.where(rs > 1e-12, rs, 1.0)
        Fx_arr = Fr_arr * (xs / safe_r)
        Fy_arr = Fr_arr * (ys / safe_r)
        # At exact r=0 the kick should be zero regardless of x/y ratio
        zero_mask = rs < 1e-12
        Fx_arr = np.where(zero_mask, 0.0, Fx_arr)
        Fy_arr = np.where(zero_mask, 0.0, Fy_arr)
        return Fx_arr, Fy_arr, Fz_arr

    def _sample_cart_2d_cyl(self, ch: FieldChannel,
                            xs: np.ndarray, ys: np.ndarray,
                            z_pos: float) -> tuple:
        """Geometry=4 or 5: bilinear (r, z) → Cartesian decomposition."""
        fz_nr = _normalize_2d_grid(ch.Fz, ch.r, ch.z)  # (n_r, n_z)
        fr_nr = _normalize_2d_grid(ch.Fr, ch.r, ch.z)   # may be None

        rs = np.hypot(xs, ys)
        n = len(xs)
        Fx_arr = np.zeros(n)
        Fy_arr = np.zeros(n)
        Fz_arr = np.zeros(n)

        for i in range(n):
            ri = float(rs[i])
            fz_val = _bilinear_2d(fz_nr, ch.r, ch.z, ri, z_pos)
            fr_val = _bilinear_2d(fr_nr, ch.r, ch.z, ri, z_pos) if fr_nr is not None else 0.0
            Fz_arr[i] = fz_val
            if ri > 1e-12:
                Fx_arr[i] = fr_val * (xs[i] / ri)
                Fy_arr[i] = fr_val * (ys[i] / ri)
        return Fx_arr, Fy_arr, Fz_arr

    def _sample_cart_quad_grad(self, ch: FieldChannel,
                               xs: np.ndarray, ys: np.ndarray,
                               z_pos: float) -> tuple:
        """Geometry=9: 1-D quad gradient G(z) in T/m → Bx=G·y, By=G·x."""
        G = float(np.interp(z_pos, ch.z, ch.Fz, left=0.0, right=0.0))
        # G in T/m; xs/ys in mm.  Convert to T: G[T/m] × r[mm] × 1e-3 = T.
        Bx = G * ys * 1e-3   # T
        By = G * xs * 1e-3   # T
        Bz = np.zeros(len(xs))
        return Bx, By, Bz

    def _sample_cart_cross(self, ch_enum: Channel, ch: FieldChannel,
                           xs: np.ndarray, ys: np.ndarray,
                           z_pos: float):
        """Return the cross-field-component (Fq) in Cartesian form, or None.

        For RF_E geometry=4 (TM mode): Fq is Bθ; returns (Bx, By, Bz).
        For RF_B geometry=5 (TE mode): Fq is Eθ; returns (Ex, Ey, Ez).
        Phasor for the cross-component follows its own field type:
          - Bθ from RF_E TM → magnetic → sin phasor
          - Eθ from RF_B TE → electric → cos phasor
        The caller is responsible for routing to the correct accumulator.

        Returns None if there is no Fq component.
        """
        if ch.Fq is None:
            return None
        g = ch.geometry
        if g not in (4, 5):
            return None

        fq_nr = _normalize_2d_grid(ch.Fq, ch.r, ch.z)  # (n_r, n_z)
        rs = np.hypot(xs, ys)
        n = len(xs)
        Fx = np.zeros(n)
        Fy = np.zeros(n)
        Fz = np.zeros(n)

        for i in range(n):
            ri = float(rs[i])
            fq_val = _bilinear_2d(fq_nr, ch.r, ch.z, ri, z_pos)
            # Fq is a θ-component; convert to Cartesian:
            # F_x = -Fq · (y/r),  F_y = +Fq · (x/r)  (manual §18143)
            if ri > 1e-12:
                Fx[i] = -fq_val * (ys[i] / ri)
                Fy[i] = +fq_val * (xs[i] / ri)
        return Fx, Fy, Fz

    # ------------------------------------------------------------------ #
    #  track_rk4
    # ------------------------------------------------------------------ #

    def track_rk4(self, beam, ds: float) -> None:
        """Track beam through one slice of length *ds* (mm).

        Step summary:
          1. Sample all channels at slice midpoint.
          2. Advance reference (on-axis Ez from electric channels).
          3. Per particle: accumulate E and B from all channels with
             correct phasor; longitudinal energy kick; transverse Lorentz
             kick; drift.
        """
        ref = beam.ref

        # Snapshot cavity entrance phase for p_flag=1 (sync-phase mode).
        if self._step_idx == 0:
            # FREQ-jump rescale of per-particle dphi.  See field_map_3d._track_kd
            # for the full rationale; mirrors envelope.py:528-537.  Without
            # this MP εnz collapses by f_old/f_new at the FREQ jump.
            eff_freq = self.effective_frequency
            if (eff_freq > 0
                    and eff_freq != ref.frequency
                    and self._has_electric_channel()):
                ratio = eff_freq / ref.frequency
                beam.particles[beam.alive_mask, 4] *= ratio
            # Propagate cavity frequency to ref BEFORE calibration (FREQ-
            # jump correctness — see field_map_3d._track_kd for rationale).
            self._propagate_frequency_to_ref(ref)
            self._phi_s_at_entrance = ref.phi_s
            if self.p_flag == 1 and self._sync_offset_deg is None:
                self._calibrate_sync_phase(ref)

        z_pos = self._z_map_start + self._step_idx * ds + ds / 2.0
        self._step_idx += 1

        ds_m = ds * 1e-3
        mass_MeV = ref.species.mass
        charge_num = ref.species.charge
        wavelength = ref.wavelength

        # Phase evaluated at the slice *midpoint* in phi_s, not the
        # entrance — matters through thick cavities (240 mm QWR sweeps
        # ~700° at 162.5 MHz).
        if wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref.beta * wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref.phi_s + dphi_to_mid
        phi_sync_rad = self._phi_sync_rad(ref, phi_s_mid)

        # ---- Reference advance (on-axis electric channels only) --------
        dW_ref = 0.0
        for ch_enum, ch in self.field_data.channels.items():
            if not ch_enum.is_electric:
                continue
            scale = self._scale(ch_enum, ch)
            if ch.geometry == 9:
                continue   # quad gradient has no axial E
            Fz_ax = self._sample_Fz_on_axis(ch, z_pos)
            if ch_enum.is_static:
                phasor = 1.0
            else:
                phasor = float(np.cos(phi_sync_rad))
            dW_ref += charge_num * Fz_ax * scale * phasor * ds_m

        ref.w_kin += dW_ref
        ref.s += ds
        if wavelength > 0:
            ref.phi_s += 360.0 * ds / (ref.beta * wavelength)

        # ---- Particle loop (alive only) --------------------------------
        alive = beam.alive_mask
        if not np.any(alive):
            return

        xs = beam.particles[alive, 0]
        ys = beam.particles[alive, 2]
        dphi_deg = beam.particles[alive, 4]
        n = int(np.sum(alive))

        Ex_tot = np.zeros(n)
        Ey_tot = np.zeros(n)
        Ez_tot = np.zeros(n)
        Bx_tot = np.zeros(n)
        By_tot = np.zeros(n)
        Bz_tot = np.zeros(n)

        phi_i = phi_sync_rad + dphi_deg * (np.pi / 180.0)  # shape (n,)

        for ch_enum, ch in self.field_data.channels.items():
            scale = self._scale(ch_enum, ch)

            # Phasor for primary component
            if ch_enum.is_static:
                phasor = np.ones(n)
            elif ch_enum.is_electric:
                phasor = np.cos(phi_i)
            else:  # RF magnetic
                phasor = np.sin(phi_i)

            # Primary field components
            Fx_p, Fy_p, Fz_p = self._sample_cart(ch, xs, ys, z_pos)

            if ch_enum.is_electric:
                Ex_tot += scale * phasor * Fx_p
                Ey_tot += scale * phasor * Fy_p
                Ez_tot += scale * phasor * Fz_p
            else:
                Bx_tot += scale * phasor * Fx_p
                By_tot += scale * phasor * Fy_p
                Bz_tot += scale * phasor * Fz_p

            # Cross-component: Bθ on RF_E geometry=4, or Eθ on RF_B geometry=5
            cross = self._sample_cart_cross(ch_enum, ch, xs, ys, z_pos)
            if cross is not None:
                Fx_c, Fy_c, Fz_c = cross
                # Phasor for cross-component follows its own field type:
                # RF_E+Fq → Bθ → magnetic → sin phasor
                # RF_B+Fq → Eθ → electric → cos phasor
                if ch_enum.is_electric:
                    # Bθ component → goes to B accumulator with sin phasor
                    phasor_c = np.sin(phi_i)
                    Bx_tot += scale * phasor_c * Fx_c
                    By_tot += scale * phasor_c * Fy_c
                    Bz_tot += scale * phasor_c * Fz_c
                else:
                    # Eθ component → goes to E accumulator with cos phasor
                    phasor_c = np.cos(phi_i)
                    Ex_tot += scale * phasor_c * Fx_c
                    Ey_tot += scale * phasor_c * Fy_c
                    Ez_tot += scale * phasor_c * Fz_c

        # ---- Longitudinal energy kick (relative to reference) ----------
        dW_i = charge_num * Ez_tot * ds_m
        beam.particles[alive, 5] += dW_i - dW_ref

        # ---- Adiabatic transverse damping ------------------------------
        # ΔW grows p_z, so x' = p_x/p_z shrinks.  Small-signal factor
        #     Δp_z/p_z ≈ ΔW / (β²γ·mc²)
        # Required for geometric emittance to track |p_z| through
        # acceleration — see field_map_3d.track_rk4 for the long-form
        # derivation.
        if ref.beta > 0 and ref.gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_i / (ref.beta ** 2 * ref.gamma * mass_MeV))
            beam.particles[alive, 1] *= damp
            beam.particles[alive, 3] *= damp

        # ---- Longitudinal drift slip (per-particle phase vs ref) -------
        # First-order drift-in-cavity phase slip matching Drift.transfer_matrix:
        #     Δφ += -360·ds/(β³γ³·m·λ) · ΔW
        if wavelength > 0 and ref.beta > 0 and ref.gamma > 0:
            slip = -360.0 * ds / (ref.beta ** 3 * ref.gamma ** 3
                                  * mass_MeV * wavelength)
            beam.particles[alive, 4] += slip * beam.particles[alive, 5]

        # ---- Transverse Lorentz kick -----------------------------------
        beta = ref.beta
        gamma = ref.gamma
        if beta > 0 and gamma > 0:
            # E contribution: Δx' = q·Δs·Ex / (γβ²·mc²)
            factor_E = ds_m / (gamma * beta * beta * mass_MeV)
            # B contribution: Δx' = q·Δs·(y'·Bz − By)·c / (γβ·mc²)
            factor_B = ds_m * _C_LIGHT / (gamma * beta * mass_MeV)

            xp_rad = beam.particles[alive, 1] * 1e-3  # mrad → rad
            yp_rad = beam.particles[alive, 3] * 1e-3

            dxp_rad = charge_num * (
                factor_E * Ex_tot
                + factor_B * (yp_rad * Bz_tot - By_tot)
            )
            dyp_rad = charge_num * (
                factor_E * Ey_tot
                + factor_B * (Bx_tot - xp_rad * Bz_tot)
            )
            beam.particles[alive, 1] += dxp_rad * 1e3  # rad → mrad
            beam.particles[alive, 3] += dyp_rad * 1e3

        # ---- Drift -----------------------------------------------------
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_m

    # ------------------------------------------------------------------ #
    #  Exact backward step (algebraic inverse of track_rk4)
    # ------------------------------------------------------------------ #

    def _onaxis_dW_ref(self, z_pos: float, phi_sync_rad: float,
                       ds_m: float, charge_num: float) -> float:
        """Reference-particle energy gain of one slice.

        BIT-MIRROR of the ``track_rk4`` reference-advance block (the
        ``dW_ref`` loop) — same channel iteration order, same operation
        order.  Keep in lockstep with any change there.
        """
        dW_ref = 0.0
        for ch_enum, ch in self.field_data.channels.items():
            if not ch_enum.is_electric:
                continue
            scale = self._scale(ch_enum, ch)
            if ch.geometry == 9:
                continue   # quad gradient has no axial E
            Fz_ax = self._sample_Fz_on_axis(ch, z_pos)
            if ch_enum.is_static:
                phasor = 1.0
            else:
                phasor = float(np.cos(phi_sync_rad))
            dW_ref += charge_num * Fz_ax * scale * phasor * ds_m
        return dW_ref

    def untrack_rk4(self, beam, ds: float, ref_entry, step_idx: int,
                    *, freq_ratio: float | None = None) -> None:
        """Exact algebraic inverse of one ``track_rk4`` slice.

        ``track_rk4`` is a first-order kick–drift step whose every
        sub-operation is closed-form invertible: positions recover by
        un-drifting (angles are unchanged after the kick), the phase
        slip used the already-updated ΔW (so it un-applies with the
        current column), and the transverse kick is a 2×2 linear system
        in (x′, y′) with det = 1 + a² ≥ 1.  Residual is float add/sub
        round-off (~1 ulp/step) — no iteration needed.

        Parameters mirror the forward call except:

        * ``ref_entry`` — the reference state at THIS slice's entrance
          (from the backtracker's zero-particle forward replay).  Never
          mutated; ``beam.ref`` is not touched either — the caller owns
          reference bookkeeping.
        * ``step_idx`` — the forward ``_step_idx`` of this slice
          (``self._step_idx`` is not consulted or advanced).
        * ``freq_ratio`` — the FREQ-jump rescale applied at slice 0 of
          the forward pass; the inverse divides it back out.  Pass only
          for step 0 (the backward walker undoes step 0 LAST, mirroring
          the forward "first op" placement).

        Element state contract: ``_sync_offset_deg`` / ``_phi_s_at_entrance``
        must already hold the forward-run calibration (the replay
        pre-pass leaves them in place), so ``_phi_sync_rad`` reproduces
        the forward synchronous phase exactly.
        """
        ref0 = ref_entry.copy()
        if step_idx == 0:
            # Mirrors the forward step-0 order: frequency propagates to
            # the (local) ref BEFORE any phase arithmetic.
            self._propagate_frequency_to_ref(ref0)

        z_pos = self._z_map_start + step_idx * ds + ds / 2.0
        ds_m = ds * 1e-3
        mass_MeV = ref0.species.mass
        charge_num = ref0.species.charge
        wavelength = ref0.wavelength

        if wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref0.beta * wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref0.phi_s + dphi_to_mid
        phi_sync_rad = self._phi_sync_rad(ref0, phi_s_mid)

        dW_ref = self._onaxis_dW_ref(z_pos, phi_sync_rad, ds_m, charge_num)
        # Post-advance reference (β₁γ₁) — the forward damp/slip/Lorentz
        # factors all read ref AFTER the energy update.
        ref1 = ref0.copy()
        ref1.w_kin += dW_ref

        alive = beam.alive_mask
        if not np.any(alive):
            return

        # ---- a. un-drift (angles unchanged since the forward drift) ----
        beam.particles[alive, 0] -= beam.particles[alive, 1] * ds_m
        beam.particles[alive, 2] -= beam.particles[alive, 3] * ds_m

        # ---- b. un-slip (forward slip used the post-kick ΔW, which is
        #        the current column — exact) ----------------------------
        if wavelength > 0 and ref1.beta > 0 and ref1.gamma > 0:
            slip = -360.0 * ds / (ref1.beta ** 3 * ref1.gamma ** 3
                                  * mass_MeV * wavelength)
            beam.particles[alive, 4] -= slip * beam.particles[alive, 5]

        # ---- c. re-sample fields at the recovered positions/phases -----
        # BIT-MIRROR of the forward accumulation loop.
        xs = beam.particles[alive, 0]
        ys = beam.particles[alive, 2]
        dphi_deg = beam.particles[alive, 4]
        n = int(np.sum(alive))

        Ex_tot = np.zeros(n)
        Ey_tot = np.zeros(n)
        Ez_tot = np.zeros(n)
        Bx_tot = np.zeros(n)
        By_tot = np.zeros(n)
        Bz_tot = np.zeros(n)

        phi_i = phi_sync_rad + dphi_deg * (np.pi / 180.0)

        for ch_enum, ch in self.field_data.channels.items():
            scale = self._scale(ch_enum, ch)
            if ch_enum.is_static:
                phasor = np.ones(n)
            elif ch_enum.is_electric:
                phasor = np.cos(phi_i)
            else:
                phasor = np.sin(phi_i)
            Fx_p, Fy_p, Fz_p = self._sample_cart(ch, xs, ys, z_pos)
            if ch_enum.is_electric:
                Ex_tot += scale * phasor * Fx_p
                Ey_tot += scale * phasor * Fy_p
                Ez_tot += scale * phasor * Fz_p
            else:
                Bx_tot += scale * phasor * Fx_p
                By_tot += scale * phasor * Fy_p
                Bz_tot += scale * phasor * Fz_p
            cross = self._sample_cart_cross(ch_enum, ch, xs, ys, z_pos)
            if cross is not None:
                Fx_c, Fy_c, Fz_c = cross
                if ch_enum.is_electric:
                    phasor_c = np.sin(phi_i)
                    Bx_tot += scale * phasor_c * Fx_c
                    By_tot += scale * phasor_c * Fy_c
                    Bz_tot += scale * phasor_c * Fz_c
                else:
                    phasor_c = np.cos(phi_i)
                    Ex_tot += scale * phasor_c * Fx_c
                    Ey_tot += scale * phasor_c * Fy_c
                    Ez_tot += scale * phasor_c * Fz_c

        dW_i = charge_num * Ez_tot * ds_m

        # ---- d/e. un-Lorentz ⊕ un-damp -----------------------------------
        # Forward: damp (post-advance β₁γ₁), then kick reading POST-damp
        # angles:  x'₊ = x'_d + Cx + a·y'_d ;  y'₊ = y'_d + Cy − a·x'_d.
        # Solve the 2×2, then divide out damp.
        beta1, gamma1 = ref1.beta, ref1.gamma
        if beta1 > 0 and gamma1 > 0:
            factor_E = ds_m / (gamma1 * beta1 * beta1 * mass_MeV)
            factor_B = ds_m * _C_LIGHT / (gamma1 * beta1 * mass_MeV)
            a_coef = charge_num * factor_B * Bz_tot            # per particle
            Cx = 1e3 * charge_num * (factor_E * Ex_tot - factor_B * By_tot)
            Cy = 1e3 * charge_num * (factor_E * Ey_tot + factor_B * Bx_tot)
            u = beam.particles[alive, 1] - Cx
            v = beam.particles[alive, 3] - Cy
            det = 1.0 + a_coef * a_coef
            xp_d = (u - a_coef * v) / det
            yp_d = (v + a_coef * u) / det
        else:
            xp_d = beam.particles[alive, 1]
            yp_d = beam.particles[alive, 3]
        if beta1 > 0 and gamma1 > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_i / (beta1 ** 2 * gamma1 * mass_MeV))
            xp_d = xp_d / damp
            yp_d = yp_d / damp
        beam.particles[alive, 1] = xp_d
        beam.particles[alive, 3] = yp_d

        # ---- f. un-energy ----------------------------------------------
        beam.particles[alive, 5] -= dW_i - dW_ref

        # ---- g. (step 0) un-rescale the FREQ-jump dphi ------------------
        if step_idx == 0 and freq_ratio is not None and freq_ratio != 1.0:
            beam.particles[alive, 4] /= freq_ratio

    # ------------------------------------------------------------------ #
    #  advance_ref
    # ------------------------------------------------------------------ #

    def advance_ref(self, ref) -> None:
        """Step the reference particle through the full field map.

        Only electric channels contribute to the energy integral;
        magnetic fields do no work on the reference.
        """
        n = self.n_steps
        ds = self.length / n
        ds_m = ds * 1e-3

        # Propagate cavity frequency to ref BEFORE calibration so the loop
        # advances phi_s at this cavity's wavelength.  Only RF cavities
        # propagate (`_propagate_frequency_to_ref` is a no-op for static-
        # only maps, so solenoids don't clobber ref.frequency).
        self._propagate_frequency_to_ref(ref)
        # Snapshot entrance phase for p_flag=1 / SET_SYNC_PHASE mode.
        self._phi_s_at_entrance = ref.phi_s
        if self.p_flag == 1 and self._sync_offset_deg is None:
            self._calibrate_sync_phase(ref)

        for i in range(n):
            z_pos = self._z_map_start + (i + 0.5) * ds
            if ref.wavelength > 0:
                dphi_to_mid = 180.0 * ds / (ref.beta * ref.wavelength)
            else:
                dphi_to_mid = 0.0
            phi_s_mid = ref.phi_s + dphi_to_mid
            phi_rad = self._phi_sync_rad(ref, phi_s_mid)
            dW = 0.0
            for ch_enum, ch in self.field_data.channels.items():
                if not ch_enum.is_electric:
                    continue
                if ch.geometry == 9:
                    continue
                Fz_ax = self._sample_Fz_on_axis(ch, z_pos)
                if ch_enum.is_static:
                    phasor = 1.0
                else:
                    phasor = float(np.cos(phi_rad))
                scale = self._scale(ch_enum, ch)
                dW += ref.species.charge * Fz_ax * scale * phasor * ds_m
            ref.w_kin += dW
            ref.s += ds
            if ref.wavelength > 0:
                ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

    def _propagate_frequency_to_ref(self, ref) -> None:
        eff_freq = self.effective_frequency
        if (eff_freq > 0
                and eff_freq != ref.frequency
                and self._has_electric_channel()):
            ref.frequency = eff_freq

    def reset_run_state(self) -> None:
        super().reset_run_state()
        # SET_SYNC_PHASE calibration is tuned to the reference β at cavity
        # entry of *this* run; clearing forces re-fit on the next pass.
        self._sync_offset_deg = None
        self._phi_s_at_entrance = 0.0

    def _has_electric_channel(self) -> bool:
        fd = getattr(self, "field_data", None)
        if fd is None:
            return False
        return any(ch_enum.is_electric for ch_enum in fd.channels)

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        """Advance *ref* by integrating on-axis E_z over a sub-range
        [z_from_mm, z_to_mm] of the element (element-local coordinates,
        0 = entrance, self.length = exit).

        Used by the envelope solver so that sigma sub-steps inside an
        accelerating element see the locally-correct β, γ when building
        the space-charge kick matrix.  Without this, ``self._ref`` stays
        frozen at the cavity entrance across a 250 mm HWR cavity where
        β actually rises ~30 %, giving a ~40 % over-strong SC kick at
        mid-cavity and a systematic envelope-vs-TW bias.

        Uses the native grid granularity (length / n_steps).  Does NOT
        touch self._step_idx — the tracker's integration cursor is
        independent from the reference's s position.
        """
        length_slice = z_to_mm - z_from_mm
        if length_slice <= 0.0:
            return
        native_ds = self.length / max(self.n_steps, 1)
        n_sub = max(1, int(round(length_slice / native_ds)))
        ds = length_slice / n_sub
        ds_m = ds * 1e-3

        # p_flag=1 lazy-calibration guard: if we're advancing without a
        # prior advance_ref/track_rk4 having triggered it, do so now.
        if self.p_flag == 1 and self._sync_offset_deg is None:
            self._calibrate_sync_phase(ref)

        for i in range(n_sub):
            z_pos = self._z_map_start + z_from_mm + (i + 0.5) * ds
            if ref.wavelength > 0:
                dphi_to_mid = 180.0 * ds / (ref.beta * ref.wavelength)
            else:
                dphi_to_mid = 0.0
            phi_s_mid = ref.phi_s + dphi_to_mid
            phi_rad = self._phi_sync_rad(ref, phi_s_mid)
            dW = 0.0
            for ch_enum, ch in self.field_data.channels.items():
                if not ch_enum.is_electric:
                    continue
                if ch.geometry == 9:
                    continue
                Fz_ax = self._sample_Fz_on_axis(ch, z_pos)
                if ch_enum.is_static:
                    phasor = 1.0
                else:
                    phasor = float(np.cos(phi_rad))
                scale = self._scale(ch_enum, ch)
                dW += ref.species.charge * Fz_ax * scale * phasor * ds_m
            ref.w_kin += dW
            ref.s += ds
            if ref.wavelength > 0:
                ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

    # ------------------------------------------------------------------ #
    #  fitted_matrix
    # ------------------------------------------------------------------ #

    def fitted_matrix(self, ref) -> np.ndarray:
        """Compute the linearised 6×6 transfer matrix by numerical Jacobian.

        Tracks 12 test particles (+/- epsilon in each of 6 coords) through
        the full element and builds the Jacobian from central differences.
        The reference particle is *not* modified.

        For magnetic-only 1-D field maps (solenoids) the per-sub-step
        thin-slice error is O((K·ds)²) and the default ``n_steps`` (set
        from the map's z-grid, e.g. 60 for a 300 mm / 5 mm-grid solenoid)
        leaves ~1-5 % error in the matrix at low β / strong B.  We refine
        the sub-step count locally here (ds ≲ 0.2 mm) so the matrix
        matches TraceWin.  RF cavities converge at the default rate.
        """
        from linac_gen.core.beam import Beam

        n = max(self.n_steps, 1)
        fd = getattr(self, "field_data", None)
        if fd is not None and fd.channels:
            has_E = any(ch_enum.is_electric for ch_enum in fd.channels)
            all_1d = all(getattr(ch, "geometry", None) == 1
                         for ch in fd.channels.values())
            if (not has_E) and all_1d:
                min_n = max(1, int(round(self.length / 0.2)))
                if min_n > n:
                    n = min_n
        ds = self.length / n

        def _track_single(state):
            ref_copy = ref.copy()
            b = Beam(ref=ref_copy, n_particles=1, current=0.0)
            b.particles[0, :] = state
            saved_idx = self._step_idx
            self._step_idx = 0
            for _ in range(n):
                self.track_rk4(b, ds)
            self._step_idx = saved_idx
            return b.particles[0, :].copy()

        ref_state = np.zeros(6)
        M = numerical_jacobian(_track_single, ref_state)
        return M

    def fitted_matrix_slice(self, ref, ds_mm: float) -> np.ndarray:
        """Linearised 6×6 map for a *ds_mm* slice starting at the current
        ``_step_idx``.

        Each probe uses ``ref.copy()`` pre-advanced to the slice's
        cumulative RF phase so that the transit-time integration is
        preserved across chained slices.

        For magnetic-only 1-D field maps (solenoids) the per-sub-step
        thin-slice error is O((K·ds)²) and becomes significant at strong
        B and low β.  We enforce a minimum of ~5000 sub-steps/metre here
        so that the envelope solver's linearised map matches TraceWin.
        RF cavities are not refined (they converge at the default rate).
        """
        from linac_gen.core.beam import Beam

        native_ds = self.length / self.n_steps if self.n_steps > 0 else ds_mm
        n_sub = max(1, int(round(ds_mm / native_ds)))

        # Solenoid auto-refinement: bump n_sub until ds < 0.2 mm.
        fd = getattr(self, "field_data", None)
        if fd is not None and fd.channels:
            has_E = any(ch_enum.is_electric for ch_enum in fd.channels)
            all_1d = all(getattr(ch, "geometry", None) == 1
                         for ch in fd.channels.values())
            if (not has_E) and all_1d:
                min_sub = max(1, int(round(ds_mm / 0.2)))
                if min_sub > n_sub:
                    n_sub = min_sub

        sub_ds = ds_mm / n_sub

        saved_idx = self._step_idx
        # NOTE: the envelope solver now advances ``ref.phi_s`` in step with
        # the bundle loop (via ``advance_ref_over``), so the ref passed in
        # is already at the current slice entrance — we trust its phi_s
        # directly rather than re-deriving an offset from saved_idx, which
        # would otherwise double-count the phase advance and blow up σ_φ.

        def _track_single(state):
            ref_copy = ref.copy()
            b = Beam(ref=ref_copy, n_particles=1, current=0.0)
            b.particles[0, :] = state
            self._step_idx = saved_idx
            for _ in range(n_sub):
                self.track_rk4(b, sub_ds)
            return b.particles[0, :].copy()

        M = numerical_jacobian(_track_single, np.zeros(6))
        self._step_idx = saved_idx + n_sub
        return M
