"""3-D Cartesian field-map tracking element.

Interpolates E and B components on the particle's (x, y, z) via
scipy.interpolate.RegularGridInterpolator (linear).  Each step applies
the Lorentz-force kick q(E + v × B) using the engineering-unit formula
derived from the TraceWin manual (§18122 / §18066 / §18180) and then drifts.

Channel phasor rules (manual §18122)
--------------------------------------
* Static E / Static B : phasor = 1  (no time dependence)
* RF electric         : phasor = cos(ωt + φ)
* RF magnetic         : phasor = sin(ωt + φ)

Lorentz kick (engineering units)
----------------------------------
Field units: E in MV/m, B in T, positions in mm, angles in mrad, Δs in m.
The speed-of-light constant c converts B to the same scale as E:

    C_LIGHT_CONV = 299.792458   (= c [m/s] × 10⁻⁶ [MeV/V conversion])

    Δx' [rad] = q · Δs · E_x / (γ β² · mc²)
              + q · Δs · (y'·Bz − By) · C_LIGHT_CONV / (γ β · mc²)

where q is in units of e, mc² in MeV.

The element only handles geometry=7 (3-D Cartesian) channels.
1-D / 2-D cylindrical channels belong to FieldMap (Task 7).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI

from linac_gen.elements.base import FieldMapElement as FieldMapBase
from linac_gen.io.field_map_data import FieldMapData
from linac_gen.io.tracewin_geom import Channel

# Speed-of-light conversion: c [m/s] × 1e-6 [MeV/V].
# Used so that B[T] × C_LIGHT_CONV has the same numerical scale as E[MV/m].
_C_LIGHT_CONV = 299.792458

# Fused C++ trilinear sampler: one call computes cell indices/weights per
# particle and applies them to ALL components of a channel (scipy RGI
# recomputes them per component — measured ~11x slower on real maps).
# BITWISE identical to the RGI linear path (pinned by
# tests/elements/test_fieldmap_kernels.py).  Falls back to per-component
# RGI when the compiled module is absent.
#
# Default ON when available; switchable at runtime (GUI/CLI/API expose
# "legacy scipy sampling") via use_fused_kernel(), and at process start
# via LINAC_GEN_FIELDMAP_KERNEL=0.  Because both paths are bitwise
# identical, the switch exists for verification/debugging, not physics.
import os as _os
try:
    from linac_gen._fieldmap_kernels import interp3_multi as _interp3_multi
    _KERNEL_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    _interp3_multi = None
    _KERNEL_AVAILABLE = False
_USE_FUSED = (_KERNEL_AVAILABLE
              and _os.environ.get("LINAC_GEN_FIELDMAP_KERNEL", "1") != "0")


def kernel_available() -> bool:
    """True when the compiled _fieldmap_kernels module is importable."""
    return _KERNEL_AVAILABLE


def fused_kernel_enabled() -> bool:
    """Current state of the fused-sampler switch."""
    return _USE_FUSED


def use_fused_kernel(enabled: bool) -> None:
    """Enable/disable the fused C++ sampler at runtime (global).

    Disabling routes every FieldMap3D sample through the original
    per-component scipy RegularGridInterpolator path.  Results are
    bitwise identical either way; only speed differs.
    """
    global _USE_FUSED
    _USE_FUSED = bool(enabled) and _KERNEL_AVAILABLE


def set_fieldmap_numerics(integrator: str | None = None,
                          interp: str | None = None) -> None:
    """Set FieldMap3D integrator / interpolation kind globally AND mirror
    the choice into env vars so SPAWNED workers (error studies, CLI
    scans re-import in fresh processes) inherit it.  Bare class-attribute
    assignment does not cross the spawn boundary — measured: parent sets
    dkd/cubic, spawned worker sees kd/linear."""
    if integrator is not None:
        FieldMap3D.integrator_kind = str(integrator)
        _os.environ["LINAC_GEN_FIELDMAP_INTEGRATOR"] = str(integrator)
    if interp is not None:
        FieldMap3D.interp_kind = str(interp)
        _os.environ["LINAC_GEN_FIELDMAP_INTERP"] = str(interp)


# Content-keyed cache of stacked component arrays for the fused sampler.
# TraceWin lattices re-use the same field-map FILES across many elements,
# but the parser builds a private FieldMapData per element — without this
# cache every element stacked its own copy (measured 1.28 GB on the full
# PIP-II linac, 299 packs / 164 elements).  Keying by content (axes +
# component bytes) collapses that to one copy per unique map.
_PACK_CACHE: dict = {}


def _shared_pack(ax, ch, names):
    """Return the cached (gx, gy, gz, stacked, names) pack for this map
    content, building it on first sight."""
    import hashlib
    h = hashlib.sha1()
    for a in ax:
        h.update(a.tobytes())
    for c in names:
        h.update(np.ascontiguousarray(
            np.asarray(getattr(ch, c), dtype=float)).tobytes())
    h.update(",".join(names).encode())
    key = h.digest()
    pack = _PACK_CACHE.get(key)
    if pack is None:
        stacked = np.ascontiguousarray(np.stack(
            [np.asarray(getattr(ch, c), dtype=float) for c in names]))
        pack = (ax[0], ax[1], ax[2], stacked, list(names))
        _PACK_CACHE[key] = pack
    return pack


from linac_gen.elements.mixins import Misalignment, FieldError


class FieldMap3D(FieldMapBase, Misalignment, FieldError):
    """3-D Cartesian field-map element.

    Parameters
    ----------
    name : str
        Element name.
    length : float
        Physical length in mm.
    field_data : FieldMapData
        All channels must have ``geometry == 7`` (3-D Cartesian).
    scale : float
        Global amplitude multiplier (applied to all channels).
    phase : float
        RF synchronous phase offset in degrees.
    frequency : float
        RF frequency in MHz (0 = static, no phasor).
    aperture : float
        Aperture radius in mm; 0 = no aperture check.
    n_steps : int
        Number of integration sub-steps.
    ke : float
        Additional electric-field amplitude scale (default 1.0).
    kb : float
        Additional magnetic-field amplitude scale (default 1.0).
    p_flag : int
        Phase reference flag (0 = relative to ref.phi_s, 1 = absolute).
    """

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  fitted_matrix runs an RK4 jacobian through the field map —
    # it reads length, scale, ke, kb, phase, frequency, voltage_rel,
    # phase_offset, frequency_offset, ki, n_steps, p_flag.  field_data
    # is immutable per-instance (not in the fingerprint; the id(element)
    # part of the cache key disambiguates instances that share params
    # but were loaded from different files).
    _cache_keys: tuple[str, ...] = (
        "length", "scale", "phase", "phase_offset",
        "frequency", "frequency_offset",
        "ke", "kb", "voltage_rel", "ki", "n_steps", "p_flag",
    )

    def __init__(self, name: str, length: float, field_data: FieldMapData,
                 scale: float = 1.0, phase: float = 0.0,
                 frequency: float = 0.0, aperture: float = 0.0,
                 n_steps: int = 100,
                 ke: float = 1.0, kb: float = 1.0,
                 ki: float = 0.0,
                 p_flag: int = 0,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 voltage_rel: float = 0.0,
                 phase_offset: float = 0.0,
                 frequency_offset: float = 0.0,
                 ka: int = 1,
                 field_file: str | None = None, geom: int | None = None):
        super().__init__(name=name, length=length,
                         aperture=aperture, n_steps=n_steps)
        self.field_data = field_data
        self.scale = scale
        self.phase = phase
        self.frequency = frequency
        self.ke = ke
        self.kb = kb
        self._init_misalignment(dx=dx, dy=dy, dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        self._init_field_error(voltage_rel=voltage_rel,
                               phase_offset=phase_offset,
                               frequency_offset=frequency_offset)
        # ki: TraceWin's per-element SC-compensation flag.  Per the manual
        # (§FIELD_MAP): when Ki>0, the reader loads ``<filename>.scc`` and
        # stores it on ``field_data.scc_profile`` with ``scc_scale = Ki``;
        # the tracker then uses ``Ki·Scc(z_local)`` to override
        # ``_sc_factor`` at each SC kick inside this element.  Ki=0
        # (default) is a no-op — the global ``SPACE_CHARGE_COMP`` marker
        # controls compensation.
        self.ki = float(ki)
        self.p_flag = p_flag
        # ka: TraceWin aperture-shape flag — ignored by the 3-D tracker but
        # retained so the FIELD_MAP card round-trips faithfully.
        self.ka = int(ka)
        # Provenance for round-tripping back to a FIELD_MAP .dat card (the
        # resolved field-file prefix and raw geom code, set by the parser via
        # field_map_factory).  None ⇒ writer emits a comment, not a card.
        self.field_file = field_file
        self.geom = geom
        self._z_map_start = float(field_data.z[0])
        self._z_map_end   = float(field_data.z[-1])
        self._step_idx = 0
        # Cavity-intrinsic phase offset ψ used to realise TraceWin's
        # SET_SYNC_PHASE convention (p_flag == 1).  Populated lazily
        # on the first track_rk4/advance_ref call from a scan-fit at the
        # reference particle's β at cavity entrance.  None until then.
        self._sync_offset_deg: float | None = None
        self._interpolators = self._build_channel_interpolators(field_data)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phi_sync_rad(self, ref, phi_s_midpoint_deg: float) -> float:
        """Synchronous phase in radians at the slice midpoint.

        * p_flag == 0 (relative): φ_sync = element.phase + phi_s_midpoint.
        * p_flag == 1 (absolute / SET_SYNC_PHASE): TraceWin's θ_s is
          NOT a geometric anchor point — it's the **inverse** of a
          scan-fit over input phases that yields ``dW = R·cos(φ + ψ) + a₀``
          for the reference particle.  ψ is an intrinsic property of
          the cavity + entrance β (it captures the cavity's transit-time
          factor).  At tracking time we just use the same formula as
          p_flag=0 but with the effective phase shifted by ψ:

              phi_sync_rad = (self.phase − ψ) + phi_s_midpoint.

          ψ is computed once per cavity by :meth:`_calibrate_sync_phase`
          and stored in ``self._sync_offset_deg``.
        """
        if self.p_flag == 1:
            offset = self._sync_offset_deg or 0.0
            # Internal phase advance from cavity entrance — independent of
            # the lattice-global ref.phi_s, so sweeps of upstream elements
            # can't shift this cavity's operating point.
            internal = phi_s_midpoint_deg - self._phi_s_at_entrance
            return (self.effective_phase - offset + internal) * np.pi / 180.0
        return (self.effective_phase + phi_s_midpoint_deg) * np.pi / 180.0

    # -- Iterative self-consistent calibration for SET_SYNC_PHASE --------
    def _probe_voltage_integral(self, ref_entry,
                                 phi_input_deg: float) -> tuple[float, float]:
        """Integrate the complex voltage V_c = Σ q·E_z(z)·e^{i·φ_gen(z)}·ds
        while self-consistently advancing the reference with ``phi_input_deg``
        as the cavity-entrance RF phase.

        Returns ``(Re, Im)`` in MeV.  dW can be recovered as
        ``Re·cos(φ_in) − Im·sin(φ_in) = |V_c|·cos(φ_in + atan2(Im, Re))``.
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
            for ch_enum, comp_interps in self._interpolators.items():
                if not ch_enum.is_electric:
                    continue
                ch_data = self.field_data.channels[ch_enum]
                Ez_ax = self._sample_onaxis(ch_enum, comp_interps, z_pos)
                amp = self._scale_factor(ch_enum, ch_data)
                Ez_sum += Ez_ax * amp
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
        """Iterative self-consistent SET_SYNC_PHASE calibration, evaluated
        at the **actual operating point** ``φ_in = θ_s − ψ``.

        The cavity's dW response is

            dW(φ_in) = Re_V(φ_in)·cos(φ_in) − Im_V(φ_in)·sin(φ_in)
                     = |V_c(φ_in)|·cos(φ_in + Ψ(φ_in))

        where ``Ψ(φ_in) = atan2(Im_V, Re_V)``.  Because β evolves during
        transit, V_c and hence Ψ depend on the probe's input phase.  The
        cavity is tracked with ``φ_in = θ_s − ψ``, so we solve the fixed
        point

            ψ = Ψ(θ_s − ψ)

        so that the β-trajectory the calibration sees is exactly the
        β-trajectory the actual tracking will produce.  This matches
        TraceWin's "synchronous phase of the generatrix" convention
        evaluated at operating conditions.
        """
        if self.p_flag != 1 or self._sync_offset_deg is not None:
            return
        theta_s = float(self.effective_phase)
        psi = 0.0
        relax = 0.7   # under-relaxation stabilises the stiff β-feedback loop
        for _ in range(max_iter):
            probe_input = theta_s - psi
            re_v, im_v = self._probe_voltage_integral(ref_entry, probe_input)
            psi_new = float(np.rad2deg(np.arctan2(im_v, re_v)))
            if abs(psi_new - psi) < tol_deg:
                psi = psi_new
                break
            psi = (1.0 - relax) * psi + relax * psi_new
        self._sync_offset_deg = float(psi)

    # Interpolation order for the field map.  ``"linear"`` is trilinear
    # (8-point), the legacy default; ``"cubic"`` uses SciPy's tricubic
    # which gives smoother off-grid evaluation at the cost of building
    # the spline coefficient table once per element instantiation.  Used
    # for sharp-gradient solenoid maps where trilinear introduces visible
    # focusing kinks.
    # Initialized from the env so SPAWNED workers (error studies / CLI
    # scans re-import in fresh processes) inherit the parent's choice —
    # a bare class attribute set in the parent does not cross the spawn
    # boundary.  Use set_fieldmap_numerics() to change it at runtime.
    interp_kind: str = _os.environ.get("LINAC_GEN_FIELDMAP_INTERP",
                                       "linear")

    def _build_channel_interpolators(self, fd: FieldMapData) -> dict:
        """Build RGI interpolators per channel, component.

        Uses :attr:`interp_kind` (class- or instance-level) to choose
        between ``"linear"`` (trilinear) and ``"cubic"`` (tricubic).

        Returns
        -------
        dict[Channel, dict[str, RGI]]
            Outer key is Channel enum; inner keys are 'Fx', 'Fy', 'Fz'.
            Missing or None components are absent from the inner dict.

        Raises
        ------
        ValueError
            If any channel has geometry != 7.
        """
        result = {}
        method = self.interp_kind if self.interp_kind in ("linear", "cubic") else "linear"
        common = dict(method=method, bounds_error=False, fill_value=0.0)
        self._fused_packs = {}
        for ch_enum, ch in fd.channels.items():
            if ch.geometry != 7:
                raise ValueError(
                    f"FieldMap3D only handles geometry=7 (3-D Cartesian) channels; "
                    f"got geometry={ch.geometry} on {ch_enum.name}. "
                    f"Use FieldMap for 1-D / 2-D cylindrical channels."
                )
            axes = (ch.x, ch.y, ch.z)
            comp_interps = {}
            for comp_name in ("Fx", "Fy", "Fz"):
                arr = getattr(ch, comp_name)
                if arr is not None:
                    comp_interps[comp_name] = RGI(axes, arr, **common)
            result[ch_enum] = comp_interps
            # Fused C++ path (linear only): stack the present components
            # so one kernel call samples them all.  Packs are built
            # whenever the kernel exists (not gated on the runtime switch,
            # so the switch can be flipped mid-session both ways); usage
            # is gated in _sample_channel_at.  Requires strictly ascending
            # axes (always true for TraceWin exports); anything else
            # silently keeps the scipy path.
            if _KERNEL_AVAILABLE and method == "linear" and comp_interps:
                ax = [np.ascontiguousarray(a, dtype=float) for a in axes]
                if all(a.ndim == 1 and a.size >= 2 and np.all(np.diff(a) > 0)
                       for a in ax):
                    names = list(comp_interps.keys())
                    # shared across elements loading identical map content
                    self._fused_packs[ch_enum] = _shared_pack(ax, ch, names)
        return result

    def _phasor(self, channel: Channel, phi_rad: np.ndarray) -> np.ndarray:
        """Per-channel time phasor evaluated at total phase phi_rad.

        Static channels: phasor = 1 (no time dependence, phase ignored).
        RF electric    : phasor = cos(phi_rad)
        RF magnetic    : phasor = sin(phi_rad)
        """
        if channel.is_static:
            return np.ones_like(phi_rad)
        if channel.is_electric:
            return np.cos(phi_rad)
        return np.sin(phi_rad)   # RF magnetic

    def _scale_factor(self, channel: Channel, ch_data) -> float:
        """Combined scale factor: global × k_e/k_b × 1/norm_factor.

        Uses ``effective_ke`` / ``effective_kb`` to honour per-seed
        ``voltage_rel`` cavity-amplitude errors.
        """
        k = self.effective_ke if channel.is_electric else self.effective_kb
        return self.scale * k / ch_data.norm_factor

    def _sample_channel_at(self, ch_enum: Channel, comp_interps: dict,
                            xs: np.ndarray, ys: np.ndarray,
                            zs: np.ndarray) -> tuple:
        """Sample Fx, Fy, Fz for one channel at N particle positions.

        Uses the full 3-D map values for all components — both RF_E and
        RF_B — faithfully tracking whatever the field solver exported.

        Returns (Fx, Fy, Fz) each shape (N,); zeros for absent components.
        """
        n = len(xs)
        pack = (getattr(self, "_fused_packs", {}).get(ch_enum)
                if _USE_FUSED else None)
        if pack is not None:
            gx, gy, gz, stacked, names = pack
            out = _interp3_multi(
                np.ascontiguousarray(xs, dtype=float),
                np.ascontiguousarray(ys, dtype=float),
                np.ascontiguousarray(zs, dtype=float),
                gx, gy, gz, stacked)
            by_name = dict(zip(names, out))
            return (by_name.get("Fx", np.zeros(n)),
                    by_name.get("Fy", np.zeros(n)),
                    by_name.get("Fz", np.zeros(n)))
        pts = np.column_stack([xs, ys, zs])
        Fx = comp_interps["Fx"](pts) if "Fx" in comp_interps else np.zeros(n)
        Fy = comp_interps["Fy"](pts) if "Fy" in comp_interps else np.zeros(n)
        Fz = comp_interps["Fz"](pts) if "Fz" in comp_interps else np.zeros(n)
        return Fx, Fy, Fz

    def _sample_onaxis(self, ch_enum: Channel, comp_interps: dict,
                       z_pos: float) -> float:
        """Sample Fz component on-axis (0, 0, z_pos) for one channel."""
        if "Fz" not in comp_interps:
            return 0.0
        pt = np.array([[0.0, 0.0, z_pos]])
        return float(comp_interps["Fz"](pt)[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Integrator selection (class-level so callers can flip globally).
    # ``kd`` is the legacy first-order kick-then-drift integrator.
    # ``dkd`` is the second-order symplectic Drift–Kick–Drift integrator
    # (a.k.a. "velocity Verlet").  Position-dependent fields are sampled
    # at the half-drifted location, which removes the O(ds²) phase-advance
    # asymmetry that drives long-trajectory emittance growth in strongly
    # focusing solenoid maps.  Both schemes evaluate the field once per
    # call so the cost is essentially the same.
    # env-initialized for spawn inheritance — see interp_kind note.
    integrator_kind: str = _os.environ.get("LINAC_GEN_FIELDMAP_INTEGRATOR",
                                           "kd")

    def track_rk4(self, beam, ds: float) -> None:
        """Advance the beam by one slice of length *ds* (mm).

        Dispatches to the integrator selected by :attr:`integrator_kind`:
        ``"kd"`` (default, legacy first-order) or ``"dkd"`` (second-order
        symplectic).  See class docstring for trade-offs.
        """
        if self.integrator_kind == "dkd":
            self._track_dkd(beam, ds)
        else:
            self._track_kd(beam, ds)

    # ------------------------------------------------------------------
    def _track_kd(self, beam, ds: float) -> None:
        """Legacy first-order kick-then-drift integrator.

        Thin-slice split:
          1. Advance reference particle using on-axis Ez (midpoint sample).
          2. For each alive particle, accumulate fields from all channels.
          3. Longitudinal energy kick   ΔW  = q · Ez_tot · Δs  (MeV).
          4. Transverse angular kick    using full Lorentz formula.
          5. Drift.
        """
        ref = beam.ref
        ds_m = ds * 1e-3

        # Snapshot the reference phase at the very entrance of the cavity.
        # Used by p_flag=1 (SET_SYNC_PHASE / absolute) to peg the starting
        # phase to self.phase while still advancing through the cavity.
        if self._step_idx == 0:
            # FREQ-jump rescale of per-particle dphi.  ``beam.particles[:,4]``
            # is in deg-at-ref.frequency.  When this cavity's frequency
            # differs from ref.frequency, the same physical phase deviation
            # corresponds to a different number of degrees at the new
            # frequency — multiply by ratio = f_new/f_old BEFORE updating
            # ref.frequency.  Mirrors envelope.py:528-537 σ rescale; without
            # this MP εnz collapses by f_old/f_new at SSR1 entry (162.5 →
            # 325 MHz Liouville violation).
            eff_freq = self.effective_frequency
            if (eff_freq > 0
                    and eff_freq != ref.frequency
                    and self._has_electric_channel()):
                ratio = eff_freq / ref.frequency
                beam.particles[beam.alive_mask, 4] *= ratio
            # Propagate the cavity's RF frequency to ref BEFORE calibration
            # and BEFORE the first slice's φ_s advance, so phi_s tracks at
            # this cavity's wavelength.  Critical at FREQ jumps (e.g. the
            # first 325 MHz SSR1 cavity entered with ref still at 162.5
            # MHz — calibration would otherwise absorb the wrong λ and
            # bias the cavity's transverse defocus by ~6 %).
            self._propagate_frequency_to_ref(ref)
            self._phi_s_at_entrance = ref.phi_s
            # Lazy one-shot SET_SYNC_PHASE calibration using ref at entry.
            if self.p_flag == 1 and self._sync_offset_deg is None:
                self._calibrate_sync_phase(ref)

        z_pos = self._z_map_start + self._step_idx * ds + ds / 2.0
        self._step_idx += 1

        charge = ref.species.charge
        beta   = ref.beta
        gamma  = ref.gamma
        mass_MeV = ref.species.mass

        # --- Reference advance (on-axis, electric channels only) -------
        # Evaluate the phasor at the slice midpoint in phi-s terms, not
        # at the slice entrance.  The extra half-slice of phi_s advance
        # matters for long thick cavities (240 mm QWR at 162.5 MHz sweeps
        # ~700° through its full length).
        if ref.wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref.beta * ref.wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref.phi_s + dphi_to_mid
        phi_sync_rad = self._phi_sync_rad(ref, phi_s_mid)
        dW_ref = 0.0
        for ch_enum, comp_interps in self._interpolators.items():
            if not ch_enum.is_electric:
                continue
            ch_data = self.field_data.channels[ch_enum]
            Ez_ax = self._sample_onaxis(ch_enum, comp_interps, z_pos)
            phasor_val = float(self._phasor(ch_enum, np.array([phi_sync_rad]))[0])
            amp = self._scale_factor(ch_enum, ch_data)
            dW_ref += charge * Ez_ax * amp * phasor_val * ds_m

        ref.w_kin += dW_ref
        ref.s += ds
        if ref.wavelength > 0:
            ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

        alive = beam.alive_mask
        if not np.any(alive):
            return

        xs = beam.particles[alive, 0]
        ys = beam.particles[alive, 2]
        zs = np.full_like(xs, z_pos)
        dphi_deg = beam.particles[alive, 4]

        # Angles in rad for B-cross-v coupling
        xp_rad = beam.particles[alive, 1] * 1e-3
        yp_rad = beam.particles[alive, 3] * 1e-3

        # Total phase per particle (sync + deviation)
        phi_total_rad = phi_sync_rad + dphi_deg * (np.pi / 180.0)

        # Accumulate total E and B fields across all channels
        n = int(np.sum(alive))
        Ex_tot = np.zeros(n)
        Ey_tot = np.zeros(n)
        Ez_tot = np.zeros(n)
        Bx_tot = np.zeros(n)
        By_tot = np.zeros(n)
        Bz_tot = np.zeros(n)

        for ch_enum, comp_interps in self._interpolators.items():
            ch_data = self.field_data.channels[ch_enum]
            amp = self._scale_factor(ch_enum, ch_data)
            phasor = self._phasor(ch_enum, phi_total_rad)  # shape (n,)
            Fx, Fy, Fz = self._sample_channel_at(ch_enum, comp_interps, xs, ys, zs)
            scaled_x = Fx * amp * phasor
            scaled_y = Fy * amp * phasor
            scaled_z = Fz * amp * phasor
            if ch_enum.is_electric:
                Ex_tot += scaled_x
                Ey_tot += scaled_y
                Ez_tot += scaled_z
            else:
                Bx_tot += scaled_x
                By_tot += scaled_y
                Bz_tot += scaled_z

        # --- Longitudinal energy kick (relative to reference) ----------
        dW_i = charge * Ez_tot * ds_m          # MeV, shape (n,)
        beam.particles[alive, 5] += dW_i - dW_ref

        # --- Adiabatic transverse damping ------------------------------
        # A ΔW kick grows p_z (the particle's longitudinal momentum), so
        # x' = p_x / p_z must shrink proportionally even though p_x is
        # unchanged.  Using the small-signal expansion  Δp_z / p_z ≈
        # ΔW / (β²γ·mc²)  per particle.  Without this factor the
        # "geometric" emittance ε_x = √⟨x²⟩⟨x'²⟩ − ⟨xx'⟩² fails to
        # shrink during acceleration (a classic missing-damping bug —
        # a beam accelerated 2 → 10 MeV would look as if its geometric
        # emittance doubled instead of halving as βγ grows).
        if beta > 0 and gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_i / (beta * beta * gamma * mass_MeV))
            beam.particles[alive, 1] *= damp
            beam.particles[alive, 3] *= damp

        # --- Longitudinal drift slip (per-particle phase vs ref) --------
        # A particle with ΔW ≠ 0 moves at a different velocity than the
        # reference, so its RF phase slips through each slice.  The same
        # first-order relation used by Drift.transfer_matrix:
        #     Δφ += -360·ds/(β³γ³·m·λ) · ΔW
        # with ds in mm, λ in mm, m in MeV, ΔW in MeV, Δφ in deg.
        if ref.wavelength > 0 and beta > 0 and gamma > 0:
            slip = -360.0 * ds / (beta ** 3 * gamma ** 3 * mass_MeV
                                  * ref.wavelength)
            beam.particles[alive, 4] += slip * beam.particles[alive, 5]

        # --- Transverse angular kick (Lorentz) -------------------------
        if beta > 0 and gamma > 0:
            # E contribution: Δx' = q·Δs·Ex / (γ β² mc²)
            factor_E = ds_m / (gamma * beta * beta * mass_MeV)
            # B contribution: Δx' = q·Δs·(y'·Bz−By)·c / (γ β mc²)
            # the c factor is _C_LIGHT_CONV = 299.792458 for B[T],E[MV/m]
            factor_B = ds_m * _C_LIGHT_CONV / (gamma * beta * mass_MeV)

            dxp_rad = charge * (factor_E * Ex_tot
                                + factor_B * (yp_rad * Bz_tot - By_tot))
            dyp_rad = charge * (factor_E * Ey_tot
                                + factor_B * (Bx_tot - xp_rad * Bz_tot))
            beam.particles[alive, 1] += dxp_rad * 1e3   # back to mrad
            beam.particles[alive, 3] += dyp_rad * 1e3

        # --- Drift -------------------------------------------------------
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_m

    # ------------------------------------------------------------------
    def _track_dkd(self, beam, ds: float) -> None:
        """Second-order symplectic Drift–Kick–Drift integrator.

        Half-drift positions, sample fields at the new (x, y, z_mid),
        apply the full kick (energy + adiabatic damping + transverse
        Lorentz), then half-drift again.  Reference advance and z-bookkeeping
        are identical to the legacy KD scheme so the per-element call
        contract (one slice → one call) is preserved.
        """
        ref = beam.ref
        ds_m = ds * 1e-3
        ds_half_m = 0.5 * ds_m

        if self._step_idx == 0:
            # FREQ-jump rescale of per-particle dphi (see _track_kd for the
            # full rationale; without this εnz collapses at SSR1 entry).
            eff_freq = self.effective_frequency
            if (eff_freq > 0
                    and eff_freq != ref.frequency
                    and self._has_electric_channel()):
                ratio = eff_freq / ref.frequency
                beam.particles[beam.alive_mask, 4] *= ratio
            self._propagate_frequency_to_ref(ref)
            self._phi_s_at_entrance = ref.phi_s
            if self.p_flag == 1 and self._sync_offset_deg is None:
                self._calibrate_sync_phase(ref)

        z_pos = self._z_map_start + self._step_idx * ds + ds / 2.0
        self._step_idx += 1

        charge = ref.species.charge
        beta = ref.beta
        gamma = ref.gamma
        mass_MeV = ref.species.mass

        # ---- Reference advance (identical to legacy: midpoint Ez) -----
        if ref.wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref.beta * ref.wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref.phi_s + dphi_to_mid
        phi_sync_rad = self._phi_sync_rad(ref, phi_s_mid)
        dW_ref = 0.0
        for ch_enum, comp_interps in self._interpolators.items():
            if not ch_enum.is_electric:
                continue
            ch_data = self.field_data.channels[ch_enum]
            Ez_ax = self._sample_onaxis(ch_enum, comp_interps, z_pos)
            phasor_val = float(self._phasor(ch_enum, np.array([phi_sync_rad]))[0])
            amp = self._scale_factor(ch_enum, ch_data)
            dW_ref += charge * Ez_ax * amp * phasor_val * ds_m

        ref.w_kin += dW_ref
        ref.s += ds
        if ref.wavelength > 0:
            ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

        alive = beam.alive_mask
        if not np.any(alive):
            return

        # ---- First half-drift: x, y, dphi ------------------------------
        # Phase slip for half a slice (uses pre-kick dW per particle).
        if ref.wavelength > 0 and beta > 0 and gamma > 0:
            slip_half = -360.0 * (ds * 0.5) / (
                beta ** 3 * gamma ** 3 * mass_MeV * ref.wavelength
            )
            beam.particles[alive, 4] += slip_half * beam.particles[alive, 5]
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_half_m

        # ---- Sample fields at half-drifted positions -------------------
        xs = beam.particles[alive, 0]
        ys = beam.particles[alive, 2]
        zs = np.full_like(xs, z_pos)
        dphi_deg = beam.particles[alive, 4]
        xp_rad = beam.particles[alive, 1] * 1e-3
        yp_rad = beam.particles[alive, 3] * 1e-3
        phi_total_rad = phi_sync_rad + dphi_deg * (np.pi / 180.0)

        n = int(np.sum(alive))
        Ex_tot = np.zeros(n)
        Ey_tot = np.zeros(n)
        Ez_tot = np.zeros(n)
        Bx_tot = np.zeros(n)
        By_tot = np.zeros(n)
        Bz_tot = np.zeros(n)

        for ch_enum, comp_interps in self._interpolators.items():
            ch_data = self.field_data.channels[ch_enum]
            amp = self._scale_factor(ch_enum, ch_data)
            phasor = self._phasor(ch_enum, phi_total_rad)
            Fx, Fy, Fz = self._sample_channel_at(ch_enum, comp_interps, xs, ys, zs)
            sx = Fx * amp * phasor
            sy = Fy * amp * phasor
            sz = Fz * amp * phasor
            if ch_enum.is_electric:
                Ex_tot += sx
                Ey_tot += sy
                Ez_tot += sz
            else:
                Bx_tot += sx
                By_tot += sy
                Bz_tot += sz

        # ---- Full kick (energy + damping + transverse Lorentz) ---------
        dW_i = charge * Ez_tot * ds_m
        beam.particles[alive, 5] += dW_i - dW_ref

        if beta > 0 and gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_i / (beta * beta * gamma * mass_MeV))
            beam.particles[alive, 1] *= damp
            beam.particles[alive, 3] *= damp

        if beta > 0 and gamma > 0:
            factor_E = ds_m / (gamma * beta * beta * mass_MeV)
            factor_B = ds_m * _C_LIGHT_CONV / (gamma * beta * mass_MeV)
            dxp_rad = charge * (factor_E * Ex_tot
                                + factor_B * (yp_rad * Bz_tot - By_tot))
            dyp_rad = charge * (factor_E * Ey_tot
                                + factor_B * (Bx_tot - xp_rad * Bz_tot))
            beam.particles[alive, 1] += dxp_rad * 1e3
            beam.particles[alive, 3] += dyp_rad * 1e3

        # ---- Second half-drift: x, y, dphi (using post-kick momenta) ---
        if ref.wavelength > 0 and beta > 0 and gamma > 0:
            beam.particles[alive, 4] += slip_half * beam.particles[alive, 5]
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_half_m

    # ------------------------------------------------------------------
    #  Exact backward step (algebraic inverse of track_rk4)
    # ------------------------------------------------------------------
    def _onaxis_dW_ref(self, z_pos: float, phi_sync_rad: float,
                       ds_m: float, charge: float) -> float:
        """Reference energy gain of one slice.  BIT-MIRROR of the
        ``_track_kd`` / ``_track_dkd`` reference-advance loop — keep in
        lockstep with any change there."""
        dW_ref = 0.0
        for ch_enum, comp_interps in self._interpolators.items():
            if not ch_enum.is_electric:
                continue
            ch_data = self.field_data.channels[ch_enum]
            Ez_ax = self._sample_onaxis(ch_enum, comp_interps, z_pos)
            phasor_val = float(
                self._phasor(ch_enum, np.array([phi_sync_rad]))[0])
            amp = self._scale_factor(ch_enum, ch_data)
            dW_ref += charge * Ez_ax * amp * phasor_val * ds_m
        return dW_ref

    def _accumulate_fields(self, xs, ys, z_pos, phi_total_rad):
        """Total (E, B) at the given positions/phases.  BIT-MIRROR of the
        per-particle accumulation loop shared by ``_track_kd`` and
        ``_track_dkd``."""
        n = xs.shape[0]
        zs = np.full_like(xs, z_pos)
        Ex_tot = np.zeros(n)
        Ey_tot = np.zeros(n)
        Ez_tot = np.zeros(n)
        Bx_tot = np.zeros(n)
        By_tot = np.zeros(n)
        Bz_tot = np.zeros(n)
        for ch_enum, comp_interps in self._interpolators.items():
            ch_data = self.field_data.channels[ch_enum]
            amp = self._scale_factor(ch_enum, ch_data)
            phasor = self._phasor(ch_enum, phi_total_rad)
            Fx, Fy, Fz = self._sample_channel_at(
                ch_enum, comp_interps, xs, ys, zs)
            sx = Fx * amp * phasor
            sy = Fy * amp * phasor
            sz = Fz * amp * phasor
            if ch_enum.is_electric:
                Ex_tot += sx
                Ey_tot += sy
                Ez_tot += sz
            else:
                Bx_tot += sx
                By_tot += sy
                Bz_tot += sz
        return Ex_tot, Ey_tot, Ez_tot, Bx_tot, By_tot, Bz_tot

    def untrack_rk4(self, beam, ds: float, ref_entry, step_idx: int,
                    *, freq_ratio: float | None = None) -> None:
        """Exact algebraic inverse of one forward slice.

        See :meth:`FieldMap.untrack_rk4` for the full contract (entry
        ref from the replay, ``self._step_idx`` untouched, freq_ratio
        only for step 0).  The 3-D specifics mirrored here: the forward
        integrators capture β, γ at slice ENTRY (before the reference
        advance) for damp/slip/Lorentz, and the Lorentz kick reads the
        PRE-damp angles — so the angle undo is one 2×2 solve
        ``[damp a; −a damp]`` with det = damp² + a².
        """
        if self.integrator_kind == "dkd":
            self._untrack_dkd(beam, ds, ref_entry, step_idx,
                              freq_ratio=freq_ratio)
        else:
            self._untrack_kd(beam, ds, ref_entry, step_idx,
                             freq_ratio=freq_ratio)

    def _scalars_for_untrack(self, ds: float, ref_entry, step_idx: int):
        """Common forward-scalar reconstruction for both inverses."""
        ref0 = ref_entry.copy()
        if step_idx == 0:
            self._propagate_frequency_to_ref(ref0)
        z_pos = self._z_map_start + step_idx * ds + ds / 2.0
        ds_m = ds * 1e-3
        charge = ref0.species.charge
        beta = ref0.beta            # ENTRY β₀γ₀ — the 3-D convention
        gamma = ref0.gamma
        mass_MeV = ref0.species.mass
        wavelength = ref0.wavelength
        if wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref0.beta * wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref0.phi_s + dphi_to_mid
        phi_sync_rad = self._phi_sync_rad(ref0, phi_s_mid)
        dW_ref = self._onaxis_dW_ref(z_pos, phi_sync_rad, ds_m, charge)
        return (z_pos, ds_m, charge, beta, gamma, mass_MeV, wavelength,
                phi_sync_rad, dW_ref)

    def _untrack_kd(self, beam, ds: float, ref_entry, step_idx: int,
                    *, freq_ratio: float | None = None) -> None:
        (z_pos, ds_m, charge, beta, gamma, mass_MeV, wavelength,
         phi_sync_rad, dW_ref) = self._scalars_for_untrack(
            ds, ref_entry, step_idx)

        alive = beam.alive_mask
        if not np.any(alive):
            return

        # a. un-drift (post-kick angles unchanged since the drift)
        beam.particles[alive, 0] -= beam.particles[alive, 1] * ds_m
        beam.particles[alive, 2] -= beam.particles[alive, 3] * ds_m
        # b. un-slip (entry β₀γ₀; forward slip used the post-kick ΔW)
        if wavelength > 0 and beta > 0 and gamma > 0:
            slip = -360.0 * ds / (beta ** 3 * gamma ** 3 * mass_MeV
                                  * wavelength)
            beam.particles[alive, 4] -= slip * beam.particles[alive, 5]
        # c. re-sample fields at the recovered state
        xs = beam.particles[alive, 0]
        ys = beam.particles[alive, 2]
        dphi_deg = beam.particles[alive, 4]
        phi_total_rad = phi_sync_rad + dphi_deg * (np.pi / 180.0)
        Ex_tot, Ey_tot, Ez_tot, Bx_tot, By_tot, Bz_tot = \
            self._accumulate_fields(xs, ys, z_pos, phi_total_rad)
        dW_i = charge * Ez_tot * ds_m
        # d/e. un-kick: forward was damp·(pre) then kick with PRE-damp
        # angles:  x'₊ = damp·x' + Cx + a·y' ;  y'₊ = damp·y' + Cy − a·x'.
        if beta > 0 and gamma > 0:
            factor_E = ds_m / (gamma * beta * beta * mass_MeV)
            factor_B = ds_m * _C_LIGHT_CONV / (gamma * beta * mass_MeV)
            a_coef = charge * factor_B * Bz_tot
            Cx = 1e3 * charge * (factor_E * Ex_tot - factor_B * By_tot)
            Cy = 1e3 * charge * (factor_E * Ey_tot + factor_B * Bx_tot)
            u = beam.particles[alive, 1] - Cx
            v = beam.particles[alive, 3] - Cy
        else:
            a_coef = np.zeros_like(dW_i)
            u = beam.particles[alive, 1]
            v = beam.particles[alive, 3]
        if beta > 0 and gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_i / (beta * beta * gamma * mass_MeV))
        else:
            damp = np.ones_like(dW_i)
        det = damp * damp + a_coef * a_coef
        beam.particles[alive, 1] = (damp * u - a_coef * v) / det
        beam.particles[alive, 3] = (damp * v + a_coef * u) / det
        # f. un-energy
        beam.particles[alive, 5] -= dW_i - dW_ref
        # g. (step 0) un-rescale
        if step_idx == 0 and freq_ratio is not None and freq_ratio != 1.0:
            beam.particles[alive, 4] /= freq_ratio

    def _untrack_dkd(self, beam, ds: float, ref_entry, step_idx: int,
                     *, freq_ratio: float | None = None) -> None:
        (z_pos, ds_m, charge, beta, gamma, mass_MeV, wavelength,
         phi_sync_rad, dW_ref) = self._scalars_for_untrack(
            ds, ref_entry, step_idx)
        ds_half_m = 0.5 * ds_m

        alive = beam.alive_mask
        if not np.any(alive):
            return

        if wavelength > 0 and beta > 0 and gamma > 0:
            slip_half = -360.0 * (ds * 0.5) / (
                beta ** 3 * gamma ** 3 * mass_MeV * wavelength)
        else:
            slip_half = 0.0

        # a. un-second-half (post-kick momenta; slip used post-kick ΔW)
        beam.particles[alive, 0] -= beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] -= beam.particles[alive, 3] * ds_half_m
        if slip_half != 0.0:
            beam.particles[alive, 4] -= slip_half * beam.particles[alive, 5]
        # b. re-sample at the half-drifted state (= the forward sample point)
        xs = beam.particles[alive, 0]
        ys = beam.particles[alive, 2]
        dphi_deg = beam.particles[alive, 4]
        phi_total_rad = phi_sync_rad + dphi_deg * (np.pi / 180.0)
        Ex_tot, Ey_tot, Ez_tot, Bx_tot, By_tot, Bz_tot = \
            self._accumulate_fields(xs, ys, z_pos, phi_total_rad)
        dW_i = charge * Ez_tot * ds_m
        # c. un-kick (same PRE-damp-angle structure as KD)
        if beta > 0 and gamma > 0:
            factor_E = ds_m / (gamma * beta * beta * mass_MeV)
            factor_B = ds_m * _C_LIGHT_CONV / (gamma * beta * mass_MeV)
            a_coef = charge * factor_B * Bz_tot
            Cx = 1e3 * charge * (factor_E * Ex_tot - factor_B * By_tot)
            Cy = 1e3 * charge * (factor_E * Ey_tot + factor_B * Bx_tot)
            u = beam.particles[alive, 1] - Cx
            v = beam.particles[alive, 3] - Cy
        else:
            a_coef = np.zeros_like(dW_i)
            u = beam.particles[alive, 1]
            v = beam.particles[alive, 3]
        if beta > 0 and gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_i / (beta * beta * gamma * mass_MeV))
        else:
            damp = np.ones_like(dW_i)
        det = damp * damp + a_coef * a_coef
        beam.particles[alive, 1] = (damp * u - a_coef * v) / det
        beam.particles[alive, 3] = (damp * v + a_coef * u) / det
        # d. un-energy → ΔW is now the pre-kick value
        beam.particles[alive, 5] -= dW_i - dW_ref
        # e. un-first-half (pre-kick momenta; slip used pre-kick ΔW)
        beam.particles[alive, 0] -= beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] -= beam.particles[alive, 3] * ds_half_m
        if slip_half != 0.0:
            beam.particles[alive, 4] -= slip_half * beam.particles[alive, 5]
        # f. (step 0) un-rescale
        if step_idx == 0 and freq_ratio is not None and freq_ratio != 1.0:
            beam.particles[alive, 4] /= freq_ratio

    # ------------------------------------------------------------------
    def advance_ref(self, ref) -> None:
        """Step the reference particle through the full 3-D map on-axis.

        Only electric channels contribute to energy gain; magnetic fields
        do no work on the reference.
        """
        n = self.n_steps
        ds = self.length / n
        ds_m = ds * 1e-3

        # Propagate cavity frequency to ref BEFORE calibration so phi_s
        # tracks at this cavity's wavelength inside the loop (matters at
        # FREQ jumps; see _track_kd for the rationale).
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
            for ch_enum, comp_interps in self._interpolators.items():
                if not ch_enum.is_electric:
                    continue
                ch_data = self.field_data.channels[ch_enum]
                Ez_ax = self._sample_onaxis(ch_enum, comp_interps, z_pos)
                phasor_val = float(self._phasor(ch_enum, np.array([phi_rad]))[0])
                amp = self._scale_factor(ch_enum, ch_data)
                dW += ref.species.charge * Ez_ax * amp * phasor_val * ds_m
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
        [z_from_mm, z_to_mm] of the element (element-local coordinates).

        See the matching docstring on ``FieldMap.advance_ref_over`` — this
        is the 3-D-map analogue used by the envelope solver to keep
        ``self._ref`` in sync with sigma sub-steps so the SC kick uses
        the local β, γ inside an accelerating cavity.
        """
        length_slice = z_to_mm - z_from_mm
        if length_slice <= 0.0:
            return
        native_ds = self.length / max(self.n_steps, 1)
        n_sub = max(1, int(round(length_slice / native_ds)))
        ds = length_slice / n_sub
        ds_m = ds * 1e-3

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
            for ch_enum, comp_interps in self._interpolators.items():
                if not ch_enum.is_electric:
                    continue
                ch_data = self.field_data.channels[ch_enum]
                Ez_ax = self._sample_onaxis(ch_enum, comp_interps, z_pos)
                phasor_val = float(self._phasor(ch_enum, np.array([phi_rad]))[0])
                amp = self._scale_factor(ch_enum, ch_data)
                dW += ref.species.charge * Ez_ax * amp * phasor_val * ds_m
            ref.w_kin += dW
            ref.s += ds
            if ref.wavelength > 0:
                ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

    # ------------------------------------------------------------------
    def fitted_matrix(self, ref) -> np.ndarray:
        """Linearised 6×6 transfer matrix (numerical Jacobian via track_rk4).

        For static-B-only 3-D maps (solenoids) the first-order thin-slice
        integrator has O((K·ds)²) error per step — same situation as the
        1-D solenoid fix.  Auto-refine to ds < 0.2 mm (~5000 steps/m).
        """
        import copy
        eps_x    = 1e-3   # mm
        eps_xp   = 1e-3   # mrad
        eps_dphi = 1e-2   # deg  (Δφ → Δφ, ΔW coupling)
        eps_dW   = 1e-4   # MeV  (ΔW → ΔW, Δφ coupling)

        deltas = [eps_x, eps_xp, eps_x, eps_xp, eps_dphi, eps_dW]
        M = np.eye(6)

        from linac_gen.core.beam import Beam
        from linac_gen.core.reference import ReferenceParticle

        # Refine n_steps locally for static-B-only maps so the solenoid's
        # linear matrix matches TraceWin.  RF cavities (electric channel
        # present) already converge at the default rate.
        n = max(self.n_steps, 1)
        fd = getattr(self, "field_data", None)
        if fd is not None and fd.channels:
            has_E = any(ch_enum.is_electric for ch_enum in fd.channels)
            if not has_E:
                min_n = max(1, int(round(self.length / 0.2)))
                if min_n > n:
                    n = min_n
        ds = self.length / n

        def _track_single(coord_vec):
            ref_c = ref.copy()
            beam  = Beam(ref=ref_c, n_particles=1, current=0.0)
            beam.particles[0] = coord_vec
            self._step_idx = 0
            for _ in range(n):
                self.track_rk4(beam, ds)
            return beam.particles[0].copy()

        origin = _track_single(np.zeros(6))
        for j, dj in enumerate(deltas):
            if dj == 0:
                continue
            v_plus  = np.zeros(6); v_plus[j]  = +dj
            v_minus = np.zeros(6); v_minus[j] = -dj
            y_plus  = _track_single(v_plus)
            y_minus = _track_single(v_minus)
            M[:, j] = (y_plus - y_minus) / (2.0 * dj)

        self._step_idx = 0   # reset after Jacobian evaluation
        return M

    # ------------------------------------------------------------------
    def fitted_matrix_slice(self, ref, ds_mm: float) -> np.ndarray:
        """Linearised 6×6 map for a *ds_mm* slice starting at the current
        ``_step_idx``.

        Uses a numerical Jacobian over ``n_sub`` calls to ``track_rk4``,
        each of length ``sub_ds = ds_mm / n_sub``.  After the Jacobian is
        built the counter is advanced by ``n_sub`` so successive calls in
        the envelope SC loop cover the full element in order.

        The caller's ``ref`` is NOT modified.  Each probe uses a
        ``ref.copy()`` that we pre-advance to the slice's cumulative RF
        phase — otherwise every slice would think it is at the cavity
        entrance, destroying the transit-time integration.
        """
        from linac_gen.core.beam import Beam
        from linac_gen.tracking.rk4 import numerical_jacobian

        native_ds = self.length / self.n_steps if self.n_steps > 0 else ds_mm
        n_sub = max(1, int(round(ds_mm / native_ds)))

        # Static-B 3-D maps (solenoids): refine so ds < 0.2 mm, same rule
        # the 1-D path uses.  Skip when an electric channel is present —
        # RF cavities converge at the default n_steps rate.
        fd = getattr(self, "field_data", None)
        if fd is not None and fd.channels:
            has_E = any(ch_enum.is_electric for ch_enum in fd.channels)
            if not has_E:
                min_sub = max(1, int(round(ds_mm / 0.2)))
                if min_sub > n_sub:
                    n_sub = min_sub

        sub_ds = ds_mm / n_sub

        saved_idx = self._step_idx
        # NOTE: the envelope solver now advances ``ref.phi_s`` inline via
        # ``advance_ref_over`` so the incoming ref is already at the slice
        # entrance.  Trust it directly; adding another offset here would
        # double-count the RF phase advance and blow up σ_φ.

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
