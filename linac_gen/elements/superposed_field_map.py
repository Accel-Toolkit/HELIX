"""Superposed field maps — TraceWin ``SUPERPOSE_MAP`` clusters.

A *cluster* is a run of ``SUPERPOSE_MAP Z0`` + ``FIELD_MAP`` card pairs
ended by the next ordinary element.  TraceWin places each map's origin
at ``Z0`` (mm) from the cluster entrance and integrates ONCE through
the vector-summed field of every map covering each z (manual,
"Superpose field maps" / "Field map with curved reference trajectory").

:class:`SuperposedFieldMap` is the container element that owns the
cluster's single disjoint s-span (``max_i(Z0_i + L_i)``) and sums its
children's fields at a shared container-local z.  Children — plain
:class:`~linac_gen.elements.field_map.FieldMap` /
:class:`~linac_gen.elements.field_map_3d.FieldMap3D` instances — become
PURE SAMPLERS: their own integrators, step cursors and reference
advances are never invoked; the container drives their low-level
samplers (``_sample_Fz_on_axis`` / ``_sample_cart`` /
``_sample_cart_cross`` / ``_sample_channel_at``) at
``z_file = child._z_map_start + (z_local − Z0_i)`` and applies ONE
energy kick, ONE adiabatic damping, ONE phase slip, ONE Lorentz kick
and ONE drift per slice — single application of the shared-slice
physics is what makes overlap correct.

Scope (v1, per the design round):
* Straight reference axis only — the transverse/rotation operands of
  ``SUPERPOSE_MAP`` are ignored by TraceWin itself without
  ``SUPERPOSE_MAP_OUT`` (curved clusters stay refused at parse).
* All RF children must share one frequency (mixed-clock clusters are
  refused at construction — a single ``ref.phi_s`` clock exists).
* Aperture / ``.ouv`` / ``.scc`` follow TraceWin's position-0
  convention: the z0 == 0 child carries them (``field_data`` property).
* Misalignment / ERROR_CAV apply to the cluster as a whole (TraceWin
  applies them per card — documented divergence).
* 3-D children are integrated with the container's kick-drift walk
  (the global DKD selection for standalone ``FieldMap3D`` does not
  apply inside a cluster).
"""
from __future__ import annotations

import math

import numpy as np

from linac_gen.elements.base import FieldMapElement
from linac_gen.elements.field_map import FieldMap, _C_LIGHT
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.mixins import FieldError, Misalignment
from linac_gen.tracking.rk4 import numerical_jacobian

__all__ = ["SuperposedFieldMap"]


class SuperposedFieldMap(FieldMapElement, Misalignment, FieldError):
    """Container for a TraceWin ``SUPERPOSE_MAP`` cluster.

    Parameters
    ----------
    name : str
        Element name.
    children : list[tuple[float, FieldMap | FieldMap3D]]
        ``(z0_mm, element)`` pairs in ORIGINAL CARD ORDER (preserved
        for writer round-trips).  ``z0`` may be negative ("start at +Z
        inside the map": TraceWin ``Z0 = −Z``); the pre-entrance part
        of such a map is simply outside the tracked span.
    aperture : float, optional
        Override; default = the z0 == 0 child's aperture (TraceWin's
        position-0 carrier convention), 0.0 if no such child.
    n_steps : int, optional
        Override; default resolves the FINEST child native grid over
        the whole span so no child's transit-time integral is
        undercut.
    interior_markers : list[tuple[float, object]], optional
        ``(dz_mm, Marker)`` diagnostics consumed from
        ``SHIFT_IN_FIELD_MAP`` (phase D) — recorded, not tracked.
    """

    # NO _cache_keys: get_element_matrix(cache=) fingerprinting
    # auto-disables for containers (matrix_tracking._element_fingerprint
    # returns None), which is the safe default.

    def __init__(self, name: str,
                 children: list,
                 *, aperture: float | None = None,
                 n_steps: int | None = None,
                 interior_markers: list | None = None,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 voltage_rel: float = 0.0,
                 phase_offset: float = 0.0,
                 frequency_offset: float = 0.0):
        if not children:
            raise ValueError("SuperposedFieldMap needs at least one child")
        for z0, child in children:
            if not isinstance(child, (FieldMap, FieldMap3D)):
                raise ValueError(
                    f"SUPERPOSE cluster child '{getattr(child, 'name', child)}'"
                    f" is not a FieldMap/FieldMap3D")
        span = max(float(z0) + float(child.length) for z0, child in children)
        if span <= 0.0:
            raise ValueError(
                f"SUPERPOSE cluster '{name}' has non-positive span "
                f"{span:.6g} mm (every map ends at or before the cluster "
                f"entrance)")

        # One ref.phi_s clock exists: all RF children must share one
        # frequency.  TraceWin's behavior for mixed-clock clusters is
        # unverifiable without a ground-truth run — refuse (the parser
        # degrades to the legacy end-to-end fallback in permissive mode).
        rf_freqs = sorted({
            float(child.effective_frequency)
            for _z0, child in children
            if any(not ch.is_static for ch in child.field_data.channels)
        })
        if len(rf_freqs) > 1:
            raise ValueError(
                f"SUPERPOSE cluster '{name}' mixes RF frequencies "
                f"{rf_freqs} MHz — a single reference clock cannot "
                f"represent this (unsupported in v1)")
        self._rf_frequency = rf_freqs[0] if rf_freqs else 0.0

        # TraceWin position-0 convention: the z0==0 child (first in card
        # order) carries aperture / .ouv / .scc / Ki for the cluster.
        pos0 = next((child for z0, child in children
                     if abs(float(z0)) < 1e-12), None)
        if aperture is None:
            aperture = float(pos0.aperture) if pos0 is not None else 0.0

        if n_steps is None:
            finest = min(child.length / max(child.n_steps, 1)
                         for _z0, child in children)
            n_steps = max(1, int(math.ceil(span / max(finest, 1e-9))))

        super().__init__(name=name, length=float(span),
                         aperture=float(aperture), n_steps=int(n_steps))
        self.children = [(float(z0), child) for z0, child in children]
        self._pos0_child = pos0
        self.ka = int(getattr(pos0, "ka", 1)) if pos0 is not None else 1
        self.ki = float(getattr(pos0, "ki", 0.0)) if pos0 is not None else 0.0
        self.p_flag = 0          # container-level; children keep theirs
        self.interior_markers = [(float(dz_mm), m)
                                 for dz_mm, m in (interior_markers or [])]

        # Per-run walk state (see reset_run_state).
        self._step_idx = 0
        self._z_cursor = 0.0
        self._z_history: list = []
        self._phi_s_at_entrance = 0.0
        self._ref_entry = None

        self._init_misalignment(dx=dx, dy=dy, dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        self._init_field_error(voltage_rel=voltage_rel,
                               phase_offset=phase_offset,
                               frequency_offset=frequency_offset)

    # ------------------------------------------------------------------ #
    #  Cluster-level properties (getattr seams read these)
    # ------------------------------------------------------------------ #
    @property
    def frequency(self) -> float:
        return self._rf_frequency

    @property
    def effective_frequency(self) -> float:
        return self._rf_frequency + self.frequency_offset

    @property
    def field_data(self):
        """The z0 == 0 child's data — TraceWin's position-0 carrier for
        the cluster's .ouv aperture profile and .scc compensation map
        (tracker/envelope read ``element.field_data`` for both)."""
        return (self._pos0_child.field_data
                if self._pos0_child is not None else None)

    def _has_electric_channel(self) -> bool:
        return any(ch.is_electric
                   for _z0, child in self.children
                   for ch in child.field_data.channels)

    # ------------------------------------------------------------------ #
    #  Shared-walk helpers
    # ------------------------------------------------------------------ #
    def _active(self, z_mid: float):
        """Children whose window [z0, z0+L) covers *z_mid* (container-
        local mm), yielding (z0, child, z_file)."""
        for z0, child in self.children:
            if z0 <= z_mid < z0 + child.length:
                yield z0, child, child._z_map_start + (z_mid - z0)

    def _child_phase_rad(self, child, ref, phi_s_mid: float) -> float:
        """Synchronous phase of *child* at the shared-slice midpoint,
        including the container-level ERROR_CAV phase offset."""
        return (child._phi_sync_rad(ref, phi_s_mid)
                + self.phase_offset * math.pi / 180.0)

    def _child_amp(self) -> float:
        """Container-level ERROR_CAV amplitude factor applied on top of
        each child's own scaling."""
        return 1.0 + self.voltage_rel

    def _propagate_frequency_to_ref(self, ref) -> None:
        eff = self.effective_frequency
        if eff > 0 and eff != ref.frequency and self._has_electric_channel():
            ref.frequency = eff

    def _entrance_ops(self, ref, beam=None) -> None:
        """Cluster-entrance bookkeeping (mirrors FieldMap.track_rk4
        step-0 block): FREQ-jump dphi rescale, frequency propagation,
        entrance-clock snapshots, SET_SYNC_PHASE calibrations."""
        eff = self.effective_frequency
        if (beam is not None and eff > 0 and eff != ref.frequency
                and self._has_electric_channel()):
            ratio = eff / ref.frequency
            beam.particles[beam.alive_mask, 4] *= ratio
        self._propagate_frequency_to_ref(ref)
        self._phi_s_at_entrance = ref.phi_s
        self._ref_entry = ref.copy()
        # Children whose window begins at (or before) the entrance see
        # the entrance clock; later windows are snapshotted at crossing.
        for z0, child in self.children:
            if z0 <= 0.0:
                child._phi_s_at_entrance = ref.phi_s
        # SET_SYNC_PHASE children: container-aware calibration in card
        # order (each sees earlier calibrations through the probe walk).
        for _z0, child in self.children:
            if (child.p_flag == 1
                    and getattr(child, "_sync_offset_deg", None) is None
                    and any(ch.is_electric
                            for ch in child.field_data.channels)):
                self._calibrate_child_sync_phase(child, self._ref_entry)

    def _snapshot_crossings(self, ref, z_prev: float, z_cur: float) -> None:
        """Child window entries crossed in (z_prev, z_cur]: snapshot the
        interpolated clock so p_flag=1 phasing is referenced to the
        child's own entry (first-order in the slice — exact at fixed β)."""
        for z0, child in self.children:
            entry = max(z0, 0.0)
            if z_prev < entry <= z_cur:
                if ref.wavelength > 0:
                    child._phi_s_at_entrance = (
                        ref.phi_s
                        + 360.0 * (entry - z_prev)
                        / (ref.beta * ref.wavelength))
                else:
                    child._phi_s_at_entrance = ref.phi_s

    def _onaxis_dW(self, ref, z_mid: float, phi_s_mid: float,
                   ds_m: float) -> float:
        """Summed on-axis electric energy gain for one shared slice."""
        charge = ref.species.charge
        amp = self._child_amp()
        dW = 0.0
        for _z0, child, z_file in self._active(z_mid):
            phi_rad = self._child_phase_rad(child, ref, phi_s_mid)
            if isinstance(child, FieldMap3D):
                for ch_enum, interps in child._interpolators.items():
                    if not ch_enum.is_electric:
                        continue
                    ch = child.field_data.channels[ch_enum]
                    if getattr(ch, "geometry", None) == 9:
                        continue
                    Fz = child._sample_onaxis(ch_enum, interps, z_file)
                    scale = child._scale_factor(ch_enum, ch) * amp
                    phasor = 1.0 if ch_enum.is_static else math.cos(phi_rad)
                    dW += charge * Fz * scale * phasor * ds_m
            else:
                for ch_enum, ch in child.field_data.channels.items():
                    if not ch_enum.is_electric:
                        continue
                    if ch.geometry == 9:
                        continue
                    Fz = child._sample_Fz_on_axis(ch, z_file)
                    scale = child._scale(ch_enum, ch) * amp
                    phasor = 1.0 if ch_enum.is_static else math.cos(phi_rad)
                    dW += charge * Fz * scale * phasor * ds_m
        return dW

    # ------------------------------------------------------------------ #
    #  track_rk4 — the shared-slice kick-drift kernel
    # ------------------------------------------------------------------ #
    def track_rk4(self, beam, ds: float) -> None:
        """One shared slice of length *ds* (mm): sum every covering
        child's E and B at the slice midpoint, then apply the
        single-slice physics ONCE (mirrors FieldMap.track_rk4)."""
        ref = beam.ref

        if self._step_idx == 0:
            self._entrance_ops(ref, beam=beam)

        z_prev = self._z_cursor
        z_mid = z_prev + ds / 2.0
        self._z_history.append(z_prev)
        self._z_cursor = z_prev + ds
        self._step_idx += 1
        self._snapshot_crossings(ref, z_prev, self._z_cursor)

        ds_m = ds * 1e-3
        mass_MeV = ref.species.mass
        charge_num = ref.species.charge
        wavelength = ref.wavelength

        if wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref.beta * wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref.phi_s + dphi_to_mid

        # ---- Reference advance (summed on-axis electric) ---------------
        dW_ref = self._onaxis_dW(ref, z_mid, phi_s_mid, ds_m)
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

        amp = self._child_amp()
        for _z0, child, z_file in self._active(z_mid):
            phi_sync_rad = self._child_phase_rad(child, ref, phi_s_mid)
            phi_i = phi_sync_rad + dphi_deg * (np.pi / 180.0)

            if isinstance(child, FieldMap3D):
                zs = np.full(n, z_file)
                for ch_enum, interps in child._interpolators.items():
                    ch = child.field_data.channels[ch_enum]
                    scale = child._scale_factor(ch_enum, ch) * amp
                    phasor = child._phasor(ch_enum, phi_i)
                    Fx, Fy, Fz = child._sample_channel_at(
                        ch_enum, interps, xs, ys, zs)
                    if ch_enum.is_electric:
                        Ex_tot += scale * phasor * Fx
                        Ey_tot += scale * phasor * Fy
                        Ez_tot += scale * phasor * Fz
                    else:
                        Bx_tot += scale * phasor * Fx
                        By_tot += scale * phasor * Fy
                        Bz_tot += scale * phasor * Fz
                continue

            # 1-D / 2-D-cyl / quad-grad child (FieldMap kernel,
            # field_map.py:610-651, generalized to a shared z).
            for ch_enum, ch in child.field_data.channels.items():
                scale = child._scale(ch_enum, ch) * amp

                if ch_enum.is_static:
                    phasor = np.ones(n)
                elif ch_enum.is_electric:
                    phasor = np.cos(phi_i)
                else:
                    phasor = np.sin(phi_i)

                Fx_p, Fy_p, Fz_p = child._sample_cart(ch, xs, ys, z_file)
                if ch_enum.is_electric:
                    Ex_tot += scale * phasor * Fx_p
                    Ey_tot += scale * phasor * Fy_p
                    Ez_tot += scale * phasor * Fz_p
                else:
                    Bx_tot += scale * phasor * Fx_p
                    By_tot += scale * phasor * Fy_p
                    Bz_tot += scale * phasor * Fz_p

                cross = child._sample_cart_cross(ch_enum, ch, xs, ys, z_file)
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

        # ---- Single-slice physics (byte-mirror of field_map.py:653-701)
        dW_i = charge_num * Ez_tot * ds_m
        beam.particles[alive, 5] += dW_i - dW_ref

        if ref.beta > 0 and ref.gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_i / (ref.beta ** 2 * ref.gamma * mass_MeV))
            beam.particles[alive, 1] *= damp
            beam.particles[alive, 3] *= damp

        if wavelength > 0 and ref.beta > 0 and ref.gamma > 0:
            slip = -360.0 * ds / (ref.beta ** 3 * ref.gamma ** 3
                                  * mass_MeV * wavelength)
            beam.particles[alive, 4] += slip * beam.particles[alive, 5]

        beta = ref.beta
        gamma = ref.gamma
        if beta > 0 and gamma > 0:
            factor_E = ds_m / (gamma * beta * beta * mass_MeV)
            factor_B = ds_m * _C_LIGHT / (gamma * beta * mass_MeV)

            xp_rad = beam.particles[alive, 1] * 1e-3
            yp_rad = beam.particles[alive, 3] * 1e-3

            dxp_rad = charge_num * (
                factor_E * Ex_tot
                + factor_B * (yp_rad * Bz_tot - By_tot)
            )
            dyp_rad = charge_num * (
                factor_E * Ey_tot
                + factor_B * (Bx_tot - xp_rad * Bz_tot)
            )
            beam.particles[alive, 1] += dxp_rad * 1e3
            beam.particles[alive, 3] += dyp_rad * 1e3

        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_m

    # ------------------------------------------------------------------ #
    #  Reference-particle walks
    # ------------------------------------------------------------------ #
    def advance_ref(self, ref) -> None:
        """Step the reference through the whole cluster (summed on-axis
        electric fields; native grid granularity)."""
        n = max(self.n_steps, 1)
        ds = self.length / n
        self._propagate_frequency_to_ref(ref)
        self._phi_s_at_entrance = ref.phi_s
        self._ref_entry = ref.copy()
        for z0, child in self.children:
            if z0 <= 0.0:
                child._phi_s_at_entrance = ref.phi_s
        for _z0, child in self.children:
            if (child.p_flag == 1
                    and getattr(child, "_sync_offset_deg", None) is None
                    and any(ch.is_electric
                            for ch in child.field_data.channels)):
                self._calibrate_child_sync_phase(child, self._ref_entry)
        self._ref_walk(ref, 0.0, self.length, ds)

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        """Sub-range reference advance (envelope SC bundles) — mirrors
        FieldMap.advance_ref_over; container-local z."""
        length_slice = z_to_mm - z_from_mm
        if length_slice <= 0.0:
            return
        if z_from_mm <= 0.0:
            # Bundle walk starting at the entrance: run the entrance
            # bookkeeping exactly once (mirrors NCells.advance_ref_over).
            self._propagate_frequency_to_ref(ref)
            self._phi_s_at_entrance = ref.phi_s
            self._ref_entry = ref.copy()
            for z0, child in self.children:
                if z0 <= 0.0:
                    child._phi_s_at_entrance = ref.phi_s
            for _z0, child in self.children:
                if (child.p_flag == 1
                        and getattr(child, "_sync_offset_deg", None) is None
                        and any(ch.is_electric
                                for ch in child.field_data.channels)):
                    self._calibrate_child_sync_phase(child, self._ref_entry)
        native_ds = self.length / max(self.n_steps, 1)
        n_sub = max(1, int(round(length_slice / native_ds)))
        ds = length_slice / n_sub
        self._ref_walk(ref, z_from_mm, z_to_mm, ds)

    def _ref_walk(self, ref, z_from: float, z_to: float, ds: float) -> None:
        ds_m = ds * 1e-3
        n = max(1, int(round((z_to - z_from) / ds)))
        z = z_from
        for _ in range(n):
            z_mid = z + ds / 2.0
            if ref.wavelength > 0:
                dphi_to_mid = 180.0 * ds / (ref.beta * ref.wavelength)
            else:
                dphi_to_mid = 0.0
            phi_s_mid = ref.phi_s + dphi_to_mid
            self._snapshot_crossings(ref, z, z + ds)
            ref.w_kin += self._onaxis_dW(ref, z_mid, phi_s_mid, ds_m)
            ref.s += ds
            if ref.wavelength > 0:
                ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)
            z += ds

    # ------------------------------------------------------------------ #
    #  SET_SYNC_PHASE — container-aware calibration
    # ------------------------------------------------------------------ #
    def _probe_cluster_voltage_integral(self, target, ref_entry,
                                        phi_input_deg: float):
        """(Re_V, Im_V) of the TARGET child's own E-channels while β
        evolves under the SUMMED fields of the whole cluster.

        The phase reference for the voltage integral is the target's
        window entry (mirrors the frame-matching rule documented on
        NCells._probe_voltage_integral); other children run at their
        operating phases (earlier calibrations already applied, card
        order)."""
        rf = ref_entry.copy()
        rf.phi_s = 0.0
        n = max(self.n_steps, 1)
        ds = self.length / n
        ds_m = ds * 1e-3
        charge = rf.species.charge
        phi_input_rad = phi_input_deg * math.pi / 180.0
        amp = self._child_amp()

        target_z0 = next(z0 for z0, child in self.children
                         if child is target)
        target_entry = max(target_z0, 0.0)
        phi_at_target_entry = None

        # Probe-local entrance snapshots (restored by the caller's
        # save/restore in _calibrate_child_sync_phase).
        for z0, child in self.children:
            if z0 <= 0.0:
                child._phi_s_at_entrance = 0.0

        Re_V = 0.0
        Im_V = 0.0
        z = 0.0
        for _ in range(n):
            z_mid = z + ds / 2.0
            if rf.wavelength > 0:
                dphi_to_mid = 180.0 * ds / (rf.beta * rf.wavelength)
            else:
                dphi_to_mid = 0.0
            phi_s_mid = rf.phi_s + dphi_to_mid
            # Window-entry crossings for the probe clock.
            for z0, child in self.children:
                entry = max(z0, 0.0)
                if z < entry <= z + ds:
                    snap = rf.phi_s
                    if rf.wavelength > 0:
                        snap += 360.0 * (entry - z) / (rf.beta * rf.wavelength)
                    child._phi_s_at_entrance = snap
                    if child is target:
                        phi_at_target_entry = snap
            if phi_at_target_entry is None and z >= target_entry:
                phi_at_target_entry = 0.0

            dW = 0.0
            for _z0, child, z_file in self._active(z_mid):
                is_target = child is target
                if isinstance(child, FieldMap3D):
                    items = [(ch_enum, child.field_data.channels[ch_enum],
                              interps)
                             for ch_enum, interps
                             in child._interpolators.items()]
                else:
                    items = [(ch_enum, ch, None)
                             for ch_enum, ch
                             in child.field_data.channels.items()]
                for ch_enum, ch, interps in items:
                    if not ch_enum.is_electric:
                        continue
                    if getattr(ch, "geometry", None) == 9:
                        continue
                    if isinstance(child, FieldMap3D):
                        Fz = child._sample_onaxis(ch_enum, interps, z_file)
                        scale = child._scale_factor(ch_enum, ch) * amp
                    else:
                        Fz = child._sample_Fz_on_axis(ch, z_file)
                        scale = child._scale(ch_enum, ch) * amp
                    if is_target:
                        # Target runs at the probe input phase referenced
                        # to ITS window entry.
                        entry_phi = (phi_at_target_entry or 0.0)
                        local = (phi_s_mid - entry_phi) * math.pi / 180.0
                        if ch_enum.is_static:
                            dW += charge * Fz * scale * ds_m
                        else:
                            dW += (charge * Fz * scale
                                   * math.cos(phi_input_rad + local) * ds_m)
                            Re_V += (charge * Fz * scale
                                     * math.cos(local) * ds_m)
                            Im_V += (charge * Fz * scale
                                     * math.sin(local) * ds_m)
                    else:
                        phi_rad = self._child_phase_rad(child, rf, phi_s_mid)
                        phasor = (1.0 if ch_enum.is_static
                                  else math.cos(phi_rad))
                        dW += charge * Fz * scale * phasor * ds_m
            rf.w_kin += dW
            rf.s += ds
            if rf.wavelength > 0:
                rf.phi_s += 360.0 * ds / (rf.beta * rf.wavelength)
            z += ds
        return Re_V, Im_V

    def _calibrate_child_sync_phase(self, child, ref_entry,
                                    tol_deg: float = 0.005,
                                    max_iter: int = 30) -> None:
        """Fixed-point ψ scan (mirrors FieldMap._calibrate_sync_phase)
        with the probe walking the WHOLE cluster."""
        if child.p_flag != 1 or getattr(child, "_sync_offset_deg",
                                        None) is not None:
            return
        saved = {id(c): c._phi_s_at_entrance for _z, c in self.children}
        try:
            theta_s = float(child.effective_phase)
            psi = 0.0
            relax = 0.7
            for _ in range(max_iter):
                re_v, im_v = self._probe_cluster_voltage_integral(
                    child, ref_entry, theta_s - psi)
                psi_new = float(np.rad2deg(np.arctan2(im_v, re_v)))
                if abs(psi_new - psi) < tol_deg:
                    psi = psi_new
                    break
                psi = (1.0 - relax) * psi + relax * psi_new
            child._sync_offset_deg = float(psi)
        finally:
            for _z, c in self.children:
                c._phi_s_at_entrance = saved[id(c)]

    # ------------------------------------------------------------------ #
    #  Fitted matrices (envelope / matrix modes)
    # ------------------------------------------------------------------ #
    def _all_children_magnetic_only_1d(self) -> bool:
        """True iff every child is a magnetic-only 1-D map — the class
        whose thin-slice error motivates the 0.2 mm fitted-matrix
        refinement (mirrors FieldMap.fitted_matrix).  3-D magnetic
        children are excluded on purpose: FieldMap3D applies no such
        refinement, and 0.2 mm sub-steps over 3-D interpolators would
        be pathologically slow."""
        for _z0, child in self.children:
            fd = child.field_data
            if any(ch_enum.is_electric for ch_enum in fd.channels):
                return False
            if not all(getattr(ch, "geometry", None) == 1
                       for ch in fd.channels.values()):
                return False
        return True

    def _save_walk_state(self):
        return (self._step_idx, self._z_cursor, len(self._z_history),
                self._phi_s_at_entrance,
                {id(c): c._phi_s_at_entrance for _z, c in self.children})

    def _restore_walk_state(self, state) -> None:
        (self._step_idx, self._z_cursor, hist_len,
         self._phi_s_at_entrance, snaps) = state
        del self._z_history[hist_len:]
        for _z, c in self.children:
            c._phi_s_at_entrance = snaps[id(c)]

    def fitted_matrix(self, ref) -> np.ndarray:
        """Central-difference 6×6 over the full cluster walk (mirrors
        FieldMap.fitted_matrix, incl. the magnetic-only refinement)."""
        from linac_gen.core.beam import Beam

        n = max(self.n_steps, 1)
        if self._all_children_magnetic_only_1d():
            min_n = max(1, int(round(self.length / 0.2)))
            if min_n > n:
                n = min_n
        ds = self.length / n

        saved = self._save_walk_state()

        def _track_single(state):
            ref_copy = ref.copy()
            b = Beam(ref=ref_copy, n_particles=1, current=0.0)
            b.particles[0, :] = state
            self._step_idx = 0
            self._z_cursor = 0.0
            for _ in range(n):
                self.track_rk4(b, ds)
            return b.particles[0, :].copy()

        try:
            M = numerical_jacobian(_track_single, np.zeros(6))
        finally:
            self._restore_walk_state(saved)
        return M

    def fitted_matrix_slice(self, ref, ds_mm: float,
                            _z_from_mm=None) -> np.ndarray:
        """6×6 for one slice; accepts the envelope's explicit z-cursor
        (multi-cell plumbing, see envelope._fitted_matrix_slice_at)."""
        from linac_gen.core.beam import Beam

        native_ds = self.length / self.n_steps if self.n_steps > 0 else ds_mm
        n_sub = max(1, int(round(ds_mm / native_ds)))
        if self._all_children_magnetic_only_1d():
            min_sub = max(1, int(round(ds_mm / 0.2)))
            if min_sub > n_sub:
                n_sub = min_sub
        sub_ds = ds_mm / n_sub

        z_from = self._z_cursor if _z_from_mm is None else float(_z_from_mm)
        saved = self._save_walk_state()

        def _track_single(state):
            ref_copy = ref.copy()
            b = Beam(ref=ref_copy, n_particles=1, current=0.0)
            b.particles[0, :] = state
            self._restore_walk_state(saved)
            self._z_cursor = z_from
            self._step_idx = saved[0]
            for _ in range(n_sub):
                self.track_rk4(b, sub_ds)
            return b.particles[0, :].copy()

        try:
            M = numerical_jacobian(_track_single, np.zeros(6))
        finally:
            self._restore_walk_state(saved)
        # Advance the walk cursor past this slice (envelope SC loop
        # contract, mirrors FieldMap.fitted_matrix_slice).
        self._step_idx = saved[0] + n_sub
        self._z_cursor = z_from + ds_mm
        return M

    # ------------------------------------------------------------------ #
    #  Exact backward step (backtracking)
    # ------------------------------------------------------------------ #
    def untrack_rk4(self, beam, ds: float, ref_entry, step_idx: int,
                    *, freq_ratio: float | None = None) -> None:
        """Exact algebraic inverse of one shared ``track_rk4`` slice —
        the structural mirror of :meth:`FieldMap.untrack_rk4` with the
        summed-children field accumulation.

        Entry z comes from ``self._z_history[step_idx]`` (recorded by
        the forward walk / the backtracker's zero-particle replay; the
        accumulated cursor cannot be reconstructed as ``step_idx·ds``
        when trailing sub-steps use a different ds).  Children's
        ``_phi_s_at_entrance`` / ``_sync_offset_deg`` must hold the
        forward-run state (the replay pre-pass leaves them in place)."""
        ref0 = ref_entry.copy()
        if step_idx == 0:
            self._propagate_frequency_to_ref(ref0)

        z_local = self._z_history[step_idx]
        z_mid = z_local + ds / 2.0
        ds_m = ds * 1e-3
        mass_MeV = ref0.species.mass
        charge_num = ref0.species.charge
        wavelength = ref0.wavelength

        if wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref0.beta * wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref0.phi_s + dphi_to_mid

        dW_ref = self._onaxis_dW(ref0, z_mid, phi_s_mid, ds_m)
        ref1 = ref0.copy()
        ref1.w_kin += dW_ref

        alive = beam.alive_mask
        if not np.any(alive):
            return

        # a. un-drift (angles unchanged since the forward drift)
        beam.particles[alive, 0] -= beam.particles[alive, 1] * ds_m
        beam.particles[alive, 2] -= beam.particles[alive, 3] * ds_m

        # b. un-slip (forward slip used the post-kick ΔW — current column)
        if wavelength > 0 and ref1.beta > 0 and ref1.gamma > 0:
            slip = -360.0 * ds / (ref1.beta ** 3 * ref1.gamma ** 3
                                  * mass_MeV * wavelength)
            beam.particles[alive, 4] -= slip * beam.particles[alive, 5]

        # c. re-sample summed fields at the recovered state (bit-mirror
        #    of the forward per-child accumulation).
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

        amp = self._child_amp()
        for _z0, child, z_file in self._active(z_mid):
            phi_sync_rad = self._child_phase_rad(child, ref0, phi_s_mid)
            phi_i = phi_sync_rad + dphi_deg * (np.pi / 180.0)

            if isinstance(child, FieldMap3D):
                zs = np.full(n, z_file)
                for ch_enum, interps in child._interpolators.items():
                    ch = child.field_data.channels[ch_enum]
                    scale = child._scale_factor(ch_enum, ch) * amp
                    phasor = child._phasor(ch_enum, phi_i)
                    Fx, Fy, Fz = child._sample_channel_at(
                        ch_enum, interps, xs, ys, zs)
                    if ch_enum.is_electric:
                        Ex_tot += scale * phasor * Fx
                        Ey_tot += scale * phasor * Fy
                        Ez_tot += scale * phasor * Fz
                    else:
                        Bx_tot += scale * phasor * Fx
                        By_tot += scale * phasor * Fy
                        Bz_tot += scale * phasor * Fz
                continue

            for ch_enum, ch in child.field_data.channels.items():
                scale = child._scale(ch_enum, ch) * amp
                if ch_enum.is_static:
                    phasor = np.ones(n)
                elif ch_enum.is_electric:
                    phasor = np.cos(phi_i)
                else:
                    phasor = np.sin(phi_i)
                Fx_p, Fy_p, Fz_p = child._sample_cart(ch, xs, ys, z_file)
                if ch_enum.is_electric:
                    Ex_tot += scale * phasor * Fx_p
                    Ey_tot += scale * phasor * Fy_p
                    Ez_tot += scale * phasor * Fz_p
                else:
                    Bx_tot += scale * phasor * Fx_p
                    By_tot += scale * phasor * Fy_p
                    Bz_tot += scale * phasor * Fz_p
                cross = child._sample_cart_cross(ch_enum, ch, xs, ys,
                                                 z_file)
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

        # d/e. un-Lorentz ⊕ un-damp (2×2 solve, then divide out damp)
        beta1, gamma1 = ref1.beta, ref1.gamma
        if beta1 > 0 and gamma1 > 0:
            factor_E = ds_m / (gamma1 * beta1 * beta1 * mass_MeV)
            factor_B = ds_m * _C_LIGHT / (gamma1 * beta1 * mass_MeV)
            a_coef = charge_num * factor_B * Bz_tot
            Cx = 1e3 * charge_num * (factor_E * Ex_tot
                                     - factor_B * By_tot)
            Cy = 1e3 * charge_num * (factor_E * Ey_tot
                                     + factor_B * Bx_tot)
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

        # f. un-energy
        beam.particles[alive, 5] -= dW_i - dW_ref

        # g. (step 0) un-rescale the FREQ-jump dphi
        if step_idx == 0 and freq_ratio is not None and freq_ratio != 1.0:
            beam.particles[alive, 4] /= freq_ratio

    # ------------------------------------------------------------------ #
    def reset_run_state(self) -> None:
        super().reset_run_state()          # clears _step_idx
        self._z_cursor = 0.0
        self._z_history = []
        self._phi_s_at_entrance = 0.0
        self._ref_entry = None
        for _z0, child in self.children:
            child.reset_run_state()

    def __repr__(self) -> str:
        kids = ", ".join(f"{child.name}@{z0:g}mm"
                         for z0, child in self.children)
        return (f"SuperposedFieldMap({self.name!r}, span={self.length:g} mm, "
                f"[{kids}])")
