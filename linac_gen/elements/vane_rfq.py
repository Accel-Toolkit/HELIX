"""TraceWin/Toutatis-equivalent RFQ tracker driven by a ``.vane`` file.

Wraps the entire RFQ as one :class:`FieldMapElement`, using per-z geometry
(apertures ``a₁..a₄``, voltages, ``Tc``) sampled from the ``.vane`` data
and cell-level RFQ_CELL parameters (``V``, ``A₁₀``, ``m``, ``L``, ``θₛ``,
``Type``) for the Crandall coefficients.

This is the M1 milestone toward full Toutatis equivalence:

* **M1 (this version)** — 2-term Crandall potential with cell-level
  coefficients ``(X, A, k=π/L, φ_s)`` and per-substep ``r₀(z)`` from the
  ``.vane`` interpolation.  Same per-particle physics as :class:`RfqCell`
  except the DC quadrupole coefficient ``A_quad = (1−A₁₀)/r₀(z)²`` is
  re-evaluated each substep.  Modest accuracy gain over the cell-constant
  ``r₀`` baseline; primarily a framework for higher milestones.
* **M2** — 8-term Crandall multipole expansion with coefficients fit
  from the ``.vane`` modulation envelope.
* **M3** — 2-D numerical Laplace at each ``z`` with vane-shape Dirichlet
  BCs (true Toutatis equivalent).

Activation
----------
Not used by the default parser path.  Construct manually, or use
:func:`linac_gen.io.vane_rfq_helper.replace_rfq_cells_with_vane` to swap
a contiguous chain of :class:`RfqCell` elements in a parsed lattice for
one :class:`VaneRFQ` driven by a sibling ``.vane`` file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from linac_gen.elements.base import FieldMapElement
from linac_gen.io.tracewin_vane import VaneGeometry


@dataclass
class CellSpan:
    """Per-cell parameters from a TraceWin RFQ_CELL line, plus z extent.

    A :class:`VaneRFQ` holds an ordered list of ``CellSpan`` covering its
    full length; physics at any element-local ``z_local`` looks up the
    ``CellSpan`` whose ``[z_start_mm, z_end_mm]`` contains it.  All
    lengths in mm, voltages in volts, phases in degrees — same units as
    the rest of LG.

    ``r0_dat_mm`` is the cell's design ``r₀`` from the RFQ_CELL line.
    The 2-term Crandall coefficients ``(X = 1−A₁₀, A_quad = (1−A₁₀)/r₀²)``
    are derived assuming a constant ``r₀_dat`` over the cell — using a
    per-z value from the .vane data here breaks self-consistency in
    matcher cells where the apertures flare.  Per-z geometry support
    arrives in M2 / M3 with a coefficient model that accommodates
    aperture variation correctly.
    """
    z_start_mm: float
    z_end_mm: float
    voltage_V: float
    A10: float
    modulation: float
    length_mm: float
    phi_s_deg: float
    cell_type: int
    type_prev: int
    type_next: int
    r0_dat_mm: float = 5.0
    Tc_mm: float = 0.0
    dP_deg: float = 0.0


class VaneRFQ(FieldMapElement):
    """Whole RFQ as one element with per-z geometry from a ``.vane`` file.

    Parameters
    ----------
    name : str
    vane : VaneGeometry
        Parsed ``.vane`` contents — provides ``r₀(z)``, voltages, etc.
    cells : list[CellSpan]
        Ordered cell list covering the full element length.  Cells must
        be contiguous and start at ``z=0`` of the element.
    n_steps : int, optional
        Total number of substeps over the whole element.  Default
        ``max(2 · vane.n_slices, 1000)`` — about half a vane-slice per
        substep, similar to the 0.1-mm RfqCell auto-default.
    aperture : float
        Loss-tracking aperture (mm).  ``0`` disables aperture checks.

    Notes
    -----
    *Substeps that straddle a cell boundary* are classified by the cell
    containing the substep midpoint; a substep > cell length will
    misbehave, so always run with ``n_steps`` large enough that
    ``self.length / n_steps`` ≪ shortest cell.

    *Voltage* is read per-z from the .vane (``V(z) = V₁ − V₂``) so
    voltage-modulated experiments track correctly.  For symmetric
    constant-V four-vane RFQs this equals the user's V parameter.
    """

    #: Set of allowed values for the ``field_model`` constructor keyword.
    #:
    #: Only the default ``"2term"`` is production-ready on the PXIE
    #: LEBT+RFQ benchmark.  The others are diagnostic/reference paths
    #: that document specific failure modes (see memory note
    #: ``project_rfq_m3_findings.md``).
    #:
    #: * ``"2term"``      — M1, cell-constant Wangler short-form
    #:                      A_quad = (1−A₁₀)/r₀².  Stable; σ_x within
    #:                      ~30 % of TraceWin reference.  See
    #:                      ``project_rfq_m1_manual_audit.md`` in
    #:                      auto-memory for why the literal TW-manual
    #:                      formula (A_01 = m·(1−A₁₀·I₀(ka))/r₀²,
    #:                      C₂ = sin(πz/L), V/4 in defoc) diverges —
    #:                      each of the three corrections individually
    #:                      pushes the substep AG kick past the
    #:                      Mathieu stability boundary.
    #: * ``"8term"``      — M2, matcher-aware per-z ``r₀`` from .vane.
    #: * ``"8term_full"`` — M3.3, full 8-term Crandall analytic
    #:                      expansion.  *Unstable* (AG resonance).
    #: * ``"laplace2d"``  — M3, numerical 2-D Laplace per .vane slice.
    #:                      σ_x converges; σ_y blows up.
    #: * ``"laplace3d"``  — M3.2, full 3-D Laplace with Shortley-Weller
    #:                      embedded boundary (Toutatis Eq. 4.17) +
    #:                      pyamg/GMRES.  K_xx, K_yy values are physical
    #:                      (~1900 V/mm² with K_xx + K_yy ≈ 0).  But the
    #:                      cache is single-polarity (constant DC vane
    #:                      voltages) and applying M1's cell-type S sign
    #:                      to recover AG triggers Mathieu resonance.
    #:                      Without S, σ_y diverges.  Either way σ → ∞.
    FIELD_MODELS = {"2term", "8term", "8term_full",
                    "laplace2d", "laplace3d"}

    def __init__(self,
                 name: str,
                 vane: VaneGeometry,
                 cells: List[CellSpan],
                 n_steps: Optional[int] = None,
                 aperture: float = 0.0,
                 field_model: str = "2term",
                 laplace_cache=None,
                 laplace_kwargs: Optional[dict] = None):
        if not cells:
            raise ValueError("VaneRFQ needs at least one CellSpan")
        if field_model not in self.FIELD_MODELS:
            raise ValueError(
                f"unknown field_model {field_model!r}; "
                f"valid: {sorted(self.FIELD_MODELS)}"
            )
        self.field_model = str(field_model)
        # Element length is the cell-list span — that's the region with
        # well-defined physics.  The .vane file may extend beyond the
        # cell list (e.g., post-RFQ matcher geometry); we still use it
        # for r₀(z) lookups, but the element only covers the cell range.
        length_mm = float(cells[-1].z_end_mm - cells[0].z_start_mm)
        if n_steps is None:
            # ~2× the vane resolution within the element's z-span.
            vane_length_mm = (vane.z[-1] - vane.z[0]) * 1000.0
            slices_in_span = max(1, int(round(
                vane.n_slices * length_mm / vane_length_mm
            ))) if vane_length_mm > 0 else vane.n_slices
            n_steps = max(2 * slices_in_span, 1000)
        super().__init__(name=name, length=length_mm,
                         aperture=aperture, n_steps=int(n_steps))
        self.vane = vane
        self.cells = list(cells)

        # Pre-compute lookup arrays for fast per-substep queries.
        self._cell_z_starts = np.array([c.z_start_mm for c in self.cells])
        self._cell_z_ends   = np.array([c.z_end_mm   for c in self.cells])

        # Substep state — ``_step_idx`` is reset to 0 by the lattice
        # tracker on element entry (same convention as RfqCell).
        self._step_idx: int = 0
        self._current_cell_idx: int = -1
        self._cell_phi_deg: float = 0.0

        # Independent cursor for ``advance_ref_over``: the EnvelopeSolver
        # bundle loop calls advance_ref_over for non-overlapping segments,
        # and we need to carry the cell-local phase across calls.
        self._adv_cell_phi_deg: float = 0.0
        self._adv_cur_cell: int = -1

        # ---- M3 (laplace2d / laplace3d) cache -----------------------
        # Eagerly built so the first track_rk4 / fitted_matrix_slice
        # call doesn't pay the Laplace solve cost.  Pass an already-
        # built cache via ``laplace_cache=`` to share across multiple
        # VaneRFQ elements or to keep test fixtures fast.
        self._laplace_cache = None
        if self.field_model in ("laplace2d", "laplace3d"):
            kwargs = dict(laplace_kwargs or {})
            if laplace_cache is not None:
                self._laplace_cache = laplace_cache
            elif self.field_model == "laplace2d":
                from linac_gen.elements.vane_rfq_laplace2d import Laplace2DCache
                self._laplace_cache = Laplace2DCache(self.vane, **kwargs)
            else:  # laplace3d
                from linac_gen.elements.vane_rfq_laplace3d import Laplace3DCache
                self._laplace_cache = Laplace3DCache(self.vane, **kwargs)

        # ---- M3.3 (8term_full) per-cell coefficients ----------------
        # Analytic 2-term Crandall coefficients from each cell's .dat
        # parameters (m, A₁₀, r₀, L).  No BC fit on .vane data — that
        # produces wild higher-order coefficients in matcher / type-3
        # cells where the .vane profile diverges from the Crandall
        # ansatz.  See module docstring of vane_rfq_8term_full.
        self._cell_coeffs_8term = None
        if self.field_model == "8term_full":
            from linac_gen.elements.vane_rfq_8term_full import (
                solve_cell_coeffs_dat,
            )
            V_amp = float(self._vane_voltage_V(
                0.5 * (self.cells[0].z_start_mm + self.cells[-1].z_end_mm)
            ))
            if abs(V_amp) < 1e-9:
                V_amp = float(self.vane.inter_vane_voltage())
            self._cell_coeffs_8term = [
                solve_cell_coeffs_dat(
                    c.z_start_mm, c.z_end_mm,
                    c.modulation, c.A10, c.r0_dat_mm, V_amp,
                    S_sign=float(self._type_coeffs(0.5 * c.length_mm, c)[2]),
                )
                for c in self.cells
            ]

    # ------------------------------------------------------------------
    # Geometry lookups
    # ------------------------------------------------------------------
    def _vane_r0_mm(self, z_mm: float) -> float:
        """Interpolated ``r₀(z) = √(a₁ a₂)`` (mm) at element-local z (mm).

        ``vane.aperture_v*`` are stored in metres in :class:`VaneGeometry`;
        we return mm so the caller can use it as the ``r₀`` in the same
        units RfqCell does.
        """
        z_m = z_mm * 1e-3
        a1_m = float(np.interp(z_m, self.vane.z, self.vane.aperture_v1))
        a2_m = float(np.interp(z_m, self.vane.z, self.vane.aperture_v2))
        return float(np.sqrt(a1_m * a2_m)) * 1000.0

    def _vane_voltage_V(self, z_mm: float) -> float:
        """Inter-vane voltage ``V(z) = V₁(z) − V₂(z)`` at element-local z."""
        z_m = z_mm * 1e-3
        v1 = float(np.interp(z_m, self.vane.z, self.vane.voltage_v1))
        v2 = float(np.interp(z_m, self.vane.z, self.vane.voltage_v2))
        return v1 - v2

    def _cell_index(self, z_mm: float) -> int:
        """Index of the :class:`CellSpan` containing element-local ``z_mm``.

        Uses :func:`numpy.searchsorted` on the cell-start array; returns
        ``-1`` if ``z_mm`` lies outside the cell list (shouldn't happen
        for a well-formed VaneRFQ, but the caller should fall back to a
        pure drift in that case).
        """
        idx = int(np.searchsorted(self._cell_z_starts, z_mm, side='right') - 1)
        if 0 <= idx < len(self.cells) and z_mm <= self._cell_z_ends[idx] + 1e-9:
            return idx
        return -1

    # ------------------------------------------------------------------
    # Per-Type C₁(z), C₂(z), S coefficients — cloned from RfqCell.
    # ``z_in_cell_mm`` is z relative to *cell entry*, in mm.
    # ------------------------------------------------------------------
    @staticmethod
    def _type_coeffs(z_in_cell_mm: float, cell: CellSpan) -> tuple[float, float, float]:
        L = cell.length_mm
        if L <= 0:
            return 1.0, 0.0, 0.0
        arg = np.pi * z_in_cell_mm / L
        sign_type = 1 if cell.cell_type > 0 else -1
        abs_type = abs(cell.cell_type)

        if abs_type == 2:
            return 1.0, float(np.cos(arg)), -float(sign_type)
        if abs_type == 3:
            if sign_type > 0:
                S = -1.0 if cell.type_next > 0 else +1.0
            else:
                S = -1.0 if cell.type_prev > 0 else +1.0
            return 1.0, 0.0, S
        if abs_type == 4:
            if sign_type > 0:
                C2 = 0.5 * (float(np.cos(arg)) + 1.0)
                S  = -1.0 if cell.type_next > 0 else +1.0
            else:
                C2 = 0.5 * (1.0 - float(np.cos(arg)))
                S  = -1.0 if cell.type_prev > 0 else +1.0
            return 1.0, C2, S
        return 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _Ez_onaxis(z_in_cell_mm: float, phase_rad: float, cell: CellSpan,
                   V_local: float) -> float:
        L = cell.length_mm
        if L <= 0:
            return 0.0
        return ((np.pi * cell.A10 * V_local) / (2.0 * L)
                * np.sin(np.pi * z_in_cell_mm / L)
                * np.sin(phase_rad))

    # ------------------------------------------------------------------
    # FieldMapElement contract
    # ------------------------------------------------------------------
    def track_rk4(self, beam, ds: float) -> None:
        """Advance the beam by a single substep of length ``ds`` (mm)."""
        ref = beam.ref
        ds_m = ds * 1e-3
        ds_half_m = 0.5 * ds_m

        z_mid_mm = self._step_idx * ds + ds * 0.5
        # Reset cell-phase tracking on element entry (first substep).
        if self._step_idx == 0:
            self._current_cell_idx = -1
            self._cell_phi_deg = 0.0
        self._step_idx += 1

        cidx = self._cell_index(z_mid_mm)
        if cidx < 0:
            # Outside cell list ⇒ drift.  Still advance ref so downstream
            # elements stay synchronised.
            alive = beam.alive_mask
            if np.any(alive):
                beam.particles[alive, 0] += beam.particles[alive, 1] * ds_m
                beam.particles[alive, 2] += beam.particles[alive, 3] * ds_m
            ref.s += ds
            return

        cell = self.cells[cidx]
        # Cell-boundary crossing → reset cell-local synchronous phase.
        if cidx != self._current_cell_idx:
            self._current_cell_idx = cidx
            self._cell_phi_deg = 0.0
        z_in_cell_mm = z_mid_mm - cell.z_start_mm

        # DC quadrupole coefficient.  Default (``field_model="2term"``)
        # uses the cell-constant short-form ``A_quad = (1−A₁₀)/r₀²``
        # for bit-identical M1 behaviour.  ``"8term"`` adds matcher-
        # aware per-z r₀ from the .vane (M2).  ``"laplace2d"`` (M3)
        # replaces the analytic Crandall transverse coefficients with
        # the in-plane second derivatives ``K_xx, K_yy`` of a numerical
        # Laplace solution per .vane slice.  The longitudinal Ez stays
        # on the M1 analytic path because the per-slice 2-D solver,
        # combined with sub-grid mask quantisation, does not produce a
        # numerically clean ``∂Φ/∂z`` at axis on the typical .vane
        # resolution.
        V_local = self._vane_voltage_V(z_mid_mm)
        if self.field_model in ("laplace2d", "laplace3d"):
            # K_xx, K_yy from the cached 2-D Laplace per-slice solution.
            A_quad_local = 0.0
            K_xx_static = float(self._laplace_cache.K_xx_axis(z_mid_mm))
            K_yy_static = float(self._laplace_cache.K_yy_axis(z_mid_mm))
            Ez_static_axis = 0.0  # not used; M1 analytic E_z still drives ref
        elif self.field_model == "8term_full":
            from linac_gen.elements.vane_rfq_8term_full import (
                K_xx_axis as _Kxx_8tf, K_yy_axis as _Kyy_8tf,
            )
            cc = self._cell_coeffs_8term[cidx]
            A_quad_local = 0.0
            K_xx_static = float(_Kxx_8tf(cc, z_mid_mm))
            K_yy_static = float(_Kyy_8tf(cc, z_mid_mm))
            Ez_static_axis = 0.0  # M1 analytic E_z still drives ref (matches M1 normalisation)
        elif self.field_model == "2term":
            denom = cell.r0_dat_mm * cell.r0_dat_mm
            A_quad_local = (max(0.0, (1.0 - cell.A10)) / denom
                            if denom > 0 else 0.0)
            K_xx_static = K_yy_static = Ez_static_axis = 0.0
        else:  # "8term"
            from linac_gen.elements.vane_rfq_8term import (
                A_quad_local as _A_quad_8term, effective_r0_mm,
            )
            r0_eff = effective_r0_mm(z_mid_mm, cell.r0_dat_mm,
                                     cell.A10, self._vane_r0_mm)
            A_quad_local = _A_quad_8term(cell.A10, cell.modulation,
                                         r0_eff, cell.r0_dat_mm,
                                         cell.length_mm)
            K_xx_static = K_yy_static = Ez_static_axis = 0.0

        beta = ref.beta
        gamma = ref.gamma
        mass_MeV = ref.species.mass
        charge   = ref.species.charge
        wl_mm    = ref.wavelength
        q_abs    = abs(charge)

        # Cell-local synchronous phase at the substep midpoint
        if wl_mm > 0:
            dphi_to_mid = 180.0 * ds * 0.5 / (beta * wl_mm)
        else:
            dphi_to_mid = 0.0
        phi_s_mid_deg = self._cell_phi_deg + dphi_to_mid + cell.phi_s_deg
        phi_s_mid_rad = np.deg2rad(phi_s_mid_deg)

        # E_z always comes from the M1 analytic on-axis form; M3 only
        # changes the transverse kick coefficients.
        E_z = self._Ez_onaxis(z_in_cell_mm, phi_s_mid_rad,
                              cell, V_local)
        dW_ref_MeV = q_abs * E_z * ds * 1e-6
        ref.w_kin += dW_ref_MeV
        ref.s += ds
        if wl_mm > 0:
            full_step_dphi = 360.0 * ds / (beta * wl_mm)
            ref.phi_s += full_step_dphi
            self._cell_phi_deg += full_step_dphi
            # Cell-end ``dP`` shift, applied on the substep that crosses
            # the cell exit.
            if (z_mid_mm + ds * 0.5 >= cell.z_end_mm
                    and cell.dP_deg != 0.0):
                ref.phi_s += cell.dP_deg

        alive = beam.alive_mask
        if not np.any(alive):
            return

        # Half-drift transverse (Strang)
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_half_m

        # Per-particle longitudinal kick
        dphi_deg = beam.particles[alive, 4]
        phi_part_rad = phi_s_mid_rad + np.deg2rad(dphi_deg)
        E_z_part = ((np.pi * cell.A10 * V_local) / (2.0 * cell.length_mm)
                    * np.sin(np.pi * z_in_cell_mm / cell.length_mm)
                    * np.sin(phi_part_rad))
        dW_part_MeV = q_abs * E_z_part * ds * 1e-6
        beam.particles[alive, 5] += dW_part_MeV - dW_ref_MeV

        # Adiabatic damping
        if beta > 0 and gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_part_MeV / (beta * beta * gamma * mass_MeV))
            beam.particles[alive, 1] *= damp
            beam.particles[alive, 3] *= damp

        # Per-particle transverse kick
        cos_part = np.cos(phi_part_rad)
        if self.field_model in ("laplace2d", "laplace3d", "8term_full"):
            # M3 / M3.3: linearised quadrupole kick from K_xx, K_yy
            # already computed at the substep midpoint.  K_xx_static is
            # ``∂²Φ/∂x²`` of the FULL potential (vane voltages applied
            # directly, no extra V/2 factor).  The 0.5 in the kick line
            # below recovers the M1 ``(V/2)·A_quad`` convention so
            # numerical and analytic paths agree in the 2-term limit.
            #
            # NOTE: the .vane geometry has constant DC vane voltages
            # (±V/2) so K_xx, K_yy are single-polarity along z and the
            # alternating-gradient (AG) sign flip that M1 introduces via
            # the cell-type ``S`` factor is NOT present here.  Multiplying
            # by S empirically pushes the AG strength above the Mathieu
            # stability limit (4× M1 K → parametric resonance, σ → ∞).
            # See ``project_rfq_m3_findings.md`` for the outcome map.
            if beta > 0 and gamma > 0 and mass_MeV > 0:
                factor = q_abs * ds / (gamma * beta * beta * mass_MeV * 1e6)
                kx_per_x = -factor * cos_part * K_xx_static * 0.5
                ky_per_y = -factor * cos_part * K_yy_static * 0.5
                xs_mm = beam.particles[alive, 0]
                ys_mm = beam.particles[alive, 2]
                beam.particles[alive, 1] += kx_per_x * xs_mm * 1e3
                beam.particles[alive, 3] += ky_per_y * ys_mm * 1e3
        else:
            C1, C2, S = self._type_coeffs(z_in_cell_mm, cell)
            rf_quad_part = cos_part * S * (V_local * 0.5) * A_quad_local * C1
            rf_defoc     = (np.pi / cell.length_mm) ** 2 * (cell.A10 * V_local * 0.5) * C2

            if beta > 0 and gamma > 0 and mass_MeV > 0:
                factor = q_abs * ds / (gamma * beta * beta * mass_MeV * 1e6)
                kx_per_x = -factor * (rf_quad_part - rf_defoc)
                ky_per_y = -factor * (-rf_quad_part - rf_defoc)
                xs_mm = beam.particles[alive, 0]
                ys_mm = beam.particles[alive, 2]
                beam.particles[alive, 1] += kx_per_x * xs_mm * 1e3
                beam.particles[alive, 3] += ky_per_y * ys_mm * 1e3

        # Half-drift transverse (Strang)
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_half_m

    # ------------------------------------------------------------------
    def _ez_at(self, z_mid_mm: float, z_in_cell_mm: float,
               cell: CellSpan, phi_s_mid_rad: float, V_local: float
               ) -> float:
        """E_z on axis (V/mm) at the substep midpoint.

        All field-model branches use the M1 analytic ``_Ez_onaxis`` form
        for the longitudinal field.  M3's per-slice 2-D Laplace solve
        does not produce a numerically clean ``∂Φ/∂z`` at axis on the
        typical .vane native sampling, so the longitudinal physics is
        handed back to the M1 cell-level expression while M3 still
        drives the transverse kick via cached ``K_xx``, ``K_yy``.
        """
        return self._Ez_onaxis(z_in_cell_mm, phi_s_mid_rad, cell, V_local)

    def advance_ref(self, ref) -> None:
        """Advance ``ref`` through the entire RFQ.

        Mirrors :meth:`track_rk4` for ref-only physics — used in plain
        (non-envelope) tracking when a snapshot of ref state is needed
        without going through the substep loop.
        """
        n = self.n_steps
        ds = self.length / n
        cur_cell = -1
        cell_phi_deg = 0.0
        for i in range(n):
            z_mid_mm = (i + 0.5) * ds
            cidx = self._cell_index(z_mid_mm)
            if cidx < 0:
                ref.s += ds
                continue
            cell = self.cells[cidx]
            if cidx != cur_cell:
                cur_cell = cidx
                cell_phi_deg = 0.0
            z_in_cell = z_mid_mm - cell.z_start_mm
            V_local = self._vane_voltage_V(z_mid_mm)
            if ref.wavelength > 0:
                dphi_mid = 180.0 * ds * 0.5 / (ref.beta * ref.wavelength)
            else:
                dphi_mid = 0.0
            phi_s_mid_deg = cell_phi_deg + dphi_mid + cell.phi_s_deg
            E_z = self._ez_at(z_mid_mm, z_in_cell, cell,
                              np.deg2rad(phi_s_mid_deg), V_local)
            ref.w_kin += abs(ref.species.charge) * E_z * ds * 1e-6
            ref.s += ds
            if ref.wavelength > 0:
                full = 360.0 * ds / (ref.beta * ref.wavelength)
                ref.phi_s += full
                cell_phi_deg += full
                if (z_mid_mm + ds * 0.5 >= cell.z_end_mm
                        and cell.dP_deg != 0.0):
                    ref.phi_s += cell.dP_deg

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        """Advance ``ref`` over a sub-range — used by the EnvelopeSolver.

        The bundle loop calls this for non-overlapping segments; we carry
        the cell-local phase on the instance (``_adv_*``) so the slice
        cursor produces the same E_z(z, t_s) sampling as a single
        :meth:`advance_ref` call would.
        """
        length_slice = z_to_mm - z_from_mm
        if length_slice <= 0.0:
            return
        if z_from_mm == 0.0:
            self._adv_cell_phi_deg = 0.0
            self._adv_cur_cell = -1
        n_full = max(1, self.n_steps)
        native_ds = self.length / n_full
        n_sub = max(1, int(round(length_slice / native_ds)))
        ds = length_slice / n_sub
        cur_cell = self._adv_cur_cell
        cell_phi_deg = self._adv_cell_phi_deg
        for i in range(n_sub):
            z_mid_mm = z_from_mm + (i + 0.5) * ds
            cidx = self._cell_index(z_mid_mm)
            if cidx < 0:
                ref.s += ds
                continue
            cell = self.cells[cidx]
            if cidx != cur_cell:
                cur_cell = cidx
                cell_phi_deg = 0.0
            z_in_cell = z_mid_mm - cell.z_start_mm
            V_local = self._vane_voltage_V(z_mid_mm)
            if ref.wavelength > 0:
                dphi_mid = 180.0 * ds * 0.5 / (ref.beta * ref.wavelength)
            else:
                dphi_mid = 0.0
            phi_s_mid_deg = cell_phi_deg + dphi_mid + cell.phi_s_deg
            E_z = self._ez_at(z_mid_mm, z_in_cell, cell,
                              np.deg2rad(phi_s_mid_deg), V_local)
            ref.w_kin += abs(ref.species.charge) * E_z * ds * 1e-6
            ref.s += ds
            if ref.wavelength > 0:
                full = 360.0 * ds / (ref.beta * ref.wavelength)
                ref.phi_s += full
                cell_phi_deg += full
                if (z_mid_mm + ds * 0.5 >= cell.z_end_mm
                        and cell.dP_deg != 0.0):
                    ref.phi_s += cell.dP_deg
        self._adv_cell_phi_deg = cell_phi_deg
        self._adv_cur_cell = cur_cell

    # ------------------------------------------------------------------
    def fitted_matrix_slice(self, ref, ds_mm: float,
                            _z_from_mm: float | None = None) -> np.ndarray:
        """Substep DKD transfer matrix for a sub-range, mirroring track_rk4.

        Parameters
        ----------
        ref : ReferenceParticle
            Entry-side reference (β, γ, λ at slice start).
        ds_mm : float
            Slice length in mm.
        _z_from_mm : float, optional
            Element-local position of slice entry (mm).  ``None`` means
            slice covers the full element starting at ``z=0``.
        """
        if ds_mm <= 0:
            return np.eye(6)
        z_from = 0.0 if _z_from_mm is None else float(_z_from_mm)

        n_full = max(1, self.n_steps)
        native_ds = self.length / n_full
        n_sub = max(1, int(round(ds_mm / native_ds)))
        ds = ds_mm / n_sub

        wl = ref.wavelength
        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass
        q_abs = abs(ref.species.charge)

        if wl > 0 and beta > 0 and gamma > 0 and mass > 0:
            r45_full = -360.0 * ds / (beta**3 * gamma**3 * mass * wl)
        else:
            r45_full = 0.0
        d_half_xprime = ds * 0.5e-3   # mm per mrad

        Mh = np.eye(6)
        Mh[0, 1] = d_half_xprime
        Mh[2, 3] = d_half_xprime
        Mh[4, 5] = 0.5 * r45_full

        M_total = np.eye(6)
        cur_cell = -1
        cell_phi_deg = 0.0
        for i in range(n_sub):
            z_mid_mm = z_from + (i + 0.5) * ds
            cidx = self._cell_index(z_mid_mm)
            if cidx < 0:
                # Pure drift sub-block
                Md = np.eye(6)
                Md[0, 1] = ds * 1e-3
                Md[2, 3] = ds * 1e-3
                Md[4, 5] = r45_full
                M_total = Md @ M_total
                continue
            cell = self.cells[cidx]
            if cidx != cur_cell:
                cur_cell = cidx
                # Reset cell-local phase on cell entry; the phase
                # accumulator is rebuilt from the substep's distance into
                # the cell using the entry-side β.  Same approximation
                # RfqCell.fitted_matrix_slice uses.
                if wl > 0 and beta > 0:
                    cell_phi_deg = (
                        360.0 * (z_mid_mm - cell.z_start_mm - 0.5 * ds)
                        / (beta * wl)
                    )
                else:
                    cell_phi_deg = 0.0
            z_in_cell = z_mid_mm - cell.z_start_mm
            V_local = self._vane_voltage_V(z_mid_mm)
            # Same field-model dispatch as track_rk4 — 2term keeps the
            # short-form (1−A₁₀)/r₀²; 8term uses matcher-aware per-z r₀;
            # 8term_full uses the analytic Crandall multipole basis;
            # laplace2d/3d substitutes the numerical Φ_static gradients.
            if self.field_model in ("laplace2d", "laplace3d"):
                A_quad_local = 0.0
                K_xx_static = float(self._laplace_cache.K_xx_axis(z_mid_mm))
                K_yy_static = float(self._laplace_cache.K_yy_axis(z_mid_mm))
            elif self.field_model == "8term_full":
                from linac_gen.elements.vane_rfq_8term_full import (
                    K_xx_axis as _Kxx_8tf, K_yy_axis as _Kyy_8tf,
                )
                cc = self._cell_coeffs_8term[cidx]
                A_quad_local = 0.0
                K_xx_static = float(_Kxx_8tf(cc, z_mid_mm))
                K_yy_static = float(_Kyy_8tf(cc, z_mid_mm))
            elif self.field_model == "2term":
                denom = cell.r0_dat_mm * cell.r0_dat_mm
                A_quad_local = (max(0.0, (1.0 - cell.A10)) / denom
                                if denom > 0 else 0.0)
                K_xx_static = K_yy_static = 0.0
            else:
                from linac_gen.elements.vane_rfq_8term import (
                    A_quad_local as _A_quad_8term, effective_r0_mm,
                )
                r0_eff = effective_r0_mm(z_mid_mm, cell.r0_dat_mm,
                                         cell.A10, self._vane_r0_mm)
                A_quad_local = _A_quad_8term(cell.A10, cell.modulation,
                                             r0_eff, cell.r0_dat_mm,
                                             cell.length_mm)
                K_xx_static = K_yy_static = 0.0

            if wl > 0 and beta > 0:
                dphi_mid = 180.0 * ds * 0.5 / (beta * wl)
            else:
                dphi_mid = 0.0
            phi_s_mid_rad = np.deg2rad(cell_phi_deg + dphi_mid + cell.phi_s_deg)
            cos_phase = float(np.cos(phi_s_mid_rad))

            if self.field_model in ("laplace2d", "laplace3d", "8term_full"):
                # Transverse from numerical Laplace or analytic 8-term;
                # longitudinal R[5,4] still uses M1's analytic form
                # (see _ez_at docstring).  See track_rk4 NOTE on why no
                # cell-type S factor is applied — multiplying by S pushes
                # the AG above the Mathieu stability limit and σ explodes.
                if beta > 0 and gamma > 0 and mass > 0:
                    factor = q_abs * ds / (gamma * beta * beta * mass * 1e6)
                    kx_per_x = -factor * cos_phase * K_xx_static * 0.5
                    ky_per_y = -factor * cos_phase * K_yy_static * 0.5
                else:
                    kx_per_x = 0.0
                    ky_per_y = 0.0
                if cell.length_mm > 0:
                    Ez_amp_per_phi = (
                        (np.pi * cell.A10 * V_local) / (2.0 * cell.length_mm)
                        * np.sin(np.pi * z_in_cell / cell.length_mm)
                        * cos_phase
                    )
                    k54 = q_abs * Ez_amp_per_phi * ds * 1e-6 * (np.pi / 180.0)
                else:
                    k54 = 0.0
            else:
                C1, C2, S = self._type_coeffs(z_in_cell, cell)
                rf_quad   = cos_phase * S * (V_local * 0.5) * A_quad_local * C1
                rf_defoc  = ((np.pi / cell.length_mm) ** 2
                             * (cell.A10 * V_local * 0.5) * C2)

                if beta > 0 and gamma > 0 and mass > 0:
                    factor = q_abs * ds / (gamma * beta * beta * mass * 1e6)
                    kx_per_x = -factor * (rf_quad - rf_defoc)
                    ky_per_y = -factor * (-rf_quad - rf_defoc)
                else:
                    kx_per_x = 0.0
                    ky_per_y = 0.0

                if cell.length_mm > 0:
                    Ez_amp_per_phi = (
                        (np.pi * cell.A10 * V_local) / (2.0 * cell.length_mm)
                        * np.sin(np.pi * z_in_cell / cell.length_mm)
                        * cos_phase
                    )
                    k54 = q_abs * Ez_amp_per_phi * ds * 1e-6 * (np.pi / 180.0)
                else:
                    k54 = 0.0

            Mk = np.eye(6)
            Mk[1, 0] = kx_per_x * 1e3
            Mk[3, 2] = ky_per_y * 1e3
            Mk[5, 4] = k54

            M_sub = Mh @ Mk @ Mh
            M_total = M_sub @ M_total

            if wl > 0 and beta > 0:
                cell_phi_deg += 360.0 * ds / (beta * wl)

        return M_total

    def fitted_matrix(self, ref) -> np.ndarray:
        return self.fitted_matrix_slice(ref, self.length, _z_from_mm=0.0)
