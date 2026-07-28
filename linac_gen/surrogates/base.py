"""Base classes for HELIX surrogate elements.

M1 scope (per ``docs/plans/surrogates.md``):
* ``SurrogateFieldMap(FieldMapElement, nn.Module)`` — a drop-in for
  any ``FieldMapElement`` instance in **envelope-mode** tracking.
* ``fitted_matrix(ref)`` is implemented by an MLP that predicts the
  flattened 6x6 transfer matrix from (ref kinematics, element params).
* ``track_rk4(beam, ds)`` and ``fitted_matrix_slice(ref, ds_mm)``
  FALL BACK to the wrapped element — M7 / future work will replace
  these with per-particle NN tracking and slice-aware surrogates.
* Out-of-scope inputs raise :class:`OutOfScopeError`; callers fall
  back to the wrapped element.
"""
from __future__ import annotations

import copy as _copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from linac_gen.elements.base import FieldMapElement


F64 = torch.float64


# ---------------------------------------------------------------------------
class OutOfScopeError(RuntimeError):
    """Raised when a surrogate is queried outside its trained scope."""


# ---------------------------------------------------------------------------
class MlpHead(nn.Module):
    """Configurable MLP: input -> hidden(s) -> output.

    FP64 throughout, smooth activation (default SiLU) so the Jacobian
    stays clean for downstream gradient-matcher use (M7).
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: Sequence[int] = (128, 128, 128),
                 activation: str = "silu") -> None:
        super().__init__()
        act_cls = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h), dtype=F64))
            layers.append(act_cls())
            prev = int(h)
        layers.append(nn.Linear(prev, int(output_dim), dtype=F64))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
@dataclass
class Scope:
    """Training-set scope bounds for OOD admissibility.

    The check is elementwise on the input vector: each entry must
    lie in ``[lo, hi]`` (with a small ``eps`` tolerance) to be in
    scope.
    """
    input_names: list[str]
    input_lo: np.ndarray   # shape (input_dim,)
    input_hi: np.ndarray   # shape (input_dim,)

    def is_in_scope(self, x: np.ndarray, eps: float = 1e-9) -> bool:
        x = np.asarray(x, dtype=np.float64)
        return bool(np.all(x >= self.input_lo - eps)
                    and np.all(x <= self.input_hi + eps))

    def to_dict(self) -> dict:
        return {
            "input_names": list(self.input_names),
            "input_lo": [float(v) for v in self.input_lo],
            "input_hi": [float(v) for v in self.input_hi],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scope":
        return cls(
            input_names=list(d["input_names"]),
            input_lo=np.asarray(d["input_lo"], dtype=np.float64),
            input_hi=np.asarray(d["input_hi"], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
@dataclass
class SurrogateMetadata:
    """Bookkeeping persisted alongside a trained weights file.

    Required for reproducibility (seed, commit SHA, lattice hash),
    runtime admissibility (scope), and inference (normalisation).
    """
    element_key: str
    element_class: str
    architecture: dict
    scope: Scope
    input_norm: dict   # {"mean": [...], "std": [...]}
    output_norm: dict  # {"mean": [...], "std": [...]}
    training_seed: int
    n_samples: int
    epochs: int
    val_mape: float
    helix_commit_sha: str
    lattice_hash: str
    created_iso: str
    # Training kinematics (added 2026-07-18 with the H- mass fix): the
    # species the transfer maps were sampled with.  Defaults keep legacy
    # metadata loadable; load_surrogate warns when they are absent or when
    # the recorded mass no longer matches the current species table.
    species_name: str = ""
    species_mass_mev: float = 0.0
    # SHA-256 of weights.pt, computed by load_surrogate() at READ time
    # (never trusted from metadata.json): carried on the metadata so
    # every load route — CLI, GUI, and the plain Python API — feeds
    # the HDF5 provenance manifest uniformly.  Empty for a surrogate
    # never loaded from disk (in-memory training).
    weights_sha256: str = ""

    def to_json(self) -> dict:
        return {
            "element_key": self.element_key,
            "element_class": self.element_class,
            "architecture": self.architecture,
            "scope": self.scope.to_dict(),
            "input_norm": self.input_norm,
            "output_norm": self.output_norm,
            "training_seed": self.training_seed,
            "n_samples": self.n_samples,
            "epochs": self.epochs,
            "val_mape": self.val_mape,
            "helix_commit_sha": self.helix_commit_sha,
            "lattice_hash": self.lattice_hash,
            "created_iso": self.created_iso,
            "species_name": self.species_name,
            "species_mass_mev": self.species_mass_mev,
            # informational only — load_surrogate() recomputes from
            # the actual file bytes and overwrites this.
            "weights_sha256": self.weights_sha256,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SurrogateMetadata":
        return cls(
            element_key=str(d["element_key"]),
            element_class=str(d["element_class"]),
            architecture=dict(d["architecture"]),
            scope=Scope.from_dict(d["scope"]),
            input_norm=dict(d["input_norm"]),
            output_norm=dict(d["output_norm"]),
            training_seed=int(d["training_seed"]),
            n_samples=int(d["n_samples"]),
            epochs=int(d["epochs"]),
            val_mape=float(d["val_mape"]),
            helix_commit_sha=str(d["helix_commit_sha"]),
            lattice_hash=str(d["lattice_hash"]),
            species_name=str(d.get("species_name", "")),
            species_mass_mev=float(d.get("species_mass_mev", 0.0)),
            created_iso=str(d["created_iso"]),
        )


# ---------------------------------------------------------------------------
class SurrogateFieldMap(FieldMapElement, nn.Module):
    """ML surrogate for any FieldMapElement, envelope-mode (M1)."""

    def __init__(self,
                 wrapped: FieldMapElement,
                 mlp: MlpHead,
                 metadata: SurrogateMetadata) -> None:
        FieldMapElement.__init__(
            self,
            name=wrapped.name,
            length=wrapped.length,
            aperture=wrapped.aperture,
            n_steps=wrapped.n_steps,
        )
        nn.Module.__init__(self)
        self._wrapped = wrapped
        self.mlp = mlp
        self.metadata = metadata
        # Normalisation tensors (FP64) — kept on CPU.
        self._in_mean = torch.as_tensor(metadata.input_norm["mean"], dtype=F64)
        self._in_std = torch.as_tensor(metadata.input_norm["std"], dtype=F64)
        self._out_mean = torch.as_tensor(metadata.output_norm["mean"], dtype=F64)
        self._out_std = torch.as_tensor(metadata.output_norm["std"], dtype=F64)
        self._param_names: list[str] = list(
            metadata.architecture.get("param_names", []))
        # Ensure inference mode by default.
        self.mlp.eval()
        # M3 envelope hook reads this — set False to bypass surrogate.
        self._enabled: bool = True
        # M7 hybrid MP knob: how many RK4 substeps the residual pass
        # uses inside :meth:`track_rk4`.  Lower = faster + less
        # accurate; higher = slower + closer to baseline RK4.
        # Default 15 is the "accuracy first" choice from the plan.
        self.residual_n_steps: int = 15
        # M7 fast-path cache: ``scipy.linalg.logm + expm`` is ~10 ms on
        # a 6x6 matrix (eigendecomposition).  Across ~160 substeps per
        # cavity × 20 cavities that dominates the run wall-clock.
        # Slice matrices are constant per ``(ds, ref.w_kin)`` within
        # one element's inner loop, so cache the most recent.
        self._slice_cache_key: tuple | None = None
        self._slice_cache_M: np.ndarray | None = None
        # M7-followup fast-path per-element state.  All cleared by
        # `reset_run_state` so each element traversal re-inits cleanly.
        self._fast_init_done: bool = False
        self._fast_init_failed: bool = False
        self._M_full: np.ndarray | None = None
        self._log_M_full: np.ndarray | None = None
        self._electric_channels: list | None = None

    # ------------------------------------------------------------------
    # Envelope-mode contract
    # ------------------------------------------------------------------
    def fitted_matrix(self, ref) -> np.ndarray:
        """Predict the linearised 6x6 transfer matrix.

        Raises :class:`OutOfScopeError` if the input falls outside the
        training-set scope (caller falls back to the wrapped element).
        """
        if not self._enabled:
            return self._wrapped.fitted_matrix(ref)
        x = self._make_input(ref)
        if not self.metadata.scope.is_in_scope(x):
            raise OutOfScopeError(
                f"surrogate '{self.metadata.element_key}': input "
                f"{x.tolist()} outside training scope")
        x_t = torch.as_tensor(x, dtype=F64)
        x_norm = (x_t - self._in_mean) / self._in_std
        with torch.no_grad():
            y_norm = self.mlp(x_norm)
        y = y_norm * self._out_std + self._out_mean
        M = y.detach().numpy().reshape(6, 6)
        # Output sanity gate: a corrupted/NaN weight file must not
        # inject garbage into the beamline — raise OutOfScopeError so
        # every caller takes its existing wrapped-element fallback.
        # Two unambiguous rejections only (a condition-number threshold
        # would be arbitrary and could refuse legitimate strongly-
        # focusing optics): non-finite entries, and an outright
        # singular matrix — no physical transfer map is singular
        # (Liouville: |det| ~ 1 up to adiabatic damping).
        if not np.all(np.isfinite(M)):
            raise OutOfScopeError(
                f"surrogate '{self.metadata.element_key}': NN produced "
                "a non-finite transfer matrix — falling back to the "
                "wrapped element (weights may be corrupt)")
        if abs(np.linalg.det(M)) < 1e-9:
            raise OutOfScopeError(
                f"surrogate '{self.metadata.element_key}': NN produced "
                "a SINGULAR transfer matrix (|det|="
                f"{abs(np.linalg.det(M)):.3g}) — falling back to the "
                "wrapped element (weights may be corrupt)")
        return M

    def fitted_matrix_torch(self, kin_tensor: torch.Tensor) -> torch.Tensor:
        """Autograd-differentiable 6x6 transfer matrix (M8 matcher arm).

        Parameters
        ----------
        kin_tensor : torch.Tensor, shape (3,), dtype F64
            ``[w_kin, beta, gamma]`` of the reference particle.  May
            carry ``requires_grad=True`` so the matcher's chain rule
            connects (today the matcher tunes upstream linear knobs;
            surrogated cavities are passive blocks the gradient flows
            through).

        Returns
        -------
        torch.Tensor, shape (6, 6), dtype F64
            Same MLP output as :meth:`fitted_matrix` but with the
            autograd graph intact -- no ``torch.no_grad``.  Element
            params (``ke``, ``kb``, ``phase``, ...) are read once from
            the wrapped element as constants in this iteration; future
            work can lift them to differentiable overrides if matching
            ever needs to tune them directly.

        Notes
        -----
        Caller is responsible for the OOD scope check -- the matcher
        arm in :mod:`linac_gen.tracking.torch_tracking` does this
        before invoking, raising ``ValueError`` (hard error, never identity)
        when the surrogate would refuse.  This method itself never
        raises ``OutOfScopeError``; it computes whatever the MLP
        gives, which is what the matcher's gradient path needs.
        """
        params = torch.as_tensor(
            [float(getattr(self._wrapped, p)) for p in self._param_names],
            dtype=F64,
        )
        x = torch.cat([kin_tensor, params])
        x_norm = (x - self._in_mean) / self._in_std
        # NO torch.no_grad -- autograd graph stays connected.
        y_norm = self.mlp(x_norm)
        y = y_norm * self._out_std + self._out_mean
        return y.reshape(6, 6)

    def fitted_matrix_slice(self, ref, ds_mm: float) -> np.ndarray:
        """Surrogate-NN slice when transporting the FULL element;
        delegate to wrapped RK4 for partial slices.

        Why the split:

        * **Full-element transport** (``ds_mm == self.length``): the
          surrogate's ``fitted_matrix(ref)`` IS the trained NN
          prediction for the whole element.  Returning it here gives
          the order-of-magnitude speedup the user expected when
          ticking Use, with no accuracy loss vs the wrapped RK4
          ``fitted_matrix_slice`` evaluated over the same full
          element (both predict the same M_full, surrogate via NN +
          training data, wrapped via direct RK4 integration).
        * **Partial slice** (``ds_mm < self.length``, e.g. an envelope
          SC bundle that splits the element to apply SC kicks every
          few substeps): the surrogate's ``M_full`` is the LINEAR
          end-to-end matrix.  Composing a sub-slice via
          ``expm(logm(M_full) × ratio)`` ONLY equals the true sub-slice
          when the per-substep dynamics is constant-coefficient
          (Drift / Quad / Solenoid).  For FieldMap RF cavities the
          field is z-varying ``sin(ωt)`` and ``expm(logm)`` gives a
          time-averaged approximation that diverges from RK4 -- this
          is what caused the env-with-surrogate vs env-without-
          surrogate result divergence in user testing.  RK4 is the
          honest answer for these partial slices; we delegate.

        Net effect for users:

        * **No-SC envelope** (any lattice) -- speedup, surrogate used.
        * **SC-active envelope on Drift/Quad/Solenoid-only lattice** --
          partial slices through CONSTANT-COEFF elements would
          actually be correct via logm/expm, but the wrapped
          fitted_matrix_slice on Drift/Quad is already analytical
          (not RK4) so the surrogate doesn't help anyway.
        * **SC-active envelope through FieldMap RF cavities** (the
          user's PIP-II HWR case) -- partial slices delegate to RK4,
          NO surrogate speedup.  Future work: train a slice-aware
          surrogate that takes ds_mm as an input to gain speedup
          here too.

        OOD safety: surrogate's ``fitted_matrix`` raises
        :class:`OutOfScopeError` for inputs outside the trained scope;
        we catch and fall back to RK4 in that case too.
        """
        if abs(float(ds_mm) - float(self.length)) < 1e-9:
            try:
                return self.fitted_matrix(ref)
            except OutOfScopeError:
                pass  # OOD -- fall through to wrapped RK4.
        # Partial slice OR OOD -- delegate to RK4 for accuracy.
        return self._wrapped.fitted_matrix_slice(ref, ds_mm)

    def _slice_matrix(self, ref, ds_mm: float) -> np.ndarray:
        """Matrix log/exp slice of the surrogate's full-element matrix.

        Used by the M7 hybrid MP path (and any other slice-aware
        caller that wants a surrogate-predicted slice).  Mathematics:

            M_slice = expm(logm(M_full) * (ds / L))

        Slices for any partition of the element compose to M_full by
        construction.  Valid under the smooth-dynamics assumption;
        falls back to the wrapped element's RK4 slice if the matrix
        log isn't real (very rare for symplectic transport).

        ``scipy.linalg.logm + expm`` cost ~5-10 ms each on a 6x6 (an
        eigendecomposition + reconstruction), so a per-substep call
        across ~3000 substeps dominates the run.  The result is
        constant per ``(ds, ref.w_kin)`` within one element's inner
        loop, so cache the most recent and reuse if the inputs match.
        """
        from scipy.linalg import expm, logm
        # Cache key: (ds, w_kin rounded to 1e-9 MeV).  Within an SC
        # bundle the reference doesn't change between same-ds calls;
        # across elements both ds and w_kin shift so the cache misses
        # naturally.
        key = (round(float(ds_mm), 9), round(float(ref.w_kin), 9))
        if self._slice_cache_key == key and self._slice_cache_M is not None:
            return self._slice_cache_M

        try:
            if abs(float(ds_mm) - float(self.length)) < 1e-9:
                M = self.fitted_matrix(ref)
            else:
                M_full = self.fitted_matrix(ref)
                try:
                    log_M = logm(M_full)
                except Exception:
                    # Pathological matrix -- can't take a real log.
                    return self._wrapped.fitted_matrix_slice(ref, ds_mm)
                ratio = float(ds_mm) / max(float(self.length), 1e-12)
                M = np.real(expm(log_M * ratio))
        except OutOfScopeError:
            # OOD (or non-finite NN output): the wrapped element is
            # always the safe answer — never abort the caller's run
            # over a matrix that would not have been used.
            return self._wrapped.fitted_matrix_slice(ref, ds_mm)

        self._slice_cache_key = key
        self._slice_cache_M = M
        return M

    # ------------------------------------------------------------------
    # Multi-particle contract (M7 + M7-followup)
    #
    # Two opt-in gates from the registry control behaviour here:
    #   - is_mp_enabled()       (M7)       -- if False, delegate to
    #                                          wrapped (bit-identical
    #                                          to baseline).
    #   - is_fast_path_enabled() (M7-followup) -- if False (default),
    #                                          delegate.  If True, run
    #                                          the linear-matrix fast
    #                                          path (analytic ref
    #                                          advance + batched M @
    #                                          particles, no per-substep
    #                                          wrapped.track_rk4 call).
    # The fast path also falls back to the wrapped element on any
    # error (logm failure, OOD, missing helpers on the wrapped) so
    # the run is always physically defensible.
    # ------------------------------------------------------------------
    def _init_fast_path(self, ref) -> None:
        """Element-entry initialisation for the fast path.

        Runs once per element traversal (cleared by reset_run_state).
        Predicts M_full, caches logm(M_full), triggers the wrapped's
        SET_SYNC_PHASE calibration so per-substep `_phi_sync_rad`
        returns the right thing, and caches the electric-channel list.
        """
        from scipy.linalg import logm
        try:
            self._M_full = self.fitted_matrix(ref)
        except OutOfScopeError:
            self._fast_init_failed = True
            return
        try:
            self._log_M_full = logm(self._M_full)
            # Reject if logm returned complex with non-trivial imag.
            if np.max(np.abs(np.imag(self._log_M_full))) > 1e-6:
                # Real-log fallback: use the real part and accept
                # small per-slice unitarity error.
                self._log_M_full = np.real(self._log_M_full)
        except Exception:
            self._fast_init_failed = True
            return

        wrapped = self._wrapped
        # Trigger the cavity-entry sync-phase calibration so
        # `_phi_sync_rad` returns the right value under p_flag == 1.
        if hasattr(wrapped, "_calibrate_sync_phase"):
            try:
                wrapped._calibrate_sync_phase(ref)
            except Exception:
                pass
        if hasattr(wrapped, "_step_idx"):
            wrapped._step_idx = 0
        if hasattr(wrapped, "_phi_s_at_entrance"):
            wrapped._phi_s_at_entrance = ref.phi_s
        # NOTE: no lazy interpolator build is needed — FieldMap3D creates
        # ``_interpolators`` eagerly in __init__ via
        # ``_build_channel_interpolators(field_data)``.  (An old
        # ``hasattr(wrapped, "_build_interpolators")`` guard here matched a
        # method that never existed, so it was permanently-dead code.)
        # Cache electric-channel list for the per-substep dW_ref sum.
        interps = getattr(wrapped, "_interpolators", None)
        if interps is None:
            self._fast_init_failed = True
            return
        self._electric_channels = [
            (ch_enum, comp_interps)
            for ch_enum, comp_interps in interps.items()
            if getattr(ch_enum, "is_electric", False)
        ]
        # Reset the slice cache so the entry's M_slice is computed
        # fresh -- prevents bleed-through from a previous element.
        self._slice_cache_key = None
        self._slice_cache_M = None

    def _get_cached_slice_from_log(self, ds: float, ref_w_kin: float) -> np.ndarray:
        """Like `_slice_matrix` but uses the cached log_M_full.

        Skips the per-substep `logm` (the expensive part); only the
        much cheaper `expm` runs on a cache miss.
        """
        from scipy.linalg import expm
        key = (round(float(ds), 9), round(float(ref_w_kin), 9))
        if self._slice_cache_key == key and self._slice_cache_M is not None:
            return self._slice_cache_M
        if abs(float(ds) - float(self.length)) < 1e-9:
            M = self._M_full
        else:
            ratio = float(ds) / max(float(self.length), 1e-12)
            M = np.real(expm(self._log_M_full * ratio))
        self._slice_cache_key = key
        self._slice_cache_M = M
        return M

    def track_rk4(self, beam, ds: float) -> None:
        """MP push -- safe delegate (M7) or fast path (M7-followup).

        Gating (cheap checks, top to bottom):

          * surrogate disabled                  -> delegate
          * MP-engagement (registry) off        -> delegate
          * fast-path (registry) off            -> delegate (safe M7)
          * ref OOD                             -> delegate
          * fast init failed earlier            -> delegate
          * wrapped lacks required helpers      -> delegate (defensive)

        If all checks pass, run the linear-matrix fast path:
        analytic ref advance using wrapped's helpers + batched
        M_slice @ alive particles.
        """
        if not self._enabled:
            self._wrapped.track_rk4(beam, ds)
            return
        from linac_gen.surrogates import registry as _reg
        if not _reg.is_mp_enabled() or not _reg.is_fast_path_enabled():
            self._wrapped.track_rk4(beam, ds)
            return
        # Cheap ref-only scope check (per-particle would be too slow).
        x = self._make_input(beam.ref)
        if not self.metadata.scope.is_in_scope(x):
            self._wrapped.track_rk4(beam, ds)
            return

        # First call within this element traversal: initialise.
        if not self._fast_init_done:
            self._init_fast_path(beam.ref)
            self._fast_init_done = True
        if self._fast_init_failed:
            self._wrapped.track_rk4(beam, ds)
            return

        wrapped = self._wrapped
        # Defensive: the fast path relies on these private helpers.
        # If the wrapped class doesn't expose them, fall back.
        for attr in ("_z_map_start", "_step_idx", "_phi_sync_rad",
                     "_phasor", "_sample_onaxis", "_scale_factor",
                     "field_data"):
            if not hasattr(wrapped, attr):
                self._wrapped.track_rk4(beam, ds)
                return

        # ---- Per-substep ref advance (analytic) -------------------
        ref = beam.ref
        z_pos = (wrapped._z_map_start + wrapped._step_idx * ds + ds / 2.0)
        wrapped._step_idx += 1

        if ref.wavelength > 0:
            dphi_to_mid = 180.0 * ds / (ref.beta * ref.wavelength)
        else:
            dphi_to_mid = 0.0
        phi_s_mid = ref.phi_s + dphi_to_mid
        try:
            phi_sync_rad = wrapped._phi_sync_rad(ref, phi_s_mid)
        except Exception:
            # Calibration not ready, etc.  Roll back the _step_idx
            # bump and fall back.
            wrapped._step_idx -= 1
            self._wrapped.track_rk4(beam, ds)
            return

        dW_ref = 0.0
        ds_m = ds * 1e-3
        for ch_enum, comp_interps in self._electric_channels:
            ch_data = wrapped.field_data.channels[ch_enum]
            Ez_ax = wrapped._sample_onaxis(ch_enum, comp_interps, z_pos)
            phasor = float(wrapped._phasor(
                ch_enum, np.array([phi_sync_rad]))[0])
            amp = wrapped._scale_factor(ch_enum, ch_data)
            dW_ref += ref.species.charge * Ez_ax * amp * phasor * ds_m

        ref.w_kin += dW_ref
        ref.s += ds
        if ref.wavelength > 0:
            ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

        # ---- Per-substep particle push (batched matmul) -----------
        alive = beam.alive_mask
        if alive.any():
            M_slice = self._get_cached_slice_from_log(ds, ref.w_kin)
            beam.particles[alive] = (M_slice @ beam.particles[alive].T).T

    def advance_ref(self, ref) -> None:
        """Delegate reference-particle advance to the wrapped element.

        Without this, a directly-substituted surrogate inherited the
        FieldMapElement no-op placeholder — the reference energy AND s
        froze across the element, silently corrupting every downstream
        map (latent hazard; the registry route never hit it because it
        advances via the native lattice element)."""
        self._wrapped.advance_ref(ref)

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        """Delegate sub-range reference advance (envelope SC sub-steps)
        to the wrapped element."""
        self._wrapped.advance_ref_over(ref, z_from_mm, z_to_mm)

    def reset_run_state(self) -> None:
        super().reset_run_state()
        if hasattr(self._wrapped, "reset_run_state"):
            self._wrapped.reset_run_state()
        # M7-followup fast-path state: each element traversal starts
        # fresh, so the next _init_fast_path() re-runs the sync-phase
        # calibration + M_full prediction against the current ref
        # state (not whatever the previous element left behind).
        self._fast_init_done = False
        self._fast_init_failed = False
        self._M_full = None
        self._log_M_full = None
        self._electric_channels = None
        self._slice_cache_key = None
        self._slice_cache_M = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_input(self, ref) -> np.ndarray:
        """Build the MLP input vector from (ref kinematics + element params).

        Convention: [w_kin (MeV), beta, gamma, *param_values].
        ``param_values`` are read by name from the wrapped element
        (see ``self._param_names``).
        """
        params = [float(getattr(self._wrapped, p)) for p in self._param_names]
        return np.array(
            [float(ref.w_kin), float(ref.beta), float(ref.gamma)] + params,
            dtype=np.float64,
        )
