"""Differentiable matcher objective for the ``gradient`` algorithm.

Reproduces the SET_TWISS / SET_SIZE matcher residuals through the
differentiable (PyTorch) matrix-tracking engine, so the matcher can
optimise with an exact autograd Jacobian instead of finite differences.

Scope — the ``gradient`` algorithm supports:

* variables tuning a Quadrupole ``gradient``, Solenoid ``field`` or
  Dipole ``angle``;
* ``SET_TWISS`` / ``SET_SIZE`` constraints (end-of-lattice equality —
  the most common matching constraints), plus the zero-residual stub
  cards (``SET_ACHROMAT`` etc.);
* lattices whose every element is linear (drift / quad / solenoid /
  dipole / edge) or a pure no-op (markers, apertures, command cards);
* matching **without** space charge.

Anything outside this — ``SET_SIZE_MAX`` / ``SET_SIZE_MIN`` /
``SET_BEAM_PHASE_ADV``, centroid constraints (``SET_POSITION`` /
``DIAG_POSITION`` targets — the torch mirror propagates Σ only, no
first moment, so it cannot see steerer kicks or misalignment
feed-down), longitudinal ``SET_TWISS`` flags (kaz/kbz — no tracking
mode records longitudinal Twiss), runtime-active control cards that
mutate the reference kinematics (``FREQ`` / ``SET_BEAM_ENERGY`` /
``SET_BEAM_E0_P0`` — the mirror composes commands as identity), RF /
field-map elements, beam-Twiss variables, space charge — raises a
clear :class:`ValueError` telling the caller to use
``least_squares``.  The matcher additionally self-validates the
torch residual against the numpy residual at ``x0``.
"""
from __future__ import annotations

import numpy as np
import torch

from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.edge import Edge
from linac_gen.elements.lattice_commands import (
    Freq, LatticeCommand, SetAchromat, SetAdv, SetBeamE0P0,
    SetBeamEnergy, SetBeamPhaseAdv, SetPosition, SetSeparation, SetSize,
    SetSizeMax, SetSizeMin, SetTwiss,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.tracking.envelope import _build_sigma_matrix
from linac_gen.tracking.matrix_tracking import _compute_element_matrix
from linac_gen.tracking.torch_matrices import F64
from linac_gen.tracking.torch_step_tracker import track_beam_torch_stepwise
from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch

__all__ = ["check_gradient_supported", "build_torch_residual",
           "build_torch_residual_sc"]

# The 5 element types the differentiable engine can transfer-map.
_LINEAR_TYPES = (Drift, Quadrupole, Solenoid, Dipole, Edge)
# Variable target type -> the one attribute the engine differentiates.
_TUNABLE_ATTR = {Quadrupole: "gradient", Solenoid: "field", Dipole: "angle"}
# Constraint source types whose residual the torch path reproduces.
# SET_POSITION is NOT here: since the envelope gained a real first
# moment (2026-07-19) its numpy residual is no longer a zero stub, and
# the torch mirror tracks Σ only — it cannot reproduce centroid
# residuals (no constant steerer kicks, no misalignment feed-down).
_SUPPORTED_CONSTRAINTS = (SetTwiss, SetSize,
                          SetAchromat, SetSeparation, SetAdv)


# ---------------------------------------------------------------------------
# Support check
# ---------------------------------------------------------------------------
def check_gradient_supported(lattice, ref, variables, constraints) -> None:
    """Raise :class:`ValueError` if the ``gradient`` algorithm cannot
    represent this matching problem exactly."""
    # --- variables: quad gradient / solenoid field / dipole angle only ---
    for var in variables:
        want = _TUNABLE_ATTR.get(type(var.target))
        if want is None or var.attr != want:
            raise ValueError(
                f"the 'gradient' algorithm tunes Quadrupole gradient, "
                f"Solenoid field or Dipole angle only; variable "
                f"'{var.label}' is not one of these — use 'least_squares'."
            )

    # --- constraints: SET_TWISS / SET_SIZE (+ zero stubs) only ---
    for c in constraints:
        src = c.source
        if isinstance(src, (SetSizeMax, SetSizeMin, SetBeamPhaseAdv)):
            raise ValueError(
                f"the 'gradient' algorithm supports SET_TWISS and SET_SIZE "
                f"constraints; this lattice has {type(src).__name__} "
                f"('{c.label}') — use 'least_squares'."
            )
        if (isinstance(src, SetSize)
                and getattr(src, "phi_or_z", 0.0) < 0):
            # The numpy evaluator honours the negative-operand σ_z(mm)
            # form (2026-07 review round); the torch mirror does not —
            # refuse rather than silently drop the target (SC path has
            # no self-validation to catch the divergence).
            raise ValueError(
                f"the 'gradient' algorithm does not implement the "
                f"SET_SIZE negative-operand σ_z(mm) form "
                f"('{c.label}') — use 'least_squares'."
            )
        if isinstance(src, SetPosition) or c.label.startswith(
                ("SET_POSITION", "DIAG_POSITION")):
            raise ValueError(
                f"the 'gradient' algorithm cannot enforce centroid "
                f"constraint '{c.label}': its torch mirror propagates "
                f"the envelope only, not the first moment — use "
                f"'least_squares', 'bo', 'cmaes' or 'sequential_scan'."
            )
        if isinstance(src, SetTwiss) and (src.kaz == 1 or src.kbz == 1):
            # Longitudinal Twiss IS recorded (and honoured by the numpy
            # evaluator) since 2026-07, but this torch mirror does not
            # implement the longitudinal SET_TWISS residual (its z-block
            # Twiss would also need the FREQ-jump rescale verified on
            # the torch Σ propagation).  Refuse rather than optimize a
            # problem missing a requested axis.
            raise ValueError(
                f"the 'gradient' algorithm's torch mirror does not "
                f"implement the longitudinal flags (kaz/kbz) of "
                f"'{c.label}' — use 'least_squares' (which honours "
                f"them) or the transverse flags only."
            )
        if not isinstance(src, _SUPPORTED_CONSTRAINTS):
            raise ValueError(
                f"the 'gradient' algorithm cannot represent constraint "
                f"'{c.label}' — use 'least_squares'."
            )

    # --- elements: every element must be linear or a pure no-op ---
    eye = np.eye(6)
    # M8 matcher arm: SurrogateFieldMap exposes an autograd-differentiable
    # 6x6 via `fitted_matrix_torch(kin_tensor)`; accept it even though it
    # subclasses FieldMapElement (which would otherwise fail the linear
    # check below).  The `element_matrix_torch` arm in
    # `linac_gen/tracking/torch_tracking.py` handles the per-element
    # gradient path; OOD inputs raise (hard error, never identity).
    from linac_gen.surrogates.base import SurrogateFieldMap   # lazy
    # Running machine frequency: a FREQ card whose value EQUALS the
    # incoming frequency is a no-op (jump ratio 1 — the freq-jump D
    # matrix is the identity), so the torch identity composition is
    # exactly faithful.  This exempts the header FREQ that opens
    # virtually every real TraceWin deck; only a genuine mid-lattice
    # frequency CHANGE is refused.
    cur_freq = float(getattr(ref, "frequency", 0.0) or 0.0)
    for el in lattice.elements:
        if isinstance(el, Freq):
            f_new = float(getattr(el, "frequency_mhz", 0.0) or 0.0)
            if abs(f_new - cur_freq) <= 1e-9 * max(1.0, abs(cur_freq)):
                continue                       # ratio-1 no-op: faithful
        if isinstance(el, (Freq, SetBeamEnergy, SetBeamE0P0)):
            # The torch mirror composes every LatticeCommand as IDENTITY
            # — faithful for the passive SET_*/ADJUST_* markers, but
            # these three mutate the reference frequency/energy at the
            # card, changing downstream focusing (K ∝ 1/Bρ) and the
            # longitudinal coordinate.  Silently optimizing without
            # them means solving a different lattice (PRAB review
            # finding: FREQ + gradient + SC ran with zero warnings).
            raise ValueError(
                f"the 'gradient' algorithm cannot represent the "
                f"runtime-active card {el.KEYWORD} ('{el.name}'): the "
                f"torch mirror treats command cards as identity, but "
                f"this card changes the reference kinematics mid-"
                f"lattice — use 'least_squares'."
            )
        if isinstance(el, (_LINEAR_TYPES, LatticeCommand)):
            # Passive constraint/variable markers, plus the remaining
            # active commands (SET_SYNC_PHASE, SET_BEAM_PHASE_ERROR,
            # SET_GAUSSIAN_CUT_OFF) which act only through RF cavities
            # or the error study — both unreachable here (the linear-
            # element check below refuses cavities), so identity is
            # faithful for them.
            continue
        if isinstance(el, SurrogateFieldMap):
            # M8: autograd-differentiable — but only faithful for
            # NON-ACCELERATING maps (solenoids, ke = 0).  The torch
            # composition holds the reference energy fixed, so a
            # surrogate wrapping an accelerating cavity would silently
            # produce wrong downstream optics.  Gate on the wrapped
            # element's electric amplitude; no wrapped element = refuse
            # loudly rather than guess.
            wrapped = getattr(el, "_wrapped", None)
            ke = getattr(wrapped, "ke", None) if wrapped is not None else None
            if ke is None or abs(float(ke)) > 1e-12:
                raise ValueError(
                    f"the 'gradient' algorithm accepts only "
                    f"non-accelerating surrogates; '{el.name}' wraps "
                    f"{'an unknown element' if ke is None else f'a cavity (ke={float(ke):g})'}"
                    f" whose energy gain the torch composition cannot "
                    f"model — use 'least_squares'."
                )
            continue
        try:
            M = _compute_element_matrix(el, ref.copy())
        except Exception:                                  # noqa: BLE001
            M = None
        if M is None or not np.allclose(M, eye, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"the 'gradient' algorithm needs an all-linear lattice; "
                f"element '{getattr(el, 'name', '?')}' "
                f"({type(el).__name__}) carries dynamics it cannot "
                f"differentiate — use 'least_squares'."
            )


# ---------------------------------------------------------------------------
# Torch residual construction
# ---------------------------------------------------------------------------
def _twiss_torch(sigma: torch.Tensor, i: int, j: int):
    """(alpha, beta) of a 2x2 sigma sub-block — mirrors envelope._sigma_to_twiss."""
    s11, s12, s22 = sigma[i, i], sigma[i, j], sigma[j, j]
    emit = torch.sqrt(torch.clamp(s11 * s22 - s12 * s12, min=0.0))
    return -s12 / emit, s11 / emit


def _constraint_residual_torch(constraint, sigma: torch.Tensor) -> torch.Tensor:
    """Residual sub-vector for one constraint, from the end-of-lattice
    sigma matrix — the differentiable mirror of the numpy evaluators in
    ``linac_gen.matching.constraints``."""
    src = constraint.source
    w = float(constraint.weight)

    if isinstance(src, SetTwiss):
        if src.kaz == 1 or src.kbz == 1:
            # Defense in depth: check_gradient_supported refuses these
            # — a longitudinal flag here would be a zero-gradient
            # constant the optimizer can never act on.
            raise ValueError(
                "SET_TWISS kaz/kbz reached the torch residual — "
                "check_gradient_supported should have refused them."
            )
        ax, bx = _twiss_torch(sigma, 0, 1)
        ay, by = _twiss_torch(sigma, 2, 3)
        twiss = {"alpha_x": ax, "beta_x": bx, "alpha_y": ay, "beta_y": by}
        target = {
            "alpha_x": src.alpha_x, "beta_x": src.beta_x,
            "alpha_y": src.alpha_y, "beta_y": src.beta_y,
        }
        flags = {
            "alpha_x": src.kax, "beta_x": src.kbx,
            "alpha_y": src.kay, "beta_y": src.kby,
        }
        out = [twiss[k] - target[k] for k, f in flags.items() if f == 1]
        r = torch.stack(out) if out else torch.zeros(1, dtype=F64)
        return r * w

    if isinstance(src, SetSize):
        out = []
        if src.x_mm > 0:
            out.append(torch.sqrt(torch.clamp(sigma[0, 0], min=0.0)) - src.x_mm)
        if src.y_mm > 0:
            out.append(torch.sqrt(torch.clamp(sigma[2, 2], min=0.0)) - src.y_mm)
        if src.phi_or_z > 0:
            out.append(torch.sqrt(torch.clamp(sigma[4, 4], min=0.0))
                       - src.phi_or_z)
        r = torch.stack(out) if out else torch.zeros(1, dtype=F64)
        return r * w

    if isinstance(src, SetPosition):
        # Defense in depth: check_gradient_supported refuses these
        # up front — the numpy residual is a real 4-vector the Σ-only
        # torch mirror cannot reproduce.
        raise ValueError(
            "SET_POSITION reached the torch residual — "
            "check_gradient_supported should have refused it."
        )

    # SetAchromat / SetSeparation / SetAdv — numpy stub: a single zero.
    return torch.zeros(1, dtype=F64)


def build_torch_residual(lattice, beam_cfg, ref, variables, constraints,
                         col_for_var, n_cols):
    """Return a pure torch function ``residual(x) -> residual_vector``.

    ``x`` is the optimiser column vector (length ``n_cols``).  The
    returned tensor concatenates the per-constraint residuals in the
    same order as the numpy matcher, but is autograd-differentiable
    with respect to ``x`` — so ``torch.autograd.functional.jacobian``
    yields the exact matcher Jacobian.

    Call :func:`check_gradient_supported` first.
    """
    # Initial beam sigma — constant; built exactly as engine._run_envelope
    # (mismatch-scaled geometric emittances via the shared helper).
    from linac_gen.distributions.factory import geometric_emittances
    _ex, _ey, _ez = geometric_emittances(beam_cfg, max(float(ref.bg), 1e-9))
    sigma_in = torch.as_tensor(
        _build_sigma_matrix(
            beam_cfg.alpha_x, beam_cfg.beta_x, _ex,
            beam_cfg.alpha_y, beam_cfg.beta_y, _ey,
            beam_cfg.alpha_z, beam_cfg.beta_z, _ez,
        ),
        dtype=F64,
    )
    var_target_ids = [id(v.target) for v in variables]

    def residual(x: torch.Tensor) -> torch.Tensor:
        # Inject the optimiser column into the tunable elements.
        overrides = {var_target_ids[i]: x[col_for_var[i]]
                     for i in range(len(variables))}
        M = compute_transfer_matrix_torch(
            lattice, ref, overrides=overrides, on_nonlinear="ignore")
        sigma_out = M @ sigma_in @ M.T
        chunks = [_constraint_residual_torch(c, sigma_out)
                  for c in constraints]
        if not chunks:
            return torch.zeros(1, dtype=F64)
        return torch.cat(chunks)

    return residual


# ---------------------------------------------------------------------------
# Space-charge-aware residual — gradient matching through non-linear PIC SC
# ---------------------------------------------------------------------------
def _sample_bunch(sigma_in: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sample ``n`` macro-particles whose 6-D covariance is
    ``sigma_in + 1e-12·I`` — a whitened Gaussian recoloured through
    jittered Cholesky factors (the +1e-12·I keeps near-singular Σ
    factorable), deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 6))
    z = z - z.mean(axis=0, keepdims=True)
    # Whiten to identity covariance, then colour to sigma_in.
    cz = (z.T @ z) / n
    lz = np.linalg.cholesky(cz + 1e-12 * np.eye(6))
    ls = np.linalg.cholesky(sigma_in + 1e-12 * np.eye(6))
    return (z @ np.linalg.inv(lz).T) @ ls.T


def build_torch_residual_sc(lattice, beam_cfg, ref, variables, constraints,
                            col_for_var, n_cols, *, sc_cfg,
                            bunch_size: int = 1500, seed: int = 42):
    """Return a torch residual ``residual(x)`` that matches **through
    non-linear PIC space charge**.

    Same contract as :func:`build_torch_residual`, but the forward model
    is particle tracking with the differentiable PIC space-charge kick
    (:func:`linac_gen.tracking.torch_step_tracker.track_beam_torch_stepwise`)
    instead of a linear sigma-matrix transport. A fixed-seed macro-particle
    bunch keeps the residual — and its autograd Jacobian — deterministic.

    Raises
    ------
    ValueError
        If any element carries a misalignment (shift/roll/pitch/yaw):
        the stepwise tracker composes ideal-frame maps only and would
        silently DROP the misalignment from the physics (2026-07-25
        review, claim 8).  The torch matrix path (no-SC gradient) does
        apply tilt rotations; this limitation is specific to the SC
        step tracker.
    """
    for el in lattice.elements:
        if getattr(el, "is_misaligned", False):
            raise ValueError(
                f"gradient+SC matching cannot model misalignments: element "
                f"'{getattr(el, 'name', '?')}' carries a non-zero "
                "shift/roll/pitch/yaw which the stepwise torch tracker "
                "would silently ignore.  Remove the misalignment or use "
                "'least_squares' (the full tracker applies it)."
            )
    from linac_gen.distributions.factory import geometric_emittances
    _ex, _ey, _ez = geometric_emittances(beam_cfg, max(float(ref.bg), 1e-9))
    sigma_in = _build_sigma_matrix(
        beam_cfg.alpha_x, beam_cfg.beta_x, _ex,
        beam_cfg.alpha_y, beam_cfg.beta_y, _ey,
        beam_cfg.alpha_z, beam_cfg.beta_z, _ez,
    )
    bunch = torch.as_tensor(_sample_bunch(sigma_in, bunch_size, seed),
                            dtype=F64)
    # ENTRANCE snapshot: ref.frequency here (pre-tracking) IS the
    # bunch repetition frequency — exactly what Beam.bunch_frequency
    # freezes at creation.  Never read ref.frequency after tracking
    # starts: FREQ cards advance the RF clock, the bunch rate is fixed.
    from linac_gen.pic.macrocharge import macro_charge_coulombs
    bunch_frequency_mhz = float(
        getattr(beam_cfg, "bunch_frequency_MHz", 0.0) or 0.0) \
        or float(ref.frequency)
    macro_charge = macro_charge_coulombs(
        float(beam_cfg.current), bunch_frequency_mhz, bunch_size)
    var_target_ids = [id(v.target) for v in variables]

    def residual(x: torch.Tensor) -> torch.Tensor:
        overrides = {var_target_ids[i]: x[col_for_var[i]]
                     for i in range(len(variables))}
        # "ignore" (identity map, silent) — the same policy as the no-SC
        # residual above, and safe for the same reason: the pre-run
        # check_gradient_supported audit has already REFUSED every
        # element whose matrix is not the identity, so the only elements
        # this can ignore are markers/apertures/steerers the audit
        # certified as pure no-ops.  With "error" those audit-approved
        # elements aborted the match at the FIRST residual evaluation —
        # after the optimiser had started (2026-07-25 review, claim 7).
        tracked = track_beam_torch_stepwise(
            lattice, ref, bunch, overrides=overrides, sc_cfg=sc_cfg,
            macro_charge=macro_charge, on_nonlinear="ignore",
            checkpoint=True)
        centred = tracked - tracked.mean(dim=0, keepdim=True)
        sigma_out = (centred.T @ centred) / bunch.shape[0]
        chunks = [_constraint_residual_torch(c, sigma_out)
                  for c in constraints]
        if not chunks:
            return torch.zeros(1, dtype=F64)
        return torch.cat(chunks)

    return residual
