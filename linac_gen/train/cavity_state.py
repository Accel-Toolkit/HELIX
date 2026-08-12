"""Cavity parameters and per-train mutable state (multibunch study).

Static parameters (R/Q, loaded Q, detuning, design voltage/phase) come
from a SIDECAR file matched to elements by name pattern — physics inputs
are explicit, never silently defaulted (plan 3b).  Mutable phasor STATE
lives here, external to the elements, keyed by (element_index, name):
element attributes would be forked by the error model's lattice deepcopy,
and reset_run_state's contract does not cover train state.

Conventions (anchored by tests/train/test_beam_loading.py):
  * R/Q is the LINAC (accelerator) convention: R/Q = V^2 / (omega U)
    [Ohm], so a point charge q induces |dV| = omega (R/Q) q / 2 per
    passage and loses W_self = q^2 omega (R/Q) / 4 to the mode
    (fundamental theorem of beam loading: a bunch sees HALF its own
    induced voltage).
  * Rotating frame at the cavity RF frequency, zero phase = the DESIGN
    synchronous arrival.  The generator holds V_design (real axis).  A
    bunch arriving with cos-argument phi_s induces
    dV_b = -(omega (R/Q) q / 2) e^{-i phi_s}; the phasor decays and
    rotates between arrivals as e^{-dt/tau} e^{i 2 pi df dt} with
    tau = 2 Q_L / omega.
  * The kick a bunch sees maps onto the existing FieldError slots via
    V_tot = V_design + V_b:  voltage_rel = |V_tot|/V_design - 1,
    phase_offset = arg(V_tot) [deg]  (Re[V e^{i phi}] = |V| cos(phi +
    arg V), matching effective_voltage * cos(effective_phase + dphi)).

Dipole-HOM state (M4) also lives on `_CavityState` (``hom`` /
``hom_last_slot`` + the design-pass records the fast centroid model
needs); the transverse R/Q convention and kick law are pinned in
linac_gen/train/hom.py and anchored by tests/train/test_hom.py.
"""
from __future__ import annotations

import fnmatch
import json
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CavityMode:
    """Fundamental-mode beam-loading parameters for one cavity, plus the
    cavity's dipole-HOM table (M4; ``hom.HOMMode`` entries).  When only
    the HOM channel is enabled the fundamental fields may be inert
    (0.0) — ``load_sidecar``'s ``need_fundamental``/``need_hom`` flags
    say which blocks are required physics inputs."""
    r_over_q: float                    # Ohm (linac convention)
    q_loaded: float
    detuning_Hz: float = 0.0
    v_design_MV: Optional[float] = None   # None -> derive from design pass
    phi_s_deg: Optional[float] = None     # None -> derive from the element
    hom_modes: list = field(default_factory=list)   # list[hom.HOMMode]


@dataclass
class _CavityState:
    mode: CavityMode
    frequency_MHz: float = 0.0
    v_design_MV: float = 0.0
    phi_s_deg: float = 0.0
    v_beam: complex = 0j               # MV, rotating frame
    last_slot: Optional[int] = None
    dw_design_MeV: float = 0.0
    # ---- dipole-HOM state (M4) — separate slot clock from the
    # fundamental's so BeamLoadingManager and HomManager can share one
    # registry without clobbering each other's decay bookkeeping.
    hom: list = field(default_factory=list)         # list[hom._HomState]
    hom_last_slot: Optional[int] = None
    # Design-pass records for the fast driver's centroid model: cavity
    # entry position, reference betagamma, and the design-beam centroid
    # (x mm, xp mrad, y mm, yp mrad) at entry.
    s_design_mm: float = float("nan")
    bg_design: float = float("nan")
    centroid_design: Optional[object] = None        # np.ndarray (4,)


class CavityStateRegistry:
    """Per-train registry of cavity beam-loading state."""

    def __init__(self):
        self._by_key: dict[tuple[int, str], _CavityState] = {}

    # ---- construction -------------------------------------------------
    @staticmethod
    def load_sidecar(path: str, *, need_fundamental: bool = True,
                     need_hom: bool = False) -> dict[str, CavityMode]:
        """Read {name_pattern: {r_over_q, q_loaded, ..., hom_modes}} from
        JSON (or YAML when PyYAML is available).

        Physics inputs are never silently defaulted (plan 3b), so which
        blocks are REQUIRED follows the enabled channels:
          * ``need_fundamental`` (beam_loading on): every entry must
            carry r_over_q + q_loaded — the M2 contract, unchanged.
          * ``need_hom`` (hom on): every entry must carry a NON-EMPTY
            ``hom_modes`` list [{f_MHz, r_over_q_t, q_loaded[,
            polarization_deg]}, ...] — an empty or missing table under
            hom physics is a contradiction and refused loudly.
        Blocks that are present are parsed strictly either way; absent
        non-required fundamental fields become inert (0.0, never used).
        """
        text = open(path).read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                raw = yaml.safe_load(text)
            except ImportError as exc:
                raise ValueError(
                    f"{path}: YAML sidecar but PyYAML is not installed — "
                    "use JSON") from exc
        else:
            raw = json.loads(text)
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"{path}: sidecar must be a non-empty mapping")
        from linac_gen.train.hom import HOMMode
        out = {}
        for pat, d in raw.items():
            if need_fundamental:
                missing = [k for k in ("r_over_q", "q_loaded") if k not in d]
                if missing:
                    raise ValueError(
                        f"{path}: cavity entry {pat!r} missing required "
                        f"key(s): {', '.join(missing)}")
            hom_raw = d.get("hom_modes", None)
            if need_hom and not hom_raw:
                raise ValueError(
                    f"{path}: cavity entry {pat!r}: physics.hom is enabled "
                    "but 'hom_modes' is "
                    + ("empty" if hom_raw is not None else "missing")
                    + " — declare the dipole-mode table [{f_MHz, "
                    "r_over_q_t, q_loaded[, polarization_deg]}, ...] or "
                    "disable the hom channel")
            hom_modes = []
            for i, h in enumerate(hom_raw or ()):
                try:
                    hom_modes.append(HOMMode.from_dict(h))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}: cavity entry {pat!r} hom_modes[{i}]: "
                        f"{exc}") from exc
            out[pat] = CavityMode(
                r_over_q=float(d.get("r_over_q", 0.0)),
                q_loaded=float(d.get("q_loaded", 0.0)),
                detuning_Hz=float(d.get("detuning_Hz", 0.0)),
                v_design_MV=(float(d["v_design_MV"])
                             if "v_design_MV" in d else None),
                phi_s_deg=(float(d["phi_s_deg"])
                           if "phi_s_deg" in d else None),
                hom_modes=hom_modes,
            )
        return out

    def bind(self, lattice, modes_by_pattern: dict[str, CavityMode]) -> int:
        """Match sidecar patterns to lattice cavities by element name."""
        n = 0
        for idx, el in enumerate(lattice.elements):
            name = getattr(el, "name", "")
            freq = float(getattr(el, "frequency", 0.0)
                         or getattr(el, "frequency_mhz", 0.0) or 0.0)
            for pat, mode in modes_by_pattern.items():
                if fnmatch.fnmatch(name, pat):
                    if freq <= 0.0:
                        raise ValueError(
                            f"cavity {name!r} matched pattern {pat!r} but "
                            "has no RF frequency — beam loading undefined")
                    st = _CavityState(
                        mode=mode, frequency_MHz=freq,
                        phi_s_deg=(mode.phi_s_deg
                                   if mode.phi_s_deg is not None
                                   else float("nan")))
                    if mode.hom_modes:
                        from linac_gen.train.hom import _HomState
                        st.hom = [_HomState(mode=m) for m in mode.hom_modes]
                    self._by_key[(idx, name)] = st
                    n += 1
                    break
        return n

    # ---- access -------------------------------------------------------
    def get(self, idx: int, name: str) -> Optional[_CavityState]:
        return self._by_key.get((idx, name))

    def items(self):
        return self._by_key.items()

    def __len__(self):
        return len(self._by_key)

    # ---- physics helpers ----------------------------------------------
    @staticmethod
    def tau_s(st: _CavityState) -> float:
        omega = 2.0 * math.pi * st.frequency_MHz * 1e6
        return 2.0 * st.mode.q_loaded / omega

    @staticmethod
    def decay(st: _CavityState, dt_s: float) -> None:
        """Advance the beam-induced phasor by dt (decay + detuning)."""
        if dt_s <= 0.0 or st.v_beam == 0j:
            return
        tau = CavityStateRegistry.tau_s(st)
        rot = 2.0 * math.pi * st.mode.detuning_Hz * dt_s
        st.v_beam *= math.exp(-dt_s / tau) * complex(math.cos(rot),
                                                     math.sin(rot))

    @staticmethod
    def induced_dv_MV(st: _CavityState, charge_C: float) -> complex:
        """Bunch-induced voltage phasor (MV), rotating frame."""
        omega = 2.0 * math.pi * st.frequency_MHz * 1e6
        mag_V = omega * st.mode.r_over_q * charge_C / 2.0
        phi = math.radians(st.phi_s_deg)
        return -(mag_V * 1e-6) * complex(math.cos(-phi), math.sin(-phi))
