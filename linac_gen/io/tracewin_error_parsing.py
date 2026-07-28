"""Translate TraceWin ``ERROR_*`` directives into HELIX ``ErrorDef`` entries.

TraceWin's error commands are stateful: ``ERROR_QUAD_NCPL_STAT N r dx dy …``
applies to *the next N quadrupoles in the .dat stream*.  This helper holds
a small state machine that the main TraceWin parser drives:

* ``add_directive(keyword, params)`` — called on each ``ERROR_*`` line.
* ``on_element(element_type, element_name)`` — called whenever the parser
  emits a new element; matching pending directives consume one of their
  N counts and produce real ErrorDef entries that target this specific
  element's name.
* ``finalize(lattice)`` — assigns the accumulated ``ErrorDef`` /
  ``BeamErrorDef`` lists onto the lattice.

The mapping from TraceWin parameter slots to HELIX ``ErrorDef`` rows is:

* ``ERROR_QUAD_NCPL_STAT N r dx dy φx φy φz dG dG3 dG4 dG5 dG6 [Nb]``
    →  ErrorDef("Q1", "dx", ...), ErrorDef("Q1", "dy", ...),
       ErrorDef("Q1", "tilt_deg", ...), ErrorDef("Q1", "gradient_rel", ...),
       ErrorDef("Q1", "g3_rel", ...), …

* ``ERROR_CAV_NCPL_STAT N r dx dy φx φy E φ dz [Nb]``
    →  ErrorDef("CAV1", "dx", …), ErrorDef("CAV1", "tilt_deg", …),
       ErrorDef("CAV1", "voltage_rel", σ=E/100), ErrorDef("CAV1",
       "phase_offset", σ=φ°), ErrorDef("CAV1", "dz", ...)

* ``ERROR_BEND_NCPL_STAT N r dx dy φx φy φz dg dz``
    →  alignment + ``field_rel`` (dg%) + dz

Distribution code ``r`` (see ``_DIST_FROM_R`` — the authority):
    0 = TraceWin constant amplitude — NOT implemented; approximated as
        a truncated gaussian draw with σ = the declared value (warns)
    1 = uniform   (half_width=value, distribution="uniform")
    2 = gaussian  (sigma=value, distribution="gaussian")
    4, 5 = binary ±/∓ and 0/+ modes — coarsely approximated as gaussian

Beam errors (``ERROR_BEAM_STAT``) and the global cutoff /
ratio-sweep / file-loader directives are also handled here.
"""
from __future__ import annotations

import glob
import warnings
from dataclasses import dataclass, field
from typing import Optional

from linac_gen.errors.error_model import ErrorDef
from linac_gen.errors.beam_error import BeamErrorDef


# Distribution code from the TraceWin ``r`` parameter.
_DIST_FROM_R = {
    0: "gaussian",   # TraceWin r=0 = CONSTANT amplitude — HELIX has no
                     # constant mode; approximated as a gaussian draw
                     # (see the warning in _dist_from_r).
    1: "uniform",
    2: "gaussian",
    4: "gaussian",   # binary ±/∓ — coarse approximation
    5: "gaussian",   # binary 0/+ — coarse approximation
}


def _dist_from_r(r: int) -> str:
    """Resolve the TraceWin ``r`` distribution code — LOUDLY.

    The old ``_DIST_FROM_R.get(r, "gaussian")`` silently absorbed both
    the unimplemented constant mode (r=0) and unknown codes; every seed
    then drew a full gaussian where the .dat declared something else.
    """
    dist = _DIST_FROM_R.get(r)
    if dist is None:
        warnings.warn(
            f"ERROR_* distribution code r={r} is unknown — "
            "treating as gaussian.", stacklevel=3)
        return "gaussian"
    if r == 0:
        warnings.warn(
            "ERROR_* r=0 (constant amplitude) is not implemented — "
            "approximating with a truncated gaussian draw (σ = the "
            "declared value).  Error amplitudes will differ from "
            "TraceWin.", stacklevel=3)
    return dist


def _build_error_defs(pattern: str, dist: str, alignment_value_set,
                      field_value_pairs) -> list[ErrorDef]:
    """Helper: build the ErrorDef list for one quad/cav/bend element.

    ``alignment_value_set`` is a dict like ``{"dx": 0.1, "dy": 0.0,
    "tilt_deg": 0.0, "dz": 0.0}`` — only non-zero entries become
    ErrorDefs.

    ``field_value_pairs`` is a list of ``(param_name, value)`` for
    field errors (gradient_rel, voltage_rel, etc.).  Same convention.
    """
    out: list[ErrorDef] = []
    for name, val in alignment_value_set.items():
        if val == 0.0:
            continue
        kw = (dict(sigma=val) if dist == "gaussian"
              else dict(half_width=val))
        out.append(ErrorDef(pattern=pattern, parameter=name,
                            distribution=dist, **kw))
    for name, val in field_value_pairs:
        if val == 0.0:
            continue
        kw = (dict(sigma=val) if dist == "gaussian"
              else dict(half_width=val))
        out.append(ErrorDef(pattern=pattern, parameter=name,
                            distribution=dist, **kw))
    return out


# ---------------------------------------------------------------------------
@dataclass
class _PendingErrorDirective:
    """One ERROR_*_STAT/DYN waiting to consume the next N elements."""
    element_type: str            # "QUAD" | "CAV" | "BEND"
    remaining: int               # decrement on each consumption
    distribution: str            # "gaussian" | "uniform"
    alignment_values: dict       # {"dx": ..., "dy": ..., "tilt_deg": ..., "dz": ...}
    field_values: list           # [("gradient_rel", val), ("g3_rel", val), ...]


# ---------------------------------------------------------------------------
class TraceWinErrorState:
    """Per-parse-pass state machine for ERROR_* directives."""

    def __init__(self):
        self._pending: list[_PendingErrorDirective] = []
        self.errors: list[ErrorDef] = []
        self.beam_errors: list[BeamErrorDef] = []
        self.ratios: list[float] = []
        self.cutoff: float = 0.0

    # ----- element-side hook --------------------------------------------
    def on_element(self, element_type: str, element_name: str) -> None:
        """Called when the parser emits a new element.

        Walks all pending error directives that target this element's
        type and decrements their counts — producing concrete ErrorDef
        entries with this element's exact name as the pattern.
        """
        for pend in self._pending[:]:
            if pend.remaining <= 0 or pend.element_type != element_type:
                continue
            # glob.escape: the exact element name is used as an fnmatch
            # PATTERN downstream — a user-named element containing
            # ``*``/``?``/``[`` would otherwise never match itself and
            # its errors would silently no-op.
            self.errors.extend(_build_error_defs(
                glob.escape(element_name), pend.distribution,
                pend.alignment_values, pend.field_values,
            ))
            pend.remaining -= 1
            if pend.remaining <= 0:
                self._pending.remove(pend)

    # ----- directive-side hooks -----------------------------------------
    def add_quad_directive(self, params: list[float]) -> None:
        """ERROR_QUAD_NCPL_STAT N r dx dy φx φy φz dG dG3 dG4 dG5 dG6 [Nb]

        12 required slots + an optional 13th (``Nb``, random-subset
        count, TraceWin v2.16+).  ``Nb`` is parsed-and-ignored for now.
        """
        if len(params) < 12:
            warnings.warn(
                f"malformed ERROR_QUAD_* directive ({len(params)} numeric "
                "params, expected ≥ 12) — SKIPPED.  The quad tolerance "
                "family it declared is absent from the study.",
                stacklevel=3)
            return
        N = int(params[0])
        r = int(params[1])
        dx, dy, phix, phiy, phiz = params[2:7]
        dG, dG3, dG4, dG5, dG6 = params[7:12]
        if len(params) >= 13 and params[12] > 0:
            warnings.warn(
                "ERROR_QUAD Nb (random-subset count) is not implemented — "
                "errors are applied to ALL N matched elements.",
                stacklevel=3)
        if phix != 0.0 or phiy != 0.0:
            warnings.warn(
                "ERROR_QUAD φx/φy (pitch/yaw) are parsed but the tracker "
                "does not apply them — these tolerances have NO effect.",
                stacklevel=3)
        if N <= 0:
            return
        dist = _dist_from_r(r)
        alignment = {
            "dx": float(dx),
            "dy": float(dy),
            "tilt_deg": float(phiz),    # φz = roll about z
            # pitch/yaw (φx, φy) reserved but not honoured by the tracker yet
        }
        # TraceWin gradient errors are in %; convert to *_rel fractional.
        fields = [
            ("gradient_rel", float(dG) / 100.0),
            ("g3_rel",       float(dG3) / 100.0),
            ("g4_rel",       float(dG4) / 100.0),
            ("g5_rel",       float(dG5) / 100.0),
            ("g6_rel",       float(dG6) / 100.0),
        ]
        self._pending.append(_PendingErrorDirective(
            element_type="QUAD", remaining=N, distribution=dist,
            alignment_values=alignment, field_values=fields,
        ))

    def add_cav_directive(self, params: list[float]) -> None:
        """ERROR_CAV_NCPL_STAT N r dx dy φx φy E φ dz [Nb]"""
        if len(params) < 9:
            warnings.warn(
                f"malformed ERROR_CAV_* directive ({len(params)} numeric "
                "params, expected ≥ 9) — SKIPPED.  (Note: ERROR_CAV_*_FILE "
                "variants also land here — they are not implemented.)",
                stacklevel=3)
            return
        N = int(params[0])
        r = int(params[1])
        dx, dy, phix, phiy = params[2:6]
        E_pct, phi_deg, dz = params[6:9]
        if len(params) >= 10 and params[9] > 0:
            warnings.warn(
                "ERROR_CAV Nb (random-subset count) is not implemented — "
                "errors are applied to ALL N matched elements.",
                stacklevel=3)
        if phix != 0.0 or phiy != 0.0:
            warnings.warn(
                "ERROR_CAV φx/φy (pitch/yaw) are parsed but the tracker "
                "does not apply them — these tolerances have NO effect.",
                stacklevel=3)
        if dz != 0.0:
            warnings.warn(
                "ERROR_CAV dz is stored on the element but the tracker "
                "does not apply longitudinal displacements — this "
                "tolerance has NO effect.", stacklevel=3)
        if N <= 0:
            return
        dist = _dist_from_r(r)
        alignment = {
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            # φx, φy not honoured yet — store as tilt_deg only when
            # explicitly asked (TraceWin treats CAV-φz separately).
        }
        fields = [
            ("voltage_rel",  float(E_pct) / 100.0),
            ("phase_offset", float(phi_deg)),
        ]
        self._pending.append(_PendingErrorDirective(
            element_type="CAV", remaining=N, distribution=dist,
            alignment_values=alignment, field_values=fields,
        ))

    def add_bend_directive(self, params: list[float]) -> None:
        """ERROR_BEND_NCPL_STAT N r dx dy φx φy φz dg dz [Nb]

        9 required slots + an optional 10th (``Nb``, random-subset
        count, TraceWin v2.16+).  ``Nb`` is parsed-and-ignored, same
        as the quad/cav handlers — but it must WARN, not vanish.
        """
        if len(params) < 9:
            warnings.warn(
                f"malformed ERROR_BEND_* directive ({len(params)} numeric "
                "params, expected ≥ 9) — SKIPPED.",
                stacklevel=3)
            return
        N = int(params[0])
        r = int(params[1])
        dx, dy, phix, phiy, phiz = params[2:7]
        dg_pct, dz = params[7:9]
        if len(params) >= 10 and params[9] > 0:
            warnings.warn(
                "ERROR_BEND Nb (random-subset count) is not implemented — "
                "errors are applied to ALL N matched elements.",
                stacklevel=3)
        if phix != 0.0 or phiy != 0.0:
            warnings.warn(
                "ERROR_BEND φx/φy (pitch/yaw) are parsed but the tracker "
                "does not apply them — these tolerances have NO effect.",
                stacklevel=3)
        if dz != 0.0:
            warnings.warn(
                "ERROR_BEND dz is stored on the element but the tracker "
                "does not apply longitudinal displacements — this "
                "tolerance has NO effect.", stacklevel=3)
        if N <= 0:
            return
        dist = _dist_from_r(r)
        alignment = {
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "tilt_deg": float(phiz),
        }
        fields = [("field_rel", float(dg_pct) / 100.0)]
        self._pending.append(_PendingErrorDirective(
            element_type="BEND", remaining=N, distribution=dist,
            alignment_values=alignment, field_values=fields,
        ))

    def add_beam_directive(self, params: list[float]) -> None:
        """ERROR_BEAM_STAT r dx dy dφ dxp dyp de dEx dEy dEz mx my mz dIb …

        Only the first 14 slots are honoured here (centroid + emittance +
        mismatch + current).  The Twiss-α/β min/max ranges are TraceWin
        niche features — left for a follow-up.
        """
        if len(params) < 14:
            warnings.warn(
                f"malformed ERROR_BEAM_* directive ({len(params)} numeric "
                "params, expected ≥ 14) — SKIPPED.",
                stacklevel=3)
            return
        r = int(params[0])
        dist = _dist_from_r(r)
        # Centroid offsets
        dx, dy, dphi, dxp, dyp, de = params[1:7]
        # Emittance growth (%)
        dEx, dEy, dEz = params[7:10]
        # Mismatch (%)
        mx, my, mz = params[10:13]
        # Current jitter (mA)
        dIb = params[13]

        def _add(pname: str, val: float) -> None:
            if val == 0.0:
                return
            kw = (dict(sigma=val) if dist == "gaussian"
                  else dict(half_width=val))
            self.beam_errors.append(BeamErrorDef(
                parameter=pname, distribution=dist, **kw,
            ))

        _add("centroid_x",    float(dx))
        _add("centroid_y",    float(dy))
        _add("centroid_dphi", float(dphi))
        _add("centroid_xp",   float(dxp))
        _add("centroid_yp",   float(dyp))
        _add("centroid_dw",   float(de))
        _add("emit_nx_rel", float(dEx) / 100.0)
        _add("emit_ny_rel", float(dEy) / 100.0)
        _add("emit_z_rel",  float(dEz) / 100.0)
        _add("mismatch_x", float(mx))
        _add("mismatch_y", float(my))
        _add("mismatch_z", float(mz))
        _add("current_rel", float(dIb) / 100.0)

    def add_cutoff(self, params: list[float]) -> None:
        """ERROR_GAUSSIAN_CUT_OFF σ_max — sets the global truncation."""
        if not params:
            return
        self.cutoff = float(params[0])

    def add_ratios(self, params: list[float]) -> None:
        """ERROR_SET_RATIO r1 r2 r3 …"""
        self.ratios = [float(x) for x in params]

    # ----- finalisation --------------------------------------------------
    def finalize(self, lattice) -> None:
        """Push accumulated errors onto the lattice."""
        # Apply global cutoff to every quadrupole/cavity/bend ErrorDef
        # whose distribution is gaussian.
        if self.cutoff > 0.0:
            for ed in self.errors:
                if ed.distribution == "gaussian":
                    ed.cutoff = self.cutoff
            for be in self.beam_errors:
                if be.distribution == "gaussian":
                    be.cutoff = self.cutoff
        lattice.errors = list(self.errors)
        lattice.beam_errors = list(self.beam_errors)
        lattice.error_ratios = list(self.ratios)
        lattice.error_cutoff = float(self.cutoff)
        # ERROR_SET_RATIO is parsed for round-trip but nothing consumes
        # lattice.error_ratios (TraceWin uses the ratios to scale linked
        # error families; HELIX's ErrorStudy does not implement that yet).
        if self.ratios:
            warnings.warn(
                "ERROR_SET_RATIO is parsed but not implemented — the "
                "ratios are stored on lattice.error_ratios and ignored by "
                "ErrorStudy (error amplitudes will NOT be scaled).",
                UserWarning,
                stacklevel=2,
            )
