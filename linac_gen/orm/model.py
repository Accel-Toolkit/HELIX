"""Model orbit response matrix from the envelope phase probe.

The centroid of a beam is carried by the same per-element transfer maps the
envelope uses (``EnvelopeResults.element_maps_bare``), and linear space charge
does not move it; so the response of every BPM to every trim follows from ONE
probe run at zero current: a unit angle kick at the trim exit is chained
through the downstream maps and read at each BPM.  Kick-and-read tracking with
the envelope solver (:meth:`OrmModel.tracked_response`) is the slow cross-check
and agrees to ~1e-8 (verified on fnalscl at 23.7 mA).

Units: the map response is mm per mrad of actual deflection; the physical
response is mm per T·m of integrated field, ``kick_mrad_per_Tm = sign(q)·1e3/Bρ``
at the trim (``Steerer._kick_mrad``: Δx' = sign(q)·by_l/Bρ, Δy' = sign(q)·bx_l/Bρ).
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np

from linac_gen.core.cancelled import OperationCancelled
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.orm.devices import OrmSelection, element_label

_BLOCKS = (("x", "x"), ("x", "y"), ("y", "x"), ("y", "y"))       # (kick plane, read plane)
_ROW = {"x": 0, "y": 2}                                          # position row of the 6-vector
_COL = {"x": 1, "y": 3}                                          # angle column of the unit kick


@dataclass
class ModelOrm:
    """Response blocks in mm per T·m (``per_Tm``) and mm per mrad (``per_mrad``), keyed ``(kick, read)``."""

    per_Tm: dict
    per_mrad: dict
    gscale: np.ndarray                      # quad scale factors the model was built with
    method: str = "maps"                    # "maps" | "tracking"
    notes: list = field(default_factory=list)

    def block(self, kick: str, read: str) -> np.ndarray:
        return self.per_Tm[(kick, read)]


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


class OrmModel:
    """Model ORM of ``lattice`` for the BPMs/trims of ``selection``.

    ``response(gscale)`` sets every fitted quadrupole's ``gradient_rel`` so
    that ``effective_gradient = design·gscale`` (the design ``gradient`` is
    never touched), runs one probe, and restores the previous ``gradient_rel``
    even when cancelled.
    """

    def __init__(self, lattice, beam_cfg, selection: OrmSelection, *, quads="upstream_of_last_bpm"):
        from linac_gen.cli.common import build_ref, _envelope_initial
        self.lattice = lattice
        self.beam_cfg = beam_cfg
        self.sel = selection
        self.ref = build_ref(beam_cfg)
        self.initial = _envelope_initial(beam_cfg, self.ref)
        self.kick_sign = -1.0 if getattr(self.ref.species, "charge", 1) < 0 else 1.0
        els = lattice.elements
        last_bpm = int(selection.bpm_idx.max()) if selection.n_bpm else len(els)
        first_trim = int(selection.trim_idx.min()) if selection.n_trim else 0
        if quads == "upstream_of_last_bpm":
            self.quad_idx = np.array([i for i, e in enumerate(els) if isinstance(e, Quadrupole) and first_trim < i < last_bpm], dtype=int)
        elif quads == "all":
            self.quad_idx = np.array([i for i, e in enumerate(els) if isinstance(e, Quadrupole)], dtype=int)
        else:
            self.quad_idx = np.asarray(quads, dtype=int)
        self.quads = [els[i] for i in self.quad_idx]
        self.quad_labels = [element_label(q) for q in self.quads]
        self.g0 = np.array([float(q.gradient) for q in self.quads])           # design gradients (T/m)
        self.rel0 = np.array([float(getattr(q, "gradient_rel", 0.0)) for q in self.quads])
        self.n_probe = 0
        self._brho_trim = None
        self._w_trim = None

    # ------------------------------------------------------------------ probes
    def _probe(self, should_stop=None):
        from linac_gen.analysis.phase_advance import run_phase_probe
        if should_stop is not None and should_stop():
            raise OperationCancelled("ORM model probe cancelled")
        self.n_probe += 1
        with _quiet():
            res = run_phase_probe(self.lattice, self.ref, self.initial, current=0.0, should_abort=should_stop)
        if getattr(res, "aborted", False) or (should_stop is not None and should_stop()):
            raise OperationCancelled("ORM model probe cancelled")           # never use a partial probe
        return res

    def _set_scale(self, gscale):
        for q, r0, g in zip(self.quads, self.rel0, gscale):
            q.gradient_rel = float((1.0 + r0) * g - 1.0)

    def _restore(self):
        for q, r0 in zip(self.quads, self.rel0):
            q.gradient_rel = float(r0)

    @property
    def brho_trim(self) -> np.ndarray:
        """Rigidity (T·m) at every trim, from the reference energy recorded by the probe."""
        if self._brho_trim is None:
            self.response(np.ones(len(self.quads)))
        return self._brho_trim

    @property
    def w_trim_MeV(self) -> np.ndarray:
        if self._w_trim is None:
            self.response(np.ones(len(self.quads)))
        return self._w_trim

    # --------------------------------------------------------------- response
    def response(self, gscale=None, *, should_stop=None) -> ModelOrm:
        """All four blocks for quad scale factors ``gscale`` (default: all 1)."""
        gscale = np.ones(len(self.quads)) if gscale is None else np.asarray(gscale, dtype=float)
        if gscale.shape != (len(self.quads),):
            raise ValueError(f"gscale must have {len(self.quads)} entries (one per fitted quad), got {gscale.shape}")
        self._set_scale(gscale)
        try:
            res = self._probe(should_stop)
        finally:
            self._restore()
        maps = res.element_maps_bare
        exit_idx = np.asarray(res.element_exit_idx, dtype=int)
        w_kin = np.asarray(res.ref_w_kin, dtype=float)
        sel = self.sel
        nb, nt = sel.n_bpm, sel.n_trim
        # rigidity at every trim: the reference energy at the trim's exit row
        w_trim = np.array([w_kin[exit_idx[i]] for i in sel.trim_idx]) if nt else np.zeros(0)
        brho = np.array([ReferenceParticle(self.ref.species, float(w), self.ref.frequency).brho for w in w_trim])
        self._w_trim, self._brho_trim = w_trim, brho
        per_mrad = {blk: np.zeros((nb, nt)) for blk in _BLOCKS}
        bpm_pos = {int(i): n for n, i in enumerate(sel.bpm_idx)}
        last = int(sel.bpm_idx.max()) if nb else -1
        for j, i_t in enumerate(sel.trim_idx):
            v = np.zeros((6, 2)); v[1, 0] = 1.0; v[3, 1] = 1.0            # unit x' and y' kicks (mrad) at the trim exit
            for k in range(int(i_t) + 1, last + 1):
                v = maps[k] @ v
                if k in bpm_pos:
                    n = bpm_pos[k]
                    for kp, c in _COL.items():
                        for rp, r in _ROW.items():
                            per_mrad[(kp, rp)][n, j] = v[r, 0 if kp == "x" else 1]
        per_Tm = {blk: per_mrad[blk] * (self.kick_sign * 1e3 / brho)[None, :] for blk in _BLOCKS}
        return ModelOrm(per_Tm=per_Tm, per_mrad=per_mrad, gscale=gscale.copy(), method="maps")

    def tracked_response(self, *, dbl_Tm: float = 1e-3, current_mA=None, should_stop=None) -> ModelOrm:
        """Kick-and-read cross-check: one envelope run per trim and plane, centroid read at the BPMs."""
        from linac_gen.errors.correction import _live_bpm_reading
        from linac_gen.tracking.envelope import EnvelopeSolver
        current = float(self.beam_cfg.current) if current_mA is None else float(current_mA)
        sel = self.sel; els = self.lattice.elements
        bpms = [els[i] for i in sel.bpm_idx]; notes = []

        def run():
            if should_stop is not None and should_stop():
                raise OperationCancelled("ORM tracking cross-check cancelled")
            with _quiet():
                rec = EnvelopeSolver(self.lattice, self.ref.copy(), self.initial, current=current, should_abort=should_stop).run()
            out = np.full((len(bpms), 2), np.nan)
            for n, b in enumerate(bpms):
                r = _live_bpm_reading(self.lattice, rec, b)
                if r is not None:
                    out[n] = r
            return out
        r0 = run()
        nb, nt = sel.n_bpm, sel.n_trim
        per_Tm = {blk: np.zeros((nb, nt)) for blk in _BLOCKS}
        for j, i_t in enumerate(sel.trim_idx):
            st = els[int(i_t)]
            for kp, attr in (("x", "by_l"), ("y", "bx_l")):
                old = float(getattr(st, attr)); setattr(st, attr, old + dbl_Tm)
                try:
                    r = run()
                finally:
                    setattr(st, attr, old)
                for rp, c in (("x", 0), ("y", 1)):
                    per_Tm[(kp, rp)][:, j] = (r[:, c] - r0[:, c]) / dbl_Tm
        if not np.isfinite(np.concatenate([b.ravel() for b in per_Tm.values()])).all():
            notes.append("dead beam at one or more BPMs during the kick-and-read runs (NaN entries)")
        brho = self.brho_trim
        per_mrad = {blk: per_Tm[blk] / (self.kick_sign * 1e3 / brho)[None, :] for blk in _BLOCKS}
        return ModelOrm(per_Tm=per_Tm, per_mrad=per_mrad, gscale=np.ones(len(self.quads)), method="tracking", notes=notes)

    def selfcheck(self, model_a: ModelOrm, model_b: ModelOrm) -> float:
        """Largest |Δ| between two models relative to the largest entry of each in-plane block."""
        worst = 0.0
        for blk in _BLOCKS:
            a, b = model_a.per_Tm[blk], model_b.per_Tm[blk]
            scale = max(np.nanmax(np.abs(a)) if a.size else 0.0, np.nanmax(np.abs(b)) if b.size else 0.0, 1e-300)
            worst = max(worst, float(np.nanmax(np.abs(a - b)) / scale) if a.size else 0.0)
        return worst
