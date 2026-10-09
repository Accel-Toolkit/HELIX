# linac_gen/errors/orm_correction.py
"""Model-ORM SVD orbit correction (one probe instead of 2N tracking runs).

:func:`linac_gen.errors.correction.apply_correction` builds its response
matrix by finite-difference tracking — two runs per steerer per pass.  For
a transfer line with ~50 correctors and an ensemble of hundreds of seeds
that dominates the study.  Here the response comes from the envelope phase
probe instead (:class:`linac_gen.orm.model.OrmModel`, one zero-current run,
all four blocks x→x, x→y, y→x, y→y including coupling), built on the
DESIGN lattice by default — the realistic case: the control room corrects
with the model, not with the unknown errors.

The orbit itself is read from the envelope centroid (``c → M c + o``,
which carries steerer kicks, misalignment feed-down and the dipole
field-error / roll terms); linear space charge does not move the centroid
in HELIX's model, so the readings are taken at zero current.  A BPM reads
``true − offset + noise``: ``bpm_offsets`` (mm, fixed per seed — e.g. BPM
misalignment / electrical offset) and ``bpm_noise`` (mm rms, resolution,
one draw per reading).

Each BPM reads only its own plane(s) (``bpm_planes``: ``"x"``, ``"y"`` or
``"xy"``, default from the MAD-style label ``HP…``/``VP…``) and each
steerer kicks in its plane(s) (``steerer_planes``, default from ``HT…``/
``VT…``).  The kicks solve ``R·Δθ = −(reading − target)`` with a truncated
SVD (``n_sv`` largest singular values and/or an ``rcond`` relative cut,
default 1e-9), clipped to ``vmax`` [T·m], iterated ``n_iter`` times on
fresh readings.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


def _plane_from_label(label: str | None, h_prefixes, v_prefixes, default="xy") -> str:
    lab = (label or "").upper()
    if any(lab.startswith(p) for p in h_prefixes):
        return "x"
    if any(lab.startswith(p) for p in v_prefixes):
        return "y"
    return default


def bpm_plane(elem) -> str:
    """Measured plane of a BPM from its MAD-style label (HP…/VP…), else both."""
    return _plane_from_label(getattr(elem, "label", None), ("HP", "IHP"), ("VP", "IVP"))


def steerer_plane(elem) -> str:
    """Kick plane of a steerer from its label (HT…/HK…/XCOR, VT…/VK…/YCOR), else both."""
    return _plane_from_label(getattr(elem, "label", None),
                             ("HT", "HK", "XCOR", "LBTL:ARC1:XCOR"),
                             ("VT", "VK", "YCOR"))


def _selection(lattice, bpm_idx, trim_idx):
    """Minimal :class:`OrmSelection` for model use (no measured data)."""
    from linac_gen.orm.devices import OrmSelection, element_label
    els = lattice.elements
    lengths = [float(getattr(e, "length", 0.0) or 0.0) for e in els]
    s_ent = np.concatenate([[0.0], np.cumsum(lengths)]) / 1000.0
    nb, nt = len(bpm_idx), len(trim_idx)
    return OrmSelection(
        bpm_idx=np.asarray(bpm_idx, dtype=int),
        bpm_labels=[element_label(els[i]) for i in bpm_idx],
        bpm_devices={"x": [None] * nb, "y": [None] * nb},
        bpm_sign={"x": np.ones(nb), "y": np.ones(nb)},
        bpm_include={"x": np.ones(nb, bool), "y": np.ones(nb, bool)},
        trim_idx=np.asarray(trim_idx, dtype=int),
        trim_labels=[element_label(els[i]) for i in trim_idx],
        trim_devices={"x": [None] * nt, "y": [None] * nt},
        trim_include={"x": np.ones(nt, bool), "y": np.ones(nt, bool)},
        s_bpm_m=s_ent[np.asarray(bpm_idx, dtype=int) + 1] if nb else np.zeros(0),
        s_trim_m=s_ent[np.asarray(trim_idx, dtype=int)] if nt else np.zeros(0),
        down=(np.asarray(bpm_idx)[:, None] > np.asarray(trim_idx)[None, :]),
        unmatched_devices={"bpm": [], "trim": []},
        unmatched_elements={"bpm": [], "trim": []},
        dead_devices=[], notes=[])


@dataclass
class OrmCorrector:
    """Reusable model-ORM SVD corrector for one lattice layout.

    Build once per study (the response depends only on the model lattice),
    then call :meth:`correct` on each seed's errored copy — same element
    indices, so the steerers/BPMs are found by index.
    """
    model_lattice: object
    beam_config: object
    steerer_idx: list
    bpm_idx: list
    steerer_planes: dict = field(default_factory=dict)   # index -> "x"|"y"|"xy"
    bpm_planes: dict = field(default_factory=dict)       # index -> "x"|"y"|"xy"
    n_sv: int | None = None
    rcond: float | None = None
    vmax_Tm: float | None = None

    def __post_init__(self):
        from linac_gen.orm.model import OrmModel
        els = self.model_lattice.elements
        self.steerer_idx = [int(i) for i in self.steerer_idx]
        self.bpm_idx = [int(i) for i in self.bpm_idx]
        for i in self.steerer_idx:
            self.steerer_planes.setdefault(i, steerer_plane(els[i]))
        for i in self.bpm_idx:
            self.bpm_planes.setdefault(i, bpm_plane(els[i]))
        sel = _selection(self.model_lattice, self.bpm_idx, self.steerer_idx)
        cfg0 = copy.deepcopy(self.beam_config)
        cfg0.current = 0.0
        orm = OrmModel(self.model_lattice, cfg0, sel, quads=[]).response()
        # columns: (steerer j, kick plane); rows: (bpm n, read plane)
        self.cols = [(j, p) for j, i in enumerate(self.steerer_idx)
                     for p in "xy" if p in self.steerer_planes[i]]
        self.rows = [(n, p) for n, i in enumerate(self.bpm_idx)
                     for p in "xy" if p in self.bpm_planes[i]]
        R = np.zeros((len(self.rows), len(self.cols)))
        for c, (j, kp) in enumerate(self.cols):
            for r, (n, rp) in enumerate(self.rows):
                R[r, c] = orm.per_Tm[(kp, rp)][n, j]          # mm per T·m
        self.R = R
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        keep = np.ones_like(S, dtype=bool)
        if self.n_sv is not None:
            keep[int(self.n_sv):] = False
        # Relative cut: the requested one, else 1e-9 — a steerer with no
        # BPM downstream gives an exactly-zero column on the design model
        # but a ~1e-17 singular value on a perturbed one, whose inverse
        # would blow the correction up.
        rcond = 1e-9 if self.rcond is None else float(self.rcond)
        if S.size:
            keep &= (S >= rcond * S[0]) & (S > 0.0)
        self.singular_values = S
        self.n_kept = int(keep.sum())
        Sinv = np.zeros_like(S)
        Sinv[keep] = 1.0 / S[keep]
        self.Rpinv = (Vt.T * Sinv) @ U.T                     # T·m per mm

    def _check_layout(self, lattice) -> None:
        """The corrected lattice must have the model's layout (indices)."""
        from linac_gen.elements.steerer import Steerer
        els = lattice.elements
        if len(els) != len(self.model_lattice.elements):
            raise ValueError(f"OrmCorrector: lattice has {len(els)} elements, the "
                             f"model {len(self.model_lattice.elements)} — correct a "
                             "copy of the model lattice")
        bad = [i for i in self.steerer_idx if not isinstance(els[i], Steerer)]
        bad += [i for i in self.bpm_idx if not getattr(els[i], "is_bpm", False)]
        if bad:
            raise ValueError(f"OrmCorrector: element(s) {bad[:5]} are not the "
                             "model's steerers / BPMs")

    # ------------------------------------------------------------- reading
    def read(self, lattice, *, offsets=None, noise_mm=0.0, rng=None,
             beam_config=None):
        """BPM readings (rows order) and the true orbit, from one envelope run."""
        from linac_gen.cli.common import run_envelope_sim
        cfg = copy.deepcopy(beam_config if beam_config is not None else self.beam_config)
        cfg.current = 0.0
        res = run_envelope_sim(lattice, cfg)
        exit_idx = res.element_exit_idx
        cen = res.centroid
        true = np.empty(len(self.rows))
        for r, (n, p) in enumerate(self.rows):
            c = cen[exit_idx[self.bpm_idx[n]]]
            true[r] = c[0] if p == "x" else c[2]
        meas = true.copy()
        if offsets is not None:
            meas -= offsets
        noise = np.broadcast_to(np.asarray(noise_mm if noise_mm is not None else 0.0,
                                           dtype=float), meas.shape)
        if rng is not None and np.any(noise > 0):
            meas += rng.normal(0.0, 1.0, size=meas.size) * noise
        return meas, true, res

    # ---------------------------------------------------------- correction
    def correct(self, lattice, *, n_iter=2, offsets=None, noise_mm=0.0,
                rng=None, beam_config=None, targets=None):
        """Correct ``lattice`` in place (steerer integrated fields).

        ``offsets``: per-row BPM offsets [mm] (``self.rows`` order) or None;
        ``noise_mm``: reading noise σ [mm], a scalar or per row.
        Returns a dict with the history (rms of the TRUE orbit and of the
        readings per pass), the final kicks [T·m], and ``n_clipped`` — the
        number of kicks clipped at ``vmax`` in the LAST pass (i.e. sitting
        at the limit).
        """
        els = lattice.elements
        self._check_layout(lattice)
        tgt = np.zeros(len(self.rows)) if targets is None else np.asarray(targets, float)
        hist = []
        n_clipped = 0
        for it in range(int(n_iter) + 1):
            meas, true, _ = self.read(lattice, offsets=offsets, noise_mm=noise_mm,
                                      rng=rng, beam_config=beam_config)
            hist.append({"iter": it, "rms_true_mm": float(np.sqrt(np.mean(true ** 2))),
                         "rms_meas_mm": float(np.sqrt(np.mean((meas - tgt) ** 2))),
                         "max_true_mm": float(np.max(np.abs(true))) if true.size else 0.0})
            if it == int(n_iter):
                break
            dk = -self.Rpinv @ (meas - tgt)
            n_clipped = 0
            for c, (j, kp) in enumerate(self.cols):
                st = els[self.steerer_idx[j]]
                attr = "by_l" if kp == "x" else "bx_l"
                v = float(getattr(st, attr)) + float(dk[c])
                if self.vmax_Tm is not None and abs(v) > self.vmax_Tm:
                    v = float(np.sign(v) * self.vmax_Tm)
                    n_clipped += 1
                setattr(st, attr, v)
        kicks = {els[self.steerer_idx[j]].name:
                 (float(els[self.steerer_idx[j]].bx_l), float(els[self.steerer_idx[j]].by_l))
                 for j in range(len(self.steerer_idx))}
        return {"history": hist, "kicks_Tm": kicks, "n_clipped": n_clipped,
                "n_sv_kept": self.n_kept,
                "rms_before_mm": hist[0]["rms_true_mm"],
                "rms_after_mm": hist[-1]["rms_true_mm"]}
