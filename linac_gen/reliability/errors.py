"""Leg C glue: an error budget → a populated ``ErrorStudy``, the pre-run
audit policy, and the per-seed draw worker the campaign runs in a pool.

The budget is the same vocabulary as the error engine (``add_error`` /
``add_beam_error``): a list of element rows ``{pattern, parameter,
sigma | half_width, distribution, cutoff, kind}`` and beam rows
``{parameter, sigma | half_width, distribution, cutoff}``.
:func:`audit_or_raise` mirrors the misalignment study driver's policy —
refuse to run on a selector that matches nothing, a warn-skipped or
inert parameter, an unknown distribution, or overlapping selectors — so
a malformed budget never silently draws nothing.  :func:`draw_seed` is
the pool worker: it re-parses the deck, draws one seed through
:meth:`ErrorStudy.draw_overrides` (the engine's own RNG streams), runs
the optional orbit correction on the errored copy and returns the seed
as JSON-serialisable overrides (element draws, beam config, steerer
kicks) for the scan-pool points.
"""
from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import asdict, dataclass, field

__all__ = ["ErrorBudget", "build_error_study", "audit_or_raise",
           "draw_seed", "steerer_kick_overrides", "auto_paired_correction"]

_ROW_KEYS = ("pattern", "parameter", "distribution", "sigma", "half_width",
             "cutoff", "kind")


@dataclass
class ErrorBudget:
    """Element and beam error rows (see the module docstring)."""
    element: list = field(default_factory=list)
    beam: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorBudget":
        el = [dict(r) for r in (d.get("element") or [])]
        bm = [dict(r) for r in (d.get("beam") or [])]
        for r in el:
            if "pattern" not in r or "parameter" not in r:
                raise ValueError(f"error budget: element row needs pattern and parameter: {r}")
        for r in bm:
            if "parameter" not in r:
                raise ValueError(f"error budget: beam row needs parameter: {r}")
        return cls(element=el, beam=bm)

    def to_dict(self) -> dict:
        return {"element": [dict(r) for r in self.element],
                "beam": [dict(r) for r in self.beam]}

    @classmethod
    def from_study_budget(cls, budget: dict, beam_budget=()) -> "ErrorBudget":
        """Convert the misalignment-study driver's ``BUDGET`` /
        ``BEAM_BUDGET`` shape (spec → patterns + (parameter, sigma, unit)
        tuples, all Gaussian at 3σ)."""
        el = []
        for spec in budget.values():
            pats = spec.get("patterns") or [spec["pattern"]]
            for pat in pats:
                for parameter, sigma, _unit in spec["errors"]:
                    if sigma <= 0:
                        continue
                    el.append({"pattern": pat, "parameter": parameter,
                               "distribution": "gaussian", "sigma": float(sigma),
                               "cutoff": 3.0, "kind": spec.get("kind")})
        bm = [{"parameter": p, "distribution": "gaussian", "sigma": float(s), "cutoff": 3.0}
              for p, s, _u in beam_budget if s > 0]
        return cls(element=el, beam=bm)


def build_error_study(lattice, beam_cfg, budget: ErrorBudget, *, n_seeds: int = 1,
                      base_seed: int = 0, sc_config=None, correction: dict | None = None):
    """A populated ``ErrorStudy`` (nothing run).  ``correction`` = the
    ``enable_correction`` kwargs (``method, n_iter, tol_mm, bpm_noise,
    rcond, targets, reading_backend``) or None."""
    from linac_gen.errors.error_model import ErrorStudy
    study = ErrorStudy(lattice=lattice, beam_config=beam_cfg, n_seeds=int(n_seeds),
                       sc_config=sc_config, base_seed=int(base_seed))
    for r in budget.element:
        study.add_error(pattern=r["pattern"], parameter=r["parameter"],
                        distribution=r.get("distribution", "gaussian"),
                        sigma=float(r.get("sigma", 0.0)),
                        half_width=float(r.get("half_width", 0.0)),
                        cutoff=float(r.get("cutoff", 3.0)),
                        element_kind=r.get("kind"))
    for r in budget.beam:
        study.add_beam_error(parameter=r["parameter"],
                             distribution=r.get("distribution", "gaussian"),
                             sigma=float(r.get("sigma", 0.0)),
                             half_width=float(r.get("half_width", 0.0)),
                             cutoff=float(r.get("cutoff", 3.0)))
    if correction:
        kw = {k: v for k, v in correction.items() if k != "enabled"}
        study.enable_correction(**kw)
    return study


def audit_problems(audit: dict) -> list[str]:
    """The anomaly list the misalignment-study driver refuses on."""
    problems: list[str] = []
    for row in audit["defs"]:
        tag = f"def {row['index']} ({row['pattern']!r}, {row['parameter']})"
        if not row["matched"]:
            problems.append(f"{tag} matches no element")
        if row["skipped"]:
            classes = sorted({c for _n, c in row["skipped"]})
            problems.append(f"{tag} would be warn-skipped on "
                            f"{len(row['skipped'])} matched element(s) {classes}")
        if row["inert"]:
            problems.append(f"{tag} is inert — the tracker never reads it")
        if row.get("bad_distribution"):
            problems.append(f"{tag} has an unknown distribution — it would draw NOTHING")
    for row in audit.get("beam_defs", ()):
        tag = f"beam def {row['index']} ({row['parameter']!r})"
        if not row["applicable"]:
            problems.append(f"{tag} does not exist on BeamConfig")
        if row["bad_distribution"]:
            problems.append(f"{tag} has an unknown distribution — it would draw NOTHING")
    for (name, parameter), defs in sorted(audit.get("multi", {}).items()):
        problems.append(f"element {name} draws {parameter!r} from {len(defs)} "
                        f"different defs {defs} — overlapping selectors")
    return problems


def audit_or_raise(study) -> dict:
    """Run ``study.audit_errors()`` and raise ``ValueError`` listing every
    anomaly; returns the audit when clean."""
    audit = study.audit_errors()
    problems = audit_problems(audit)
    if problems:
        raise ValueError("error budget audit refused:\n  - " + "\n  - ".join(problems))
    return audit


def steerer_kick_overrides(design, corrected) -> tuple:
    """``@index.bx_l`` / ``by_l`` overrides for every Steerer whose kick
    differs between the design lattice and the corrected copy."""
    from linac_gen.elements.steerer import Steerer
    out: list = []
    for i, (a, b) in enumerate(zip(design.elements, corrected.elements)):
        if not isinstance(b, Steerer):
            continue
        for attr in ("bx_l", "by_l"):
            if float(getattr(a, attr)) != float(getattr(b, attr)):
                out.append((f"@{i + 1}.{attr}", float(getattr(b, attr))))
    return tuple(out)


def auto_paired_correction(lattice, beam_factory, *, override_method=None,
                           n_iter: int = 5, tol_mm: float = 0.05, bpm_noise: float = 0.0,
                           rcond=None, noise_seed: int = 0, targets=None,
                           reading_backend: str = "envelope", beam_config=None,
                           should_stop=None) -> dict:
    """Orbit correction WITHOUT ``ADJUST_STEERER`` cards: every magnetic
    ``Steerer`` of the lattice against every ``is_bpm`` marker, in place.
    One-to-one (each steerer to its nearest downstream BPM) when the two
    counts match, otherwise a global SVD over all BPMs — the same
    ``apply_correction`` the card-driven driver calls, so a deck whose
    cards pair each steerer with its next BPM gives the same kicks.
    Without cards there is no per-steerer ``vmax`` clip.  Returns the
    ``run_correction_from_lattice`` dict (``method`` is ``"none"`` when the
    deck has no steerer or no BPM; ``n_pairs`` counts the steerers that
    took part — one-to-one drops a steerer with no BPM downstream)."""
    from linac_gen.elements.steerer import Steerer
    from linac_gen.errors.correction import apply_correction, correction_status
    steerers = [e for e in lattice.elements if isinstance(e, Steerer) and not getattr(e, "elec", False)]
    bpms = [e for e in lattice.elements if getattr(e, "is_bpm", False)]
    if not steerers or not bpms:
        return {"kicks": {}, "history": [], "method": "none", "n_pairs": 0,
                "status": "none", "converged": False, "beam_lost_at": None, "pairing": "auto"}
    method = override_method or ("one_to_one" if len(steerers) == len(bpms) else "svd")
    kicks, hist = apply_correction(
        lattice, beam_factory, method=method, bpm_noise=bpm_noise, rcond=rcond,
        n_iter=n_iter, tol_mm=tol_mm, steerers=steerers, bpms=bpms,
        noise_seed=noise_seed, history=True, should_stop=should_stop, targets=targets,
        reading_backend=reading_backend, beam_config=beam_config)
    st = correction_status(hist)
    return {"kicks": kicks, "history": hist, "method": method, "n_pairs": len(kicks),
            "status": st["status"], "converged": st["converged"],
            "beam_lost_at": st["beam_lost_at"], "pairing": "auto"}


def draw_seed(job: dict) -> dict:
    """Pool worker: one seed's draws (+ optional orbit correction) as
    JSON-serialisable overrides.

    ``job``: ``{"lattice_path", "beam_config" (BeamConfig dict), "budget"
    (ErrorBudget dict), "seed", "base_seed" (default 0), "correction"
    (enable_correction kwargs plus ``pairing`` = ``"cards"`` — the deck's
    ``ADJUST_STEERER`` cards, the default — or ``"auto"`` — every steerer
    against every BPM, :func:`auto_paired_correction` — or None),
    "tracewin_ini" (optional)}``.
    Returns ``{"seed", "overrides", "beam_config", "untransportable",
    "applied", "beam_applied", "correction" (the driver's dict with its
    per-pass ``history`` and ``rms_before_mm``, the uncorrected orbit rms
    over every BPM — envelope backend), "kick_overrides", "error"}``;
    a raised exception is captured into ``error`` with the traceback.
    """
    import time
    t0 = time.time()
    try:
        from linac_gen.cli import common
        from linac_gen.core.config import BeamConfig
        from linac_gen.distributions.factory import create_beam
        with contextlib.redirect_stdout(io.StringIO()):
            lattice, _cfg, _conv = common.load_input(
                job["lattice_path"], tracewin_ini=job.get("tracewin_ini"))
        beam_cfg = BeamConfig(**job["beam_config"])
        budget = ErrorBudget.from_dict(job["budget"])
        seed = int(job["seed"])
        study = build_error_study(lattice, beam_cfg, budget, n_seeds=1,
                                  base_seed=int(job.get("base_seed", 0)))
        draw = study.draw_overrides(seed)
        out = {
            "seed": seed,
            "overrides": [list(o) for o in draw.overrides],
            "beam_config": asdict(draw.beam_config),
            "untransportable": [list(u) for u in draw.untransportable],
            "applied": draw.applied,
            "beam_applied": draw.beam_applied,
            "correction": None,
            "kick_overrides": [],
            "elapsed": None,
            "error": None,
        }
        corr = job.get("correction")
        if corr and corr.get("enabled", True):
            from linac_gen.errors.correction import run_correction_from_lattice
            kw = {k: v for k, v in corr.items() if k not in ("enabled", "pairing")}
            kw.setdefault("reading_backend", "envelope")
            if "method" in kw:
                kw["override_method"] = kw.pop("method")
            bcfg = draw.beam_config
            def _factory(_b=bcfg, _s=seed):
                return create_beam(_b, seed=_s + 1000)
            rms_before = None
            if kw.get("reading_backend") == "envelope":
                # the uncorrected orbit over EVERY BPM (one envelope run):
                # the seed row reports it next to the history's after-pass rms
                from linac_gen.cli.common import run_envelope_sim
                from linac_gen.errors.correction import _orbit_rms_mm
                bpms = [e for e in draw.lattice.elements if getattr(e, "is_bpm", False)]
                rms_before = float(_orbit_rms_mm(draw.lattice, lambda: run_envelope_sim(draw.lattice, bcfg),
                                                 bpms, 0.0, None))
            pairing = (corr.get("pairing") or "cards")
            if pairing == "auto":
                info = auto_paired_correction(
                    draw.lattice, _factory, noise_seed=seed, beam_config=bcfg, **kw)
            elif pairing == "cards":
                info = run_correction_from_lattice(
                    draw.lattice, _factory, noise_seed=seed, beam_config=bcfg, **kw)
            else:
                raise ValueError(f"correction.pairing must be cards|auto, got {pairing!r}")
            out["correction"] = {**info, "rms_before_mm": rms_before}   # history = per-pass rms, plain values
            out["kick_overrides"] = [list(o) for o in
                                     steerer_kick_overrides(lattice, draw.lattice)]
        out["elapsed"] = time.time() - t0
        return out
    except Exception as exc:
        import traceback
        return {"seed": job.get("seed"), "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "elapsed": time.time() - t0}
