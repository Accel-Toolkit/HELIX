"""``python -m linac_gen orm`` — orbit-response matrices: compare a measurement with the model,
fit quad gradients / trim calibrations / BPM gains (LOCO-style), export a recalibrated deck.

    python -m linac_gen orm compare  INPUT --measured DIR [--measured DIR2 | --measured FILE.csv] [options]
    python -m linac_gen orm fit      INPUT --measured … [--stage trims+quads] [--calibration cal.json] [--export-deck out.dat]
    python -m linac_gen orm export   INPUT --calibration cal.json --out-deck out.dat [--lgproj]
    python -m linac_gen orm validate INPUT --measured … [--seed 7 --g-sigma 0.03]
    python -m linac_gen orm model    INPUT --out DIR

``INPUT`` is a ``.lgproj`` project or a lattice file (``--energy/--freq/--species`` for a bare deck).
Manual: docs/manual/06_running/13_cli_orm.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from linac_gen.cli import common
from linac_gen.orm.fit import STAGES


def add_arguments(p) -> None:
    sub = p.add_subparsers(dest="orm_mode", required=True, metavar="MODE")

    def _common(sp, measured=True, mapping=True, fit_opts=False):
        sp.add_argument("input", help="a .lgproj project or a lattice file (.dat …)")
        common.add_tracewin_ini_argument(sp)
        sp.add_argument("--energy", type=float, help="kinetic energy (MeV) for a bare lattice file")
        sp.add_argument("--freq", type=float, help="beam frequency (MHz) for a bare lattice file")
        sp.add_argument("--species", help="proton / deuteron / H-")
        sp.add_argument("--out", metavar="DIR", help="output folder (JSON, CSV, calibration; PNG with --plots)")
        sp.add_argument("--plots", action="store_true", help="write the figures into --out")
        sp.add_argument("-q", "--quiet", action="store_true", help="print only the per-plane summary lines")
        if measured:
            sp.add_argument("--measured", action="append", default=[], metavar="DIR|CSV",
                            help="FORMA scan folder (one driven plane) or HELIX ORM CSV; repeat for the second plane")
        if mapping:
            sp.add_argument("--map", metavar="JSON", help="device map JSON (overrides the built-in naming rules)")
            sp.add_argument("--write-map", metavar="JSON", help="write the resolved device map")
            sp.add_argument("--exclude-bpm", action="append", default=[], metavar="DEV|LABEL", help="leave this BPM out of the comparison/fit")
            sp.add_argument("--exclude-trim", action="append", default=[], metavar="DEV|LABEL", help="leave this trim out")
            sp.add_argument("--sys-floor", type=float, default=0.05, help="systematic floor as a fraction of the column maximum (default 0.05)")
        if fit_opts:
            sp.add_argument("--stage", default="trims+quads", choices=list(STAGES) + ["all"], help="which parameters to fit (default trims+quads)")
            sp.add_argument("--prior-g", type=float, default=0.10, help="Gaussian prior width on the quad scale factors (default 0.10)")
            sp.add_argument("--prior-G", type=float, default=0.05, help="prior width on the BPM gains (default 0.05)")
            sp.add_argument("--fix-quad", action="append", default=[], metavar="LABEL", help="hold this quad at scale 1")
            sp.add_argument("--max-iter", type=int, default=12)

    c = sub.add_parser("compare", help="measured vs model: per-trim calibration, correlation, residual"); _common(c)
    c.add_argument("--check-tracking", action="store_true", help="also compute the kick-and-read (tracking) model and report the largest deviation")
    f = sub.add_parser("fit", help="LOCO-style fit of quad scale factors, trim calibrations and BPM gains"); _common(f, fit_opts=True)
    f.add_argument("--calibration", metavar="JSON", help="write the calibration record here (default <out>/orm_calibration.json)")
    f.add_argument("--export-deck", metavar="DAT", help="also write the recalibrated deck (text surgery on the original .dat)")
    f.add_argument("--lgproj", action="store_true", help="with --export-deck: write a sibling .lgproj pointing at the new deck")
    e = sub.add_parser("export", help="write a recalibrated deck from a calibration JSON"); _common(e, measured=False, mapping=False)
    e.add_argument("--calibration", required=True, metavar="JSON"); e.add_argument("--out-deck", required=True, metavar="DAT")
    e.add_argument("--lgproj", action="store_true", help="write a sibling .lgproj pointing at the new deck")
    v = sub.add_parser("validate", help="synthetic closed loop: inject random errors, refit, report the recovery"); _common(v, fit_opts=True)
    v.add_argument("--seed", type=int, default=7); v.add_argument("--g-sigma", type=float, default=0.03, help="rms of the injected quad errors (default 0.03)")
    v.add_argument("--G-sigma", type=float, default=0.03, help="rms of the injected BPM gain errors (default 0.03)")
    m = sub.add_parser("model", help="write the model response matrix of every BPM to every steerer"); _common(m, measured=False, mapping=False)
    m.add_argument("--method", choices=("maps", "tracking"), default="maps")
    m.add_argument("--unit", choices=("mm/Tm", "mm/mrad"), default="mm/Tm")


# ---------------------------------------------------------------------------
def _load(args):
    if not Path(args.input).is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr); return None
    lat, cfg, _conv = common.load_input(
        args.input, tracewin_ini=getattr(args, "tracewin_ini", None))
    if getattr(args, "energy", None) is not None:
        cfg.energy = float(args.energy)
    if getattr(args, "freq", None) is not None:
        cfg.frequency = float(args.freq)
    if getattr(args, "species", None) is not None:
        cfg.species = args.species
    common.note_tracewin_ini_overrides(
        getattr(args, "tracewin_ini", None),
        {"energy": getattr(args, "energy", None), "freq": getattr(args, "freq", None),
         "species": getattr(args, "species", None)})
    return lat, cfg


def _deck_path(args, lat):
    """The TraceWin deck behind INPUT (a .lgproj points at it) — needed for the text-surgery export."""
    p = Path(args.input)
    if p.suffix.lower() == ".lgproj":
        from linac_gen.io.project import load_project
        proj = load_project(p); q = Path(proj.lattice_path)
        return q if q.is_absolute() else (p.parent / q)
    return p


def _lgproj_src(args):
    """The project to clone next to an exported deck (``--lgproj``); None (with a note) for a bare lattice input."""
    if not getattr(args, "lgproj", False):
        return None
    if Path(args.input).suffix.lower() == ".lgproj":
        return Path(args.input)
    print("note: --lgproj ignored (INPUT is a lattice file, not a .lgproj project)", file=sys.stderr)
    return None


def _setup(args, lat, cfg):
    from linac_gen.orm import DeviceMap, default_device_map, load_measured, resolve_devices
    if not args.measured:
        print("error: --measured is required (FORMA folder or HELIX ORM CSV)", file=sys.stderr); return None
    meas = load_measured(args.measured)
    dm = DeviceMap.from_json(args.map) if args.map else default_device_map(lat, meas)
    dm.exclude_bpms |= set(args.exclude_bpm); dm.exclude_trims |= set(args.exclude_trim)
    sel = resolve_devices(lat, meas, dm)
    if args.write_map:
        dm.to_json(args.write_map)
    if sel.n_trim == 0 or sel.n_bpm == 0:
        print("error: no measured device could be matched to the lattice (see the mapping report below)", file=sys.stderr)
        _print_mapping(sel, quiet=False); return None
    return meas, dm, sel


def _print_mapping(sel, quiet):
    print(sel.summary())
    if quiet:
        return
    for kind in ("bpm", "trim"):
        if sel.unmatched_devices[kind]:
            print(f"  unmatched {kind} devices: {', '.join(sel.unmatched_devices[kind])}")
        if sel.unmatched_elements[kind]:
            print(f"  lattice {kind}s without data: {', '.join(sel.unmatched_elements[kind])}")
    for n in sel.notes:
        print("  note:", n)


_TRIM_COLS = ("k_Tm_per_A", "dk", "kick_mrad_per_A", "r", "nrms_resid", "n_down", "noise_upstream_mm_per_A", "cross_over_inplane", "W_MeV", "brho_Tm", "included")


def _write_compare(outdir: Path, cmp, sel, plots):
    from linac_gen.orm.measured import write_orm_csv
    out = {"reason": cmp["reason"], "sys_floor": cmp["sys_floor"], "planes": {}}
    for p, d in cmp["planes"].items():
        out["planes"][p] = {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
        write_orm_csv(outdir / f"orm_model_{p}{p}.csv", d["model"], sel.bpm_labels, sel.trim_labels, kick_plane=p, read_plane=p, units="mm/Tm")
        write_orm_csv(outdir / f"orm_measured_{p}{p}.csv", d["aligned"], sel.bpm_labels, sel.trim_labels, kick_plane=p, read_plane=p, units="mm/A", errors=d["aligned_err"])
    (outdir / "orm_compare.json").write_text(json.dumps(_jsonable(out), indent=1), encoding="utf-8")
    with open(outdir / "orm_trim_calibration.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write("trim,plane,device," + ",".join(_TRIM_COLS) + "\n")
        for p, d in cmp["planes"].items():
            for r in d["per_trim"]:
                fh.write(",".join([r["trim"], p, r["device"] or ""] + [_cell(r[k]) for k in _TRIM_COLS]) + "\n")
    if plots:
        from linac_gen.orm import plots as P
        P.fig_heatmaps(cmp, sel).savefig(outdir / "orm_fig1_heatmaps.png", dpi=160, bbox_inches="tight")
        for p in cmp["planes"]:
            P.fig_trends(cmp, sel, p).savefig(outdir / f"orm_fig2_trends_{'H' if p == 'x' else 'V'}.png", dpi=160, bbox_inches="tight")
        P.fig_calibration(cmp, sel).savefig(outdir / "orm_fig4_calibration.png", dpi=160, bbox_inches="tight")


def _cell(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.10g}"
    return str(v)


def _jsonable(o):
    """Strict-JSON copy: numpy scalars/arrays to Python, non-finite floats to null."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _fit_options(args, stage):
    from linac_gen.orm import OrmFitOptions
    return OrmFitOptions(stage=stage, prior_g=args.prior_g, prior_G=args.prior_G, sys_floor=args.sys_floor, fix_quads=tuple(args.fix_quad), max_iter=args.max_iter)


def run(args) -> int:
    from linac_gen.core.cancelled import OperationCancelled
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr); return 130
    except OperationCancelled:
        return 130
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr); return 1


def _run(args) -> int:
    loaded = _load(args)
    if loaded is None:
        return 2
    lat, cfg = loaded
    outdir = Path(args.out) if args.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    mode = args.orm_mode
    if mode == "export":
        from linac_gen.orm import export_recalibrated_deck, load_calibration
        cal = load_calibration(args.calibration); src = _deck_path(args, lat)
        rep = export_recalibrated_deck(src, args.out_deck, cal, lgproj_src=_lgproj_src(args))
        print(f"wrote {rep['dst']}: {len(rep['changed'])} quad(s) rescaled, {rep['n_quads_checked']} verified on reload" + (f", project {rep['lgproj']}" if "lgproj" in rep else ""))
        return 0
    if mode == "model":
        from linac_gen.orm import DeviceMap, OrmModel, resolve_devices
        from linac_gen.orm.measured import MeasuredOrm, write_orm_csv
        from linac_gen.orm.devices import element_label
        from linac_gen.elements.marker import Marker
        from linac_gen.elements.steerer import Steerer
        if not outdir:
            print("error: --out DIR is required for the model matrices", file=sys.stderr); return 2
        bpms = [element_label(e) for e in lat.elements if isinstance(e, Marker) and getattr(e, "is_bpm", False)]
        trims = [element_label(e) for e in lat.elements if isinstance(e, Steerer)]
        if not bpms or not trims:
            print("error: the lattice has no BPM markers or no steerers", file=sys.stderr); return 1
        stub = {p: MeasuredOrm(kick_plane=p, trims=trims, bpms={"x": bpms, "y": bpms}, values={q: np.zeros((len(bpms), len(trims))) for q in "xy"},
                               errors={q: np.zeros((len(bpms), len(trims))) for q in "xy"}) for p in "xy"}
        sel = resolve_devices(lat, stub, DeviceMap(bpms={b: b for b in bpms}, trims={t: t for t in trims}))
        model = OrmModel(lat, cfg, sel); R = model.response() if args.method == "maps" else model.tracked_response()
        for (kp, rp), M in (R.per_Tm if args.unit == "mm/Tm" else R.per_mrad).items():
            write_orm_csv(outdir / f"orm_model_{kp}{rp}.csv", M, sel.bpm_labels, sel.trim_labels, kick_plane=kp, read_plane=rp, units=args.unit)
        print(f"wrote 4 blocks ({sel.n_bpm} BPMs × {sel.n_trim} steerers, {args.unit}, {R.method}) to {outdir}")
        return 0
    st = _setup(args, lat, cfg)
    if st is None:
        return 1
    meas, dm, sel = st
    from linac_gen.orm import OrmModel, compare_orm, summary_lines
    model = OrmModel(lat, cfg, sel); R = model.response()
    _print_mapping(sel, args.quiet)
    cmp = compare_orm(meas, R, sel, model.brho_trim, model.w_trim_MeV, sys_floor=args.sys_floor)
    for ln in summary_lines(cmp):
        print(ln)
    if mode == "compare":
        if args.check_tracking:
            T = model.tracked_response(); print(f"kick-and-read cross-check: largest relative deviation {model.selfcheck(R, T):.2e}")
        if outdir:
            _write_compare(outdir, cmp, sel, args.plots)
        return 0 if cmp["reason"] is None else 1
    if mode == "validate":
        from linac_gen.orm import synthetic_validation
        stage = "trims+quads" if args.stage == "all" else args.stage
        v = synthetic_validation(model, meas, _fit_options(args, stage), seed=args.seed, g_sigma=args.g_sigma, G_sigma=args.G_sigma)
        print(f"synthetic validation (seed {v['seed']}): injected quad errors {100*v['g_rms_injected']:.2f} % rms → recovered to "
              f"{100*v['g_rms_recovered']:.2f} % rms (max {100*v['g_max_recovered']:.2f} %); trims to {100*v['k_rel_rms']:.1f} %"
              + (f"; BPM gains to {100*v['G_rms_recovered']:.2f} %" if stage == "trims+quads+bpms" else ""))
        for lab, t, f in v["worst_quads"]:
            print(f"   {lab}: injected {100*t:+.2f} %, recovered {100*f:+.2f} %")
        if outdir:
            (outdir / "orm_validation.json").write_text(json.dumps(_jsonable({k: v[k] for k in v if k not in ("fit", "g_true", "k_true", "G_true")}), indent=1), encoding="utf-8")
        return 0
    # ---- fit
    from linac_gen.orm import calibration_from_fit, export_recalibrated_deck, fit_orm, run_stages, save_calibration
    if args.stage == "all":
        results = run_stages(model, meas, _fit_options(args, "trims"))
        for st_name, fr in results.items():
            print(f"--- {st_name}"); [print("   " + ln) for ln in fr.summary_lines()]
        fit = results[STAGES[-1]]
    else:
        fit = fit_orm(model, meas, _fit_options(args, args.stage))
        for ln in fit.summary_lines():
            print(ln)
    cal = calibration_from_fit(fit, model, meas, lattice_path=_deck_path(args, lat), beam_cfg=cfg, device_map=dm)
    cal_path = Path(args.calibration) if args.calibration else (outdir / "orm_calibration.json" if outdir else None)
    if cal_path:
        save_calibration(cal_path, cal); print(f"calibration written: {cal_path}")
    else:
        print("note: calibration not saved (give --out DIR or --calibration FILE)", file=sys.stderr)
    if outdir:
        _write_compare(outdir, cmp, sel, False)
        (outdir / "orm_fit_summary.txt").write_text("\n".join(fit.summary_lines()) + "\n", encoding="utf-8")
        if args.plots:
            from linac_gen.orm import plots as P
            P.fig_fit_quads(fit, model).savefig(outdir / "fit_fig1_quads.png", dpi=160, bbox_inches="tight")
            P.fig_fit_residuals(fit, sel, model.brho_trim).savefig(outdir / "fit_fig2_residuals.png", dpi=160, bbox_inches="tight")
            P.fig_residual_maps(fit, sel, cmp).savefig(outdir / "fit_fig4_residual_maps.png", dpi=160, bbox_inches="tight")
            for p in fit.metrics_after:
                P.fig_trends(cmp, sel, p, fitted=fit.fitted_block(p), r_per_trim=fit.metrics_after[p]["r_per_trim"],
                             model_label=f"fitted model ({fit.stage})").savefig(outdir / f"fit_fig3_trends_{'H' if p == 'x' else 'V'}.png", dpi=160, bbox_inches="tight")
    if args.export_deck:
        rep = export_recalibrated_deck(_deck_path(args, lat), args.export_deck, cal, lgproj_src=_lgproj_src(args))
        print(f"recalibrated deck written: {rep['dst']} ({len(rep['changed'])} quads rescaled, verified on reload)" + (f", project {rep['lgproj']}" if "lgproj" in rep else ""))
    return 0 if fit.converged or fit.stage == "trims" else 1
