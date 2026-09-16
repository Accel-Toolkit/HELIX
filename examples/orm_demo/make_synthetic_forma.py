"""Write a synthetic FORMA-layout measurement from the HELIX model of a deck (default: the demo FODO next to this script).

    PYTHONPATH=.:gui python examples/orm_demo/make_synthetic_forma.py [--input examples/orm_demo/fodo_orm.lgproj]
        [--out examples/orm_demo/synthetic_forma] [--g-sigma 0.03] [--noise 0.01] [--seed 1]

Every BPM marker and steerer of the deck becomes a device (``L:<label>H``-style
names are NOT used — the HELIX labels are the device names, so the built-in
identity mapping applies); the quads get random scale errors, the trims random
signed calibrations (T·m per A), the BPM readings Gaussian noise.  The folder
pair (``<stamp>_H``, ``<stamp>_V``) can then be fed to ``python -m linac_gen orm``
exactly like a machine measurement, and the injected errors are recorded in
``truth.json`` for comparison.
"""
from __future__ import annotations
import argparse, contextlib, io, json
from pathlib import Path
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(Path(__file__).with_name("fodo_orm.lgproj")), help="a .lgproj (default: the demo FODO)")
    ap.add_argument("--out", default=str(Path(__file__).with_name("synthetic_forma")))
    ap.add_argument("--g-sigma", type=float, default=0.03); ap.add_argument("--noise", type=float, default=0.01, help="rms noise as a fraction of each column's maximum")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    from linac_gen.cli import common
    from linac_gen.elements.marker import Marker
    from linac_gen.elements.steerer import Steerer
    from linac_gen.orm import DeviceMap, OrmModel, resolve_devices
    from linac_gen.orm.devices import element_label
    from linac_gen.orm.measured import MeasuredOrm
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, _ = common.load_input(a.input)
    bpms = [element_label(e) for e in lat.elements if isinstance(e, Marker) and getattr(e, "is_bpm", False)]
    trims = [element_label(e) for e in lat.elements if isinstance(e, Steerer)]
    stub = {p: MeasuredOrm(kick_plane=p, trims=trims, bpms={"x": bpms, "y": bpms}, values={q: np.zeros((len(bpms), len(trims))) for q in "xy"},
                           errors={q: np.zeros((len(bpms), len(trims))) for q in "xy"}) for p in "xy"}
    sel = resolve_devices(lat, stub, DeviceMap(bpms={b: b for b in bpms}, trims={t: t for t in trims}))
    model = OrmModel(lat, cfg, sel)
    rng = np.random.default_rng(a.seed)
    g_true = 1.0 + rng.normal(0, a.g_sigma, len(model.quads)); R = model.response(g_true)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); truth = {"quads": dict(zip(model.quad_labels, g_true.tolist())), "trims": {}}
    stamp = "20260101_000000"
    for p, tag in (("x", "H"), ("y", "V")):
        k = model.kick_sign * rng.uniform(3e-4, 8e-4, sel.n_trim); truth["trims"][p] = dict(zip(sel.trim_labels, k.tolist()))
        folder = out / f"{stamp}_{tag}"; folder.mkdir(exist_ok=True)
        for rp, kind in (("x", "horizontal"), ("y", "vertical")):
            A = k[None, :] * R.per_Tm[(p, rp)]; E = a.noise * np.max(np.abs(A), axis=0)[None, :] * np.ones_like(A) + 1e-6
            A = A + rng.normal(0, 1, A.shape) * E
            for name, M in ((f"response_matrix_{stamp}_{kind}.csv", A), (f"response_matrix_{stamp}_{kind}_error.csv", E)):
                with open(folder / name, "w", encoding="utf-8", newline="") as fh:
                    fh.write("Reading Device," + ",".join(sel.trim_labels) + "\n")
                    for b, row in zip(sel.bpm_labels, M):
                        fh.write(b + "," + ",".join(f"{v:.6g}" for v in row) + "\n")
        (folder / "scan_info.json").write_text(json.dumps({"forma_version": "synthetic (HELIX model)", "baseline": {"dead_bpms": []}, "kick_plane": p}), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"wrote {out}: {sel.n_bpm} BPMs × {sel.n_trim} trims per plane, {len(model.quads)} quads perturbed ({a.g_sigma:.0%} rms), truth.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
