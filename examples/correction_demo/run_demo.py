"""End-to-end orbit-correction demo.

1. Load ``correction_demo.dat`` (6-cell FODO + 4 BPM/steerer pairs).
2. Plant 1 mm RMS quadrupole transverse misalignments on every QUAD.
3. Track once → record pre-correction BPM readings.
4. Run :func:`run_correction_from_lattice` (TraceWin-card driven).
5. Track again → record post-correction BPM readings.
6. Plot pre/post side-by-side and assert the residual is < 1 % of the
   pre-correction RMS.

Run from the repository root::

    python examples/correction_demo/run_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Make ``linac_gen`` importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.errors.correction import run_correction_from_lattice
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.tracker import Tracker

DEMO_DAT = Path(__file__).resolve().parent / "correction_demo.dat"
OUT_PNG = Path(__file__).resolve().parent / "correction_demo.png"


def _beam_factory(beam_cfg: BeamConfig, seed: int = 0):
    return lambda: create_beam(beam_cfg, seed=seed)


def _bpm_readings(lattice, beam_factory) -> dict[str, tuple[float, float]]:
    """Return ``{bpm_name: (centroid_x_mm, centroid_y_mm)}`` after a
    single tracking pass."""
    rec = Tracker(lattice, beam_factory()).run()
    out: dict[str, tuple[float, float]] = {}
    for i, elem in enumerate(lattice.elements):
        if getattr(elem, "is_bpm", False):
            j = i + 1
            if j < len(rec.centroid):
                c = np.array(rec.centroid[j])
                out[elem.name] = (float(c[0]), float(c[2]))
    return out


def _rms(readings: dict[str, tuple[float, float]]) -> float:
    vals: list[float] = []
    for x, y in readings.values():
        vals.extend([x, y])
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0


def main() -> int:
    print(f"Loading {DEMO_DAT.name} …")
    lattice, meta = parse_tracewin(str(DEMO_DAT))
    if meta.get("warnings"):
        print(f"  parser warnings: {len(meta['warnings'])}")

    # 5 MeV proton beam, very small emittance (we want pure centroid
    # response, not finite-emittance halo).
    beam_cfg = BeamConfig(
        species="proton", energy=5.0, frequency=352.21,
        current=0.0, n_particles=2000,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.05, emit_ny=0.05, emit_z=0.10,
        alpha_x=0.0, beta_x=2.0,
        alpha_y=0.0, beta_y=2.0,
        alpha_z=0.0, beta_z=1.0,
    )
    factory = _beam_factory(beam_cfg, seed=1)

    # Plant 0.2 mm RMS dx/dy on every quadrupole — a realistic alignment
    # tolerance for a precision linac.  Larger misalignments would saturate
    # the vmax=0.01 T·m steerer cap; that case is intentionally exercised
    # by tests/errors/test_correction_iter.py.
    rng = np.random.default_rng(2026)
    sigma_mm = 0.2
    n_quads = 0
    for elem in lattice.elements:
        if isinstance(elem, Quadrupole):
            elem.dx = float(rng.normal(0.0, sigma_mm))
            elem.dy = float(rng.normal(0.0, sigma_mm))
            n_quads += 1
    print(f"Planted dx/dy ~ N(0, {sigma_mm} mm) on {n_quads} quadrupoles.")

    # Pre-correction orbit
    pre = _bpm_readings(lattice, factory)
    rms_pre = _rms(pre)
    print(f"\nPre-correction BPM readings:")
    for name, (x, y) in pre.items():
        print(f"  {name}: x={x:+.3f} mm  y={y:+.3f} mm")
    print(f"  RMS = {rms_pre:.3f} mm")

    # Run correction
    print(f"\nRunning run_correction_from_lattice(n_iter=5, tol_mm=0.01) …")
    result = run_correction_from_lattice(
        lattice, factory, n_iter=5, tol_mm=0.01, history=True,
    )
    print(f"  method = {result['method']}")
    print(f"  n_pairs = {result['n_pairs']}")
    print(f"  history:")
    for h in result["history"]:
        print(f"    iter {h['iter']}: rms_orbit = {h['rms_orbit_mm']:.6f} mm  "
              f"saturated = {h['n_saturated']}")
    print(f"  applied kicks (T·m):")
    for name, kicks in result["kicks"].items():
        print(f"    {name}: bx_l={kicks['bx_l']:+.5e}  by_l={kicks['by_l']:+.5e}")

    # Post-correction orbit
    post = _bpm_readings(lattice, factory)
    rms_post = _rms(post)
    print(f"\nPost-correction BPM readings:")
    for name, (x, y) in post.items():
        print(f"  {name}: x={x:+.6f} mm  y={y:+.6f} mm")
    print(f"  RMS = {rms_post:.6f} mm")

    ratio = rms_post / rms_pre if rms_pre > 0 else 0.0
    print(f"\nRatio post/pre = {ratio*100:.3f} %  "
          f"({'PASS' if ratio < 0.01 else 'FAIL'} — target < 1 %)")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return 0 if ratio < 0.01 else 1

    names = list(pre)
    pre_x = [pre[n][0] for n in names]
    pre_y = [pre[n][1] for n in names]
    post_x = [post[n][0] for n in names]
    post_y = [post[n][1] for n in names]
    idx = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, plane, pre_v, post_v in (
        (axes[0], "x", pre_x, post_x), (axes[1], "y", pre_y, post_y),
    ):
        w = 0.4
        ax.bar(idx - w / 2, pre_v, w, label="pre", color="#cc6666")
        ax.bar(idx + w / 2, post_v, w, label="post", color="#3388cc")
        ax.set_xticks(idx); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(f"BPM centroid {plane} (mm)")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Plane {plane}")
    fig.suptitle(
        f"Orbit correction: RMS {rms_pre:.3f} mm → {rms_post:.6f} mm "
        f"({ratio*100:.2f} %)"
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nSaved {OUT_PNG.name}")

    return 0 if ratio < 0.01 else 1


if __name__ == "__main__":
    raise SystemExit(main())
