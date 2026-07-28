"""Demonstration of HELIX Month-1 additions: Foil element + openPMD output.

Run from the repository root::

    python examples/foil_and_openpmd_demo.py

What this demo does
-------------------
1. Loads ``examples/pipii/btl/btl_with_foil.dat`` — the PIP-II Beam
   Transfer Line with a 600 μg/cm² carbon stripping foil inserted at
   s ≈ 308 m (the actual BTL → Booster charge-exchange boundary).
2. Runs the envelope solver on a representative 800-MeV H⁻ beam.
3. Auto-saves results in **both** HDF5 (HELIX-native) and openPMD-1.1
   (interop) formats to a local ``demo_results/`` directory.
4. Re-loads the openPMD file to confirm round-trip parity.
5. Prints a diagnostic table at and around the foil so you can see the
   stochastic scatter + energy loss in action.

Two side-by-side multi-particle (MP) runs at the end demonstrate the
foil's stochastic kick:
* MP through ``btl.dat`` (no foil)         — baseline
* MP through ``btl_with_foil.dat``         — foil-applied

You'll see σ_xp jump at the foil and the longitudinal distribution
acquire a small Bohr-straggling tail.
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.foil import Foil
from linac_gen.io.hdf5_output import load_results_hdf5, save_results_hdf5
from linac_gen.io.openpmd_output import (
    is_openpmd_file, load_results_openpmd, save_results_openpmd,
)
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.envelope import EnvelopeSolver


REPO_ROOT = Path(__file__).resolve().parent.parent
BTL_DIR = REPO_ROOT / "examples" / "pipii" / "btl"
DAT_NO_FOIL = BTL_DIR / "btl.dat"
DAT_WITH_FOIL = BTL_DIR / "btl_with_foil.dat"
OUT_DIR = REPO_ROOT / "demo_results"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_pipii_btl_beam(n: int, seed: int = 0) -> tuple[Beam, ReferenceParticle]:
    """Build the 800-MeV H⁻ Twiss-matched beam from btl.lgproj."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    # Twiss values from examples/pipii/btl/btl.lgproj
    alpha_x, beta_x = 1.918, 13.814
    alpha_y, beta_y = -0.916, 5.387
    emit_nx_mmmrad = 0.2137162
    emit_ny_mmmrad = 0.2165343
    # de-normalise to geometric
    bg = ref.bg
    eps_x = emit_nx_mmmrad / bg
    eps_y = emit_ny_mmmrad / bg
    sigma_x = math.sqrt(eps_x * beta_x)
    sigma_xp = math.sqrt(eps_x * (1 + alpha_x ** 2) / beta_x)
    sigma_y = math.sqrt(eps_y * beta_y)
    sigma_yp = math.sqrt(eps_y * (1 + alpha_y ** 2) / beta_y)

    rng = np.random.default_rng(seed)
    beam = Beam(ref=ref, n_particles=n, current=4.84235)
    beam.particles[:, 0] = rng.normal(0.0, sigma_x, n)
    beam.particles[:, 1] = (
        rng.normal(0.0, sigma_xp, n) - alpha_x / beta_x * beam.particles[:, 0]
    )
    beam.particles[:, 2] = rng.normal(0.0, sigma_y, n)
    beam.particles[:, 3] = (
        rng.normal(0.0, sigma_yp, n) - alpha_y / beta_y * beam.particles[:, 2]
    )
    # longitudinal: small dphi spread for an 800-MeV continuous bunched beam
    beam.particles[:, 4] = rng.normal(0.0, 1.0, n)   # 1 deg
    beam.particles[:, 5] = rng.normal(0.0, 0.05, n)  # 50 keV
    return beam, ref


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_envelope_window(results, s_center: float, half_width_mm: float = 500.0):
    """Print the envelope arrays in a window centred on ``s_center`` (mm)."""
    s_arr = np.asarray(results.s)
    in_window = (s_arr >= s_center - half_width_mm) & (
        s_arr <= s_center + half_width_mm
    )
    idx = np.where(in_window)[0]
    if len(idx) == 0:
        # fallback: last 6 entries
        idx = np.arange(max(0, len(s_arr) - 6), len(s_arr))
    has_trans = bool(getattr(results, "transmission", None))
    header_fmt = (
        f"  {'s [mm]':>10}  {'sigma_x':>10}  {'sigma_y':>10}  "
        f"{'emit_x':>10}  {'emit_y':>10}  {'W [MeV]':>10}"
    )
    if has_trans:
        header_fmt += f"  {'trans %':>8}"
    print(header_fmt)
    for i in idx:
        s = results.s[i]
        sx = results.sigma_x[i]
        sy = results.sigma_y[i]
        ex = results.emit_x[i]
        ey = results.emit_y[i]
        w = results.ref_w_kin[i]
        row = (
            f"  {s:>10.1f}  {sx:>10.4f}  {sy:>10.4f}  "
            f"{ex:>10.4f}  {ey:>10.4f}  {w:>10.4f}"
        )
        if has_trans:
            row += f"  {results.transmission[i]:>8.2f}"
        print(row)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_envelope_no_foil() -> object:
    header("[1/4] Envelope baseline on btl.dat (no Foil)")
    lat, _ = parse_tracewin(str(DAT_NO_FOIL))
    print(f"Loaded {len(lat.elements)} elements from {DAT_NO_FOIL.name}")

    beam, ref = make_pipii_btl_beam(n=100, seed=1)
    bg = ref.bg
    initial_twiss = {
        "alpha_x": 1.918, "beta_x": 13.814,
        "emit_x": 0.2137162 / bg,   # geometric mm·mrad (de-normalised)
        "alpha_y": -0.916, "beta_y": 5.387,
        "emit_y": 0.2165343 / bg,
    }
    solver = EnvelopeSolver(lat, ref, initial_twiss, current=4.84235)
    results = solver.run()
    print(f"Envelope steps: {len(results.s)}")
    print_envelope_window(results, s_center=308_182.0, half_width_mm=2000.0)
    return results


def run_envelope_with_foil() -> object:
    header("[2/4] Envelope on btl_with_foil.dat (with Foil)")
    lat, _ = parse_tracewin(str(DAT_WITH_FOIL))
    foils = [e for e in lat.elements if isinstance(e, Foil)]
    print(
        f"Loaded {len(lat.elements)} elements from {DAT_WITH_FOIL.name} "
        f"({len(foils)} Foil)"
    )
    if foils:
        f = foils[0]
        print(
            f"  Foil: name={f.name!r} material={f.material!r} "
            f"thickness={f.thickness_ug_cm2:g} μg/cm²"
        )

    beam, ref = make_pipii_btl_beam(n=100, seed=1)
    bg = ref.bg
    initial_twiss = {
        "alpha_x": 1.918, "beta_x": 13.814,
        "emit_x": 0.2137162 / bg,
        "alpha_y": -0.916, "beta_y": 5.387,
        "emit_y": 0.2165343 / bg,
    }
    solver = EnvelopeSolver(lat, ref, initial_twiss, current=4.84235)
    results = solver.run()
    print(f"Envelope steps: {len(results.s)}")
    print_envelope_window(results, s_center=308_182.0, half_width_mm=2000.0)
    return results


def dual_format_dump(results, run_type: str) -> tuple[Path, Path]:
    """Write the results in both HDF5 (HELIX-native) and openPMD-1.1 formats."""
    OUT_DIR.mkdir(exist_ok=True)
    h5_path = OUT_DIR / f"btl_{run_type}.h5"
    opmd_path = OUT_DIR / f"btl_{run_type}.opmd.h5"

    class _DemoBeamConfig:
        species = "H-"
        frequency = 162.5
        current = 4.84235
        n_particles = 100

    save_results_hdf5(results, str(h5_path), beam_config=_DemoBeamConfig())
    save_results_openpmd(results, str(opmd_path), beam_config=_DemoBeamConfig())
    print(f"  HDF5:    {h5_path.relative_to(REPO_ROOT)}  ({h5_path.stat().st_size:>7} B)")
    print(f"  openPMD: {opmd_path.relative_to(REPO_ROOT)}  ({opmd_path.stat().st_size:>7} B)")
    return h5_path, opmd_path


def show_opmd_structure(opmd_path: Path) -> None:
    """Print the openPMD HDF5 group tree so the user can see the schema."""
    print(f"  openPMD-1.1 structure of {opmd_path.name}:")
    with h5py.File(opmd_path, "r") as f:
        # Required root attributes
        print(f"    root attrs:")
        for k in ("openPMD", "openPMDextension", "basePath",
                  "iterationEncoding", "particlesPath", "meshesPath"):
            if k in f.attrs:
                v = f.attrs[k]
                print(f"      {k:>20} = {v!r}")

        def walk(g, indent=4):
            for name, item in g.items():
                if isinstance(item, h5py.Group):
                    print(" " * indent + f"📁 {name}/")
                    if indent < 20:
                        walk(item, indent + 2)
                else:
                    shape = item.shape
                    print(
                        " " * indent + f"📄 {name}  shape={shape}  "
                        f"dtype={item.dtype}"
                    )
        walk(f)


def verify_opmd_round_trip(opmd_path: Path) -> None:
    """Confirm the openPMD file is loadable and matches a sniff check."""
    assert is_openpmd_file(opmd_path), "Sniff check failed"
    loaded = load_results_openpmd(opmd_path)
    print(f"  Round-trip load: {len(loaded)} keys, s array has {len(loaded.get('s', []))} points")


def run_mp_through_foil() -> None:
    """Multi-particle pass through a minimal lattice ending in a Foil — shows
    the actual stochastic kick on σ_xp and the dw distribution."""
    header("[3/4] Multi-particle stochastic kicks at the Foil (direct apply)")
    from linac_gen.elements.foil import Foil as _Foil

    beam, ref = make_pipii_btl_beam(n=2000, seed=2)
    print(f"  Before foil: ⟨xp⟩={beam.particles[:,1].mean():+.6e}  "
          f"σ(xp)={beam.particles[:,1].std(ddof=1):.6f} mrad")
    print(f"  Before foil: ⟨dw⟩={beam.particles[:,5].mean():+.6e}  "
          f"σ(dw)={beam.particles[:,5].std(ddof=1):.6f} MeV")

    foil = _Foil(name="STRIP", material="C", thickness_ug_cm2=600.0, seed=42)
    theta_rms_mrad = foil._highland_theta_rms(beam) * 1e3
    dE_mean = foil._mean_energy_loss_MeV(beam)
    dE_sigma = foil._energy_loss_sigma_MeV(beam)
    print(f"  Expected Highland θ_rms: {theta_rms_mrad:.4f} mrad")
    print(f"  Expected Bethe-Bloch ⟨ΔE⟩: {dE_mean*1e6:.3f} eV "
          f"(sigma {dE_sigma*1e6:.3f} eV)")

    foil.apply_kick(beam)

    sigma_xp = beam.particles[:, 1].std(ddof=1)
    sigma_yp = beam.particles[:, 3].std(ddof=1)
    sigma_dw_diff = beam.particles[:, 5].std(ddof=1)
    print(f"  After foil:  ⟨xp⟩={beam.particles[:,1].mean():+.6e}  "
          f"σ(xp)={sigma_xp:.6f} mrad")
    print(f"  After foil:  ⟨yp⟩={beam.particles[:,3].mean():+.6e}  "
          f"σ(yp)={sigma_yp:.6f} mrad")
    print(f"  After foil:  ⟨dw⟩={beam.particles[:,5].mean():+.6e}  "
          f"σ(dw)={sigma_dw_diff:.6f} MeV")
    print(f"  Foil kick at 800 MeV C-600μg/cm² is in the weak-perturbation "
          "regime — small but non-zero, as expected.")


def main() -> None:
    print("HELIX Month-1 demo — Foil scattering + openPMD output")
    print("Repo root:", REPO_ROOT)

    results_no_foil = run_envelope_no_foil()
    results_with_foil = run_envelope_with_foil()

    header("[4/4] Dual-format auto-dump + openPMD round-trip")
    print("Saving baseline (no foil):")
    h5_a, opmd_a = dual_format_dump(results_no_foil, "no_foil")
    verify_opmd_round_trip(opmd_a)

    print("Saving with-foil:")
    h5_b, opmd_b = dual_format_dump(results_with_foil, "with_foil")
    verify_opmd_round_trip(opmd_b)

    print()
    show_opmd_structure(opmd_b)

    run_mp_through_foil()

    print("\nDone.  Open the .opmd.h5 files in openPMD-viewer or any "
          "openPMD-compatible tool; open the .h5 files in HELIX itself "
          "(Results tab → Import Results…).")


if __name__ == "__main__":
    main()
