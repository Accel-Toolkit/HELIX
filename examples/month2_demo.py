"""Demonstration of HELIX Month-2 additions: CSR + MAD-X import.

Run from the repository root::

    python examples/month2_demo.py

What this demo does
-------------------
Part A — MAD-X import
  Loads the two bundled MAD-X files (examples/madx/fodo.madx and
  examples/madx/transport.madx) through HELIX's in-house MAD-X subset
  parser and prints the converted HELIX lattices.  In the GUI the same
  thing happens via  File -> Open Lattice...  (the dialog accepts
  *.madx / *.seq).

Part B — CSR (Coherent Synchrotron Radiation)
  Loads examples/csr_chicane.dat (a compact 6-bend chicane) and runs the
  multi-particle solver twice — CSR off, then CSR on — and prints the
  resulting longitudinal-emittance and energy-spread growth.  In the GUI
  this is the "CSR in bends" checkbox on the Convergence tab; open
  examples/csr_chicane.lgproj (CSR already enabled) and Run Multi-particle.

The CSR effect also shows up in the Results-tab plots: open the
Emittance popup (epsilon_z), the RMS popup (sigma_w energy spread) and
the Energy popup after a run — CSR-on curves sit above CSR-off.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.io.madx_parser import parse_madx
from linac_gen.io.tracewin_parser import parse_tracewin

REPO = Path(__file__).resolve().parent.parent


def header(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
# Part A — MAD-X import
# --------------------------------------------------------------------------

def demo_madx_import() -> None:
    header("[A] MAD-X import — in-house subset parser")
    for rel in ("examples/madx/fodo.madx", "examples/madx/transport.madx"):
        lat, meta = parse_madx(str(REPO / rel))
        types = Counter(type(e).__name__ for e in lat.elements)
        print(f"\n  {rel}")
        print(f"    title       : {meta['title']}")
        ref = meta["reference"]
        print(f"    beam        : {ref.species.name}, "
              f"W_kin = {ref.w_kin:.1f} MeV")
        print(f"    elements    : {len(lat.elements)}  "
              f"(total {lat.total_length:.0f} mm)")
        print(f"    composition : {dict(types)}")
        if meta["warnings"]:
            for w in meta["warnings"]:
                print(f"    warning     : {w}")
        else:
            print(f"    warnings    : none")
    print("\n  -> In the GUI: File -> Open Lattice... accepts these "
          ".madx files directly.")


# --------------------------------------------------------------------------
# Part B — CSR
# --------------------------------------------------------------------------

def _short_bunch(n: int, seed: int = 0) -> Beam:
    """An 800-MeV H- bunch — deliberately short so CSR is visible."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=5.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, 1.0, n)     # x  mm
    beam.particles[:, 1] = rng.normal(0.0, 0.3, n)     # xp mrad
    beam.particles[:, 2] = rng.normal(0.0, 1.0, n)     # y  mm
    beam.particles[:, 3] = rng.normal(0.0, 0.3, n)     # yp mrad
    beam.particles[:, 4] = rng.normal(0.0, 0.6, n)     # dphi deg (short)
    beam.particles[:, 5] = rng.normal(0.0, 0.005, n)   # dw  MeV
    return beam


def demo_csr() -> None:
    header("[B] CSR — 6-bend chicane, multi-particle, CSR off vs on")
    lat_path = REPO / "examples" / "csr_chicane.dat"
    lattice, _ = parse_tracewin(str(lat_path))
    n_bends = sum(1 for e in lattice.elements
                  if type(e).__name__ == "Dipole")
    print(f"  Lattice: {lat_path.name} — {len(lattice.elements)} elements, "
          f"{n_bends} dipoles, {lattice.total_length:.0f} mm")

    def run(csr_on: bool):
        # Fresh lattice + beam each run so the two are independent.
        lat, _ = parse_tracewin(str(lat_path))
        sc = SpaceChargeConfig(
            nx=48, ny=48, nz=48, grid_extent=5.0, kernel="cic",
            use_gpu="cpu", csr_enabled=csr_on, csr_bins=200,
        )
        res = Simulation(lat, _short_bunch(20_000, seed=1),
                         space_charge=sc).run()
        return res

    res_off = run(csr_on=False)
    res_on = run(csr_on=True)

    ez_off, ez_on = res_off.emit_z[-1], res_on.emit_z[-1]
    sw_off, sw_on = res_off.sigma_w[-1], res_on.sigma_w[-1]
    w_off, w_on = res_off.ref_w_kin[-1], res_on.ref_w_kin[-1]

    print()
    print(f"  {'quantity':<26}{'CSR off':>14}{'CSR on':>14}{'delta':>12}")
    print(f"  {'-'*64}")
    print(f"  {'emit_z [deg.MeV]':<26}{ez_off:>14.6f}{ez_on:>14.6f}"
          f"{(ez_on-ez_off)/ez_off*100:>11.2f}%")
    print(f"  {'sigma_w [MeV]':<26}{sw_off:>14.6f}{sw_on:>14.6f}"
          f"{(sw_on-sw_off)/max(sw_off,1e-12)*100:>11.2f}%")
    print(f"  {'ref W_kin [MeV]':<26}{w_off:>14.6f}{w_on:>14.6f}"
          f"{(w_on-w_off)*1e3:>10.3f} keV")
    print()
    if ez_on >= ez_off:
        print("  -> CSR raises the longitudinal emittance, as expected.")
    else:
        print("  -> NOTE: CSR did not raise emit_z for this configuration.")
    print("  -> In the GUI: open examples/csr_chicane.lgproj (CSR already "
          "enabled),\n     Run Multi-particle, then view the Emittance / RMS "
          "/ Energy popups.")


def main() -> None:
    print("HELIX Month-2 demo — CSR + MAD-X import")
    print("Repo root:", REPO)
    demo_madx_import()
    demo_csr()
    print("\nDone.")


if __name__ == "__main__":
    main()
