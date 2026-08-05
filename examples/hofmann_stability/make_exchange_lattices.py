"""Generate the emittance-exchange validation pair (decks + GUI projects).

Two 150-cell channels, identical beam (20 mA proton, eps_z/eps_x = 3,
SC-matched injection) and identical geometry except the quad gradient:

  exchange_resonant.(dat|lgproj)  quads ±40 T/m — depressed working point
      (R = 1.16, Y = 0.41) ON the Hofmann l=2 coupling band
      (gamma/nu_0x = 0.099, S^2 = 5.9): multiparticle tracking shows
      z -> x/y emittance exchange over the first ~20 cells.
  exchange_control.(dat|lgproj)   quads ±44 T/m — (R = 0.89, Y = 0.45),
      chart quiet: no exchange beyond the initial redistribution
      transient.

The .lgproj beams carry the SPACE-CHARGE-MATCHED per-plane Twiss at
20 mA (find_matched_period_sigma), so a GUI envelope or MP run starts on
the periodic orbit.  Run from the repo root:

    PYTHONPATH=. python examples/hofmann_stability/make_exchange_lattices.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.matching.periodic import find_matched_period_sigma
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix,
    compute_twiss,
)

HERE = Path(__file__).resolve().parent
CURRENT = 20.0
N_CELLS = 150
EMIT_T = 2.0      # geometric transverse emittance, mm.mrad
EMIT_Z = 0.080    # native longitudinal emittance, deg.MeV (eps_z/eps_x ~ 3)
APERTURE = 20.0

CHANNELS = [
    ("exchange_resonant", 40.0,
     "ON the l=2 coupling band: (R=1.16, Y=0.41), gamma/nu_0x=0.099"),
    ("exchange_control", 44.0,
     "chart-quiet control: (R=0.89, Y=0.45), all branches zero"),
]


def _cell_lines(grad: float) -> list[str]:
    v = 0.10 * 1e6                          # GAP voltage is in volts
    return [
        f"DRIFT 30.0 {APERTURE}",
        f"QUAD 40.0 {grad:+.1f} {APERTURE}",
        f"DRIFT 30.0 {APERTURE}",
        f"GAP {v:.1f} -90.0 {APERTURE}",
        f"DRIFT 30.0 {APERTURE}",
        f"QUAD 40.0 {-grad:+.1f} {APERTURE}",
        f"DRIFT 30.0 {APERTURE}",
    ]


def _build_api(grad: float, n_cells: int):
    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=30.0, aperture=APERTURE))
        lat.add(Quadrupole(name="QF", length=40.0, gradient=+grad,
                           aperture=APERTURE))
        lat.add(Drift(name="D", length=30.0, aperture=APERTURE))
        lat.add(RFGap(name="G", voltage=0.10, phase=-90.0, frequency=162.5))
        lat.add(Drift(name="D", length=30.0, aperture=APERTURE))
        lat.add(Quadrupole(name="QD", length=40.0, gradient=-grad,
                           aperture=APERTURE))
        lat.add(Drift(name="D", length=30.0, aperture=APERTURE))
    per = PeriodicStructure(start=0, end=7 * n_cells, inner_period_length=7,
                            inner_slice_end=7, n_repeats=n_cells,
                            label="cell", source="manual")
    return lat, per


def _matched_beam(grad: float) -> dict:
    """SC-matched per-plane Twiss at CURRENT (10-cell probe lattice)."""
    lat, per = _build_api(grad, 10)
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    M = compute_transfer_matrix(lat, ref, start=0, end=6)
    twx, twy, twz = (compute_twiss(M, p) for p in ("x", "y", "z"))
    base = dict(alpha_x=twx["alpha"], beta_x=twx["beta"], emit_x=EMIT_T,
                alpha_y=twy["alpha"], beta_y=twy["beta"], emit_y=EMIT_T,
                alpha_z=twz["alpha"], beta_z=twz["beta"], emit_z=EMIT_Z)
    ms = find_matched_period_sigma(lat, ref, per, CURRENT, base)
    assert ms["converged"], f"matched Sigma not converged for grad={grad}"
    S = ms["sigma_entry"]

    def tw(i):
        blk = S[np.ix_([2 * i, 2 * i + 1], [2 * i, 2 * i + 1])]
        e = float(np.sqrt(max(np.linalg.det(blk), 1e-30)))
        return float(-blk[0, 1] / e), float(blk[0, 0] / e), e

    ax, bx, ex = tw(0)
    ay, by, ey = tw(1)
    az, bz, ez = tw(2)
    bg = ref.beta * ref.gamma
    return {
        "species": "proton", "energy": 2.5, "frequency": 162.5,
        "current": CURRENT, "duty_cycle": 100.0, "n_particles": 20000,
        "distribution": "gaussian", "cutoff": 4.0,
        "emit_nx": round(ex * bg, 6), "alpha_x": round(ax, 6),
        "beta_x": round(bx, 6),
        "emit_ny": round(ey * bg, 6), "alpha_y": round(ay, 6),
        "beta_y": round(by, 6),
        "emit_z": round(ez, 6), "alpha_z": round(az, 6),
        "beta_z": round(bz, 6),
        "centroid_x": 0.0, "centroid_xp": 0.0, "centroid_y": 0.0,
        "centroid_yp": 0.0, "centroid_dphi": 0.0, "centroid_dw": 0.0,
        "mismatch_x": 0.0, "mismatch_y": 0.0, "mismatch_z": 0.0,
        "source": "generate", "distribution_file": None,
        "continuous": False, "dc_energy_spread_keV": 0.0,
    }


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, grad, note in CHANNELS:
            lines = [
                f"; {name} — generated by make_exchange_lattices.py",
                f"; {note}",
                "TITLE Hofmann emittance-exchange validation "
                f"({name.split('_')[1]})",
                "FREQ 162.5",
                "LATTICE 7 0",
            ]
            for _ in range(N_CELLS):
                lines += _cell_lines(grad)
            lines += ["LATTICE_END", "END"]
            (HERE / f"{name}.dat").write_text("\n".join(lines) + "\n")

            proj = {
                "__kind__": "linac_gen_project",
                "__version__": 1,
                "lattice_path": f"{name}.dat",
                "beam": _matched_beam(grad),
            }
            with open(HERE / f"{name}.lgproj", "w") as fh:
                json.dump(proj, fh, indent=2)
                fh.write("\n")
            print(f"wrote {name}.dat + {name}.lgproj "
                  f"(quads ±{grad:.0f} T/m, {N_CELLS} cells)")


if __name__ == "__main__":
    main()
