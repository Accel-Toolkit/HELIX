"""Reference-data loaders for the RFQ benchmark suite.

Ground-truth sources (present on the development machine only — every
test that uses them must skip cleanly when they are absent; none of
these files may ever be committed):

- In-repo, read-only: ``Tracewin_code/LEBT+RFQ_ENV+{NOSC,SC}.txt``
  (26-column TraceWin "Save data" envelope export of the full LEBT+RFQ
  line, 0 – 6.402673 m) and ``Tracewin_code/LEBT+RFQEnergy.txt``.
- External PXIE project: ``~/Desktop/Projects/PIP_II/Paper/LEBT+RFQ/``
  — the complete TraceWin+Toutatis project of the PIP-II/PXIE RFQ
  (162.5 MHz, 30 keV H⁻, V=60 kV, 4.447 m).

One-time axis audit (2026-07-30) of what each reference file covers:

- ``Chart_Transmission(%).txt`` ends at x = 1.9561 m = the LEBT exit.
  Its final 77.191 % is 100 % minus LEBT scraping of the MEASURED
  Allison-scanner input distribution (all loss occurs between 0.169 m
  and 0.312 m).  It contains NO RFQ information.
- ``partran1.out`` is the LEBT-ONLY multiparticle run (174 rows, ends
  z = 1.9561 m, 77 287 / 100 000 alive).
- ``Chart_Energy(MeV).txt``, ``tracewin.out`` and the two ENV exports
  cover the full LEBT+RFQ line to 6.402673 m; final γ−1 = 2.082166e−3
  → 1.955717 MeV (envelope mode).
**NEVER infer a .dst file's PLANE from its name.**  TraceWin names the
particle files it writes after the run/section they belong to, not after
the location the particles sit at, so an "rfq" file need not be at the
RFQ.  Both .dst files in this project are mis-suggestive; check the
Twiss before using either.

- ``part_rfq.dst`` is the **LINE INPUT at z = 0** — the LEBT entrance,
  NOT the RFQ input, despite the name (100 000 particles at 30 keV,
  standard 6-column layout, readable by
  :func:`linac_gen.io.tracewin_dst.load_dst`).  Three independent
  checks (2026-07-31): its 100 000 particles are the LAUNCH count, and
  ``partran1.out`` records 77 287 / 100 000 still alive at the LEBT
  *exit*; its σ_x = 4.8749 mm and β_x = 1.38705 reproduce the ENV
  export's row at position 0 to all printed digits; and 4.87 mm rms
  cannot physically fit an RFQ bore of a few mm.
  USE IT AT z = 0, paired with the FULL ``lebt_plus_rfq.dat`` deck —
  which is what ``conftest.py``, ``test_phase4_sc.py`` and
  ``test_rfq_losses.py`` already do.  Seeding the first RFQ cell with
  it directly injects a 4.87 mm beam into a few-mm bore and produces
  nonsense.
  CONSEQUENCE: the project holds NO TraceWin particle file at the RFQ
  input, so a seeded "RFQ-only" benchmark is not possible with this
  data; compare envelopes against the ENV exports instead.
- ``part_dtl1.dst`` is likewise NOT an RFQ-exit file: 23-byte header +
  SEVEN float64 per particle (x, x', y, y', phi_rad, W_MeV, extra) +
  trailing mass, and every particle sits at W ≈ 30 keV.  The project
  folder holds no Toutatis RFQ-exit distribution.
- ``Transfer_matrix1.dat`` stores the CUMULATIVE 6×6 transfer matrix at
  each of 242 element exits (``ELE# n : <s> m``).  Per-element matrices
  follow from M_elem = M_n · M_{n−1}⁻¹.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

H_MINUS_MC2_MEV = 939.294308      # TraceWin's H⁻ mass (partran1.out header)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACEWIN_CODE = REPO_ROOT / "Tracewin_code"
PXIE_PROJECT = (Path.home() / "Desktop" / "Projects" / "PIP_II"
                / "Paper" / "LEBT+RFQ")
PXIE_DECK = REPO_ROOT / "examples" / "lebt_plus_rfq" / "lebt_plus_rfq.dat"


def reference_path(name: str) -> Path | None:
    """Resolve a reference file by basename across both ground-truth
    roots; ``None`` when not present on this machine (callers skip)."""
    for root in (TRACEWIN_CODE, PXIE_PROJECT):
        p = root / name
        if p.is_file():
            return p
    return None


def load_env_chart(path: str | Path,
                   mass_mev: float = H_MINUS_MC2_MEV) -> dict:
    """Read a 26-column TraceWin "Save data" envelope export.

    Column map (verified against the writer in
    ``linac_gen/io/tracewin_outputs.py`` and the assist
    ``compare_to_tracewin`` tool): 0 = position (m), 1 = γ−1,
    2-11 = centroid 10-vector, 12-21 = rms 10-vector
    (x, x', y, y', z, dp/p, z', phase, time, energy), 22-25 =
    (dispX, dispY, betX, betY).
    """
    a = np.genfromtxt(path, skip_header=1)
    a = a[np.isfinite(a[:, 0])]
    return {
        "s_m": a[:, 0],
        "gam1": a[:, 1],
        "ref_W_MeV": a[:, 1] * mass_mev,
        "sigma_x_mm": a[:, 12] * 1e3,
        "sigma_y_mm": a[:, 14] * 1e3,
        "sigma_phi_deg": a[:, 19],
        "sigma_w_MeV": a[:, 21],
        "beta_x_m": a[:, 24],
        "beta_y_m": a[:, 25],
    }


def load_chart_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column TraceWin chart export (``Position`` header)."""
    a = np.genfromtxt(path, skip_header=1)
    a = a[np.isfinite(a[:, 0])]
    return a[:, 0], a[:, 1]


_ELE_RE = re.compile(r"ELE#\s*(\d+)\s*:\s*([0-9.eE+-]+)\s*m")


def load_transfer_matrices(path: str | Path):
    """Parse ``Transfer_matrix1.dat``: cumulative 6×6 at each element.

    Returns ``(elem_no, s_m, mats)`` with ``mats.shape == (N, 6, 6)``.
    """
    nums, positions, mats = [], [], []
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _ELE_RE.search(line)
            if m:
                if rows:
                    mats.append(np.array(rows[:6], dtype=float))
                    rows = []
                nums.append(int(m.group(1)))
                positions.append(float(m.group(2)))
                continue
            toks = line.split()
            if len(toks) == 6:
                try:
                    rows.append([float(t) for t in toks])
                except ValueError:
                    pass
    if rows:
        mats.append(np.array(rows[:6], dtype=float))
    return (np.array(nums, dtype=int), np.array(positions, dtype=float),
            np.stack(mats))


def per_element_matrices(cumulative: np.ndarray) -> np.ndarray:
    """Per-element matrices from cumulative ones:
    M_elem[0] = cum[0]; M_elem[n] = cum[n] · cum[n−1]⁻¹."""
    out = np.empty_like(cumulative)
    out[0] = cumulative[0]
    for n in range(1, len(cumulative)):
        out[n] = cumulative[n] @ np.linalg.inv(cumulative[n - 1])
    return out
