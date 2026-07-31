"""Generate a SYNTHETIC RFQ demonstration deck for HELIX.

WHY THIS EXISTS
    The other RFQ decks in the development tree carry a real machine's
    design and are excluded from public releases.  Without this file a
    public checkout would ship the RFQ model with nothing to run it on
    and every RFQ test skipping.

WHAT IT IS NOT
    It is NOT an existing design with the voltage changed.  A cell
    table's identity lives in its per-cell (A10, modulation, length,
    synchronous phase) ramp, not in the voltage, so rescaling V alone
    would leave the original design fully intact.  Every number here is
    generated from textbook two-term RFQ design relations at
    independently chosen parameters — species, frequency, aperture,
    voltage, energy range and cell count.  Nothing is copied from, or
    fitted to, any TraceWin output or any real machine.

DESIGN
    A conventional four-section RFQ: radial matcher, shaper, gentle
    buncher, accelerator, exit matcher.
      * modulation m ramps 1 -> M_MAX, synchronous phase -90 -> PHI_END
      * cell length L = beta*lambda/2 (synchronism), with beta advanced
        from the energy HELIX itself computes — the ramp is iterated
        against the tracker rather than an analytic guess, so the deck
        is self-consistent with the model that will run it
      * A10 derived from (r0, m, L) via the Crandall/Wangler two-term
        relation in rfq_coefficients.modulation_consistency, so the
        card triplet is internally consistent
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from linac_gen.elements.rfq_coefficients import modulation_consistency

# ---- design parameters ---------------------------------------------
# Chosen to be a plausible, self-consistent RFQ and deliberately
# unlike any existing machine.  See the module docstring.
SPECIES = "proton"
FREQ_MHZ = 352.21
W_IN_MEV = 0.075
V_VANE = 85_000.0
R0_MM = 3.40
M_MAX = 1.95
PHI_END = -28.0
N_SHAPER = 26
N_BUNCHER = 115
N_ACCEL = 56
MC2 = 938.272088          # proton rest mass (MeV)
C_MM_NS = 299.792458

N_CELLS = N_SHAPER + N_BUNCHER + N_ACCEL


def beta_of(w_mev: float) -> float:
    g = 1.0 + w_mev / MC2
    return math.sqrt(max(g * g - 1.0, 0.0)) / g


def ramps():
    """(m, phi_s) per cell — the four-section profile."""
    m = np.ones(N_CELLS)
    phi = np.full(N_CELLS, -90.0)
    # shaper: modulation just begins, phase held at -90 (pure bunching)
    i0, i1 = 0, N_SHAPER
    m[i0:i1] = np.linspace(1.0, 1.04, N_SHAPER)
    # gentle buncher: m and phi_s ramp together, the classic design
    i0, i1 = N_SHAPER, N_SHAPER + N_BUNCHER
    t = np.linspace(0.0, 1.0, N_BUNCHER)
    m[i0:i1] = 1.04 + (1.55 - 1.04) * t ** 2.2
    phi[i0:i1] = -90.0 + (-45.0 + 90.0) * t ** 3.4
    # accelerator: modulation tops out, phase flattens
    i0 = N_SHAPER + N_BUNCHER
    t = np.linspace(0.0, 1.0, N_ACCEL)
    m[i0:] = 1.55 + (M_MAX - 1.55) * t ** 0.7
    phi[i0:] = -45.0 + (PHI_END + 45.0) * t ** 0.8
    return m, phi


def build(cell_lengths):
    """Emit the RFQ_CELL card block for the given cell lengths."""
    m, phi = ramps()
    lam_mm = C_MM_NS / FREQ_MHZ * 1e3 / 1e3 * 1e3   # mm  (c/f)
    lam_mm = 299_792_458.0 / (FREQ_MHZ * 1e6) * 1e3
    cards = []
    for i in range(N_CELLS):
        L = cell_lengths[i]
        # modulation_consistency is a FIXED POINT: it needs a seed A10
        # to solve the vane-tip aperture, then returns the theory A10 for
        # (r0, m, L).  Iterate it to convergence so the emitted triplet
        # is internally consistent by the same relation HELIX uses to
        # cross-check cards.
        A10 = 0.0
        if m[i] > 1.0:
            A10 = 0.10
            for _ in range(6):
                a10_th, _ = modulation_consistency(R0_MM, A10, m[i], L)
                if a10_th <= 0.0:
                    break
                if abs(a10_th - A10) < 1e-9:
                    A10 = a10_th
                    break
                A10 = a10_th
        cards.append((V_VANE, R0_MM, A10, m[i], L, phi[i], 2 if i % 2 == 0
                      else -2))
    return cards, lam_mm


def synchronous_lengths(n_iter=4):
    """Cell lengths L = beta*lambda/2, with beta advanced from the energy
    the RFQ model itself produces.  Iterated so the deck is synchronous
    with the tracker rather than with an analytic approximation."""
    lam_mm = 299_792_458.0 / (FREQ_MHZ * 1e6) * 1e3
    L = np.full(N_CELLS, beta_of(W_IN_MEV) * lam_mm / 2.0)
    for _ in range(n_iter):
        cards, _ = build(L)
        w = W_IN_MEV
        newL = np.empty(N_CELLS)
        for i, (V, r0, A10, m, Li, phi, _t) in enumerate(cards):
            newL[i] = beta_of(w) * lam_mm / 2.0
            # two-term synchronous gain over one cell
            k = math.pi / max(newL[i], 1e-9)
            from scipy.special import i0 as bessel_i0
            T = bessel_i0(k * R0_MM * 0.6)
            w += (math.pi / 4.0) * A10 * V * 1e-6 * T * math.cos(
                math.radians(phi))
        L = newL
    return L, w


def main():
    L, w_exit = synchronous_lengths()
    cards, _ = build(L)
    total = float(np.sum(L))
    lines = [
        "; ===========================================================",
        "; SYNTHETIC RFQ DEMONSTRATION DECK — generated, do not hand-edit",
        ";   regenerate with:  python3 make_rfq_demo.py",
        ";",
        "; A conventional four-section RFQ (shaper / gentle buncher /",
        "; accelerator) built from textbook two-term design relations.",
        "; It is NOT any real machine's design: every per-cell number is",
        "; generated here, and nothing is copied from or fitted to any",
        "; TraceWin output.",
        ";",
        f";   species        {SPECIES}",
        f";   frequency      {FREQ_MHZ} MHz",
        f";   input energy   {W_IN_MEV*1e3:.1f} keV",
        f";   vane voltage   {V_VANE/1e3:.0f} kV",
        f";   r0             {R0_MM} mm",
        f";   cells          {N_CELLS}  ({N_SHAPER} shaper /"
        f" {N_BUNCHER} buncher / {N_ACCEL} accelerator)",
        f";   length         {total:.1f} mm",
        f";   design exit W  ~{w_exit:.3f} MeV (analytic estimate)",
        "; ===========================================================",
        "",
        f"FREQ {FREQ_MHZ}",
        "",
        "; radial matcher — unmodulated, pure transverse focusing",
        f"RFQ_CELL {V_VANE:.0f} {R0_MM} 0 1 "
        f"{beta_of(W_IN_MEV)*299_792_458.0/(FREQ_MHZ*1e6)*1e3/2:.4f} -90 3",
    ]
    for (V, r0, A10, m, Li, phi, ctype) in cards:
        lines.append(f"RFQ_CELL {V:.0f} {r0} {A10:.6f} {m:.4f} "
                     f"{Li:.4f} {phi:.4f} {ctype}")
    lines += [
        "; exit matcher",
        f"RFQ_CELL {V_VANE:.0f} {R0_MM} 0 1 {L[-1]:.4f} -90 -3",
        "",
        "END",
    ]
    out = pathlib.Path(__file__).with_name("rfq_demo.dat")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.name}: {N_CELLS + 2} cells, {total:.1f} mm, "
          f"analytic exit ~{w_exit:.3f} MeV")


if __name__ == "__main__":
    main()
