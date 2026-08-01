"""Detect — and optionally repair — RFQ_CELL cards whose θs operand is
inconsistent with the deck's own cell lengths.

WHY THIS EXISTS
    TraceWin's RFQ_CELL card carries θs, and the two-term card model
    (HELIX's ``tw2term``, and TraceWin's own ENVELOPE model) takes it at
    face value: the per-cell energy gain is ``|q|·(π/4)·A10·V·cos θs``.
    **Toutatis does not.**  It builds the field from the vane geometry —
    the TraceWin manual states outright that Toutatis *"does not own
    phase reference"* (in its definition of the card's ``dP`` operand) —
    so a wrong θs is invisible to Toutatis and fatal to the card model.

    On the PXIE deck that difference is 7 %.  Cells 195–199 carry
    θs = −90° ("do not accelerate") while **every other column** — A10,
    m, L, dP — continues its smooth ramp, and the ``pxie-rfq.vane``
    geometry is fully modulated there (measured: same |E_z| as the
    neighbouring cells, declining ~1 %/cell).  The card model gains
    0.4–1.8 keV in those cells where Toutatis gains 27–42 keV, which is
    92 % of the total 137.6 keV deficit.

WHAT IT DOES NOT DO
    It does not touch the default path.  Parsing a deck is unchanged;
    you must call :func:`repair_rfq_phases` explicitly, exactly like
    :func:`~linac_gen.io.vane_rfq_helper.replace_rfq_cells_with_vane`.
    Nothing here is fitted to any TraceWin or Toutatis output — the
    replacement θs is derived from the deck's own cell lengths through
    the synchronism condition ``L = β·λ/2``.

    A repaired deck is a DIFFERENT deck.  Say so in anything you publish
    from it, and keep the original alongside.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from linac_gen.core.lattice import Lattice
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.rfq_coefficients import synchronous_phase_from_lengths


class RfqPhaseInconsistencyWarning(UserWarning):
    """A card's θs disagrees with the θs its own cell lengths imply."""


#: |θs_card − θs_derived| above which a card is called inconsistent.
#: The derivation itself is good to ~2° (see
#: :func:`synchronous_phase_from_lengths`), and genuine designs ramp θs
#: by well under this per cell, so 25° flags only gross disagreement —
#: on PXIE the five bad cards are out by 66°.
DEFAULT_TOL_DEG = 25.0

#: Cards below this A10 carry no meaningful accelerating field, so θs is
#: unconstrained and the derivation is ill-conditioned.  Shaper and
#: matcher cells legitimately sit at −90° with A10 ≈ 0.
DEFAULT_MIN_A10 = 0.10


@dataclass
class PhaseFinding:
    """One card whose θs disagrees with its own cell lengths."""
    index: int                  #: position within the RfqCell chain
    name: str
    phi_card_deg: float
    phi_derived_deg: float
    A10: float
    length_mm: float

    @property
    def delta_deg(self) -> float:
        return self.phi_derived_deg - self.phi_card_deg


def _chain(lattice: Lattice) -> List[RfqCell]:
    return [e for e in lattice.elements if isinstance(e, RfqCell)]


def derived_phases(lattice: Lattice, mass_MeV: float, charge: float,
                   frequency_MHz: float):
    """``(phi_deg, valid)`` for every RfqCell in *lattice*, in order.

    Thin wrapper over
    :func:`~linac_gen.elements.rfq_coefficients.synchronous_phase_from_lengths`
    that pulls the card operands off the parsed chain.
    """
    cells = _chain(lattice)
    if not cells:
        return np.zeros(0), np.zeros(0, dtype=bool)
    if frequency_MHz <= 0:
        raise ValueError("frequency_MHz must be > 0 to derive theta_s")
    wavelength_mm = 299_792_458.0 / (frequency_MHz * 1e6) * 1e3
    return synchronous_phase_from_lengths(
        [c.length for c in cells],
        [c.A10 for c in cells],
        [c.voltage_V for c in cells],
        mass_MeV=mass_MeV, charge=charge, wavelength_mm=wavelength_mm)


def inconsistent_phase_cells(lattice: Lattice, mass_MeV: float,
                             charge: float, frequency_MHz: float,
                             tol_deg: float = DEFAULT_TOL_DEG,
                             min_A10: float = DEFAULT_MIN_A10,
                             ) -> List[PhaseFinding]:
    """Cards whose θs is grossly inconsistent with their own cell lengths.

    Read-only.  Returns an empty list for a self-consistent deck.
    """
    cells = _chain(lattice)
    phi, valid = derived_phases(lattice, mass_MeV, charge, frequency_MHz)
    out: List[PhaseFinding] = []
    for i, c in enumerate(cells):
        if not valid[i] or c.A10 < min_A10:
            continue
        # Both this card and its successor must be genuinely accelerating
        # for the length-synchronism derivation to mean anything (see
        # synchronous_phase_from_lengths).
        if i + 1 >= len(cells) or cells[i + 1].A10 < min_A10:
            continue
        # cos is even, so the derivation fixes only |θs|.  Compare
        # against BOTH signs: an above-crest deck (θs > 0) is unusual
        # but must not be flagged wholesale as inconsistent.
        if min(abs(phi[i] - c.phi_s_deg),
               abs(-phi[i] - c.phi_s_deg)) > tol_deg:
            out.append(PhaseFinding(index=i, name=c.name,
                                    phi_card_deg=float(c.phi_s_deg),
                                    phi_derived_deg=float(phi[i]),
                                    A10=float(c.A10),
                                    length_mm=float(c.length)))
    return out


def repair_rfq_phases(lattice: Lattice, mass_MeV: float, charge: float,
                      frequency_MHz: float,
                      tol_deg: float = DEFAULT_TOL_DEG,
                      min_A10: float = DEFAULT_MIN_A10,
                      n_anchor: int = 25,
                      dry_run: bool = False) -> List[PhaseFinding]:
    """Replace θs on inconsistent cards, IN PLACE.  Returns what changed.

    The replacement is ``θs_derived − offset``, where *offset* is the
    median ``(derived − card)`` over the ``n_anchor`` nearest CONSISTENT
    accelerating cards that precede the run.  That anchor removes the
    derivation's known ~2° systematic (the dropped transit-time factor)
    using the deck's own trustworthy cards rather than any external
    reference — so a deck with no trustworthy neighbours gets the raw
    derived value and a warning, never a silent guess.

    A card that CANNOT be derived (the last cell has no successor to give
    ΔW; an exit matcher makes |cos θs| > 1) is still repaired when it is
    strongly modulated AND carries the same θs as an adjacent confirmed-
    bad card — i.e. it belongs to the same run of suspicious values.  Its
    θs is then linearly extrapolated from that run.  PXIE's cell 199 is
    exactly this case.

    Pass ``dry_run=True`` to see the findings without mutating anything.
    """
    cells = _chain(lattice)
    findings = inconsistent_phase_cells(lattice, mass_MeV, charge,
                                        frequency_MHz, tol_deg, min_A10)
    if not findings:
        return []
    phi, valid = derived_phases(lattice, mass_MeV, charge, frequency_MHz)
    bad = {f.index for f in findings}

    # Grow each run through neighbours that share the same suspicious
    # theta_s and are still modulated, but which the derivation cannot
    # reach.  Without this the LAST card of a bad run survives, because
    # a cell needs a successor to yield dW.
    grew = True
    while grew:
        grew = False
        for i in sorted(bad):
            for j in (i - 1, i + 1):
                if not (0 <= j < len(cells)) or j in bad:
                    continue
                if valid[j] or cells[j].A10 < min_A10:
                    continue
                if abs(cells[j].phi_s_deg - cells[i].phi_s_deg) > 1e-6:
                    continue
                findings.append(PhaseFinding(
                    index=j, name=cells[j].name,
                    phi_card_deg=float(cells[j].phi_s_deg),
                    phi_derived_deg=float("nan"),
                    A10=float(cells[j].A10),
                    length_mm=float(cells[j].length)))
                bad.add(j)
                grew = True
    findings.sort(key=lambda f: f.index)

    first = min(bad)
    anchors = [i for i in range(max(0, first - n_anchor), first)
               if valid[i] and cells[i].A10 >= min_A10 and i not in bad]
    # The derivation fixes only |θs|; take the deck's own sign convention
    # from its trustworthy cards rather than assuming below-crest.
    if anchors:
        sgn = -1.0 if np.median([cells[i].phi_s_deg
                                 for i in anchors]) <= 0.0 else 1.0
    else:
        sgn = -1.0
    phi = sgn * np.abs(phi)

    if anchors:
        offset = float(np.median([phi[i] - cells[i].phi_s_deg
                                  for i in anchors]))
    else:
        offset = 0.0
        warnings.warn(
            "repair_rfq_phases: no consistent accelerating cards precede "
            f"cell {first}, so the ~2 deg systematic of the derivation "
            "cannot be anchored out; using the raw derived theta_s.",
            RfqPhaseInconsistencyWarning, stacklevel=2)

    for f in findings:
        if valid[f.index]:
            f.phi_derived_deg = float(phi[f.index] - offset)
    # Extrapolate the ones the derivation could not reach, from the
    # repaired members of their own run.
    known = [(f.index, f.phi_derived_deg) for f in findings
             if np.isfinite(f.phi_derived_deg)]
    for f in findings:
        if np.isfinite(f.phi_derived_deg):
            continue
        if len(known) >= 2:
            xs = np.array([k[0] for k in known], dtype=float)
            ys = np.array([k[1] for k in known], dtype=float)
            f.phi_derived_deg = float(np.polyval(np.polyfit(xs, ys, 1),
                                                 f.index))
        elif len(known) == 1:
            f.phi_derived_deg = float(known[0][1])
        else:
            warnings.warn(
                f"repair_rfq_phases: cell {f.index} is suspicious but no "
                "derivable card in its run gives a value to extrapolate "
                "from; leaving it untouched.",
                RfqPhaseInconsistencyWarning, stacklevel=2)
    findings = [f for f in findings if np.isfinite(f.phi_derived_deg)]
    if not findings:
        return []
    if not dry_run:
        for f in findings:
            cells[f.index].phi_s_deg = f.phi_derived_deg

    warnings.warn(
        f"repair_rfq_phases: {len(findings)} RFQ_CELL card(s) carry a "
        f"theta_s inconsistent with their own cell lengths by more than "
        f"{tol_deg:g} deg"
        + ("" if dry_run else " and were REPLACED")
        + f" (anchor offset {offset:+.2f} deg from {len(anchors)} "
          f"neighbouring cards): "
        + ", ".join(f"cell {f.index} {f.phi_card_deg:+.2f} -> "
                    f"{f.phi_derived_deg:+.2f} deg" for f in findings[:8])
        + (" ..." if len(findings) > 8 else "")
        + ".  This is a MODIFIED deck — report it as such.",
        RfqPhaseInconsistencyWarning, stacklevel=2)
    return findings
