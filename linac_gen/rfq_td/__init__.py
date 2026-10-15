"""Time-domain RFQ solver.

The RFQ is tracked with TIME as the independent variable through the
quasi-static 3-D field of its real vanes, the way Toutatis and
LINACSrfqSIM do it, instead of cell by cell along z:

* :mod:`.geometry` — the electrodes: four vane tips (tip distance and
  transverse radius per z, from a TraceWin/Toutatis ``.vane`` file or
  generated from the ``RFQ_CELL`` cards), flanks, grounded end plates;
  signed distances and the electrode-contact loss test.
* :mod:`.laplace` — Shortley–Weller finite differences of Laplace's
  equation on a quadrant grid, solved in overlapping z windows.
* :mod:`.fieldmap` — the near-axis potential of those solves, cached
  on disk by content hash, evaluated tricubically (E = −∇φ).
* :mod:`.space_charge` — the self-field at a common time, periodic in
  z with the local bunch period βλ (the bunch train) and open in x, y.
* :mod:`.tracker` — kick-drift-kick leapfrog in (r, γβ) with the RF
  time dependence, electrode losses, plane-crossing records.
* :mod:`.chain` — runs a whole chain of ``RfqCell`` elements once and
  hands the tracker each cell's exit state, so every cell keeps its
  recorder row, name and loss bookkeeping.

``StepConfig.rfq_solver``: ``"auto"`` (the default) uses it whenever it
can run the lattice and beam and otherwise tracks the RFQ cell by cell,
with a notice saying why; ``"time_domain"`` always uses it (a case it
cannot run is an error); ``"cells"`` never does.
"""
