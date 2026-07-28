"""HALO-PIC (EXPERIMENTAL / RESEARCH): anchored defect correction for the
coarse PIC space-charge loop.

STATUS (2026-07-16): the ML speedup claim was retired after a two-scale
cost-matched control study — even a perfect corrector ties or loses to
plain grid refinement at equal wall-clock in HELIX (the field solve is
too cheap relative to deposit/gather for a solve-side surrogate to pay).
Full verdict and decision tables: docs/halo_pic/m3_gate_status.md.

What remains useful in production:

* ANCHORED AUDIT MODE — ``sc_backend="halo"`` with anchors enabled and no
  network gives a coarse MP+SC run that measures its own grid error
  in-flight (fine-grid solves every K-th kick; ``log["e_raw"]`` is the
  beam-weighted relative grid defect).  Use it to validate grid/N choices
  on a new lattice without a full fine-grid comparison run.
* The sigma-adaptive, beam-density-weighted delta-rho basis
  (:mod:`.basis`) and the trust-region anchor machinery (:mod:`.solver`)
  as a scaffold for any future revisit (e.g. codes whose Poisson solve
  is expensive enough to dominate the kick).

With ``alpha=0``/no net the solver is bit-identical to the production
``PicSolver`` (pinned by test).  Nothing here runs unless
``SpaceChargeConfig.sc_backend == "halo"`` is explicitly selected.
"""
