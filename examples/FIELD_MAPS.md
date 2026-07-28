# Field-map data (`Fields/`)

Several PIP-II example decks reference superconducting-cavity and
solenoid **field maps** under a `Fields/` directory (HWR, SSR1, β=0.63
and β=0.92 650 MHz cavities).  That data is **third-party (ANL / CEA)
and is not distributed with this repository.**

What this means for a fresh clone:

- Decks that contain `FIELD_MAP` cards pointing into `Fields/`
  (e.g. `examples/pipii/*.dat`, `examples/MEBT_To_Foil/`,
  `examples/pipii_tunable/`) will refuse to load until you place the
  corresponding field-map files in a `Fields/` directory at the repo
  root (it is `.gitignore`d).
- Four test modules that exercise field-map I/O skip automatically when
  `Fields/` is absent — the rest of the suite (3,500+ tests) runs
  without it.
- Everything else — the BTL examples, FODO/chicane decks, the matching
  and error-study examples, the full GUI — works out of the box.

The same applies to the **PXIE/PIP2IT LEBT solenoid maps**
(`examples/lebt_pxie/SOL*-PXIE.*`): they are not distributed, so the
LEBT and LEBT+RFQ examples need them supplied locally before they run.

If you have access to the PIP-II lattice distribution (TraceWin decks +
field maps), drop its field-map files into `Fields/` (and the PXIE
solenoid maps into `examples/lebt_pxie/`) and every example runs.
