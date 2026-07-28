# Contributing to HELIX

Thanks for your interest! HELIX is a physics code first — correctness and
honesty outrank convenience.

## Ground rules

- **Every change runs the full suite** (`pytest -q`, 3,600+ tests, zero
  failures expected) before a PR.
- **Numerics changes need an external anchor**: compare against TraceWin
  output, an analytic identity, or a finite-difference check — never only
  against the code's own previous output (round-trip tests cancel
  symmetric errors).
- **Physics conventions are load-bearing**: lengths mm, energies MeV,
  emittances π·mm·mrad (normalized transverse), `alpha_z` is the negative
  of TraceWin's. When a formula has two regimes (DC/bunched, I=0/I>0),
  test both.
- **Refuse loudly**: when an input can't be represented faithfully, raise
  with a clear message — never approximate silently.

## Development setup

```bash
git clone https://github.com/Accel-Toolkit/HELIX.git
cd HELIX
pip install -e ".[gui,dev]"
pytest -q
```

GUI tests need Qt in offscreen mode:

```bash
QT_QPA_PLATFORM=offscreen pytest tests/gui -q
```

## Pull requests

- One logical change per PR, with tests that pin the behavior.
- Match the surrounding style (the code is heavily commented where physics
  constraints live — keep that habit).
- Update `docs/manual/` when behavior changes and check
  `mkdocs build --strict` stays green.
