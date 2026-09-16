# orm_demo — a synthetic orbit-response measurement

`fodo_orm.dat` / `fodo_orm.lgproj` is a six-cell labelled FODO (a steerer `DnnT`,
quads `Qnn1`/`Qnn2` and BPM markers `Dnn1BPM`/`Dnn2BPM` per cell, proton 100 MeV).
`make_synthetic_forma.py` writes a FORMA-layout folder pair (`<stamp>_H`, `<stamp>_V`)
from the HELIX model of that deck (or of any `--input` project) with random quad
scale errors, random trim calibrations and Gaussian noise, plus a `truth.json`
holding the injected values.  Device names are the deck labels, so the built-in
mapping resolves them directly:

```
PYTHONPATH=.:gui python examples/orm_demo/make_synthetic_forma.py
PYTHONPATH=.:gui python -m linac_gen orm compare examples/orm_demo/fodo_orm.lgproj \
    --measured examples/orm_demo/synthetic_forma/20260101_000000_H \
    --measured examples/orm_demo/synthetic_forma/20260101_000000_V --out examples/orm_demo/orm_out --plots
PYTHONPATH=.:gui python -m linac_gen orm fit examples/orm_demo/fodo_orm.lgproj --measured … \
    --out examples/orm_demo/orm_out --plots --export-deck examples/orm_demo/orm_out/fodo_orm_fit.dat --lgproj
```

Compare `orm_out/orm_calibration.json` (`quads[].scale`) with `synthetic_forma/truth.json`.
Manual: `docs/manual/06_running/13_cli_orm.md`.
