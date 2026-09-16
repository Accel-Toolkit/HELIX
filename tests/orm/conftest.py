"""Fixtures for the ORM tests: a synthetic FORMA export and small labelled lattices."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer


def make_forma_folder(root: Path, *, stamp: str, kick_plane: str, trims, bpms, values, errors=None,
                      dead=(), amplitude=0.15, with_gains=True) -> Path:
    """Write a folder in the exact FORMA v8 layout.

    ``values``/``errors``: ``{"x": (n_bpm_x, n_trim), "y": (n_bpm_y, n_trim)}``;
    ``bpms``: ``{"x": [devices], "y": [devices]}``; ``trims``: column devices.
    """
    folder = root / stamp; folder.mkdir(parents=True, exist_ok=True)
    kind = {"x": "horizontal", "y": "vertical"}
    for rp in ("x", "y"):
        if rp not in values:
            continue
        for tag, A in ((kind[rp], values[rp]), (kind[rp] + "_error", None if errors is None else errors.get(rp))):
            if A is None:
                continue
            with open(folder / f"response_matrix_{stamp}_{tag}.csv", "w", encoding="utf-8", newline="") as fh:
                fh.write("Reading Device," + ",".join(trims) + "\n")
                for dev, row in zip(bpms[rp], np.asarray(A)):
                    fh.write(dev + "," + ",".join(repr(float(v)) for v in row) + "\n")
    if with_gains:
        with open(folder / f"response_matrix_{stamp}_corrector_gains.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write("device,commanded_amp,achieved_amp,gain,coherent_r2,offdrive_noise,reliable\n")
            for t in trims:
                fh.write(f"{t},{amplitude},{amplitude*1.02},1.02,0.99,0.0005,True\n")
    setup = [{"device": t, "amplitude": amplitude, "periods": 10 + k} for k, t in enumerate(trims)]
    (folder / "scan_setup.json").write_text(json.dumps(setup), encoding="utf-8")
    (folder / "scan_info.json").write_text(json.dumps({"forma_version": "FORMA v8", "started": "2026-08-26T13:34:54",
                                                       "scan": {"devices": setup}, "baseline": {"measured": True, "dead_bpms": list(dead)}}), encoding="utf-8")
    (folder / "baseline_rms.json").write_text(json.dumps({"dead_bpms": list(dead), "baselines": {}}), encoding="utf-8")
    return folder


def fodo_with_devices(n_cells: int = 4, *, labels: bool = True, w_kin: float = 100.0) -> Lattice:
    """``[Steerer, Drift, QF, Drift, BPM, Drift, QD, Drift, BPM] × n``, labelled like the SCL deck.

    Cell k carries trim ``D{k}T``, quads ``Q{k}1``/``Q{k}2`` and BPMs ``D{k}1BPM``/``D{k}2BPM``.
    """
    lat = Lattice()
    for k in range(1, n_cells + 1):
        st = Steerer(f"STEER_{k:03d}"); st.label = f"D{k:02d}T" if labels else None; lat.add(st)
        lat.add(Drift(f"DRIFT_{k}a", 300.0, 20.0))
        q = Quadrupole(f"QUAD_{2*k-1:03d}", 100.0, 12.0, 20.0); q.label = f"Q{k:02d}1" if labels else None; lat.add(q)
        lat.add(Drift(f"DRIFT_{k}b", 300.0, 20.0))
        b = Marker(f"D{k:02d}1BPM", is_bpm=True); b.label = f"D{k:02d}1BPM" if labels else None; lat.add(b)
        lat.add(Drift(f"DRIFT_{k}c", 300.0, 20.0))
        q = Quadrupole(f"QUAD_{2*k:03d}", 100.0, -12.0, 20.0); q.label = f"Q{k:02d}2" if labels else None; lat.add(q)
        lat.add(Drift(f"DRIFT_{k}d", 300.0, 20.0))
        b = Marker(f"D{k:02d}2BPM", is_bpm=True); b.label = f"D{k:02d}2BPM" if labels else None; lat.add(b)
    return lat


@pytest.fixture
def fodo_lattice():
    return fodo_with_devices()
