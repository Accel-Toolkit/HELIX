"""Linac results on public decks, stored once: the numbers ring mode must
never change (tests/ring/test_linac_untouched.py compares them).

    PYTHONPATH=.:gui python tests/ring/linac_golden.py --write

regenerates ``fixtures/linac_golden.npz`` — only after a deliberate change to
linac results, said so in its commit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "linac_golden.npz"

ENVELOPE_DECKS = ("examples/fodo_cell.dat", "examples/dtl_section.dat",
                  "examples/solenoid_channel.dat")
MATRIX_DECKS = ("examples/bend_line.dat", "examples/csr_chicane.dat", "examples/madx/fodo.madx",
                "examples/madx/transport.madx", "tests/ring/fixtures/simple_booster.madx")
ENVELOPE_FIELDS = ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w", "emit_x", "emit_y",
                   "emit_z")


def compute() -> dict:
    from linac_gen.cli import common
    from linac_gen.io.formats import parse_lattice_file
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    out: dict = {}
    for rel in ENVELOPE_DECKS:
        lat, cfg, _ = common.load_input(str(ROOT / rel))
        res = common.run_envelope_sim(lat, cfg)
        for f in ENVELOPE_FIELDS:
            out[f"{rel}:{f}"] = np.asarray(getattr(res, f), dtype=float)
    for rel in MATRIX_DECKS:
        lat, meta = parse_lattice_file(str(ROOT / rel))
        ref = (meta or {}).get("reference") if isinstance(meta, dict) else None
        if ref is None:
            _lat, cfg, _ = common.load_input(str(ROOT / rel))
            ref = common.build_ref(cfg)
        out[f"{rel}:matrix"] = np.asarray(compute_transfer_matrix(lat, ref.copy()), dtype=float)
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(__doc__)
    np.savez_compressed(GOLDEN, **compute())
    print(f"wrote {GOLDEN}")
