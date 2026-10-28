"""Ring mode must not change the linac (the guarantee in the ring-mode plan).

The full proof is the before/after probe (``scripts/bit_identity_probe.py``,
run at every ring-mode milestone).  These permanent checks guard the cheap
structural parts of the contract:

* importing HELIX and running a linac simulation never imports ring mode or
  its optional compiled dependency (numba);
* an ordinary bend or RF gap built by an importer carries no ring-only
  attribute, so linac objects are exactly what they were;
* a TraceWin deck without ``;@LG ring`` is never reported as a declared ring;
* envelope runs and transfer matrices on public decks equal stored numbers.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_linac_run_does_not_import_ring_mode_or_numba():
    code = (
        "import sys\n"
        "from linac_gen.cli import common\n"
        "lat, cfg, _ = common.load_input('examples/fodo_cell.dat')\n"
        "cfg.species='H-'; cfg.energy=2.1226695; cfg.frequency=162.5\n"
        "common.run_envelope_sim(lat, cfg)\n"
        "bad = sorted(m for m in sys.modules if m.startswith('linac_gen.ring') or m == 'numba')\n"
        "print('IMPORTED:' + ','.join(bad))\n")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "gui")]),
                "OMP_NUM_THREADS": "1"})
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("IMPORTED:")][-1]
    assert line == "IMPORTED:", f"linac run imported ring-mode modules: {line}"


def test_ordinary_bends_and_gaps_carry_no_ring_attributes(tmp_path):
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen.io.madx_parser import parse_madx

    deck = tmp_path / "plain.madx"
    deck.write_text("beam, particle=proton, energy=1.738272;\n"
                    "b: sbend, l=1, angle=0.1, k1=0.01; rf: rfcavity, l=0, volt=0.1, freq=100, lag=0;\n"
                    "s: sequence, l=3, refer=entry; b, at=0; rf, at=2; endsequence; use, sequence=s;\n")
    lat, _ = parse_madx(str(deck))
    bends = [e for e in lat.elements if isinstance(e, Dipole)]
    gaps = [e for e in lat.elements if isinstance(e, RFGap)]
    assert bends and gaps
    for e in bends:
        assert not hasattr(e, "k2") and not hasattr(e, "k0")
    for g in gaps:
        assert not hasattr(g, "harmonic")


def test_tracewin_deck_is_not_declared_a_ring():
    from linac_gen.io.formats import parse_lattice_file
    from linac_gen.ring.detect import detect_ring

    lat, meta = parse_lattice_file(str(ROOT / "examples" / "fodo_cell.dat"))
    v = detect_ring(lat, meta)
    assert not v.declared and not v.is_ring


def test_linac_results_match_the_stored_numbers():
    """Envelope runs and transfer matrices on public decks equal the numbers
    stored when ring mode began (bit-identical to the code before it; the
    tolerance only absorbs other platforms' round-off).  Regenerate with
    ``tests/ring/linac_golden.py --write`` after a deliberate linac change."""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import linac_golden
    finally:
        sys.path.pop(0)
    stored = np.load(linac_golden.GOLDEN)
    now = linac_golden.compute()
    assert sorted(now) == sorted(stored.files)
    for key, value in now.items():
        np.testing.assert_allclose(value, stored[key], rtol=1e-11, atol=1e-12, err_msg=key)
