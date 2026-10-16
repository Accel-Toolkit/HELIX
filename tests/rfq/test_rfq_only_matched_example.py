"""The RFQ-alone example with the matched input (examples/lebt_plus_rfq):
the deck and project on disk are what make_rfq_only_matched.py writes; the
deck is the RFQ section of the matched LEBT + RFQ deck, unchanged; the
project reads the particle file re-shaped onto the matched alpha / beta; the
Toutatis reader keeps the particles that entered the RFQ.  The particle
file and the vane file are not in the repository: the tests that need them
skip without them, and the example is not in the public checkout."""
from __future__ import annotations

import importlib.util
import json
import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parents[2] / "examples" / "lebt_plus_rfq"
GEN = EX / "make_rfq_only_matched.py"

pytestmark = pytest.mark.skipif(
    not (GEN.is_file() and (EX / "lebt_plus_rfq_matched.dat").is_file()),
    reason="lebt_plus_rfq example not present")


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("make_rfq_only_matched", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_tracewin(str(path))[0]


def test_files_are_what_the_generator_writes(gen):
    assert gen.DECK.read_text() == gen.build()
    assert json.loads(gen.PROJECT.read_text()) == gen.project()


def test_deck_is_the_rfq_section_of_the_matched_deck(gen):
    from linac_gen.elements.rfq_cell import RfqCell
    r, m = _parse(gen.DECK), _parse(gen.SRC_DECK)
    first = next(i for i, e in enumerate(m.elements)
                 if type(e).__name__ == "Drift" and e.aperture == 5.0)
    tail = m.elements[first:]
    assert ([(type(e), e.length, e.aperture) for e in r.elements]
            == [(type(e), e.length, e.aperture) for e in tail])
    cr = [e for e in r.elements if isinstance(e, RfqCell)]
    cm = [e for e in tail if isinstance(e, RfqCell)]
    assert len(cr) == len(cm) == 203
    assert all(vars(a) == vars(b) for a, b in zip(cr, cm))
    assert Path(r.rfq_geom_file) == gen.HERE / "pxie-rfq.vane"


def test_project_reads_the_file_reshaped_onto_the_matched_twiss(gen):
    from linac_gen.io.project import load_project
    p, s = load_project(gen.PROJECT), load_project(gen.SRC_PROJECT)
    assert Path(p.lattice_path) == gen.DECK
    b = p.beam
    assert (b.source, Path(b.distribution_file), b.reshape_to_twiss) == (
        "file", gen.DST, True)
    assert {k: getattr(b, k) for k in gen.MATCHED} == gen.MATCHED
    assert (b.current, b.n_particles, b.continuous) == (gen.CURRENT_MA, gen.N_IN, True)
    assert p.convergence == s.convergence
    assert p.convergence["rfq_solver"] == "time_domain"


def _toutatis_file(path, n_in, n_lost, rng):
    """A synthetic 7-column Toutatis input .dst: n_in particles kept,
    n_lost flagged lost upstream, shuffled."""
    a = np.zeros((n_in + n_lost, 7))
    a[:, 0], a[:, 2] = rng.normal(0, 0.1, (2, len(a)))            # cm
    a[:, 1], a[:, 3] = rng.normal(0, 0.01, (2, len(a)))           # rad
    a[:, 4] = rng.uniform(-np.pi, np.pi, len(a))
    a[:, 5] = 0.030 + rng.normal(0, 1e-5, len(a))                  # MeV
    a[n_in:, 6] = 1.0
    a = a[rng.permutation(len(a))]
    head = struct.pack("<2sidd1s", b"\x7d\x50", len(a), 5.0, 162.5, b"\x00")
    path.write_bytes(head + a.astype("<f8").tobytes() + struct.pack("<d", 939.294))
    return a[a[:, 6] == 0]


def test_toutatis_reader_keeps_the_particles_that_entered(gen, tmp_path):
    from linac_gen.io.tracewin_dst import load_dst, write_dst
    rng = np.random.default_rng(3)
    kept = _toutatis_file(tmp_path / "toutatis.dst", gen.N_IN, 10000 - gen.N_IN, rng)
    p = gen.read_toutatis(tmp_path / "toutatis.dst")
    assert p.shape == (gen.N_IN, 6)
    assert np.allclose(p[:, 0], kept[:, 0] * 10.0) and np.allclose(p[:, 3], kept[:, 3] * 1e3)
    assert np.allclose(p[:, 4], np.degrees(kept[:, 4]))
    assert np.allclose(p[:, 5], kept[:, 5] - gen.W_IN)
    write_dst(str(tmp_path / "in.dst"), p, current_mA=gen.CURRENT_MA,
              frequency_MHz=gen.FREQ, mass_MeV=939.294, w_kin_ref=gen.W_IN)
    q, head = load_dst(str(tmp_path / "in.dst"))
    assert head["n_particles"] == gen.N_IN
    assert np.allclose(q[:, 0], p[:, 0]) and np.allclose(q[:, 1], p[:, 1])
    _toutatis_file(tmp_path / "short.dst", gen.N_IN - 1, 10, rng)
    with pytest.raises(ValueError, match="expected 7786"):
        gen.read_toutatis(tmp_path / "short.dst")


def _need(gen, *names):
    missing = [n for n in names if not (gen.HERE / n).is_file()]
    if missing:
        pytest.skip(f"not in the repository: {', '.join(missing)}")


def test_the_beam_starts_on_the_matched_twiss(gen):
    _need(gen, gen.DST.name)
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.project import load_project
    p = create_beam(load_project(gen.PROJECT).beam).particles
    for (i, j), pl in (((0, 1), "x"), ((2, 3), "y")):
        c = np.cov(p[:, i], p[:, j])
        e = np.sqrt(np.linalg.det(c))
        assert -c[0, 1] / e == pytest.approx(gen.MATCHED[f"alpha_{pl}"], rel=1e-9)
        assert c[0, 0] / e == pytest.approx(gen.MATCHED[f"beta_{pl}"], rel=1e-9)


@pytest.mark.slow
def test_the_matched_rfq_run(gen):
    """The project as the GUI and CLI run it (75 s): nearly every particle
    is accelerated and the exit emittances are those of a matched beam."""
    _need(gen, gen.DST.name, "pxie-rfq.vane")
    from linac_gen.cli import common
    from linac_gen.io.project import load_project
    p = load_project(gen.PROJECT)
    common.apply_fieldmap_settings(p.convergence, {})
    rec, beam = common.run_mp_sim(
        _parse(gen.DECK), p.beam, common.make_sc_config(p.beam, p.convergence, {}),
        common.make_step_config(p.convergence, {}), seed=42)
    w = beam.ref.w_kin + beam.particles[beam.alive_mask, 5]
    assert (w > 2.0).sum() / gen.N_IN > 0.999
    assert rec.emit_nx[-1] < 0.145 and rec.emit_ny[-1] < 0.145
    assert rec.emit_z[-1] < 0.046
    assert rec.rfq_td["n_left_map"] == 0
