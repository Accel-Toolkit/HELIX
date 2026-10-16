"""The tuned LEBT + RFQ example (examples/lebt_plus_rfq, files named
"matched"): the deck and project on disk are what make_matched_deck.py
writes; the deck is the phifix deck with only the SOL2 / SOL3 strengths
changed (both read the vane file of this folder); the project is the phifix
project with the particles' own field for the LEBT's space charge and the
time-domain RFQ solver asked for, which stops with the vane file named when
that file is missing.  The example
is not in the public checkout: everything skips there."""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

EX = Path(__file__).resolve().parents[2] / "examples" / "lebt_plus_rfq"
GEN = EX / "make_matched_deck.py"

pytestmark = pytest.mark.skipif(
    not (GEN.is_file() and (EX / "lebt_plus_rfq_phifix.dat").is_file()),
    reason="lebt_plus_rfq example not present")


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("make_matched_deck", GEN)
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


def test_deck_changes_only_the_two_strengths(gen):
    from linac_gen.elements.rfq_cell import RfqCell
    m, s = _parse(gen.DECK), _parse(gen.SRC_DECK)
    assert ([(type(e), e.name, e.length) for e in m.elements]
            == [(type(e), e.name, e.length) for e in s.elements])
    kb = [(a.kb, b.kb) for a, b in zip(m.elements, s.elements)
          if a.name.startswith("FMAP")]
    assert kb[0][0] == kb[0][1]                       # SOL1 unchanged
    assert kb[1:] == [(gen.KB_SOL2, gen.KB_SOL2_SRC),
                      (gen.KB_SOL3, gen.KB_SOL3_SRC)]
    cm = [e for e in m.elements if isinstance(e, RfqCell)]
    cs = [e for e in s.elements if isinstance(e, RfqCell)]
    assert len(cm) == len(cs) == 203
    assert all(vars(a) == vars(b) for a, b in zip(cm, cs))
    assert Path(m.rfq_geom_file) == Path(s.rfq_geom_file) == gen.HERE / gen.VANE


def test_project_is_the_phifix_project_with_the_time_domain_solver(gen):
    from linac_gen.io.project import load_project
    m, s = load_project(gen.PROJECT), load_project(gen.SRC_PROJECT)
    assert Path(m.lattice_path) == gen.DECK
    assert m.beam == s.beam
    assert m.convergence == {**s.convergence, "dc_kernel": "pic2d",
                             "rfq_solver": "time_domain",
                             "rfq_td_steps_per_period": 48,
                             "rfq_td_sc_every": 4}


def test_missing_vane_file_stops_the_run_and_names_it(gen, tmp_path):
    from dataclasses import replace

    from linac_gen.cli import common
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.project import load_project
    p = load_project(gen.PROJECT)
    lat = _parse(gen.DECK)
    lat.rfq_geom_file = str(tmp_path / gen.VANE)          # not there
    lat.step_config = common.make_step_config(p.convergence, {})
    beam = create_beam(replace(p.beam, n_particles=200), seed=1)
    with pytest.raises(Exception, match="cannot read the vane geometry") as e:
        Simulation(lat, beam).run()
    assert str(tmp_path / gen.VANE) in str(e.value)


@pytest.mark.slow
def test_the_tuned_run(gen):
    """The run the GUI shows for this project (20000 particles, seed 42)
    leaves the RFQ as make_matched_deck.py documents it.  About 10 particles
    leave unaccelerated; the statistics leave them out (they would set the
    exit energy spread: 57 keV against 8.1).  The rms rows along the RFQ
    stay on the bunch's scale: a particle that left the stored field region
    used to fly on to 1 m off axis and read 2.9 mm mrad there (the rule that
    loses it is pinned in tests/rfq_td/test_left_field_region.py)."""
    import numpy as np
    from linac_gen.cli import common
    from linac_gen.io.project import load_project
    if not (gen.HERE / gen.VANE).is_file():
        pytest.skip("the RFQ vane file is not in examples/lebt_plus_rfq")
    p = load_project(gen.PROJECT)
    lat = _parse(gen.DECK)
    common.apply_fieldmap_settings(p.convergence, {})
    step = common.make_step_config(p.convergence, {})
    lat.step_config = step
    rec, beam = common.run_mp_sim(
        lat, p.beam, common.make_sc_config(p.beam, p.convergence, {}), step,
        seed=42)
    info = rec.rfq_td
    assert info["outside_map_steps"] == info["n_left_map"]
    assert info["n_unaccelerated"] >= 1
    names = list(rec.element_names)
    i = next(k for k, n in enumerate(names) if n.startswith("RFQ_"))
    assert max(rec.emit_nx[i:]) < 0.35 and max(rec.emit_ny[i:]) < 0.35
    assert max(rec.sigma_x[i:]) < 1.5 and max(rec.sigma_y[i:]) < 1.5
    assert 0.160 < rec.emit_nx[-1] < 0.175 and 0.160 < rec.emit_ny[-1] < 0.175
    assert 1e3 * rec.sigma_w[-1] < 10.0           # keV, the accelerated beam
    assert 76.7 < rec.transmission[-1] < 76.9      # everything alive counts
    w = beam.ref.w_kin + beam.particles[:, 5]
    accelerated = np.count_nonzero(beam.alive_mask & (w > 2.0))
    assert accelerated / (rec.transmission[i - 1] / 100 * len(w)) > 0.99
