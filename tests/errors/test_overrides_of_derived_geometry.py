"""An error study's draws travel as ``@index.attr`` overrides: a reliability campaign replays them in the scan pool on a
lattice read again from its file.  An attribute that follows from others, a bend's length (from its angle and radius)
or a cavity's length and synchronous beta (from its cells), is left out of them: the attributes it follows from carry
it, and writing it is refused."""
from __future__ import annotations

import contextlib
import copy
import io
import warnings

import pytest

from linac_gen.cli import common
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.ncells import NCells
from linac_gen.errors.error_model import ErrorStudy, lattice_overrides


def _lattice():
    lat = Lattice()
    for e in (Drift("D0", 100.0), Dipole("BEND_1", angle=10.0, rho=800.0), Drift("D1", 100.0),
              NCells("CAV_1", mode=0, n_cells=3, beta_g=0.5, eot_v_per_m=1.0e6, theta_s_deg=-30.0,
                     frequency_mhz=650.0),
              Drift("D2", 100.0)):
        lat.add(e)
    return lat


def _public_numbers(el):
    return {k: v for k, v in vars(el).items()
            if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)}


@pytest.mark.parametrize("pattern, parameter, sigma", [
    ("BEND_1", "angle_rel", 0.01), ("BEND_1", "rho", 1.0), ("CAV_1", "beta_g_rel", 0.01),
    ("CAV_1", "eot_v_per_m_rel", 0.01)])
def test_a_draw_replays_as_the_errored_lattice(pattern, parameter, sigma):
    lat = _lattice()
    study = ErrorStudy(lat, BeamConfig(), n_seeds=1)
    study.add_error(pattern=pattern, parameter=parameter, sigma=sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draw = study.draw_overrides(0)
    assert draw.overrides and draw.untransportable == ()
    assert not any(sel.endswith((".length", ".beta_s")) for sel, _v in draw.overrides)
    fresh = copy.deepcopy(lat)
    for sel, val in draw.overrides:                     # as the scan pool and the campaign apply them
        common.apply_element_override(fresh, sel, val)
    for x, y in zip(fresh.elements, draw.lattice.elements, strict=True):
        assert _public_numbers(x) == _public_numbers(y), x.name


def test_a_pipe_given_a_length_carries_it():
    """A bend with no angle takes a length: then it is an input like any other."""
    design = Lattice()
    design.add(Dipole("P", angle=0.0, rho=1.0e9))
    errored = copy.deepcopy(design)
    errored.elements[0].length = 300.0
    assert lattice_overrides(design, errored) == ((("@1.length", 300.0),), ())


def test_a_campaign_point_replays_a_bend_draw(tmp_path):
    """Through the scan pool's worker, which reads the deck again and applies the overrides."""
    from linac_gen.parallel.scan_pool import _run_one_point_worker
    deck = tmp_path / "line.dat"
    deck.write_text("DRIFT 100 30\nBEND 10 800 0 30 0\nDRIFT 100 30\nEND\n")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _cfg, _conv = common.load_input(str(deck))
        bend = next(e for e in lat.elements if isinstance(e, Dipole))
        cfg = BeamConfig(current=0.0)
        study = ErrorStudy(lat, cfg, n_seeds=1)
        study.add_error(pattern=bend.name, parameter="angle_rel", sigma=0.05)
        draw = study.draw_overrides(0)
        point = common.build_scan_point(str(deck), element_overrides=draw.overrides, mode="envelope",
                                        beam_overrides={"current": 0.0})
        got = _run_one_point_worker(point)
        want = common.run_envelope_sim(draw.lattice, BeamConfig(**point.beam_config), env_solver=point.env_solver)
    assert got["error"] is None, got.get("error")
    assert got["sigma_x"] == float(want.sigma_x[-1]) and got["sigma_phi"] == float(want.sigma_phi[-1])
