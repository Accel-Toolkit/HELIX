"""The reliability demo deck: generator <-> committed files, parser, and the
analytic truth (re-phased cavity-OFF deficits, exit energy, clock pins)."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"
GENERATED = ("reliability_demo.dat", "reliability_demo.lgproj",
             "circuits.json", "truth.json")


@pytest.fixture(scope="module")
def truth() -> dict:
    return json.loads((DEMO / "truth.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loaded():
    from linac_gen.cli import common
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, conv = common.load_input(str(DEMO / "reliability_demo.lgproj"))
    return lat, cfg, conv


def test_generator_reproduces_committed_files(tmp_path):
    """The generator run in a scratch copy writes byte-identical files."""
    gen = DEMO / "make_reliability_demo.py"
    (tmp_path / "make_reliability_demo.py").write_bytes(gen.read_bytes())
    r = subprocess.run([sys.executable, str(tmp_path / "make_reliability_demo.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    for name in GENERATED:
        assert (tmp_path / name).read_bytes() == (DEMO / name).read_bytes(), name


def test_parses_without_warnings():
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, meta = parse_tracewin(str(DEMO / "reliability_demo.dat"))
    assert meta.get("warnings") == []
    names = [getattr(e, "name", None) for e in lat.elements]
    for n in ("GAP_001", "GAP_004", "SOL_003", "STEER_003", "BPM_003",
              "APER_004", "FOIL_001"):
        assert n in names
    for bad in ("pipii", "pip2", "fnalscl", "mebt", "hebt"):
        assert not any(bad in str(p).lower() for p in DEMO.iterdir()), bad


def test_envelope_exit_energy_matches_truth(loaded, truth):
    from linac_gen.cli import common
    lat, cfg, _ = loaded
    res = common.run_envelope_sim(copy.deepcopy(lat), cfg)
    w = common.result_summary(res)["ref_w_kin"]
    want = truth["nominal"]["w_end_envelope_mev"]
    assert abs(w - want) <= truth["tolerances"]["energy_rel"] * want


def test_rephased_cavity_off_deficits_match_truth(loaded, truth):
    from linac_gen.failures.scenario import enumerate_scenarios
    from linac_gen.failures.study import FailureStudy
    lat, cfg, _ = loaded
    scen, n2c, names = enumerate_scenarios(lat, types=["cavity"])
    out = FailureStudy(lattice=copy.deepcopy(lat), beam_config=cfg,
                       mode="env").run(scen, names, n2c, serial=True)
    tol = truth["tolerances"]["energy_rel"]
    seen = set()
    for imp in out.impacts:
        g = imp.scenario.element_names[0]
        want = truth["cavity_off_rephased"][g]["d_energy_mev"]
        assert abs(imp.d_energy_mev - want) <= tol * want, g
        seen.add(g)
    assert seen == set(truth["cavity_off_rephased"])
    assert out.top(1)[0].scenario.element_names == (
        truth["ranking"]["top_single_cavity_rephased"],)


def test_solenoid_off_leaves_energy_exactly(loaded, truth):
    from linac_gen.failures.scenario import enumerate_scenarios
    from linac_gen.failures.study import FailureStudy
    lat, cfg, _ = loaded
    scen, n2c, names = enumerate_scenarios(lat, types=["solenoid"])
    out = FailureStudy(lattice=copy.deepcopy(lat), beam_config=cfg,
                       mode="env").run(scen, names, n2c, serial=True)
    assert len(out.impacts) == 3
    assert all(imp.d_energy_mev == truth["solenoid_off"]["d_energy_mev"]
               for imp in out.impacts)


def test_mp_clock_at_gap_entrances_matches_truth(loaded, truth):
    """The reference RF clock the frozen-phase pins are built on."""
    from linac_gen.cli import common
    lat, cfg, conv = loaded
    cfg = copy.deepcopy(cfg)
    cfg.n_particles = 200
    sc = common.make_sc_config(cfg, conv, {})
    st = common.make_step_config(conv, {})
    rec, beam = common.run_mp_sim(copy.deepcopy(lat), cfg, sc, st, seed=42)
    assert rec.transmission[-1] == 100.0
    pins = truth["nominal"]["clock_at_gap_entrance_deg"]
    for i, e in enumerate(lat.elements):
        if type(e).__name__ == "RFGap":
            phi = rec.ref_phi_s[rec.element_exit_idx[i - 1]]
            assert abs(phi - pins[e.name]) <= 1e-12 * abs(pins[e.name]), e.name
    want = truth["nominal"]["w_end_mp_ref_mev"]
    assert abs(rec.ref_w_kin[-1] - want) <= 1e-12 * want
