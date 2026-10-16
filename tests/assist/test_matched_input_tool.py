"""matched_input_twiss assistant tool: the default cell (an RFQ period when
the lattice has an RFQ, else the first FODO cell), explicit bounds, and a
JSON-clean result (flags stay booleans)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from linac_gen.assist.tools import TOOLS, WorkContext

ROOT = Path(__file__).parents[2]
DEMO_PROJ = ROOT / "examples" / "rfq_demo" / "rfq_demo.lgproj"


def _ctx(tmp_path, lattice, cfg):
    c = WorkContext(calc_dir=str(tmp_path))
    c.set_lattice(lattice, "<matched-input-test-lattice>")
    c.set_beam_config(cfg)
    return c


@pytest.mark.skipif(not DEMO_PROJ.exists(), reason="examples/rfq_demo missing")
def test_rfq_lattice_defaults_to_rfq_period(tmp_path):
    from linac_gen.cli.common import build_ref, load_input
    from linac_gen.matching.periodic import find_rfq_matched_input_twiss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, cfg, _ = load_input(str(DEMO_PROJ))
    res = TOOLS["matched_input_twiss"].fn(_ctx(tmp_path, lat, cfg))
    assert res["status"] == "ok", res
    data = res["data"]
    assert data["kind"] == "rfq"
    assert data["at_lattice_entrance"] is True and data["converged"] is True
    assert data["rfq_name"] == "RFQ_001"
    tw = find_rfq_matched_input_twiss(lat, build_ref(cfg))
    for k in ("alpha_x", "beta_x", "alpha_y", "beta_y"):
        assert data[k] == pytest.approx(tw[k])
    # the demo's short radial matching section is flagged
    assert any("depends on the RF phase" in w for w in res["warnings"])
    json.dumps(res, allow_nan=False)


def _fodo_ctx(tmp_path):
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    for c in range(4):
        lat.add(Quadrupole(f"QF_{c}", 100.0, gradient=10.0, aperture=20.0))
        lat.add(Drift(f"D1_{c}", 200.0, aperture=20.0))
        lat.add(Quadrupole(f"QD_{c}", 100.0, gradient=-10.0, aperture=20.0))
        lat.add(Drift(f"D2_{c}", 200.0, aperture=20.0))
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21)
    return _ctx(tmp_path, lat, cfg)


def test_fodo_lattice_keeps_first_fodo_cell(tmp_path):
    from linac_gen.cli.common import build_ref
    from linac_gen.matching.periodic import (find_fodo_cells,
                                             find_matched_input_twiss)
    ctx = _fodo_ctx(tmp_path)
    res = TOOLS["matched_input_twiss"].fn(ctx)
    assert res["status"] == "ok", res
    data = res["data"]
    assert data["kind"] == "fodo" and data["coupled"] is False
    cs, ce = find_fodo_cells(ctx.lattice)[0]
    tw = find_matched_input_twiss(ctx.lattice, build_ref(ctx.beam_config),
                                  cs, ce)
    assert data["beta_x"] == pytest.approx(tw["beta_x"])
    json.dumps(res, allow_nan=False)


def test_explicit_bounds(tmp_path):
    ctx = _fodo_ctx(tmp_path)
    res = TOOLS["matched_input_twiss"].fn(ctx, cell_start=1, cell_end=4)
    assert res["status"] == "ok", res
    assert res["data"]["kind"] == "manual"
