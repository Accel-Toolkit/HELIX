"""The reliability campaign's tuning windows (2026-11-08).

Every element got (0, headroom) x a family median of its name prefix: one median of the signed
quadrupole gradients (+6.57 T/m on a linac-to-foil deck) left 81 of 124 quadrupoles' own setting
outside their window, and one "FMAP" family lumped every field-map cavity type into a single
nominal, so headroom keys naming a cavity type never matched.  The headroom rule is the RF's, by
cavity type; a magnet keeps the window about its own setting."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from linac_gen.failures.compensation import CompensationConfig, amp_bounds, family_nominals
from linac_gen.reliability.campaign import ReliabilityCampaign


def _fieldmap(name, ke, kb, field_file):
    from linac_gen.elements.field_map import FieldMap
    fm = FieldMap.__new__(FieldMap)        # no field file read: the families read its name
    fm.name, fm.ke, fm.kb, fm.phase, fm.field_file = name, ke, kb, -30.0, field_file
    return fm


def _line():
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap
    return SimpleNamespace(elements=[
        RFGap("GAP_001", 0.5, -30.0, 352.21), RFGap("GAP_002", 0.7, -30.0, 352.21),
        _fieldmap("FMAP_001", 0.066, 0.066, "/x/HWRDonut"),
        _fieldmap("FMAP_002", 0.070, 0.070, "/x/HWRDonut"),
        _fieldmap("FMAP_003", 18.0, 18.0, "/y/beta092"),
        _fieldmap("FMAP_004", 0.0, 0.0, "/y/beta092"),                  # a spare
        Quadrupole("QUAD_001", length=100.0, gradient=8.0),
        Quadrupole("QUAD_002", length=100.0, gradient=-6.0),
        Quadrupole("QUAD_003", length=100.0, gradient=12.0)])


def _families(names):
    stub = SimpleNamespace(spec=SimpleNamespace(compensation={}))
    return ReliabilityCampaign._cavity_families(stub, _line(), names)


def test_a_cavitys_family_is_its_type():
    fam = _families(["GAP_001", "GAP_002", "FMAP_001", "FMAP_002", "FMAP_003", "FMAP_004"])
    assert fam == {"GAP_001": "GAP", "GAP_002": "GAP", "FMAP_001": "HWRDonut",
                   "FMAP_002": "HWRDonut", "FMAP_003": "beta092", "FMAP_004": "beta092"}
    cfg = CompensationConfig(family_of=fam, amp_bounds_by_family={f: (0.0, 1.5) for f in fam.values()})
    nom = family_nominals(_line(), cfg)
    assert nom["HWRDonut"] == pytest.approx(0.068) and nom["beta092"] == 18.0
    lo, hi = amp_bounds(cfg, "FMAP_004", 0.0, nom)          # a spare rises to its type's headroom
    assert (lo, hi) == (0.0, pytest.approx(27.0))


def test_the_campaign_gives_its_magnets_no_headroom_family(tmp_path):
    """Through the campaign's own compensation jobs on the demo deck: the cavities keep their
    family, the solenoids get none."""
    import shutil
    from pathlib import Path

    from linac_gen.reliability.spec import default_spec
    demo = Path(__file__).resolve().parents[2] / "examples" / "reliability_demo"
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json"):
        shutil.copy2(demo / name, tmp_path / name)
    spec = default_spec(str(tmp_path / "reliability_demo.lgproj"), preset="quick", name="w",
                        circuits="circuits.json", landmarks={"treaty_element": "BPM_003"})
    c = ReliabilityCampaign.create(tmp_path / "campaign", spec)
    cases, _n2c = c._cases_cached()
    sol = next(k for k in cases if k.case_id.startswith("S2_SOL_") and k.case_id.endswith("_re"))
    item = {"case_id": sol.case_id, "bracket": "rephased", "mode": "envelope", "label": sol.case_id,
            "strategy": "k_out_of_n", "failures": [[n, m.kind.value, m.amp_scale, m.phase_deg]
                                                   for n, m in sol.scenario.failures]}
    job = c._comp_jobs([item])[0]
    assert job["family_of"] and set(job["family_of"].values()) == {"GAP"}
    assert not any(n.startswith("SOL_") for n in job["family_of"])
    assert set(job["amp_bounds_by_family"]) == {"GAP"}


@pytest.mark.parametrize("name, gradient", [("QUAD_001", 8.0), ("QUAD_002", -6.0), ("QUAD_003", 12.0)])
def test_a_magnet_with_no_family_keeps_the_window_about_its_own_setting(name, gradient):
    fam = _families(["GAP_001", "FMAP_003"])
    cfg = CompensationConfig(family_of=fam, amp_bounds_by_family={f: (0.0, 1.5) for f in fam.values()})
    lo, hi = amp_bounds(cfg, name, gradient, family_nominals(_line(), cfg))
    assert lo < gradient < hi and lo * hi > 0


def test_the_campaign_gives_quadrupoles_no_family(tmp_path):
    """On a line with quadrupoles: none gets a headroom family, and each one's window brackets
    its own signed gradient (one median of the signed gradients left most outside)."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.project import write_project
    from linac_gen.reliability.spec import default_spec
    deck = tmp_path / "line.dat"
    deck.write_text("""FREQ 352.21
DRIFT 100 20
GAP 300000 -30 20 0 0 0 0 0 0
DRIFT 100 20
QUAD 100 8 20
DRIFT 100 20
GAP 300000 -30 20 0 0 0 0 0 0
DRIFT 100 20
QUAD 100 -6 20
DRIFT 100 20
QUAD 100 12 20
DRIFT 100 20
GAP 300000 -30 20 0 0 0 0 0 0
DRIFT 100 20
END
""")
    proj = write_project(tmp_path / "line.lgproj", lattice_path=deck,
                         beam=BeamConfig(species="proton", energy=3.0, frequency=352.21,
                                         current=0.0, emit_nx=0.25, emit_ny=0.25, emit_z=0.3,
                                         beta_x=1.0, beta_y=1.0, beta_z=1.0))
    c = ReliabilityCampaign.create(tmp_path / "campaign",
                                   default_spec(str(proj), preset="quick", name="q"))
    cases, _n2c = c._cases_cached()
    quad = next(k for k in cases if k.case_id.startswith("S2_QUAD_") and k.case_id.endswith("_re"))
    item = {"case_id": quad.case_id, "bracket": "rephased", "mode": "envelope", "label": quad.case_id,
            "strategy": "k_out_of_n", "failures": [[n, m.kind.value, m.amp_scale, m.phase_deg]
                                                   for n, m in quad.scenario.failures]}
    job = c._comp_jobs([item])[0]
    assert set(job["family_of"].values()) == {"GAP"}
    cfg = CompensationConfig(family_of=job["family_of"],
                             amp_bounds_by_family=job["amp_bounds_by_family"])
    lat, _c, _v = c._load()
    nom = family_nominals(lat, cfg)
    for e in lat.elements:
        if type(e).__name__ == "Quadrupole":
            lo, hi = amp_bounds(cfg, e.name, e.gradient, nom)
            assert lo < e.gradient < hi and lo * hi > 0, (e.name, e.gradient, lo, hi)


def test_a_cavitys_window_holds_its_own_setting():
    """Within a type the settings ramp: a cavity above headroom x the type's median kept a
    window it could not return to its own setting in."""
    fam = {"FMAP_001": "HWRDonut", "FMAP_002": "HWRDonut", "FMAP_005": "HWRDonut"}
    cfg = CompensationConfig(family_of=fam, amp_bounds_by_family={"HWRDonut": (0.0, 1.5)})
    nominals = {"HWRDonut": 1.3}
    lo, hi = amp_bounds(cfg, "FMAP_005", 2.0, nominals)
    assert lo <= 0.0 and hi == 2.0
    assert amp_bounds(cfg, "FMAP_001", 1.0, nominals) == (0.0, pytest.approx(1.95))
