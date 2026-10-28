"""Ring optics of the simple Booster from the ring kernels, against MAD-X
(L1: tunes 1e-6 with the integer part, beta/alpha/D 1e-6, chromaticity 0.02
with and without the bends' K2, momentum compaction 1e-6) and against
MAD-X's exact-model TWISS (PTC_TWISS).

The references (``tests/ring/fixtures/refs/simple_booster_madx*.json``) come
from a dev-only script (MAD-X 5.09 via cpymad, RF
voltage zero; the no-K2 variant edits the deck's two bend definitions); their
manifests record the commands.  MAD-X's TWISS chromaticity and its exact-model
PTC_TWISS (MAD-X's default fringe settings) agree to 0.003; ``twiss, chrom``'s
finite difference differs by 0.24 — it is not the reference.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.optics import rf_bucket, twiss
from linac_gen.ring.reference import RingReference

FIX = Path(__file__).resolve().parent / "fixtures"
DECK = FIX / "simple_booster.madx"
REF = json.loads((FIX / "refs" / "simple_booster_madx.json").read_text())
REF_NOK2 = json.loads((FIX / "refs" / "simple_booster_madx_nok2.json").read_text())
REF_SF = json.loads((FIX / "refs" / "simple_booster_madx_sector_fringe.json").read_text())


def _deck(tmp_path, subs=()):
    text = DECK.read_text(encoding="latin-1")
    for sub in subs:
        old, _sep, new = sub.partition("=>")
        assert old in text
        text = text.replace(old, new)
    p = tmp_path / "deck.madx"
    p.write_text(text)
    return p


def _optics(path, **options):
    lat, meta = parse_madx(str(path))
    w = meta["reference"].w_kin
    ref = RingReference.from_kinetic("proton", w)
    rl = compile_ring(lat, ref, lattice_energy=w, **options)
    return rl, twiss(rl)


@pytest.fixture(scope="module")
def booster(tmp_path_factory):
    return _optics(DECK)


def test_references_belong_to_the_deck():
    for name in ("simple_booster_madx", "simple_booster_madx_nok2",
                 "simple_booster_madx_sector_fringe"):
        p = FIX / "refs" / f"{name}.json"
        man = json.loads(p.with_suffix(".manifest.json").read_text())
        assert man["output_sha256"] == hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert man["inputs"][DECK.name] == hashlib.sha256(
            DECK.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_tunes_with_their_integer_part(booster):
    _rl, o = booster
    assert o.qx == pytest.approx(REF["q1"], abs=1e-6)
    assert o.qy == pytest.approx(REF["q2"], abs=1e-6)
    assert math.floor(o.qx) == 6 and math.floor(o.qy) == 6


def test_twiss_and_dispersion_at_the_start(booster):
    _rl, o = booster
    st = REF["start"]
    assert o.start["betx"] == pytest.approx(st["betx"], rel=1e-6)
    assert o.start["bety"] == pytest.approx(st["bety"], rel=1e-6)
    assert o.start["alfx"] == pytest.approx(st["alfx"], abs=1e-6)
    assert o.start["alfy"] == pytest.approx(st["alfy"], abs=1e-6)
    assert o.start["dx"] == pytest.approx(st["dx_delta"], rel=1e-6)
    assert o.start["dpx"] == pytest.approx(st["dpx_delta"], abs=1e-6)


def test_twiss_along_the_ring(booster):
    # at every quadrupole and at the cell ends (rows whose exit position
    # and name both match; MAD-X's bend rows include the exit pole face,
    # HELIX writes it as a separate element, so bends are not compared)
    _rl, o = booster
    tab = REF["table"]
    s_h = np.asarray(o.table["s"])
    names_h = o.table["name"]
    n = 0
    for i, (nm, kw) in enumerate(zip(tab["name"], tab["keyword"], strict=True)):
        if kw.lower() not in ("quadrupole", "drift") or nm.split(":")[0] not in ("qsxx", "qlxx", "sc"):
            continue
        j = np.flatnonzero((np.abs(s_h - tab["s"][i]) < 1e-9)
                           & np.array([h == nm.split(":")[0] for h in names_h]))
        assert j.size == 1, nm
        j = int(j[0])
        assert o.table["betx"][j] == pytest.approx(tab["betx"][i], rel=1e-6)
        assert o.table["bety"][j] == pytest.approx(tab["bety"][i], rel=1e-6)
        assert o.table["alfx"][j] == pytest.approx(tab["alfx"][i], abs=1e-6)
        assert o.table["alfy"][j] == pytest.approx(tab["alfy"][i], abs=1e-6)
        assert o.table["dx"][j] == pytest.approx(tab["dx_delta"][i], abs=1e-6)
        assert o.table["mux"][j] == pytest.approx(tab["mux"][i], abs=1e-6)
        assert o.table["muy"][j] == pytest.approx(tab["muy"][i], abs=1e-6)
        n += 1
    assert n == 72                       # 24 cells x (two quadrupoles + the cell end)


def test_momentum_compaction_and_transition(booster):
    _rl, o = booster
    assert o.alpha_c == pytest.approx(REF["alfa"], rel=1e-6)
    assert o.gamma_t == pytest.approx(REF["gammatr"], rel=1e-6)
    assert o.eta == pytest.approx(REF["alfa"] - 1.0 / o.gamma0 ** 2, rel=1e-6)


@pytest.mark.parametrize("variant", ["with K2", "without K2"])
def test_chromaticity_against_madx_twiss_and_ptc_twiss(tmp_path, variant):
    if variant == "with K2":
        ref, (_rl, o) = REF, _optics(DECK)
    else:
        ref, (_rl, o) = REF_NOK2, _optics(_deck(tmp_path, REF_NOK2["substitutions"]))
    assert o.dqx == pytest.approx(ref["dq1_delta"], abs=0.02)
    assert o.dqy == pytest.approx(ref["dq2_delta"], abs=0.02)
    # HELIX's default pole face is that of MAD-X's exact model (PTC_TWISS):
    # the two agree closely
    assert o.dqx == pytest.approx(ref["ptc"]["dq1_delta"], abs=2e-3)
    assert o.dqy == pytest.approx(ref["ptc"]["dq2_delta"], abs=2e-3)


def test_pole_face_models_match_the_reference_optics():
    # reference code A's 4-D optics of the same deck (its default bend model,
    # 98 kicks) for its linear and full pole faces
    ref_a = json.loads((FIX / "refs" / "simple_booster_cartesian_ref.json").read_text())
    for edge in ("linear", "full"):
        _rl, o = _optics(DECK, edge_model=edge)
        assert o.qx == pytest.approx(ref_a[edge]["qx"], abs=1e-7)
        assert o.qy == pytest.approx(ref_a[edge]["qy"], abs=1e-7)
        assert o.dqx == pytest.approx(ref_a[edge]["dqx"], abs=2e-3)
        assert o.dqy == pytest.approx(ref_a[edge]["dqy"], abs=2e-3)


def test_the_bend_k2_matters():
    # (the importer's K2 bug of the plan: without the bends' K2 the vertical
    # chromaticity is -21.8, with it -8.09)
    assert REF_NOK2["dq2_delta"] < -21.0 and REF["dq2_delta"] > -8.2


def test_six_d_synchrotron_tune_and_bucket(booster):
    from linac_gen.cli.common import load_lattice
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.optics import ring_optics
    lat = load_lattice(str(DECK))
    o6 = ring_optics(lat, six_d=True)
    assert o6.mode == "6d" and np.abs(o6.closed_orbit).max() < 1e-10
    bucket = rf_bucket(compile_from_config(lat, RingConfig()), o6)
    assert bucket["harmonic"] == 84 and bucket["voltage_mv"] == pytest.approx(0.2, rel=1e-12)
    assert bucket["stable"] and bucket["lag_deg"] == pytest.approx(0.0, abs=1e-12)
    # 22 discrete cavities vs the smooth small-amplitude formula
    assert o6.qs == pytest.approx(bucket["qs"], rel=1e-3)
    # bucket half-height: 2 V / (pi h |eta| beta^2 E), analytic
    E = o6.gamma0 * 938.27208816
    want = math.sqrt(2 * 0.2 / (math.pi * 84 * abs(o6.eta) * o6.beta0 ** 2 * E))
    assert bucket["delta_max"] == pytest.approx(want, rel=1e-12)
    assert bucket["separatrix"](0.0) == pytest.approx(want, rel=1e-12)
    assert np.isnan(bucket["separatrix"](0.6 * bucket["bucket_length_m"]))


def test_auto_steps_are_converged(booster):
    # the automatic bend steps keep the tunes within 1e-8 of a twice finer run
    rl, o = booster
    lat, meta = parse_madx(str(DECK))
    w = meta["reference"].w_kin
    ref = RingReference.from_kinetic("proton", w)
    from linac_gen.ring.compile import auto_bend_steps
    steps = auto_bend_steps(2.889612, 1 / 40.847086, 0.05410921561, "yoshida4")
    fine = twiss(compile_ring(lat, ref, lattice_energy=w, bend_steps=2 * steps), chromaticity=False)
    assert abs(fine.qx - o.qx) < 1e-8 and abs(fine.qy - o.qy) < 1e-8


@pytest.mark.parametrize("options", [{}, {"edge_model": "full"}, {"edge_model": "linear"},
                                     {"model": "curvilinear"}])
def test_sector_pole_faces_with_a_fringe_field(tmp_path, options):
    # pole faces at zero angle with a gap and a fringe integral (the bends of
    # the deck made sector bends, hgap 5 cm, fint 0.5) still focus
    # vertically through their fringe field, in every edge model — MAD-X's
    # -h tan(-psi); without it Qy is 4.7e-3 higher
    _rl, o = _optics(_deck(tmp_path, REF_SF["substitutions"]), **options)
    assert o.qx == pytest.approx(REF_SF["q1"], abs=1e-6)
    assert o.qy == pytest.approx(REF_SF["q2"], abs=1e-6)
    assert o.start["bety"] == pytest.approx(REF_SF["start"]["bety"], rel=1e-6)
