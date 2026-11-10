"""The MAD-X export's own reference walk: what it does when the walk cannot go on,
which length it writes for an element a run sizes for its beam, and that it leaves
the lattice it is given as it was.

1. An energy change to zero energy or below is exported as MARKER + body DRIFT with
   one warning (a refusal that says why under strict), not a bare ``math domain
   error``: an explicit matrix element's gain or energy offset, to zero exactly (which
   the kinematics pass) or below; the orbit the maps are written about, taken there by
   a cavity; the re-export of a deck whose kick6 loses the importer's reference.
2. An element a run sizes for the beam that reaches it is written with that length.
   A curved field-map cluster's path, in a uniform field, is a circle of radius
   rho' = rho p'/p cut by the exit plane, in closed form; after an energy card the
   export skips, the deck's reference still reaches it at the card's energy.  An NCELLS
   with beta_g <= 0 builds its cells for its beam.  The length does not depend on what
   the element went through before, plain or linearised; where the deck's own walk
   cannot size it, it is the length before any run.  No element of the lattice
   changes, and a line the walk is not needed for is not copied.
3. When the export's walk loses the reference, the warning gives the cause: an energy
   card the export skipped, when the deck's own walk gets past that point; a curved
   cluster whose path cannot be traced; a negative gain; or the deck's RF decelerating
   the species.  A card that cannot apply at the energy the export stays at is a
   comment, as the others are.
"""
from __future__ import annotations

import copy
import math
import pickle
import re
import types
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import SetReferenceEnergy
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.ncells import NCells
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.thin_lens import ThinLens
from linac_gen.io import madx_writer
from linac_gen.io.madx_parser import parse_madx
from linac_gen.io.madx_writer import write_madx
from linac_gen.tracking.matrix_tracking import reference_at
from linac_gen.tracking.tracker import Tracker
from tests.elements.test_curved_field_map import FREQ, RHO, W, _sector


def _ref(w, species=PROTON, f=352.21):
    return ReferenceParticle(species=species, w_kin=w, frequency=f)


def _lattice(*elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    return lat


def _state(lat):
    """Every element's state, to the byte."""
    return [pickle.dumps(e) for e in lat.elements]


def _export(lat, path, ref, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msgs = write_madx(lat, path, ref, **kw)
    return msgs, path.read_text(encoding="utf-8")


def _mine(msgs, prefix):
    return [m for m in msgs if m.startswith(prefix)]


# --------------------------------------------------------------------------- #
#  1. To zero energy or below: MARKER + DRIFT, not a crash
# --------------------------------------------------------------------------- #
def test_an_energy_offset_below_zero_is_a_marker_not_a_crash(tmp_path):
    lat = _lattice(MatrixElement("m", np.eye(6), length=100.0,
                                 offset=np.array([0, 0, 0, 0, 0, -30.0])),
                   Drift("d", 100.0))
    msgs, txt = _export(lat, tmp_path / "a.madx", _ref(20.0), linearize=True)
    assert _mine(msgs, "'m' ") == [
        "'m' (MatrixElement): its energy change ΔW=-30 MeV takes the reference to zero "
        "energy or below (W=20 MeV) — exported as MARKER + body DRIFT instead of MATRIX"]
    assert re.search(r"^m: MARKER;$", txt, re.M)
    assert re.search(r"^m_body: DRIFT, l=0\.1;$", txt, re.M)
    assert "MATRIX" not in txt
    assert re.search(r"^\s*d, at=0\.1;$", txt, re.M)          # the line goes on after it
    with pytest.raises(ValueError, match=r"MAD-X export refused: 'm' \(MatrixElement\): "
                                         r"its energy change ΔW=-30 MeV takes the reference "
                                         r"to zero energy or below"):
        write_madx(lat, tmp_path / "b.madx", _ref(20.0), linearize=True,
                   on_unsupported="error")


@pytest.mark.parametrize("gain, off", [(-20.0, 0.0), (0.0, -20.0), (-25.0, 0.0), (-15.0, -10.0)])
def test_an_energy_change_to_zero_or_below_is_one_message(tmp_path, gain, off):
    """To zero exactly, which the kinematics pass (and the maps then cannot convert
    about), or below; from the element's gain, its offset or both: one message."""
    lat = _lattice(MatrixElement("m", np.eye(6), length=100.0, ref_gain=gain,
                                 offset=np.array([0, 0, 0, 0, 0, off]) if off else None),
                   Drift("d", 100.0))
    msgs, txt = _export(lat, tmp_path / "z.madx", _ref(20.0), linearize=True)
    assert _mine(msgs, "'m' ") == [
        f"'m' (MatrixElement): its energy change ΔW={gain + off:g} MeV takes the reference "
        "to zero energy or below (W=20 MeV) — exported as MARKER + body DRIFT instead of "
        "MATRIX"], msgs
    assert re.search(r"^m: MARKER;$", txt, re.M) and "MATRIX" not in txt


def test_an_orbit_a_cavity_takes_to_zero_energy_is_a_marker(tmp_path):
    """An explicit matrix's energy offset is the orbit's gain (the import reads it as the
    reference's), so a cavity after it can take that orbit below zero energy while
    HELIX's reference goes on: the next map cannot be written about it."""
    lat = _lattice(MatrixElement("m", np.eye(6), length=100.0,
                                 offset=np.array([0, 0, 0, 0, 0, -15.0])),
                   Drift("d1", 50.0), RFGap("g", -10.0, 0.0, 352.21), Drift("d2", 50.0),
                   ThinLens("tl", 2000.0, -2000.0), Drift("d3", 50.0))
    assert reference_at(lat, _ref(20.0), 4).w_kin == pytest.approx(10.0, rel=1e-12)
    msgs, txt = _export(lat, tmp_path / "o.madx", _ref(20.0), linearize=True)
    assert re.search(r"^m: MATRIX", txt, re.M)          # (the orbit after it: 20 - 15 MeV)
    assert _mine(msgs, "'tl' ") == [
        "'tl' (ThinLens): the reference orbit it would be written about is at zero energy "
        "or below there (W=-5 MeV, after the RF cavities before it) — exported as MARKER "
        "+ body DRIFT instead of MATRIX"], msgs
    assert re.search(r"^tl: MARKER;$", txt, re.M)
    with pytest.raises(ValueError, match=r"MAD-X export refused: 'tl' \(ThinLens\): "
                                         r"the reference orbit it would be written about"):
        write_madx(lat, tmp_path / "o2.madx", _ref(20.0), linearize=True,
                   on_unsupported="error")


def test_a_deck_whose_kick6_loses_the_reference_exports_again(tmp_path):
    deck = tmp_path / "k.madx"
    deck.write_text("beam, particle=proton, energy=0.95827208816;\n"     # W = 20 MeV
                    "k: matrix, l=0.1, kick6=-1;\nd: drift, l=0.1;\n"
                    "line_k: line=(k, d);\nuse, period=line_k;\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_madx(deck)
    assert any("kick6" in m and "lost" in m for m in meta["warnings"])
    msgs, txt = _export(lat, tmp_path / "k2.madx", meta["reference"], linearize=True)
    assert any("takes the reference to zero energy or below" in m
               and "exported as MARKER + body DRIFT instead of MATRIX" in m for m in msgs), msgs
    assert re.search(r"^k: MARKER;$", txt, re.M)
    assert re.search(r"^\s*d, at=0\.1;$", txt, re.M)


# --------------------------------------------------------------------------- #
#  2. The length a run gives an element it sizes for its beam
# --------------------------------------------------------------------------- #
def _curved_ref(w=W):
    return ReferenceParticle(PROTON, w, FREQ)


def _arc_path(w_kin):
    """The reference's path through ``_sector(30)``'s uniform field at ``w_kin``: a
    circle of radius rho' from the origin along +z, cut by the exit plane (the
    design circle's, through its 30-degree point, normal to it there):
    rho' [sin(phi - a) + sin a] = rho sin a."""
    p = math.sqrt(W * (W + 2.0 * PROTON.mass))
    p2 = math.sqrt(w_kin * (w_kin + 2.0 * PROTON.mass))
    rho2 = RHO * p2 / p
    a = math.radians(30.0)
    return rho2 * (a + math.asin((RHO / rho2 - 1.0) * math.sin(a)))


def _written(txt, name, after="d1"):
    m = re.search(rf"^{name}\S*: (DRIFT|MATRIX), l=([0-9.e+-]+)", txt, re.M | re.I)
    at = re.search(rf"^\s*{after}, at=([0-9.e+-]+);", txt, re.M | re.I)
    return m.group(1), float(m.group(2)) * 1e3, float(at.group(1)) * 1e3


@pytest.mark.parametrize("history", ["untraced", "traced at another energy"])
def test_a_curved_cluster_is_written_with_its_path_for_the_beam_reaching_it(tmp_path, history):
    cl = _sector(30.0)
    gap = RFGap("G", 0.1, -30.0, FREQ)                 # +0.0866 MeV before the cluster
    lat = _lattice(Drift("D0", 50.0), gap, cl, Drift("D1", 100.0))
    w_cl = reference_at(copy.deepcopy(lat), _curved_ref(), 2).w_kin
    assert w_cl == pytest.approx(W + 0.1 * math.cos(math.radians(30.0)), rel=1e-14)
    s = _arc_path(w_cl)
    assert abs(s - RHO * math.radians(30.0)) > 0.05     # not the design path
    if history == "untraced":
        assert cl.length_provisional and abs(cl.length - s) > 1.0
    else:
        cl.prepare_reference(_curved_ref())             # the design beam's path, rho theta
        assert cl.length == pytest.approx(RHO * math.radians(30.0), rel=1e-12)
    before = _state(lat)
    for lin, kind in ((False, "DRIFT"), (True, "MATRIX")):
        msgs, txt = _export(lat, tmp_path / f"c{int(lin)}.madx", _curved_ref(), linearize=lin)
        k, length, at_d1 = _written(txt, "c30")
        assert k == kind
        assert length == pytest.approx(s, rel=1e-12)
        assert at_d1 == pytest.approx(50.0 + s, rel=1e-12)    # and every s after it
        assert _state(lat) == before                    # the lattice is as it was given


def test_two_curved_clusters_each_have_the_path_for_their_beam(tmp_path):
    """A gap before each: the second cluster's beam has both gains, once."""
    cl, cl2 = _sector(30.0), _sector(30.0)
    cl2.name = "C30B"
    lat = _lattice(Drift("D0", 50.0), RFGap("G", 0.1, -30.0, FREQ), cl, Drift("D1", 100.0),
                   RFGap("G2", 0.2, -30.0, FREQ), cl2, Drift("D2", 100.0))
    gain = math.cos(math.radians(30.0))
    s1, s2 = _arc_path(W + 0.1 * gain), _arc_path(W + 0.3 * gain)
    for lin in (False, True):
        _msgs, txt = _export(lat, tmp_path / f"t{int(lin)}.madx", _curved_ref(), linearize=lin)
        assert _written(txt, "c30")[1] == pytest.approx(s1, rel=1e-12)
        _k, length, at_d2 = _written(txt, "c30b", after="d2")
        assert length == pytest.approx(s2, rel=1e-12)
        assert at_d2 == pytest.approx(150.0 + s1 + s2, rel=1e-12)


@pytest.mark.parametrize("w0", [2.0, 0.1])
def test_a_curved_cluster_after_a_skipped_energy_card_has_the_decks_path(tmp_path, w0):
    """The export stays at the energy before the card (MAD-X holds one reference
    energy), but a length is geometry: the deck's reference reaches the cluster at the
    card's energy, the design one, whose path is rho theta.  At 0.1 MeV the export's own
    walk cannot trace the cluster at all, and the gap after it says why."""
    cl = _sector(30.0)
    lat = _lattice(Drift("D0", 50.0), SetReferenceEnergy("E1", energy_MeV=W), cl,
                   Drift("D1", 100.0), RFGap("G", 0.01, -30.0, FREQ), Drift("D2", 10.0))
    for lin in (False, True):
        msgs, txt = _export(lat, tmp_path / f"e{int(lin)}.madx", _curved_ref(w0), linearize=lin)
        _k, length, at_d1 = _written(txt, "c30")
        assert length == pytest.approx(RHO * math.radians(30.0), rel=1e-12)
        assert at_d1 == pytest.approx(50.0 + RHO * math.radians(30.0), rel=1e-12)
        if w0 == 0.1:
            lost = _mine(msgs, "reference particle lost at 'C30'")
            assert lost and "energy card(s) 'E1'" in lost[0], msgs


@pytest.mark.parametrize("history", ["untraced", "traced for another beam"])
def test_a_curved_cluster_the_deck_cannot_trace_keeps_its_length_before_any_run(tmp_path, history):
    """The beam bends the other way and leaves the maps: no path, for it or for the
    cluster after it, which the deck's reference never reaches.  The gap after them
    walks the export's reference into the first, and the loss says why."""
    cl, cl2 = _sector(90.0), _sector(90.0)
    cl2.name = "C90B"
    if history != "untraced":
        for c in (cl, cl2):
            c.prepare_reference(_curved_ref())          # a proton at the design energy
            assert c.length == pytest.approx(RHO * math.pi / 2.0, rel=1e-12)
    lat = _lattice(Drift("D0", 50.0), cl, Drift("D1", 100.0), cl2, Drift("D2", 100.0),
                   RFGap("G", 0.01, -30.0, FREQ), Drift("D3", 10.0))
    before = _state(lat)
    msgs, txt = _export(lat, tmp_path / "u.madx", _ref(200.0, H_MINUS, FREQ))  # bends away
    # through the vertex where the entrance axis meets the exit axis
    estimate = 2.0 * RHO * math.tan(math.radians(45.0))
    assert _written(txt, "c90")[1] == pytest.approx(estimate, rel=1e-12)
    assert _written(txt, "c90b", after="d2")[1] == pytest.approx(estimate, rel=1e-12)
    tail = f"written with its length before any run, {estimate:.6g} mm"
    note = _mine(msgs, "'C90' (CurvedFieldMapCluster): its reference path cannot be traced")
    assert note and note[0].endswith(tail), msgs
    note = _mine(msgs, "'C90B' (CurvedFieldMapCluster): the deck's own reference does not "
                       "reach it (it does not get past 'C90')")
    assert note and note[0].endswith(tail), msgs
    lost = _mine(msgs, "reference particle lost at 'C90' (element 2) — its reference path "
                       "cannot be traced")
    assert lost, msgs
    assert _state(lat) == before


@pytest.mark.parametrize("history", ["fresh", "after a run for another beam"])
def test_a_cavity_sized_at_entry_is_written_with_its_length_for_the_beam(tmp_path, history):
    """An NCELLS with beta_g <= 0 builds its cells for the beam that reaches it; until
    then it carries a provisional over-estimate, and after a run that run's length."""
    nc = NCells("NC0", mode=1, n_cells=6, beta_g=0.0, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=1, frequency_mhz=352.21)
    lat = _lattice(Drift("D0", 100.0), RFGap("G", 1.0, -30.0, 352.21), nc, Drift("D1", 100.0))
    ref = _ref(20.0)

    def run(lattice, r):
        beam = Beam(r.copy(), n_particles=1, current=0.0)
        beam.particles[:] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tracker(lattice, beam).run()
        return lattice.elements[2].length

    want = run(copy.deepcopy(lat), ref)                 # the length a run gives it
    if history != "fresh":
        run(lat, _ref(25.0))
    assert abs(want - nc.length) > 1.0                  # not the length it carries now
    before = _state(lat)
    for lin in (False, True):
        _msgs, txt = _export(lat, tmp_path / f"n{int(lin)}.madx", ref, linearize=lin)
        _k, length, at_d1 = _written(txt, "nc0")
        assert length == pytest.approx(want, rel=1e-12)
        assert at_d1 == pytest.approx(100.0 + want, rel=1e-12)
        assert _state(lat) == before


def test_a_cavity_the_deck_brings_no_energy_to_keeps_its_length_before_any_run(tmp_path):
    """A gap that takes the deck's reference to zero energy exactly (which the kinematics
    pass) before the cavity: there is no beam to size it for."""
    def cavity():
        return NCells("NC0", mode=1, n_cells=6, beta_g=0.0, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                      aperture_mm=15.0, p_flag=1, frequency_mhz=352.21)
    nc = cavity()
    nc.prepare_reference(_ref(25.0))                    # sized for another beam
    provisional = cavity().length
    assert abs(nc.length - provisional) > 1.0
    lat = _lattice(Drift("D0", 100.0), RFGap("G0", -20.0, 0.0, 352.21), nc, Drift("D1", 100.0))
    msgs, txt = _export(lat, tmp_path / "w.madx", _ref(20.0))
    assert _written(txt, "nc0")[1] == pytest.approx(provisional, rel=1e-12)
    note = _mine(msgs, "'NC0' (NCells): the deck's own reference does not reach it (W = 0 MeV)")
    assert note, msgs


def test_the_walk_copies_the_lattice_once_and_only_when_it_is_needed(tmp_path, monkeypatch):
    real, copied = copy.deepcopy, []

    def spy(x, *a, **k):
        if isinstance(x, Lattice):
            copied.append(x)
        return real(x, *a, **k)

    monkeypatch.setattr(madx_writer, "copy", types.SimpleNamespace(deepcopy=spy))
    write_madx(_lattice(Drift("d0", 100.0), Quadrupole("q", 50.0, 10.0), Drift("d1", 100.0)),
               tmp_path / "p.madx", _ref(20.0))
    assert copied == []                                 # drifts and magnets: no walk
    write_madx(_lattice(Drift("d0", 100.0), RFGap("g1", 0.1, -30.0, 352.21), Drift("d1", 100.0),
                        RFGap("g2", 0.1, -30.0, 352.21), Drift("d2", 100.0)),
               tmp_path / "g.madx", _ref(20.0))
    assert len(copied) == 1


# --------------------------------------------------------------------------- #
#  3. The loss names its cause
# --------------------------------------------------------------------------- #
def _gaps(*first, v1=-3.0):
    return (*first, Drift("d1", 50.0), RFGap("G1", v1, 0.0, 352.21), Drift("d2", 50.0),
            RFGap("G2", 0.5, -30.0, 352.21), Drift("d3", 50.0))


def test_a_loss_after_a_skipped_energy_card_names_the_card(tmp_path):
    lat = _lattice(*_gaps(Drift("d0", 50.0), SetReferenceEnergy("E1", energy_MeV=5.0, shift=True)))
    # the deck's own walk keeps its reference: 1 + 5 - 3 + 0.5 cos 30
    assert reference_at(lat, _ref(1.0), len(lat.elements)).w_kin == pytest.approx(
        3.0 + 0.5 * math.cos(math.radians(30.0)), rel=1e-12)
    msgs, _ = _export(lat, tmp_path / "e.madx", _ref(1.0))
    lost = _mine(msgs, "reference particle lost at 'G1'")
    assert lost and "energy card(s) 'E1'" in lost[0], msgs
    assert "decelerates" not in lost[0]
    # without the card the deck's RF does decelerate the species below zero
    msgs, _ = _export(_lattice(*_gaps(Drift("d0", 50.0))), tmp_path / "n.madx", _ref(1.0))
    lost = _mine(msgs, "reference particle lost at 'G1'")
    assert lost and "decelerates below zero energy in the deck's RF" in lost[0], msgs
    # and with the card, when the deck's own walk loses the reference there too
    lat = _lattice(*_gaps(Drift("d0", 50.0), SetReferenceEnergy("E1", energy_MeV=5.0, shift=True),
                          v1=-10.0))
    with pytest.raises(ValueError):
        reference_at(lat, _ref(1.0), len(lat.elements))
    msgs, _ = _export(lat, tmp_path / "x.madx", _ref(1.0))
    lost = _mine(msgs, "reference particle lost at 'G1'")
    assert lost and "decelerates below zero energy in the deck's RF" in lost[0], msgs
    assert "'E1'" not in lost[0]
    # or takes it to zero energy exactly, which the kinematics pass: 1 + 5 - 6
    lat = _lattice(*_gaps(Drift("d0", 50.0), SetReferenceEnergy("E1", energy_MeV=5.0, shift=True),
                          v1=-6.0))
    assert reference_at(lat, _ref(1.0), 4).w_kin == 0.0
    msgs, _ = _export(lat, tmp_path / "y.madx", _ref(1.0))
    lost = _mine(msgs, "reference particle lost at 'G1'")
    assert lost and "'E1'" not in lost[0], msgs


@pytest.mark.parametrize("gain", [-25.0, -20.0])
def test_a_loss_at_a_negative_gain_names_the_gain(tmp_path, gain):
    """Below zero or to zero exactly: lost at the element itself, not at the drift after
    it, and not blamed on RF the line does not have before it."""
    lat = _lattice(MatrixElement("m", np.eye(6), length=100.0, ref_gain=gain), Drift("d1", 50.0),
                   RFGap("G", 0.1, -30.0, 352.21), Drift("d2", 50.0))
    msgs, _ = _export(lat, tmp_path / "g.madx", _ref(20.0))
    lost = [m for m in msgs if m.startswith("reference particle lost at")]
    assert lost == [f"reference particle lost at 'm' (element 1) — its reference gain of {gain:g} "
                    "MeV takes the reference to zero energy or below; the energy-dependent notes "
                    "(line gain, linearised maps) stop there, the static conversions are "
                    "unaffected"], msgs


def test_an_energy_card_that_cannot_apply_at_the_exports_energy_is_a_comment(tmp_path):
    """A shift down after a shift up: the deck ends where it began, but the export, which
    stays at the energy before the first card, cannot apply the second either."""
    lat = _lattice(Drift("d0", 50.0), SetReferenceEnergy("E1", energy_MeV=5.0, shift=True),
                   Drift("d1", 50.0), SetReferenceEnergy("E2", energy_MeV=-5.0, shift=True),
                   Drift("d2", 50.0))
    assert reference_at(lat, _ref(1.0), len(lat.elements)).w_kin == pytest.approx(1.0, rel=1e-12)
    msgs, txt = _export(lat, tmp_path / "s.madx", _ref(1.0))
    for nm in ("E1", "E2"):
        mine = _mine(msgs, f"'{nm}' (")
        assert mine and "MAD-X holds one reference energy" in mine[0], msgs
    assert re.search(r"^\s*d2, at=0\.1;$", txt, re.M)
