"""A bend edited after it was built is the bend built with the new value.

Its arc length is |ρ|·|θ|.  It used to be set once, in ``__init__``, so after an edit of the angle or the radius (from
the inspector, a scan, ADJUST, the assistant or the API) the map took the new arc while positions, the reference
phase, space-charge slices, the roll offset and the MAD-X length kept the old one.  Now the length follows from the
angle and the radius, and while the bend has an angle it cannot be written: the two inputs commute, so edits in any
order and their undo are exact.  A bend with no angle is a field-free pipe and takes a length.
"""
import copy
import math
import pickle
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_data import FieldMapData
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
from linac_gen.tracking.tracker import Tracker

BASE = dict(angle=10.0, rho=800.0, e1=3.0, e2=2.0, field_index=0.3, aperture=40.0)
INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
            emit_z=0.2)


def _bend(**kw):
    return Dipole("B", **dict(BASE, **kw))


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=325.0)


def _cavity():
    """A cavity on the absolute RF phase: it sees the arrival time, so the length of the line before it."""
    z = np.linspace(0.0, 120.0, 121)
    fd = FieldMapData(z=z, Ez=2.0 * np.sin(np.pi * z / 120.0), symmetry="1d")
    return FieldMap("FM", length=120.0, field_data=fd, phase=-25.0, frequency=325.0, aperture=30.0, n_steps=125,
                    p_flag=0)


def _engines(bend):
    """What every engine gives a drift, the bend, a drift and a cavity on the absolute phase."""
    def line():
        lat = Lattice()
        for e in (Drift("D0", 100.0), bend, Drift("D1", 100.0), _cavity()):
            lat.add(e)
        return lat
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beam = Beam(_ref(), n_particles=40, current=0.0)
        beam.particles[:] = np.random.default_rng(4).normal(size=(40, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        Tracker(line(), beam).run()
        out["tracker"] = np.concatenate([beam.particles.ravel(), [beam.ref.w_kin, beam.ref.phi_s]])
        for current in (0.0, 20.0):
            res = EnvelopeSolver(line(), _ref(), dict(INIT), current=current).run()
            out[f"envelope {current} mA"] = np.concatenate([np.asarray(res.sigma_matrix[-1]).ravel(),
                                                           [res.ref_w_kin[-1], res.s[-1]]])
        out["matrix"] = compute_transfer_matrix(line(), _ref())
    return out


def _same(a, b):
    for engine in a:
        np.testing.assert_array_equal(a[engine], b[engine], err_msg=engine)


def _state(b):
    return (b.angle, b.rho, b.length)


@pytest.mark.parametrize("name, value", [("angle", 11.03), ("angle", -10.0), ("rho", 700.0), ("rho", -800.0)])
def test_an_edited_bend_is_the_bend_built_with_the_value(name, value):
    edited = _bend()
    setattr(edited, name, value)
    fresh = _bend(**{name: value})
    assert edited.length == fresh.length
    _same(_engines(edited), _engines(fresh))


def test_the_length_of_a_bend_follows_and_is_not_written():
    b = _bend()
    with pytest.raises(AttributeError, match="follows from the angle and the radius"):
        b.length = 150.0
    assert _state(b) == _state(_bend())


def test_edits_of_angle_and_radius_commute_and_undo_exactly():
    """Two inputs, the length derived from both: any order gives the same bend, and writing the old values back, in
    any order, gives the bend back to the last bit."""
    a, b = _bend(), _bend()
    a.angle, a.rho = 11.03, 700.0
    b.rho, b.angle = 700.0, 11.03
    assert _state(a) == _state(b) == _state(_bend(angle=11.03, rho=700.0))
    a.angle, a.rho = 10.0, 800.0
    b.rho, b.angle = 800.0, 10.0
    assert _state(a) == _state(b) == _state(_bend())


def test_a_plan_that_sets_a_bends_length_applies_nothing():
    """An assistant plan or tuning plan with the length of a bend that has an angle is refused as a whole."""
    from linac_gen.assist.tools import WorkContext
    b, d = _bend(), Drift("D1", 100.0)
    lat = Lattice()
    lat.add(b)
    lat.add(d)
    ctx = WorkContext()
    ctx.lattice = lat
    with pytest.raises(AttributeError):
        ctx.apply_param_changes([(d, "length", 150.0), (b, "angle", 11.0), (b, "length", 150.0)])
    assert d.length == 100.0 and _state(b) == _state(_bend())


def test_a_bend_with_no_angle_keeps_a_length_as_a_pipe():
    """Ring injection bumps are built so: no field, a length.  The bend stays a drift of that length, whatever is
    written to its radius or angle, and a pipe given an angle and turned back keeps its length."""
    b = Dipole("B", angle=0.0, rho=1.0e9)
    b.length = 300.0
    b.rho = 1.0e9
    b.angle = 0.0
    b.rho = 5.0e8
    assert b.length == 300.0
    b.angle = 1.0e-3
    assert b.length == pytest.approx(5.0e8 * math.radians(1.0e-3), rel=1e-15)
    b.angle = 0.0
    assert b.length == 300.0
    lat_b, lat_d = Lattice(), Lattice()
    lat_b.add(b)
    lat_d.add(Drift("D", 300.0))
    np.testing.assert_allclose(compute_transfer_matrix(lat_b, _ref()), compute_transfer_matrix(lat_d, _ref()),
                               rtol=1e-12, atol=1e-15)


def test_a_bend_set_to_no_angle_has_no_arc():
    """Its arc is |ρ|·|θ|, none at angle 0; its angle back gives it back exactly.  A field error of -1 is what switches
    a magnet off at its length."""
    b = _bend()
    b.angle = 0.0
    assert b.length == 0.0
    b.angle = 10.0
    assert _state(b) == _state(_bend())
    assert _bend(field_rel=-1.0).length == _bend().length


def _inputs(b):
    """Everything a bend's behaviour follows from: angle, radius, length and the length it was given as a pipe (an
    unset one and a pipe of no length behave alike)."""
    return (b.angle, b.rho, b.length, getattr(b, "_pipe_length", None) or 0.0)


def _pipe(length=300.0, rho=1.0e9):
    p = Dipole("B", angle=0.0, rho=rho)
    p.length = length
    return p


@pytest.mark.parametrize("seed", range(4))
def test_any_sequence_of_edits_undoes_and_redoes_step_by_step_exactly(seed):
    """Random edits of the angle (zero among them), the radius and a pipe's length, and of a drift beside it, through
    the GUI's command bus, which merges consecutive edits of one parameter into one step as a dragged editor makes
    them: every undo and every redo lands on the state of its step, to the last bit."""
    pytest.importorskip("PyQt6")
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.state import AppState
    rng = np.random.default_rng(seed)
    for b in (_bend(), _pipe()):
        d = Drift("D", 100.0)
        lat = Lattice()
        lat.add(b)
        lat.add(d)
        st = AppState()
        st.set_lattice(lat, None)
        levels = [(_inputs(b), d.length)]
        for _ in range(30):
            el, attr = [(b, "angle"), (b, "angle"), (b, "rho"), (b, "length"), (d, "length")][rng.integers(5)]
            value = {"angle": float(rng.choice([0.0, 1.0, 5.0, 10.0, -7.5])),
                     "rho": float(rng.choice([500.0, 800.0, 1.0e9])),
                     "length": float(rng.choice([150.0, 300.0]))}[attr]
            depth = len(st.bus._undo)
            try:
                st.bus.do(ParamChangeCommand(el, attr, getattr(el, attr), value))
            except AttributeError:
                assert (el, attr) == (b, "length") and b.angle != 0.0
                assert (_inputs(b), d.length) == levels[-1]
                continue
            if len(st.bus._undo) == depth:                # merged into the step before
                levels[-1] = (_inputs(b), d.length)
            else:
                levels.append((_inputs(b), d.length))
        for k in range(len(levels) - 1, 0, -1):
            st.bus.undo()
            assert (_inputs(b), d.length) == levels[k - 1]
        for k in range(1, len(levels)):
            st.bus.redo()
            assert (_inputs(b), d.length) == levels[k]


def test_merged_edits_to_no_angle_redo_as_they_were_made():
    """Three steps of an editor take a bend from 3 to 0 degrees in one undo step: undo and redo give the bend before
    and after."""
    pytest.importorskip("PyQt6")
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.state import AppState
    b = Dipole("B", angle=3.0, rho=800.0)
    lat = Lattice()
    lat.add(b)
    st = AppState()
    st.set_lattice(lat, None)
    for old, new in ((3.0, 2.0), (2.0, 1.0), (1.0, 0.0)):
        st.bus.do(ParamChangeCommand(b, "angle", old, new))
    assert len(st.bus._undo) == 1 and b.length == 0.0
    st.bus.undo()
    assert _state(b) == _state(Dipole("B", angle=3.0, rho=800.0))
    st.bus.redo()
    assert b.angle == 0.0 and b.length == 0.0


@pytest.mark.parametrize("make, changes", [
    (_pipe, lambda b: [(b, "length", 500.0), (b, "angle", 10.0)]),
    (_pipe, lambda b: [(b, "angle", 10.0), (b, "rho", 700.0)]),
    (_bend, lambda b: [(b, "angle", 0.0), (b, "length", 500.0)]),      # made a pipe
], ids=["pipe given a length and an angle", "pipe given an angle and a radius", "bend made a pipe"])
def test_a_rollback_puts_a_bend_back_exactly(make, changes):
    """A tuning plan rolled back, given the values as the plan's caller saw them before it: the bend as it was, and it
    behaves as before afterwards."""
    from linac_gen.assist.tools import WorkContext
    b, twin = make(), make()
    ctx = WorkContext()
    lat = Lattice()
    lat.add(b)
    ctx.lattice = lat
    before = _inputs(b)
    plan = changes(b)
    olds = [(e, a, getattr(e, a)) for e, a, _v in plan]
    handle = ctx.apply_param_changes(plan)
    ctx.revert_param_changes(handle, olds)
    assert _inputs(b) == before
    b.angle, twin.angle = 0.0, 0.0
    assert _inputs(b) == _inputs(twin)


def test_a_pipe_is_written_as_the_drift_it_is(tmp_path):
    """A BEND card's length is ρ·θ: a bend with no angle and a length is written as a drift, so the line keeps its
    length through a save and a reload."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin
    pipe = Dipole("B", angle=0.0, rho=1.0e9, aperture=30.0)
    pipe.length = 300.0
    lat = Lattice()
    for e in (Drift("D0", 100.0), pipe, Drift("D1", 100.0), Dipole("B0", angle=0.0, rho=1000.0)):
        lat.add(e)
    out = tmp_path / "line.dat"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_tracewin(lat, out)
        back, _ = parse_tracewin(str(out))
    assert back.total_length == pytest.approx(lat.total_length, rel=1e-12)
    assert "BEND" in out.read_text(encoding="latin-1")   # the bend with no angle and no length stays a BEND


def test_the_matrix_cache_follows_a_pipes_length():
    from linac_gen.tracking.matrix_tracking import get_element_matrix
    pipe = Dipole("B", angle=0.0, rho=1.0e9)
    pipe.length = 300.0
    cache = {}
    m300 = get_element_matrix(pipe, _ref(), cache=cache)
    pipe.length = 500.0
    assert get_element_matrix(pipe, _ref(), cache=cache)[0, 1] == pytest.approx(0.5, rel=1e-12)
    assert m300[0, 1] == pytest.approx(0.3, rel=1e-12)


def test_a_command_line_override_of_a_bends_length_is_a_clean_error():
    from linac_gen.cli.common import apply_element_override
    lat = Lattice()
    lat.add(_bend())
    with pytest.raises(ValueError, match="follows from the angle and the radius"):
        apply_element_override(lat, "B.length", "100")


def test_the_assistant_does_not_scan_a_bends_length(tmp_path):
    """Every point would refuse the value: the scan is refused before it starts, with the reason."""
    from linac_gen.assist.tools import TOOLS, WorkContext
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "line.dat"
    deck.write_text("DRIFT 100 30\nBEND 10 800 0 30 0\nDRIFT 100 30\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
    ctx = WorkContext(lattice=lat, lattice_path=str(deck), beam_config=BeamConfig())
    bend = next(e for e in lat.elements if isinstance(e, Dipole))
    r = TOOLS["parameter_scan"].fn(ctx, element=bend.name, attribute="length", start=100.0, stop=200.0)
    assert r["status"] == "refused" and "follows from" in r["data"]["message"]


def test_a_face_behind_an_aperture_is_recorded(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "b.dat"
    deck.write_text("EDGE 3 800 50 0.45 2.8 40 0\nAPERTURE 20 20 0\nBEND 10 800 0 40 0\nEDGE 2 800 50 0.45 2.8 40 0\n"
                    "END\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
    assert len(next(e for e in lat.elements if isinstance(e, Dipole))._edge_names) == 2


def test_a_rolled_bend_steers_by_its_edited_arc():
    edited = _bend(tilt_deg=2.0)
    edited.angle = 11.03
    fresh = _bend(tilt_deg=2.0, angle=11.03)
    np.testing.assert_array_equal(np.asarray(edited.roll_exit_offset(_ref()), dtype=float),
                                  np.asarray(fresh.roll_exit_offset(_ref()), dtype=float))


def test_the_mad_x_export_of_an_edited_bend_is_the_bend_built_with_the_value(tmp_path):
    from linac_gen.io.madx_writer import write_madx
    texts = []
    for bend, tag in ((_bend(), "edited"), (_bend(angle=11.03), "fresh")):
        if tag == "edited":
            bend.angle = 11.03
        lat = Lattice()
        for e in (Drift("D0", 100.0), bend, Drift("D1", 100.0)):
            lat.add(e)
        (tmp_path / tag).mkdir()
        out = tmp_path / tag / "line.madx"            # the file name goes into the title
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_madx(lat, out, _ref())
        texts.append(out.read_text(encoding="utf-8"))
    assert texts[0] == texts[1]


def test_a_copy_and_a_pickle_keep_the_edit_and_edit_alone():
    b = _bend()
    b.angle = 11.03
    fresh = _engines(_bend(angle=11.03))
    for other in (copy.deepcopy(b), pickle.loads(pickle.dumps(b))):
        _same(_engines(other), fresh)
    d = copy.deepcopy(b)
    d.rho = 700.0
    assert d.length == _bend(angle=11.03, rho=700.0).length and b.length == _bend(angle=11.03).length


def _face_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in caught if "pole faces" in str(w.message)]


def _imports(tmp_path):
    """The same bend with separate faces from MAD-X, TraceWin (an ADJUST card between face and bend) and elegant."""
    from linac_gen.io.elegant_parser import parse_elegant
    from linac_gen.io.madx_parser import parse_madx
    from linac_gen.io.tracewin_parser import parse_tracewin
    src = tmp_path / "b.madx"
    src.write_text("beam, particle=proton, energy=1.738272;\n"
                   "b1: sbend, l=1.0, angle=0.1, e1=0.03, e2=0.02;\n"
                   "s1: sequence, l=2.0, refer=centre;\n  b1, at=1.0;\nendsequence;\nuse, sequence=s1;\n")
    deck = tmp_path / "b.dat"
    deck.write_text("EDGE 3 800 50 0.45 2.8 40 0\nADJUST 1 1\nBEND 10 800 0 40 0\nEDGE 2 800 50 0.45 2.8 40 0\n"
                    "END\n")
    lte = tmp_path / "b.lte"
    lte.write_text("b1: sben, l=0.3, angle=0.25, e1=0.1, e2=0.1, hgap=0.02\nl: line=(b1)\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [parse_madx(str(src))[0], parse_tracewin(str(deck))[0], parse_elegant(str(lte), name="l")[0]]


def test_a_bend_read_with_separate_faces_warns_once_that_they_stay(tmp_path):
    """The importers make a bend's faces separate elements; an edit of the bend does not move them, and says so, once
    for the bend however often a scan or a match edits it."""
    from linac_gen.elements.edge import Edge
    for lat in _imports(tmp_path):
        bend = next(e for e in lat.elements if isinstance(e, Dipole))
        faces = [e.name for e in lat.elements if isinstance(e, Edge)]
        assert bend._edge_names == tuple(faces)

        def edits(b=bend):
            for v in (6.0, 7.0, 8.0):
                b.angle = v
        _, msgs = _face_warnings(edits)
        assert len(msgs) == 1 and all(f in msgs[0] for f in faces)


def test_no_warning_for_a_bend_without_faces_or_for_a_value_that_does_not_change(tmp_path):
    _, msgs = _face_warnings(lambda: setattr(_bend(), "angle", 11.0))
    assert msgs == []
    bend = next(e for e in _imports(tmp_path)[0].elements if isinstance(e, Dipole))
    _, msgs = _face_warnings(lambda: setattr(bend, "rho", bend.rho))
    assert msgs == []


def test_inert_faces_are_not_recorded(tmp_path):
    from linac_gen.io.madx_parser import parse_madx
    from linac_gen.io.tracewin_parser import parse_tracewin
    src = tmp_path / "plain.madx"
    src.write_text("beam, particle=proton, energy=1.738272;\n"
                   "b1: sbend, l=1.0, angle=0.1;\n"
                   "s1: sequence, l=2.0, refer=centre;\n  b1, at=1.0;\nendsequence;\nuse, sequence=s1;\n")
    deck = tmp_path / "plain.dat"
    deck.write_text("EDGE 0 800 0 0.45 2.8 40 0\nBEND 10 800 0 40 0\nEDGE 0 800 0 0.45 2.8 40 0\nEND\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lats = [parse_madx(str(src))[0], parse_tracewin(str(deck))[0]]
    for lat in lats:
        assert next(e for e in lat.elements if isinstance(e, Dipole))._edge_names == ()


def test_a_mad8_import_does_not_warn(tmp_path):
    """MAD8's import re-asserts a bend's radius after it is built; that is no edit of the bend."""
    from linac_gen.io.mad8_parser import parse_mad8
    p = tmp_path / "b.mad8"
    p.write_text("BRHO := 4.881\nB1: SBEND, L=1.0, ANGLE=0.1, E1=0.03, E2=0.07\nB2: RBEND, L=1.0, ANGLE=0.1\n"
                 "D1: DRIFT, L=0.5\nTOP: LINE=(D1, B1, D1, B2, D1)\n")
    (lat, _), msgs = _face_warnings(lambda: parse_mad8(str(p)))
    assert msgs == []
    assert any(e._edge_names for e in lat.elements if isinstance(e, Dipole))


def test_a_lattix_import_records_the_faces(tmp_path):
    """Bmad, SciBmad and PALS come in through lattix, which makes a bend's faces separate elements too."""
    from linac_gen.elements.edge import Edge
    from linac_gen.io.lattix_bridge import parse_with_lattix
    p = tmp_path / "b.bmad"
    p.write_text("parameter[particle] = proton\nparameter[p0c] = 1e9\n"
                 "b1: sbend, l = 1.0, angle = 0.1, e1 = 0.03, e2 = 0.02\n"
                 "d1: drift, l = 0.3\nl1: line = (d1, b1, d1)\nuse, l1\n")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lat, _ = parse_with_lattix(str(p))
    except Exception as exc:                                      # noqa: BLE001
        if "lattix" in str(exc).lower():
            pytest.skip(f"lattix not available: {exc}")
        raise
    bend = next(e for e in lat.elements if isinstance(e, Dipole))
    assert bend._edge_names == tuple(e.name for e in lat.elements if isinstance(e, Edge))
    assert bend._edge_names
