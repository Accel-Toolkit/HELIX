"""The Sacherer envelope reads a field map as every other physics path does.

The amplitude is the channel's own (scale x kb x (1 + voltage_rel) / norm_factor), so kb = 0
and the failures' OFF (voltage_rel = -1) are no field; B_z is read on the axis (a 3-D map's
interpolated at x = 0, y = 0); a map is a solenoid by the failure compensation's rule
(``field_map_data.on_axis_field_kind``), so a G(z) gradient map is none, and a 2-D cylindrical
solenoid map and a SUPERPOSE_MAP cluster focus; a steering corrector (a dipole field on the
axis with no quadrupole gradient of a quadrupole's size there) is left out, standing alone or
in a cluster, while the correctors of the map or of the cluster together bend the beam by at
most 20 mrad and their gradient on the axis, as the beam sees it (turned by its Larmor angle in
the element's B_z, the largest value along the element), focuses it by at most 1/f = 0.0088 /m,
and a B_z a corrector holds on the axis focuses as a solenoid's; a powered static map the
solver has no model of is refused, naming the element; an RF channel stays out (the DC
approximation); a cluster sums its children's B_z over their windows at the cluster's
amplitude.  Every map here is synthetic: the matrix envelope (``env_solver='matrix'``) through
the same map, on the same round DC beam, is the reference."""
import copy
import math
import re

import numpy as np
import pytest

from linac_gen.cli import common
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking.sacherer import SachererSolver

L = 200.0          # map length, mm
KB = 0.3           # the profiles peak at 1, so the solenoid's peak is 0.3 T
REL = 1e-3         # the Sacherer ODE against the matrix envelope's fitted kick-drift maps
EXACT = 1e-12      # the same field: the same ODE
ODE = 1e-6         # the same field up to rounding: the adaptive ODE's own tolerance


def _bump(z):
    return np.exp(-((np.asarray(z, dtype=float) - 0.5 * L) / 35.0) ** 2)


def _dbump(z):
    return -2.0 * (np.asarray(z, dtype=float) - 0.5 * L) / 35.0 ** 2 * _bump(z)


def _z(length=L, n=401):
    return np.linspace(0.0, length, n)


def _map1d(kb=KB, name="SOL", norm=1.0, **kw):
    z = _z()
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=_bump(z), norm_factor=norm)
    return FieldMap(name, length=L, field_data=fd, kb=kb, ke=0.0, aperture=30.0, **kw)


def _sol3(x=None, y=None, tilt=0.0):
    """A 3-D solenoid channel: B_z on the axis and the radial field of its slope off it.
    ``tilt`` (1/mm) makes B_z grow linearly off the axis -- the linear optics do not feel it,
    but a grid point off the axis then reads another B_z."""
    x = np.linspace(-10.0, 10.0, 5) if x is None else np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    z = _z()
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return FieldChannel(geometry=7, x=x, y=y, z=z, Fx=-0.5 * X * _dbump(Z),
                        Fy=-0.5 * Y * _dbump(Z), Fz=_bump(Z) * (1.0 + tilt * (X + Y)))


def _map3d(ch=None, kb=KB, name="SOL", norm=1.0, **kw):
    ch = _sol3() if ch is None else ch
    ch.norm_factor = norm
    fd = FieldMapData(z=ch.z)
    fd.channels[Channel.STAT_B] = ch
    return FieldMap3D(name, length=L, field_data=fd, kb=kb, ke=0.0, aperture=30.0, **kw)


def _grid3(name, fx, fy, fz, kb=1.0, x=None, y=None, **kw):
    """A 3-D static magnetic map sampled from functions of (X, Y, Z) [mm], on the grid ``x``
    in x and ``y`` in y (5 points over +-10 mm unless given; ``y`` is ``x`` unless given)."""
    x = np.linspace(-10.0, 10.0, 5) if x is None else np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    z = _z()
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=y, z=z, Fx=fx(X, Y, Z),
                                               Fy=fy(X, Y, Z), Fz=fz(X, Y, Z))
    return FieldMap3D(name, length=L, field_data=fd, kb=kb, ke=0.0, aperture=30.0, **kw)


def _quad3(**kw):            # a quadrupole: Bx = g y, By = g x, nothing on the axis
    return _grid3("Q3", lambda X, Y, Z: 0.05 * Y * _bump(Z), lambda X, Y, Z: 0.05 * X * _bump(Z),
                  lambda X, Y, Z: 0.0 * Z, **kw)


def _corrector3(name="B3", label=None, **kw):    # a dipole field across the axis
    m = _grid3(name, lambda X, Y, Z: 0.0 * Z, lambda X, Y, Z: 0.01 * _bump(Z) + 0.0 * X,
               lambda X, Y, Z: 0.0 * Z, **kw)
    m.label = label
    return m


def _gradient(**kw):         # a 1-D quadrupole gradient G(z), T/m
    kw.setdefault("kb", 1.0)
    z = _z()
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=9, z=z, Fz=5.0 * _bump(z))
    return FieldMap("QG", length=L, field_data=fd, ke=0.0, aperture=30.0, **kw)


def _electric(**kw):         # a static electric lens
    kw.setdefault("ke", 1.0)
    z = _z()
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_E] = FieldChannel(geometry=1, z=z, Fz=0.2 * _bump(z))
    return FieldMap("E1", length=L, field_data=fd, kb=0.0, aperture=30.0, **kw)


def _cart2d(**kw):           # a 2-D Cartesian (transverse) map
    x = np.linspace(-10.0, 10.0, 5)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fd = FieldMapData(z=_z())
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=6, x=x, y=x, Fx=0.01 * Y, Fy=0.01 * X)
    return FieldMap("C2", length=L, field_data=fd, kb=1.0, ke=0.0, aperture=30.0, **kw)


def _cyl2d():
    """A 2-D cylindrical solenoid map (B-type, geometry 5), stepped at 0.2 mm (the matrix
    envelope refines only a 1-D or 3-D magnetic map so).  B_z grows linearly with r off the
    axis, as ``_sol3``'s tilt: the linear optics do not feel it, another row reads it."""
    z = _z()
    r = np.linspace(0.0, 10.0, 6)
    Zg, Rg = np.meshgrid(z, r, indexing="ij")             # (n_z, n_r), as the reader lays it
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=5, z=z, r=r,
                                               Fz=_bump(Zg) * (1.0 + 0.02 * Rg),
                                               Fr=-0.5 * Rg * _dbump(Zg))
    return FieldMap("SOL2D", length=L, field_data=fd, kb=KB, ke=0.0, aperture=30.0,
                    n_steps=1000)


def _rf(stat_b=False, **kw):
    z = _z()
    fd = FieldMapData(z=z, frequency=162.5)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=0.5 * np.sin(np.pi * z / L))
    if stat_b:
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=_bump(z))
    return FieldMap("CAV", length=L, field_data=fd, kb=KB, ke=1.0, frequency=162.5,
                    phase=-30.0, aperture=30.0, **kw)


def _deck(tmp_path, norm=1.0):
    """The one-solenoid line as a TraceWin deck reading its map from sol.bsz, whose header
    gives ``norm`` (the file holds the field times it)."""
    z = _z()
    lines = [f"{len(z) - 1} {L * 1e-3:.6f}", f"{norm}"] + [f"{v:.15g}" for v in norm * _bump(z)]
    (tmp_path / "sol.bsz").write_text("\n".join(lines) + "\n")
    deck = tmp_path / "line.dat"
    deck.write_text(f"DRIFT 100 30 0\nSOL : FIELD_MAP 10 {L:g} 0 30 {KB} 0 0 0 sol\n"
                    "DRIFT 300 30 0\nEND\n")
    return deck


def _line(*middle, after=300.0):
    lat = Lattice()
    for e in (Drift("D0", 100.0, aperture=30.0), *middle, Drift("D1", after, aperture=30.0)):
        lat.add(e)
    return lat


def _drift_line(length=L):
    return _line(Drift("DM", length, aperture=30.0))


def _cfg(current=0.0, species="H-"):
    """A round DC beam at 30 keV (H- unless ``species`` says otherwise): the solenoids' x-y
    rotation leaves its sizes alone."""
    return BeamConfig(species=species, energy=0.030, frequency=162.5, current=current,
                      emit_nx=0.15, alpha_x=-1.0, beta_x=0.6, emit_ny=0.15, alpha_y=-1.0,
                      beta_y=0.6, emit_z=0.06, alpha_z=0.0, beta_z=800.0, continuous=True)


def _exit(lat, solver="sacherer", current=0.0, species="H-"):
    res = common.run_envelope_sim(copy.deepcopy(lat), _cfg(current, species), env_solver=solver)
    return np.array([res.sigma_x[-1], res.sigma_y[-1]])


def _ref(species=H_MINUS):
    return ReferenceParticle(species=species, w_kin=0.030, frequency=162.5)


_INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=1.0, alpha_y=0.0, beta_y=1.0, emit_y=1.0)


# ---------------------------------------------------------------------------
# the amplitude
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("current", [0.0, 5.0], ids=["I=0", "5 mA DC"])
@pytest.mark.parametrize("make", [_map1d, _map3d], ids=["1-D", "3-D"])
def test_at_every_amplitude_setting_a_solenoid_map_matches_the_matrix_envelope(make, current):
    """Nominal, an error draw (voltage_rel), a global scale and the file's normalisation:
    the matrix envelope through the same map is where the Sacherer lands."""
    nominal = _exit(_line(make()), current=current)
    assert nominal[0] < 0.8 * _exit(_drift_line(), current=current)[0]      # it focuses
    for kw in ({}, {"voltage_rel": 0.05}, {"scale": 0.5}, {"norm": 2.0}):
        lat = _line(make(**kw))
        np.testing.assert_allclose(_exit(lat, current=current),
                                   _exit(lat, "matrix", current), rtol=REL, err_msg=str(kw))


@pytest.mark.parametrize("make", [_map1d, _map3d], ids=["1-D", "3-D"])
def test_voltage_rel_scale_and_norm_are_the_amplitude_kb_is(make):
    np.testing.assert_allclose(_exit(_line(make(voltage_rel=0.05))),
                               _exit(_line(make(kb=KB * 1.05))), rtol=EXACT)
    half = _exit(_line(make(kb=KB * 0.5)))
    np.testing.assert_allclose(_exit(_line(make(scale=0.5))), half, rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(make(norm=2.0))), half, rtol=EXACT)
    assert abs(half[0] / _exit(_line(make()))[0] - 1.0) > 0.05            # it moved


@pytest.mark.parametrize("current", [0.0, 5.0], ids=["I=0", "5 mA DC"])
@pytest.mark.parametrize("make", [_map1d, _map3d], ids=["1-D", "3-D"])
def test_a_map_switched_off_is_no_field(make, current):
    """kb = 0 read as kb = 1 (a 1.0 T peak for a 0.3 T solenoid), and voltage_rel = -1 --
    a failure's OFF -- left the map on."""
    drift = _exit(_drift_line(), current=current)
    for kw in ({"kb": 0.0}, {"voltage_rel": -1.0}):
        lat = _line(make(**kw))
        np.testing.assert_allclose(_exit(lat, current=current), drift, rtol=EXACT,
                                   err_msg=str(kw))
        np.testing.assert_allclose(_exit(lat, "matrix", current), drift, rtol=REL,
                                   err_msg=str(kw))


def test_a_bsz_file_with_norm_2_reads_as_the_matrix_envelope_reads_it(tmp_path):
    """A TraceWin 1-D map whose header says Norm = 2, through the parser: the field is the
    file's over 2 (the Sacherer had it at twice the matrix envelope's)."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _meta = parse_tracewin(str(_deck(tmp_path, norm=2.0)))
    assert [type(e).__name__ for e in lat.elements] == ["Drift", "FieldMap", "Drift"]
    assert lat.elements[1].field_data.channels[Channel.STAT_B].norm_factor == 2.0
    got = _exit(lat)
    np.testing.assert_allclose(got, _exit(lat, "matrix"), rtol=REL)
    np.testing.assert_allclose(got, _exit(_line(_map1d())), rtol=ODE)


def test_a_flat_map_is_the_solenoid_card_at_its_error_up_to_the_end_of_the_element():
    """A map of constant B_z is the hard-edge SOLENOID card of that field, to the last stage
    of the last step: at 350 mm the ODE's z rounds past the element's end (the map's last
    node, beyond which a map has no field), and is held at it."""
    length = 350.0
    z = _z(length, 351)
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=np.ones_like(z))
    flat = FieldMap("FLAT", length=length, field_data=fd, kb=KB, ke=0.0, aperture=30.0,
                    voltage_rel=0.05)
    card = Solenoid("CARD", length=length, field=KB * 1.05, aperture=30.0)
    np.testing.assert_allclose(_exit(_line(flat)), _exit(_line(card)), rtol=1e-13)


# ---------------------------------------------------------------------------
# the axis
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [
    [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0],      # the middle index is at x = 10
    [-12.5, -7.5, -2.5, 2.5, 7.5, 12.5, 17.5],             # the axis falls between points
], ids=["axis-off-the-middle", "axis-between-points"])
def test_a_3d_maps_field_is_read_on_the_axis(x):
    m = _map3d(_sol3(x=x, tilt=0.02))
    got = _exit(_line(m))
    np.testing.assert_allclose(got, _exit(_line(_map3d())), rtol=ODE)
    np.testing.assert_allclose(got, _exit(_line(m), "matrix"), rtol=REL)


def test_a_3d_map_whose_grid_misses_the_axis_is_refused():
    m = _map3d(_sol3(x=[5.0, 10.0, 15.0]), name="OFFAXIS")
    with pytest.raises(ValueError, match=r"grid does not reach the axis.*'OFFAXIS'.*"
                                         r"env_solver='matrix'"):
        SachererSolver(_line(m), _ref(), dict(_INIT), current=0.0)


def test_a_2d_cylindrical_solenoid_map_is_read_on_its_axis():
    """Read on its r = 0 row, a 2-D cylindrical solenoid map focuses as the 1-D map of the
    same B_z does."""
    lat = _line(_cyl2d())
    got = _exit(lat)
    np.testing.assert_allclose(got, _exit(lat, "matrix"), rtol=REL)
    np.testing.assert_allclose(got, _exit(_line(_map1d())), rtol=EXACT)


# ---------------------------------------------------------------------------
# what is a solenoid
# ---------------------------------------------------------------------------
def test_the_on_axis_rule_names_each_kind_of_map():
    from linac_gen.io.field_map_data import on_axis_field_kind

    def ch(m):
        return next(iter(m.field_data.channels.values()))
    assert on_axis_field_kind(ch(_map1d())) == "longitudinal"
    assert on_axis_field_kind(ch(_cyl2d())) == "longitudinal"
    assert on_axis_field_kind(ch(_map3d())) == "longitudinal"
    assert on_axis_field_kind(ch(_quad3())) == "vanishing"
    assert on_axis_field_kind(ch(_corrector3())) == "transverse"
    assert on_axis_field_kind(ch(_gradient())) == "gradient"
    assert on_axis_field_kind(ch(_cart2d())) == "other"
    zero = _grid3("Z3", *(lambda X, Y, Z: 0.0 * Z,) * 3)
    assert on_axis_field_kind(ch(zero)) == "none"


_MATRIX = "use env_solver='matrix'"
_QUAD = "a 3-D map with a quadrupole gradient on the axis"


@pytest.mark.parametrize("make, what, instead", [
    (_gradient, "a 1-D quadrupole-gradient map G(z) ('QG')", _MATRIX),
    (_quad3, f"{_QUAD} ('Q3')", _MATRIX),
    (lambda: _corrector3(label="BEND1"),
     "a 3-D map whose dipole field on the axis bends the beam by 24.8 mrad (a steering "
     "corrector, up to 20 mrad, is left out) ('BEND1')", _MATRIX),
    (_electric, "a static electric field map ('E1')", _MATRIX),
    (_cart2d, "a 2-D Cartesian field map ('C2')", "no HELIX mode tracks one"),
], ids=["G(z)", "3-D quadrupole", "3-D dipole", "static electric", "2-D Cartesian"])
def test_a_powered_map_the_solver_has_no_model_of_is_refused_before_the_run(make, what,
                                                                            instead):
    """Refused before the run, by the solver and through run_envelope_sim alike, naming the
    element and pointing to the mode that tracks it (none for a 2-D Cartesian map)."""
    lat = _line(make())
    msg = re.escape(f"the Sacherer envelope solver has no model of {what} — {instead}")
    with pytest.raises(ValueError, match=msg):
        SachererSolver(lat, _ref(), dict(_INIT), current=0.0)          # no run asked
    with pytest.raises(ValueError, match=msg):
        common.run_envelope_sim(lat, _cfg(), env_solver="sacherer")


def test_an_unpowered_map_of_any_kind_is_no_field():
    """Nothing to model: a drift, as the matrix envelope has it (an unpowered G(z) map read
    as a 1 T solenoid)."""
    drift = _exit(_drift_line())
    zero = _grid3("Z3", *(lambda X, Y, Z: 0.0 * Z,) * 3)
    for m in (_gradient(kb=0.0), _quad3(voltage_rel=-1.0), _electric(ke=0.0), zero):
        np.testing.assert_allclose(_exit(_line(m)), drift, rtol=EXACT, err_msg=m.name)


def test_an_rf_map_is_a_drift_and_keeps_its_solenoid_channel():
    """The DC approximation leaves the RF channel out; a static solenoid channel on the same
    map focuses at its own amplitude."""
    np.testing.assert_allclose(_exit(_line(_rf())), _exit(_drift_line()), rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(_rf(stat_b=True))), _exit(_line(_map1d())),
                               rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(_rf(stat_b=True, voltage_rel=0.05))),
                               _exit(_line(_map1d(kb=KB * 1.05))), rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(_rf(stat_b=True, voltage_rel=-1.0))),
                               _exit(_drift_line()), rtol=EXACT)


# ---------------------------------------------------------------------------
# SUPERPOSE_MAP clusters
# ---------------------------------------------------------------------------
def _cluster(*children, name="CL", **kw):
    return SuperposedFieldMap(name, list(children), **kw)


@pytest.mark.parametrize("current", [0.0, 5.0], ids=["I=0", "5 mA DC"])
def test_a_solenoid_with_its_correctors_is_the_solenoid_alone(current):
    """The DC LEBT idiom: a solenoid map with a horizontal and a vertical corrector map
    superposed on it.  A corrector's field on the axis is a dipole's, and it is left out,
    unpowered or within its bounds; this one's is uniform across the axis, with no gradient: it
    steers the beam (by 12 mrad here) and adds no rms focusing, as the matrix envelope has it."""
    cl = _cluster((0.0, _map3d(name="SOL")),
                  (0.0, _corrector3(name="CORH", kb=0.0)),
                  (0.0, _corrector3(name="CORV", kb=0.5)))
    lat = _line(cl)
    got = _exit(lat, current=current)
    np.testing.assert_allclose(got, _exit(_line(_map3d()), current=current), rtol=EXACT)
    np.testing.assert_allclose(got, _exit(lat, "matrix", current), rtol=REL)


def test_a_clusters_voltage_rel_scales_its_solenoid():
    """The cluster's own amplitude factor (1 + voltage_rel, where an error study's draw on
    the cluster lands) multiplies every child."""
    def cl(vr):
        return _cluster((0.0, _map1d()), (0.0, _corrector3(name="CORH", kb=0.5)),
                        voltage_rel=vr)
    up = _exit(_line(cl(0.05)))
    np.testing.assert_allclose(up, _exit(_line(_map1d(kb=KB * 1.05))), rtol=EXACT)
    np.testing.assert_allclose(up, _exit(_line(cl(0.05)), "matrix"), rtol=REL)
    np.testing.assert_allclose(_exit(_line(cl(-1.0))), _exit(_drift_line()), rtol=EXACT)


def _two_bumps(second=True):
    """A 300 mm map holding two solenoid bumps, at 60 and 240 mm, whose card length (150 mm)
    keeps the first."""
    z = _z(300.0, 601)
    bz = np.exp(-((z - 60.0) / 25.0) ** 2)
    if second:
        bz = bz + np.exp(-((z - 240.0) / 25.0) ** 2)
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=bz)
    return FieldMap("TWO", length=150.0, field_data=fd, kb=KB, ke=0.0, aperture=30.0)


def test_each_child_counts_over_its_own_window():
    """A solenoid placed at z0 = 50 mm, overlapping the first bump of a map cut to 150 mm by
    its card: the sum the tracker takes, each child over its own window -- the second bump
    of the cut map, inside the cluster's 250 mm but beyond its window, counts for nothing."""
    cl = _cluster((50.0, _map1d(name="LATE", kb=0.2)), (0.0, _two_bumps()))
    assert cl.length == pytest.approx(250.0)
    lat = _line(cl)
    got = _exit(lat)
    np.testing.assert_allclose(got, _exit(lat, "matrix"), rtol=REL)
    first_only = _cluster((50.0, _map1d(name="LATE", kb=0.2)), (0.0, _two_bumps(second=False)))
    np.testing.assert_allclose(got, _exit(_line(first_only)), rtol=ODE)
    late_only = _cluster((50.0, _map1d(name="LATE", kb=0.2)))
    assert abs(got[0] / _exit(_line(late_only))[0] - 1.0) > 0.05      # the cut map counts
    # the same field split into other elements: other ODE steps
    np.testing.assert_allclose(_exit(_line(late_only)),
                               _exit(_line(Drift("D50", 50.0, aperture=30.0),
                                           _map1d(kb=0.2))), rtol=1e-5)


@pytest.mark.parametrize("child, what", [
    (lambda: _gradient(), "a 1-D quadrupole-gradient map G(z) ('QG' in the SUPERPOSE_MAP "
                          "cluster 'CL')"),
    (lambda: _quad3(), f"{_QUAD} ('Q3' in the SUPERPOSE_MAP cluster 'CL')"),
    (lambda: _electric(), "a static electric field map ('E1' in the SUPERPOSE_MAP cluster "
                          "'CL')"),
], ids=["G(z)", "3-D quadrupole", "static electric"])
def test_a_child_the_solver_has_no_model_of_is_refused(child, what):
    lat = _line(_cluster((0.0, _map1d()), (0.0, child())))
    with pytest.raises(ValueError, match=re.escape(f"no model of {what} — use "
                                                   "env_solver='matrix'")):
        SachererSolver(lat, _ref(), dict(_INIT), current=0.0)


# ---------------------------------------------------------------------------
# steering correctors, and the 3-D maps refused with them
# ---------------------------------------------------------------------------
#: the deflection of the test beam (30 keV H-) per unit kb of ``_corrector3``'s 0.01 T dipole
#: field along the bump (rad): its integral over Brho
_KICK = 0.01 * 35.0 * math.sqrt(math.pi) * math.erf(100.0 / 35.0) * 1e-3 / _ref().brho


def _refused(lat, what, ref=None):
    """The refusal of ``what`` before the run (for the test beam unless ``ref`` is given),
    the element named in it."""
    with pytest.raises(ValueError, match=re.escape(f"no model of {what} — {_MATRIX}")):
        SachererSolver(lat, _ref() if ref is None else ref, dict(_INIT), current=0.0)


def _alone_and_superposed(make, name):
    """The map standing alone, and superposed on the 3-D solenoid: (line, refusal's name)."""
    return [(_line(make()), f"'{name}'"),
            (_line(_cluster((0.0, _map3d()), (0.0, make()))),
             f"'{name}' in the SUPERPOSE_MAP cluster 'CL'")]


def _quad_grid(n, centre=(0.0, 0.0), rot=0.0, g=0.05):
    """An ideal quadrupole of gradient ``g`` (T/mm) on an n-point grid over +-10 mm in both
    planes, its magnetic centre at ``centre`` (mm), rotated by ``rot`` (rad) about it."""
    c, s = np.cos(2.0 * rot), np.sin(2.0 * rot)
    dx, dy = centre

    def fx(X, Y, Z):
        return g * (c * (Y - dy) + s * (X - dx)) * _bump(Z)

    def fy(X, Y, Z):
        return g * (c * (X - dx) - s * (Y - dy)) * _bump(Z)

    return _grid3("Q3", fx, fy, lambda X, Y, Z: 0.0 * Z, x=np.linspace(-10.0, 10.0, n))


_QUADS = ([dict(n=n, g=5e-5) for n in range(4, 42)]
          + [dict(n=n, centre=c, rot=r) for n in (10, 41)
             for c in ((1.2, 0.0), (0.0, 5.0), (10.0, 10.0))
             for r in (0.0, np.pi / 8.0, np.pi / 4.0)])


@pytest.mark.parametrize("kw", _QUADS, ids=[
    "n{}-centre{:g},{:g}-rot{:g}".format(kw["n"], *kw.get("centre", (0.0, 0.0)),
                                         np.degrees(kw.get("rot", 0.0))) for kw in _QUADS])
def test_a_quadrupole_whose_magnetic_centre_lies_on_its_grid_is_refused(kw):
    """An ideal quadrupole, standing alone or superposed on a solenoid, centred on a grid of
    4 to 41 points (a weak one: 0.05 T/m), or with its magnetic centre off the grid's 0 up to
    the grid's half-width, at any rotation.  On an even grid the point nearest the axis reads
    the field off the axis, and an off-centre quadrupole has a dipole field on the axis: the
    gradient there, more than a quarter of the peak field at the grid's half-width, tells
    either from a corrector."""
    for lat, where in _alone_and_superposed(lambda: _quad_grid(**kw), "Q3"):
        _refused(lat, f"{_QUAD} ({where})")


@pytest.mark.parametrize("centre, x, y", [
    ((0.0, 0.0), np.linspace(-10.0, 10.0, 5), None),
    ((0.0, 0.0), np.linspace(-10.0, 10.0, 4), None),
    ((0.0, 0.0), np.linspace(-10.0, 10.0, 5), np.linspace(-2.0, 2.0, 5)),
    ((0.0, 0.0), np.linspace(-2.0, 2.0, 5), np.linspace(-10.0, 10.0, 5)),
    ((1.2, 0.0), np.linspace(-10.0, 10.0, 10), None),
], ids=["5 points", "4 points", "narrower in y", "narrower in x", "centre off the axis"])
def test_a_quadrupole_on_its_grid_is_refused_however_weak(centre, x, y):
    """At 0.002 T/m a quadrupole focuses the beam by 1/f = 0.005 /m, less than a corrector may;
    the share of its gradient in its peak field, a property of its shape, refuses it.  On a grid
    over +-10 mm in one plane and +-2 mm in the other the peak field is the one 10 mm off the
    axis, and the gradient counts at that half-width."""
    g, (dx, dy) = 2e-6, centre

    def quad():
        return _grid3("Q3", lambda X, Y, Z: g * (Y - dy) * _bump(Z),
                      lambda X, Y, Z: g * (X - dx) * _bump(Z), lambda X, Y, Z: 0.0 * Z, x=x, y=y)
    for lat, where in _alone_and_superposed(quad, "Q3"):
        _refused(lat, f"{_QUAD} ({where})")


@pytest.mark.parametrize("n", [4, 5])
def test_a_field_that_vanishes_on_the_axis_with_no_gradient_is_refused_as_a_multipole(n):
    """A sextupole: no field and no gradient on the axis.  On the 4-point grid the point
    nearest the axis reads a field off the axis, which the axis itself does not have."""
    def sextupole():
        return _grid3("S3", lambda X, Y, Z: 2e-3 * X * Y * _bump(Z),
                      lambda X, Y, Z: 1e-3 * (X ** 2 - Y ** 2) * _bump(Z),
                      lambda X, Y, Z: 0.0 * Z, x=np.linspace(-10.0, 10.0, n))
    for lat, where in _alone_and_superposed(sextupole, "S3"):
        _refused(lat, "a 3-D map whose field vanishes on the axis, with no gradient there (a "
                      f"multipole) ({where})")


def _sextupole_with_bz(bz, n):
    """The sextupole of 0.2 T peak at the grid's corners, with a solenoid field ``bz`` x the bump
    on the axis, on n points over +-10 mm (on 4, the point nearest the axis reads the
    sextupole's 0.022 T there)."""
    return _grid3("S3", lambda X, Y, Z: 2e-3 * X * Y * _bump(Z),
                  lambda X, Y, Z: 1e-3 * (X ** 2 - Y ** 2) * _bump(Z),
                  lambda X, Y, Z: bz * _bump(Z) + 0.0 * X, x=np.linspace(-10.0, 10.0, n))


@pytest.mark.parametrize("bz, n", [(0.01, 4), (0.01, 5), (0.018, 4)],
                         ids=["a twentieth, 4 points", "a twentieth, 5 points", "0.09, 4 points"])
def test_a_weak_solenoid_field_on_a_multipoles_axis_leaves_it_a_multipole(bz, n):
    """The sextupole with a solenoid field on the axis a twentieth of its peak (an end field's
    size), or 0.09 of it: under a tenth, the field on the axis is no solenoid's, and the map is
    refused as a multipole."""
    for lat, where in _alone_and_superposed(lambda: _sextupole_with_bz(bz, n), "S3"):
        _refused(lat, "a 3-D map whose field vanishes on the axis, with no gradient there (a "
                      f"multipole) ({where})")


def test_a_solenoid_field_above_a_tenth_of_a_multipoles_peak_is_a_solenoids():
    """The sextupole on 4 points with a solenoid field on the axis 0.11 of its peak: the point
    nearest the axis reads the sextupole's field there above it, so the map is no solenoid by
    the nearest point's rule, and it is refused as a solenoid's field on the axis that the point
    nearest it does not show."""
    for lat, where in _alone_and_superposed(lambda: _sextupole_with_bz(0.022, 4), "S3"):
        _refused(lat, "a 3-D map whose field on the axis is a solenoid's, which its grid point "
                      f"nearest the axis does not show ({where})")


@pytest.mark.parametrize("current", [0.0, 5.0], ids=["I=0", "5 mA DC"])
def test_a_steering_corrector_standing_alone_is_a_drift(current):
    """A powered corrector map with no gradient (12 mrad) standing alone adds no rms focusing:
    the drift, as the matrix envelope has it and as the same map in a cluster is."""
    lat = _line(_corrector3(kb=0.5))
    drift = _exit(_drift_line(), current=current)
    np.testing.assert_allclose(_exit(lat, current=current), drift, rtol=EXACT)
    np.testing.assert_allclose(_exit(lat, "matrix", current), drift, rtol=REL)


def _dipole3(plane, sign, kb):
    """A corrector of ``_corrector3``'s field on the axis: B_y (``plane`` 'y') or B_x."""
    def across(X, Y, Z):
        return sign * 0.01 * _bump(Z) + 0.0 * X

    def zero(X, Y, Z):
        return 0.0 * Z

    return _grid3("COR", zero if plane == "y" else across, across if plane == "y" else zero,
                  zero, kb=kb)


@pytest.mark.parametrize("plane", ["y", "x"], ids=["B_y", "B_x"])
@pytest.mark.parametrize("field_sign, kb_sign", [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)],
                         ids=["+", "field -", "kb -"])
def test_a_dipole_map_is_a_steering_corrector_up_to_20_mrad(plane, field_sign, kb_sign):
    """Left out up to 20 mrad (θ = |amplitude| x the largest |∫(B_x, B_y) dz| along the map, over
    Bρ), standing alone or in a cluster; refused beyond, the message giving the angle."""
    def dipole(theta):
        return _dipole3(plane, field_sign, kb_sign * theta / _KICK)
    np.testing.assert_allclose(_exit(_line(dipole(0.0195))), _exit(_drift_line()), rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(_cluster((0.0, _map3d()), (0.0, dipole(0.0195))))),
                               _exit(_line(_map3d())), rtol=EXACT)
    for lat, where in _alone_and_superposed(lambda: dipole(0.0205), "COR"):
        _refused(lat, "a 3-D map whose dipole field on the axis bends the beam by 20.5 mrad (a "
                      f"steering corrector, up to 20 mrad, is left out) ({where})")


def _sextupole_part(s, x, kb):
    """A corrector whose field grows by ``s`` out to 10 mm (a sextupole part, no gradient on the
    axis), on the grid ``x`` in both planes."""
    return _grid3("COR", lambda X, Y, Z: 0.01 * 2.0 * s * X * Y / 100.0 * _bump(Z),
                  lambda X, Y, Z: 0.01 * (1.0 + s * (X ** 2 - Y ** 2) / 100.0) * _bump(Z),
                  lambda X, Y, Z: 0.0 * Z, kb=kb, x=x)


def _axis_point_at(at):
    """21 points over +-10 mm, the middle one at ``at`` (mm): 0, or off it by rounding."""
    x = np.linspace(-10.0, 10.0, 21)
    x[10] = at
    return x


@pytest.mark.parametrize("s, x, kb", [(0.2, np.linspace(-7.0, 13.0, 4), 0.04),
                                      (0.1, _axis_point_at(0.0), 0.016 / _KICK),
                                      (0.1, _axis_point_at(1e-15), 0.016 / _KICK),
                                      (0.1, _axis_point_at(-1e-15), 0.016 / _KICK)],
                         ids=["off-centre grid", "a point on the axis",
                              "a point 1e-15 mm off the axis", "a point -1e-15 mm off the axis"])
def test_a_corrector_with_a_sextupole_part_is_left_out(s, x, kb):
    """On 4 points from -7 to 13 mm the slope across the points around the axis reads the
    sextupole part as a gradient making 6 % of the peak at the half-width, under a
    quadrupole's 25 %, and the map's sampler reads it so: at kb = 0.04 it focuses the beam by
    1/f = 0.0084 /m, less than a corrector may.  On 21 points over +-10 mm, one at the axis, the
    slope is taken across both its neighbours: no gradient (at 16 mrad the slope on either side
    alone reads the sextupole part as a gradient focusing the beam by 1/f = 0.0113 /m).  A
    point 1e-15 mm off the axis, where a grid built from a file's bounds in metres can place it,
    is at the axis.  The matrix envelope through each map moves the rms sizes by less than
    0.5 %."""
    lat = _line(_sextupole_part(s, x, kb))
    np.testing.assert_allclose(_exit(lat), _exit(_drift_line()), rtol=EXACT)
    np.testing.assert_allclose(_exit(lat, "matrix"), _exit(_drift_line(), "matrix"), rtol=5e-3)


#: the integral of ``_bump`` over the map (mm), and the focusing (1/m) of a gradient of 1 T/mm
#: along it, per unit kb, on the test beam
_LENGTH = 35.0 * math.sqrt(math.pi) * math.erf(100.0 / 35.0)
_FOCUS = _LENGTH / _ref().brho


def _gradient_part(q, inv_f):
    """A dipole field with a gradient part, 0.01 T x (1 + q x / 10 mm) across the axis (with
    B_x = 0.01 T x q y / 10 mm), at the kb whose gradient focuses the beam by ``inv_f`` (1/m);
    it bends the beam by 0.01 m x inv_f / q."""
    return _grid3("CF", lambda X, Y, Z: 0.01 * q * Y / 10.0 * _bump(Z),
                  lambda X, Y, Z: 0.01 * (1.0 + q * X / 10.0) * _bump(Z),
                  lambda X, Y, Z: 0.0 * Z, kb=inv_f / (0.01 * q / 10.0 * _FOCUS))


def _quad_off_grid(inv_f, centre=35.0):
    """An ideal quadrupole whose magnetic centre lies at x = ``centre`` (mm; 35 mm is 3.5
    half-widths off the grid), at the gradient that focuses the beam by ``inv_f`` (1/m)."""
    g = inv_f / _FOCUS
    return _grid3("QOFF", lambda X, Y, Z: g * Y * _bump(Z),
                  lambda X, Y, Z: g * (X - centre) * _bump(Z), lambda X, Y, Z: 0.0 * Z)


def _focuses(inv_f):
    return (f"a 3-D map whose quadrupole gradient on the axis, as the beam sees it, focuses the "
            f"beam by up to 1/f = {inv_f} /m (a steering corrector, up to 0.0088 /m, is left out)")


#: the share of a gradient along the bump that the beam sees in the solenoid of ``_map3d``: B_z
#: and the gradient share the bump's shape, so |∫ bump exp(-2i θ_L) dz| / ∫ bump dz is
#: sin(Θ) / Θ, Θ the angle the solenoid turns the beam by over the map (21.3 deg)
_TURN = KB * _LENGTH * 1e-3 / (2.0 * _ref().brho)
_SEEN = math.sin(_TURN) / _TURN


def _alone_and_seen(make, name, inv_f):
    """The map standing alone, focusing the beam by ``inv_f`` (1/m), and superposed on the 3-D
    solenoid, where the beam sees ``_SEEN`` of it: (line, refusal)."""
    return [(_line(make()), f"{_focuses(f'{inv_f:.3g}')} ('{name}')"),
            (_line(_cluster((0.0, _map3d()), (0.0, make()))),
             f"{_focuses(f'{inv_f * _SEEN:.3g}')} ('{name}' in the SUPERPOSE_MAP cluster 'CL')")]


@pytest.mark.parametrize("make, name, inv_f", [
    (lambda: _gradient_part(0.1, 0.1), "CF", 0.1),
    (lambda: _gradient_part(0.1, -0.1), "CF", 0.1),
    (lambda: _quad_off_grid(5e-5 * _FOCUS), "QOFF", 5e-5 * _FOCUS),
    (lambda: _sextupole_part(0.2, np.linspace(-7.0, 13.0, 4), 0.5), "COR",
     0.5 * math.hypot(6e-5, 6e-5) * _FOCUS),
], ids=["dipole with a gradient part", "the same, kb < 0", "quadrupole centred off its grid",
        "sextupole part on an off-centre grid"])
def test_a_map_whose_gradient_focuses_the_beam_more_than_a_corrector_is_refused(make, name,
                                                                                inv_f):
    """The integrated normal and skew quadrupole gradient on the axis, times the amplitude's
    size, over Bρ: a dipole field with a gradient part (a tenth of the field at 10 mm, 10 mrad;
    at kb > 0 and kb < 0), an ideal quadrupole of 0.05 T/m whose magnetic centre lies 3.5
    half-widths off its grid (its gradient makes 22 % of its peak field at the half-width;
    4.3 mrad), the sextupole part of ``test_a_corrector_with_a_sextupole_part_is_left_out`` on
    its off-centre grid at kb = 0.5 (the slope across the points around the axis reads 6e-5 T/mm
    in both the normal and the skew gradient).  Each bends the beam by less than 20 mrad, and the
    matrix envelope through it moves the rms sizes by more than 2 %.  Superposed on the
    solenoid, which turns the beam by 21.3 deg along the bump, the beam sees ``_SEEN`` of it."""
    for lat, what in _alone_and_seen(make, name, inv_f):
        _refused(lat, what)
    moved = _exit(_line(make()), "matrix") / _exit(_drift_line(), "matrix") - 1.0
    assert np.max(np.abs(moved)) > 0.02


@pytest.mark.parametrize("make", [
    lambda inv_f: _gradient_part(0.01, inv_f), lambda inv_f: _gradient_part(0.2, inv_f),
    _quad_off_grid,
], ids=["gradient part 1 %", "gradient part 20 %", "quadrupole centred off its grid"])
def test_a_map_with_a_gradient_is_a_steering_corrector_up_to_1_f_of_0_0088_per_m(make):
    """Left out while its gradient focuses the beam by at most 1/f = 0.0088 /m, standing alone
    or in a cluster, and refused beyond: a dipole field with a gradient part of 1 % or 20 % of
    the field at 10 mm (bending the beam by 8.4 or 0.42 mrad at 0.0084 /m; the gradient makes 1 %
    or 17 % of the peak field at the half-width) and the quadrupole centred off its grid (22 %).
    Superposed on the solenoid the beam sees ``_SEEN`` of a gradient along the bump: at
    0.0092 /m, 0.00899."""
    np.testing.assert_allclose(_exit(_line(make(0.0084))), _exit(_drift_line()), rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(_cluster((0.0, _map3d()), (0.0, make(0.0084))))),
                               _exit(_line(_map3d())), rtol=EXACT)
    for lat, what in _alone_and_seen(lambda: make(0.0092), make(0.0092).name, 0.0092):
        _refused(lat, what)


# ---------------------------------------------------------------------------
# a corrector's gradient as the beam sees it
# ---------------------------------------------------------------------------
def _turning_solenoid(turn, kb=1.0, factor=1.0, ref=None):
    """A 400 mm 1-D solenoid map whose B_z, sin² over 120 to 280 mm and zero elsewhere, turns
    the beam (of ``ref``, the test beam unless given) by ``turn`` (rad) at the amplitude
    kb x ``factor``: its file holds that field over kb x ``factor``."""
    ref = _ref() if ref is None else ref
    b0 = turn * 2.0 * ref.brho / 0.080            # T: the integral of sin² over 160 mm is 80 mm
    z = _z(400.0, 801)
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=1, z=z, Fz=b0 / (kb * factor) * _lobe(z, 120.0, 280.0))
    return FieldMap("SOL", length=400.0, field_data=fd, kb=kb, ke=0.0, aperture=30.0)


def _lobed(inv_f, phi=0.0, ref=None, name="COR", start=0.0, theta=0.012):
    """A 400 mm corrector map: a dipole field across the axis, 0.01 T x sin² over 40 to 360 mm
    (at the kb that bends the beam by ``theta``, rad), and a quadrupole gradient on the axis in
    two lobes outside the field of ``_turning_solenoid``, sin² over 40 to 120 mm (normal) and
    over 280 to 360 mm (-exp(i phi) times the first), each focusing the beam (of ``ref``) by
    ``inv_f`` (1/m).  The beam sees the second lobe turned by twice the angle the solenoid
    turns it by.  Its grid runs from ``start``, the field from ``start`` on as from 0."""
    ref = _ref() if ref is None else ref
    kb = theta / (0.01 * 160.0 * 1e-3 / ref.brho)
    g = inv_f * ref.brho / (kb * 40.0)            # T/mm: the integral of a lobe is 40 mm
    z = _z(400.0, 801) + start
    x = np.linspace(-10.0, 10.0, 5)
    X, Y, Z = np.meshgrid(x, x, z - start, indexing="ij")
    grad = g * (_lobe(Z, 40.0, 120.0) - np.exp(1j * phi) * _lobe(Z, 280.0, 360.0))
    b = 0.01 * _lobe(Z, 40.0, 360.0) + grad * (X + 1j * Y)          # B_y + i B_x
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=b.imag,
                                               Fy=b.real, Fz=0.0 * Z)
    return FieldMap3D(name, length=400.0, field_data=fd, kb=kb, ke=0.0, aperture=30.0)


def _quadrupolar(lat, base, species="H-"):
    """|dQ| / (Sxx + Syy) of the matrix envelope's exit beam against ``base``'s, Q = Sxx - Syy
    + 2i Sxy: how much a gradient has made the round beam elliptical, in any frame."""
    def moments(line):
        res = common.run_envelope_sim(copy.deepcopy(line), _cfg(0.0, species),
                                      env_solver="matrix")
        m = np.asarray(res.sigma_matrix[-1])
        return m[0, 0], m[2, 2], m[0, 2]
    (xx, yy, xy), (bxx, byy, bxy) = moments(lat), moments(base)
    return abs((xx - yy - bxx + byy) + 2j * (xy - bxy)) / (bxx + byy)


def test_a_correctors_gradient_counts_as_the_beam_sees_it():
    """The two opposite lobes of ``_lobed(0.006)``, as a corrector's two end fields have, cancel
    in the laboratory frame (net 0; the largest value along the map 0.006 /m), and standing
    alone the map is left out.  Superposed on a solenoid that turns the beam by 90 deg between
    the lobes, the beam sees the second turned by 180 deg: they add, 0.012 /m, and the map is
    refused (also with the corrector's grid running from 100 to 500 mm: its field read where it
    lies in the cluster); turned by 360 deg they cancel again, and it is left out.  The matrix
    envelope moves the rms sizes by less than 0.05 % with the map standing alone, and makes the
    round beam elliptical through the 90 deg cluster 12 times more than through the 180 deg
    one."""
    def cl(turn, lobes=True, start=0.0):
        kids = [(0.0, _turning_solenoid(turn))]
        if lobes:
            kids.append((0.0, _lobed(0.006, start=start)))
        return _line(_cluster(*kids))
    np.testing.assert_allclose(_exit(_line(_lobed(0.006))), _exit(_drift_line(400.0)),
                               rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(_lobed(0.006)), "matrix"),
                               _exit(_drift_line(400.0), "matrix"), rtol=5e-4)
    for start in (0.0, 100.0):
        _refused(cl(0.5 * math.pi, start=start),
                 f"{_focuses('0.012')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")
    np.testing.assert_allclose(_exit(cl(math.pi)), _exit(cl(math.pi, lobes=False)), rtol=EXACT)
    assert (_quadrupolar(cl(0.5 * math.pi), cl(0.5 * math.pi, lobes=False))
            > 8.0 * _quadrupolar(cl(math.pi), cl(math.pi, lobes=False)))


def _flat_solenoid(nodes, card):
    """A 1-D solenoid map whose file holds a flat B_z on nodes from 0 to ``nodes`` mm, at the kb
    whose first 80 mm turn the test beam by 90 deg, behind a card ``card`` mm long."""
    z = np.linspace(0.0, nodes, int(2 * nodes) + 1)
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=1.0 + 0.0 * z)
    return FieldMap("SOL", length=card, field_data=fd, kb=math.pi * _ref().brho / 0.080, ke=0.0,
                    aperture=30.0)


@pytest.mark.parametrize("nodes, card", [(80.0, 280.0), (280.0, 80.0)],
                         ids=["its nodes end", "its card ends"])
def test_the_beam_turns_in_each_childs_field_over_that_childs_window(nodes, card):
    """A solenoid child placed at z0 = 120 mm, its flat field reaching to 200 mm in the cluster,
    where its nodes or its card end (the other running on to the cluster's exit): it turns the
    beam by 90 deg between the lobes of ``_lobed(0.006)``, which add (refused, 1/f =
    0.012 /m).  Read past its nodes or its card, or from its own start rather than its place
    in the cluster, its field would turn the beam on through the second lobe, or within the
    first, and the lobes would not add."""
    cl = _cluster((120.0, _flat_solenoid(nodes, card)), (0.0, _lobed(0.006)))
    _refused(_line(cl), f"{_focuses('0.012')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")


def test_a_corrector_standing_alone_counts_the_largest_value_its_gradient_reaches():
    """Standing alone, the lobes of ``_lobed`` count each, net 0 though they are: at 0.0084 /m
    each the map is left out, at 0.0092 refused.  A corrector whose gradient lobes follow its
    end fields, the slope of its 16 mrad bump (each lobe 0.06 /m), is refused at 0.06."""
    np.testing.assert_allclose(_exit(_line(_lobed(0.0084))), _exit(_drift_line(400.0)),
                               rtol=EXACT)
    _refused(_line(_lobed(0.0092)), f"{_focuses('0.0092')} ('COR')")
    kb = 0.016 / _KICK
    g = 0.06 * _ref().brho / (kb * 35.0)       # the largest |∫ 35 _dbump dz| is 35 mm
    ends = _grid3("CE", lambda X, Y, Z: g * 35.0 * _dbump(Z) * Y,
                  lambda X, Y, Z: 0.01 * _bump(Z) + g * 35.0 * _dbump(Z) * X,
                  lambda X, Y, Z: 0.0 * Z, kb=kb)
    _refused(_line(ends), f"{_focuses('0.06')} ('CE')")


@pytest.mark.parametrize("share, refused", [(0.999, False), (1.001, True)],
                         ids=["1/f = 0.999 of the bound", "1/f = 1.001 of the bound"])
def test_a_corrector_is_left_out_up_to_0_0088_per_m_as_the_beam_sees_it(share, refused):
    """The lobes of ``_lobed`` turned together by the 90 deg solenoid, each at half ``share``
    times 0.0088 /m: just under the bound the map is left out (the solenoid alone), just over it
    refused."""
    cl = _cluster((0.0, _turning_solenoid(0.5 * math.pi)), (0.0, _lobed(0.5 * share * 0.0088)))
    if refused:
        _refused(_line(cl), f"{_focuses('0.00881')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")
    else:
        np.testing.assert_allclose(
            _exit(_line(cl)), _exit(_line(_cluster((0.0, _turning_solenoid(0.5 * math.pi))))),
            rtol=EXACT)


@pytest.mark.parametrize("species", [H_MINUS, PROTON], ids=["H-", "proton"])
def test_the_beam_turns_its_own_way_in_the_solenoids_field(species):
    """A normal lobe, then a skew one (comparable parts: the second lobe is -exp(i phi) times
    the first, phi = +-90 deg), each 0.006 /m, superposed on a solenoid that turns the beam by
    45 deg between them, one way for H- and the other way for a proton (θ_L = sign(q) ∫ B_z dz
    / (2 Bρ)).  The beam sees the second lobe turned by twice that: for H- the lobes add at phi =
    +90 deg (refused, 0.012 /m) and cancel at -90 (left out), for a proton the other way round;
    the matrix envelope makes the round beam elliptical 6 times more where they add."""
    ref = _ref(species)
    name = "H-" if species is H_MINUS else "proton"
    add, cancel = (0.5 * math.pi, -0.5 * math.pi)[::1 if species is H_MINUS else -1]

    def cl(phi=None):
        kids = [(0.0, _turning_solenoid(0.25 * math.pi, ref=ref))]
        if phi is not None:
            kids.append((0.0, _lobed(0.006, phi, ref)))
        return _line(_cluster(*kids))
    _refused(cl(add), f"{_focuses('0.012')} ('COR' in the SUPERPOSE_MAP cluster 'CL')", ref)
    np.testing.assert_allclose(_exit(cl(cancel), species=name), _exit(cl(), species=name),
                               rtol=EXACT)
    assert (_quadrupolar(cl(add), cl(), name)
            > 4.0 * _quadrupolar(cl(cancel), cl(), name))


def test_the_solenoid_turns_the_beam_at_its_amplitude_and_the_clusters():
    """The beam turns in the cluster's B_z at the amplitude every path uses: a solenoid child at
    kb = 2 whose file holds the field of ``_turning_solenoid(90 deg)``, in a cluster at
    voltage_rel = -0.5, turns it by 90 deg, and the lobes of ``_lobed(0.012)``, 0.006 /m each at
    the cluster's amplitude, add: the map is refused (1/f = 0.012 /m).  Read at kb = 1 the
    solenoid would turn the beam by 45 deg, at the cluster's full amplitude by 180 deg."""
    cl = _cluster((0.0, _turning_solenoid(0.5 * math.pi, kb=2.0, factor=0.5)),
                  (0.0, _lobed(0.012)), voltage_rel=-0.5)
    _refused(_line(cl), f"{_focuses('0.012')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")


def test_the_beam_turns_where_each_child_lies_in_the_cluster():
    """The solenoid and the corrector of ``test_a_correctors_gradient_counts_as_the_beam_sees_it``
    (each lobe 0.0047 /m) both placed at z0 = 100 mm: the solenoid turns the beam by 90 deg
    between the corrector's lobes where they lie in the cluster, and they add, 0.0094 /m
    (refused).  Turned as at the corrector's own z, 100 mm before where it lies, the beam would
    see its second lobe inside the solenoid's field, and the lobes would not add (0.0081 /m)."""
    cl = _cluster((100.0, _turning_solenoid(0.5 * math.pi)), (100.0, _lobed(0.0047)))
    _refused(_line(cl), f"{_focuses('0.0094')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")


def test_each_solenoid_child_turns_the_beam():
    """Two solenoid children, each turning the beam by 45 deg between the lobes of
    ``_lobed(0.0055)``: 90 deg in all, and the lobes add (refused, 1/f = 0.011 /m).  Turned by
    one child alone, the beam would see them add to 0.0078 /m."""
    cl = _cluster((0.0, _turning_solenoid(0.25 * math.pi)),
                  (0.0, _turning_solenoid(0.25 * math.pi)), (0.0, _lobed(0.0055)))
    _refused(_line(cl), f"{_focuses('0.011')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")


def test_a_corrector_on_a_coarse_grid_along_z_is_turned_within_its_cells():
    """A uniform normal gradient on three z nodes, 0, 100 and 200 mm (with a dipole field of
    1 mT, 8 mrad), superposed on a flat solenoid that turns the beam by 90 deg over the map, or
    on two that turn it by 9 and 81 deg: the beam sees the gradient turn by 180 deg, and it
    focuses the beam by 2/pi of its laboratory-frame integral, 0.0097 /m (refused).  Turned only
    at the nodes, by 0, 90 and 180 deg, it would focus the beam by half that integral,
    0.0076 /m."""
    brho = _ref().brho
    g = 0.0097 * 0.5 * math.pi * brho / 200.0       # T/mm
    z = np.array([0.0, 100.0, 200.0])
    x = np.linspace(-10.0, 10.0, 5)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")

    def cor():
        fd = FieldMapData(z=z)
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=g * Y + 0.0 * Z,
                                                   Fy=0.001 + g * X + 0.0 * Z, Fz=0.0 * Z)
        return FieldMap3D("COR", length=200.0, field_data=fd, kb=1.0, ke=0.0, aperture=30.0)

    def sol(turn):
        zs = _z()
        fd = FieldMapData(z=zs)
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=zs,
                                                   Fz=turn * 2.0 * brho / 0.2 + 0.0 * zs)
        return FieldMap("SOL", length=200.0, field_data=fd, kb=1.0, ke=0.0, aperture=30.0)
    for turns in ((0.5 * math.pi,), (0.05 * math.pi, 0.45 * math.pi)):
        _refused(_line(_cluster(*[(0.0, sol(t)) for t in turns], (0.0, cor()))),
                 f"{_focuses('0.0097')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")


@pytest.mark.parametrize("what", ["deflection", "focusing"])
def test_a_field_that_changes_sign_within_a_cell_is_read_where_it_does(what):
    """A corrector on three z nodes, 0, 100 and 200 mm, whose dipole field (or gradient, with a
    uniform dipole field 150 times it x 1 mm) on the axis is +a, -3a and +a, linear between
    them: the beam reaches its largest angle, or meets its largest focusing, a x 212.5 mm over
    Brho, at 175 mm, where the field changes sign three quarters into the second cell.  It is
    read there: left out at 0.99 of the bound, refused at 1.01.  (The nodes alone read 0.94 of
    it, and 0 of a field +a, -a and +a.)"""
    brho = _ref().brho
    z = np.array([0.0, 100.0, 200.0])
    sign = np.array([1.0, -3.0, 1.0])
    x = np.linspace(-10.0, 10.0, 5)
    X, Y, _Z = np.meshgrid(x, x, z, indexing="ij")

    def cor(share):
        if what == "deflection":
            a = share * 0.020 * brho / 212.5e-3             # T
            fx, fy = 0.0 * X, a * sign + 0.0 * X
        else:
            a = share * 0.0088 * brho / 212.5               # T/mm
            fx, fy = a * sign * Y, 150.0 * a + a * sign * X
        fd = FieldMapData(z=z)
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=fx, Fy=fy,
                                                   Fz=0.0 * X)
        return FieldMap3D("COR", length=200.0, field_data=fd, kb=1.0, ke=0.0, aperture=30.0)
    np.testing.assert_allclose(_exit(_line(cor(0.99))), _exit(_drift_line()), rtol=EXACT)
    _refused(_line(cor(1.01)),
             "a 3-D map whose dipole field on the axis bends the beam by 20.2 mrad (a steering "
             "corrector, up to 20 mrad, is left out) ('COR')" if what == "deflection" else
             f"{_focuses('0.00889')} ('COR')")


def test_a_field_in_two_planes_is_read_where_the_beam_is_bent_most():
    """A corrector on three z nodes, 0, 100 and 200 mm, whose dipole field on the axis is
    (B_x, B_y) = (-3a, -a), (0, 2a) and (2a, 0), linear between them: the beam reaches its
    largest angle, a x 125 sqrt(2) mm over Brho, at 150 mm, where neither part of the field
    changes sign; the nodes give a x 158 mm.  It is read there: left out at 0.99 of the bound,
    refused at 1.01."""
    z = np.array([0.0, 100.0, 200.0])
    x = np.linspace(-10.0, 10.0, 5)
    X, _Y, _Z = np.meshgrid(x, x, z, indexing="ij")

    def cor(share):
        a = share * 0.020 * _ref().brho / (125.0 * math.sqrt(2.0) * 1e-3)      # T
        fd = FieldMapData(z=z)
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z,
                                                   Fx=a * np.array([-3.0, 0.0, 2.0]) + 0.0 * X,
                                                   Fy=a * np.array([-1.0, 2.0, 0.0]) + 0.0 * X,
                                                   Fz=0.0 * X)
        return FieldMap3D("COR", length=200.0, field_data=fd, kb=1.0, ke=0.0, aperture=30.0)
    np.testing.assert_allclose(_exit(_line(cor(0.99))), _exit(_drift_line()), rtol=EXACT)
    _refused(_line(cor(1.01)), "a 3-D map whose dipole field on the axis bends the beam by 20.2 "
                               "mrad (a steering corrector, up to 20 mrad, is left out) ('COR')")


@pytest.mark.parametrize("where, b, largest", [("starts", 1.5, 25.0), ("ends", 3.0, 37.5)],
                         ids=["where B starts", "where A ends"])
def test_a_field_that_changes_sign_next_to_another_childs_end_is_read_where_it_does(where, b,
                                                                                    largest):
    """Two corrector children: A on nodes 0 and 100 mm, its dipole field +a and -a, and B, placed
    at z0 = 100 mm on nodes 0 and 50 mm, +b and -b.  The beam reaches its largest angle where a
    field changes sign in a cell next to the other child's end: with b = 1.5a, A's (a x 25 mm
    over Brho, at 50 mm, in the cell that ends where B's field starts); with b = 3a, B's (a x
    37.5 mm, at 125 mm, in the cell that starts where A's field ends).  The sum there is that
    child's field alone, and it is read where it changes sign: left out at 0.99 of the bound,
    refused at 1.01."""
    brho = _ref().brho
    x = np.linspace(-10.0, 10.0, 5)

    def child(name, z, by):
        X, _Y, _Z = np.meshgrid(x, x, z, indexing="ij")
        fd = FieldMapData(z=z)
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.0 * X,
                                                   Fy=np.asarray(by) + 0.0 * X, Fz=0.0 * X)
        return FieldMap3D(name, length=float(z[-1]), field_data=fd, kb=1.0, ke=0.0,
                          aperture=30.0)

    def cl(share):
        a = share * 0.020 * brho / (largest * 1e-3)        # T
        return _cluster((0.0, child("A", np.array([0.0, 100.0]), [a, -a])),
                        (100.0, child("B", np.array([0.0, 50.0]), [b * a, -b * a])))
    np.testing.assert_allclose(_exit(_line(cl(0.99))), _exit(_drift_line(150.0)), rtol=EXACT)
    _refused(_line(cl(1.01)), "3-D maps whose dipole fields on the axis bend the beam by 20.2 mrad "
                              "together (steering correctors, up to 20 mrad together, are left "
                              "out) ('A' and 'B' in the SUPERPOSE_MAP cluster 'CL')")


def _pair(theta, inv_f, v_sign, name_h="CORH", name_v="CORV"):
    """An H corrector (B_y on the axis) and a V corrector (B_x) of ``_corrector3``'s bump, each
    bending the beam by ``theta`` (rad) and focusing it by ``inv_f`` (1/m) through a normal
    gradient along the bump, the V corrector's of the other sign, at kb x ``v_sign``."""
    kb = theta / _KICK
    g = inv_f / (kb * _FOCUS)

    def cor(name, dipole, grad, k):
        def b(X, Y, Z):                           # B_y + i B_x
            return (dipole * 0.01 + grad * g * (X + 1j * Y)) * _bump(Z)
        return _grid3(name, lambda X, Y, Z: b(X, Y, Z).imag, lambda X, Y, Z: b(X, Y, Z).real,
                      lambda X, Y, Z: 0.0 * Z, kb=k)
    return cor(name_h, 1.0, 1.0, kb), cor(name_v, 1j, -1.0, v_sign * kb)


def test_a_correctors_deflection_holds_past_the_end_of_its_window_in_the_sum():
    """The two-bump corrector map cut to 150 mm by its card (its first bump's 15 mrad) and a
    300 mm corrector map whose one bump, at 240 mm, bends the beam by 10 mrad the same way:
    together 25 mrad, the first child's angle carried past the end of its window."""
    z = _z(300.0, 601)
    x = np.linspace(-10.0, 10.0, 5)
    X, _Y, Z = np.meshgrid(x, x, z, indexing="ij")
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=7, x=x, y=x, z=z, Fx=0.0 * Z,
        Fy=0.01 * np.exp(-((Z - 240.0) / 25.0) ** 2) + 0.0 * X, Fz=0.0 * Z)
    one = 0.01 * 25.0 * math.sqrt(math.pi) * 1e-3 / _ref().brho          # rad per unit kb
    late = FieldMap3D("LATE", length=300.0, field_data=fd, kb=0.010 / one, ke=0.0,
                      aperture=30.0)
    _refused(_line(_cluster((0.0, _dipole_bumps(150.0)), (0.0, late))),
             "3-D maps whose dipole fields on the axis bend the beam by 25.0 mrad together "
             "(steering correctors, up to 20 mrad together, are left out) ('COR' and 'LATE' in "
             "the SUPERPOSE_MAP cluster 'CL')")


def test_a_clusters_correctors_bend_the_beam_together():
    """The corrector children of a cluster are bounded together, as the beam meets their summed
    field: an H and a V child at 14.5 mrad each bend the beam by 20.5 mrad and are refused, as
    one child with B_x = B_y is; two H children at 10.5 mrad each, 21.0 mrad.  Two H children
    at 25 mrad each and of opposite signs hold no field together and are left out.  Two
    separate maps in a row are bounded one at a time: the H and the V corrector at 19.9 mrad
    each are left out."""
    h, v = _pair(0.0205 / math.sqrt(2.0), 0.0, 1.0)
    _refused(_line(_cluster((0.0, _map3d()), (0.0, h), (0.0, v))),
             "3-D maps whose dipole fields on the axis bend the beam by 20.5 mrad together "
             "(steering correctors, up to 20 mrad together, are left out) ('CORH' and 'CORV' "
             "in the SUPERPOSE_MAP cluster 'CL')")
    a, _v = _pair(0.0105, 0.0, 1.0, name_h="CORA")
    b, _v = _pair(0.0105, 0.0, 1.0, name_h="CORB")
    _refused(_line(_cluster((0.0, _map3d()), (0.0, a), (0.0, b))),
             "3-D maps whose dipole fields on the axis bend the beam by 21.0 mrad together "
             "(steering correctors, up to 20 mrad together, are left out) ('CORA' and 'CORB' "
             "in the SUPERPOSE_MAP cluster 'CL')")
    np.testing.assert_allclose(
        _exit(_line(_cluster((0.0, _map3d()), (0.0, _dipole3("y", 1.0, 0.025 / _KICK)),
                             (0.0, _dipole3("y", 1.0, -0.025 / _KICK))))),
        _exit(_line(_map3d())), rtol=EXACT)
    h, v = _pair(0.0199, 0.0, 1.0)
    np.testing.assert_allclose(_exit(_line(h, v)),
                               _exit(_line(Drift("DH", L, aperture=30.0),
                                           Drift("DV", L, aperture=30.0))), rtol=EXACT)


@pytest.mark.parametrize("inv_f, v_sign, refused", [
    (0.006, -1.0, True), (0.004, -1.0, False), (0.006, 1.0, False),
], ids=["0.006 /m each, opposite signs", "0.004 /m each, opposite signs",
        "0.006 /m each, the same sign"])
def test_a_clusters_correctors_focus_the_beam_together(inv_f, v_sign, refused):
    """An H and a V corrector child of the solenoid, each bending the beam by 10 mrad and
    focusing it by ``inv_f`` alone: at kb of opposite signs their gradients add (2 inv_f, of
    which the beam sees ``_SEEN``), at the same sign they cancel."""
    h, v = _pair(0.010, inv_f, v_sign)
    lat = _line(_cluster((0.0, _map3d()), (0.0, h), (0.0, v)))
    if refused:
        _refused(lat, "3-D maps whose quadrupole gradients on the axis, as the beam sees them, "
                      f"focus the beam by up to 1/f = {2 * inv_f * _SEEN:.3g} /m together "
                      "(steering correctors, up to 0.0088 /m together, are left out) ('CORH' and "
                      "'CORV' in the SUPERPOSE_MAP cluster 'CL')")
    else:
        np.testing.assert_allclose(_exit(lat), _exit(_line(_map3d())), rtol=EXACT)


def test_a_clusters_correctors_focus_the_beam_where_each_lies():
    """Two corrector children with the lobes of ``_lobed(0.006)`` (each bending the beam by
    4 mrad), the second placed at z0 = 240 mm, where its first lobe lies on the first child's
    second: no solenoid turns the beam, those two cancel, and the largest value the beam meets is
    one lobe's, 0.006 /m: they are left out (the drift).  Summed as at their own z, their lobes
    would add, 0.012 /m."""
    cl = _cluster((0.0, _lobed(0.006, name="A", theta=0.004)),
                  (240.0, _lobed(0.006, name="B", theta=0.004)))
    np.testing.assert_allclose(_exit(_line(cl)), _exit(_drift_line(640.0)), rtol=EXACT)


def _dipole_lobes(name, *lobes, length=300.0):
    """A corrector map ``length`` mm long whose dipole field across the axis lies in sin² lobes
    ``(a, b, theta)``, over a to b mm, each bending the beam by theta (rad) at kb = 1."""
    z = _z(length, int(2 * length) + 1)
    x = np.linspace(-10.0, 10.0, 5)
    X, _Y, Z = np.meshgrid(x, x, z, indexing="ij")
    by = sum(theta * _ref().brho / (0.5e-3 * (b - a)) * _lobe(Z, a, b) for a, b, theta in lobes)
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.0 * Z,
                                               Fy=by + 0.0 * X, Fz=0.0 * Z)
    return FieldMap3D(name, length=length, field_data=fd, kb=1.0, ke=0.0, aperture=30.0)


def test_a_clusters_correctors_bend_the_beam_where_each_lies():
    """Two corrector children: one bends the beam by 15 mrad over 0 to 100 mm and back over 200
    to 300 mm, the other, placed at z0 = 200 mm, by 15 mrad over its first 100 mm, on the first
    one's return.  The beam reaches 15 mrad, and they are left out (the drift).  Summed as at
    their own z, the two first lobes would add to 30 mrad."""
    cl = _cluster((0.0, _dipole_lobes("A", (0.0, 100.0, 0.015), (200.0, 300.0, -0.015))),
                  (200.0, _dipole_lobes("B", (0.0, 100.0, 0.015), length=100.0)))
    np.testing.assert_allclose(_exit(_line(cl)), _exit(_drift_line(300.0)), rtol=EXACT)


def test_a_refusal_just_past_a_bound_reads_past_it():
    """Just past a bound the refusal gives the angle or the focusing with as many digits as it
    takes to read past the bound: a corrector bending the beam by 20.04 mrad, an H and a V
    corrector by 20.04 mrad together, the lobes of ``_lobed`` turned together by the 90 deg
    solenoid focusing it by 1.0005 times 0.0088 /m, and an H and a V corrector focusing it by
    0.008802 /m together."""
    _refused(_line(_dipole3("y", 1.0, 0.02004 / _KICK)),
             "a 3-D map whose dipole field on the axis bends the beam by 20.04 mrad (a steering "
             "corrector, up to 20 mrad, is left out) ('COR')")
    h, v = _pair(0.02004 / math.sqrt(2.0), 0.0, 1.0)
    _refused(_line(_cluster((0.0, _map3d()), (0.0, h), (0.0, v))),
             "3-D maps whose dipole fields on the axis bend the beam by 20.04 mrad together "
             "(steering correctors, up to 20 mrad together, are left out) ('CORH' and 'CORV' "
             "in the SUPERPOSE_MAP cluster 'CL')")
    cl = _cluster((0.0, _turning_solenoid(0.5 * math.pi)), (0.0, _lobed(0.5 * 1.0005 * 0.0088)))
    _refused(_line(cl), f"{_focuses('0.008804')} ('COR' in the SUPERPOSE_MAP cluster 'CL')")
    h, v = _pair(0.010, 0.008802 / (2.0 * _SEEN), -1.0)
    _refused(_line(_cluster((0.0, _map3d()), (0.0, h), (0.0, v))),
             "3-D maps whose quadrupole gradients on the axis, as the beam sees them, focus the "
             "beam by up to 1/f = 0.008802 /m together (steering correctors, up to 0.0088 /m "
             "together, are left out) ('CORH' and 'CORV' in the SUPERPOSE_MAP cluster 'CL')")


@pytest.mark.parametrize("label", [None, "COR1H"], ids=["no label", "a label"])
def test_a_refusal_names_a_child_with_no_label_by_its_field_file_too(label):
    """The parser names a straight SUPERPOSE_MAP cluster's children FMAP_001, FMAP_002, ...
    whatever the deck calls them, and keeps each one's field file (its absolute path, without
    an extension): a refusal names such a child by that file's name too, and one with a label
    of its own by its label alone."""
    cor = _dipole3("y", 1.0, 0.0205 / _KICK)
    cor.name, cor.label, cor.field_file = "FMAP_002", label, "/maps/CORH11"
    named = "'FMAP_002' (CORH11)" if label is None else "'COR1H'"
    _refused(_line(_cluster((0.0, _map3d()), (0.0, cor), name="SUPERP_001")),
             "a 3-D map whose dipole field on the axis bends the beam by 20.5 mrad (a steering "
             f"corrector, up to 20 mrad, is left out) ({named} in the SUPERPOSE_MAP cluster "
             "'SUPERP_001')")


def _lobe(Z, a, b):
    """sin² over [a, b] (mm), zero outside: its integral is (b - a) / 2."""
    u = (np.asarray(Z, dtype=float) - a) / (b - a)
    return np.where((u >= 0.0) & (u <= 1.0), np.sin(np.pi * u) ** 2, 0.0)


@pytest.mark.parametrize("field, unit", [
    (lambda Z: (_bump(Z) / math.sqrt(2.0), _bump(Z) / math.sqrt(2.0)), _KICK),
    (lambda Z: (0.0 * Z, _lobe(Z, 20.0, 100.0) - _lobe(Z, 100.0, 180.0)),
     0.01 * 40.0 * 1e-3 / _ref().brho),
], ids=["rotated by 45 deg", "two opposite lobes"])
def test_a_correctors_deflection_is_the_largest_angle_the_beam_reaches_in_it(field, unit):
    """Both planes together, and the largest angle along the map, not its exit's: a corrector
    rotated by 45 deg (B_x = B_y) bends the beam by 20.5 mrad, 14.5 mrad in each plane; a map
    whose dipole field changes sign half way bends it by each lobe's angle and by nothing
    overall.  Left out at 19.5 mrad, refused at 20.5 (``unit``: the angle per unit kb of the
    0.01 T field, in total or of a lobe)."""

    def cor(theta):
        return _grid3("COR", lambda X, Y, Z: 0.01 * field(Z)[0] + 0.0 * X,
                      lambda X, Y, Z: 0.01 * field(Z)[1] + 0.0 * X, lambda X, Y, Z: 0.0 * Z,
                      kb=theta / unit)
    np.testing.assert_allclose(_exit(_line(cor(0.0195))), _exit(_drift_line()), rtol=EXACT)
    np.testing.assert_allclose(_exit(_line(_cluster((0.0, _map3d()), (0.0, cor(0.0195))))),
                               _exit(_line(_map3d())), rtol=EXACT)
    for lat, where in _alone_and_superposed(lambda: cor(0.0205), "COR"):
        _refused(lat, "a 3-D map whose dipole field on the axis bends the beam by 20.5 mrad (a "
                      f"steering corrector, up to 20 mrad, is left out) ({where})")


def _dipole_bumps(length):
    """A 300 mm corrector map holding two dipole bumps, at 60 and 240 mm, bending the beam by
    15 and 25 mrad, its card ``length`` mm long."""
    z = _z(300.0, 601)
    x = np.linspace(-10.0, 10.0, 5)
    X, _Y, Z = np.meshgrid(x, x, z, indexing="ij")
    by = 0.01 * (np.exp(-((Z - 60.0) / 25.0) ** 2) + 5.0 / 3.0 * np.exp(-((Z - 240.0) / 25.0) ** 2))
    one = 0.01 * 25.0 * math.sqrt(math.pi) * 1e-3 / _ref().brho          # rad per unit kb
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.0 * Z,
                                               Fy=by + 0.0 * X, Fz=0.0 * Z)
    return FieldMap3D("COR", length=length, field_data=fd, kb=0.015 / one, ke=0.0,
                      aperture=30.0)


def test_a_correctors_deflection_counts_over_its_window():
    """The two-bump corrector map (``_dipole_bumps``): the deflection counts the field the
    solver reads.  Cut to 150 mm by its card, the map bends the beam by the first bump's
    15 mrad and is left out; superposed on the 200 mm solenoid from z0 = -150 mm (the cluster
    starts at the map's 150 mm), by the second's 25 mrad; whole, by 40 mrad."""
    tail = "mrad (a steering corrector, up to 20 mrad, is left out)"
    np.testing.assert_allclose(_exit(_line(_dipole_bumps(150.0))), _exit(_drift_line(150.0)),
                               rtol=EXACT)
    _refused(_line(_cluster((0.0, _map3d()), (-150.0, _dipole_bumps(300.0)))),
             f"a 3-D map whose dipole field on the axis bends the beam by 25.0 {tail} ('COR' in "
             "the SUPERPOSE_MAP cluster 'CL')")
    _refused(_line(_dipole_bumps(300.0)),
             f"a 3-D map whose dipole field on the axis bends the beam by 40.0 {tail} ('COR')")


def test_a_clusters_voltage_rel_scales_its_correctors_deflection():
    """The cluster's amplitude factor (1 + voltage_rel) multiplies its corrector child too: at
    +10 % a 19 mrad corrector bends the beam by 20.9 mrad and is refused; at -10 % a 21 mrad
    one bends it by 18.9 mrad and is left out (the cluster is its solenoid at 0.9 x kb)."""
    def cl(theta, vr):
        return _cluster((0.0, _map3d()), (0.0, _corrector3(name="COR", kb=theta / _KICK)),
                        voltage_rel=vr)
    _refused(_line(cl(0.019, 0.1)), "a 3-D map whose dipole field on the axis bends the beam by "
                                    "20.9 mrad (a steering corrector, up to 20 mrad, is left "
                                    "out) ('COR' in the SUPERPOSE_MAP cluster 'CL')")
    np.testing.assert_allclose(_exit(_line(cl(0.021, -0.1))), _exit(_line(_map3d(kb=KB * 0.9))),
                               rtol=EXACT)


def test_a_child_cut_by_its_card_bends_the_beam_by_its_field_up_to_the_cut():
    """The two-bump corrector map superposed from z0 = 0 on a 300 mm solenoid map.  Cut to
    150 mm by its card, it reads no field past its card's length in the 300 mm cluster: it bends
    the beam by the first bump's 15 mrad and is left out.  Whole, it bends it by 40 mrad."""
    z = _z(300.0, 601)
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z,
                                               Fz=np.exp(-((z - 150.0) / 35.0) ** 2))

    def sol():
        return FieldMap("SOL", length=300.0, field_data=copy.deepcopy(fd), kb=KB, ke=0.0,
                        aperture=30.0)
    np.testing.assert_allclose(_exit(_line(_cluster((0.0, sol()), (0.0, _dipole_bumps(150.0))))),
                               _exit(_line(_cluster((0.0, sol())))), rtol=EXACT)
    _refused(_line(_cluster((0.0, sol()), (0.0, _dipole_bumps(300.0)))),
             "a 3-D map whose dipole field on the axis bends the beam by 40.0 mrad (a steering "
             "corrector, up to 20 mrad, is left out) ('COR' in the SUPERPOSE_MAP cluster 'CL')")


def test_a_child_ending_at_the_clusters_exit_counts_its_field_to_the_end():
    """A 150 mm corrector map superposed from z0 = 50 mm on the 200 mm solenoid ends at the
    cluster's exit: it reads its field up to there, where its 25 mrad dipole lobe lies (its last
    50 mm), and is refused."""
    z = _z(150.0, 301)
    x = np.linspace(-10.0, 10.0, 5)
    X, _Y, Z = np.meshgrid(x, x, z, indexing="ij")
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.0 * Z,
                                               Fy=0.01 * _lobe(Z, 100.0, 150.0) + 0.0 * X,
                                               Fz=0.0 * Z)
    kb = 0.025 / (0.01 * 25.0 * 1e-3 / _ref().brho)
    cor = FieldMap3D("COR", length=150.0, field_data=fd, kb=kb, ke=0.0, aperture=30.0)
    _refused(_line(_cluster((0.0, _map3d()), (50.0, cor))),
             "a 3-D map whose dipole field on the axis bends the beam by 25.0 mrad (a steering "
             "corrector, up to 20 mrad, is left out) ('COR' in the SUPERPOSE_MAP cluster 'CL')")


def test_a_solenoids_fringe_on_a_correctors_axis_is_no_quadrupole():
    """A corrector map that also holds a weaker solenoid field with sharp ends: across the axis
    its fringe is the gradient's trace, which a quadrupole's gradient is not.  Its dipole field
    is left out, and its B_z focuses the beam as the same map's without the dipole field (a
    solenoid map) does; the matrix envelope through it agrees within 1e-3."""
    def ends(Z):
        return 0.5 * (np.tanh((Z - 50.0) / 1.0) - np.tanh((Z - 150.0) / 1.0))

    def dends(Z):
        return 0.5 * (1.0 / np.cosh((Z - 50.0) / 1.0) ** 2 - 1.0 / np.cosh((Z - 150.0) / 1.0) ** 2)

    def cor(dipole):
        return _grid3("COR", lambda X, Y, Z: -0.5 * X * 0.009 * dends(Z),
                      lambda X, Y, Z: dipole * _bump(Z) - 0.5 * Y * 0.009 * dends(Z),
                      lambda X, Y, Z: 0.009 * ends(Z) + 0.0 * X, kb=0.5)
    lat = _line(cor(0.01))
    np.testing.assert_allclose(_exit(lat), _exit(_line(cor(0.0))), rtol=EXACT)
    np.testing.assert_allclose(_exit(lat), _exit(lat, "matrix"), rtol=REL)


def _spiked(spike, card=L):
    """A 3-D solenoid map, B_z = 0.3 T x ``_bump`` with its paraxial radial field, that also
    holds a dipole field B_y = ``spike`` T x a Gaussian 0.3 mm long at z = 60 mm, uniform across
    the grid; its card ``card`` mm long (its grid 200 mm)."""
    def spiked(Z):
        return spike * np.exp(-0.5 * ((Z - 60.0) / 0.3) ** 2)
    m = _grid3("MIX", lambda X, Y, Z: -0.5 * X * 0.3 * _dbump(Z),
               lambda X, Y, Z: spiked(Z) - 0.5 * Y * 0.3 * _dbump(Z),
               lambda X, Y, Z: 0.3 * _bump(Z) + 0.0 * X)
    return m if card == L else FieldMap3D("MIX", length=card, field_data=m.field_data, kb=1.0,
                                          ke=0.0, aperture=30.0)


@pytest.mark.parametrize("superposed", [False, True], ids=["alone", "superposed"])
def test_a_solenoid_map_with_a_dipole_spike_on_its_axis_focuses_by_its_b_z(superposed):
    """A solenoid map whose dipole spike on the axis (0.305 T, bending the beam by 9 mrad) peaks
    above its B_z (0.3 T): its field on the axis is read as a corrector's, the spike is left
    out, and its B_z focuses the beam as the same map's without the spike does, standing alone
    or superposed on a weak 3-D solenoid (0.03 T); the matrix envelope through it agrees within
    1e-3.
    (Left out whole, with its B_z, it would be the drift: 7.50 mm against the matrix
    envelope's 4.76.)"""
    def line(spike):
        return _line(_cluster((0.0, _map3d(kb=0.03)), (0.0, _spiked(spike))) if superposed
                     else _spiked(spike))
    np.testing.assert_allclose(_exit(line(0.305)), _exit(line(0.0)), rtol=EXACT)
    np.testing.assert_allclose(_exit(line(0.305)), _exit(line(0.305), "matrix"), rtol=REL)


@pytest.mark.parametrize("z0, card", [(100.0, L), (0.0, 150.0)],
                         ids=["placed at 100 mm", "its card cut to 150 mm"])
def test_a_correctors_b_z_is_read_over_its_own_window(z0, card):
    """The spiked solenoid map as a cluster child placed at z0 = 100 mm, or with its card cut
    to 150 mm of its 200 mm grid (the cluster runs on to 200 mm), superposed on the weak 3-D
    solenoid: its B_z is read from where the child starts to where its card ends, as the same
    map's without the spike is; placed at 100 mm, the matrix envelope through it agrees within
    1e-3."""
    def line(spike):
        return _line(_cluster((0.0, _map3d(kb=0.03)), (z0, _spiked(spike, card))))
    np.testing.assert_allclose(_exit(line(0.305)), _exit(line(0.0)), rtol=EXACT)
    if card == L:
        np.testing.assert_allclose(_exit(line(0.305)), _exit(line(0.305), "matrix"), rtol=REL)


def test_a_corrector_whose_grid_misses_the_axis_is_refused():
    def cor():
        return _corrector3(kb=0.5, x=[5.0, 10.0, 15.0])
    for lat, where in _alone_and_superposed(cor, "B3"):
        _refused(lat, f"a 3-D map whose grid does not reach the axis x = 0, y = 0 ({where})")


def test_a_corrector_whose_field_on_the_axis_is_a_sixth_of_its_peak_is_left_out():
    """A corrector (12.4 mrad) whose field departs from a uniform one as the sixth power of the
    distance from the axis, on 17 points over +-40 mm: at the grid's corners, beyond the beam,
    B_x is 6 times the field on the axis, which is a sixth of the peak, above a tenth.  The
    matrix envelope through it is the drift within 1e-3."""
    def b(X, Y, Z):            # B_y + i B_x
        return 0.01 * _bump(Z) * (1.0 + 0.75 * ((X + 1j * Y) / 40.0) ** 6)
    lat = _line(_grid3("COR", lambda X, Y, Z: b(X, Y, Z).imag, lambda X, Y, Z: b(X, Y, Z).real,
                       lambda X, Y, Z: 0.0 * Z, kb=0.5, x=np.linspace(-40.0, 40.0, 17)))
    np.testing.assert_allclose(_exit(lat), _exit(_drift_line()), rtol=EXACT)
    np.testing.assert_allclose(_exit(lat, "matrix"), _exit(_drift_line(), "matrix"), rtol=REL)


def _ends(Z):
    """A solenoid's B_z profile with ends 0.5 mm long, at 50 and 150 mm."""
    return 0.5 * (np.tanh((Z - 50.0) / 0.5) - np.tanh((Z - 150.0) / 0.5))


def _dends(Z):
    """d _ends / dz (1/mm): 1 at 50 mm."""
    return 1.0 / np.cosh((Z - 50.0) / 0.5) ** 2 - 1.0 / np.cosh((Z - 150.0) / 0.5) ** 2


@pytest.mark.parametrize("dipole", [0.8, 1.0 / 1.05], ids=["0.8 of B_z", "B_z 1.05 times it"])
def test_a_map_whose_axis_holds_more_solenoid_than_dipole_field_is_refused(dipole):
    """A solenoid field with ends 0.5 mm long and a weaker dipole field (0.8 of it, 9.9 mrad;
    or B_z 1.05 times it), on 4 points over +-10 mm: the grid point nearest the axis, 4.7 mm off
    it, reads the ends' radial field above the solenoid's, so the map is no solenoid by the
    nearest point's rule; on the axis its B_z exceeds the dipole field.  Left out as a
    corrector, it would lose that B_z: it is refused, as a solenoid's field on the axis that the
    point nearest it does not show."""
    def mixed():
        return _grid3("SB", lambda X, Y, Z: -0.5 * X * _dends(Z),
                      lambda X, Y, Z: dipole * _bump(Z) - 0.5 * Y * _dends(Z),
                      lambda X, Y, Z: _ends(Z) + 0.0 * X, kb=0.005,
                      x=np.linspace(-10.0, 10.0, 4))
    for lat, where in _alone_and_superposed(mixed, "SB"):
        _refused(lat, "a 3-D map whose field on the axis is a solenoid's, which its grid point "
                      f"nearest the axis does not show ({where})")


def _nearest_reads_bz(bz, dipole, s, kb=1.0):
    """B_y = ``dipole`` x _ends across the grid (a dipole field on the axis) and B_z = ``bz`` x
    _ends + ``s`` y _dends, on 4 points over +-10 mm: the point nearest the axis, 3.3 mm off it
    in x and in y, reads the ends' B_z there (up to 0.5 ``bz`` + 3.3 ``s``), which the axis
    itself does not have; the peak is 0.5 ``bz`` + 10 ``s``, at y = 10 mm and z = 50 mm."""
    return _grid3("NB", lambda X, Y, Z: 0.0 * Z,
                  lambda X, Y, Z: dipole * _ends(Z) + 0.0 * X,
                  lambda X, Y, Z: bz * _ends(Z) + s * Y * _dends(Z), kb=kb,
                  x=np.linspace(-10.0, 10.0, 4))


_DIPOLE_NOT_SHOWN = ("a 3-D map whose field on the axis is a dipole's, which its grid point "
                     "nearest the axis does not show")
_MULTIPOLE = "a 3-D map whose field vanishes on the axis, with no gradient there (a multipole)"


@pytest.mark.parametrize("bz, dipole, s, what", [
    (0.0, 0.01, 0.01 / 1.1, _DIPOLE_NOT_SHOWN),
    (0.0, 0.01, 0.01 / 0.9, _MULTIPOLE),
    (0.0095, 0.01, 0.002, _DIPOLE_NOT_SHOWN),
    (0.02, 0.002, 0.024, _MULTIPOLE),
], ids=["dipole 0.11 of the peak", "dipole 0.09 of the peak", "B_z 0.95 of the dipole",
        "B_z 0.08 of the peak"])
def test_a_map_whose_nearest_point_reads_a_solenoid_the_axis_does_not_hold_is_refused(
        bz, dipole, s, what):
    """The grid point nearest the axis reads the ends' B_z, above the field across it and a
    tenth of the peak, so the map is a solenoid by the nearest point's rule; on the axis the
    field is a dipole's (transverse, at least its B_z and above a tenth of the peak) or vanishes
    (no field above a tenth of the peak).  Read as a solenoid it would be one of next to no
    field: it is refused, as a dipole's field on the axis that the point nearest it does not
    show, or as a multipole."""
    for lat, where in _alone_and_superposed(lambda: _nearest_reads_bz(bz, dipole, s), "NB"):
        _refused(lat, f"{what} ({where})")


def test_a_map_whose_nearest_point_and_axis_read_a_solenoid_is_a_solenoid():
    """On the axis B_z 1.05 times the dipole field (and 0.42 of the peak): a solenoid by both
    rules, read as the 1-D map of its on-axis B_z (up to rounding in the interpolation)."""
    z = _z()
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=0.0105 * _ends(z))
    one_d = FieldMap("NB", length=L, field_data=fd, kb=20.0, ke=0.0, aperture=30.0)
    got = _exit(_line(_nearest_reads_bz(0.0105, 0.01, 0.002, kb=20.0)))
    np.testing.assert_allclose(got, _exit(_line(one_d)), rtol=ODE)
    assert got[0] < 0.95 * _exit(_drift_line())[0]


def test_a_corrector_at_its_bounds_is_left_out(monkeypatch):
    """The bounds are inclusive.  At bounds of 0 mrad and 0 /m: a corrector map whose field, a
    dipole field with a gradient part, lies past its card's length bends and focuses the beam by
    nothing and is left out; with the card over the field, it bends the beam and is refused."""
    from linac_gen.tracking import sacherer
    monkeypatch.setattr(sacherer, "_CORRECTOR_MAX_DEFLECTION", 0.0)
    monkeypatch.setattr(sacherer, "_CORRECTOR_MAX_FOCUSING", 0.0)
    z = _z(400.0, 801)
    x = np.linspace(-10.0, 10.0, 5)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")

    def cor(length):
        fd = FieldMapData(z=z)
        fd.channels[Channel.STAT_B] = FieldChannel(
            geometry=7, x=x, y=x, z=z, Fx=1e-4 * Y * _lobe(Z, 250.0, 350.0),
            Fy=(0.01 + 1e-4 * X) * _lobe(Z, 250.0, 350.0), Fz=0.0 * Z)
        return FieldMap3D("COR", length=length, field_data=fd, kb=0.5, ke=0.0, aperture=30.0)
    np.testing.assert_allclose(_exit(_line(cor(200.0))), _exit(_drift_line()), rtol=EXACT)
    _refused(_line(cor(400.0)), "a 3-D map whose dipole field on the axis bends the beam by "
                                "10.0 mrad (a steering corrector, up to 0 mrad, is left out) "
                                "('COR')")


def test_the_run_reads_each_map_as_it_stands():
    """A corrector left out when the solver is built, pushed past the bound before the run,
    is refused by the run, at the beam's rigidity."""
    lat = _line(_corrector3(name="COR", kb=0.015 / _KICK))
    solver = SachererSolver(lat, _ref(), dict(_INIT), current=0.0)
    lat.elements[1].kb = 0.025 / _KICK
    with pytest.raises(ValueError, match=re.escape(
            "no model of a 3-D map whose dipole field on the axis bends the beam by 25.0 mrad (a "
            f"steering corrector, up to 20 mrad, is left out) ('COR') — {_MATRIX}")):
        solver.run()


def test_a_corrector_reads_no_field_outside_its_grid_or_its_cluster():
    """A corrector map of a uniform 0.01 T field across the axis over its grid, 0 to 100 mm,
    behind a 200 mm card: it reads no field past its grid's last node, so at 15 mrad it is left
    out.  The same field over 200 mm, at the kb that bends the beam by 50 mrad, superposed from
    z0 = -300 mm on the solenoid, lies before the cluster's start: it reads no field there."""
    def flat(length):
        z = _z(length, int(2 * length) + 1)
        x = np.linspace(-10.0, 10.0, 5)
        X, _Y, Z = np.meshgrid(x, x, z, indexing="ij")
        fd = FieldMapData(z=z)
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.0 * Z,
                                                   Fy=0.01 + 0.0 * X * Z, Fz=0.0 * Z)
        return fd
    per_kb = 0.01 * 100.0 * 1e-3 / _ref().brho                # rad per unit kb, over 100 mm
    short = FieldMap3D("COR", length=200.0, field_data=flat(100.0), kb=0.015 / per_kb, ke=0.0,
                       aperture=30.0)
    np.testing.assert_allclose(_exit(_line(short)), _exit(_drift_line()), rtol=EXACT)
    early = FieldMap3D("COR", length=200.0, field_data=flat(200.0), kb=0.025 / per_kb, ke=0.0,
                       aperture=30.0)
    np.testing.assert_allclose(_exit(_line(_cluster((0.0, _map3d()), (-300.0, early)))),
                               _exit(_line(_map3d())), rtol=EXACT)


def test_a_corrector_on_a_coarse_grid_along_z_counts_its_first_and_last_cells():
    """A uniform dipole field on three z nodes, 0, 50 and 100 mm, behind a 100 mm card: the
    deflection runs from the window's start to its end, both cells, 25 mrad in all."""
    z = np.array([0.0, 50.0, 100.0])
    x = np.linspace(-10.0, 10.0, 5)
    X, _Y, Z = np.meshgrid(x, x, z, indexing="ij")
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.0 * Z,
                                               Fy=0.01 + 0.0 * X * Z, Fz=0.0 * Z)
    kb = 0.025 / (0.01 * 100.0 * 1e-3 / _ref().brho)
    cor = FieldMap3D("COR", length=100.0, field_data=fd, kb=kb, ke=0.0, aperture=30.0)
    _refused(_line(cor), "a 3-D map whose dipole field on the axis bends the beam by 25.0 mrad "
                         "(a steering corrector, up to 20 mrad, is left out) ('COR')")


def test_a_maps_window_starts_at_its_grids_first_node():
    """A corrector map whose grid runs from z = 100 to 400 mm, cut to 150 mm by its card: it
    reads its field from 100 to 250 mm, where its 25 mrad dipole bump lies."""
    z = np.linspace(100.0, 400.0, 601)
    x = np.linspace(-10.0, 10.0, 5)
    X, _Y, Z = np.meshgrid(x, x, z, indexing="ij")
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=0.0 * Z,
                                               Fy=0.01 * _lobe(Z, 150.0, 250.0) + 0.0 * X,
                                               Fz=0.0 * Z)
    kb = 0.025 / (0.01 * 50.0 * 1e-3 / _ref().brho)
    cor = FieldMap3D("COR", length=150.0, field_data=fd, kb=kb, ke=0.0, aperture=30.0)
    _refused(_line(cor), "a 3-D map whose dipole field on the axis bends the beam by 25.0 mrad "
                         "(a steering corrector, up to 20 mrad, is left out) ('COR')")


def test_no_mode_tracks_a_2d_cartesian_map():
    """Why its refusal points to no other mode: the field-map sampler has no such geometry
    (and the deck parser drops the card)."""
    with pytest.raises(ValueError, match="unsupported geometry 6"):
        common.run_envelope_sim(_line(_cart2d()), _cfg(), env_solver="matrix")


# ---------------------------------------------------------------------------
# the studies
# ---------------------------------------------------------------------------
def _study_line():
    return _line(_map1d(name="SOL"))


@pytest.mark.parametrize("path_mode", [False, True], ids=["in-memory", "deck file"])
def test_a_failure_study_turns_the_solenoid_map_off(path_mode, tmp_path):
    """FailureStudy(mode='envelope', env_solver='sacherer') with the solenoid map OFF
    (voltage_rel = -1), in memory (the Failure Study tab) and from a deck through the scan
    pool (the CLI): the exit moves as the matrix envelope's does, to the drift's, within
    1e-3 (it stayed at the baseline)."""
    from linac_gen.failures import FailureStudy, enumerate_scenarios
    from linac_gen.io.tracewin_parser import parse_tracewin
    if path_mode:
        deck = _deck(tmp_path)
        lat, _meta = parse_tracewin(str(deck))
    else:
        lat = _study_line()
    scenarios, n2c, names = enumerate_scenarios(lat, types={"solenoid"})
    assert len(names) == 1
    moved = {}
    for solver in ("sacherer", "matrix"):
        if path_mode:
            study = FailureStudy(str(deck), beam_overrides=_beam_overrides(),
                                 mode="envelope", env_solver=solver)
            res = study.run(scenarios, names, n2c, serial=True)
        else:
            study = FailureStudy(lattice=lat, beam_config=_cfg(), mode="envelope",
                                 env_solver=solver)
            res = study.run(scenarios, names, n2c)
        moved[solver] = (res.baseline["sigma_x"], res.impacts[0].metrics["sigma_x"])
    base_s, off_s = moved["sacherer"]
    base_m, off_m = moved["matrix"]
    assert off_s == pytest.approx(_exit(_drift_line())[0], rel=REL)
    assert base_s == pytest.approx(base_m, rel=REL) and off_s == pytest.approx(off_m, rel=REL)
    assert (off_s - base_s) == pytest.approx(off_m - base_m, rel=1e-2)
    assert off_s > 1.3 * base_s


def _beam_overrides():
    cfg = _cfg()
    return {k: getattr(cfg, k) for k in (
        "species", "energy", "frequency", "current", "emit_nx", "alpha_x", "beta_x",
        "emit_ny", "alpha_y", "beta_y", "emit_z", "alpha_z", "beta_z", "continuous")}


def test_an_error_studys_voltage_rel_draws_move_its_seeds():
    """ErrorStudy(mode='envelope', env_solver='sacherer') drawing voltage_rel on a solenoid
    map: each seed is the Sacherer at its draw and the matrix envelope's seed from the same
    draw (the seeds all sat at the nominal exit)."""
    from linac_gen.errors.error_model import ErrorStudy
    nominal = _exit(_study_line())
    seeds, draws = {}, {}
    for solver in ("sacherer", "matrix"):
        study = ErrorStudy(_study_line(), _cfg(), n_seeds=3, mode="envelope",
                           env_solver=solver)
        study.add_error("SOL", "voltage_rel", sigma=0.05, cutoff=3.0)
        res = study.run()
        seeds[solver] = [np.array([r.sigma_x[-1], r.sigma_y[-1]]) for r in res._recorders]
        draws[solver] = [[d["value"] for d in study.applied_log[k]
                          if d["parameter"] == "voltage_rel"] for k in range(3)]
    assert draws["sacherer"] == draws["matrix"]
    assert all(len(d) == 1 for d in draws["sacherer"])
    for (draw,), got, want in zip(draws["sacherer"], seeds["sacherer"], seeds["matrix"],
                                  strict=True):
        assert abs(got[0] / nominal[0] - 1.0) > 0.5 * abs(draw) > 0.0
        np.testing.assert_allclose(got, want, rtol=REL)
        np.testing.assert_allclose(got, _exit(_line(_map1d(voltage_rel=draw))), rtol=EXACT)
