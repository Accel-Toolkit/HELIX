"""Tests for TraceWin .dat writer (Task 7.4)."""
import os

import pytest

if not os.path.isdir("Fields"):
    pytest.skip("Fields/ field-map data is not distributed with the "
                "repository (third-party ANL/CEA data) — see "
                "examples/FIELD_MAPS.md", allow_module_level=True)

import os
import tempfile

import numpy as np
import pytest

from linac_gen.io.tracewin_writer import write_tracewin
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.steerer import Steerer
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.marker import Marker
from linac_gen.elements.space_charge_comp import SpaceChargeComp

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tmp_path():
    f = tempfile.NamedTemporaryFile(suffix='.dat', delete=False)
    f.close()
    return f.name


def _build_fodo():
    lat = Lattice()
    lat.add(Drift("DRIFT_001", 100.0, aperture=20.0, n_steps=5))
    lat.add(Quadrupole("QUAD_001", 50.0, 5.2, aperture=20.0, n_steps=10))
    lat.add(Drift("DRIFT_002", 200.0, aperture=20.0, n_steps=5))
    lat.add(Quadrupole("QUAD_002", 50.0, -5.2, aperture=20.0, n_steps=10))
    lat.add(Drift("DRIFT_003", 100.0, aperture=20.0, n_steps=5))
    return lat


# ── Basic output structure ────────────────────────────────────────────────────

class TestBasicOutput:
    def test_creates_file(self):
        path = _tmp_path()
        try:
            write_tracewin(_build_fodo(), path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_ends_with_end_card(self):
        path = _tmp_path()
        try:
            write_tracewin(_build_fodo(), path)
            with open(path) as f:
                content = f.read()
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            assert lines[-1] == "END"
        finally:
            os.unlink(path)

    def test_drift_line_format(self):
        lat = Lattice()
        lat.add(Drift("DRIFT_001", 100.0, aperture=20.0, n_steps=5))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "DRIFT" in content
        finally:
            os.unlink(path)

    def test_quad_line_format(self):
        lat = Lattice()
        lat.add(Quadrupole("QUAD_001", 50.0, 5.2, aperture=20.0, n_steps=10))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "QUAD" in content
        finally:
            os.unlink(path)

    def test_solenoid_line_format(self):
        lat = Lattice()
        lat.add(Solenoid("SOL_001", 200.0, 0.5, aperture=20.0, n_steps=10))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "SOLENOID" in content
        finally:
            os.unlink(path)

    def test_gap_line_format(self):
        lat = Lattice()
        lat.add(RFGap("GAP_001", 1.5, -30.0, 352.21, ttf=0.9, aperture=20.0))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "GAP" in content
        finally:
            os.unlink(path)

    def test_bend_line_format(self):
        lat = Lattice()
        lat.add(Dipole("BEND_001", 45.0, 500.0, e1=22.5, e2=22.5, aperture=20.0, n_steps=5))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "BEND" in content
        finally:
            os.unlink(path)

    def test_steerer_line_format(self):
        lat = Lattice()
        lat.add(Steerer("STEER_001", bx_l=0.001, by_l=-0.002))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "THIN_STEERING" in content
        finally:
            os.unlink(path)

    def test_steerer_electric_round_trip(self):
        """elec=1 must survive write -> parse (4-field form)."""
        from linac_gen.io.tracewin_parser import parse_tracewin
        lat = Lattice()
        lat.add(Steerer("STEER_001", bx_l=500.0, by_l=-200.0, elec=True))
        lat.add(Steerer("STEER_002", bx_l=0.001, by_l=0.0))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "THIN_STEERING 500 -200 0 1" in content
            # Magnetic steerers keep the historical 2-field form.
            assert "THIN_STEERING 0.001 0\n" in content
            lat2, _ = parse_tracewin(path)
            s1, s2 = [e for e in lat2.elements
                      if isinstance(e, Steerer)]
            assert s1.elec is True and s1.bx_l == 500.0
            assert s2.elec is False
        finally:
            os.unlink(path)

    def test_aperture_line_format(self):
        lat = Lattice()
        lat.add(Aperture("APER_001", aperture_type=0, dx=20.0))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "APERTURE" in content
        finally:
            os.unlink(path)

    def test_marker_line_format(self):
        lat = Lattice()
        lat.add(Marker("MARK_001", snapshot=False))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "MARKER" in content
        finally:
            os.unlink(path)

    def test_snapshot_marker_becomes_diag_phase(self):
        lat = Lattice()
        lat.add(Marker("MARK_001", snapshot=True))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "DIAG_PHASE" in content
        finally:
            os.unlink(path)

    def test_space_charge_comp_line_format(self):
        lat = Lattice()
        lat.add(SpaceChargeComp("SCCOMP_001", factor=0.9))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "SPACE_CHARGE_COMP" in content
        finally:
            os.unlink(path)


# ── FREQ card handling ────────────────────────────────────────────────────────

class TestFreqWriting:
    def test_freq_written_before_gap(self):
        lat = Lattice()
        lat.add(RFGap("GAP_001", 1.5, -30.0, 352.21))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            # FREQ line should appear before GAP line
            freq_idx = next(i for i, l in enumerate(lines) if l.startswith("FREQ"))
            gap_idx = next(i for i, l in enumerate(lines) if l.startswith("GAP"))
            assert freq_idx < gap_idx
        finally:
            os.unlink(path)

    def test_freq_value_written_correctly(self):
        lat = Lattice()
        lat.add(RFGap("GAP_001", 1.5, -30.0, 704.42))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "704.42" in content
        finally:
            os.unlink(path)

    def test_freq_not_repeated_for_same_frequency(self):
        lat = Lattice()
        lat.add(RFGap("GAP_001", 1.5, -30.0, 352.21))
        lat.add(RFGap("GAP_002", 1.5, -30.0, 352.21))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            freq_count = content.count("FREQ")
            assert freq_count == 1
        finally:
            os.unlink(path)

    def test_freq_written_again_on_frequency_change(self):
        lat = Lattice()
        lat.add(RFGap("GAP_001", 1.5, -30.0, 352.21))
        lat.add(RFGap("GAP_002", 1.5, -30.0, 704.42))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            freq_count = content.count("FREQ")
            assert freq_count == 2
        finally:
            os.unlink(path)


# ── Round-trip tests ──────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_fodo_round_trip_element_count(self):
        original = _build_fodo()
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            parsed, _ = parse_tracewin(path)
            assert len(parsed) == len(original)
        finally:
            os.unlink(path)

    def test_fodo_round_trip_element_types(self):
        original = _build_fodo()
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            parsed, _ = parse_tracewin(path)
            orig_types = [type(e) for e in original.elements]
            parsed_types = [type(e) for e in parsed.elements]
            assert orig_types == parsed_types
        finally:
            os.unlink(path)

    def test_fodo_round_trip_drift_length(self):
        original = _build_fodo()
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            parsed, _ = parse_tracewin(path)
            orig_drifts = [e for e in original.elements if isinstance(e, Drift)]
            new_drifts = [e for e in parsed.elements if isinstance(e, Drift)]
            for od, nd in zip(orig_drifts, new_drifts):
                assert nd.length == pytest.approx(od.length)
        finally:
            os.unlink(path)

    def test_fodo_round_trip_quad_gradient(self):
        original = _build_fodo()
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            parsed, _ = parse_tracewin(path)
            orig_quads = [e for e in original.elements if isinstance(e, Quadrupole)]
            new_quads = [e for e in parsed.elements if isinstance(e, Quadrupole)]
            for oq, nq in zip(orig_quads, new_quads):
                assert nq.gradient == pytest.approx(oq.gradient)
        finally:
            os.unlink(path)

    def test_simple_fodo_fixture_round_trip(self):
        """Parse fixture, write, re-parse, compare element count and types."""
        original, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            parsed, _ = parse_tracewin(path)
            assert len(parsed) == len(original)
            orig_types = [type(e) for e in original.elements]
            parsed_types = [type(e) for e in parsed.elements]
            assert orig_types == parsed_types
        finally:
            os.unlink(path)

    def test_rf_gap_round_trip_voltage(self):
        # TraceWin's GAP card carries only E0TL = V * T (the transit-time
        # factor is folded into the effective voltage on the card).  The
        # writer therefore emits V*T, and the parser reads back a gap with
        # voltage=V*T and ttf=1.0.  The effective kick is preserved.
        V, T = 2.3, 0.85
        lat = Lattice()
        lat.add(RFGap("GAP_001", V, -35.0, 352.21, ttf=T, aperture=15.0))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            gap = next(e for e in parsed.elements if isinstance(e, RFGap))
            assert gap.voltage == pytest.approx(V * T)
            assert gap.ttf == pytest.approx(1.0)
            assert gap.phase == pytest.approx(-35.0)
            assert gap.frequency == pytest.approx(352.21)
            # Effective V*T is the physically-meaningful quantity.
            assert (gap.voltage * gap.ttf) == pytest.approx(V * T)
        finally:
            os.unlink(path)

    def test_solenoid_round_trip(self):
        # TraceWin SOLENOID carries L B R only — ``n_steps`` is a Linac_Gen
        # runtime-only attribute and is NOT round-trippable via the .dat file.
        lat = Lattice()
        lat.add(Solenoid("SOL_001", 300.0, 0.8, aperture=25.0, n_steps=8))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            sol = parsed.elements[0]
            assert isinstance(sol, Solenoid)
            assert sol.length == pytest.approx(300.0)
            assert sol.field == pytest.approx(0.8)
            assert sol.aperture == pytest.approx(25.0)
        finally:
            os.unlink(path)

    def test_dipole_round_trip(self):
        # TraceWin BEND carries angle/rho/field_index/aperture/hv only; pole-face
        # rotations (e1/e2) live on separate EDGE cards and therefore do NOT
        # survive a bare Dipole -> BEND -> Dipole round-trip.
        lat = Lattice()
        lat.add(Dipole(
            "BEND_001", 30.0, 800.0,
            field_index=0.5, aperture=20.0, hv=0, n_steps=5,
        ))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            d = parsed.elements[0]
            assert isinstance(d, Dipole)
            assert d.angle == pytest.approx(30.0)
            assert d.rho == pytest.approx(800.0)
            assert d.field_index == pytest.approx(0.5)
            assert d.aperture == pytest.approx(20.0)
            assert d.hv == 0
        finally:
            os.unlink(path)

    def test_steerer_round_trip(self):
        lat = Lattice()
        lat.add(Steerer("STEER_001", bx_l=0.002, by_l=-0.001))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            s = parsed.elements[0]
            assert isinstance(s, Steerer)
            assert s.bx_l == pytest.approx(0.002)
            assert s.by_l == pytest.approx(-0.001)
        finally:
            os.unlink(path)

    def test_aperture_round_trip(self):
        # Circular aperture (n=1) with dx=radius; dy ignored on circular but
        # preserved through the round trip so the card keeps 3 positionals.
        lat = Lattice()
        lat.add(Aperture("APER_001", aperture_type=1, dx=18.0, dy=12.0))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            ap = parsed.elements[0]
            assert isinstance(ap, Aperture)
            assert ap.aperture_type == 1
            assert ap.dx == pytest.approx(18.0)
            assert ap.dy == pytest.approx(12.0)
        finally:
            os.unlink(path)

    def test_space_charge_comp_round_trip(self):
        lat = Lattice()
        lat.add(SpaceChargeComp("SCCOMP_001", factor=0.75))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            sc = parsed.elements[0]
            assert isinstance(sc, SpaceChargeComp)
            assert sc.factor == pytest.approx(0.75)
        finally:
            os.unlink(path)

    def test_marker_round_trip(self):
        lat = Lattice()
        lat.add(Marker("MARK_001", snapshot=False))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            m = parsed.elements[0]
            assert isinstance(m, Marker)
            assert not m.snapshot
        finally:
            os.unlink(path)

    def test_snapshot_marker_round_trip(self):
        lat = Lattice()
        lat.add(Marker("MARK_001", snapshot=True))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            m = parsed.elements[0]
            assert isinstance(m, Marker)
            assert m.snapshot
        finally:
            os.unlink(path)


# ── Empty lattice ─────────────────────────────────────────────────────────────

class TestEmptyLattice:
    def test_empty_lattice_writes_just_end(self):
        lat = Lattice()
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert lines == ["END"]
        finally:
            os.unlink(path)


# ── Numerical precision ───────────────────────────────────────────────────────

class TestNumericalPrecision:
    def test_drift_length_preserved(self):
        lat = Lattice()
        lat.add(Drift("DRIFT_001", 123.456, n_steps=3))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            assert parsed.elements[0].length == pytest.approx(123.456, rel=1e-6)
        finally:
            os.unlink(path)

    def test_quad_gradient_preserved(self):
        lat = Lattice()
        lat.add(Quadrupole("QUAD_001", 50.0, -7.654321, n_steps=5))
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            parsed, _ = parse_tracewin(path)
            assert parsed.elements[0].gradient == pytest.approx(-7.654321, rel=1e-6)
        finally:
            os.unlink(path)


# ── End-to-end round-trip on an example .dat ─────────────────────────────────

def test_roundtrip_fodo_example(tmp_path):
    """Parse examples/fodo_cell.dat, write it back, re-parse, elements match."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin
    src = os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"
    )
    lat1, _ = parse_tracewin(src)
    dst = tmp_path / "roundtrip.dat"
    write_tracewin(lat1, str(dst))
    lat2, _ = parse_tracewin(str(dst))
    assert len(lat1.elements) == len(lat2.elements)
    for a, b in zip(lat1.elements, lat2.elements):
        assert type(a) is type(b)
        assert a.length == pytest.approx(b.length, rel=1e-10)
        # Drifts / quads / etc: aperture round-trips
        if hasattr(a, "aperture"):
            assert a.aperture == pytest.approx(b.aperture, rel=1e-10)


# ── Field-map round-trip ──────────────────────────────────────────────────────
# Regression for the writer silently DROPPING field-map elements: it used to
# emit a comment for FieldMap and an "unsupported element type" comment for
# FieldMap3D, so on reload every RF cavity AND every field-map solenoid
# vanished — taking any matched amplitudes/phases with it.

FIELD_FIXTURE = os.path.join(FIXTURE_DIR, "lattice_with_fieldmap.dat")


def _field_maps(lat):
    return [e for e in lat.elements if "FieldMap" in type(e).__name__]


class TestFieldMapRoundTrip:
    def test_fieldmap_survives_round_trip(self):
        original, _ = parse_tracewin(FIELD_FIXTURE)
        assert len(_field_maps(original)) == 1          # fixture has one
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            parsed, meta = parse_tracewin(path)
            assert not meta.get("warnings"), meta.get("warnings")
            assert len(_field_maps(parsed)) == 1        # pre-fix: 0
            assert len(parsed) == len(original)
        finally:
            os.unlink(path)

    def test_fieldmap_emits_card_not_comment(self):
        original, _ = parse_tracewin(FIELD_FIXTURE)
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            with open(path) as f:
                content = f.read()
            assert "FIELD_MAP " in content
            assert "re-export not supported" not in content
            assert "unsupported element type" not in content
        finally:
            os.unlink(path)

    def test_fieldmap_geom_and_params_preserved(self):
        original, _ = parse_tracewin(FIELD_FIXTURE)
        o = _field_maps(original)[0]
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            n = _field_maps(parse_tracewin(path)[0])[0]
            assert n.geom == o.geom
            assert n.length == pytest.approx(o.length)
            assert n.aperture == pytest.approx(o.aperture)
            assert n.phase == pytest.approx(o.phase)
            assert n.kb == pytest.approx(o.kb)
            assert n.ke == pytest.approx(o.ke)
        finally:
            os.unlink(path)

    def test_matched_values_persist(self):
        """Mutating phase/kb/ke (what ADJUST FMAP tunes) survives a save."""
        original, _ = parse_tracewin(FIELD_FIXTURE)
        fm = _field_maps(original)[0]
        fm.phase, fm.kb, fm.ke = -42.7, 0.835, 1.27
        path = _tmp_path()
        try:
            write_tracewin(original, path)
            n = _field_maps(parse_tracewin(path)[0])[0]
            assert n.phase == pytest.approx(-42.7)
            assert n.kb == pytest.approx(0.835)
            assert n.ke == pytest.approx(1.27)
        finally:
            os.unlink(path)

    def test_no_provenance_falls_back_to_comment(self):
        """A programmatically-built map (no geom/field_file) must not emit a
        broken card — it falls back to a comment instead."""
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.io.field_map_data import FieldMapData
        fm = FieldMap("FMAP_X", length=10.0,
                      field_data=FieldMapData(z=np.linspace(0.0, 10.0, 11)))
        assert fm.geom is None and fm.field_file is None
        lat = Lattice()
        lat.add(fm)
        path = _tmp_path()
        try:
            write_tracewin(lat, path)
            with open(path) as f:
                content = f.read()
            assert "; FIELD_MAP" in content
            assert not any(l.startswith("FIELD_MAP ")
                           for l in content.splitlines())
        finally:
            os.unlink(path)


def _write_3d_component(path, x, y, z, values):
    """Minimal 3-D TraceWin component file (x fastest, norm=1)."""
    nx, ny, nz = len(x), len(y), len(z)
    with open(path, "w") as f:
        f.write(f"{nz - 1}  {z[-1]:.6e}\n")
        f.write(f"{nx - 1}  {x[0]:.6e}  {x[-1]:.6e}\n")
        f.write(f"{ny - 1}  {y[0]:.6e}  {y[-1]:.6e}\n")
        f.write("1.000000e+00\n")
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{values[ix, iy, iz]:.6e}\n")


def test_fieldmap3d_round_trip(tmp_path):
    """geom=700 (3-D RF electric) FieldMap3D survives parse→write→parse.

    Self-contained: writes the .edx/.edy/.edz component files into tmp_path
    so the test does not depend on the repo's (untracked) Fields/ data.
    """
    nx, ny, nz = 5, 5, 11
    x = np.linspace(-0.02, 0.02, nx)
    y = np.linspace(-0.02, 0.02, ny)
    z = np.linspace(0.0, 0.1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Ez = 1e6 * np.cos(np.pi * Z / 0.1)
    Ex = -X * 1e5
    Ey = -Y * 1e5
    prefix = str(tmp_path / "cav3d")
    _write_3d_component(prefix + ".edx", x, y, z, Ex)
    _write_3d_component(prefix + ".edy", x, y, z, Ey)
    _write_3d_component(prefix + ".edz", x, y, z, Ez)

    dat = tmp_path / "lat3d.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "DRIFT 50 15 3\n"
        f"FIELD_MAP 700 100 -25 18 1.0 1.1 0 0 {prefix}\n"
        "DRIFT 50 15 3\n"
        "END\n"
    )
    lat1, _ = parse_tracewin(str(dat))
    fm1 = _field_maps(lat1)
    assert len(fm1) == 1 and type(fm1[0]).__name__ == "FieldMap3D"

    out = tmp_path / "lat3d_out.dat"
    write_tracewin(lat1, str(out))
    lat2, meta2 = parse_tracewin(str(out))
    assert not meta2.get("warnings"), meta2.get("warnings")
    fm2 = _field_maps(lat2)
    assert len(fm2) == 1 and type(fm2[0]).__name__ == "FieldMap3D"
    assert fm2[0].phase == pytest.approx(-25.0)
    assert fm2[0].ke == pytest.approx(1.1)
    assert len(lat2) == len(lat1)


def test_fieldmap_spaced_path_round_trip(tmp_path):
    """A field-file path with a space round-trips: the writer quotes it so the
    parser keeps it as one token (the user's workspace had spaced filenames)."""
    d = tmp_path / "my maps"
    d.mkdir()
    (d / "cav.edz").write_text("3\n0.0 0.1\n0.5\n1.0\n0.5\n")
    dat = tmp_path / "lat.dat"
    dat.write_text(
        'TITLE t\nFREQ 352.21\n'
        'FIELD_MAP 100 10 0 20 1 1 0 0 "my maps/cav.edz" 0\nEND\n'
    )
    lat1, m1 = parse_tracewin(str(dat))
    assert len(_field_maps(lat1)) == 1 and not m1.get("warnings")
    out = tmp_path / "out.dat"
    write_tracewin(lat1, str(out))
    assert '"' in out.read_text()          # spaced path emitted quoted
    lat2, m2 = parse_tracewin(str(out))
    assert not m2.get("warnings"), m2.get("warnings")
    assert len(_field_maps(lat2)) == 1
