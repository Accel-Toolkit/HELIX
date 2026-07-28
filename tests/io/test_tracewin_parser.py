"""Tests for TraceWin .dat parser (Task 7.3)."""
import os
import pytest
import tempfile

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
from linac_gen.elements.field_map import FieldMap

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ── simple_fodo.dat parsing ─────────────────────────────────────────────────

class TestSimpleFodoParsing:
    def test_returns_lattice_and_metadata(self):
        result = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        assert isinstance(result, tuple) and len(result) == 2
        lattice, metadata = result
        assert isinstance(lattice, Lattice)
        assert isinstance(metadata, dict)

    def test_element_count(self):
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        # FREQ materializes as a Freq command element (machine-clock switch)
        assert len(lattice) == 6

    def test_element_types(self):
        from linac_gen.elements.lattice_commands import Freq
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        types = [type(e) for e in lattice.elements]
        assert types == [Freq, Drift, Quadrupole, Drift, Quadrupole, Drift]

    def test_drift_length(self):
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        drifts = [e for e in lattice.elements if isinstance(e, Drift)]
        assert drifts[0].length == pytest.approx(100.0)
        assert drifts[1].length == pytest.approx(200.0)
        assert drifts[2].length == pytest.approx(100.0)

    def test_drift_aperture(self):
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        drifts = [e for e in lattice.elements if isinstance(e, Drift)]
        for d in drifts:
            assert d.aperture == pytest.approx(20.0)

    def test_drift_aperture_y(self):
        """4th positional in TraceWin DRIFT is aperture_y (mm)."""
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        drifts = [e for e in lattice.elements if isinstance(e, Drift)]
        for d in drifts:
            assert d.aperture_y == pytest.approx(5.0)

    def test_quad_gradient(self):
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        quads = [e for e in lattice.elements if isinstance(e, Quadrupole)]
        assert quads[0].gradient == pytest.approx(5.2)
        assert quads[1].gradient == pytest.approx(-5.2)

    def test_quad_length(self):
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        quads = [e for e in lattice.elements if isinstance(e, Quadrupole)]
        for q in quads:
            assert q.length == pytest.approx(50.0)

    def test_quad_skew_angle(self):
        """4th positional in TraceWin QUAD is skew_angle (deg)."""
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        quads = [e for e in lattice.elements if isinstance(e, Quadrupole)]
        for q in quads:
            assert q.skew_angle == pytest.approx(10.0)

    def test_title_in_metadata(self):
        _, metadata = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        assert metadata["title"] == "Test FODO Cell"

    def test_no_warnings(self):
        _, metadata = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        assert metadata["warnings"] == []


# ── Element naming ───────────────────────────────────────────────────────────

class TestElementNaming:
    def test_drift_names(self):
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        drifts = [e for e in lattice.elements if isinstance(e, Drift)]
        assert drifts[0].name == "DRIFT_001"
        assert drifts[1].name == "DRIFT_002"
        assert drifts[2].name == "DRIFT_003"

    def test_quad_names(self):
        lattice, _ = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        quads = [e for e in lattice.elements if isinstance(e, Quadrupole)]
        assert quads[0].name == "QUAD_001"
        assert quads[1].name == "QUAD_002"


# ── FREQ card is stateful ────────────────────────────────────────────────────

class TestFreqCard:
    def _make_gap_file(self, content):
        """Write a temp .dat file and return the path."""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write(content)
        f.flush()
        f.close()
        return f.name

    def test_gap_uses_freq(self):
        dat = self._make_gap_file(
            "FREQ 704.42\nGAP 1.5 -30.0 20.0\nEND\n"
        )
        try:
            lattice, _ = parse_tracewin(dat)
            gap = next(e for e in lattice.elements if isinstance(e, RFGap))
            assert gap.frequency == pytest.approx(704.42)
        finally:
            os.unlink(dat)

    def test_gap_before_freq_uses_default_and_warns(self):
        dat = self._make_gap_file("GAP 1.5 -30.0 20.0\nEND\n")
        try:
            lattice, metadata = parse_tracewin(dat)
            gap = next(e for e in lattice.elements if isinstance(e, RFGap))
            assert gap.frequency == pytest.approx(352.21)
            assert any("GAP before FREQ" in w for w in metadata["warnings"])
        finally:
            os.unlink(dat)

    def test_freq_updates_subsequent_gaps(self):
        dat = self._make_gap_file(
            "FREQ 100.0\nGAP 1.0 0.0\nFREQ 200.0\nGAP 1.0 0.0\nEND\n"
        )
        try:
            lattice, _ = parse_tracewin(dat)
            gaps = [e for e in lattice.elements if isinstance(e, RFGap)]
            assert gaps[0].frequency == pytest.approx(100.0)
            assert gaps[1].frequency == pytest.approx(200.0)
        finally:
            os.unlink(dat)


# ── strict vs non-strict mode ────────────────────────────────────────────────

class TestStrictMode:
    def _make_file_with_unknown(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("DRIFT 100.0 20.0\nUNKNOWN_CARD foo bar\nEND\n")
        f.flush()
        f.close()
        return f.name

    def test_non_strict_adds_warning(self):
        fpath = self._make_file_with_unknown()
        try:
            lattice, metadata = parse_tracewin(fpath, strict=False)
            assert any("UNKNOWN_CARD" in w for w in metadata["warnings"])
            # Known element still parsed
            assert len(lattice) == 1
        finally:
            os.unlink(fpath)

    def test_strict_raises_value_error(self):
        fpath = self._make_file_with_unknown()
        try:
            with pytest.raises(ValueError, match="unsupported card"):
                parse_tracewin(fpath, strict=True)
        finally:
            os.unlink(fpath)


# ── Comment lines ────────────────────────────────────────────────────────────

class TestCommentHandling:
    def test_comment_lines_skipped(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("; This is a full comment line\nDRIFT 100.0 ; inline comment\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, metadata = parse_tracewin(f.name)
            assert len(lattice) == 1
            assert isinstance(lattice.elements[0], Drift)
            assert lattice.elements[0].length == pytest.approx(100.0)
        finally:
            os.unlink(f.name)

    def test_inline_comment_stripped(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("QUAD 50.0 3.0 20.0 5 ; focusing quad\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            q = lattice.elements[0]
            assert isinstance(q, Quadrupole)
            assert q.gradient == pytest.approx(3.0)
        finally:
            os.unlink(f.name)


# ── Empty file ───────────────────────────────────────────────────────────────

class TestEmptyFile:
    def test_empty_file_returns_empty_lattice(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("")
        f.flush()
        f.close()
        try:
            lattice, metadata = parse_tracewin(f.name)
            assert len(lattice) == 0
            assert metadata["warnings"] == []
        finally:
            os.unlink(f.name)

    def test_only_comments_returns_empty_lattice(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("; comment 1\n; comment 2\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            assert len(lattice) == 0
        finally:
            os.unlink(f.name)

    def test_end_card_stops_parsing(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("DRIFT 100.0\nEND\nDRIFT 200.0\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            assert len(lattice) == 1
        finally:
            os.unlink(f.name)


# ── All element types ────────────────────────────────────────────────────────

class TestAllElementTypes:
    def test_solenoid_parsed(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("SOLENOID 200.0 0.5 20.0 10\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            sol = lattice.elements[0]
            assert isinstance(sol, Solenoid)
            assert sol.length == pytest.approx(200.0)
            assert sol.field == pytest.approx(0.5)
            assert sol.aperture == pytest.approx(20.0)
            # TraceWin SOLENOID has no n_steps positional; default kept.
            assert sol.n_steps == 5
            assert sol.name == "SOL_001"
        finally:
            os.unlink(f.name)

    def test_steerer_parsed(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("THIN_STEERING 0.001 -0.002\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            steer = lattice.elements[0]
            assert isinstance(steer, Steerer)
            assert steer.bx_l == pytest.approx(0.001)
            assert steer.by_l == pytest.approx(-0.002)
            assert steer.name == "STEER_001"
        finally:
            os.unlink(f.name)

    def test_steerer_magnetic_default_flag(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("THIN_STEERING 0.001 -0.002 20\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            assert lattice.elements[0].elec is False
        finally:
            os.unlink(f.name)

    def test_steerer_electric_flag_parsed(self):
        """THIN_STEERING elec=1: the flag reaches the element (operands
        are volts, same-plane electric kick) — it used to be silently
        dropped and tracked with the magnetic 1/Brho law."""
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("THIN_STEERING 500 -200 20 1\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            steer = lattice.elements[0]
            assert isinstance(steer, Steerer)
            assert steer.elec is True
            assert steer.bx_l == pytest.approx(500.0)
            assert steer.by_l == pytest.approx(-200.0)
        finally:
            os.unlink(f.name)

    def test_bend_parsed(self):
        # TraceWin BEND schema: angle(deg) rho(mm) field_index aperture(mm) hv
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("BEND 45.0 500.0 0.5 20.0 0\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            dipole = lattice.elements[0]
            assert isinstance(dipole, Dipole)
            assert dipole.angle == pytest.approx(45.0)
            assert dipole.rho == pytest.approx(500.0)
            assert dipole.field_index == pytest.approx(0.5)
            assert dipole.aperture == pytest.approx(20.0)
            assert dipole.hv == 0
            # Edge angles not present on BEND card itself in TraceWin
            assert dipole.e1 == 0.0
            assert dipole.e2 == 0.0
            assert dipole.name == "BEND_001"
        finally:
            os.unlink(f.name)

    def test_aperture_circular_parsed(self):
        # TraceWin order: dx dy n  (n=1 -> circular)
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("APERTURE 20.0 0 1\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            ap = lattice.elements[0]
            assert isinstance(ap, Aperture)
            assert ap.aperture_type == 1
            assert ap.dx == pytest.approx(20.0)
            assert ap.name == "APER_001"
        finally:
            os.unlink(f.name)

    def test_aperture_rectangular_parsed(self):
        # TraceWin order: dx dy n  (n=0 -> rectangular)
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("APERTURE 15.0 10.0 0\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            ap = lattice.elements[0]
            assert isinstance(ap, Aperture)
            assert ap.aperture_type == 0
            assert ap.dx == pytest.approx(15.0)
            assert ap.dy == pytest.approx(10.0)
        finally:
            os.unlink(f.name)

    def test_marker_parsed(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("MARKER\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            m = lattice.elements[0]
            assert isinstance(m, Marker)
            assert not m.snapshot
            assert m.name == "MARK_001"
        finally:
            os.unlink(f.name)

    def test_diag_phase_creates_snapshot_marker(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("DIAG_PHASE 1\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            m = lattice.elements[0]
            assert isinstance(m, Marker)
            assert m.snapshot is True
        finally:
            os.unlink(f.name)

    def test_diag_size_creates_non_snapshot_marker(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("DIAG_SIZE 1\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            m = lattice.elements[0]
            assert isinstance(m, Marker)
            assert not m.snapshot
        finally:
            os.unlink(f.name)

    def test_space_charge_comp_parsed(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("SPACE_CHARGE_COMP 0.9\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            sc = lattice.elements[0]
            assert isinstance(sc, SpaceChargeComp)
            assert sc.factor == pytest.approx(0.9)
            assert sc.name == "SCCOMP_001"
        finally:
            os.unlink(f.name)

    def test_gap_parsed(self):
        # TraceWin GAP positionals: E0TL (V), phase (deg), aperture (mm), p_flag (int).
        # E0TL in V is divided by 1e6 to give voltage in MV (our internal unit).
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("FREQ 352.21\nGAP 1.5e6 -30.0 20.0 1\nEND\n")
        f.flush()
        f.close()
        try:
            lattice, _ = parse_tracewin(f.name)
            gap = next(e for e in lattice.elements if isinstance(e, RFGap))
            assert gap.voltage == pytest.approx(1.5)        # MV (E0TL / 1e6)
            assert gap.phase == pytest.approx(-30.0)
            assert gap.frequency == pytest.approx(352.21)
            assert gap.ttf == pytest.approx(1.0)            # TraceWin embeds TTF in E0TL
            assert gap.aperture == pytest.approx(20.0)
            assert gap.p_flag == 1
            assert gap.name == "GAP_001"
        finally:
            os.unlink(f.name)


# ── Field map parsing ────────────────────────────────────────────────────────

class TestFieldMapParsing:
    def test_field_map_found_creates_element(self):
        lattice, metadata = parse_tracewin(
            os.path.join(FIXTURE_DIR, "lattice_with_fieldmap.dat"),
            base_dir=FIXTURE_DIR,
        )
        fmaps = [e for e in lattice.elements if isinstance(e, FieldMap)]
        assert len(fmaps) == 1

    def test_field_map_missing_file_adds_warning(self):
        # TraceWin FIELD_MAP: geom, L, phase, R, kb, ke, ki, ka, filename [, p_flag]
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("FREQ 352.21\n"
                "FIELD_MAP 100 100 -30.0 20.0 1.0 1.0 0.0 0 nonexistent.edz\n"
                "END\n")
        f.flush()
        f.close()
        try:
            lattice, metadata = parse_tracewin(f.name)
            assert any("missing" in w.lower() or "not found" in w.lower()
                       for w in metadata["warnings"])
            # No FieldMap added for missing file (FREQ marker still present)
            from linac_gen.elements.field_map import FieldMap as _FM
            assert not any(isinstance(e, _FM) for e in lattice.elements)
        finally:
            os.unlink(f.name)

    def test_field_map_phase_set(self):
        lattice, _ = parse_tracewin(
            os.path.join(FIXTURE_DIR, "lattice_with_fieldmap.dat"),
            base_dir=FIXTURE_DIR,
        )
        fmaps = [e for e in lattice.elements if isinstance(e, FieldMap)]
        assert fmaps[0].phase == pytest.approx(-30.0)

    def test_field_map_frequency_set(self):
        lattice, _ = parse_tracewin(
            os.path.join(FIXTURE_DIR, "lattice_with_fieldmap.dat"),
            base_dir=FIXTURE_DIR,
        )
        fmaps = [e for e in lattice.elements if isinstance(e, FieldMap)]
        assert fmaps[0].frequency == pytest.approx(352.21)


# ── lattice_with_fieldmap.dat full parse ────────────────────────────────────

class TestLatticeWithFieldmap:
    def test_correct_number_of_elements(self):
        lattice, _ = parse_tracewin(
            os.path.join(FIXTURE_DIR, "lattice_with_fieldmap.dat"),
            base_dir=FIXTURE_DIR,
        )
        # FREQ->Freq, DRIFT, FIELD_MAP, DRIFT, QUAD, DIAG_PHASE->Marker,
        # MARKER, SC_COMP, THIN_STEER, BEND, APERTURE
        assert len(lattice) == 11

    def test_element_sequence(self):
        from linac_gen.elements.lattice_commands import Freq
        lattice, _ = parse_tracewin(
            os.path.join(FIXTURE_DIR, "lattice_with_fieldmap.dat"),
            base_dir=FIXTURE_DIR,
        )
        expected_types = [
            Freq, Drift, FieldMap, Drift, Quadrupole,
            Marker, Marker, SpaceChargeComp, Steerer, Dipole, Aperture,
        ]
        actual_types = [type(e) for e in lattice.elements]
        assert actual_types == expected_types


# ── base_dir resolution ──────────────────────────────────────────────────────

class TestBaseDirResolution:
    def test_base_dir_defaults_to_file_directory(self):
        """If base_dir not given, field map files resolve relative to the .dat file."""
        # This should find test_1d.edz in the same directory as lattice_with_fieldmap.dat
        lattice, metadata = parse_tracewin(
            os.path.join(FIXTURE_DIR, "lattice_with_fieldmap.dat")
        )
        fmaps = [e for e in lattice.elements if isinstance(e, FieldMap)]
        assert len(fmaps) == 1
        assert not any("not found" in w for w in metadata["warnings"])


# ── Metadata structure ───────────────────────────────────────────────────────

class TestMetadata:
    def test_metadata_has_warnings_key(self):
        lattice, metadata = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        assert "warnings" in metadata

    def test_metadata_has_title_key(self):
        lattice, metadata = parse_tracewin(os.path.join(FIXTURE_DIR, "simple_fodo.dat"))
        assert "title" in metadata

    def test_title_empty_when_not_present(self):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False)
        f.write("DRIFT 100.0\nEND\n")
        f.flush()
        f.close()
        try:
            _, metadata = parse_tracewin(f.name)
            assert metadata["title"] == ""
        finally:
            os.unlink(f.name)


@pytest.mark.parametrize("line,expected", [
    ("DRIFT 50 30", dict(length=50.0, aperture=30.0, aperture_y=None,
                         x_shift=0.0, y_shift=0.0)),
    ("DRIFT 50 30 20", dict(length=50.0, aperture=30.0, aperture_y=20.0,
                            x_shift=0.0, y_shift=0.0)),
    ("DRIFT 100 20 10 0.5 -0.3", dict(length=100.0, aperture=20.0,
                                       aperture_y=10.0,
                                       x_shift=0.5, y_shift=-0.3)),
])
def test_drift_parsing(tmp_path, line, expected):
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text(f"FREQ 352.21\n{line}\nEND\n")
    lat, _ = parse_tracewin(str(dat))
    d = next(e for e in lat.elements if isinstance(e, Drift))
    assert d.length == expected["length"]
    assert d.aperture == expected["aperture"]
    assert d.aperture_y == expected["aperture_y"]
    assert d.x_shift == expected["x_shift"]
    assert d.y_shift == expected["y_shift"]


def test_quad_parses_skew_angle(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text("FREQ 352.21\nQUAD 50 5 20 30\nEND\n")
    lat, _ = parse_tracewin(str(dat))
    q = next(e for e in lat.elements if isinstance(e, Quadrupole))
    assert q.length == 50.0
    assert q.gradient == 5.0
    assert q.aperture == 20.0
    assert q.skew_angle == 30.0


@pytest.mark.parametrize("line,expected", [
    ("APERTURE 10 5 0", dict(dx=10.0, dy=5.0, aperture_type=0)),
    ("APERTURE 8 0 1", dict(dx=8.0, dy=8.0, aperture_type=1)),    # dy defaults to dx when <=0
])
def test_aperture_card(tmp_path, line, expected):
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text(f"FREQ 352.21\n{line}\nEND\n")
    lat, _ = parse_tracewin(str(dat))
    ap = next(e for e in lat.elements if isinstance(e, Aperture))
    for k, v in expected.items():
        assert getattr(ap, k) == v


def test_partran_step_updates_lattice_step_config(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "p.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "PARTRAN_STEP 200 80\n"
        "DRIFT 100 30\n"
        "END\n"
    )
    lat, meta = parse_tracewin(str(dat))
    assert lat.step_config.integration_steps_per_metre == 200.0
    assert lat.step_config.sc_steps_per_metre == 80.0


@pytest.mark.parametrize("keyword", ["STEERER", "THIN_STEERING"])
def test_steerer_card_parses(tmp_path, keyword):
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text(f"FREQ 352.21\n{keyword} 0.01 -0.02 20 0\nEND\n")
    lat, _ = parse_tracewin(str(dat))
    s = next(e for e in lat.elements if isinstance(e, Steerer))
    assert s.bx_l == 0.01
    assert s.by_l == -0.02


def test_unknown_card_logs_warning(tmp_path):
    """Parser should skip unsupported cards with a warning, not raise."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "DRIFT 50 30\n"
        "FOOBAR 1 2 3\n"          # NCELLS is now a supported card — use a
        "DRIFT 50 30\n"          # genuinely-unknown keyword to test skipping
        "END\n"
    )
    lat, meta = parse_tracewin(str(dat))
    # Both drifts present; FOOBAR skipped with warning.  (FREQ emits a
    # zero-length Marker carrying ref.frequency for the section boundary.)
    drifts = [e for e in lat.elements if isinstance(e, Drift)]
    assert len(drifts) == 2
    assert any("FOOBAR" in w for w in meta["warnings"])


def test_strict_mode_still_raises_on_unknown_card(tmp_path):
    """When the parser runs in strict mode, unknown cards are fatal."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "DRIFT 50 30\n"
        "FOOBAR 1 2 3\n"          # NCELLS is now supported — use an unknown card
        "END\n"
    )
    with pytest.raises(ValueError, match="FOOBAR"):
        parse_tracewin(str(dat), strict=True)


def test_glued_colon_labeled_quad(tmp_path):
    """``LABEL: QUAD …`` (colon glued to label) must drop the label and
    keep the QUAD card.  TraceWin treats this as a labeled QUAD; the
    parser previously silently dropped the line because the elif branch
    that detects the glued form was a stub.  Used in the PIP-II PDR
    lattice for ``skew: QUAD 200 0 22 …`` placeholders.
    """
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text(
        "FREQ 650\n"
        "skew: QUAD 200 0 22 0 0 0 0 0\n"
        "END\n"
    )
    lat, meta = parse_tracewin(str(dat))
    quads = [e for e in lat.elements if isinstance(e, Quadrupole)]
    assert len(quads) == 1, (
        f"expected 1 QUAD, got {len(quads)}; warnings={meta.get('warnings')}"
    )
    assert quads[0].length == pytest.approx(200.0)
    assert quads[0].gradient == pytest.approx(0.0)


def test_glued_colon_numbered_labels(tmp_path):
    """``Q01: QUAD …`` — labels containing DIGITS (the standard TraceWin
    naming: Q01, D02BPM, M11) must drop the label and keep the card.

    Regression: the glued-colon branch used to require the label to have
    NO digit, so every numbered element in a real deck was mis-read as an
    unsupported card named after its own label (fnalscl.dat: 80 dropped
    QUAD/STEERING/DIAG elements).  Now gated on 'starts with a letter'.
    """
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.elements.steerer import Steerer
    dat = tmp_path / "x.dat"
    dat.write_text(
        "FREQ 804.96\n"
        "Q01: QUAD 85.3 -8.09 20 0 0 0 0 0\n"
        "D01BPM: DIAG_POSITION 12 0.0035 -0.0007\n"
        "D01T: THIN_STEERING 4.56E-5 0.0 20 0\n"
        "END\n"
    )
    lat, meta = parse_tracewin(str(dat))
    kinds = {type(e).__name__ for e in lat.elements}
    assert "Quadrupole" in kinds, meta.get("warnings")
    assert "Steerer" in kinds
    # the card keyword, not the label, was used (no 'Q01'/'D01BPM' warning)
    assert not any("Q01" in w or "D01BPM" in w or "D01T" in w
                   for w in meta.get("warnings", []))
    q = next(e for e in lat.elements if isinstance(e, Quadrupole))
    assert q.length == pytest.approx(85.3)
    # the DIAG_POSITION label + operands are captured on the BPM marker
    m = next(e for e in lat.elements if getattr(e, "is_bpm", False))
    assert m.name == "D01BPM"
    assert m.diag_family == 12
    assert m.x_target_mm == pytest.approx(0.0035)
    assert m.y_target_mm == pytest.approx(-0.0007)
    assert m.accuracy_mm == 1.0


def test_glued_colon_standalone_marker_still_works(tmp_path):
    """``BPM:`` (standalone glued-colon marker) must continue to be
    recognised — the fix for labeled glued-colon QUADs must not break
    standalone glued-colon markers.
    """
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "x.dat"
    dat.write_text(
        "FREQ 352.21\n"
        "DRIFT 50 30\n"
        "BPM:\n"
        "DRIFT 50 30\n"
        "END\n"
    )
    lat, _ = parse_tracewin(str(dat))
    markers = [e for e in lat.elements if isinstance(e, Marker)]
    # At least one BPM marker should be present (the parser may also emit
    # a FREQ marker; both are Markers — so just check that one of them
    # carries a name suggesting BPM).
    assert any("BPM" in (getattr(m, "name", "") or "").upper() for m in markers)
