"""FIELD_MAP filenames with quoted spaced paths must parse as one token."""

from linac_gen.io.tracewin_parser import parse_tracewin


def test_quoted_filename_with_space(tmp_path):
    # Make a subdir with a space in its name and put a minimal canonical
    # 1-D .edz (nz=3) inside.
    dir_with_space = tmp_path / "my maps"
    dir_with_space.mkdir()
    edz = dir_with_space / "cav.edz"
    # Legacy Linac_Gen layout: N_pts / zmin zmax [cm] / N_pts values
    edz.write_text("3\n0.0 0.1\n0.5\n1.0\n0.5\n")

    dat = tmp_path / "lattice.dat"
    # geom=100 → RF_E 1-D → reader looks for .edz (matches filename).
    # Ka=0 means no .ouv file required.
    dat.write_text(
        'TITLE t\nFREQ 352.21\n'
        'FIELD_MAP 100 10 0 20 1 1 0 0 "my maps/cav.edz" 0\nEND\n'
    )
    lat, meta = parse_tracewin(str(dat))

    # The FIELD_MAP parsed and the filename token was kept intact, so the
    # resulting element loaded (no file-not-found warning).
    assert any(type(e).__name__ == "FieldMap" for e in lat.elements), (
        f"FIELD_MAP did not produce a FieldMap element; warnings: "
        f"{meta.get('warnings', [])}"
    )
    # No warnings about missing files or parse errors.
    assert meta.get("warnings", []) == [], meta["warnings"]


def test_apostrophe_in_title_does_not_crash(tmp_path):
    """TITLE containing an apostrophe must not crash with 'No closing quotation'."""
    dat = tmp_path / "lattice.dat"
    dat.write_text("TITLE Fred's linac\nFREQ 352.21\nEND\n")

    lat, meta = parse_tracewin(str(dat))

    # Must parse without raising ValueError.
    # Check that no parser-error entries appear in warnings.
    parser_errors = [w for w in meta.get("warnings", []) if "closing quotation" in w.lower()]
    assert parser_errors == [], (
        f"Parser crashed on apostrophe: {parser_errors}"
    )


def test_unquoted_backslash_path_survives(tmp_path):
    """Unquoted Windows-style paths with backslashes must be preserved literally."""
    dat = tmp_path / "lattice.dat"
    # Write a FIELD_MAP line with a literal backslash in the path.
    # We use raw string to ensure the backslash is preserved when written to file.
    dat.write_text("TITLE t\nFREQ 352.21\nFIELD_MAP 100 10 0 20 1 1 0 0 maps\\mycav.edz 0\nEND\n")

    lat, meta = parse_tracewin(str(dat))

    # The file doesn't exist, so we expect a "file missing" / "not found" warning.
    # But crucially, the warning message must contain the backslash in the path.
    file_not_found_warnings = [
        w for w in meta.get("warnings", [])
        if "file missing" in w.lower() or "file not found" in w.lower() or "not found" in w.lower()
    ]
    assert file_not_found_warnings, "Expected 'file missing' warning for missing FIELD_MAP"

    # Check that at least one warning contains the backslash path.
    # The backslash must NOT be stripped (maps\mycav not mapsmycav).
    assert any("maps\\mycav" in w or r"maps\mycav" in w for w in file_not_found_warnings), (
        f"Backslash in path was stripped. Warnings: {file_not_found_warnings}"
    )
