"""FIELD_MAP_PATH directive resolves subsequent FIELD_MAP filenames."""
import os
from linac_gen.io.tracewin_parser import parse_tracewin


def _minimal_canonical_1d_edz(path: str, Nz: int = 3, Zmax_m: float = 0.01) -> None:
    """Write a TraceWin-canonical 1-D .edz with ``Nz+1`` values."""
    # Format: Nz, z_start_cm z_end_cm, then Nz values
    with open(path, "w") as f:
        f.write(f"{Nz}\n")
        f.write(f"0.0 {Zmax_m * 100:.6e}\n")  # cm
        for v in (0.0, 0.5, 1.0, 0.5):
            f.write(f"{v:.6e}\n")


def test_absolute_field_map_path(tmp_path):
    sub = tmp_path / "maps"
    sub.mkdir()
    _minimal_canonical_1d_edz(str(sub / "cav.edz"))
    dat = tmp_path / "lattice.dat"
    dat.write_text(
        "TITLE t\nFREQ 352.21\n"
        f"FIELD_MAP_PATH {sub}\n"
        "FIELD_MAP 100 10 0 20 1 1 0 0 cav.edz 0\nEND\n"
    )
    lat, meta = parse_tracewin(str(dat))
    assert any(type(e).__name__ == "FieldMap" for e in lat.elements), meta.get("warnings", [])
    assert meta.get("warnings", []) == [], meta["warnings"]


def test_relative_field_map_path(tmp_path):
    """Relative paths resolve against the .dat's directory."""
    sub = tmp_path / "maps"
    sub.mkdir()
    _minimal_canonical_1d_edz(str(sub / "cav.edz"))
    dat = tmp_path / "lattice.dat"
    dat.write_text(
        "TITLE t\nFREQ 352.21\n"
        "FIELD_MAP_PATH maps\n"
        "FIELD_MAP 100 10 0 20 1 1 0 0 cav.edz 0\nEND\n"
    )
    lat, meta = parse_tracewin(str(dat))
    assert meta.get("warnings", []) == [], meta["warnings"]
    assert any(type(e).__name__ == "FieldMap" for e in lat.elements)
