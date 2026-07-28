# tests/io/test_review_round2_parser.py
"""Review round 2 parser fixes: the negative-geom guard's digit-family
gate (claim 5) and the SET_SIZE k2 per-parse warning (claim 7)."""
import numpy as np
import pytest

from linac_gen.io.tracewin_parser import parse_tracewin


def _w_1d(path, vals, zmax_m):
    lines = [f"{len(vals) - 1} {zmax_m:.6f}", "1.0"]
    lines += [f"{v:.9g}" for v in vals]
    path.write_text("\n".join(lines) + "\n")


def _w_3d(path, Nz, Zmax, Nx, Xmin, Xmax, Ny, Ymin, Ymax, norm, vals):
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax:.6e}\n{Nx} {Xmin:.6e} {Xmax:.6e}\n"
                f"{Ny} {Ymin:.6e} {Ymax:.6e}\n{norm:.6e}\n")
        for iz in range(Nz + 1):
            for iy in range(Ny + 1):
                for ix in range(Nx + 1):
                    f.write(f"{vals[iz, iy, ix]:.6e}\n")


def _parse(tmp_path, text, **kw):
    p = tmp_path / "deck.dat"
    p.write_text(text)
    return parse_tracewin(str(p), **kw)


# ── claim 5: negative geom must only warn for on-axis-only digits ────────
def test_negative_geom_3d_parses_clean(tmp_path):
    """geom=-70 (3-D Cartesian static B): the second-order flag is
    meaningless — HELIX samples the map directly at full fidelity — so
    neither a warning nor a strict refusal is allowed (used to
    strict-refuse valid 3-D decks on the sign alone)."""
    Nz, Nx, Ny = 5, 2, 2
    vals = np.full((Nz + 1, Ny + 1, Nx + 1), 0.1)
    for suf in (".bsx", ".bsy", ".bsz"):
        _w_3d(tmp_path / f"sol{suf}", Nz, 0.1, Nx, -0.01, 0.01,
              Ny, -0.01, 0.01, 1.0, vals)
    deck = "FIELD_MAP -70 100 0 15 1.0 1.0 0 0 sol\nDRIFT 10 15 0\nEND\n"
    lat, meta = _parse(tmp_path, deck)
    assert type(lat.elements[0]).__name__ == "FieldMap3D"
    assert not any("second-order" in w for w in meta["warnings"]), \
        meta["warnings"]
    lat2, _ = _parse(tmp_path, deck, strict=True)   # must NOT raise
    assert type(lat2.elements[0]).__name__ == "FieldMap3D"


def test_negative_geom_1d_still_warns_and_strict_raises(tmp_path):
    z = np.linspace(0.0, 0.1, 11)
    _w_1d(tmp_path / "sol.bsz", 0.3 * np.sin(np.pi * z / 0.1), 0.1)
    deck = "FIELD_MAP -10 100 0 15 1.0 0 0 0 sol\nDRIFT 10 15 0\nEND\n"
    _, meta = _parse(tmp_path, deck)
    assert any("second-order" in w for w in meta["warnings"]), \
        meta["warnings"]
    with pytest.raises(ValueError, match="second-order"):
        _parse(tmp_path, deck, strict=True)


# ── claim 7: k2 warning is per-parse metadata (GUI sees it every load) ───
def test_set_size_k2_warns_per_parse(tmp_path):
    deck = ("DRIFT 100 15 0\nSET_SIZE 1 3 3 5 1\nDRIFT 100 15 0\nEND\n")
    for _ in range(2):     # every parse, not once per process
        _, meta = _parse(tmp_path, deck)
        assert any("k2" in w and "not modelled" in w
                   for w in meta["warnings"]), meta["warnings"]
    # k2=0 (the TW default) parses clean.
    _, meta0 = _parse(tmp_path,
                      "DRIFT 100 15 0\nSET_SIZE 1 3 3 5 0\nEND\n")
    assert not any("k2" in w for w in meta0["warnings"])
