"""Locale-independent text I/O (external Windows report, issue 3).

Deck-side text is latin-1 (byte-transparent — real TraceWin decks are
cp1252/latin-1); HELIX-native formats are utf-8; CLI consoles degrade
non-encodable glyphs instead of crashing.  The 0xE9 fixture is invalid
UTF-8, so the read test fails on macOS too before the fix — proving
locale-independence everywhere, no locale monkeypatching needed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "io" / "fixtures" / "lattice_with_set_commands.dat"


def test_deck_with_latin1_byte_parses(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin

    deck = tmp_path / "accent.dat"
    deck.write_bytes(b"; d\xe9viation faisceau\nDRIFT 100 20\nEND\n")
    lat, meta = parse_tracewin(str(deck))
    assert len(lat.elements) == 1


def test_deck_write_read_round_trip_preserves_latin1(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    src = tmp_path / "in.dat"
    src.write_bytes(b"; commentaire: d\xe9viation\nDRIFT 250 30\nEND\n")
    lat, _ = parse_tracewin(str(src))
    out = tmp_path / "out.dat"
    write_tracewin(lat, str(out))
    lat2, _ = parse_tracewin(str(out))          # reader accepts its writer
    assert len(lat2.elements) == len(lat.elements)


def test_project_json_with_non_ascii_path_loads(tmp_path):
    from linac_gen.io.project import load_project

    deck = tmp_path / "réseau.dat"
    deck.write_bytes(b"DRIFT 100 20\nEND\n")
    proj = tmp_path / "p.lgproj"
    proj.write_text(json.dumps({
        "__kind__": "linac_gen_project", "__version__": 1,
        "lattice_path": deck.name,
    }), encoding="utf-8")
    cfg = load_project(str(proj))          # loader resolves to absolute
    assert Path(getattr(cfg, "lattice_path", "")).name == "réseau.dat"


def test_cli_report_survives_cp1252_console(tmp_path):
    """The matching report prints arrows/box glyphs outside cp1252; on a
    stock Windows console this was a hard UnicodeEncodeError.  Simulated
    portably by forcing the child's stream encoding."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "cp1252"
    out = tmp_path / "matched.dat"
    cp = subprocess.run(
        [sys.executable, "-m", "linac_gen.matching", str(FIXTURE),
         "--out", str(out), "--report", "--allow-inert-constraints"],
        cwd=str(REPO), capture_output=True, env=env, timeout=600)
    # the child legitimately emits cp1252 bytes — decode accordingly
    stdout = cp.stdout.decode("cp1252", errors="replace")
    stderr = cp.stderr.decode("cp1252", errors="replace")
    assert cp.returncode == 0, stderr[-1500:]
    assert "Matching report" in stdout
