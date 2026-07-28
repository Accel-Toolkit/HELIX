"""Phase 1+2 core-side tools: element resolution / cursor / gui-context
inert defaults, and the offline manual search (index + markdown fallback)."""
from __future__ import annotations

from linac_gen.assist import tools as T
from linac_gen.assist.tools import TOOLS, WorkContext, _resolve_element


def _lat():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QF1", 50.0, gradient=5.0, aperture=20.0))
    lat.add(Quadrupole("QD1", 50.0, gradient=-5.0, aperture=20.0))
    return lat


# ---- element resolution ----------------------------------------------
def test_resolve_element_by_index_name_substring_ambiguous():
    lat = _lat()
    assert _resolve_element(lat, 0)[0] == 0
    assert _resolve_element(lat, "QF1")[0] == 1          # exact
    assert _resolve_element(lat, "qd")[0] == 2           # unique substring
    idx, el, cand = _resolve_element(lat, "Q")           # ambiguous
    assert idx is None and set(cand) == {"QF1", "QD1"}
    assert _resolve_element(lat, "nope") == (None, None, [])


def test_highlight_tool_reports_position_without_gui():
    ctx = WorkContext(calc_dir=".")
    ctx.lattice = _lat()
    r = TOOLS["highlight_element"].fn(ctx, element="QF1")
    assert r["status"] == "ok"
    d = r["data"]
    assert d["index"] == 1 and d["type"] == "Quadrupole"
    assert d["s_start_m"] == 0.2 and d["s_end_m"] == 0.25
    assert d["highlighted_in_gui"] is False              # no GUI hook


def test_cursor_and_context_refuse_without_gui():
    ctx = WorkContext(calc_dir=".")
    assert TOOLS["set_cursor"].fn(ctx, s_m=1.0)["status"] == "refused"
    assert TOOLS["get_gui_context"].fn(ctx)["status"] == "refused"


# ---- manual search ---------------------------------------------------
def test_search_manual_real_index_hits_ncells():
    ctx = WorkContext(calc_dir=".")
    r = TOOLS["search_manual"].fn(ctx, query="NCELLS synchronism", k=5)
    assert r["status"] == "ok"
    titles = " | ".join(h["title"] + h["location"]
                        for h in r["data"]["results"])
    assert "ncells" in titles.lower()


def test_search_manual_markdown_fallback(monkeypatch, tmp_path):
    man = tmp_path / "docs" / "manual" / "03_elements"
    man.mkdir(parents=True)
    (man / "99_widget.md").write_text(
        "# Frobnicator element\n\nThe frobnicator inverts the polarity.\n")
    monkeypatch.setattr(T, "_manual_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(T, "_MANUAL_CACHE", {})
    ctx = WorkContext(calc_dir=".")
    r = TOOLS["search_manual"].fn(ctx, query="frobnicator polarity")
    assert r["status"] == "ok"
    assert r["data"]["results"][0]["title"] == "Frobnicator element"


def test_search_manual_no_manual_is_error(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_manual_roots", lambda: [str(tmp_path)])
    monkeypatch.setattr(T, "_MANUAL_CACHE", {})
    ctx = WorkContext(calc_dir=".")
    assert TOOLS["search_manual"].fn(ctx, query="x y")["status"] == "error"
