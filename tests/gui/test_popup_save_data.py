"""Round-trip tests for the Results-tab popup data exporter.

The exporter lives on the shared ``_PopupPlot`` base class; every
concrete popup (RMS, emittance, phase space, …) inherits the same
``_collect_panels`` walk.  These tests build a stripped-down popup with
a known curve and a known image and confirm CSV / NPZ / JSON round-trip
identically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest

from PyQt6.QtWidgets import QVBoxLayout

from linac_gen_gui.interphase.tabs.results_tab import (
    _Curve, _Image, _Panel, _PopupPlot, _drop_anonymous_mirrors,
)


def _make_popup_with_one_line_plot(qapp):
    pop = _PopupPlot(parent=None, title="TEST line", size=(400, 300))
    layout = QVBoxLayout(pop)
    pw = pg.PlotWidget()
    pw.getAxis("left").setLabel("y_quantity", units="mm")
    pw.getAxis("bottom").setLabel("s", units="mm")
    pw.plot([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0], name="square")
    layout.addWidget(pw)
    return pop


def test_collect_panels_line_plot(qapp):
    pop = _make_popup_with_one_line_plot(qapp)
    panels = pop._collect_panels()
    assert len(panels) == 1
    pn = panels[0]
    assert pn.curves and pn.curves[0].name == "square"
    np.testing.assert_array_equal(pn.curves[0].x, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(pn.curves[0].y, [0.0, 1.0, 4.0, 9.0])
    assert pn.images == []


def test_csv_roundtrip(qapp, tmp_path: Path):
    pop = _make_popup_with_one_line_plot(qapp)
    out = tmp_path / "data.csv"
    pop._write_csv(out, pop._collect_panels())
    text = out.read_text(encoding="utf-8").strip().splitlines()
    # Header + 4 data rows.
    assert len(text) == 5
    header = [h.strip() for h in text[0].split(",")]
    # Column name carries the curve name (possibly panel-prefixed in
    # wide-table mode: "plot_1:square").
    assert any("square" in h for h in header)
    rows = np.loadtxt(out, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(rows[:, 0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(rows[:, 1], [0.0, 1.0, 4.0, 9.0])


def test_npz_roundtrip(qapp, tmp_path: Path):
    pop = _make_popup_with_one_line_plot(qapp)
    out = tmp_path / "data.npz"
    pop._write_npz(out, pop._collect_panels())
    loaded = np.load(out)
    keys = list(loaded.keys())
    x_keys = [k for k in keys if k.endswith("_x")]
    y_keys = [k for k in keys if k.endswith("_y")]
    assert x_keys and y_keys
    np.testing.assert_array_equal(loaded[x_keys[0]], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(loaded[y_keys[0]], [0.0, 1.0, 4.0, 9.0])


def test_json_roundtrip(qapp, tmp_path: Path):
    pop = _make_popup_with_one_line_plot(qapp)
    out = tmp_path / "data.json"
    pop._write_json(out, pop._collect_panels())
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc) == 1
    panel = next(iter(doc.values()))
    assert panel["curves"][0]["name"] == "square"
    assert panel["curves"][0]["y"] == [0.0, 1.0, 4.0, 9.0]


def test_anonymous_mirror_dropped():
    s = np.linspace(0.0, 1.0, 5)
    y = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    curves = [
        _Curve(name="σ_x", x=s, y=y),
        _Curve(name="",    x=s, y=-y),    # anonymous mirror
        _Curve(name="σ_y", x=s, y=y * 2), # different magnitude → keep its name
    ]
    kept = _drop_anonymous_mirrors(curves)
    names = [c.name for c in kept]
    assert "σ_x" in names and "σ_y" in names
    assert "" not in names
    assert len(kept) == 2


def test_image_export_via_npz(qapp, tmp_path: Path):
    pop = _PopupPlot(parent=None, title="TEST image", size=(400, 300))
    QVBoxLayout(pop)
    pw = pg.PlotWidget()
    pw.getAxis("left").setLabel("density")
    img = pg.ImageItem(np.arange(12, dtype=float).reshape(3, 4))
    pw.getPlotItem().addItem(img)
    pop.layout().addWidget(pw)
    out = tmp_path / "data.npz"
    pop._write_npz(out, pop._collect_panels())
    loaded = np.load(out)
    image_keys = [k for k in loaded.keys() if "image" in k
                   and "extent" not in k]
    assert image_keys
    arr = loaded[image_keys[0]]
    np.testing.assert_array_equal(arr, np.arange(12).reshape(3, 4))


def test_save_with_no_data_is_safe(qapp):
    pop = _PopupPlot(parent=None, title="empty", size=(300, 200))
    QVBoxLayout(pop)
    panels = pop._collect_panels()
    assert panels == []


def test_png_export(qapp, tmp_path: Path):
    pop = _make_popup_with_one_line_plot(qapp)
    pop.show()  # PNG capture needs a real (offscreen) widget
    qapp.processEvents()
    out = tmp_path / "snap.png"
    pop._write_raster(out, ".png")
    pop.hide()
    assert out.exists() and out.stat().st_size > 0
    head = out.read_bytes()[:8]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"


def test_svg_export(qapp, tmp_path: Path):
    try:
        from PyQt6.QtSvg import QSvgGenerator  # noqa: F401
    except Exception:
        pytest.skip("PyQt6.QtSvg not available")
    pop = _make_popup_with_one_line_plot(qapp)
    pop.show()
    qapp.processEvents()
    out = tmp_path / "snap.svg"
    pop._write_svg(out)
    pop.hide()
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8", errors="ignore")[:200]
    assert "<svg" in text


def test_pdf_export(qapp, tmp_path: Path):
    try:
        from PyQt6.QtPrintSupport import QPrinter  # noqa: F401
    except Exception:
        pytest.skip("PyQt6.QtPrintSupport not available")
    pop = _make_popup_with_one_line_plot(qapp)
    pop.show()
    qapp.processEvents()
    out = tmp_path / "snap.pdf"
    pop._write_pdf(out)
    pop.hide()
    assert out.exists() and out.stat().st_size > 0
    head = out.read_bytes()[:5]
    assert head == b"%PDF-"
