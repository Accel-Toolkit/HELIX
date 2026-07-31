"""Results tab — launcher of plot pop-ups.

Each button opens a modeless dialog with the corresponding plot
populated from ``AppState.results`` (envelope or multi-particle).
Dialogs stay open so users can compare multiple quantities side-by-side.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QDialog, QFileDialog, QMenu, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QComboBox, QSpinBox,
)

from linac_gen_gui.interphase.app_settings import make_settings
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.panels import kpi_card, kpi_set
from linac_gen_gui.interphase.plots.envelope_plot import EnvelopeTriple


# ---------------------------------------------------------------------------
from linac_gen_gui.interphase.plots.plot_style import (
    style_plot, add_legend, curve_pen, filled_curve,
    density_heatmap, update_density, DensityPanel,
)


def _mk_plot(ylabel: str, ylabel_units: str = "") -> pg.PlotWidget:
    p = pg.PlotWidget()
    style_plot(p, ylabel, ylabel_units)
    add_legend(p)
    return p


class _LoadedResults:
    """In-memory adapter for an HDF5 file saved by :func:`save_results_hdf5`.

    The popups and cards read fields via ``getattr(results, name, default)``.
    We expose every key in the file as an attribute so those lookups
    succeed (and unknown keys fall back to the caller's default, which
    is the existing behaviour for envelope-mode runs that don't record
    every quantity).  A ``source_path`` attr lets the status bar show
    which file is currently loaded.
    """

    def __init__(self, data: dict, source_path):
        for k, v in data.items():
            # 1-D numpy arrays → plain lists, so an imported run's per-step
            # series behave exactly like a live run's.  The GUI reads them
            # with plain truthiness / iteration (e.g. ``if emit_x``), which
            # a numpy array breaks with "ambiguous truth value" — that
            # exception silently aborted the post-import refresh.
            if isinstance(v, np.ndarray) and v.ndim == 1:
                v = v.tolist()
            setattr(self, k, v)
        self.source_path = source_path


@dataclass
class _Curve:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass
class _Image:
    name: str
    data: np.ndarray
    extent: tuple[float, float, float, float] | None = None


@dataclass
class _Panel:
    label: str
    xlabel: str = ""
    ylabel: str = ""
    curves: list[_Curve] = field(default_factory=list)
    images: list[_Image] = field(default_factory=list)


def _axis_label(plot_item: pg.PlotItem, side: str) -> str:
    try:
        ax = plot_item.getAxis(side)
        return (ax.labelText or "").strip()
    except Exception:
        return ""


def _panel_label(pw: pg.PlotItem | pg.PlotWidget, fallback: str = "panel") -> str:
    try:
        item = pw.getPlotItem() if isinstance(pw, pg.PlotWidget) else pw
        title = ""
        if item.titleLabel is not None:
            title = (item.titleLabel.toPlainText() or "").strip()
        if not title:
            title = _axis_label(item, "left")
        if not title:
            title = _axis_label(item, "bottom")
        return title or fallback
    except Exception:
        return fallback


def _curves_from_plotitem(item: pg.PlotItem) -> list[_Curve]:
    out: list[_Curve] = []
    for i, di in enumerate(list(item.listDataItems())):
        try:
            x, y = di.getData()
        except Exception:
            continue
        if x is None or y is None:
            continue
        x = np.asarray(x); y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            continue
        name = (di.opts.get("name") if hasattr(di, "opts") else None) or ""
        if not name:
            name = f"{_axis_label(item, 'left') or 'y'}_{i}"
        out.append(_Curve(name=str(name), x=x, y=y))
    # Bar items aren't returned by listDataItems on older pyqtgraph;
    # walk all PlotItem children for BarGraphItem too.
    for it in list(item.items):
        if isinstance(it, pg.BarGraphItem):
            opts = getattr(it, "opts", {}) or {}
            x = opts.get("x"); h = opts.get("height")
            if x is None or h is None:
                continue
            x = np.asarray(x); h = np.asarray(h)
            if x.size == 0 or h.size == 0:
                continue
            out.append(_Curve(name=str(opts.get("name") or "bars"), x=x, y=h))
    return out


def _images_from_plotitem(item: pg.PlotItem, default_name: str) -> list[_Image]:
    out: list[_Image] = []
    for it in list(item.items):
        if isinstance(it, pg.ImageItem):
            arr = it.image
            if arr is None:
                continue
            data = np.asarray(arr)
            if data.size == 0:
                continue
            extent: tuple[float, float, float, float] | None = None
            try:
                rect = it.boundingRect()
                extent = (float(rect.left()), float(rect.right()),
                          float(rect.top()),  float(rect.bottom()))
            except Exception:
                pass
            out.append(_Image(name=default_name, data=data, extent=extent))
    return out


def _drop_anonymous_mirrors(curves: list[_Curve]) -> list[_Curve]:
    """Strip the unnamed lower curve of a symmetric ±y band.

    EnvelopeTriple draws ±σ as two `PlotDataItem`s on the same plot: an
    upper curve carrying the ``name='σ_x'`` (or similar) plus an unnamed
    lower curve at exactly ``-y``. For data export the mirror is pure
    redundancy, so drop unnamed curves whose `(x, y)` matches a named
    sibling at `(x, -y)` within tight tolerance.
    """
    if not curves:
        return curves
    keep: list[_Curve] = []
    drop_idx: set[int] = set()
    for j, c in enumerate(curves):
        if j in drop_idx:
            continue
        if c.name and not c.name.startswith("y_"):
            # Look for an unnamed mirror partner with same x and y == -c.y.
            for k, d in enumerate(curves):
                if k == j or k in drop_idx:
                    continue
                if d.name and not d.name.startswith("y_"):
                    continue
                if d.x.shape != c.x.shape or d.y.shape != c.y.shape:
                    continue
                if not np.array_equal(d.x, c.x):
                    continue
                if np.allclose(d.y, -c.y, rtol=1e-7, atol=1e-12):
                    drop_idx.add(k)
                    break
    for j, c in enumerate(curves):
        if j not in drop_idx:
            keep.append(c)
    return keep


# Strong references to worker threads that outlived their popup: a live
# QThread whose last Python reference is dropped gets destroyed while
# running, which aborts the whole process.  Pruned when each finishes.
_ZOMBIE_WORKERS: list = []


def _park_zombie(worker) -> None:
    _ZOMBIE_WORKERS.append(worker)

    def _prune() -> None:
        try:
            _ZOMBIE_WORKERS.remove(worker)
        except ValueError:
            pass
    worker.finished.connect(_prune)
    if not worker.isRunning():   # finished between wait() and here
        _prune()


class _PopupPlot(QDialog):
    """Shared plot-popup shell: title bar, close via Esc/Ctrl-W, dark theme.

    Also carries the universal "Save plot data" pathway: every popup
    inherits a Ctrl+S shortcut and a right-click "Save plot data…"
    action that walks any embedded ``pg.PlotWidget`` /
    ``pg.GraphicsLayoutWidget`` / ``QTableWidget`` and writes the data
    to CSV / NPZ / JSON / HDF5.
    """
    def __init__(self, parent=None, title: str = "", size=(900, 560)):
        super().__init__(parent)
        self._raw_title = title
        self.setWindowTitle(f"{title}  —  Ctrl+S to save" if title else
                            "Ctrl+S to save")
        self.resize(*size)
        self.setStyleSheet(
            f"QDialog {{ background:{theme.BG_0}; }}"
        )
        QShortcut(QKeySequence("Ctrl+W"), self, activated=self.close)
        QShortcut(QKeySequence("Esc"),    self, activated=self.close)
        QShortcut(QKeySequence.StandardKey.Save, self,
                  activated=self._save_plot_data)
        # Optional lattice-strip widget — set by attach_lattice_strip when
        # the subclass calls it.  Used by _on_lattice_changed to refresh.
        self._lattice_strip = None

    def closeEvent(self, ev) -> None:                        # noqa: N802
        """Stop any background worker this popup owns before it goes.

        The σ₀ popups spawn a ``_StructWorker`` (a ~30 s lattice walk);
        closing the popup used to leave that thread running headless.
        Cooperative stop + short bounded wait; a straggler is parked in
        a module list so its QThread is never garbage-collected alive.
        """
        w = getattr(self, "_worker", None)
        if w is not None and w.isRunning():
            if hasattr(w, "request_stop"):
                w.request_stop()
            w.requestInterruption()
            if not w.wait(2000):
                _park_zombie(w)
        super().closeEvent(ev)

    # ------------------------------------------------------------------
    # Lattice-impression strip
    # ------------------------------------------------------------------
    def attach_lattice_strip(self, main_plot) -> None:
        """Insert a thin lattice-strip widget above ``main_plot``.

        Idempotent — calling twice is a no-op.  Reads the lattice from
        ``self.parent().state.lattice`` and subscribes to
        ``state.lattice_changed`` so the strip rebuilds when the user
        loads a new lattice.

        If the parent has no ``state`` attribute (e.g. the popup is being
        unit-tested in isolation), this method is a silent no-op.
        """
        if main_plot is None or self._lattice_strip is not None:
            return
        layout = self.layout()
        if not isinstance(layout, QVBoxLayout):
            return
        idx = layout.indexOf(main_plot)
        if idx < 0:
            return

        parent = self.parent()
        state = getattr(parent, "state", None)
        lattice = getattr(state, "lattice", None) if state is not None else None

        from linac_gen_gui.interphase.plots.lattice_strip import (
            make_lattice_strip,
        )
        strip = make_lattice_strip(self, main_plot, lattice)
        layout.insertWidget(idx, strip)
        self._lattice_strip = strip

        # Re-render the strip when the user loads a new lattice.
        if state is not None and hasattr(state, "lattice_changed"):
            try:
                state.lattice_changed.connect(self._on_lattice_changed)
            except Exception:
                pass

    def _on_lattice_changed(self, new_lattice) -> None:
        """Slot for the parent state's ``lattice_changed`` signal."""
        if self._lattice_strip is None:
            return
        try:
            self._lattice_strip.set_lattice(new_lattice)
        except Exception:
            pass

    def set_lattice_strip_visible(self, visible: bool) -> None:
        """Hide/show the lattice strip.  No-op if it was never attached."""
        if self._lattice_strip is not None:
            self._lattice_strip.setVisible(bool(visible))

    # ------------------------------------------------------------------
    # Save-data plumbing
    # ------------------------------------------------------------------
    def _extra_panels(self) -> list[_Panel]:
        """Subclass hook: append panels the universal walk can't reach.

        Default is empty — only popups that retain raw data not visible
        in any ``PlotDataItem``/``ImageItem`` need to override (currently
        only ``_PhaseSpacePopup`` whose density panels are histogrammed
        on the fly and lose the source coordinates).
        """
        return []

    def _collect_panels(self) -> list[_Panel]:
        """Discover every plotted curve/image/table inside the popup."""
        panels: list[_Panel] = []
        seen_plot_items: set[int] = set()

        # 1. Standalone PlotWidgets (covers EnvelopeTriple — which contains
        #    three `PlotWidget` children — and every other line popup).
        for pw in self.findChildren(pg.PlotWidget):
            item = pw.getPlotItem()
            if id(item) in seen_plot_items:
                continue
            seen_plot_items.add(id(item))
            label = _panel_label(item, fallback=f"plot_{len(panels)+1}")
            panel = _Panel(
                label=label,
                xlabel=_axis_label(item, "bottom"),
                ylabel=_axis_label(item, "left"),
                curves=_drop_anonymous_mirrors(_curves_from_plotitem(item)),
                images=_images_from_plotitem(item, default_name=label),
            )
            if panel.curves or panel.images:
                panels.append(panel)

        # 2. GraphicsLayoutWidgets (DensityPanel inherits this; its inner
        #    PlotItems aren't reachable as PlotWidget instances).
        for glw in self.findChildren(pg.GraphicsLayoutWidget):
            for item in list(glw.ci.items.keys()):
                if not isinstance(item, pg.PlotItem):
                    continue
                if id(item) in seen_plot_items:
                    continue
                seen_plot_items.add(id(item))
                label = _panel_label(item, fallback=f"panel_{len(panels)+1}")
                panel = _Panel(
                    label=label,
                    xlabel=_axis_label(item, "bottom"),
                    ylabel=_axis_label(item, "left"),
                    curves=_drop_anonymous_mirrors(_curves_from_plotitem(item)),
                    images=_images_from_plotitem(item, default_name=label),
                )
                if panel.curves or panel.images:
                    panels.append(panel)

        # 3. QTableWidgets (covers _BpmsPopup).
        for tbl in self.findChildren(QTableWidget):
            ncols = tbl.columnCount(); nrows = tbl.rowCount()
            if ncols == 0 or nrows == 0:
                continue
            headers: list[str] = []
            for c in range(ncols):
                hdr = tbl.horizontalHeaderItem(c)
                headers.append(hdr.text() if hdr else f"col_{c}")
            cols = [[] for _ in range(ncols)]
            for r in range(nrows):
                for c in range(ncols):
                    it = tbl.item(r, c)
                    cols[c].append(it.text() if it is not None else "")
            curves: list[_Curve] = []
            x_idx = np.arange(nrows)
            for c, h in enumerate(headers):
                # Numeric-coerce per CELL (a single text cell must not
                # drop the whole column — the beam-parameters table
                # mixes numbers with species/location strings); skip
                # columns with no numeric content at all.
                y = []
                for v in cols[c]:
                    try:
                        y.append(float(v))
                    except (TypeError, ValueError):
                        y.append(np.nan)
                y = np.asarray(y, dtype=float)
                # not-all-NaN gate (NOT isfinite): an all-inf column is
                # a diverged quantity that must still export; pure-text
                # columns (all NaN) are dropped.
                if (~np.isnan(y)).any():
                    curves.append(_Curve(name=h, x=x_idx, y=y))
            if curves:
                panels.append(_Panel(label=f"table_{len(panels)+1}",
                                     xlabel="row", ylabel="",
                                     curves=curves))

        # 4. Subclass-supplied extras (e.g. raw particles for phase space).
        try:
            panels.extend(self._extra_panels())
        except Exception:
            pass
        return panels

    def _save_plot_data(self) -> None:
        panels = self._collect_panels()
        # Detect optional helpers up front; missing ones drop their filters.
        try:
            import h5py  # noqa: F401
            have_h5 = True
        except Exception:
            have_h5 = False
        try:
            from PyQt6.QtSvg import QSvgGenerator  # noqa: F401
            have_svg = True
        except Exception:
            have_svg = False
        try:
            from PyQt6.QtPrintSupport import QPrinter  # noqa: F401
            have_pdf = True
        except Exception:
            have_pdf = False

        # Data filters first (only if there is data); then image filters,
        # which work even when the popup hasn't received any results yet.
        filters: list[str] = []
        if panels:
            filters += ["CSV (*.csv)", "NumPy (*.npz)", "JSON (*.json)"]
            if have_h5:
                filters.append("HDF5 (*.h5 *.hdf5)")
        filters.append("PNG image (*.png)")
        filters.append("JPEG image (*.jpg *.jpeg)")
        if have_svg:
            filters.append("SVG (*.svg)")
        if have_pdf:
            filters.append("PDF (*.pdf)")

        default = (self._raw_title or "plot").split("—")[0].strip() \
            .replace(" ", "_").replace("·", "")
        if not default:
            default = "plot"
        path, sel = QFileDialog.getSaveFileName(
            self, "Save plot", default, ";;".join(filters))
        if not path:
            return
        p = Path(path)
        ext = p.suffix.lower()
        known = (".csv", ".npz", ".json", ".h5", ".hdf5",
                 ".png", ".jpg", ".jpeg", ".svg", ".pdf")
        # If user didn't supply an extension, derive from the picked filter.
        if ext not in known:
            if "npz" in sel:    ext = ".npz"
            elif "json" in sel: ext = ".json"
            elif "h5" in sel or "hdf5" in sel: ext = ".h5"
            elif "png" in sel:  ext = ".png"
            elif "jpg" in sel or "jpeg" in sel: ext = ".jpg"
            elif "svg" in sel:  ext = ".svg"
            elif "pdf" in sel:  ext = ".pdf"
            else:               ext = ".csv"
            p = p.with_suffix(ext)

        is_data = ext in (".csv", ".npz", ".json", ".h5", ".hdf5")
        if is_data and not panels:
            QMessageBox.information(self, "Save plot",
                                    "Nothing to save — no plotted data found.")
            return
        try:
            if   ext == ".csv":  self._write_csv(p, panels)
            elif ext == ".npz":  self._write_npz(p, panels)
            elif ext == ".json": self._write_json(p, panels)
            elif ext in (".h5", ".hdf5"): self._write_h5(p, panels)
            elif ext in (".png", ".jpg", ".jpeg"):
                self._write_raster(p, ext)
            elif ext == ".svg":  self._write_svg(p)
            elif ext == ".pdf":  self._write_pdf(p)
            else:
                raise ValueError(f"Unsupported extension {ext!r}")
        except Exception as e:
            QMessageBox.critical(self, "Save plot",
                                 f"Failed to write {p.name}:\n{e}")
            return
        if is_data:
            msg = f"Saved {len(panels)} panel(s) to {p}"
        else:
            msg = f"Saved image to {p}"
        QMessageBox.information(self, "Save plot", msg)

    # ------------------------------------------------------------------
    # Format writers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_key(s: str) -> str:
        out = []
        for ch in s:
            if ch.isalnum() or ch in "._-":
                out.append(ch)
            else:
                out.append("_")
        return "".join(out).strip("_") or "x"

    def _write_csv(self, path: Path, panels: list[_Panel]) -> None:
        # Try wide-table mode if all curves across all panels share one x.
        all_curves = [c for p in panels for c in p.curves]
        wide = False
        if all_curves:
            x0 = all_curves[0].x
            wide = all(c.x.shape == x0.shape and np.array_equal(c.x, x0)
                       for c in all_curves)
        with open(path, "w", encoding="utf-8") as f:
            if wide and all_curves:
                # Header: x then panel:curve names.
                xname = (panels[0].xlabel or "x")
                cols = [xname] + [f"{p.label}:{c.name}"
                                  for p in panels for c in p.curves]
                f.write(",".join(cols) + "\n")
                arrs = [all_curves[0].x] + [c.y for c in all_curves]
                for row in zip(*arrs):
                    f.write(",".join(f"{v:.7g}" for v in row) + "\n")
            else:
                for pn in panels:
                    if not pn.curves and not pn.images:
                        continue
                    f.write(f"# panel: {pn.label}\n")
                    if pn.xlabel: f.write(f"# xlabel: {pn.xlabel}\n")
                    if pn.ylabel: f.write(f"# ylabel: {pn.ylabel}\n")
                    if pn.curves:
                        # Each panel: x | y_curve1 | y_curve2 | … aligned
                        # on the first curve's x; non-matching curves get
                        # their own block.
                        x0 = pn.curves[0].x
                        same_x = all(c.x.shape == x0.shape and
                                     np.array_equal(c.x, x0)
                                     for c in pn.curves)
                        if same_x:
                            cols = [pn.xlabel or "x"] + [c.name for c in pn.curves]
                            f.write(",".join(cols) + "\n")
                            arrs = [x0] + [c.y for c in pn.curves]
                            for row in zip(*arrs):
                                f.write(",".join(f"{v:.7g}" for v in row) + "\n")
                        else:
                            for c in pn.curves:
                                f.write(f"# curve: {c.name}\n")
                                f.write(f"{pn.xlabel or 'x'},{c.name}\n")
                                for xv, yv in zip(c.x, c.y):
                                    f.write(f"{xv:.7g},{yv:.7g}\n")
                    for img in pn.images:
                        side = path.with_suffix("")
                        side = side.parent / (
                            side.name + "_" + self._safe_key(pn.label) +
                            "_" + self._safe_key(img.name) + ".csv")
                        f.write(f"# image dumped to {side.name}\n")
                        np.savetxt(side, img.data, delimiter=",", fmt="%.7g")
                    f.write("\n")

    def _write_npz(self, path: Path, panels: list[_Panel]) -> None:
        out: dict[str, np.ndarray] = {}
        for pn in panels:
            base = self._safe_key(pn.label)
            for c in pn.curves:
                key = f"{base}__{self._safe_key(c.name)}"
                out[key + "_x"] = np.asarray(c.x)
                out[key + "_y"] = np.asarray(c.y)
            for img in pn.images:
                key = f"{base}__image"
                if img.name and img.name != pn.label:
                    key = f"{base}__{self._safe_key(img.name)}_image"
                out[key] = np.asarray(img.data)
                if img.extent is not None:
                    out[key + "_extent"] = np.asarray(img.extent, dtype=float)
        np.savez_compressed(path, **out)

    def _write_json(self, path: Path, panels: list[_Panel]) -> None:
        doc: dict = {}
        for pn in panels:
            entry: dict = {"xlabel": pn.xlabel, "ylabel": pn.ylabel,
                            "curves": [], "images": []}
            for c in pn.curves:
                entry["curves"].append({
                    "name": c.name,
                    "x": np.asarray(c.x).tolist(),
                    "y": np.asarray(c.y).tolist(),
                })
            for img in pn.images:
                entry["images"].append({
                    "name": img.name,
                    "data": np.asarray(img.data).tolist(),
                    "extent": list(img.extent) if img.extent else None,
                })
            doc[pn.label] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)

    def _write_h5(self, path: Path, panels: list[_Panel]) -> None:
        import h5py
        with h5py.File(path, "w") as h:
            for pn in panels:
                gname = self._safe_key(pn.label)
                if gname in h:
                    gname = f"{gname}_{id(pn) & 0xffff:04x}"
                g = h.create_group(gname)
                g.attrs["label"]  = pn.label
                g.attrs["xlabel"] = pn.xlabel
                g.attrs["ylabel"] = pn.ylabel
                for c in pn.curves:
                    name = self._safe_key(c.name)
                    sub = g.create_group(name) if name not in g else g[name]
                    sub.create_dataset("x", data=np.asarray(c.x))
                    sub.create_dataset("y", data=np.asarray(c.y))
                    sub.attrs["name"] = c.name
                for img in pn.images:
                    iname = self._safe_key(img.name) or "image"
                    if iname in g:
                        iname = f"{iname}_{id(img) & 0xffff:04x}"
                    ds = g.create_dataset(iname, data=np.asarray(img.data))
                    ds.attrs["name"] = img.name
                    if img.extent is not None:
                        ds.attrs["extent"] = np.asarray(img.extent, dtype=float)

    # ------------------------------------------------------------------
    # Image writers — render the entire popup widget (plots + chrome).
    # ------------------------------------------------------------------
    def _write_raster(self, path: Path, ext: str) -> None:
        pix = self.grab()
        fmt = "PNG" if ext == ".png" else "JPEG"
        ok = pix.save(str(path), fmt, 95 if fmt == "JPEG" else -1)
        if not ok:
            raise IOError(f"QPixmap.save returned False for {path}")

    def _write_svg(self, path: Path) -> None:
        from PyQt6.QtCore import QRectF, QSize
        from PyQt6.QtGui import QPainter
        from PyQt6.QtSvg import QSvgGenerator
        gen = QSvgGenerator()
        gen.setFileName(str(path))
        size = self.size()
        gen.setSize(size)
        gen.setViewBox(QRectF(0.0, 0.0, float(size.width()),
                              float(size.height())))
        gen.setTitle(self._raw_title or "linac_gen plot")
        painter = QPainter(gen)
        try:
            self.render(painter)
        finally:
            painter.end()

    def _write_pdf(self, path: Path) -> None:
        from PyQt6.QtGui import QPainter, QPageSize
        from PyQt6.QtCore import QSizeF
        from PyQt6.QtPrintSupport import QPrinter
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(str(path))
        # Match page size to the popup, scaled from px → points (72 dpi).
        size = self.size()
        pt = QSizeF(size.width() * 72.0 / 96.0,
                    size.height() * 72.0 / 96.0)
        printer.setPageSize(QPageSize(pt, QPageSize.Unit.Point))
        printer.setFullPage(True)
        painter = QPainter(printer)
        try:
            self.render(painter)
        finally:
            painter.end()

    # ------------------------------------------------------------------
    # Right-click menu — only fires on dialog chrome / empty margins so
    # pyqtgraph's per-plot Export menu remains the primary path on the
    # plot canvas itself.
    # ------------------------------------------------------------------
    def contextMenuEvent(self, event):  # noqa: N802 (Qt convention)
        target = self.childAt(event.pos())
        w = target
        while w is not None and w is not self:
            if isinstance(w, (pg.PlotWidget, pg.GraphicsLayoutWidget,
                               pg.GraphicsView)):
                event.ignore()
                return
            w = w.parentWidget()
        menu = QMenu(self)
        act = QAction("Save plot…  (Ctrl+S)", menu)
        act.triggered.connect(self._save_plot_data)
        menu.addAction(act)
        menu.exec(event.globalPos())


# ---------------------------------------------------------------------------
# Per-quantity popups.  Each owns a `refresh(results)` method so the popup
# stays in sync if results change while it's open.
# ---------------------------------------------------------------------------
class _BetatronResultsView:
    """Lightweight wrapper that overrides ``sigma_x``/``sigma_y`` (and
    optionally other transverse fields) with their dispersion-corrected
    betatron-only values while transparently forwarding every other
    attribute to the original results object.

    Used by the popups' Raw / Dispersion-corrected dropdowns so the
    plotting widgets (e.g. ``EnvelopeTriple``) don't need to know
    which mode they're in — they just call ``getattr(view, 'sigma_x')``
    and get the right thing.
    """
    __slots__ = ("_inner", "_overrides")

    def __init__(self, inner, overrides: dict):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        ov = self._overrides
        if name in ov:
            return ov[name]
        return getattr(self._inner, name)


def _build_betatron_view(results, fields=("sigma_x", "sigma_y")):
    """Compute a ``_BetatronResultsView`` for ``results`` with the
    requested transverse RMS fields replaced by their Schur-complement
    betatron values.  Returns ``None`` if ``results`` has no σ-matrix
    record (in which case the caller should fall back to raw).
    """
    S = _sigma_stack(results)
    if S is None or S.size == 0:
        return None
    sxx, sxxp, sxpxp, syy, syyp, sypyp = _betatron_sigma_blocks(S)
    overrides: dict = {}
    if "sigma_x" in fields:
        overrides["sigma_x"] = np.sqrt(sxx)
    if "sigma_y" in fields:
        overrides["sigma_y"] = np.sqrt(syy)
    if "sigma_xp" in fields:
        overrides["sigma_xp"] = np.sqrt(sxpxp)
    if "sigma_yp" in fields:
        overrides["sigma_yp"] = np.sqrt(sypyp)
    return _BetatronResultsView(results, overrides)


# Shared text for the Raw/Corrected dropdown so every popup uses
# identical wording.
_MODE_RAW = "Raw (includes dispersion)"
_MODE_BETA = "Dispersion-corrected (betatron only)"


def _mk_mode_combo(callback):
    """Build the standard Raw / Dispersion-corrected QComboBox.

    ``callback`` is connected to ``currentIndexChanged``.  Returns
    the QComboBox.
    """
    cb = QComboBox()
    cb.addItem(_MODE_RAW)
    cb.addItem(_MODE_BETA)
    cb.setCurrentIndex(0)
    cb.setToolTip(
        "Raw shows the σ-matrix entry as recorded (includes the\n"
        "dispersive contribution D · σ_δ in dispersive regions).\n"
        "Dispersion-corrected subtracts the Schur-complement piece\n"
        "  Σ_β,ii = Σ_ii − Σ_i5² / Σ_55\n"
        "giving the pure-betatron quantity — the right number to\n"
        "compare against the design β·ε in matched-Twiss workflows."
    )
    cb.currentIndexChanged.connect(callback)
    return cb


class _RmsPopup(_PopupPlot):
    def __init__(self, parent):
        super().__init__(parent, "RMS σ  —  σ_x · σ_y · σ_z", size=(1000, 700))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Top control row: aperture-overlay + lattice-strip + mode toggles.
        from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self._chk_ap = QCheckBox("Show aperture")
        self._chk_ap.setChecked(True)
        self._chk_ap.toggled.connect(lambda b: self._triple.set_aperture_visible(b))
        ctrl.addWidget(self._chk_ap)
        self._chk_lat = QCheckBox("Show lattice")
        self._chk_lat.setChecked(True)
        self._chk_lat.toggled.connect(self.set_lattice_strip_visible)
        ctrl.addWidget(self._chk_lat)
        ctrl.addWidget(QLabel("Display:"))
        self._mode_combo = _mk_mode_combo(lambda _i: self._redraw())
        ctrl.addWidget(self._mode_combo)
        ctrl.addStretch(1)
        v.addLayout(ctrl)
        self._last_results = None

        self._triple = EnvelopeTriple()
        v.addWidget(self._triple, stretch=1)
        # Pull the current lattice (if any) from the parent ResultsTab and
        # subscribe to changes so the aperture overlay stays in sync.
        state = getattr(parent, "state", None)
        if state is not None:
            self._triple.set_lattice(getattr(state, "lattice", None))
            try:
                state.lattice_changed.connect(self._triple.set_lattice)
            except Exception:
                pass

        # Lattice impression strip above the triple — EnvelopeTriple's
        # first sub-plot drives the shared x-axis.
        first_plot = self._triple.findChildren(pg.PlotWidget)
        if first_plot:
            self.attach_lattice_strip(self._triple)

    def refresh(self, results):
        self._last_results = results
        self._redraw()

    def _redraw(self):
        results = self._last_results
        if results is None:
            self._triple.set_data(None)
            return
        if self._mode_combo.currentIndex() == 1:
            view = _build_betatron_view(results, fields=("sigma_x", "sigma_y"))
            self._triple.set_data(view if view is not None else results)
        else:
            self._triple.set_data(results)


class _EmittancePopup(_PopupPlot):
    def __init__(self, parent):
        super().__init__(parent,
                         "Emittance  —  ε_x · ε_y · ε_z · ε_t (4-D)",
                         size=(1000, 620))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Top control row: lattice-strip + Raw/Corrected mode toggles.
        from PyQt6.QtWidgets import QCheckBox, QHBoxLayout
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self._chk_lat = QCheckBox("Show lattice")
        self._chk_lat.setChecked(True)
        self._chk_lat.toggled.connect(self.set_lattice_strip_visible)
        ctrl.addWidget(self._chk_lat)
        ctrl.addWidget(QLabel("Display:"))
        self._mode_combo = _mk_mode_combo(lambda _i: self._redraw())
        ctrl.addWidget(self._mode_combo); ctrl.addStretch(1)
        v.addLayout(ctrl)
        self._last_results = None

        self._px = _mk_plot("ε_x  (with ε_t overlay)", "mm·mrad")
        self._py = _mk_plot("ε_y", "mm·mrad")
        self._pz = _mk_plot("ε_z", "deg·MeV")
        for p in (self._py, self._pz):
            p.setXLink(self._px)
        self._cx = filled_curve(self._px, theme.ACCENT, name="ε_x")
        # 4-D coupling-invariant transverse emittance (= TraceWin εt) lives
        # on the same axis as ε_x because both share mm·mrad.  In an
        # uncoupled lattice ε_t = ε_x · ε_y; under solenoid coupling it
        # stays smooth while the 2-D projections wobble.
        self._ct = self._px.plot(pen=pg.mkPen("#c08cff", width=2),
                                  name="ε_t (4-D)")
        self._cy = filled_curve(self._py, "#a3e635", name="ε_y")
        self._cz = filled_curve(self._pz, "#fbbf24", name="ε_z")
        for p in (self._px, self._py, self._pz):
            v.addWidget(p, stretch=1)

        # Lattice strip above the first plot — all three share x-axis.
        self.attach_lattice_strip(self._px)

    def refresh(self, results):
        self._last_results = results
        self._redraw()

    def _redraw(self):
        results = self._last_results
        if results is None:
            for c in (self._cx, self._cy, self._cz, self._ct): c.setData([], [])
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        def arr(a): return np.asarray(getattr(results, a, []), dtype=float)
        def pair(c, v):
            if s.size and v.size == s.size: c.setData(s, v)
        if self._mode_combo.currentIndex() == 1:
            # Dispersion-corrected: derive ε_x,β / ε_y,β from the Schur-
            # complement Σ_β sub-blocks; ε_4D,β from det(Σ_β,4×4).
            # ε_z left raw -- z is already the dispersive plane.
            S = _sigma_stack(results)
            if S is None or S.size == 0 or S.shape[0] != s.size:
                # No Σ recorded → fall back to raw quietly.
                pair(self._cx, arr("emit_x"))
                pair(self._cy, arr("emit_y"))
                pair(self._ct, arr("emit_4d"))
            else:
                sxx, sxxp, sxpxp, syy, syyp, sypyp = _betatron_sigma_blocks(S)
                ex_b = np.sqrt(np.clip(sxx * sxpxp - sxxp ** 2, 0.0, None))
                ey_b = np.sqrt(np.clip(syy * sypyp - syyp ** 2, 0.0, None))
                pair(self._cx, ex_b)
                pair(self._cy, ey_b)
                # ε_4D,β = √det(Σ_β,4×4) — build the 4×4 betatron Σ
                # from the six independent entries and take the
                # determinant per step.
                S_b = np.zeros_like(S[:, :4, :4])
                S_b[:, 0, 0] = sxx;    S_b[:, 1, 1] = sxpxp
                S_b[:, 2, 2] = syy;    S_b[:, 3, 3] = sypyp
                S_b[:, 0, 1] = sxxp;   S_b[:, 1, 0] = sxxp
                S_b[:, 2, 3] = syyp;   S_b[:, 3, 2] = syyp
                # Off-diagonal x↔y blocks: copy from raw (coupling
                # term, NOT dispersion).
                for (i, j) in ((0, 2), (0, 3), (1, 2), (1, 3)):
                    S_b[:, i, j] = S[:, i, j] - (
                        S[:, i, 5] * S[:, j, 5]
                        / np.where(S[:, 5, 5] > 1e-30, S[:, 5, 5], 1.0)
                    ) * (S[:, 5, 5] > 1e-30).astype(float)
                    S_b[:, j, i] = S_b[:, i, j]
                e4d_b_sq = np.linalg.det(S_b)
                e4d_b = np.sqrt(np.clip(e4d_b_sq, 0.0, None))
                pair(self._ct, e4d_b)
        else:
            pair(self._cx, arr("emit_x"))
            pair(self._cy, arr("emit_y"))
            pair(self._ct, arr("emit_4d"))
        pair(self._cz, arr("emit_z_mmmrad") if hasattr(results, "emit_z_mmmrad") else arr("emit_z"))


def _sigma_stack(results) -> np.ndarray | None:
    """Return the recorded sigma matrices as a 3-D array (n, 6, 6), or None."""
    if results is None:
        return None
    sm = getattr(results, "sigma_matrix", None)
    if not sm:
        return None
    try:
        return np.asarray(sm, dtype=float)
    except Exception:
        return None


def _betatron_sigma_blocks(S: np.ndarray, eps: float = 1e-30):
    """Schur-complement dispersion correction on a 6×6 Σ stack.

    Given a stack of 6×6 σ matrices ``S`` (shape ``(n, 6, 6)``),
    return six 1-D arrays of length ``n`` giving the betatron (i.e.
    dispersion-corrected) entries of the transverse 4×4 sub-block:

        sxx,β  = Σ[0,0] − Σ[0,5]² / Σ[5,5]
        sxxp,β = Σ[0,1] − Σ[0,5]·Σ[1,5] / Σ[5,5]
        sxpxp,β= Σ[1,1] − Σ[1,5]² / Σ[5,5]
        syy,β  = Σ[2,2] − Σ[2,5]² / Σ[5,5]
        syyp,β = Σ[2,3] − Σ[2,5]·Σ[3,5] / Σ[5,5]
        sypyp,β= Σ[3,3] − Σ[3,5]² / Σ[5,5]

    Where Σ[5,5] ≤ ``eps`` (DC beam / zero energy spread / empty
    record), the dispersive contribution is zero by construction so
    the raw entries are returned unchanged.

    Numerical clamp: results are clipped to ≥ 0 for the diagonal
    entries (positivity is a property of the true betatron Σ; tiny
    negative numbers from float subtraction would otherwise NaN the
    downstream √).

    Returns ``(sxx, sxxp, sxpxp, syy, syyp, sypyp)``.

    Mirrors the inner ``_state`` helper at
    ``linac_gen/matching/periodic.py:487`` (which uses the same
    Schur complement for 8-state SC matching).  Kept self-contained
    in this module so results_tab.py has no upward dependency on
    matching code.
    """
    sww = S[:, 5, 5]
    safe = sww > eps
    inv = np.where(safe, 1.0 / np.where(safe, sww, 1.0), 0.0)

    def _diag(i):
        return np.clip(S[:, i, i] - (S[:, i, 5] ** 2) * inv, 0.0, None)

    def _off(i, j):
        return S[:, i, j] - S[:, i, 5] * S[:, j, 5] * inv

    sxx = _diag(0)
    sxpxp = _diag(1)
    sxxp = _off(0, 1)
    syy = _diag(2)
    sypyp = _diag(3)
    syyp = _off(2, 3)
    return sxx, sxxp, sxpxp, syy, syyp, sypyp


def _twiss_from_block(s11: np.ndarray, s12: np.ndarray, s22: np.ndarray):
    """Extract (α, β, γ, ε) per step from three 1-D arrays of σ-matrix entries."""
    emit_sq = s11 * s22 - s12 * s12
    emit_sq = np.clip(emit_sq, 0.0, None)
    emit = np.sqrt(emit_sq)
    safe = emit > 1e-30
    beta = np.where(safe, s11 / np.where(safe, emit, 1.0), 0.0)
    alpha = np.where(safe, -s12 / np.where(safe, emit, 1.0), 0.0)
    gamma_t = np.where(safe, s22 / np.where(safe, emit, 1.0), 0.0)
    return alpha, beta, gamma_t, emit


class _LongTwissPopup(_PopupPlot):
    """Longitudinal Twiss α_z, β_z, γ_z (derived from σ_matrix[4:6, 4:6])."""

    def __init__(self, parent):
        super().__init__(parent,
                         "Longitudinal Twiss  —  α_z · β_z · γ_z (from [Δφ,ΔW])",
                         size=(1000, 620))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Top control row: lattice-strip toggle.
        from PyQt6.QtWidgets import QCheckBox, QHBoxLayout
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self._chk_lat = QCheckBox("Show lattice")
        self._chk_lat.setChecked(True)
        self._chk_lat.toggled.connect(self.set_lattice_strip_visible)
        ctrl.addWidget(self._chk_lat); ctrl.addStretch(1)
        v.addLayout(ctrl)

        self._pa = _mk_plot("α_z", "")
        self._pb = _mk_plot("β_z", "deg/MeV")
        self._pg = _mk_plot("γ_z", "MeV/deg")
        for p in (self._pb, self._pg): p.setXLink(self._pa)
        self._ca = self._pa.plot(pen=curve_pen(theme.ACCENT), name="α_z")
        self._cb = filled_curve(self._pb, "#a3e635", name="β_z")
        self._cg = filled_curve(self._pg, "#fbbf24", name="γ_z")
        for p in (self._pa, self._pb, self._pg):
            v.addWidget(p, stretch=1)

        self.attach_lattice_strip(self._pa)

    def refresh(self, results):
        S = _sigma_stack(results)
        if S is None or S.size == 0:
            for c in (self._ca, self._cb, self._cg): c.setData([], [])
            return
        s = np.asarray(results.s, dtype=float)
        a_z, b_z, g_z, _ = _twiss_from_block(S[:, 4, 4], S[:, 4, 5], S[:, 5, 5])
        self._ca.setData(s, a_z)
        self._cb.setData(s, b_z)
        self._cg.setData(s, g_z)


class _DivergencePopup(_PopupPlot):
    """RMS divergence σ_x', σ_y' (mrad)."""

    def __init__(self, parent):
        super().__init__(parent, "Divergence  —  σ_x' · σ_y'", size=(1000, 480))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Top control row: lattice-strip + Raw/Corrected mode toggles.
        from PyQt6.QtWidgets import QCheckBox, QHBoxLayout
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self._chk_lat = QCheckBox("Show lattice")
        self._chk_lat.setChecked(True)
        self._chk_lat.toggled.connect(self.set_lattice_strip_visible)
        ctrl.addWidget(self._chk_lat)
        ctrl.addWidget(QLabel("Display:"))
        self._mode_combo = _mk_mode_combo(lambda _i: self._redraw())
        ctrl.addWidget(self._mode_combo); ctrl.addStretch(1)
        v.addLayout(ctrl)
        self._last_results = None

        self._px = _mk_plot("σ_x'", "mrad")
        self._py = _mk_plot("σ_y'", "mrad")
        self._py.setXLink(self._px)
        self._cx = filled_curve(self._px, theme.ACCENT, name="σ_x'")
        self._cy = filled_curve(self._py, "#a3e635", name="σ_y'")
        for p in (self._px, self._py): v.addWidget(p, stretch=1)

        self.attach_lattice_strip(self._px)

    def refresh(self, results):
        self._last_results = results
        self._redraw()

    def _redraw(self):
        results = self._last_results
        S = _sigma_stack(results)
        if S is None or S.size == 0:
            self._cx.setData([], []); self._cy.setData([], []); return
        s = np.asarray(results.s, dtype=float)
        if self._mode_combo.currentIndex() == 1:
            _, _, sxpxp, _, _, sypyp = _betatron_sigma_blocks(S)
            self._cx.setData(s, np.sqrt(sxpxp))
            self._cy.setData(s, np.sqrt(sypyp))
        else:
            self._cx.setData(s, np.sqrt(np.clip(S[:, 1, 1], 0, None)))
            self._cy.setData(s, np.sqrt(np.clip(S[:, 3, 3], 0, None)))


class _PeakExcursionPopup(_PopupPlot):
    """Peak excursion — ±X_max and ±Y_max of any surviving particle.

    Plotted symmetrically about y = 0 so the values read against the
    beam pipe context.  The aperture overlay is on by default and can
    be toggled off via the checkbox.
    """

    def __init__(self, parent):
        super().__init__(parent, "Peak excursion  —  X_max · Y_max",
                         size=(1000, 540))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Top control row.
        from PyQt6.QtWidgets import QCheckBox, QHBoxLayout
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self._chk_ap = QCheckBox("Show aperture")
        self._chk_ap.setChecked(True)
        self._chk_ap.toggled.connect(self._on_aperture_toggled)
        ctrl.addWidget(self._chk_ap)
        self._chk_lat = QCheckBox("Show lattice")
        self._chk_lat.setChecked(True)
        self._chk_lat.toggled.connect(self.set_lattice_strip_visible)
        ctrl.addWidget(self._chk_lat)
        ctrl.addWidget(QLabel("Display:"))
        self._mode_combo = _mk_mode_combo(lambda _i: self._redraw())
        self._mode_combo.setToolTip(
            self._mode_combo.toolTip() + "\n\nFor _PeakExcursionPopup: "
            "Corrected mode only affects the envelope-fallback path "
            "(5·σ_x → 5·σ_x,β).  When MP-tracked x_max is available it "
            "is used as-is in both modes -- per-particle dispersion "
            "correction is not supported."
        )
        ctrl.addWidget(self._mode_combo)
        ctrl.addStretch(1)
        v.addLayout(ctrl)
        self._last_results = None

        self._px = _mk_plot("X_max", "mm")
        self._py = _mk_plot("Y_max", "mm")
        self._py.setXLink(self._px)
        # Symmetric ±X_max / ±Y_max bands using FillBetweenItem so the
        # full range from -value to +value is filled (not just above 0).
        from linac_gen_gui.interphase.plots.envelope_plot import (
            _make_symmetric_band, _make_aperture_curves, set_aperture_data,
            set_band_data, _autorange,
        )
        self._set_band_data = set_band_data
        self._autorange = _autorange
        self._cx_pos, self._cx_neg, _ = _make_symmetric_band(
            self._px, "#f87171", name="X_max")
        self._cy_pos, self._cy_neg, _ = _make_symmetric_band(
            self._py, "#fb923c", name="Y_max")

        # Aperture overlay — sourced from the lattice.
        from linac_gen_gui.interphase.plots.envelope_plot import (
            _set_aperture_visibility,
        )
        self._set_aperture_visibility = _set_aperture_visibility
        self._make_aperture_curves = _make_aperture_curves
        self._set_aperture_data = set_aperture_data
        self._ap_x = _make_aperture_curves(self._px)
        self._ap_y = _make_aperture_curves(self._py)
        self._ap_visible = True
        self._ap_x_cache = None
        self._ap_y_cache = None

        for p in (self._px, self._py): v.addWidget(p, stretch=1)

        state = getattr(parent, "state", None)
        if state is not None:
            self._set_lattice(getattr(state, "lattice", None))
            try:
                state.lattice_changed.connect(self._set_lattice)
            except Exception:
                pass

        # Lattice impression strip above the X_max plot.
        self.attach_lattice_strip(self._px)

    def _set_lattice(self, lattice) -> None:
        from linac_gen.analysis.aperture_profile import aperture_profile
        if lattice is None:
            self._ap_x_cache = None
            self._ap_y_cache = None
        else:
            s_mm, rx_mm, ry_mm = aperture_profile(lattice)
            if s_mm.size == 0:
                self._ap_x_cache = None
                self._ap_y_cache = None
            else:
                self._ap_x_cache = (s_mm, rx_mm)
                self._ap_y_cache = (s_mm, ry_mm)
        self._on_aperture_toggled(self._ap_visible)

    def _on_aperture_toggled(self, visible: bool) -> None:
        self._ap_visible = bool(visible)
        self._set_aperture_visibility(self._ap_x, self._ap_x_cache, visible)
        self._set_aperture_visibility(self._ap_y, self._ap_y_cache, visible)
        for p in (self._px, self._py):
            self._autorange(p)

    def refresh(self, results):
        self._last_results = results
        self._redraw()

    def _redraw(self):
        results = self._last_results
        if results is None:
            for c in (self._cx_pos, self._cx_neg,
                      self._cy_pos, self._cy_neg):
                c.setData([], [])
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        xm = np.asarray(getattr(results, "x_max", []), dtype=float)
        ym = np.asarray(getattr(results, "y_max", []), dtype=float)
        # Envelope mode doesn't record per-particle extremes — fall back to a
        # 5-σ envelope (TraceWin convention) so the aperture overlay is still
        # meaningful for design studies.  In Dispersion-corrected mode use
        # σ_β instead of raw σ for the fallback so the 5-σ envelope shows
        # the pure betatron content (matches the other transverse popups).
        corrected = self._mode_combo.currentIndex() == 1
        if corrected:
            S = _sigma_stack(results)
            if S is not None and S.size and S.shape[0] == s.size:
                sxx, _, _, syy, _, _ = _betatron_sigma_blocks(S)
                sx_arr = np.sqrt(sxx)
                sy_arr = np.sqrt(syy)
            else:
                sx_arr = np.asarray(getattr(results, "sigma_x", []), dtype=float)
                sy_arr = np.asarray(getattr(results, "sigma_y", []), dtype=float)
        else:
            sx_arr = np.asarray(getattr(results, "sigma_x", []), dtype=float)
            sy_arr = np.asarray(getattr(results, "sigma_y", []), dtype=float)
        if s.size and xm.size != s.size:
            if sx_arr.size == s.size:
                xm = 5.0 * sx_arr
        if s.size and ym.size != s.size:
            if sy_arr.size == s.size:
                ym = 5.0 * sy_arr
        if s.size and xm.size == s.size:
            self._set_band_data(self._cx_pos, self._cx_neg, s, xm)
        if s.size and ym.size == s.size:
            self._set_band_data(self._cy_pos, self._cy_neg, s, ym)


class _DensityPopup(_PopupPlot):
    """2-D particle-density histogram along the lattice.

    The y-axis is one of the beam coordinates (x, x', y, y', φ, W); the
    x-axis is the longitudinal s.  Counts come from the recorder's
    :meth:`density_array` cache, which is populated only when the user
    opted in via the Tracking tab's "Record particle density" toggle.
    A log-intensity colour map keeps the diffuse halo visible alongside
    the dense core.

    The popup also overlays the corresponding ±σ envelope curve and (for
    the transverse position axes) the lattice aperture profile, so the
    heatmap reads against the same context as the other envelope plots.
    """

    # Display units for each axis: (label, internal→display scale, units).
    # The beam particle array is already stored in mm / mrad / deg / MeV
    # (sigma_x and the other recorder columns are computed without scaling),
    # so the display scale is 1.0 across the board.
    _AXIS_INFO = {
        "x":   ("x",   1.0, "mm"),
        "y":   ("y",   1.0, "mm"),
        "xp":  ("x'",  1.0, "mrad"),
        "yp":  ("y'",  1.0, "mrad"),
        "phi": ("φ",   1.0, "deg"),
        "w":   ("W",   1.0, "MeV"),
    }
    # Sigma column on results that pairs with each axis (None = no overlay).
    _SIGMA_ATTR = {
        "x":   "sigma_x",      # mm
        "y":   "sigma_y",      # mm
        "xp":  None,
        "yp":  None,
        "phi": "sigma_phi",    # deg
        "w":   "sigma_w",      # MeV
    }

    def __init__(self, parent):
        super().__init__(parent, "Particle density  —  axis vs s",
                         size=(1100, 600))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Top control row.
        from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QComboBox
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("axis:"))
        self._axis_cb = QComboBox()
        for key, (label, _scale, units) in self._AXIS_INFO.items():
            self._axis_cb.addItem(f"{label}  [{units}]", userData=key)
        self._axis_cb.currentIndexChanged.connect(lambda _: self._redraw())
        ctrl.addWidget(self._axis_cb)
        ctrl.addSpacing(12)
        self._chk_log = QCheckBox("log scale")
        self._chk_log.setChecked(True)
        self._chk_log.toggled.connect(lambda _: self._redraw())
        ctrl.addWidget(self._chk_log)
        ctrl.addSpacing(12)
        self._chk_sigma = QCheckBox("overlay ±σ")
        self._chk_sigma.setChecked(True)
        self._chk_sigma.toggled.connect(lambda _: self._redraw())
        ctrl.addWidget(self._chk_sigma)
        self._chk_ap = QCheckBox("show aperture")
        self._chk_ap.setChecked(True)
        self._chk_ap.toggled.connect(self._on_aperture_toggled)
        ctrl.addWidget(self._chk_ap)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        # Hint shown when density wasn't recorded for the active run.
        self._hint = QLabel(
            "No density recorded.  Enable “Record particle density” on "
            "the Tracking tab and re-run."
        )
        self._hint.setStyleSheet(
            f"color:{theme.TEXT_2}; padding:2px 0;"
        )
        self._hint.setVisible(False)
        v.addWidget(self._hint)

        # Plot.
        self._plot = _mk_plot("y-axis", "")
        self._plot.setLabel("bottom", "s", units="mm")
        v.addWidget(self._plot, stretch=1)

        # Heatmap image (filled in by ``_redraw``).
        self._image = pg.ImageItem(axisOrder="row-major")
        self._image.setZValue(-10)
        self._plot.addItem(self._image)
        from linac_gen_gui.interphase.plots.plot_style import _density_colormap
        self._cmap = _density_colormap()
        self._image.setLookupTable(self._cmap.getLookupTable(0.0, 1.0, 256))

        # ±σ overlay band (FillBetweenItem-based).
        from linac_gen_gui.interphase.plots.envelope_plot import (
            _make_symmetric_band, _make_aperture_curves, set_aperture_data,
            set_band_data, _autorange, _set_aperture_visibility,
        )
        self._set_band_data = set_band_data
        self._autorange = _autorange
        self._set_aperture_visibility = _set_aperture_visibility
        self._set_aperture_data = set_aperture_data
        self._sigma_pos, self._sigma_neg, _ = _make_symmetric_band(
            self._plot, theme.ACCENT, name="±σ")

        # Aperture curves — shown only for transverse position axes (x, y).
        self._ap_pair = _make_aperture_curves(self._plot)
        self._ap_visible = True
        self._ap_x_cache = None
        self._ap_y_cache = None
        # Cache of the (s, density, edges) most-recently rendered so a
        # toggle (log / σ overlay / aperture) doesn't have to re-fetch.
        self._results = None

        # Pull the current lattice (if any) for aperture overlay.
        state = getattr(parent, "state", None)
        if state is not None:
            self._set_lattice(getattr(state, "lattice", None))
            try:
                state.lattice_changed.connect(self._set_lattice)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _set_lattice(self, lattice) -> None:
        from linac_gen.analysis.aperture_profile import aperture_profile
        if lattice is None:
            self._ap_x_cache = None
            self._ap_y_cache = None
        else:
            s_mm, rx_mm, ry_mm = aperture_profile(lattice)
            if s_mm.size == 0:
                self._ap_x_cache = None
                self._ap_y_cache = None
            else:
                self._ap_x_cache = (s_mm, rx_mm)
                self._ap_y_cache = (s_mm, ry_mm)
        self._on_aperture_toggled(self._ap_visible)

    def _on_aperture_toggled(self, visible: bool) -> None:
        self._ap_visible = bool(visible)
        # Aperture only meaningful for x / y axes.
        axis = self._axis_cb.currentData()
        cache = (self._ap_x_cache if axis == "x"
                 else self._ap_y_cache if axis == "y"
                 else None)
        self._set_aperture_visibility(self._ap_pair, cache, visible)
        self._autorange(self._plot)

    # ------------------------------------------------------------------
    def refresh(self, results):
        self._results = results
        self._redraw()

    def _redraw(self) -> None:
        results = self._results
        axis = self._axis_cb.currentData() or "x"
        label, scale, units = self._AXIS_INFO.get(
            axis, ("?", 1.0, ""))
        self._plot.setLabel("left", label, units=units)

        # Hide overlays then short-circuit if nothing to draw.
        density = (getattr(results, "density_array", lambda _a: None)(axis)
                   if results is not None else None)
        s = (np.asarray(getattr(results, "s", []), dtype=float)
             if results is not None else np.array([]))
        edges = None
        if results is not None:
            edges_dict = getattr(results, "density_edges", {}) or {}
            edges = edges_dict.get(axis)

        if density is None or s.size == 0 or edges is None:
            # Render an empty image so any previous heatmap is cleared.
            self._image.setImage(np.zeros((2, 2)))
            self._sigma_pos.setData([], []); self._sigma_neg.setData([], [])
            self._hint.setVisible(True)
            self._on_aperture_toggled(self._ap_visible)
            return
        self._hint.setVisible(False)

        # Trim to a common length in case density was recorded only on
        # the per-element record sites and ``s`` has more entries from
        # substeps (or vice-versa).
        n_steps = min(len(s), density.shape[0])
        if n_steps < 2:
            self._image.setImage(np.zeros((2, 2)))
            self._sigma_pos.setData([], []); self._sigma_neg.setData([], [])
            return
        s = s[:n_steps]
        density = density[:n_steps, :]

        # ImageItem expects (rows, cols) with row-major axisOrder.  Rows
        # are the y-axis (bins) and cols are s — so transpose density,
        # which is stored as (n_steps, n_bins).
        H = density.T.astype(float)
        # Gaussian-smooth the heatmap so adjacent-bin transitions read as
        # a continuous distribution rather than a stair-step.  Mirrors the
        # treatment in DensityPanel (phase-space popup).  Sigma is in
        # array indices: ~0.9 px in the bin direction, ~0.6 px along s.
        # ``mode="constant"`` (zero-pad) fades the edges to zero rather
        # than mirroring the boundary row — kills the sharp step at the
        # image rect when the beam reaches the far bins.
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=(0.9, 0.6),
                                mode="constant", cval=0.0)
        except ImportError:
            pass
        if self._chk_log.isChecked():
            H = np.log1p(H)
        self._image.setImage(H, autoLevels=False)
        # Levels: clip the colour scale to the [1 %, 99.5 %] quantile so
        # a single high-density spike doesn't compress the rest of the
        # image into the lowest colour-bin.  Falls back gracefully on
        # all-zero arrays.
        nonzero = H[H > 0.0]
        if nonzero.size:
            lo = float(np.quantile(nonzero, 0.01))
            hi = float(np.quantile(nonzero, 0.995))
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        self._image.setLevels((lo, hi))

        # Map the image rect to data coords.  Edges are in internal units;
        # scale to display units for the y-axis.
        y_lo = float(edges[0]) * scale
        y_hi = float(edges[-1]) * scale
        s_lo = float(s[0]); s_hi = float(s[-1])
        self._image.setRect(pg.QtCore.QRectF(
            s_lo, y_lo, s_hi - s_lo, y_hi - y_lo,
        ))

        # ±σ overlay.
        sigma_attr = self._SIGMA_ATTR.get(axis)
        if self._chk_sigma.isChecked() and sigma_attr is not None:
            sig = np.asarray(
                getattr(results, sigma_attr, []), dtype=float)
            if sig.size >= n_steps:
                self._set_band_data(self._sigma_pos, self._sigma_neg,
                                    s, sig[:n_steps])
            else:
                self._sigma_pos.setData([], [])
                self._sigma_neg.setData([], [])
        else:
            self._sigma_pos.setData([], [])
            self._sigma_neg.setData([], [])

        # Aperture (transverse axes only).
        self._on_aperture_toggled(self._ap_visible)


class _LongOffsetPopup(_PopupPlot):
    """Δφ_s (centroid − ref) and ΔW_s over s — captures centroid drift."""

    def __init__(self, parent):
        super().__init__(parent, "Longitudinal offset  —  Δφ_s · ΔW_s",
                         size=(1000, 480))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._pp = _mk_plot("Δφ_s", "deg")
        self._pw = _mk_plot("ΔW_s", "MeV")
        self._pw.setXLink(self._pp)
        self._cp = self._pp.plot(pen=curve_pen("#fbbf24"), name="Δφ_s")
        self._cw = self._pw.plot(pen=curve_pen("#a78bfa"), name="ΔW_s")
        for p in (self._pp, self._pw): v.addWidget(p, stretch=1)
        self.attach_lattice_strip(self._pp)

    def refresh(self, results):
        if results is None:
            self._cp.setData([], []); self._cw.setData([], []); return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        c = getattr(results, "centroid", None)
        if c and len(c) == s.size:
            arr = np.asarray(c)
            self._cp.setData(s, arr[:, 4])   # ⟨φ⟩ already = Δφ_s
            self._cw.setData(s, arr[:, 5])   # ⟨ΔW⟩
        elif s.size:
            # Results without a centroid (legacy/foreign) → flat zero.
            zeros = np.zeros_like(s)
            self._cp.setData(s, zeros)
            self._cw.setData(s, zeros)


class _DispersionWorker(_QThread2 := __import__("PyQt6.QtCore", fromlist=["QThread"]).QThread):
    """Background thread for the transfer-matrix dispersion walk.

    Field-map matrices are RK4-integrated on first evaluation, so a full
    linac walk can take tens of seconds — same reason ``_StructWorker``
    exists.  Cooperative stop + stale-key protocol mirror that class.
    """
    from PyQt6.QtCore import pyqtSignal as _Signal
    finished_signal = _Signal(object, object)
    failed_signal   = _Signal(object, str)

    def __init__(self, lattice, ref, eta0, key, matrix_cache=None):
        super().__init__()
        # numpy/BLAS on the 544 KB default macOS QThread stack → SIGBUS
        # (house pattern — see workers._MatchWorker).
        self.setStackSize(16 * 1024 * 1024)
        self._lattice = lattice
        self._ref = ref
        self._eta0 = eta0
        self._key = key
        self._matrix_cache = matrix_cache
        import threading
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self):
        from linac_gen.core.cancelled import OperationCancelled
        try:
            from linac_gen.analysis.dispersion import dispersion_along_s
            out = dispersion_along_s(
                self._lattice, self._ref, eta0=self._eta0,
                cache=self._matrix_cache, should_stop=self._stopping,
            )
            self.finished_signal.emit(self._key, out)
        except OperationCancelled:
            return          # cancelled — emit nothing (no stale caching)
        except Exception as exc:                              # noqa: BLE001
            self.failed_signal.emit(self._key, str(exc))


class _DispersionPopup(_PopupPlot):
    """Dispersion D_x, D_y (metres) from σ-matrix cross terms, with an
    optional transfer-matrix (model) overlay.

    Statistical curve: δ ≡ Δp/p = ΔW / (β²γ·mc²);
    D_u = ⟨u·δ⟩ / ⟨δ²⟩ for u ∈ {x, y} — what the *beam* actually
    carries (includes space charge, seeded input dispersion, MP halo).

    Model curve (checkbox): the unit energy-offset ray propagated by the
    element transfer matrices (``analysis.dispersion``), seeded with the
    beam config's ``disp_*`` — pure linear machine optics, independent
    of the tracked beam.  On a static SC-free line fed a dispersion-free
    beam the two coincide exactly; where they split, the difference IS
    the beam physics (space charge, nonlinearity, losses).
    """

    def __init__(self, parent, state=None):
        super().__init__(parent, "Dispersion  —  D_x · D_y", size=(1000, 520))
        self._state = state if state is not None else getattr(
            parent, "state", None)
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        from PyQt6.QtWidgets import QCheckBox
        row = QHBoxLayout(); row.setSpacing(8)
        self._chk_model = QCheckBox("Transfer-matrix model")
        self._chk_model.setToolTip(
            "Overlay the dispersion of the lattice itself — the unit "
            "energy-offset ray propagated by the element transfer "
            "matrices, seeded with the beam config's input dispersion.  "
            "Independent of the tracked beam (no space charge)."
        )
        self._chk_model.toggled.connect(self._on_model_toggled)
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px;")
        row.addWidget(self._chk_model)
        row.addWidget(self._lbl_status)
        row.addStretch(1)
        v.addLayout(row)

        self._px = _mk_plot("D_x", "m")
        self._py = _mk_plot("D_y", "m")
        self._py.setXLink(self._px)
        self._cx = self._px.plot(pen=curve_pen(theme.ACCENT), name="D_x")
        self._cy = self._py.plot(pen=curve_pen("#a3e635"),    name="D_y")
        # Model pens must NOT share a hue with the statistical curves
        # (ACCENT is #22d3ee — a dashed overlay in the same cyan is
        # invisible on top of the solid curve): warm colours instead.
        _dash = Qt.PenStyle.DashLine
        self._cx_m = self._px.plot(
            pen=pg.mkPen("#fbbf24", width=2, style=_dash),
            name="D_x model (matrix)")
        self._cy_m = self._py.plot(
            pen=pg.mkPen("#f472b6", width=2, style=_dash),
            name="D_y model (matrix)")
        for p in (self._px, self._py): v.addWidget(p, stretch=1)
        self.attach_lattice_strip(self._px)

        self._model_out = None       # last finished walk result
        self._model_key = None       # key the stored result belongs to
        self._pending_key = None     # key the running worker computes
        self._worker = None          # generic name — base closeEvent stops it
        if self._state is None:
            self._chk_model.setEnabled(False)
            self._chk_model.setToolTip(
                "No app state (isolated popup) — model curve unavailable.")
        else:
            # attach_lattice_strip only subscribes via parent().state;
            # when the popup was handed a state explicitly (tests,
            # embedding) subscribe here instead — never twice.
            parent_state = getattr(parent, "state", None)
            if (self._state is not parent_state
                    and hasattr(self._state, "lattice_changed")):
                self._state.lattice_changed.connect(self._on_lattice_changed)

    # ---- model-curve plumbing ---------------------------------------
    def _model_inputs(self):
        """(lattice, ref, eta0, key) from app state, or None + reason."""
        state = self._state
        lattice = getattr(state, "lattice", None)
        cfg = getattr(state, "beam_config", None)
        if lattice is None or not getattr(lattice, "elements", None):
            return None, "Load a lattice first."
        if cfg is None:
            return None, "Set a beam config (Beam tab) first."
        try:
            from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
            from linac_gen.core.reference import ReferenceParticle
            sp_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
            sp = sp_map.get(cfg.species, PROTON)
            ref = ReferenceParticle(species=sp, w_kin=cfg.energy,
                                    frequency=cfg.frequency)
        except Exception as exc:                              # noqa: BLE001
            return None, f"Reference particle build failed: {exc}"
        eta0 = tuple(float(getattr(cfg, k, 0.0) or 0.0)
                     for k in ("disp_x", "disp_xp", "disp_y", "disp_yp"))
        key = (id(lattice), len(lattice.elements), float(cfg.energy),
               float(cfg.frequency), str(cfg.species), eta0)
        return (lattice, ref, eta0, key), None

    def _on_model_toggled(self, checked: bool) -> None:
        if not checked:
            self._cx_m.setData([], []); self._cy_m.setData([], [])
            self._lbl_status.setText("")
            return
        self._ensure_model()

    def _ensure_model(self) -> None:
        inputs, why = self._model_inputs()
        if inputs is None:
            self._lbl_status.setText(why)
            return
        lattice, ref, eta0, key = inputs
        if self._model_out is not None and self._model_key == key:
            self._draw_model(self._model_out)
            return
        if self._worker is not None and self._worker.isRunning():
            if self._pending_key == key:
                return                       # already computing this key
            self._worker.request_stop()
            self._worker.requestInterruption()
            # Overwriting the only reference to a RUNNING QThread lets
            # GC destroy it alive → Qt abort.  Park it until it exits.
            _park_zombie(self._worker)
        self._pending_key = key
        self._lbl_status.setText(
            "Computing model dispersion (field-map matrices are slow "
            "on first run)…")
        w = _DispersionWorker(
            lattice, ref, eta0, key,
            matrix_cache=getattr(self._state, "matrix_cache", None))
        w.finished_signal.connect(self._on_model_done)
        w.failed_signal.connect(self._on_model_failed)
        self._worker = w
        w.start()

    def _on_model_done(self, key, out) -> None:
        if key != self._pending_key:
            return                            # stale (lattice changed)
        self._model_out = out
        self._model_key = key
        if self._chk_model.isChecked():
            self._draw_model(out)

    def _on_model_failed(self, key, msg: str) -> None:
        if key != self._pending_key:
            return
        self._lbl_status.setText(f"Model dispersion failed: {msg}")

    def _draw_model(self, out) -> None:
        s = np.asarray(out.get("s", []), dtype=float)
        dx = np.asarray(out.get("disp_x_m", []), dtype=float)
        dy = np.asarray(out.get("disp_y_m", []), dtype=float)
        self._cx_m.setData(s, dx, connect="finite")
        self._cy_m.setData(s, dy, connect="finite")
        self._lbl_status.setText(
            "Model overlay drawn (dashed)." if out.get("complete", True) else
            "Model curve partial — chain broke at an unsupported element.")

    def _on_lattice_changed(self, new_lattice) -> None:
        try:
            super()._on_lattice_changed(new_lattice)
            self._model_out = None
            self._model_key = None
            if self._chk_model.isChecked():
                self._ensure_model()
        except RuntimeError:
            # Popup already destroyed but the state signal outlived it
            # (house rule: bound methods + RuntimeError guard).
            pass

    def refresh(self, results):
        S = _sigma_stack(results)
        if S is None or S.size == 0:
            self._cx.setData([], []); self._cy.setData([], []); return
        s = np.asarray(results.s, dtype=float)
        beta  = np.asarray(getattr(results, "ref_beta",  []), dtype=float)
        gamma = np.asarray(getattr(results, "ref_gamma", []), dtype=float)
        mass  = float(getattr(results, "mass_mev", 0.0))
        if not mass and gamma.size and gamma[0] > 1 + 1e-6:
            # Fall back to recovering mass from (γ-1) and ref_w_kin.
            w0 = float(getattr(results, "ref_w_kin", [0.0])[0])
            mass = w0 / max(gamma[0] - 1.0, 1e-9)
        if not mass or beta.size == 0: return
        # σ[0,5] is ⟨x·ΔW⟩ in mm·MeV; σ[5,5] is ⟨ΔW²⟩ in MeV²
        # D_x [mm] = ⟨x·δ⟩/⟨δ²⟩ = (σ[0,5]/σ[5,5]) · β²γ·mc²
        denom = S[:, 5, 5]
        safe = denom > 1e-30
        factor = np.where(safe, (beta ** 2) * gamma * mass, 1.0)
        dx_mm = np.where(safe, S[:, 0, 5] / np.where(safe, denom, 1.0), 0.0) * factor
        dy_mm = np.where(safe, S[:, 2, 5] / np.where(safe, denom, 1.0), 0.0) * factor
        self._cx.setData(s, dx_mm * 1e-3)      # mm → m
        self._cy.setData(s, dy_mm * 1e-3)


class _DpPRmsPopup(_PopupPlot):
    """RMS momentum spread σ(Δp/p) along s -- the beam momentum
    spread normalised to the reference particle.

    Derivation: for small longitudinal offsets the dispersion relation
    E² = (pc)² + (mc²)² gives ``dp/p = dE/(β²γ·mc²)``, so for the
    RMS:

        σ(Δp/p)[i] = sigma_w[i] / (ref_beta[i]² · ref_gamma[i] · mass_mev)

    This is the *inverse* conversion the dispersion popup applies
    when going from σ-matrix cross terms to D_x [m]; reuses the same
    physics + the same fallback for ``mass_mev`` when it's not in the
    results (recover from γ and ref_w_kin).

    Dimensionless on the y axis -- typical values are 1e-4 to 1e-3
    for a proton beam at MeV energies.  Useful for matching design
    (longitudinal acceptance), dispersion gymnastics, and watching
    adiabatic damping shrink dp/p through accelerating sections.
    """

    def __init__(self, parent):
        super().__init__(parent, "Momentum spread  —  σ(Δp/p)",
                         size=(1000, 480))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._p = _mk_plot("σ(Δp/p)", "")
        self._c = self._p.plot(pen=curve_pen("#f97316"),     # orange
                               name="σ(Δp/p)")
        v.addWidget(self._p, stretch=1)
        self.attach_lattice_strip(self._p)

    def refresh(self, results):
        if results is None:
            self._c.setData([], []); return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        sigw = np.asarray(getattr(results, "sigma_w", []), dtype=float)
        beta = np.asarray(getattr(results, "ref_beta", []), dtype=float)
        gamma = np.asarray(getattr(results, "ref_gamma", []), dtype=float)
        if (s.size == 0 or sigw.size != s.size
                or beta.size != s.size or gamma.size != s.size):
            self._c.setData([], []); return
        mass = float(getattr(results, "mass_mev", 0.0))
        if not mass and gamma.size and gamma[0] > 1 + 1e-9:
            # Fall back to recovering mass from γ-1 and ref_w_kin
            # (same trick _DispersionPopup uses).
            w0 = float(getattr(results, "ref_w_kin", [0.0])[0])
            mass = w0 / max(gamma[0] - 1.0, 1e-9)
        if not mass:
            self._c.setData([], []); return
        # Avoid 0/0 at the entrance of a DC beam (β = 0 would NaN).
        denom = beta ** 2 * gamma * mass
        safe = denom > 1e-30
        dpp = np.where(safe, sigw / np.where(safe, denom, 1.0), 0.0)
        self._c.setData(s, dpp)


class _Emit6DPopup(_PopupPlot):
    """6-D normalised emittance ε_6D = ε_nx · ε_ny · ε_nz (mm³·mrad³)."""

    def __init__(self, parent):
        super().__init__(parent, "6-D emittance  —  ε_nx · ε_ny · ε_nz",
                         size=(1000, 420))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._p = _mk_plot("ε_6D", "mm³·mrad³")
        self._c = filled_curve(self._p, "#e879f9", name="ε_6D")
        v.addWidget(self._p, stretch=1)
        self.attach_lattice_strip(self._p)

    def refresh(self, results):
        if results is None: self._c.setData([], []); return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        nx = np.asarray(getattr(results, "emit_nx", []), dtype=float)
        ny = np.asarray(getattr(results, "emit_ny", []), dtype=float)
        nz = np.asarray(getattr(results, "emit_nz", []), dtype=float)
        # Envelope mode: derive ε_n = βγ · ε_geom for any plane that's missing.
        if not (s.size == nx.size == ny.size == nz.size) or s.size == 0:
            beta = np.asarray(getattr(results, "ref_beta", []), dtype=float)
            gamma = np.asarray(getattr(results, "ref_gamma", []), dtype=float)
            if beta.size == s.size and gamma.size == s.size:
                bg = beta * gamma
                ex = np.asarray(getattr(results, "emit_x", []), dtype=float)
                ey = np.asarray(getattr(results, "emit_y", []), dtype=float)
                ez = np.asarray(getattr(results, "emit_z_mmmrad", []), dtype=float)
                if ex.size == s.size:  nx = ex * bg
                if ey.size == s.size:  ny = ey * bg
                if ez.size == s.size:  nz = ez * bg
        if not (s.size == nx.size == ny.size == nz.size) or s.size == 0: return
        self._c.setData(s, nx * ny * nz)


class _Emit4DPopup(_PopupPlot):
    """4-D transverse invariant emittance ε_4D = √det(σ_4D).

    Conserved under linear coupling (solenoid rotation).  Spikes in the
    2-D ε_xx' / ε_yy' projections from solenoid fringes DO NOT appear
    here — they are purely a projection effect.  Real 4-D growth
    (nonlinear RF defocus, SC, aperture losses) shows up cleanly.
    """

    def __init__(self, parent):
        super().__init__(parent, "4-D invariant emittance  —  ε_4D = √det(σ_4D)",
                         size=(1000, 420))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        banner = QLabel(
            "ε_4D is conserved under linear transverse coupling (solenoid "
            "rotation) — unlike ε_x / ε_y, which oscillate with the rotation "
            "because they're 2-D projections of the 4-D phase-space ellipsoid."
        )
        banner.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px; padding:4px 8px;"
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_0};"
            f"border-radius:3px;"
        )
        banner.setWordWrap(True)
        v.addWidget(banner)
        self._p = _mk_plot("ε_4D", "(mm·mrad)²")
        self._c = filled_curve(self._p, "#a78bfa", name="ε_4D")
        v.addWidget(self._p, stretch=1)
        self.attach_lattice_strip(self._p)

    def refresh(self, results):
        if results is None: self._c.setData([], []); return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        e = np.asarray(getattr(results, "emit_4d", []), dtype=float)
        if s.size and e.size == s.size:
            self._c.setData(s, e)


class _EigenEmitPopup(_PopupPlot):
    """6-D eigenemittances ε₁, ε₂, ε₃ (Balandin invariants).

    Constants of motion under any linear symplectic transport — including
    the dispersive, x-y, and longitudinally-coupled regimes where ε_x /
    ε_y / ε_z visibly oscillate.  In an uncoupled lattice they reduce
    to (ε_x, ε_y, ε_z); under coupling the projections wobble but the
    eigenemittances stay flat.  Use these to distinguish *coupling*
    (constant ε_i, oscillating ε_x/y/z) from *real phase-space growth*
    (rising ε_i, e.g. SC mismatch or RF non-linearity).
    """

    def __init__(self, parent):
        super().__init__(parent, "Eigenemittances  —  ε₁ · ε₂ · ε₃ (6-D, Balandin)",
                         size=(1000, 420))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        banner = QLabel(
            "Eigenemittances are invariants of any linear symplectic "
            "transport.  Flat ε₁/₂/₃ + oscillating ε_x/ε_y/ε_z = pure "
            "coupling (no growth).  Rising ε₁/₂/₃ = real 6-D phase-space "
            "growth (SC mismatch, RF non-linearity, aperture losses)."
        )
        banner.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px; padding:4px 8px;"
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_0};"
            f"border-radius:3px;"
        )
        banner.setWordWrap(True)
        v.addWidget(banner)
        self._p = _mk_plot("ε_i", "mm·mrad / deg·MeV")
        self._p.addLegend(offset=(10, 10))
        self._c1 = self._p.plot(pen=pg.mkPen("#c084fc", width=2), name="ε₁ (≈ε_x)")
        self._c2 = self._p.plot(pen=pg.mkPen("#a3e635", width=2), name="ε₂ (≈ε_y)")
        self._c3 = self._p.plot(pen=pg.mkPen("#fbbf24", width=2), name="ε₃ (≈ε_z)")
        v.addWidget(self._p, stretch=1)
        self.attach_lattice_strip(self._p)

    def refresh(self, results):
        if results is None:
            self._c1.setData([], []); self._c2.setData([], []); self._c3.setData([], [])
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        e1 = np.asarray(getattr(results, "emit_e1", []), dtype=float)
        e2 = np.asarray(getattr(results, "emit_e2", []), dtype=float)
        e3 = np.asarray(getattr(results, "emit_e3", []), dtype=float)
        if s.size and e1.size == s.size:
            self._c1.setData(s, e1)
        else:
            self._c1.setData([], [])
        if s.size and e2.size == s.size:
            self._c2.setData(s, e2)
        else:
            self._c2.setData([], [])
        if s.size and e3.size == s.size:
            self._c3.setData(s, e3)
        else:
            self._c3.setData([], [])


class _BeamPowerPopup(_PopupPlot):
    """Beam power P = I · W_kin along the lattice.

    Uses beam.current and the recorded reference kinetic energy per step;
    losses reduce the power proportionally via transmission.
    """

    def __init__(self, parent, state: AppState):
        super().__init__(parent, "Beam power  —  P = I · W × transmission",
                         size=(1000, 420))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._p = _mk_plot("P", "W")
        self._c = filled_curve(self._p, "#4ade80", name="P [W]")
        v.addWidget(self._p, stretch=1)
        self.attach_lattice_strip(self._p)

    def refresh(self, results):
        if results is None: self._c.setData([], []); return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        w = np.asarray(getattr(results, "ref_w_kin", []), dtype=float)
        t = np.asarray(getattr(results, "transmission", []), dtype=float)
        cfg = self._state.beam_config
        I_mA = float(getattr(cfg, "current", 0.0)) if cfg is not None else 0.0
        if s.size == 0 or w.size != s.size or I_mA <= 0:
            self._c.setData([], []); return
        # Envelope mode doesn't track transmission — assume 100% so the
        # P = I·W·(t/100) curve is well-defined.
        if t.size != s.size:
            t = np.full_like(s, 100.0)
        # P [W] = I [A] · W [MeV] · 1e6 · transmission/100
        P = (I_mA * 1e-3) * (w * 1e6) * (t / 100.0)
        self._c.setData(s, P)


class _NormEmittancePopup(_PopupPlot):
    """Normalised RMS emittance ε_nx, ε_ny, ε_nz in mm·mrad.

    Convention used:
        ε_n = (βγ) · ε_geom                  (transverse)
        ε_nz = (βγ) · ε_z,mm·mrad            (longitudinal, [z, δ=Δp/p])
    The longitudinal geometric emittance is the TraceWin ``ε_zδ`` — the
    native (Δφ, ΔW) emittance converted to (z, δ = Δp/p) via the
    Jacobian (β·λ / 360) × (1 / β²γ·mc²), then multiplied by βγ to
    normalise.  That is why ε_nz is not 1-to-1 comparable to the input
    ``emit_z`` (which is in deg·MeV — a different phase-space pair).
    """

    def __init__(self, parent):
        super().__init__(parent,
                         "Normalised RMS emittance  —  ε_nx · ε_ny · ε_nz  (mm·mrad)",
                         size=(1000, 700))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        convention = QLabel(
            "Convention:  ε_n = (βγ)·ε_geom   (all three planes).\n"
            "Longitudinal ε_z is in the (z, δ=Δp/p) basis, so ε_nz is "
            "NOT directly comparable to BeamConfig.emit_z (which is in deg·MeV)."
        )
        convention.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px; padding:4px 8px;"
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_0};"
            f"border-radius:3px;"
        )
        v.addWidget(convention)

        # Mode toggle row
        from PyQt6.QtWidgets import QHBoxLayout
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Display:"))
        self._mode_combo = _mk_mode_combo(lambda _i: self._redraw())
        ctrl.addWidget(self._mode_combo); ctrl.addStretch(1)
        v.addLayout(ctrl)
        self._last_results = None

        self._px = _mk_plot("ε_nx", "mm·mrad")
        self._py = _mk_plot("ε_ny", "mm·mrad")
        self._pz = _mk_plot("ε_nz", "mm·mrad")
        for p in (self._py, self._pz):
            p.setXLink(self._px)
        self._cx = filled_curve(self._px, theme.ACCENT, name="ε_nx")
        self._cy = filled_curve(self._py, "#a3e635",    name="ε_ny")
        self._cz = filled_curve(self._pz, "#fbbf24",    name="ε_nz")
        for p in (self._px, self._py, self._pz):
            v.addWidget(p, stretch=1)
        self.attach_lattice_strip(self._px)

    def refresh(self, results):
        self._last_results = results
        self._redraw()

    def _redraw(self):
        results = self._last_results
        if results is None:
            for c in (self._cx, self._cy, self._cz): c.setData([], [])
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        def arr(a): return np.asarray(getattr(results, a, []), dtype=float)
        def pair(c, v):
            if s.size and v.size == s.size: c.setData(s, v)

        corrected = self._mode_combo.currentIndex() == 1
        if corrected:
            # Use the betatron 4×4 Σ sub-block to compute ε_x,β / ε_y,β
            # and then normalise via βγ.  ε_nz is z-plane → leave raw
            # (z is already the dispersive plane).
            S = _sigma_stack(results)
            beta = arr("ref_beta")
            gamma = arr("ref_gamma")
            if (S is not None and S.size and S.shape[0] == s.size
                    and beta.size == s.size and gamma.size == s.size):
                sxx, sxxp, sxpxp, syy, syyp, sypyp = _betatron_sigma_blocks(S)
                ex_b = np.sqrt(np.clip(sxx * sxpxp - sxxp ** 2, 0.0, None))
                ey_b = np.sqrt(np.clip(syy * sypyp - syyp ** 2, 0.0, None))
                bg = beta * gamma
                nx = ex_b * bg
                ny = ey_b * bg
            else:
                # No Σ recorded → fall back to raw silently
                nx = arr("emit_nx")
                ny = arr("emit_ny")
            nz = arr("emit_nz")
            if nz.size != s.size:
                ezmm = arr("emit_z_mmmrad")
                if beta.size == s.size and gamma.size == s.size and ezmm.size == s.size:
                    nz = ezmm * beta * gamma
        else:
            nx = arr("emit_nx"); ny = arr("emit_ny"); nz = arr("emit_nz")
            # Envelope mode does not record normalised emittances; derive on
            # the fly from geometric ε × βγ so the plot still works.
            if nx.size != s.size or ny.size != s.size or nz.size != s.size:
                beta = arr("ref_beta")
                gamma = arr("ref_gamma")
                if beta.size == s.size and gamma.size == s.size:
                    bg = beta * gamma
                    ex = arr("emit_x")
                    ey = arr("emit_y")
                    ezmm = arr("emit_z_mmmrad")
                    if ex.size == s.size:  nx = ex * bg
                    if ey.size == s.size:  ny = ey * bg
                    if ezmm.size == s.size: nz = ezmm * bg

        pair(self._cx, nx)
        pair(self._cy, ny)
        pair(self._cz, nz)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpers to extract scalar "lattice-parameter" values out of a FieldMap /
# FieldMap3D element.  These are used by the E_acc, V_gap and B_peak
# popups so that lattices built from FIELD_MAP cards (not RFGap / SOLENOID
# lumped elements) still populate those charts.  Raw electric / magnetic
# field data is stored in ``el.field_data.channels``; the physical
# amplitude is reached by multiplying by ``el.ke`` (or ``el.kb``) and
# dividing by ``ch.norm_factor`` — the same rule the tracker uses inside
# ``FieldMap._scale``.
# ---------------------------------------------------------------------------
def _fieldmap_onaxis_fz(channel) -> np.ndarray | None:
    """Return the on-axis (x=y=r=0) Fz samples as a 1-D numpy array for a
    field-map channel of any geometry, or None if unavailable."""
    if channel is None or channel.Fz is None:
        return None
    fz = np.asarray(channel.Fz)
    g = int(getattr(channel, "geometry", 0) or 0)
    if g == 1:
        return fz.reshape(-1)
    if g in (4, 5):
        # 2-D cyl (r, z) — on-axis is the r = 0 slice, index 0 along the r
        # axis.  Grid shape may be (n_r, n_z) or (n_z, n_r) depending on
        # how the reader laid it out; take whichever slicing gives the
        # same length as the z-axis.
        z = channel.z
        if z is None:
            return None
        if fz.ndim == 2:
            if fz.shape[0] == len(z):
                return fz[:, 0].reshape(-1)     # layout (n_z, n_r)
            return fz[0, :].reshape(-1)          # layout (n_r, n_z)
        return fz.reshape(-1)
    if g == 7:
        # 3-D Cart (x, y, z) — on-axis is the x=0, y=0 pillar.  Find the
        # mid-index along each transverse axis and slice.
        x = channel.x; y = channel.y
        if x is None or y is None:
            return fz.reshape(-1)
        ix = int(np.argmin(np.abs(x)))
        iy = int(np.argmin(np.abs(y)))
        if fz.ndim == 3:
            # Shape can be (n_x, n_y, n_z) or (n_z, n_y, n_x) — pick the
            # axis with length == len(z).
            z = channel.z
            if z is not None:
                nz = len(z)
                if fz.shape[2] == nz:
                    return fz[ix, iy, :].reshape(-1)
                if fz.shape[0] == nz:
                    return fz[:, iy, ix].reshape(-1)
            return fz[ix, iy, :].reshape(-1)
        return fz.reshape(-1)
    return None


def _fieldmap_efield_channel(el):
    """Return the first electric (non-quad-gradient) channel of a field map
    element, or None if it carries no accelerating E-field."""
    fd = getattr(el, "field_data", None)
    if fd is None:
        return None
    for ch_enum, ch in fd.channels.items():
        if not ch_enum.is_electric:
            continue
        if int(getattr(ch, "geometry", 0) or 0) == 9:
            continue          # quad gradient has no axial E
        return ch
    return None


def _fieldmap_bfield_channel(el):
    """Return the first magnetic channel of a field map element (used for
    solenoid peak-B plots).  None if the element has no B channel."""
    fd = getattr(el, "field_data", None)
    if fd is None:
        return None
    for ch_enum, ch in fd.channels.items():
        if ch_enum.is_electric:
            continue
        if int(getattr(ch, "geometry", 0) or 0) == 9:
            continue
        return ch
    return None


def _fieldmap_eacc_MV_per_m(el) -> float | None:
    """Peak axial accelerating gradient E_acc = |ke| · max|E_z_axis| /
    |norm_factor|, in MV/m.  Returns None if no E-channel present.

    The absolute values matter: TraceWin uses negative ``ke`` (and
    occasionally negative ``norm_factor``) to encode a cavity polarity
    flip — the physical *amplitude* is always positive, and the user
    expects the plotted bar to reflect that amplitude.
    """
    ch = _fieldmap_efield_channel(el)
    fz = _fieldmap_onaxis_fz(ch)
    if fz is None or fz.size == 0:
        return None
    ke = float(getattr(el, "ke", 1.0))
    scale_global = float(getattr(el, "scale", 1.0))
    norm = float(getattr(ch, "norm_factor", 1.0) or 1.0)
    return abs(ke * scale_global / norm) * float(np.max(np.abs(fz)))


def _fieldmap_vgap_MV(el) -> float | None:
    """Integrated (peak) voltage V_0 = |ke| · ∫|E_z_axis|·dz / |norm_factor|,
    in MV.  Uses the on-axis E_z samples and trapezoidal integration
    across the element length.  See :func:`_fieldmap_eacc_MV_per_m`
    for why the scaling factors are taken in absolute value.
    """
    ch = _fieldmap_efield_channel(el)
    fz = _fieldmap_onaxis_fz(ch)
    if fz is None or fz.size < 2:
        return None
    ke = float(getattr(el, "ke", 1.0))
    scale_global = float(getattr(el, "scale", 1.0))
    norm = float(getattr(ch, "norm_factor", 1.0) or 1.0)
    L_m = float(getattr(el, "length", 0.0)) * 1e-3       # mm → m
    if L_m <= 0:
        return None
    dz = L_m / (fz.size - 1)
    # NumPy 2.0 dropped ``trapz`` in favour of ``trapezoid``; fall back so
    # this works on both.
    _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    integ = float(_trap(np.abs(fz), dx=dz))              # (MV/m)·m = MV
    return abs(ke * scale_global / norm) * integ


def _fieldmap_bpeak_T(el) -> float | None:
    """Peak axial |B_z| for a magnetic-only field map, = |kb| · max|B_z| /
    |norm_factor| in tesla.  Returns None if the element has an E channel
    (RF cavity) — those belong on the E_acc chart, not this one.  See
    :func:`_fieldmap_eacc_MV_per_m` for the rationale on absolute values.
    """
    if _fieldmap_efield_channel(el) is not None:
        return None
    ch = _fieldmap_bfield_channel(el)
    fz = _fieldmap_onaxis_fz(ch)
    if fz is None or fz.size == 0:
        return None
    kb = float(getattr(el, "kb", 1.0))
    scale_global = float(getattr(el, "scale", 1.0))
    norm = float(getattr(ch, "norm_factor", 1.0) or 1.0)
    return abs(kb * scale_global / norm) * float(np.max(np.abs(fz)))


def _fieldmap_int_b2(el) -> float | None:
    """Solenoid focusing strength ∫B_z²·dz [T²·m] from the on-axis field map.

    The longitudinal integral of B_z² sets a solenoid's transverse focusing
    (the thin-lens kick ∝ (∫B·dz/2Bρ)² for a hard edge, ∝ ∫B²dz for the
    distributed field), so it's the field-map-shape-aware measure of strength.
    Returns None for RF cavities (those have an E channel) and for maps with no
    magnetic channel.  Scaling (kb·scale/norm) matches ``_fieldmap_bpeak_T``.
    """
    if _fieldmap_efield_channel(el) is not None:
        return None
    ch = _fieldmap_bfield_channel(el)
    fz = _fieldmap_onaxis_fz(ch)
    if fz is None or fz.size < 2:
        return None
    kb = float(getattr(el, "kb", 1.0))
    scale_global = float(getattr(el, "scale", 1.0))
    norm = float(getattr(ch, "norm_factor", 1.0) or 1.0)
    L_m = float(getattr(el, "length", 0.0)) * 1e-3       # mm → m
    if L_m <= 0:
        return None
    dz = L_m / (fz.size - 1)
    bz = abs(kb * scale_global / norm) * np.abs(np.asarray(fz, dtype=float))  # |B_z(z)| [T]
    _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trap(bz * bz, dx=dz))                  # T²·m


def _solenoid_int_b2(el) -> float | None:
    """∫B_z²·dz [T²·m] for any solenoid element.

    A lumped ``Solenoid`` is hard-edge uniform (B₀ over its length), so
    ∫B²·dz = B₀²·L; field-map solenoids integrate the on-axis B_z(z)² profile
    via :func:`_fieldmap_int_b2`.  Returns None for non-solenoid elements
    (RF cavities, quads, …) so the popup plots only solenoids.
    """
    from linac_gen.elements.solenoid import Solenoid
    if isinstance(el, Solenoid):
        b0 = abs(float(getattr(el, "field", 0.0)))
        L_m = float(getattr(el, "length", 0.0)) * 1e-3
        return b0 * b0 * L_m if L_m > 0 else None
    return _fieldmap_int_b2(el)


def _ncells_v0_MV(el) -> float | None:
    """Total effective cavity voltage V₀ = Σ|EoT·Lc| per gap [MV] for NCELLS,
    including any ERROR_CAV amplitude factor (mirrors how the RFGap chart
    shows the errored ``voltage``).  βg≤0 cavities have no gap list until a
    run resolves their geometry — None until then (skipped by the plots)."""
    gaps = getattr(el, "_gaps", None)
    if not gaps:
        return None
    fac = 1.0 + float(getattr(el, "voltage_rel", 0.0) or 0.0)
    return float(sum(abs(g.voltage_mv) for g in gaps)) * abs(fac)


def _ncells_phase_deg(el) -> float | None:
    """Per-cavity RF phase for the φ_s chart (wrapped to [−180°, 180°)).

    * SET_SYNC_PHASE / P=0 / P=2 : θs (+ any ERROR_CAV phase offset) IS the
      synchronous/relative phase — plot it directly.
    * P=1 (absolute)             : θs is the raw RF-clock setting (a
      meaningless ramp on a φ_s chart, e.g. 62/242/422/… on fnalscl) — plot
      the RUN-RESOLVED phase the beam actually saw at gap 1,
      wrap(φ_clock(gap 1) − θs − offset), which the element snapshots during
      tracking.  None before any run so the raw ramp is never shown.
    """
    theta = float(getattr(el, "theta_s_deg", 0.0))
    off = float(getattr(el, "phase_offset", 0.0) or 0.0)
    if getattr(el, "sync_phase", False) or getattr(el, "p_flag", 0) != 1:
        return ((theta + off + 180.0) % 360.0) - 180.0
    gap1 = float(getattr(el, "_phi_s_at_gap1", 0.0) or 0.0)
    if gap1 == 0.0 and getattr(el, "_step_idx", 0) == 0:
        return None                          # P=1 with no run yet
    return ((gap1 - theta - off + 180.0) % 360.0) - 180.0


def _rf_phase_value(el) -> float | None:
    """Dispatch for the synchronous-phase chart: NCELLS via
    :func:`_ncells_phase_deg`, everything else via its flat ``phase``."""
    from linac_gen.elements.ncells import NCells
    if isinstance(el, NCells):
        return _ncells_phase_deg(el)
    v = getattr(el, "phase", None)
    return float(v) if v is not None else None


class _LatticeParamPopup(_PopupPlot):
    """Per-element lattice parameter plot: stems at each element's s_mid.

    Used for quad gradient, gradient-integral, RF voltage, synch phase,
    Eacc — any attribute that decorates a subset of the lattice elements.
    Gracefully shows a "no data" label when no element of the requested
    type is present in the lattice.
    """

    def __init__(self, parent, state: AppState,
                 title: str,
                 element_types: tuple[type, ...],
                 attr: str,
                 ylabel: str, yunits: str,
                 color: str = theme.ACCENT,
                 transform=None,
                 type_name: str = "matching",
                 value_fn=None):
        super().__init__(parent, title, size=(1000, 480))
        self._state = state
        self._types = element_types
        self._attr  = attr
        self._transform = transform
        self._type_name = type_name
        self._color = color
        # ``value_fn(element) -> float | None`` takes precedence over
        # ``getattr(element, attr)`` when provided.  Used for FieldMap /
        # FieldMap3D popups where the relevant scalar (peak E_z, peak B_z,
        # V_gap = ∫E_z·dz) has to be computed from the field-map grid,
        # not read off a flat Python attribute.
        self._value_fn = value_fn
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._plot = _mk_plot(ylabel, yunits)
        v.addWidget(self._plot, stretch=1)
        # Overlay a large centred "no data" label when the lattice has no
        # matching elements — hidden once data shows up.
        self._empty_lbl = QLabel("", self._plot.getViewBox().scene().views()[0])
        self._empty_lbl.setStyleSheet(
            f"color:{theme.TEXT_3}; font-size:14px; letter-spacing:1px;"
            f"background:transparent; padding:24px;"
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.hide()
        self.attach_lattice_strip(self._plot)

    def refresh(self, results):
        # Data comes from the **lattice**, not the recorded results — the
        # values are element attributes set at load time.  We still refresh
        # when results change because lattice may have been loaded after.
        lat = getattr(self._state, "lattice", None)
        self._plot.clear()
        if lat is None:
            self._show_empty(f"No lattice loaded")
            return
        s_mids, values, lengths, names = [], [], [], []
        s_cursor = 0.0
        for el in lat.elements:
            L = float(getattr(el, "length", 0.0) or 0.0)
            if isinstance(el, self._types):
                if self._value_fn is not None:
                    try:
                        val = self._value_fn(el)
                    except Exception:
                        val = None
                else:
                    val = getattr(el, self._attr, None)
                if val is not None:
                    if self._transform is not None:
                        val = self._transform(el, val)
                    s_mids.append(s_cursor + 0.5 * L)
                    values.append(float(val))
                    lengths.append(max(L, 10.0))  # min bar width so thin lenses show
                    names.append(getattr(el, "name", ""))
            s_cursor += L
        if not s_mids:
            self._show_empty(f"No {self._type_name} elements in the current lattice")
            return
        self._empty_lbl.hide()
        # Bar chart with element-length-sized bars
        import pyqtgraph as pg
        r, g, b = _hex_to_rgb(self._color)
        bar = pg.BarGraphItem(
            x=s_mids, height=values, width=lengths,
            brush=pg.mkBrush(r, g, b, 160),
            pen=pg.mkPen(self._color, width=1.2),
        )
        self._plot.addItem(bar)
        # Marker on top for readability
        sc = pg.ScatterPlotItem(
            x=s_mids, y=values,
            pen=pg.mkPen(self._color, width=1.2),
            brush=pg.mkBrush(r, g, b, 230),
            size=8, symbol="o",
        )
        self._plot.addItem(sc)
        self._plot.autoRange()

    def _show_empty(self, msg: str) -> None:
        self._empty_lbl.setText(msg)
        self._empty_lbl.adjustSize()
        # Centre over the plot widget
        geom = self._plot.geometry()
        self._empty_lbl.setGeometry(
            geom.x(), geom.y() + geom.height() // 2 - 30,
            geom.width(), 60,
        )
        self._empty_lbl.show()
        self._empty_lbl.raise_()


def _hex_to_rgb(color: str):
    c = color.lstrip("#")
    return int(c[:2], 16), int(c[2:4], 16), int(c[4:6], 16)


# ---------------------------------------------------------------------------
class _TwissPopup(_PopupPlot):
    def __init__(self, parent):
        super().__init__(parent, "Twiss  —  α · β", size=(1000, 620))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Mode toggle row
        from PyQt6.QtWidgets import QHBoxLayout
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Display:"))
        self._mode_combo = _mk_mode_combo(lambda _i: self._redraw())
        ctrl.addWidget(self._mode_combo); ctrl.addStretch(1)
        v.addLayout(ctrl)
        self._last_results = None

        self._pa_x = _mk_plot("α_x")
        self._pb_x = _mk_plot("β_x", "mm/mrad")
        self._pa_y = _mk_plot("α_y")
        self._pb_y = _mk_plot("β_y", "mm/mrad")
        for p in (self._pb_x, self._pa_y, self._pb_y):
            p.setXLink(self._pa_x)
        self._cax = self._pa_x.plot(pen=curve_pen(theme.ACCENT))
        self._cbx = self._pb_x.plot(pen=curve_pen(theme.ACCENT))
        self._cay = self._pa_y.plot(pen=curve_pen("#a3e635"))
        self._cby = self._pb_y.plot(pen=curve_pen("#a3e635"))
        for p in (self._pa_x, self._pb_x, self._pa_y, self._pb_y):
            v.addWidget(p, stretch=1)
        self.attach_lattice_strip(self._pa_x)

    def refresh(self, results):
        self._last_results = results
        self._redraw()

    def _redraw(self):
        results = self._last_results
        if results is None:
            for c in (self._cax, self._cbx, self._cay, self._cby): c.setData([], [])
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        def arr(a): return np.asarray(getattr(results, a, []), dtype=float)
        def pair(c, v):
            if s.size and v.size == s.size: c.setData(s, v)
        if self._mode_combo.currentIndex() == 1:
            # Compute α_β, β_β from the betatron 2×2 sub-blocks via
            # _twiss_from_block — same routine the dispersed code uses
            # internally on the raw Σ.
            S = _sigma_stack(results)
            if S is not None and S.size and S.shape[0] == s.size:
                sxx, sxxp, sxpxp, syy, syyp, sypyp = _betatron_sigma_blocks(S)
                ax, bx, _, _ = _twiss_from_block(sxx, sxxp, sxpxp)
                ay, by, _, _ = _twiss_from_block(syy, syyp, sypyp)
                pair(self._cax, ax); pair(self._cbx, bx)
                pair(self._cay, ay); pair(self._cby, by)
                return
            # No Σ recorded → silent fallback to raw
        pair(self._cax, arr("alpha_x")); pair(self._cbx, arr("beta_x"))
        pair(self._cay, arr("alpha_y")); pair(self._cby, arr("beta_y"))


class _StructWorker(_QThread := __import__("PyQt6.QtCore", fromlist=["QThread"]).QThread):
    """Background thread that computes the (cache-miss) structure σ₀
    curves so the GUI thread doesn't freeze on a 30 s lattice walk.

    Emits ``finished_signal(key_tuple, curves_dict, sigma0_dict)`` on
    success and ``failed_signal(key_tuple, msg)`` on failure.  The
    caller passes a ``key_tuple`` it can use to ignore stale results
    (e.g. user changed period mid-flight).
    """
    from PyQt6.QtCore import pyqtSignal as _Signal
    finished_signal = _Signal(object, object, object)
    failed_signal   = _Signal(object, str)

    def __init__(self, lattice, ref, period, key, fallback_seed=None,
                 matrix_cache=None):
        super().__init__()
        # numpy/BLAS on the 544 KB default macOS QThread stack → SIGBUS
        # (house pattern — see workers._MatchWorker).
        self.setStackSize(16 * 1024 * 1024)
        self._lattice = lattice
        self._ref = ref
        self._period = period
        self._key = key
        self._fallback_seed = fallback_seed
        # Opt-in per-element transfer-matrix cache (see
        # linac_gen.tracking.matrix_tracking.get_element_matrix).  Lives
        # on the popup and is replaced when lattice or beam_config
        # changes.  Threading-safe because each worker owns its own
        # reference to the dict, and we only read+write entries that
        # have a fingerprint matching the current ref state — no
        # cross-thread mutation of the same entry.
        self._matrix_cache = matrix_cache
        # Cooperative stop — popup close / app teardown sets this; the
        # three core walks poll it per element / per cell.
        import threading
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self):
        from linac_gen.core.cancelled import OperationCancelled
        try:
            from linac_gen.analysis.phase_advance import (
                structure_phase_advance, structure_phase_advance_along_s,
                coupled_phase_advance_along_s,
            )
            cache = self._matrix_cache
            sigma0 = structure_phase_advance(
                self._lattice, self._ref, self._period, cache=cache,
                should_stop=self._stopping,
            )
            # Per-plane fallback: any plane whose periodic Twiss came
            # back None (because the period was unstable or coupled in
            # that pair) falls back to the user-supplied beam_config
            # initial Twiss so the user always gets a curve.
            seed_for_curves = dict(sigma0)
            used_planes: list[str] = []
            if self._fallback_seed is not None:
                for plane in ("x", "y", "z"):
                    if seed_for_curves.get(f"alpha_{plane}") is None:
                        seed_for_curves[f"alpha_{plane}"] = (
                            self._fallback_seed.get(f"alpha_{plane}")
                        )
                        seed_for_curves[f"beta_{plane}"] = (
                            self._fallback_seed.get(f"beta_{plane}")
                        )
                        used_planes.append(plane)
            curves = structure_phase_advance_along_s(
                self._lattice, self._ref, self._period, seed=seed_for_curves,
                cache=cache, should_stop=self._stopping,
            )
            curves["fallback_planes"] = used_planes
            curves["used_fallback_seed"] = bool(used_planes)
            # Coupled along-s for the eigenmode μ_I, μ_II curves —
            # only meaningful when the period is xy-coupled
            # (solenoid + HWR cavity).
            if sigma0.get("coupled_xy"):
                try:
                    coupled = coupled_phase_advance_along_s(
                        self._lattice, self._ref, self._period,
                        cache=cache, should_stop=self._stopping,
                    )
                    curves["coupled_along_s"] = coupled
                except OperationCancelled:
                    raise   # never swallow a cancel into a ⅔ result
                except Exception:                              # noqa: BLE001
                    pass
            self.finished_signal.emit(self._key, curves, sigma0)
        except OperationCancelled:
            # Cancelled — emit NOTHING: a partial result delivered via
            # finished_signal would be cached as valid σ₀ data.
            return
        except Exception as exc:                              # noqa: BLE001
            self.failed_signal.emit(self._key, str(exc))


class _PhaseAdvancePopup(_PopupPlot):
    """Cumulative μ(s) plots — structure σ₀ from periodic Twiss
    propagation, beam σ from envelope-output β(s).

    Three stacked plots (x · y · z); each plot has *two* y-axes —
    structure σ₀ on the left (cyan) and beam σ on the right (lime) —
    so the two curves don't visually collapse when they have very
    different magnitudes (e.g. small RF-driven σ_z vs large structure
    σ₀_z over many cells, or vice-versa under heavy SC depression).

    Auto-detects period candidates from the loaded lattice; the
    selected one drives both the seed Twiss and the dashed period
    markers.
    """

    def __init__(self, parent, state):
        super().__init__(parent, "Phase advance σ₀ · σ", size=(1600, 920))
        from PyQt6.QtWidgets import QComboBox, QPushButton, QGridLayout
        self._state = state
        self._periods: list = []
        # Cache of structure σ₀ curves (slow — full lattice walk).  Keyed
        # on (lattice id, period start/inner, ref tuple) so a results-only
        # refresh doesn't trigger recomputation.
        self._struct_cache: dict = {}
        # Per-element transfer-matrix cache lives on AppState
        # (``state.matrix_cache``) — shared across tabs and the
        # background warmer.  We just read it at use-time.
        # Subscribe to state changes directly so this popup stays in
        # sync even if the parent results-tab refresh loop misses us.
        if state is not None:
            try:
                state.results_changed.connect(self._on_results_changed)
            except Exception:                                # noqa: BLE001
                pass
            try:
                state.beam_config_changed.connect(self._on_beam_changed)
            except Exception:                                # noqa: BLE001
                pass
            try:
                state.lattice_changed.connect(self._on_lattice_changed)
            except Exception:                                # noqa: BLE001
                pass

        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)

        # Period picker row
        pr = QHBoxLayout(); pr.setSpacing(8)
        pr.addWidget(QLabel("Period:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(280)
        self._combo.currentIndexChanged.connect(self.refresh_with_state)
        pr.addWidget(self._combo, stretch=1)
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self.refresh_with_state)
        pr.addWidget(recompute)
        self._info = QLabel("")
        self._info.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;"
        )
        self._info.setWordWrap(True)
        pr.addWidget(self._info)
        v.addLayout(pr)

        # Column headers
        hdr = QGridLayout(); hdr.setSpacing(8)
        h0 = QLabel("STRUCTURE  σ₀(s)"); h0.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h0.setStyleSheet(f"color:{theme.ACCENT}; font-weight:600; font-size:11px; "
                         f"letter-spacing:1px;")
        hb = QLabel("BEAM  σ(s)"); hb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hb.setStyleSheet(f"color:#a3e635; font-weight:600; font-size:11px; "
                         f"letter-spacing:1px;")
        hp0 = QLabel("PER-CELL  Δσ₀"); hp0.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hp0.setStyleSheet(f"color:{theme.ACCENT}; font-weight:600; font-size:11px; "
                          f"letter-spacing:1px;")
        hpb = QLabel("PER-CELL  Δσ"); hpb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hpb.setStyleSheet(f"color:#a3e635; font-weight:600; font-size:11px; "
                          f"letter-spacing:1px;")
        hdr.addWidget(h0, 0, 0); hdr.addWidget(hb, 0, 1)
        hdr.addWidget(hp0, 0, 2); hdr.addWidget(hpb, 0, 3)
        for c in range(4):
            hdr.setColumnStretch(c, 1)
        v.addLayout(hdr)

        # 3 (planes) × 4 (σ₀(s) | σ(s) | Δσ₀/cell | Δσ/cell) grid.
        # One curve per plot — no overlay, no crowding.
        grid = QGridLayout(); grid.setSpacing(6)
        for c in range(4):
            grid.setColumnStretch(c, 1)
        self._rows: dict[str, dict] = {}
        for r, (plane, title) in enumerate((("x", "μ_x"), ("y", "μ_y"), ("z", "μ_z"))):
            p_struct = _mk_plot(f"{title} σ₀", "deg")
            p_beam   = _mk_plot(f"{title} σ",  "deg")
            p_per0   = _mk_plot(f"{title} Δσ₀", "deg"); p_per0.setLabel("bottom", "cell #")
            p_perb   = _mk_plot(f"{title} Δσ",  "deg"); p_perb.setLabel("bottom", "cell #")
            c0 = p_struct.plot(pen=curve_pen(theme.ACCENT, width=2.0),
                                name=f"{title} σ₀ (structure)")
            cb = p_beam.plot(pen=curve_pen("#a3e635", width=2.0),
                              name=f"{title} σ (beam)")
            c0_pp = p_per0.plot(pen=curve_pen(theme.ACCENT, width=2.0),
                                 symbol="o", symbolSize=7,
                                 symbolBrush=theme.ACCENT, symbolPen=theme.ACCENT,
                                 name=f"{title} Δσ₀")
            cb_pp = p_perb.plot(pen=curve_pen("#a3e635", width=2.0),
                                 symbol="s", symbolSize=7,
                                 symbolBrush="#a3e635", symbolPen="#a3e635",
                                 name=f"{title} Δσ")
            grid.addWidget(p_struct, r, 0)
            grid.addWidget(p_beam,   r, 1)
            grid.addWidget(p_per0,   r, 2)
            grid.addWidget(p_perb,   r, 3)
            self._rows[plane] = {
                "plot_struct": p_struct, "plot_beam": p_beam,
                "plot_per0": p_per0, "plot_perb": p_perb,
                "c0": c0, "cb": cb, "c0_pp": c0_pp, "cb_pp": cb_pp,
            }
        # Link s-axis plots; per-cell plots share their own cell-# x-axis.
        anchor = self._rows["x"]["plot_struct"]
        for plane in ("x", "y", "z"):
            for key in ("plot_struct", "plot_beam"):
                if self._rows[plane][key] is not anchor:
                    self._rows[plane][key].setXLink(anchor)
        anchor_pp0 = self._rows["x"]["plot_per0"]
        for plane in ("x", "y", "z"):
            for key in ("plot_per0", "plot_perb"):
                if self._rows[plane][key] is not anchor_pp0:
                    self._rows[plane][key].setXLink(anchor_pp0)
        v.addLayout(grid, stretch=1)

        # Holders for period boundary InfiniteLines (cleared on each refresh).
        self._period_marks: list = []

        self._populate_periods()

    # ------------------------------------------------------------------
    def _populate_periods(self) -> None:
        from linac_gen.analysis.period_detect import detect_periods
        self._combo.blockSignals(True)
        self._combo.clear()
        self._periods = []
        lattice = self._state.lattice if self._state else None
        if lattice is not None:
            try:
                self._periods = detect_periods(lattice)
            except Exception:                                # noqa: BLE001
                self._periods = []
        if not self._periods:
            self._combo.addItem("(no lattice)")
        else:
            for p in self._periods:
                tag = {
                    "lattice_card":            "[LATTICE]",
                    "lattice_card_recovered":  "[LATTICE+]",
                    "type_sequence":           "[auto]",
                    "fallback":                "[full]",
                }.get(p.source, "[?]")
                self._combo.addItem(f"{tag}  {p.label}")
        self._combo.blockSignals(False)

    def refresh_with_state(self, *_args) -> None:
        """Convenience wrapper for buttons / signals."""
        self.refresh(self._state.results if self._state else None)

    def _on_results_changed(self, results) -> None:
        """Handle state.results_changed — auto-refreshes whether or not
        the parent tab's loop got to us first.  Hidden popups skip: a
        just-invalidated σ₀ cache would otherwise spawn a ~30 s
        _StructWorker off-screen; showEvent re-refreshes on reopen."""
        if not self.isVisible():
            return
        self.refresh(results)

    def _on_beam_changed(self, *_args) -> None:
        """Beam config affects ref-particle energy → invalidate σ₀ cache.
        ``state.matrix_cache`` is replaced by AppState itself; we just
        drop the local σ₀ memo."""
        self._struct_cache.clear()
        self.refresh_with_state()

    def _on_lattice_changed(self, *_args) -> None:
        """Lattice mutation → drop σ₀ cache and re-detect periods.
        ``state.matrix_cache`` is replaced by AppState itself."""
        self._struct_cache.clear()
        self._populate_periods()
        self.refresh_with_state()

    def _struct_cache_key(self, lattice, period, ref) -> tuple:
        cfg = self._state.beam_config if self._state else None
        cfg_sig = (
            float(getattr(cfg, "alpha_x", 0.0)) if cfg else 0.0,
            float(getattr(cfg, "beta_x",  1.0)) if cfg else 0.0,
            float(getattr(cfg, "alpha_y", 0.0)) if cfg else 0.0,
            float(getattr(cfg, "beta_y",  1.0)) if cfg else 0.0,
            float(getattr(cfg, "alpha_z", 0.0)) if cfg else 0.0,
            float(getattr(cfg, "beta_z",  1.0)) if cfg else 0.0,
        )
        return (
            id(lattice),
            period.start, period.inner_slice_end, period.n_repeats,
            float(ref.w_kin), float(ref.frequency),
            type(ref.species).__name__ if ref.species else "",
            cfg_sig,
        )

    def _build_fallback_seed(self) -> dict | None:
        """Return a Twiss seed dict from beam_config, or None if absent.
        Used when the periodic Twiss is unavailable (non-periodic
        transport like MEBT+HWR with no LATTICE card)."""
        cfg = self._state.beam_config if self._state else None
        if cfg is None:
            return None
        return {
            "alpha_x": float(getattr(cfg, "alpha_x", 0.0)),
            "beta_x":  float(getattr(cfg, "beta_x",  1.0)),
            "alpha_y": float(getattr(cfg, "alpha_y", 0.0)),
            "beta_y":  float(getattr(cfg, "beta_y",  1.0)),
            "alpha_z": float(getattr(cfg, "alpha_z", 0.0)),
            "beta_z":  float(getattr(cfg, "beta_z",  1.0)),
        }

    def _get_or_compute_struct(self, lattice, ref, period):
        """Return ``(curves, sigma0)`` from cache, or ``None`` if a
        background worker is computing it (caller should bail out and
        the worker's ``finished_signal`` will trigger another refresh).
        """
        key = self._struct_cache_key(lattice, period, ref)
        cached = self._struct_cache.get(key)
        if cached is not None:
            return cached
        # Cache miss — kick off (or already running) background worker.
        # Track the active key to ignore stale results.
        self._pending_key = key
        if getattr(self, "_worker", None) is not None and self._worker.isRunning():
            return None
        self._worker = _StructWorker(
            lattice, ref, period, key,
            fallback_seed=self._build_fallback_seed(),
            matrix_cache=(self._state.matrix_cache
                          if self._state is not None else None),
        )
        self._worker.finished_signal.connect(self._on_struct_ready)
        self._worker.failed_signal.connect(self._on_struct_failed)
        self._worker.start()
        return None

    def _on_struct_ready(self, key, curves, sigma0) -> None:
        self._struct_cache[key] = (curves, sigma0)
        # Re-trigger refresh only if user is still on the same selection.
        if getattr(self, "_pending_key", None) == key:
            self._pending_key = None
            self.refresh_with_state()

    def _on_struct_failed(self, key, msg) -> None:
        if getattr(self, "_pending_key", None) == key:
            self._pending_key = None
            self._info.setText(f"σ₀ computation failed: {msg}")

    def showEvent(self, event):                              # noqa: N802
        """Pull in the latest results / lattice every time we're shown."""
        super().showEvent(event)
        # Lattice may have changed while we were hidden — re-detect periods.
        self._populate_periods()
        self.refresh_with_state()

    def refresh(self, results) -> None:
        """Repaint all six curves (σ₀ × σ × {x,y,z})."""
        # Reset curves so stale data never lingers.
        for plane in ("x", "y", "z"):
            self._rows[plane]["c0"].setData([], [])
            self._rows[plane]["cb"].setData([], [])
            self._rows[plane]["c0_pp"].setData([], [])
            self._rows[plane]["cb_pp"].setData([], [])
        self._clear_period_marks()
        self._info.setText("")

        state = self._state
        lattice = state.lattice if state else None
        if lattice is None:
            self._info.setText("Load a lattice first.")
            return
        if not self._periods or self._combo.count() != len(self._periods):
            self._populate_periods()
        idx = max(0, self._combo.currentIndex())
        if idx >= len(self._periods):
            return
        period = self._periods[idx]

        cfg = state.beam_config
        if cfg is None:
            self._info.setText("Set a beam config (Beam tab) to seed Twiss.")
            return

        # Build a ReferenceParticle from the beam config.
        try:
            from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
            from linac_gen.core.reference import ReferenceParticle
            sp_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
            sp = sp_map.get(cfg.species, PROTON)
            ref = ReferenceParticle(species=sp, w_kin=cfg.energy,
                                    frequency=cfg.frequency)
        except Exception as exc:                              # noqa: BLE001
            self._info.setText(f"Reference particle build failed: {exc}")
            return

        # ---- Structure σ₀(s) for all three planes -----------------
        try:
            from linac_gen.analysis.phase_advance import (
                beam_phase_advance_along_s,
            )
            out = self._get_or_compute_struct(lattice, ref, period)
            if out is None:
                # Background worker is computing — show a placeholder
                # and bail; finished_signal will re-trigger refresh().
                for plane in ("x", "y", "z"):
                    self._rows[plane]["c0"].setData([], [])
                    self._rows[plane]["cb"].setData([], [])
                    self._rows[plane]["c0_pp"].setData([], [])
                    self._rows[plane]["cb_pp"].setData([], [])
                self._info.setText(
                    "Computing σ₀ in background "
                    "(field-map matrices are slow on first run)…"
                )
                return
            curves, sigma0 = out
        except Exception as exc:                              # noqa: BLE001
            self._info.setText(f"σ₀ failed: {exc}")
            return

        s_struct = np.asarray(curves["s"], dtype=float)
        coupled_along = curves.get("coupled_along_s")

        for plane in ("x", "y", "z"):
            mu = np.asarray(curves[f"mu_{plane}_deg"], dtype=float)
            self._rows[plane]["c0"].setData(s_struct, mu)

        # When the period is xy-coupled, the decoupled per-plane
        # propagation gave NaN for x/y above.  Overlay the eigenmode
        # cumulative μ_I(s)/μ_II(s) on the x/y rows so the user sees
        # the physically-meaningful curves (computed from per-cell 4×4
        # eigenvalues, not Courant-Snyder propagation).
        if coupled_along is not None:
            s_cells = np.asarray(coupled_along["s_cells"], dtype=float)
            mu_I = np.asarray(coupled_along["mu_I_deg"], dtype=float)
            mu_II = np.asarray(coupled_along["mu_II_deg"], dtype=float)
            self._rows["x"]["c0"].setData(s_cells, mu_I)
            self._rows["x"]["plot_struct"].setTitle(
                "<span style='color:#fbbf24;'>μ_I (eigenmode I, coupled)</span>"
            )
            self._rows["y"]["c0"].setData(s_cells, mu_II)
            self._rows["y"]["plot_struct"].setTitle(
                "<span style='color:#fbbf24;'>μ_II (eigenmode II, coupled)</span>"
            )
        else:
            # Restore default titles in case we previously rendered
            # a coupled lattice and now we're back to a decoupled one.
            self._rows["x"]["plot_struct"].setTitle("μ_x σ₀  (deg)")
            self._rows["y"]["plot_struct"].setTitle("μ_y σ₀  (deg)")

        notes = []
        fb_planes = curves.get("fallback_planes") or []
        if fb_planes:
            notes.append(
                f"⚠ {','.join(fb_planes)}: σ₀ unavailable — μ(s) seeded from "
                f"beam_config α/β (transport μ, not periodic σ₀)"
            )
        # Per-cell σ₀ headline (skip planes with no seed).
        head_parts = []
        for plane in ("x", "y", "z"):
            mu_p = sigma0.get(f"mu_{plane}_deg")
            if mu_p is not None:
                head_parts.append(f"{plane}={mu_p:.2f}°")
            else:
                head_parts.append(f"{plane}=—")
        notes.append(
            f"per-cell σ₀ ({', '.join(head_parts)}) × {period.n_repeats}"
        )
        if sigma0.get("dw"):
            notes.append(f"ΔW={sigma0['dw']:+.4f} MeV")
        # Gate the heavy per-cell eigenmode walk on visibility, exactly as the
        # tune-depression popup does: refresh() re-runs on every
        # results/beam/lattice change even while this popup is hidden, and
        # coupled_phase_advance_per_cell rebuilds per-cell transfer matrices
        # synchronously on the UI thread.  showEvent re-refreshes on open, so
        # a visible popup still gets the note.
        if sigma0.get("coupled_xy") and self.isVisible():
            try:
                from linac_gen.analysis.phase_advance import (
                    coupled_phase_advance_per_cell,
                )
                cpc = coupled_phase_advance_per_cell(lattice, ref, period)
                I_med = float(np.nanmedian(cpc["mu_I_deg"]))
                II_med = float(np.nanmedian(cpc["mu_II_deg"]))
                notes.append(
                    f"xy coupled — eigenmode tunes: "
                    f"μ_I={I_med:.2f}°, μ_II={II_med:.2f}° per cell"
                )
            except Exception:                                # noqa: BLE001
                notes.append("xy coupled — σ₀_x/y unavailable; σ₀_z still valid")

        # ---- Stale-results banner ---------------------------------
        # Compare the current beam_config.current against the current
        # that was used to produce the envelope results.  If they
        # differ, the σ curves don't reflect the user's latest input —
        # warn explicitly so the user knows to press Ctrl+R.
        if results is not None:
            res_I = float(getattr(results, "current_mA", 0.0))
            cfg_I = float(getattr(cfg, "current", res_I))
            if abs(res_I - cfg_I) > 1e-6:
                notes.append(
                    f"⚠ STALE: σ run at {res_I:.3f} mA, "
                    f"beam_config now {cfg_I:.3f} mA — press Ctrl+R to re-run"
                )

        # ---- Beam σ(s) — only if envelope results exist -----------
        if results is not None and getattr(results, "beta_x", None):
            try:
                s_env = np.asarray(results.s, dtype=float)
                start_idx = int(np.searchsorted(s_env, s_struct[period.start]))
                start_idx = max(0, min(start_idx, s_env.size - 1))
                bcurves = beam_phase_advance_along_s(results, start_index=start_idx)
                s_b = np.asarray(bcurves["s"])
                for plane in ("x", "y", "z"):
                    mu = np.asarray(bcurves[f"mu_{plane}_deg"], dtype=float)
                    if np.any(np.isfinite(mu)):
                        self._rows[plane]["cb"].setData(s_b, mu)
                # If z-plane has no σ-matrix data, flag it.
                if not np.any(np.isfinite(np.asarray(bcurves["mu_z_deg"]))):
                    notes.append("σ_z unavailable (no σ-matrix in results)")
            except Exception as exc:                          # noqa: BLE001
                notes.append(f"σ failed: {exc}")
        else:
            notes.append("Run an envelope to populate σ.")

        # ---- Period boundary markers --------------------------------
        # Cell boundaries come from period.spans() — explicit per-repeat
        # element ranges (significant-element walk); constant stride only
        # as fallback for manually-built periods.
        boundary_idx: list[int] = []
        if period.n_repeats > 1:
            cell_spans = period.spans()
            boundaries = [cell_spans[0][0]] + [b for (_a, b) in cell_spans]
            for el_idx in boundaries:
                if 0 <= el_idx < len(s_struct):
                    boundary_idx.append(el_idx)
                    x_mm = float(s_struct[el_idx])
                    for plane in ("x", "y", "z"):
                        for col_key in ("plot_struct", "plot_beam"):
                            line = pg.InfiniteLine(
                                x_mm, angle=90,
                                pen=pg.mkPen(theme.TEXT_3, width=1,
                                              style=Qt.PenStyle.DashLine),
                            )
                            self._rows[plane][col_key].addItem(line)
                            self._period_marks.append((plane, col_key, line))

        # ---- Per-period Δσ -----------------------------------------
        # Δμ per cell = μ(boundary[k+1]) − μ(boundary[k]) using the
        # cumulative curves we already drew.  Cell index 1..n_repeats.
        if len(boundary_idx) >= 2:
            cells = np.arange(1, len(boundary_idx))  # 1..n_repeats
            for plane in ("x", "y", "z"):
                mu_struct = np.asarray(curves[f"mu_{plane}_deg"], dtype=float)
                d0 = np.array([
                    mu_struct[boundary_idx[k + 1]] - mu_struct[boundary_idx[k]]
                    for k in range(len(boundary_idx) - 1)
                ], dtype=float)
                if np.any(np.isfinite(d0)):
                    self._rows[plane]["c0_pp"].setData(cells, d0)
            # Coupled-mode per-cell tunes overlay the x/y per-cell columns.
            if coupled_along is not None:
                mu_I = np.asarray(coupled_along["mu_I_deg"], dtype=float)
                mu_II = np.asarray(coupled_along["mu_II_deg"], dtype=float)
                d_I = np.diff(mu_I); d_II = np.diff(mu_II)
                cells_c = np.arange(1, len(mu_I))
                if d_I.size:
                    self._rows["x"]["c0_pp"].setData(cells_c, d_I)
                    self._rows["x"]["plot_per0"].setTitle(
                        "<span style='color:#fbbf24;'>Δμ_I per cell  (deg)</span>"
                    )
                if d_II.size:
                    self._rows["y"]["c0_pp"].setData(cells_c, d_II)
                    self._rows["y"]["plot_per0"].setTitle(
                        "<span style='color:#fbbf24;'>Δμ_II per cell  (deg)</span>"
                    )
            # Beam per-cell — sample bcurves at boundary s-positions.
            if results is not None and getattr(results, "beta_x", None):
                try:
                    s_env = np.asarray(results.s, dtype=float)
                    s_b = np.asarray(bcurves["s"])
                    boundary_s = np.array([s_struct[i] for i in boundary_idx],
                                           dtype=float)
                    sample_idx = np.clip(
                        np.searchsorted(s_b, boundary_s), 0, s_b.size - 1
                    )
                    for plane in ("x", "y", "z"):
                        mu_beam = np.asarray(bcurves[f"mu_{plane}_deg"],
                                             dtype=float)
                        if not np.any(np.isfinite(mu_beam)):
                            continue
                        mu_at_b = mu_beam[sample_idx]
                        db = np.diff(mu_at_b)
                        if np.any(np.isfinite(db)):
                            self._rows[plane]["cb_pp"].setData(cells, db)
                except Exception:                              # noqa: BLE001
                    pass
        self._info.setText("  ·  ".join(notes))

    def _clear_period_marks(self) -> None:
        for plane, col_key, line in self._period_marks:
            self._rows[plane][col_key].removeItem(line)
        self._period_marks.clear()


def _beam_inputs_from_config(cfg):
    """``(ref, initial)`` from a BeamConfig, exactly as
    ``app._run_envelope`` builds them (species map, normalized→geometric
    ε via βγ, DC/continuous metadata included)."""
    from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    sp = {"proton": PROTON, "deuteron": DEUTERON,
          "H-": H_MINUS}.get(getattr(cfg, "species", "proton"), PROTON)
    ref = ReferenceParticle(species=sp, w_kin=cfg.energy,
                            frequency=cfg.frequency)
    bg = max(float(ref.bg), 1e-30)
    # Mismatch-scaled geometric emittances (shared helper) — without
    # this the envelope OVERLAY on MP results showed an artifactual
    # MP/envelope discrepancy for any beam with mismatch_{x,y,z} set.
    # The helper reads BeamConfig fields directly; this seam must stay
    # DEFENSIVE (callers pass partial/mock configs when only `ref` is
    # needed — e.g. the bend-field card series), so getattr-extract
    # with the historical defaults first.
    from types import SimpleNamespace
    from linac_gen.distributions.factory import geometric_emittances
    _ex, _ey, _ez = geometric_emittances(SimpleNamespace(
        emit_nx=float(getattr(cfg, "emit_nx", 0.2)),
        emit_ny=float(getattr(cfg, "emit_ny", 0.2)),
        emit_z=float(getattr(cfg, "emit_z", 0.0)),
        mismatch_x=getattr(cfg, "mismatch_x", 0.0),
        mismatch_y=getattr(cfg, "mismatch_y", 0.0),
        mismatch_z=getattr(cfg, "mismatch_z", 0.0),
    ), bg)
    initial = dict(
        alpha_x=float(cfg.alpha_x), beta_x=float(cfg.beta_x),
        emit_x=float(_ex),
        alpha_y=float(cfg.alpha_y), beta_y=float(cfg.beta_y),
        emit_y=float(_ey),
        alpha_z=float(getattr(cfg, "alpha_z", 0.0)),
        beta_z=float(getattr(cfg, "beta_z", 1.0)),
        emit_z=float(_ez),
        continuous=bool(getattr(cfg, "continuous", False)),
        dc_energy_spread_keV=float(
            getattr(cfg, "dc_energy_spread_keV", 0.0)),
    )
    return ref, initial


def _probe_sig(lattice, cfg) -> tuple:
    """Cache signature for a companion probe run — the lattice object and
    every beam-config field the envelope depends on."""
    return (
        id(lattice),
        str(getattr(cfg, "species", "")),
        float(getattr(cfg, "energy", 0.0)),
        float(getattr(cfg, "frequency", 0.0)),
        float(getattr(cfg, "current", 0.0)),
        float(getattr(cfg, "emit_nx", 0.0)),
        float(getattr(cfg, "alpha_x", 0.0)),
        float(getattr(cfg, "beta_x", 0.0)),
        float(getattr(cfg, "emit_ny", 0.0)),
        float(getattr(cfg, "alpha_y", 0.0)),
        float(getattr(cfg, "beta_y", 0.0)),
        float(getattr(cfg, "emit_z", 0.0)),
        float(getattr(cfg, "alpha_z", 0.0)),
        float(getattr(cfg, "beta_z", 0.0)),
        bool(getattr(cfg, "continuous", False)),
        float(getattr(cfg, "dc_energy_spread_keV", 0.0)),
        # Mismatch now feeds the envelope seed — a change must
        # invalidate the cached probe run.
        float(getattr(cfg, "mismatch_x", 0.0)),
        float(getattr(cfg, "mismatch_y", 0.0)),
        float(getattr(cfg, "mismatch_z", 0.0)),
    )


#: Single-slot cache of the last companion envelope-probe run — shared
#: across popups (tune-depression computes it; Hofmann reuses it), so
#: MP-results sessions pay the ~30 s probe once, not per popup.
_COMPANION_PROBE: dict = {}


def _companion_probe_results(lattice, cfg):
    """The cached companion probe results, or None if stale/absent."""
    if (_COMPANION_PROBE.get("results") is not None
            and _COMPANION_PROBE.get("sig") == _probe_sig(lattice, cfg)):
        return _COMPANION_PROBE["results"]
    return None


class _ProbeWorker(
        __import__("PyQt6.QtCore", fromlist=["QThread"]).QThread):
    """Background companion envelope probe (Option A for MP results):
    runs ``run_phase_probe`` with the current Beam-tab config so the
    channel-model curves (σ₀/σ_model/η_model) can be shown next to a
    multi-particle run's beam markers."""
    from PyQt6.QtCore import pyqtSignal as _Signal
    finished_ok = _Signal(object)
    failed = _Signal(str)

    def __init__(self, lattice, ref, initial, current):
        super().__init__()
        # numpy/BLAS on the 544 KB default macOS QThread stack → SIGBUS
        # (house pattern — see workers._MatchWorker).
        self.setStackSize(16 * 1024 * 1024)
        import threading
        self._args = (lattice, ref, initial, current)
        self._stop = threading.Event()

    def request_stop(self):
        self._stop.set()

    def run(self):
        try:
            from linac_gen.analysis.phase_advance import run_phase_probe
            lat, ref, initial, current = self._args
            out = run_phase_probe(lat, ref, initial, current,
                                  should_abort=self._stop.is_set)
            self.finished_ok.emit(out)
        except Exception as exc:                              # noqa: BLE001
            self.failed.emit(str(exc))


class _TuneDepressionPopup(_PopupPlot):
    """Tune-depression η = σ / σ₀ along s, per plane.

    σ₀(s) is the cumulative structure phase advance from periodic Twiss
    propagation; σ(s) is the cumulative beam phase advance from
    envelope-output β_beam(s).  Their ratio quantifies how much
    space-charge is depressing the lattice's bare focusing — η → 1 means
    SC-free, η → 0 means fully depressed.

    Three stacked plots (x · y · z), each a single-curve scatter+line
    showing η at every s-point where σ₀ > 0.  Period boundaries are
    marked with vertical dashed lines.
    """

    def __init__(self, parent, state):
        super().__init__(parent, "Tune depression  η = σ / σ₀",
                         size=(1100, 920))
        from PyQt6.QtWidgets import QComboBox, QPushButton
        self._state = state
        self._periods: list = []
        # Cache slow structure-phase-advance computations (independent
        # of envelope results — only depends on lattice / ref / period).
        self._struct_cache: dict = {}
        # Per-element transfer-matrix cache lives on AppState
        # (``state.matrix_cache``); see _PhaseAdvancePopup.
        if state is not None:
            try:
                state.results_changed.connect(self._on_results_changed)
            except Exception:                                # noqa: BLE001
                pass
            try:
                state.beam_config_changed.connect(self._on_beam_changed)
            except Exception:                                # noqa: BLE001
                pass
            try:
                state.lattice_changed.connect(self._on_lattice_changed)
            except Exception:                                # noqa: BLE001
                pass

        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)

        # Period picker row.
        pr = QHBoxLayout(); pr.setSpacing(8)
        pr.addWidget(QLabel("Period:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(280)
        self._combo.currentIndexChanged.connect(self.refresh_with_state)
        pr.addWidget(self._combo, stretch=1)
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self.refresh_with_state)
        pr.addWidget(recompute)
        self._info = QLabel("")
        self._info.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;"
        )
        self._info.setWordWrap(True)
        pr.addWidget(self._info)
        # Companion envelope probe (Option A): multi-particle results
        # carry no probe maps, so the model η cannot be extracted from
        # them — this button runs a fast envelope probe with the current
        # Beam-tab config and plots the channel model alongside the MP
        # beam markers.  Hidden while the loaded results carry their own
        # probe maps.
        self._probe_btn = QPushButton("Compute channel model")
        self._probe_btn.setToolTip(
            "Run a companion envelope probe (current Beam-tab config) to "
            "obtain σ₀/σ_model/η_model when the loaded results carry no "
            "probe maps (e.g. multi-particle runs).  ~30–60 s; runs in "
            "the background.")
        self._probe_btn.clicked.connect(self._compute_companion_probe)
        self._probe_btn.hide()
        pr.addWidget(self._probe_btn)
        self._probe_worker = None
        self._probe_sig_pending = None
        v.addLayout(pr)

        # 3 stacked plots, one per plane — η per CELL (the canonical
        # per-period definition; the running ratio σ(s)/σ₀(s) is not
        # standard and breaks down for unmatched beams).
        self._rows: dict[str, dict] = {}
        anchor = None
        for plane, title in (("x", "η_x"), ("y", "η_y"), ("z", "η_z")):
            p = _mk_plot(f"{title}  =  σ_{plane} / σ₀_{plane}  per cell", "")
            p.setLabel("bottom", "cell #")
            p.addLegend(offset=(-8, 8))
            # PRIMARY: model η from the phase-probe channel monodromy
            # (matched-channel definition, exact w.r.t. the run).
            model = p.plot(pen=curve_pen("#a3e635", width=2.2),
                           symbol="s", symbolSize=8,
                           symbolBrush="#a3e635", symbolPen="#a3e635",
                           name=f"{title} (model)")
            # SECONDARY: beam ∫ds/β ratio — TraceWin-comparable, valid
            # only for a matched beam.
            curve = p.plot(pen=curve_pen("#fbbf24", width=1.2),
                           symbol="o", symbolSize=7,
                           symbolBrush=None, symbolPen="#fbbf24",
                           name=f"{title} (beam)")
            ref = pg.InfiniteLine(1.0, angle=0,
                                   pen=pg.mkPen(theme.TEXT_3, width=1,
                                                 style=Qt.PenStyle.DashLine))
            p.addItem(ref)
            if anchor is None:
                anchor = p
            else:
                p.setXLink(anchor)
            v.addWidget(p, stretch=1)
            self._rows[plane] = {"plot": p, "curve": curve, "model": model}

        self._populate_periods()

    def _populate_periods(self) -> None:
        from linac_gen.analysis.period_detect import detect_periods
        self._combo.blockSignals(True)
        self._combo.clear()
        self._periods = []
        lattice = self._state.lattice if self._state else None
        if lattice is not None:
            try:
                self._periods = detect_periods(lattice)
            except Exception:                                # noqa: BLE001
                self._periods = []
        if not self._periods:
            self._combo.addItem("(no lattice)")
        else:
            for p in self._periods:
                tag = {
                    "lattice_card":            "[LATTICE]",
                    "lattice_card_recovered":  "[LATTICE+]",
                    "type_sequence":           "[auto]",
                    "fallback":                "[full]",
                }.get(p.source, "[?]")
                self._combo.addItem(f"{tag}  {p.label}")
        self._combo.blockSignals(False)

    def refresh_with_state(self, *_args) -> None:
        self.refresh(self._state.results if self._state else None)

    def _on_results_changed(self, results) -> None:
        # Hidden-popup guard: see _PhaseAdvancePopup._on_results_changed.
        if not self.isVisible():
            return
        self.refresh(results)

    def _on_beam_changed(self, *_args) -> None:
        self._struct_cache.clear()
        self.refresh_with_state()

    def _on_lattice_changed(self, *_args) -> None:
        self._struct_cache.clear()
        self._populate_periods()
        self.refresh_with_state()

    def _struct_cache_key(self, lattice, period, ref) -> tuple:
        cfg = self._state.beam_config if self._state else None
        cfg_sig = (
            float(getattr(cfg, "alpha_x", 0.0)) if cfg else 0.0,
            float(getattr(cfg, "beta_x",  1.0)) if cfg else 0.0,
            float(getattr(cfg, "alpha_y", 0.0)) if cfg else 0.0,
            float(getattr(cfg, "beta_y",  1.0)) if cfg else 0.0,
            float(getattr(cfg, "alpha_z", 0.0)) if cfg else 0.0,
            float(getattr(cfg, "beta_z",  1.0)) if cfg else 0.0,
        )
        return (
            id(lattice),
            period.start, period.inner_slice_end, period.n_repeats,
            float(ref.w_kin), float(ref.frequency),
            type(ref.species).__name__ if ref.species else "",
            cfg_sig,
        )

    def _build_fallback_seed(self) -> dict | None:
        cfg = self._state.beam_config if self._state else None
        if cfg is None:
            return None
        return {
            "alpha_x": float(getattr(cfg, "alpha_x", 0.0)),
            "beta_x":  float(getattr(cfg, "beta_x",  1.0)),
            "alpha_y": float(getattr(cfg, "alpha_y", 0.0)),
            "beta_y":  float(getattr(cfg, "beta_y",  1.0)),
            "alpha_z": float(getattr(cfg, "alpha_z", 0.0)),
            "beta_z":  float(getattr(cfg, "beta_z",  1.0)),
        }

    def _get_or_compute_struct(self, lattice, ref, period):
        """Return ``(curves, sigma0)`` from cache, or ``None`` if a
        background worker is computing it."""
        key = self._struct_cache_key(lattice, period, ref)
        cached = self._struct_cache.get(key)
        if cached is not None:
            return cached
        self._pending_key = key
        if getattr(self, "_worker", None) is not None and self._worker.isRunning():
            return None
        self._worker = _StructWorker(
            lattice, ref, period, key,
            fallback_seed=self._build_fallback_seed(),
            matrix_cache=(self._state.matrix_cache
                          if self._state is not None else None),
        )
        self._worker.finished_signal.connect(self._on_struct_ready)
        self._worker.failed_signal.connect(self._on_struct_failed)
        self._worker.start()
        return None

    def _on_struct_ready(self, key, curves, sigma0) -> None:
        self._struct_cache[key] = (curves, sigma0)
        if getattr(self, "_pending_key", None) == key:
            self._pending_key = None
            self.refresh_with_state()

    def _on_struct_failed(self, key, msg) -> None:
        if getattr(self, "_pending_key", None) == key:
            self._pending_key = None
            self._info.setText(f"σ₀ computation failed: {msg}")


    # ---- Companion envelope probe (model curves for MP results) -----
    def _compute_companion_probe(self) -> None:
        state = self._state
        lattice = state.lattice if state else None
        cfg = state.beam_config if state else None
        if lattice is None or cfg is None:
            self._info.setText("Load a lattice and set a beam config first.")
            return
        if (self._probe_worker is not None
                and self._probe_worker.isRunning()):
            return
        try:
            ref, initial = _beam_inputs_from_config(cfg)
        except Exception as exc:                              # noqa: BLE001
            self._info.setText(f"probe setup failed: {exc}")
            return
        self._probe_btn.setEnabled(False)
        self._probe_btn.setText("Computing…")
        self._probe_sig_pending = _probe_sig(lattice, cfg)
        self._probe_worker = _ProbeWorker(
            lattice, ref, initial, float(getattr(cfg, "current", 0.0)))
        self._probe_worker.finished_ok.connect(self._on_probe_ready)
        self._probe_worker.failed.connect(self._on_probe_failed)
        self._probe_worker.start()

    def _on_probe_ready(self, probe_results) -> None:
        _COMPANION_PROBE.clear()
        _COMPANION_PROBE["sig"] = self._probe_sig_pending
        _COMPANION_PROBE["results"] = probe_results
        self._probe_btn.setText("Compute channel model")
        self._probe_btn.setEnabled(True)
        self.refresh_with_state()

    def _on_probe_failed(self, msg) -> None:
        self._probe_btn.setText("Compute channel model")
        self._probe_btn.setEnabled(True)
        self._info.setText(f"channel-model probe failed: {msg}")

    def closeEvent(self, ev) -> None:                        # noqa: N802
        # Stop the companion-probe worker too — the base handler only
        # knows about self._worker (the sigma_0 struct worker).
        w = self._probe_worker
        if w is not None and w.isRunning():
            if hasattr(w, "request_stop"):
                w.request_stop()
            w.requestInterruption()
            if not w.wait(2000):
                _park_zombie(w)
        super().closeEvent(ev)

    def showEvent(self, event):                              # noqa: N802
        super().showEvent(event)
        self._populate_periods()
        self.refresh_with_state()

    def refresh(self, results) -> None:
        for plane in ("x", "y", "z"):
            self._rows[plane]["curve"].setData([], [])
            self._rows[plane]["model"].setData([], [])
        self._info.setText("")
        # Withdraw the companion-probe offer on every pass; the model
        # branch below re-shows it when (and only when) it applies —
        # otherwise an early return leaves a stale button visible.
        self._probe_btn.hide()

        state = self._state
        lattice = state.lattice if state else None
        if lattice is None:
            self._info.setText("Load a lattice first.")
            return
        if results is None or not getattr(results, "beta_x", None):
            self._info.setText("η needs envelope σ(s) — run an envelope first.")
            return
        if not self._periods or self._combo.count() != len(self._periods):
            self._populate_periods()
        idx = max(0, self._combo.currentIndex())
        if idx >= len(self._periods):
            return
        period = self._periods[idx]
        if period.n_repeats < 1:
            self._info.setText("Period has zero repeats — cannot form η.")
            return

        cfg = state.beam_config
        if cfg is None:
            self._info.setText("Set a beam config (Beam tab) to seed Twiss.")
            return

        try:
            from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
            from linac_gen.core.reference import ReferenceParticle
            sp_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
            sp = sp_map.get(cfg.species, PROTON)
            ref = ReferenceParticle(species=sp, w_kin=cfg.energy,
                                    frequency=cfg.frequency)
        except Exception as exc:                              # noqa: BLE001
            self._info.setText(f"Reference particle build failed: {exc}")
            return

        # Structure cumulative σ₀(s) (cached) and beam cumulative σ(s).
        try:
            from linac_gen.analysis.phase_advance import (
                beam_phase_advance_along_s,
            )
            out = self._get_or_compute_struct(lattice, ref, period)
            if out is None:
                self._info.setText(
                    "Computing σ₀ in background "
                    "(field-map matrices are slow on first run)…"
                )
                return
            scurves, sigma0 = out
            s_struct = np.asarray(scurves["s"], dtype=float)
            s_env = np.asarray(results.s, dtype=float)
            start_idx = int(np.searchsorted(s_env, s_struct[period.start]))
            start_idx = max(0, min(start_idx, s_env.size - 1))
            bcurves = beam_phase_advance_along_s(results, start_index=start_idx)
            s_b = np.asarray(bcurves["s"], dtype=float)
        except Exception as exc:                              # noqa: BLE001
            self._info.setText(f"phase-advance failed: {exc}")
            return
        if s_b.size == 0:
            # Degenerate results (zero-step run): np.clip(…, 0, -1) below
            # would yield index −1 and the β-mismatch guard would index an
            # empty array — an IndexError OUTSIDE the try above.
            self._info.setText("η unavailable — beam μ(s) is empty.")
            return

        # Boundary indices of each cell (in lattice-element s space) —
        # explicit per-repeat spans from period.spans() (significant-
        # element walk; constant stride only as manual-period fallback).
        cell_spans = period.spans()
        boundary_idx = [
            b for b in [cell_spans[0][0]] + [sp[1] for sp in cell_spans]
            if b < len(s_struct)
        ]
        if len(boundary_idx) < 2:
            self._info.setText("Need ≥2 cell boundaries to form per-cell η.")
            return

        # For each boundary, sample beam μ at the matching s-position.
        boundary_s = np.array([s_struct[i] for i in boundary_idx], dtype=float)
        sample_idx = np.clip(np.searchsorted(s_b, boundary_s), 0, s_b.size - 1)

        # Matched-beam guard: |β_end − β_start| / β_start over the WHOLE
        # span; if it exceeds 5 % the per-cell η ratio is unreliable
        # because σ_beam reflects mismatch oscillations rather than the
        # SC-depressed periodic phase advance.
        bx = np.asarray(getattr(results, "beta_x", []), dtype=float)
        by = np.asarray(getattr(results, "beta_y", []), dtype=float)
        i_lo = sample_idx[0]; i_hi = sample_idx[-1]
        mismatch_msg = ""
        if bx.size > i_hi and bx[i_lo] > 0:
            mx = abs(bx[i_hi] - bx[i_lo]) / bx[i_lo]
            if mx > 0.05:
                mismatch_msg = f"β_x mismatch {100*mx:.1f}%"
        if by.size > i_hi and by[i_lo] > 0:
            my = abs(by[i_hi] - by[i_lo]) / by[i_lo]
            if my > 0.05:
                if mismatch_msg:
                    mismatch_msg += f" · β_y {100*my:.1f}%"
                else:
                    mismatch_msg = f"β_y mismatch {100*my:.1f}%"

        cells = np.arange(1, len(boundary_idx))  # 1..n_repeats
        notes = []
        any_data = False
        skipped = []

        # ---- PRIMARY: model η per cell from the phase-probe channel
        # monodromies (matched-channel definition; exact w.r.t. the
        # envelope run and valid for coupled sections via modes I/II).
        # MP results carry no probe maps — fall back to the cached
        # companion envelope probe (same beam config) when available,
        # else surface the "Compute channel model" button.
        model_src = (results if getattr(results, "element_maps_dep", None)
                     else _companion_probe_results(lattice, cfg))
        if model_src is not None:
            self._probe_btn.hide()
            if model_src is not results:
                notes.append("model η from companion envelope probe "
                             "(current Beam-tab config)")
            try:
                from linac_gen.analysis.phase_advance import (
                    channel_phase_advance,
                )
                ch = channel_phase_advance(model_src, period)
                cells_m = np.asarray(ch["cells"], dtype=float)
                if ch["coupled_xy"]:
                    series = {"x": ("eta_I", "η_I"), "y": ("eta_II", "η_II"),
                              "z": ("eta_z", "η_z")}
                else:
                    series = {p: (f"eta_{p}", f"η_{p}")
                              for p in ("x", "y", "z")}
                meds = []
                for plane, (key, tag) in series.items():
                    eta_m = np.asarray(ch.get(key, []), dtype=float)
                    fin_m = np.isfinite(eta_m)
                    if fin_m.any():
                        self._rows[plane]["model"].setData(
                            cells_m[fin_m], eta_m[fin_m])
                        meds.append(
                            f"{tag}={float(np.nanmedian(eta_m[fin_m])):.3f}")
                        any_data = True
                if meds:
                    notes.append("model η_med: " + " ".join(meds))
            except Exception as exc:                          # noqa: BLE001
                notes.append(f"model η failed: {exc}")
        else:
            if (self._probe_worker is not None
                    and self._probe_worker.isRunning()):
                notes.append("computing channel model (envelope probe)…")
            else:
                self._probe_btn.show()
                notes.append(
                    "model η: these results carry no probe maps (MP run?) "
                    "— press 'Compute channel model' for a companion "
                    "envelope probe, or re-run the envelope")

        for plane in ("x", "y", "z"):
            mu0_full = np.asarray(scurves[f"mu_{plane}_deg"], dtype=float)
            mub_full = np.asarray(bcurves[f"mu_{plane}_deg"], dtype=float)
            mu0_ok = np.any(np.isfinite(mu0_full))
            mub_ok = np.any(np.isfinite(mub_full))
            if not (mu0_ok and mub_ok):
                # Tell the user *why* this plane is empty.
                if not mu0_ok and not mub_ok:
                    skipped.append(f"{plane}: σ₀ & σ unavailable")
                elif not mu0_ok:
                    skipped.append(f"{plane}: σ₀ unavailable (xy-coupled?)")
                else:
                    skipped.append(f"{plane}: σ unavailable (no β_{plane})")
                continue
            # Per-cell σ₀_k = mu0[boundary_{k+1}] − mu0[boundary_k].
            sigma0_k = np.array([
                mu0_full[boundary_idx[k + 1]] - mu0_full[boundary_idx[k]]
                for k in range(len(boundary_idx) - 1)
            ], dtype=float)
            sigma_k = np.array([
                mub_full[sample_idx[k + 1]] - mub_full[sample_idx[k]]
                for k in range(len(boundary_idx) - 1)
            ], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                eta = np.where(np.abs(sigma0_k) > 1e-6, sigma_k / sigma0_k, np.nan)
            finite = np.isfinite(eta)
            if not finite.any():
                continue
            any_data = True
            eta_f = eta[finite]; cells_f = cells[finite]
            self._rows[plane]["curve"].setData(cells_f, eta_f)
            med = float(np.nanmedian(eta_f))
            notes.append(f"{plane}: η_med = {med:.3f}")
            # y-range encompasses both the data and the η=1 reference.
            ymin = float(np.nanmin(eta_f))
            ymax = float(np.nanmax(eta_f))
            lo = min(0.0, ymin)
            hi = max(1.1, ymax)
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            self._rows[plane]["plot"].setYRange(lo - pad, hi + pad)
            # x-range covers cell 1..n_repeats with a small margin.
            self._rows[plane]["plot"].setXRange(0.5, len(boundary_idx) - 0.5)

        # If x/y were skipped due to xy-coupling, plot the coupled
        # eigenmode tune depression: η_I = σ_I / σ₀_I, η_II = σ_II /
        # σ₀_II per cell, with σ_I,II from the depressed transfer
        # matrix M_dep_period built by chaining bare element matrices
        # with the SC kick at the recorded σ.  Verified against I=0:
        # η_I = η_II = 1.0 exactly when there's no space charge.
        coupled_msg = ""
        try:
            from linac_gen.analysis.phase_advance import (
                coupled_phase_advance_per_cell,
                coupled_beam_phase_advance_per_cell_via_M,
            )
            # Reuse the σ₀ dict the background worker already computed instead
            # of recomputing structure_phase_advance() synchronously on the UI
            # thread — that rebuilt the full one-period transfer matrix on
            # EVERY refresh (each run / beam-apply / lattice-load) and froze the
            # GUI for seconds on field-map lattices, even while the popup was
            # hidden.
            s_full = sigma0
            # The coupled per-cell walks are still heavy; only run them when the
            # popup is actually visible (showEvent re-refreshes on open) so a
            # hidden popup never freezes on a background refresh.
            if s_full.get("coupled_xy") and self.isVisible():
                cpc = coupled_phase_advance_per_cell(lattice, ref, period)
                bpc = coupled_beam_phase_advance_per_cell_via_M(
                    lattice, ref, period, results,
                )
                s0_I  = np.asarray(cpc["mu_I_deg"],  dtype=float)
                s0_II = np.asarray(cpc["mu_II_deg"], dtype=float)
                s_I  = np.asarray(bpc["mu_I_deg"],  dtype=float)
                s_II = np.asarray(bpc["mu_II_deg"], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    eta_I  = np.where(np.abs(s0_I)  > 1e-6, s_I  / s0_I,  np.nan)
                    eta_II = np.where(np.abs(s0_II) > 1e-6, s_II / s0_II, np.nan)
                cells = np.arange(1, len(s0_I) + 1)
                finI = np.isfinite(eta_I)
                finII = np.isfinite(eta_II)
                if finI.any():
                    self._rows["x"]["curve"].setData(cells[finI], eta_I[finI])
                    eta_f = eta_I[finI]
                    ymin, ymax = float(np.nanmin(eta_f)), float(np.nanmax(eta_f))
                    lo = min(0.0, ymin); hi = max(1.1, ymax)
                    pad = 0.05 * (hi - lo) if hi > lo else 0.1
                    self._rows["x"]["plot"].setYRange(lo - pad, hi + pad)
                    self._rows["x"]["plot"].setXRange(0.5, len(s0_I) + 0.5)
                    self._rows["x"]["plot"].setTitle(
                        "<span style='color:#fbbf24;'>η_I = σ_I / σ₀_I  "
                        "per cell  (coupled mode I)</span>"
                    )
                    any_data = True
                if finII.any():
                    self._rows["y"]["curve"].setData(cells[finII], eta_II[finII])
                    eta_f = eta_II[finII]
                    ymin, ymax = float(np.nanmin(eta_f)), float(np.nanmax(eta_f))
                    lo = min(0.0, ymin); hi = max(1.1, ymax)
                    pad = 0.05 * (hi - lo) if hi > lo else 0.1
                    self._rows["y"]["plot"].setYRange(lo - pad, hi + pad)
                    self._rows["y"]["plot"].setXRange(0.5, len(s0_II) + 0.5)
                    self._rows["y"]["plot"].setTitle(
                        "<span style='color:#fbbf24;'>η_II = σ_II / σ₀_II  "
                        "per cell  (coupled mode II)</span>"
                    )
                    any_data = True
                eI_med = float(np.nanmedian(eta_I[finI])) if finI.any() else float("nan")
                eII_med = float(np.nanmedian(eta_II[finII])) if finII.any() else float("nan")
                I_med = float(np.nanmedian(s0_I)) if np.any(np.isfinite(s0_I)) else float("nan")
                II_med = float(np.nanmedian(s0_II)) if np.any(np.isfinite(s0_II)) else float("nan")
                coupled_msg = (
                    f"xy-coupled — σ₀_I={I_med:.2f}°/σ₀_II={II_med:.2f}°  "
                    f"·  η_I_med={eI_med:.3f}, η_II_med={eII_med:.3f}"
                )
        except Exception as exc:                              # noqa: BLE001
            coupled_msg = f"coupled η failed: {exc}"

        if not any_data:
            msg = "η unavailable — no overlapping σ₀ / σ data."
            if skipped:
                msg += "  ·  " + "  ·  ".join(skipped)
            if coupled_msg:
                msg += "  ·  ⚠ " + coupled_msg
            self._info.setText(msg)
            return
        if skipped:
            notes.append("skipped — " + " / ".join(skipped))
        if coupled_msg:
            notes.append("⚠ " + coupled_msg)
        # Stale-results banner.
        res_I = float(getattr(results, "current_mA", 0.0))
        cfg_I = float(getattr(cfg, "current", res_I))
        if abs(res_I - cfg_I) > 1e-6:
            notes.append(
                f"⚠ STALE: σ run at {res_I:.3f} mA, "
                f"beam_config now {cfg_I:.3f} mA — Ctrl+R to re-run"
            )
        if mismatch_msg:
            notes.append(f"⚠ {mismatch_msg} — η unreliable for unmatched beams")
        self._info.setText("  ·  ".join(notes))


class _HofmannPopup(_PopupPlot):
    """Channel-tune trajectory on the Hofmann stability chart.

    Conventions pinned to Hofmann et al., PRST-AB 6, 024202 (2003):
    abscissa k_z/k_x (depressed tune ratio), ordinate k_x/k_0x
    (transverse tune depression), resonance families at k_z/k_x = m/n.
    Band WIDTHS drawn here are indicative (see analysis.hofmann); the
    ε_z/ε_x family selector is reported so the published chart for the
    right family can be consulted for quantitative edges.
    """

    def __init__(self, parent, state):
        super().__init__(parent, "Hofmann stability chart", size=(900, 640))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)
        pr = QHBoxLayout(); pr.setSpacing(8)
        pr.addWidget(QLabel("Period:"))
        self._combo = QComboBox(); self._combo.setMinimumWidth(280)
        self._combo.currentIndexChanged.connect(
            lambda *_: self.refresh(self._state.results))
        pr.addWidget(self._combo, stretch=1)
        self._info = QLabel("")
        self._info.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;")
        self._info.setWordWrap(True)
        pr.addWidget(self._info, stretch=2)
        v.addLayout(pr)

        self._plot = _mk_plot(
            "k_z/k_x  vs  k_x/k_0x   (trajectory: one point per cell)", "")
        self._plot.setLabel("bottom", "tune ratio k_z / k_x")
        self._plot.setLabel("left", "tune depression k_x / k_0x")
        self._plot.setXRange(0.0, 2.2)
        self._plot.setYRange(0.0, 1.05)
        self._band_items: list = []
        self._traj = self._plot.plot(
            pen=curve_pen("#f472b6", width=1.2), symbol="o", symbolSize=9,
            symbolBrush="#f472b6", symbolPen="w")
        v.addWidget(self._plot, stretch=1)
        self._periods: list = []
        self._populate_periods()
        # Bound methods only on the long-lived AppState signals — a
        # lambda here outlives the popup's C++ object and fires on a
        # deleted widget (see the PyQt6 lambda-connection rule).
        state.results_changed.connect(self.refresh)
        state.lattice_changed.connect(self._on_lattice_changed)

    def _on_lattice_changed(self, *_a):
        try:
            self._populate_periods()
        except RuntimeError:                                 # C++ side gone
            pass

    def _populate_periods(self):
        from linac_gen.analysis.period_detect import detect_periods
        self._combo.blockSignals(True)
        self._combo.clear()
        self._periods = []
        lat = self._state.lattice if self._state else None
        if lat is not None:
            try:
                self._periods = detect_periods(lat)
            except Exception:                                # noqa: BLE001
                self._periods = []
        for p in self._periods:
            self._combo.addItem(p.label)
        self._combo.blockSignals(False)

    def refresh(self, results):
        for it in self._band_items:
            self._plot.removeItem(it)
        self._band_items = []
        self._traj.setData([], [])
        model_note = ""
        if results is None or not getattr(results, "element_maps_dep", None):
            # MP results carry no probe maps — reuse the companion
            # envelope probe computed from the tune-depression popup
            # (shared single-slot cache) when its signature matches.
            lat = self._state.lattice if self._state else None
            cfg = self._state.beam_config if self._state else None
            companion = (_companion_probe_results(lat, cfg)
                         if (lat is not None and cfg is not None) else None)
            if companion is None:
                self._info.setText(
                    "Needs a probe-bearing envelope run — re-run the "
                    "envelope, or press 'Compute channel model' in the "
                    "Tune-depression popup (the result is shared).")
                return
            results = companion
            model_note = " · companion envelope probe"
        if not self._periods:
            self._info.setText("No periodic structure detected.")
            return
        idx = max(0, min(self._combo.currentIndex(), len(self._periods) - 1))
        period = self._periods[idx]
        try:
            from linac_gen.analysis.hofmann import (
                hofmann_trajectory, resonance_bands,
            )
            from linac_gen.analysis.phase_advance import channel_phase_advance
            ch = channel_phase_advance(results, period)
            tr = hofmann_trajectory(ch, results)
        except Exception as exc:                              # noqa: BLE001
            self._info.setText(f"chart failed: {exc}")
            return
        ratio = np.asarray(tr["ratio"], float)
        dep = np.asarray(tr["depression"], float)
        fin = np.isfinite(ratio) & np.isfinite(dep)
        med_dep = float(np.nanmedian(dep[fin])) if fin.any() else 1.0
        # Resonance bands: center line + indicative shaded region at the
        # trajectory's median depression.
        for band in resonance_bands(tr["emit_ratio"]):
            r0 = band["ratio"]
            w = band["width"](med_dep)
            region = pg.LinearRegionItem(
                values=(r0 - w, r0 + w), orientation="vertical",
                movable=False,
                brush=pg.mkBrush(244, 114, 182, 28),
                pen=pg.mkPen(244, 114, 182, 60))
            self._plot.addItem(region)
            line = pg.InfiniteLine(
                r0, angle=90,
                pen=pg.mkPen("#f472b6", width=1,
                             style=Qt.PenStyle.DashLine),
                label=band["label"],
                labelOpts={"position": 0.95, "color": "#f472b6"})
            self._plot.addItem(line)
            self._band_items += [region, line]
        if fin.any():
            self._traj.setData(ratio[fin], dep[fin])
        er = tr["emit_ratio"]
        er_txt = f"{er:.2f}" if np.isfinite(er) else "n/a"
        self._info.setText(
            f"transverse = {tr['transverse_label']} · ε_z/ε_x = {er_txt} "
            f"(pick this family on the published charts) · shaded widths "
            f"are INDICATIVE, drawn at the median depression "
            f"{med_dep:.2f}{model_note}")


class _FootprintWorker(
        __import__("PyQt6.QtCore", fromlist=["QThread"]).QThread):
    from PyQt6.QtCore import pyqtSignal as _Signal
    finished_ok = _Signal(object)
    failed = _Signal(str)
    progress_turn = _Signal(int, int)          # (turns done, total)

    def __init__(self, lattice, ref, period, initial, current,
                 n_turns: int = 128, n_particles: int = 60):
        super().__init__()
        # numpy/BLAS on the 544 KB default macOS QThread stack → SIGBUS
        # (house pattern — see workers._MatchWorker).
        self.setStackSize(16 * 1024 * 1024)
        import threading
        self._args = (lattice, ref, period, initial, current)
        self._n_turns = int(n_turns)
        self._n_particles = int(n_particles)
        self._stop = threading.Event()

    def request_stop(self):
        self._stop.set()

    def run(self):
        try:
            from linac_gen.analysis.footprint import tune_footprint
            lat, ref, period, initial, current = self._args
            out = tune_footprint(lat, ref, period, initial, current,
                                 n_turns=self._n_turns,
                                 n_particles=self._n_particles,
                                 should_stop=self._stop.is_set,
                                 progress=self.progress_turn.emit)
            self.finished_ok.emit(out)
        except Exception as exc:                              # noqa: BLE001
            self.failed.emit(str(exc))


class _FootprintPopup(_PopupPlot):
    """Frozen-SC incoherent tune footprint (see analysis.footprint —
    2-D Gaussian-equivalent frozen field, real nonlinear transport)."""

    def __init__(self, parent, state):
        super().__init__(parent, "Tune footprint (frozen SC)",
                         size=(860, 640))
        self._state = state
        self._worker = None
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)
        pr = QHBoxLayout(); pr.setSpacing(8)
        pr.addWidget(QLabel("Period:"))
        self._combo = QComboBox(); self._combo.setMinimumWidth(240)
        pr.addWidget(self._combo, stretch=1)
        # Compute budget: cost = turns × particles × per-turn transport;
        # field-map cells (RK4) are ~two orders costlier per turn than
        # hard-edge cells, so the budget must be user-visible (the old
        # hardwired 256×120 ran >20 min on the PIP-II HWR cell).
        pr.addWidget(QLabel("Turns:"))
        self._turns = QSpinBox(); self._turns.setRange(32, 2048)
        self._turns.setValue(128); self._turns.setSingleStep(32)
        self._turns.setToolTip(
            "FFT tune resolution ≈ 360°/turns per cell.  128 → ~2.8°.")
        pr.addWidget(self._turns)
        pr.addWidget(QLabel("Particles:"))
        self._nparts = QSpinBox(); self._nparts.setRange(12, 600)
        self._nparts.setValue(60); self._nparts.setSingleStep(12)
        self._nparts.setToolTip(
            "Amplitude-ladder size (3 rays: x / y / diagonal).")
        pr.addWidget(self._nparts)
        self._btn = QPushButton("Compute footprint")
        self._btn.clicked.connect(self._compute)
        pr.addWidget(self._btn)
        v.addLayout(pr)
        self._info = QLabel(
            "Tracks a test distribution repeatedly through the cell with "
            "the SC field frozen from the first pass — per-particle tunes "
            "via NAFF-lite.  Cost scales with turns × particles; field-map "
            "cells are far costlier per turn than hard-edge cells.")
        self._info.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;")
        self._info.setWordWrap(True)
        v.addWidget(self._info)
        self._plot = _mk_plot("per-particle tunes (deg per cell)", "")
        self._plot.setLabel("bottom", "μ_x (deg)")
        self._plot.setLabel("left", "μ_y (deg)")
        self._scatter = pg.ScatterPlotItem(size=7, pen=pg.mkPen("w", width=0.4))
        self._plot.addItem(self._scatter)
        v.addWidget(self._plot, stretch=1)
        self._periods: list = []
        self._populate_periods()
        # Bound method only (long-lived AppState signal ↔ destroyable
        # popup — the PyQt6 lambda-connection rule).
        state.lattice_changed.connect(self._on_lattice_changed)

    def _on_lattice_changed(self, *_a):
        try:
            self._populate_periods()
        except RuntimeError:                                 # C++ side gone
            pass

    def refresh(self, results):                              # noqa: ARG002
        """Popup contract: ``_open_popup`` calls ``refresh(results)`` on
        every open (a missing method surfaces as a warning dialog before
        the popup shows).  The footprint re-tracks the cell only when the
        user presses Compute — nothing is plotted from ``results`` — so
        this just keeps the button state coherent."""
        if self._worker is not None and self._worker.isRunning():
            return                          # keep the "computing…" state
        lat = self._state.lattice if self._state else None
        self._btn.setEnabled(lat is not None)

    def _populate_periods(self):
        from linac_gen.analysis.period_detect import detect_periods
        self._combo.clear()
        self._periods = []
        lat = self._state.lattice if self._state else None
        if lat is not None:
            try:
                self._periods = detect_periods(lat)
            except Exception:                                # noqa: BLE001
                self._periods = []
        for p in self._periods:
            self._combo.addItem(p.label)

    def _compute(self):
        state = self._state
        cfg = state.beam_config if state else None
        lat = state.lattice if state else None
        if lat is None or cfg is None or not self._periods:
            self._info.setText("Load a lattice and set a beam config first.")
            return
        idx = max(0, min(self._combo.currentIndex(), len(self._periods) - 1))
        period = self._periods[idx]
        try:
            ref, initial = _beam_inputs_from_config(cfg)
        except Exception as exc:                              # noqa: BLE001
            self._info.setText(f"setup failed: {exc}")
            return
        n_turns = int(self._turns.value())
        n_parts = int(self._nparts.value())
        self._btn.setEnabled(False)
        self._info.setText(
            f"Computing matched Σ + record pass ({n_turns} turns × "
            f"{n_parts} particles)…")
        self._worker = _FootprintWorker(
            lat, ref, period, initial, float(getattr(cfg, "current", 0.0)),
            n_turns=n_turns, n_particles=n_parts)
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.progress_turn.connect(self._progress)
        self._worker.start()

    def _progress(self, done: int, total: int):
        if done == 0:
            self._info.setText(
                f"Matched Σ + field tape done — tracking turn 1/{total}…")
        else:
            self._info.setText(f"Tracking — turn {done}/{total}…")

    def _done(self, out):
        self._btn.setEnabled(True)
        qx = np.asarray(out["qx"], float) * 360.0
        qy = np.asarray(out["qy"], float) * 360.0
        amp = (np.asarray(out["ax_sigma"], float)
               + np.asarray(out["ay_sigma"], float))
        fin = np.isfinite(qx) & np.isfinite(qy)
        if not fin.any():
            self._info.setText("No finite tunes — beam lost?")
            return
        a = amp[fin]
        a_norm = (a - a.min()) / max(a.max() - a.min(), 1e-12)
        spots = [{"pos": (float(qx[fin][i]), float(qy[fin][i])),
                  "brush": pg.mkBrush(56 + int(180 * a_norm[i]),
                                      189 - int(120 * a_norm[i]), 248, 200)}
                 for i in range(int(fin.sum()))]
        self._scatter.setData(spots)
        self._info.setText(
            f"core μ_x={out['mu_x_core_deg']:.2f}° μ_y="
            f"{out['mu_y_core_deg']:.2f}° · spread(max−min) "
            f"x: {np.nanmax(qx[fin]) - np.nanmin(qx[fin]):.2f}° · "
            f"{out['model']} · color = launch amplitude")

    def _fail(self, msg):
        self._btn.setEnabled(True)
        self._info.setText(f"footprint failed: {msg}")


class _EnergyPopup(_PopupPlot):
    def __init__(self, parent):
        super().__init__(parent, "Energy · γ · Transmission", size=(1000, 620))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._pw = _mk_plot("W_kin", "MeV")
        self._pg = _mk_plot("γ")
        self._pt = _mk_plot("Transmission", "%")
        self._pt.setYRange(0, 102)
        for p in (self._pg, self._pt):
            p.setXLink(self._pw)
        self._cw = filled_curve(self._pw, theme.ACCENT)
        self._cg = filled_curve(self._pg, "#a78bfa")
        self._ct = filled_curve(self._pt, "#4ade80", fill_alpha=48)
        for p in (self._pw, self._pg, self._pt):
            v.addWidget(p, stretch=1)
        self.attach_lattice_strip(self._pw)

    def refresh(self, results):
        if results is None:
            for c in (self._cw, self._cg, self._ct): c.setData([], [])
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        def arr(a): return np.asarray(getattr(results, a, []), dtype=float)
        def pair(c, v):
            if s.size and v.size == s.size: c.setData(s, v)
        pair(self._cw, arr("ref_w_kin"))
        pair(self._cg, arr("ref_gamma"))
        pair(self._ct, arr("transmission"))


class _LossPopup(_PopupPlot):
    def __init__(self, parent):
        super().__init__(parent, "Loss", size=(900, 500))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._p = _mk_plot("Cumulative loss", "%")
        self._c = filled_curve(self._p, "#f87171", fill_alpha=80)
        v.addWidget(self._p)

    def refresh(self, results):
        if results is None: self._c.setData([], []); return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        tr = np.asarray(getattr(results, "transmission", []), dtype=float)
        if s.size and tr.size == s.size:
            self._c.setData(s, 100 - tr)
        elif s.size:
            # Envelope mode tracks no losses; transmission is implicitly
            # 100% throughout, so cumulative loss is 0% along the line.
            self._c.setData(s, np.zeros_like(s))


class _PartranComparePopup(_PopupPlot):
    """Side-by-side overlay of HELIX and a TraceWin partran.out file.

    Top control row: file picker, axis selector (σ_x / σ_y / σ_φ /
    ε_x / ε_y / W), and a "clear overlay" button.  Below: two stacked
    plots — top is the absolute curves (HELIX solid, TraceWin dashed),
    bottom is the per-step relative difference (HELIX − TW) / TW.
    """

    _AXES = [
        ("sigma_x_mm",     "σ_x",    "mm",     "sigma_x"),
        ("sigma_y_mm",     "σ_y",    "mm",     "sigma_y"),
        ("sigma_phi_deg",  "σ_φ",    "deg",    "sigma_phi"),
        ("emit_x_mm_mrad", "ε_x",    "mm·mrad","emit_x"),
        ("emit_y_mm_mrad", "ε_y",    "mm·mrad","emit_y"),
        ("emit_z_deg_MeV", "ε_z",    "deg·MeV","emit_z"),
        ("ref_W_MeV",      "W",      "MeV",    "ref_w_kin"),
    ]

    def __init__(self, parent, state: AppState):
        super().__init__(parent,
                         "Compare with TraceWin partran",
                         size=(1100, 760))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        # Control row.
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        from PyQt6.QtWidgets import QComboBox, QPushButton
        load_btn = QPushButton("Open partran.out…")
        load_btn.clicked.connect(self._on_load)
        ctrl.addWidget(load_btn)
        clear_btn = QPushButton("Clear overlay")
        clear_btn.clicked.connect(self._on_clear)
        ctrl.addWidget(clear_btn)
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("axis:"))
        self._axis = QComboBox()
        for _, label, units, _src in self._AXES:
            self._axis.addItem(f"{label}  ({units})", userData=label)
        self._axis.currentIndexChanged.connect(self._redraw)
        ctrl.addWidget(self._axis)
        ctrl.addStretch(1)
        self._status = QLabel("no overlay loaded")
        self._status.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        ctrl.addWidget(self._status)
        v.addLayout(ctrl)
        # Two stacked plots.
        self._p_abs = _mk_plot("HELIX vs TraceWin", "")
        self._p_diff = _mk_plot("(HELIX − TW) / TW", "%")
        self._p_diff.setXLink(self._p_abs)
        self._c_helix = self._p_abs.plot(
            pen=pg.mkPen(theme.ACCENT, width=2.0), name="HELIX")
        self._c_tw = self._p_abs.plot(
            pen=pg.mkPen("#f97316", width=1.6,
                          style=Qt.PenStyle.DashLine), name="TraceWin")
        self._p_abs.addLegend(offset=(-10, 10))
        self._c_diff = self._p_diff.plot(
            pen=pg.mkPen("#a78bfa", width=1.4))
        v.addWidget(self._p_abs, stretch=2)
        v.addWidget(self._p_diff, stretch=1)

    def _on_load(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open TraceWin partran.out", "",
            "Partran output (*.out *.txt);;All files (*.*)")
        if not fname:
            return
        try:
            from linac_gen.io.tracewin_outputs import read_partran_out
            data = read_partran_out(fname)
        except Exception as exc:
            QMessageBox.critical(self, "Read failed", str(exc))
            return
        self._state.partran_overlay = data
        n = int(np.asarray(data.get("s_m", [])).size)
        self._status.setText(f"loaded {n} rows from {fname}")
        self._redraw()

    def _on_clear(self):
        self._state.partran_overlay = None
        self._status.setText("no overlay loaded")
        self._redraw()

    def refresh(self, results):
        self._redraw()

    def _redraw(self):
        results = self._state.results
        ovl = self._state.partran_overlay
        sel = self._axis.currentData()
        # Find the axis spec.
        spec = next((a for a in self._AXES if a[1] == sel), self._AXES[0])
        ovl_key, label, units, helix_attr = spec
        # HELIX curve.
        if results is None:
            self._c_helix.setData([], [])
        else:
            s_h = np.asarray(getattr(results, "s", []), dtype=float) * 1e-3
            y_h = np.asarray(getattr(results, helix_attr, []), dtype=float)
            if s_h.size and y_h.size == s_h.size:
                self._c_helix.setData(s_h, y_h)
            else:
                self._c_helix.setData([], [])
        # TraceWin overlay.
        if ovl is None:
            self._c_tw.setData([], [])
            self._c_diff.setData([], [])
            self._p_abs.setLabel("left", label, units=units)
            self._p_abs.setLabel("bottom", "s", units="m")
            return
        s_t = np.asarray(ovl.get("s_m", []), dtype=float)
        y_t = np.asarray(ovl.get(ovl_key, []), dtype=float)
        if s_t.size and y_t.size == s_t.size:
            self._c_tw.setData(s_t, y_t)
        else:
            self._c_tw.setData([], [])
        self._p_abs.setLabel("left", label, units=units)
        self._p_abs.setLabel("bottom", "s", units="m")
        # Relative difference (HELIX − TW) / TW × 100, resampled onto the
        # HELIX s-grid.  Use linear interpolation on the TW curve.
        if (results is not None and s_h.size and y_h.size == s_h.size and
                s_t.size and y_t.size == s_t.size):
            try:
                y_t_interp = np.interp(s_h, s_t, y_t)
                with np.errstate(divide="ignore", invalid="ignore"):
                    diff = 100.0 * (y_h - y_t_interp) / np.where(
                        y_t_interp != 0, y_t_interp, np.nan)
                self._c_diff.setData(s_h, diff)
            except Exception:
                self._c_diff.setData([], [])
        else:
            self._c_diff.setData([], [])
        self._p_diff.setLabel("bottom", "s", units="m")


class _HaloPopup(_PopupPlot):
    """Halo parameter H = ⟨u⁴⟩/⟨u²⟩² − 1 vs s, both transverse planes.

    H = 0 for a uniformly distributed core (round waterbag); H = 2 for a
    Gaussian; H ≫ 2 means heavy non-Gaussian tails.  The popup overlays
    the x- and y-plane traces with shared x-axis so a halo growth event
    is visible against the matching σ growth in the rms popup.
    """

    def __init__(self, parent):
        super().__init__(parent, "Halo parameter  —  H_x · H_y vs s",
                         size=(960, 540))
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._p = _mk_plot("Halo H = ⟨u⁴⟩/⟨u²⟩² − 1", "")
        self._cx = self._p.plot(pen=pg.mkPen("#60a5fa", width=2.0),
                                 name="H_x")
        self._cy = self._p.plot(pen=pg.mkPen("#f472b6", width=2.0),
                                 name="H_y")
        # Reference horizontal lines at H=0 (uniform) and H=2 (Gaussian).
        self._p.addLine(y=0.0, pen=pg.mkPen("#475569",
                                             width=1.0,
                                             style=Qt.PenStyle.DashLine))
        self._p.addLine(y=2.0, pen=pg.mkPen("#94a3b8",
                                             width=1.0,
                                             style=Qt.PenStyle.DashLine))
        self._p.addLegend(offset=(-10, 10))
        self._p.plot([], [], pen=pg.mkPen("#60a5fa", width=2.0), name="H_x")
        self._p.plot([], [], pen=pg.mkPen("#f472b6", width=2.0), name="H_y")
        v.addWidget(self._p)
        self.attach_lattice_strip(self._p)

    def refresh(self, results):
        if results is None:
            self._cx.setData([], []); self._cy.setData([], []); return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        hx = np.asarray(getattr(results, "halo_x", []), dtype=float)
        hy = np.asarray(getattr(results, "halo_y", []), dtype=float)
        if s.size and hx.size == s.size:
            self._cx.setData(s, hx)
        else:
            self._cx.setData([], [])
        if s.size and hy.size == s.size:
            self._cy.setData(s, hy)
        else:
            self._cy.setData([], [])


class _ApertureLossPopup(_PopupPlot):
    """Per-particle loss locations overlaid on the aperture profile.

    Two stacked plots (x and y).  Solid envelope lines trace the bore
    radius rx(s)/ry(s); orange scatter dots mark each (s_loss, x_loss)
    or (s_loss, y_loss) where a particle hit the aperture.  A side
    table groups losses by element so hot spots are obvious at a glance.
    """

    def __init__(self, parent, state: AppState):
        super().__init__(parent,
                         "Aperture-profile losses  —  per-particle hit map",
                         size=(1200, 760))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # Top half: two stacked s-vs-{x,y} plots with aperture envelope.
        plots_row = QHBoxLayout(); plots_row.setSpacing(8)
        plots_col = QVBoxLayout(); plots_col.setSpacing(6)
        self._px = _mk_plot("x   (mm)", "mm")
        self._py = _mk_plot("y   (mm)", "mm")
        self._py.setXLink(self._px)
        # Aperture envelope curves (thick, dim).
        self._ap_x = self._px.plot(pen=pg.mkPen("#f97316", width=2.0))
        self._ap_x_neg = self._px.plot(pen=pg.mkPen("#f97316", width=2.0))
        self._ap_y = self._py.plot(pen=pg.mkPen("#f97316", width=2.0))
        self._ap_y_neg = self._py.plot(pen=pg.mkPen("#f97316", width=2.0))
        # Loss-location scatter (small bright orange dots).
        self._loss_x = pg.ScatterPlotItem(
            pen=pg.mkPen("#fde047", width=0.5),
            brush=pg.mkBrush("#facc15"), size=5, pxMode=True,
        )
        self._loss_y = pg.ScatterPlotItem(
            pen=pg.mkPen("#fde047", width=0.5),
            brush=pg.mkBrush("#facc15"), size=5, pxMode=True,
        )
        self._px.addItem(self._loss_x)
        self._py.addItem(self._loss_y)
        plots_col.addWidget(self._px); plots_col.addWidget(self._py)
        plots_row.addLayout(plots_col, stretch=3)

        # Right side: per-element loss summary list.
        from PyQt6.QtWidgets import QListWidget
        self._summary = QListWidget()
        self._summary.setStyleSheet(
            f"QListWidget {{ background:{theme.BG_INSET}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:3px; "
            f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; "
            f"font-size:11px; padding:4px; }}"
        )
        self._summary.setMinimumWidth(260)
        plots_row.addWidget(self._summary, stretch=1)
        v.addLayout(plots_row, stretch=1)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        v.addWidget(self._status)

    def refresh(self, results):
        # Aperture envelope from the live lattice.
        from linac_gen.analysis.aperture_profile import aperture_profile
        lat = self._state.lattice
        if lat is not None:
            try:
                s_mm, rx, ry = aperture_profile(lat)
                self._ap_x.setData(s_mm, rx)
                self._ap_x_neg.setData(s_mm, -rx)
                self._ap_y.setData(s_mm, ry)
                self._ap_y_neg.setData(s_mm, -ry)
            except Exception:
                pass
        # Per-particle loss table from the recorder's beam reference.
        beam = getattr(results, "beam", None) if results is not None else None
        table = getattr(beam, "loss_table", None) if beam is not None else None
        self._summary.clear()
        if table is None or len(table) == 0:
            for sc in (self._loss_x, self._loss_y):
                sc.setData([], [])
            self._status.setText(
                "No losses recorded.  Run a multi-particle simulation "
                "with apertures (Aperture / DIAG_APERTURE) for hit "
                "locations.")
            return
        s_loss = np.asarray(table["s"]).astype(float)
        x_loss = np.asarray(table["x"]).astype(float)
        y_loss = np.asarray(table["y"]).astype(float)
        elem = np.asarray(table["element_name"]).astype(str)
        self._loss_x.setData(s_loss, x_loss)
        self._loss_y.setData(s_loss, y_loss)
        # Group by element.
        from collections import Counter
        counts = Counter(elem.tolist())
        n_total = int(len(table))
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / max(n_total, 1)
            self._summary.addItem(f"{name:<28} {n:>6}  ({pct:5.1f} %)")
        self._status.setText(
            f"{n_total} particle losses across {len(counts)} elements "
            f"(s ∈ [{float(s_loss.min()):.0f}, "
            f"{float(s_loss.max()):.0f}] mm)"
        )


class _IbsPopup(_PopupPlot):
    """Intra-beam stripping (H⁻ only) — Lebedev arXiv:1207.5492.

    Computes per-step IBS loss curves from the existing tracking results.
    Has its OWN duty-factor setting (independent of ``BeamConfig.duty_cycle``)
    persisted to QSettings — IBS radiation budgets are usually evaluated
    at a different duty point than the run's beam current.
    """

    _QSETTINGS_GROUP = "linac_gen/ibs"

    def __init__(self, parent, state: AppState):
        super().__init__(parent, "Intra-beam stripping  —  H⁻ losses",
                         size=(1100, 800))
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import (
            QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
        )
        self._state = state
        self._settings = make_settings("linac_gen", "linac_gen_gui")

        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # ----- top control row ---------------------------------------------
        ctrl = QHBoxLayout(); ctrl.setSpacing(14)

        f1 = QFormLayout(); f1.setSpacing(4)
        self._duty_spin = QDoubleSpinBox()
        self._duty_spin.setRange(0.0, 1.0)
        self._duty_spin.setDecimals(4)
        self._duty_spin.setSingleStep(0.01)
        self._duty_spin.setValue(float(self._settings.value(
            f"{self._QSETTINGS_GROUP}/duty_factor", 0.02, type=float)))
        self._duty_spin.setToolTip(
            "IBS duty factor (independent of the BeamConfig duty cycle).\n"
            "Multiplies the W/m and integral-W curves only.")
        f1.addRow("IBS duty factor", self._duty_spin)
        ctrl.addLayout(f1)

        f2 = QFormLayout(); f2.setSpacing(4)
        self._curr_spin = QDoubleSpinBox()
        self._curr_spin.setRange(0.0, 1000.0)
        self._curr_spin.setDecimals(4)
        self._curr_spin.setSingleStep(0.1)
        self._curr_spin.setSuffix(" mA")
        # Default to whatever the active run used.
        cfg = getattr(state, "beam_config", None)
        self._curr_spin.setValue(float(getattr(cfg, "current", 5.0))
                                  if cfg is not None else 5.0)
        f2.addRow("Peak current", self._curr_spin)
        ctrl.addLayout(f2)

        f3 = QFormLayout(); f3.setSpacing(4)
        self._theta_combo = QComboBox()
        self._theta_combo.addItem("dp/p (paper)", userData="dp_over_p")
        self._theta_combo.addItem("dv/v (Ostiguy)", userData="dv_over_v")
        self._theta_combo.setToolTip(
            "Longitudinal coordinate inside the radical of Lebedev Eq. (7).\n"
            "Paper convention is dp/p; Ostiguy's script substitutes dv/v.")
        f3.addRow("θ_s convention", self._theta_combo)
        ctrl.addLayout(f3)

        self._chk_nonconst = QCheckBox("Non-constant σ_strip")
        self._chk_nonconst.setToolTip(
            "Use the per-step β-dependent σ_H from Eq. (8) instead of a\n"
            "constant σ_max (~the plateau value).  Recommended outside the\n"
            "plateau (PIP-II SC linac: β ≳ 0.5).")
        ctrl.addWidget(self._chk_nonconst)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        # ----- summary line -------------------------------------------------
        self._summary = QLabel("—")
        self._summary.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:12px; padding:2px 0;")
        v.addWidget(self._summary)

        # ----- four stacked plots, x-linked --------------------------------
        self._p_rate = _mk_plot("Local fractional loss", "1/m")
        self._p_pwr  = _mk_plot("Local power loss", "W/m")
        self._p_iL   = _mk_plot("Cumulative fractional loss", "")
        self._p_iP   = _mk_plot("Cumulative power loss", "W")
        for p in (self._p_pwr, self._p_iL, self._p_iP):
            p.setXLink(self._p_rate)
        self._p_rate.setLogMode(False, True)
        self._c_rate = self._p_rate.plot(pen=curve_pen("#f87171"),
                                         name="dN/N/ds")
        self._c_pwr  = filled_curve(self._p_pwr, "#fbbf24", name="dP/ds")
        self._c_iL   = self._p_iL.plot(pen=curve_pen("#a3e635"),
                                       name="∫ dN/N")
        self._c_iP   = filled_curve(self._p_iP, "#60a5fa", name="∫ dP")
        for p in (self._p_rate, self._p_pwr, self._p_iL, self._p_iP):
            p.setLabel("bottom", "s", units="m")
            v.addWidget(p, stretch=1)

        # ----- in-plot diagnostic overlay (visible only on guard branches) -
        self._overlay = pg.TextItem(
            "", anchor=(0.5, 0.5), color="#fca5a5",
            border=pg.mkPen("#7f1d1d", width=1),
            fill=pg.mkBrush(20, 20, 30, 220),
        )
        # html lets us bump font-size; pyqtgraph TextItem accepts setHtml.
        self._overlay.setVisible(False)
        # Add to the rate-plot's central viewbox.
        self._p_rate.addItem(self._overlay, ignoreBounds=True)

        # ----- wire control changes to recompute ---------------------------
        self._duty_spin.valueChanged.connect(self._on_controls_changed)
        self._curr_spin.valueChanged.connect(self._on_controls_changed)
        self._theta_combo.currentIndexChanged.connect(self._on_controls_changed)
        self._chk_nonconst.toggled.connect(self._on_controls_changed)

    # ------------------------------------------------------------------
        self.attach_lattice_strip(self._p_rate)
    def _on_controls_changed(self, *args) -> None:
        # Persist the duty factor independently of any BeamConfig.
        self._settings.setValue(f"{self._QSETTINGS_GROUP}/duty_factor",
                                 float(self._duty_spin.value()))
        self.refresh(self._state.results)

    def _show_overlay(self, html: str) -> None:
        """Display a centered diagnostic message and clear all curves."""
        for c in (self._c_rate, self._c_pwr, self._c_iL, self._c_iP):
            c.setData([], [])
        self._overlay.setHtml(
            f"<div style='font-size:14pt; color:#fca5a5; "
            f"padding:10px; line-height:1.4'>{html}</div>"
        )
        vb = self._p_rate.getViewBox()
        # Anchor the overlay at the centre of the current view; pin it
        # there with a sigRangeChanged hook so panning doesn't move it.
        rng = vb.viewRange()
        cx = 0.5 * (rng[0][0] + rng[0][1])
        cy = 0.5 * (rng[1][0] + rng[1][1])
        self._overlay.setPos(cx, cy)
        self._overlay.setVisible(True)

    def _hide_overlay(self) -> None:
        self._overlay.setVisible(False)

    def refresh(self, results) -> None:
        cfg = getattr(self._state, "beam_config", None)
        if cfg is None or results is None:
            self._summary.setText("Run a tracking simulation first.")
            self._show_overlay(
                "No tracking results yet.<br>"
                "Click <b>Run</b> in the Beam tab, then re-open this popup.")
            return
        if getattr(cfg, "species", None) != "H-":
            self._summary.setText(
                f"IBS analysis applies only to H⁻ beams; current species = "
                f"{cfg.species!r}.")
            self._show_overlay(
                f"IBS applies to <b>H⁻ beams only</b>.<br>"
                f"Current species: <code>{cfg.species!r}</code>.<br>"
                f"Switch Species → <b>H-</b> in the Beam tab, click "
                f"<b>Apply</b>, and re-run.")
            return
        # Need σ-matrix entries (1,1)/(3,3) and σ_W to compute θ's.
        sm = getattr(results, "sigma_matrix", None)
        if sm is None or len(sm) == 0:
            self._summary.setText(
                "Tracking results do not include σ_matrix — re-run with "
                "diagnostics enabled to populate σ_x', σ_y'.")
            self._show_overlay(
                "σ-matrix missing from results.<br>"
                "Re-run tracking with diagnostics enabled.")
            return

        from linac_gen.analysis.intrabeam_stripping import ibs_loss
        try:
            out = ibs_loss(
                results, cfg,
                duty_factor=float(self._duty_spin.value()),
                current_mA=float(self._curr_spin.value()),
                non_constant_xs=bool(self._chk_nonconst.isChecked()),
                theta_z_convention=str(self._theta_combo.currentData()),
            )
        except Exception as exc:
            self._summary.setText(f"IBS calculation failed: {exc}")
            self._show_overlay(
                f"IBS computation failed:<br><code>{exc}</code><br><br>"
                f"Re-run tracking after the latest fix; if persistent, "
                f"check that <code>mass_mev</code> and "
                f"<code>ref_frequency</code> are populated on Results.")
            return

        s_m = out.s
        # Plot positive-only loss rate on log axis; pyqtgraph drops <=0.
        rate = np.where(out.loss_rate_per_m > 0,
                        out.loss_rate_per_m, np.nan)
        self._c_rate.setData(s_m, rate)
        self._c_pwr.setData(s_m, out.power_loss_per_m_W)
        self._c_iL.setData(s_m, out.integral_loss)
        self._c_iP.setData(s_m, out.integral_power_loss_W)
        self._hide_overlay()

        n_str = (f"{out.n_per_bunch:.3e}"
                  if np.isfinite(out.n_per_bunch) else "—")
        self._summary.setText(
            f"DF = {out.duty_factor:.3g}   I_peak = {out.current_mA:.3f} mA   "
            f"N/bunch = {n_str}   total loss = "
            f"{out.integral_loss[-1]:.3e}   total ⟨P⟩ = "
            f"{out.integral_power_loss_W[-1]:.3f} W   "
            f"({'non-const σ' if out.non_constant_xs else 'const σ_max'}, "
            f"θ_s = {out.theta_z_convention.replace('_over_', '/')})"
        )


class _EnsemblePopup(_PopupPlot):
    """Error-study ensemble plots: σ_x, σ_y mean ± std envelopes plus a
    transmission histogram across all Monte Carlo seeds.

    Reads ``state.error_study_results`` populated by the Error Study tab.
    Renders nothing if no study has run yet (placeholder banner).
    """

    def __init__(self, parent, state: AppState):
        super().__init__(parent, "Error study  —  ensemble σ envelopes",
                         size=(1100, 700))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        banner = QLabel(
            "Mean ± 1σ across all Monte Carlo seeds.  Bottom panel: final "
            "transmission histogram.  Run the Error Study tab first to populate "
            "this view."
        )
        banner.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px; padding:4px 8px;"
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_0};"
            f"border-radius:3px;"
        )
        banner.setWordWrap(True)
        v.addWidget(banner)

        # Top: σ_x mean + ±1σ band
        self._p_sx = _mk_plot("σ_x ensemble", "mm")
        self._sx_mean = self._p_sx.plot(pen=pg.mkPen("#60a5fa", width=2),
                                         name="mean")
        self._sx_band_hi = self._p_sx.plot(pen=pg.mkPen("#60a5fa", width=1,
                                                          style=Qt.PenStyle.DashLine))
        self._sx_band_lo = self._p_sx.plot(pen=pg.mkPen("#60a5fa", width=1,
                                                          style=Qt.PenStyle.DashLine))
        v.addWidget(self._p_sx, stretch=1)

        # Middle: σ_y
        self._p_sy = _mk_plot("σ_y ensemble", "mm")
        self._sy_mean = self._p_sy.plot(pen=pg.mkPen("#a3e635", width=2),
                                         name="mean")
        self._sy_band_hi = self._p_sy.plot(pen=pg.mkPen("#a3e635", width=1,
                                                          style=Qt.PenStyle.DashLine))
        self._sy_band_lo = self._p_sy.plot(pen=pg.mkPen("#a3e635", width=1,
                                                          style=Qt.PenStyle.DashLine))
        v.addWidget(self._p_sy, stretch=1)

        # Bottom: transmission histogram
        self._p_th = _mk_plot("Final transmission histogram", "%")
        self._th_bars = pg.BarGraphItem(x=[], height=[], width=0.5,
                                         brush=pg.mkBrush("#fb923c"))
        self._p_th.addItem(self._th_bars)
        v.addWidget(self._p_th, stretch=1)

        # Status / summary line
        self._summary = QLabel("")
        self._summary.setStyleSheet(
            f"color:{theme.TEXT_1}; font-family:{theme.FONT_MONO};"
            f" font-size:12px; padding:4px;"
        )
        v.addWidget(self._summary)

    def refresh(self, results) -> None:  # ``results`` ignored — read from state
        study = getattr(self._state, "error_study_results", None)
        if study is None or getattr(study, "n_seeds", 0) == 0:
            self._sx_mean.setData([], []); self._sx_band_hi.setData([], [])
            self._sx_band_lo.setData([], [])
            self._sy_mean.setData([], []); self._sy_band_hi.setData([], [])
            self._sy_band_lo.setData([], [])
            self._th_bars.setOpts(x=[], height=[])
            self._summary.setText("No error study results — run the Error Study tab.")
            return
        try:
            sx_mean = study.mean("sigma_x"); sx_std = study.std("sigma_x")
            sy_mean = study.mean("sigma_y"); sy_std = study.std("sigma_y")
            # Pull s from the first recorder (all seeds share the lattice
            # element list, hence identical s arrays).
            s = np.asarray(study._recorders[0].s, dtype=float)
        except Exception as exc:                                 # pragma: no cover
            self._summary.setText(f"render failed: {exc}")
            return
        self._sx_mean.setData(s, sx_mean)
        self._sx_band_hi.setData(s, sx_mean + sx_std)
        self._sx_band_lo.setData(s, np.maximum(sx_mean - sx_std, 0.0))
        self._sy_mean.setData(s, sy_mean)
        self._sy_band_hi.setData(s, sy_mean + sy_std)
        self._sy_band_lo.setData(s, np.maximum(sy_mean - sy_std, 0.0))

        # Final transmission histogram.  Skip any seed whose recorder has an
        # empty transmission array (a degenerate/zero-step seed) so a single
        # bad seed can't raise IndexError here (this block is outside the
        # try/except above) and abort the whole refresh.
        finals_list = []
        for r in study._recorders:
            t = np.asarray(r.transmission)
            if t.size:
                finals_list.append(float(t[-1]))
        finals = np.array(finals_list, dtype=float)
        if finals.size:
            bins = np.linspace(min(finals.min(), 95.0), 100.0, 21)
            hist, edges = np.histogram(finals, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = float(edges[1] - edges[0]) * 0.9
            self._th_bars.setOpts(x=centers, height=hist, width=width)
            # Compute the summary from the already-filtered ``finals`` array —
            # study.transmission_stats() indexes r.transmission[-1] over ALL
            # recorders, so the same degenerate empty-transmission seed the
            # histogram loop above skips would raise IndexError here and leave
            # the summary line stale.  np.mean/min/max/std over finals is
            # numerically identical for non-degenerate ensembles.
            self._summary.setText(
                f"n_seeds = {study.n_seeds}   "
                f"transmission: mean = {float(np.mean(finals)):.3f} %, "
                f"min = {float(np.min(finals)):.3f} %, "
                f"max = {float(np.max(finals)):.3f} %, "
                f"σ = {float(np.std(finals)):.3f} %"
            )
        else:
            self._th_bars.setOpts(x=[], height=[])


class _MagStripPopup(_PopupPlot):
    """Magnetic / Lorentz stripping (H⁻) — Folsom 2021 PRAB 24:074201 Eq. (5).

    Sister to ``_IbsPopup``: same 4-stack plot layout (rate · power ·
    cumulative loss · cumulative power), same control-row idiom, same
    QSettings persistence — but evaluating the design |B| at each step
    via element-type dispatch instead of beam-distribution moments.
    """

    _QSETTINGS_GROUP = "linac_gen/magnetic_stripping"

    def __init__(self, parent, state: AppState):
        super().__init__(parent, "Magnetic stripping  —  H⁻ Lorentz losses",
                         size=(1100, 800))
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import (
            QComboBox, QDoubleSpinBox, QFormLayout,
        )
        self._state = state
        self._settings = make_settings("linac_gen", "linac_gen_gui")

        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        # ----- top control row ---------------------------------------------
        ctrl = QHBoxLayout(); ctrl.setSpacing(14)

        f1 = QFormLayout(); f1.setSpacing(4)
        self._duty_spin = QDoubleSpinBox()
        self._duty_spin.setRange(0.0, 1.0)
        self._duty_spin.setDecimals(4)
        self._duty_spin.setSingleStep(0.01)
        self._duty_spin.setValue(float(self._settings.value(
            f"{self._QSETTINGS_GROUP}/duty_factor", 0.02, type=float)))
        self._duty_spin.setToolTip(
            "Duty factor for the W/m and integral-W curves.\n"
            "Independent of the BeamConfig duty cycle.")
        f1.addRow("Duty factor", self._duty_spin)
        ctrl.addLayout(f1)

        f2 = QFormLayout(); f2.setSpacing(4)
        self._curr_spin = QDoubleSpinBox()
        self._curr_spin.setRange(0.0, 1000.0)
        self._curr_spin.setDecimals(4)
        self._curr_spin.setSingleStep(0.1)
        self._curr_spin.setSuffix(" mA")
        cfg = getattr(state, "beam_config", None)
        self._curr_spin.setValue(float(getattr(cfg, "current", 5.0))
                                  if cfg is not None else 5.0)
        f2.addRow("Peak current", self._curr_spin)
        ctrl.addLayout(f2)

        f3 = QFormLayout(); f3.setSpacing(4)
        self._quad_combo = QComboBox()
        self._quad_combo.addItem("|B| at 2σ (default)", userData="2sigma")
        self._quad_combo.addItem("|B| at 1σ",          userData="1sigma")
        self._quad_combo.addItem("|B| at pole tip",    userData="pole")
        self._quad_combo.setToolTip(
            "Quadrupole |B| evaluation radius.  In a quad, |B|=G·r — particles\n"
            "near the axis see almost no field; this picks the representative\n"
            "radius for the loss-rate integration.  '2σ' is a typical halo\n"
            "approximation; 'pole tip' gives a conservative upper bound.")
        f3.addRow("Quad B", self._quad_combo)
        ctrl.addLayout(f3)

        f4 = QFormLayout(); f4.setSpacing(4)
        self._fmap_combo = QComboBox()
        self._fmap_combo.addItem("|B| max over grid", userData="max")
        self._fmap_combo.addItem("|B| on axis",        userData="on_axis")
        self._fmap_combo.setToolTip(
            "How to sample |B| inside FIELD_MAP elements.\n"
            "'max' is conservative for off-centred halos; 'on_axis' is\n"
            "appropriate for tightly centred reference orbits.")
        f4.addRow("FieldMap B", self._fmap_combo)
        ctrl.addLayout(f4)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        # ----- summary line -------------------------------------------------
        self._summary = QLabel("—")
        self._summary.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:12px; padding:2px 0;")
        v.addWidget(self._summary)

        # ----- four stacked plots, x-linked --------------------------------
        self._p_rate = _mk_plot("Local fractional loss", "1/m")
        self._p_pwr  = _mk_plot("Local power loss", "W/m")
        self._p_iL   = _mk_plot("Cumulative fractional loss", "")
        self._p_iP   = _mk_plot("Cumulative power loss", "W")
        for p in (self._p_pwr, self._p_iL, self._p_iP):
            p.setXLink(self._p_rate)
        self._p_rate.setLogMode(False, True)
        self._c_rate = self._p_rate.plot(pen=curve_pen("#fb923c"),
                                         name="dN/N/ds")
        self._c_pwr  = filled_curve(self._p_pwr, "#fbbf24", name="dP/ds")
        self._c_iL   = self._p_iL.plot(pen=curve_pen("#a3e635"),
                                       name="∫ dN/N")
        self._c_iP   = filled_curve(self._p_iP, "#60a5fa", name="∫ dP")
        for p in (self._p_rate, self._p_pwr, self._p_iL, self._p_iP):
            p.setLabel("bottom", "s", units="m")
            v.addWidget(p, stretch=1)

        # ----- diagnostic overlay (visible on guard branches) --------------
        self._overlay = pg.TextItem(
            "", anchor=(0.5, 0.5), color="#fca5a5",
            border=pg.mkPen("#7f1d1d", width=1),
            fill=pg.mkBrush(20, 20, 30, 220),
        )
        self._overlay.setVisible(False)
        self._p_rate.addItem(self._overlay, ignoreBounds=True)

        # ----- wire control changes to recompute ---------------------------
        self._duty_spin.valueChanged.connect(self._on_controls_changed)
        self._curr_spin.valueChanged.connect(self._on_controls_changed)
        self._quad_combo.currentIndexChanged.connect(self._on_controls_changed)
        self._fmap_combo.currentIndexChanged.connect(self._on_controls_changed)

    # ------------------------------------------------------------------
        self.attach_lattice_strip(self._p_rate)
    def _on_controls_changed(self, *args) -> None:
        self._settings.setValue(f"{self._QSETTINGS_GROUP}/duty_factor",
                                 float(self._duty_spin.value()))
        self.refresh(self._state.results)

    def _show_overlay(self, html: str) -> None:
        for c in (self._c_rate, self._c_pwr, self._c_iL, self._c_iP):
            c.setData([], [])
        self._overlay.setHtml(
            f"<div style='font-size:14pt; color:#fca5a5; "
            f"padding:10px; line-height:1.4'>{html}</div>"
        )
        vb = self._p_rate.getViewBox()
        rng = vb.viewRange()
        cx = 0.5 * (rng[0][0] + rng[0][1])
        cy = 0.5 * (rng[1][0] + rng[1][1])
        self._overlay.setPos(cx, cy)
        self._overlay.setVisible(True)

    def _hide_overlay(self) -> None:
        self._overlay.setVisible(False)

    def refresh(self, results) -> None:
        cfg = getattr(self._state, "beam_config", None)
        lattice = getattr(self._state, "lattice", None)
        if cfg is None or results is None:
            self._summary.setText("Run a tracking simulation first.")
            self._show_overlay(
                "No tracking results yet.<br>"
                "Click <b>Run</b> in the Beam tab, then re-open this popup.")
            return
        if lattice is None:
            self._summary.setText("Load a lattice to evaluate magnetic fields.")
            self._show_overlay(
                "No lattice loaded.<br>"
                "Open a TraceWin .dat file, then re-open this popup.")
            return
        if getattr(cfg, "species", None) != "H-":
            self._summary.setText(
                f"Lorentz stripping applies only to H⁻ beams; current "
                f"species = {cfg.species!r}.")
            self._show_overlay(
                f"Lorentz stripping applies to <b>H⁻ beams only</b>.<br>"
                f"Current species: <code>{cfg.species!r}</code>.<br>"
                f"Switch Species → <b>H-</b> in the Beam tab, click "
                f"<b>Apply</b>, and re-run.")
            return

        from linac_gen.analysis.magnetic_stripping import (
            magnetic_stripping_loss,
        )
        try:
            out = magnetic_stripping_loss(
                results, lattice, cfg,
                duty_factor=float(self._duty_spin.value()),
                current_mA=float(self._curr_spin.value()),
                quad_b_scale=str(self._quad_combo.currentData()),
                fieldmap_sample=str(self._fmap_combo.currentData()),
            )
        except Exception as exc:
            self._summary.setText(f"Magnetic-stripping calculation failed: {exc}")
            self._show_overlay(
                f"Computation failed:<br><code>{exc}</code><br><br>"
                f"Re-run tracking; if persistent, check that "
                f"<code>mass_mev</code>, <code>element_names</code>, and "
                f"<code>ref_gamma/beta</code> are populated on Results.")
            return

        s_m = out.s
        # Plot positive-only loss rate on log axis; pyqtgraph drops <=0.
        rate = np.where(out.loss_rate_per_m > 0,
                        out.loss_rate_per_m, np.nan)
        self._c_rate.setData(s_m, rate)
        self._c_pwr.setData(s_m, out.power_loss_per_m_W)
        self._c_iL.setData(s_m, out.integral_loss)
        self._c_iP.setData(s_m, out.integral_power_loss_W)
        self._hide_overlay()

        b_max = float(np.max(out.B_T)) if out.B_T.size else 0.0
        peak_rate = float(np.max(out.loss_rate_per_m)) if out.loss_rate_per_m.size else 0.0
        peak_idx = (int(np.argmax(out.loss_rate_per_m))
                    if out.loss_rate_per_m.size else 0)
        peak_s = float(s_m[peak_idx]) if out.loss_rate_per_m.size else 0.0
        self._summary.setText(
            f"DF = {out.duty_factor:.3g}   I_peak = {out.current_mA:.3f} mA   "
            f"|B|_max = {b_max:.3f} T   peak rate = {peak_rate:.3e} /m at "
            f"s = {peak_s:.2f} m   total loss = "
            f"{out.integral_loss[-1]:.3e}   total ⟨P⟩ = "
            f"{out.integral_power_loss_W[-1]:.3f} W   "
            f"(quad: {out.quad_b_scale}, fmap: {out.fieldmap_sample})"
        )


class _CentroidPopup(_PopupPlot):
    """Achieved orbit ⟨x⟩/⟨y⟩/⟨φ⟩ along s, with the DIAG_POSITION goal
    orbit overlaid when the loaded lattice carries targets — pink points
    are what the deck (or a loaded BPM-targets file) asked for, curves
    are what the beam actually did, and the banner reports the rms gap.
    """

    _GOAL = "#f472b6"

    def __init__(self, parent, state: AppState = None):
        super().__init__(parent, "Centroid  —  ⟨x⟩ · ⟨y⟩ · ⟨φ⟩", size=(1000, 620))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._goal_lbl = QLabel("")
        self._goal_lbl.setStyleSheet(
            f"color:{self._GOAL}; font-family:{theme.FONT_MONO}; "
            f"font-size:12px; padding:0 2px;")
        self._goal_lbl.setVisible(False)
        v.addWidget(self._goal_lbl)
        self._px = _mk_plot("⟨x⟩", "mm")
        self._py = _mk_plot("⟨y⟩", "mm")
        self._pp = _mk_plot("⟨φ⟩", "deg")
        for p in (self._py, self._pp): p.setXLink(self._px)
        self._cx = self._px.plot(pen=curve_pen(theme.ACCENT))
        self._cy = self._py.plot(pen=curve_pen("#a3e635"))
        self._cp = self._pp.plot(pen=curve_pen("#fbbf24"))
        # Goal-orbit overlays (DIAG_POSITION targets at the BPM rows).
        self._tx = self._px.plot(pen=None, symbol="o", symbolSize=8,
                                 symbolPen=pg.mkPen(self._GOAL, width=1.5),
                                 symbolBrush=None)
        self._ty = self._py.plot(pen=None, symbol="o", symbolSize=8,
                                 symbolPen=pg.mkPen(self._GOAL, width=1.5),
                                 symbolBrush=None)
        for p in (self._px, self._py, self._pp):
            v.addWidget(p, stretch=1)
        self.attach_lattice_strip(self._px)

    def refresh(self, results):
        if results is None:
            for c in (self._cx, self._cy, self._cp,
                      self._tx, self._ty):
                c.setData([], [])
            self._goal_lbl.setVisible(False)
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        c = getattr(results, "centroid", None)
        have_mp = bool(c and len(c) == s.size)
        if have_mp:
            arr = np.array(c)
            self._cx.setData(s, arr[:, 0])
            self._cy.setData(s, arr[:, 2])
            self._cp.setData(s, arr[:, 4])
        elif s.size:
            # Results without a centroid list (legacy/foreign objects —
            # envelope results carry one now) → plot flat zeros.
            zeros = np.zeros_like(s)
            self._cx.setData(s, zeros)
            self._cy.setData(s, zeros)
            self._cp.setData(s, zeros)
        self._refresh_goals(results, s, np.array(c) if have_mp else None)

    def _refresh_goals(self, results, s, arr):
        """Overlay the DIAG_POSITION goal orbit and report the rms gap."""
        lat = self._state.lattice if self._state is not None else None
        sx_pts, tx_pts, sy_pts, ty_pts = [], [], [], []
        res_x, res_y = [], []
        if lat is not None and s.size:
            from linac_gen.matching.constraints import _effective_targets
            exit_idx = getattr(results, "element_exit_idx", None) or []
            for ei, el in enumerate(lat.elements):
                if not getattr(el, "is_bpm", False):
                    continue
                tx, ty, _w = _effective_targets(el)
                if tx is None and ty is None:
                    continue
                row = int(exit_idx[ei]) if ei < len(exit_idx) else ei + 1
                if row >= s.size:
                    continue
                if tx is not None:
                    sx_pts.append(s[row]); tx_pts.append(tx)
                    if arr is not None:
                        res_x.append(arr[row, 0] - tx)
                if ty is not None:
                    sy_pts.append(s[row]); ty_pts.append(ty)
                    if arr is not None:
                        res_y.append(arr[row, 2] - ty)
        self._tx.setData(sx_pts, tx_pts)
        self._ty.setData(sy_pts, ty_pts)
        n_goal = len(sx_pts) + len(sy_pts)
        if not n_goal:
            self._goal_lbl.setVisible(False)
            return
        if arr is not None:
            rx = float(np.sqrt(np.mean(np.square(res_x)))) if res_x else 0.0
            ry = float(np.sqrt(np.mean(np.square(res_y)))) if res_y else 0.0
            self._goal_lbl.setText(
                f"○ goal orbit (DIAG_POSITION / targets file): "
                f"{len(sx_pts)}x + {len(sy_pts)}y points — achieved-vs-goal "
                f"rms Δx = {rx:.4f} mm, Δy = {ry:.4f} mm")
        else:
            self._goal_lbl.setText(
                f"○ goal orbit shown ({len(sx_pts)}x + {len(sy_pts)}y "
                "points) — these results carry no centroid, run a "
                "simulation to compare")
        self._goal_lbl.setVisible(True)


class _PhaseSpacePopup(_PopupPlot):
    _FOLD_TIP = (
        "Fold Δφ into ONE RF period about the bunch centroid.\n"
        "Particles a full period away are in the neighbouring bucket "
        "of the same bunch train — the same physical bunch — and "
        "this is what TraceWin/Toutatis display.\n"
        "Uncheck to see the raw unwrapped phase (the bunch train).\n\n"
        "This is a DISPLAY fold: it changes the plot and the numbers "
        "beside it, not the run.  To remove the train at the source, "
        "enable Periodic phase in the Beam tab — then this control is "
        "disabled because there is nothing left to fold.")

    def __init__(self, parent, state: AppState):
        super().__init__(parent, "Phase Space  —  x-x' · y-y' · x-y · φ-W", size=(1000, 760))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(10, 10, 10, 10); v.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(QLabel("basis:"))
        from PyQt6.QtWidgets import QComboBox
        self._basis = QComboBox()
        self._basis.addItem("(Δφ, ΔW) — our code",  userData="ours")
        self._basis.addItem("(z, δ) — TraceWin",    userData="tracewin")
        self._basis.currentIndexChanged.connect(self._redraw)
        top.addWidget(self._basis)
        top.addSpacing(12)
        top.addWidget(QLabel("colour by:"))
        self._colour = QComboBox()
        self._colour.addItem("density (heatmap)",  userData="density")
        self._colour.addItem("ΔW (energy)",        userData="dw")
        self._colour.addItem("particle index",     userData="idx")
        self._colour.addItem("|r| (transverse)",   userData="r")
        self._colour.setToolTip(
            "density: 2-D log-intensity histogram (default).\n"
            "ΔW: per-particle scatter coloured by relative energy.\n"
            "particle index: scatter coloured by index — useful for "
            "tracking individual particles through a halo formation.\n"
            "|r|: transverse radius √(x² + y²); highlights tail "
            "particles."
        )
        self._colour.currentIndexChanged.connect(self._redraw)
        top.addWidget(self._colour)
        top.addSpacing(12)
        top.addWidget(QLabel("location:"))
        # Which recorded distribution to plot — the exit beam (default) or
        # any snapshot captured along the lattice (via "snapshot every N" /
        # "snapshot at elements" / Marker(snapshot=True)).
        self._location = QComboBox()
        self._location.addItem("exit (final)", userData=None)
        self._location.setToolTip(
            "Phase space at the exit (default) or at any location "
            "snapshotted during the run.  Enable snapshots in the Numerics "
            "tab ('Snapshot every N' or 'Snapshot at').")
        self._location.currentIndexChanged.connect(self._redraw)
        top.addWidget(self._location)
        top.addSpacing(12)
        # Toggle: swap the four density panels for a full beam-parameter
        # table of the SELECTED distribution (Twiss, RMS, centroid,
        # emittances, halo, extents…).  Ctrl+S export picks the table up
        # automatically (QTableWidget discovery in _collect_panels).
        self._params_btn = QPushButton("Beam parameters")
        self._params_btn.setCheckable(True)
        self._params_btn.setToolTip(
            "Show a table of every beam parameter of the selected "
            "distribution (Twiss all planes, RMS sizes, centroid, "
            "energies, emittances, halo, max extents) instead of the "
            "density plots.  Follows the location selector; Ctrl+S "
            "exports the table.")
        self._params_btn.toggled.connect(self._toggle_view)
        top.addWidget(self._params_btn)
        # Bunch-train fold.  An RFQ turns a DC beam into a train of
        # bunches one RF period apart; particles that slipped into a
        # neighbouring bucket are drawn 360° away, so the φ–ΔW panel
        # shows a row of stripes instead of one bunch.  Folding restores
        # the single-bucket view (TraceWin/Toutatis convention) and
        # matches the σ_φ / ε_z the parameters table reports.
        from PyQt6.QtWidgets import QCheckBox
        self._wrap_phi = QCheckBox("fold φ")
        self._wrap_phi.setChecked(True)
        self._wrap_phi.setToolTip(self._FOLD_TIP)
        self._wrap_phi.toggled.connect(self._redraw)
        top.addWidget(self._wrap_phi)
        top.addStretch(1)
        v.addLayout(top)
        # The 2x2 density grid lives in a container widget so the
        # parameters-table toggle can swap the whole block at once.
        self._plots_box = QWidget()
        grid = QGridLayout(self._plots_box)
        grid.setSpacing(8); grid.setContentsMargins(0, 0, 0, 0)
        # Four 2-D density panels (log-intensity, magma colormap with
        # transparent ramp, vertical colorbar on the right).
        self._panel_xx  = DensityPanel(title="x – x'", xlabel="x",   xunits="mm",
                                       ylabel="x'",  yunits="mrad")
        self._panel_yy  = DensityPanel(title="y – y'", xlabel="y",   xunits="mm",
                                       ylabel="y'",  yunits="mrad")
        self._panel_xy  = DensityPanel(title="x – y",  xlabel="x",   xunits="mm",
                                       ylabel="y",   yunits="mm")
        self._panel_phi = DensityPanel(title="φ – W",  xlabel="φ",   xunits="deg",
                                       ylabel="W",   yunits="MeV")
        grid.addWidget(self._panel_xx,  0, 0); grid.addWidget(self._panel_yy,  0, 1)
        grid.addWidget(self._panel_xy,  1, 0); grid.addWidget(self._panel_phi, 1, 1)
        v.addWidget(self._plots_box, stretch=1)
        # Beam-parameters table (hidden until toggled) — style mirrors
        # _BpmsPopup.
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setVisible(False)
        v.addWidget(self._table, stretch=1)

    def refresh(self, results):
        # Populate the location dropdown from any snapshots captured this
        # run, then redraw the currently-selected location.
        self._populate_locations(results)
        self._sync_fold_to_run(results)
        self._redraw()

    def _sync_fold_to_run(self, results) -> None:
        """Disable the display fold for a run that folded while tracking.

        ``wrap_phase_column`` folds about the MEDIAN with a hard 360°
        period.  A run made with ``periodic_phase`` has already folded
        about the synchronous particle using the true bunch spacing —
        720° downstream of a 162.5 → 325 MHz jump — so applying the
        display fold on top would slice a legitimately ±360°-wide bunch
        in half.  Nothing to fold, so take the control away rather than
        leave a checkbox that silently corrupts the picture.
        """
        folded = bool(getattr(results, "periodic_phase", False))
        self._wrap_phi.blockSignals(True)
        if folded:
            self._wrap_phi.setChecked(False)
            self._wrap_phi.setEnabled(False)
            self._wrap_phi.setToolTip(
                "Disabled: this run tracked with periodic phase "
                "coordinates (Beam tab → Periodic phase), so Δφ was "
                "already folded into one bunch spacing during tracking "
                "and the plot below is the single-bunch view.\n\n"
                "The display fold assumes a 360° period, which would be "
                "wrong here downstream of a frequency jump.")
        else:
            self._wrap_phi.setEnabled(True)
            self._wrap_phi.setToolTip(self._FOLD_TIP)
        self._wrap_phi.blockSignals(False)

    def _populate_locations(self, results):
        """Rebuild the location combo: 'exit (final)' + one entry per
        snapshot captured along the lattice (labelled by element name)."""
        combo = self._location
        prev = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("exit (final)", userData=None)
        snaps = getattr(results, "_snapshots", None) if results is not None else None
        if snaps:
            try:
                name_by_s = dict(zip(results.s, results.element_names))
            except Exception:
                name_by_s = {}
            for s in sorted(snaps):
                nm = name_by_s.get(s, "?")
                combo.addItem(f"{nm}  ·  s={s / 1000:.3f} m", userData=s)
        if prev is not None:                        # keep prior selection
            idx = combo.findData(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _current_particles(self):
        """The (N,6) particle array for the selected location — a captured
        snapshot (alive-filtered) or the exit beam, falling back to a
        freshly-generated beam from the BeamConfig."""
        return self._current_particles_with_ref()[0]

    def _current_particles_with_ref(self):
        """(particles, ref_or_None, source_label, n_total_or_None) for the
        selected location.  ``ref`` is the per-snapshot ReferenceParticle
        (captured with the snapshot), the exit beam's ref, or None for
        the regenerated-input fallback; ``n_total`` is the launched
        macroparticle count (for the transmission row)."""
        results = getattr(self._state, "results", None)
        loc = self._location.currentData() if hasattr(self, "_location") else None
        if loc is not None and results is not None:
            try:
                alive = results.alive_at(loc)            # snapshot at s
                ref = None
                n_total = None
                try:
                    all_p, ref = results.beam_at(loc)
                    n_total = len(all_p)
                except (KeyError, AttributeError):
                    pass
                return alive, ref, "snapshot", n_total
            except (KeyError, AttributeError):
                pass                                     # fall through to exit
        beam = getattr(results, "beam", None) if results is not None else None
        if beam is not None and hasattr(beam, "alive_particles"):
            n_total = getattr(getattr(beam, "particles", None),
                              "shape", (None,))[0]
            return (beam.alive_particles, getattr(beam, "ref", None),
                    "exit beam", n_total)
        if self._state.beam_config is not None:
            try:
                from linac_gen.distributions.factory import create_beam
                gen = create_beam(self._state.beam_config, seed=42)
                return (gen.alive_particles, getattr(gen, "ref", None),
                        "input beam (regenerated — no particle data "
                        "in results)", gen.particles.shape[0])
            except Exception:
                return None, None, "", None
        return None, None, "", None

    def _toggle_view(self, checked: bool):
        """Swap the density-panel grid for the beam-parameters table."""
        self._plots_box.setVisible(not checked)
        self._table.setVisible(checked)
        if checked:
            self._fill_table()
        else:
            # Clear on hide: _collect_panels discovers QTableWidgets
            # regardless of visibility, so a populated hidden table
            # would attach STALE (wrong-location) data to a plots-view
            # Ctrl+S export; an empty table is skipped (nrows==0).
            self._table.setRowCount(0)

    def _fill_table(self):
        from linac_gen.diagnostics.beam_summary import summarize_particles
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont

        q, ref, src, n_total = self._current_particles_with_ref()
        loc_label = self._location.currentText()
        if src and not src.startswith("snapshot"):
            loc_label = f"{loc_label}  [{src}]"
        # Follow the "fold φ" checkbox so the table can never disagree
        # with the plot beside it (a train would otherwise show
        # σ_φ = 183° next to a picture of a 4° bunch).  Folded rows are
        # labelled so the number is never mistaken for the raw one.
        if q is not None and len(q) and getattr(self, "_wrap_phi", None) \
                is not None and self._wrap_phi.isChecked():
            from linac_gen.diagnostics.moments import wrap_phase_column
            q, n_folded = wrap_phase_column(q)
            if n_folded:
                loc_label = (f"{loc_label}  [φ folded: {n_folded} "
                             "particles from adjacent RF buckets]")
        cfg = self._state.beam_config
        rows = summarize_particles(
            q, ref,
            species_name=getattr(cfg, "species", "") if cfg else "",
            current_ma=getattr(cfg, "current", None) if cfg else None,
            n_total=n_total, location=loc_label)

        tbl = self._table
        tbl.setRowCount(0)
        tbl.setRowCount(len(rows))
        bold = QFont(); bold.setBold(True)
        for r, (group, name, value, unit) in enumerate(rows):
            if name == "":                       # group header row
                item = QTableWidgetItem(group)
                item.setFont(bold)
                tbl.setItem(r, 0, item)
                tbl.setSpan(r, 0, 1, 3)
                continue
            tbl.setItem(r, 0, QTableWidgetItem(f"  {name}"))
            vitem = QTableWidgetItem(value)
            vitem.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                   | Qt.AlignmentFlag.AlignVCenter)
            tbl.setItem(r, 1, vitem)
            tbl.setItem(r, 2, QTableWidgetItem(unit))

    def _redraw(self):
        # Keep the parameters table in sync with the location selector
        # while it is the selected view (basis/colour don't affect it).
        # isHidden(), not isVisible(): the latter is False for a popup
        # that has not been show()n yet (offscreen tests, pre-show
        # refresh) even when the table IS the selected view.
        if getattr(self, "_table", None) is not None \
                and not self._table.isHidden():
            self._fill_table()
        particles = self._current_particles()
        if particles is None or len(particles) == 0:
            for panel in (self._panel_xx, self._panel_yy,
                          self._panel_xy, self._panel_phi):
                panel.set_data([], [])
            return
        q = particles
        # Fold BEFORE the (z, δ) basis conversion below, otherwise z
        # would be folded instead of φ.  Same helper as the physics
        # moments, so plot and parameters table always agree.
        n_folded = 0
        if getattr(self, "_wrap_phi", None) is not None \
                and self._wrap_phi.isChecked():
            from linac_gen.diagnostics.moments import wrap_phase_column
            q, n_folded = wrap_phase_column(q)
        self._wrap_phi.setToolTip(
            self._wrap_phi.toolTip().split("\n\nCurrently")[0]
            + (f"\n\nCurrently folding {n_folded} particles from "
               "adjacent RF buckets." if n_folded else
               "\n\nCurrently nothing to fold (single bucket)."))
        phi, dw = q[:, 4], q[:, 5]
        basis = self._basis.currentData()
        cfg = self._state.beam_config
        # Color-by mode dispatch — density (default) keeps the existing
        # histogram render path so older snapshots still look identical.
        mode = self._colour.currentData()
        if mode != "density":
            n = q.shape[0]
            if mode == "dw":
                cvec = dw; clabel = "ΔW (MeV)"
            elif mode == "idx":
                cvec = np.arange(n, dtype=float); clabel = "particle #"
            elif mode == "r":
                cvec = np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2)
                clabel = "|r| (mm)"
            else:
                cvec = None; clabel = ""
            self._panel_xx.set_scatter(q[:, 0], q[:, 1], cvec, label=clabel)
            self._panel_yy.set_scatter(q[:, 2], q[:, 3], cvec, label=clabel)
            self._panel_xy.set_scatter(q[:, 0], q[:, 2], cvec, label=clabel)
        else:
            self._panel_xx.set_data(q[:, 0], q[:, 1])
            self._panel_yy.set_data(q[:, 2], q[:, 3])
            self._panel_xy.set_data(q[:, 0], q[:, 2])
        if basis == "tracewin" and cfg is not None:
            import math
            from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
            from linac_gen.core.constants import C_LIGHT
            sp = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}.get(cfg.species, PROTON)
            gamma = 1 + cfg.energy / sp.mass
            beta = math.sqrt(max(1 - 1 / (gamma * gamma), 0.0))
            wl_m = C_LIGHT / (cfg.frequency * 1e6)
            z = -phi * beta * wl_m / 360.0
            delta = dw / (beta * beta * gamma * sp.mass)
            if mode != "density":
                self._panel_phi.set_scatter(z, delta, cvec, label=clabel)
            else:
                self._panel_phi.set_data(z, delta)
            self._panel_phi.plot.setLabel("bottom", "z", units="m")
            self._panel_phi.plot.setLabel("left", "δ = Δp/p", units="")
        else:
            if mode != "density":
                self._panel_phi.set_scatter(phi, dw, cvec, label=clabel)
            else:
                self._panel_phi.set_data(phi, dw)
            self._panel_phi.plot.setLabel("bottom", "φ", units="deg")
            self._panel_phi.plot.setLabel("left", "W", units="MeV")

    def _extra_panels(self):
        """Expose raw per-particle coordinates — DensityPanels histogram
        their inputs and don't retain (x, y) themselves.  Uses the SAME
        selected location as the plot so Ctrl+S export matches the view."""
        particles = self._current_particles()
        if particles is None or len(particles) == 0:
            return []
        q = np.asarray(particles)
        # Same fold as the view, so a Ctrl+S CSV never mixes a folded
        # picture with a raw φ column.
        if getattr(self, "_wrap_phi", None) is not None \
                and self._wrap_phi.isChecked():
            from linac_gen.diagnostics.moments import wrap_phase_column
            q, _ = wrap_phase_column(q)
        n = q.shape[0]
        idx = np.arange(n, dtype=float)
        names = ("x_mm", "xprime_mrad", "y_mm", "yprime_mrad",
                 "phi_deg", "dW_MeV")
        curves = [_Curve(name=names[i], x=idx, y=q[:, i].astype(float))
                  for i in range(min(6, q.shape[1]))]
        return [_Panel(label="phase_space_particles",
                       xlabel="particle index", ylabel="",
                       curves=curves)]


_CHANNEL_LABEL = {
    "STAT_E": "Static E  (DC electric)",
    "STAT_B": "Static B  (solenoid / DC magnetic)",
    "RF_E":   "RF E      (cavity electric)",
    "RF_B":   "RF B      (cavity magnetic)",
}


def _expand_channel_to_grid(ch, *, oversample_xy: int = 25,
                             r_max: float | None = None,
                             ) -> "tuple[dict, dict, list[str]]":
    """Convert any FieldChannel into a uniform ``(axes, comps, ax_order)``.

    Returns:
        axes : ``{'x': ndarray, 'y': ndarray, 'z': ndarray}`` (mm).
        comps: ``{'Fx': arr, 'Fy': arr, 'Fz': arr, 'Fr': arr | None}``.
               Each ``arr`` has shape ``(nx, ny, nz)`` for Cartesian, or
               ``(nz, nr)`` for the 2-D-cyl Fr/Fz cross-views.
        ax_order: 3-element list naming the array's axes in order.

    Geometries:
        1  (1-D on-axis)        → expand to 3-D Cart via paraxial.
        4/5 (2-D cyl)           → expand to 3-D Cart via Fr/Fz on r.
        6  (2-D Cart x-y)       → return as-is, broadcast over a single z.
        7  (3-D Cart)           → return as-is.
        9  (1-D quad gradient)  → expand to 3-D Cart with Bx=G·y, By=G·x.
    """
    g = ch.geometry
    z = np.asarray(ch.z, dtype=float) if ch.z is not None else None

    # -------- helper: build an x-y grid of half-radius r_max --------
    def _xy_grid(r_max_mm: float, n: int):
        if r_max_mm <= 0:
            r_max_mm = 1.0
        xs = np.linspace(-r_max_mm, r_max_mm, n)
        ys = np.linspace(-r_max_mm, r_max_mm, n)
        return xs, ys

    # ------------------- 1-D on-axis (Fz only) ----------------------
    if g == 1 and z is not None and ch.Fz is not None:
        Fz_axis = np.asarray(ch.Fz, dtype=float)
        # Paraxial off-axis: Fr = -(r/2)·dFz/dz.  Expand on a synthetic
        # x-y grid; r_max defaults to a small fraction of the longitudinal
        # span so the off-axis derivative stays valid.
        if r_max is None:
            r_max = max(5.0, 0.05 * (z[-1] - z[0]))
        xs, ys = _xy_grid(r_max, oversample_xy)
        dFz_dz = np.gradient(Fz_axis, z)
        # Build (nx, ny, nz) arrays.
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        R = np.hypot(X, Y)
        # Fz uniform across xy on-axis paraxial.
        Fz = np.broadcast_to(Fz_axis[None, None, :], (xs.size, ys.size, z.size)).copy()
        # Fr(x, y, z) = -(R / 2) · dFz/dz.
        Fr = -0.5 * R[:, :, None] * dFz_dz[None, None, :]
        # Decompose to Fx, Fy with safe-r guard.
        safe_R = np.where(R > 1e-12, R, 1.0)
        Fx = Fr * (X / safe_R)[:, :, None]
        Fy = Fr * (Y / safe_R)[:, :, None]
        zero = R < 1e-12
        Fx[zero, :] = 0.0
        Fy[zero, :] = 0.0
        return ({"x": xs, "y": ys, "z": z},
                {"Fx": Fx, "Fy": Fy, "Fz": Fz, "Fr": Fr},
                ["x", "y", "z"])

    # ------------------- 1-D quad gradient G(z) ---------------------
    if g == 9 and z is not None and ch.Fz is not None:
        Gz = np.asarray(ch.Fz, dtype=float)   # T/m
        if r_max is None:
            r_max = 5.0
        xs, ys = _xy_grid(r_max, oversample_xy)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        # Bx = G·y, By = G·x.  G in T/m, x/y in mm → factor 1e-3 → T.
        Bx = (Y[:, :, None] * 1e-3) * Gz[None, None, :]
        By = (X[:, :, None] * 1e-3) * Gz[None, None, :]
        Bz = np.zeros((xs.size, ys.size, z.size))
        return ({"x": xs, "y": ys, "z": z},
                {"Fx": Bx, "Fy": By, "Fz": Bz, "Fr": None},
                ["x", "y", "z"])

    # ------------------- 2-D cyl (z-r) ------------------------------
    if g in (4, 5) and z is not None and ch.r is not None:
        r = np.asarray(ch.r, dtype=float)
        Fz_zr = np.asarray(ch.Fz, dtype=float) if ch.Fz is not None else None
        Fr_zr = np.asarray(ch.Fr, dtype=float) if ch.Fr is not None else None
        # Loader convention: (n_z, n_r) — match _sample_cart_2d_cyl.
        if Fz_zr is not None and Fz_zr.shape != (z.size, r.size):
            # Some loaders may emit (n_r, n_z); transpose to (n_z, n_r).
            if Fz_zr.shape == (r.size, z.size):
                Fz_zr = Fz_zr.T
        if Fr_zr is not None and Fr_zr.shape != (z.size, r.size):
            if Fr_zr.shape == (r.size, z.size):
                Fr_zr = Fr_zr.T
        # Build a synthetic Cart grid covering ±r[-1].
        rmax = float(r[-1]) if r_max is None else r_max
        xs, ys = _xy_grid(rmax, oversample_xy)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        R = np.hypot(X, Y)
        # Bilinear interpolate Fz(z, r) and Fr(z, r) for every (x, y, z).
        # Vectorise: for each z slice, interp over R.
        nz = z.size
        Fz_full = np.zeros((xs.size, ys.size, nz))
        Fr_full = np.zeros((xs.size, ys.size, nz)) if Fr_zr is not None else None
        for k in range(nz):
            if Fz_zr is not None:
                Fz_full[..., k] = np.interp(R, r, Fz_zr[k, :], left=Fz_zr[k, 0],
                                              right=Fz_zr[k, -1])
            if Fr_full is not None:
                Fr_full[..., k] = np.interp(R, r, Fr_zr[k, :], left=Fr_zr[k, 0],
                                              right=Fr_zr[k, -1])
        # Decompose Fr → Fx, Fy.
        if Fr_full is not None:
            safe_R = np.where(R > 1e-12, R, 1.0)
            Fx = Fr_full * (X / safe_R)[:, :, None]
            Fy = Fr_full * (Y / safe_R)[:, :, None]
            zero = R < 1e-12
            Fx[zero, :] = 0.0
            Fy[zero, :] = 0.0
        else:
            Fx = np.zeros_like(Fz_full)
            Fy = np.zeros_like(Fz_full)
        return ({"x": xs, "y": ys, "z": z},
                {"Fx": Fx, "Fy": Fy, "Fz": Fz_full, "Fr": Fr_full},
                ["x", "y", "z"])

    # ------------------- 2-D Cart (x-y, no z) -----------------------
    if g == 6 and ch.x is not None and ch.y is not None:
        xs = np.asarray(ch.x, dtype=float)
        ys = np.asarray(ch.y, dtype=float)
        # Single z-slice; broadcast to (nx, ny, 1).
        z_arr = z if z is not None and z.size > 0 else np.array([0.0])
        nz = z_arr.size
        comps = {}
        for k in ("Fx", "Fy", "Fz"):
            arr = getattr(ch, k, None)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.shape == (xs.size, ys.size):
                comps[k] = np.broadcast_to(arr[:, :, None], (xs.size, ys.size, nz)).copy()
            else:
                comps[k] = None
        return ({"x": xs, "y": ys, "z": z_arr}, comps, ["x", "y", "z"])

    # ------------------- 3-D Cart (native) --------------------------
    if g == 7 and ch.x is not None and ch.y is not None and z is not None:
        xs = np.asarray(ch.x, dtype=float)
        ys = np.asarray(ch.y, dtype=float)
        comps = {}
        for k in ("Fx", "Fy", "Fz"):
            arr = getattr(ch, k, None)
            if arr is None:
                comps[k] = None; continue
            comps[k] = np.asarray(arr, dtype=float)
        comps["Fr"] = None    # not native
        return ({"x": xs, "y": ys, "z": z}, comps, ["x", "y", "z"])

    # Fallback: nothing we recognise.
    return ({}, {}, [])


class _FieldMapPopup(_PopupPlot):
    """Field-map visualiser: 2D contour + 1D line cuts.

    Lets the user pick (a) a FieldMap / FieldMap3D element from the
    lattice, (b) a Channel (STAT_E / STAT_B / RF_E / RF_B), and (c) a
    component (Fx, Fy, Fz, Fr, Fq).  Renders:

    * a 2D pseudocolour heatmap of the chosen component on the two
      "in-plane" axes, sliced at the user-picked third-axis coordinate
      (slider).  For native 2-D maps the heatmap is the whole map; for
      1-D maps the heatmap stays empty.
    * three 1-D line plots showing F along each axis at the chosen
      coordinates on the other two.

    Sliders pick the slice indices in each available axis; the line
    plots and the 2-D slice update live.
    """

    def __init__(self, parent, state):
        super().__init__(parent, "Field Map · 2D contour + 1D cuts",
                         size=(1500, 920))
        from PyQt6.QtWidgets import (
            QComboBox, QSlider, QPushButton, QSplitter, QGroupBox,
        )
        from linac_gen_gui.interphase.plots.plot_style import (
            configure_pyqtgraph_defaults,
        )
        configure_pyqtgraph_defaults()

        self._state = state
        self._channel_obj_map: dict = {}     # combo idx → FieldChannel
        self._component_keys: list = []      # combo idx → "Fx"/"Fy"/...
        self._suspend_signals = False

        v = QVBoxLayout(self); v.setContentsMargins(10, 10, 10, 10); v.setSpacing(6)

        # ── Top toolbar: element / channel / component selectors ──
        tb = QHBoxLayout(); tb.setSpacing(8)
        tb.addWidget(QLabel("Element:"))
        self._el_combo = QComboBox(); self._el_combo.setMinimumWidth(220)
        self._el_combo.currentIndexChanged.connect(self._on_element_changed)
        tb.addWidget(self._el_combo)
        tb.addWidget(QLabel("Channel:"))
        self._ch_combo = QComboBox(); self._ch_combo.setMinimumWidth(120)
        self._ch_combo.currentIndexChanged.connect(self._on_channel_changed)
        tb.addWidget(self._ch_combo)
        tb.addWidget(QLabel("Component:"))
        self._cp_combo = QComboBox(); self._cp_combo.setMinimumWidth(80)
        self._cp_combo.currentIndexChanged.connect(self._redraw)
        tb.addWidget(self._cp_combo)
        tb.addWidget(QLabel("2D plane:"))
        self._plane_combo = QComboBox(); self._plane_combo.setMinimumWidth(80)
        self._plane_combo.currentIndexChanged.connect(self._redraw)
        tb.addWidget(self._plane_combo)
        recompute = QPushButton("Reload")
        recompute.clicked.connect(self._populate_elements)
        tb.addWidget(recompute)
        tb.addStretch(1)
        self._info = QLabel("")
        self._info.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;"
        )
        tb.addWidget(self._info, stretch=2)
        v.addLayout(tb)

        # ── Slice-coordinate sliders (one per axis) ──
        sb = QGridLayout(); sb.setSpacing(4)
        self._sliders: dict[str, QSlider] = {}
        self._slider_labels: dict[str, QLabel] = {}
        for r, ax in enumerate(("x", "y", "z")):
            lab = QLabel(f"{ax}:"); lab.setMinimumWidth(20)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setMinimum(0); sl.setMaximum(0)
            sl.setSingleStep(1); sl.setPageStep(5)
            sl.setEnabled(False)
            val = QLabel("—"); val.setMinimumWidth(120)
            val.setStyleSheet(f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO};")
            sl.valueChanged.connect(self._on_slider_changed)
            sb.addWidget(lab, r, 0)
            sb.addWidget(sl,  r, 1)
            sb.addWidget(val, r, 2)
            self._sliders[ax] = sl
            self._slider_labels[ax] = val
        sb.setColumnStretch(1, 1)
        sb_group = QGroupBox("Slice coordinates")
        sb_group.setStyleSheet(
            f"QGroupBox {{ color:{theme.TEXT_2}; border:1px solid {theme.BORDER_2};"
            f" border-radius:4px; margin-top:6px; padding:4px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left:8px;"
            f" padding:0 4px; }}"
        )
        sb_group.setLayout(sb)
        v.addWidget(sb_group)

        # ── Main split: heatmap (left) + 3 line plots (right) ──
        split = QSplitter(Qt.Orientation.Horizontal)
        # 2D heatmap shell — pyqtgraph GraphicsLayoutWidget gives us
        # a PlotItem + ColorBar pair just like DensityPanel.
        from pyqtgraph import GraphicsLayoutWidget, ImageItem, ColorBarItem
        self._gl = GraphicsLayoutWidget()
        self._gl.setBackground(theme.BG_INSET)
        self._heat_plot = self._gl.addPlot(row=0, col=0)
        self._heat_plot.setMenuEnabled(False)
        self._heat_plot.showGrid(x=True, y=True, alpha=0.25)
        self._heat_plot.setAspectLocked(False)
        self._image = ImageItem(axisOrder="row-major")
        self._heat_plot.addItem(self._image)
        # Diverging colormap (blue → white → red) suits ± field components.
        from pyqtgraph import colormap as pgcmap
        try:
            self._cmap = pgcmap.get("CET-D1A")
        except Exception:                                   # noqa: BLE001
            self._cmap = pgcmap.get("magma")
        self._image.setLookupTable(self._cmap.getLookupTable(0.0, 1.0, 256))
        self._cbar = ColorBarItem(
            colorMap=self._cmap, values=(-1.0, 1.0),
            label="F", interactive=False,
            orientation="v", width=14,
        )
        self._cbar.setFixedWidth(56)
        self._cbar.setImageItem(self._image, insert_in=self._heat_plot)
        # Cross-hair lines on the heatmap to show the 1D cut locations.
        self._vline = pg.InfiniteLine(0.0, angle=90,
                                       pen=pg.mkPen("#facc15", width=1))
        self._hline = pg.InfiniteLine(0.0, angle=0,
                                       pen=pg.mkPen("#facc15", width=1))
        self._heat_plot.addItem(self._vline); self._heat_plot.addItem(self._hline)
        split.addWidget(self._gl)

        # Right column: three line plots (F vs x, F vs y, F vs z).
        right = QWidget(); rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
        self._line_plots: dict[str, pg.PlotWidget] = {}
        self._line_curves: dict[str, pg.PlotDataItem] = {}
        self._line_markers: dict[str, pg.InfiniteLine] = {}
        for ax in ("x", "y", "z"):
            p = _mk_plot(f"F vs {ax}", "")
            p.setLabel("bottom", ax, units="mm")
            curve = p.plot(pen=curve_pen(theme.ACCENT, width=2.0))
            mark = pg.InfiniteLine(0.0, angle=90,
                                    pen=pg.mkPen("#facc15", width=1,
                                                  style=Qt.PenStyle.DashLine))
            p.addItem(mark)
            rl.addWidget(p, stretch=1)
            self._line_plots[ax] = p
            self._line_curves[ax] = curve
            self._line_markers[ax] = mark
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        v.addWidget(split, stretch=1)

        # Subscribe to lattice changes so the element combo stays fresh.
        if state is not None:
            try:
                state.lattice_changed.connect(self._populate_elements)
            except Exception:                              # noqa: BLE001
                pass

        self._populate_elements()

    # ------------------------------------------------------------------
    def _populate_elements(self, *_args) -> None:
        """Rebuild the element dropdown from the current lattice."""
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        self._suspend_signals = True
        try:
            self._el_combo.clear()
            lat = self._state.lattice if self._state else None
            self._fm_elements = []
            if lat is not None:
                for i, el in enumerate(lat.elements):
                    if isinstance(el, (FieldMap, FieldMap3D)):
                        self._fm_elements.append((i, el))
                        cls = type(el).__name__
                        # Quick "what is it" label from the channel set:
                        #   STAT_B only      → solenoid
                        #   RF_E (+RF_B)     → cavity
                        #   STAT_E only      → DC E
                        kind = ""
                        fd = getattr(el, "field_data", None)
                        if fd is not None and fd.channels:
                            chs = {c.name for c in fd.channels.keys()}
                            if "STAT_B" in chs and not (chs & {"RF_E", "RF_B"}):
                                kind = "solenoid"
                            elif chs & {"RF_E", "RF_B"}:
                                kind = "cavity"
                            elif "STAT_E" in chs:
                                kind = "DC electric"
                        suffix = f" — {kind}" if kind else ""
                        self._el_combo.addItem(
                            f"[{i}] {el.name}  ({cls}){suffix}"
                        )
            if not self._fm_elements:
                self._el_combo.addItem("(no field-map elements)")
                self._info.setText("Load a lattice with FIELD_MAP elements.")
                self._ch_combo.clear(); self._cp_combo.clear()
                self._reset_plots()
                return
            self._el_combo.setCurrentIndex(0)
        finally:
            self._suspend_signals = False
        self._on_element_changed()

    def _on_element_changed(self, *_args) -> None:
        if self._suspend_signals or not getattr(self, "_fm_elements", None):
            return
        idx = max(0, self._el_combo.currentIndex())
        if idx >= len(self._fm_elements):
            return
        _, el = self._fm_elements[idx]
        fd = getattr(el, "field_data", None)
        self._suspend_signals = True
        try:
            self._ch_combo.clear()
            self._channel_obj_map.clear()
            if fd is None or not getattr(fd, "channels", None):
                self._ch_combo.addItem("(no channels)")
                self._cp_combo.clear()
                self._reset_plots()
                self._info.setText("Element has no channel data.")
                return
            for ch_enum, ch in fd.channels.items():
                name = getattr(ch_enum, "name", str(ch_enum))
                pretty = _CHANNEL_LABEL.get(name, name)
                geom_lbl = {
                    1: "1D on-axis",
                    4: "2D cyl  (TM)", 5: "2D cyl  (TE)",
                    6: "2D Cart", 7: "3D Cart",
                    9: "1D quad gradient",
                }.get(ch.geometry, f"geom={ch.geometry}")
                self._ch_combo.addItem(f"{pretty}  ·  {geom_lbl}")
                self._channel_obj_map[self._ch_combo.count() - 1] = ch
            self._ch_combo.setCurrentIndex(0)
        finally:
            self._suspend_signals = False
        self._on_channel_changed()

    def _on_channel_changed(self, *_args) -> None:
        if self._suspend_signals:
            return
        ch_idx = max(0, self._ch_combo.currentIndex())
        ch = self._channel_obj_map.get(ch_idx)
        if ch is None:
            self._cp_combo.clear()
            self._reset_plots()
            return
        # Build the synthetic uniform-grid representation once; all
        # downstream plotting operates on this.
        axes, comps, ax_order = _expand_channel_to_grid(ch)
        self._axes = axes
        self._comps = comps
        self._ax_order = ax_order
        self._suspend_signals = True
        try:
            self._cp_combo.clear()
            self._component_keys = []
            # Always offer the standard set; only those with data in
            # ``comps`` are kept.  ``Fr`` is offered only when natively
            # available or computed from a 1-D / 2-D-cyl expansion.
            for key in ("Fx", "Fy", "Fz", "Fr"):
                arr = comps.get(key)
                if arr is not None and arr.size > 0 and np.any(np.isfinite(arr)):
                    self._cp_combo.addItem(key)
                    self._component_keys.append(key)
            # Available 2-D plane choices: any pair from the axes dict.
            self._plane_combo.clear()
            self._plane_combo.addItem("(none)")
            for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
                if a in axes and b in axes:
                    self._plane_combo.addItem(f"{a}-{b}")
            # Default plane: x-z is usually the most physically meaningful
            # for a solenoid / RF cavity (longitudinal slice through axis).
            for cand in ("x-z", "x-y", "y-z"):
                idx = self._plane_combo.findText(cand)
                if idx > 0:
                    self._plane_combo.setCurrentIndex(idx); break
            # Configure sliders for each axis.
            for ax in ("x", "y", "z"):
                arr = axes.get(ax)
                sl = self._sliders[ax]
                if arr is None or len(arr) == 0:
                    sl.setEnabled(False); sl.setMaximum(0); sl.setValue(0)
                    self._slider_labels[ax].setText("—")
                else:
                    sl.setEnabled(True)
                    sl.setMaximum(len(arr) - 1)
                    # Default cuts: x = y = 0 (on-axis), z = midpoint.
                    if ax == "z":
                        i0 = len(arr) // 2
                    else:
                        i0 = int(np.argmin(np.abs(arr)))
                    sl.setValue(i0)
                    self._slider_labels[ax].setText(f"{arr[i0]:.3f} mm")
        finally:
            self._suspend_signals = False
        self._redraw()

    def _channel_axes(self, ch) -> dict[str, np.ndarray]:
        out = {}
        for ax in ("x", "y", "z", "r"):
            arr = getattr(ch, ax, None)
            if arr is not None and np.asarray(arr).size > 0:
                out[ax] = np.asarray(arr, dtype=float)
        return out

    def _on_slider_changed(self, *_args) -> None:
        if self._suspend_signals:
            return
        # Refresh label values then redraw (cheap).
        ch = self._channel_obj_map.get(self._ch_combo.currentIndex())
        if ch is None:
            return
        axes = self._channel_axes(ch)
        for ax in ("x", "y", "z"):
            arr = axes.get(ax)
            if arr is None or len(arr) == 0:
                continue
            i = min(self._sliders[ax].value(), len(arr) - 1)
            self._slider_labels[ax].setText(f"{arr[i]:.3f} mm")
        self._redraw()

    def _reset_plots(self) -> None:
        self._image.setImage(np.zeros((2, 2)))
        for ax in ("x", "y", "z"):
            self._line_curves[ax].setData([], [])

    def _redraw(self, *_args) -> None:
        if self._suspend_signals:
            return
        cp_idx = self._cp_combo.currentIndex()
        if not getattr(self, "_comps", None) or cp_idx < 0:
            self._reset_plots(); return
        if cp_idx >= len(self._component_keys):
            self._reset_plots(); return
        key = self._component_keys[cp_idx]
        F = self._comps.get(key)
        axes = self._axes
        if F is None or F.size == 0:
            self._reset_plots(); return
        ax_order = self._ax_order   # always ['x', 'y', 'z']
        # Indices on each axis (clamped to slider).
        idx = {}
        for ax in ("x", "y", "z"):
            if ax in axes and len(axes[ax]) > 0:
                idx[ax] = min(self._sliders[ax].value(), len(axes[ax]) - 1)

        # ----- 2-D heatmap on the chosen plane ------------------------
        plane = self._plane_combo.currentText()
        if plane and plane != "(none)" and "-" in plane:
            a, b = plane.split("-")
            self._draw_heatmap(F, ax_order, axes, idx, a, b)
            self._heat_plot.setLabel("bottom", a, units="mm")
            self._heat_plot.setLabel("left",   b, units="mm")
        else:
            self._image.setImage(np.zeros((2, 2)))

        # ----- 1-D line plots -----------------------------------------
        for ax in ("x", "y", "z"):
            self._draw_line_along(ax, F, ax_order, axes, idx)
        # Sync cross-hairs.
        if plane and plane != "(none)" and "-" in plane:
            a, b = plane.split("-")
            if a in axes:
                self._vline.setValue(float(axes[a][idx.get(a, 0)]))
            if b in axes:
                self._hline.setValue(float(axes[b][idx.get(b, 0)]))
        # Info ribbon.
        ch = self._channel_obj_map.get(self._ch_combo.currentIndex())
        geom = ch.geometry if ch is not None else "?"
        derived = ""
        if geom in (1, 9, 4, 5) and key in ("Fx", "Fy", "Fr"):
            derived = " (derived from on-axis Fz)"
        finite = np.isfinite(F)
        if finite.any():
            stats = (f"|F|max={float(np.nanmax(np.abs(F))):.3g}  "
                     f"min={float(np.nanmin(F)):.3g}  "
                     f"max={float(np.nanmax(F)):.3g}")
        else:
            stats = "(no finite values)"
        self._info.setText(
            f"{key}{derived}  ·  shape={F.shape}  ·  geom={geom}  ·  {stats}"
        )

    def _infer_axis_order(self, shape, axes, geometry) -> list[str]:
        """Best-effort mapping of F's array dims to grid axes.

        Loader conventions in this codebase:
          * geometry==1: F has shape (nz,)              → ['z']
          * geometry==4 or 5 (cyl): F has shape (nz, nr) → ['z', 'r']
          * geometry==6 (2D Cart): F has shape (nx, ny)  → ['x', 'y']
          * geometry==7 (3D Cart): F has shape (nx, ny, nz) → ['x','y','z']
          * geometry==9: F has shape (nz,)              → ['z']
        """
        if geometry in (1, 9):
            return ["z"]
        if geometry in (4, 5):
            return ["z", "r"]
        if geometry == 6:
            return ["x", "y"]
        if geometry == 7:
            return ["x", "y", "z"]
        # Fallback: match dims to axes by length.
        order = []
        used = set()
        for d in shape:
            for ax, arr in axes.items():
                if ax in used:
                    continue
                if len(arr) == d:
                    order.append(ax); used.add(ax); break
            else:
                order.append("?")
        return order

    def _slice_to_2d(self, F: np.ndarray, ax_order: list[str],
                     a: str, b: str, idx: dict) -> "tuple[np.ndarray, str, str] | None":
        """Reduce F to a 2-D array over (a, b) by fixing the other dims.

        Returns ``(F2d_with_a_along_rows_b_along_cols, a, b)`` or None
        if the chosen pair isn't extractable.
        """
        if a not in ax_order or b not in ax_order:
            return None
        # Move (a, b) to the front of the array.
        keep_axes = [ax_order.index(a), ax_order.index(b)]
        slice_axes = [i for i in range(F.ndim) if i not in keep_axes]
        # Reduce slice axes by indexing.
        sl = [slice(None)] * F.ndim
        for ai in slice_axes:
            ax_name = ax_order[ai]
            sl[ai] = idx.get(ax_name, 0)
        F2 = F[tuple(sl)]
        # F2 still has dims in (a, b) order if a-axis-index < b-axis-index;
        # if reversed, transpose.
        if ax_order.index(a) > ax_order.index(b):
            F2 = F2.T
        return F2, a, b

    def _draw_heatmap(self, F, ax_order, axes, idx, a, b) -> None:
        out = self._slice_to_2d(F, ax_order, a, b, idx)
        if out is None:
            self._image.setImage(np.zeros((2, 2)))
            return
        F2, a, b = out
        # ImageItem: axisOrder='row-major' → image[r, c] with rows=y.
        # We want axis a along the *horizontal* direction → transpose.
        img = F2.T if F2.shape[0] == len(axes[a]) else F2
        # Symmetric levels for diverging colormap when sign-changing.
        finite = np.isfinite(F2)
        if not finite.any():
            self._image.setImage(np.zeros((2, 2))); return
        vmax = float(np.nanmax(np.abs(F2)))
        if vmax == 0.0:
            vmax = 1.0
        if (F2[finite] < 0).any() and (F2[finite] > 0).any():
            lo, hi = -vmax, vmax
        else:
            lo, hi = float(np.nanmin(F2)), float(np.nanmax(F2))
            if hi <= lo:
                hi = lo + 1.0
        self._image.setImage(img, autoLevels=False)
        self._image.setLevels((lo, hi))
        self._cbar.setLevels((lo, hi))
        # Position the image rectangle in data coordinates.
        a_arr = axes[a]; b_arr = axes[b]
        a0 = float(a_arr[0]); aN = float(a_arr[-1])
        b0 = float(b_arr[0]); bN = float(b_arr[-1])
        self._image.setRect(pg.QtCore.QRectF(
            a0, b0, aN - a0, bN - b0,
        ))

    def _draw_line_along(self, line_ax, F, ax_order, axes, idx) -> None:
        plot = self._line_plots[line_ax]; curve = self._line_curves[line_ax]
        marker = self._line_markers[line_ax]
        if line_ax not in ax_order:
            curve.setData([], []); return
        # Build slicer that varies line_ax and fixes all others.
        sl = [slice(None)] * F.ndim
        for i, ax_name in enumerate(ax_order):
            if ax_name == line_ax:
                continue
            sl[i] = idx.get(ax_name, 0)
        line = F[tuple(sl)]
        x_axis = axes[line_ax] if line_ax in axes else np.arange(line.size)
        if line.size == 0 or x_axis.size != line.size:
            curve.setData([], []); return
        curve.setData(x_axis, line)
        # Marker = where the OTHER cuts are placed for cross-reference.
        # On F-vs-x, mark the y-coordinate at which we sliced (if y exists).
        # That visualisation is not very meaningful — mark the slice point on
        # this axis itself (always 0 mm or the chosen index value).
        cur_idx = idx.get(line_ax, 0) if line_ax in idx else 0
        if line_ax in axes and len(axes[line_ax]) > cur_idx:
            marker.setValue(float(axes[line_ax][cur_idx]))

    def refresh(self, results) -> None:
        """API parity with other popups; field maps don't depend on results."""
        self._populate_elements()


class _TtfPopup(_PopupPlot):
    """Transit-time factor T(β) plotted across an RF cavity's β range.

    For each FieldMap / FieldMap3D RF cavity (carrying an RF_E channel)
    this popup computes the **phase-optimised TTF**::

        I_c(β) = ∫ E_z(z) · cos(2π z / (β λ)) dz
        I_s(β) = ∫ E_z(z) · sin(2π z / (β λ)) dz
        T(β)   = √(I_c² + I_s²) / ∫|E_z(z)| dz

    The peak of T(β) is annotated as ``β_opt``.  When tracking results
    are available, the *current beam β* at the cavity is also drawn as
    a vertical reference so the user can see how far the design β sits
    from the actual operating β.
    """

    def __init__(self, parent, state):
        super().__init__(parent,
                         "Transit-time factor T(β)  —  RF cavities",
                         size=(1200, 720))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)

        from PyQt6.QtWidgets import QComboBox, QPushButton, QDoubleSpinBox
        tb = QHBoxLayout(); tb.setSpacing(8)
        tb.addWidget(QLabel("Cavity:"))
        self._el_combo = QComboBox(); self._el_combo.setMinimumWidth(280)
        self._el_combo.currentIndexChanged.connect(self._redraw)
        tb.addWidget(self._el_combo)
        tb.addWidget(QLabel("β range:"))
        self._beta_min = QDoubleSpinBox()
        self._beta_min.setRange(0.005, 0.99); self._beta_min.setDecimals(3)
        self._beta_min.setSingleStep(0.01); self._beta_min.setValue(0.05)
        self._beta_min.valueChanged.connect(self._redraw)
        tb.addWidget(self._beta_min)
        tb.addWidget(QLabel("→"))
        self._beta_max = QDoubleSpinBox()
        self._beta_max.setRange(0.01, 1.0); self._beta_max.setDecimals(3)
        self._beta_max.setSingleStep(0.01); self._beta_max.setValue(1.0)
        self._beta_max.valueChanged.connect(self._redraw)
        tb.addWidget(self._beta_max)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._populate)
        tb.addWidget(reload_btn)
        tb.addStretch(1)
        self._info = QLabel("")
        self._info.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:11px;"
        )
        tb.addWidget(self._info, stretch=2)
        v.addLayout(tb)

        self._plot = _mk_plot("T(β)", "")
        self._plot.setLabel("bottom", "β = v/c")
        self._plot.setLabel("left", "T(β)")
        self._curve = self._plot.plot(
            pen=pg.mkPen(theme.ACCENT, width=2.0))
        self._opt_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("#facc15", width=1.4,
                                    style=Qt.PenStyle.DashLine),
            label="β_opt", labelOpts={"position": 0.92,
                                       "color": "#facc15"},
        )
        self._opt_line.hide()
        self._plot.addItem(self._opt_line)
        self._beam_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen("#a3e635", width=1.4,
                                    style=Qt.PenStyle.DotLine),
            label="β_beam", labelOpts={"position": 0.08,
                                        "color": "#a3e635"},
        )
        self._beam_line.hide()
        self._plot.addItem(self._beam_line)
        v.addWidget(self._plot, stretch=1)

        self._fm_elements: list = []
        if state is not None:
            try:
                state.lattice_changed.connect(self._populate)
            except Exception:
                pass
        self._populate()

    def _populate(self, *_):
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        self._el_combo.blockSignals(True)
        self._el_combo.clear()
        self._fm_elements = []
        lat = self._state.lattice if self._state is not None else None
        if lat is not None:
            for i, el in enumerate(lat.elements):
                if not isinstance(el, (FieldMap, FieldMap3D)):
                    continue
                fd = getattr(el, "field_data", None)
                if fd is None:
                    continue
                channel_names = {c.name for c in fd.channels.keys()}
                if "RF_E" not in channel_names:
                    continue
                if float(getattr(el, "ke", 0.0)) == 0.0:
                    continue
                if float(getattr(el, "frequency", 0.0)) <= 0.0:
                    continue
                self._fm_elements.append((i, el))
                self._el_combo.addItem(
                    f"[{i}] {el.name}  (f={el.frequency:.1f} MHz, "
                    f"L={el.length:.0f} mm)"
                )
        self._el_combo.blockSignals(False)
        if not self._fm_elements:
            self._info.setText(
                "No RF cavities with field maps in the lattice.")
            self._curve.setData([], [])
            self._opt_line.hide(); self._beam_line.hide()
            return
        self._redraw()

    def _redraw(self, *_):
        if not self._fm_elements:
            return
        i = max(0, self._el_combo.currentIndex())
        if i >= len(self._fm_elements):
            return
        elem_idx, el = self._fm_elements[i]
        ch = _fieldmap_efield_channel(el)
        if ch is None:
            self._info.setText("No electric channel on this element.")
            return
        fz = _fieldmap_onaxis_fz(ch)
        if fz is None or fz.size < 4:
            self._info.setText("On-axis E_z grid too small for TTF.")
            return
        L_m = float(el.length) * 1e-3
        if L_m <= 0:
            self._info.setText("Element length is zero.")
            return
        n = fz.size
        z_m = np.linspace(-L_m / 2.0, L_m / 2.0, n)
        ke = float(getattr(el, "ke", 1.0))
        scale_global = float(getattr(el, "scale", 1.0))
        norm = float(getattr(ch, "norm_factor", 1.0) or 1.0)
        amp = abs(ke * scale_global / norm)
        ez = amp * fz.astype(float)
        c_mps = 299_792_458.0
        f_hz = float(el.frequency) * 1e6
        wl = c_mps / f_hz
        b_lo = max(1e-3, float(self._beta_min.value()))
        b_hi = max(b_lo + 1e-3, float(self._beta_max.value()))
        betas = np.linspace(b_lo, b_hi, 600)
        T = np.zeros_like(betas)
        _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        denom = float(_trap(np.abs(ez), z_m))
        if denom <= 0:
            self._info.setText("∫|E_z|·dz = 0 — cannot normalise TTF.")
            return
        for j, b in enumerate(betas):
            kz = 2.0 * np.pi * z_m / (b * wl)
            ic = float(_trap(ez * np.cos(kz), z_m))
            isn = float(_trap(ez * np.sin(kz), z_m))
            T[j] = np.sqrt(ic * ic + isn * isn) / denom
        self._curve.setData(betas, T)
        j_opt = int(np.argmax(T))
        beta_opt = float(betas[j_opt]); T_opt = float(T[j_opt])
        self._opt_line.setValue(beta_opt)
        self._opt_line.show()
        beam_beta_str = ""
        results = self._state.results if self._state is not None else None
        try:
            ref_bg = list(getattr(results, "ref_bg", []) or [])
            elem_names = list(getattr(results, "element_names", []) or [])
            if elem_names and ref_bg and el.name in elem_names:
                k = elem_names.index(el.name)
                bg = float(ref_bg[k])
                bb = bg / np.sqrt(1.0 + bg * bg)
                self._beam_line.setValue(bb); self._beam_line.show()
                beam_beta_str = f"  β_beam={bb:.4f}"
            else:
                self._beam_line.hide()
        except Exception:
            self._beam_line.hide()
        self._info.setText(
            f"β_opt = {beta_opt:.4f}   T(β_opt) = {T_opt:.4f}   "
            f"f = {el.frequency:.1f} MHz   L = {L_m*1e3:.1f} mm   "
            f"∫|E_z|·dz = {denom:.3f} MV{beam_beta_str}"
        )

    def refresh(self, results):
        """Rebuild on results change so β_beam tracks the latest run."""
        self._populate()


class _BpmsPopup(_PopupPlot):
    def __init__(self, parent, state: AppState):
        super().__init__(parent, "BPMs", size=(720, 500))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(10, 10, 10, 10)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["BPM", "s [mm]", "σ_x [mm]", "σ_y [mm]", "⟨x⟩ [mm]"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        v.addWidget(self._table)

    def refresh(self, results):
        self._table.setRowCount(0)
        if results is None: return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        sx = np.asarray(getattr(results, "sigma_x", []), dtype=float)
        sy = np.asarray(getattr(results, "sigma_y", []), dtype=float)
        names = getattr(results, "element_names", [])
        centroid = getattr(results, "centroid", None)
        lat = self._state.lattice
        rows = []
        # Source of truth: the lattice's is_bpm flags mapped to record
        # rows via element_exit_idx — correct even with substep
        # recording on, where rows ≠ elements+1 and any positional
        # guess shows drift-interior rows as "BPM readings".  Labeled
        # BPMs ("D01BPM") don't start with "BPM", so the old name-
        # prefix scan stays only as the no-lattice fallback.
        exit_idx = getattr(results, "element_exit_idx", None) or []
        if lat is not None:
            for ei, el in enumerate(lat.elements):
                if not getattr(el, "is_bpm", False):
                    continue
                i = int(exit_idx[ei]) if ei < len(exit_idx) else ei + 1
                if i >= len(names):
                    continue
                rows.append((getattr(el, "name", names[i]),
                             s[i] if i < len(s) else 0.0,
                             sx[i] if i < len(sx) else 0.0,
                             sy[i] if i < len(sy) else 0.0,
                             centroid[i][0] if centroid and i < len(centroid) else 0.0))
        else:
            for i, n in enumerate(names):
                if n and n.upper().startswith("BPM"):
                    rows.append((n,
                                 s[i] if i < len(s) else 0.0,
                                 sx[i] if i < len(sx) else 0.0,
                                 sy[i] if i < len(sy) else 0.0,
                                 centroid[i][0] if centroid and i < len(centroid) else 0.0))
        self._table.setRowCount(len(rows))
        for r, (name, pos, six, siy, cx) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(str(name)))
            for c, v in enumerate((pos, six, siy, cx), start=1):
                it = QTableWidgetItem(f"{v:.3f}")
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, c, it)


class _FloorPlanPopup(_PopupPlot):
    """Design-trajectory survey: top view (x–z, aspect-locked) with the
    horizontal dipole arcs highlighted, and a side view (y vs path
    length s) for vertical bends.  Pure lattice geometry — no run
    required."""

    def __init__(self, parent, state: AppState):
        super().__init__(parent, "Floor plan — design trajectory survey",
                         size=(1100, 640))
        self._state = state
        v = QVBoxLayout(self); v.setContentsMargins(12, 12, 12, 12); v.setSpacing(6)
        self._info = QLabel("")
        self._info.setStyleSheet(f"color:{theme.TEXT_2}; font-size:12px;")
        v.addWidget(self._info)
        self._top = _mk_plot("x (horizontal)", "m")
        self._top.setLabel("bottom", "z", units="m")
        self._top.getViewBox().setAspectLocked(True)
        v.addWidget(self._top, stretch=3)
        # side view is y(s) — after strong horizontal bending the floor
        # plan folds back over itself and y(z) would be multivalued.
        self._side = _mk_plot("y (vertical)", "m")
        self._side.setLabel("bottom", "s along line", units="m")
        v.addWidget(self._side, stretch=1)

    def refresh(self, results) -> None:
        lat = getattr(self._state, "lattice", None)
        self._top.clear(); self._side.clear()
        sv = _survey_lattice(lat, ds_mm=50.0)
        if sv is None or sv["s"].size < 2:
            self._info.setText("No lattice loaded")
            return
        z, x, y, sarr = sv["z"], sv["x"], sv["y"], sv["s"]
        pen = pg.mkPen(theme.ACCENT, width=2)
        self._top.plot(z, x, pen=pen, name="design orbit")
        self._side.plot(sarr, y, pen=pen)
        dip_pen_h = pg.mkPen("#f97316", width=5)
        dip_pen_v = pg.mkPen("#f472b6", width=5)
        for i0, i1 in sv["dip_h"]:
            self._top.plot(z[i0:i1 + 1], x[i0:i1 + 1], pen=dip_pen_h)
        for i0, i1 in sv["dip_v"]:
            self._side.plot(sarr[i0:i1 + 1], y[i0:i1 + 1], pen=dip_pen_v)
        self._top.plot([z[0]], [x[0]], pen=None, symbol="o", symbolSize=9,
                       symbolBrush="#34d399", name="entrance")
        self._top.plot([z[-1]], [x[-1]], pen=None, symbol="s", symbolSize=9,
                       symbolBrush="#f87171", name="exit")
        self._side.plot([sarr[0]], [y[0]], pen=None, symbol="o", symbolSize=7,
                        symbolBrush="#34d399")
        self._side.plot([sarr[-1]], [y[-1]], pen=None, symbol="s", symbolSize=7,
                        symbolBrush="#f87171")
        self._info.setText(
            f"path length {sv['s'][-1]:.3f} m   ·   Σ|θ| = "
            f"{sv['bend_deg']:.2f}°   ·   {len(sv['dip_h'])} horizontal + "
            f"{len(sv['dip_v'])} vertical dipoles   ·   exit at "
            f"(z {z[-1]:+.2f}, x {x[-1]:+.2f}, y {y[-1]:+.3f}) m")


# ---------------------------------------------------------------------------
# Results tab: button grid → spawns popups.
# ---------------------------------------------------------------------------
_BUTTONS_LEGACY: list[tuple[str, str, str]] = [  # preserved for reference
    # (key, label, icon)
    ("rms",         "RMS σ (x · y · z)",                "wave"),
    ("emit",        "Emittance (geometric)",            "chart"),
    ("emit_n",      "Normalised emittance",             "chart"),
    ("twiss",       "Twiss α · β",                      "sliders"),
    ("energy",      "Energy · γ · Transmission",        "gauge"),
    ("loss",        "Loss",                             "collision"),
    ("centroid",    "Centroid ⟨x⟩, ⟨y⟩, ⟨φ⟩",            "target"),
    ("phase",       "Phase Space (4-panel)",            "scatter"),
    ("bpms",        "BPMs",                             "target"),
    ("long_twiss",  "Longitudinal Twiss (α_z · β_z · γ_z)", "sliders"),
    ("divergence",  "Divergence (σ_x' · σ_y')",          "wave"),
    ("peak",        "Peak excursion (X_max · Y_max)",    "chart"),
    ("long_offset", "Longitudinal offset (Δφ_s · ΔW_s)", "target"),
    ("dispersion",  "Dispersion (D_x · D_y)",            "wave"),
    ("dpp",         "σ(Δp/p) along s",                   "wave"),
    ("emit6d",      "6-D emittance (ε_nx·ε_ny·ε_nz)",    "chart"),
    ("power",       "Beam power (I · W · trans)",        "gauge"),
    # Lattice-parameter plots — show N/A when the lattice has no matching
    # elements (e.g. a FODO has quads but no RF, a cavity chain has no quads).
    ("quad_grad",   "Quadrupole gradient",              "sliders"),
    ("quad_gl",     "Quadrupole ∫G·ds",                  "sliders"),
    ("rf_volt",     "RF voltage",                       "gauge"),
    ("sync_phase",  "Synchronous phase",                "target"),
    ("sigma",       "Sigma Matrix (6×6)",               "grid"),
    ("tmatrix",     "Transfer Matrix (6×6)",            "grid"),
    ("convergence", "SC Convergence",                   "heatmap"),
]


# ---------------------------------------------------------------------------
# Derived sparkline series — for tiles whose thumbnail is not a plain
# results attribute.  Built per-tab (closures over AppState so lattice
# tiles render live values even before any run).  Every function returns
# ``ys`` (aligned with results.s), ``(xs, ys)``, or None → "—" placeholder.
# ---------------------------------------------------------------------------
def _bpeak_value(el):
    """Peak |B_z|: lumped Solenoid axis field first (|field|/Bz/B0 — the
    popup's extraction), field-map on-axis peak as the fallback."""
    for a in ("field", "Bz", "B0"):
        v = getattr(el, a, None)
        if v is not None:
            return abs(float(v))
    return _fieldmap_bpeak_T(el)


def _brho_for_element(state, el):
    """Beam rigidity Bρ [T·m] at ``el``.

    Species (mass, |q|) always comes from the beam config.  βγ is the
    run's per-element reference value when results exist (exact through
    accelerating sections); without a run it falls back to the entrance
    energy — exact for fixed-energy transfer lines, a live preview
    otherwise.  Returns None without a beam config: a BEND card stores
    only geometry (θ, ρ), so B is undefined until a rigidity is known.
    """
    cfg = getattr(state, "beam_config", None) if state is not None else None
    if cfg is None:
        return None
    try:
        ref, _ = _beam_inputs_from_config(cfg)
    except Exception:                                        # noqa: BLE001
        return None
    m = float(ref.species.mass)
    q = abs(float(getattr(ref.species, "charge", 1) or 1))
    bg = float(ref.bg)                       # entrance βγ (preview)
    res = getattr(state, "results", None)
    lat = getattr(state, "lattice", None)
    if res is not None and lat is not None:
        try:
            # NOTE: no ``x or []`` here — ref_gamma is an ndarray and
            # ambiguous-truth would raise, silently killing this branch.
            g_raw = getattr(res, "ref_gamma", None)
            g = np.asarray(g_raw if g_raw is not None else [], float)
            ei_raw = getattr(res, "element_exit_idx", None)
            exit_idx = list(ei_raw) if ei_raw is not None else []
            els = list(getattr(lat, "elements", []))
            if g.size and len(exit_idx) == len(els):
                i = next(j for j, e in enumerate(els) if e is el)
                gamma = float(g[min(int(exit_idx[i]), g.size - 1)])
                if gamma > 1.0:
                    bg = float(np.sqrt(gamma * gamma - 1.0))
        except Exception:                                    # noqa: BLE001
            pass                # stale/mismatched results → entrance βγ
    return bg * m / (299.792458 * q)


def _bend_field_T(state):
    """value_fn factory: hard-edge |B| [T] of a BEND, B = Bρ/|ρ|."""
    def fn(el):
        brho = _brho_for_element(state, el)
        rho_m = abs(float(getattr(el, "rho", 0.0) or 0.0)) * 1e-3
        if brho is None or rho_m <= 0:
            return None
        return brho / rho_m
    return fn


def _survey_lattice(lattice, ds_mm: float = 100.0):
    """3-D design-trajectory survey of the lattice (floor-plan geometry).

    Walks the reference orbit with an orthonormal frame (tangent t,
    horizontal normal n_h, vertical normal n_v).  Straight elements
    translate along t; each Dipole rotates the frame about n_v
    (horizontal bend, hv=0) or n_h (vertical bend, hv=1) via Rodrigues
    steps, sampling the arc every ~``ds_mm``.  Positive horizontal angle
    curves toward +x; positive vertical angle toward +y — matching the
    sign convention of the BEND cards.

    Returns None when the lattice is missing, else a dict of float
    arrays in metres: ``s, z, x, y`` plus ``dip_h`` / ``dip_v`` lists of
    (i0, i1) point-index spans covering each horizontal / vertical
    dipole (for highlighting), and ``bend_deg`` = Σ|θ|.
    """
    from linac_gen.elements.dipole import Dipole
    if lattice is None:
        return None

    def rot(v, axis, ang):
        return (v * np.cos(ang) + np.cross(axis, v) * np.sin(ang)
                + axis * np.dot(axis, v) * (1.0 - np.cos(ang)))

    p = np.zeros(3)                        # (z, x, y) in mm
    t = np.array([1.0, 0.0, 0.0])          # tangent: along +z
    n_h = np.array([0.0, 1.0, 0.0])        # +x
    n_v = np.array([0.0, 0.0, 1.0])        # +y
    pts, svals = [p.copy()], [0.0]
    dip_h, dip_v = [], []
    s = 0.0
    total_bend = 0.0
    for el in getattr(lattice, "elements", []):
        L = float(getattr(el, "length", 0.0) or 0.0)
        if L == 0.0:
            continue
        # Negative drifts (MAD-style overlap bookkeeping) fall through to
        # the straight-translation branch and step backward — skipping
        # them would inflate the path length and shift everything
        # downstream.
        if L > 0.0 and isinstance(el, Dipole) and abs(float(el.angle)) > 1e-12:
            th = float(np.radians(el.angle))
            total_bend += abs(float(el.angle))
            vertical = int(getattr(el, "hv", 0)) == 1
            axis = -n_h if vertical else n_v
            n = max(4, int(L / ds_mm))
            i0 = len(pts) - 1
            dth, dl = th / n, L / n
            for _ in range(n):
                # rotate half, advance, rotate half (midpoint stepping)
                t = rot(t, axis, dth / 2.0)
                p = p + t * dl
                t = rot(t, axis, dth / 2.0)
                n_h = rot(n_h, axis, dth)
                n_v = rot(n_v, axis, dth)
                s += dl
                pts.append(p.copy()); svals.append(s)
            (dip_v if vertical else dip_h).append((i0, len(pts) - 1))
        else:
            p = p + t * L
            s += L
            pts.append(p.copy()); svals.append(s)
    pts = np.asarray(pts) * 1e-3
    return {"s": np.asarray(svals) * 1e-3,
            "z": pts[:, 0], "x": pts[:, 1], "y": pts[:, 2],
            "dip_h": dip_h, "dip_v": dip_v, "bend_deg": total_bend}


def _floorplan_series(state):
    """Sparkline series for the floor-plan tile: plan-view x(z) [m].

    Coarse sampling (1 m) — the thumbnail only needs the shape; the
    popup resamples finely.  A straight lattice gives a flat line at 0,
    which is the honest picture rather than a placeholder.
    """
    def fn(_results):
        lat = state.lattice if state is not None else None
        sv = _survey_lattice(lat, ds_mm=1000.0)
        if sv is None or sv["s"].size < 2:
            return None
        return (sv["z"], sv["x"])
    return fn


def _lattice_param_series(state, element_types, attr=None, value_fn=None,
                          transform=None):
    """(mid-s [m], value) per matching lattice element — the same data the
    ``_LatticeParamPopup`` stems show."""
    def fn(_results):
        lat = state.lattice if state is not None else None
        if lat is None:
            return None
        xs, ys = [], []
        s_mm = 0.0
        for el in getattr(lat, "elements", []):
            L = float(getattr(el, "length", 0.0) or 0.0)
            mid = (s_mm + L / 2.0) * 1e-3
            s_mm += L
            if not isinstance(el, element_types):
                continue
            try:
                if value_fn is not None:
                    v = value_fn(el)
                else:
                    v = getattr(el, attr, None)
                if v is None:
                    continue
                v = float(v)
                if transform is not None:
                    v = transform(el, v)
            except Exception:                                # noqa: BLE001
                continue          # one bad element must not kill the tile
            xs.append(mid); ys.append(v)
        return (np.asarray(xs), np.asarray(ys)) if ys else None
    return fn


def _build_series_fns(state) -> dict:
    """key → series_fn for every tile whose thumbnail is derivable."""
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.ncells import NCells

    def _arr(results, name):
        a = np.asarray(getattr(results, name, []) or [], dtype=float)
        return a

    def _mass(results):
        m = float(getattr(results, "mass_mev", 0.0) or 0.0)
        if not m:
            g = _arr(results, "ref_gamma"); w = _arr(results, "ref_w_kin")
            if g.size and w.size and g[0] > 1 + 1e-9:
                m = float(w[0]) / max(float(g[0]) - 1.0, 1e-9)
        return m

    def emit6d(results):
        if results is None:
            return None
        e1, e2, e3 = (_arr(results, "emit_e1"), _arr(results, "emit_e2"),
                      _arr(results, "emit_e3"))
        if e1.size and e1.size == e2.size == e3.size:
            return e1 * e2 * e3
        return None

    def phase_adv(results):
        # Cumulative beam μ_x(s) [deg] — s is mm, β is mm/mrad.
        if results is None:
            return None
        s = _arr(results, "s"); bx = _arr(results, "beta_x")
        if s.size < 2 or bx.size != s.size:
            return None
        inv = np.where(bx > 0, 1.0 / np.clip(bx, 1e-30, None), 0.0)
        mu = np.concatenate(
            [[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))])
        return mu * (180.0 / np.pi) * 1e-3

    def _channel(results):
        from linac_gen.analysis.period_detect import detect_periods
        from linac_gen.analysis.phase_advance import channel_phase_advance
        if results is None or not getattr(results, "element_maps_dep", None):
            return None
        lat = state.lattice if state is not None else None
        if lat is None:
            return None
        periods = detect_periods(lat)
        multi = [p for p in periods if p.n_repeats > 1]
        if multi:
            # A genuine repeating block (e.g. the 8-cell HWR bracket).
            period = max(multi, key=lambda p: p.n_repeats)
        else:
            # Single-repeat LATTICE brackets are still declared periods;
            # the "(whole lattice)" catch-all is NOT — the eigenphase of
            # an arbitrary aperiodic line is not a tune, so show the
            # placeholder rather than a misleading number.
            declared = [p for p in periods if p.source != "fallback"]
            if not declared:
                return None
            period = declared[0]
        return channel_phase_advance(results, period)

    def tune_depr(results):
        ch = _channel(results)
        if ch is None:
            return None
        eta = np.asarray(
            ch.get("eta_I" if ch["coupled_xy"] else "eta_x", []), float)
        cells = np.asarray(ch.get("cells", []), float)
        return (cells, eta) if eta.size else None

    def hofmann(results):
        # The chart's abscissa k_z/k_x per cell (the depression ordinate
        # is already the tune_depr tile) — NaN for DC beams (no z tune),
        # rendering the placeholder.
        from linac_gen.analysis.hofmann import hofmann_trajectory
        ch = _channel(results)
        if ch is None:
            return None
        tr = hofmann_trajectory(ch, results)
        return (np.asarray(tr["cells"], float),
                np.asarray(tr["ratio"], float))

    def long_twiss(results):
        from linac_gen.analysis.phase_advance import _beta_z_eff_from_sigma
        if results is None:
            return None
        bz = _beta_z_eff_from_sigma(results)
        return None if bz is None else np.asarray(bz, float)

    def _sigma_diag(results, i):
        mats = getattr(results, "sigma_matrix", None) or []
        s = _arr(results, "s")
        if len(mats) != s.size or not len(mats):
            return None
        out = np.full(s.size, np.nan)
        for k, S in enumerate(mats):
            S = np.asarray(S, dtype=float)
            if S.shape == (6, 6) and S[i, i] >= 0:
                out[k] = np.sqrt(S[i, i])
        return out

    def divergence(results):
        return None if results is None else _sigma_diag(results, 1)

    def power(results):
        if results is None:
            return None
        w = _arr(results, "ref_w_kin")
        if not w.size:
            return None
        cfg = state.beam_config if state is not None else None
        i_ma = float(getattr(cfg, "current",
                             getattr(results, "current_mA", 0.0) or 0.0)
                     if cfg is not None else
                     getattr(results, "current_mA", 0.0) or 0.0)
        tr = _arr(results, "transmission")
        frac = tr / 100.0 if tr.size == w.size else np.ones_like(w)
        return i_ma * 1e-3 * w * 1e6 * frac

    def aperture_loss(results):
        if results is None:
            return None
        tr = _arr(results, "transmission")
        if not tr.size:                      # envelope: lossless
            s = _arr(results, "s")
            return np.zeros(s.size) if s.size else None
        return 100.0 - tr

    def _centroid_comp(results, comp):
        if results is None:
            return None
        s = _arr(results, "s")
        cents = getattr(results, "centroid", None) or []
        if len(cents) != s.size:
            # Centroid-less results (loaded archives, foreign objects,
            # backtrack): no first moment recorded → show 0.
            return np.zeros(s.size) if s.size else None
        out = np.full(s.size, np.nan)
        for k, c in enumerate(cents):
            c = np.asarray(c, dtype=float).flatten()
            if c.size >= 6:
                out[k] = c[comp]
        return out

    def centroid(results):
        return _centroid_comp(results, 0)

    def long_offset(results):
        return _centroid_comp(results, 4)

    def dispersion(results):
        # D_x [m] = (Σ[0,5]/Σ[5,5])·β²γ·mc² · 1e-3  (same as the popup).
        if results is None:
            return None
        mats = getattr(results, "sigma_matrix", None) or []
        s = _arr(results, "s")
        beta = _arr(results, "ref_beta"); gamma = _arr(results, "ref_gamma")
        m = _mass(results)
        if (len(mats) != s.size or not len(mats) or not m
                or beta.size != s.size or gamma.size != s.size):
            return None
        out = np.full(s.size, np.nan)
        for k, S in enumerate(mats):
            S = np.asarray(S, dtype=float)
            if S.shape == (6, 6) and S[5, 5] > 0:
                out[k] = (S[0, 5] / S[5, 5]
                          * beta[k] * beta[k] * gamma[k] * m) * 1e-3
        return out

    def dpp(results):
        if results is None:
            return None
        sw = _arr(results, "sigma_w")
        beta = _arr(results, "ref_beta"); gamma = _arr(results, "ref_gamma")
        m = _mass(results)
        if not (sw.size and sw.size == beta.size == gamma.size and m):
            return None
        return sw / (beta * beta * gamma * m)

    return {
        "emit6d": emit6d,
        "phase_adv": phase_adv,
        "tune_depr": tune_depr,
        "hofmann": hofmann,
        "long_twiss": long_twiss,
        "divergence": divergence,
        "power": power,
        "aperture_loss": aperture_loss,
        "centroid": centroid,
        "long_offset": long_offset,
        "dispersion": dispersion,
        "dpp": dpp,
        "quad_grad": _lattice_param_series(
            state, (Quadrupole,), attr="gradient"),
        "quad_gl": _lattice_param_series(
            state, (Quadrupole,), attr="gradient",
            transform=lambda el, v: v * (float(el.length) * 1e-3)),
        "rf_volt": _lattice_param_series(
            state, (RFGap, FieldMap, FieldMap3D, NCells), attr=None,
            value_fn=lambda el: (
                _ncells_v0_MV(el) if isinstance(el, NCells)
                else abs(float(el.voltage))
                if getattr(el, "voltage", None) is not None
                else _fieldmap_vgap_MV(el))),
        "eacc": _lattice_param_series(
            state, (FieldMap, FieldMap3D, NCells), attr=None,
            value_fn=lambda el: (
                float(el.eot_v_per_m) * 1e-6 if isinstance(el, NCells)
                else _fieldmap_eacc_MV_per_m(el))),
        "bpeak": _lattice_param_series(
            state, (Solenoid, FieldMap, FieldMap3D), attr=None,
            value_fn=_bpeak_value),
        "int_b2": _lattice_param_series(
            state, (Solenoid, FieldMap, FieldMap3D), attr=None,
            value_fn=_solenoid_int_b2),
        "sync_phase": _lattice_param_series(
            state, (RFGap, FieldMap, FieldMap3D, NCells), attr=None,
            value_fn=_rf_phase_value),
        "bend_b": _lattice_param_series(
            state, (Dipole,), attr=None, value_fn=_bend_field_T(state)),
        "bend_bl": _lattice_param_series(
            state, (Dipole,), attr=None, value_fn=_bend_field_T(state),
            # B[T] × arc length[mm] → T·m needs L in metres
            transform=lambda el, v: v * (float(el.length) * 1e-3)),
        "floorplan": _floorplan_series(state),
    }


# ---------------------------------------------------------------------------
# Redesigned layout: sectioned cards with sparklines
# ---------------------------------------------------------------------------
# Each tuple is (key, label, icon, series_attr, unit, fmt, accent_color).
# ``series_attr`` of None means no sparkline UNLESS the key has a derived
# series in ``_build_series_fns`` (per-cell tunes, dispersion, lattice
# parameters…); truly plain tiles (phase space, matrix viewers) stay
# thumbnail-less because their data isn't a 1-D curve.

_SECTIONS: list[tuple[str, list[tuple]]] = [
    ("BEAM SIZE & EMITTANCE", [
        ("rms",     "RMS σ (x · y · z)",            "wave",   "sigma_x",
            "mm",        "{:.3f}", theme.ACCENT),
        ("emit",    "Geometric emittance",          "chart",  "emit_x",
            "mm·mrad",   "{:.3f}", "#a3e635"),
        ("emit_n",  "Normalised emittance",         "chart",  "emit_nx",
            "mm·mrad",   "{:.3f}", "#a3e635"),
        ("emit6d",  "6-D emittance",                "chart",  None,
            "",          "{:.2e}", "#e879f9"),
    ]),
    ("TWISS · DIVERGENCE · HALO", [
        ("twiss",       "Transverse Twiss α · β",   "sliders", "beta_x",
            "mm/mrad",  "{:.3f}", theme.ACCENT),
        ("phase_adv",   "Phase advance σ₀ · σ",      "sliders", None,
            "deg",      "{:.1f}", "#a3e635"),
        ("tune_depr",   "Tune depression η = σ/σ₀",  "gauge",   None,
            "",         "{:.3f}", "#fbbf24"),
        ("hofmann",     "Hofmann stability chart",   "scatter", None,
            "k_z/k_x",  "{:.3f}", "#f472b6"),
        ("footprint",   "Tune footprint (frozen SC)", "scatter", None,
            "",         "",       "#38bdf8"),
        ("long_twiss",  "Longitudinal Twiss",        "sliders", None,
            "mm/mrad",  "{:.3g}", "#fbbf24"),
        ("divergence",  "Divergence σ_x' · σ_y'",    "wave",    None,
            "mrad",     "{:.3f}", theme.ACCENT),
        ("peak",        "Peak excursion X / Y_max",  "chart",   "x_max",
            "mm",       "{:.3f}", "#f87171"),
        ("halo",        "Halo parameter H_x · H_y",  "scatter", "halo_x",
            "",         "{:.3f}", "#fde047"),
    ]),
    ("ENERGY · KINEMATICS", [
        ("energy",      "Energy · γ · Transmission", "gauge",   "ref_w_kin",
            "MeV",      "{:.4f}", theme.ACCENT),
        ("power",       "Beam power",                "gauge",   None,
            "W",        "{:.2f}", "#4ade80"),
        ("emit4d",      "4-D invariant ε_4D",        "chart",   "emit_4d",
            "(mm·mrad)²", "{:.3g}", "#a78bfa"),
        ("eigenemit",   "Eigenemittances ε₁·ε₂·ε₃",  "chart",   "emit_e1",
            "mm·mrad",  "{:.3g}", "#c084fc"),
    ]),
    ("LOSSES · TRANSMISSION", [
        ("loss",        "Loss profile",              "collision","transmission",
            "%",        "{:.2f}", "#f87171"),
        ("aperture_loss", "Aperture-profile losses", "collision", None,
            "%",        "{:.2f}", "#f97316"),
        ("ibs",         "Intra-beam stripping (H⁻)", "collision", None,
            "W",        "{:.2f}", "#fbbf24"),
        ("magstrip",    "Magnetic stripping (H⁻)",    "magnet",    None,
            "W",        "{:.2f}", "#fb923c"),
        ("ensemble",    "Error study ensemble",      "scatter",   None,
            "",         "",       "#60a5fa"),
    ]),
    ("CENTROID · DISPERSION", [
        ("centroid",    "Centroid ⟨x⟩ · ⟨y⟩ · ⟨φ⟩",   "target",  None,
            "mm",       "{:.3g}", "#fbbf24"),
        ("long_offset", "Long. offset Δφ_s · ΔW_s",  "target",  None,
            "deg",      "{:.3g}", "#a78bfa"),
        ("dispersion",  "Dispersion D_x · D_y",      "wave",    None,
            "m",        "{:.3e}", "#a3e635"),
        ("dpp",         "σ(Δp/p) along s",           "wave",    None,
            "",         "{:.3e}", "#f97316"),
    ]),
    ("PHASE SPACE · DIAGNOSTICS", [
        ("phase",       "Phase space (4-panel)",     "scatter", None,
            "",         "",       theme.ACCENT),
        ("density_s",   "Density vs s · heatmap",    "heatmap", None,
            "",         "",       "#e879f9"),
        ("bpms",        "BPMs",                      "target",  None,
            "",         "",       "#f472b6"),
        ("field_map",   "Field maps · 2D + cuts",    "magnet",  None,
            "",         "",       "#60a5fa"),
        ("ttf",         "Cavity TTF · T(β)",         "wave",    None,
            "",         "",       "#fbbf24"),
    ]),
    ("LATTICE PARAMETERS", [
        ("quad_grad",   "Quadrupole gradient",       "sliders", None,
            "T/m",      "{:.2f}", theme.ACCENT),
        ("quad_gl",     "Quadrupole ∫G·ds",           "sliders", None,
            "T",        "{:.3f}", "#a3e635"),
        ("rf_volt",     "RF voltage (V₀)",           "gauge",   None,
            "MV",       "{:.3f}", "#fbbf24"),
        ("eacc",        "Peak E_acc",                "zap",     None,
            "MV/m",     "{:.2f}", "#fbbf24"),
        ("bpeak",       "Peak solenoid |B_z|",       "magnet",  None,
            "T",        "{:.3f}", "#60a5fa"),
        ("int_b2",      "Solenoid ∫B²·dz",           "magnet",  None,
            "T²·m",     "{:.3g}", "#38bdf8"),
        ("bend_b",      "Dipole field |B|",          "magnet",  None,
            "T",        "{:.4f}", "#f97316"),
        ("bend_bl",     "Dipole ∫B·dl",              "magnet",  None,
            "T·m",      "{:.4f}", "#fb7185"),
        ("floorplan",   "Floor plan (survey)",       "scatter", None,
            "",         "",       "#f97316"),
        ("sync_phase",  "Synchronous phase",         "target",  None,
            "deg",      "{:.1f}", "#f472b6"),
    ]),
    ("CROSS-CHECKS · COMPARE", [
        ("partran",     "Compare with TraceWin partran", "scatter", None,
            "",         "",       "#f97316"),
    ]),
    ("ADVANCED · MATRIX VIEWERS", [
        ("sigma",       "Σ matrix (6×6)",             "grid",   None,
            "",         "",       "#a78bfa"),
        ("tmatrix",     "Transfer matrix (6×6)",     "grid",   None,
            "",         "",       "#a78bfa"),
        ("convergence", "SC convergence",            "heatmap", None,
            "",         "",       "#e879f9"),
    ]),
]


class ResultsTab(QWidget):
    open_sigma_matrix_requested     = None  # wired by app
    open_transfer_matrix_requested  = None
    open_convergence_requested      = None

    def __init__(self, state: AppState,
                 open_sigma_cb, open_tmatrix_cb, open_convergence_cb):
        super().__init__()
        self.state = state
        self._popups: dict[str, QDialog] = {}

        # Subtle top-to-bottom gradient so the results canvas has depth
        # instead of a flat BG_0 — reads as a "stage" the cards sit on.
        self.setStyleSheet(
            "ResultsTab, QWidget#ResultsTabRoot {"
            " background: qlineargradient(x1:0 y1:0 x2:0 y2:1,"
            f"    stop:0 {theme.BG_1}, stop:0.25 {theme.BG_0},"
            f"    stop:1 #05070a);"
            "}"
        )
        self.setObjectName("ResultsTabRoot")

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # --- KPI header row --------------------------------------------
        kpi_lay = QHBoxLayout(); kpi_lay.setSpacing(8)
        self._k_sx = kpi_card("σ_x end", "—", "mm")
        self._k_sy = kpi_card("σ_y end", "—", "mm")
        self._k_sz = kpi_card("σ_z end", "—", "mm")
        self._k_ex = kpi_card("ε_x growth", "—")
        self._k_loss = kpi_card("loss", "—", "%")
        self._k_trans = kpi_card("transmission", "—", "%")
        for k in (self._k_sx, self._k_sy, self._k_sz, self._k_ex,
                  self._k_trans, self._k_loss):
            kpi_lay.addWidget(k, stretch=1)
        v.addLayout(kpi_lay)

        hint = QLabel(
            "Tap any card to open its full-size plot — windows stay open so "
            "you can compare side-by-side.  The sparkline inside each card is "
            "a live preview of the underlying quantity along s."
        )
        hint.setStyleSheet(f"color:{theme.TEXT_2}; font-size:13px;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        # --- Import / load saved run -----------------------------------
        # Loads any HDF5 produced by save_results_hdf5 (auto-saved after
        # every run into the configured calc dir) so all the cards and
        # popups below re-populate from the on-disk arrays.
        import_row = QHBoxLayout(); import_row.setSpacing(6)
        import_row.addStretch(1)
        self._import_btn = QPushButton("  Import Results…")
        self._import_btn.setIcon(icon("folder-open", 12))
        self._import_btn.setStyleSheet(
            f"background:{theme.BG_2}; color:{theme.TEXT_0};"
            f"border:1px solid {theme.BORDER_1}; border-radius:3px;"
            f"padding:6px 12px;"
        )
        self._import_btn.clicked.connect(self._import_results)
        import_row.addWidget(self._import_btn)
        v.addLayout(import_row)

        # --- Sectioned card grid -------------------------------------
        from linac_gen_gui.interphase.panels.result_card import (
            ResultCard, section_header,
        )
        # Scroll area so the redesigned layout stays readable on small screens
        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Inherit the gradient from the tab — transparent scroll viewport
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: 0; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        inner = QWidget(); inner.setStyleSheet("background: transparent;")
        scroll_lay = QVBoxLayout(inner)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(12)

        # Callbacks for classic dialogs (routed through app.py)
        classic_routes = {
            "sigma":       open_sigma_cb,
            "tmatrix":     open_tmatrix_cb,
            "convergence": open_convergence_cb,
        }
        self._cards: list[ResultCard] = []
        COLS = 4

        # Derived-series providers (per-cell tunes, dispersion, lattice
        # parameters…) — closures over AppState so lattice tiles render
        # live values even before any run.
        self._series_fns = _build_series_fns(state)

        self._ibs_card: Optional[ResultCard] = None
        for section_name, cards_def in _SECTIONS:
            scroll_lay.addWidget(section_header(section_name))
            grid = QGridLayout(); grid.setSpacing(10)
            for idx, (key, label, ic, attr, unit, fmt, accent) in enumerate(cards_def):
                card = ResultCard(
                    key=key, title=label, icon_name=ic, accent=accent,
                    series_attr=attr, unit=unit,
                    value_fmt=fmt if fmt else "{:.3g}",
                    series_fn=self._series_fns.get(key),
                )
                if key in classic_routes:
                    card.clicked.connect(
                        lambda _k, cb=classic_routes[key]: cb()
                    )
                else:
                    card.clicked.connect(self._open_popup)
                grid.addWidget(card, idx // COLS, idx % COLS)
                self._cards.append(card)
                if key == "ibs":
                    self._ibs_card = card
            # Make the grid columns stretch uniformly
            for c in range(COLS):
                grid.setColumnStretch(c, 1)
            scroll_lay.addLayout(grid)

        scroll_lay.addStretch(1)
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)

        state.results_changed.connect(self._refresh)
        # Tile is H⁻-only — re-evaluate whenever the beam config changes
        # (Apply on the Beam tab) and once at construction time.
        state.beam_config_changed.connect(self._update_ibs_tile_state)
        self._update_ibs_tile_state(getattr(state, "beam_config", None))
        # Lattice-derived sparklines (quad gradients, RF voltages, …) read
        # the lattice live — refresh the cards when it changes, and once
        # now so they render before any run.
        state.lattice_changed.connect(self._refresh_cards_from_state)
        # beam-config edits change the dipole-field tiles (B = Bρ/ρ needs
        # the beam rigidity); the other lattice tiles ignore the config.
        state.beam_config_changed.connect(self._refresh_cards_from_state)
        self._refresh_cards_from_state()

    def _refresh_cards_from_state(self, *_a) -> None:
        results = getattr(self.state, "results", None)
        for card in self._cards:
            card.set_results(results)

    # ------------------------------------------------------------------
    # Load a previously-saved HDF5 run and feed it back through the
    # same AppState.set_results path that a live simulation uses, so the
    # KPI cards, sparklines, and popups all re-populate automatically.
    def _import_results(self) -> None:
        from PyQt6.QtCore import QSettings as _QS

        from linac_gen.io.hdf5_output import load_results_hdf5
        from linac_gen.io.openpmd_output import (
            load_results_openpmd, is_openpmd_file,
        )

        s = _QS("Linac_Gen", "Interphase")
        # Default to the configured calc dir (same setting used by
        # auto-save) so users land where their dumps live.
        calc_dir = str(s.value("calcDir", "")) or str(Path.cwd() / "runs")
        path, _ = QFileDialog.getOpenFileName(
            self, "Import results…",
            calc_dir,
            "All results (*.h5 *.opmd.h5);;"
            "HELIX HDF5 (*.h5);;openPMD HDF5 (*.opmd.h5);;All files (*)",
        )
        if not path:
            return
        try:
            # Auto-detect format: openPMD files carry a root-level
            # ``openPMD`` attribute we can sniff cheaply.
            if is_openpmd_file(path):
                data = load_results_openpmd(path)
            else:
                data = load_results_hdf5(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        if not data:
            QMessageBox.warning(self, "Empty file",
                                f"{Path(path).name} contained no recognised "
                                "envelope or reference data.")
            return
        # Wrap the dict as an attribute object so every plot's
        # ``getattr(results, "sigma_x", [])`` lookup just works.
        loaded = _LoadedResults(data, Path(path))
        self.state.set_results(loaded)
        self.state.status_message.emit(f"Loaded {Path(path).name}")

    def _update_ibs_tile_state(self, cfg) -> None:
        """Enable the IBS tile only when the active beam is H⁻."""
        if self._ibs_card is None:
            return
        is_hminus = (cfg is not None
                     and getattr(cfg, "species", None) == "H-")
        self._ibs_card.setEnabled(is_hminus)
        if is_hminus:
            self._ibs_card.setToolTip(
                "Lebedev intra-beam stripping (H⁻).  Click to open the "
                "popup with per-step loss curves."
            )
        else:
            cur = (getattr(cfg, "species", "—") if cfg is not None
                   else "—")
            self._ibs_card.setToolTip(
                f"Intra-beam stripping applies to H⁻ beams only "
                f"(current species: {cur!r}).\n"
                f"Set Species → H- in the Beam tab and click Apply."
            )

    # ------------------------------------------------------------------
    def shutdown_begin(self) -> list:
        """App teardown: signal every popup-owned σ₀ worker plus any
        parked stragglers from earlier popup closes, and hand them to
        the window's bounded-wait loop."""
        out = []
        for dlg in getattr(self, "_popups", {}).values():
            w = getattr(dlg, "_worker", None)
            if w is not None and hasattr(w, "isRunning") and w.isRunning():
                if hasattr(w, "request_stop"):
                    w.request_stop()
                w.requestInterruption()
                out.append(w)
        for w in list(_ZOMBIE_WORKERS):
            if hasattr(w, "isRunning") and w.isRunning():
                out.append(w)
        return out

    def plot_catalog(self):
        """[(key, label)] of every result plot card (from _SECTIONS) — the
        catalogue the assistant resolves plot names against."""
        return [(c[0], c[1]) for _sec, cards in _SECTIONS for c in cards]

    def open_plot(self, key) -> bool:
        """Open a result plot by its card key, routed EXACTLY like a user
        card click (popup dialog or classic viewer) — including the card's
        enabled state (a disabled card, e.g. IBS for a non-H⁻ beam, never
        receives clicks, so the assistant must not bypass it either).
        GUI thread only."""
        valid = {c[0] for _sec, cards in _SECTIONS for c in cards}
        if key not in valid:
            return False
        for card in self._cards:
            if getattr(card, "_key", None) == key:
                if not card.isEnabled():
                    return False                # gated off, like a real click
                card.clicked.emit(key)          # -> _open_popup or classic cb
                return True
        return False

    def _open_popup(self, key: str) -> None:
        dlg = self._popups.get(key)
        if dlg is None:
            if key == "rms":       dlg = _RmsPopup(self)
            elif key == "emit":    dlg = _EmittancePopup(self)
            elif key == "emit_n":  dlg = _NormEmittancePopup(self)
            elif key == "twiss":   dlg = _TwissPopup(self)
            elif key == "phase_adv": dlg = _PhaseAdvancePopup(self, self.state)
            elif key == "tune_depr": dlg = _TuneDepressionPopup(self, self.state)
            elif key == "hofmann":   dlg = _HofmannPopup(self, self.state)
            elif key == "footprint": dlg = _FootprintPopup(self, self.state)
            elif key == "field_map": dlg = _FieldMapPopup(self, self.state)
            elif key == "ttf":     dlg = _TtfPopup(self, self.state)
            elif key == "energy":  dlg = _EnergyPopup(self)
            elif key == "loss":    dlg = _LossPopup(self)
            elif key == "aperture_loss": dlg = _ApertureLossPopup(self, self.state)
            elif key == "ibs":     dlg = _IbsPopup(self, self.state)
            elif key == "magstrip":dlg = _MagStripPopup(self, self.state)
            elif key == "ensemble":dlg = _EnsemblePopup(self, self.state)
            elif key == "centroid":dlg = _CentroidPopup(self, self.state)
            elif key == "phase":   dlg = _PhaseSpacePopup(self, self.state)
            elif key == "density_s": dlg = _DensityPopup(self)
            elif key == "bpms":    dlg = _BpmsPopup(self, self.state)
            elif key == "long_twiss":  dlg = _LongTwissPopup(self)
            elif key == "halo":        dlg = _HaloPopup(self)
            elif key == "partran":     dlg = _PartranComparePopup(self, self.state)
            elif key == "divergence":  dlg = _DivergencePopup(self)
            elif key == "peak":        dlg = _PeakExcursionPopup(self)
            elif key == "long_offset": dlg = _LongOffsetPopup(self)
            elif key == "dispersion":  dlg = _DispersionPopup(self, self.state)
            elif key == "dpp":         dlg = _DpPRmsPopup(self)
            elif key == "emit6d":      dlg = _Emit6DPopup(self)
            elif key == "emit4d":      dlg = _Emit4DPopup(self)
            elif key == "eigenemit":   dlg = _EigenEmitPopup(self)
            elif key == "power":       dlg = _BeamPowerPopup(self, self.state)
            elif key == "quad_grad":
                from linac_gen.elements.quadrupole import Quadrupole
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="Quadrupole gradient  —  G [T/m] per QUAD",
                    element_types=(Quadrupole,), attr="gradient",
                    ylabel="G", yunits="T/m",
                    color=theme.ACCENT, type_name="quadrupole",
                )
            elif key == "quad_gl":
                from linac_gen.elements.quadrupole import Quadrupole
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="Quadrupole gradient integral  —  ∫G·ds per QUAD",
                    element_types=(Quadrupole,), attr="gradient",
                    ylabel="∫G·ds", yunits="T",
                    color="#a3e635", type_name="quadrupole",
                    # G[T/m] × length[mm] → T·m needs L in metres
                    transform=lambda el, v: v * (el.length * 1e-3),
                )
            elif key == "rf_volt":
                from linac_gen.elements.rf_gap import RFGap
                from linac_gen.elements.field_map import FieldMap
                from linac_gen.elements.field_map_3d import FieldMap3D
                from linac_gen.elements.ncells import NCells
                def _rf_volt(el):
                    if isinstance(el, NCells):
                        # multi-gap cavity: V₀ = Σ|per-gap EoT·Lc|
                        return _ncells_v0_MV(el)
                    # Lumped RFGap carries a flat ``voltage`` attribute (MV);
                    # FieldMap / FieldMap3D store the raw axial profile, so
                    # we compute V_0 = |ke| · ∫|E_z|·dz / |norm| from the map.
                    # ``voltage`` can be negative on a TraceWin GAP card whose
                    # E0TL was set negative to encode a 180° phase flip — we
                    # take abs here because this plot is "RF amplitude".
                    v = getattr(el, "voltage", None)
                    if v is not None:
                        return abs(float(v))
                    return _fieldmap_vgap_MV(el)
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="RF voltage  —  V₀ [MV] per RF cavity / gap",
                    element_types=(RFGap, FieldMap, FieldMap3D, NCells),
                    attr=None,
                    value_fn=_rf_volt,
                    ylabel="V₀", yunits="MV",
                    color="#fbbf24", type_name="RF",
                )
            elif key == "eacc":
                from linac_gen.elements.field_map import FieldMap
                from linac_gen.elements.field_map_3d import FieldMap3D
                from linac_gen.elements.ncells import NCells
                def _eacc(el):
                    if isinstance(el, NCells):
                        return float(el.eot_v_per_m) * 1e-6   # EoT [MV/m]
                    return _fieldmap_eacc_MV_per_m(el)
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="Peak accelerating gradient  —  E_acc / EoT [MV/m] per cavity",
                    element_types=(FieldMap, FieldMap3D, NCells), attr=None,
                    value_fn=_eacc,
                    ylabel="E_acc", yunits="MV/m",
                    color="#fbbf24", type_name="cavity",
                )
            elif key == "bpeak":
                from linac_gen.elements.field_map import FieldMap
                from linac_gen.elements.solenoid import Solenoid
                def _bpeak(el):
                    # Lumped ``Solenoid`` stores its axis field in
                    # ``el.field`` (attribute name chosen to match the
                    # TraceWin SOLENOID card); magnetic field-map has a
                    # raw on-axis B_z profile we evaluate at peak.  We
                    # abs() the result because TraceWin uses negative
                    # ``field`` / ``kb`` to encode a polarity flip; the
                    # physical *amplitude* is always positive.
                    for a in ("field", "Bz", "B0"):
                        v = getattr(el, a, None)
                        if v is not None:
                            return abs(float(v))
                    return _fieldmap_bpeak_T(el)
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="Peak solenoid field  —  |B_z| [T] per solenoid",
                    element_types=(Solenoid, FieldMap), attr=None,
                    value_fn=_bpeak,
                    ylabel="|B_z|", yunits="T",
                    color="#60a5fa", type_name="solenoid",
                )
            elif key == "int_b2":
                from linac_gen.elements.field_map import FieldMap
                from linac_gen.elements.field_map_3d import FieldMap3D
                from linac_gen.elements.solenoid import Solenoid
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="Solenoid focusing strength  —  ∫B_z²·dz [T²·m] per solenoid",
                    element_types=(Solenoid, FieldMap, FieldMap3D), attr=None,
                    value_fn=_solenoid_int_b2,
                    ylabel="∫B_z²·dz", yunits="T²·m",
                    color="#38bdf8", type_name="solenoid",
                )
            elif key == "floorplan":
                dlg = _FloorPlanPopup(self, self.state)
            elif key == "bend_b":
                from linac_gen.elements.dipole import Dipole
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="Dipole field  —  |B| = Bρ/|ρ| [T] per BEND "
                          "(rigidity: per-element from the run, "
                          "entrance energy before one)",
                    element_types=(Dipole,), attr=None,
                    value_fn=_bend_field_T(self.state),
                    ylabel="|B|", yunits="T",
                    color="#f97316", type_name="dipole (BEND)",
                )
            elif key == "bend_bl":
                from linac_gen.elements.dipole import Dipole
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title="Dipole field integral  —  ∫B·dl = B·L [T·m] per BEND",
                    element_types=(Dipole,), attr=None,
                    value_fn=_bend_field_T(self.state),
                    # B[T] × arc length[mm] → T·m needs L in metres
                    transform=lambda el, v: v * (float(el.length) * 1e-3),
                    ylabel="∫B·dl", yunits="T·m",
                    color="#fb7185", type_name="dipole (BEND)",
                )
            elif key == "sync_phase":
                from linac_gen.elements.rf_gap import RFGap
                from linac_gen.elements.field_map import FieldMap
                from linac_gen.elements.field_map_3d import FieldMap3D
                from linac_gen.elements.ncells import NCells
                dlg = _LatticeParamPopup(
                    self, self.state,
                    title=("Synchronous phase  —  φ_s [deg] per RF element "
                           "(NCELLS P=1: run-resolved phase at gap 1)"),
                    element_types=(RFGap, FieldMap, FieldMap3D, NCells),
                    attr=None, value_fn=_rf_phase_value,
                    ylabel="φ_s", yunits="deg",
                    color="#f472b6", type_name="RF or field-map",
                )
            else:
                QMessageBox.warning(self, "Unknown", f"No popup for {key}")
                return
            self._install_live_preview_toggle(dlg)
            self._popups[key] = dlg
        try:
            dlg.refresh(self.state.results)
        except Exception as exc:
            # Refresh failures should NOT prevent the popup from showing —
            # the user can still close it cleanly and an error message is
            # better than a silently broken click.
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, f"{key} refresh failed",
                f"{type(exc).__name__}: {exc}",
            )
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _refresh(self, results) -> None:
        # Update KPIs
        def _fmt(arr, fmt="{:.3f}"):
            try: return fmt.format(arr[-1])
            except Exception: return "—"
        def _has(x, n=1):
            # Length-aware truthiness usable for BOTH python lists and numpy
            # arrays.  A bare ``if array:`` raises "truth value of an array
            # is ambiguous" — which it does when results are loaded from HDF5
            # (the loader returns ndarrays, whereas a live run yields lists).
            try: return x is not None and len(x) >= n
            except TypeError: return bool(x)
        if results is None:
            for k in (self._k_sx, self._k_sy, self._k_sz, self._k_ex,
                      self._k_loss, self._k_trans):
                kpi_set(k, "—")
        else:
            sx = getattr(results, "sigma_x", [])
            sy = getattr(results, "sigma_y", [])
            sphi = getattr(results, "sigma_phi", [])
            ex = getattr(results, "emit_x", [])
            trans = getattr(results, "transmission", [])
            kpi_set(self._k_sx, _fmt(sx))
            kpi_set(self._k_sy, _fmt(sy))
            # σ_z from σ_φ * β * λ /360
            sz_end = "—"
            try:
                if _has(sphi) and hasattr(results, "ref_beta") and hasattr(results, "ref_frequency"):
                    from linac_gen.core.constants import C_LIGHT
                    wl_mm = C_LIGHT / (results.ref_frequency * 1e6) * 1000.0
                    sz_end = f"{sphi[-1] * results.ref_beta[-1] * wl_mm / 360.0:.3f}"
            except Exception: pass
            kpi_set(self._k_sz, sz_end)
            if _has(ex, 2):
                growth = ex[-1] / max(ex[0], 1e-12)
                kpi_set(self._k_ex, f"{growth:.2f}×")
            if _has(trans):
                kpi_set(self._k_trans, f"{trans[-1]:.2f}")
                kpi_set(self._k_loss, f"{100 - trans[-1]:.2f}")
        # Refresh sparklines + footer values on every result card
        for card in self._cards:
            card.set_results(results)
        # Refresh any open popups.  Isolate each so one popup raising (e.g. a
        # degenerate ensemble seed with an empty transmission array) can't
        # abort the loop and leave every later popup stale for this cycle.
        for d in self._popups.values():
            if not d.isVisible():
                continue
            try:
                d.refresh(results)
            except Exception:   # noqa: BLE001 — best-effort live refresh
                # Keep the loop isolation but preserve observability: before
                # this guard these exceptions reached the excepthook (stderr
                # traceback + dialog); silent swallowing would turn any
                # future popup bug into an invisibly stale plot.
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Live match preview — stream the matcher's current iterate into
    # popups that opted in.  End-of-run refresh (above) is always-on and
    # unchanged; the preview channel is per-popup opt-in (default OFF)
    # and NEVER goes through state.set_results (it must not clobber the
    # committed results the rest of the tab reads).
    # ------------------------------------------------------------------
    def _install_live_preview_toggle(self, dlg) -> None:
        """Insert the per-popup 'live match preview' checkbox (row 0).

        Every popup's root layout is a ``QVBoxLayout(self)`` (verified
        across the file), so a uniform ``insertWidget(0, …)`` works; a
        popup with an exotic layout simply gets no toggle (preview off).
        """
        from PyQt6.QtWidgets import QBoxLayout, QCheckBox
        lay = dlg.layout()
        if not isinstance(lay, QBoxLayout):
            dlg._live_cb = None
            return
        cb = QCheckBox("live match preview")
        cb.setChecked(False)
        cb.setToolTip(
            "While a Matching-tab optimization runs, stream its current "
            "iterate into this plot (~1 update/s).  Off: the plot stays "
            "still during the match.  Either way it refreshes with the "
            "final committed results — end-of-run sync is always on."
        )
        cb.setStyleSheet(
            f"QCheckBox {{ color:{theme.TEXT_1}; font-size:11px; }}")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(cb)
        lay.insertLayout(0, row)
        dlg._live_cb = cb

    def preview_refresh(self, results, iter_idx: int) -> None:
        """Fan a mid-match iterate out to visible, opted-in popups."""
        if results is None:
            return
        for d in self._popups.values():
            try:
                cb = getattr(d, "_live_cb", None)
                if cb is None or not cb.isChecked() or not d.isVisible():
                    continue
                d.refresh(results)
                d.setWindowTitle(
                    f"{getattr(d, '_raw_title', d.windowTitle())}"
                    f"  —  LIVE match iter {int(iter_idx)}")
                d._previewing = True
            except RuntimeError:
                continue        # C++ side deleted — skip, never crash
            except Exception:   # noqa: BLE001 — same policy as _refresh
                import traceback
                traceback.print_exc()

    def end_preview(self, *_args) -> None:
        """Match ended (finished/failed/stopped): restore titles and put
        previewed popups back on the committed results so none lingers
        on a mid-match iterate."""
        for d in self._popups.values():
            try:
                if not getattr(d, "_previewing", False):
                    continue
                d._previewing = False
                raw = getattr(d, "_raw_title", None)
                if raw:
                    d.setWindowTitle(f"{raw}  —  Ctrl+S to save")
                if d.isVisible():
                    d.refresh(self.state.results)
            except RuntimeError:
                continue
            except Exception:   # noqa: BLE001
                import traceback
                traceback.print_exc()
