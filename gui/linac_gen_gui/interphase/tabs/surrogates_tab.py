"""Surrogates tab — train, use and compare ML surrogate elements (M6).

Workflow:

1. Pick a FieldMap3D element from the lattice dropdown.
2. Hit **Train surrogate**: configure samples / epochs / hidden dims
   / param sweep in the modal dialog; training runs in a background
   ``QThread`` with a progress label.  A non-modal **progress
   dialog** with live matplotlib subplots (data-gen progress,
   training loss, val MAPE, per-entry MAPE heatmap) tracks the run.
3. On completion the new surrogate is added to the **Trained
   surrogates** table:
   - **Use** checkbox: register the surrogate in the runtime registry
     (the envelope-mode hook (M3) engages automatically on the next
     envelope run).
   - **Compare** button: run baseline vs surrogate-enabled envelope
     (via :func:`linac_gen.surrogates.compare.compare_envelope`),
     show a summary dialog and offer to save the σ-curves PNG.

**Persistence.**  Weights and ``metadata.json`` are written under
``linac_gen/surrogates/weights/<lattice-hash-16>/<element-name>/``.
On lattice load the tab auto-scans that directory and repopulates
the Trained-surrogates table; clicking **Train** for an element with
cached weights opens a three-way dialog (Load cached / Retrain /
Cancel) so re-running the smoke cycle is a deliberate act.

For full plan + scope see ``docs/plans/surrogates.md``.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase.app_settings import make_settings
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.state import AppState


# ---------------------------------------------------------------------------
def _surrogatable_types():
    """The Element subclasses the GUI currently knows how to surrogate.

    Both ``FieldMap`` (1-D/2-D channels) and ``FieldMap3D`` share the
    ``FieldMapElement`` contract that :class:`SurrogateFieldMap` wraps;
    the difference is only in their internal field-data representation.
    RFQ subclasses (``VaneRFQ``, ``RfqCell``) carry per-instance state
    the surrogate doesn't replicate yet and are excluded.
    """
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D
    return (FieldMap, FieldMap3D)


def _detect_sweep_params(element) -> list[tuple[str, float, str]]:
    """Return ``[(name, current_value, kind)]`` for every sweepable
    knob the element actually carries with a non-trivial value.

    ``kind`` is one of:
      * ``"rel"`` — relative sweep (param * (1 ± rel)); used for the
        amplitude knobs ``ke``/``kb``.
      * ``"abs_deg"`` — absolute degree sweep (param ± Δ°); used for
        ``phase``.

    Heuristics applied (so the sweep matches the element's actual
    degrees of freedom):

    * ``ke`` / ``kb`` — included when present **and** non-zero.  Zero
      means the channel is disabled; sweeping it would produce a
      degenerate window and dilute the training set.
    * ``phase`` — included only if there's an active electric channel
      (``ke != 0``); for a pure-magnetic solenoid (``ke == 0``) the
      .scc has no E-component so phase doesn't enter the dynamics.
    * ``scale`` — **not** in the default sweep set.  It's a global
      multiplier whose default is 1.0 and is rarely tuned per element;
      including it would balloon the input dim with little signal.
    """
    out: list[tuple[str, float, str]] = []
    ke_active = False
    kb_active = False
    if hasattr(element, "ke"):
        ke_val = float(getattr(element, "ke"))
        if abs(ke_val) > 1e-15:
            out.append(("ke", ke_val, "rel"))
            ke_active = True
    if hasattr(element, "kb"):
        kb_val = float(getattr(element, "kb"))
        if abs(kb_val) > 1e-15:
            out.append(("kb", kb_val, "rel"))
            kb_active = True
    if hasattr(element, "phase") and ke_active:
        # Solenoid-only (kb_active, !ke_active) doesn't see phase.
        out.append(("phase", float(element.phase), "abs_deg"))
    # `scale` deliberately omitted — see docstring.
    _ = kb_active   # silence unused-variable lint; kept for future use
    return out


def _adjust_bounds_for_element(lattice, element) -> dict:
    """Return ``{attr_name: (vmin, vmax)}`` for each ADJUST card in the
    loaded lattice that targets ``element``.

    Reuses ``linac_gen.matching.variables`` machinery:

    * ``_PARAM_INDEX_MAP`` -- maps each element class's ``param_idx``
      slot to the actual attribute name (``ke``, ``phase``, ``kb``, ...).
    * ``_resolve_adjust_target`` -- the same name-prefix + index
      resolution the matcher itself uses, so the ADJUST → element
      binding the dialog sees matches what the matcher will see at
      run time.

    Bounds where both ``vmin`` and ``vmax`` are zero are TraceWin's
    "unbounded" convention; those rows are skipped so the dialog
    keeps its legacy default for that attribute.

    Falls back to an empty dict on any error (missing lattice,
    unresolvable target, etc.) so the caller can always use ``.get``
    with a graceful fallback.
    """
    if lattice is None:
        return {}
    try:
        from linac_gen.elements.lattice_commands import Adjust
        from linac_gen.matching.variables import (
            _PARAM_INDEX_MAP, _resolve_adjust_target,
        )
    except Exception:    # noqa: BLE001
        return {}
    param_map = _PARAM_INDEX_MAP.get(type(element), {})
    if not param_map:
        return {}
    out: dict = {}
    for i, cmd in enumerate(lattice.elements):
        if not isinstance(cmd, Adjust):
            continue
        try:
            target = _resolve_adjust_target(cmd, lattice, i)
        except Exception:    # noqa: BLE001
            continue
        if target is not element:
            continue
        attr = param_map.get(int(cmd.param_idx))
        if not attr:
            continue
        # TraceWin's "vmin=0, vmax=0" = unbounded; skip so the legacy
        # default sweep applies for that attribute.
        if cmd.vmin == 0.0 and cmd.vmax == 0.0:
            continue
        lo, hi = sorted((float(cmd.vmin), float(cmd.vmax)))
        out[attr] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
class _TrainDialog(QDialog):
    """Modal: configure training params for the chosen element."""

    def __init__(self, element, lattice=None, ref_template=None,
                 envelope_result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Train surrogate — {element.name}")
        self._element = element

        # ---- Auto-detect w_kin range from a forward envelope pass.
        # The dialog used to default to (2.0, 2.5) MeV regardless of
        # which cavity the user picked — a silent OOD-fallback footgun
        # for downstream elements in cryomodules where the beam has
        # accelerated.  We read the synchronous particle's w_kin from
        # an envelope forward pass at the element's lattice index, and
        # bracket entry/exit values with a 5 % margin.
        #
        # The caller (SurrogatesTab._on_train_clicked) ideally passes a
        # cached envelope_result -- the pass is expensive (~22 s on
        # mebt+hwr) so caching by lattice hash avoids the GUI freeze
        # on repeated dialog opens.  If no cached result is supplied,
        # we fall back to running it inline (slow but always correct).
        # Any failure (no lattice, envelope errors, element not in the
        # live lattice) falls back silently to the legacy (2.0, 2.5)
        # defaults so the dialog never blocks.
        w_lo_default, w_hi_default = 2.0, 2.5
        if lattice is not None:
            try:
                r = envelope_result
                if r is None and ref_template is not None:
                    from linac_gen.tracking.envelope import EnvelopeSolver
                    initial = dict(
                        alpha_x=0.0, beta_x=1.0, emit_x=0.25,
                        alpha_y=0.0, beta_y=1.0, emit_y=0.25,
                        alpha_z=0.0, beta_z=1.0, emit_z=0.30,
                    )
                    r = EnvelopeSolver(lattice, ref_template,
                                       initial=initial, current=0.0).run()
                if r is not None:
                    idx = lattice.elements.index(element)
                    w_entry = float(r.ref_w_kin[idx])
                    w_exit  = float(r.ref_w_kin[min(idx + 1,
                                                    len(r.ref_w_kin) - 1)])
                    if w_entry > w_exit:
                        w_entry, w_exit = w_exit, w_entry
                    w_lo_default = max(0.001, w_entry * 0.95)
                    w_hi_default = w_exit * 1.05
            except Exception:
                pass   # keep legacy defaults

        # ---- Auto-detect ADJUST bounds for each tunable parameter.
        # If the loaded lattice declares an ADJUST card for this
        # element's ke / kb / phase, use that range (+25 % margin) as
        # the sweep default instead of the flat ±20 % / ±20 deg fallback.
        # Eliminates the "trained tighter than ADJUST, silent OOD"
        # mistake and the "trained much wider than ADJUST, wasted
        # samples" inefficiency.
        adjust_bounds = (_adjust_bounds_for_element(lattice, element)
                         if lattice is not None else {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self._samples = QSpinBox()
        self._samples.setRange(50, 200_000)
        self._samples.setValue(200)
        self._samples.setSingleStep(50)
        form.addRow("Samples (LHS):", self._samples)

        self._epochs = QSpinBox()
        self._epochs.setRange(5, 5000)
        self._epochs.setValue(40)
        form.addRow("Epochs:", self._epochs)

        self._hidden = QLineEdit("64,64")
        form.addRow("Hidden dims:", self._hidden)

        # CPU workers for data generation (the RK4 fitted_matrix loop
        # dominates wall-clock; this is the only meaningful speedup
        # knob the user has).  Default = total cores - 2 (leave room
        # for the OS + Qt event loop); 1 = serial.
        import os as _os
        default_workers = max(1, (_os.cpu_count() or 2) - 2)
        self._workers = QSpinBox()
        self._workers.setRange(1, 64)
        self._workers.setValue(default_workers)
        form.addRow("Workers (CPU):", self._workers)

        # w_kin lo/hi — auto-detected above when possible.
        w_kin_label_suffix = (" (auto from envelope)"
                              if (w_lo_default, w_hi_default) != (2.0, 2.5)
                              else "")
        self._w_lo = QDoubleSpinBox()
        self._w_lo.setRange(0.001, 10000.0)
        self._w_lo.setDecimals(3)
        self._w_lo.setValue(w_lo_default)
        self._w_lo.setSuffix(" MeV")
        form.addRow(f"w_kin lo{w_kin_label_suffix}:", self._w_lo)

        self._w_hi = QDoubleSpinBox()
        self._w_hi.setRange(0.001, 10000.0)
        self._w_hi.setDecimals(3)
        self._w_hi.setValue(w_hi_default)
        self._w_hi.setSuffix(" MeV")
        form.addRow(f"w_kin hi{w_kin_label_suffix}:", self._w_hi)

        # ---- Dynamic per-element parameter sweeps -------------------
        # _param_widgets[name] = (current_value, kind, QDoubleSpinBox)
        self._param_widgets: dict[str, tuple[float, str, QDoubleSpinBox]] = {}
        for name, current, kind in _detect_sweep_params(element):
            sp = QDoubleSpinBox()
            bounds = adjust_bounds.get(name)
            if bounds is not None:
                lo, hi = bounds
                half_range = max(0.0, (hi - lo) / 2.0) * 1.25   # 25 % margin
            else:
                half_range = None
            if kind == "rel":
                sp.setRange(0.0, 0.5)
                sp.setDecimals(3)
                if half_range is not None and abs(current) > 1e-12:
                    rel = min(0.5, half_range / abs(current))
                    sp.setValue(rel)
                    form.addRow(
                        f"{name} ± rel  (from ADJUST [{bounds[0]:.4g}, "
                        f"{bounds[1]:.4g}]):", sp)
                else:
                    sp.setValue(0.20)
                    form.addRow(f"{name} ± rel  (current={current:.4g}):",
                                sp)
            else:   # abs_deg
                sp.setRange(0.0, 180.0)
                sp.setDecimals(1)
                sp.setSuffix(" deg")
                if half_range is not None:
                    sp.setValue(min(180.0, half_range))
                    form.addRow(
                        f"{name} ±  (from ADJUST [{bounds[0]:.4g}, "
                        f"{bounds[1]:.4g}]):", sp)
                else:
                    sp.setValue(20.0)
                    form.addRow(
                        f"{name} ±  (current={current:+.1f} deg):", sp)
            self._param_widgets[name] = (current, kind, sp)

        info_bits = [f"length={element.length:.0f} mm"]
        if hasattr(element, "frequency"):
            info_bits.append(f"freq={float(element.frequency):.1f} MHz")
        info_bits.append(f"type={type(element).__name__}")
        info = QLabel("   ".join(info_bits))
        info.setStyleSheet(f"color:{theme.TEXT_2}; font-size:10px;")
        layout.addWidget(info)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_options(self) -> dict:
        try:
            hidden = tuple(int(h.strip()) for h in self._hidden.text().split(",")
                           if h.strip())
        except ValueError:
            hidden = (64, 64)
        param_ranges: dict[str, tuple[float, float]] = {}
        for name, (current, kind, sp) in self._param_widgets.items():
            delta = float(sp.value())
            if delta <= 0.0:
                continue   # user disabled this sweep
            if kind == "rel":
                lo, hi = current * (1 - delta), current * (1 + delta)
                if lo > hi:
                    lo, hi = hi, lo
            else:   # abs_deg
                lo, hi = current - delta, current + delta
            param_ranges[name] = (lo, hi)
        return dict(
            n_samples=int(self._samples.value()),
            epochs=int(self._epochs.value()),
            hidden_dims=hidden,
            ref_w_kin_range=(float(self._w_lo.value()),
                              float(self._w_hi.value())),
            param_ranges=param_ranges,
            n_workers=int(self._workers.value()),
        )


# ---------------------------------------------------------------------------
class _BatchTrainDialog(QDialog):
    """Multi-element selector for the batch trainer.

    Shows one row per FieldMap-class element with a checkbox (all
    checked by default).  A single shared sample / epoch / hidden /
    workers row at the top sets the values used for every element in
    the batch.  Per-element w_kin and sweep ranges are still auto-
    detected at worker construction (Patch A applies transparently).
    """

    def __init__(self, elements, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch-train surrogates")
        self._elements = list(elements)

        layout = QVBoxLayout(self)

        # ---- Shared knobs (apply to every checked element) -----------
        shared = QGroupBox("Shared training settings")
        sform = QFormLayout(shared)
        self._samples = QSpinBox()
        self._samples.setRange(50, 200_000)
        self._samples.setValue(5000)        # production-grade default
        self._samples.setSingleStep(500)
        sform.addRow("Samples (LHS):", self._samples)
        self._epochs = QSpinBox()
        self._epochs.setRange(5, 5000)
        self._epochs.setValue(200)
        sform.addRow("Epochs:", self._epochs)
        self._hidden = QLineEdit("128,128")
        sform.addRow("Hidden dims:", self._hidden)
        import os as _os
        self._workers = QSpinBox()
        self._workers.setRange(1, 64)
        self._workers.setValue(max(1, (_os.cpu_count() or 2) - 2))
        sform.addRow("Workers (CPU):", self._workers)
        layout.addWidget(shared)

        # ---- Element checkboxes --------------------------------------
        pick = QGroupBox(f"Elements ({len(elements)} candidates)")
        plist = QVBoxLayout(pick)
        self._boxes: list[QCheckBox] = []
        for el in self._elements:
            label = (f"{el.name}   ({type(el).__name__}, "
                     f"L={getattr(el, 'length', 0):.0f} mm)")
            cb = QCheckBox(label)
            cb.setChecked(True)
            plist.addWidget(cb)
            self._boxes.append(cb)
        # All / None convenience buttons.
        row = QHBoxLayout()
        b_all = QPushButton("Select all")
        b_all.clicked.connect(lambda: [cb.setChecked(True)
                                       for cb in self._boxes])
        row.addWidget(b_all)
        b_none = QPushButton("Select none")
        b_none.clicked.connect(lambda: [cb.setChecked(False)
                                        for cb in self._boxes])
        row.addWidget(b_none)
        row.addStretch(1)
        plist.addLayout(row)
        layout.addWidget(pick)

        # ---- OK / Cancel ---------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self):
        """Return ``(selected_elements, shared_opts_dict)``."""
        selected = [el for el, cb in zip(self._elements, self._boxes)
                    if cb.isChecked()]
        try:
            hidden = tuple(int(h.strip())
                           for h in self._hidden.text().split(",")
                           if h.strip())
        except ValueError:
            hidden = (128, 128)
        shared = dict(
            n_samples=int(self._samples.value()),
            epochs=int(self._epochs.value()),
            hidden_dims=hidden,
            n_workers=int(self._workers.value()),
        )
        return selected, shared


class _BatchTrainController:
    """Serialise ``_TrainWorker`` runs across multiple elements.

    One element at a time; cancel stops the current element and skips
    the rest.  Owned by the tab so the live ``_TrainProgressDialog``
    can still update as each element trains.
    """

    def __init__(self, *, elements, ref_template, lattice, lattice_hash,
                 shared_opts, tab):
        self._elements = list(elements)
        self._ref_template = ref_template
        self._lattice = lattice
        self._lattice_hash = lattice_hash
        self._shared_opts = shared_opts
        self._tab = tab
        self._idx = 0
        self._cancelled = False
        self._worker = None
        self._dlg = None

    def start(self) -> None:
        self._tab._status.setText(
            f"Batch training: 0 / {len(self._elements)} elements done")
        self._train_next()

    def cancel(self) -> None:
        self._cancelled = True
        if self._worker is not None and self._worker.isRunning():
            # Real cooperative stop: the core polls should_stop per
            # sample / per epoch and raises OperationCancelled; the
            # worker then emits `cancelled` and _train_next ends the
            # batch (self._cancelled is already set).
            try:
                self._worker.request_stop()
                self._worker.requestInterruption()
            except Exception:    # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def _train_next(self) -> None:
        if self._cancelled or self._idx >= len(self._elements):
            done = self._idx
            self._tab._status.setText(
                f"Batch training: {done} / {len(self._elements)} "
                f"elements done"
                + ("  (cancelled)" if self._cancelled else "")
            )
            self._tab._train_btn.setEnabled(True)
            self._tab._batch_train_btn.setEnabled(True)
            return

        element = self._elements[self._idx]
        # Per-element auto-detection (Patch A) -- bounds + w_kin
        # ranges are derived from this element's ADJUST card / its
        # position in the lattice.  Reuse the helpers directly so we
        # don't have to spin up another _TrainDialog modally.
        per_param = _adjust_bounds_for_element(self._lattice, element)
        param_ranges: dict = {}
        for name, current, kind in _detect_sweep_params(element):
            bounds = per_param.get(name)
            if bounds is not None:
                lo, hi = bounds
                half_range = max(0.0, (hi - lo) / 2.0) * 1.25
            else:
                half_range = None
            if kind == "rel":
                if half_range is not None and abs(current) > 1e-12:
                    rel = min(0.5, half_range / abs(current))
                else:
                    rel = 0.20
                if rel <= 0.0:
                    continue
                lo_v = current * (1.0 - rel)
                hi_v = current * (1.0 + rel)
                if lo_v > hi_v:
                    lo_v, hi_v = hi_v, lo_v
            else:   # abs_deg
                delta = (half_range if half_range is not None
                         else 20.0)
                if delta <= 0.0:
                    continue
                lo_v = current - delta
                hi_v = current + delta
            param_ranges[name] = (lo_v, hi_v)

        # w_kin range from envelope forward pass (Patch A1 logic).
        # Reuses the tab's cached envelope result so the batch trainer
        # only pays the ~22 s envelope cost ONCE for the whole batch,
        # not 16x.  Cache is populated either by an earlier dialog
        # open OR by this call itself.
        w_lo, w_hi = 2.0, 2.5
        try:
            r = self._tab._get_or_compute_envelope(self._ref_template)
            if r is not None:
                idx = self._lattice.elements.index(element)
                we = float(r.ref_w_kin[idx])
                wx = float(r.ref_w_kin[min(idx + 1,
                                           len(r.ref_w_kin) - 1)])
                if we > wx:
                    we, wx = wx, we
                w_lo = max(0.001, we * 0.95)
                w_hi = wx * 1.05
        except Exception:    # noqa: BLE001
            pass

        opts = dict(self._shared_opts)
        opts["param_ranges"] = param_ranges
        opts["ref_w_kin_range"] = (w_lo, w_hi)

        out_dir = self._tab._weights_dir_for_element(
            element.name, self._lattice_hash)
        self._worker = _TrainWorker(
            element=element,
            ref_template=self._ref_template,
            out_dir=out_dir,
            lattice_hash=self._lattice_hash,
            options=opts,
            parent=self._tab,
        )

        # Live progress dialog -- same one used by the per-element
        # train flow.  Close the previous dialog (if any) so we don't
        # leak windows.  Clear its cancel hook first: this close is
        # programmatic housekeeping, not a user cancel.
        if self._dlg is not None:
            try:
                self._dlg.cancel_cb = None
                self._dlg.close()
            except Exception:    # noqa: BLE001
                pass
        self._dlg = _TrainProgressDialog(
            element_name=element.name,
            n_samples=int(opts["n_samples"]),
            epochs=int(opts["epochs"]),
            parent=self._tab,
        )
        self._dlg.setWindowTitle(
            f"Batch training [{self._idx + 1}/{len(self._elements)}] -- "
            f"{element.name}")
        # Closing the live dialog IS the batch cancel gesture.
        self._dlg.cancel_cb = self.cancel
        self._dlg.show()
        self._worker.progress.connect(self._dlg.on_progress)
        self._worker.finished_ok.connect(self._on_element_done)
        self._worker.failed.connect(self._on_element_failed)
        self._worker.cancelled.connect(self._on_element_cancelled)
        self._tab._train_btn.setEnabled(False)
        self._tab._batch_train_btn.setEnabled(False)
        self._tab._status.setText(
            f"Batch training [{self._idx + 1}/{len(self._elements)}]: "
            f"{element.name}")
        self._worker.start()

    def _on_element_done(self, meta) -> None:
        # Capture the trained surrogate into the tab's table just like
        # the single-element flow does.
        try:
            from linac_gen.surrogates.base import SurrogateFieldMap
            from linac_gen.surrogates.training import load_surrogate
            element = self._elements[self._idx]
            out_dir = self._tab._weights_dir_for_element(
                element.name, self._lattice_hash)
            mlp, meta_loaded = load_surrogate(out_dir)
            surr = SurrogateFieldMap(element, mlp, meta_loaded)
            surr.weights_dir = str(out_dir)     # provenance: weights hash
            # Key by the element-name STRING, exactly as the single-element
            # and cached-load paths do (``meta.element_key`` / ``name``).  A
            # tuple key here made batch-trained rows invisible to every
            # name-based lookup (Use / Compare) and crashed the next
            # ``_refresh_table`` on ``QTableWidgetItem(tuple)`` — an
            # exception that then escaped a later single-train's finished slot.
            key = element.name
            self._tab._trained[key] = (surr, out_dir, meta_loaded)
            # Same retrain-while-engaged rule as the single-element path:
            # replace a live registration with the fresh weights, else the
            # registry keeps serving the pre-retrain surrogate behind a
            # ticked Use box.
            from linac_gen.surrogates import registry as _reg
            if _reg.get(meta_loaded.lattice_hash, key) is not None:
                _reg.register(surr)
            self._tab._refresh_table()
        except Exception as exc:    # noqa: BLE001
            print(f"[batch-train] post-train load failed for "
                  f"{self._elements[self._idx].name}: "
                  f"{type(exc).__name__}: {exc}")
        self._idx += 1
        self._train_next()

    def _on_element_failed(self, message: str) -> None:
        QMessageBox.warning(
            self._tab,
            f"Training failed for {self._elements[self._idx].name}",
            message,
        )
        # Skip this element and move on (don't abort the whole batch).
        self._idx += 1
        self._train_next()

    def _on_element_cancelled(self) -> None:
        # User-initiated stop — no modal warning per element; cancel()
        # already set _cancelled, so _train_next ends the batch and
        # re-enables the buttons.
        self._idx += 1
        self._train_next()


# ---------------------------------------------------------------------------
class _TrainWorker(QThread):
    """Background trainer; emits progress / finished / failed / cancelled."""

    finished_ok = pyqtSignal(object)   # SurrogateMetadata
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()           # user stop — NOT a failure
    progress = pyqtSignal(dict)        # {"stage": "data_gen"|"epoch", ...}

    def __init__(self, element, ref_template, out_dir, lattice_hash,
                 options: dict, parent=None):
        super().__init__(parent)
        self._element = element
        self._ref_template = ref_template
        self._out_dir = out_dir
        self._lattice_hash = lattice_hash
        self._options = options
        self.out_dir = out_dir   # exposed for the receiver
        # Cooperative stop — threading.Event like the envelope/MP
        # workers; polled per sample / per epoch inside the core.
        import threading
        self._stop_event = threading.Event()
        # See `_MpCompareWorker.__init__` for the rationale.  Training
        # itself doesn't directly call LAPACK, but `train_surrogate_for_element`
        # eventually triggers `fitted_matrix(ref)` (via the RK4
        # ground-truth generator) which uses NumPy linalg internally;
        # the same OpenBLAS / stack-overflow risk applies on macOS.
        self.setStackSize(16 * 1024 * 1024)

    def request_stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            from linac_gen.core.cancelled import OperationCancelled
            from linac_gen.surrogates.training import train_surrogate_for_element
            # Bridge the library's progress_callback to a Qt signal.
            # The dialog drives its own paint cadence via a QTimer
            # (see `_TrainProgressDialog`), so we don't need to yield
            # here -- signals just queue up and the timer drains them
            # at ~20 Hz regardless of the worker's emit rate.
            def _cb(info: dict) -> None:
                self.progress.emit(info)
            try:
                _, meta = train_surrogate_for_element(
                    element=self._element,
                    ref_template=self._ref_template,
                    out_dir=self._out_dir,
                    lattice_hash=self._lattice_hash,
                    element_key=self._element.name,
                    verbose=False,
                    progress_callback=_cb,
                    should_stop=self._stopping,
                    **self._options,
                )
            except OperationCancelled:
                # No partial weights exist — save runs after both stages.
                self.cancelled.emit()
                return
            self.finished_ok.emit(meta)
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(repr(exc))


# ---------------------------------------------------------------------------
def _apply_dark_theme(fig, axes) -> None:
    """Recolour a matplotlib Figure + axes to match the Interphase dark theme.

    Matplotlib's default is a white-bg, blue-line look that fights with
    the rest of the app.  Set face / spine / text colours from theme.py.
    """
    fig.patch.set_facecolor(theme.BG_1)
    for ax in axes:
        ax.set_facecolor(theme.BG_INSET)
        for sp in ax.spines.values():
            sp.set_color(theme.BORDER_0)
        ax.tick_params(colors=theme.TEXT_2, which="both")
        ax.xaxis.label.set_color(theme.TEXT_2)
        ax.yaxis.label.set_color(theme.TEXT_2)
        ax.title.set_color(theme.TEXT_0)
        ax.grid(color=theme.BORDER_0, alpha=0.35, linestyle=":")


class _TrainProgressDialog(QDialog):
    """Non-modal live-plot dialog for the training run.

    The 2x2 figure layout (Interphase dark-themed):
      * (0,0)  Data-gen progress -- bar (done / total) + sample rate.
               This is the only panel active during the data-gen phase
               (which is usually 90 %+ of total wall-clock); the
               training panels show "waiting for training" placeholder
               text until the epoch loop kicks in.
      * (0,1)  Training loss vs epoch (log y).
      * (1,0)  Validation MAPE vs epoch (log y) + running best.
      * (1,1)  Per-entry val MAPE heatmap (6x6) -- which matrix
               entries the MLP struggles with most.

    A stage strip at the top shows the overall progress
    (Data-gen -> Training -> Done).  Redraws are throttled to ~10 Hz.
    """

    def __init__(self, element_name: str, n_samples: int,
                 epochs: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Training progress -- {element_name}")
        self.resize(900, 680)
        self.setModal(False)
        # Optional zero-arg callable fired when the USER closes the
        # dialog — the owner wires it to the training cancel (batch or
        # single).  Cleared before programmatic closes.
        self.cancel_cb = None
        # Match the panel background so the dialog frame blends in.
        self.setStyleSheet(f"QDialog {{ background: {theme.BG_0}; }}")

        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas)
        from matplotlib.figure import Figure

        self._n_samples = max(1, int(n_samples))
        self._epochs = max(1, int(epochs))
        self._element_name = element_name

        # Buffers.
        self._epoch_xs: list[int] = []
        self._train_loss: list[float] = []
        self._val_mape: list[float] = []
        self._best_val_mape: list[float] = []
        self._data_done = 0
        self._data_elapsed = 0.0
        self._per_entry_heatmap = None
        self._training_started = False
        # Paint throttle: the worker connects with a
        # BlockingQueuedConnection, so every emit blocks the worker until
        # this slot returns.  We cap the *expensive* canvas paint to ~25 Hz
        # so a fast, high-frequency data-gen phase doesn't stall the worker
        # on a full redraw per sample (see on_progress).
        self._last_paint = 0.0
        self._last_stage = None

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        # ---- Stage strip --------------------------------------------
        self._stage_label = QLabel(
            "<b>Stage:</b> "
            f"<span style='color:{theme.ACCENT};'>"
            "&#x25CF; Data generation</span>"
            f"&nbsp;&nbsp;<span style='color:{theme.TEXT_DIM};'>"
            "&#x25CB; Training</span>"
            f"&nbsp;&nbsp;<span style='color:{theme.TEXT_DIM};'>"
            "&#x25CB; Done</span>")
        self._stage_label.setStyleSheet(
            f"color:{theme.TEXT_1}; "
            f"font-family:{theme.FONT_SANS}; font-size:12px; "
            f"padding:4px 0;")
        v.addWidget(self._stage_label)

        # ---- Status line --------------------------------------------
        self._status = QLabel("Waiting for first progress sample...")
        self._status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; "
            f"font-size:11px; padding:2px 0;")
        v.addWidget(self._status)

        # ---- Figure -------------------------------------------------
        # Larger top-row data-gen panel because that's where the
        # interesting action is most of the time.
        self._fig = Figure(figsize=(8.5, 6.0), constrained_layout=True)
        self._canvas = FigureCanvas(self._fig)
        v.addWidget(self._canvas, stretch=1)

        gs = self._fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
        self._ax_data = self._fig.add_subplot(gs[0, 0])
        self._ax_loss = self._fig.add_subplot(gs[0, 1])
        self._ax_mape = self._fig.add_subplot(gs[1, 0])
        self._ax_heat = self._fig.add_subplot(gs[1, 1])

        # Apply theme to figure + axes BEFORE adding artists so colours
        # the axes are styled from the start.
        _apply_dark_theme(
            self._fig,
            [self._ax_data, self._ax_loss, self._ax_mape, self._ax_heat],
        )

        # ---- Data-gen subplot --------------------------------------
        self._ax_data.set_title("Data generation (RK4 samples)", fontsize=10)
        self._ax_data.set_xlabel("samples")
        self._ax_data.set_xlim(0, self._n_samples)
        self._ax_data.set_ylim(-0.5, 0.5)
        self._ax_data.set_yticks([])
        # Subdued background bar showing capacity.
        self._ax_data.barh(
            [0], [self._n_samples], color=theme.BG_2, height=0.8,
            zorder=1)
        self._data_bar = self._ax_data.barh(
            [0], [0], color=theme.ACCENT, height=0.8, zorder=2)
        self._data_text = self._ax_data.text(
            self._n_samples / 2, 0, f"0 / {self._n_samples}",
            va="center", ha="center",
            color=theme.TEXT_0, fontsize=11, fontweight="bold",
            zorder=3)

        # ---- Train loss subplot ------------------------------------
        self._ax_loss.set_title("MLP train loss", fontsize=10)
        self._ax_loss.set_xlabel("epoch")
        self._ax_loss.set_ylabel("loss (MSE)")
        self._ax_loss.set_yscale("log")
        self._loss_line, = self._ax_loss.plot(
            [], [], color=theme.PLOT_PALETTE[0], lw=1.5)

        # ---- Val MAPE subplot --------------------------------------
        self._ax_mape.set_title("Validation MAPE", fontsize=10)
        self._ax_mape.set_xlabel("epoch")
        self._ax_mape.set_ylabel("MAPE")
        self._ax_mape.set_yscale("log")
        self._mape_line, = self._ax_mape.plot(
            [], [], color=theme.PLOT_PALETTE[1], lw=1.5, label="val")
        self._best_line, = self._ax_mape.plot(
            [], [], color=theme.PLOT_PALETTE[2], lw=1.0, linestyle="--",
            label="best so far")
        leg = self._ax_mape.legend(fontsize=8, loc="upper right",
                                    facecolor=theme.BG_1,
                                    edgecolor=theme.BORDER_0,
                                    labelcolor=theme.TEXT_1)
        for text in leg.get_texts():
            text.set_color(theme.TEXT_1)

        # ---- Heatmap subplot ---------------------------------------
        self._ax_heat.set_title("Per-entry val MAPE (6x6)", fontsize=10)
        self._ax_heat.set_xticks(range(6))
        self._ax_heat.set_yticks(range(6))
        self._ax_heat.set_xticklabels(
            ["x", "x'", "y", "y'", "$\\phi$", "W"], fontsize=8,
            color=theme.TEXT_2)
        self._ax_heat.set_yticklabels(
            ["x", "x'", "y", "y'", "$\\phi$", "W"], fontsize=8,
            color=theme.TEXT_2)
        import numpy as _np
        # "magma" sits well on dark backgrounds; vmin=0, vmax adapts.
        self._heatmap_img = self._ax_heat.imshow(
            _np.zeros((6, 6)), cmap="magma", aspect="equal",
            vmin=0, vmax=0.1)
        self._heatmap_cbar = self._fig.colorbar(
            self._heatmap_img, ax=self._ax_heat,
            fraction=0.046, pad=0.04)
        self._heatmap_cbar.outline.set_edgecolor(theme.BORDER_0)
        self._heatmap_cbar.ax.yaxis.set_tick_params(color=theme.TEXT_2)
        for lbl in self._heatmap_cbar.ax.yaxis.get_ticklabels():
            lbl.set_color(theme.TEXT_2)

        # ---- "Waiting for training" placeholders ------------------
        # Drawn into each training subplot so the user knows the
        # empty axes are intentional during the data-gen phase.
        self._placeholders = [
            ax.text(
                0.5, 0.5,
                "Waiting for training to start\n"
                "(data generation in progress)",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=10, color=theme.TEXT_DIM,
                style="italic",
            )
            for ax in (self._ax_loss, self._ax_mape, self._ax_heat)
        ]

        self._canvas.draw()

    # ----------------------------------------------------------------
    def _set_stage(self, current: str) -> None:
        """Update the stage strip; current is 'data', 'train', or 'done'."""
        on = theme.ACCENT
        off = theme.TEXT_DIM
        done = theme.OK
        stages = [("data", "Data generation"),
                  ("train", "Training"),
                  ("done", "Done")]
        order = [s for s, _ in stages]
        ci = order.index(current)
        parts = []
        for i, (sid, lbl) in enumerate(stages):
            if i < ci:
                col = done
                glyph = "&#x25CF;"
            elif i == ci:
                col = on
                glyph = "&#x25CF;"
            else:
                col = off
                glyph = "&#x25CB;"
            parts.append(
                f"<span style='color:{col};'>{glyph}&nbsp;{lbl}</span>")
        self._stage_label.setText(
            "<b>Stage:</b> &nbsp;" + "&nbsp;&nbsp;&rarr;&nbsp;&nbsp;".join(parts))

    def _clear_placeholders(self) -> None:
        """Remove the 'waiting for training' overlays — call once on
        the first epoch event."""
        for ph in self._placeholders:
            ph.remove()
        self._placeholders = []

    def closeEvent(self, ev) -> None:
        """Closing the live dialog is the cancel gesture for the run it
        monitors (the old behavior — dialog gone, training silently
        continues with no way to stop it — was the bug)."""
        cb = self.cancel_cb
        if cb is not None:
            try:
                cb()
            except Exception:    # noqa: BLE001
                pass
        super().closeEvent(ev)

    def on_progress(self, info: dict) -> None:
        """Slot: ingest a progress event AND repaint synchronously.

        The connection from `_TrainWorker.progress` is set up as
        ``Qt.ConnectionType.BlockingQueuedConnection`` in the tab so
        the worker emits, then BLOCKS waiting for this slot to fully
        return.  Inside the slot we:

        1. Ingest the event into our buffers.
        2. Update every matplotlib artist that the event affects.
        3. Force a synchronous canvas paint (``canvas.draw`` +
           ``canvas.repaint`` + ``QApplication.processEvents``) so
           the user actually SEES the new frame before the worker
           continues to the next epoch.

        Why blocking instead of a QTimer-driven repaint?  At the GUI
        smoke defaults (200 samples, 40 epochs, hidden=64,64) the
        training loop takes ~430 ms total -- about 1-2 ms per
        epoch.  Any timer-based redraw at 10-20 Hz fires only a
        handful of times in that window, so the user sees the
        training panels jump from empty to fully-populated and
        perceives "all updates happen at the end".  Blocking the
        worker per emit throttles training to the canvas paint cost
        (~30 ms / frame), turning 40 epochs into ~1.2 s of visibly
        animated training -- exactly the live feedback the user
        wants.  For production cycles (50 000 samples / 300 epochs
        at ~100 ms / epoch), the per-emit block is invisible
        overhead.
        """
        stage = info.get("stage")
        if stage == "data_gen":
            self._data_done = int(info["done"])
            self._data_elapsed = float(info["elapsed_s"])
            if self._data_done <= 0:
                # Initial tick: workers are spawning, no samples done yet.
                self._status.setText(
                    f"[data-gen] starting — spinning up workers "
                    f"for {self._n_samples} samples…")
            else:
                rate = (self._data_done / self._data_elapsed
                        if self._data_elapsed > 0 else 0.0)
                eta = ((self._n_samples - self._data_done) / rate
                       if rate > 0 else float("nan"))
                self._status.setText(
                    f"[data-gen] {self._data_done}/{self._n_samples}  "
                    f"({self._data_done / self._n_samples * 100:.1f}%)  "
                    f"{rate:.1f} samples/s  "
                    f"elapsed {self._data_elapsed:.1f} s  "
                    f"ETA {eta:.0f} s")
            self._set_stage("data")
        elif stage == "epoch":
            if not self._training_started:
                self._training_started = True
                self._clear_placeholders()
            ep = int(info["epoch"])
            self._epoch_xs.append(ep)
            self._train_loss.append(float(info["train_loss"]))
            self._val_mape.append(float(info["val_mape"]))
            self._best_val_mape.append(float(info["best_val_mape"]))
            self._per_entry_heatmap = info.get("per_entry_val_mape")
            self._status.setText(
                f"[train] epoch {ep}/{self._epochs}  "
                f"train loss {info['train_loss']:.3e}  "
                f"val MAPE {info['val_mape']:.3e}  "
                f"(best {info['best_val_mape']:.3e})  "
                f"elapsed {info['elapsed_s']:.1f} s")
            self._set_stage("train" if ep < self._epochs else "done")

        # ---- Artist updates ---------------------------------------
        self._data_bar[0].set_width(self._data_done)
        self._data_text.set_x(self._n_samples / 2)
        self._data_text.set_text(
            f"{self._data_done} / {self._n_samples}")

        if self._epoch_xs:
            self._loss_line.set_data(self._epoch_xs, self._train_loss)
            self._mape_line.set_data(self._epoch_xs, self._val_mape)
            self._best_line.set_data(self._epoch_xs, self._best_val_mape)
            for ax, ys in [
                (self._ax_loss, list(self._train_loss)),
                (self._ax_mape,
                 list(self._val_mape) + list(self._best_val_mape)),
            ]:
                if ys:
                    lo, hi = float(min(ys)), float(max(ys))
                    if lo == hi:
                        lo = lo * 0.5
                        hi = hi * 1.5 if hi > 0 else 1.0
                    ax.set_xlim(0, max(self._epoch_xs) + 1)
                    ax.set_ylim(max(lo * 0.5, 1e-12), hi * 2.0)

        if self._per_entry_heatmap is not None:
            arr = self._per_entry_heatmap
            self._heatmap_img.set_data(arr)
            self._heatmap_img.set_clim(
                vmin=0.0, vmax=max(float(arr.max()), 1e-6))
            self._heatmap_cbar.update_normal(self._heatmap_img)

        # ---- Synchronous paint (throttled for data-gen only) ------
        # The cheap artist updates above run every event; the EXPENSIVE
        # trio below (draw -> agg buffer, repaint -> push to screen,
        # processEvents -> drain) costs ~10-30 ms and, under the
        # BlockingQueuedConnection, stalls the worker for that long on
        # EVERY emit.
        #
        # Training ("epoch") emits are deliberately painted every time:
        # that per-epoch block IS the live training animation (esp. the
        # fast smoke loop at ~1-2 ms/epoch, which would otherwise jump
        # from empty to full).  Data generation, by contrast, fires
        # progress every ~1% of samples -- painting per emit froze the
        # bar in choppy, worker-blocking bursts -- so we throttle the
        # real paint there to ~25 Hz.  Always paint the terminal event
        # (done == total) and any stage change (data_gen -> training)
        # so the final frame and the hand-off are never dropped.
        import time
        now = time.monotonic()
        done = info.get("done")
        total = info.get("total")
        terminal = (done is not None and total is not None and done >= total)
        stage_changed = (stage != self._last_stage)
        self._last_stage = stage
        paint = (stage == "epoch" or terminal or stage_changed
                 or (now - self._last_paint) >= 0.04)
        if paint:
            self._canvas.draw()
            self._canvas.repaint()
            QApplication.processEvents()
            self._last_paint = now


# ---------------------------------------------------------------------------
class SurrogatesTab(QWidget):
    """Train, register and compare ML surrogate elements."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._worker: _TrainWorker | None = None
        self._progress_dlg: _TrainProgressDialog | None = None
        # element_name -> (SurrogateFieldMap, weights_dir, metadata)
        self._trained: dict[str, tuple[object, Path, object]] = {}
        # Cache for the auto-detect envelope forward pass keyed on
        # lattice hash.  The first dialog open on a given lattice pays
        # the full envelope-pass cost (~22 s on mebt+hwr); subsequent
        # opens reuse the cached `EnvelopeResults` and the dialog
        # constructs instantly.  Invalidated on every
        # `lattice_changed` signal.
        self._envelope_cache: dict[str, object] = {}

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        hint = QLabel(
            "Train ML surrogates for field-map elements (1-D FieldMap "
            "or 3-D FieldMap3D) in the loaded lattice.  Tick 'Use' to "
            "engage the surrogate in envelope-mode runs; 'Compare' "
            "runs the envelope twice (baseline vs surrogate) and shows "
            "the diff plot."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:12px; padding:4px 2px;")
        v.addWidget(hint)

        # --- element picker + train button ----------------------------
        pick_box = QGroupBox("Train a new surrogate")
        pick = QHBoxLayout(pick_box)
        pick.addWidget(QLabel("Element:"))
        self._elem_combo = QComboBox()
        self._elem_combo.setMinimumWidth(300)
        pick.addWidget(self._elem_combo, 1)
        self._train_btn = QPushButton("  Train surrogate")
        self._train_btn.setIcon(icon("play", 12, "#00161c"))
        self._train_btn.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; "
            f"border-radius:3px; padding:6px 14px; font-weight:600;")
        self._train_btn.clicked.connect(self._on_train_clicked)
        pick.addWidget(self._train_btn)
        # Batch-train every FieldMap-class element in the lattice.
        # Saves the user from 8x clicking through the per-element
        # dialog for a full cryomodule.  Auto-detection from Patch A
        # applies per element (each gets its own w_kin range + ADJUST
        # sweep bounds).  Workers run sequentially -- training is
        # already multi-process via the data-gen pool, so parallelising
        # *across* elements would oversubscribe.
        self._batch_train_btn = QPushButton("  Train all FieldMap*")
        self._batch_train_btn.setStyleSheet(
            f"background:{theme.BG_2}; color:{theme.TEXT_0}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:3px; "
            f"padding:6px 14px;")
        self._batch_train_btn.setToolTip(
            "Train surrogates for every FieldMap / FieldMap3D / RFGap "
            "element in the lattice in one go.  Per-element auto-"
            "detected w_kin and ADJUST sweep ranges still apply.  "
            "Workers run sequentially -- inner data-gen is already "
            "multi-process."
        )
        self._batch_train_btn.clicked.connect(self._on_batch_train_clicked)
        pick.addWidget(self._batch_train_btn)
        v.addWidget(pick_box)

        # --- progress -------------------------------------------------
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate
        self._progress_bar.setVisible(False)
        v.addWidget(self._progress_label)
        v.addWidget(self._progress_bar)

        # --- trained-surrogates table --------------------------------
        v.addWidget(QLabel("Trained surrogates"))

        # Small toolbar above the table: bulk-tick / untick all Use
        # checkboxes.  On a 20-element lattice like MEBT+HWR clicking
        # 20 boxes individually is tedious; these two buttons just
        # iterate the rows and call `setChecked(...)` on every Use
        # checkbox, which fires the existing `_on_use_toggled` per
        # row to register / unregister each surrogate.
        bulk_row = QHBoxLayout()
        bulk_row.setContentsMargins(0, 0, 0, 0)
        self._select_all_btn = QPushButton("Select all")
        self._select_all_btn.setToolTip(
            "Tick the Use checkbox on every trained surrogate -- "
            "engages all of them in the next envelope / MP run.")
        self._select_all_btn.clicked.connect(
            lambda: self._set_all_use(True))
        self._deselect_all_btn = QPushButton("Deselect all")
        self._deselect_all_btn.setToolTip(
            "Untick every Use checkbox -- removes all surrogates "
            "from the registry; next run reverts to baseline RK4.")
        self._deselect_all_btn.clicked.connect(
            lambda: self._set_all_use(False))
        bulk_row.addWidget(self._select_all_btn)
        bulk_row.addWidget(self._deselect_all_btn)
        bulk_row.addStretch(1)
        v.addLayout(bulk_row)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Element", "Val MAPE", "Scope", "Use", "Action"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(160)
        # No more stretch=1 -- the new collapsible MP section below
        # needs layout room too.  Table keeps its setMinimumHeight so
        # it stays usable.
        v.addWidget(self._table)

        # ---- M7: Multi-particle surrogates (collapsible) -----------
        # Opt-in MP-mode engagement of the SAME trained surrogates.
        # Collapsed by default so the existing envelope workflow stays
        # the visual focus.  See `_build_mp_section` for the layout.
        self._mp_section = self._build_mp_section()
        v.addWidget(self._mp_section)

        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:10px; "
            f"font-family:{theme.FONT_MONO};")
        v.addWidget(self._status)

        state.lattice_changed.connect(self._on_lattice_changed)
        # Populate now in case a lattice is already loaded.
        self._on_lattice_changed(state.lattice)

        # Restore persisted MP-section UI state (master toggle,
        # substeps, open/closed) -- done after the rest of __init__
        # so child widgets exist.
        self._restore_mp_section_state()

    # ----------------------------------------------------------------
    def _lattice_hash(self) -> str:
        """Hash of the currently-loaded lattice file (or ``'unknown'``)."""
        from linac_gen.surrogates import registry as _reg
        lp = getattr(self.state, "lattice_path", None)
        if not lp:
            return "unknown"
        try:
            return _reg.hash_lattice_file(lp)
        except Exception:
            return "unknown"

    def _weights_root(self, lattice_hash: str | None = None) -> Path:
        """Per-lattice weights directory (created on demand by training)."""
        lh = lattice_hash if lattice_hash is not None else self._lattice_hash()
        return (Path("linac_gen/surrogates/weights")
                / (lh[:16] if lh != "unknown" else "unknown"))

    def _weights_dir_for_element(self, element_name: str,
                                  lattice_hash: str | None = None) -> Path:
        return self._weights_root(lattice_hash) / element_name

    def _get_or_compute_envelope(self, ref_template):
        """Return cached or freshly-computed envelope result for the
        loaded lattice.

        Used by both the per-element train flow and the batch trainer
        to feed `w_kin` auto-detection in the train dialog.  Computing
        is expensive (~22 s on mebt+hwr.dat with all 20 field maps),
        so we cache by lattice hash.  ``_on_lattice_changed`` clears
        the cache so a freshly-loaded lattice is re-computed.

        Returns ``None`` on any failure (no lattice loaded, envelope
        errors, etc.) -- callers must handle that path; the dialog
        falls back to the legacy (2.0, 2.5) w_kin defaults.
        """
        lat = self.state.lattice
        if lat is None or ref_template is None:
            return None
        key = self._lattice_hash()
        cached = self._envelope_cache.get(key)
        if cached is not None:
            return cached
        try:
            from linac_gen.tracking.envelope import EnvelopeSolver
            initial = dict(
                alpha_x=0.0, beta_x=1.0, emit_x=0.25,
                alpha_y=0.0, beta_y=1.0, emit_y=0.25,
                alpha_z=0.0, beta_z=1.0, emit_z=0.30,
            )
            r = EnvelopeSolver(lat, ref_template,
                               initial=initial, current=0.0).run()
            self._envelope_cache[key] = r
            return r
        except Exception:    # noqa: BLE001
            return None

    # ----------------------------------------------------------------
    def _on_lattice_changed(self, lattice) -> None:
        self._elem_combo.clear()
        # Drop any rows bound to the previous lattice; their Element
        # references are stale and the user shouldn't operate on them.
        self._trained.clear()
        # The cached envelope forward pass (used for auto-detect
        # w_kin in the train dialog) is also bound to the old lattice;
        # invalidate so the next dialog open re-computes for whichever
        # lattice is now loaded.
        self._envelope_cache.clear()
        # CRITICAL: clear the runtime surrogate registry.  The envelope
        # hook's get_by_element_name() lookup is keyed by element name
        # only (not by lattice hash), so a surrogate registered under
        # project A's FMAP_001 would be silently used for project B's
        # FMAP_001 if we left the registry alone -- producing wrong
        # envelope / MP results when the user switches projects without
        # restarting the GUI.  Forcing a registry clear matches the
        # GUI-state clear above (the Use checkboxes are also reset);
        # the user re-ticks Use after loading the new project's
        # surrogates, registering them under the new lattice hash.
        from linac_gen.surrogates import registry as _surr_reg
        _surr_reg.clear()
        self._refresh_table()
        if lattice is None:
            return
        types = _surrogatable_types()
        for i, elem in enumerate(lattice.elements):
            if isinstance(elem, types):
                kind = type(elem).__name__
                self._elem_combo.addItem(
                    f"{i:>3d}: {elem.name}  [{kind}, {elem.length:.0f} mm]", i)
        # Re-populate the table from any cached weights on disk that
        # match elements present in the new lattice.
        self._discover_cached_for_lattice(lattice)

    def _discover_cached_for_lattice(self, lattice) -> None:
        """Scan the per-lattice weights dir and add a row per match."""
        from linac_gen.surrogates.base import SurrogateFieldMap
        from linac_gen.surrogates.training import discover_cached_surrogates

        types = _surrogatable_types()
        fm_by_name: dict[str, object] = {
            e.name: e for e in lattice.elements
            if isinstance(e, types)}
        if not fm_by_name:
            return
        root = self._weights_root()
        try:
            cached = discover_cached_surrogates(
                root, element_names=list(fm_by_name.keys()))
        except Exception as exc:                          # noqa: BLE001
            self._status.setText(f"cache scan failed: {exc!r}")
            return
        if not cached:
            return
        for key, (mlp, meta, dir_path) in cached.items():
            element = fm_by_name.get(key)
            if element is None:
                continue
            surr = SurrogateFieldMap(element, mlp, meta)
            surr.weights_dir = str(dir_path)    # provenance: weights hash
            self._trained[key] = (surr, dir_path, meta)
        self._refresh_table()
        self._status.setText(
            f"auto-loaded {len(cached)} cached surrogate(s) from {root}")

    # ----------------------------------------------------------------
    def _on_batch_train_clicked(self) -> None:
        """Open the multi-element selector + serial trainer.

        Workflow:
        1. Enumerate FieldMap / FieldMap3D / RFGap elements in the
           current lattice.
        2. Show a checkbox list dialog; user picks which to train and
           sets shared sample/epoch/hidden defaults.
        3. Dispatch ``_TrainWorker`` sequentially per element; each
           worker re-uses the per-element auto-detection added in
           Patch A (w_kin from envelope, sweep bounds from ADJUST).
        4. Cancel button stops the running element and skips the rest.
        """
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice",
                                "Load a lattice first.")
            return

        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        from linac_gen.elements.rf_gap import RFGap
        targets = [
            e for e in self.state.lattice.elements
            if isinstance(e, (FieldMap, FieldMap3D, RFGap))
        ]
        if not targets:
            QMessageBox.information(
                self, "No field-map elements",
                "No FieldMap / FieldMap3D / RFGap elements in the "
                "current lattice -- nothing to batch-train."
            )
            return

        dlg = _BatchTrainDialog(targets, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected, shared_opts = dlg.get_selection()
        if not selected:
            return

        from linac_gen.core.particle import H_MINUS
        from linac_gen.core.reference import ReferenceParticle
        ref_template = ReferenceParticle(
            species=H_MINUS, w_kin=2.12, frequency=162.5)

        # Build the controller; it manages a worker per element in
        # sequence and reports progress back to this tab.
        lh = self._lattice_hash()
        controller = _BatchTrainController(
            elements=selected,
            ref_template=ref_template,
            lattice=self.state.lattice,
            lattice_hash=lh,
            shared_opts=shared_opts,
            tab=self,
        )
        controller.start()
        # Hold a reference so Qt doesn't garbage-collect mid-run.
        self._batch_controller = controller

    # ----------------------------------------------------------------
    def _on_train_clicked(self) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice",
                                "Load a lattice first.")
            return
        if self._elem_combo.count() == 0:
            QMessageBox.warning(
                self, "No element",
                "The loaded lattice contains no FieldMap or FieldMap3D "
                "elements (the surrogatable types).")
            return

        elem_idx = self._elem_combo.currentData()
        element = self.state.lattice.elements[elem_idx]

        # ---- Cache check ------------------------------------------
        from linac_gen.surrogates.training import find_cached_surrogate
        lh = self._lattice_hash()
        out_dir = self._weights_dir_for_element(element.name, lh)
        cached = find_cached_surrogate(out_dir)
        if cached is not None:
            mlp_c, meta_c = cached
            choice = QMessageBox.question(
                self, "Cached surrogate found",
                f"A trained surrogate for '{element.name}' already "
                f"exists at\n  {out_dir}\n"
                f"(val MAPE = {meta_c.val_mape:.3e}, "
                f"{meta_c.n_samples} samples × {meta_c.epochs} epochs)\n\n"
                f"Load the cached weights, or retrain from scratch?",
                QMessageBox.StandardButton.Open
                | QMessageBox.StandardButton.Retry
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Open,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                return
            if choice == QMessageBox.StandardButton.Open:
                self._load_cached_into_table(element, mlp_c, meta_c, out_dir)
                return
            # else: Retry => fall through to the normal training path.

        # Build the ref template up front so the dialog can use it for
        # the w_kin / ADJUST auto-detect pass.  Same construction as
        # the worker uses below (single source of truth).
        from linac_gen.core.particle import H_MINUS
        from linac_gen.core.reference import ReferenceParticle
        ref_template = ReferenceParticle(
            species=H_MINUS, w_kin=2.12, frequency=162.5)

        # Look up (or build) the cached envelope forward pass for this
        # lattice.  First click on a fresh lattice pays the full ~22 s
        # cost on mebt+hwr; later clicks reuse the cached result and
        # the dialog opens instantly.  The cache is invalidated in
        # ``_on_lattice_changed`` whenever the loaded lattice changes.
        envelope_result = self._get_or_compute_envelope(ref_template)

        dlg = _TrainDialog(
            element,
            lattice=self.state.lattice,
            ref_template=ref_template,
            envelope_result=envelope_result,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        opts = dlg.get_options()

        # Disable BOTH training entry points: leaving the batch button
        # live allowed a concurrent batch to start next to this run.
        self._train_btn.setEnabled(False)
        self._batch_train_btn.setEnabled(False)
        self._progress_label.setText(
            f"Training '{element.name}' ({opts['n_samples']} samples, "
            f"{opts['epochs']} epochs)...")
        self._progress_bar.setVisible(True)
        self._status.setText("")

        # A previous run's "[done]" dialog may still be open — disarm
        # its cancel hook so closing it later can't cancel THIS run.
        prev_dlg = getattr(self, "_progress_dlg", None)
        if prev_dlg is not None:
            prev_dlg.cancel_cb = None
            # Release the previous run's dialog (a 2×2 Figure + canvas) instead
            # of leaking it for the session; a fresh one is created just below.
            # cancel_cb was nulled first, so closing it can't cancel THIS run.
            try:
                prev_dlg.close()
                prev_dlg.deleteLater()
            except Exception:    # noqa: BLE001
                pass

        self._worker = _TrainWorker(
            element=element,
            ref_template=ref_template,
            out_dir=out_dir,
            lattice_hash=lh,
            options=opts,
            parent=self,
        )
        # Open a non-modal live-plot dialog and wire signals so the
        # user can watch data-gen progress, training loss curves, val
        # MAPE, and the per-entry MAPE heatmap as the QThread runs.
        try:
            self._progress_dlg = _TrainProgressDialog(
                element_name=element.name,
                n_samples=int(opts["n_samples"]),
                epochs=int(opts["epochs"]),
                parent=self,
            )
            # BlockingQueuedConnection: each emit on the worker
            # thread waits for the slot to fully return on the main
            # thread before the worker continues.  This makes the
            # per-epoch paint visible to the user even when the
            # smoke training loop runs in ~1-2 ms per epoch
            # (otherwise paint events pile up and only the last
            # one is visible).  Deadlock-safe here because emitter
            # (worker thread) and receiver (main thread) are
            # different threads and the main thread is in its
            # event loop, not blocked on the worker.
            self._worker.progress.connect(
                self._progress_dlg.on_progress,
                type=Qt.ConnectionType.BlockingQueuedConnection,
            )
            # Closing the live dialog cancels THIS run — bind to the
            # worker instance, not the tab's mutable _worker attribute
            # (which a later run rebinds).
            self._progress_dlg.cancel_cb = (
                lambda w=self._worker: self._cancel_train_worker(w))
            self._progress_dlg.show()
        except Exception as exc:                       # noqa: BLE001
            # If matplotlib / Qt-Agg backend unavailable, fall back
            # to the (existing) progress-bar-only flow gracefully.
            self._progress_dlg = None
            self._status.setText(
                f"(progress popup unavailable: {exc!r})")
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_train_cancelled)
        self._worker.start()

    def _cancel_train_worker(self, w) -> None:
        if isinstance(w, _TrainWorker) and w.isRunning():
            w.request_stop()
            w.requestInterruption()
            self._status.setText(
                "cancelling training — nothing will be saved…")

    def shutdown_begin(self) -> list:
        """App teardown: close progress dialogs FIRST (a worker blocked
        in a BlockingQueuedConnection emit is released when its receiver
        dies — otherwise the window's wait loop would deadlock on it),
        then signal every surrogate worker."""
        # 1. Progress dialogs — programmatic close, not a user cancel.
        for dlg_attr in ("_progress_dlg",):
            dlg = getattr(self, dlg_attr, None)
            if dlg is not None:
                try:
                    dlg.cancel_cb = None
                    dlg.close()
                    dlg.deleteLater()
                except Exception:    # noqa: BLE001
                    pass
        ctl = getattr(self, "_batch_controller", None)
        if ctl is not None:
            try:
                if ctl._dlg is not None:
                    ctl._dlg.cancel_cb = None
                    ctl._dlg.close()
                    ctl._dlg.deleteLater()
                ctl.cancel()
            except Exception:    # noqa: BLE001
                pass
        # 2. Workers.
        out = []
        candidates = [getattr(self, "_worker", None),
                      getattr(self, "_env_compare_worker", None),
                      getattr(self, "_mp_compare_worker", None)]
        if ctl is not None:
            candidates.append(getattr(ctl, "_worker", None))
        for w in candidates:
            if w is not None and hasattr(w, "isRunning") and w.isRunning():
                if hasattr(w, "request_stop"):
                    w.request_stop()
                w.requestInterruption()
                out.append(w)
        return out

    def _load_cached_into_table(self, element, mlp, meta, dir_path) -> None:
        """Add a row from a cached (mlp, meta) without running training."""
        from linac_gen.surrogates.base import SurrogateFieldMap
        surr = SurrogateFieldMap(element, mlp, meta)
        surr.weights_dir = str(dir_path)        # provenance: weights hash
        self._trained[meta.element_key] = (surr, Path(dir_path), meta)
        self._refresh_table()
        self._status.setText(
            f"loaded cached surrogate '{meta.element_key}' "
            f"(val MAPE {meta.val_mape:.3e}) from {dir_path}")

    def _on_finished(self, meta) -> None:
        self._train_btn.setEnabled(True)
        self._batch_train_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        # Flag the progress dialog as done -- leave it open so the
        # user can inspect the final curves; they can close it via
        # the window close button.  Disarm its cancel hook: the run is
        # over, so closing the leftover window must be inert.
        if self._progress_dlg is not None:
            self._progress_dlg.cancel_cb = None
            self._progress_dlg.setWindowTitle(
                self._progress_dlg.windowTitle() + "  [done]")
        from linac_gen.surrogates.base import SurrogateFieldMap
        from linac_gen.surrogates.training import load_surrogate
        out_dir = self._worker.out_dir
        mlp, _ = load_surrogate(out_dir)
        element = next(
            (e for e in self.state.lattice.elements
             if e.name == meta.element_key), None)
        if element is None:
            QMessageBox.warning(
                self, "Element gone",
                f"Element '{meta.element_key}' no longer in the loaded "
                f"lattice; discarding the trained surrogate.")
            return
        surr = SurrogateFieldMap(element, mlp, meta)
        surr.weights_dir = str(out_dir)         # provenance: weights hash
        self._trained[meta.element_key] = (surr, out_dir, meta)
        # If this element's surrogate is currently ENGAGED, replace the
        # registry entry with the fresh weights.  Without this, the rebuilt
        # table shows a ticked Use box (the registry probe finds the OLD
        # registration under the same key) but no stateChanged fires, so the
        # envelope/MP hook would keep serving the pre-retrain surrogate.
        from linac_gen.surrogates import registry as _reg
        if _reg.get(meta.lattice_hash, meta.element_key) is not None:
            _reg.register(surr)
        self._refresh_table()
        self._status.setText(
            f"trained {meta.element_key}: val MAPE {meta.val_mape:.3e}, "
            f"saved {out_dir}")

    def _on_failed(self, msg: str) -> None:
        self._train_btn.setEnabled(True)
        self._batch_train_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        if self._progress_dlg is not None:
            self._progress_dlg.cancel_cb = None
            self._progress_dlg.setWindowTitle(
                self._progress_dlg.windowTitle() + "  [failed]")
        QMessageBox.critical(self, "Training failed", msg)

    def _on_train_cancelled(self) -> None:
        """User stop — not a failure: no modal error, nothing saved."""
        self._train_btn.setEnabled(True)
        self._batch_train_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        if self._progress_dlg is not None:
            self._progress_dlg.cancel_cb = None
            self._progress_dlg.setWindowTitle(
                self._progress_dlg.windowTitle() + "  [cancelled]")
        self._status.setText("training cancelled — nothing was saved")

    # ----------------------------------------------------------------
    def _refresh_table(self) -> None:
        from linac_gen.surrogates import registry as _reg
        self._table.setRowCount(len(self._trained))
        for row, (name, (surr, out_dir, meta)) in enumerate(
                self._trained.items()):
            self._table.setItem(row, 0, QTableWidgetItem(name))
            self._table.setItem(row, 1, QTableWidgetItem(
                f"{meta.val_mape:.2e}"))
            scope = "  ".join(
                f"{n}∈[{lo:.3g},{hi:.3g}]"
                for n, lo, hi in zip(meta.scope.input_names,
                                       meta.scope.input_lo,
                                       meta.scope.input_hi))
            self._table.setItem(row, 2, QTableWidgetItem(scope))
            use_cb = QCheckBox()
            # Reflect the ACTUAL runtime-registry state (set BEFORE connecting
            # ``stateChanged`` so it does not spuriously fire register()).
            # Hard-coding False here let a still-registered surrogate show
            # unchecked after any table rebuild — the envelope/MP hook would
            # then silently engage a surrogate the UI reported as OFF, and
            # "Deselect all" could not clear it (setChecked(False) on an
            # already-unchecked box emits no stateChanged → no unregister).
            use_cb.setChecked(_reg.get(meta.lattice_hash, name) is not None)
            use_cb.stateChanged.connect(
                lambda s, n=name: self._on_use_toggled(
                    n, s == int(Qt.CheckState.Checked.value)))
            self._table.setCellWidget(row, 3, use_cb)
            cmp_btn = QPushButton("Compare")
            cmp_btn.clicked.connect(
                lambda _=False, n=name: self._on_compare_clicked(n))
            self._table.setCellWidget(row, 4, cmp_btn)

    def _on_use_toggled(self, name: str, checked: bool) -> None:
        from linac_gen.surrogates import registry as _reg
        surr, _, meta = self._trained[name]
        if checked:
            _reg.register(surr)
            self._status.setText(f"registered surrogate '{name}'")
        else:
            _reg.unregister(meta.lattice_hash, name)
            self._status.setText(f"unregistered surrogate '{name}'")

    def _set_all_use(self, checked: bool) -> None:
        """Bulk tick / untick every Use checkbox in the Trained-
        surrogates table.

        Drives the per-row `_on_use_toggled` slot (via Qt's
        `stateChanged` signal) so each surrogate is correctly
        register()ed or unregister()ed in the runtime registry.  No
        signal-blocking here -- we WANT the side effects.

        With many rows (e.g. 20 on MEBT+HWR) Qt fires the slot 20
        times in rapid succession; each call updates `self._status`,
        so the final status line ends up showing the last name.
        Override it with a summary at the end.
        """
        if not self._trained:
            self._status.setText("(no trained surrogates to select)")
            return
        for row in range(self._table.rowCount()):
            cb = self._table.cellWidget(row, 3)   # column 3 == "Use"
            if cb is not None:
                cb.setChecked(bool(checked))
        n = len(self._trained)
        action = "registered" if checked else "unregistered"
        self._status.setText(
            f"{action} all {n} surrogate(s)")

    def _on_compare_clicked(self, name: str) -> None:
        if self.state.lattice is None or self.state.beam_config is None:
            QMessageBox.warning(
                self, "Need lattice + beam",
                "Load a lattice and apply a beam config first.")
            return
        if self.state.running:
            QMessageBox.warning(
                self, "Busy",
                "Another run is in progress — compare swaps the global "
                "surrogate registry and cannot overlap a solve.")
            return
        surr, _, meta = self._trained[name]
        from linac_gen.distributions.factory import create_beam
        cfg = self.state.beam_config
        try:
            beam = create_beam(cfg, seed=42)
        except Exception as exc:                      # noqa: BLE001
            QMessageBox.critical(self, "Beam config error", str(exc))
            return
        ref = beam.ref
        from linac_gen.distributions.factory import geometric_emittances
        _ex, _ey, _ez = geometric_emittances(cfg, max(float(ref.bg), 1e-9))
        twiss = dict(
            alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=_ex,
            alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=_ey,
            alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=_ez,
        )
        # Run the two envelope solves in a worker — inline this froze
        # the GUI for the full ~20 s compare.  The running flag excludes
        # toolbar solves while the registry is swapped (and vice versa).
        self.state.set_running(True)
        for b in (self._train_btn, self._batch_train_btn,
                  self._mp_compare_btn):
            b.setEnabled(False)
        self._status.setText(
            f"Comparing '{name}' (baseline vs surrogate envelope)…")
        self._env_compare_worker = _EnvCompareWorker(
            lattice=self.state.lattice, ref=ref, twiss=twiss,
            current=float(cfg.current), surrogate=surr,
            name=name, parent=self,
        )
        self._env_compare_worker.finished_ok.connect(
            self._on_env_compare_finished)
        self._env_compare_worker.failed.connect(self._on_env_compare_failed)
        self._env_compare_worker.cancelled.connect(
            self._on_env_compare_cancelled)
        self._env_compare_worker.start()

    def _restore_after_env_compare(self) -> None:
        self.state.set_running(False)
        for b in (self._train_btn, self._batch_train_btn,
                  self._mp_compare_btn):
            b.setEnabled(True)

    def _on_env_compare_finished(self, name: str, report) -> None:
        self._restore_after_env_compare()
        self._status.setText(
            f"compare '{name}': worst rel.diff "
            f"{report.worst_rel_diff():.2e}")
        # File dialog + matplotlib plotting stay on the GUI thread.
        from linac_gen.surrogates.compare import plot_compare_report
        text = report.summary_text()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save compare PNG",
            f"{name}_compare.png", "PNG (*.png);;All Files (*)")
        if path:
            try:
                plot_compare_report(report, path)
                text += f"\n\nPlot saved: {path}"
            except Exception as exc:                  # noqa: BLE001
                text += f"\n\n(plot save failed: {exc})"
        QMessageBox.information(self, f"Compare — {name}", text)

    def _on_env_compare_failed(self, msg: str) -> None:
        self._restore_after_env_compare()
        self._status.setText("compare failed")
        QMessageBox.critical(self, "Compare failed", msg)

    def _on_env_compare_cancelled(self) -> None:
        self._restore_after_env_compare()
        self._status.setText("compare cancelled")

    # ----------------------------------------------------------------
    # M7 — Multi-particle surrogates (hybrid mode) section
    # ----------------------------------------------------------------
    def _build_mp_section(self) -> "CollapsibleSection":
        """Construct the collapsed-by-default 'Multi-particle surrogates'
        panel that sits beneath the Trained-surrogates table.

        Layout:
          * Hint label explaining the hybrid contract.
          * "Engage in MP runs" master QCheckBox -> registry.set_mp_enabled.
          * "Residual RK4 substeps" QSpinBox (default 15, accuracy-first).
          * "Compare MP" button -> background worker runs compare_mp on
             the loaded lattice + beam config.
        """
        from linac_gen_gui.interphase.tabs.convergence_tab import (
            CollapsibleSection)

        sec = CollapsibleSection("Multi-particle surrogates (hybrid mode)")
        sec.setExpanded(False)

        # Hint label spans both form columns via addRow(widget).
        hint = QLabel(
            "Engage trained surrogates in multi-particle runs.  Safe "
            "mode (default) delegates to native RK4 — bit-identical, "
            "no speedup; tick the experimental fast path for "
            "linear-matrix surrogate transport.  (The planned hybrid "
            "RK4-residual mode is not implemented — the substeps "
            "control below is reserved.)")
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px; padding:6px 0;")
        sec.addRow(hint)

        self._mp_engage = QCheckBox("Engage in MP runs")
        self._mp_engage.setChecked(False)
        self._mp_engage.toggled.connect(self._on_mp_engage_toggled)
        sec.addRow("Master:", self._mp_engage)

        # M7-followup: second opt-in for the actual linear-matrix
        # fast path.  Stays off by default; with both this AND the
        # master toggle on, the per-substep wrapped.track_rk4 call
        # is replaced by an analytic ref advance + batched M_slice @
        # particles.  See `docs/manual/13_surrogates/01_overview.md`
        # for the accuracy / speedup trade-offs.
        self._mp_fast_path = QCheckBox(
            "Experimental: linear-matrix fast path")
        self._mp_fast_path.setChecked(False)
        self._mp_fast_path.setToolTip(
            "Bypass per-substep RK4 in favor of a cached matrix-slice "
            "apply.  ~10-15x speedup expected with production-quality "
            "training; accuracy degrades for halo particles and with "
            "smoke training.  Always run Compare MP after toggling on.")
        self._mp_fast_path.toggled.connect(self._on_mp_fast_path_toggled)
        sec.addRow("", self._mp_fast_path)

        self._mp_substeps = QSpinBox()
        self._mp_substeps.setRange(0, 100)
        self._mp_substeps.setValue(15)
        self._mp_substeps.setToolTip(
            "RESERVED — no tracking path reads this value yet (the "
            "planned hybrid linear-anchor + RK4-residual mode is not "
            "implemented).  The setting is stored and propagated to "
            "registered surrogates but has no effect on any run.")
        self._mp_substeps.valueChanged.connect(self._on_mp_substeps_changed)
        sec.addRow("Residual RK4 substeps (reserved):", self._mp_substeps)

        self._mp_compare_btn = QPushButton("Compare MP (baseline vs hybrid)")
        self._mp_compare_btn.setIcon(icon("play", 12, "#00161c"))
        self._mp_compare_btn.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; "
            f"border-radius:3px; padding:6px 14px; font-weight:600;")
        self._mp_compare_btn.clicked.connect(self._on_mp_compare_clicked)
        sec.addRow("", self._mp_compare_btn)

        # Indeterminate progress bar -- visible only while the worker
        # runs.  Before this was added, a Compare-MP click looked
        # like "nothing happened" for ~15 s while the worker silently
        # executed two full MP simulations.
        self._mp_progress = QProgressBar()
        self._mp_progress.setRange(0, 0)        # indeterminate (busy)
        self._mp_progress.setTextVisible(False)
        self._mp_progress.setMaximumHeight(6)
        self._mp_progress.setVisible(False)
        sec.addRow("", self._mp_progress)

        self._mp_status = QLabel("")
        # Slightly larger / accent-coloured so it draws the eye when
        # the run starts (previously 10 px tertiary text -- easy to
        # miss).  Returned to muted styling at idle in the slots
        # below.
        self._mp_status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px; "
            f"font-family:{theme.FONT_MONO};")
        self._mp_status.setWordWrap(True)
        sec.addRow(self._mp_status)

        # Persist open/closed via QSettings (same pattern as
        # convergence_tab's _persist_section_state).
        sec.toggled_changed.connect(self._persist_mp_section_open)

        return sec

    def _restore_mp_section_state(self) -> None:
        s = make_settings("HELIX", "linac_gen_gui")
        # Master toggle
        eng = s.value("surrogates/mp_enabled", False, type=bool)
        if bool(eng):
            # Set state without triggering the toggled handler twice.
            self._mp_engage.blockSignals(True)
            self._mp_engage.setChecked(True)
            self._mp_engage.blockSignals(False)
            # Push to the registry directly.
            from linac_gen.surrogates import registry as _reg
            _reg.set_mp_enabled(True)
        # Fast-path toggle (M7-followup)
        fp = s.value("surrogates/mp_fast_path", False, type=bool)
        if bool(fp):
            self._mp_fast_path.blockSignals(True)
            self._mp_fast_path.setChecked(True)
            self._mp_fast_path.blockSignals(False)
            from linac_gen.surrogates import registry as _reg
            _reg.set_fast_path_enabled(True)
        # Substeps
        sub = s.value("surrogates/mp_residual_substeps", 15, type=int)
        self._mp_substeps.blockSignals(True)
        self._mp_substeps.setValue(int(sub))
        self._mp_substeps.blockSignals(False)
        self._propagate_substeps_to_registered(int(sub))
        # Section open/closed
        sec_key = "surrogates/section/Multi-particle surrogates"
        if s.contains(sec_key):
            self._mp_section.setExpanded(bool(s.value(sec_key, type=bool)))

    def _persist_mp_section_open(self, title: str, expanded: bool) -> None:
        s = make_settings("HELIX", "linac_gen_gui")
        s.setValue(f"surrogates/section/{title}", bool(expanded))

    def _on_mp_engage_toggled(self, checked: bool) -> None:
        from linac_gen.surrogates import registry as _reg
        _reg.set_mp_enabled(bool(checked))
        make_settings("HELIX", "linac_gen_gui").setValue(
            "surrogates/mp_enabled", bool(checked))
        # Honest status: in safe mode (fast path OFF) an "engaged"
        # surrogate DELEGATES every call to native RK4 — bit-identical,
        # zero neural inference.  Only the fast path changes physics.
        fast = self._mp_fast_path.isChecked()
        self._mp_status.setText(
            ("MP-mode engaged (fast path ON): surrogate linear-matrix "
             "transport replaces per-substep RK4 for registered "
             "elements."
             if fast else
             "MP-mode engaged (safe mode): registered surrogates "
             "DELEGATE to native RK4 — bit-identical to baseline, no "
             "neural inference.  Tick the fast path for actual "
             "surrogate transport.")
            if checked else
            "MP-mode disengaged: MP runs use the baseline RK4 path.")

    def _on_mp_fast_path_toggled(self, checked: bool) -> None:
        from linac_gen.surrogates import registry as _reg
        _reg.set_fast_path_enabled(bool(checked))
        make_settings("HELIX", "linac_gen_gui").setValue(
            "surrogates/mp_fast_path", bool(checked))
        if checked:
            self._mp_status.setText(
                "Fast path ENGAGED: per-substep RK4 replaced by "
                "M_slice @ particles.  Run Compare MP to validate "
                "accuracy on your lattice.")
        else:
            self._mp_status.setText(
                "Fast path disengaged: MP runs use the M7 safe-delegate "
                "(bit-identical to baseline).")

    def _on_mp_substeps_changed(self, val: int) -> None:
        make_settings("HELIX", "linac_gen_gui").setValue(
            "surrogates/mp_residual_substeps", int(val))
        self._propagate_substeps_to_registered(int(val))
        self._mp_status.setText(
            f"Residual RK4 substeps set to {val} (propagated to "
            f"{len(self._trained)} registered surrogate(s)).")

    def _propagate_substeps_to_registered(self, n: int) -> None:
        """Apply ``n`` to every currently-tracked SurrogateFieldMap.

        The trained-surrogates dict (`self._trained`) is the source of
        truth; we set residual_n_steps on each, whether or not it's
        currently in the registry.  That way ticking 'Use' later
        engages the surrogate with the right substep value too.
        """
        for _name, (surr, _dir, _meta) in self._trained.items():
            try:
                surr.residual_n_steps = int(n)
            except Exception:                                # noqa: BLE001
                pass

    def _on_mp_compare_clicked(self) -> None:
        """Run compare_mp on the loaded lattice + beam config in a
        background worker; show the summary dialog + offer PNG save."""
        if self.state.lattice is None or self.state.beam_config is None:
            QMessageBox.warning(
                self, "Need lattice + beam",
                "Load a lattice and apply a beam config first.")
            return
        if not self._trained:
            QMessageBox.warning(
                self, "No trained surrogates",
                "Train at least one surrogate (or load cached weights) "
                "before running an MP comparison.")
            return
        if self.state.running:
            QMessageBox.warning(
                self, "Busy",
                "Another run is in progress — compare swaps the global "
                "surrogate registry and cannot overlap a solve.")
            return

        surrogates = [t[0] for t in self._trained.values()]
        residual_n_steps = int(self._mp_substeps.value())

        # Exclude toolbar solves while the registry is swapped (the
        # worker was racing them before — same physics hazard as the
        # envelope compare).
        self.state.set_running(True)
        self._mp_compare_btn.setEnabled(False)
        self._mp_compare_btn.setText("Running comparison...  (~15 s)")
        self._mp_progress.setVisible(True)
        # Accent the status line during the run so it actually catches
        # the user's eye (10 px tertiary text is too easy to miss).
        self._mp_status.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:11px; "
            f"font-family:{theme.FONT_MONO}; font-weight:600;")
        self._mp_status.setText(
            f"Running Compare MP: 2 simulations on "
            f"{self.state.beam_config.n_particles}-particle beam "
            f"with {len(surrogates)} surrogate(s), "
            f"residual_n_steps={residual_n_steps}.  Save-PNG dialog "
            f"will appear when done."
        )

        self._mp_compare_worker = _MpCompareWorker(
            lattice=self.state.lattice,
            beam_config=self.state.beam_config,
            surrogates=surrogates,
            residual_n_steps=residual_n_steps,
            parent=self,
        )
        self._mp_compare_worker.finished_ok.connect(
            self._on_mp_compare_finished)
        self._mp_compare_worker.failed.connect(self._on_mp_compare_failed)
        self._mp_compare_worker.start()

    def _restore_compare_btn_idle(self) -> None:
        """Reset Compare-MP button label + style + progress bar
        after the worker completes (success or failure)."""
        self.state.set_running(False)
        self._mp_compare_btn.setEnabled(True)
        self._mp_compare_btn.setText("Compare MP (baseline vs hybrid)")
        self._mp_progress.setVisible(False)
        self._mp_status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:11px; "
            f"font-family:{theme.FONT_MONO};")

    def _on_mp_compare_finished(self, report) -> None:
        self._restore_compare_btn_idle()
        text = report.summary_text()
        self._mp_status.setText(
            f"Compare MP done: speedup {report.speedup():.2f}x  "
            f"worst rel.diff {report.worst_rel_diff():.2e}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MP-compare PNG",
            "mp_compare.png", "PNG (*.png);;All Files (*)")
        if path:
            try:
                from linac_gen.surrogates.compare import (
                    plot_compare_mp_report)
                plot_compare_mp_report(report, path)
                text += f"\n\nPlot saved: {path}"
            except Exception as exc:                       # noqa: BLE001
                text += f"\n\n(plot save failed: {exc})"
        QMessageBox.information(self, "Compare MP", text)

    def _on_mp_compare_failed(self, msg: str) -> None:
        self._restore_compare_btn_idle()
        self._mp_status.setText(f"Compare MP FAILED: {msg}")
        QMessageBox.critical(self, "Compare MP failed", msg)


# ---------------------------------------------------------------------------
class _EnvCompareWorker(QThread):
    """Background thread for `compare_envelope` — the per-row Compare
    button used to run its two envelope solves INLINE on the GUI thread,
    freezing the app for the full ~20 s."""

    finished_ok = pyqtSignal(str, object)   # (surrogate name, CompareReport)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, lattice, ref, twiss, current, surrogate, name,
                 parent=None):
        super().__init__(parent)
        self._lattice = lattice
        self._ref = ref
        self._twiss = dict(twiss)
        self._current = float(current)
        self._surrogate = surrogate
        self._name = str(name)
        import threading
        self._stop_event = threading.Event()
        # Same OpenBLAS stack-headroom rationale as _MpCompareWorker.
        self.setStackSize(16 * 1024 * 1024)

    def request_stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            from linac_gen.core.cancelled import OperationCancelled
            from linac_gen.surrogates.compare import compare_envelope
            try:
                report = compare_envelope(
                    self._lattice, self._ref, self._twiss,
                    current=self._current, surrogates=[self._surrogate],
                    should_abort=self._stopping,
                )
            except OperationCancelled:
                self.cancelled.emit()
                return
            self.finished_ok.emit(self._name, report)
        except Exception as exc:                              # noqa: BLE001
            self.failed.emit(repr(exc))


# ---------------------------------------------------------------------------
class _MpCompareWorker(QThread):
    """Background thread that runs `compare_mp` so the GUI stays
    responsive during what can be a 30-60 s job."""

    finished_ok = pyqtSignal(object)     # CompareMpReport
    failed = pyqtSignal(str)

    def __init__(self, lattice, beam_config, surrogates,
                 residual_n_steps: int, parent=None):
        super().__init__(parent)
        self._lattice = lattice
        self._beam_config = beam_config
        self._surrogates = list(surrogates)
        self._residual_n_steps = int(residual_n_steps)
        # The fast-path's per-element `scipy.linalg.expm` call uses
        # `np.linalg.inv`, which on macOS routes through OpenBLAS's
        # `dgetrf_parallel`.  That routine consumes a large amount of
        # stack space, easily blowing past QThread's default ~512 KB
        # macOS thread stack (we crashed in `___chkstk_darwin` from
        # OpenBLAS's parallel LU).  16 MB is comfortable headroom and
        # costs nothing at idle; the main thread already gets 8 MB
        # from the OS by default.
        self.setStackSize(16 * 1024 * 1024)

    def run(self) -> None:
        try:
            from linac_gen.distributions.factory import create_beam
            from linac_gen.surrogates.compare import compare_mp
            beam = create_beam(self._beam_config, seed=42)
            report = compare_mp(
                self._lattice, beam,
                surrogates=self._surrogates,
                residual_n_steps=self._residual_n_steps,
            )
            self.finished_ok.emit(report)
        except Exception as exc:                              # noqa: BLE001
            self.failed.emit(repr(exc))
