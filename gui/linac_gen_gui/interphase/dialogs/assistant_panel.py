"""GUI surface for the OPTIONAL AI assistant.

A NON-MODAL QDialog (deliberately not the modal Python-console
pattern: a modal exec() would freeze the main window and deadlock the
worker↔approval bridge).  The agent loop runs in a QThread; tool calls
that change session state flow through AppState setters so the tabs
stay in sync.

This module lazy-imports ``linac_gen.assist`` inside methods — nothing
here is imported at GUI startup, so a HELIX with no assistant
configured (or the subpackage removed) launches and runs untouched.
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme


# ---------------------------------------------------------------------------
# WorkContext that proxies the live AppState (assist stays Qt-free; the
# proxying lives here in the GUI layer).
# ---------------------------------------------------------------------------
def _grab_png(widget, label: str, max_w: int = 1024,
              max_bytes: int = 600_000):
    """GUI-thread helper: render a widget to an image payload (downscaled
    to ≤max_w).  PNG first; if the encoding exceeds ``max_bytes`` (dense
    heatmaps compress poorly and can overflow the SDK's stdio transport),
    fall back to JPEG, then to a further-downscaled JPEG."""
    import base64

    from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt

    def _encode(pix, fmt, quality=-1):
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        pix.save(buf, fmt, quality)
        buf.close()
        return bytes(ba)

    pix = widget.grab()
    if pix.isNull():
        return None
    if pix.width() > max_w:
        pix = pix.scaledToWidth(
            max_w, Qt.TransformationMode.SmoothTransformation)
    data, mime = _encode(pix, "PNG"), "image/png"
    if len(data) > max_bytes:
        data, mime = _encode(pix, "JPEG", 85), "image/jpeg"
    if len(data) > max_bytes:
        pix = pix.scaledToWidth(
            640, Qt.TransformationMode.SmoothTransformation)
        data, mime = _encode(pix, "JPEG", 80), "image/jpeg"
    return {"img_b64": base64.b64encode(data).decode("ascii"),
            "mime": mime, "w": pix.width(), "h": pix.height(),
            "label": label}


def _make_context(state, calc_dir: str, nav=None):
    from linac_gen.assist.tools import WorkContext

    class _AppStateContext(WorkContext):
        def __init__(self):
            super().__init__(calc_dir=calc_dir)
            self._state = state
            self._nav = nav          # the panel (queued nav signal + labels)

        # reads reflect the live app
        @property
        def lattice(self):
            return self._state.lattice

        @lattice.setter
        def lattice(self, v):            # WorkContext.__init__ sets this
            pass

        @property
        def lattice_path(self):
            return getattr(self._state, "lattice_path", "") or ""

        @lattice_path.setter
        def lattice_path(self, v):
            pass

        @property
        def beam_config(self):
            return self._state.beam_config

        @beam_config.setter
        def beam_config(self, v):
            pass

        @property
        def results(self):
            return self._state.results

        @results.setter
        def results(self, v):
            pass

        # writes go through AppState setters → tabs update via signals
        def set_lattice(self, lattice, path):
            self._state.set_lattice(lattice, path)

        def set_beam_config(self, cfg):
            self._state.set_beam_config(cfg)

        def set_results(self, results, path=""):
            self._results_path = path
            self._state.set_results(results)

        @property
        def results_path(self):
            return getattr(self, "_results_path", "")

        @results_path.setter
        def results_path(self, v):
            self._results_path = v

        # GUI navigation — resolve against the live tab labels, then hop to
        # the GUI thread via the panel's queued signal (never touch widgets
        # from the agent worker thread).
        def available_tabs(self):
            return list(getattr(self._nav, "tab_labels", []) or []) \
                if self._nav is not None else []

        def available_subtabs(self):
            return dict(getattr(self._nav, "subtab_map", {}) or {}) \
                if self._nav is not None else {}

        def available_plots(self):
            return [lab for _k, lab in
                    (getattr(self._nav, "plot_catalog", []) or [])] \
                if self._nav is not None else []

        def show_tab(self, tab, subtab=None):
            if self._nav is None:
                return None
            labels = list(getattr(self._nav, "tab_labels", []) or [])
            want = str(tab).strip().casefold()
            match = next((t for t in labels if t.casefold() == want), None)
            if match is None:
                try:
                    from linac_gen_gui.interphase.state import TABS
                    id2 = {tid.casefold(): lab for tid, lab in TABS}
                    if want in id2 and id2[want] in labels:
                        match = id2[want]
                except Exception:                # noqa: BLE001
                    pass
            if match is None:
                match = next((t for t in labels
                              if want and want in t.casefold()), None)
            if match is None:
                return None
            self._nav.navigate_requested.emit(match, str(subtab or ""))
            if subtab:                           # describe the subtab too
                subs = (getattr(self._nav, "subtab_map", {})
                        or {}).get(match, [])
                sw = str(subtab).casefold()
                sm = next((s for s in subs if s.casefold() == sw
                           or sw in s.casefold()), None)
                return f"{match} › {sm}" if sm else match
            return match

        def _match_plot(self, name):
            catalog = list(getattr(self._nav, "plot_catalog", []) or [])
            want = str(name).strip().casefold()
            match = next((c for c in catalog if c[0].casefold() == want
                          or c[1].casefold() == want), None)
            if match is None:                    # label substring
                match = next((c for c in catalog
                              if want and want in c[1].casefold()), None)
            if match is None:                    # key substring
                match = next((c for c in catalog
                              if want and want in c[0].casefold()), None)
            return match

        def open_plot(self, name):
            if self._nav is None:
                return None
            match = self._match_plot(name)
            if match is not None:
                self._nav.plot_requested.emit(match[0])   # emit the KEY
                return match[1]                            # return the label
            return None

        # -- Sight: grab rendered figures as PNG (GUI thread) -----------
        def grab_plot(self, name):
            if self._nav is None or self._nav._app is None:
                return None
            match = self._match_plot(name)
            if match is None:
                return None
            key, label = match
            app = self._nav._app

            def _do():
                from PyQt6.QtCore import QEventLoop
                from PyQt6.QtWidgets import QApplication
                app.show_result_plot(key)          # opens + switches
                QApplication.processEvents(        # let it lay out — but
                    QEventLoop.ProcessEventsFlag   # never re-enter user
                    .ExcludeUserInputEvents)       # input (click storms)
                dlg = getattr(app.results_tab, "_popups", {}).get(key)
                if dlg is None:
                    return None
                return _grab_png(dlg, label)

            return self._nav.run_on_gui(_do, timeout=8.0)

        def grab_screen(self):
            if self._nav is None or self._nav._app is None:
                return None
            app = self._nav._app

            def _do():
                # frontmost visible popup wins, else the current tab
                for key, dlg in getattr(app.results_tab, "_popups",
                                        {}).items():
                    try:
                        if dlg.isVisible() and dlg.isActiveWindow():
                            labels = dict(app.results_tab.plot_catalog())
                            return _grab_png(dlg, labels.get(key, key))
                    except RuntimeError:
                        continue
                w = app._tabs.currentWidget()
                if w is None:
                    return None
                label = app._tabs.tabText(app._tabs.currentIndex())
                return _grab_png(w, f"{label} tab")

            return self._nav.run_on_gui(_do, timeout=8.0)

        def highlight_element(self, index, s_mm):
            if self._nav is None:
                return False
            self._nav.highlight_requested.emit(int(index), float(s_mm))
            return True

        def set_cursor(self, s_mm):
            if self._nav is None:
                return False
            self._nav.cursor_requested.emit(float(s_mm))
            return True

        def run_gui_simulation(self, kind):
            if self._nav is None or self._nav._app is None \
                    or not hasattr(self._nav._app,
                                   "assistant_run_simulation"):
                return None
            app = self._nav._app
            return self._nav.run_on_gui(
                lambda: app.assistant_run_simulation(kind), timeout=10.0)

        def gui_context(self):
            st = self._state
            snap = {
                "lattice_loaded": st.lattice is not None,
                "lattice_path": getattr(st, "lattice_path", "") or "",
                "results_loaded": st.results is not None,
                "selected_element": getattr(st.selected, "name", None)
                    if st.selected is not None else None,
                "s_cursor_m": round(float(st.s_cursor) * 1e-3, 4),
                "current_tab": None,
                "open_plot_windows": [],
            }
            try:                       # tab title from the registry (no Qt)
                from linac_gen_gui.interphase.state import TABS
                if 0 <= st.tab < len(TABS):
                    snap["current_tab"] = TABS[st.tab][1]
            except Exception:                    # noqa: BLE001
                pass
            # live widget bits (open popups) — GUI-thread round-trip
            if self._nav is not None and self._nav._app is not None \
                    and hasattr(self._nav._app, "assistant_open_plots"):
                app = self._nav._app
                live = self._nav.run_on_gui(
                    lambda: app.assistant_open_plots(), timeout=2.0)
                if live is not None:
                    snap["open_plot_windows"] = live
            return snap

    return _AppStateContext()


# ---------------------------------------------------------------------------
# Thread-bridged approval: the worker thread blocks on request(); the GUI
# thread renders buttons and sets the answer.
# ---------------------------------------------------------------------------
class _GuiApprover:
    def __init__(self, panel, timeout_s: float = 0.0):
        self._panel = panel
        self._timeout = float(timeout_s or 0.0)   # 0 = wait forever
        self._event = threading.Event()
        self._decision = None
        self._pending = None
        # Generation counter: a button click that lands AFTER a timeout
        # auto-deny must not resolve the NEXT confirmation (the strip may
        # already show a different call).
        self._gen = 0

    @property
    def generation(self) -> int:
        return self._gen

    @property
    def pending(self):
        return self._pending

    def __call__(self, req):
        import time as _time
        from linac_gen.assist.agent import Decision
        self._gen += 1
        self._event.clear()
        self._decision = None
        self._pending = req
        # ask the GUI thread to show the strip (queued connection)
        self._panel.confirmation_needed.emit(req)
        t0 = _time.time()
        # block the worker until the user answers or the panel closes
        try:
            while not self._event.wait(0.1):
                if self._panel._closing:
                    return Decision.ABORT
                if self._timeout and _time.time() - t0 > self._timeout:
                    self._panel.confirm_timed_out.emit(
                        f"auto-denied after {self._timeout:.0f} s "
                        "without an answer")
                    return Decision.DENY
            return self._decision or Decision.DENY
        finally:
            self._pending = None

    def resolve(self, decision, gen: int | None = None):
        if gen is not None and gen != self._gen:
            return                       # stale click — belongs to a past ask
        self._decision = decision
        self._event.set()


class _AgentWorker(QThread):
    assistant_text = pyqtSignal(str)
    assistant_delta = pyqtSignal(str)        # live streaming token(s)
    assistant_delta_done = pyqtSignal()      # end of a streamed segment
    event = pyqtSignal(object)
    turn_done = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, session, prompt, parent=None):
        super().__init__(parent)
        self.setStackSize(16 * 1024 * 1024)      # numpy in tool calls
        self._session = session
        self._prompt = prompt

    def run(self):
        # NOTHING may escape QThread.run — an exception here surfaces
        # as the mangled 'sipBadCatcherResult' TypeError (observed);
        # emits can hit a deleted C++ object during panel teardown.
        try:
            self._session.ask(self._prompt)
        except Exception as exc:                 # noqa: BLE001
            try:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
            except RuntimeError:
                pass
        finally:
            try:
                self.turn_done.emit()
            except RuntimeError:
                pass


class _EventWorker(_AgentWorker):
    """An unprompted model turn reacting to queued machine events.
    Same signal plumbing as a user turn; drives ``ask_event()`` so the
    transcript shows no '▶ you' line."""

    def __init__(self, session, parent=None):
        super().__init__(session, "", parent)

    def run(self):
        try:
            self._session.ask_event()
        except Exception as exc:                 # noqa: BLE001
            try:
                self.failed.emit(f"{type(exc).__name__}: {exc}")
            except RuntimeError:
                pass
        finally:
            self.turn_done.emit()


class _TranscribeWorker(QThread):
    """Off-thread speech-to-text (Whisper model load + inference).

    Given a ``recorder``, its ``stop()`` (which JOINS the reader thread
    — up to a blocking read away) also runs here: the GUI thread must
    never wait on an audio device, or the ● rec cue visibly freezes."""
    transcribed = pyqtSignal(str)
    failed = pyqtSignal(str)
    too_short = pyqtSignal(float)                # captured seconds

    def __init__(self, audio, parent=None, recorder=None):
        super().__init__(parent)
        self.setStackSize(16 * 1024 * 1024)      # numpy/ctranslate2
        self._audio = audio
        self._recorder = recorder

    def run(self):
        try:
            import time as _time
            audio = self._audio
            if self._recorder is not None:
                audio = self._recorder.stop()    # join OFF the GUI thread
            n = int(getattr(audio, "size", 0))
            if n < int(0.3 * 16000):             # under the STT gate:
                self.too_short.emit(n / 16000.0)  # say so honestly
                return
            from linac_gen.assist.voice import WhisperSTT
            t0 = _time.monotonic()
            text = WhisperSTT().transcribe(audio)
            self.stt_s = _time.monotonic() - t0
            self.transcribed.emit(text)
        except RuntimeError:
            pass                                 # panel torn down mid-turn
        except Exception as exc:                 # noqa: BLE001
            try:
                self.failed.emit(f"transcription failed: {exc}")
            except RuntimeError:
                pass


class _FastWorker(QThread):
    """Executes one intent fast-path tool call off the GUI thread."""
    done = pyqtSignal(object)                    # (FastIntent, result)

    def __init__(self, session, intent, parent=None):
        super().__init__(parent)
        self.setStackSize(16 * 1024 * 1024)
        self._session = session
        self._intent = intent

    def run(self):
        try:
            res = self._session.run_fast(self._intent.tool,
                                         self._intent.params)
        except Exception:                        # noqa: BLE001
            res = None
        try:
            self.done.emit((self._intent, res))
        except RuntimeError:
            pass                                 # panel torn down mid-call


class AssistantPanel(QDialog):
    confirmation_needed = pyqtSignal(object)
    confirm_timed_out = pyqtSignal(str)        # approver timeout → hide strip
    navigate_requested = pyqtSignal(str, str)  # assistant → tab, subtab
    plot_requested = pyqtSignal(str)           # assistant → open a Results plot
    highlight_requested = pyqtSignal(int, float)  # element index, s_mm
    cursor_requested = pyqtSignal(float)          # s_mm
    gui_call = pyqtSignal(object)              # generic run-on-GUI-thread
    state_changed = pyqtSignal(str)            # worker-thread → orb/lamp
    level_changed = pyqtSignal(float)          # mic RMS → orb bars
    mic_died = pyqtSignal()                    # reader thread → recovery
    speaking_changed = pyqtSignal(bool)        # TTS active → orb state
    models_listed = pyqtSignal(str, list)      # local server → model dropdown
    event_line = pyqtSignal(str)               # machine events (any thread)
    progress_changed = pyqtSignal(str)         # long-run progress text
    capture_saved = pyqtSignal(str, str)       # path, tool → inline image
    wake_text = pyqtSignal(str)                # wake/follow-up utterance
    wake_status = pyqtSignal(str)              # listening cues → status

    # dropdown suggestions per backend (the box stays editable — gateway
    # aliases, full IDs and "URL model" for local servers all still type)
    _MODEL_SUGGESTIONS = {
        "anthropic": ["claude-fable-5", "claude-sonnet-5",
                      "claude-opus-4-8", "claude-haiku-4-5-20251001"],
        "claude_sdk": ["fable", "opus", "sonnet", "haiku"],
        "openai": [],
    }

    def __init__(self, parent, state):
        super().__init__(parent)
        self.setWindowTitle("HELIX Assistant  (AI · optional)")
        self.resize(760, 640)
        self.setModal(False)
        self._state = state
        self._session = None
        self._worker = None
        self._approver = None
        self._closing = False
        # tab navigation: the main window (parent) exposes show_tab + labels
        self._app = parent if hasattr(parent, "show_tab") else None
        self.tab_labels = (self._app.assistant_tab_labels()
                           if self._app is not None else [])
        self.subtab_map = (self._app.assistant_subtabs()
                           if self._app is not None
                           and hasattr(self._app, "assistant_subtabs")
                           else {})
        self.plot_catalog = (self._app.result_plot_catalog()
                             if self._app is not None
                             and hasattr(self._app, "result_plot_catalog")
                             else [])

        # the panel's design system: 14 px body type, rounded controls,
        # accent-filled Send, ghost secondaries with hover states
        self.setStyleSheet(f"""
            QDialog {{ background:{theme.BG_0}; }}
            QLabel {{ color:{theme.TEXT_1}; }}
            QLineEdit {{
                background:{theme.BG_INSET}; color:{theme.TEXT_0};
                border:1px solid {theme.BORDER_0}; border-radius:9px;
                padding:9px 12px; font-size:14px;
            }}
            QLineEdit:focus {{ border:1px solid {theme.ACCENT}; }}
            QPushButton {{
                background:transparent; color:{theme.TEXT_1};
                border:1px solid {theme.BORDER_0}; border-radius:9px;
                padding:8px 14px; font-size:13px;
            }}
            QPushButton:hover {{
                border:1px solid {theme.ACCENT}; color:{theme.TEXT_0};
            }}
            QPushButton:pressed {{ background:{theme.BG_INSET}; }}
            QPushButton#sendBtn {{
                background:{theme.ACCENT}; color:{theme.BG_0};
                border:none; font-weight:600;
            }}
            QPushButton#sendBtn:hover {{ background:{theme.ACCENT_2}; }}
            QPushButton#chipBtn {{
                background:{theme.BG_INSET}; color:{theme.TEXT_1};
                border:1px solid {theme.BG_2}; border-radius:11px;
                padding:2px 10px; font-size:11px;
            }}
            QPushButton#chipBtn:hover {{
                border:1px solid {theme.ACCENT}; color:{theme.TEXT_0};
            }}
            QPushButton#stopBtn:hover {{
                border:1px solid #f87171; color:#f87171;
            }}
            QPushButton:checked {{
                background:{theme.BG_INSET};
                border:1px solid {theme.ACCENT}; color:{theme.ACCENT};
            }}
            QCheckBox {{ color:{theme.TEXT_2}; font-size:12px;
                         spacing:6px; }}
            QCheckBox:hover {{ color:{theme.TEXT_0}; }}
            QComboBox {{
                background:{theme.BG_INSET}; color:{theme.TEXT_1};
                border:1px solid {theme.BORDER_0}; border-radius:8px;
                padding:6px 10px;
            }}
            QFrame#confirmCard {{
                background:{theme.BG_INSET};
                border:1px solid #b45309; border-radius:10px;
            }}
        """)
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        self._vbox = v          # voice-only view re-weights the stretch

        # animated state orb + lamp (optional visual; degrades if it fails)
        try:
            from linac_gen_gui.interphase.dialogs.assistant_orb import (
                AssistantOrb)
            self._orb = AssistantOrb()
            v.addWidget(self._orb)
            self._lamp = QLabel("● idle")
            self._lamp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._lamp.setStyleSheet(
                f"color:{theme.TEXT_2}; font-size:11px; font-weight:600;")
            v.addWidget(self._lamp)
        except Exception:                        # noqa: BLE001
            self._orb = None
            self._lamp = None

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        self._perf = QLabel("")
        self._perf.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:10px;")
        self._perf.setToolTip("turn latency (first token / instant "
                              "command) and speech-to-text time")
        status_row = QHBoxLayout()
        status_row.addWidget(self._status, stretch=1)
        status_row.addWidget(self._perf)
        # the provider row hides once connected — this re-opens it so the
        # backend can be switched (e.g. subscription login -> API key)
        self._backend_btn = QPushButton("backend…")
        self._backend_btn.setToolTip(
            "Change the assistant backend / API key.  Shows the provider "
            "row; pick a backend and press Connect to switch.")
        self._backend_btn.clicked.connect(self._on_show_backend)
        status_row.addWidget(self._backend_btn)
        v.addLayout(status_row)

        # ---- provider settings (key entered HERE, never in a chat) ----
        self._settings_box = QWidget()
        srow = QHBoxLayout(self._settings_box)
        srow.setContentsMargins(0, 0, 0, 0)
        self._provider = QComboBox()
        self._provider.addItem("Anthropic (cloud, API key)", "anthropic")
        self._provider.addItem("Local / OpenAI-compatible", "openai")
        self._provider.addItem(
            "Claude (subscription login — no API key)", "claude_sdk")
        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("Anthropic API key (sk-…)")
        self._model_edit = QComboBox()
        self._model_edit.setEditable(True)
        self._model_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model_edit.setMinimumWidth(180)
        self._model_edit.lineEdit().setPlaceholderText(
            "model (blank = default)")
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect)
        srow.addWidget(QLabel("backend:"))
        srow.addWidget(self._provider)
        srow.addWidget(self._key_edit, stretch=1)
        srow.addWidget(self._model_edit)
        srow.addWidget(self._connect_btn)
        v.addWidget(self._settings_box)
        self._repopulate_models(self._provider.currentData())
        self._provider.currentIndexChanged.connect(self._on_provider_changed)
        self.models_listed.connect(self._on_models_listed)

        from linac_gen_gui.interphase.dialogs.assistant_chat import (
            ChatView,
        )
        self._transcript = ChatView(self)
        try:                     # user-chosen conversation text size
            saved_px = int(self._settings().value("assist/chat_px", 0))
            if saved_px:
                self._transcript.set_text_size(saved_px)
        except Exception:                    # noqa: BLE001
            pass
        v.addWidget(self._transcript, stretch=1)

        # confirmation card (hidden until a compute/mutate call): a
        # warm-bordered rounded frame so an approval request is
        # unmistakable without being an alarm
        from PyQt6.QtWidgets import QFrame
        self._confirm_frame = QFrame(self)
        self._confirm_frame.setObjectName("confirmCard")
        self._confirm_row = QHBoxLayout(self._confirm_frame)
        self._confirm_row.setContentsMargins(12, 8, 12, 8)
        self._confirm_label = QLabel("")
        self._confirm_label.setStyleSheet(
            f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO};"
            f"font-size:12px; background:transparent; border:none;")
        self._confirm_label.setWordWrap(True)
        self._btn_approve = QPushButton("Approve")
        self._btn_always = QPushButton("Approve (session)")
        self._btn_deny = QPushButton("Deny")
        for b, slot in ((self._btn_approve, self._on_approve),
                        (self._btn_always, self._on_always),
                        (self._btn_deny, self._on_deny)):
            b.clicked.connect(slot)          # bound methods, not lambdas
        self._confirm_row.addWidget(self._confirm_label, stretch=1)
        self._confirm_row.addWidget(self._btn_approve)
        self._confirm_row.addWidget(self._btn_always)
        self._confirm_row.addWidget(self._btn_deny)
        self._confirm_widget = QLabel()      # placeholder container
        v.addWidget(self._confirm_frame)
        self._set_confirm_visible(False)

        # quick-action chips: one-click entry to the marquee features
        # (discoverability — the tour/drills/python exist but nobody
        # finds a feature they have to know to ask for)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self._chip_btns = []
        for label, prompt in (
                ("🎓 Tour", "start the guided tour"),
                ("🧩 Drill", "start a training drill"),
                ("🐍 Python", "use run_python to analyze the current "
                              "results and show me a plot"),
                ("❓ What can you do", "what can you do? list your "
                                       "capabilities briefly")):
            b = QPushButton(label)
            b.setObjectName("chipBtn")
            b.setProperty("chip_prompt", prompt)
            b.clicked.connect(self._on_chip_clicked)
            chips.addWidget(b)
            self._chip_btns.append(b)
        chips.addStretch(1)
        v.addLayout(chips)

        row = QHBoxLayout()
        # push-to-talk mic (offline voice — shown only if the audio
        # stack is available; hold to speak, release to transcribe)
        self._mic_btn = QPushButton("🎤 Hold")
        self._mic_btn.setToolTip(
            "Push-to-talk: press and hold to speak, release to send "
            "(offline: faster-whisper).")
        self._mic_btn.pressed.connect(self._mic_pressed)
        self._mic_btn.released.connect(self._mic_released)
        # hands-free wake listening — say "HELIX", hear the chime, talk
        self._wake_btn = QPushButton("👂 HELIX")
        self._wake_btn.setCheckable(True)
        self._wake_btn.setToolTip(
            "Hands-free: keep the mic open and wake the assistant by "
            "saying 'HELIX' (chime confirms).  Detection pauses while "
            "the assistant speaks or you hold the mic.  Fully local.")
        self._wake_btn.toggled.connect(self._on_wake_toggled)
        self._input = QLineEdit()
        self._input.setPlaceholderText(
            "Ask the assistant…  (e.g. 'run the envelope and report σx')")
        self._input.returnPressed.connect(self._send)
        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.clicked.connect(self._send)
        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setToolTip(
            "Interrupt the current reply / tool run (Esc works too).")
        self._stop_btn.clicked.connect(self._on_stop)
        row.addWidget(self._mic_btn)
        row.addWidget(self._wake_btn)
        row.addWidget(self._input, stretch=1)
        row.addWidget(self._send_btn)
        row.addWidget(self._stop_btn)
        v.addLayout(row)

        opts = QHBoxLayout()
        self._auto = QCheckBox("Auto-approve compute (this session)")
        self._auto.stateChanged.connect(self._on_auto_toggle)
        self._speak = QCheckBox("Speak replies")
        self._speak.setToolTip(
            "Read a concise spoken summary of each reply aloud "
            "(exact numbers stay on screen).")
        self._fast_chk = QCheckBox("Instant commands")
        self._fast_chk.setToolTip(
            "Run unambiguous read-only requests ('status', 'show the "
            "RMS plot', tour 'next') directly — instant, no model "
            "round-trip.  Anything nuanced still goes to the model.")
        self._fast_chk.setChecked(
            str(self._settings().value("assist/fast_intents", "1")) == "1")
        self._fast_chk.toggled.connect(self._on_fast_toggled)
        self._narrate = QCheckBox("Narrate events")
        self._narrate.setToolTip(
            "When a machine event arrives while the assistant is idle "
            "(job finished, run-watch alert), give it a turn to react "
            "out loud instead of waiting for your next message.")
        self._watch_chk = QCheckBox("Watch runs")
        self._watch_chk.setToolTip(
            "Inspect every finished run (including ones YOU start) for "
            "transmission drops, σ blow-ups, emittance growth and "
            "baseline drift; alerts appear as events — pure local "
            "numpy, no tokens until something fires.")
        self._watch_chk.setChecked(
            self._settings().value("assist/watch_runs", "true") in
            (True, "true", "1"))
        self._watch_chk.toggled.connect(self._on_watch_toggled)
        self._prog = QLabel("")
        self._prog.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO};"
            f"font-size:10px;")
        # voice-only view: untick to hide the conversation text — the
        # orb, mic controls and confirm strip stay (user request)
        self._text_chk = QCheckBox("Text")
        self._text_chk.setToolTip(
            "Show the conversation text.  Untick for a voice-only view "
            "— the orb, voice controls and approval strip stay; the "
            "conversation keeps recording underneath.")
        self._text_chk.setChecked(
            str(self._settings().value("assist/show_text", "1")) == "1")
        self._text_chk.toggled.connect(self._on_text_toggled)
        opts.addWidget(self._auto)
        opts.addWidget(self._speak)
        opts.addWidget(self._fast_chk)
        opts.addWidget(self._narrate)
        opts.addWidget(self._watch_chk)
        opts.addWidget(self._text_chk)
        opts.addWidget(self._prog, stretch=1)
        # conversation text size — the user dials it, it persists
        self._font_minus = QPushButton("A−")
        self._font_plus = QPushButton("A+")
        for b, tip in ((self._font_minus, "smaller conversation text"),
                       (self._font_plus, "larger conversation text")):
            b.setToolTip(tip)
            b.setFixedWidth(44)
        self._font_minus.clicked.connect(self._on_font_smaller)
        self._font_plus.clicked.connect(self._on_font_bigger)
        opts.addWidget(self._font_minus)
        opts.addWidget(self._font_plus)
        v.addLayout(opts)

        # proactive run watching: every results_changed (assistant- OR
        # user-initiated run) is inspected by pure numpy; alerts ride
        # the machine-event channel
        self._run_watch = None
        try:
            self._state.results_changed.connect(self._on_results_changed)
        except Exception:                    # noqa: BLE001
            pass

        # voice engines (lazy, optional) — created on first use
        self._recorder = None
        self._tts = None
        self._voice_worker = None
        # hands-free listening stack (wake / follow-up / barge-in).
        # _tts_speaking is a THREAD-SAFE pause flag (set directly in the
        # Speaker's on_speaking callback, not only via the queued signal).
        self._mic_stream = None
        self._wake = None
        self._followup = None
        self._barge = None
        self._capture_tap = None
        self._wake_stt = None
        self._tts_speaking = False
        self._last_turn_was_voice = False
        self._pending_voice = False
        self._t_turn0 = None
        self._t_first_tok = None
        self._perf_parts = {}
        self._fast_worker = None
        self._fast_text = ""
        # never-drop inbox: utterances arriving mid-turn are held (small
        # FIFO — the one-slot version silently OVERWROTE the first
        # message when a second arrived) and run in order at turn end
        import collections as _co
        self._inbox: _co.deque = _co.deque(maxlen=3)
        self._barged_this_turn = False
        self._dismissed = False
        # everything the assistant SPOKE recently (post-normalization),
        # for the own-echo filter — the semantic backstop that breaks
        # the she-answers-herself loop no matter how the audio leaks in
        import collections as _coll
        self._spoken_recent: _coll.deque = _coll.deque()
        self._orb_state = "starting"
        self._responding_latch = False
        self._followup_lock = threading.Lock()
        self._runwatch_lock = threading.Lock()
        self._listen_gen = 0
        self._warned_no_speaker = False
        # turn watchdog: a wedged/dead worker must surface within
        # seconds, never as a silent forever-hang
        self._turn_open_ui = False
        self._stall_notified = False
        self._t_activity = 0.0
        self._turn_watchdog = QTimer(self)
        self._turn_watchdog.setInterval(5000)
        self._turn_watchdog.timeout.connect(self._check_turn_health)
        self._init_voice()

        self._append("[starting] warming up in the background (brain "
                     "+ voice models) — anything you type meanwhile is "
                     "queued, not lost.")
        try:                     # prove the VAD hears on THIS machine —
            from linac_gen.assist import vad as _vadmod
            _vadmod.self_test_async()            # else pure RMS gates
        except Exception:                        # noqa: BLE001
            pass
        # MIRAGE parity (2026-07-28): hands-free listening starts WITH
        # the assistant — the orb must react to the room immediately,
        # never sit gray until a button press.  Every relaunch used to
        # reset the toggle to OFF, so sessions began silently deaf and
        # the wake word "mostly didn't work".  Deferred one event-loop
        # turn so panel construction finishes first; the user's choice
        # is persisted (turning it off stays off).
        import os as _os
        if (str(self._settings().value("assist/wake_on", "1")) == "1"
                and _os.environ.get("HELIX_ASSIST_NO_PREWARM") != "1"):
            # NO_PREWARM (tests/soaks) also gates auto-listen: panels
            # constructed in tests were opening the REAL microphone
            QTimer.singleShot(0, self._auto_enable_wake)
        self.confirmation_needed.connect(self._show_confirmation)
        self.confirm_timed_out.connect(self._on_confirm_timeout)
        self.mic_died.connect(self._on_mic_died)
        self.event_line.connect(self._on_event_line)
        self.progress_changed.connect(self._on_progress)
        self.capture_saved.connect(self._on_capture_saved)
        self.wake_text.connect(self._on_voice_text)
        self.wake_status.connect(self._on_wake_status)
        self.navigate_requested.connect(self._on_navigate)
        self.plot_requested.connect(self._on_plot)
        self.highlight_requested.connect(self._on_highlight)
        self.cursor_requested.connect(self._on_cursor)
        self.gui_call.connect(self._on_gui_call)
        self.state_changed.connect(self._set_state)
        self.state_changed.connect(self._popup_on_wake)
        self.level_changed.connect(self._on_level)
        self.speaking_changed.connect(self._on_speaking)
        self._speaker = None                     # streaming TTS, lazy
        self._speak_pend = ""                    # unspoken streamed text
        self._spoke_streaming = False
        # streaming state must exist before the first _send (Stop/Esc can
        # fire on a fresh panel)
        self._stream_buf = ""
        self._streaming = False
        self._last_reply = ""
        self._speak_carry = ""
        self._speak_in_fence = False
        if not self._text_chk.isChecked():       # voice-only view persisted
            self._apply_text_visible(False)
        self._init_session()

    def _on_level(self, rms: float):
        if self._orb is not None:
            self._orb.set_level(rms)

    def _on_speaking(self, active: bool):
        # keep the orb in the voiceprint while audio actually plays
        if active:
            self._set_state("responding")
        elif not (self._worker and self._worker.isRunning()):
            self._set_state("idle")

    def _on_text_toggled(self, on: bool):
        try:
            self._settings().setValue("assist/show_text",
                                      "1" if on else "0")
        except Exception:                        # noqa: BLE001
            pass
        self._apply_text_visible(on)

    def _apply_text_visible(self, on: bool):
        """Voice-only view: hide the transcript (it keeps recording
        underneath) and hand its stretch to the ORB — the sphere fills
        the whole window, maximized included (user request: it used to
        sit small at the top of the freed space)."""
        if on:
            self._transcript.setVisible(True)
            self._vbox.setStretchFactor(self._transcript, 1)
            if self._orb is not None:
                self._vbox.setStretchFactor(self._orb, 0)
        else:
            self._transcript.setVisible(False)
            self._vbox.setStretchFactor(self._transcript, 0)
            if self._orb is not None:
                self._vbox.setStretchFactor(self._orb, 1)

    def _on_fast_toggled(self, on: bool):
        try:
            self._settings().setValue("assist/fast_intents",
                                      "1" if on else "0")
        except Exception:                        # noqa: BLE001
            pass

    def _perf_update(self, **parts):
        """Tiny latency HUD in the status row (QSettings-gated) —
        smoothness as a number, not a feeling."""
        try:
            if str(self._settings().value("assist/latency_hud",
                                          "1")) != "1":
                self._perf.setText("")
                return
            self._perf_parts.update(parts)
            self._perf.setText("   ".join(
                f"⏱ {v}" for v in self._perf_parts.values()))
        except Exception:                        # noqa: BLE001
            pass

    def _speaker_turn(self, opening: bool):
        """begin_turn/end_turn on the Speaker (None-safe): while a turn
        is open, sentence gaps in the streamed reply must not flap the
        speaking state (barge-in calibration + wake pause + orb)."""
        sp = self._speaker
        if sp is None:
            return
        try:
            (sp.begin_turn if opening else sp.end_turn)()
        except Exception:                        # noqa: BLE001
            pass

    def _get_speaker(self):
        """The pre-warmed Speaker, or None while it is still loading.
        NEVER constructs on the GUI thread — the kokoro backend loads a
        ~325 MB ONNX model, which must not freeze the UI."""
        return self._speaker

    def _prewarm_speaker(self):
        """Build the Speaker (and its heavy TTS backend) off-thread."""
        import threading as _th

        def _build():
            try:
                from functools import partial

                from linac_gen.assist.voice import Speaker
                sp = Speaker(on_speaking=self._on_speaking_threadsafe)
                self._speaker = sp
                # surface the live TTS backend in the mic tooltip
                self.gui_call.emit(partial(self._set_tts_tooltip,
                                           sp.backend_name))
            except Exception as exc:             # noqa: BLE001
                self._speaker = None
                # fail-loud: a dead speech engine must be VISIBLE, not
                # a mystery of missing audio
                self.wake_status.emit(
                    f"speech engine failed to load: {exc} — text only")

        _th.Thread(target=_build, daemon=True,
                   name="assist-tts-warmup").start()

    def _on_speaking_threadsafe(self, active: bool):
        """Player-thread callback: keep the pause flag current for the
        wake listener (no queued-signal latency — a stale flag would let
        the assistant wake itself), arm/disarm barge-in, open the
        follow-up window when a spoken reply finishes, then update the
        orb via the queued signal."""
        self._tts_speaking = bool(active)
        barge = self._barge
        if barge is not None:
            try:
                if active and self._mic_stream is not None:
                    barge.arm()
                elif not active:
                    barge.disarm()
            except Exception:                    # noqa: BLE001
                pass
        if not active:
            w = self._worker
            ap = self._approver
            # a worker BLOCKED in a confirmation counts as idle — the
            # spoken confirm echo must hand the mic back for the "yes"
            confirm_pending = ap is not None and ap.pending is not None
            if confirm_pending or not (w and w.isRunning()):
                self._open_followup_if_voice()
        self.speaking_changed.emit(bool(active))

    # -- generic run-on-GUI-thread bridge --------------------------------
    def _on_gui_call(self, fn):
        # queued slot: executes an arbitrary thunk on the GUI thread
        try:
            fn()
        except Exception:                        # noqa: BLE001
            pass

    def run_on_gui(self, fn, timeout: float = 3.0):
        """Run ``fn()`` on the GUI thread and return its result (or None on
        timeout).  Safe from any thread; direct call when already on the
        GUI thread (avoids self-deadlock)."""
        if self._closing:
            return None                          # shutting down: no round-trip
        import threading as _th
        from PyQt6.QtCore import QThread
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None or QThread.currentThread() is app.thread():
            try:
                return fn()
            except Exception:                    # noqa: BLE001
                return None
        ev = _th.Event()
        box = {}

        def _wrapped(_fn=fn):
            try:
                box["r"] = _fn()
            except Exception as exc:             # noqa: BLE001
                box["err"] = str(exc)
            finally:
                ev.set()

        self.gui_call.emit(_wrapped)
        if not ev.wait(timeout):
            return None
        return box.get("r")

    # -- state orb + tab navigation -------------------------------------
    def _set_state(self, state: str):
        """Drive the orb + lamp.  Must run on the GUI thread (callers on a
        worker thread emit ``state_changed`` instead of calling directly)."""
        self._orb_state = state
        if self._orb is not None:
            self._orb.set_state(state)
        if self._lamp is not None:
            colors = {"idle": theme.TEXT_2, "thinking": "#fbbf24",
                      "responding": theme.ACCENT, "listening": "#42a5f5",
                      "awaiting-confirm": "#f472b6", "error": "#ef4444"}
            self._lamp.setText("● " + state)
            self._lamp.setStyleSheet(
                f"color:{colors.get(state, theme.TEXT_2)}; font-size:11px;"
                f" font-weight:600;")

    def _on_navigate(self, tab: str, subtab: str = ""):
        # GUI thread (queued from the agent worker) — safe to touch widgets
        if self._app is not None:
            try:
                self._app.show_tab(tab, subtab or None)
            except Exception:                    # noqa: BLE001
                pass

    def _on_plot(self, key: str):
        if self._app is not None and hasattr(self._app, "show_result_plot"):
            try:
                self._app.show_result_plot(key)
            except Exception:                    # noqa: BLE001
                pass

    def _on_highlight(self, idx: int, s_mm: float):
        # GUI thread: switch to Lattice, select the element, move the cursor
        try:
            lat = self._state.lattice
            if lat is None or not (0 <= idx < len(lat.elements)):
                return
            if self._app is not None:
                self._app.show_tab("Lattice")
            self._state.set_selected(lat.elements[idx])
            self._state.set_s_cursor(float(s_mm))
        except Exception:                        # noqa: BLE001
            pass

    def _on_cursor(self, s_mm: float):
        try:
            self._state.set_s_cursor(float(s_mm))
        except Exception:                        # noqa: BLE001
            pass

    # -- voice (offline, optional) --------------------------------------
    def _init_voice(self):
        try:
            from linac_gen.assist import voice
        except Exception:                        # noqa: BLE001
            self._mic_btn.setEnabled(False)
            self._wake_btn.setEnabled(False)
            self._speak.setEnabled(False)
            return
        if not voice.stt_available():
            self._mic_btn.setEnabled(False)
            self._wake_btn.setEnabled(False)
            self._mic_btn.setToolTip(
                "Voice input needs the optional audio stack: "
                "pip install linac_gen[assist-voice]")
        else:
            # MIRAGE lesson the first port missed: warm Whisper the
            # moment the panel opens (background), not on first use —
            # a cold model load made the first push-to-talk 'take time
            # to listen'.  HELIX_ASSIST_NO_PREWARM=1 disables (tests).
            import os as _os
            if _os.environ.get("HELIX_ASSIST_NO_PREWARM") != "1":
                self._ensure_stt_warm()
        if not voice.tts_available():
            self._speak.setEnabled(False)
        else:
            self._speak.toggled.connect(self._on_speak_toggled)
            # pre-warm the Speaker at PANEL BUILD (off-thread — kokoro
            # loads a ~325 MB model): the first spoken confirm/reply
            # used to lose its audio because the speaker was still None
            import os as _os
            if _os.environ.get("HELIX_ASSIST_NO_PREWARM") != "1":
                self._prewarm_speaker()

    def _on_speak_toggled(self, on: bool):
        if on and self._speaker is None:
            self._prewarm_speaker()
        elif not on and self._speaker is not None:
            try:
                self._speaker.stop()
            except Exception:                    # noqa: BLE001
                pass

    def _mic_pressed(self):
        if self._recorder is not None or self._capture_tap is not None:
            return                               # already recording (re-entry)
        if self._speaker is not None:            # barge-in: cut the TTS
            self._barged_this_turn = True        # …and STAY quiet: late
            try:                                 # sentences must not talk
                self._speaker.stop()             # into the recording
            except Exception:                    # noqa: BLE001
                pass
        if self._followup is not None:           # PTT takes over the mic
            self._followup.cancel()
        # while wake listening owns the microphone, PTT records through a
        # TAP on that stream — never a second InputStream (CoreAudio
        # HAL-mutex rule); the wake toggle is frozen during the hold so
        # ownership cannot change mid-recording
        if self._mic_stream is not None and self._mic_stream.running:
            try:
                from linac_gen.assist.listen import CaptureTap
                self._wake_btn.setEnabled(False)
                self._capture_tap = CaptureTap(
                    self._mic_stream, on_level=self.level_changed.emit)
                self._capture_tap.start()
                self._mic_btn.setText("● rec")
                self._set_state("listening")
            except Exception as exc:             # noqa: BLE001
                self._append(f"[voice] microphone unavailable: {exc}")
                self._capture_tap = None
                self._wake_btn.setEnabled(True)
            return
        try:
            from functools import partial

            from linac_gen.assist.voice import PushToTalkRecorder
            rec = PushToTalkRecorder(
                on_level=self.level_changed.emit)   # queued → orb bars
            self._recorder = rec
            self._mic_btn.setText("● rec")
            self._set_state("listening")

            # start OFF the GUI thread: the CoreAudio open can block for
            # seconds and used to freeze the UI mid-press.  A release
            # racing the open is safe: stop() marks the recorder dead
            # and the reader closes the stream itself.
            def _start(r=rec):
                try:
                    r.start()
                except Exception as exc:         # noqa: BLE001
                    self.gui_call.emit(partial(
                        self._ptt_start_failed, r, str(exc)))

            threading.Thread(target=_start, daemon=True,
                             name="assist-ptt-start").start()
        except Exception as exc:                 # noqa: BLE001
            self._append(f"[voice] microphone unavailable: {exc}")
            self._recorder = None

    def _ptt_start_failed(self, rec, err: str):
        if self._recorder is not rec:
            return                               # already released/replaced
        self._recorder = None
        self._mic_btn.setText("🎤 Hold")
        self._set_state("idle")
        self._append(f"[voice] microphone unavailable: {err}")
        self._append("  · if macOS never asked for permission: "
                     "System Settings → Privacy & Security → "
                     "Microphone → allow this app, then relaunch.")

    def _mic_released(self):
        self._mic_btn.setText("🎤 Hold")
        tap, self._capture_tap = self._capture_tap, None
        if tap is not None:                      # wake-mode PTT via tap
            self._wake_btn.setEnabled(True)
            audio = tap.stop()
            self._set_state("thinking")
            self._transcribe(audio)
            return
        rec, self._recorder = self._recorder, None
        if rec is None:
            self._set_state("idle")
            return
        self._set_state("thinking")              # stopping + transcribing
        self._transcribe(None, recorder=rec)     # stop() runs off-GUI

    def _transcribe(self, audio, recorder=None):
        # transcribe off the GUI thread; the transcript lands in the input
        # box and AUTO-SENDS (fluid voice flow) — it stays visible in the
        # transcript, and compute/mutate actions still confirm before
        # running, so a misheard utterance cannot act unreviewed
        self._voice_worker = _TranscribeWorker(audio, self,
                                               recorder=recorder)
        self._voice_worker.transcribed.connect(self._on_transcribed)
        self._voice_worker.failed.connect(self._on_voice_failed)
        self._voice_worker.too_short.connect(self._on_too_short)
        self._voice_worker.start()

    def _on_too_short(self, seconds: float):
        self._append("[voice] (too short — keep holding 🎤 while "
                     "you speak)")
        self._set_state("idle")

    def _on_voice_failed(self, msg: str):
        self._append(f"[voice] {msg}")
        self._set_state("idle")

    def _on_transcribed(self, text: str):
        stt = getattr(self._voice_worker, "stt_s", None)
        if stt is not None:
            self._perf_update(stt=f"STT {stt:.1f} s")
        text = (text or "").strip()
        if not text:
            self._append("[voice] (nothing heard)")
            self._set_state("idle")
            return
        self._on_voice_text(text)

    # -- own-echo filter (the she-answers-herself loop breaker) ----------
    def _note_spoken(self, text: str):
        """Remember what WE just said (any speak path) — captured audio
        that transcribes to mostly these words is our own echo."""
        import time as _time
        t = (text or "").strip()
        if not t:
            return
        try:                 # store what the TTS will actually SAY —
            from linac_gen.assist.voice import speechify
            t = speechify(t)         # numerals as words etc., so the
        except Exception:            # echo's transcript matches
            pass
        d = self._spoken_recent
        d.append((_time.monotonic(), t))
        while len(d) > 80:
            d.popleft()

    def _is_own_echo(self, text: str) -> bool:
        """True when the utterance is a CONTIGUOUS replay of something
        the assistant itself spoke in the last ~10 s (4-gram phrase
        match, ≥5 content words).  v1 used a 45 s token-SET union at
        80% membership — it ate every legitimate reply that reused her
        words, e.g. picking "show the transmission plot" from a menu
        she had just offered.  A real echo replays a contiguous
        stretch; a menu pick recombines words in a new order."""
        import re as _re
        import time as _time
        toks = _re.findall(r"[a-z']+", (text or "").lower())
        content = [t for t in toks if len(t) > 2]
        if len(content) < 5:
            return False                 # short replies always pass
        grams = {" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)}
        if not grams:
            return False
        now = _time.monotonic()
        # 30 s window: entries are stamped at QUEUE time but a long
        # reply takes 20-30 s to actually play — 10 s missed the tail.
        # Threshold 0.35 with >=2 gram hits: ONE Whisper mis-hearing in
        # a 12-token echo destroys 4 of 9 grams (ratio 0.56) — 0.6 let
        # the loop straight back in; a menu-pick recombination shares
        # ~0 contiguous grams, so 0.35 stays safe for real replies.
        for ts, s in reversed(self._spoken_recent):
            if now - ts > 30.0:
                break
            stoks = _re.findall(r"[a-z']+", s.lower())
            sg = {" ".join(stoks[i:i + 4])
                  for i in range(len(stoks) - 3)}
            if not sg:
                continue
            hit = len(grams & sg)
            if hit >= 2 and hit / len(grams) >= 0.35:
                return True
        return False

    # -- hands-free listening stack --------------------------------------
    def _on_voice_text(self, text: str):
        """Every spoken utterance (PTT, wake, follow-up) routes here.
        A pending confirmation is resolved LOCALLY (yes/no words — the
        model is never in that loop); anything else auto-sends."""
        text = (text or "").strip()
        if not text:
            return
        if self._is_own_echo(text):
            self._append(f"[voice] (ignored own echo: «{text[:48]}»)")
            return
        ap = self._approver
        if ap is not None and ap.pending is not None:
            from linac_gen.assist.agent import Decision
            from linac_gen.assist.listen import interpret_confirm
            verdict = interpret_confirm(text)
            if verdict == "yes":
                self._append(f"[voice] «{text}» → approved")
                self._resolve_confirm(Decision.APPROVE)
            elif verdict == "no":
                self._append(f"[voice] «{text}» → denied")
                self._resolve_confirm(Decision.DENY)
            else:
                self._speak_line("Please say confirm, or cancel.")
                # open the mic NOW only when nothing will be spoken —
                # otherwise the TTS drain opens it (opening while our
                # own voice plays let the echo's "cancel" self-deny)
                if not (self._speak.isChecked()
                        and self._get_speaker() is not None):
                    fl = self._ensure_followup()
                    if fl is not None:
                        self._open_fl_window(fl)
            return
        self._pending_voice = True
        self._input.setText(text)
        self._send()

    def _speak_line(self, text: str):
        sp = self._get_speaker()
        if sp is not None and self._speak.isChecked():
            try:
                sp.say(text)
                self._note_spoken(text)
            except Exception:                    # noqa: BLE001
                pass

    def _tour_active(self) -> bool:
        try:
            from linac_gen.assist.guide import get_state
            return bool(self._session is not None
                        and get_state(self._session.context).active)
        except Exception:                        # noqa: BLE001
            return False

    def _open_followup_if_voice(self):
        """Reopen the mic briefly after a spoken reply to a VOICE turn
        (or after a spoken confirm echo) — natural back-and-forth.
        During an ACTIVE guided tour the window opens after EVERY
        station no matter how the turn started: the tour IS a
        back-and-forth ('next', 'back', questions) — clicking the mic
        once per station killed it."""
        if self._closing:
            return
        if self._dismissed:
            return          # user ✕-dismissed mid-turn: no hot mic —
                            # saying "HELIX" (or reopening) re-engages.
                            # NOT isVisible(): the warm panel legitimately
                            # runs voice turns while never yet shown.
        sticky = False
        try:
            # STICKY voice mode: while hands-free (👂) is on, a typed
            # turn no longer silently ends the voice loop — the mic is
            # armed anyway, so the natural back-and-forth continues
            sticky = self._wake_btn.isChecked()
        except RuntimeError:
            pass
        if (not self._last_turn_was_voice and not sticky
                and not self._tour_active()):
            return
        fl = self._ensure_followup()
        if fl is not None:
            self._open_fl_window(fl)

    def _open_fl_window(self, fl) -> bool:
        """Open a follow-up window AND make the open mic visible: the
        orb shows the listening hue while the window is armed (the user
        could never tell whether the mic was open)."""
        try:
            ok = fl.open_window()
        except Exception:                        # noqa: BLE001
            return False
        if ok:
            self.state_changed.emit("listening")
        return ok

    def _on_fl_timeout(self):
        # follow-up thread: window expired silently before — reflect it
        self.state_changed.emit("idle")
        self.wake_status.emit("say 'HELIX'")

    def _ensure_followup(self):
        if self._followup is not None:
            return self._followup
        try:
            # vad.make() BEFORE the lock (it may wait for the self-test
            # proof off-GUI; holding the build lock through that would
            # convoy other callers)
            from linac_gen.assist import vad as _vadmod
            from linac_gen.assist.listen import (
                FollowUpListener, TransientMic, _SharedSource,
            )
            from linac_gen.assist.voice import WhisperSTT
            vad = _vadmod.make()
        except Exception:                        # noqa: BLE001
            return None

        def _source():
            ms = self._mic_stream
            if ms is not None and ms.running:
                return _SharedSource(ms)
            return TransientMic()

        # LOCKED build: GUI thread + TTS player thread both come through
        # here — a race once built two listeners over two streams (the
        # CoreAudio HAL-mutex deadlock)
        def _speaker_audible():
            sp = self._speaker
            try:
                return sp is not None and sp.busy()
            except Exception:                    # noqa: BLE001
                return False

        with self._followup_lock:
            if self._followup is None:
                try:
                    self._followup = FollowUpListener(
                        self._wake_stt or WhisperSTT(),
                        on_text=self.wake_text.emit,
                        source_factory=_source,
                        on_status=self.wake_status.emit,
                        on_level=self.level_changed.emit,
                        on_timeout=self._on_fl_timeout,
                        vad=vad,                 # None -> RMS gates
                        busy_probe=_speaker_audible)
                except Exception as exc:         # noqa: BLE001
                    self._followup = None
                    self.wake_status.emit(
                        f"follow-up listening unavailable: {exc}")
            return self._followup

    def _auto_enable_wake(self):
        # bound method on a timer (never a lambda — destroyed-widget rule)
        try:
            if not self._closing:
                self._wake_btn.setChecked(True)
        except RuntimeError:
            pass

    def _on_wake_toggled(self, on: bool):
        try:
            self._settings().setValue("assist/wake_on", "1" if on else "0")
        except Exception:                        # noqa: BLE001
            pass
        if on:
            self._start_listening_async()
        else:
            self._listen_gen += 1                # invalidate an in-flight start
            self._shutdown_listening()
            self._prog.setText("")
            self.state_changed.emit("off")       # orb: gray, not hearing

    def _start_listening_async(self):
        """Build the hands-free stack OFF the GUI thread.  CoreAudio's
        InputStream open can block for seconds (device wake, permission
        prompt, HAL contention) — it used to run in the toggle handler
        and freeze the whole UI at every launch."""
        self._listen_gen += 1
        gen = self._listen_gen
        self._prog.setText("starting hands-free listening …")
        try:
            from linac_gen.assist.voice import WhisperSTT
            self._wake_stt = self._wake_stt or WhisperSTT()
            self._prewarm_stt()
        except Exception:                        # noqa: BLE001
            pass

        def _build():
            from functools import partial
            try:
                from linac_gen.assist import vad as _vadmod
                from linac_gen.assist.listen import MicStream
                vad = _vadmod.make()             # off-GUI: may wait for proof
                # the barge listener gets its OWN instance: silero
                # carries recurrent state, and sharing one model between
                # the wake tap and the barge tap fed every block twice —
                # smearing the probabilities exactly when the user spoke
                barge_vad = _vadmod.make() if vad is not None else None
                mic = MicStream(on_died=self.mic_died.emit)
                mic.on_overflow = self._on_mic_overflow
                mic.open()                       # the blocking CoreAudio call
            except Exception as exc:             # noqa: BLE001
                import traceback as _tb
                _tb.print_exc()      # land the real error in the launch log
                try:
                    self.gui_call.emit(partial(self._listen_start_failed,
                                               gen, str(exc)))
                except RuntimeError:             # panel died while we built
                    pass
                return
            try:
                self.gui_call.emit(partial(self._listen_started,
                                           gen, mic, vad, barge_vad))
            except RuntimeError:
                # panel destroyed while the mic was opening: WE still
                # own the stream — close it, never leak a hot mic
                try:
                    mic.close()
                except Exception:                # noqa: BLE001
                    pass

        threading.Thread(target=_build, daemon=True,
                         name="assist-listen-start").start()

    def _listen_started(self, gen: int, mic, vad, barge_vad=None):
        """GUI thread: adopt the freshly opened mic (or discard it if the
        toggle flipped / panel began closing while it was starting)."""
        if (gen != self._listen_gen or self._closing
                or not self._wake_btn.isChecked()):
            threading.Thread(target=mic.close, daemon=True,
                             name="assist-mic-discard").start()
            return
        try:
            from linac_gen.assist.listen import (
                BargeListener, WakeListener, barge_in_enabled,
            )
            self._mic_stream = mic
            self._reopen_attempt = 0             # recovery backoff reset
            self._wake = WakeListener(
                mic, self._wake_stt,
                on_command=self.wake_text.emit,
                is_paused=self._listening_paused,
                on_status=self.wake_status.emit,
                on_wake=self._on_wake_word,
                on_level=self.level_changed.emit,
                vad=vad)                         # None -> RMS gates
            self._wake.start()
            self.state_changed.emit("idle")      # orb: alive + hearing
            if barge_in_enabled():
                self._barge = BargeListener(on_barge=self._on_barge,
                                            vad=barge_vad)
                mic.subscribe(self._barge.feed)
            if vad is None:
                # fail-loud: RMS-only wake is a real degradation the
                # user must see — and RETRY in the background so a lost
                # first-launch self-test race no longer locks the whole
                # session into loudness-only hearing
                self._prog.setText("VAD warming up — wake uses loudness "
                                   "gates until it lands")
                threading.Thread(target=self._vad_hot_swap,
                                 args=(gen,), daemon=True,
                                 name="assist-vad-hotswap").start()
        except Exception as exc:                 # noqa: BLE001
            self._listen_start_failed(gen, str(exc))

    def _vad_hot_swap(self, gen: int):
        import time as _time
        from linac_gen.assist import vad as _vadmod
        for _ in range(6):
            _time.sleep(3.0)
            if gen != self._listen_gen or self._closing:
                return
            v = _vadmod.make()
            if v is not None:
                # a ref swap is NOT enough: the wake thread chose
                # cadence-vs-event mode once at start — rebuild the
                # stack so event-driven wake actually engages
                self.gui_call.emit(self._restart_listening_for_vad)
                return

    def _restart_listening_for_vad(self):
        try:
            if not self._closing and self._wake_btn.isChecked():
                self.wake_status.emit("voice detection ready — "
                                      "re-arming the mic")
                self._listen_gen += 1
                self._shutdown_listening()
                self._start_listening_async()
        except RuntimeError:
            pass

    def _listen_start_failed(self, gen: int, err: str):
        if gen != self._listen_gen or self._closing:
            return
        self._append(f"[voice] wake listening unavailable: {err}")
        self._shutdown_listening()
        # RETRY with backoff (2→30 s) instead of giving up after one
        # attempt: a device mid-switch cured itself seconds later while
        # the assistant stayed permanently deaf
        self._reopen_attempt = getattr(self, "_reopen_attempt", 0) + 1
        if self._reopen_attempt <= 5:
            delay = min(30, 2 ** self._reopen_attempt)
            self._prog.setText(f"mic unavailable — retrying in {delay} s")
            QTimer.singleShot(delay * 1000, self._retry_listen_start)
        else:
            self._append("  · if macOS never asked for permission: "
                         "System Settings → Privacy & Security → "
                         "Microphone → allow this app, then "
                         "relaunch.")
            self._wake_btn.setChecked(False)

    def _retry_listen_start(self):
        try:
            if (not self._closing and self._wake_btn.isChecked()
                    and self._mic_stream is None):
                self._start_listening_async()
        except RuntimeError:
            pass

    def _on_mic_overflow(self):
        # reader thread, first overflow only → visible once
        self.wake_status.emit("mic overloaded — audio may be choppy")

    def _on_mic_died(self):
        """The wake stream's reader exited on a device error (sleep/
        wake, headset change) or the data watchdog saw it starve.
        Without this the hands-free stack goes SILENTLY deaf.  Recover
        by cycling the toggle — DEBOUNCED 1.5 s so device churn settles
        first (an instant reopen lands on the device mid-switch and
        dies again, looping)."""
        if self._closing or self._mic_stream is None:
            return
        self._append("[voice] microphone stream lost (device change or "
                     "sleep) — reopening")
        self._wake_btn.setChecked(False)
        QTimer.singleShot(1500, self._reopen_wake)

    def _reopen_wake(self):
        try:
            if not self._closing and not self._wake_btn.isChecked():
                self._wake_btn.setChecked(True)
        except RuntimeError:
            pass

    def _ensure_stt_warm(self):
        """Create the shared WhisperSTT and load its model off-thread."""
        try:
            from linac_gen.assist.voice import WhisperSTT
            if self._wake_stt is None:
                self._wake_stt = WhisperSTT()
            self._prewarm_stt()
        except Exception:                        # noqa: BLE001
            pass

    def _prewarm_stt(self):
        """First transcription loads the Whisper model (~15 s) — warm it
        off-thread so the first press/wake isn't swallowed."""
        import threading as _th
        import numpy as _np
        stt = self._wake_stt
        if stt is None or getattr(stt, "ready", False):
            return

        def _warm():
            try:
                stt.transcribe(_np.zeros(1600, dtype="float32"))
            except Exception:                    # noqa: BLE001
                pass

        _th.Thread(target=_warm, daemon=True,
                   name="assist-stt-warmup").start()

    def _listening_paused(self) -> bool:
        """Thread-safe pause gate for wake detection: never listen for
        the wake word while WE are speaking, PTT records, or a follow-up
        window owns the conversation.

        Speaking is DERIVED from the live Speaker (busy()), not from the
        stored flag — one missed on_speaking(False) once wedged the gate
        (and with it the whole hands-free stack) forever."""
        fl = self._followup
        sp = self._speaker
        if sp is not None:
            try:
                speaking = sp.busy()
            except Exception:                    # noqa: BLE001
                speaking = self._tts_speaking
        else:
            speaking = self._tts_speaking
        return (speaking or self._recorder is not None
                or self._capture_tap is not None
                or (fl is not None and fl.active))

    def _popup_on_wake(self, state: str) -> None:
        """Wake word while the panel is hidden (app-startup warm panel)
        → bring the conversation into view."""
        if state == "listening" and not self.isVisible():
            self.show()
            self.raise_()

    def _on_wake_word(self):
        # reader-thread callback → queued signal for the orb, plus a
        # visible acknowledgment so a heard wake is never a mystery
        self.state_changed.emit("listening")
        self.gui_call.emit(self._ack_wake)

    def _ack_wake(self):
        self._append("[voice] heard 'HELIX' — go ahead")

    def _on_barge(self):
        # spoken-over: cut the TTS, ABANDON the answer, and hand the
        # floor to the human.  Muting alone kept the old turn computing
        # silently while the user's redirect waited in the queue — the
        # "does not gear well" experience.
        self._barged_this_turn = True
        sp = self._get_speaker()
        if sp is not None:
            try:
                sp.stop()
            except Exception:                    # noqa: BLE001
                pass
        if self._session is not None:
            try:
                self._session.request_stop()     # interrupting = redirect
            except Exception:                    # noqa: BLE001
                pass
        self.gui_call.emit(self._note_interrupted)
        self._last_turn_was_voice = True
        fl = self._ensure_followup()
        if fl is not None:
            self._open_fl_window(fl)

    def _note_interrupted(self):
        self._append("  · interrupted — go ahead")

    def _on_wake_status(self, text: str):
        self._last_wake_status = text
        self._prog.setText(text)
        # an EMPTY wake capture leaves no command to advance the state —
        # when the listener re-arms, pull the orb out of "listening"
        if (text.startswith("say 'HELIX'")
                and self._orb_state == "listening"):
            self._set_state("idle")

    # -- proactive run watching ------------------------------------------
    def _on_watch_toggled(self, on: bool):
        self._settings().setValue("assist/watch_runs",
                                  "true" if on else "false")

    def _get_run_watch(self):
        if self._run_watch is None:
            try:
                from linac_gen.assist.watch import RunWatch

                def _baseline():
                    import json as _js
                    import os as _os
                    if self._session is None:
                        return None
                    from linac_gen.assist.tools_analysis import (
                        _baseline_path,
                    )
                    p = _baseline_path(self._session.context)
                    if not _os.path.isfile(p):
                        return None
                    with open(p, encoding="utf-8") as f:
                        return _js.load(f)

                self._run_watch = RunWatch(baseline_loader=_baseline)
            except Exception:                # noqa: BLE001
                self._run_watch = None
        return self._run_watch

    def _on_results_changed(self, results):
        try:
            if (results is None or self._closing
                    or not self._watch_chk.isChecked()
                    or self._session is None):
                return
            watch = self._get_run_watch()
            if watch is None:
                return
            sess = self._session

            # numpy over full MP results — OFF the GUI thread (it used
            # to run inline in this slot and stall the UI after every run)
            def _inspect(w=watch, res=results, s=sess):
                try:
                    from linac_gen.assist.tools_analysis import _identity
                    with self._runwatch_lock:    # RunWatch mutates its
                        alerts = w.inspect(      # baselines — serialize
                            res, identity=_identity(s.context))
                    for a in alerts:
                        s.submit_event(a)
                except Exception:                # noqa: BLE001
                    pass

            threading.Thread(target=_inspect, daemon=True,
                             name="assist-runwatch").start()
        except RuntimeError:
            # widget destroyed while the state signal outlived us
            pass
        except Exception:                    # noqa: BLE001
            pass

    def _shutdown_listening(self):
        """Teardown order matters: passive analyzers first, then the
        listeners, then the ONE stream (its reader closes it)."""
        if self._barge is not None:
            try:
                self._barge.disarm()
                if self._mic_stream is not None:
                    self._mic_stream.unsubscribe(self._barge.feed)
            except Exception:                    # noqa: BLE001
                pass
            self._barge = None
        if self._followup is not None:
            try:
                self._followup.cancel()
            except Exception:                    # noqa: BLE001
                pass
            self._followup = None
        if self._wake is not None:
            try:
                self._wake.shutdown()
            except Exception:                    # noqa: BLE001
                pass
            self._wake = None
        if self._mic_stream is not None:
            try:
                self._mic_stream.close()
            except Exception:                    # noqa: BLE001
                pass
            self._mic_stream = None

    def _maybe_speak(self, reply: str):
        if not self._speak.isChecked() or self._barged_this_turn:
            return
        try:
            from linac_gen.assist.voice import summarize_for_speech
            sp = self._get_speaker()
            if sp is not None:
                spoken = summarize_for_speech(reply)
                sp.say(spoken)
                self._note_spoken(spoken)
        except Exception:                        # noqa: BLE001
            pass

    # -- settings persistence (QSettings — the key stays on THIS machine)
    def _settings(self):
        from linac_gen_gui.interphase.app_settings import make_settings
        return make_settings("Linac_Gen", "Interphase")

    def _load_saved_config(self):
        """Build an AssistConfig from GUI-entered settings, if any.
        assist stays Qt-free — the panel passes an explicit config."""
        from linac_gen.assist.config import AssistConfig
        s = self._settings()
        provider = s.value("assist/provider", "") or ""
        key = s.value("assist/api_key", "") or ""
        base = s.value("assist/base_url", "") or ""
        model = s.value("assist/model", "") or ""
        # pre-fill the fields for visibility (key stays password-masked)
        idx = self._provider.findData(provider or "anthropic")
        if idx >= 0:
            self._provider.setCurrentIndex(idx)
        if key:
            self._key_edit.setText(key)
        if model:
            self._model_edit.setCurrentText(model)
        if provider == "anthropic" and key:
            return AssistConfig(provider="anthropic",
                                model=model or "claude-sonnet-5",
                                api_key=key)
        if provider == "openai" and base:
            return AssistConfig(provider="openai",
                                model=model or "llama3.1",
                                api_key=key, base_url=base)
        if provider == "claude_sdk":      # keyless — Claude Code login
            return AssistConfig(provider="claude_sdk", model=model)
        return None

    def _on_show_backend(self):
        """Re-open the provider row (hidden after a successful connect)
        so the backend can be changed; Connect performs the switch (the
        running session is closed and replaced)."""
        self._settings_box.setVisible(True)
        if self._provider.currentData() == "openai":
            base = self._settings().value("assist/base_url", "") or ""
            if base:
                self._fetch_local_models(base)

    # -- model dropdown (editable combo; free text always wins) ---------
    def _repopulate_models(self, provider, extra=()):
        """Fill the dropdown with per-backend suggestions (+ the saved
        model and any server-reported ids) without touching the typed
        text — a blank first item keeps 'blank = default'."""
        cur = self._model_edit.currentText()
        items = [""]
        items += self._MODEL_SUGGESTIONS.get(provider, [])
        for m in extra:
            if m and m not in items:
                items.append(m)
        saved = self._settings().value("assist/model", "") or ""
        if saved and saved not in items:
            items.append(saved)
        self._model_edit.blockSignals(True)
        self._model_edit.clear()
        self._model_edit.addItems(items)
        self._model_edit.setCurrentText(cur)
        self._model_edit.blockSignals(False)

    def _on_provider_changed(self, _idx):
        provider = self._provider.currentData()
        self._repopulate_models(provider)
        if provider == "openai":
            base = self._settings().value("assist/base_url", "") or ""
            if base:
                self._fetch_local_models(base)

    def _fetch_local_models(self, base):
        """Ask an OpenAI-compatible server for its model list (ollama:
        GET <base>/models) on a daemon thread; results arrive through the
        queued ``models_listed`` signal.  Failures are silent — the
        dropdown simply keeps its curated entries."""
        import threading

        def _run(sig=self.models_listed, base=base):
            try:
                import json as _json
                import urllib.request
                url = base.rstrip("/") + "/models"
                with urllib.request.urlopen(url, timeout=2.0) as r:
                    data = _json.load(r)
                ids = sorted(str(d.get("id", ""))
                             for d in data.get("data", []) if d.get("id"))
                if ids:
                    sig.emit(base, ids)
            except Exception:                    # noqa: BLE001
                pass                             # incl. panel torn down
        threading.Thread(target=_run, daemon=True,
                         name="assist-model-list").start()

    def _on_models_listed(self, base, ids):
        if self._provider.currentData() != "openai":
            return                               # stale — backend changed
        self._repopulate_models("openai", extra=ids)
        self._status.setText(f"local server offers {len(ids)} model(s) — "
                             "see the model dropdown")

    def _on_connect(self):
        provider = self._provider.currentData()
        key = self._key_edit.text().strip()
        model = self._model_edit.currentText().strip()
        s = self._settings()
        s.setValue("assist/provider", provider)
        s.setValue("assist/api_key", key)        # local store only
        if provider != "openai":                 # openai saves PARSED model
            s.setValue("assist/model", model)    # (box may hold "URL model")
        # for the local backend the "model" field accepts "URL",
        # "model" (URL from saved settings), or "URL model" in one go
        from linac_gen.assist.config import AssistConfig
        if provider == "anthropic":
            if not key:
                self._status.setText("enter an Anthropic API key first")
                return
            cfg = AssistConfig(provider="anthropic",
                               model=model or "claude-sonnet-5",
                               api_key=key)
        elif provider == "claude_sdk":
            # keyless: reuse the Claude Code login. Verify it's reachable
            # before starting so the user gets guidance, not a dead session.
            from linac_gen.assist.sdk_backend import sdk_available
            ok, why = sdk_available()
            if not ok:
                self._status.setText(why)
                self._append("[setup] " + why)
                return
            cfg = AssistConfig(provider="claude_sdk", model=model)
        else:
            # One box, two facts: accept "URL", "model", or "URL model".
            base = s.value("assist/base_url", "") or ""
            mdl = s.value("assist/model", "") or ""
            if model.startswith("http"):
                parts = model.split()
                base = parts[0]
                if len(parts) > 1:
                    mdl = parts[1]
            elif model:
                mdl = model
            if not base:
                self._status.setText(
                    "for a local backend put the server URL in the model "
                    "box — 'http://localhost:11434/v1' or "
                    "'http://localhost:11434/v1 qwen2.5' (URL + model)")
                return
            s.setValue("assist/base_url", base)
            s.setValue("assist/model", mdl)      # parsed, not the raw box
            cfg = AssistConfig(provider="openai", model=mdl or "llama3.1",
                               api_key=key, base_url=base)
        self._start_session(cfg)

    # -- session bring-up (surfaces the optionality contract) -----------
    def _init_session(self, provider=None):
        try:
            import linac_gen.assist as assist   # noqa: F401
        except Exception as exc:                 # noqa: BLE001
            self._status.setText(f"assistant unavailable: {exc}")
            self._set_enabled(False)
            return
        if provider is not None:                 # test injection (mock)
            self._start_session(None, provider=provider)
            return
        # GUI-entered settings first, then environment
        cfg = self._load_saved_config()
        if cfg is None:
            from linac_gen.assist.config import resolve_config
            cfg = resolve_config()
        if cfg is None:
            self._status.setText("Enter an Anthropic API key above and "
                                 "click Connect (or a local server URL).")
            self._append("[setup] No backend configured yet — paste "
                         "your key in the field above; it is stored on "
                         "this machine and never sent anywhere except "
                         "the provider you choose.")
            self._set_enabled(False)
            return
        if cfg.provider == "claude_sdk":     # keyless — verify the login
            from linac_gen.assist.sdk_backend import sdk_available
            ok, why = sdk_available()
            if not ok:
                self._status.setText(why)
                self._append("[setup] " + why)
                self._set_enabled(False)
                return
        self._start_session(cfg)

    def _start_session(self, cfg, provider=None):
        from linac_gen.assist.agent import AgentSession, Decision
        # a confirmation still blocking the OLD session's worker must
        # resolve now, or that executor thread waits forever (leak that
        # also blocks the SDK loop's shutdown at quit)
        if self._approver is not None and self._approver.pending is not None:
            try:
                self._set_confirm_visible(False)
                self._approver.resolve(Decision.ABORT)
            except Exception:                    # noqa: BLE001
                pass
        if self._session is not None:            # replace: don't leak the old
            try:                                 # session's SDK thread/process
                self._session.close()
            except Exception:                    # noqa: BLE001
                pass
            self._session = None
        try:
            from linac_gen_gui.interphase.app import _resolve_calc_dir
            calc_dir = str(_resolve_calc_dir())
        except Exception:                        # noqa: BLE001
            calc_dir = "runs"
        ctx = _make_context(self._state, calc_dir, nav=self)
        self._approver = _GuiApprover(
            self, timeout_s=getattr(cfg, "confirm_timeout_s", 0.0)
            if cfg else 0.0)
        self._session = AgentSession(
            cfg, ctx, approver=self._approver, provider=provider,
            on_event=self._on_event_threadsafe)
        # prewarm the SDK client OFF-thread: the first command of every
        # launch used to pay the 1.5–5 s CLI startup inside ask()
        import os as _os
        sdk = getattr(self._session, "_sdk", None)
        if (sdk is not None
                and _os.environ.get("HELIX_ASSIST_NO_PREWARM") != "1"):
            # gated: the prewarm spawned REAL claude CLI processes in
            # every test that constructs a panel (the constructor reads
            # the user's saved claude_sdk settings before mock injection)
            def _warm_sdk(b=sdk):
                try:
                    b._ensure_started()
                except Exception:                # noqa: BLE001
                    pass                         # surfaces on first ask
            threading.Thread(target=_warm_sdk, daemon=True,
                             name="assist-sdk-prewarm").start()
        self._set_enabled(True)
        self._settings_box.setVisible(False)     # hide once connected
        who = (cfg.provider + f" ({cfg.model})") if cfg else "mock"
        self._status.setText(f"provider: {who}   ·   ledger: "
                             f"{self._session.ledger.path.name}")
        self._append("[ready] Ask me to run simulations, matches, or "
                     "query results.  Compute and mutate actions ask "
                     "for your confirmation first.")
        try:
            first = not bool(self._settings().value("assist/welcomed"))
        except Exception:                        # noqa: BLE001
            first = False
        if first:
            self._append(
                "[hello] A quick map of what I can do — run envelope/MP "
                "simulations, matches and scans · read your plots with "
                "vision · tune knobs and run multi-step campaigns (each "
                "mutation confirms first) · execute Python analysis in "
                "a sandbox · watch runs and speak up when something "
                "drifts · keep a lab notebook across sessions · give a "
                "guided tour (🎓) · run hidden-fault training drills "
                "(🧩).  Voice: hold 🎤 or SPACE to talk, or toggle 👂 "
                "and just say 'HELIX'.")
            try:
                self._settings().setValue("assist/welcomed", 1)
            except Exception:                    # noqa: BLE001
                pass

    def _set_enabled(self, on: bool):
        self._input.setEnabled(on)
        self._send_btn.setEnabled(on)

    # -- messaging ------------------------------------------------------
    def _append(self, text: str):
        self._transcript.appendPlainText(text)

    def _on_chip_clicked(self):
        btn = self.sender()
        prompt = btn.property("chip_prompt") if btn is not None else None
        if prompt:
            self._input.setText(str(prompt))
            self._send()

    def _set_tts_tooltip(self, backend_name: str):
        self._mic_btn.setToolTip(
            "Push-to-talk: press and hold to speak, release to send "
            "(offline: faster-whisper).  Or hold SPACE with the input "
            f"line unfocused.  TTS: {backend_name}.")

    def _send(self):
        text = self._input.text().strip()
        if not text:
            return
        if self._session is None:
            # NEVER swallow a command silently (a dead backend used to
            # eat wake+command with zero feedback of any kind).  The
            # text STAYS in the box, ready to send once connected.
            self._append("[error] the assistant backend is not "
                         "connected — open backend… and press Connect")
            self._speak_line("The assistant backend is not connected.")
            return
        busy = False
        try:
            busy = bool(self._worker and self._worker.isRunning())
        except RuntimeError:
            busy = False
        if busy:
            # NEVER silently drop input mid-turn.  FIFO of 3: the old
            # one-slot version destroyed message A when B arrived.
            if len(self._inbox) == self._inbox.maxlen:
                lost = self._inbox.popleft()[0]
                self._append(f"  · inbox full — dropped «{lost[:40]}»")
            self._inbox.append((text, self._pending_voice))
            self._pending_voice = False
            self._input.clear()
            self._append(f"\n▷ {text}    · queued "
                         f"({len(self._inbox)}) — runs after this turn")
            return
        # a voice-initiated turn earns a follow-up window after the
        # spoken reply; a typed turn never opens the mic
        self._last_turn_was_voice = self._pending_voice
        self._pending_voice = False
        self._input.clear()
        self._append(f"\n▶ {text}")
        import time as _time
        self._t_turn0 = _time.monotonic()
        self._t_first_tok = None
        if self._fast_chk.isChecked() and self._try_fast(text):
            return                               # instant, no LLM turn
        self._dispatch_model(text)

    def _try_fast(self, text: str) -> bool:
        """Instant commands: unambiguous read-only requests (and tour
        navigation) execute directly — no LLM round trip.  Returns True
        when the intent was taken; nuance falls back to the model."""
        try:
            from linac_gen.assist.intents import match
            fi = match(text, self._session.context)
        except Exception:                        # noqa: BLE001
            fi = None
        if fi is None:
            return False
        self._set_state("thinking")
        self._fast_text = text
        w = _FastWorker(self._session, fi, self)
        w.done.connect(self._on_fast_done)
        self._fast_worker = w
        w.start()
        return True

    def _on_fast_done(self, payload):
        fi, res = payload
        if not res or res.get("status") != "ok":
            # not eligible / tool refused → the model handles the nuance
            self._dispatch_model(self._fast_text)
            return
        import time as _time
        from linac_gen.assist.intents import render_result
        chat, speech = render_result(fi, res)
        self._append(f"  · instant: {fi.tool}")
        try:
            self._transcript.add_message("assistant", chat)
        except Exception:                        # noqa: BLE001
            self._append(chat)
        self._perf_update(
            turn=f"instant {(_time.monotonic() - self._t_turn0)*1e3:.0f} ms")
        self._set_state("idle")
        spoke = False
        if (speech and self._speak.isChecked()
                and self._get_speaker() is not None):
            self._speak_line(speech)
            spoke = True                 # its drain opens the follow-up
        if not spoke:
            self._open_followup_if_voice()

    def _reap_finished_worker(self):
        """Delete the PREVIOUS turn's finished QThread wrapper before
        replacing the reference (they used to accumulate for the
        panel's whole life)."""
        old = self._worker
        if old is None:
            return
        try:
            if not old.isRunning():
                old.deleteLater()
        except RuntimeError:
            pass

    def _begin_turn_ui(self):
        """Shared per-turn state reset + the turn watchdog."""
        import time as _time
        self._set_state("thinking")
        try:
            self._transcript.show_thinking()
        except Exception:                        # noqa: BLE001
            pass
        self._last_reply = ""
        self._stream_buf = ""
        self._streaming = False
        self._speak_pend = ""
        self._speak_carry = ""
        self._speak_in_fence = False
        self._spoke_streaming = False
        self._barged_this_turn = False
        self._responding_latch = False
        self._turn_open_ui = True
        self._stall_notified = False
        self._t_activity = _time.monotonic()
        self._turn_watchdog.start()
        self._speaker_turn(True)

    def _dispatch_model(self, text: str):
        self._begin_turn_ui()
        self._reap_finished_worker()
        self._worker = _AgentWorker(self._session, text, self)
        self._worker.assistant_text.connect(self._append)
        self._worker.assistant_text.connect(self._track_reply)
        self._worker.assistant_delta.connect(self._stream_delta)
        self._worker.assistant_delta_done.connect(self._stream_done)
        self._worker.turn_done.connect(self._on_turn_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _stream_delta(self, text: str):
        """Live-typing insert; the ChatView re-renders the completed
        segment as a markdown card at end_stream."""
        if not self._streaming:
            self._streaming = True
        if self._t_first_tok is None and self._t_turn0 is not None:
            import time as _time
            self._t_first_tok = _time.monotonic()
            self._perf_update(
                turn=f"first token {self._t_first_tok - self._t_turn0:.1f} s")
        self._transcript.stream_delta(text)
        self._stream_buf += text
        # speak-as-it-goes: queue each COMPLETED sentence while streaming
        if self._speak.isChecked():
            self._ingest_speech_text(text)
            import re as _re
            parts = _re.split(r"(?<=[.!?])\s+", self._speak_pend)
            if len(parts) > 1:
                for sent in parts[:-1]:
                    self._speak_sentence(sent)
                self._speak_pend = parts[-1]

    def _ingest_speech_text(self, text: str):
        """Feed streamed text into the speech buffer with PERSISTENT
        code-fence tracking: fenced content must never be spoken (the
        per-sentence summariser can't see a fence opened sentences ago).
        A 2-char carry handles a ``` split across two deltas."""
        buf = getattr(self, "_speak_carry", "") + text
        self._speak_carry = ""
        while True:
            i = buf.find("```")
            if i < 0:
                break
            if not getattr(self, "_speak_in_fence", False):
                self._speak_pend += buf[:i]      # keep prose before fence
            self._speak_in_fence = not getattr(self, "_speak_in_fence",
                                               False)
            buf = buf[i + 3:]
        if getattr(self, "_speak_in_fence", False):
            # discard fenced content — but keep trailing partial backticks
            # (a closing ``` may be split across deltas)
            n_tail = len(buf) - len(buf.rstrip("`"))
            if 0 < n_tail < 3:
                self._speak_carry = buf[-n_tail:]
            return
        # hold back trailing backticks that may start a split fence
        n_tail = len(buf) - len(buf.rstrip("`"))
        if 0 < n_tail < 3:
            self._speak_carry = buf[-n_tail:]
            buf = buf[:-n_tail]
        self._speak_pend += buf

    def _speak_sentence(self, sent: str):
        sent = (sent or "").strip()
        if not sent or self._barged_this_turn:
            return                   # barged: stay quiet for this turn
        sp = self._get_speaker()
        if sp is None:
            return
        try:
            from linac_gen.assist.voice import summarize_for_speech
            spoken = summarize_for_speech(sent)
            if spoken:
                sp.say(spoken)
                self._note_spoken(spoken)
                self._spoke_streaming = True
        except Exception:                        # noqa: BLE001
            pass

    def _stream_done(self):
        try:
            self._transcript.end_stream()    # re-render as markdown card
        except Exception:                    # noqa: BLE001
            pass
        if self._stream_buf.strip():
            self._last_reply = self._stream_buf.strip()   # for speak-back
        if self._speak.isChecked() and self._speak_pend.strip():
            self._speak_sentence(self._speak_pend)        # flush remainder
        self._speak_pend = ""
        self._speak_carry = ""
        self._speak_in_fence = False
        self._stream_buf = ""
        self._streaming = False

    def _track_reply(self, text: str):
        # the agent emits assistant text with a leading newline; keep the
        # most substantive line to speak back
        t = (text or "").strip()
        if t and not t.startswith(("·", "✓")):
            self._last_reply = t

    def _on_turn_done(self):
        if not self._turn_open_ui:
            return          # already finalized (watchdog recovery raced
        #                     the worker's own queued turn_done)
        self._turn_watchdog.stop()
        self._turn_open_ui = False
        self._set_state("idle")
        self._prog.setText("")
        try:
            self._transcript.hide_thinking()
        except Exception:                        # noqa: BLE001
            pass
        # speak the reply only if it wasn't already spoken while streaming
        if not self._spoke_streaming and getattr(self, "_last_reply", ""):
            self._maybe_speak(self._last_reply)
        self._speaker_turn(False)                # queue drain now = done
        if self._drain_queued():
            return                               # a held utterance runs now
        sp = self._speaker
        speaking = self._tts_speaking
        if sp is not None:
            try:
                # LIVE derived read, never the mirrored flag: the flag
                # is False during the first sentence's synthesis, and
                # opening the mic there recorded the assistant's own
                # reply (audit R1 — the cascade behind every complaint)
                speaking = sp.busy()
            except Exception:                    # noqa: BLE001
                pass
        if not speaking:                         # nothing spoken -> the
            self._open_followup_if_voice()       # drain path won't fire

    def _drain_queued(self) -> bool:
        """Run the next held utterance (FIFO order)."""
        if not self._inbox or self._closing:
            return False
        qt, was_voice = self._inbox.popleft()
        self._pending_voice = was_voice
        self._input.setText(qt)
        QTimer.singleShot(0, self._send_queued)
        return True

    def _send_queued(self):
        try:
            if not self._closing:
                self._send()
        except RuntimeError:
            pass

    def _check_turn_health(self):
        """Watchdog while a turn runs: a dead worker or a long-silent
        one must SURFACE — a wedge may last minutes, never forever, and
        never silently."""
        if not self._turn_open_ui:
            self._turn_watchdog.stop()
            return
        w = self._worker
        try:
            running = bool(w is not None and w.isRunning())
        except RuntimeError:
            running = False
        if not running:
            # hard-crashed worker: turn_done never fired — recover the UI
            self._append("[error] the reasoning worker stopped "
                         "unexpectedly — input re-enabled")
            self._on_turn_done()
            return
        import time as _time
        if (_time.monotonic() - self._t_activity > 300.0
                and not self._stall_notified):
            self._stall_notified = True
            self._append("  · still working (no output for 5 min) — "
                         "press ■ Stop to abort if this looks wrong")

    # -- stop / interrupt ------------------------------------------------
    def _on_stop(self):
        """Interrupt the current turn (button or Esc).  Session stays
        usable — ``ask()`` re-arms the abort flag on the next turn."""
        self._barged_this_turn = True            # Stop = SILENCE: late
        #                                          streamed sentences must
        #                                          not speak or re-open
        #                                          the mic-pause gate
        fl = self._followup                      # Stop also closes an open
        if fl is not None and fl.active:         # mic window (MIRAGE
            try:                                 # behavior; ours left it
                fl.cancel()                      # listening for 10-30 s)
            except Exception:                    # noqa: BLE001
                pass
            self.state_changed.emit("idle")
        if self._inbox:                          # Stop drops the inbox —
            self._append(f"  · dropped {len(self._inbox)} queued "
                         "message(s)")           # but never silently
            self._inbox.clear()
        if self._session is not None:
            try:
                self._session.request_stop()
            except Exception:                    # noqa: BLE001
                pass
        # a pending confirmation must RESOLVE (deny) — Stop used to
        # leave the approver blocked and the turn wedged behind it
        ap = self._approver
        if ap is not None and ap.pending is not None:
            from linac_gen.assist.agent import Decision
            self._set_confirm_visible(False)
            ap.resolve(Decision.DENY)
        sp = self._get_speaker()
        if sp is not None:
            try:
                sp.stop()
            except Exception:                    # noqa: BLE001
                pass
        self._stream_done()                      # close any open stream line
        try:
            self._transcript.hide_thinking()
        except Exception:                        # noqa: BLE001
            pass
        self._append("  · stopped.")
        self._prog.setText("")
        self._set_state("idle")

    def keyPressEvent(self, ev):                             # noqa: N802
        # Esc = STOP, not close (QDialog's default reject would close the
        # panel — mid-conversation that is exactly wrong).  Ctrl+W still
        # closes via the base shortcuts.  The body is sealed: an audio
        # exception escaping a Qt virtual is the sipBadCatcherResult bug.
        try:
            if ev.key() == Qt.Key.Key_Escape:
                self._on_stop()
                ev.accept()
                return
            # hold-SPACE push-to-talk when the input line isn't focused
            if (ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat()
                    and not self._input.hasFocus()
                    and self._mic_btn.isEnabled()
                    and self._recorder is None
                    and self._capture_tap is None):
                self._mic_pressed()
                ev.accept()
                return
        except Exception:                        # noqa: BLE001
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):                           # noqa: N802
        try:
            if (ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat()
                    and (self._recorder is not None
                         or self._capture_tap is not None)):
                self._mic_released()
                ev.accept()
                return
        except Exception:                        # noqa: BLE001
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def showEvent(self, ev):                                 # noqa: N802
        # any way of coming back into view (toolbar reopen, wake popup)
        # clears the dismissal — hands-free follow-ups resume
        try:
            self._dismissed = False
        except Exception:                        # noqa: BLE001
            pass
        super().showEvent(ev)

    def focusOutEvent(self, ev):                             # noqa: N802
        # SPACE-held PTT + focus stolen (dialog, cmd-tab) = the release
        # never arrives and the mic stays hot — force-release instead
        try:
            if (self._recorder is not None
                    or self._capture_tap is not None):
                self._mic_released()
        except Exception:                        # noqa: BLE001
            pass
        super().focusOutEvent(ev)

    # -- machine events / progress --------------------------------------
    def _on_event_line(self, text: str):
        """GUI-thread: render a machine event; optionally narrate it."""
        self._append(f"  ‣ [event] {text}")
        if (self._narrate.isChecked() and self._session is not None
                and not (self._worker and self._worker.isRunning())):
            self._start_event_worker()

    def _start_event_worker(self):
        """Give the model an unprompted turn to react to queued events."""
        self._begin_turn_ui()
        self._reap_finished_worker()
        # machine-initiated: must not inherit a stale voice flag and
        # reopen the mic after ITS reply
        self._last_turn_was_voice = False
        w = _EventWorker(self._session, self)
        w.assistant_text.connect(self._append)
        w.assistant_text.connect(self._track_reply)
        w.assistant_delta.connect(self._stream_delta)
        w.assistant_delta_done.connect(self._stream_done)
        w.turn_done.connect(self._on_turn_done)
        w.failed.connect(self._on_worker_failed)
        self._worker = w
        w.start()

    def _on_worker_failed(self, msg: str):
        self._append(f"[error] {msg}")

    def _on_progress(self, text: str):
        import time as _time
        self._t_activity = _time.monotonic()     # long tools stay "alive"
        if not text:
            # progress cleared: restore the listening cue it displaced
            self._prog.setText(getattr(self, "_last_wake_status", ""))
        else:
            self._prog.setText(text)

    # -- conversation text size -----------------------------------------
    def _bump_font(self, delta: int):
        try:
            px = self._transcript.body_px + delta
            self._transcript.set_text_size(px)
            self._settings().setValue("assist/chat_px",
                                      self._transcript.body_px)
        except Exception:                    # noqa: BLE001
            pass

    def _on_font_smaller(self):
        self._bump_font(-2)

    def _on_font_bigger(self):
        self._bump_font(+2)

    def _on_capture_saved(self, path: str, tool: str):
        """Inline thumbnail of what the assistant just looked at."""
        try:
            self._transcript.add_image(path, caption=f"({tool})")
        except Exception:                    # noqa: BLE001
            pass

    def _on_confirm_timeout(self, note: str):
        self._set_confirm_visible(False)
        self._append(f"  · confirmation {note}")
        self._set_state("thinking")
        self._speaker_turn(True)                 # reply stream resumes

    def _on_event_threadsafe(self, event: dict):
        # called from the worker thread — marshal via Qt signals (GUI-thread)
        t = event.get("type")
        # events and progress are worker-independent (a watcher alert can
        # arrive while the assistant is idle)
        if t == "event":
            self.event_line.emit(str(event.get("text", "")))
            return
        if t == "progress":
            self.progress_changed.emit(str(event.get("text", "")))
            return
        if t == "capture":
            self.capture_saved.emit(str(event.get("path", "")),
                                    str(event.get("tool", "")))
            return
        if not self._worker:
            return
        import time as _time
        self._t_activity = _time.monotonic()     # turn-watchdog heartbeat
        if t == "assistant_delta":
            self._worker.assistant_delta.emit(event.get("text", ""))
            # LATCHED: one state emission per streamed segment, not one
            # per token (the per-delta storm re-styled the orb/lamp
            # hundreds of times per reply)
            if not self._responding_latch:
                self._responding_latch = True
                self.state_changed.emit("responding")
        elif t == "assistant_delta_done":
            self._responding_latch = False
            self._worker.assistant_delta_done.emit()
        elif t == "assistant_text":
            self._worker.assistant_text.emit("\n" + event["text"])
            self._responding_latch = False
            self.state_changed.emit("responding")
        elif t == "tool_start":
            self._worker.assistant_delta_done.emit()   # close any open line
            self._worker.assistant_text.emit(
                f"  · {event['tool']} …")
            self._responding_latch = False
            self.state_changed.emit("thinking")
        elif t == "error":
            self._worker.assistant_text.emit(
                f"[error] {event.get('message', '')}")
            self._responding_latch = False
            self.state_changed.emit("error")
            if ("turn error" in str(event.get("message", ""))
                    and not getattr(self, "_opus_tip_shown", False)):
                try:
                    model = str(self._settings().value(
                        "assist/model", "")).lower()
                except Exception:                # noqa: BLE001
                    model = ""
                if "opus" in model:
                    self._opus_tip_shown = True
                    self._worker.assistant_text.emit(
                        "[tip] opus looks unavailable right now — "
                        "backend… → model 'sonnet' → Connect usually "
                        "answers immediately")
        elif t == "job_submitted" and self._worker:
            self._worker.assistant_text.emit(
                f"  · {event['tool']} → {event['job_id']} (background)")

    # -- confirmation ---------------------------------------------------
    def _set_confirm_visible(self, on: bool):
        self._confirm_frame.setVisible(on)
        self._confirm_label.setVisible(on)
        self._btn_approve.setVisible(on)
        self._btn_always.setVisible(on)
        self._btn_deny.setVisible(on)

    def _show_confirmation(self, req):
        if self._closing:
            return          # late worker emit after shutdown began — the
                            # approver's own closing check resolves it
        self._btn_always.setVisible(req.allow_session_auto)
        self._confirm_label.setText(
            f"Confirm {req.tier}:\n{req.pretty}")
        self._set_confirm_visible(True)
        self._btn_always.setVisible(req.allow_session_auto)
        # stamp the buttons with the CURRENT confirmation's generation so
        # a click that lands after a timeout auto-deny cannot resolve the
        # next request
        self._confirm_gen = (self._approver.generation
                             if self._approver else 0)
        self._set_state("awaiting-confirm")
        # spoken echo + a follow-up window so a bare "yes"/"cancel" is
        # hands-free (the yes/no is interpreted LOCALLY — never the model)
        if self._speak.isChecked() or self._mic_stream is not None:
            self._speaker_turn(False)  # echo drain must report + open mic
            self._speak_line(f"Approval needed: {req.tool}. "
                             "Say confirm, or cancel.")
            # open the mic NOW unless our own echo is about to play (the
            # TTS drain opens it then) — hands-free must survive a
            # missing/failed speaker
            if not (self._speak.isChecked()
                    and self._get_speaker() is not None):
                self._last_turn_was_voice = True
                fl = self._ensure_followup()
                if fl is not None:
                    self._open_fl_window(fl)
            else:
                self._last_turn_was_voice = True

    def _resolve_confirm(self, decision):
        self._set_confirm_visible(False)
        self._set_state("thinking")
        self._speaker_turn(True)                 # reply stream resumes
        if self._approver:
            self._approver.resolve(decision,
                                   gen=getattr(self, "_confirm_gen", None))

    def _on_approve(self):
        from linac_gen.assist.agent import Decision
        self._resolve_confirm(Decision.APPROVE)

    def _on_always(self):
        from linac_gen.assist.agent import Decision
        self._auto.setChecked(True)
        self._resolve_confirm(Decision.APPROVE_SESSION)

    def _on_deny(self):
        from linac_gen.assist.agent import Decision
        self._resolve_confirm(Decision.DENY)

    def _on_auto_toggle(self, _state):
        if self._session is not None:
            self._session.auto_approve_compute = self._auto.isChecked()

    # -- teardown -------------------------------------------------------
    def closeEvent(self, ev):                                # noqa: N802
        """✕ HIDES the panel — it never tears the session down.

        The old behavior was THE hang: closing the window killed the
        SDK session and the listeners, but the app kept REUSING the
        dead panel — the worker then blocked forever on a closed
        backend and every later message was silently swallowed.  The
        assistant is an app-level service now: it dies only at app
        shutdown (via :meth:`shutdown`).

        Dismissing the window also SILENCES it (user report: "I closed
        it but still can hear it"): speech is cut, the rest of the turn
        stays muted, and any open follow-up mic window is cancelled.
        The turn itself keeps computing — its text lands in the
        transcript — and the wake word still reopens the panel."""
        if not self._closing:
            ev.ignore()
            self.hide()
            self._dismissed = True               # explicit user dismissal
            self._barged_this_turn = True        # mute the rest of the turn
            sp = self._speaker
            if sp is not None:
                try:
                    sp.stop()                    # cut current speech NOW
                except Exception:                # noqa: BLE001
                    pass
            fl = self._followup
            if fl is not None:
                try:
                    fl.cancel()                  # no open mic on a hidden panel
                except Exception:                # noqa: BLE001
                    pass
            return
        super().closeEvent(ev)

    def shutdown(self):
        """REAL teardown — called by the app at quit (never by ✕)."""
        if self._closing:
            return
        self._closing = True
        self._turn_watchdog.stop()
        try:
            self._state.results_changed.disconnect(
                self._on_results_changed)
        except Exception:                        # noqa: BLE001
            pass
        # hands-free stack first: passive analyzers, listeners, then the
        # single stream (its reader closes it) — see _shutdown_listening
        try:
            self._shutdown_listening()
        except Exception:                        # noqa: BLE001
            pass
        if getattr(self, "_orb", None) is not None:
            self._orb.stop()                     # halt the animation timer
        if self._session is not None:
            self._session.request_stop()
            try:
                # Close FIRST: this cancels a blocked SDK turn so the worker
                # thread can actually exit before we wait on it — otherwise
                # the worker hangs on fut.result() and Qt aborts on quit.
                self._session.close()
            except Exception:                    # noqa: BLE001
                pass
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.wait(3000)
        except (RuntimeError, AttributeError):   # deleted / test double
            pass
        try:
            if (getattr(self, "_fast_worker", None) is not None
                    and self._fast_worker.isRunning()):
                self._fast_worker.wait(2000)     # else qFatal at app quit
        except (RuntimeError, AttributeError):
            pass
        # voice teardown
        if getattr(self, "_recorder", None) is not None:
            try:
                self._recorder.stop()
            except Exception:                    # noqa: BLE001
                pass
        if getattr(self, "_voice_worker", None) is not None \
                and self._voice_worker.isRunning():
            self._voice_worker.wait(1500)
        if getattr(self, "_tts", None) is not None:
            try:
                self._tts.stop()
            except Exception:                    # noqa: BLE001
                pass
        if getattr(self, "_speaker", None) is not None:
            try:
                self._speaker.stop()             # flush queue + cut audio
            except Exception:                    # noqa: BLE001
                pass
        self.close()                             # real close (closing=True)
