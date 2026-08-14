"""The assistant transcript as a designed, animated chat.

Widget-based (a QScrollArea of message cards): rounded cards with a
fade-in, role chips ("you" / "HELIX") with timestamps, markdown
rendering with Pygments code, inline capture thumbnails, an animated
thinking indicator — and a USER-CONTROLLED text size: the whole
conversation re-renders live when the panel's A− / A+ controls change
``set_text_size`` (default 18 px body; every other size derives from
it, so the hierarchy scales together).

API contract preserved (all pre-existing tests): ``add_message /
append_line / appendPlainText / stream_delta / end_stream / add_image /
toPlainText`` — plain text keeps every legacy prefix, and streamed raw
text is visible in it until ``end_stream``.
"""
from __future__ import annotations

import html as _html
import os
import re
import time as _time

from PyQt6.QtCore import (
    Qt, QEasingCurve, QPropertyAnimation, QTimer, QUrl,
)
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme

DEFAULT_BODY_PX = 18
MIN_BODY_PX = 12
MAX_BODY_PX = 28

_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.S)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _inline(text: str, body_px: int = DEFAULT_BODY_PX) -> str:
    t = _html.escape(text)
    t = _BOLD_RE.sub(r"<b>\1</b>", t)
    t = _ITALIC_RE.sub(r"<i>\1</i>", t)
    t = _CODE_RE.sub(
        rf'<code style="background:{theme.BG_INSET}; '
        rf'font-family:{theme.FONT_MONO}; '
        rf'font-size:{max(body_px - 2, 10)}px;">\1</code>', t)
    t = _LINK_RE.sub(
        rf'<a style="color:{theme.ACCENT};" href="\2">\1</a>', t)
    return t


def _highlight_code(code: str, lang: str,
                    body_px: int = DEFAULT_BODY_PX) -> str:
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name, guess_lexer
        try:
            lexer = get_lexer_by_name(lang or "python")
        except Exception:                                   # noqa: BLE001
            lexer = guess_lexer(code)
        body = highlight(code, lexer,
                         HtmlFormatter(noclasses=True, style="monokai",
                                       nowrap=True))
    except Exception:                                       # noqa: BLE001
        body = _html.escape(code)
    # white-space:pre-wrap wraps long code lines WITHOUT mutating the
    # text (a hard-wrap variant put real newlines into copy-pasted code
    # — split string literals; adversarial review, measured)
    return (f'<pre style="background:{theme.BG_INSET}; '
            f'color:{theme.TEXT_1}; font-family:{theme.FONT_MONO}; '
            f'font-size:{max(body_px - 3, 10)}px; padding:10px; '
            f'margin:8px 0; white-space:pre-wrap;">' + body + "</pre>")


def md_to_html(text: str, body_px: int = DEFAULT_BODY_PX) -> str:
    """Small, total-failure-safe markdown → Qt-HTML; every size derives
    from ``body_px`` so the whole hierarchy scales together."""
    try:
        parts: list[str] = []
        last = 0
        src = str(text)
        for m in _FENCE_RE.finditer(src):
            parts.append(_prose_to_html(src[last:m.start()], body_px))
            parts.append(_highlight_code(m.group(2), m.group(1),
                                         body_px))
            last = m.end()
        parts.append(_prose_to_html(src[last:], body_px))
        return "".join(p for p in parts if p)
    except Exception:                                       # noqa: BLE001
        return "<p>" + _html.escape(str(text)) + "</p>"


def _prose_to_html(text: str, body_px: int) -> str:
    if not text.strip():
        return ""
    hsize = {1: body_px + 7, 2: body_px + 4, 3: body_px + 2,
             4: body_px + 1}
    out: list[str] = []
    in_list = None
    for raw in text.split("\n"):
        line = raw.rstrip()
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        number = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        want = "ul" if bullet else ("ol" if number else None)
        if in_list and want != in_list:
            out.append(f"</{in_list}>")
            in_list = None
        if h:
            out.append(
                f'<div style="font-size:{hsize[len(h.group(1))]}px; '
                f'font-weight:bold; color:{theme.TEXT_0}; '
                'margin:10px 0 5px 0;">'
                + _inline(h.group(2), body_px) + "</div>")
        elif bullet or number:
            if not in_list:
                in_list = want
                out.append(f'<{want} style="margin:5px 0 5px 20px;">')
            out.append("<li style='margin:3px 0;'>"
                       + _inline((bullet or number).group(1), body_px)
                       + "</li>")
        elif line:
            out.append("<p style='margin:5px 0; line-height:1.5;'>"
                       + _inline(line, body_px) + "</p>")
    if in_list:
        out.append(f"</{in_list}>")
    return "".join(out)


def _html_to_plain(html_text: str) -> str:
    from PyQt6.QtGui import QTextDocument
    doc = QTextDocument()
    doc.setHtml(html_text)
    return doc.toPlainText()


# ---------------------------------------------------------------------------
# message cards
# ---------------------------------------------------------------------------
class _Card(QFrame):
    """One rounded message card; fades in on arrival."""

    def __init__(self, style: str, parent=None, animate: bool = True):
        super().__init__(parent)
        self.setStyleSheet(style)
        # vertical Preferred, NEVER Minimum: Minimum has no ShrinkFlag,
        # so the scroll range is computed from rich-text sizeHints that
        # overestimate height (~+77 px per card, measured) — the surplus
        # accumulated in the trailing stretch and the pinned viewport
        # ended up showing pure blank space ("the text box goes empty")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(14, 10, 14, 10)
        self._lay.setSpacing(4)
        if animate:
            eff = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity", self)
            anim.setDuration(220)
            # floor 0.4, not 0.0: while the GUI thread is busy laying
            # out a long transcript the animation gets no slice — a
            # 0-opacity card is a full-height INVISIBLE block (the
            # second confirmed "transcript looks empty" mechanism)
            anim.setStartValue(0.4)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.finished.connect(self._drop_effect)
            self._anim = anim
            anim.start()
            # backup: if the animation timer is starved, force full
            # opacity soon regardless (bound method — timer rule)
            QTimer.singleShot(600, self._drop_effect)

    def _drop_effect(self):
        self.setGraphicsEffect(None)         # normal painting afterwards

    def add_rich(self, html_text: str) -> QLabel:
        lab = QLabel(self)
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setWordWrap(True)
        lab.setText(html_text)
        lab.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse)
        lab.setOpenExternalLinks(True)
        lab.setStyleSheet("background:transparent; border:none;")
        self._lay.addWidget(lab)
        return lab


class _ThinkingCard(QFrame):
    """Animated ● ● ● while the model works."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background:{theme.BG_INSET}; border:none; "
            "border-radius:12px; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        self._lab = QLabel("●")
        self._lab.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:15px; background:"
            "transparent; border:none;")
        lay.addWidget(self._lab)
        lay.addStretch(1)
        self._n = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(350)

    def _tick(self):
        self._n = (self._n + 1) % 3
        self._lab.setText("● " * (self._n + 1))

    def stop(self):
        self._timer.stop()


# ---------------------------------------------------------------------------
# the chat view
# ---------------------------------------------------------------------------
class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"QScrollArea {{ background:{theme.BG_0}; border:1px solid "
            f"{theme.BORDER_0}; border-radius:10px; }}"
            f"QScrollBar:vertical {{ background:{theme.BG_0}; width:8px; "
            "margin:0; }"
            f"QScrollBar::handle:vertical {{ background:{theme.BORDER_0};"
            " border-radius:4px; min-height:30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical"
            " { height:0; }")
        self._host = QWidget(self)
        # NOTE: do NOT set the host's vertical policy to Ignored — with
        # no height-for-width item in the layout (an image-only feed)
        # the host then collapses to the viewport and the content is
        # clipped with no scrollbar (adversarial review, measured).
        # The _Card Preferred policy alone removes the phantom height.
        self._host.setStyleSheet(f"background:{theme.BG_0};")
        self._col = QVBoxLayout(self._host)
        self._col.setContentsMargins(12, 12, 12, 12)
        self._col.setSpacing(10)
        self._col.addStretch(1)              # cards pack from the top
        #: Python-side card registry — _content_bottom measures THESE, not
        #: QLayoutItems: a QLayoutItem is not a QObject, so a stale item is
        #: an unguardable dangling pointer (deterministic full-suite
        #: segfault, 2026-08-14), while a dead QWidget wrapper raises a
        #: catchable RuntimeError via sip.
        self._cards: list = []
        self.setWidget(self._host)

        self._body_px = DEFAULT_BODY_PX
        self._plain: list[str] = []          # the toPlainText mirror
        self._history: list[tuple] = []      # replayable message log
        self._img_n = 0
        self._stream_card: _Card | None = None
        self._stream_lab: QLabel | None = None
        self._stream_raw = ""
        self._thinking: _ThinkingCard | None = None
        # delta coalescing: tokens arrive one queued signal each, and
        # re-rendering the WHOLE segment per token is O(n²) — repaint at
        # most every 33 ms instead (imperceptible, storm-proof)
        self._stream_dirty = False
        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(33)
        self._stream_timer.timeout.connect(self._flush_stream)
        # CONTENT-AWARE follow-the-bottom.  Two proven rules:
        # 1. never pin to bar.maximum() — the range can carry phantom
        #    height; pin to the real content bottom instead;
        # 2. unpin ONLY on user gestures (wheel / scrollbar drag /
        #    steps), never by inferring from valueChanged — programmatic
        #    scrolls fire it too and the logic fights itself.
        self._pinned = True
        bar = self.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range_changed)
        bar.sliderMoved.connect(self._on_user_scroll)
        bar.actionTriggered.connect(self._on_user_scroll)

    # -- sizing ----------------------------------------------------------
    @property
    def body_px(self) -> int:
        return self._body_px

    def set_text_size(self, px: int) -> None:
        """Change the conversation type size and RE-RENDER the whole
        transcript live (the panel's A− / A+ controls)."""
        px = max(MIN_BODY_PX, min(MAX_BODY_PX, int(px)))
        if px == self._body_px:
            return
        self._body_px = px
        self._rebuild()

    def _rebuild(self) -> None:
        history = list(self._history)
        stream_raw = self._stream_raw        # NEVER destroy an in-flight
        img_n = 0                            # reply (A+/A- mid-stream
        #                                      used to silently eat it)
        while self._col.count() > 1:         # keep the trailing stretch
            item = self._col.takeAt(0)
            w = item.widget()
            if w is not None:
                if isinstance(w, _ThinkingCard):
                    w.stop()
                w.hide()                     # no zombie frame until the
                w.deleteLater()              # deferred delete lands
        self._thinking = None
        self._cards = []                     # re-render re-registers
        self._stream_timer.stop()
        self._stream_dirty = False
        self._stream_card = None
        self._stream_lab = None
        self._stream_raw = ""
        self._plain = []
        self._history = []
        self._img_n = img_n
        for entry in history:
            try:
                if entry[0] == "msg":
                    self._render_message(entry[1], entry[2], entry[3],
                                         animate=False)
                elif entry[0] == "img":
                    self._render_image(entry[1], entry[2], animate=False)
                self._history.append(entry)  # AFTER success — a bad
                #                              entry must not re-raise
                #                              on every future rebuild
            except Exception:                               # noqa: BLE001
                continue                     # one bad entry must never
                #                              blank the rest of the feed
        if stream_raw:                       # recreate the live card
            self.stream_delta(stream_raw)

    # -- internals -------------------------------------------------------
    def _add_widget(self, w: QWidget, at_index: int | None = None) -> None:
        self._col.insertWidget(self._col.count() - 1
                               if at_index is None else at_index, w)
        self._cards.append(w)
        # follow only while pinned — a reader who scrolled up must not
        # be yanked down by every new card
        QTimer.singleShot(0, self._maybe_scroll_bottom)

    def _maybe_scroll_bottom(self) -> None:
        try:
            if self._pinned:
                self._scroll_bottom()
        except RuntimeError:
            pass

    def _content_bottom(self) -> int:
        """Bottom edge of the REAL content (max over visible cards) —
        the truth the pin targets; ``bar.maximum()`` lies whenever
        size-hints overestimate (measured +77 px per turn).

        Iterates the Python-side card registry, NOT the layout items:
        ``itemAt()`` hands back QLayoutItem pointers with no liveness
        guard (not QObjects), and a stale one segfaulted the full test
        suite deterministically once test teardown started actually
        destroying widgets.  Dead cards raise RuntimeError here (sip
        knows) and are pruned; hidden cards are skipped as before.
        """
        b = 0
        live = []
        for w in self._cards:
            try:
                if not w.isHidden():
                    b = max(b, w.y() + w.height())
                live.append(w)
            except RuntimeError:             # C++ side gone — prune
                continue
        self._cards = live
        return b + self._col.contentsMargins().bottom()

    def _scroll_bottom(self) -> None:
        """Explicit jump to the newest content — always re-pins."""
        try:
            self._pinned = True
            bar = self.verticalScrollBar()
            target = self._content_bottom() - self.viewport().height()
            bar.setValue(max(0, min(bar.maximum(), target)))
        except RuntimeError:
            pass

    def _on_range_changed(self, _lo: int, _hi: int) -> None:
        try:
            if self._pinned:
                self._scroll_bottom()
        except RuntimeError:
            pass

    def _on_user_scroll(self, *_a) -> None:
        # gesture in flight: evaluate the pin AFTER the value settles
        QTimer.singleShot(0, self._eval_pin)

    def _eval_pin(self, tol: int | None = None) -> None:
        try:
            bar = self.verticalScrollBar()
            target = self._content_bottom() - self.viewport().height()
            if tol is None:
                tol = max(4, bar.singleStep() // 2)   # BELOW one arrow
            self._pinned = bar.value() >= target - tol   # step (20 px)
        except RuntimeError:
            pass

    def wheelEvent(self, ev):                                # noqa: N802
        down = False
        try:
            down = ev.angleDelta().y() < 0
        except Exception:                                   # noqa: BLE001
            pass
        super().wheelEvent(ev)
        if down:
            # scrolling TOWARD the bottom re-pins generously: during a
            # stream the bottom runs away and the tight band needed ~13
            # wheel notches to re-catch (adversarial review, measured)
            self._eval_pin(tol=max(60, self.viewport().height() // 3))
        else:
            self._eval_pin()

    def _card_style(self, role: str) -> str:
        if role == "assistant":
            bg = getattr(theme, "BG_1", theme.BG_INSET)
            return ("QFrame { border-radius:12px; border:none; "
                    f"background:{bg}; }}")
        if role == "user":
            return ("QFrame { border-radius:12px; border:none; "
                    "background:qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                    f" stop:0 {theme.BG_INSET}, stop:1 {theme.BG_0}); "
                    f"border-left:3px solid {theme.ACCENT}; }}")
        return "QFrame { background:transparent; border:none; }"

    def _chip(self, label: str, color: str) -> str:
        return (f'<span style="color:{color}; '
                f'font-size:{max(self._body_px - 6, 10)}px; '
                'font-weight:600; letter-spacing:1px;">'
                + _html.escape(label) + "</span>")

    def _stamp_html(self, stamp: str) -> str:
        return (f'<span style="color:{theme.TEXT_3}; '
                f'font-size:{max(self._body_px - 7, 9)}px;">'
                + _html.escape(stamp) + "</span>")

    # -- public API ------------------------------------------------------
    def add_message(self, role: str, text: str) -> None:
        self.hide_thinking()
        stamp = _time.strftime("%H:%M")
        self._history.append(("msg", role, text, stamp))
        self._render_message(role, text, stamp, animate=True)

    def _render_message(self, role: str, text: str, stamp: str,
                        animate: bool, at_index: int | None = None) -> None:
        px = self._body_px
        if role == "assistant":
            html_body = md_to_html(text, px)
            card = _Card(self._card_style(role), self._host,
                         animate=animate)
            card.add_rich(
                '<table width="100%"><tr><td align="left">'
                + self._chip("HELIX", theme.ACCENT)
                + f'</td><td align="right">{self._stamp_html(stamp)}'
                "</td></tr></table>"
                f'<div style="color:{theme.TEXT_0}; '
                f'font-size:{px}px;">' + html_body + "</div>")
            self._add_widget(card, at_index)
            self._plain.append(_html_to_plain(html_body))
        elif role == "user":
            card = _Card(self._card_style(role), self._host,
                         animate=animate)
            card.add_rich(
                '<table width="100%"><tr><td align="left">'
                + self._chip("you", theme.TEXT_3)
                + f'</td><td align="right">{self._stamp_html(stamp)}'
                "</td></tr></table>"
                f'<span style="color:{theme.ACCENT}; '
                f'font-size:{px}px;">▶ ' + _inline(text, px)
                + "</span>")
            self._add_widget(card)
            self._plain.append("▶ " + text)
        else:
            color = {"tool": theme.TEXT_3, "event": "#4ade80",
                     "system": theme.TEXT_2}.get(role, theme.TEXT_2)
            lab = QLabel(self._host)
            lab.setWordWrap(True)
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setText(
                f'<span style="color:{color}; '
                f'font-family:{theme.FONT_MONO}; '
                f'font-size:{max(px - 4, 11)}px;">'
                + _inline(text, px) + "</span>")
            lab.setStyleSheet("background:transparent; border:none; "
                              "margin-left:8px;")
            lab.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            self._add_widget(lab)
            self._plain.append(text)

    def add_image(self, path: str, caption: str = "") -> None:
        self._history.append(("img", path, caption))
        self._render_image(path, caption, animate=True)

    def _render_image(self, path: str, caption: str,
                      animate: bool) -> None:
        try:
            img = QImage(path)
            if img.isNull():
                return
            if img.width() > 460:
                img = img.scaledToWidth(
                    460, Qt.TransformationMode.SmoothTransformation)
            self._img_n += 1
            card = _Card(self._card_style("assistant"), self._host,
                         animate=animate)
            pic = QLabel(card)
            pic.setPixmap(QPixmap.fromImage(img))
            pic.setStyleSheet("background:transparent; border:none;")
            pic.setCursor(Qt.CursorShape.PointingHandCursor)
            pic.setToolTip("click to open the saved capture")
            full = os.path.abspath(path)
            pic.mousePressEvent = (                        # noqa: E731
                lambda ev, p=full:
                QDesktopServices.openUrl(QUrl.fromLocalFile(p)))
            card._lay.addWidget(pic)
            if caption:
                cap = QLabel(caption, card)
                cap.setStyleSheet(
                    f"color:{theme.TEXT_3}; "
                    f"font-size:{max(self._body_px - 5, 10)}px;"
                    "background:transparent; border:none;")
                card._lay.addWidget(cap)
            self._add_widget(card)
            self._plain.append(caption or "(image)")
        except Exception:                                   # noqa: BLE001
            pass

    # -- thinking indicator ----------------------------------------------
    def show_thinking(self) -> None:
        if self._thinking is not None:
            return
        self._thinking = _ThinkingCard(self._host)
        self._add_widget(self._thinking)

    def hide_thinking(self) -> None:
        t, self._thinking = self._thinking, None
        if t is not None:
            try:
                t.stop()
                t.hide()                     # removeWidget leaves the
                self._col.removeWidget(t)    # widget VISIBLE until the
                t.deleteLater()              # deferred delete otherwise
            except Exception:                               # noqa: BLE001
                pass

    # -- streaming -------------------------------------------------------
    def stream_delta(self, text: str) -> None:
        self.hide_thinking()
        first = self._stream_card is None
        if first:
            self._stream_card = _Card(self._card_style("assistant"),
                                      self._host, animate=False)
            self._stream_lab = self._stream_card.add_rich("")
            self._stream_raw = ""
            # chronology anchors: a tool/event line arriving MID-stream
            # is inserted after this card; the final markdown card and
            # its history/plain entries must land back HERE, not at the
            # end (the swap used to reorder the transcript permanently)
            self._stream_hist_pos = len(self._history)
            self._stream_plain_pos = len(self._plain)
            self._add_widget(self._stream_card)
            self._stream_timer.start()
        self._stream_raw += text
        self._stream_dirty = True
        if first:
            self._flush_stream()         # first token shows instantly

    def _flush_stream(self) -> None:
        if not self._stream_dirty or self._stream_lab is None:
            return
        self._stream_dirty = False
        try:
            self._stream_lab.setText(
                f'<div style="color:{theme.TEXT_0}; '
                f'font-size:{self._body_px}px; line-height:1.5;">'
                + _html.escape(self._stream_raw).replace("\n", "<br>")
                + f'<span style="color:{theme.ACCENT};">▌</span></div>')
        except RuntimeError:             # label died mid-render
            return
        # no scroll here: rangeChanged pins the view to the bottom the
        # moment the layout pass actually grows the range (any scroll
        # issued NOW runs too early and lands one step short — the
        # "text goes up beyond the chat window" report)

    def end_stream(self) -> str:
        self._stream_timer.stop()
        self._stream_dirty = False
        raw = self._stream_raw
        card, self._stream_card = self._stream_card, None
        self._stream_lab = None
        self._stream_raw = ""
        if card is None or not raw.strip():
            if card is not None:
                card.hide()
                self._col.removeWidget(card)
                card.deleteLater()
            return raw
        # replace the live card with a full markdown card AT THE SAME
        # POSITION (visual + history + plain) — appending at the end
        # reordered the transcript whenever a tool/event line had
        # landed mid-stream (adversarial review, measured)
        pos = self._col.indexOf(card)
        card.hide()                          # no ghost frame during the
        self._col.removeWidget(card)         # shrink→grow swap
        card.deleteLater()
        stamp = _time.strftime("%H:%M")
        self._render_message("assistant", raw, stamp, animate=False,
                             at_index=pos if pos >= 0 else None)
        entry = ("msg", "assistant", raw, stamp)
        hp = min(getattr(self, "_stream_hist_pos", len(self._history)),
                 len(self._history))
        self._history.insert(hp, entry)
        pp = min(getattr(self, "_stream_plain_pos", len(self._plain)),
                 max(0, len(self._plain) - 1))
        moved = self._plain.pop()            # _render_message appended it
        self._plain.insert(pp, moved)
        return raw

    # -- legacy compatibility -------------------------------------------
    def append_line(self, text: str) -> None:
        t = str(text).strip("\n")
        if not t.strip():
            return
        if t.startswith("▶ "):
            self.add_message("user", t[2:])
        elif t.lstrip().startswith(("·", "✓")):
            self.add_message("tool", t.strip())
        elif t.lstrip().startswith("‣"):
            self.add_message("event", t.strip())
        else:
            self.add_message("system" if t.startswith("[")
                             else "assistant", t)

    appendPlainText = append_line

    def toPlainText(self) -> str:                            # noqa: N802
        parts = list(self._plain)
        if self._stream_raw:
            parts.append(self._stream_raw)
        return "\n".join(parts)
