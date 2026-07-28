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
    return (f'<pre style="background:{theme.BG_INSET}; '
            f'color:{theme.TEXT_1}; font-family:{theme.FONT_MONO}; '
            f'font-size:{max(body_px - 3, 10)}px; padding:10px; '
            f'margin:8px 0;">' + body + "</pre>")


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
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Minimum)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(14, 10, 14, 10)
        self._lay.setSpacing(4)
        if animate:
            eff = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity", self)
            anim.setDuration(220)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.finished.connect(self._drop_effect)
            self._anim = anim
            anim.start()

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
        self._host.setStyleSheet(f"background:{theme.BG_0};")
        self._col = QVBoxLayout(self._host)
        self._col.setContentsMargins(12, 12, 12, 12)
        self._col.setSpacing(10)
        self._col.addStretch(1)              # cards pack from the top
        self.setWidget(self._host)

        self._body_px = DEFAULT_BODY_PX
        self._plain: list[str] = []          # the toPlainText mirror
        self._history: list[tuple] = []      # replayable message log
        self._img_n = 0
        self._stream_card: _Card | None = None
        self._stream_lab: QLabel | None = None
        self._stream_raw = ""
        self._thinking: _ThinkingCard | None = None

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
        img_n = 0
        while self._col.count() > 1:         # keep the trailing stretch
            item = self._col.takeAt(0)
            w = item.widget()
            if w is not None:
                if isinstance(w, _ThinkingCard):
                    w.stop()
                w.deleteLater()
        self._thinking = None
        self._stream_card = None
        self._stream_lab = None
        self._stream_raw = ""
        self._plain = []
        self._history = []
        self._img_n = img_n
        for entry in history:
            if entry[0] == "msg":
                self._history.append(entry)
                self._render_message(entry[1], entry[2], entry[3],
                                     animate=False)
            elif entry[0] == "img":
                self._history.append(entry)
                self._render_image(entry[1], entry[2], animate=False)

    # -- internals -------------------------------------------------------
    def _add_widget(self, w: QWidget) -> None:
        self._col.insertWidget(self._col.count() - 1, w)
        QTimer.singleShot(0, self._scroll_bottom)

    def _scroll_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

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
                        animate: bool) -> None:
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
            self._add_widget(card)
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
                self._col.removeWidget(t)
                t.deleteLater()
            except Exception:                               # noqa: BLE001
                pass

    # -- streaming -------------------------------------------------------
    def stream_delta(self, text: str) -> None:
        self.hide_thinking()
        if self._stream_card is None:
            self._stream_card = _Card(self._card_style("assistant"),
                                      self._host, animate=False)
            self._stream_lab = self._stream_card.add_rich("")
            self._stream_raw = ""
            self._add_widget(self._stream_card)
        self._stream_raw += text
        self._stream_lab.setText(
            f'<div style="color:{theme.TEXT_0}; '
            f'font-size:{self._body_px}px; line-height:1.5;">'
            + _html.escape(self._stream_raw).replace("\n", "<br>")
            + f'<span style="color:{theme.ACCENT};">▌</span></div>')
        QTimer.singleShot(0, self._scroll_bottom)

    def end_stream(self) -> str:
        raw = self._stream_raw
        card, self._stream_card = self._stream_card, None
        self._stream_lab = None
        self._stream_raw = ""
        if card is None or not raw.strip():
            if card is not None:
                self._col.removeWidget(card)
                card.deleteLater()
            return raw
        # replace the live card with a full markdown card (history too)
        self._col.removeWidget(card)
        card.deleteLater()
        stamp = _time.strftime("%H:%M")
        self._history.append(("msg", "assistant", raw, stamp))
        self._render_message("assistant", raw, stamp, animate=False)
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
