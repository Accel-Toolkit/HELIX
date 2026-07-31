"""Phase-7 chat view: markdown rendering, role styling, streaming
re-render, inline images, and the legacy plain-text contract."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from linac_gen_gui.interphase.dialogs.assistant_chat import (  # noqa: E402
    ChatView, md_to_html,
)


def test_md_to_html_prose_and_code():
    h = md_to_html("# Title\n**bold** and *italic* and `code`\n"
                   "- item one\n- item two\n"
                   "```python\nx = 1\n```\ntail")
    assert "Title" in h and "<b>bold</b>" in h and "<i>italic</i>" in h
    assert "item one</li>" in h
    assert "<pre" in h and "x = 1" in h or "x" in h
    assert "tail" in h


def test_md_to_html_escapes_html():
    h = md_to_html("<script>alert(1)</script>")
    assert "<script>" not in h
    assert "&lt;script&gt;" in h


def test_md_to_html_never_raises():
    assert md_to_html(None) is not None
    assert md_to_html("```unclosed\nfence") is not None


def test_roles_render_and_plaintext_preserved(qapp):
    v = ChatView()
    v.append_line("▶ run the envelope")
    v.append_line("  · run_envelope …")
    v.append_line("  ‣ [event] job finished")
    v.append_line("[ready] hello")
    v.add_message("assistant", "σx is **2.1 mm** at exit")
    t = v.toPlainText()
    assert "▶ run the envelope" in t
    assert "· run_envelope" in t
    assert "‣ [event] job finished" in t
    assert "[ready] hello" in t
    assert "2.1 mm" in t


def test_stream_rerenders_as_markdown(qapp):
    v = ChatView()
    v.stream_delta("The beam is ")
    v.stream_delta("**matched** at exit.")
    assert "The beam is **matched** at exit." in v.toPlainText()
    raw = v.end_stream()
    assert raw == "The beam is **matched** at exit."
    # after re-render the literal asterisks are GONE (rendered bold)
    t = v.toPlainText()
    assert "**matched**" not in t
    assert "matched" in t


def test_end_stream_without_stream_is_safe(qapp):
    v = ChatView()
    assert v.end_stream() == ""


def test_add_image_inserts_resource(qapp, tmp_path):
    from PyQt6.QtGui import QImage
    p = tmp_path / "cap.png"
    img = QImage(500, 300, QImage.Format.Format_RGB32)
    img.fill(0xFF2244AA)
    img.save(str(p))
    v = ChatView()
    v.add_image(str(p), caption="(look_at_plot)")
    assert "(look_at_plot)" in v.toPlainText()
    assert v._img_n == 1


def test_add_image_missing_file_is_safe(qapp, tmp_path):
    v = ChatView()
    v.add_image(str(tmp_path / "nope.png"))
    assert v._img_n == 0


def test_capture_event_flows_to_inline_image(qapp, tmp_path):
    """End-to-end seam: a tool result with saved_to + image mime emits a
    capture event and the panel inlines the thumbnail."""
    from PyQt6.QtGui import QImage
    from tests.gui.test_assistant_stop_events import _mock_panel
    from linac_gen.assist.testing import turn_text
    p = tmp_path / "shot.png"
    img = QImage(64, 64, QImage.Format.Format_RGB32)
    img.fill(0xFFFFFFFF)
    img.save(str(p))
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._session._emit(type="capture", path=str(p),
                         tool="look_at_plot")
    qapp.processEvents()
    assert panel._transcript._img_n == 1
    assert "(look_at_plot)" in panel._transcript.toPlainText()
    panel.shutdown()


def test_text_size_rerenders_live(qapp):
    """The A-/A+ path: set_text_size re-renders EXISTING messages at
    the new size and clamps to sane bounds."""
    v = ChatView()
    v.add_message("assistant", "the exit sigma is **2.1 mm**")
    v.add_message("user", "thanks")
    assert v.body_px == 18                   # bigger default
    v.set_text_size(24)
    assert v.body_px == 24
    t = v.toPlainText()                      # history survived rebuild
    assert "2.1 mm" in t and "▶ thanks" in t
    v.set_text_size(999)
    assert v.body_px == 28                   # clamped
    v.set_text_size(1)
    assert v.body_px == 12


def test_font_buttons_persist(qapp, tmp_path):
    from tests.gui.test_assistant_stop_events import _mock_panel
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    px0 = panel._transcript.body_px
    panel._on_font_bigger()
    assert panel._transcript.body_px == px0 + 2
    assert int(panel._settings().value("assist/chat_px")) == px0 + 2
    panel._on_font_smaller()
    assert panel._transcript.body_px == px0
    panel.shutdown()
