"""HELIX's warnings reach the status bar, and the terminal keeps them too.

They used to reach only stderr, which a GUI user does not see, and the status
message clears after six seconds, so the bar keeps a count until it is cleared.
"""
import gc
import logging
import threading
import warnings

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from linac_gen_gui.interphase import warning_log  # noqa: E402
from linac_gen_gui.interphase.chrome.statusbar import MAX_WARNINGS, StatusBar  # noqa: E402
from linac_gen_gui.interphase.state import AppState  # noqa: E402
from linac_gen_gui.interphase.warning_log import install  # noqa: E402
from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QPlainTextEdit  # noqa: E402

from linac_gen.core.config import BeamConfig  # noqa: E402
from linac_gen.core.lattice import Lattice  # noqa: E402
from linac_gen.elements.drift import Drift  # noqa: E402
from tests.io.test_tracewin_dst_flagged import _flagged_file  # noqa: E402

LOG = logging.getLogger("linac_gen.distributions.factory")
HELIX_FILE = "/somewhere/linac_gen/io/some_module.py"


@pytest.fixture
def bar(qapp):
    state = AppState()
    widget = StatusBar(state)
    install(state)
    widget.test_state = state       # kept alive with the bar
    yield widget
    widget.clear_warnings()
    widget.deleteLater()


def _shown(widget):
    return not widget._warn_seg.isHidden()


def _helix_warning(text, category=UserWarning, filename=HELIX_FILE):
    """Through the warnings module's own route, as a ``warnings.warn`` would go."""
    warnings.warn_explicit(text, category, filename, 1, registry={})


# ---------------------------------------------------------------------------
# What the bar shows
# ---------------------------------------------------------------------------
def test_nothing_shows_until_there_is_a_warning(bar):
    assert not _shown(bar)
    LOG.info("not a warning")
    assert not _shown(bar)


def test_a_logged_warning_shows_its_count_and_its_text(bar):
    LOG.warning("the beam has a problem")
    assert _shown(bar)
    assert bar._warn_seg.text() == "1 warning"
    assert bar._msg_seg.text() == "the beam has a problem"
    LOG.warning("another problem")
    assert bar._warn_seg.text() == "2 warnings"


def test_the_list_outlasts_the_message_segment(bar):
    LOG.warning("the beam has a problem")
    bar._clear_message()
    assert bar._msg_seg.text() == ""
    assert _shown(bar)
    assert "the beam has a problem" in bar.warning_text()


def test_a_repeat_counts_once_and_says_how_often(bar):
    for _ in range(3):
        LOG.warning("the same problem")
    assert bar._warn_seg.text() == "1 warning"
    assert "the same problem  (×3)" in bar.warning_text()


def test_other_levels_of_the_event_are_ignored(bar):
    bar.test_state.log_event.emit("info", "just saying")
    assert not _shown(bar)


def test_the_list_is_capped(bar):
    for i in range(MAX_WARNINGS + 5):
        LOG.warning("problem %d", i)
    assert bar._warn_seg.text() == f"{MAX_WARNINGS}+ warnings"
    assert "problem 0" in bar.warning_text()
    assert f"problem {MAX_WARNINGS + 4}" not in bar.warning_text()
    LOG.warning("problem 0")          # an entry already listed still counts
    assert "problem 0  (×2)" in bar.warning_text()


def test_a_click_opens_a_window_that_does_not_block(bar):
    LOG.warning("the beam has a problem")
    bar._warn_seg.click()
    dlg = bar._warn_dialog
    assert dlg is not None and dlg.isVisible() and not dlg.isModal()
    assert "the beam has a problem" in dlg.findChild(QPlainTextEdit).toPlainText()
    LOG.warning("while it is open")
    assert "while it is open" in dlg.findChild(QPlainTextEdit).toPlainText()


def test_return_in_the_window_does_not_wipe_the_list(bar):
    LOG.warning("the beam has a problem")
    bar._warn_seg.click()
    text = bar._warn_dialog.findChild(QPlainTextEdit)
    text.setFocus()
    QTest.keyClick(text, Qt.Key.Key_Return)
    assert bar.warning_text() == "the beam has a problem"
    assert bar._warn_dialog.isVisible()


def test_a_new_warning_leaves_the_reader_where_they_scrolled_to(bar):
    for i in range(150):
        LOG.warning("problem %d", i)
    bar._warn_seg.click()
    bar._warn_dialog.resize(300, 120)
    text = bar._warn_dialog.findChild(QPlainTextEdit)
    scroll = text.verticalScrollBar()
    assert scroll.maximum() > 50
    scroll.setValue(40)
    LOG.warning("a new one")
    assert "a new one" in text.toPlainText()
    assert scroll.value() == 40


def test_clear_empties_the_list_and_hides_the_button(bar):
    LOG.warning("the beam has a problem")
    bar._warn_seg.click()
    bar.clear_warnings()
    assert not _shown(bar)
    assert bar.warning_text() == ""
    assert not bar._warn_dialog.isVisible()
    LOG.warning("a new one")
    assert bar._warn_seg.text() == "1 warning"


def test_a_warning_from_another_thread_arrives(bar, qapp):
    t = threading.Thread(target=LOG.warning, args=("from a worker",))
    t.start()
    t.join()
    for _ in range(50):
        qapp.processEvents()
        if _shown(bar):
            break
    assert "from a worker" in bar.warning_text()


# ---------------------------------------------------------------------------
# The terminal keeps them
# ---------------------------------------------------------------------------
def test_stderr_still_gets_the_line_when_nothing_else_prints_it(bar, monkeypatch, capsys):
    """Python prints through ``logging.lastResort`` only while no handler exists at
    all, so adding one would silence the terminal."""
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    LOG.warning("for the terminal")
    assert capsys.readouterr().err.count("for the terminal") == 1
    assert _shown(bar)


def test_stderr_does_not_get_it_twice_with_a_root_handler(bar, monkeypatch, capsys):
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
    LOG.warning("for the root handler")
    assert capsys.readouterr().err == ""
    assert _shown(bar)


# ---------------------------------------------------------------------------
# warnings.warn
# ---------------------------------------------------------------------------
def test_a_warning_raised_by_helix_code_is_shown(bar):
    _helix_warning("a surrogate is stale")
    assert "a surrogate is stale" in bar.warning_text()


def test_a_subclass_of_user_warning_is_shown(bar):
    class Odd(UserWarning):
        pass
    _helix_warning("odd but ours", category=Odd)
    assert "odd but ours" in bar.warning_text()


def test_numpy_style_warnings_are_not_shown(bar):
    """numpy attributes "invalid value encountered" to the calling line, which may
    well be ours."""
    _helix_warning("invalid value encountered in divide", category=RuntimeWarning)
    assert not _shown(bar)


def test_another_librarys_warning_is_not_shown(bar):
    _helix_warning("something in a library", filename="/site-packages/lib/x.py")
    assert not _shown(bar)


def test_whatever_else_shows_warnings_still_runs(qapp):
    state = AppState()
    seen = []
    with warnings.catch_warnings():
        warnings.showwarning = lambda *a, **k: seen.append(a[0])
        install(state)
        _helix_warning("for the previous one too")
    assert [str(m) for m in seen] == ["for the previous one too"]


def test_a_recording_block_gets_its_warning_and_the_bar_too(bar):
    """The backtrack worker records its warnings for itself, in its own thread."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _helix_warning("recorded")
    assert [str(w.message) for w in seen] == ["recorded"]
    assert bar.warning_text() == "recorded"


def test_overlapping_recording_blocks_do_not_disconnect_the_bar(bar):
    """Two threads' blocks that end out of order leave showwarning and its recorder
    wrong for the rest of the session, here restored by the outer block."""
    with warnings.catch_warnings():
        first, second = warnings.catch_warnings(record=True), warnings.catch_warnings(record=True)
        first.__enter__()
        second.__enter__()
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)
        _helix_warning("after the overlap")
    assert bar.warning_text() == "after the overlap"


def test_a_python_without_the_private_name_still_gets_the_logged_warnings(qapp, monkeypatch):
    monkeypatch.delattr(warnings, "_showwarnmsg")
    state = AppState()
    widget = StatusBar(state)
    install(state)
    LOG.warning("logged anyway")
    assert widget.warning_text() == "logged anyway"
    widget.deleteLater()


# ---------------------------------------------------------------------------
# Installing and lifetime
# ---------------------------------------------------------------------------
def test_installing_twice_forwards_once(bar):
    install(bar.test_state)
    install(bar.test_state)
    LOG.warning("once")
    _helix_warning("once too")
    assert bar.warning_text() == "once\nonce too"


def test_a_window_built_inside_a_block_keeps_getting_warnings_after_it(qapp):
    """pytest wraps each test in such a block."""
    state = AppState()
    widget = StatusBar(state)
    with warnings.catch_warnings():
        install(state)
    _helix_warning("after the block")
    assert widget.warning_text() == "after the block"
    widget.deleteLater()


def test_a_second_window_takes_over_and_the_first_gets_nothing(bar, qapp):
    other_state = AppState()
    other = StatusBar(other_state)
    install(other_state)
    LOG.warning("for the new window")
    assert "for the new window" in other.warning_text()
    assert bar.warning_text() == ""
    other.deleteLater()


def test_a_collected_state_is_harmless(qapp):
    state = AppState()
    install(state)
    del state
    gc.collect()
    assert warning_log._state_ref() is None       # the handler does not keep it alive
    LOG.warning("nobody home")


def test_a_deleted_state_is_harmless(qapp):
    state = AppState()
    install(state)
    sip.delete(state)
    LOG.warning("nobody home")
    _helix_warning("nobody home either")


# ---------------------------------------------------------------------------
# Through the real run slot
# ---------------------------------------------------------------------------
def _flagged_beam_rows():
    """200 particles in the beam and 50 flagged lost: cm, rad, MeV, flag."""
    rng = np.random.default_rng(3)
    rows = np.zeros((250, 7))
    rows[:, 0:4] = rng.normal(0.0, [0.1, 0.001, 0.1, 0.001], (250, 4))
    rows[:, 4] = rng.normal(0.5, 0.05, 250)
    rows[:, 5] = rng.normal(2.1, 0.001, 250)
    rows[200:, 6] = 29.0
    return rows


def test_the_run_slot_shows_the_beam_warning(win, tmp_path):
    path = _flagged_file(tmp_path / "beam.dst", _flagged_beam_rows())
    lat = Lattice()
    lat.add(Drift("D", length=50.0, aperture=50.0))
    win.state.set_lattice(lat)
    win.state.set_beam_config(BeamConfig(
        species="H-", energy=2.1, frequency=162.5, current=6.8, n_particles=200,
        source="file", distribution_file=str(path)))
    win._run_mp()
    win.wait_worker(win._mp_worker)
    assert not win.message_boxes
    text = win._statusbar.warning_text()
    assert "50 of 250" in text and "5.44" in text and "6.8" in text
    assert not win._statusbar._warn_seg.isHidden()


def test_a_stale_surrogate_reaches_the_bar_when_it_is_loaded(bar, tmp_path):
    """By the route the Surrogates tab loads a cached file."""
    from linac_gen.surrogates.training import find_cached_surrogate
    from tests.surrogates.test_frequency_jump import _cavity
    from tests.surrogates.test_surrogate_provenance import _save, _z_block
    assert find_cached_surrogate(_save(tmp_path, _cavity(), _z_block(2.0))) is not None
    assert _shown(bar)
    assert "stale" in bar.warning_text()
