"""The GUI's backtrack write keeps the rule of ``write_final_dst``: the current in
the file is the one the particles in it carry."""
import os

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402

from linac_gen.io.tracewin_dst import load_dst  # noqa: E402

FODO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))


class _Worker(QObject):
    """The sender the slot checks for, with the beam a worker would hold."""
    done = pyqtSignal(object)

    def __init__(self, beam):
        super().__init__()
        self.beam = beam


@pytest.mark.parametrize("lost", [0, 30], ids=["nothing lost", "some lost"])
def test_the_file_holds_the_current_of_its_particles(win, tmp_path, lost):
    win.open_lattice(FODO)
    win.beam_tab._current.setValue(5.0)
    win.beam_tab._npart.setValue(300)
    win.beam_tab._apply()
    win._run_mp()
    win.wait_worker(win._mp_worker)
    results = win.state.results
    beam = win._mp_worker.beam
    beam.lost[:] = False            # the count this test decides, not the run's
    beam.lost[:lost] = True
    worker = _Worker(beam)
    worker.done.connect(win._backtrack_done)
    win._backtrack_worker = worker
    out = tmp_path / "entrance.dst"
    win._backtrack_write_dst = str(out)
    worker.done.emit(results)
    parts, head = load_dst(str(out))
    assert parts.shape[0] == 300 - lost
    assert head["current_mA"] == pytest.approx(5.0 * (300 - lost) / 300, rel=1e-12)
    if not lost:
        assert head["current_mA"] == 5.0
