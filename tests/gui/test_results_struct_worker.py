"""Regression tests: the Results-tab σ₀ worker is cancellable and a
closing popup never abandons (or GC-kills) a live worker thread."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402

from linac_gen.core.lattice import Lattice  # noqa: E402
from linac_gen.core.particle import PROTON  # noqa: E402
from linac_gen.core.reference import ReferenceParticle  # noqa: E402
from linac_gen.elements.drift import Drift  # noqa: E402
from linac_gen.elements.quadrupole import Quadrupole  # noqa: E402
from linac_gen.analysis.period_detect import detect_periods  # noqa: E402


def _fodo():
    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=10.0))
    return lat


def test_struct_worker_emits_nothing_on_cancel(qapp):
    from linac_gen_gui.interphase.tabs.results_tab import _StructWorker

    lat = _fodo()
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    period = next(p for p in detect_periods(lat)
                  if p.source == "type_sequence")

    w = _StructWorker(lat, ref, period, key=("k",))
    emissions = []
    w.finished_signal.connect(lambda *a: emissions.append(("ok", a)))
    w.failed_signal.connect(lambda *a: emissions.append(("fail", a)))

    w.request_stop()
    w.run()                        # synchronous — house pattern
    # Cancel emits NOTHING: a partial σ₀ result delivered through
    # finished_signal would be cached by the popup as valid data.
    assert emissions == []

    # Sanity: without the stop the same worker computes and emits.
    w2 = _StructWorker(lat, ref, period, key=("k",))
    got = []
    w2.finished_signal.connect(lambda *a: got.append(a))
    w2.run()
    assert len(got) == 1


def test_popup_close_parks_unstoppable_worker(qapp):
    """closeEvent: request stop, bounded wait; on timeout the thread is
    parked in the module zombie list (dropping the last reference to a
    live QThread aborts the process) and pruned when it finishes."""
    from linac_gen_gui.interphase.tabs import results_tab as rt

    class _FakeWorker(QObject):
        finished = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.stop_requested = False
            self.running = True

        def isRunning(self):
            return self.running

        def request_stop(self):
            self.stop_requested = True

        def requestInterruption(self):
            pass

        def wait(self, _ms):
            return False           # simulates a straggler

    popup = rt._PopupPlot(title="t")
    fake = _FakeWorker()
    popup._worker = fake

    popup.close()

    assert fake.stop_requested
    assert fake in rt._ZOMBIE_WORKERS
    # Thread eventually exits → pruned from the parking list.
    fake.running = False
    fake.finished.emit()
    assert fake not in rt._ZOMBIE_WORKERS
    popup.deleteLater()
