"""Background simulation worker thread."""
from PyQt6.QtCore import QThread, pyqtSignal


class SimulationWorker(QThread):
    progress = pyqtSignal(int)       # percentage
    finished = pyqtSignal(object)    # results (DiagnosticRecorder or EnvelopeResults)
    error = pyqtSignal(str)

    def __init__(self, simulation, mode="multiparticle"):
        super().__init__()
        self.simulation = simulation
        self.mode = mode
        self._stop_flag = False

    def run(self):
        try:
            if self.mode == "envelope":
                result = self.simulation.run_envelope()
            else:
                result = self.simulation.run()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._stop_flag = True
