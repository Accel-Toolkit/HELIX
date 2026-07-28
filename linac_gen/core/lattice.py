"""Lattice: ordered container of elements plus a global step configuration."""
import numpy as np
from linac_gen.elements.base import Element
from linac_gen.core.step_config import StepConfig


class Lattice:
    def __init__(self):
        self.elements: list[Element] = []
        # Default step density -- overridden by ``PARTRAN_STEP`` when parsing
        # a TraceWin .dat file.
        self.step_config: StepConfig = StepConfig()
        # ErrorDef list parsed from TraceWin ERROR_* directives in the .dat
        # file (or attached programmatically).  ErrorStudy.run() reads this
        # alongside any user-added ``add_error`` calls.  Empty by default —
        # a lattice with no error directives runs deterministically.
        self.errors: list = []
        # Beam-error specs from ``ERROR_BEAM_STAT`` — perturbation sigmas /
        # half-widths on BeamConfig per seed.  See linac_gen.errors for shape.
        self.beam_errors: list = []
        # Multi-scale ratio sweep (``ERROR_SET_RATIO`` directive).  Empty
        # = single 100 % sweep; populated = run the study at each ratio
        # (e.g. [0, 0.25, 0.5, 1.0]) and compare envelopes.
        self.error_ratios: list[float] = []
        # Global Gaussian truncation (``ERROR_GAUSSIAN_CUT_OFF σ``); 0 = no
        # truncation (use ErrorDef.cutoff defaults instead).
        self.error_cutoff: float = 0.0

    def add(self, element: Element) -> None:
        self.elements.append(element)

    def insert(self, index: int, element: Element) -> None:
        """Insert element at the given index (clamped to [0, len])."""
        i = max(0, min(int(index), len(self.elements)))
        self.elements.insert(i, element)

    def remove(self, element: Element) -> int:
        """Remove element by identity. Returns the index it occupied.

        Raises ``ValueError`` if the element is not present (matching the
        Python list ``remove`` convention).
        """
        for i, e in enumerate(self.elements):
            if e is element:
                del self.elements[i]
                return i
        raise ValueError("element not present in lattice")

    @property
    def total_length(self) -> float:
        return sum(e.length for e in self.elements)

    def get_s_positions(self) -> tuple[np.ndarray, np.ndarray]:
        s = 0.0
        s_start = []
        s_end = []
        for e in self.elements:
            s_start.append(s)
            s += e.length
            s_end.append(s)
        return np.array(s_start), np.array(s_end)

    def get_element(self, name: str) -> Element:
        for e in self.elements:
            if e.name == name:
                return e
        raise KeyError(f"Element '{name}' not found in lattice")

    def __len__(self) -> int:
        return len(self.elements)
