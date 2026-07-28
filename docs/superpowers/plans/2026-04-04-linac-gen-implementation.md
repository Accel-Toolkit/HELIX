# Linac_Gen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-featured proton/ion linac particle tracking code with 3D PIC space charge, TraceWin-compatible I/O, and a PyQt6 desktop GUI.

**Architecture:** Two-package monorepo: `linac_gen` (computation library, zero GUI deps) and `linac_gen_gui` (PyQt6 desktop app). Performance-critical PIC kernels in C++ via pybind11/FFTW. Particles stored as (N,6) deviation arrays relative to a ReferenceParticle.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pybind11, FFTW3, PyQt6, pyqtgraph, h5py, pytest

**Spec:** `docs/superpowers/specs/2026-04-03-linac-gen-particle-tracking-design.md`

---

## Phase Overview

| Phase | What it delivers | Depends on |
|-------|-----------------|------------|
| 1 | Project scaffold, core data model, configs, element interfaces | — |
| 2 | Drift + Quadrupole (TransferMapElement), matrix tracking, diagnostics | Phase 1 |
| 3 | Distributions, beam generation from BeamConfig | Phase 1 |
| 4 | Multi-particle tracker (no SC), aperture, marker, steerer | Phase 2, 3 |
| 5 | PIC space charge (Python-only, incl. dphi↔z conversion) | Phase 4 |
| 6 | RF gap, solenoid, dipole, multipole elements | Phase 2 |
| 7 | Field map reader, FieldMapElement+RK4, TraceWin .dat parser/writer | Phase 4, 6 |
| 8 | C++ PIC kernels (pybind11 + FFTW) | Phase 5 |
| 9 | Envelope solver (Sacherer equations) | Phase 2, 6 |
| 10 | Matching module (envelope + multi-particle modes) | Phase 4, 9 |
| 11 | Error study framework + orbit correction | Phase 4, 10 |
| 12 | Distribution I/O + HDF5 results output | Phase 3, 4 |
| 13 | GUI: main window, lattice editor, beam config panel | Phase 3, 7 |
| 14 | GUI: plot widgets (envelope, phase space, loss map, emittance) | Phase 13 |
| 15 | GUI: simulation runner, matching dialog, error study dialog | Phase 4, 5, 10, 11, 14 |

---

## Phase 1: Project Scaffold & Core Data Model

### Task 1.1: Project scaffold and pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `linac_gen/__init__.py`
- Create: `linac_gen/core/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/core/__init__.py`

- [ ] **Step 1: Create two pyproject.toml files (true two-package monorepo)**

`pyproject.toml` (root — library package):
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "linac_gen"
version = "0.1.0"
description = "Particle tracking code with PIC space charge for proton/ion linacs"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "h5py>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.setuptools.packages.find]
include = ["linac_gen*"]
exclude = ["linac_gen_gui*"]
```

`gui/pyproject.toml` (GUI package — separate distribution, in its own subdirectory):
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "linac_gen_gui"
version = "0.1.0"
description = "Desktop GUI for Linac_Gen particle tracking"
requires-python = ">=3.10"
dependencies = [
    "linac_gen>=0.1.0",
    "PyQt6>=6.5",
    "pyqtgraph>=0.13",
]

[tool.setuptools.packages.find]
include = ["linac_gen_gui*"]
```

The GUI package lives under `gui/` with this layout:
```
gui/
├── pyproject.toml
└── linac_gen_gui/
    ├── __init__.py
    ├── app.py
    ├── main_window.py
    ├── widgets/
    └── dialogs/
```

This ensures `pip install linac_gen` installs only the library (no Qt), and `pip install gui/` pulls in Qt + the library. During development: `pip install -e . && pip install -e gui/`.

- [ ] **Step 2: Create package init files**

`linac_gen/__init__.py`:
```python
"""Linac_Gen: Particle tracking with PIC space charge for proton/ion linacs."""
__version__ = "0.1.0"
```

`linac_gen/core/__init__.py`:
```python
"""Core data structures: Particle, ReferenceParticle, Beam, Lattice."""
```

`gui/linac_gen_gui/__init__.py`:
```python
"""Linac_Gen GUI: Desktop application for particle tracking simulation."""
__version__ = "0.1.0"
```

`tests/__init__.py` and `tests/core/__init__.py`: empty files.

- [ ] **Step 3: Verify scaffold**

Run: `cd /mnt/c/Project/Linac_Gen && python -c "import linac_gen; print(linac_gen.__version__)"`
Expected: `0.1.0`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml linac_gen/ gui/ tests/
git commit -m "feat: two-package project scaffold with pyproject.toml"
```

---

### Task 1.2: Physics constants

**Files:**
- Create: `linac_gen/core/constants.py`
- Create: `tests/core/test_constants.py`

- [ ] **Step 1: Write the test**

```python
# tests/core/test_constants.py
from linac_gen.core.constants import C_LIGHT, E_CHARGE, M_PROTON, AMU, EPSILON_0, PI

def test_speed_of_light_mm_per_ns():
    # c = 299792458 m/s = 299.792458 mm/ns
    assert abs(C_LIGHT - 299792458.0) < 1.0  # m/s

def test_proton_mass_mev():
    assert abs(M_PROTON - 938.272) < 0.01  # MeV/c^2

def test_amu_mev():
    assert abs(AMU - 931.494) < 0.01  # MeV/c^2

def test_elementary_charge():
    assert abs(E_CHARGE - 1.602176634e-19) < 1e-25  # Coulombs
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/core/test_constants.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement constants**

```python
# linac_gen/core/constants.py
"""Physics constants in SI units unless noted."""
import math

PI = math.pi
C_LIGHT = 299792458.0           # speed of light (m/s)
E_CHARGE = 1.602176634e-19      # elementary charge (C)
M_PROTON = 938.27208816         # proton mass (MeV/c^2)
M_ELECTRON = 0.51099895000      # electron mass (MeV/c^2)
AMU = 931.49410242              # atomic mass unit (MeV/c^2)
EPSILON_0 = 8.8541878128e-12    # vacuum permittivity (F/m)
MU_0 = 1.25663706212e-6        # vacuum permeability (H/m)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/core/test_constants.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/core/constants.py tests/core/test_constants.py
git commit -m "feat: physics constants module"
```

---

### Task 1.3: Particle species

**Files:**
- Create: `linac_gen/core/particle.py`
- Create: `tests/core/test_particle.py`

- [ ] **Step 1: Write the test**

```python
# tests/core/test_particle.py
import pytest
from linac_gen.core.particle import Particle, PROTON, DEUTERON, H_MINUS

def test_proton_mass():
    assert abs(PROTON.mass - 938.272) < 0.01

def test_proton_charge():
    assert PROTON.charge == 1

def test_deuteron():
    assert abs(DEUTERON.mass - 1875.613) < 0.01
    assert DEUTERON.charge == 1

def test_h_minus():
    assert H_MINUS.charge == -1
    assert abs(H_MINUS.mass - PROTON.mass) < 0.01  # same mass as proton

def test_custom_ion():
    # Carbon-12, charge state 6+
    carbon = Particle(mass=12 * 931.494, charge=6, name="C12_6+")
    assert carbon.charge == 6
    assert carbon.name == "C12_6+"

def test_particle_repr():
    assert "proton" in repr(PROTON)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/core/test_particle.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Particle**

```python
# linac_gen/core/particle.py
"""Particle species definitions."""
from dataclasses import dataclass
from linac_gen.core.constants import M_PROTON, AMU

@dataclass(frozen=True)
class Particle:
    """A particle species defined by rest mass and charge state."""
    mass: float    # rest mass (MeV/c^2)
    charge: int    # charge state (units of e, can be negative)
    name: str = ""

    def __repr__(self) -> str:
        return f"Particle(name='{self.name}', mass={self.mass:.3f} MeV/c², charge={self.charge:+d})"

# Built-in species
PROTON = Particle(mass=M_PROTON, charge=1, name="proton")
DEUTERON = Particle(mass=1875.61294, charge=1, name="deuteron")  # CODATA 2018 value
H_MINUS = Particle(mass=M_PROTON, charge=-1, name="H-")
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/core/test_particle.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/core/particle.py tests/core/test_particle.py
git commit -m "feat: Particle species with proton, deuteron, H-minus presets"
```

---

### Task 1.4: ReferenceParticle

**Files:**
- Create: `linac_gen/core/reference.py`
- Create: `tests/core/test_reference.py`

- [ ] **Step 1: Write the test**

```python
# tests/core/test_reference.py
import pytest
import math
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.constants import C_LIGHT

def test_create_reference():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    assert ref.w_kin == 3.0
    assert ref.phi_s == 0.0
    assert ref.s == 0.0

def test_gamma_from_kinetic_energy():
    ref = ReferenceParticle(species=PROTON, w_kin=938.272, frequency=352.21)
    # gamma = 1 + W_kin / mass = 1 + 1.0 = 2.0
    assert abs(ref.gamma - 2.0) < 0.001

def test_beta_from_gamma():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    gamma = 1.0 + 3.0 / PROTON.mass
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    assert abs(ref.beta - beta) < 1e-10

def test_bg():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    assert abs(ref.bg - ref.beta * ref.gamma) < 1e-10

def test_brho():
    # Brho = p / q = m*c*beta*gamma / q (in T.m)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    # p in MeV/c = mass * beta * gamma
    p_mev = PROTON.mass * ref.bg  # MeV/c
    # Brho = p (eV/c) / (c * q) = p * 1e6 / (C_LIGHT * 1) in T.m
    brho_expected = p_mev * 1e6 / (C_LIGHT * abs(PROTON.charge))
    assert abs(ref.brho - brho_expected) < 1e-6

def test_wavelength():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    # wavelength = c / f in mm: (299792458 m/s) / (352.21e6 Hz) * 1000 mm/m
    wl_expected = C_LIGHT / (352.21e6) * 1000.0  # mm
    assert abs(ref.wavelength - wl_expected) < 0.01

def test_update_energy():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    old_gamma = ref.gamma
    ref.w_kin = 10.0
    assert ref.gamma > old_gamma
    assert abs(ref.gamma - (1.0 + 10.0 / PROTON.mass)) < 1e-10

def test_advance_s():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    ref.s += 100.0
    assert ref.s == 100.0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/core/test_reference.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ReferenceParticle**

```python
# linac_gen/core/reference.py
"""ReferenceParticle: synchronous particle state tracking."""
import math
from linac_gen.core.particle import Particle
from linac_gen.core.constants import C_LIGHT

class ReferenceParticle:
    """Tracks the synchronous particle's absolute state through the lattice.

    All derived quantities (beta, gamma, brho, etc.) are recomputed whenever
    w_kin is updated via the property setter.
    """

    def __init__(self, species: Particle, w_kin: float, frequency: float,
                 phi_s: float = 0.0, s: float = 0.0):
        self.species = species
        self.frequency = frequency  # MHz
        self.phi_s = phi_s          # absolute RF phase (deg)
        self.s = s                  # position along lattice (mm)
        self._w_kin = w_kin
        self._update_derived()

    @property
    def w_kin(self) -> float:
        """Absolute kinetic energy (MeV)."""
        return self._w_kin

    @w_kin.setter
    def w_kin(self, value: float) -> None:
        self._w_kin = value
        self._update_derived()

    def _update_derived(self) -> None:
        """Recompute beta, gamma, bg, brho, wavelength from current w_kin."""
        mass = self.species.mass
        self.gamma = 1.0 + self._w_kin / mass
        self.beta = math.sqrt(1.0 - 1.0 / (self.gamma * self.gamma))
        self.bg = self.beta * self.gamma
        # brho = p / (|q| * e) in T.m: p in eV/c, divide by c in m/s
        p_ev = mass * self.bg * 1e6  # eV/c
        self.brho = p_ev / (C_LIGHT * abs(self.species.charge))  # T.m
        # wavelength = c / f in mm
        self.wavelength = C_LIGHT / (self.frequency * 1e6) * 1000.0  # mm

    def copy(self) -> "ReferenceParticle":
        """Return an independent copy."""
        return ReferenceParticle(
            species=self.species, w_kin=self._w_kin,
            frequency=self.frequency, phi_s=self.phi_s, s=self.s,
        )
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/core/test_reference.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/core/reference.py tests/core/test_reference.py
git commit -m "feat: ReferenceParticle with auto-computed relativistic quantities"
```

---

### Task 1.5: Beam class

**Files:**
- Create: `linac_gen/core/beam.py`
- Create: `tests/core/test_beam.py`

- [ ] **Step 1: Write the test**

```python
# tests/core/test_beam.py
import numpy as np
import pytest
from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle

def test_create_beam():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=1000, current=60.0)
    assert beam.particles.shape == (1000, 6)
    assert beam.n_particles == 1000
    assert beam.current == 60.0

def test_particles_initialized_to_zero():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=100, current=0.0)
    assert np.all(beam.particles == 0.0)

def test_lost_mask():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=100, current=0.0)
    assert beam.lost.shape == (100,)
    assert not np.any(beam.lost)
    beam.lost[5] = True
    assert beam.n_alive == 99

def test_alive_particles():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    beam.lost[3] = True
    beam.lost[7] = True
    alive = beam.alive_particles
    assert alive.shape == (8, 6)

def test_record_loss():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    beam.particles[3, 0] = 25.0  # x = 25 mm
    beam.particles[3, 2] = 10.0  # y = 10 mm
    beam.particles[3, 5] = 0.5   # dW = 0.5 MeV
    beam.record_loss(particle_id=3, s=100.0, element_name="QUAD_01")
    assert beam.lost[3] == True
    assert len(beam.loss_table) == 1
    assert beam.loss_table[0]["s"] == 100.0
    assert beam.loss_table[0]["element_name"] == "QUAD_01"

def test_species_shortcut():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    assert beam.species is PROTON

def test_frequency_shortcut():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    assert beam.frequency == 352.21
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/core/test_beam.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Beam**

```python
# linac_gen/core/beam.py
"""Beam class: particle ensemble with reference state."""
import numpy as np
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import Particle

# Column indices for the (N, 6) particle array
X, XP, Y, YP, DPHI, DW = 0, 1, 2, 3, 4, 5

# Loss table dtype
LOSS_DTYPE = np.dtype([
    ("particle_id", np.int32),
    ("s", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("energy", np.float64),
    ("element_name", "U32"),
])

class Beam:
    """A beam of macro-particles with deviations from a reference particle."""

    def __init__(self, ref: ReferenceParticle, n_particles: int, current: float):
        self.ref = ref
        self.current = current  # mA
        self.particles = np.zeros((n_particles, 6), dtype=np.float64)
        self.lost = np.zeros(n_particles, dtype=bool)
        self._loss_list: list = []

    @property
    def n_particles(self) -> int:
        return self.particles.shape[0]

    @property
    def n_alive(self) -> int:
        return int(np.count_nonzero(~self.lost))

    @property
    def alive_mask(self) -> np.ndarray:
        return ~self.lost

    @property
    def alive_particles(self) -> np.ndarray:
        return self.particles[self.alive_mask]

    @property
    def species(self) -> Particle:
        return self.ref.species

    @property
    def frequency(self) -> float:
        return self.ref.frequency

    def record_loss(self, particle_id: int, s: float, element_name: str) -> None:
        """Mark a particle as lost and record its loss data."""
        self.lost[particle_id] = True
        p = self.particles[particle_id]
        self._loss_list.append((
            particle_id, s, p[X], p[Y],
            self.ref.w_kin + p[DW], element_name,
        ))

    @property
    def loss_table(self) -> np.ndarray:
        """Structured array of all recorded losses."""
        if not self._loss_list:
            return np.array([], dtype=LOSS_DTYPE)
        return np.array(self._loss_list, dtype=LOSS_DTYPE)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/core/test_beam.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/core/beam.py tests/core/test_beam.py
git commit -m "feat: Beam class with deviation-based particle array and loss tracking"
```

---

### Task 1.6: Lattice class and Element base

**Files:**
- Create: `linac_gen/elements/__init__.py`
- Create: `linac_gen/elements/base.py`
- Create: `linac_gen/core/lattice.py`
- Create: `tests/core/test_lattice.py`
- Create: `tests/elements/__init__.py`

- [ ] **Step 1: Write the test**

```python
# tests/core/test_lattice.py
import numpy as np
import pytest
from linac_gen.core.lattice import Lattice
from linac_gen.elements.base import Element

class DummyElement(Element):
    """Minimal concrete element for testing (not capability-specific)."""
    def __init__(self, name: str, length: float):
        super().__init__(name=name, length=length, aperture=0.0, n_steps=1)

def test_create_empty_lattice():
    lat = Lattice()
    assert len(lat.elements) == 0
    assert lat.total_length == 0.0

def test_add_elements():
    lat = Lattice()
    lat.add(DummyElement("D1", 100.0))
    lat.add(DummyElement("Q1", 50.0))
    assert len(lat.elements) == 2
    assert lat.total_length == 150.0

def test_s_positions():
    lat = Lattice()
    lat.add(DummyElement("D1", 100.0))
    lat.add(DummyElement("Q1", 50.0))
    lat.add(DummyElement("D2", 200.0))
    s_start, s_end = lat.get_s_positions()
    np.testing.assert_array_almost_equal(s_start, [0.0, 100.0, 150.0])
    np.testing.assert_array_almost_equal(s_end, [100.0, 150.0, 350.0])

def test_get_element_by_name():
    lat = Lattice()
    lat.add(DummyElement("Q1", 50.0))
    lat.add(DummyElement("D1", 100.0))
    assert lat.get_element("Q1").name == "Q1"

def test_get_element_not_found():
    lat = Lattice()
    with pytest.raises(KeyError):
        lat.get_element("NONEXISTENT")
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/core/test_lattice.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Element base and Lattice**

```python
# linac_gen/elements/base.py
"""Abstract base classes for lattice elements."""
from abc import ABC

class Element(ABC):
    """Base class for all lattice elements."""

    def __init__(self, name: str, length: float, aperture: float, n_steps: int):
        self.name = name
        self.length = length      # mm
        self.aperture = aperture  # mm (0 = no aperture check)
        self.n_steps = n_steps    # integration sub-steps
```

```python
# linac_gen/core/lattice.py
"""Lattice: ordered container of elements."""
import numpy as np
from linac_gen.elements.base import Element

class Lattice:
    """An ordered sequence of accelerator elements."""

    def __init__(self):
        self.elements: list[Element] = []

    def add(self, element: Element) -> None:
        self.elements.append(element)

    @property
    def total_length(self) -> float:
        return sum(e.length for e in self.elements)

    def get_s_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (s_start, s_end) arrays for each element."""
        s = 0.0
        s_start = []
        s_end = []
        for e in self.elements:
            s_start.append(s)
            s += e.length
            s_end.append(s)
        return np.array(s_start), np.array(s_end)

    def get_element(self, name: str) -> Element:
        """Find element by name. Raises KeyError if not found."""
        for e in self.elements:
            if e.name == name:
                return e
        raise KeyError(f"Element '{name}' not found in lattice")

    def __len__(self) -> int:
        return len(self.elements)
```

`linac_gen/elements/__init__.py` and `tests/elements/__init__.py`: empty files.

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/core/test_lattice.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/elements/ linac_gen/core/lattice.py tests/core/test_lattice.py tests/elements/
git commit -m "feat: Element base class and Lattice container"
```

---

### Task 1.7: Capability-based element interfaces

**Files:**
- Modify: `linac_gen/elements/base.py`
- Create: `tests/elements/test_base.py`

- [ ] **Step 1: Write the test**

```python
# tests/elements/test_base.py
import numpy as np
from linac_gen.elements.base import (
    Element, TransferMapElement, ThinKickElement,
    FieldMapElement, PassiveElement,
)
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.beam import Beam

def test_transfer_map_element_is_element():
    assert issubclass(TransferMapElement, Element)

def test_thin_kick_element_zero_length():
    """ThinKickElement.__init__ forces length=0 via super().__init__."""
    class DummyKick(ThinKickElement):
        def apply_kick(self, beam): pass
        def kick_matrix(self, ref): return np.eye(6)
    k = DummyKick(name="K1")
    assert k.length == 0.0

def test_passive_element_zero_length():
    assert issubclass(PassiveElement, Element)

def test_field_map_element_is_element():
    assert issubclass(FieldMapElement, Element)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/elements/test_base.py -v`
Expected: FAIL (classes not defined)

- [ ] **Step 3: Implement capability-based interfaces**

Replace `linac_gen/elements/base.py` with:

```python
# linac_gen/elements/base.py
"""Abstract base classes for lattice elements (capability-based)."""
from abc import ABC, abstractmethod
import numpy as np

class Element(ABC):
    """Base class for all lattice elements."""
    def __init__(self, name: str, length: float, aperture: float, n_steps: int):
        self.name = name
        self.length = length      # mm
        self.aperture = aperture  # mm (0 = no aperture check)
        self.n_steps = n_steps

class TransferMapElement(Element):
    """Elements with a linear 6x6 transfer matrix (drift, quad, solenoid, dipole).
    The tracker calls transfer_matrix(ref, ds) with ds = half-step length for
    the split-operator scheme. track() advances beam.ref then applies the matrix.
    """
    @abstractmethod
    def transfer_matrix(self, ref, ds: float = None) -> np.ndarray:
        """Return 6x6 matrix for a slice of length ds (mm). None = full element."""
        ...

    @abstractmethod
    def track(self, beam, ds: float = None) -> None:
        """Advance beam.ref and apply transfer matrix to alive particles."""
        ...

class ThinKickElement(Element):
    """Zero-length elements with instantaneous kicks (RF gap, multipole, steerer).
    For RF gaps: updates beam.ref.w_kin and phi_s.
    For steerers/multipoles: beam.ref unchanged.
    """
    def __init__(self, name: str, aperture: float = 0.0):
        super().__init__(name=name, length=0.0, aperture=aperture, n_steps=0)

    @abstractmethod
    def apply_kick(self, beam) -> None:
        ...

    @abstractmethod
    def kick_matrix(self, ref) -> np.ndarray:
        """Linearized 6x6 matrix for envelope tracking."""
        ...

    def advance_ref(self, ref) -> None:
        """Advance reference particle state (energy, phase) through this element.
        Override in RF gaps to apply synchronous energy gain.
        Default: no-op (steerers, multipoles don't change ref)."""
        pass

class FieldMapElement(Element):
    """Elements tracked via RK4 through imported field data."""
    @abstractmethod
    def track_rk4(self, beam, ds: float) -> None:
        """RK4 step: integrates ref particle first, then all particles."""
        ...

    def fitted_matrix(self, ref) -> np.ndarray:
        """Linearized 6x6 matrix for envelope/matching mode.
        Implemented in Phase 7 Task 7.2: tracks a small test beam via RK4
        and computes the Jacobian. Returns np.eye(6) as placeholder."""
        return np.eye(6)  # placeholder until Phase 7 Task 7.2

    def advance_ref(self, ref) -> None:
        """Advance reference particle through the field map.
        Implemented in Phase 7 Task 7.2: runs ref through full RK4,
        updating ref.w_kin, ref.phi_s, ref.s. No-op as placeholder."""
        pass  # placeholder until Phase 7 Task 7.2

class PassiveElement(Element):
    """Zero-length elements with no dynamics (aperture, marker, diag, SC comp)."""
    def __init__(self, name: str):
        super().__init__(name=name, length=0.0, aperture=0.0, n_steps=0)

    @abstractmethod
    def apply(self, beam) -> None:
        ...
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/elements/test_base.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/elements/base.py tests/elements/test_base.py
git commit -m "feat: capability-based element interfaces (TransferMapElement, ThinKickElement, FieldMapElement, PassiveElement)"
```

---

### Task 1.8: BeamConfig, SpaceChargeConfig, and Simulation facade

**Files:**
- Create: `linac_gen/core/config.py`
- Create: `linac_gen/core/simulation.py`
- Create: `tests/core/test_config.py`

- [ ] **Step 1: Write the test**

```python
# tests/core/test_config.py
from linac_gen.core.config import BeamConfig, SpaceChargeConfig

def test_beam_config_defaults():
    bc = BeamConfig(species="proton", energy=3.0, frequency=352.21, current=60.0)
    assert bc.n_particles == 10000
    assert bc.distribution == "waterbag"
    assert bc.cutoff == 3.0

def test_beam_config_custom():
    bc = BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=60.0,
        n_particles=100000, distribution="gaussian",
        emit_nx=0.25, alpha_x=1.0, beta_x=0.12,
        emit_ny=0.25, alpha_y=-0.5, beta_y=0.08,
        emit_z=0.30, alpha_z=0.0, beta_z=1.5,
    )
    assert bc.n_particles == 100000
    assert bc.distribution == "gaussian"
    assert bc.emit_nx == 0.25

def test_sc_config_defaults():
    sc = SpaceChargeConfig()
    assert sc.nx == 64
    assert sc.boundary == "open"
    assert sc.grid_mode == "fixed"

def test_sc_config_custom():
    sc = SpaceChargeConfig(nx=128, ny=128, nz=128, boundary="conducting")
    assert sc.nx == 128
    assert sc.boundary == "conducting"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/core/test_config.py -v`
Expected: FAIL

- [ ] **Step 3: Implement configs and Simulation facade**

```python
# linac_gen/core/config.py
"""Configuration dataclasses for beam and space charge."""
from dataclasses import dataclass, field

@dataclass
class BeamConfig:
    """Initial beam parameters for distribution generation."""
    species: str = "proton"
    energy: float = 3.0             # initial kinetic energy (MeV)
    frequency: float = 352.21       # RF frequency (MHz)
    current: float = 0.0            # beam current (mA)
    n_particles: int = 10000
    distribution: str = "waterbag"  # waterbag, kv, gaussian, parabolic, uniform, file
    cutoff: float = 3.0             # sigma cutoff for gaussian
    emit_nx: float = 0.25           # normalized RMS emittance x (mm.mrad)
    alpha_x: float = 0.0
    beta_x: float = 0.1             # Twiss beta (m)
    emit_ny: float = 0.25
    alpha_y: float = 0.0
    beta_y: float = 0.1
    emit_z: float = 0.3             # longitudinal emittance (deg.MeV)
    alpha_z: float = 0.0
    beta_z: float = 1.0             # (deg/MeV)
    source: str = "generate"        # "generate" or "file"
    distribution_file: str = None

@dataclass
class SpaceChargeConfig:
    """PIC space charge solver configuration."""
    nx: int = 64
    ny: int = 64
    nz: int = 64
    boundary: str = "open"          # "open" or "conducting"
    solver: str = "fft"
    grid_extent: float = 3.0        # grid = N * sigma_rms per dimension
    shape_order: int = 1            # 1=CIC, 2=TSC
    grid_mode: str = "fixed"        # "fixed" or "adaptive"
    adaptive_interval: int = 50
    adaptive_threshold: float = 0.3
```

```python
# linac_gen/core/simulation.py
"""Simulation facade: wires lattice + beam + config into a single run API."""
from linac_gen.core.lattice import Lattice
from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig

class Simulation:
    """Top-level simulation controller.

    Usage:
        sim = Simulation(lattice, beam, space_charge=SpaceChargeConfig())
        sim.run()               # multi-particle tracking
        sim.run_envelope()      # envelope-only mode
        results = sim.get_results()
    """

    def __init__(self, lattice: Lattice, beam: Beam,
                 space_charge: SpaceChargeConfig = None,
                 snapshot_locations: list = None,
                 snapshot_every_n: int = None):
        self.lattice = lattice
        self.beam = beam
        self.sc_config = space_charge
        self.snapshot_locations = snapshot_locations
        self.snapshot_every_n = snapshot_every_n
        self._results = None

    def run(self):
        """Run multi-particle tracking. Implemented in Phase 4."""
        raise NotImplementedError("Multi-particle tracking: see Phase 4")

    def run_envelope(self):
        """Run envelope-only tracking. Implemented in Phase 10."""
        raise NotImplementedError("Envelope tracking: see Phase 10")

    def get_results(self):
        """Return DiagnosticRecorder from the last run."""
        if self._results is None:
            raise RuntimeError("No simulation has been run yet")
        return self._results
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/core/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/core/config.py linac_gen/core/simulation.py tests/core/test_config.py
git commit -m "feat: BeamConfig, SpaceChargeConfig dataclasses, Simulation facade"
```

---

## Phase 2: Drift + Quadrupole + Matrix Tracking + Basic Diagnostics

### Task 2.1: Drift element with 6x6 transfer matrix

**Files:**
- Create: `linac_gen/elements/drift.py`
- Create: `tests/elements/test_drift.py`

- [ ] **Step 1: Write the test**

```python
# tests/elements/test_drift.py
import numpy as np
import pytest
from linac_gen.elements.drift import Drift
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.constants import C_LIGHT, PI

def test_drift_transfer_matrix_shape():
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    assert M.shape == (6, 6)

def test_drift_identity_like():
    """A zero-length drift is identity."""
    d = Drift(name="D0", length=0.0, aperture=0.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    np.testing.assert_array_almost_equal(M, np.eye(6))

def test_drift_x_transport():
    """Particle with x'=1 mrad drifts L=100mm -> x += 0.1 mm."""
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    # M[0,1] should be L in mm (but x' is in mrad, so M[0,1] = L_mm * 1e-3 to convert)
    # Actually: x_new = x + x' * L where x in mm, x' in mrad -> need L in mm * 1e-3 = L in m? No.
    # Convention: x(mm), x'(mrad). x_new(mm) = x(mm) + x'(mrad) * L(mm) * 1e-3
    # So M[0,1] = L * 1e-3 = 0.1
    assert abs(M[0, 1] - 0.1) < 1e-10  # 100mm * 1e-3 = 0.1

def test_drift_longitudinal_coupling():
    """The (4,5) element couples energy deviation to phase slip."""
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    # M[4,5] = -2*pi*L / (beta^2 * gamma^3 * mass * wavelength) in deg/MeV
    # This should be nonzero and negative
    assert M[4, 5] < 0

def test_drift_slice():
    """Half-length drift matrix should give half the transport."""
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M_full = d.transfer_matrix(ref)
    M_half = d.transfer_matrix(ref, ds=50.0)
    assert abs(M_half[0, 1] - M_full[0, 1] / 2) < 1e-10
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/elements/test_drift.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Drift**

```python
# linac_gen/elements/drift.py
"""Drift space element."""
import numpy as np
from linac_gen.elements.base import TransferMapElement
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.constants import PI

class Drift(TransferMapElement):
    """Field-free drift space."""

    def __init__(self, name: str, length: float, aperture: float = 0.0, n_steps: int = 1):
        super().__init__(name=name, length=length, aperture=aperture, n_steps=n_steps)

    def transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        """6x6 transfer matrix for a drift of length ds (mm). If ds is None, uses self.length."""
        L = ds if ds is not None else self.length  # mm
        M = np.eye(6)
        # x(mm) += x'(mrad) * L(mm) * 1e-3; same for y
        L_m = L * 1e-3  # convert mm to m for mrad->mm coupling
        M[0, 1] = L_m   # x += x' * L (mm, mrad -> mm when L in m)
        M[2, 3] = L_m   # y += y' * L
        # Longitudinal: dphi(deg) += dW(MeV) * (-2*pi*L) / (beta^2 * gamma^3 * mass * wavelength)
        # converted to deg/MeV
        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass  # MeV/c^2
        wl = ref.wavelength      # mm
        # Phase slip: dphi = -360 * L / (beta^2 * gamma^3 * mass * wl) * dW
        # Factor: L and wl in mm cancel; 360 = 2*pi in degrees
        M[4, 5] = -360.0 * L / (beta**2 * gamma**3 * mass * wl)
        return M

    def track(self, beam: Beam, ds: float = None) -> None:
        """Apply drift to all alive particles and advance beam.ref."""
        L = ds if ds is not None else self.length
        # Advance reference particle
        beam.ref.s += L
        # Phase slip of reference (zero for reference particle by definition)
        # But ref.phi_s advances by the phase accumulated over drift:
        # dphi_ref = 360 * L / (beta * wavelength)
        beam.ref.phi_s += 360.0 * L / (beam.ref.beta * beam.ref.wavelength)
        # Apply transfer matrix to alive particles
        M = self.transfer_matrix(beam.ref, ds=L)
        alive = beam.alive_mask
        beam.particles[alive] = (M @ beam.particles[alive].T).T
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/elements/test_drift.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/elements/drift.py tests/elements/test_drift.py
git commit -m "feat: Drift element with 6x6 transfer matrix and tracking"
```

---

### Task 2.2: Quadrupole element with 6x6 transfer matrix

**Files:**
- Create: `linac_gen/elements/quadrupole.py`
- Create: `tests/elements/test_quadrupole.py`

- [ ] **Step 1: Write the test**

```python
# tests/elements/test_quadrupole.py
import numpy as np
import math
import pytest
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON

def test_quad_matrix_shape():
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    assert M.shape == (6, 6)

def test_quad_symplecticity():
    """Transfer matrix must be symplectic: M^T J M = J."""
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    # 2x2 symplectic check per plane: det(M_plane) = 1
    det_x = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    det_y = M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]
    assert abs(det_x - 1.0) < 1e-10
    assert abs(det_y - 1.0) < 1e-10

def test_quad_focusing_defocusing():
    """Positive gradient -> focusing in x, defocusing in y."""
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    # Focusing plane (x): M[0,0] = cos(kL) < 1
    assert M[0, 0] < 1.0
    # Defocusing plane (y): M[2,2] = cosh(kL) > 1
    assert M[2, 2] > 1.0

def test_quad_negative_gradient():
    """Negative gradient -> defocusing in x, focusing in y."""
    q = Quadrupole(name="Q1", length=100.0, gradient=-5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    assert M[0, 0] > 1.0  # defocusing in x
    assert M[2, 2] < 1.0  # focusing in y

def test_quad_thin_lens_limit():
    """Short quad should approximate thin lens: f = 1/(k^2*L)."""
    L_mm = 1.0  # very short
    G = 5.0
    q = Quadrupole(name="Q1", length=L_mm, gradient=G, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    # M[1,0] should be approximately -k^2 * L (in appropriate units)
    # k^2 = G / brho (1/m^2), L in m
    k2 = G / ref.brho  # 1/m^2
    L_m = L_mm * 1e-3
    # Thin lens: M[1,0] ~ -k^2 * L (units: 1/m -> mrad/mm needs factor 1e3 * 1e-3 = 1)
    expected_kick = -k2 * L_m  # this is in 1/m = mrad/mm
    assert abs(M[1, 0] - expected_kick) < abs(expected_kick) * 0.01  # 1% accuracy

def test_quad_slice():
    """Half-length quad should give different matrix."""
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M_full = q.transfer_matrix(ref)
    M_half = q.transfer_matrix(ref, ds=50.0)
    assert not np.allclose(M_full, M_half)
    # Half * Half should equal Full
    M_product = M_half @ M_half
    np.testing.assert_array_almost_equal(M_product, M_full, decimal=10)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/elements/test_quadrupole.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Quadrupole**

```python
# linac_gen/elements/quadrupole.py
"""Quadrupole magnet element."""
import numpy as np
import math
from linac_gen.elements.base import TransferMapElement
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam

class Quadrupole(TransferMapElement):
    """Magnetic quadrupole with hard-edge model.

    gradient > 0: horizontally focusing, vertically defocusing.
    gradient < 0: horizontally defocusing, vertically focusing.
    """

    def __init__(self, name: str, length: float, gradient: float,
                 aperture: float = 0.0, n_steps: int = 5):
        super().__init__(name=name, length=length, aperture=aperture, n_steps=n_steps)
        self.gradient = gradient  # T/m

    def transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        """6x6 transfer matrix. ds (mm) for sliced maps; None = full element."""
        L_mm = ds if ds is not None else self.length
        L_m = L_mm * 1e-3  # convert to meters for computation
        M = np.eye(6)

        if L_m == 0.0:
            return M

        # k^2 = |G| / Brho (1/m^2)
        k2 = abs(self.gradient) / ref.brho
        k = math.sqrt(k2)
        kL = k * L_m

        if self.gradient > 0:
            # x: focusing (cos/sin), y: defocusing (cosh/sinh)
            cos_kL = math.cos(kL)
            sin_kL = math.sin(kL)
            cosh_kL = math.cosh(kL)
            sinh_kL = math.sinh(kL)
            # Focusing plane (x): units -> M[0,1] in m (so x(mm) += x'(mrad) * M01)
            M[0, 0] = cos_kL
            M[0, 1] = sin_kL / k          # m (mrad->mm coupling)
            M[1, 0] = -k * sin_kL         # 1/m (mm->mrad coupling)
            M[1, 1] = cos_kL
            # Defocusing plane (y)
            M[2, 2] = cosh_kL
            M[2, 3] = sinh_kL / k
            M[3, 2] = k * sinh_kL
            M[3, 3] = cosh_kL
        else:
            # x: defocusing, y: focusing
            cosh_kL = math.cosh(kL)
            sinh_kL = math.sinh(kL)
            cos_kL = math.cos(kL)
            sin_kL = math.sin(kL)
            M[0, 0] = cosh_kL
            M[0, 1] = sinh_kL / k
            M[1, 0] = k * sinh_kL
            M[1, 1] = cosh_kL
            M[2, 2] = cos_kL
            M[2, 3] = sin_kL / k
            M[3, 2] = -k * sin_kL
            M[3, 3] = cos_kL

        # Longitudinal: same as drift (no direct longitudinal effect from pure quad)
        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass
        wl = ref.wavelength
        M[4, 5] = -360.0 * L_mm / (beta**2 * gamma**3 * mass * wl)

        return M

    def track(self, beam: Beam, ds: float = None) -> None:
        """Apply quadrupole to all alive particles and advance beam.ref."""
        L = ds if ds is not None else self.length
        beam.ref.s += L
        beam.ref.phi_s += 360.0 * L / (beam.ref.beta * beam.ref.wavelength)
        M = self.transfer_matrix(beam.ref, ds=L)
        alive = beam.alive_mask
        beam.particles[alive] = (M @ beam.particles[alive].T).T
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/elements/test_quadrupole.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/elements/quadrupole.py tests/elements/test_quadrupole.py
git commit -m "feat: Quadrupole element with hard-edge 6x6 transfer matrix"
```

---

### Task 2.3: Matrix tracking and basic diagnostics

**Files:**
- Create: `linac_gen/tracking/__init__.py`
- Create: `linac_gen/tracking/matrix_tracking.py`
- Create: `linac_gen/diagnostics/__init__.py`
- Create: `linac_gen/diagnostics/moments.py`
- Create: `tests/tracking/__init__.py`
- Create: `tests/tracking/test_matrix_tracking.py`

- [ ] **Step 1: Write the test**

```python
# tests/tracking/test_matrix_tracking.py
import numpy as np
import pytest
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, compute_twiss

def test_single_drift_matrix():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = compute_transfer_matrix(lat, ref)
    assert M.shape == (6, 6)
    assert abs(M[0, 1] - 0.1) < 1e-10  # 100mm drift

def test_fodo_matrix():
    """FODO cell: QF - D - QD - D should produce net focusing."""
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0))
    lat.add(Drift("D2", 200.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = compute_transfer_matrix(lat, ref)
    # Stability: |Tr(M_x)| < 2
    trace_x = M[0, 0] + M[1, 1]
    assert abs(trace_x) < 2.0, f"Unstable FODO: trace_x = {trace_x}"

def test_twiss_from_periodic():
    """Compute matched Twiss parameters for a FODO cell."""
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0))
    lat.add(Drift("D2", 200.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = compute_transfer_matrix(lat, ref)
    twiss_x = compute_twiss(M, plane="x")
    assert twiss_x["beta"] > 0
    assert 0 < twiss_x["mu"] < 180  # phase advance in degrees
    # beta * gamma - alpha^2 = 1
    bg_check = twiss_x["beta"] * twiss_x["gamma_t"] - twiss_x["alpha"]**2
    assert abs(bg_check - 1.0) < 1e-10
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/tracking/test_matrix_tracking.py -v`
Expected: FAIL

- [ ] **Step 3: Implement matrix tracking and Twiss computation**

```python
# linac_gen/tracking/matrix_tracking.py
"""Linear transfer matrix tracking and Twiss parameter computation."""
import numpy as np
import math
from linac_gen.core.lattice import Lattice
from linac_gen.core.reference import ReferenceParticle

from linac_gen.elements.base import (
    TransferMapElement, ThinKickElement, PassiveElement, FieldMapElement,
)

def get_element_matrix(element, ref_copy: ReferenceParticle) -> np.ndarray:
    """Get the linearized 6x6 matrix for any element type.

    Dispatches by interface:
    - TransferMapElement: calls transfer_matrix(ref)
    - ThinKickElement: calls kick_matrix(ref)
    - PassiveElement: returns identity (no dynamics)
    - FieldMapElement: calls fitted_matrix(ref) (linearized from RK4 tracking)

    Raises TypeError for unrecognized element types.
    """
    if isinstance(element, TransferMapElement):
        return element.transfer_matrix(ref_copy)
    elif isinstance(element, ThinKickElement):
        return element.kick_matrix(ref_copy)
    elif isinstance(element, FieldMapElement):
        return element.fitted_matrix(ref_copy)
    elif isinstance(element, PassiveElement):
        return np.eye(6)
    else:
        raise TypeError(
            f"Cannot compute transfer matrix for {type(element).__name__} '{element.name}': "
            f"element must be TransferMapElement, ThinKickElement, FieldMapElement, or PassiveElement"
        )

def compute_transfer_matrix(lattice: Lattice, ref: ReferenceParticle) -> np.ndarray:
    """Compute the cumulative 6x6 transfer matrix of the full lattice.

    Uses get_element_matrix() to dispatch correctly for all element types
    (TransferMapElement, ThinKickElement, PassiveElement, FieldMapElement).
    """
    ref_copy = ref.copy()
    M = np.eye(6)
    for element in lattice.elements:
        M_elem = get_element_matrix(element, ref_copy)
        M = M_elem @ M
        # Advance reference through element.
        # FieldMapElement: advance_ref() handles s, phi_s, AND w_kin (via RK4).
        # ThinKickElement: generic s/phi_s advance (zero, since length=0),
        #   then advance_ref() updates w_kin/phi_s for RF gaps (no-op for steerers).
        # TransferMapElement/PassiveElement: generic s/phi_s advance only.
        if isinstance(element, FieldMapElement):
            element.advance_ref(ref_copy)  # handles s, phi_s, w_kin — skip generic advance
        else:
            ref_copy.s += element.length
            if element.length > 0:
                ref_copy.phi_s += 360.0 * element.length / (ref_copy.beta * ref_copy.wavelength)
            if isinstance(element, ThinKickElement):
                element.advance_ref(ref_copy)  # RF gap: updates w_kin, phi_s
    return M

def compute_twiss(M: np.ndarray, plane: str = "x") -> dict:
    """Compute matched Twiss parameters from a one-period transfer matrix.

    Args:
        M: 6x6 transfer matrix for one period.
        plane: "x" (uses rows/cols 0,1) or "y" (uses rows/cols 2,3).

    Returns:
        dict with keys: alpha, beta, gamma_t (Twiss gamma, not relativistic), mu (deg).

    Raises:
        ValueError: if the matrix is unstable (|trace| >= 2).
    """
    if plane == "x":
        i, j = 0, 1
    elif plane == "y":
        i, j = 2, 3
    else:
        raise ValueError(f"plane must be 'x' or 'y', got '{plane}'")

    m11 = M[i, i]
    m12 = M[i, j]
    m21 = M[j, i]
    m22 = M[j, j]

    cos_mu = 0.5 * (m11 + m22)
    if abs(cos_mu) >= 1.0:
        raise ValueError(f"Unstable: cos(mu) = {cos_mu}")

    mu = math.acos(cos_mu)  # radians
    sin_mu = math.sin(mu)

    # Sign of sin_mu: if m12 > 0, sin_mu > 0 (standard convention)
    if m12 < 0:
        mu = 2 * math.pi - mu
        sin_mu = math.sin(mu)

    beta = m12 / sin_mu
    alpha = (m11 - m22) / (2.0 * sin_mu)
    gamma_t = (1.0 + alpha**2) / beta

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma_t": gamma_t,
        "mu": math.degrees(mu),
    }
```

```python
# linac_gen/diagnostics/__init__.py
"""Beam diagnostics: moments, Twiss, emittance, losses."""
```

```python
# linac_gen/diagnostics/moments.py
"""Statistical moment computation from particle arrays."""
import numpy as np

def compute_moments(particles: np.ndarray) -> dict:
    """Compute first and second moments from (N,6) particle array.

    Returns dict with keys: mean (6,), sigma (6x6), sigma_x, sigma_y, sigma_phi, sigma_w.
    """
    mean = np.mean(particles, axis=0)
    centered = particles - mean
    sigma = (centered.T @ centered) / len(particles)

    return {
        "mean": mean,
        "sigma_matrix": sigma,
        "sigma_x": np.sqrt(sigma[0, 0]),
        "sigma_xp": np.sqrt(sigma[1, 1]),
        "sigma_y": np.sqrt(sigma[2, 2]),
        "sigma_yp": np.sqrt(sigma[3, 3]),
        "sigma_phi": np.sqrt(sigma[4, 4]),
        "sigma_w": np.sqrt(sigma[5, 5]),
    }

def compute_emittance(particles: np.ndarray, plane: str = "x") -> float:
    """Compute geometric RMS emittance for a given plane.

    Args:
        particles: (N, 6) array.
        plane: "x" (cols 0,1), "y" (cols 2,3), "z" (cols 4,5).

    Returns:
        RMS emittance in mm.mrad (transverse) or deg.MeV (longitudinal).
    """
    col_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    i, j = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    up = particles[:, j] - np.mean(particles[:, j])
    uu = np.mean(u * u)
    upup = np.mean(up * up)
    uup = np.mean(u * up)
    emit = np.sqrt(abs(uu * upup - uup * uup))
    return float(emit)

def compute_twiss_from_particles(particles: np.ndarray, plane: str = "x") -> dict:
    """Compute Twiss parameters from particle distribution."""
    col_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    i, j = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    up = particles[:, j] - np.mean(particles[:, j])
    emit = compute_emittance(particles, plane)
    if emit < 1e-30:
        return {"alpha": 0.0, "beta": 0.0, "gamma_t": 0.0, "emittance": 0.0}
    beta = np.mean(u * u) / emit
    alpha = -np.mean(u * up) / emit
    gamma_t = np.mean(up * up) / emit
    return {"alpha": float(alpha), "beta": float(beta), "gamma_t": float(gamma_t), "emittance": float(emit)}

def compute_halo(particles: np.ndarray, plane: str = "x") -> float:
    """Compute halo parameter: H = <u^4>/<u^2>^2 - 1."""
    col_map = {"x": 0, "y": 2, "z": 4}
    i = col_map[plane]
    u = particles[:, i] - np.mean(particles[:, i])
    u2 = np.mean(u**2)
    u4 = np.mean(u**4)
    if u2 < 1e-30:
        return 0.0
    return float(u4 / u2**2 - 1.0)
```

`linac_gen/tracking/__init__.py` and `tests/tracking/__init__.py`: empty files.

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/tracking/test_matrix_tracking.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add linac_gen/tracking/ linac_gen/diagnostics/ tests/tracking/ tests/elements/
git commit -m "feat: matrix tracking, Twiss computation, beam moment diagnostics"
```

---

## Phases 3-16: Remaining Implementation

Due to the massive scope, the remaining phases follow the same TDD pattern. Each phase is summarized with its tasks, files, and key implementation details. The full step-by-step breakdown (tests, implementation, verification, commit) follows the identical format as Phases 1-2.

---

## Phase 3: Distributions

### Task 3.1: Gaussian distribution generator

**Files:**
- Create: `linac_gen/distributions/gaussian.py`
- Create: `tests/distributions/__init__.py`
- Create: `tests/distributions/test_gaussian.py`

Generate 6D Gaussian distribution using Cholesky decomposition of the sigma matrix derived from Twiss parameters and emittances. Truncate at `cutoff` sigma. Verify: RMS emittance matches input within 5%, Twiss parameters match within 10%, no particles beyond cutoff.

### Task 3.2: Waterbag distribution generator

**Files:**
- Create: `linac_gen/distributions/waterbag.py`
- Create: `tests/distributions/test_waterbag.py`

Uniform sampling inside a 6D hyperellipsoid. Algorithm: generate 6 standard normals, normalize to unit 6-sphere surface, scale by `r^(1/6)` where r ~ U(0,1), then apply Twiss scaling. Verify: halo parameter H near 1/3, emittance matches input.

### Task 3.3: KV distribution generator

**Files:**
- Create: `linac_gen/distributions/kv.py`
- Create: `tests/distributions/test_kv.py`

Uniform on 4D transverse hyperellipsoid surface. Generate 4 normals, normalize to unit 4-sphere surface, scale by Twiss. Longitudinal added separately. Verify: halo parameter H near 0, uniform projected density.

### Task 3.4: Parabolic and Uniform distributions

**Files:**
- Create: `linac_gen/distributions/parabolic.py`
- Create: `linac_gen/distributions/uniform.py`
- Create: `tests/distributions/test_parabolic.py`
- Create: `tests/distributions/test_uniform.py`

Parabolic: density proportional to `(1 - r²/R²)` inside ellipsoid. Uniform: flat density inside ellipsoid (6D). Both use rejection sampling. Verify emittance and halo parameter.

### Task 3.5: From-file distribution loader

**Files:**
- Create: `linac_gen/distributions/from_file.py`
- Create: `tests/distributions/test_from_file.py`

Reads ASCII 6-column file with header (absolute coordinates), subtracts reference to produce deviations. Handles missing header with warning.

### Task 3.6: BeamConfig and beam factory

**Files:**
- Create: `linac_gen/distributions/factory.py`
- Create: `tests/distributions/test_factory.py`

`create_beam(config: BeamConfig) -> Beam`: dispatches to the correct distribution generator based on `config.distribution`. Converts normalized emittance to geometric using `ref.bg`. Produces (N,6) deviation array centered at zero.

---

## Phase 4: Multi-Particle Tracker

### Task 4.1: Main tracker with element dispatch

**Files:**
- Create: `linac_gen/tracking/tracker.py`
- Create: `linac_gen/diagnostics/recorder.py`
- Create: `tests/tracking/test_tracker.py`

The `Tracker` class iterates over lattice elements, dispatches to element-specific tracking strategies (TransferMapElement split-operator, ThinKick, FieldMap, Passive), records diagnostics at each element exit, and handles aperture checking. Stores moments + reference history in a `DiagnosticRecorder`.

### Task 4.2: Aperture element

**Files:**
- Create: `linac_gen/elements/aperture.py`
- Create: `tests/elements/test_aperture.py`

PassiveElement that checks particles against circular (`r > R`), rectangular (`|x|>a or |y|>b`), or elliptical (`(x/a)^2+(y/b)^2>1`) limits. Calls `beam.record_loss()` for each lost particle.

### Task 4.3: Marker/Diag element

**Files:**
- Create: `linac_gen/elements/marker.py`
- Create: `tests/elements/test_marker.py`

PassiveElement with zero length, no dynamics. Triggers diagnostic snapshot if configured.

### Task 4.4: Steerer element

**Files:**
- Create: `linac_gen/elements/steerer.py`
- Create: `tests/elements/test_steerer.py`

ThinKickElement: applies `dx' = q*By*L/p` and `dy' = q*Bx*L/p` using `beam.ref.brho`. Does not update `beam.ref`.

---

## Phase 5: PIC Space Charge (Python)

### Task 5.1: Phase-space coordinate conversion (dphi,dW) ↔ (z,pz)

**Files:**
- Create: `linac_gen/pic/coordinates.py`
- Create: `tests/pic/__init__.py`
- Create: `tests/pic/test_coordinates.py`

The PIC solver operates on spatial coordinates (x, y, z in mm), but the beam stores longitudinal deviations as (dphi in deg, dW in MeV). This task implements the bidirectional conversion:

**dphi → z_lab (mm):**
```
z_lab_i = -dphi_i * (beta_ref * wavelength) / 360.0
```
where wavelength = c/f in mm. The sign convention: negative dphi (particle ahead in phase) = positive z (ahead in space).

**dW → delta_beta (for velocity-dependent z-offset):**
Not needed for the Lorentz boost — the PIC solver uses z_lab directly. The energy deviation affects the SC kick denominator but is neglected per the spec (paraxial beam approximation).

**After PIC kick, convert z-kick back to dphi-kick:**
The PIC solver produces `dW_kick` (energy kick in MeV) directly — no phase kick needed since phase changes come from the energy-dependent drift in the split-operator half-step.

Verify: round-trip dphi → z → dphi preserves values within 1e-12.

### Task 5.2: Lorentz boost

**Files:**
- Create: `linac_gen/pic/lorentz_boost.py`
- Create: `tests/pic/test_lorentz_boost.py`

Boost spatial coordinates (x, y, z_lab) to beam rest frame: `z_rest = gamma_ref * z_lab`. Transverse coordinates unchanged. Back-boost after field solve. Operates on a temporary (N,3) spatial array extracted via Task 5.1's converter, not on the (N,6) deviation array directly.

### Task 5.3: Charge deposition (CIC, Python)

**Files:**
- Create: `linac_gen/pic/charge_deposition.py`
- Create: `tests/pic/test_charge_deposition.py`

CIC (Cloud-In-Cell) weighting: each particle deposits charge to 8 neighboring nodes with trilinear weights. Verify: total deposited charge equals sum of particle charges.

### Task 5.4: Poisson solver (FFT, Python)

**Files:**
- Create: `linac_gen/pic/poisson_solver.py`
- Create: `tests/pic/test_poisson_solver.py`

Hockney's method: zero-pad to 2N, FFT, multiply by precomputed integrated Green's function, inverse FFT. Verify against analytical potential for a uniform sphere.

### Task 5.5: Field interpolation (CIC, Python)

**Files:**
- Create: `linac_gen/pic/field_interpolation.py`
- Create: `tests/pic/test_field_interpolation.py`

Same CIC weights as deposition to interpolate E-field from grid to particles.

### Task 5.6: PIC solver integration

**Files:**
- Create: `linac_gen/pic/pic_solver.py`
- Create: `tests/pic/test_pic_solver.py`

`PicSolver.kick(beam, ds)`: orchestrates the full PIC cycle (boost, deposit, solve, interpolate, boost back, apply kicks). Implements the field-to-slope conversion formulas from the spec. Uses `SpaceChargeConfig` from `linac_gen.core.config` (defined in Phase 1 Task 1.8 — no separate `pic/config.py`).

---

## Phase 6: Remaining Elements

### Task 6.1: RF Gap (ThinKickElement)

**Files:**
- Create: `linac_gen/elements/rf_gap.py`
- Create: `tests/elements/test_rf_gap.py`

Energy kick: `dW_i = q*V0*T*cos(phi_s + dphi_i) - q*V0*T*cos(phi_s)`. Updates `beam.ref.w_kin` with synchronous gain. Transverse RF defocusing. Adiabatic damping of x', y'. Linearized kick_matrix for envelope mode.

### Task 6.2: Solenoid (TransferMapElement)

**Files:**
- Create: `linac_gen/elements/solenoid.py`
- Create: `tests/elements/test_solenoid.py`

4x4 coupled x-y matrix with Larmor rotation. `k_s = qB/(2mc*beta*gamma)`. Thin fringe kicks at entry/exit. Verify: round beam stays round, phase advance matches analytical.

### Task 6.3: Dipole (TransferMapElement)

**Files:**
- Create: `linac_gen/elements/dipole.py`
- Create: `tests/elements/test_dipole.py`

Sector bend 6x6 with dispersion terms. Edge focusing via thin matrices at entrance/exit with `1/f = tan(e)/rho`. Combined-function via optional `k1` parameter.

### Task 6.4: Multipole (ThinKickElement)

**Files:**
- Create: `linac_gen/elements/multipole.py`
- Create: `tests/elements/test_multipole.py`

General 2n-pole thin kick. `dx' + i*dy' = -L/(Brho) * sum (b_n + i*a_n)/(n-1)! * (x + iy)^(n-1)`.

### Task 6.5: SpaceChargeComp and ThinLens

**Files:**
- Create: `linac_gen/elements/space_charge_comp.py`
- Create: `linac_gen/elements/thin_lens.py`
- Create: `tests/elements/test_misc_elements.py`

SpaceChargeComp: PassiveElement that sets a SC reduction factor. ThinLens: ThinKickElement with focal length `f`, applies `dx' = -x/f`.

---

## Phase 7: Field Map Reader + TraceWin I/O

Field map reader comes FIRST because the TraceWin parser needs to resolve FIELD_MAP cards into FieldMapElement instances, which requires loaded field data.

### Task 7.1: Field map reader (1D and 2D .edz + CSV)

**Files:**
- Create: `linac_gen/io/__init__.py`
- Create: `linac_gen/io/field_map_reader.py`
- Create: `tests/io/__init__.py`
- Create: `tests/io/test_field_map_reader.py`
- Create: `tests/io/fixtures/test_cavity_1d.edz`
- Create: `tests/io/fixtures/test_cavity_2d.edz`
- Create: `tests/io/fixtures/test_fields.csv`

Read TraceWin .edz format: 1D on-axis (Ez(z)), 2D electric (Ez(r,z), Er(r,z)), 2D E+B type 7 (Ez, Er, Bz, Br). Also read generic CSV with header columns. Return `FieldMapData` dataclass. For 1D: implement Bessel off-axis expansion for Er from Ez. Auto-detect format from header.

### Task 7.2: FieldMapElement with RK4 tracking

**Files:**
- Create: `linac_gen/elements/field_map.py`
- Create: `linac_gen/tracking/rk4.py`
- Create: `tests/elements/test_field_map.py`

FieldMapElement (inherits FieldMapElement base) loads field data, implements `track_rk4(beam, ds)` using 4th-order Runge-Kutta through interpolated fields. Reference particle tracked first, then all particles. Tricubic interpolation on grid.

Also implements `fitted_matrix(ref)` for envelope/matching mode: tracks a small test beam (12 particles offset by +/- epsilon in each of 6 coordinates) through the full field map via RK4, computes the 6x6 Jacobian from input/output differences. Also implements `advance_ref(ref)`: runs the reference particle through the full field map via RK4, updating `ref.w_kin`, `ref.phi_s`, and `ref.s`. This is called by `compute_transfer_matrix()` (via the same `advance_ref` path as ThinKickElement) so that downstream elements see correct post-field-map energy/phase.

Override the placeholder `advance_ref` and `fitted_matrix` from the FieldMapElement base class (defined in Phase 1 Task 1.7) with real implementations in `linac_gen/elements/field_map.py`.

### Task 7.3: TraceWin .dat parser

**Files:**
- Create: `linac_gen/io/tracewin_parser.py`
- Create: `tests/io/test_tracewin_parser.py`
- Create: `tests/io/fixtures/simple_fodo.dat`
- Create: `tests/io/fixtures/lattice_with_fieldmap.dat`

Line-by-line parser for v1 supported cards. FREQ is stateful. FIELD_MAP cards use the field map reader (Task 7.1) to load field data and create FieldMapElement instances. Returns `(Lattice, metadata_dict)`. Unsupported cards emit warnings. Test with both a FODO lattice and a lattice containing a FIELD_MAP reference.

### Task 7.4: TraceWin .dat writer

**Files:**
- Create: `linac_gen/io/tracewin_writer.py`
- Create: `tests/io/test_tracewin_writer.py`

Serialize lattice back to `.dat` format. Round-trip test: parse -> write -> parse -> compare lattice (for non-field-map elements; field-map elements preserve the filename reference).

---

## Phase 8: C++ PIC Kernels

### Task 8.1: pybind11 build setup

**Files:**
- Create: `linac_gen/csrc/CMakeLists.txt`
- Modify: `pyproject.toml` (add cmake build)

Set up pybind11 + FFTW3 build. Verify: `import linac_gen._pic_kernels` works.

### Task 8.2: C++ charge deposition and field interpolation

**Files:**
- Create: `linac_gen/csrc/pic_kernels.cpp`
- Create: `tests/pic/test_pic_kernels_cpp.py`

CIC deposit and interpolation in C++. Test: results match Python implementation within 1e-10.

### Task 8.3: C++ FFT Poisson solver

**Files:**
- Create: `linac_gen/csrc/poisson_fft.cpp`
- Create: `tests/pic/test_poisson_fft_cpp.py`

3D FFT Poisson with FFTW. Test: results match Python implementation.

### Task 8.4: Switch PIC solver to C++ backend

**Files:**
- Modify: `linac_gen/pic/pic_solver.py`

Auto-detect C++ extension. Use if available, fall back to Python. Verify: existing PIC tests still pass with both backends.

---

## Phase 9: Envelope Solver

### Task 9.1: Sacherer envelope equations

**Files:**
- Create: `linac_gen/tracking/envelope.py`
- Create: `tests/tracking/test_envelope.py`

Solve RMS envelope equations including analytical SC term. Track sigma_x, sigma_y, sigma_z through the lattice. Uses transfer matrices for external focusing, uniform-ellipsoid for SC.

---

## Phase 10: Matching

### Task 10.1: Matcher with envelope and multi-particle modes

**Files:**
- Create: `linac_gen/matching/__init__.py`
- Create: `linac_gen/matching/matcher.py`
- Create: `linac_gen/matching/objectives.py`
- Create: `linac_gen/matching/periodic.py`
- Create: `tests/matching/__init__.py`
- Create: `tests/matching/test_matcher.py`

`Matcher.add_variable()`, `add_objective()`, `solve()`. Uses SciPy `least_squares` for envelope matching (requires Phase 9 envelope solver), `differential_evolution` for multi-particle matching (requires Phase 4 tracker + Phase 5 PIC). Periodic matching via eigenvalue extraction from one-period matrix.

---

## Phase 11: Error Studies

### Task 11.1: Error model and Monte Carlo engine

**Files:**
- Create: `linac_gen/errors/__init__.py`
- Create: `linac_gen/errors/error_model.py`
- Create: `linac_gen/errors/monte_carlo.py`
- Create: `tests/errors/__init__.py`
- Create: `tests/errors/test_error_study.py`

`ErrorStudy.add_error()`, `.run(n_workers)`. Applies random errors to lattice copies per seed. Runs tracking per seed using `multiprocessing.Pool`. Aggregates results: mean, std, percentiles.

### Task 11.2: Orbit correction

**Files:**
- Create: `linac_gen/errors/correction.py`
- Create: `tests/errors/test_correction.py`

One-to-one and SVD correction. Build response matrix (steerer->BPM), invert, compute corrector settings.

---

## Phase 12: Distribution I/O and HDF5

### Task 12.1: Distribution import/export

**Files:**
- Create: `linac_gen/io/distribution_io.py`
- Create: `tests/io/test_distribution_io.py`

ASCII 6-column with header (absolute coordinates). Import: subtract header ref to get deviations. Export: add beam.ref to deviations. Round-trip test.

### Task 12.2: HDF5 results output

**Files:**
- Create: `linac_gen/io/hdf5_output.py`
- Create: `tests/io/test_hdf5_output.py`

Save DiagnosticRecorder to HDF5 with groups: lattice/, beam_config/, reference/, envelope/, particles/ (with ref per snapshot), losses/.

---

## Phase 13: GUI — Main Window

### Task 13.1: Application entry point and main window skeleton

**Files:**
- Modify: `gui/linac_gen_gui/__init__.py` (already created in Phase 1 Task 1.1)
- Create: `gui/linac_gen_gui/app.py`
- Create: `gui/linac_gen_gui/main_window.py`

PyQt6 QMainWindow with menu bar (File/Edit/Beam/Simulation/Analysis/Help), toolbar, and docked panel layout. Three-column: lattice editor (left), plot area (center), beam config (right).

### Task 13.2: Lattice editor widget

**Files:**
- Create: `gui/linac_gen_gui/widgets/lattice_editor.py`

QTreeWidget showing element list. Color-coded by type. Right-click context menu for insert/delete. Double-click to select -> shows properties. Drag to reorder.

### Task 13.3: Beam config panel

**Files:**
- Create: `gui/linac_gen_gui/widgets/beam_config.py`

QDockWidget with form fields for all BeamConfig parameters. Auto-computes beta/gamma/beam sizes on energy change. Distribution type dropdown. Preview button.

---

## Phase 14: GUI — Plot Widgets

### Task 14.1: Envelope plot

**Files:**
- Create: `gui/linac_gen_gui/widgets/envelope_plot.py`

pyqtgraph PlotWidget showing sigma_x(s), sigma_y(s) with aperture overlay. Clickable cursor synchronized with other plots.

### Task 14.2: Phase space plot

**Files:**
- Create: `gui/linac_gen_gui/widgets/phase_space_plot.py`

2D scatter/density plot for (x,x'), (y,y'), (phi,W) at selected s-position. Density colormap for >10k particles.

### Task 14.3: Loss map and emittance plots

**Files:**
- Create: `gui/linac_gen_gui/widgets/loss_map_plot.py`
- Create: `gui/linac_gen_gui/widgets/emittance_plot.py`

Loss map: histogram of lost particles vs s. Emittance: emit_x(s), emit_y(s), emit_z(s) curves.

### Task 14.4: Lattice layout strip

**Files:**
- Create: `gui/linac_gen_gui/widgets/lattice_layout.py`

Thin horizontal widget showing colored rectangles for each element. Synchronized cursor line.

---

## Phase 15: GUI — Simulation Runner and Dialogs

### Task 15.1: Simulation worker thread

**Files:**
- Create: `gui/linac_gen_gui/workers.py`

QThread subclass that runs `Simulation.run()` (the facade from Phase 1 Task 1.8, fully wired by Phase 4). Emits signals: progress(int), step_complete(dict), finished(DiagnosticRecorder). Stop flag for cancellation. Uses SpaceChargeConfig from beam config panel.

### Task 15.2: Simulation settings dialog

**Files:**
- Create: `gui/linac_gen_gui/dialogs/simulation_settings.py`

QDialog for SpaceChargeConfig, tracking mode (envelope/multi-particle), snapshot settings.

### Task 15.3: Matching dialog

**Files:**
- Create: `gui/linac_gen_gui/dialogs/matching_dialog.py`

QDialog for variable/objective selection, algorithm choice, run button. Displays results.

### Task 15.4: Error study dialog and results viewer

**Files:**
- Create: `gui/linac_gen_gui/dialogs/error_study_dialog.py`
- Create: `gui/linac_gen_gui/widgets/error_study_view.py`

Error definition UI, seed count, run button. Results viewer with percentile envelopes and statistical summaries.
