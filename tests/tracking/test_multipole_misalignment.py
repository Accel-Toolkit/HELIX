"""Regression: MP tracking must apply Multipole misalignment ONCE.

The tracker wraps non-passive elements in the dx/dy/tilt transform, but
Multipole ALSO self-handles its offset and tilt inside apply_kick (the
manual's contract: kick evaluated at (x−dx, y−dy) in the tilted frame).
Both running double-applied everything: a 1 mm offset kicked like 2 mm
(feed-down errors 2×/4×) and a 45° skew tracked as 90°.  The envelope
solver always excluded Multipole from the wrap — MP now matches it,
the element's own semantics, and the manual.
"""
from __future__ import annotations

import numpy as np

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.tracker import Tracker


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _one_particle_beam(x_mm=3.0):
    beam = Beam(ref=_ref(), n_particles=1, current=0.0)
    beam.particles[:] = 0.0
    beam.particles[0, 0] = x_mm
    return beam


def _track(element, x_mm=3.0):
    lat = Lattice()
    lat.add(element)
    beam = _one_particle_beam(x_mm)
    Tracker(lat, beam).run()
    return beam.particles[0].copy()


def test_offset_applied_once_matches_element_semantics():
    """dx=1 mm on knl=[0,1] at x=3 mm: kick = −k1L·(x−dx) = −2 mrad.
    The doubled (pre-fix) value was −1 mrad."""
    out = _track(Multipole("M", knl=[0.0, 1.0], dx=1.0))
    assert abs(out[1] - (-2.0)) < 1e-12
    assert abs(out[3]) < 1e-12


def test_tracker_equals_bare_apply_kick_with_misalignment():
    """THE doubling seam: through the Tracker must equal the element's
    own apply_kick — the tracker adds NO extra transform for Multipole."""
    m = Multipole("M", knl=[0.0, 1.0], dx=0.7, dy=-0.4, tilt_deg=22.5)
    via_tracker = _track(m)
    beam = _one_particle_beam()
    m2 = Multipole("M", knl=[0.0, 1.0], dx=0.7, dy=-0.4, tilt_deg=22.5)
    m2.apply_kick(beam)
    direct = beam.particles[0]
    assert np.allclose(via_tracker[[1, 3]], direct[[1, 3]], atol=1e-14)


def test_tilt_not_doubled():
    """tilt=22.5° quadrupole harmonic: lab kick rotates by 2θ=45° →
    (−3cos45°, −3sin45°).  The pre-fix double gave the 90° pattern
    (0, −3)."""
    out = _track(Multipole("M", knl=[0.0, 1.0], tilt_deg=22.5))
    expect = 3.0 / np.sqrt(2.0)
    assert abs(out[1] + expect) < 1e-9
    assert abs(out[3] + expect) < 1e-9
    assert abs(out[3] + 3.0) > 0.5          # NOT the doubled pattern


def test_quadrupole_wrap_unchanged_control():
    """Quadrupole (mixin, tracker-applied misalignment) must still be
    wrapped by the tracker: an offset quad at x equals a centered quad
    seen at (x − dx) — angles identical, position restored by +dx."""
    off = _track(Quadrupole("Q", length=100.0, gradient=8.0, dx=1.0),
                 x_mm=3.0)
    cen = _track(Quadrupole("Q", length=100.0, gradient=8.0), x_mm=2.0)
    assert abs(off[1] - cen[1]) < 1e-12          # same kick physics
    assert abs(off[0] - (cen[0] + 1.0)) < 1e-12  # offset restored


def test_backtrack_closure_through_misaligned_multipole():
    """Forward → exact backward must close bit-tight with the exclusion
    applied SYMMETRICALLY on both sides."""
    from linac_gen.tracking.backtrack import backtrack_distribution
    lat = Lattice()
    lat.add(Multipole("M", knl=[0.0, 1.0], dx=0.5, tilt_deg=10.0))
    beam = _one_particle_beam()
    x0 = beam.particles.copy()
    entrance_ref = _ref()
    Tracker(lat, beam).run()
    backtrack_distribution(lat, beam, entrance_ref,
                           start=0, end=len(lat.elements) - 1)
    assert np.allclose(beam.particles, x0, atol=1e-10)
