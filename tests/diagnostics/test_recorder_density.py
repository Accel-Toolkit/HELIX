"""Tests for the 2-D density-vs-s recording path on :class:`DiagnosticRecorder`.

Density recording is opt-in: by default :attr:`density_axes` is empty and
:meth:`record_density` is a no-op.  These tests cover:

* opt-in default off
* :meth:`configure_density` sets axes / extent / n_bins and resets state
* per-call shapes — one column per recorded step, ``n_bins`` rows
* edge auto-fit on first record when no explicit extent is given
* explicit extent honoured exactly
* empty beam appends a zero column so the array stays rectangular vs ``s``
* unknown axis labels are skipped silently
* :meth:`record` invokes density alongside scalar diagnostics so the two
  arrays remain aligned
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.diagnostics.recorder import DiagnosticRecorder


# --------------------------------------------------------------------- #
def _seeded_beam(n: int = 200, seed: int = 7,
                 sigma_x: float = 2.0,
                 sigma_y: float = 2.0) -> Beam:
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=10.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, sigma_x, n)
    beam.particles[:, 1] = rng.normal(0.0, 0.5, n)
    beam.particles[:, 2] = rng.normal(0.0, sigma_y, n)
    beam.particles[:, 3] = rng.normal(0.0, 0.5, n)
    beam.particles[:, 4] = rng.normal(0.0, 5.0, n)
    beam.particles[:, 5] = rng.normal(0.0, 0.01, n)
    return beam


# --------------------------------------------------------------------- #
class TestOptIn:
    def test_default_axes_empty(self):
        rec = DiagnosticRecorder()
        assert rec.density_axes == ()

    def test_record_density_noop_when_unconfigured(self):
        rec = DiagnosticRecorder()
        beam = _seeded_beam()
        rec.record_density(beam)
        assert rec.density == {}
        assert rec.density_edges == {}

    def test_record_via_record_method_is_noop_when_unconfigured(self):
        # Calling record() should not implicitly turn density on.
        rec = DiagnosticRecorder()
        beam = _seeded_beam()
        rec.record(beam, s_position=0.0)
        assert rec.density == {}


class TestConfigure:
    def test_configure_sets_axes(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x", "y"), n_bins=128)
        assert rec.density_axes == ("x", "y")
        assert rec.density_n_bins == 128
        # Configure resets the per-axis storage to empty lists.
        assert rec.density == {"x": [], "y": []}

    def test_configure_with_explicit_extent(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), extent={"x": (-3.0, 3.0)},
                              n_bins=50)
        beam = _seeded_beam(sigma_x=1.0)
        rec.record_density(beam)
        edges = rec.density_edges["x"]
        assert edges.shape == (51,)
        assert edges[0] == pytest.approx(-3.0)
        assert edges[-1] == pytest.approx(3.0)

    def test_reconfigure_resets_state(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",))
        beam = _seeded_beam()
        rec.record_density(beam)
        assert rec.density["x"]
        rec.configure_density(axes=("y",))
        assert rec.density == {"y": []}
        assert rec.density_edges == {}


class TestHistogramShape:
    def test_one_call_produces_one_column(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=64)
        beam = _seeded_beam()
        rec.record_density(beam)
        assert len(rec.density["x"]) == 1
        assert rec.density["x"][0].shape == (64,)

    def test_n_calls_produce_n_columns(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x", "y"), n_bins=32)
        beam = _seeded_beam()
        for _ in range(5):
            rec.record_density(beam)
        assert len(rec.density["x"]) == 5
        assert len(rec.density["y"]) == 5

    def test_density_array_returns_2d(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=16)
        beam = _seeded_beam()
        for _ in range(7):
            rec.record_density(beam)
        arr = rec.density_array("x")
        assert arr.shape == (7, 16)
        assert arr.dtype == np.int32

    def test_density_array_returns_none_for_unrecorded_axis(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",))
        assert rec.density_array("y") is None
        assert rec.density_array("phi") is None

    def test_total_counts_equal_n_alive(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=200)
        # Use a very wide explicit extent so no particles fall outside the
        # range — np.histogram would otherwise drop them.
        rec.configure_density(axes=("x",), n_bins=200,
                              extent={"x": (-1e6, 1e6)})
        beam = _seeded_beam(n=300)
        rec.record_density(beam)
        column = rec.density["x"][0]
        assert column.sum() == beam.n_alive


class TestAutoFit:
    def test_first_call_auto_fits_when_no_extent(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=100)
        beam = _seeded_beam(sigma_x=2.0)
        rec.record_density(beam)
        edges = rec.density_edges["x"]
        x_data = beam.alive_particles[:, 0]
        # Auto-fit (5σ × 1.5 margin, centred on the mean) brackets the data.
        assert edges[0] < x_data.min()
        assert edges[-1] > x_data.max()
        # Centred on the data mean, not necessarily on zero.
        midpoint = 0.5 * (edges[0] + edges[-1])
        assert midpoint == pytest.approx(float(np.mean(x_data)), abs=1e-9)

    def test_edges_are_pinned_after_first_call(self):
        # Subsequent calls must keep the same bin grid so the 2-D image is
        # geometrically meaningful — we don't want a moving y-axis.
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=64)
        beam1 = _seeded_beam(seed=1, sigma_x=1.0)
        rec.record_density(beam1)
        edges1 = rec.density_edges["x"].copy()
        beam2 = _seeded_beam(seed=2, sigma_x=10.0)
        rec.record_density(beam2)
        np.testing.assert_array_equal(rec.density_edges["x"], edges1)


class TestEdgeCases:
    def test_empty_beam_appends_zero_column(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=40,
                              extent={"x": (-5.0, 5.0)})
        beam = _seeded_beam(n=10)
        for i in range(beam.n_particles):
            beam.record_loss(i, s=0.0, element_name="APERTURE")
        assert beam.n_alive == 0

        rec.record_density(beam)
        column = rec.density["x"][0]
        assert column.shape == (40,)
        assert column.sum() == 0

    def test_unknown_axis_is_skipped_silently(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("bogus", "x"), n_bins=20)
        beam = _seeded_beam()
        rec.record_density(beam)
        # 'bogus' is silently dropped; 'x' still records.
        assert "x" in rec.density and len(rec.density["x"]) == 1
        # 'bogus' may or may not be a key; if present it must be empty.
        assert rec.density.get("bogus", []) == []

    def test_record_keeps_density_aligned_with_s(self):
        # The whole point of hooking density into ``record()`` is so
        # density columns and the s-array advance in lock-step.
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=32)
        beam = _seeded_beam()
        for s in (0.0, 10.0, 20.0, 30.0, 40.0):
            rec.record(beam, s_position=s)
        assert len(rec.s) == len(rec.density["x"]) == 5

    def test_record_all_lost_beam_keeps_density_aligned(self):
        rec = DiagnosticRecorder()
        rec.configure_density(axes=("x",), n_bins=32,
                              extent={"x": (-5.0, 5.0)})
        beam = _seeded_beam(n=10)
        for i in range(beam.n_particles):
            beam.record_loss(i, s=0.0, element_name="APERTURE")
        rec.record(beam, s_position=0.0)
        assert len(rec.s) == len(rec.density["x"]) == 1
        assert rec.density["x"][0].sum() == 0
