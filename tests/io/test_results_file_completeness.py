"""A results file carries everything the Results tab plots.

Imported runs showed blank tiles: the files never stored the longitudinal
normalised / 4-D / eigen-emittances, the 6×6 beam matrix (behind the 6-D
emittance, dispersion, longitudinal-Twiss and divergence plots), centroids,
peak excursions, element names, the RF frequency, the rest mass (IBS and
magnetic-stripping popups) or the recorded density / tail series; openPMD
files also dropped the loss record and the halo action scan; and the GUI
adapter turned the structured loss table into a list, so the aperture-loss
and loss-power popups failed.  Every per-step series now round-trips
EXACTLY through both formats, and a loaded run re-saves unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.hdf5_output import (
    _EXTRA_PER_STEP, load_results_hdf5, save_results_hdf5,
)
from linac_gen.io.openpmd_output import (
    load_results_openpmd, save_results_openpmd,
)


def make_run():
    """Small MP run WITH losses (tight quad aperture), density, tail
    quantiles and the halo action scan recorded."""
    lat = Lattice()
    for i in range(6):
        lat.add(Quadrupole(f"Q{i}", length=50.0,
                           gradient=(20.0 if i % 2 else -20.0),
                           aperture=4.0))
        lat.add(Drift(f"D{i}", length=100.0, aperture=4.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=3000, current=0.0)
    rng = np.random.default_rng(5)
    beam.particles[:] = rng.standard_normal((3000, 6)) * np.array(
        [1.2, 0.8, 1.2, 0.8, 5.0, 0.002])
    rec = Simulation(lat, beam, space_charge="off", density_axes=("x", "y"),
                     tail_fractions=(0.99,), record_action_scan=True).run()
    assert rec.loss_table is not None and len(rec.loss_table) > 0
    return rec


@pytest.fixture(scope="module")
def run():
    return make_run()


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.names or b.dtype.names:            # structured (loss table)
        return (a.dtype.names == b.dtype.names and a.shape == b.shape
                and all(np.array_equal(a[k], b[k]) for k in a.dtype.names))
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


@pytest.mark.parametrize("fmt", ["h5", "opmd"])
def test_every_per_step_series_round_trips_exactly(run, tmp_path, fmt):
    if fmt == "h5":
        p = tmp_path / "r.h5"
        save_results_hdf5(run, str(p))
        d = load_results_hdf5(str(p))
    else:
        p = tmp_path / "r.opmd.h5"
        save_results_openpmd(run, str(p))
        d = load_results_openpmd(str(p))
    n = len(run.s)
    for key in _EXTRA_PER_STEP:
        assert key in d, key
        if key == "element_names":
            assert d[key] == [str(x) for x in run.element_names]
        else:
            assert len(d[key]) == n and _same(d[key], getattr(run, key)), key
    assert _same(d["ref_frequency"], run.ref_frequency)
    assert d["mass_mev"] == run.mass_mev
    assert d["periodic_phase"] == run.periodic_phase
    # loss record, launched count and halo action scan (openPMD too)
    assert _same(d["loss_table"], run.loss_table)
    assert d["n_macro"] == run.n_macro
    assert "action_scan" in d and "action_scan_error" not in d


def test_density_and_tail_round_trip(run, tmp_path):
    p = tmp_path / "r.h5"
    save_results_hdf5(run, str(p))
    d = load_results_hdf5(str(p))
    for axis in ("x", "y"):
        assert _same(np.asarray(d["density"][axis], dtype=np.int32),
                     run.density_array(axis))
        assert _same(d["density_edges"][axis], run.density_edges[axis])
    assert set(d["tail"]) == set(run.tail)
    for k in run.tail:
        assert _same(d["tail"][k], run.tail[k])
    assert d["tail_fractions"] == tuple(run.tail_fractions)


def test_files_written_before_the_change_still_load(run, tmp_path):
    """A file without the new datasets loads as before (keys absent)."""
    import h5py
    p = tmp_path / "old.h5"
    save_results_hdf5(run, str(p))
    with h5py.File(p, "a") as f:
        for key in _EXTRA_PER_STEP:
            if key in f["envelope"]:
                del f["envelope"][key]
        for grp in ("density", "tail"):
            if grp in f:
                del f[grp]
        del f["reference"]["frequency"]
        for a in ("mass_mev", "periodic_phase"):
            f["envelope"].attrs.pop(a, None)
    d = load_results_hdf5(str(p))
    assert "sigma_matrix" not in d and "density" not in d
    assert _same(d["emit_nx"], run.emit_nx)


def test_scalar_ref_frequency_does_not_break_saving(run, tmp_path):
    """A results object whose ref_frequency is ONE number (older GUI runs
    overwrote the per-step list with it) still saves in both formats; the
    per-step frequency is simply not written."""
    import copy
    import h5py
    rec = copy.copy(run)
    rec.ref_frequency = 162.5
    save_results_hdf5(rec, str(tmp_path / "a.h5"))
    save_results_openpmd(rec, str(tmp_path / "a.opmd.h5"))
    with h5py.File(tmp_path / "a.h5") as f:
        assert "frequency" not in f["reference"]
        assert "action_scan" in f and "losses" in f        # file complete


def test_openpmd_export_of_loaded_results(run, tmp_path):
    """The assistant exports a loaded run as SimpleNamespace(**dict), whose
    ``s`` is a numpy array — openPMD export must accept it."""
    from types import SimpleNamespace
    p = tmp_path / "r.h5"
    save_results_hdf5(run, str(p))
    ns = SimpleNamespace(**load_results_hdf5(str(p)))
    save_results_openpmd(ns, str(tmp_path / "r.opmd.h5"))
    d = load_results_openpmd(str(tmp_path / "r.opmd.h5"))
    assert _same(d["emit_nz"], run.emit_nz)


def test_none_element_name_and_compression(run, tmp_path):
    import copy
    import h5py
    rec = copy.copy(run)
    rec.element_names = [None] + list(run.element_names[1:])
    p = tmp_path / "n.h5"
    save_results_hdf5(rec, str(p))
    d = load_results_hdf5(str(p))
    assert d["element_names"][0] == ""
    from linac_gen.io.hdf5_output import _create
    with h5py.File(tmp_path / "c.h5", "w") as f:
        big = _create(f, "big", np.zeros((300, 6, 6)))     # 10 800 values
        small = _create(f, "small", np.zeros((13, 6, 6)))  # 468 values
        assert big.compression == "gzip" and small.compression is None
        assert _same(big[:], np.zeros((300, 6, 6)))
