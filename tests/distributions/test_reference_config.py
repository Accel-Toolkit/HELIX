"""Every mode starts a beam read from a TraceWin .dst file where a multiparticle run does.

The multiparticle run takes the file's energy, frequency, emittances and Twiss (the file
wins over the project's beam, as in TraceWin); the envelope run and the CLI's envelope,
matrix, export and other modes took the project's.
"""
import copy

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.particle import PROTON
from linac_gen.distributions.factory import create_beam, reference_config
from linac_gen.io.tracewin_dst import load_dst, write_dst

W_WRITTEN, F_FILE = 7.0003, 325.0


def _file_energy(cfg):
    """The reference energy the file gives (its reader takes it from the particles)."""
    return load_dst(cfg.distribution_file)[1]["w_kin_ref"]


@pytest.fixture
def dst_config(tmp_path):
    rng = np.random.default_rng(5)
    p = rng.normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    path = tmp_path / "beam.dst"
    write_dst(str(path), p, current_mA=0.0, frequency_MHz=F_FILE, mass_MeV=PROTON.mass,
              w_kin_ref=W_WRITTEN)
    return BeamConfig(species="proton", energy=6.5, frequency=162.5, n_particles=2000,
                      source="file", distribution_file=str(path), emit_nx=0.25, alpha_x=0.0,
                      beta_x=1.0, emit_ny=0.25, alpha_y=0.0, beta_y=1.0, emit_z=0.3,
                      alpha_z=0.0, beta_z=1.0)


FIELDS = ("energy", "frequency", "emit_nx", "emit_ny", "emit_z", "alpha_x", "beta_x",
          "alpha_y", "beta_y", "alpha_z", "beta_z")


def test_it_is_what_create_beam_takes(dst_config):
    start = reference_config(dst_config)
    taken = copy.deepcopy(dst_config)
    beam = create_beam(taken, seed=1)            # sets the file's Twiss on its config
    w_file = _file_energy(dst_config)
    assert abs(w_file - W_WRITTEN) < 1e-3 and w_file != dst_config.energy
    assert (start.energy, start.frequency) == (beam.ref.w_kin, beam.ref.frequency) == (w_file, F_FILE)
    for f in FIELDS[2:]:
        assert getattr(start, f) == getattr(taken, f), f


def test_the_projects_beam_is_left_as_it_is(dst_config):
    before = {f: getattr(dst_config, f) for f in FIELDS}
    reference_config(dst_config)
    assert {f: getattr(dst_config, f) for f in FIELDS} == before


def test_a_generated_beam_is_its_own_start():
    cfg = BeamConfig(species="proton", energy=6.5, frequency=162.5, n_particles=10)
    assert reference_config(cfg) is cfg


def test_the_cli_envelope_starts_from_the_file(dst_config):
    from linac_gen.cli.common import build_ref, run_envelope_sim
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    lat = Lattice()
    lat.add(Drift("D", 100.0, aperture=30.0))
    res = run_envelope_sim(lat, dst_config)
    w_file = _file_energy(dst_config)
    assert np.asarray(res.ref_w_kin)[0] == w_file
    assert np.asarray(res.ref_frequency)[0] == F_FILE
    ref = build_ref(dst_config)
    assert (ref.w_kin, ref.frequency) == (w_file, F_FILE)
    from linac_gen.cli.common import _envelope_initial
    start, init = reference_config(dst_config), _envelope_initial(dst_config, ref)
    assert init["beta_x"] == start.beta_x != dst_config.beta_x
    assert init["alpha_z"] == start.alpha_z and init["beta_z"] == start.beta_z


def test_a_files_emittance_takes_no_mismatch(dst_config):
    """create_beam uses a file's emittance as it is, whatever the mismatch fields say."""
    import dataclasses

    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.distributions.factory import geometric_emittances
    dst_config.mismatch_x = 0.2
    start = reference_config(dst_config)
    assert start.mismatch_x == 0.0 and dst_config.mismatch_x == 0.2
    ref = build_ref(dst_config)
    plain = dataclasses.replace(start, mismatch_x=0.0, mismatch_y=0.0, mismatch_z=0.0)
    assert _envelope_initial(dst_config, ref)["emit_x"] == geometric_emittances(plain, ref.bg)[0]


def test_the_start_is_at_the_files_centroid(tmp_path):
    from linac_gen.cli.common import _envelope_initial, build_ref
    p = np.random.default_rng(6).normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    p += [1.5, 0.0, -0.7, 0.0, 0.0, 0.0]
    path = tmp_path / "off.dst"
    write_dst(str(path), p, current_mA=0.0, frequency_MHz=F_FILE, mass_MeV=PROTON.mass,
              w_kin_ref=W_WRITTEN)
    cfg = BeamConfig(species="proton", energy=6.5, frequency=162.5, n_particles=2000, source="file",
                     distribution_file=str(path), centroid_x=0.1)
    beam = create_beam(copy.deepcopy(cfg), seed=1)
    start = reference_config(cfg)
    np.testing.assert_allclose([start.centroid_x, start.centroid_y],
                               [beam.particles[:, 0].mean(), beam.particles[:, 2].mean()], rtol=1e-12)
    assert abs(start.centroid_x - 1.6) < 0.1
    assert _envelope_initial(cfg, build_ref(cfg))["centroid"][0] == start.centroid_x


def test_a_file_rewritten_with_the_same_size_and_time_is_read_again(dst_config):
    import os
    path = dst_config.distribution_file
    st = os.stat(path)
    before = reference_config(dst_config).energy
    p = np.random.default_rng(7).normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    write_dst(path, p, current_mA=0.0, frequency_MHz=F_FILE, mass_MeV=PROTON.mass, w_kin_ref=8.0)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(path).st_size == st.st_size and os.stat(path).st_mtime_ns == st.st_mtime_ns
    assert reference_config(dst_config).energy != before
