"""linac_gen.pic.factory.make_sc_solver: every run path builds the solver
its space-charge config describes."""
from __future__ import annotations

import pytest

from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.pic import factory
from linac_gen.pic.pic_solver import PicSolver


def test_none_and_unknown_purpose():
    assert factory.make_sc_solver(None) is None
    with pytest.raises(ValueError, match="purpose"):
        factory.make_sc_solver(SpaceChargeConfig(), purpose="sideways")


def test_numpy_config_gives_the_plain_pic():
    pic = factory.make_sc_solver(SpaceChargeConfig(nx=8, ny=8, nz=8))
    assert type(pic) is PicSolver
    assert pic.exact_backward


def test_torch_config_gives_the_torch_pic():
    pytest.importorskip("torch")
    from linac_gen.pic.torch.solver import TorchPicSolver
    cfg = SpaceChargeConfig(method="differentiable_pic", nx=8, ny=8, nz=8)
    pic = factory.make_sc_solver(cfg)
    assert type(pic) is TorchPicSolver and pic.exact_backward
    # a backtrack walk rebuilds the plain numpy PIC, as before
    assert type(factory.make_sc_solver(cfg, purpose="backward")) is PicSolver


def test_halo_config_loads_its_corrector(monkeypatch, tmp_path):
    from linac_gen.pic.ml.solver import HaloPicSolver
    loaded = []
    monkeypatch.setattr(HaloPicSolver, "load",
                        lambda self, d: loaded.append(str(d)))
    cfg = SpaceChargeConfig(sc_backend="halo", nx=8, ny=8, nz=8,
                            halo={"corrector_dir": str(tmp_path)})
    pic = factory.make_sc_solver(cfg)
    assert type(pic) is HaloPicSolver and not pic.exact_backward
    assert loaded == [str(tmp_path)]
    assert type(factory.make_sc_solver(cfg, purpose="backward")) is PicSolver


def test_simulation_builds_through_the_factory():
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    lat = Lattice()
    lat.add(Drift(name="D1", length=20.0, aperture=50.0))
    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5, current=5.0,
                     n_particles=200, emit_nx=0.2, beta_x=0.5, emit_ny=0.2,
                     beta_y=0.5, emit_z=0.06, beta_z=500.0)
    sim = Simulation(lat, create_beam(cfg, seed=3),
                     space_charge=SpaceChargeConfig(nx=8, ny=8, nz=8,
                                                    use_gpu="cpu"))
    sim.run()
    assert type(sim._pic_solver) is PicSolver


def test_error_study_runs_the_configured_engine(monkeypatch):
    """The error study used to build the numpy PicSolver whatever engine
    the config selected."""
    pytest.importorskip("torch")
    from linac_gen.errors.error_model import ErrorStudy
    from linac_gen.pic.torch.solver import TorchPicSolver
    built = []
    real = factory.make_sc_solver

    def spy(cfg, **kw):
        pic = real(cfg, **kw)
        built.append(pic)
        return pic

    monkeypatch.setattr(factory, "make_sc_solver", spy)
    lat = Lattice()
    lat.add(Drift(name="D1", length=20.0, aperture=50.0))
    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5, current=5.0,
                     n_particles=200, emit_nx=0.2, beta_x=0.5, emit_ny=0.2,
                     beta_y=0.5, emit_z=0.06, beta_z=500.0)
    sc = SpaceChargeConfig(method="differentiable_pic", nx=8, ny=8, nz=8,
                           use_gpu="cpu")
    ErrorStudy(lat, cfg, n_seeds=1, sc_config=sc).run()
    assert built and all(type(p) is TorchPicSolver for p in built)


def test_torch_run_records_the_adaptive_grid_it_used():
    """The torch PIC always resizes its grid: a fixed-grid config is run
    adaptive (with a warning) and the results say so; the caller's config
    object is left as it was."""
    pytest.importorskip("torch")
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.pic.torch.solver import AdaptiveOnlyBackendWarning
    lat = Lattice()
    lat.add(Drift(name="D1", length=20.0, aperture=50.0))
    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5, current=5.0,
                     n_particles=200, emit_nx=0.2, beta_x=0.5, emit_ny=0.2,
                     beta_y=0.5, emit_z=0.06, beta_z=500.0)
    sc = SpaceChargeConfig(nx=8, ny=8, nz=8, sc_backend="torch",
                           use_gpu="cpu")
    with pytest.warns(AdaptiveOnlyBackendWarning):
        rec = Simulation(lat, create_beam(cfg, seed=3), space_charge=sc).run()
    assert sc.grid_mode == "fixed"
    assert rec.sc_config.grid_mode == "adaptive"
    assert rec.sc_config.method == "differentiable_pic"


def _spy(monkeypatch):
    built = []
    real = factory.make_sc_solver

    def spy(cfg, **kw):
        pic = real(cfg, **kw)
        built.append((kw.get("purpose", "forward"), pic))
        return pic

    monkeypatch.setattr(factory, "make_sc_solver", spy)
    return built


def test_simulation_goes_through_the_factory(monkeypatch):
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    built = _spy(monkeypatch)
    lat = Lattice()
    lat.add(Drift(name="D1", length=20.0, aperture=50.0))
    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5, current=5.0,
                     n_particles=200, emit_nx=0.2, beta_x=0.5, emit_ny=0.2,
                     beta_y=0.5, emit_z=0.06, beta_z=500.0)
    sim = Simulation(lat, create_beam(cfg, seed=3),
                     space_charge=SpaceChargeConfig(nx=8, ny=8, nz=8,
                                                    use_gpu="cpu"))
    sim.run()
    assert [p for _, p in built] == [sim._pic_solver]


def test_backward_rebuild_records_the_numpy_engine():
    cfg = SpaceChargeConfig(sc_backend="halo", nx=8, ny=8, nz=8)
    pic = factory.make_sc_solver(cfg, purpose="backward")
    assert type(pic) is PicSolver and pic.config.sc_backend == "numpy"
    assert cfg.sc_backend == "halo"                    # caller's untouched


def test_validate_replay_goes_through_the_factory(monkeypatch, tmp_path):
    """backtrack --validate used to build PicSolver directly."""
    from linac_gen.__main__ import main as cli_main
    built = _spy(monkeypatch)
    deck = tmp_path / "d.dat"
    deck.write_text("DRIFT 100 20 0\nEND\n")
    rc = cli_main(["backtrack", str(deck), "--mode", "mp", "--species", "H-",
                   "--energy", "2.1", "--freq", "162.5", "--current", "1",
                   "--n-particles", "200", "--sc-on", "--validate", "-q",
                   "--out", str(tmp_path)])
    assert rc in (0, 1)                      # closure verdict is not the point
    assert any(purpose == "forward" for purpose, _ in built)
