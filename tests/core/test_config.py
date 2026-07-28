import pytest

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
    # Defaults chosen per the H- 5 mA convergence study; see docs/convergence/.
    sc = SpaceChargeConfig()
    assert sc.nx == 96
    assert sc.grid_extent == 5.0
    assert sc.boundary == "open"
    assert sc.grid_mode == "fixed"

def test_sc_config_custom():
    sc = SpaceChargeConfig(nx=128, ny=128, nz=128, boundary="open")
    assert sc.nx == 128
    assert sc.boundary == "open"


# ---------------------------------------------------------------------------
# Validation (__post_init__) tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"energy": 0.0},
    {"energy": -1.0},
    {"frequency": 0.0},
    {"frequency": -10.0},
    {"current": -5.0},
    {"n_particles": 0},
    {"n_particles": -100},
    {"cutoff": 0.0},
    {"emit_nx": -0.1},
    {"beta_x": 0.0},
    {"beta_x": -0.5},
    {"source": "bogus"},
    # Duty cycle must be in (0, 100] %
    {"duty_cycle": 0.0},
    {"duty_cycle": -10.0},
    {"duty_cycle": 100.01},
    # Mismatch must be > -100 % (avoid zero / negative emittance)
    {"mismatch_x": -100.0},
    {"mismatch_y": -150.0},
    {"mismatch_z": -100.0},
])
def test_beam_config_rejects_invalid(kwargs):
    base = dict(species="proton", energy=3.0, frequency=352.21, current=10.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        BeamConfig(**base)


def test_beam_config_accepts_new_fields():
    """Duty cycle, centroid offsets, and mismatch are accepted when valid."""
    bc = BeamConfig(
        species="H-", energy=2.1226695, frequency=162.5, current=5.0,
        duty_cycle=50.0,
        centroid_x=0.5, centroid_xp=-0.1,
        centroid_y=-0.3, centroid_yp=0.2,
        centroid_dphi=1.0, centroid_dw=0.001,
        mismatch_x=10.0, mismatch_y=-5.0, mismatch_z=0.0,
    )
    assert bc.duty_cycle == 50.0
    assert bc.centroid_x == 0.5
    assert bc.mismatch_x == 10.0


@pytest.mark.parametrize("kwargs", [
    {"nx": 0}, {"ny": -1},
    {"grid_extent": 0.0}, {"grid_extent": -1.0},
    {"shape_order": 0},
    {"boundary": "conducting"},     # not yet implemented
    # "periodic" used to VALIDATE and then be silently ignored (the
    # Poisson solve is hard-wired open-BC Hockney) — refused since the
    # 2026-07 review round.
    {"boundary": "periodic"},
    {"solver": "multigrid"},        # not yet implemented
    {"grid_mode": "sliding"},
])
def test_sc_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        SpaceChargeConfig(**kwargs)


def test_sc_config_shape_order_deprecated():
    """shape_order was never read by any solver (kernel= selects the
    deposition order) — any value != 1 now warns."""
    with pytest.warns(DeprecationWarning, match="shape_order"):
        SpaceChargeConfig(shape_order=2)


def test_space_charge_config_use_gpu_default_and_validation():
    """SpaceChargeConfig.use_gpu defaults to 'auto' and validates enum."""
    assert SpaceChargeConfig().use_gpu == "auto"
    assert SpaceChargeConfig(use_gpu="cpu").use_gpu == "cpu"
    assert SpaceChargeConfig(use_gpu="gpu").use_gpu == "gpu"
    with pytest.raises(ValueError, match="use_gpu"):
        SpaceChargeConfig(use_gpu="tpu")
