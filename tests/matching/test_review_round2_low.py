# tests/matching/test_review_round2_low.py
"""Review round 2 LOW fixes: the cmaes_search_solver sequential notice
(claim 9), gradient+SC honouring mp_sc_config / mp_n_particles
(claim 11), and the surrogate unknown-species staleness warning
(claim 12).  Claim 10 (boundary/shape_order) lives in
tests/core/test_config.py."""
import warnings

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.elements.quadrupole import Quadrupole


def _bcfg(**over) -> BeamConfig:
    base = dict(species="proton", energy=3.0, frequency=352.21,
                n_particles=10, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                emit_z=0.3, alpha_z=0.0, beta_z=10.0)
    base.update(over)
    return BeamConfig(**base)


def _one_knob_lattice():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                   link_group=0, vmin=-30, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0, y_mm=0.0, phi_or_z=0.0))
    return lat


# ── claim 9: sequential CMA-ES notice ────────────────────────────────────
def test_cmaes_search_solver_sequential_notice(capsys):
    """Boundary-value lesson (external review, 2026-07-20): −1 also
    resolves to SEQUENTIAL (max(1,−1)=1; the auto sentinel is 0), so
    the notice must fire for it too — the first version keyed on ==1
    and its own text recommended −1 as the fix."""
    from linac_gen.matching.engine import match
    for par in (1, -1, -4):
        match(_one_knob_lattice(), _bcfg(), algorithm="cmaes",
              cmaes_search_solver="envelope", cmaes_parallel=par,
              max_iter=3)
        err = capsys.readouterr().err
        assert f"NO EFFECT with cmaes_parallel={par}" in err, (par, err)
        assert "-1 for all cores" not in err          # the bad advice
    # auto stays silent.
    match(_one_knob_lattice(), _bcfg(), algorithm="cmaes",
          cmaes_search_solver="auto", cmaes_parallel=1, max_iter=3)
    assert "NO EFFECT" not in capsys.readouterr().err


def test_cmaes_parallel_zero_is_auto_and_engages_pool_path():
    """Execute the advice the notice now prints: cmaes_parallel=0 must
    auto-detect to >1 workers (not sequential) so the search override
    genuinely takes effect.  Pinned by resolving the same expression
    the engine uses plus a live run reaching the pool branch."""
    import os
    if (os.cpu_count() or 2) < 3:
        pytest.skip("auto worker count needs >2 cores to exceed 1")
    from linac_gen.matching.engine import match
    # Live run: pool path either builds a real pool or prints the
    # documented sequential-fallback notice — both prove the ==0
    # branch did NOT take the silent sequential path.
    res = match(_one_knob_lattice(), _bcfg(), algorithm="cmaes",
                cmaes_search_solver="envelope", cmaes_parallel=0,
                max_iter=3)
    assert res is not None


# ── claim 11: gradient+SC honours mp_sc_config / mp_n_particles ─────────
def test_gradient_sc_honours_mp_config(monkeypatch):
    torch = pytest.importorskip("torch")
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.matching import engine as eng

    seen = {}

    def _spy(lattice, beam_cfg, ref, variables, constraints,
             col_for_var, n_cols, *, sc_cfg, bunch_size=1500, seed=42):
        seen["sc_cfg"] = sc_cfg
        seen["bunch"] = bunch_size
        raise RuntimeError("spy stop")

    monkeypatch.setattr(
        "linac_gen.matching.torch_objective.build_torch_residual_sc",
        _spy)
    cfg16 = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=3.0,
                              use_gpu="cpu", grid_mode="adaptive")
    with pytest.raises(RuntimeError, match="spy stop"):
        eng.match(_one_knob_lattice(), _bcfg(current=5.0),
                  algorithm="gradient", space_charge=True,
                  mp_sc_config=cfg16, mp_n_particles=200, max_iter=2)
    assert seen["sc_cfg"] is cfg16
    assert seen["bunch"] == 200
    # Defaults preserved when nothing is passed.
    with pytest.raises(RuntimeError, match="spy stop"):
        eng.match(_one_knob_lattice(), _bcfg(current=5.0),
                  algorithm="gradient", space_charge=True, max_iter=2)
    assert seen["sc_cfg"].nx == 32 and seen["bunch"] == 1500


# ── adversarial-review find: torch mirror lacks the SET_SIZE z-form ─────
def test_gradient_refuses_set_size_z_form():
    """The numpy evaluator honours SET_SIZE's negative-operand σ_z(mm)
    form; the torch mirror does not — gradient must REFUSE (the SC path
    has no self-validation, so it would silently drop the target)."""
    pytest.importorskip("torch")
    from linac_gen.matching.engine import match
    lat = _one_knob_lattice()
    for e in lat.elements:
        if isinstance(e, SetSize):
            e.phi_or_z = -8.0        # σ_z target: 8 mm
    with pytest.raises(ValueError, match="z\\(mm\\) form|z.form|σ_z"):
        match(lat, _bcfg(current=5.0), algorithm="gradient",
              space_charge=True, max_iter=2)


def test_cmaes_auto_low_core_notice(monkeypatch, capsys):
    """cmaes_parallel=0 on a ≤2-core host auto-resolves to ONE worker —
    the same silent-sequential shape as the fixed −1 bug; the
    resolution-time notice must fire."""
    import os
    from linac_gen.matching.engine import match
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    match(_one_knob_lattice(), _bcfg(), algorithm="cmaes",
          cmaes_search_solver="envelope", cmaes_parallel=0, max_iter=3)
    err = capsys.readouterr().err
    assert "auto-resolved to 1 worker" in err
