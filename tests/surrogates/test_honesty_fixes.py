
from tests.dataguard import needs, require  # noqa: E402
# tests/surrogates/test_honesty_fixes.py
"""Surrogate/backend honesty fixes (PRAB-review round, phase 4).

* register-multi used to CRASH (AttributeError: Scope has no
  'w_kin_lo') before registering anything — empirically reproduced,
  now a regression test against real shipped weights.
* SurrogateFieldMap inherited the FieldMapElement advance_ref NO-OP —
  a directly-substituted surrogate froze the reference energy and s
  across the element (latent hazard).  Now delegates to the wrapped
  element.
* TorchPicSolver silently ignored grid_mode='fixed' (always adaptive)
  while calling itself a drop-in replacement — now warns once.
"""
import pytest

torch = pytest.importorskip("torch")

WEIGHTS = "linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_001"
DECK = "examples/pipii/mebt/mebt.dat"


@needs("examples/pipii/mebt/mebt.dat")
def test_register_multi_completes(capsys):
    import os
    if not os.path.isdir(WEIGHTS):
        pytest.skip("shipped weights not present")
    from linac_gen.surrogates import cli, registry
    try:
        rc = cli.main(["register-multi",
                       "--source", WEIGHTS,
                       "--lattice", DECK,
                       "--names", "FMAP_002", "FMAP_003"])
        err = capsys.readouterr().err
        assert rc == 0, err
        assert "scope:" in err                     # formatted, not crashed
        assert "w_kin" in err                      # real Scope fields
        assert "IN-PROCESS only" in err            # honesty note
    finally:
        registry.clear()


def test_surrogate_advance_ref_delegates_to_wrapped():
    from linac_gen.surrogates.base import SurrogateFieldMap
    calls = []

    class _Wrapped:
        def advance_ref(self, ref):
            calls.append(("advance_ref", ref))

        def advance_ref_over(self, ref, z_from, z_to):
            calls.append(("advance_ref_over", z_from, z_to))

    s = SurrogateFieldMap.__new__(SurrogateFieldMap)   # bypass heavy ctor
    s._wrapped = _Wrapped()
    s.advance_ref("REF")
    s.advance_ref_over("REF", 0.0, 5.0)
    assert calls == [("advance_ref", "REF"),
                     ("advance_ref_over", 0.0, 5.0)]


def test_torch_solver_warns_on_fixed_grid():
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.pic.torch.solver import (AdaptiveOnlyBackendWarning,
                                            TorchPicSolver)
    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_mode="fixed",
                            sc_backend="torch")
    with pytest.warns(AdaptiveOnlyBackendWarning, match="ADAPTIVE-ONLY"):
        TorchPicSolver(cfg)


def test_torch_solver_quiet_on_adaptive_grid():
    import warnings
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.pic.torch.solver import TorchPicSolver
    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_mode="adaptive",
                            sc_backend="torch")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TorchPicSolver(cfg)
    assert not caught, [str(w.message) for w in caught]
