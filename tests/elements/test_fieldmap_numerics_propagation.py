"""set_fieldmap_numerics: class attrs + env mirror, and the env actually
reaches a FRESH interpreter (the spawn-worker scenario — bare class-attr
assignment provably does not cross the spawn boundary)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore():
    from linac_gen.elements.field_map_3d import FieldMap3D
    ik, ip = FieldMap3D.integrator_kind, FieldMap3D.interp_kind
    saved = {k: os.environ.get(k) for k in
             ("LINAC_GEN_FIELDMAP_INTEGRATOR", "LINAC_GEN_FIELDMAP_INTERP")}
    yield
    FieldMap3D.integrator_kind, FieldMap3D.interp_kind = ik, ip
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_helper_sets_attrs_and_env():
    from linac_gen.elements import field_map_3d as fm3
    fm3.set_fieldmap_numerics(integrator="dkd", interp="cubic")
    assert fm3.FieldMap3D.integrator_kind == "dkd"
    assert fm3.FieldMap3D.interp_kind == "cubic"
    assert os.environ["LINAC_GEN_FIELDMAP_INTEGRATOR"] == "dkd"
    assert os.environ["LINAC_GEN_FIELDMAP_INTERP"] == "cubic"
    fm3.set_fieldmap_numerics(integrator="kd")      # partial update
    assert fm3.FieldMap3D.integrator_kind == "kd"
    assert fm3.FieldMap3D.interp_kind == "cubic"


def test_fresh_interpreter_inherits_choice():
    """A fresh python process (== a spawned scan/error-study worker) must
    see the parent's integrator/interp via the env-initialized class
    attributes."""
    from linac_gen.elements import field_map_3d as fm3
    fm3.set_fieldmap_numerics(integrator="dkd", interp="cubic")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); "
         "from linac_gen.elements.field_map_3d import FieldMap3D; "
         "print(FieldMap3D.integrator_kind, FieldMap3D.interp_kind)"
         % repo],
        capture_output=True, text=True, env=os.environ.copy(), timeout=120)
    assert out.stdout.split() == ["dkd", "cubic"], (out.stdout, out.stderr)
