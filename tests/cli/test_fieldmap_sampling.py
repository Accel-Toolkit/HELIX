"""apply_fieldmap_settings: sampling-switch semantics (all regimes)."""
from __future__ import annotations

import os

import pytest

from linac_gen.cli.common import apply_fieldmap_settings
from linac_gen.elements import field_map_3d as fm3

pytestmark = pytest.mark.skipif(not fm3.kernel_available(),
                                reason="compiled kernel not built")


@pytest.fixture(autouse=True)
def _restore_switch():
    before = fm3.fused_kernel_enabled()
    env = os.environ.get("LINAC_GEN_FIELDMAP_KERNEL")
    yield
    fm3.use_fused_kernel(before)
    if env is None:
        os.environ.pop("LINAC_GEN_FIELDMAP_KERNEL", None)
    else:
        os.environ["LINAC_GEN_FIELDMAP_KERNEL"] = env


@pytest.mark.parametrize("cli, conv, expect", [
    ({"fieldmap_sampling": "scipy"}, {}, False),
    ({"fieldmap_sampling": "kernel"}, {}, True),
    ({}, {"fieldmap_sampling": "scipy"}, False),
    ({}, {"fieldmap_kernel": False}, False),          # legacy bool key
    ({}, {"fieldmap_kernel": True}, True),
    ({}, {"fieldmap_kernel": "false"}, False),        # stringly bool
    ({}, {"fieldmap_kernel": "true"}, True),
    # CLI wins over project
    ({"fieldmap_sampling": "kernel"}, {"fieldmap_sampling": "scipy"}, True),
])
def test_sampling_regimes(cli, conv, expect):
    fm3.use_fused_kernel(not expect)      # start from the opposite state
    apply_fieldmap_settings(conv, cli)
    assert fm3.fused_kernel_enabled() is expect
    # env mirror for spawned workers
    assert os.environ["LINAC_GEN_FIELDMAP_KERNEL"] == ("1" if expect
                                                       else "0")


def test_absent_keys_leave_switch_untouched():
    fm3.use_fused_kernel(False)
    apply_fieldmap_settings({}, {})
    assert fm3.fused_kernel_enabled() is False
    fm3.use_fused_kernel(True)
    apply_fieldmap_settings({}, {"nx": 32})
    assert fm3.fused_kernel_enabled() is True


def test_unrecognized_value_warns_and_defaults_to_kernel():
    fm3.use_fused_kernel(False)
    with pytest.warns(UserWarning, match="unrecognized fieldmap_sampling"):
        apply_fieldmap_settings({}, {"fieldmap_sampling": "scippy"})
    assert fm3.fused_kernel_enabled() is True
