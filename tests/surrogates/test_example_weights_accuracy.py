"""The weights the development tree keeps for five of its example decks are within their
accuracy limits on those decks, and a study uses them: every set passes its check, and the
design lattice at 0 mA run with them agrees with the native run (``transport.study_spec``, what
the CLI's ``--surrogates`` runs).  Before they were retrained with the training recipe of
``test_training_recipe``, 40 of these 72 sets were refused on their own decks.  The twin that runs
everywhere: a file trained on a synthetic cavity passes its check (the data generation, the
training and the check together, on data that ships with HELIX; the earlier recipe passes it
too, so it is not a test of the recipe)."""
import warnings
from pathlib import Path

import pytest

from tests.dataguard import needs

ROOT = Path(__file__).resolve().parents[2]
_CAVITY = tuple(f"Fields/HWRDonut.{c}" for c in ("edx", "edy", "edz", "bdx", "bdy", "bdz"))
_SOLENOID = ("Fields/HWR-SOL-ANLMAP.bsz",)
_BUNCHER = tuple(f"Fields/QWR-2012-02.{c}" for c in ("edx", "edy", "edz", "bdx", "bdy", "bdz"))
_CASES = [
    ("examples/emittance_min/pipii_hwr_cm_match.lgproj", "4e24bace9139f4b5", 16,
     _CAVITY + _SOLENOID),
    ("examples/emittance_min/pipii_hwr_cm_match_4d.lgproj", "11a899849b426ef5", 16,
     _CAVITY + _SOLENOID),
    ("examples/emittance_min/pipii_hwr_cm_match_4d_relaxed.lgproj", "8c68dc256ce52adf", 16,
     _CAVITY + _SOLENOID),
    ("examples/pipii/mebt/mebt.lgproj", "7429fb95147c412d", 4, _BUNCHER),
    ("examples/pipii/mebt+hwr/mebt+hwr.lgproj", "e495f501743f11ea", 20,
     _BUNCHER + _CAVITY + _SOLENOID),
]


@pytest.mark.parametrize(
    "project, folder, n_sets",
    [pytest.param(p, f, n, marks=needs(p, *fields), id=f) for p, f, n, fields in _CASES])
def test_the_kept_weights_pass_their_check_and_a_study_uses_them(project, folder, n_sets,
                                                                monkeypatch):
    weights = ROOT / "linac_gen" / "surrogates" / "weights" / folder
    if not weights.is_dir():
        pytest.skip("the weights are not in this checkout")
    monkeypatch.chdir(ROOT)
    from linac_gen.cli.common import apply_fieldmap_settings, lattice_file_of, load_input
    from linac_gen.surrogates import registry, transport
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, cfg, conv = load_input(project)
        apply_fieldmap_settings(conv, {})
        cfg.current = 0.0
        try:
            check, spec, _notes = transport.study_spec(lat, cfg, lattice_file_of(project),
                                                       [str(weights)])
        finally:
            registry.clear()
    metas = {e.element_key: e.meta for e in spec.entries}
    assert len(metas) == n_sets
    assert {k: m.accuracy_check for k, m in metas.items() if m.accuracy_check != ""} == {}
    assert check.use, check.note
    # each set answered the design lattice, none was left to its native element
    assert check.nn_calls == n_sets and check.refused == {}


def test_a_file_the_training_writes_passes_its_check(tmp_path):
    from linac_gen.io.tracewin_geom import Channel
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.surrogates.training import load_surrogate, train_surrogate_for_element
    from tests.surrogates.test_frequency_jump import F_NEW, _map, _ref

    def cavity():
        el = _map(Channel.RF_E, F_NEW, p_flag=1)
        el.name = "CAV"
        return el

    el = cavity()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_surrogate_for_element(
            element=el, ref_template=_ref(F_NEW, 5.0), n_samples=1000,
            ref_w_kin_range=(4.5, 5.5),
            param_ranges={"ke": (el.ke * 0.8, el.ke * 1.2),
                          "phase": (el.phase - 20.0, el.phase + 20.0)},
            out_dir=tmp_path / "CAV", hidden_dims=(64, 64), epochs=300, seed=42)
        mlp, meta = load_surrogate(tmp_path / "CAV")
        SurrogateFieldMap(cavity(), mlp, meta).fitted_matrix(_ref(F_NEW, 5.0))
    assert meta.accuracy_check == ""
    assert meta.accuracy_block <= SurrogateFieldMap.ACCURACY_TOLERANCE
    assert meta.accuracy_det <= SurrogateFieldMap.ACCURACY_DET_TOLERANCE
