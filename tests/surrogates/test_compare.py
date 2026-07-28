"""M4 — comparison framework: CompareReport dataclass + the
``compare_envelope`` orchestration that engages / disengages the
registry between the two runs."""
import numpy as np
import pytest

from linac_gen.surrogates import registry
from linac_gen.surrogates.compare import (
    CompareReport,
    compare_envelope,
    plot_compare_report,
)


# ---------------------------------------------------------------------------
def test_compare_report_dataclass_methods():
    """speedup, worst_rel_diff, and summary_text behave."""
    s = np.linspace(0.0, 100.0, 11)
    sig_b = {"x": np.full(11, 1.0), "y": np.full(11, 1.0),
             "phi": np.full(11, 2.0), "w": np.full(11, 0.01)}
    sig_s = {"x": np.full(11, 1.01), "y": np.full(11, 1.0),
             "phi": np.full(11, 2.0), "w": np.full(11, 0.01)}
    end = {
        "sigma_x_base": 1.0, "sigma_x_surr": 1.01, "rel_diff_x": 0.01,
        "sigma_y_base": 1.0, "sigma_y_surr": 1.0,   "rel_diff_y": 0.0,
        "sigma_phi_base": 2.0, "sigma_phi_surr": 2.0, "rel_diff_phi": 0.0,
        "sigma_w_base": 0.01, "sigma_w_surr": 0.01, "rel_diff_w": 0.0,
    }
    rep = CompareReport(
        s=s, sigma_baseline=sig_b, sigma_surrogate=sig_s,
        end_of_line=end, wall_baseline_s=2.0, wall_surrogate_s=1.0,
        scope_ok=True, surrogate_names=["FMAP_001"], notes=[],
    )
    assert rep.speedup() == pytest.approx(2.0)
    assert rep.worst_rel_diff() == pytest.approx(0.01)
    text = rep.summary_text()
    assert "FMAP_001" in text
    assert "2.00x" in text
    assert "sigma_x" in text


def test_compare_report_speedup_safe_on_zero_wall():
    """speedup() returns nan rather than dividing by zero."""
    rep = CompareReport(s=np.zeros(1), sigma_baseline={"x": np.zeros(1),
        "y": np.zeros(1), "phi": np.zeros(1), "w": np.zeros(1)},
        sigma_surrogate={"x": np.zeros(1), "y": np.zeros(1),
        "phi": np.zeros(1), "w": np.zeros(1)},
        end_of_line={}, wall_baseline_s=1.0, wall_surrogate_s=0.0,
        scope_ok=True, surrogate_names=[], notes=[])
    assert np.isnan(rep.speedup())


# ---------------------------------------------------------------------------
class _FakeEnvelopeResult:
    """Stand-in for the envelope solver's result object."""
    def __init__(self):
        self.s = np.array([0.0, 100.0, 200.0])
        self.sigma_x = np.array([1.0, 1.05, 1.10])
        self.sigma_y = np.array([1.0, 1.04, 1.08])
        self.sigma_phi = np.array([2.0, 2.0, 2.0])
        self.sigma_w = np.array([0.01, 0.01, 0.01])


def test_compare_envelope_orchestration_with_no_surrogates(monkeypatch):
    """Both runs use the fake result -> rel.diff is zero; both walls
    are recorded."""
    monkeypatch.setattr(
        "linac_gen.surrogates.compare._envelope_run",
        lambda lattice, ref, init_twiss, current, should_abort=None:
            _FakeEnvelopeResult(),
    )
    registry.clear()
    report = compare_envelope(lattice=None, ref=None, init_twiss={},
                              current=0.0)
    assert isinstance(report.s, np.ndarray)
    # No surrogate engaged -> identical runs.
    assert report.worst_rel_diff() < 1e-12
    assert report.surrogate_names == []
    # Both wall-clocks recorded (non-negative).
    assert report.wall_baseline_s >= 0
    assert report.wall_surrogate_s >= 0
    # Summary text builds.
    summary = report.summary_text()
    assert "Surrogates engaged" in summary
    assert "registry empty" in summary


def test_compare_envelope_restores_registry_state(monkeypatch):
    """After compare_envelope returns, the registry is in the same
    state as before the call -- even if surrogates were temporarily
    cleared / repopulated mid-run."""
    monkeypatch.setattr(
        "linac_gen.surrogates.compare._envelope_run",
        lambda lattice, ref, init_twiss, current, should_abort=None:
            _FakeEnvelopeResult(),
    )
    # Pre-load the registry with a marker surrogate.
    from linac_gen.elements.base import FieldMapElement
    from linac_gen.surrogates.base import (
        MlpHead, Scope, SurrogateFieldMap, SurrogateMetadata,
    )

    class _Mock(FieldMapElement):
        def __init__(self):
            super().__init__(name="ORIG_KEEP", length=1.0, aperture=1.0,
                              n_steps=1)
        def track_rk4(self, beam, ds): return None
        def fitted_matrix(self, ref): return np.eye(6)
        def fitted_matrix_slice(self, ref, ds_mm): return np.eye(6)

    mlp = MlpHead(input_dim=3, output_dim=36, hidden_dims=(2,))
    meta = SurrogateMetadata(
        element_key="ORIG_KEEP", element_class="_Mock",
        architecture={"input_dim": 3, "output_dim": 36,
                      "hidden_dims": [2], "activation": "silu",
                      "param_names": []},
        scope=Scope(input_names=["w_kin", "beta", "gamma"],
                    input_lo=np.array([0.0]*3),
                    input_hi=np.array([100.0]*3)),
        input_norm={"mean": [0.0]*3, "std": [1.0]*3},
        output_norm={"mean": [0.0]*36, "std": [1.0]*36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="lh", created_iso="",
    )
    registry.clear()
    surr = SurrogateFieldMap(_Mock(), mlp, meta)
    registry.register(surr)
    pre = registry.list_registered()

    _ = compare_envelope(lattice=None, ref=None, init_twiss={},
                         current=0.0)

    # Registry state preserved across the call.
    assert registry.list_registered() == pre
    registry.clear()


# ---------------------------------------------------------------------------
def test_plot_compare_report_writes_png(tmp_path):
    """plot_compare_report writes a non-empty PNG without crashing."""
    rep = CompareReport(
        s=np.linspace(0.0, 100.0, 11),
        sigma_baseline={"x": np.full(11, 1.0), "y": np.full(11, 1.0),
                        "phi": np.full(11, 2.0), "w": np.full(11, 0.01)},
        sigma_surrogate={"x": np.full(11, 1.005), "y": np.full(11, 1.0),
                         "phi": np.full(11, 2.0), "w": np.full(11, 0.01)},
        end_of_line={
            "sigma_x_base": 1.0, "sigma_x_surr": 1.005, "rel_diff_x": 5e-3,
            "sigma_y_base": 1.0, "sigma_y_surr": 1.0,   "rel_diff_y": 0.0,
            "sigma_phi_base": 2.0, "sigma_phi_surr": 2.0, "rel_diff_phi": 0.0,
            "sigma_w_base": 0.01, "sigma_w_surr": 0.01, "rel_diff_w": 0.0,
        },
        wall_baseline_s=0.5, wall_surrogate_s=0.1,
        scope_ok=True, surrogate_names=["A"],
    )
    out = plot_compare_report(rep, tmp_path / "diff.png")
    assert out.exists()
    assert out.stat().st_size > 1000   # not a stub


# ---------------------------------------------------------------------------
# Cooperative cancellation
# ---------------------------------------------------------------------------

def test_envelope_run_raises_on_abort():
    """The solver's abort contract is 'return partial results'; the
    compare helper must convert that into a raise — a report diffing two
    differently-truncated runs would be silently wrong."""
    from linac_gen.core.cancelled import OperationCancelled
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.surrogates.compare import _envelope_run

    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Drift("D2", 100.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    twiss = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.25,
                 alpha_y=0.0, beta_y=1.0, emit_y=0.25,
                 alpha_z=0.0, beta_z=10.0, emit_z=0.3)

    with pytest.raises(OperationCancelled):
        _envelope_run(lat, ref, twiss, current=0.0,
                      should_abort=lambda: True)


def test_compare_envelope_cancel_restores_registry(monkeypatch):
    from linac_gen.core.cancelled import OperationCancelled

    def _aborting(lattice, ref, init_twiss, current, should_abort=None):
        raise OperationCancelled("cancelled")

    monkeypatch.setattr(
        "linac_gen.surrogates.compare._envelope_run", _aborting)
    registry.clear()
    before = dict(registry._REGISTRY)
    with pytest.raises(OperationCancelled):
        compare_envelope(lattice=None, ref=None, init_twiss={}, current=0.0)
    assert dict(registry._REGISTRY) == before   # swap fully unwound


def test_compare_mp_records_fast_path_flag(monkeypatch):
    """compare_mp captures the process-global linear-matrix fast-path
    flag in the report (2026-07-25 review, claim 14: any number from
    the tool silently depended on uncaptured global state)."""
    from linac_gen.surrogates.compare import compare_mp

    fake = (np.array([0.0, 1.0]),
            {k: np.array([1.0, 1.0]) for k in ("x", "y", "z")},
            {k: np.array([0.2, 0.2]) for k in ("nx", "ny")},
            np.array([100.0, 100.0]))
    monkeypatch.setattr("linac_gen.surrogates.compare._mp_run",
                        lambda lattice, beam: object())
    monkeypatch.setattr("linac_gen.surrogates.compare._extract_mp",
                        lambda res: fake)
    registry.clear()
    try:
        registry.set_fast_path_enabled(True)
        rep = compare_mp(lattice=None, beam=None)
        assert rep.fast_path_enabled is True
        assert any("fast path" in n for n in rep.notes)

        registry.set_fast_path_enabled(False)
        rep2 = compare_mp(lattice=None, beam=None)
        assert rep2.fast_path_enabled is False
        assert not any("fast path" in n for n in rep2.notes)
    finally:
        registry.set_fast_path_enabled(False)
        registry.clear()
