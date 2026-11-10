"""How a surrogate is fitted: each output's normalising spread floored, a learning rate that
falls along a half cosine, the epoch kept by its validation mean squared error, a validation
MAPE over the outputs that vary, and the recipe and the element's structure recorded in the
weights file.  With the earlier recipe (spread + 1e-12, a constant rate, the epoch kept by
validation MAPE) the round-off of a map's zero couplings held nearly all the loss and no
retraining brought the refused HWR and MEBT weights within the accuracy limits."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from linac_gen.elements.base import FieldMapElement
from linac_gen.surrogates import training
from linac_gen.surrogates.base import STRUCTURE, structure_record
from linac_gen.surrogates.training import (
    LR_FINAL_FRACTION,
    OUTPUT_STD_FLOOR,
    epoch_learning_rate,
    load_surrogate,
    mape_outputs,
    train_surrogate,
    train_surrogate_for_element,
)


def _data(n=200, seed=0, noise=0.0):
    """Two inputs and the 36 entries of a map: the identity, two entries that vary with the
    inputs (M[0,1], M[2,3]; with relative ``noise``) and one zero coupling that holds
    round-off (M[4,0])."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(2.0, 10.0, n), rng.uniform(0.8, 1.2, n)])
    Y = np.tile(np.eye(6).reshape(36), (n, 1))
    Y[:, 1] = 1e-3 * X[:, 0] * X[:, 1] * (1.0 + noise * rng.standard_normal(n))
    Y[:, 15] = 8e-4 * X[:, 0] * X[:, 1] * (1.0 + noise * rng.standard_normal(n))
    Y[:, 24] = 1e-13 * rng.standard_normal(n)
    return X, Y


def _split(X, Y, seed=0, val_frac=0.2):
    """train_surrogate's split of (X, Y): (X_train, Y_train, X_val, Y_val)."""
    perm = np.random.default_rng(seed).permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    X, Y = X[perm], Y[perm]
    return X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]


def _predict(mlp, norm, X):
    x = (X - np.asarray(norm["input_mean"])) / np.asarray(norm["input_std"])
    with torch.no_grad():
        y = mlp(torch.as_tensor(x, dtype=torch.float64)).numpy()
    return y * np.asarray(norm["output_std"]) + np.asarray(norm["output_mean"])


def test_an_output_that_does_not_vary_is_normalised_by_the_floor():
    X, Y = _data()
    _, norm, _ = train_surrogate(X, Y, hidden_dims=(8,), epochs=2, batch_size=64, seed=0)
    std = np.asarray(norm["output_std"])
    _, Yt, _, _ = _split(X, Y)
    raw = Yt.std(0) + 1e-12
    assert raw[24] < 1e-11                          # round-off of a zero coupling
    assert std[24] == OUTPUT_STD_FLOOR
    assert std[0] == std[2] == OUTPUT_STD_FLOOR     # exact constants
    assert std[1] == raw[1] and std[15] == raw[15]  # the entries that vary keep their spread


def test_the_val_mape_is_over_the_outputs_that_vary():
    X, Y = _data()
    events = []
    mlp, norm, val_mape = train_surrogate(X, Y, hidden_dims=(16,), epochs=5, batch_size=64,
                                          seed=0, progress_callback=events.append)
    on = mape_outputs(norm["output_std"])
    assert np.flatnonzero(on).tolist() == [1, 15]
    _, _, Xv, Yv = _split(X, Y)
    per = np.abs(_predict(mlp, norm, Xv) - Yv) / (np.abs(Yv) + 1e-12)
    assert val_mape == pytest.approx(float(per[:, on].mean()), rel=1e-12)
    assert per.mean() > 100 * val_mape              # what the zero and constant entries make of
                                                    # a MAPE over every output
    heat = np.asarray(events[-1]["per_entry_val_mape"]).reshape(36)
    assert (heat[~on] == 0.0).all() and (heat[on] > 0.0).all()


def test_when_no_output_varies_the_mape_is_over_the_entries_that_are_not_zero():
    flat = np.full(36, OUTPUT_STD_FLOOR)
    eye = np.eye(6).reshape(36)
    assert np.flatnonzero(mape_outputs(flat, eye)).tolist() == [0, 7, 14, 21, 28, 35]
    assert mape_outputs(flat, np.zeros(36)).all() and mape_outputs(flat).all()


def test_a_training_needs_an_epoch():
    X, Y = _data()
    with pytest.raises(ValueError, match="at least one epoch"):
        train_surrogate(X, Y, hidden_dims=(8,), epochs=0)


def test_the_learning_rate_falls_along_a_half_cosine(monkeypatch):
    seen = []
    real_step = torch.optim.Adam.step

    def step(self, *args, **kwargs):
        seen.append(self.param_groups[0]["lr"])
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    X, Y = _data()
    epochs, lr = 7, 2e-3
    events = []
    # one batch per epoch: one optimiser step per epoch
    train_surrogate(X, Y, hidden_dims=(8,), epochs=epochs, lr=lr, batch_size=1000, seed=0,
                    progress_callback=events.append)
    want = [lr * (LR_FINAL_FRACTION + (1.0 - LR_FINAL_FRACTION) * 0.5
                  * (1.0 + math.cos(math.pi * e / (epochs - 1)))) for e in range(epochs)]
    assert seen == want
    assert seen[0] == lr and seen[-1] == pytest.approx(lr * LR_FINAL_FRACTION, rel=1e-15)
    assert [ev["lr"] for ev in events] == seen
    assert epoch_learning_rate(lr, 0, 1) == lr      # a single epoch runs at lr


def test_the_epoch_kept_is_the_one_with_the_lowest_validation_mse():
    # noisy targets: the validation error passes a minimum before the last epoch
    X, Y = _data(noise=0.05)
    events, rep = [], {}
    mlp, norm, val_mape = train_surrogate(X, Y, hidden_dims=(16,), epochs=30, lr=1e-2,
                                          batch_size=16, seed=0,
                                          progress_callback=events.append, report=rep)
    mse = [ev["val_mse"] for ev in events]
    mape = [ev["val_mape"] for ev in events]
    k = int(np.argmin(mse))
    # the case tells the criteria apart: neither the last epoch nor the lowest MAPE is it
    assert k != len(events) - 1 and k != int(np.argmin(mape))
    assert rep["kept_epoch"] == events[-1]["best_epoch"] == k + 1
    assert rep["val_mse"] == mse[k] == events[-1]["best_val_mse"]
    assert val_mape == mape[k] == events[-1]["best_val_mape"]
    bests = [ev["best_val_mse"] for ev in events]
    assert all(b2 <= b1 for b1, b2 in zip(bests, bests[1:], strict=False))
    # the model returned is that epoch's: its validation error, in the training's units
    _, _, Xv, Yv = _split(X, Y)
    x = (Xv - np.asarray(norm["input_mean"])) / np.asarray(norm["input_std"])
    target = (Yv - np.asarray(norm["output_mean"])) / np.asarray(norm["output_std"])
    with torch.no_grad():
        y = mlp(torch.as_tensor(x, dtype=torch.float64))
    got = float(((y - torch.as_tensor(target, dtype=torch.float64)) ** 2).mean())
    assert got == pytest.approx(mse[k], rel=1e-12)


class _Cavity(FieldMapElement):
    """A mock field map with a known map and the structure of a SET_SYNC_PHASE cavity."""

    def __init__(self):
        super().__init__(name="CAV", length=250.0, aperture=15.0, n_steps=20)
        self.scale = 1.0
        self.field_file = "/an/absolute/place/Fields/HWRDonut"
        self.geom = 7700
        self.p_flag = 1

    def track_rk4(self, beam, ds):
        return None

    def fitted_matrix(self, ref):
        M = np.eye(6)
        M[0, 1] = 0.001 * ref.w_kin * self.scale
        M[2, 3] = 0.0008 * ref.w_kin * self.scale
        return M

    def fitted_matrix_slice(self, ref, ds_mm):
        return np.eye(6)


class _Ref:
    def __init__(self, w_kin: float = 3.0):
        self.w_kin = float(w_kin)
        self.beta = 0.1
        self.gamma = 1.005

    def copy(self):
        return _Ref(self.w_kin)


def _train(out, **kw):
    opts = dict(element=_Cavity(), ref_template=_Ref(), n_samples=64,
                ref_w_kin_range=(2.0, 10.0), param_ranges={"scale": (0.8, 1.2)}, out_dir=out,
                hidden_dims=(8,), epochs=4, lr=2e-3, batch_size=16, seed=0)
    opts.update(kw)
    return train_surrogate_for_element(**opts)


def test_the_weights_file_records_the_recipe_and_the_structure(tmp_path):
    _, meta = _train(tmp_path / "CAV")
    r = meta.train_recipe
    assert r["optimizer"] == "adam" and r["lr_schedule"] == "cosine"
    assert r["lr"] == 2e-3 and r["lr_final"] == pytest.approx(2e-5, rel=1e-15)
    assert r["batch_size"] == 16 and r["val_frac"] == 0.2
    assert r["output_std_floor"] == OUTPUT_STD_FLOOR
    assert 1 <= r["kept_epoch"] <= 4 and r["val_mse"] > 0.0
    assert r["val_mape_outputs"] == 2
    assert r["torch"] == str(torch.__version__) and r["numpy"] == str(np.__version__)
    s = meta.train_structure
    assert s["class"] == "_Cavity" and s["field_file"] == "HWRDonut"   # no machine path
    assert s["p_flag"] == 1 and s["geom"] == 7700 and s["length"] == 250.0
    assert s["n_steps"] == 20 and s["aperture"] == 15.0
    assert set(s) - {"class"} <= set(STRUCTURE)
    _, loaded = load_surrogate(tmp_path / "CAV")
    assert loaded.train_recipe == r and loaded.train_structure == s
    on_disk = json.loads((tmp_path / "CAV" / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk["train_recipe"] == r and on_disk["train_structure"] == s


def test_the_structure_recorded_is_the_one_the_targets_were_taken_at(tmp_path):
    """The GUI leaves the element open to edits during a training: what the file records of
    it is read before the data, as its settings are."""
    el = _Cavity()

    def edit(info):
        if info.get("stage") == "epoch" and info["epoch"] == 1:
            el.n_steps, el.p_flag = 99, 0

    _, meta = _train(tmp_path / "CAV", element=el, progress_callback=edit)
    assert (el.n_steps, el.p_flag) == (99, 0)
    assert (meta.train_structure["n_steps"], meta.train_structure["p_flag"]) == (20, 1)


def test_a_file_from_before_loads_without_them(tmp_path):
    _train(tmp_path / "CAV")
    path = tmp_path / "CAV" / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["train_recipe"], data["train_structure"]
    path.write_text(json.dumps(data), encoding="utf-8")
    _, loaded = load_surrogate(tmp_path / "CAV")
    assert loaded.train_recipe is None and loaded.train_structure is None
    assert loaded.to_json()["train_recipe"] is None


def test_structure_record_names_the_field_file_by_its_name():
    class _E:
        pass

    e = _E()
    e.length, e.field_file, e.geom = 250.0000000001, "/a/b/Fields/HWRDonut", 7700
    e.aperture, e.n_steps, e.p_flag = 15.0, 200, 1
    assert structure_record(e) == {"class": "_E", "length": 250.0, "field_file": "HWRDonut",
                                   "geom": 7700, "aperture": 15.0, "n_steps": 200, "p_flag": 1}


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_commit_recorded_is_the_code_checkout_not_the_working_directory(
        tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    subprocess.run(["git", "-C", str(other), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "x"],
                   check=True)
    other_head = subprocess.check_output(
        ["git", "-C", str(other), "rev-parse", "HEAD"]).decode().strip()
    monkeypatch.chdir(other)
    sha = training._git_head_sha()
    assert sha != other_head
    pkg = Path(training.__file__).resolve().parent
    try:
        want = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pkg,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        want = "unknown"
    assert sha == want


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_batch_trainer_defaults_to_1000_epochs(qapp):
    from linac_gen_gui.interphase.tabs.surrogates_tab import _BatchTrainDialog
    _, shared = _BatchTrainDialog([]).get_selection()
    assert shared["epochs"] == 1000
    assert shared["n_samples"] == 5000 and shared["hidden_dims"] == (128, 128)


def test_the_progress_window_shows_the_kept_epoch(qapp):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from linac_gen_gui.interphase.tabs.surrogates_tab import _TrainProgressDialog
    dlg = _TrainProgressDialog(element_name="T", n_samples=10, epochs=3)
    for ep, kept in ((1, 1), (2, 1)):
        dlg.on_progress({"stage": "epoch", "epoch": ep, "total": 3, "lr": 1e-3,
                         "train_loss": 1.0 / ep, "val_mse": 1.0 / ep, "best_val_mse": 1.0,
                         "val_mape": 0.1 / ep, "best_val_mape": 0.1, "best_epoch": kept,
                         "per_entry_val_mape": np.full((6, 6), 0.05), "elapsed_s": 0.1})
    assert "kept epoch" in [line.get_label() for line in dlg._ax_mape.get_lines()]
    assert "kept epoch 1:" in dlg._status.text()
    dlg.close()


def _epoch_event(e, total=1000):
    return {"stage": "epoch", "epoch": e, "total": total, "lr": 1e-3, "train_loss": 1.0,
            "val_mse": 1.0, "best_val_mse": 1.0, "val_mape": 0.1, "best_val_mape": 0.1,
            "best_epoch": 1, "per_entry_val_mape": np.full((6, 6), 0.05), "elapsed_s": 0.0}


def test_a_long_run_paints_four_times_a_second_and_its_last_epoch(qapp, monkeypatch):
    """A paint costs about 0.1 s: painting each of 1000 epochs made the training stage of the
    batch trainer 8x as long as the training.  A run of at most 200 epochs still paints each
    (test_gui_tab.test_progress_dialog_throttles_datagen_paints)."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from linac_gen_gui.interphase.tabs import surrogates_tab as st
    clock = {"t": 1000.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    dlg = st._TrainProgressDialog(element_name="T", n_samples=10, epochs=1000)
    draws = []
    dlg._canvas.draw = lambda *a, **k: draws.append(clock["t"])
    dlg._canvas.repaint = lambda *a, **k: None
    for e in range(1, 1001):
        clock["t"] += 0.01                  # 10 ms an epoch, 10 s in all
        dlg.on_progress(_epoch_event(e))
    assert draws[0] == pytest.approx(1000.01) and draws[-1] == clock["t"]   # first and last
    gaps = np.diff(draws[:-1])
    assert (gaps >= st.EPOCH_PAINT_INTERVAL_S - 1e-6).all()
    assert len(draws) <= 2 + 10.0 / st.EPOCH_PAINT_INTERVAL_S
    assert len(dlg._epoch_xs) == 1000      # every epoch is in the curves
    dlg.close()


def test_a_progress_event_inside_a_paint_is_not_painted_inside_it(qapp, monkeypatch):
    """The batch trainer's queued connection delivers the events waiting while a paint drains
    the event queue: each nested one more slot call, until the stack ran out.  Now an event
    that arrives inside a paint is recorded there, and the last epoch painted after it."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from linac_gen_gui.interphase.tabs import surrogates_tab as st
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    dlg = st._TrainProgressDialog(element_name="T", n_samples=10, epochs=1000)
    waiting = [_epoch_event(e) for e in (2, 3, 4, 5, 1000)]
    depth = {"now": 0, "max": 0}
    draws = []

    def draw(*a, **k):
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        draws.append(len(dlg._epoch_xs))
        while waiting:                      # what processEvents would deliver now
            dlg.on_progress(waiting.pop(0))
        depth["now"] -= 1

    dlg._canvas.draw = draw
    dlg._canvas.repaint = lambda *a, **k: None
    dlg.on_progress(_epoch_event(1))        # the stage change paints; the rest arrive inside
    assert depth["max"] == 1
    assert draws == [1, 6]                  # the last epoch painted once the first paint ended
    assert dlg._epoch_xs == [1, 2, 3, 4, 5, 1000]
    dlg.close()
