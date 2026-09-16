"""Comparison metrics on synthetic measurements built from the model."""
from __future__ import annotations
import numpy as np
import pytest

from linac_gen.orm import DeviceMap, resolve_devices
from linac_gen.orm.compare import column_normalize, compare_orm, sigma_eff, summary_lines
from linac_gen.orm.measured import MeasuredOrm
from linac_gen.orm.model import OrmModel
from tests.orm.conftest import fodo_with_devices
from tests.orm.test_model import _cfg


def _synthetic(k_true, noise=0.0, seed=0, flip_trim=None, extra_bpm=None):
    lat = fodo_with_devices(3)
    trims = [f"D{k:02d}T" for k in (1, 2, 3)]; bpms = [f"D{k:02d}{m}BPM" for k in (1, 2, 3) for m in (1, 2)]
    devs_b = [f"L:{b[:-3]}BPH" for b in bpms] + ([extra_bpm] if extra_bpm else [])
    devs_t = [f"L:{t}MH" for t in trims]
    rng = np.random.default_rng(seed)
    dm = DeviceMap(bpms={d: b for d, b in zip(devs_b, bpms)}, trims={d: t for d, t in zip(devs_t, trims)})
    stub = {"x": MeasuredOrm(kick_plane="x", trims=devs_t, bpms={"x": devs_b}, values={"x": np.zeros((len(devs_b), 3))}, errors={"x": np.zeros((len(devs_b), 3))})}
    sel = resolve_devices(lat, stub, dm); model = OrmModel(lat, _cfg(), sel); R = model.response()
    A = R.per_Tm[("x", "x")] * np.asarray(k_true)[None, :]
    if flip_trim is not None:
        A[:, flip_trim] *= -1
    A = A + rng.normal(0, noise, A.shape)
    A[~sel.down] = rng.normal(0, noise, A[~sel.down].shape)            # upstream entries = noise only
    if extra_bpm:
        A = np.vstack([A, rng.normal(0, noise, (1, 3))])
    E = np.full_like(A, max(noise, 1e-4))
    meas = {"x": MeasuredOrm(kick_plane="x", trims=devs_t, bpms={"x": devs_b}, values={"x": A}, errors={"x": E})}
    sel = resolve_devices(lat, meas, dm)
    return meas, sel, model, R


def test_noise_free_calibration_is_exact():
    k_true = np.array([-3e-4, 5e-4, -4e-4])
    meas, sel, model, R = _synthetic(k_true)
    cmp = compare_orm(meas, R, sel, model.brho_trim, model.w_trim_MeV)
    rows = cmp["planes"]["x"]["per_trim"]
    np.testing.assert_allclose([r["k_Tm_per_A"] for r in rows[:2]], k_true[:2], rtol=1e-10)
    assert rows[0]["r"] == pytest.approx(-1.0) and rows[0]["nrms_resid"] < 1e-9 and rows[0]["n_down"] == 6   # r is signed: k < 0
    assert np.isnan(rows[2]["r"])                                # only 2 BPMs downstream of D03T → r undefined
    assert cmp["reason"] is None and cmp["planes"]["x"]["polarity_flipped"] == ["D02T"]   # k_true[1] has the opposite sign
    assert rows[0]["kick_mrad_per_A"] == pytest.approx(k_true[0] / model.brho_trim[0] * 1e3)


def test_noise_floor_and_polarity_flip():
    k_true = np.array([-3e-4, -5e-4, -4e-4])
    meas, sel, model, R = _synthetic(k_true, noise=0.02, seed=3, flip_trim=1, extra_bpm="L:BPH2OT")
    cmp = compare_orm(meas, R, sel, model.brho_trim)
    d = cmp["planes"]["x"]
    assert d["polarity_flipped"] == ["D02T"]
    assert 0.005 < d["per_trim"][1]["noise_upstream_mm_per_A"] < 0.06        # rms of N(0, 0.02) over 2 upstream rows
    assert 0.005 < d["tank_noise_mm_per_A"] < 0.06                          # the unmatched tank BPM row
    assert abs(d["per_trim"][0]["r"]) > 0.9
    assert any("polarity flipped: D02T" in ln for ln in summary_lines(cmp))


def test_empty_plane_refused():
    meas, sel, model, R = _synthetic(np.ones(3))
    meas["x"].values["x"][:] = np.nan
    cmp = compare_orm(meas, R, sel, model.brho_trim)
    assert cmp["reason"] and "no downstream measured entries" in cmp["reason"]


def test_column_normalize_and_sigma_eff():
    A = np.array([[1.0, np.nan], [2.0, 4.0], [-4.0, 1.0]]); down = np.array([[False, False], [True, True], [True, True]])
    N = column_normalize(A, down)
    np.testing.assert_allclose(N[:, 0], [0.25, 0.5, -1.0]); np.testing.assert_allclose(N[1:, 1], [1.0, 0.25])
    S = sigma_eff(A, np.full_like(A, 0.1), down, 0.05)
    assert S[1, 0] == pytest.approx(np.sqrt(0.1 ** 2 + (0.05 * 4.0) ** 2))
    S2 = sigma_eff(A, np.full_like(A, np.nan), down, 0.05)                  # no measured errors → floor only
    assert S2[1, 0] == pytest.approx(0.2)
