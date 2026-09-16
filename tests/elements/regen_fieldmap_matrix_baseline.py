"""Regenerate the field-map Jacobian baseline used by
``test_fieldmap_matrix_baseline.py::test_fieldmap_matrix_baseline_bit_identical``.

Until 2026-09-08 no test compared a field-map transfer matrix against a
*previous tree*: every tight guard in the suite compares two runs of the
same code, so a bit-level change to ``fitted_matrix`` /
``fitted_matrix_slice`` would pass.  This fixture pins the matrices of
synthetic ``FieldMap`` (1-D geometry 1, static and RF, and a 2-D geometry-4
map), ``FieldMap3D`` (static-B and RF box maps, both integrators, fused and
scipy samplers) and ``SuperposedFieldMap`` (one- and two-child clusters —
no example deck carries a live SUPERPOSE_MAP, so synthetic is the only way
to pin it) elements, at two reference energies, for the full-element
Jacobian, three slice lengths and a two-slice chain, together with the
element walk state each call leaves behind.  Nothing here needs the
untracked ``Fields/`` data.

Run it from the tree whose behaviour is the reference — a worktree at the
commit BEFORE a change — and commit the fixture together with the change::

    PYTHONPATH=/path/to/reference/tree:/path/to/reference/tree/gui \\
        python3 tests/elements/regen_fieldmap_matrix_baseline.py

Writes ``tests/elements/fixtures/fieldmap_matrix_baseline.npz``.  The test
compares with ``np.array_equal`` on the architecture the fixture was
generated on (recorded under the ``__machine__`` key) and at
``rtol=1e-14`` elsewhere.
"""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np

from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements import field_map_3d as fm3d_mod
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.field_map_reader import Channel, FieldChannel, FieldMapData

FREQ = 352.21
REFS = {
    "p3": dict(species=PROTON, w_kin=3.0, frequency=FREQ),
    "h20": dict(species=H_MINUS, w_kin=20.0, frequency=FREQ),
}


# --- synthetic field data ---------------------------------------------------

def _rf_1d(L=100.0, nz=101, ez=1.5):
    z = np.linspace(0.0, L, nz)
    return FieldMapData(z=z, Ez=ez * np.sin(np.pi * z / L), symmetry="1d")


def _sol_1d(L=300.0, nz=151, bz=0.4):
    z = np.linspace(0.0, L, nz)
    return FieldMapData(z=z, Bz=bz * (0.5 - 0.5 * np.cos(2 * np.pi * z / L)),
                        symmetry="1d")


def _rf_2d_cyl(L=100.0, nz=51, nr=11):
    z = np.linspace(0.0, L, nz)
    r = np.linspace(0.0, 20.0, nr)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    Ez = 1.2 * np.sin(np.pi * zz / L) * (1.0 + 0.01 * rr ** 2)
    Fr = -0.6 * np.cos(np.pi * zz / L) * rr / L
    fd = FieldMapData(z=z, frequency=FREQ)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=4, z=z, r=r, Fz=Ez, Fr=Fr)
    return fd


def _box_3d(kind, value, L=200.0, f_MHz=0.0):
    n, nz = 3, 11
    x = np.linspace(-10.0, 10.0, n)
    z = np.linspace(0.0, L, nz)
    fd = FieldMapData(z=z, frequency=f_MHz)
    zeros = np.zeros((n, n, nz))
    fz = np.full((n, n, nz), value) * np.sin(np.pi * z / L)[None, None, :]
    fd.channels[kind] = FieldChannel(geometry=7, x=x, y=x.copy(), z=z,
                                     Fx=zeros, Fy=zeros.copy(), Fz=fz)
    return fd


def elements() -> dict:
    """Fresh element per key (state is reset per measurement anyway)."""
    return {
        "fm1d_rf":    lambda: FieldMap("R", length=100.0, field_data=_rf_1d(),
                                       phase=-25.0, frequency=FREQ, n_steps=50),
        "fm1d_rf_p1": lambda: FieldMap("R", length=100.0, field_data=_rf_1d(),
                                       phase=-25.0, frequency=FREQ, n_steps=50, p_flag=1),
        "fm1d_sol":   lambda: FieldMap("S", length=300.0, field_data=_sol_1d(), n_steps=30),
        "fm1d_g4":    lambda: FieldMap("G4", length=100.0, field_data=_rf_2d_cyl(),
                                       phase=-20.0, frequency=FREQ, n_steps=25),
        "fm3d_rf":    lambda: FieldMap3D("C", length=200.0,
                                         field_data=_box_3d(Channel.RF_E, 2.0, f_MHz=FREQ),
                                         phase=-30.0, frequency=FREQ, n_steps=40),
        "fm3d_rf_p1": lambda: FieldMap3D("C", length=200.0,
                                         field_data=_box_3d(Channel.RF_E, 2.0, f_MHz=FREQ),
                                         phase=-30.0, frequency=FREQ, n_steps=40, p_flag=1),
        "fm3d_sol":   lambda: FieldMap3D("S", length=200.0,
                                         field_data=_box_3d(Channel.STAT_B, 0.5), n_steps=20),
        "sup1":       lambda: SuperposedFieldMap("SUP", [
            (0.0, FieldMap("c", length=100.0, field_data=_rf_1d(),
                           phase=-25.0, frequency=FREQ, n_steps=50))]),
        "sup2":       lambda: SuperposedFieldMap("SUP", [
            (0.0, FieldMap("c1", length=100.0, field_data=_rf_1d(),
                           phase=-30.0, frequency=FREQ, n_steps=50, p_flag=1)),
            (40.0, FieldMap("c2", length=120.0, field_data=_sol_1d(L=120.0), n_steps=60))]),
    }


VARIANTS_3D = (("kd", True), ("kd", False), ("dkd", True))   # (integrator, fused sampler)


def _state(el) -> np.ndarray:
    """Walk state a Jacobian call leaves behind (must match before/after)."""
    vals = [float(getattr(el, "_step_idx", 0))]
    if isinstance(el, SuperposedFieldMap):
        vals += [float(el._z_cursor), float(len(el._z_history))]
        for _z, c in el.children:
            so = getattr(c, "_sync_offset_deg", None)
            vals += [float("nan") if so is None else float(so),
                     float(getattr(c, "_phi_s_at_entrance", 0.0) or 0.0)]
    else:
        so = getattr(el, "_sync_offset_deg", None)
        vals += [float("nan") if so is None else float(so),
                 float(getattr(el, "_phi_s_at_entrance", 0.0) or 0.0)]
    return np.asarray(vals, dtype=float)


def _fresh(make):
    el = make()
    el.reset_run_state()
    return el


def _measure(out: dict, key: str, make, ref_kw: dict) -> None:
    ref = ReferenceParticle(**ref_kw)
    # full-element Jacobian
    el = _fresh(make)
    out[f"{key}|fitted_matrix"] = el.fitted_matrix(ref.copy())
    out[f"{key}|fitted_matrix|state"] = _state(el)
    # single slices of three lengths, each from a fresh element
    L = el.length
    native = L / max(el.n_steps, 1)
    for tag, ds in (("native", native), ("half", 0.5 * L), ("full", L)):
        el = _fresh(make)
        out[f"{key}|slice_{tag}"] = el.fitted_matrix_slice(ref.copy(), ds)
        out[f"{key}|slice_{tag}|state"] = _state(el)
    # a two-slice chain (second call starts at the advanced cursor)
    el = _fresh(make)
    r = ref.copy()
    out[f"{key}|chain_1"] = el.fitted_matrix_slice(r, native)
    out[f"{key}|chain_2"] = el.fitted_matrix_slice(r, native)
    out[f"{key}|chain|state"] = _state(el)
    if isinstance(el, SuperposedFieldMap):
        # the envelope passes the cursor explicitly to clusters
        el = _fresh(make)
        out[f"{key}|slice_zfrom"] = el.fitted_matrix_slice(ref.copy(), native, _z_from_mm=0.0)
        out[f"{key}|slice_zfrom|state"] = _state(el)


def matrices() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    saved_integ = FieldMap3D.integrator_kind
    saved_fused = fm3d_mod.fused_kernel_enabled()
    try:
        for ename, make in elements().items():
            for rn, ref_kw in REFS.items():
                if ename.startswith("fm3d"):
                    for integ, fused in VARIANTS_3D:
                        if fused and not fm3d_mod.kernel_available():
                            continue
                        FieldMap3D.integrator_kind = integ
                        fm3d_mod.use_fused_kernel(fused)
                        _measure(out, f"{ename}|{integ}|{'fused' if fused else 'scipy'}|{rn}",
                                 make, ref_kw)
                else:
                    _measure(out, f"{ename}|{rn}", make, ref_kw)
    finally:
        FieldMap3D.integrator_kind = saved_integ
        fm3d_mod.use_fused_kernel(saved_fused)
    return out


def main() -> None:
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "fieldmap_matrix_baseline.npz"
    arrs = matrices()
    arrs["__machine__"] = np.array(platform.machine())
    np.savez_compressed(path, **arrs)
    print(f"wrote {path}  ({len(arrs) - 1} arrays, generated on {platform.machine()})")


if __name__ == "__main__":
    main()
