"""Weights for the accuracy-check tests: pinned maps, and a network that answers with the native
map itself, exact over any range.  (Not a test module: the tests import it.)"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import torch

from linac_gen.core.particle import PROTON
from linac_gen.surrogates.base import MlpHead, Scope, SurrogateFieldMap, SurrogateMetadata
from linac_gen.surrogates.training import _at_element_frequency, load_surrogate, save_surrogate
from tests.surrogates.test_frequency_jump import F_NEW, _ref

W = 5.0
INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2,
            alpha_z=0.0, beta_z=1.0, emit_z=0.2)


def native(el, w=W, f=F_NEW, phi_s=0.0):
    ref = _ref(f, w)
    ref.phi_s = phi_s
    return np.asarray(copy.deepcopy(el).fitted_matrix(ref), dtype=np.float64)


def meta(el, M=None, *, w=(W - 1e-9, W + 1e-9), params=(), train_frequency=F_NEW,
         train_phi=0.0):
    """Weights trained for protons on ``train_frequency`` at reference phase ``train_phi``, over
    energies ``w`` and, for each (name, lo, hi) or (name, value) of ``params``, that range (a
    value alone: a range that holds it alone).  ``M`` pins the output (spread 0); None leaves the
    output to the network as it is (mean 0, spread 1)."""
    names, lo, hi = ["w_kin", "beta", "gamma"], [w[0], 0.0, 1.0], [w[1], 1.0, 100.0]
    for spec in params:
        n, a, b = (spec[0], spec[1] - 1e-9, spec[1] + 1e-9) if len(spec) == 2 else spec
        names.append(n)
        lo.append(a)
        hi.append(b)
    out = ({"mean": [float(v) for v in np.asarray(M).ravel()], "std": [0.0] * 36} if M is not None
           else {"mean": [0.0] * 36, "std": [1.0] * 36})
    return SurrogateMetadata(
        element_key=el.name, element_class=type(el).__name__,
        architecture={"input_dim": len(names), "output_dim": 36, "hidden_dims": [4],
                      "activation": "silu", "param_names": names[3:]},
        scope=Scope(input_names=names, input_lo=np.array(lo), input_hi=np.array(hi)),
        input_norm={"mean": [0.0] * len(names), "std": [1.0] * len(names)},
        output_norm=out, training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="lh", created_iso="", species_name=PROTON.name,
        species_mass_mev=PROTON.mass, train_frequency_mhz=train_frequency,
        train_phi_s_deg=train_phi)


def mlp(n_in=3):
    return MlpHead(input_dim=n_in, output_dim=36, hidden_dims=(4,))


def from_file(tmp_path, el, M, name="w", **kw):
    """A surrogate of ``el`` whose pinned weights were saved and loaded back."""
    d = tmp_path / name
    m = meta(el, M, **kw)
    save_surrogate(mlp(m.architecture["input_dim"]), m, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp_, m = load_surrogate(d)
    return SurrogateFieldMap(el, mlp_, m)


def pinned(el, M, **kw):
    """Pinned weights built in memory (not checked)."""
    m = meta(el, M, **kw)
    return SurrogateFieldMap(el, mlp(m.architecture["input_dim"]), m)


def off(M, i=0, j=0, factor=1.01):
    out = np.array(M, dtype=np.float64)
    out[i, j] *= factor
    return out


class NativeNet(torch.nn.Module):
    """A network that answers each input with the native map there, taken as training takes
    its targets (a design copy, the inputs set, the run state reset, the training phase, the
    map's clock): exact over any range.  ``blind`` names inputs it ignores (it keeps their
    design value), as a network that did not learn them; ``from_gamma`` reads the energy from
    its γ input instead of its energy input."""

    def __init__(self, el, template, names, train_phi=0.0, blind=(), from_gamma=False):
        super().__init__()
        self._el = copy.deepcopy(el)
        for knob in ("voltage_rel", "phase_offset", "frequency_offset"):
            if hasattr(self._el, knob):
                setattr(self._el, knob, 0.0)
        self._design = {n: float(getattr(self._el, n)) for n in names}
        self._template, self._names = template, list(names)
        self._train_phi, self._blind = float(train_phi), set(blind)
        self._from_gamma = bool(from_gamma)

    def forward(self, x):
        rows = x.detach().numpy().reshape(-1, x.shape[-1])
        out = []
        for row in rows:
            for j, n in enumerate(self._names):
                setattr(self._el, n, self._design[n] if n in self._blind else float(row[3 + j]))
            self._el.reset_run_state()
            ref = self._template.copy()
            ref.w_kin = (float(row[2]) - 1.0) * ref.species.mass if self._from_gamma else float(row[0])
            ref.phi_s = self._train_phi
            _at_element_frequency(self._el, ref)
            out.append(np.asarray(self._el.fitted_matrix(ref), dtype=np.float64).reshape(36))
        return torch.as_tensor(np.array(out), dtype=torch.float64).reshape(*x.shape[:-1], 36)


def exact(el, *, w=(4.0, 6.0), params=(), train_phi=0.0, blind=(), template=None,
          from_gamma=False):
    """Weights 'from a file' (checked) whose network is the native map: ``params`` as (name, lo,
    hi) ranges of the element's inputs."""
    m = meta(el, None, w=w, params=params, train_phi=train_phi)
    m.accuracy_due = True
    net = NativeNet(el, template if template is not None else _ref(F_NEW, W),
                    m.architecture["param_names"], train_phi=train_phi, blind=blind,
                    from_gamma=from_gamma)
    return SurrogateFieldMap(el, net, m)
