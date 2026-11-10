"""The registered surrogates of a study, carried to the processes of the scan pool.

A worker process starts with an empty registry, so a study run through the pool would track
every point natively while the same study run serially used the registered weights.  A study
that uses them (its design check passed, ``study_check.check_design``) gives each point a
``SurrogateSpec``: per registered element (the weights tracking serves for its name) the folder
its weights were loaded from, the hash of the weights the parent loaded, the parent's metadata
as it stands and the structure the weights were built on.  The metadata holds what the parent
decided: the accuracy verdict (``SurrogateFieldMap._accuracy_refusal``), the phase check of a
file with no training phase, the settings the weights hold (recorded in training, or the
parent's snapshot); ``study_spec`` runs in the parent every such check still due.  Weights their
accuracy check refused are carried too, so a worker refuses them, and counts them, as the
parent does.  A worker reads only the weights (``install``), once per process, checks them
against that hash, and builds them on its own parse of the lattice before the point's
overrides with the parent's metadata (an element whose structure is not the parent's, or
weights that cannot be read or changed since, run native, with a warning): nothing is decided
again in a worker, so a point runs alike serially and in parallel.
"""
from __future__ import annotations

import hashlib
import io
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SpecEntry", "SurrogateSpec", "install", "load_folders", "spec_of_registry",
           "study_spec"]


@dataclass(frozen=True)
class SpecEntry:
    """One registered element's weights, as a worker rebuilds them: ``metadata`` is the
    parent's ``SurrogateMetadata``, pickled (``meta``)."""
    element_key: str
    weights_dir: str                 # absolute
    weights_sha256: str              # of the weights the parent loaded
    lattice_hash: str
    metadata: bytes
    structure: tuple

    @property
    def meta(self):
        """A copy of the parent's metadata."""
        return pickle.loads(self.metadata)


@dataclass(frozen=True)
class SurrogateSpec:
    """The weights of a study (``spec_of_registry``), hashable and picklable: a ``ScanPoint``
    carries it to the pool's workers."""
    entries: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.entries)


def _served() -> list:
    """``(lattice_hash, name, surrogate)`` of the weights tracking serves for each registered
    name: its last registration (``registry.get_by_element_name``)."""
    from linac_gen.surrogates import registry
    out, seen = [], set()
    for lattice_hash, key in registry.list_registered():
        surr = registry.get(lattice_hash, key)
        if key in seen or surr is None or surr is not registry.get_by_element_name(key):
            continue
        seen.add(key)
        out.append((lattice_hash, key, surr))
    return out


def spec_of_registry() -> tuple[SurrogateSpec, list[str]]:
    """The registered weights a worker can rebuild — those loaded from a folder, with the
    parent's metadata (an accuracy check's refusal included: a worker refuses them as the
    parent does) — and a note for each one it cannot carry (weights built in memory, with no
    folder: the workers run it native)."""
    entries, notes = [], []
    for lattice_hash, key, surr in _served():
        meta = surr.metadata
        folder = getattr(surr, "weights_dir", None)
        if not folder or not (Path(folder) / "weights.pt").is_file():
            notes.append(f"{key}: its weights are not in a folder, so the pool's workers run it "
                         "native")
            continue
        folder = Path(folder).resolve()
        # the hash of the bytes the load read (load_surrogate); a set built in memory and
        # saved has none, and takes the file's
        sha = getattr(meta, "weights_sha256", "") or hashlib.sha256(
            (folder / "weights.pt").read_bytes()).hexdigest()
        entries.append(SpecEntry(element_key=key, weights_dir=str(folder), weights_sha256=sha,
                                 lattice_hash=str(lattice_hash), metadata=pickle.dumps(meta),
                                 structure=tuple(surr._structure)))
    return SurrogateSpec(tuple(entries)), notes


#: in a worker process: (folder, weights hash, metadata) -> (mlp, metadata), read once; (None,
#: why) for weights that cannot be read or are not the parent's
_LOADED: dict = {}


def _read(entry: SpecEntry):
    """The entry's network, read from its folder and checked against the parent's hash, and
    its metadata; or ``(None, why)``."""
    import torch

    from linac_gen.surrogates.base import MlpHead
    path = Path(entry.weights_dir) / "weights.pt"
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, f"its weights in {path.parent} could not be read ({exc})"
    if hashlib.sha256(data).hexdigest() != entry.weights_sha256:
        return None, f"the weights in {path.parent} changed since the study loaded them"
    try:
        meta = pickle.loads(entry.metadata)
        arch = meta.architecture
        mlp = MlpHead(input_dim=int(arch["input_dim"]), output_dim=int(arch["output_dim"]),
                      hidden_dims=arch["hidden_dims"], activation=str(arch["activation"]))
        mlp.load_state_dict(torch.load(io.BytesIO(data), weights_only=True))
        mlp.eval()
    except Exception as exc:                              # noqa: BLE001 - a torn file
        return None, (f"its weights in {path.parent} could not be read "
                      f"({type(exc).__name__}: {exc})")
    return mlp, meta


def install(spec: SurrogateSpec, lattice) -> list:
    """In a worker process, before a point's overrides: the registry holds the weights of
    ``spec``, each built on its element of ``lattice`` (this point's parse) with the parent's
    metadata, and nothing else.  Returns the registered surrogates.  Weights that cannot be
    read, or changed since the parent loaded them, and an element whose structure is not the
    one the parent built them on, are left out with a warning: that element runs native."""
    from linac_gen.surrogates import registry
    from linac_gen.surrogates.base import SurrogateFieldMap, structure_of
    registry.clear()
    by_name = {}
    for el in getattr(lattice, "elements", ()):
        by_name.setdefault(getattr(el, "name", None), el)
    out = []
    for entry in spec.entries:
        element = by_name.get(entry.element_key)
        if element is None:
            continue
        if structure_of(element) != tuple(entry.structure):
            warnings.warn(f"surrogate '{entry.element_key}': this worker's element is not the "
                          "one the study built its weights on; it runs native", stacklevel=2)
            continue
        key = (entry.weights_dir, entry.weights_sha256, entry.metadata)
        loaded = _LOADED.get(key)
        if loaded is None:
            loaded = _LOADED[key] = _read(entry)
            if loaded[0] is None:
                warnings.warn(f"surrogate '{entry.element_key}': {loaded[1]}; this worker runs "
                              "it native", stacklevel=2)
        mlp, meta = loaded
        if mlp is None:
            continue
        surr = SurrogateFieldMap(element, mlp, meta)
        surr.weights_dir = entry.weights_dir
        registry.register(surr, entry.lattice_hash, element_key=entry.element_key)
        out.append(surr)
    return out


def _settle(beam_config) -> None:
    """Run here every one-time check still due on the weights tracking serves (the accuracy
    check, the phase check of a file that records no training phase), at the point of their
    training range nearest the element as it stands: weights the design lattice did not ask
    (out of range there, a setting changed) would otherwise be checked by each worker at the
    first point it meets, and the rows would depend on which point that was."""
    import numpy as np

    from linac_gen.cli.common import build_ref
    from linac_gen.surrogates.base import _CHECK_LOCK
    for _lh, _key, surr in _served():
        meta = surr.metadata
        accuracy = bool(meta.accuracy_due) and meta.accuracy_check is None
        phase = not meta.phase_recorded and meta.phase_check is None
        if not (accuracy or phase):
            continue
        scope = meta.scope
        ref = build_ref(beam_config)
        ref.w_kin = 0.5 * (float(scope.input_lo[0]) + float(scope.input_hi[0]))
        own = surr._on_own_clock(ref)
        x = np.clip(np.asarray(surr._make_input(own), dtype=float),
                    np.asarray(scope.input_lo, dtype=float), np.asarray(scope.input_hi, dtype=float))
        if phase:
            surr._unrecorded_phase_refusal(own, x)
        if accuracy:
            with _CHECK_LOCK:
                if meta.accuracy_check is None:
                    surr._run_accuracy_check(own, x)


def study_spec(lattice, beam_config, lattice_path, folders, *, mode: str = "envelope",
               env_solver: str = "matrix"):
    """For a study run through the scan pool (the CLI's ``--surrogates``): the weights of
    ``folders`` registered on ``lattice`` (``load_folders``), the study's design check
    (``study_check.check_design``), and ``(check, spec, notes)``: when the check uses them,
    the spec of the weights, every one-time check settled first (``_settle``); an empty spec
    otherwise; and what was said of them."""
    from linac_gen.surrogates.study_check import DesignCheck, check_design
    notes = load_folders(lattice, lattice_path, folders,
                         species=getattr(beam_config, "species", None))
    check = check_design(lattice, beam_config, mode=mode, env_solver=env_solver)
    if not check.note:
        check = DesignCheck(use=False, note="surrogates not used: no weights were registered")
    spec = SurrogateSpec()
    if check.use:
        _settle(beam_config)
        spec, more = spec_of_registry()
        notes += more
    return check, spec, notes


def load_folders(lattice, lattice_path, folders, species: str | None = None) -> list[str]:
    """Register on ``lattice`` the weights of ``folders`` — each a set's folder (it holds
    ``weights.pt``) or a folder of sets named by element — and, for ``"auto"``, of the weights
    folder of ``lattice_path`` and the beam's ``species`` (``training.weights_folder``, from the
    working directory or the HELIX tree), each set on the element of its name.  Returns what it
    says about them (a folder that does not exist or holds no sets, no weights folder for
    ``auto``, a set whose element the lattice does not have)."""
    from linac_gen.surrogates import registry
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.surrogates.training import (
        discover_cached_surrogates,
        find_cached_surrogate,
        weights_folder,
    )
    lattice_hash = registry.hash_lattice_file(lattice_path)
    by_name = {}
    for el in lattice.elements:
        by_name.setdefault(getattr(el, "name", None), el)
    roots, notes = [], []
    for folder in folders:
        if folder == "auto":
            rel = weights_folder(lattice_hash, species)
            tree = Path(__file__).resolve().parents[2]
            found = next((r for r in dict.fromkeys((Path.cwd() / rel, tree / rel))
                          if r.is_dir()), None)
            if found is None:
                notes.append(f"auto: no weights folder for this lattice ({rel})")
            else:
                roots.append(found)
        elif not Path(folder).exists():
            notes.append(f"{folder}: does not exist")
        else:
            roots.append(Path(folder))
    found_sets = {}
    for root in roots:
        if (root / "weights.pt").is_file():
            loaded = find_cached_surrogate(root)
            if loaded is None:
                notes.append(f"{root}: its weights do not load")
                continue
            found_sets[loaded[1].element_key] = (loaded[0], loaded[1], root)
        else:
            sets = discover_cached_surrogates(root)
            if not sets:
                notes.append(f"{root}: no weights in it")
            found_sets.update(sets)
    for key, (mlp, meta, folder) in found_sets.items():
        element = by_name.get(key)
        if element is None:
            notes.append(f"{key}: not an element of this lattice")
            continue
        surr = SurrogateFieldMap(element, mlp, meta)
        surr.weights_dir = str(Path(folder).resolve())
        registry.register(surr, lattice_hash, element_key=key)
    return notes
