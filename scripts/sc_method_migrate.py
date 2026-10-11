"""Show or change the space-charge method a HELIX project stores.

A project's space-charge settings live in its ``convergence`` block.  A
project without ``sc_method`` keeps resolving exactly as it always did; this
command can name its method explicitly, switch it to another method, or
undo a switch.  Nothing is written without ``--write``.

  python scripts/sc_method_migrate.py PROJECT.lgproj [...]
        report: the method each project resolves to and its settings
  python scripts/sc_method_migrate.py --pin PROJECT.lgproj [...] [--write]
        store every setting explicitly under the project's own method: its
        sc_method, else the method its settings match, else "custom" when
        it stores no space-charge setting (such a project follows the
        default method).  The configuration is unchanged, also when a
        command line names a method (checked for every method before
        writing), and later changes of the default method or of a method's
        defaults no longer apply.  A project that stores the torch engine
        without sc_method is refused: the GUI runs it with the PyTorch PIC,
        the command line with the numpy PIC, so a pin would change one.
  python scripts/sc_method_migrate.py --switch NAME PROJECT.lgproj [--write]
        switch to method NAME with its defaults, keeping the settings that
        belong to no method (DC kernel, CSR, FFT device, ...); the previous
        space-charge keys are kept under "sc_previous" (a GUI save keeps
        them as well)
  python scripts/sc_method_migrate.py --restore PROJECT.lgproj [--write]
        bring back the keys saved by --switch

Run from the repository root (``PYTHONPATH=.``).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from linac_gen.pic import methods

#: convergence-block keys that hold space-charge settings, as HELIX writes
#: them (grid_nx/grid_extent_sigma/backend are the project names of the
#: nx=ny=nz / grid_extent / use_gpu fields)
_ALIASES = {"grid_nx": "nx", "grid_extent_sigma": "grid_extent",
            "backend": "use_gpu"}


def _sc_keys(conv: dict) -> dict:
    """The space-charge entries of a convergence block."""
    fields = {f.name for f in dataclasses.fields(_config_class())}
    return {k: v for k, v in conv.items()
            if k in fields or k in _ALIASES or k == "sc_method"}


def _config_class():
    from linac_gen.core.config import SpaceChargeConfig
    return SpaceChargeConfig


def _fields(cfg) -> dict:
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
            if f.name != "method"}


#: written for every pinned project, as the GUI writes them
_ALWAYS = ("sc_backend", "green_kind", "kernel", "grid_mode", "dc_kernel",
           "csr_enabled", "grid_card", "green_build", "reuse_every")


def explicit_block(cfg, name: str) -> dict:
    """Convergence entries that store ``cfg`` completely under method
    ``name``: the keys HELIX writes, and any other field whose value is not
    the dataclass default."""
    if cfg.ny != cfg.nx or cfg.nz != cfg.nx:
        raise ValueError("grid sizes differ per axis; a project stores one "
                         "grid size (grid_nx)")
    defaults = {f.name: f.default for f in dataclasses.fields(cfg)}
    out = {"sc_method": name, "grid_nx": int(cfg.nx),
           "grid_extent_sigma": float(cfg.grid_extent),
           "backend": str(cfg.use_gpu)}
    for key, val in _fields(cfg).items():
        if key in ("nx", "ny", "nz", "grid_extent", "use_gpu") or val is None:
            continue
        if key in _ALWAYS or val != defaults.get(key):
            out[key] = val
    return out


def _general_block(cfg) -> dict:
    """The settings of ``cfg`` that belong to no method."""
    return {k: v for k, v in explicit_block(cfg, "").items()
            if k not in methods.METHOD_FIELDS and k not in
            ("sc_method", "grid_nx", "grid_extent_sigma")}


def _merged(conv: dict, block: dict) -> dict:
    """``conv`` with its space-charge entries replaced by ``block``: an
    entry both hold keeps its place, an entry ``block`` lacks is removed,
    new entries go at the end."""
    sc = _sc_keys(conv)
    out = {}
    for key, val in conv.items():
        if key not in sc:
            out[key] = val
        elif key in block:
            out[key] = block[key]
    for key, val in block.items():
        out.setdefault(key, val)
    return out


def resolve(conv: dict):
    return methods.resolve_sc(conv, {})


def _outcome(conv: dict, cli: dict):
    try:
        return _fields(methods.resolve_sc(conv, cli))
    except ValueError as exc:
        return f"error: {exc}"


def _check_cases(conv: dict) -> list:
    """Command lines a pin must not change: none, and every method named
    (``default`` too, unless the project follows the default: pinning
    stops that), each also with grid-size and extent flags."""
    names = list(methods.names())
    if conv.get("sc_method") != methods.DEFAULT_ALIAS:
        names.append(methods.DEFAULT_ALIAS)
    cases = [{}] + [{"sc_method": n} for n in names]
    return cases + [dict(c, nx=40, grid_extent=4.5) for c in cases]


def pin(conv: dict) -> dict:
    """``conv`` with its space-charge settings stored explicitly under the
    project's own method; the configuration is unchanged, also when a
    command line names a method."""
    if ("sc_method" not in conv
            and str(conv.get("sc_backend", "numpy")) == "torch"):
        raise ValueError(
            "the project stores sc_backend 'torch' without sc_method: the "
            "GUI runs it with the PyTorch PIC, the command line with the "
            "numpy PIC, and a pin would change one of them; choose with "
            "--switch differentiable_pic or --switch NAME")
    before = resolve(conv)
    # A project that stores no method setting follows the default method;
    # "custom" keeps a command line that names a method starting from that
    # method's own defaults, as before.
    name = methods.own_method(conv) or methods.CUSTOM
    new = _merged(conv, explicit_block(before, name))
    changed = [c for c in _check_cases(conv)
               if _outcome(conv, c) != _outcome(new, c)]
    if changed:
        flags = " ".join(f"--{k.replace('_', '-')} {v}"
                         for k, v in changed[0].items())
        raise RuntimeError(f"pinning would change the configuration "
                           f"({flags or 'no command-line options'}); "
                           "not written")
    return new


def switch(conv: dict, name: str, *, had_block: bool = True) -> dict:
    """``conv`` switched to method ``name`` with its defaults; the settings
    that belong to no method are kept, the old space-charge entries go to
    ``sc_previous`` (null when the project had no convergence block)."""
    methods.check_name(name)
    if "sc_previous" in conv:
        raise ValueError("the project already holds a switched-away block "
                         "(sc_previous); --restore it first")
    general = {k: v for k, v in _fields(resolve(conv)).items()
               if k not in methods.METHOD_FIELDS}
    cfg = _config_class()(method=name, **general)
    if name == methods.DEFAULT_ALIAS:
        # follow the default method, whatever it is now or later
        block = {"sc_method": name, **_general_block(cfg)}
    else:
        block = explicit_block(cfg, name)
    new = _merged(conv, block)
    new["sc_previous"] = _sc_keys(conv) if had_block else None
    return new


def restore(conv: dict):
    """``conv`` with the entries saved by :func:`switch` back, or ``None``
    when the project had no convergence block before the switch and the
    block now holds nothing else."""
    if "sc_previous" not in conv:
        raise ValueError("nothing to restore: the project has no "
                         "sc_previous block")
    prev = conv["sc_previous"]
    if prev is not None and not isinstance(prev, dict):
        raise ValueError("its sc_previous entry is not a JSON object")
    new = _merged({k: v for k, v in conv.items() if k != "sc_previous"},
                  prev or {})
    return None if prev is None and not new else new


def describe(conv: dict | None) -> str:
    conv = conv or {}
    cfg = resolve(conv)
    name = methods.effective_method(cfg)
    stored = conv.get("sc_method")
    how = (f"stored sc_method {stored!r}" if stored is not None else
           "no sc_method; resolved as before named methods")
    return (f"{name} ({how}): {cfg.nx}^3 grid, +-{cfg.grid_extent:g} sigma, "
            f"{cfg.grid_mode} grid, {cfg.kernel.upper()}, "
            f"{cfg.green_kind} Green's function, engine {cfg.sc_backend}")


def _load(path: Path):
    """``(text, project, convergence block or None)``; ValueError when the
    file is not a project."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("not a HELIX project (not a JSON object)")
    conv = data.get("convergence")
    if conv is not None and not isinstance(conv, dict):
        raise ValueError("its convergence entry is not a JSON object")
    return text, data, conv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("projects", nargs="+", type=Path)
    act = p.add_mutually_exclusive_group()
    act.add_argument("--pin", action="store_true",
                     help="store the method and every setting explicitly")
    act.add_argument("--switch", metavar="NAME",
                     help="switch to method NAME (old keys kept)")
    act.add_argument("--restore", action="store_true",
                     help="undo --switch")
    p.add_argument("--write", action="store_true",
                   help="write the files (default: show what would change)")
    args = p.parse_args(argv)
    rc = 0
    for path in args.projects:
        try:
            text, data, block = _load(path)
            conv = dict(block or {})
            print(f"{path}: {describe(conv)}")
            if not (args.pin or args.switch or args.restore):
                continue
            if args.pin:
                new = pin(conv)
            elif args.switch:
                new = switch(conv, args.switch, had_block=block is not None)
            else:
                new = restore(conv)
            shown = new or {}
            removed = sorted(set(conv) - set(shown))
            added = {k: v for k, v in shown.items()
                     if k not in conv or conv[k] != v}
            print(f"  -> {describe(shown)}")
            for k in removed:
                print(f"     - {k}: {conv[k]!r}")
            for k, v in added.items():
                print(f"     + {k}: {v!r}")
            if new is None:
                print("     (convergence block removed: the project had "
                      "none before --switch)")
            if args.write:
                if new is None:
                    data.pop("convergence", None)
                else:
                    data["convergence"] = new
                path.write_text(json.dumps(data, indent=2)
                                + ("\n" if text.endswith("\n") else ""),
                                encoding="utf-8")
                print("  written")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"{path}: error: {exc}", file=sys.stderr)
            rc = 2
    if not args.write and (args.pin or args.switch or args.restore):
        print("(nothing written; add --write)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
