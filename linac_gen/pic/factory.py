"""One place that turns a :class:`SpaceChargeConfig` into its solver.

Every run path builds its bunched-beam space-charge solver here, so a config
always runs the solver it describes.  Before, the error study and the
``backtrack --validate`` replay constructed ``PicSolver`` directly and ran
the numpy PIC whatever engine the config selected.
"""
from __future__ import annotations


def make_sc_solver(config, *, purpose: str = "forward"):
    """Solver for ``config`` (``None`` for ``None``).

    ``purpose="forward"``: the solver the config describes -- the numpy/C++
    PIC (solving the field every ``reuse_every`` kicks when that is above
    1), the differentiable PyTorch PIC, or HALO-PIC with its corrector
    loaded when ``config.halo["corrector_dir"]`` is set.

    ``purpose="backward"``: the solver a backtrack walk rebuilds from a
    config: the plain numpy PIC, as before (the walk decides beforehand what
    to do with a config whose solver it cannot undo exactly); its config
    says numpy, so the results record the engine that ran.
    """
    if config is None:
        return None
    if purpose not in ("forward", "backward"):
        raise ValueError(
            f"purpose must be 'forward' or 'backward', got {purpose!r}")
    from linac_gen.pic.pic_solver import PicSolver
    if purpose == "backward":
        import dataclasses
        if (getattr(config, "sc_backend", "numpy") != "numpy"
                and dataclasses.is_dataclass(config)):
            from linac_gen.pic.gpu_backend import numpy_engine_use_gpu
            config = dataclasses.replace(
                config, sc_backend="numpy",
                use_gpu=numpy_engine_use_gpu(config.use_gpu))
        return PicSolver(config)
    backend = getattr(config, "sc_backend", "numpy")
    if backend == "torch":
        from linac_gen.pic.torch.solver import TorchPicSolver
        return TorchPicSolver(config)
    if backend == "halo":
        from linac_gen.pic.ml.solver import HaloPicSolver
        pic = HaloPicSolver(config)
        cdir = (getattr(config, "halo", None) or {}).get("corrector_dir")
        if cdir:
            pic.load(cdir)
        return pic
    if int(getattr(config, "reuse_every", 1) or 1) > 1:
        from linac_gen.pic.field_reuse import FieldReusePicSolver
        return FieldReusePicSolver(config)
    return PicSolver(config)
