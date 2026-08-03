"""Parameter Study Manager — headless multi-parameter study engine.

See :mod:`linac_gen.study.spec` for the study.json schema,
:mod:`linac_gen.study.strategies` for run expansion, and
:mod:`linac_gen.study.engine` for folder-backed execution with resume.
CLI: ``python -m linac_gen study plan|run|resume|summarize``.
"""
from linac_gen.study.spec import (ObservableSpec, ParamSpec, StudySpec,
                                  load_spec, save_spec)
from linac_gen.study.strategies import RunSpec, expand_runs

__all__ = ["ObservableSpec", "ParamSpec", "StudySpec", "RunSpec",
           "expand_runs", "load_spec", "save_spec", "StudyManager",
           "StudyProgress"]


def __getattr__(name):
    # StudyManager pulls in cli.common/scan_pool lazily — keep the
    # package import light for spec-only consumers (GUI run-count label)
    if name in ("StudyManager", "StudyProgress"):
        from linac_gen.study import engine
        return getattr(engine, name)
    raise AttributeError(name)
