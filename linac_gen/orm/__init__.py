"""Orbit-response-matrix (ORM) comparison and LOCO-style calibration.

Measured response matrices (FORMA export folders or HELIX CSV) are mapped onto
the lattice's BPM markers and steerers, compared with the model response built
from the envelope phase probe, and fitted: quadrupole gradient scale factors
(``Quadrupole.gradient_rel``), signed trim calibrations (T·m per ampere) and
optional BPM gains, Levenberg–Marquardt with Gaussian priors.  The result can
be applied to the lattice, saved as a calibration JSON, and exported as a
recalibrated deck (label-keyed text surgery on the original file).

Manual: ``docs/manual/08_errors/08_orm_calibration.md``.
"""
from linac_gen.orm.measured import (MeasuredOrm, read_forma_folder, read_orm_csv,
                                    write_orm_csv, load_measured)
from linac_gen.orm.devices import (DeviceMap, OrmSelection, default_device_map,
                                   resolve_devices, align_measured, element_label,
                                   FNAL_SCL_ALIASES)

__all__ = ["MeasuredOrm", "read_forma_folder", "read_orm_csv", "write_orm_csv", "load_measured",
           "DeviceMap", "OrmSelection", "default_device_map", "resolve_devices", "align_measured",
           "element_label", "FNAL_SCL_ALIASES"]
from linac_gen.orm.model import OrmModel, ModelOrm                               # noqa: E402
from linac_gen.orm.compare import compare_orm, summary_lines, column_normalize   # noqa: E402
__all__ += ["OrmModel", "ModelOrm", "compare_orm", "summary_lines", "column_normalize"]
from linac_gen.orm.fit import (OrmFitOptions, OrmFitResult, fit_orm, run_stages,     # noqa: E402
                               synthetic_validation, degeneracy_report)
__all__ += ["OrmFitOptions", "OrmFitResult", "fit_orm", "run_stages", "synthetic_validation", "degeneracy_report"]
from linac_gen.orm.calibration import (calibration_from_fit, save_calibration, load_calibration,   # noqa: E402
                                       validate_calibration, apply_calibration, apply_changes,
                                       revert_changes, export_recalibrated_deck, verify_exported_deck)
__all__ += ["calibration_from_fit", "save_calibration", "load_calibration", "validate_calibration", "apply_calibration",
            "apply_changes", "revert_changes", "export_recalibrated_deck", "verify_exported_deck"]
