"""Every attribute the element inspector lists must exist on the element it
describes — a misspelt attribute (``n_index`` for ``field_index``, 2026-09-06)
is skipped silently by the inspector and the field becomes invisible."""
import pytest

pytest.importorskip("PyQt6")


def _instances():
    from linac_gen.elements.aperture import Aperture
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.marker import Marker
    from linac_gen.elements.multipole import Multipole
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.elements.steerer import Steerer
    from linac_gen.elements.thin_lens import ThinLens
    return {
        "Drift": Drift("d", length=100.0),
        "Quadrupole": Quadrupole("q", length=50.0, gradient=5.0),
        "Multipole": Multipole("m", knl=[0.0, 0.1]),
        "Dipole": Dipole("b", angle=10.0, rho=1000.0),
        "Solenoid": Solenoid("s", length=100.0, field=0.5),
        "RFGap": RFGap("g", voltage=0.1, phase=-30.0, frequency=352.21),
        "Aperture": Aperture("a", dx=10.0, dy=10.0),
        "Marker": Marker("mk"),
        "Steerer": Steerer("st"),
        "ThinLens": ThinLens("tl"),
    }


def test_schema_attrs_exist_on_elements():
    from linac_gen_gui.interphase.panels.element_inspector import _SCHEMA
    missing = []
    for cls_name, el in _instances().items():
        for attr, label, unit in _SCHEMA.get(cls_name, []):
            if not hasattr(el, attr):
                missing.append(f"{cls_name}.{attr} ({label})")
    assert not missing, "inspector rows without a backing attribute: " + ", ".join(missing)


def test_dipole_rows_use_degrees():
    from linac_gen_gui.interphase.panels.element_inspector import _SCHEMA
    units = {attr: unit for attr, _label, unit in _SCHEMA["Dipole"]}
    assert units["angle"] == "deg" and units["e1"] == "deg" and units["e2"] == "deg"
    assert "field_index" in units and "rho" in units
