import numpy as np
import pytest
from linac_gen.core.lattice import Lattice
from linac_gen.elements.base import Element

class DummyElement(Element):
    def __init__(self, name: str, length: float):
        super().__init__(name=name, length=length, aperture=0.0, n_steps=1)

def test_create_empty_lattice():
    lat = Lattice()
    assert len(lat.elements) == 0
    assert lat.total_length == 0.0

def test_add_elements():
    lat = Lattice()
    lat.add(DummyElement("D1", 100.0))
    lat.add(DummyElement("Q1", 50.0))
    assert len(lat.elements) == 2
    assert lat.total_length == 150.0

def test_s_positions():
    lat = Lattice()
    lat.add(DummyElement("D1", 100.0))
    lat.add(DummyElement("Q1", 50.0))
    lat.add(DummyElement("D2", 200.0))
    s_start, s_end = lat.get_s_positions()
    np.testing.assert_array_almost_equal(s_start, [0.0, 100.0, 150.0])
    np.testing.assert_array_almost_equal(s_end, [100.0, 150.0, 350.0])

def test_get_element_by_name():
    lat = Lattice()
    lat.add(DummyElement("Q1", 50.0))
    lat.add(DummyElement("D1", 100.0))
    assert lat.get_element("Q1").name == "Q1"

def test_get_element_not_found():
    lat = Lattice()
    with pytest.raises(KeyError):
        lat.get_element("NONEXISTENT")
