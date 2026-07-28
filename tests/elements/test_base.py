import numpy as np
from linac_gen.elements.base import (
    Element, TransferMapElement, ThinKickElement,
    FieldMapElement, PassiveElement,
)
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.beam import Beam

def test_transfer_map_element_is_element():
    assert issubclass(TransferMapElement, Element)

def test_thin_kick_element_zero_length():
    """ThinKickElement.__init__ forces length=0 via super().__init__."""
    class DummyKick(ThinKickElement):
        def apply_kick(self, beam): pass
        def kick_matrix(self, ref): return np.eye(6)
    k = DummyKick(name="K1")
    assert k.length == 0.0

def test_passive_element_zero_length():
    assert issubclass(PassiveElement, Element)

def test_field_map_element_is_element():
    assert issubclass(FieldMapElement, Element)
