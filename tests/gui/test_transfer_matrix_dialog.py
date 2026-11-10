"""The transfer-matrix window across a FREQ card, through the dialog itself."""
import numpy as np
import pytest
from linac_gen_gui.dialogs.transfer_matrix_dialog import TransferMatrixDialog

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Freq

GAMMA2 = ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5).gamma ** 2


def _dialog():
    lat = Lattice()
    for el in (Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0)):
        lat.add(el)
    return TransferMatrixDialog(None, lat, BeamConfig(species="proton", energy=2.1, frequency=162.5))


def _shown(dlg, start, end, basis):
    dlg._start_spin.setValue(start)
    dlg._end_spin.setValue(end)
    dlg._basis_combo.setCurrentIndex(dlg._basis_combo.findData(basis))
    dlg._compute()
    t = dlg._matrix_table
    return np.array([[float(t.item(i, j).text()) for j in range(6)] for i in range(6)])


@pytest.mark.parametrize("start, length_m", [(0, 0.25), (2, 0.15)], ids=["whole", "after_card"])
def test_the_tracewin_basis_is_physical_across_a_card(qapp, start, length_m):
    dlg = _dialog()
    M = _shown(dlg, start, 2, "tracewin")
    assert M[4, 4] == pytest.approx(1.0, abs=1e-6)
    assert M[4, 5] == pytest.approx(length_m / GAMMA2, abs=2e-6)


def test_our_basis_shows_the_factor_and_says_why(qapp):
    dlg = _dialog()
    M = _shown(dlg, 0, 2, "ours")
    assert M[4, 4] == pytest.approx(2.0, abs=1e-6)
    assert "f_out/f_in" in dlg._summary.toPlainText()
