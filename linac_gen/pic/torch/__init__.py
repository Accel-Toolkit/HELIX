"""Differentiable (PyTorch autograd) PIC space charge.

An opt-in torch mirror of the numpy/C++ PIC in :mod:`linac_gen.pic`. It is
imported only when the torch space-charge backend is explicitly selected
(``SpaceChargeConfig.sc_backend = "torch"``) or when the differentiable
step tracker is used; it never affects the default numpy/C++ PIC path.

Everything here runs FP64 on CPU — required for numerical parity with the
numpy PIC and for clean autograd gradients.
"""
