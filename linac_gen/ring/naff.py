"""Tunes from turn-by-turn data: NAFF (numerical analysis of fundamental
frequencies, J. Laskar).

The signal is multiplied by a Hann window of order p, the frequency of the
largest peak of its Fourier transform is located on the FFT grid and then
refined by maximizing the modulus of the windowed Fourier integral

    F(nu) = sum_n w_n z_n exp(-2 pi i nu n) / sum_n w_n

in the neighbourhood of the FFT peak; further terms are found the same way
after subtracting the ones already found.  A constant part (a closed-orbit
offset) is fitted together with the first term — the windowed mean also
holds that term's own leakage, sum_n w_n exp(2 pi i nu n) times its
amplitude, which is taken out of it until the two agree — so a tone a few
periods long is not pulled by the constant.  For a pure tone, with or
without a constant, the error is at rounding (1e-16) from three periods in
the record on, against 1/N for the bare FFT.  A weaker line in the signal
pulls the main one through the window's leakage: over 1000 turns (p = 1) a
line of 1 % of the amplitude 0.1 away moves it by 2e-12, one of 5 % 0.04
away by 3e-10.

A complex signal z = x - i p in normalized coordinates (x, p rotating by mu
per turn) gives the tune in [0, 1) without the mirror ambiguity of a real
signal, whose tune is only known in [0, 0.5]; for a real signal each term
is fitted with its mirror image (-nu), whose leakage into the amplitude is
solved exactly — a tune a period or so from 0 or 0.5 is not pulled by its
image either.
"""
from __future__ import annotations

import math

import numpy as np


def _window(n: int, order: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n)
    w = (1.0 + np.cos(np.pi * t)) ** int(order)
    return w / w.sum()


def _amplitude(z: np.ndarray, w: np.ndarray, nu: float) -> complex:
    n = np.arange(z.size)
    return complex(np.sum(w * z * np.exp(-2j * np.pi * nu * n)))


def _refine(z: np.ndarray, w: np.ndarray, nu0: float, half_width: float) -> float:
    """The maximum of |F(nu)| near nu0: Brent's method on the offset from nu0
    (so its tolerance is not relative to nu itself), then Newton steps on
    d|F|^2/dnu = 0 with the analytic derivatives of F."""
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda d: -abs(_amplitude(z, w, nu0 + d)),
                          bounds=(-half_width, half_width), method="bounded",
                          options={"xatol": 1e-16, "maxiter": 500})
    return _polish(z, w, nu0 + float(res.x), half_width)


def _polish(z: np.ndarray, w: np.ndarray, nu: float, half_width: float) -> float:
    """Newton steps on d|F|^2/dnu = 0 from nu (already near the maximum)."""
    n = np.arange(z.size)
    wz = w * z
    k = -2j * np.pi * n
    for _ in range(6):
        e = np.exp(-2j * np.pi * nu * n)
        F = np.sum(wz * e)
        F1 = np.sum(wz * k * e)
        F2 = np.sum(wz * k * k * e)
        g = (np.conj(F) * F1).real                     # (half) d|F|^2/dnu
        gp = (abs(F1) ** 2 + (np.conj(F) * F2).real)
        if gp >= 0.0 or not np.isfinite(gp):           # (not at a maximum)
            break
        step = -g / gp
        if abs(step) > half_width:
            break
        nu += step
        if abs(step) < 1e-16:
            break
    return float(nu)


def _spectral_peak(z: np.ndarray, w: np.ndarray, real: bool) -> tuple[float, float]:
    N = z.size
    pad = 1 << int(math.ceil(math.log2(8 * N)))       # (a finer grid for the first guess)
    spec = np.abs(np.fft.fft(z * w, pad))
    freqs = np.fft.fftfreq(pad)
    if real:
        keep = freqs >= 0.0
        spec, freqs = spec[keep], freqs[keep]
    k = int(np.argmax(spec))
    return float(freqs[k]), 2.0 / pad


def _real_amplitude(z: np.ndarray, w: np.ndarray, nu: float) -> complex:
    """The amplitude a of a real term a e^(2 pi i nu n) + conj(a) e^(-2 pi i nu n):
    F(nu) = a + conj(a) K with K = sum_n w_n e^(-4 pi i nu n), solved for a."""
    n = np.arange(z.size)
    F = _amplitude(z, w, nu)
    K = complex(np.sum(w * np.exp(-4j * np.pi * nu * n)))
    det = 1.0 - abs(K) ** 2
    if det < 1e-12:                                 # (nu at 0 or 0.5: no image to separate)
        return F
    return complex((F.real * (1.0 - K.real) - F.imag * K.imag) / det,
                   (F.imag * (1.0 + K.real) - F.real * K.imag) / det)


def _main_term(z: np.ndarray, w: np.ndarray, real: bool, nu_start, half_width):
    """(nu, a, half_width) of the strongest term of z: from the FFT peak, or —
    with ``nu_start`` — polished from there."""
    n = np.arange(z.size)
    if nu_start is None:
        nu0, half_width = _spectral_peak(z, w, real)
        nu = _refine(z, w, nu0, half_width)
    else:
        nu = _polish(z, w, nu_start, half_width)
    if not real:
        nu %= 1.0
        return nu, _amplitude(z, w, nu), half_width
    # the mirror term leaks into the estimate: refit with it removed
    for _it in range(30):
        a = _real_amplitude(z, w, nu)
        img = np.conj(a) * np.exp(-2j * np.pi * nu * n)
        nu_new = _polish(z - img, w, nu, half_width)
        done = abs(nu_new - nu) < 1e-16
        nu = nu_new
        if done:
            break
    nu = abs(nu)
    return nu, _real_amplitude(z, w, nu), half_width


def _term(nu: float, a: complex, n: np.ndarray, real: bool) -> np.ndarray:
    t = a * np.exp(2j * np.pi * nu * n)
    return t + np.conj(a) * np.exp(-2j * np.pi * nu * n) if real else t


def naff(signal, *, n_terms: int = 1, window_order: int = 1) -> list[tuple[float, complex]]:
    """The ``n_terms`` strongest frequencies of ``signal`` (turn by turn), in
    cycles per turn, with their complex amplitudes: [(nu, a), ...].  A real
    signal's frequencies are in [0, 0.5], a complex one's in [0, 1); a
    constant part is fitted with the first term (not returned), and for a
    real signal each term's mirror (-nu) with it."""
    z = np.asarray(signal)
    if z.ndim != 1 or z.size < 8:
        raise ValueError("naff: a 1-D signal of at least 8 turns is needed")
    if not np.all(np.isfinite(z)):
        raise ValueError("naff: the signal is not finite (a lost particle?)")
    real = not np.iscomplexobj(z)
    N = z.size
    w = _window(N, window_order)
    z = z.astype(complex)
    n = np.arange(N)
    # the constant c and the first term together: the windowed mean M holds
    # c and the term's leakage a S(nu), S = sum_n w_n e^(2 pi i nu n), so
    # c = M - a S (for a real signal also the image's, conj(a S)), refined
    # with the term until both settle
    M = complex(np.sum(w * z))
    c = M
    nu, hw = None, None
    for _it in range(40):
        nu, a, hw = _main_term(z - c, w, real, nu, hw)
        lead = a * complex(np.sum(w * np.exp(2j * np.pi * nu * n)))
        c_new = M - (2.0 * lead.real if real else lead)
        done = abs(c_new - c) <= 1e-16 * max(abs(a), abs(c_new), 1e-300)
        c = c_new
        if done:
            break
    nu, a, hw = _main_term(z - c, w, real, nu, hw)
    out = [(nu, a)]
    resid = z - c - _term(nu, a, n, real)
    for _ in range(int(n_terms) - 1):
        nu_k, a_k, _hw = _main_term(resid, w, real, None, None)
        out.append((nu_k, a_k))
        resid = resid - _term(nu_k, a_k, n, real)
    return out


def tune(signal, *, window_order: int = 1) -> float:
    """The main frequency of ``signal`` (see :func:`naff`)."""
    return naff(signal, n_terms=1, window_order=window_order)[0][0]


def tunes_from_normalized(xn, pn, *, window_order: int = 1) -> float:
    """The tune in [0, 1) from normalized turn-by-turn coordinates (x, p) that
    rotate by +2 pi Q per turn in the (x, p) plane as the ring optics' normal
    form defines it (x' = x cos mu + p sin mu)."""
    return tune(np.asarray(xn) - 1j * np.asarray(pn), window_order=window_order)
