"""NAFF (linac_gen.ring.naff): the main frequency of turn-by-turn data —
pure tones a few periods long, with a constant beside them, real signals
near 0 and 0.5, a weaker second line, and what is refused."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.ring import naff

N = 1001


def _tone(nu, n=N, amp=0.7, phase=0.3):
    k = np.arange(n)
    return amp * np.exp(2j * np.pi * nu * k + 1j * phase)


@pytest.mark.parametrize("periods", [3.0, 3.5, 4.5, 5.5, 10.5, 20.5, 190.3])
def test_a_complex_tone_from_three_periods_on(periods):
    nu = periods / (N - 1)
    assert abs(naff.tune(_tone(nu)) - nu) < 1e-15


@pytest.mark.parametrize("periods", [3.5, 5.5, 20.5])
def test_a_constant_does_not_pull_the_tone(periods):
    # (a closed-orbit offset: fitted with the tone, not subtracted as a mean)
    nu = periods / (N - 1)
    assert abs(naff.tune(_tone(nu) + (0.5 + 0.2j)) - nu) < 1e-15


@pytest.mark.parametrize("nu", [0.0012, 0.0035, 0.01, 0.19, 0.31, 0.49, 0.4975, 0.499])
@pytest.mark.parametrize("dc", [0.0, 0.3])
def test_a_real_signal_near_0_and_half(nu, dc):
    # the mirror image (-nu) is fitted with the term: a period or so from 0
    # or 0.5 the tune is not pulled by it
    k = np.arange(N)
    z = 0.7 * np.cos(2 * np.pi * nu * k + 0.4) + dc
    assert abs(naff.tune(z) - nu) < 1e-15


def test_the_rotation_sense_of_normalized_coordinates():
    # x' = x cos mu + p sin mu, p' = -x sin mu + p cos mu  →  Q, not 1 - Q
    mu = 2 * np.pi * 0.2311
    x, p = [1.0], [0.0]
    for _ in range(999):
        x.append(x[-1] * np.cos(mu) + p[-1] * np.sin(mu))
        p.append(-x[-2] * np.sin(mu) + p[-1] * np.cos(mu))
    assert abs(naff.tunes_from_normalized(np.array(x), np.array(p)) - 0.2311) < 1e-13


@pytest.mark.parametrize("a2, d, bound", [(0.01, 0.1, 3e-12), (0.05, 0.04, 4e-10)])
def test_a_weaker_line_beside(a2, d, bound):
    # (the documented pull of a second line through the window's leakage)
    k = np.arange(1000)
    z = np.exp(2j * np.pi * 0.31 * k) + a2 * np.exp(2j * np.pi * (0.31 - d) * k + 1.0j)
    got = naff.naff(z, n_terms=2)
    assert abs(got[0][0] - 0.31) < bound
    assert abs(got[1][0] - (0.31 - d)) < 1e-14
    assert abs(abs(got[0][1]) - 1.0) < 2e-8 and abs(abs(got[1][1]) - a2) < 1e-11


def test_what_is_refused():
    with pytest.raises(ValueError, match="at least 8"):
        naff.tune(np.ones(5))
    z = _tone(0.2)
    z[10] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        naff.tune(z)
