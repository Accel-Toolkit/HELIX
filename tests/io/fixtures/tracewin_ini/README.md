# TraceWin `.ini` fixtures

## `ads.ini`

Byte-exact copy of `src/lightwin/data/ads/ads.ini` from LightWin
(<https://github.com/AdrienPlacais/LightWin>), a TraceWin project options
file of the ADS example linac (44,824 bytes, TraceWin 2019+ layout).

LightWin is distributed under the MIT License, Copyright (c) 2025 Adrien
Plaçais, Frédéric Bouly, Bruce Yee-Rendon, Jean-Michel Lagniel, Didier
Uriot.  The copyright notice and permission notice of that licence apply
to this copy.

There is deliberately no `ads.dat` next to it: the GUI tests exercise the
"import the sibling `.ini`?" prompt with their own temporary decks.

### Ground truth

LightWin describes the same input beam in `lightwin.toml` (`[beam]`) as a
6×6 σ-matrix, which is what the reader's conversion is pinned against:

| quantity | value |
|---|---|
| `e_mev` | 20.0 |
| `e_rest_mev` | 938.27203 |
| `f_bunch_mhz` | 100.0 |
| σ (x, x′) | `[[8.409896e-06, 3.548736e-06], [3.548736e-06, 1.607857e-06]]` |
| σ (y, y′) | `[[2.941564e-06, 6.094860e-07], [6.094860e-07, 4.418911e-07]]` |
| σ (z, δ)  | `[[3.593136e-06, -2.552518e-07], [-2.552518e-07, 5.994771e-07]]` |

The file itself stores `current1` = 0.005 A (the TOML says
`i_milli_a = 0.0`); the tests pin the file's 5 mA, the TOML value is a
LightWin-side override.

### Private samples

Fermilab PIP-II project files (an SCL project at 116.1 MeV / 804.96 MHz /
23.7 mA and a 2017-layout LEBT project) are not redistributed.  The tests
that pin them run only when `HELIX_TW_INI_PRIVATE_DIR` points at a folder
containing `fnalsc_APSeminar_07142026.ini` and/or `LEBT_PXIE_JAN2017.ini`;
otherwise they are skipped.
