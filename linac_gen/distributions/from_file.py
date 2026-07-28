"""Distribution loader: read particle coordinates from an ASCII file.

File format
-----------
Header lines (optional) begin with '#'.  The loader recognises two special
header fields:

    # w_kin_ref: <value> MeV
    # phi_ref:   <value> deg

Data lines contain 6 whitespace-separated floats representing absolute
coordinates::

    x(mm)  xp(mrad)  y(mm)  yp(mrad)  phi_abs(deg)  W_abs(MeV)

The loader subtracts the reference values to produce deviations:

    dphi = phi_abs  - phi_ref
    dW   = W_abs    - w_kin_ref

Columns 0-3 (transverse) are returned unchanged.

If explicit ``ref_w_kin`` or ``ref_phi_s`` arguments are supplied they take
precedence over any values found in the file header.  If neither a header
value nor an explicit argument is provided, the reference defaults to 0.
"""
import re
from typing import Optional
import numpy as np


def load_distribution(
    filepath: str,
    ref_w_kin: Optional[float] = None,
    ref_phi_s: Optional[float] = None,
) -> tuple:
    """Load distribution from an ASCII file.

    Parameters
    ----------
    filepath : str
        Path to the distribution file.
    ref_w_kin : float or None
        Reference kinetic energy (MeV).  Overrides any value in the file
        header.  Defaults to 0 if not provided and not in the header.
    ref_phi_s : float or None
        Reference phase (deg).  Overrides any value in the file header.
        Defaults to 0 if not provided and not in the header.

    Returns
    -------
    particles : np.ndarray
        Shape (N, 6) array of phase-space deviations
        [x(mm), x'(mrad), y(mm), y'(mrad), dphi(deg), dW(MeV)].
    header : dict
        Dictionary of header metadata.  Keys ``w_kin_ref`` and ``phi_ref``
        are present only when found in the file header.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If the data section cannot be parsed as a numeric array.
    """
    _re_w_kin = re.compile(
        r"#\s*w_kin_ref\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    _re_phi = re.compile(
        r"#\s*phi_ref\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )

    header: dict = {}
    data_lines: list[str] = []

    with open(filepath, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#"):
                m_w = _re_w_kin.search(stripped)
                if m_w:
                    header["w_kin_ref"] = float(m_w.group(1))
                m_phi = _re_phi.search(stripped)
                if m_phi:
                    header["phi_ref"] = float(m_phi.group(1))
            elif stripped:
                data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data lines found in '{filepath}'")

    data = np.array(
        [list(map(float, line.split())) for line in data_lines],
        dtype=np.float64,
    )

    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(
            f"Expected 6 columns per data row in '{filepath}', "
            f"got shape {data.shape}"
        )

    # Determine reference values: explicit args > file header > default 0
    w_ref = ref_w_kin if ref_w_kin is not None else header.get("w_kin_ref", 0.0)
    phi_ref = ref_phi_s if ref_phi_s is not None else header.get("phi_ref", 0.0)

    particles = data.copy()
    particles[:, 4] = data[:, 4] - phi_ref
    particles[:, 5] = data[:, 5] - w_ref

    return particles, header
