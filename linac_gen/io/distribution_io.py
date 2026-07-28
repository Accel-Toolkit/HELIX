"""Distribution I/O: export/import beam particle distributions (Task 12.1)."""


def export_distribution(beam, filepath: str) -> None:
    """Export beam particles as absolute coordinates with reference header.

    Format
    ------
    # Linac_Gen distribution file
    # species: proton
    # w_kin_ref: 3.000000 MeV
    # phi_ref: -30.000000 deg
    # frequency: 352.210000 MHz
    # current: 60.000000 mA
    # n_particles: 100000
    # columns: x(mm) xp(mrad) y(mm) yp(mrad) phi_abs(deg) W_abs(MeV)
    0.123  -0.456  0.789  1.234  -45.600  3.0012
    ...

    Only alive particles are exported.  Columns 4-5 are ABSOLUTE:
    phi_abs = ref.phi_s + dphi,  W_abs = ref.w_kin + dW.

    Parameters
    ----------
    beam : Beam
        Beam object to export.
    filepath : str
        Destination file path.
    """
    ref = beam.ref
    alive = beam.alive_particles
    with open(filepath, "w") as f:
        f.write("# Linac_Gen distribution file\n")
        f.write(f"# species: {ref.species.name}\n")
        f.write(f"# w_kin_ref: {ref.w_kin:.6f} MeV\n")
        f.write(f"# phi_ref: {ref.phi_s:.6f} deg\n")
        f.write(f"# frequency: {ref.frequency:.6f} MHz\n")
        f.write(f"# current: {beam.current:.6f} mA\n")
        f.write(f"# n_particles: {beam.n_alive}\n")
        f.write("# columns: x(mm) xp(mrad) y(mm) yp(mrad) phi_abs(deg) W_abs(MeV)\n")
        for p in alive:
            phi_abs = ref.phi_s + p[4]
            w_abs = ref.w_kin + p[5]
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} "
                f"{phi_abs:.6f} {w_abs:.6f}\n"
            )


def import_distribution(filepath: str, ref_w_kin=None, ref_phi_s=None) -> tuple:
    """Import distribution file.

    Thin wrapper around :func:`linac_gen.distributions.from_file.load_distribution`
    provided for symmetry with :func:`export_distribution`.

    Parameters
    ----------
    filepath : str
        Path to the distribution file.
    ref_w_kin : float or None
        Reference kinetic energy (MeV).  Overrides any value in the file header.
    ref_phi_s : float or None
        Reference phase (deg).  Overrides any value in the file header.

    Returns
    -------
    particles : np.ndarray
        Shape (N, 6) relative phase-space coordinates.
    header : dict
        Parsed header metadata.
    """
    from linac_gen.distributions.from_file import load_distribution
    return load_distribution(filepath, ref_w_kin=ref_w_kin, ref_phi_s=ref_phi_s)
