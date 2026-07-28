"""Generate test fixture files for field map reader tests.

Run this script to regenerate fixture files.
"""
import numpy as np
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_1d_edz():
    """Generate a 1D .edz file with sin(pi*z/L) on-axis field.

    TraceWin 1D .edz format:
        Line 1: nz (number of points)
        Line 2: z_start  z_end  (in cm)
        Lines 3..nz+2: Ez values (one per line, normalized to peak=1)
    """
    nz = 101
    z_start_cm = 0.0
    z_end_cm = 10.0  # 100 mm
    z_cm = np.linspace(z_start_cm, z_end_cm, nz)
    L_cm = z_end_cm - z_start_cm
    Ez = np.sin(np.pi * (z_cm - z_start_cm) / L_cm)

    filepath = os.path.join(FIXTURE_DIR, "test_1d.edz")
    with open(filepath, "w") as f:
        f.write(f"{nz}\n")
        f.write(f"{z_start_cm} {z_end_cm}\n")
        for val in Ez:
            f.write(f"{val:.10e}\n")
    print(f"Generated {filepath}")


def generate_2d_edz():
    """Generate a 2D .edz file with a small 5x10 (nr x nz) grid.

    TraceWin 2D .edz format:
        Line 1: nr  nz
        Line 2: dr(cm)  dz(cm)
        Next nr*nz values: Ez(r_i, z_j), row-major (r varies slowly, z varies fast)
        Next nr*nz values: Er(r_i, z_j), same layout
    """
    nr = 5
    nz = 10
    dr_cm = 0.5   # 5 mm
    dz_cm = 1.0   # 10 mm

    r_cm = np.arange(nr) * dr_cm
    z_cm = np.arange(nz) * dz_cm
    R, Z = np.meshgrid(r_cm, z_cm, indexing='ij')  # shape (nr, nz)

    L_cm = (nz - 1) * dz_cm
    # Ez: sinusoidal along z, Bessel-like radial dependence
    Ez = np.sin(np.pi * Z / L_cm) * (1.0 - 0.25 * (R / (nr * dr_cm))**2)
    # Er: derivative-based (simplified)
    Er = -0.5 * R * np.pi / L_cm * np.cos(np.pi * Z / L_cm)

    filepath = os.path.join(FIXTURE_DIR, "test_2d.edz")
    with open(filepath, "w") as f:
        f.write(f"{nr} {nz}\n")
        f.write(f"{dr_cm} {dz_cm}\n")
        # Ez values: row-major (r varies slowly)
        for i in range(nr):
            vals = " ".join(f"{Ez[i, j]:.10e}" for j in range(nz))
            f.write(vals + "\n")
        # Er values
        for i in range(nr):
            vals = " ".join(f"{Er[i, j]:.10e}" for j in range(nz))
            f.write(vals + "\n")
    print(f"Generated {filepath}")


def generate_csv():
    """Generate a simple CSV file with z(mm) and Ez(V/m) columns."""
    nz = 51
    z_mm = np.linspace(0, 100, nz)
    Ez = np.sin(np.pi * z_mm / 100.0)

    filepath = os.path.join(FIXTURE_DIR, "test_fields.csv")
    with open(filepath, "w") as f:
        f.write("# z(mm) Ez(V/m)\n")
        for z, e in zip(z_mm, Ez):
            f.write(f"{z:.6f} {e:.10e}\n")
    print(f"Generated {filepath}")


if __name__ == "__main__":
    generate_1d_edz()
    generate_2d_edz()
    generate_csv()
    print("All fixtures generated.")
