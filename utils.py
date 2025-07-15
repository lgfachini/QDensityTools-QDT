import numpy as np
from numba import njit, prange
from numba.typed import List
import density_calc

@njit(parallel=True)
def analyze_candidates_numba(candidates, x, y, z, density,
                              d2x, d2y, d2z, dxy, dxz, dyz, grad_mag,
                              grad_threshold, min_density):
    bcp_positions = List()
    bcp_rho = List()
    bcp_laplacian = List()
    bcp_hessians = List()

    for idx in prange(candidates.shape[0]):
        i, j, k = candidates[idx]

        if grad_mag[i, j, k] >= grad_threshold:
            continue
        if density[i, j, k] < min_density:
            continue

        h00 = d2x[i, j, k]
        h01 = dxy[i, j, k]
        h02 = dxz[i, j, k]
        h10 = h01
        h11 = d2y[i, j, k]
        h12 = dyz[i, j, k]
        h20 = h02
        h21 = h12
        h22 = d2z[i, j, k]

        laplacian = h00 + h11 + h22
        hessian_flat = (h00, h01, h02, h10, h11, h12, h20, h21, h22)

        bcp_positions.append((x[i], y[j], z[k]))  # correto: x[i], y[j], z[k]
        bcp_rho.append(density[i, j, k])
        bcp_laplacian.append(laplacian)
        bcp_hessians.append(hessian_flat)

    return bcp_positions, bcp_rho, bcp_laplacian, bcp_hessians


def filter_redundant_bcps(bcp_list, distance_threshold=0.2):
    filtered = []
    positions = []

    for bcp in bcp_list:
        pos = np.array(bcp['position'])
        is_redundant = False

        for existing_pos in positions:
            if np.linalg.norm(pos - existing_pos) < distance_threshold:
                is_redundant = True
                break

        if not is_redundant:
            filtered.append(bcp)
            positions.append(pos)

    return filtered


def find_bond_critical_points_and_export_cube(
    parser, grid_points=100, gradient_threshold=1e-3, min_density=1e-4,
    cube_filename='density.cube', bcp_filename='BCPs.xyz'
):
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    padding = 3.0
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding

    x = np.linspace(x_min, x_max, grid_points)
    y = np.linspace(y_min, y_max, grid_points)
    z = np.linspace(z_min, z_max, grid_points)
    dx = (x_max - x_min) / (grid_points - 1)
    dy = (y_max - y_min) / (grid_points - 1)
    dz = (z_max - z_min) / (grid_points - 1)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    print("Calculando densidade eletrônica em grade 3D...")
    density = density_calc.calculate_density(points, parser.data).reshape(grid_points, grid_points, grid_points)

    print(integrate_density(density, dx, dy, dz, spacing_in_angstroms=True))

    print("Exportando densidade para arquivo cube...")
    write_cube_file(cube_filename, density, x, y, z, parser.data['nuclei'])

    print("Calculando derivadas...")
    drho_dx, drho_dy, drho_dz = np.gradient(density, dx, dy, dz)
    grad_mag = np.sqrt(drho_dx**2 + drho_dy**2 + drho_dz**2)

    print("Calculando Hessiana...")
    d2x = np.gradient(drho_dx, dx, axis=0)
    d2y = np.gradient(drho_dy, dy, axis=1)
    d2z = np.gradient(drho_dz, dz, axis=2)
    dxy = np.gradient(drho_dx, dy, axis=1)
    dxz = np.gradient(drho_dx, dz, axis=2)
    dyz = np.gradient(drho_dy, dz, axis=2)

    print("Buscando candidatos...")
    candidates = np.argwhere(grad_mag < gradient_threshold)
    print("Analisando candidatos em paralelo...")
    pos, rho, lap, hessians = analyze_candidates_numba(
        candidates, x, y, z, density,
        d2x, d2y, d2z, dxy, dxz, dyz, grad_mag,
        gradient_threshold, min_density
    )

    bcp_list = []
    for p, r, l, hess_flat in zip(pos, rho, lap, hessians):
        H = np.array([[hess_flat[0], hess_flat[1], hess_flat[2]],
                      [hess_flat[3], hess_flat[4], hess_flat[5]],
                      [hess_flat[6], hess_flat[7], hess_flat[8]]])
        eigvals = np.linalg.eigvalsh(H)
        if eigvals[0] < 0 and eigvals[1] < 0 and eigvals[2] > 0:
            bcp_list.append({
                'position': np.array(p),
                'rho': r,
                'laplacian': l,
                'eigvals': eigvals,
                'type': 'BCP'
            })

    bcp_list = filter_redundant_bcps(bcp_list, distance_threshold=0.2)

    print(f"{len(bcp_list)} BCP(s) encontrados.")

    total_atoms = len(parser.data['nuclei']) + len(bcp_list)
    with open(bcp_filename, 'w') as f:
        f.write(f"{total_atoms}\n")
        f.write("Molecule + Bond Critical Points (BCPs)\n")

        # Átomos reais da molécula
        for atom in parser.data['nuclei']:
            symbol = atom.get('symbol', 'X')
            x_, y_, z_ = atom['coords']
            f.write(f"{symbol} {x_:.6f} {y_:.6f} {z_:.6f}\n")

        # BCPs como átomos fictícios "X"
        for bcp in bcp_list:
            x_, y_, z_ = bcp['position']
            f.write(f"X {x_:.6f} {y_:.6f} {z_:.6f}\n")


def write_cube_file(filename, density, x, y, z, nuclei):
    ANGSTROM_TO_BOHR = 1.0 / 0.52917721067
    nx, ny, nz = density.shape
    origin = np.array([x[0], y[0], z[0]]) * ANGSTROM_TO_BOHR
    dx = (x[1] - x[0]) * ANGSTROM_TO_BOHR
    dy = (y[1] - y[0]) * ANGSTROM_TO_BOHR
    dz = (z[1] - z[0]) * ANGSTROM_TO_BOHR

    with open(filename, 'w') as f:
        f.write("Cube file generated by script\n")
        f.write("Electron density data\n")
        f.write(f"{len(nuclei):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        f.write(f"{nx:5d} {dx:12.6f} 0.000000 0.000000\n")
        f.write(f"{ny:5d} 0.000000 {dy:12.6f} 0.000000\n")
        f.write(f"{nz:5d} 0.000000 0.000000 {dz:12.6f}\n")

        for atom in nuclei:
            coords_bohr = np.array(atom['coords']) * ANGSTROM_TO_BOHR
            f.write(f"{atom['atomic_number']:5d} {0.0:12.6f} {coords_bohr[0]:12.6f} {coords_bohr[1]:12.6f} {coords_bohr[2]:12.6f}\n")

        # Transposição para ajustar o formato esperado do .cube (verifique se necessário)
        density_fixed = density.transpose(2, 1, 0)
        vals = density_fixed.flatten(order='F')
        for i in range(0, len(vals), 6):
            f.write(" ".join(f"{v:13.5e}" for v in vals[i:i+6]) + "\n")


def integrate_density(density, dx, dy, dz, spacing_in_angstroms=True):
    ANGSTROM_TO_BOHR = 1.0 / 0.52917721067
    factor = ANGSTROM_TO_BOHR**3 if spacing_in_angstroms else 1.0
    dv = dx * dy * dz * factor
    total_electrons = np.sum(density) * dv
    return total_electrons
