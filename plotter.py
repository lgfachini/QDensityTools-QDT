import numpy as np
import matplotlib.pyplot as plt
import os
import density_calc

EPS = 1e-15
BOHR_TO_ANGSTROM = 0.52917721067

def _coords_to_angstrom(coords_bohr):
    return coords_bohr * BOHR_TO_ANGSTROM

def _generate_grid(parser, plane, z_pos, grid_points, padding=None):
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    if padding is None:
        # padding proporcional ao tamanho do sistema exceto se menor do que 4 bohr
        span = coords.max(axis=0) - coords.min(axis=0)
        padding = max(0.15 * np.linalg.norm(span), 4.0)

    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding

    if plane == 'xy':
        x = np.linspace(x_min, x_max, grid_points)
        y = np.linspace(y_min, y_max, grid_points)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_pos)])
        return xx, yy, points
    elif plane == 'xz':
        x = np.linspace(x_min, x_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        xx, zz = np.meshgrid(x, z)
        points = np.column_stack([xx.ravel(), np.full(xx.size, z_pos), zz.ravel()])
        return xx, zz, points
    elif plane == 'yz':
        y = np.linspace(y_min, y_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        yy, zz = np.meshgrid(y, z)
        points = np.column_stack([np.full(yy.size, z_pos), yy.ravel(), zz.ravel()])
        return yy, zz, points
    else:
        raise ValueError("Plano inválido. Use 'xy', 'xz' ou 'yz'.")

def _draw_atoms(ax, parser, plane):
    colors = plt.cm.tab20.colors
    for nuc in parser.data['nuclei']:
        coords_angstrom = _coords_to_angstrom(nuc['coords'])
        if plane == 'xy':
            x_pos, y_pos = coords_angstrom[0], coords_angstrom[1]
        elif plane == 'xz':
            x_pos, y_pos = coords_angstrom[0], coords_angstrom[2]
        else:  # 'yz'
            x_pos, y_pos = coords_angstrom[1], coords_angstrom[2]

        color = colors[nuc['atomic_number'] % len(colors)]
        ax.plot(x_pos, y_pos, 'o', color=color, markersize=8)
        ax.text(x_pos + 0.05, y_pos + 0.05, nuc['symbol'], fontsize=10, color=color)

def plot_density_slice(parser, plane='xy', z_pos=0.0, grid_points=700):
    xx, yy, points_bohr = _generate_grid(parser, plane, z_pos, grid_points)
    density = density_calc.calculate_density(points_bohr, parser.data, spin='total')
    density_log = np.log10(density + EPS).reshape(grid_points, grid_points)

    # converter grade para angstrom só para plot
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx_ang, yy_ang, density_log, levels=60, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Log10 da densidade eletrônica (a.u.)')
    ax.set_title(f'Densidade Eletrônica (fatia {plane.upper()})')
    ax.set_xlabel('X (Å)' if plane in ['xy', 'xz'] else 'Y (Å)')
    ax.set_ylabel('Y (Å)' if plane == 'xy' else 'Z (Å)')

    _draw_atoms(ax, parser, plane)
    plt.tight_layout()

    output_dir = os.path.dirname(parser.filename)
    filename = os.path.join(output_dir, f'density_slice_{plane}_z{_coords_to_angstrom(np.array([z_pos]))[0]:.2f}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_laplacian_slice(parser, plane='xy', z_pos=0.0, grid_points=700):
    xx, yy, points_bohr = _generate_grid(parser, plane, z_pos, grid_points)
    density = density_calc.calculate_density(points_bohr, parser.data).reshape(grid_points, grid_points)

    dx = (xx[0,1] - xx[0,0]) if plane == 'xy' else (yy[1,0] - yy[0,0])
    d2x = np.gradient(np.gradient(density, axis=0), axis=0) / dx**2
    d2y = np.gradient(np.gradient(density, axis=1), axis=1) / dx**2
    laplacian = d2x + d2y

    laplacian_abs = np.abs(laplacian) + EPS
    laplacian_sign = np.sign(laplacian)
    laplacian_log = laplacian_sign * np.log10(laplacian_abs)

    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx_ang, yy_ang, laplacian_log, levels=60, cmap='seismic')
    plt.colorbar(cf, ax=ax, label='Log10 do Laplaciano da Densidade Eletrônica (a.u.)')
    ax.set_title(f'Laplaciano da Densidade Eletrônica (fatia {plane.upper()})')
    ax.set_xlabel('X (Å)' if plane in ['xy', 'xz'] else 'Y (Å)')
    ax.set_ylabel('Y (Å)' if plane == 'xy' else 'Z (Å)')

    _draw_atoms(ax, parser, plane)
    plt.tight_layout()

    output_dir = os.path.dirname(parser.filename)
    filename = os.path.join(output_dir, f'laplacian_slice_{plane}_z{_coords_to_angstrom(np.array([z_pos]))[0]:.2f}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_gradient_magnitude_slice(parser, plane='xy', z_pos=0.0, grid_points=700):
    xx, yy, points_bohr = _generate_grid(parser, plane, z_pos, grid_points)
    density = density_calc.calculate_density(points_bohr, parser.data).reshape(grid_points, grid_points)

    dx = (xx[0,1] - xx[0,0]) if plane == 'xy' else (yy[1,0] - yy[0,0])
    d_rho_dx, d_rho_dy = np.gradient(density, dx, dx)
    grad_mag = np.sqrt(d_rho_dx**2 + d_rho_dy**2) + EPS
    grad_mag_log = np.log10(grad_mag)

    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx_ang, yy_ang, grad_mag_log, levels=60, cmap='inferno')
    plt.colorbar(cf, ax=ax, label='Log10 do Módulo do Gradiente da Densidade Eletrônica (a.u.)')
    ax.set_title(f'Módulo do Gradiente da Densidade Eletrônica (fatia {plane.upper()})')
    ax.set_xlabel('X (Å)' if plane in ['xy', 'xz'] else 'Y (Å)')
    ax.set_ylabel('Y (Å)' if plane == 'xy' else 'Z (Å)')

    _draw_atoms(ax, parser, plane)
    plt.tight_layout()

    output_dir = os.path.dirname(parser.filename)
    filename = os.path.join(output_dir, f'gradient_slice_{plane}_z{_coords_to_angstrom(np.array([z_pos]))[0]:.2f}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_density_gradient_laplacian_along_path(parser, atom1_index, atom2_index, points_count=700):
    atom1 = parser.data['nuclei'][atom1_index]
    atom2 = parser.data['nuclei'][atom2_index]

    coords1 = atom1['coords']
    coords2 = atom2['coords']
    vec = coords2 - coords1
    total_dist = np.linalg.norm(vec)

    path_points = np.array([coords1 + t * vec for t in np.linspace(0, 1, points_count)])
    distances_bohr = np.linspace(0, total_dist, points_count)
    distances_angstrom = distances_bohr * BOHR_TO_ANGSTROM

    density = density_calc.calculate_density(path_points, parser.data)
    gradient = np.gradient(density, distances_bohr)
    laplacian = np.gradient(gradient, distances_bohr)

    def safe_log10(arr):
        return np.log10(np.abs(arr) + EPS)

    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + EPS)

    density_log_norm = normalize(safe_log10(density))
    gradient_log_norm = normalize(safe_log10(gradient))
    laplacian_log_norm = normalize(safe_log10(laplacian))

    plt.figure(figsize=(9, 6))
    plt.plot(distances_angstrom, density_log_norm, label='log10(Density)', color='black', linewidth=2)
    plt.plot(distances_angstrom, gradient_log_norm, label='log10(Gradient)', color='blue', linestyle='--')
    plt.plot(distances_angstrom, laplacian_log_norm, label='log10(Laplacian)', color='red', linestyle=':')
    plt.xlabel('Distance along path (Å)')
    plt.ylabel('log10-scaled & normalized')
    plt.title(f'log10(Density, Gradient, Laplacian)\nBetween {atom1["symbol"]} and {atom2["symbol"]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = os.path.dirname(parser.filename)
    filename = os.path.join(output_dir, f'density_gradient_laplacian_log_path_{atom1["symbol"]}_{atom2["symbol"]}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_spin_density_slice(parser, plane='xy', z_pos=0.0, grid_points=700):
    xx, yy, points_bohr = _generate_grid(parser, plane, z_pos, grid_points)
    alpha_density = density_calc.calculate_density(points_bohr, parser.data, spin='alpha')
    beta_density = density_calc.calculate_density(points_bohr, parser.data, spin='beta')
    spin_density = alpha_density - beta_density
    spin_grid = spin_density.reshape(grid_points, grid_points)

    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx_ang, yy_ang, spin_grid, levels=60, cmap='seismic')
    plt.colorbar(cf, ax=ax, label='Densidade de Spin (ρα − ρβ)')
    ax.set_title(f'Densidade de Spin (fatia {plane.upper()})')
    ax.set_xlabel('X (Å)' if plane in ['xy', 'xz'] else 'Y (Å)')
    ax.set_ylabel('Y (Å)' if plane == 'xy' else 'Z (Å)')

    _draw_atoms(ax, parser, plane)
    plt.tight_layout()

    output_dir = os.path.dirname(parser.filename)
    output_path = os.path.join(output_dir, f"spin_density_slice_{plane}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
