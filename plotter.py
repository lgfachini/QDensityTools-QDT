import numpy as np
import matplotlib.pyplot as plt
import os
import density_calc
import rdg_calc
import s_sign_lambda2_rho_p
from matplotlib.colors import SymLogNorm, TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.pyplot as plt

EPS = 1e-15
BOHR_TO_ANGSTROM = 0.52917721067

def _coords_to_angstrom(coords_bohr):
    return coords_bohr * BOHR_TO_ANGSTROM

def _generate_grid(parser, plane, z_pos, grid_points, padding=None):
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    if padding is None:
        span = coords.max(axis=0) - coords.min(axis=0)
        padding = max(0.15 * np.linalg.norm(span), 4.0)

    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding

    if plane == 'xy':
        x = np.linspace(x_min, x_max, grid_points)
        y = np.linspace(y_min, y_max, grid_points)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_pos)])
    elif plane == 'xz':
        x = np.linspace(x_min, x_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        xx, yy = np.meshgrid(x, z)
        points = np.column_stack([xx.ravel(), np.full(xx.size, z_pos), yy.ravel()])
    elif plane == 'yz':
        y = np.linspace(y_min, y_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        xx, yy = np.meshgrid(y, z)
        points = np.column_stack([np.full(xx.size, z_pos), xx.ravel(), yy.ravel()])
    else:
        raise ValueError("Plano inválido. Use 'xy', 'xz' ou 'yz'.")
    return xx, yy, points

def _generate_custom_plane_grid(parser, atom_indices, grid_points, padding=None):
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    p1, p2, p3 = coords[atom_indices[0]], coords[atom_indices[1]], coords[atom_indices[2]]

    # Base do plano
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    x_axis = v1 / np.linalg.norm(v1)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Projeta todos os átomos no plano
    proj_coords = []
    for coord in coords:
        rel = coord - p1
        x_proj = np.dot(rel, x_axis)
        y_proj = np.dot(rel, y_axis)
        proj_coords.append([x_proj, y_proj])
    proj_coords = np.array(proj_coords)

    # Determinar bounds com padding
    x_min, y_min = proj_coords.min(axis=0)
    x_max, y_max = proj_coords.max(axis=0)

    if padding is None:
        padding = max(0.15 * np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)), 4.0)

    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    u = np.linspace(x_min, x_max, grid_points)
    v = np.linspace(y_min, y_max, grid_points)
    uu, vv = np.meshgrid(u, v)

    points = p1 + np.outer(uu.ravel(), x_axis) + np.outer(vv.ravel(), y_axis)

    return uu, vv, points, x_axis, y_axis, p1

def _draw_atoms(ax, parser, plane=None, z_pos=None, atom_indices=None, threshold_bohr=1.0):
    coords_bohr = np.array([n['coords'] for n in parser.data['nuclei']])
    
    if atom_indices is not None:
        # Plano customizado
        p1 = coords_bohr[atom_indices[0]]
        v1 = coords_bohr[atom_indices[1]] - p1
        v2 = coords_bohr[atom_indices[2]] - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        for nuc in parser.data['nuclei']:
            r = nuc['coords']
            dist_to_plane = np.abs(np.dot(r - p1, normal))
            if dist_to_plane < threshold_bohr:
                local_coords = r - p1
                x_proj = np.dot(local_coords, v1 / np.linalg.norm(v1))
                y_proj = np.dot(local_coords, np.cross(normal, v1) / np.linalg.norm(np.cross(normal, v1)))
                x_ang = x_proj * BOHR_TO_ANGSTROM
                y_ang = y_proj * BOHR_TO_ANGSTROM
                ax.plot(x_ang, y_ang, 'o', color='black', markersize=3)
                ax.text(x_ang + 0.05, y_ang + 0.05, nuc['symbol'], fontsize=12, color='black')

    else:
        # Plano padrão
        axis_map = {'xy': 2, 'xz': 1, 'yz': 0}
        ignore_axis = axis_map[plane]
        for nuc in parser.data['nuclei']:
            coord = nuc['coords']
            if abs(coord[ignore_axis] - z_pos) < threshold_bohr:
                coord_ang = _coords_to_angstrom(coord)
                if plane == 'xy':
                    x, y = coord_ang[0], coord_ang[1]
                elif plane == 'xz':
                    x, y = coord_ang[0], coord_ang[2]
                else:
                    x, y = coord_ang[1], coord_ang[2]
                ax.plot(x, y, 'o', color='black', markersize=3)
                ax.text(x + 0.05, y + 0.05, nuc['symbol'], fontsize=12, color='black')


def _plot_scalar_field(data, xx, yy, parser, suffix, title, label, cmap, xlabel, ylabel,
                       plane=None, z_pos=None, atom_indices=None,
                       linthresh=1e-3, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    # Se for colormap customizado
    if cmap == 'custom_diverging':
        # Definir colormap verde no zero, azul negativo, vermelho positivo
        cmap = LinearSegmentedColormap.from_list("blue-green-red", ['blue', 'green', 'red'])

        # Normalização simétrica em torno de zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cf = ax.contourf(xx, yy, data, levels=60, cmap=cmap, norm=norm)
    else:
        cf = ax.contourf(xx, yy, data, levels=60, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.colorbar(cf, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _draw_atoms(ax, parser, plane=plane, z_pos=z_pos, atom_indices=atom_indices)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(parser.filename), f"{suffix}.png"), dpi=300)
    plt.close()

def _prepare_slice(parser, plane, z_pos, grid_points, atom_indices):
    if atom_indices is not None:
        xx, yy, points, *_ = _generate_custom_plane_grid(parser, atom_indices, grid_points)
        xlabel, ylabel = "Custom X (Å)", "Custom Y (Å)"
        suffix = f"custom_plane_{'_'.join(map(str, atom_indices))}"
        draw_atoms = False
    else:
        xx, yy, points = _generate_grid(parser, plane, z_pos, grid_points)
        xlabel = 'X (Å)' if plane in ['xy', 'xz'] else 'Y (Å)'
        ylabel = 'Y (Å)' if plane == 'xy' else 'Z (Å)'
        zlab = _coords_to_angstrom(np.array([z_pos]))[0]
        suffix = f"{plane}_z{zlab:.2f}"
        draw_atoms = True
    return xx, yy, points, xlabel, ylabel, suffix, draw_atoms

def plot_density_slice(parser, plane='xy', z_pos=0.0, grid_points=700, atom_indices=None):
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(parser, plane, z_pos, grid_points, atom_indices)
    rho = density_calc.calculate_density(points, parser.data, spin='total').reshape(grid_points, grid_points)
    rho_log = np.log10(rho + EPS)
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)
    _plot_scalar_field(rho_log, xx_ang, yy_ang, parser, f'density_slice_{suffix}',
                   'Electronic density', r'$\log_{10}[\rho]$', 'viridis', xlabel, ylabel,
                   plane=plane, z_pos=z_pos, atom_indices=atom_indices)

def plot_laplacian_slice(parser, plane='xy', z_pos=0.0, grid_points=700, atom_indices=None):
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(parser, plane, z_pos, grid_points, atom_indices)
    rho = density_calc.calculate_density(points, parser.data).reshape(grid_points, grid_points)
    dx = (xx[0, 1] - xx[0, 0])
    d2x = np.gradient(np.gradient(rho, axis=0), axis=0) / dx**2
    d2y = np.gradient(np.gradient(rho, axis=1), axis=1) / dx**2
    lap = d2x + d2y
    lap_log = np.sign(lap) * np.log10(np.abs(lap) + EPS)
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)
    _plot_scalar_field(lap_log, xx_ang, yy_ang, parser, f'laplacian_slice_{suffix}',
                       'Electronic density Laplacian', r'$\log_{10}[\nabla^2\rho]$', 'seismic', xlabel, ylabel,
                   plane=plane, z_pos=z_pos, atom_indices=atom_indices)

def plot_gradient_magnitude_slice(parser, plane='xy', z_pos=0.0, grid_points=700, atom_indices=None):
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(parser, plane, z_pos, grid_points, atom_indices)
    rho = density_calc.calculate_density(points, parser.data).reshape(grid_points, grid_points)
    dx = (xx[0, 1] - xx[0, 0])
    gradx, grady = np.gradient(rho, dx, dx)
    grad_mag = np.sqrt(gradx**2 + grady**2)
    grad_log = np.log10(grad_mag + EPS)
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)
    _plot_scalar_field(grad_log, xx_ang, yy_ang, parser, f'gradient_slice_{suffix}',
                       'Electronic density gradient', r'$\log_{10}[|\nabla\rho|]$', 'inferno', xlabel, ylabel,
                   plane=plane, z_pos=z_pos, atom_indices=atom_indices)

def plot_spin_density_slice(parser, plane='xy', z_pos=0.0, grid_points=700, atom_indices=None):
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(parser, plane, z_pos, grid_points, atom_indices)
    alpha = density_calc.calculate_density(points, parser.data, spin='alpha')
    beta = density_calc.calculate_density(points, parser.data, spin='beta')
    spin = (alpha - beta).reshape(grid_points, grid_points)
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)
    _plot_scalar_field(spin, xx_ang, yy_ang, parser, f'spin_density_slice_{suffix}',
                       'Spin density', r'$\rho_\alpha - \rho_\beta$', 'seismic', xlabel, ylabel,
                   plane=plane, z_pos=z_pos, atom_indices=atom_indices)

def plot_density_gradient_laplacian_along_path(parser, atom1_index, atom2_index, points_count=700):
    atom1 = parser.data['nuclei'][atom1_index]
    atom2 = parser.data['nuclei'][atom2_index]
    coords1 = atom1['coords']
    coords2 = atom2['coords']
    vec = coords2 - coords1
    total_dist = np.linalg.norm(vec)
    path_points = np.array([coords1 + t * vec for t in np.linspace(0, 1, points_count)])
    distances_bohr = np.linspace(0, total_dist, points_count)
    distances_ang = distances_bohr * BOHR_TO_ANGSTROM
    rho = density_calc.calculate_density(path_points, parser.data)
    grad = np.gradient(rho, distances_bohr)
    lap = np.gradient(grad, distances_bohr)

    def safe_log10(arr): return np.log10(np.abs(arr) + EPS)
    def normalize(arr): return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + EPS)

    rho_n = normalize(safe_log10(rho))
    grad_n = normalize(safe_log10(grad))
    lap_n = normalize(safe_log10(lap))

    plt.figure(figsize=(9, 6))
    plt.plot(distances_ang, rho_n, label=r'$\log_{10}[\rho]$', color='black')
    plt.plot(distances_ang, grad_n, label=r'$\log_{10}[|\nabla \rho|]$', color='blue', linestyle='--')
    plt.plot(distances_ang, lap_n, label=r'$\log_{10}[\nabla^2 \rho]$', color='red', linestyle=':')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Normalized values')
    plt.title(rf'$\log_{{10}}[\rho, \nabla \rho, \nabla^2 \rho]$ entre {atom1["symbol"]} e {atom2["symbol"]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(os.path.dirname(parser.filename),
                            f'density_gradient_laplacian_log_path_{atom1["symbol"]}_{atom2["symbol"]}.png')
    plt.savefig(filename, dpi=300)
    plt.close()

    
def plot_reduced_gradient_slice(parser, plane='xy', z_pos=0.0, grid_points=600, atom_indices=None):
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(
        parser, plane, z_pos, grid_points, atom_indices
    )

    grid_shape = (grid_points, grid_points)
    rho, gx, gy, gz = rdg_calc.compute_density_and_gradient(points, parser.data, grid_shape=grid_shape)
    s_vals = rdg_calc.compute_s_values(rho, gx, gy, gz).reshape(grid_shape)

    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    _plot_scalar_field(np.log10(s_vals + EPS), xx_ang, yy_ang, parser,  
                       f'reduced_gradient_slice_{suffix}',
                       'Reduced density gradient (s)',
                       r'$\log_{10}[s]$', 'plasma', xlabel, ylabel,
                       plane=plane, z_pos=z_pos, atom_indices=atom_indices)

def plot_s_sign_lambda2_rho_slice(parser, plane='xy', z_pos=0.0, grid_points=600, atom_indices=None):

    # Preparar grade e pontos no plano (cartesiano ou customizado por 3 átomos)
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(
        parser, plane, z_pos, grid_points, atom_indices
    )

    grid_shape = (grid_points, grid_points)

    # Calcular s × sign(lambda2) × rho
    s_sign_lambda2_rho_vals, rho = s_sign_lambda2_rho_p.compute_s_sign_lambda2_times_rho(parser, points, parser.data, grid_shape)

    # Converter coordenadas para angstrom
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    _plot_scalar_field(
        s_sign_lambda2_rho_vals, xx_ang, yy_ang, parser,
        f's_sign_lambda2_rho_slice_{suffix}',
        r'$\log_{10}[s \times \rho] \times \mathrm{sign}(\lambda_2)$',
        r'$\log_{10}[s \times \rho] \times \mathrm{sign}(\lambda_2)$ (a.u.)',
        'custom_diverging', xlabel, ylabel,
        plane=plane, z_pos=z_pos, atom_indices=atom_indices
    )
