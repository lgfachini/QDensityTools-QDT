import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.linalg import eigh
from joblib import Parallel, delayed, parallel_backend
from itertools import combinations
from cube import write_cube_file
import density_calc
import os

BOHR_TO_ANGSTROM = 0.52917721067

# Interpoladores globais
GLOBAL_INTERP_GRAD = None
GLOBAL_INTERP_RHO = None
GLOBAL_INTERP_HESSIAN = None
GLOBAL_X = None
GLOBAL_Y = None
GLOBAL_Z = None


def point_within_bounds(p, x, y, z):
    return (x[0] <= p[0] <= x[-1]) and (y[0] <= p[1] <= y[-1]) and (z[0] <= p[2] <= z[-1])


def create_interpolators(x, y, z, density):
    interp_density = RegularGridInterpolator((x, y, z), density, bounds_error=False, fill_value=np.nan)
    drho_dx, drho_dy, drho_dz = np.gradient(density, x, y, z, edge_order=2)
    interp_grad = [
        RegularGridInterpolator((x, y, z), drho_dx, bounds_error=False, fill_value=np.nan),
        RegularGridInterpolator((x, y, z), drho_dy, bounds_error=False, fill_value=np.nan),
        RegularGridInterpolator((x, y, z), drho_dz, bounds_error=False, fill_value=np.nan)
    ]
    hess = []
    for d in [drho_dx, drho_dy, drho_dz]:
        hess.append(np.gradient(d, x, axis=0))
        hess.append(np.gradient(d, y, axis=1))
        hess.append(np.gradient(d, z, axis=2))
    interp_hessian = [RegularGridInterpolator((x, y, z), h, bounds_error=False, fill_value=np.nan) for h in hess]
    return interp_density, interp_grad, interp_hessian


def follow_gradient_optimized_thread(p0, grad_tol, min_density):
    global GLOBAL_INTERP_GRAD, GLOBAL_INTERP_RHO, GLOBAL_INTERP_HESSIAN, GLOBAL_X, GLOBAL_Y, GLOBAL_Z

    def objective(p):
        if not point_within_bounds(p, GLOBAL_X, GLOBAL_Y, GLOBAL_Z):
            return 1e6
        grad = np.array([g(p).item() for g in GLOBAL_INTERP_GRAD])
        if np.any(np.isnan(grad)):
            return 1e6
        return np.dot(grad, grad)

    res = minimize(objective, p0, method='BFGS', tol=grad_tol, options={'maxiter': 200, 'gtol': grad_tol})
    if not res.success:
        return None

    p = res.x
    if not point_within_bounds(p, GLOBAL_X, GLOBAL_Y, GLOBAL_Z):
        return None

    rho_val = GLOBAL_INTERP_RHO(p).item()
    if rho_val < min_density or np.isnan(rho_val):
        return None

    h = [hfun(p).item() for hfun in GLOBAL_INTERP_HESSIAN]
    if np.any(np.isnan(h)):
        return None

    hessian = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]])
    eigvals = np.sort(eigh(hessian, eigvals_only=True))
    if np.sum(eigvals < 0) == 2 and np.sum(eigvals > 0) == 1:
        return {'position': p, 'rho': rho_val, 'eigvals': eigvals, 'type': 'BCP'}
    return None


def generate_initial_points_between_atoms(nuclei, n_points_per_pair=30):
    points = []
    for a1, a2 in combinations(nuclei, 2):
        p1 = np.array(a1['coords'])
        p2 = np.array(a2['coords'])
        for t in np.linspace(0.2, 0.8, n_points_per_pair):
            pt = (1 - t) * p1 + t * p2
            points.append(pt)
    return np.array(points)


def filter_close_points(points, threshold=0.2):
    if not points:
        return []
    coords = np.array([p['position'] for p in points])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(threshold)
    mask = np.ones(len(points), dtype=bool)
    for i, j in pairs:
        mask[j] = False
    return [p for i, p in enumerate(points) if mask[i]]


def export_bcps_to_xyz(parser, bcp_list, filename='BCPs.xyz'):
    output_dir = os.path.dirname(parser.filename)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        total = len(parser.data['nuclei']) + len(bcp_list)
        f.write(f"{total}\n")
        f.write("Molecule + Bond Critical Points (BCPs)\n")
        for atom in parser.data['nuclei']:
            symbol = atom.get('symbol', 'X')
            x_, y_, z_ = np.array(atom['coords']) * BOHR_TO_ANGSTROM
            f.write(f"{symbol} {x_:.6f} {y_:.6f} {z_:.6f}\n")
        for bcp in bcp_list:
            x_, y_, z_ = np.array(bcp['position']) * BOHR_TO_ANGSTROM
            f.write(f"X {x_:.6f} {y_:.6f} {z_:.6f}\n")

def find_critical_points_from_gradient_flow(parser, grid_points=80, grad_tol=1e-4, min_density=1e-4,
                                            n_jobs=-1, n_points_per_pair=30):
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    padding = 3.0
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding

    x = np.linspace(x_min, x_max, grid_points)
    y = np.linspace(y_min, y_max, grid_points)
    z = np.linspace(z_min, z_max, grid_points)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    print("Calculando densidade em grade...")
    density = density_calc.calculate_density(points, parser.data).reshape(grid_points, grid_points, grid_points)

    print("Exportando .cube...")
    write_cube_file("density.cube", density, x, y, z, parser.data['nuclei'], parser)

    print("Criando interpoladores...")
    interp_rho, interp_grad, interp_hessian = create_interpolators(x, y, z, density)

    global GLOBAL_INTERP_GRAD, GLOBAL_INTERP_RHO, GLOBAL_INTERP_HESSIAN, GLOBAL_X, GLOBAL_Y, GLOBAL_Z
    GLOBAL_INTERP_GRAD = interp_grad
    GLOBAL_INTERP_RHO = interp_rho
    GLOBAL_INTERP_HESSIAN = interp_hessian
    GLOBAL_X = x
    GLOBAL_Y = y
    GLOBAL_Z = z

    print("Gerando pontos iniciais entre pares de átomos...")
    sample_points = generate_initial_points_between_atoms(parser.data['nuclei'], n_points_per_pair=n_points_per_pair)

    print(f"Iniciando seguimento de gradiente ({len(sample_points)} pontos)...")
    from tqdm import tqdm
    with parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(follow_gradient_optimized_thread)(p, grad_tol, min_density)
            for p in tqdm(sample_points, desc="Buscando BCPs", unit="ponto")
        )

    found_points = [r for r in results if r is not None]
    print(f"Filtrando {len(found_points)} pontos redundantes...")
    filtered = filter_close_points(found_points)
    print(f"{len(filtered)} BCPs encontrados.")
    export_bcps_to_xyz(parser, filtered)
    return filtered
