import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.linalg import eigh
from joblib import Parallel, delayed, parallel_backend
from itertools import combinations
from cube import write_cube_file
from periodic_table import get_atomic_number
import density_calc
import os
import scipy.ndimage

BOHR_TO_ANGSTROM = 0.52917721067

# Global interpolators for density, gradient, and Hessian
GLOBAL_INTERP_GRAD = None
GLOBAL_INTERP_RHO = None
GLOBAL_INTERP_HESSIAN = None
GLOBAL_X = None
GLOBAL_Y = None
GLOBAL_Z = None

def point_within_bounds(p, x, y, z):
    """
    Check if a 3D point p is within the grid bounds defined by arrays x, y, z.

    Parameters:
        p (array-like): Point coordinates (x, y, z).
        x, y, z (arrays): 1D arrays defining the grid axes.

    Returns:
        bool: True if point is inside bounds, else False.
    """
    return (x[0] <= p[0] <= x[-1]) and (y[0] <= p[1] <= y[-1]) and (z[0] <= p[2] <= z[-1])

def create_interpolators(x, y, z, density):
    """
    Create interpolators for electron density, its gradient components, and Hessian matrix components.

    Parameters:
        x, y, z (arrays): 1D arrays of grid points along each axis.
        density (3D array): Electron density values on the grid.

    Returns:
        tuple: (interp_density, interp_gradient_list, interp_hessian_list)
    """
    print("→ Creating interpolators for density, gradient, and Hessian...")
    interp_density = RegularGridInterpolator((x, y, z), density, bounds_error=False, fill_value=np.nan)

    # Compute gradient components of density
    drho_dx, drho_dy, drho_dz = np.gradient(density, x, y, z, edge_order=2)
    interp_grad = [
        RegularGridInterpolator((x, y, z), drho_dx, bounds_error=False, fill_value=np.nan),
        RegularGridInterpolator((x, y, z), drho_dy, bounds_error=False, fill_value=np.nan),
        RegularGridInterpolator((x, y, z), drho_dz, bounds_error=False, fill_value=np.nan)
    ]

    # Compute Hessian matrix components (second derivatives)
    hess = []
    for d in [drho_dx, drho_dy, drho_dz]:
        hess.append(np.gradient(d, x, axis=0))
        hess.append(np.gradient(d, y, axis=1))
        hess.append(np.gradient(d, z, axis=2))
    interp_hessian = [RegularGridInterpolator((x, y, z), h, bounds_error=False, fill_value=np.nan) for h in hess]

    print("✓ Interpolators created.")
    return interp_density, interp_grad, interp_hessian

def follow_gradient_optimized_thread(p0, grad_tol, min_density):
    """
    Follow the gradient flow from an initial point to find a Bond Critical Point (BCP).

    Parameters:
        p0 (array-like): Initial 3D position.
        grad_tol (float): Gradient tolerance for convergence.
        min_density (float): Minimum density threshold to accept critical points.

    Returns:
        dict or None: Dictionary with BCP info if found, else None.
    """
    global GLOBAL_INTERP_GRAD, GLOBAL_INTERP_RHO, GLOBAL_INTERP_HESSIAN, GLOBAL_X, GLOBAL_Y, GLOBAL_Z

    def objective(p):
        # Penalize points outside bounds or with invalid gradient values
        if not point_within_bounds(p, GLOBAL_X, GLOBAL_Y, GLOBAL_Z):
            return 1e6
        grad = np.array([g(p).item() for g in GLOBAL_INTERP_GRAD])
        if np.any(np.isnan(grad)):
            return 1e6
        return np.dot(grad, grad)  # Minimize gradient norm squared

    # Minimize the objective function starting at p0
    res = minimize(objective, p0, method='BFGS', tol=grad_tol, options={'maxiter': 200, 'gtol': grad_tol})
    if not res.success:
        return None

    p = res.x
    if not point_within_bounds(p, GLOBAL_X, GLOBAL_Y, GLOBAL_Z):
        return None

    rho_val = GLOBAL_INTERP_RHO(p).item()
    if rho_val < min_density or np.isnan(rho_val):
        return None

    # Retrieve Hessian values at point p
    h = [hfun(p).item() for hfun in GLOBAL_INTERP_HESSIAN]
    if np.any(np.isnan(h)):
        return None

    hessian = np.array([[h[0], h[1], h[2]],
                        [h[3], h[4], h[5]],
                        [h[6], h[7], h[8]]])
    eigvals = np.sort(eigh(hessian, eigvals_only=True))

    # Check if Hessian eigenvalues correspond to BCP (2 negative, 1 positive eigenvalue)
    if np.sum(eigvals < 0) == 2 and np.sum(eigvals > 0) == 1:
        return {'position': p, 'rho': rho_val, 'eigvals': eigvals, 'type': 'BCP'}
    return None

def generate_initial_points_between_atoms(nuclei, n_points_per_pair=30, max_distance_bohr=6.0):
    """
    Generate initial sample points along lines between pairs of nuclei.

    Parameters:
        nuclei (list): List of nuclei dictionaries with 'coords' keys.
        n_points_per_pair (int): Number of points to sample between each atom pair.
        max_distance_bohr (float): Maximum interatomic distance to consider (in Bohr).

    Returns:
        np.ndarray: Array of sampled points.
    """
    points = []
    for a1, a2 in combinations(nuclei, 2):
        p1 = np.array(a1['coords'])
        p2 = np.array(a2['coords'])
        if np.linalg.norm(p2 - p1) > max_distance_bohr:
            continue
        for t in np.linspace(0.2, 0.8, n_points_per_pair):
            pt = (1 - t) * p1 + t * p2
            points.append(pt)
    return np.array(points)

def filter_close_points(points, threshold=0.2):
    """
    Filter out points that are closer than a threshold distance, keeping only one representative.

    Parameters:
        points (list): List of dicts with 'position' keys.
        threshold (float): Minimum allowed distance between points.

    Returns:
        list: Filtered list of points.
    """
    if not points:
        return []
    coords = np.array([p['position'] for p in points])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(threshold)
    mask = np.ones(len(points), dtype=bool)
    for i, j in pairs:
        mask[j] = False  # Remove one of the close pair
    return [p for i, p in enumerate(points) if mask[i]]

def export_bcps_to_xyz(parser, bcp_list, filename='BCPs.xyz'):
    """
    Export atoms and Bond Critical Points (BCPs) to an XYZ format file.

    Parameters:
        parser (object): Parser object containing nuclei data.
        bcp_list (list): List of BCP dictionaries with positions.
        filename (str): Output filename (default 'BCPs.xyz').
    """
    print("→ Exporting BCPs to .xyz file...")
    output_dir = os.path.dirname(parser.filename)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        total = len(parser.data['nuclei']) + len(bcp_list)
        f.write(f"{total}\n")
        f.write("Molecule + Bond Critical Points (BCPs)\n")
        # Write atoms
        for atom in parser.data['nuclei']:
            symbol = atom.get('symbol', 'X')
            x_, y_, z_ = np.array(atom['coords']) * BOHR_TO_ANGSTROM
            f.write(f"{symbol} {x_:.6f} {y_:.6f} {z_:.6f}\n")
        # Write BCPs (symbol 'X')
        for bcp in bcp_list:
            x_, y_, z_ = np.array(bcp['position']) * BOHR_TO_ANGSTROM
            f.write(f"X {x_:.6f} {y_:.6f} {z_:.6f}\n")
    print("✓ Export complete.")

def find_critical_points_from_gradient_flow(parser, ext, grid_points=80, grad_tol=1e-4, min_density=1e-4,
                                            n_jobs=-1, n_points_per_pair=30):
    """
    Main routine to find Bond Critical Points (BCPs) by following gradient flow of electron density.

    Parameters:
        parser (object): Wavefunction or cube parser with nuclei and density data.
        ext (str): File extension to distinguish .wfx or .cube input.
        grid_points (int): Number of points per axis for 3D grid.
        grad_tol (float): Gradient norm tolerance for convergence.
        min_density (float): Minimum electron density to accept BCP.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).
        n_points_per_pair (int): Number of initial points between each atom pair.

    Returns:
        list: Filtered list of found BCPs.
    """
    print(f"\n===> Starting BCP search for '{ext}'")

    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    padding = 10.0
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding

    x = np.linspace(x_min, x_max, grid_points)
    y = np.linspace(y_min, y_max, grid_points)
    z = np.linspace(z_min, z_max, grid_points)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Calculate or interpolate density on the grid depending on file extension
    if ext == '.wfx':
        print("→ Calculating density on grid using density_calc...")
        density = density_calc.calculate_density(points, parser.data).reshape(grid_points, grid_points, grid_points)

    elif ext == '.cube':
        print("→ Interpolating density from .cube file...")
        origin = parser.origin
        vectors = parser.vectors
        inv_vectors = np.linalg.inv(vectors.T)
        rel_coords = points - origin
        fractional_indices = rel_coords @ inv_vectors
        density_1d = scipy.ndimage.map_coordinates(
            parser.density,
            fractional_indices.T,
            order=3,
            mode='nearest'
        )
        density = density_1d.reshape(grid_points, grid_points, grid_points)

    else:
        raise ValueError(f"File extension '{ext}' not supported for BCP search.")

    # Ensure atomic numbers are set in nuclei data, converting from symbols if necessary
    for atom in parser.data['nuclei']:
        if 'atomic_number' not in atom:
            symbol = atom.get('symbol')
            if symbol is None:
                raise ValueError("Atom without symbol, cannot determine atomic number.")
            atomic_number = get_atomic_number(symbol)
            if atomic_number is None:
                raise ValueError(f"Symbol '{symbol}' not recognized in periodic table.")
            atom['atomic_number'] = atomic_number

    print("→ Exporting density grid to density.cube...")
    write_cube_file("density.cube", density, x, y, z, parser.data['nuclei'], parser)

    # Create interpolators for density and derivatives
    interp_rho, interp_grad, interp_hessian = create_interpolators(x, y, z, density)

    # Assign to global variables for threaded access
    global GLOBAL_INTERP_GRAD, GLOBAL_INTERP_RHO, GLOBAL_INTERP_HESSIAN, GLOBAL_X, GLOBAL_Y, GLOBAL_Z
    GLOBAL_INTERP_GRAD = interp_grad
    GLOBAL_INTERP_RHO = interp_rho
    GLOBAL_INTERP_HESSIAN = interp_hessian
    GLOBAL_X = x
    GLOBAL_Y = y
    GLOBAL_Z = z

    print("→ Generating initial sample points between atom pairs...")
    sample_points = generate_initial_points_between_atoms(parser.data['nuclei'], n_points_per_pair=n_points_per_pair)

    print(f"→ Starting gradient following for {len(sample_points)} initial points...")
    from tqdm import tqdm
    with parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(follow_gradient_optimized_thread)(p, grad_tol, min_density)
            for p in tqdm(sample_points, desc="↪ Searching BCPs", unit="point")
        )

    found_points = [r for r in results if r is not None]
    print(f"→ Filtering {len(found_points)} redundant points...")
    filtered = filter_close_points(found_points)
    print(f"✓ {len(filtered)} BCPs found.")
    export_bcps_to_xyz(parser, filtered)
    return filtered
