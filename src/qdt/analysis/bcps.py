import os
from itertools import combinations

import numpy as np
import scipy.ndimage
from joblib import Parallel, delayed, parallel_backend
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eigh
from scipy.optimize import minimize
from tqdm import tqdm

from qdt.core.density import calculate_density
from qdt.core.grid import evaluate_density, resolve_padding
from qdt.core.periodic_table import get_atomic_number
from qdt.io.cube import write_cube_file

BOHR_TO_ANGSTROM = 0.52917721067


class BCPSearchContext:
    """Interpolator context for parallel BCP gradient-following."""

    def __init__(self, x, y, z, interp_rho, interp_grad, interp_hessian):
        self.x = x
        self.y = y
        self.z = z
        self.interp_rho = interp_rho
        self.interp_grad = interp_grad
        self.interp_hessian = interp_hessian

    def within_bounds(self, p):
        return (
            self.x[0] <= p[0] <= self.x[-1]
            and self.y[0] <= p[1] <= self.y[-1]
            and self.z[0] <= p[2] <= self.z[-1]
        )


def create_interpolators(x, y, z, density):
    """Interpolators for rho, gradient, and symmetric Hessian components."""
    print("-> Creating interpolators for density, gradient, and Hessian...")
    interp_density = RegularGridInterpolator(
        (x, y, z), density, bounds_error=False, fill_value=np.nan
    )

    drho_dx, drho_dy, drho_dz = np.gradient(density, x, y, z, edge_order=2)
    interp_grad = [
        RegularGridInterpolator((x, y, z), g, bounds_error=False, fill_value=np.nan)
        for g in (drho_dx, drho_dy, drho_dz)
    ]

    hess_components = []
    for dcomp in (drho_dx, drho_dy, drho_dz):
        hess_components.append(np.gradient(dcomp, x, axis=0))
        hess_components.append(np.gradient(dcomp, y, axis=1))
        hess_components.append(np.gradient(dcomp, z, axis=2))

    interp_hessian = [
        RegularGridInterpolator((x, y, z), h, bounds_error=False, fill_value=np.nan)
        for h in hess_components
    ]
    print("Interpolators created.")
    return interp_density, interp_grad, interp_hessian


def follow_gradient_to_bcp(p0, ctx, grad_tol, min_density, max_iter=300):
    """Follow |grad rho| toward zero and classify bond critical points."""

    def objective(p):
        if not ctx.within_bounds(p):
            return 1e6
        grad = np.array([g(p).item() for g in ctx.interp_grad])
        if np.any(np.isnan(grad)):
            return 1e6
        return float(np.dot(grad, grad))

    res = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        tol=grad_tol,
        options={"maxiter": max_iter, "gtol": grad_tol},
    )
    if not res.success:
        return None

    p = res.x
    if not ctx.within_bounds(p):
        return None

    rho_val = float(ctx.interp_rho(p).item())
    if rho_val < min_density or np.isnan(rho_val):
        return None

    hvals = [hfun(p).item() for hfun in ctx.interp_hessian]
    if np.any(np.isnan(hvals)):
        return None

    hessian = np.array(
        [
            [hvals[0], hvals[1], hvals[2]],
            [hvals[3], hvals[4], hvals[5]],
            [hvals[6], hvals[7], hvals[8]],
        ]
    )
    hessian = 0.5 * (hessian + hessian.T)
    eigvals = np.sort(eigh(hessian, eigvals_only=True))

    n_neg = int(np.sum(eigvals < -grad_tol))
    n_pos = int(np.sum(eigvals > grad_tol))
    if n_neg == 2 and n_pos == 1:
        return {"position": p, "rho": rho_val, "eigvals": eigvals, "type": "BCP"}
    return None


def generate_initial_points_between_atoms(
    nuclei, n_points_per_pair=30, max_distance_bohr=6.0, t_min=0.15, t_max=0.85
):
    """Sample points along lines between nearby atom pairs."""
    points = []
    for a1, a2 in combinations(nuclei, 2):
        p1 = np.array(a1["coords"], dtype=float)
        p2 = np.array(a2["coords"], dtype=float)
        if np.linalg.norm(p2 - p1) > max_distance_bohr:
            continue
        for t in np.linspace(t_min, t_max, n_points_per_pair):
            points.append((1.0 - t) * p1 + t * p2)
    return np.array(points) if points else np.empty((0, 3))


def filter_close_points(points, threshold=0.2):
    """Keep distinct BCPs; when clustered, retain the one with highest density."""
    if not points:
        return []
    points = sorted(points, key=lambda p: p["rho"], reverse=True)
    kept = []
    kept_pos = []
    for p in points:
        pos = p["position"]
        if all(np.linalg.norm(pos - kp) >= threshold for kp in kept_pos):
            kept.append(p)
            kept_pos.append(pos)
    return kept


def ensure_atomic_numbers(nuclei):
    for atom in nuclei:
        if "atomic_number" in atom:
            continue
        symbol = atom.get("symbol")
        if not symbol:
            raise ValueError("Atom without symbol; cannot determine atomic number.")
        atom["atomic_number"] = get_atomic_number(symbol)


def export_bcps_to_xyz(parser, bcp_list, filename="BCPs.xyz"):
    print("-> Exporting BCPs to .xyz file...")
    path = os.path.join(os.path.dirname(parser.filename), filename)
    with open(path, "w", encoding="utf-8") as f:
        total = len(parser.data["nuclei"]) + len(bcp_list)
        f.write(f"{total}\n")
        f.write("Molecule + Bond Critical Points (BCPs)\n")
        for atom in parser.data["nuclei"]:
            symbol = atom.get("symbol", "X")
            x_, y_, z_ = np.array(atom["coords"]) * BOHR_TO_ANGSTROM
            f.write(f"{symbol} {x_:.6f} {y_:.6f} {z_:.6f}\n")
        for bcp in bcp_list:
            x_, y_, z_ = np.array(bcp["position"]) * BOHR_TO_ANGSTROM
            f.write(f"X {x_:.6f} {y_:.6f} {z_:.6f}\n")
    print(f"Saved {path}")


def find_critical_points_from_gradient_flow(
    parser,
    ext,
    grid_points=80,
    padding=10.0,
    grad_tol=1e-4,
    min_density=1e-4,
    n_jobs=-1,
    n_points_per_pair=30,
    max_distance_bohr=6.0,
    duplicate_threshold=0.2,
    max_optimizer_iter=300,
    export_cube=True,
    cube_filename="density.cube",
):
    """
    Find bond critical points (BCPs) by gradient flow on a 3D density grid.

    Parameters
    ----------
    padding : float or None
        Box padding around nuclei (Bohr). None = automatic.
    max_distance_bohr : float
        Max interatomic distance (Bohr) for initial seed segments.
    duplicate_threshold : float
        Merge BCPs closer than this distance (Bohr); keep higher rho.
    """
    print(f"\n===> Starting BCP search ({ext})")

    padding = resolve_padding(parser, padding)
    coords = np.array([n["coords"] for n in parser.data["nuclei"]])
    bounds_min = coords.min(axis=0) - padding
    bounds_max = coords.max(axis=0) + padding

    x = np.linspace(bounds_min[0], bounds_max[0], grid_points)
    y = np.linspace(bounds_min[1], bounds_max[1], grid_points)
    z = np.linspace(bounds_min[2], bounds_max[2], grid_points)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    if ext == ".wfx":
        print("-> Calculating density on grid...")
        density = calculate_density(points, parser.data).reshape(
            (grid_points, grid_points, grid_points)
        )
    elif ext == ".cube":
        print("-> Interpolating density from .cube...")
        density = evaluate_density(parser, points, ext=".cube").reshape(
            (grid_points, grid_points, grid_points)
        )
    else:
        raise ValueError(f"File extension '{ext}' not supported for BCP search.")

    ensure_atomic_numbers(parser.data["nuclei"])

    if export_cube:
        print(f"-> Exporting density grid to {cube_filename}...")
        write_cube_file(cube_filename, density, x, y, z, parser.data["nuclei"], parser)

    interp_rho, interp_grad, interp_hessian = create_interpolators(x, y, z, density)
    ctx = BCPSearchContext(x, y, z, interp_rho, interp_grad, interp_hessian)

    sample_points = generate_initial_points_between_atoms(
        parser.data["nuclei"],
        n_points_per_pair=n_points_per_pair,
        max_distance_bohr=max_distance_bohr,
    )
    if len(sample_points) == 0:
        print("No initial points generated (increase max_distance_bohr or check geometry).")
        export_bcps_to_xyz(parser, [])
        return []

    print(f"-> Gradient following from {len(sample_points)} seeds...")
    with parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(follow_gradient_to_bcp)(
                p, ctx, grad_tol, min_density, max_optimizer_iter
            )
            for p in tqdm(sample_points, desc="Searching BCPs", unit="point")
        )

    found = [r for r in results if r is not None]
    print(f"-> Merging {len(found)} candidates (threshold={duplicate_threshold} Bohr)...")
    filtered = filter_close_points(found, threshold=duplicate_threshold)
    print(f"{len(filtered)} BCP(s) retained.")
    export_bcps_to_xyz(parser, filtered)
    return filtered
