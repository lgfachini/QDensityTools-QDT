import numpy as np

from qdt.core.density import calculate_density
from qdt.core.grid import evaluate_density, resolve_padding


def integrate_electron_density(parser, ext=".wfx", grid_points=80, padding=10.0):
    """
    Integrate electron density over a 3D grid to estimate the total number of electrons.
    """
    padding = resolve_padding(parser, padding)
    coords = np.array([n["coords"] for n in parser.data["nuclei"]])
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding

    x = np.linspace(x_min, x_max, grid_points)
    y = np.linspace(y_min, y_max, grid_points)
    z = np.linspace(z_min, z_max, grid_points)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    if ext == ".wfx":
        density = calculate_density(points, parser.data)
    elif ext == ".cube":
        density = evaluate_density(parser, points, ext=".cube")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Must be '.wfx' or '.cube'.")

    density = density.reshape(grid_points, grid_points, grid_points)
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    dz = abs(z[1] - z[0])
    return float(np.sum(density) * dx * dy * dz)
