import numpy as np
import density_calc
from scipy.interpolate import RegularGridInterpolator

def integrate_electron_density(parser, ext=".wfx", grid_points=80, padding=10.0):
    """
    Integrates the electron density over a 3D grid to estimate the total number of electrons.

    For `.wfx`: computes density explicitly using basis functions on a user-defined grid.
    For `.cube`: interpolates the cube density onto a uniform grid around the molecule and integrates.

    Parameters
    ----------
    parser : object
        For `.wfx`: must contain `parser.data['nuclei']` and compatible with density_calc.
        For `.cube`: must have `density` (3D ndarray), `origin` (3-vector), `vectors` (3x3 matrix).
    ext : str, optional
        File extension: ".wfx" or ".cube" (default: ".wfx").
    grid_points : int, optional
        Number of points per dimension in the integration grid.
    padding : float, optional
        Extra padding around molecular coordinates for integration box (in Å or Bohr).

    Returns
    -------
    float
        Total number of electrons integrated over the volume.
    """

    if ext == ".wfx":
        coords = np.array([n['coords'] for n in parser.data['nuclei']])
        x_min, y_min, z_min = coords.min(axis=0) - padding
        x_max, y_max, z_max = coords.max(axis=0) + padding

        x = np.linspace(x_min, x_max, grid_points)
        y = np.linspace(y_min, y_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

        density = density_calc.calculate_density(points, parser.data).reshape(grid_points, grid_points, grid_points)

        dx = np.abs(x[1] - x[0])
        dy = np.abs(y[1] - y[0])
        dz = np.abs(z[1] - z[0])
        voxel_volume = dx * dy * dz

        n_electrons = np.sum(density) * voxel_volume

    elif ext == ".cube":
        density = parser.density
        origin = parser.origin        # shape (3,)
        vectors = parser.vectors      # shape (3,3)
        nx, ny, nz = density.shape

        # Coordenadas originais do cube
        x_orig = origin[0] + np.arange(nx) * vectors[0, 0]
        y_orig = origin[1] + np.arange(ny) * vectors[1, 1]
        z_orig = origin[2] + np.arange(nz) * vectors[2, 2]

        # Interpolador da densidade original do cube
        interpolator = RegularGridInterpolator((x_orig, y_orig, z_orig), density, bounds_error=False, fill_value=0.0)

        # Para o grid de integração: cria box ao redor da molécula com padding (usando coords dos núcleos)
        coords = np.array([n['coords'] for n in parser.data['nuclei']])
        x_min, y_min, z_min = coords.min(axis=0) - padding
        x_max, y_max, z_max = coords.max(axis=0) + padding

        x = np.linspace(x_min, x_max, grid_points)
        y = np.linspace(y_min, y_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

        # Avalia densidade interpolada no novo grid
        density_interp = interpolator(points).reshape(grid_points, grid_points, grid_points)

        dx = np.abs(x[1] - x[0])
        dy = np.abs(y[1] - y[0])
        dz = np.abs(z[1] - z[0])
        voxel_volume = dx * dy * dz

        # Integra densidade interpolada
        n_electrons = np.sum(density_interp) * voxel_volume

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Must be '.wfx' or '.cube'.")

    return n_electrons
