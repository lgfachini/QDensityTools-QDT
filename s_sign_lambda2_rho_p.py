import numpy as np
import density_calc
import rdg_calc
from scipy.ndimage import gaussian_filter
from numpy.linalg import eigvalsh
import scipy.ndimage

EPS = 1e-15

def compute_density_gradient_hessian(parser, points, data, grid_shape, ext='.wfx', padding=None):
    """
    Calculate electron density, its gradient and 2D Hessian (second derivatives).

    Parameters
    ----------
    parser : object
        Parsed data containing coordinates and density grids.
    points : ndarray
        Points where the density is evaluated.
    data : dict
        Data needed for density calculation.
    grid_shape : tuple
        Shape (nx, ny) of the 2D grid.
    ext : str, optional
        Data source type: '.wfx' or '.cube'. Different logic applies for each.
    padding : float or None, optional
        Padding around atomic coordinates for grid limits.

    Returns
    -------
    rho : ndarray
        Electron density on the grid.
    gx, gy : ndarray
        Gradient components of density.
    dxx, dyy, dxy : ndarray
        Second derivatives (Hessian components).
    """
    if ext == '.wfx':
        # Use existing logic (evaluate density directly)
        rho = density_calc.calculate_density(points, data).reshape(grid_shape)

    elif ext == '.cube':
        # Interpolate density on cube grid
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
        rho = density_1d.reshape(grid_shape)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    if padding is None:
        span = coords.max(axis=0) - coords.min(axis=0)
        padding = max(0.15 * np.linalg.norm(span), 4.0)

    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding
    nx, ny = grid_shape

    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    print("The gradient grid mesh in bohr:", round(dx,4),"by", round(dy,4))

    # First order gradients
    gx, gy = np.gradient(rho, dx, dy)

    # Second order derivatives (Hessian components) using gaussian_filter
    dxx = gaussian_filter(rho, sigma=1, order=(2, 0), mode='nearest')
    dyy = gaussian_filter(rho, sigma=1, order=(0, 2), mode='nearest')
    dxy = gaussian_filter(rho, sigma=1, order=(1, 1), mode='nearest')

    return rho, gx, gy, dxx, dyy, dxy


def compute_lambda2_sign(dxx, dyy, dxy):
    """
    Compute sign(λ2), where λ2 is the largest eigenvalue of the 2x2 Hessian matrix at each point.
    """
    shape = dxx.shape
    sign_lambda2 = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            H = np.array([[dxx[i, j], dxy[i, j]],
                          [dxy[i, j], dyy[i, j]]])
            ev = np.sort(eigvalsh(H))  # sort eigenvalues ascending
            lambda2 = ev[1]  # largest eigenvalue in 2D
            sign_lambda2[i, j] = np.sign(lambda2)

    return sign_lambda2


def compute_s_sign_lambda2_times_rho(parser, points, data, grid_shape, ext='.wfx', vmin=None, vmax=None):
    """
    Compute the quantity s * sign(lambda2) * rho for a 2D slice,
    where s is the reduced density gradient, lambda2 is the largest eigenvalue
    of the Hessian matrix, and rho is the electron density.

    Parameters
    ----------
    parser : object
        Parsed data with coordinates and density grids.
    points : ndarray
        Points at which to evaluate.
    data : dict
        Data needed for density calculation.
    grid_shape : tuple
        Shape of 2D grid.
    ext : str, optional
        Data source type: '.wfx' or '.cube'.
    vmin, vmax : float, optional
        Minimum and maximum values for normalization (not used here).

    Returns
    -------
    result : ndarray
        The computed quantity s * sign(lambda2) * log10(rho).
    rho : ndarray
        The electron density on the grid.
    """
    rho, gx, gy, dxx, dyy, dxy = compute_density_gradient_hessian(parser, points, data, grid_shape, ext=ext)
    gz = np.zeros_like(gx)  # zero z-component for compatibility
    s = rdg_calc.compute_s_values(rho, gx, gy, gz)
    sign_lambda2 = compute_lambda2_sign(dxx, dyy, dxy)
    result = sign_lambda2 * np.log10(rho * s + EPS)  # avoid log(0)

    return result, rho
