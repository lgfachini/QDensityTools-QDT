import numpy as np
import density_calc
import scipy.ndimage

EPS = 1e-15  # Small epsilon to avoid division by zero in s-value calculation

def compute_density_and_gradient(parser, points, data, grid_shape=None, ext='.wfx'):
    """
    Compute the electron density and its gradient components on a 2D grid.

    Depending on the file extension, the density is either explicitly calculated from basis functions
    (.wfx) or interpolated from a precomputed 3D grid (.cube).

    Parameters
    ----------
    parser : object
        Parser containing molecular and density data.
        For '.wfx', used to provide basis and coordinate info.
        For '.cube', used to access density grid, origin, and lattice vectors.
    points : ndarray
        Coordinates where density is evaluated (N_points x 3).
    data : dict
        Data needed for density calculation in the case of '.wfx'.
    grid_shape : tuple, optional
        The shape of the 2D grid (nx, ny) on which to reshape density and compute gradients.
        Required to compute gradients.
    ext : str, optional
        File extension indicating data source, either '.wfx' or '.cube'. Default is '.wfx'.

    Returns
    -------
    rho_grid : ndarray
        Electron density on the 2D grid with shape `grid_shape`.
    gx, gy, gz : ndarray
        Gradient components of the density along x, y, and z directions.
        Note: gz is zero for 2D planar slices.

    Raises
    ------
    ValueError
        If an unsupported file extension is provided or grid_shape is not given for gradient calculation.
    """

    if ext == '.wfx':
        # Calculate density explicitly from basis functions at given points
        rho = density_calc.calculate_density(points, data)

    elif ext == '.cube':
        # Interpolate the precomputed density grid on the cube
        origin = parser.origin          # Origin coordinates of the cube grid
        vectors = parser.vectors        # 3x3 matrix with cube lattice vectors
        inv_vectors = np.linalg.inv(vectors.T)  # Inverse transpose for fractional indexing

        # Compute fractional indices of points relative to cube grid
        rel_coords = points - origin
        fractional_indices = rel_coords @ inv_vectors

        # Interpolate density values at fractional indices using spline interpolation
        density_1d = scipy.ndimage.map_coordinates(
            parser.density,
            fractional_indices.T,
            order=3,
            mode='nearest'
        )

        # Reshape interpolated density to the specified 2D grid shape
        rho = density_1d.reshape(grid_shape)

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if grid_shape is not None:
        # Reshape density to 2D grid for gradient calculation
        rho_grid = rho.reshape(grid_shape)

        # Use unit spacing (1.0) for gradient calculation by default
        spacing = 1.0

        # Compute numerical gradients along x and y directions on 2D grid
        gx, gy = np.gradient(rho_grid, spacing)

        # Set z-gradient to zero for 2D slices (planar)
        gz = np.zeros_like(gx)

        return rho_grid, gx, gy, gz

    else:
        raise ValueError("grid_shape must be provided for gradient computation.")


def compute_s_values(rho, gx, gy, gz):
    """
    Calculate the reduced density gradient (s) from electron density and its gradients.

    The reduced gradient is a scalar field used in non-covalent interaction (NCI) analysis.

    Parameters
    ----------
    rho : ndarray
        Electron density values on the grid.
    gx, gy, gz : ndarray
        Gradient components of the density along x, y, and z.

    Returns
    -------
    s : ndarray
        Reduced density gradient values computed as:
        s = |∇ρ| / [2 * (3π²)^(1/3) * ρ^(4/3)]
        where |∇ρ| is the magnitude of the density gradient.
    """
    # Calculate magnitude of the density gradient vector at each point
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Compute reduced density gradient, adding EPS to avoid division by zero
    s = grad_mag / (2.0 * (3.0 * np.pi**2)**(1.0/3.0) * (rho + EPS)**(4.0/3.0))

    return s
