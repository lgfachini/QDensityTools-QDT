import numpy as np
import matplotlib.pyplot as plt
import os
import density_calc
import rdg_calc
import s_sign_lambda2_rho_p
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import scipy.ndimage

EPS = 1e-15
BOHR_TO_ANGSTROM = 0.52917721067  # Conversion factor from Bohr to Angstrom units

def _coords_to_angstrom(coords_bohr):
    """
    Convert coordinates from Bohr to Angstrom units.
    
    Parameters:
        coords_bohr (array-like): Coordinates in Bohr.
    
    Returns:
        numpy.ndarray: Coordinates converted to Angstrom.
    """
    return coords_bohr * BOHR_TO_ANGSTROM

def _generate_grid(parser, plane, z_pos, grid_points, padding=None):
    """
    Generate a 2D grid of points lying on a specified plane (xy, xz, or yz)
    at a fixed coordinate value z_pos (in Bohr).
    
    The grid bounds are automatically set based on nuclear coordinates plus optional padding.
    
    Parameters:
        parser (object): Parsed molecular data containing nuclei coordinates.
        plane (str): Plane on which to generate grid; must be 'xy', 'xz' or 'yz'.
        z_pos (float): Coordinate value along the axis perpendicular to the plane (in Bohr).
        grid_points (int): Number of points per dimension on the grid.
        padding (float, optional): Padding around molecular bounds (in Bohr). If None, computed automatically.
        
    Returns:
        tuple: (xx, yy, points)
            xx, yy (ndarray): 2D meshgrid arrays of grid coordinates on the plane.
            points (ndarray): Flattened Nx3 array of 3D Cartesian coordinates of all grid points.
    """
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    
    if padding is None:
        span = coords.max(axis=0) - coords.min(axis=0)
        padding = max(0.15 * np.linalg.norm(span), 4.0)  # Minimum padding to ensure grid covers molecule well
    
    # Define bounding box with padding
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding
    
    if plane == 'xy':
        x = np.linspace(x_min, x_max, grid_points)
        y = np.linspace(y_min, y_max, grid_points)
        xx, yy = np.meshgrid(x, y)
        # Points all at fixed z_pos
        points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_pos)])
    elif plane == 'xz':
        x = np.linspace(x_min, x_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        xx, yy = np.meshgrid(x, z)
        # Points all at fixed y = z_pos here
        points = np.column_stack([xx.ravel(), np.full(xx.size, z_pos), yy.ravel()])
    elif plane == 'yz':
        y = np.linspace(y_min, y_max, grid_points)
        z = np.linspace(z_min, z_max, grid_points)
        xx, yy = np.meshgrid(y, z)
        # Points all at fixed x = z_pos
        points = np.column_stack([np.full(xx.size, z_pos), xx.ravel(), yy.ravel()])
    else:
        raise ValueError("Invalid plane. Use 'xy', 'xz', or 'yz'.")
    
    return xx, yy, points

def _generate_custom_plane_grid(parser, atom_indices, grid_points, padding=None):
    """
    Generate a 2D grid of points on a custom plane defined by three atoms.
    
    The plane is defined by three atoms indexed by atom_indices, and grid is constructed
    spanning this plane with padding around projected nuclear positions.
    
    Parameters:
        parser (object): Parsed molecular data containing nuclei coordinates.
        atom_indices (list of int): Indices of three atoms defining the plane.
        grid_points (int): Number of points per dimension on the grid.
        padding (float, optional): Padding around projected atomic coordinates (in Bohr).
        
    Returns:
        tuple: (uu, vv, points, x_axis, y_axis, origin_point)
            uu, vv (ndarray): 2D meshgrid arrays in plane coordinates.
            points (ndarray): Nx3 array of 3D Cartesian coordinates for grid points on the plane.
            x_axis, y_axis (ndarray): Unit vectors spanning the plane.
            origin_point (ndarray): Cartesian coordinate of plane origin (first atom).
    """
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    p1, p2, p3 = coords[atom_indices[0]], coords[atom_indices[1]], coords[atom_indices[2]]

    # Define vectors spanning the plane
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)  # Normalize plane normal

    x_axis = v1 / np.linalg.norm(v1)  # First axis in plane
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)  # Second axis orthogonal to x_axis in plane

    # Project all atomic coordinates onto plane coordinates
    proj_coords = []
    for coord in coords:
        rel = coord - p1
        x_proj = np.dot(rel, x_axis)
        y_proj = np.dot(rel, y_axis)
        proj_coords.append([x_proj, y_proj])
    proj_coords = np.array(proj_coords)

    # Determine grid bounds from projections, adding padding
    x_min, y_min = proj_coords.min(axis=0)
    x_max, y_max = proj_coords.max(axis=0)

    if padding is None:
        padding = max(0.15 * np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)), 4.0)

    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Create 2D meshgrid on plane
    u = np.linspace(x_min, x_max, grid_points)
    v = np.linspace(y_min, y_max, grid_points)
    uu, vv = np.meshgrid(u, v)

    # Calculate Cartesian coordinates of all grid points on the plane
    points = p1 + np.outer(uu.ravel(), x_axis) + np.outer(vv.ravel(), y_axis)

    return uu, vv, points, x_axis, y_axis, p1

def plot_density_gradient_laplacian_along_path(parser, atom1_index, atom2_index, points_count=700, ext='.wfx'):
    """
    Plot the logarithm of electron density (ρ), gradient magnitude (|∇ρ|), 
    and Laplacian (∇²ρ) along the straight line connecting two atoms.
    
    The sampling extends slightly beyond both atoms internally (-10% extension) 
    for smoother interpolation, but only the segment strictly between atoms is plotted.
    
    Parameters:
        parser (object): Parsed data object with wavefunction or cube data.
        atom1_index (int): Index of the first atom in parser.data['nuclei'].
        atom2_index (int): Index of the second atom.
        points_count (int): Number of sampling points along the extended path.
        ext (str): File extension of data source, either '.wfx' or '.cube'.
    """
    atom1 = parser.data['nuclei'][atom1_index]
    atom2 = parser.data['nuclei'][atom2_index]
    coords1 = atom1['coords']
    coords2 = atom2['coords']

    vec = coords2 - coords1
    total_dist = np.linalg.norm(vec)
    unit_vec = vec / total_dist
    extension = -0.1 * total_dist  # Extend inward by 10% beyond atoms for sampling

    # Create extended sampling path including extension beyond atoms
    extended_vec = (total_dist + 2 * extension) * unit_vec
    start_point = coords1 - extension * unit_vec
    extended_path_points = np.array([
        start_point + t * extended_vec for t in np.linspace(0, 1, points_count)
    ])

    # Distances along extended path (Bohr and Angstrom)
    extended_distances_bohr = np.linspace(-extension, total_dist + extension, points_count)
    extended_distances_ang = extended_distances_bohr * BOHR_TO_ANGSTROM

    def get_density_along_path_wfx():
        # Calculate electron density from wavefunction data at given points
        return density_calc.calculate_density(extended_path_points, parser.data)

    def get_density_along_path_cube():
        # Calculate electron density from cube grid data by trilinear interpolation
        origin = parser.origin
        vectors = parser.vectors
        inv_vectors = np.linalg.inv(vectors.T)
        
        rel_coords = extended_path_points - origin
        fractional_indices = rel_coords @ inv_vectors
        coords_for_interp = fractional_indices.T
        
        density_along_path = scipy.ndimage.map_coordinates(
            parser.density,
            coords_for_interp,
            order=3,
            mode='nearest'
        )
        return density_along_path

    # Get density values depending on file type
    if ext == '.wfx':
        rho_extended = get_density_along_path_wfx()
        distances_bohr = extended_distances_bohr
        distances_ang = extended_distances_ang
    elif ext == '.cube':
        rho_extended = get_density_along_path_cube()
        distances_bohr = extended_distances_bohr
        distances_ang = extended_distances_ang
    else:
        raise ValueError(f'Unsupported file extension: {ext}')

    # Keep only segment strictly between the two atoms for plotting
    mask = (distances_bohr >= 0) & (distances_bohr <= total_dist)
    rho = rho_extended[mask]
    distances_bohr = distances_bohr[mask]
    distances_ang = distances_ang[mask]

    # Compute numerical gradient (first derivative) and Laplacian (second derivative)
    grad = np.gradient(rho, distances_bohr)
    lap = np.gradient(grad, distances_bohr)

    def safe_log10(arr):
        # Compute log10 of absolute values adding EPS to avoid log(0)
        return np.log10(np.abs(arr) + EPS)

    def normalize(arr):
        # Normalize array linearly to [0,1]
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + EPS)

    # Normalize log-scaled quantities for better comparative plotting
    rho_n = normalize(safe_log10(rho))
    grad_n = normalize(safe_log10(grad))
    lap_n = normalize(safe_log10(lap))

    # Create x-axis in Angstrom from 0 to total distance
    x_axis = np.linspace(0, total_dist * BOHR_TO_ANGSTROM, len(rho_n))

    # Plot normalized log quantities
    plt.figure(figsize=(9, 6))
    plt.plot(x_axis, rho_n, label=r'$\log_{10}[\rho]$', color='black')
    plt.plot(x_axis, grad_n, label=r'$\log_{10}[|\nabla \rho|]$', color='blue', linestyle='--')
    plt.plot(x_axis, lap_n, label=r'$\log_{10}[\nabla^2 \rho]$', color='red', linestyle=':')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Normalized values')
    plt.title(rf'$\log_{{10}}[\rho, \nabla \rho, \nabla^2 \rho]$ between {atom1["symbol"]} and {atom2["symbol"]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure to same directory as input file
    filename = os.path.join(
        os.path.dirname(parser.filename),
        f'density_gradient_laplacian_log_path_{atom1["symbol"]}_{atom2["symbol"]}_{ext}.png'
    )
    plt.savefig(filename, dpi=300)
    plt.close()


def _draw_atoms(ax, parser, plane=None, z_pos=None, atom_indices=None, threshold_bohr=1.0):
    """
    Plot atomic positions as points and element symbols on a given matplotlib axis.
    
    Supports either a standard plane ('xy', 'xz', 'yz') at fixed coordinate z_pos,
    or a custom plane defined by three atoms (atom_indices).
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis to draw on.
        parser (object): Parsed molecular data.
        plane (str, optional): Plane name for standard planes.
        z_pos (float, optional): Coordinate value perpendicular to plane (in Bohr).
        atom_indices (list of int, optional): Three atom indices defining a custom plane.
        threshold_bohr (float): Maximum distance from plane to consider atoms for plotting (in Bohr).
    """
    coords_bohr = np.array([n['coords'] for n in parser.data['nuclei']])
    
    if atom_indices is not None:
        # Custom plane: compute plane normal and project atoms onto plane
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
        # Standard planes: filter atoms close to the plane coordinate and plot
        axis_map = {'xy': 2, 'xz': 1, 'yz': 0}  # axis perpendicular to plane
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
    """
    Plot a scalar field (e.g., density, gradient, laplacian) on a 2D grid with contours,
    including atoms positions on the plane.
    
    Supports both regular and custom diverging colormaps centered at zero.
    
    Parameters:
        data (ndarray): 2D array of scalar values on grid.
        xx, yy (ndarray): Meshgrid arrays defining grid coordinates.
        parser (object): Parsed molecular data for atom positions.
        suffix (str): Filename suffix for saving plot.
        title (str): Plot title.
        label (str): Colorbar label.
        cmap (str or Colormap): Colormap name or object.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        plane (str, optional): Standard plane name if any.
        z_pos (float, optional): Coordinate of plane (Bohr).
        atom_indices (list of int, optional): If custom plane, indices of atoms defining it.
        linthresh (float): Threshold for symmetric norm (not used here but reserved).
        vmin, vmax (float, optional): Data min and max values for colormap scaling.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    # Use custom diverging colormap if requested
    if cmap == 'custom_diverging':
        # Blue for negative, green at zero, red for positive
        cmap = LinearSegmentedColormap.from_list("blue-green-red", ['blue', 'green', 'red'])

        # Symmetric normalization about zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cf = ax.contourf(xx, yy, data, levels=60, cmap=cmap, norm=norm)
    else:
        cf = ax.contourf(xx, yy, data, levels=60, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.colorbar(cf, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Draw atoms on plot
    _draw_atoms(ax, parser, plane=plane, z_pos=z_pos, atom_indices=atom_indices)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(parser.filename), f"{suffix}.png"), dpi=300)
    plt.close()

def _prepare_slice(parser, plane, z_pos, grid_points, atom_indices):
    """
    Prepare the necessary data for plotting a 2D slice of scalar fields:
    generates grid points and defines axis labels and filename suffix.
    
    Supports standard planes (xy, xz, yz) and custom plane defined by three atoms.
    
    Parameters:
        parser (object): Parsed molecular data.
        plane (str): Standard plane name or None for custom.
        z_pos (float): Coordinate of plane (Bohr).
        grid_points (int): Number of points per grid axis.
        atom_indices (list of int or None): Three atom indices if custom plane.
        
    Returns:
        tuple: (xx, yy, points, xlabel, ylabel, suffix, draw_atoms_flag)
            xx, yy: meshgrid coordinate arrays.
            points: Nx3 Cartesian coordinates of grid points.
            xlabel, ylabel: axis labels for plots.
            suffix: string suffix for filenames.
            draw_atoms_flag: whether to draw atoms (False for custom plane).
    """
    if atom_indices is not None:
        # Custom plane grid generation
        xx, yy, points, _, _, _ = _generate_custom_plane_grid(parser, atom_indices, grid_points)
        xlabel, ylabel = "Custom X (Å)", "Custom Y (Å)"
        suffix = f"custom_plane_{'_'.join(map(str, atom_indices))}"
        draw_atoms = False
    else:
        # Standard plane grid generation
        xx, yy, points = _generate_grid(parser, plane, z_pos, grid_points)
        xlabel = 'X (Å)' if plane in ['xy', 'xz'] else 'Y (Å)'
        ylabel = 'Y (Å)' if plane == 'xy' else 'Z (Å)'
        zlab = _coords_to_angstrom(np.array([z_pos]))[0]
        suffix = f"{plane}_z{zlab:.2f}"
        draw_atoms = True
    return xx, yy, points, xlabel, ylabel, suffix, draw_atoms


def plot_density_slice(parser, plane='xy', z_pos=0.0, grid_points=700, atom_indices=None, ext='.wfx'):
    """
    Plot a 2D slice of the electron density in the specified plane and position.

    Parameters:
        parser : object
            Parsed data object with density grid, origin, vectors, and raw data.
        plane : str
            Plane to slice ('xy', 'yz', or 'xz').
        z_pos : float
            Position along the axis perpendicular to the plane (in Bohr).
        grid_points : int
            Number of points per dimension in the slice grid.
        atom_indices : list or None
            Optional list of atom indices to highlight in the plot.
        ext : str
            File extension indicating data source ('.wfx' or '.cube').
    """
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(parser, plane, z_pos, grid_points, atom_indices)

    if ext == '.cube':
        origin = parser.origin
        vectors = parser.vectors
        inv_vectors = np.linalg.inv(vectors.T)

        # Convert 2D slice points from real space to fractional cube grid indices
        rel_coords = points - origin
        fractional_indices = rel_coords @ inv_vectors

        # Interpolate density on cube grid and reshape to 2D slice
        density_1d = scipy.ndimage.map_coordinates(
            parser.density,
            fractional_indices.T,
            order=3,
            mode='nearest'
        )
        rho = density_1d.reshape(grid_points, grid_points)

    elif ext == '.wfx':
        rho = density_calc.calculate_density(points, parser.data, spin='total').reshape(grid_points, grid_points)

    else:
        raise ValueError(f'Unsupported file extension: {ext}')

    rho_log = np.log10(rho + EPS)
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    _plot_scalar_field(
        rho_log, xx_ang, yy_ang, parser,
        f'density_slice_{suffix}',
        'Electronic density',
        r'$\log_{10}[\rho]$',
        'viridis', xlabel, ylabel,
        plane=plane, z_pos=z_pos, atom_indices=atom_indices
    )

def plot_gradient_magnitude_slice(parser, plane='xy', z_pos=0.0, grid_points=700, atom_indices=None, ext='wfx'):
    """
    Plot the magnitude of the gradient of the electronic density (|∇ρ|) on a 2D slice.

    This function computes and visualizes the gradient of the electronic density 
    in a 2D slice defined by either a standard cartesian plane (e.g., 'xy') or a 
    custom plane defined by three atoms.

    Parameters
    ----------
    parser : object
        A parsed data object that contains information about the wavefunction (.wfx or .cube).
    plane : str, optional
        Plane to slice through. Accepts 'xy', 'yz', 'xz', or 'atom' to define a custom plane via three atoms.
    z_pos : float, optional
        Position of the slicing plane in Bohr (if using cartesian slicing).
    grid_points : int, optional
        Number of grid points along each axis of the 2D slice (default: 700).
    atom_indices : list of int, optional
        If plane='atom', this must be a list of 3 atom indices defining the custom plane.
    ext : str, optional
        Type of data source: 'wfx' or 'cube'. Used to call the correct density function.

    Returns
    -------
    None
        A 2D plot of the gradient magnitude is saved to disk with appropriate naming.

    Notes
    -----
    - The output is a heatmap of log₁₀(|∇ρ|), which visually emphasizes variations in 
      gradient magnitude.
    - Atom positions are plotted as scatter points for context.
    - The grid is generated in atomic units but plotted in Angstroms.
    """

    # Prepare the grid and coordinates in the slicing plane
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(parser, plane, z_pos, grid_points, atom_indices)

    # Compute density from appropriate file format
    if ext.lower() == '.wfx':
        rho = density_calc.calculate_density(points, parser.data)
    elif ext.lower() == '.cube':
        origin = parser.origin
        vectors = parser.vectors
        inv_vectors = np.linalg.inv(vectors.T)

        # Convert 2D slice points from real space to fractional cube grid indices
        rel_coords = points - origin
        fractional_indices = rel_coords @ inv_vectors

        # Interpolate density on cube grid and reshape to 2D slice
        density_1d = scipy.ndimage.map_coordinates(
            parser.density,
            fractional_indices.T,
            order=3,
            mode='nearest'
        )
        rho = density_1d.reshape(grid_points, grid_points)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use 'wfx' or 'cube'.")

    rho = rho.reshape(grid_points, grid_points)

    # Compute gradients in x and y direction (Bohr units)
    dx = (xx[0, 1] - xx[0, 0])
    gradx, grady = np.gradient(rho, dx, dx)
    grad_mag = np.sqrt(gradx**2 + grady**2)

    # Apply log scaling for visual clarity
    grad_log = np.log10(grad_mag + EPS)

    # Convert to Angstroms for display
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    # Plot the result
    _plot_scalar_field(
        grad_log, xx_ang, yy_ang, parser, 
        f'gradient_slice_{suffix}',
        'Electronic density gradient',
        r'$\log_{10}[|\nabla\rho|]$',
        'inferno', xlabel, ylabel,
        plane=plane, z_pos=z_pos, atom_indices=atom_indices
    )


def plot_laplacian_slice(parser, ext='.wfx', plane='xy', z_pos=0.0, grid_points=700, atom_indices=None):
    """
    Plot the logarithm of the Laplacian of the electron density on a specified 2D plane.

    Parameters
    ----------
    parser : object
        Parsed wavefunction or cube data containing necessary fields like density, grid, etc.
    ext : str, default='wfx'
        File extension type. Use 'wfx' for wavefunction format or 'cube' for precomputed cube data.
    plane : str, default='xy'
        Plane on which to take the slice. Can be 'xy', 'xz', 'yz', or 'atom' (custom plane through atoms).
    z_pos : float, default=0.0
        Position (in Bohr) of the slice plane along the orthogonal axis.
    grid_points : int, default=700
        Number of grid points along each axis in the slice.
    atom_indices : list of int or None
        If provided, defines a custom plane passing through three atoms for slicing.
    """
    # Prepare the 2D grid of points and other plot metadata
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(
        parser, plane, z_pos, grid_points, atom_indices
    )

    # Compute density from appropriate file format    
    if ext == '.wfx':
        rho = density_calc.calculate_density(points, parser.data).reshape(grid_points, grid_points)
    elif ext == '.cube':
        origin = parser.origin
        vectors = parser.vectors
        inv_vectors = np.linalg.inv(vectors.T)

        # Convert 2D slice points from real space to fractional cube grid indices
        rel_coords = points - origin
        fractional_indices = rel_coords @ inv_vectors

        # Interpolate density on cube grid and reshape to 2D slice
        density_1d = scipy.ndimage.map_coordinates(
            parser.density,
            fractional_indices.T,
            order=3,
            mode='nearest'
        )
        rho = density_1d.reshape(grid_points, grid_points)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use 'wfx' or 'cube'.")
        
    # Compute second derivatives for Laplacian
    dx = (xx[0, 1] - xx[0, 0])
    d2x = np.gradient(np.gradient(rho, axis=0), axis=0) / dx**2
    d2y = np.gradient(np.gradient(rho, axis=1), axis=1) / dx**2
    lap = d2x + d2y

    # Convert to log scale: sign(lap) * log10(|lap| + EPS)
    lap_log = np.sign(lap) * np.log10(np.abs(lap) + EPS)

    # Convert coordinates to angstroms for display
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    # Plotting the final 2D Laplacian slice
    _plot_scalar_field(
        lap_log, xx_ang, yy_ang, parser, f'laplacian_slice_{suffix}',
        'Electronic density Laplacian', r'$\log_{10}[\nabla^2\rho]$', 'seismic',
        xlabel, ylabel, plane=plane, z_pos=z_pos, atom_indices=atom_indices
    )

def plot_spin_density_slice(parser, ext='.wfx', plane='xy', z_pos=0.0, grid_points=700, atom_indices=None):
    """
    Plots a 2D slice of the spin density (ρ_alpha - ρ_beta) in a given molecular plane.

    Parameters:
        parser: object
            A parser object that must contain parsed data from a .wfx or .cube file,
            including molecular orbital coefficients and occupation numbers.
        ext: str
            File extension of the input data. Must be '.wfx' or '.cube'.
            Spin density calculation is only available for '.wfx'.
        plane: str
            Plane in which to plot the slice ('xy', 'xz', or 'yz').
        z_pos: float
            Position along the orthogonal axis where the slice is taken (in bohr).
        grid_points: int
            Number of grid points along each axis of the slice.
        atom_indices: list of int or None
            Optional list of atom indices used to determine the center or orientation of the slice.

    Notes:
        - For '.wfx' files, spin density is calculated as the difference between
          alpha and beta electron densities: ρ_alpha - ρ_beta.
        - For '.cube' files, a warning is printed because spin-resolved density
          is not available unless the cube contains p(alpha) - p(beta) explicitly.
    """
    if ext == '.wfx':
        # Prepare grid coordinates and related labels for the selected slice
        xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(
            parser, plane, z_pos, grid_points, atom_indices
        )

        # Compute alpha and beta spin densities at all grid points
        alpha = density_calc.calculate_density(points, parser.data, spin='alpha')
        beta = density_calc.calculate_density(points, parser.data, spin='beta')

        # Compute the spin density (ρ_alpha - ρ_beta) and reshape to 2D grid
        spin = (alpha - beta).reshape(grid_points, grid_points)

        # Convert coordinates from atomic units to angstroms for plotting
        xx_ang = _coords_to_angstrom(xx)
        yy_ang = _coords_to_angstrom(yy)

        # Plot the 2D scalar field using a diverging colormap
        _plot_scalar_field(
            spin, xx_ang, yy_ang, parser, f'spin_density_slice_{suffix}',
            'Spin density', r'$\rho_\alpha - \rho_\beta$', 'seismic', xlabel, ylabel,
            plane=plane, z_pos=z_pos, atom_indices=atom_indices
        )

    elif ext == '.cube':
        # Inform the user that spin density cannot be computed from a standard total electron density cube file
        print("[INFO] Spin density visualization is not supported for standard .cube files, as these typically contain only total electron density without spin resolution.")
        print("[INFO] To visualize spin density from cube data, you need to generate a separate cube file representing the difference between alpha and beta spin densities (p(alpha) - p(beta)) and then plot that as a density slice.")

    
def plot_reduced_gradient_slice(parser, plane='xy', z_pos=0.0, grid_points=600, atom_indices=None, ext='.wfx'):
    """
    Plot the reduced density gradient (RDG) on a 2D slice of the molecular system.

    Parameters
    ----------
    parser : object
        Parsed data object containing wavefunction or cube data.
    plane : str, optional
        Plane to slice through: 'xy', 'yz', 'xz', or 'atom' (custom plane defined by 3 atoms).
    z_pos : float, optional
        Position of the slicing plane in Bohr (for cartesian planes).
    grid_points : int, optional
        Number of grid points along each axis of the 2D slice.
    atom_indices : list of int, optional
        Indices of 3 atoms defining a custom slicing plane (used if plane='atom').
    ext : str, optional
        Data file type, either '.wfx' or '.cube'. Determines the calculation method.

    Returns
    -------
    None
        Saves a 2D plot of the log-scaled reduced density gradient to disk.
    """

    # Prepare the grid and points for the slice plane
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(
        parser, plane, z_pos, grid_points, atom_indices
    )
    grid_shape = (grid_points, grid_points)

    if ext == '.wfx':
        # Calculate density and gradients using wavefunction data
        rho, gx, gy, gz = rdg_calc.compute_density_and_gradient(parser, points, parser.data, grid_shape=grid_shape, ext=ext)

    elif ext == '.cube':
        # Calculate density and gradients directly from cube grid data
        rho, gx, gy, gz = rdg_calc.compute_density_and_gradient(parser, points, parser.data, grid_shape=grid_shape, ext=ext)

    else:
        raise ValueError(f"Unsupported file extension: '{ext}'. Use '.wfx' or '.cube'.")

    # Compute reduced density gradient values
    s_vals = rdg_calc.compute_s_values(rho, gx, gy, gz).reshape(grid_shape)

    # Convert coordinates to Angstrom for plotting
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    # Plot the log-scaled RDG slice
    _plot_scalar_field(
        np.log10(s_vals + EPS), xx_ang, yy_ang, parser,
        f'reduced_gradient_slice_{suffix}',
        'Reduced density gradient (s)',
        r'$\log_{10}[s]$', 'plasma', xlabel, ylabel,
        plane=plane, z_pos=z_pos, atom_indices=atom_indices
    )


def plot_s_sign_lambda2_rho_slice(parser, plane='xy', z_pos=0.0, grid_points=600, atom_indices=None, ext='.wfx'):
    """
    Plot the quantity s × sign(lambda2) × rho on a 2D slice.

    Parameters
    ----------
    parser : object
        Parsed data object with wavefunction or cube data.
    plane : str, optional
        Plane to slice through ('xy', 'yz', 'xz', or 'atom' for custom plane).
    z_pos : float, optional
        Position of the slicing plane in Bohr (for cartesian planes).
    grid_points : int, optional
        Number of points along each axis in the 2D slice.
    atom_indices : list of int, optional
        List of 3 atom indices defining custom plane (if plane='atom').
    ext : str, optional
        File type indicator ('.wfx' or '.cube').

    Returns
    -------
    None
        Saves a 2D plot of the quantity to disk.
    """

    # Prepare the grid and slice points
    xx, yy, points, xlabel, ylabel, suffix, draw_atoms = _prepare_slice(
        parser, plane, z_pos, grid_points, atom_indices
    )
    grid_shape = (grid_points, grid_points)

    if ext == '.wfx':
        # Use existing function for wfx data
        s_sign_lambda2_rho_vals, rho = s_sign_lambda2_rho_p.compute_s_sign_lambda2_times_rho(parser, points, parser.data, grid_shape, ext)

    elif ext == '.cube':
        s_sign_lambda2_rho_vals, rho = s_sign_lambda2_rho_p.compute_s_sign_lambda2_times_rho(parser, points, parser.data, grid_shape, ext)


    else:
        raise ValueError(f"Unsupported file extension: '{ext}'. Use '.wfx' or '.cube'.")

    # Convert coordinates to Angstrom for plotting
    xx_ang = _coords_to_angstrom(xx)
    yy_ang = _coords_to_angstrom(yy)

    # Plot the scalar field with a diverging colormap
    _plot_scalar_field(
        s_sign_lambda2_rho_vals, xx_ang, yy_ang, parser,
        f's_sign_lambda2_rho_slice_{suffix}',
        r'$\log_{10}[s \times \rho] \times \mathrm{sign}(\lambda_2)$',
        r'$\log_{10}[s \times \rho] \times \mathrm{sign}(\lambda_2)$ (a.u.)',
        'custom_diverging', xlabel, ylabel,
        plane=plane, z_pos=z_pos, atom_indices=atom_indices
    )

