import numpy as np
import density_calc

def integrate_electron_density(parser, grid_points=80, padding=10.0):
    """
    Calcula a densidade eletrônica em uma grade 3D e integra para obter o número total de elétrons.

    Args:
        parser : objeto com dados da molécula (parser.data)
        grid_points : int, número de pontos por dimensão da grade (default=80)
        padding : float, distância extra (Å) ao redor da molécula para a grade (default=3.0)

    Returns:
        float: número total de elétrons integrado da densidade eletrônica.
    """

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

    return n_electrons
