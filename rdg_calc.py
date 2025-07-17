import numpy as np
import density_calc

EPS = 1e-15


def compute_density_and_gradient(points, data, grid_shape=None):
    rho = density_calc.calculate_density(points, data)

    if grid_shape is not None:
        rho_grid = rho.reshape(grid_shape)
        spacing = 1.0  # assume 1 Bohr unless otherwise handled externally
        gx, gy = np.gradient(rho_grid, spacing)
        gz = np.zeros_like(gx)
        return rho_grid, gx, gy, gz  # everything 2d
    else:
        raise ValueError("grid_shape must be provided for gradient computation.")


def compute_s_values(rho, gx, gy, gz):
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    s = grad_mag / (2.0 * (3.0 * np.pi**2)**(1.0/3.0) * (rho + EPS)**(4.0/3.0))
    return s