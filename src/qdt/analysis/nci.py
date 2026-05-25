"""
2D-slice NCI analysis helpers.

For volumetric plots on a 2D cut through 3D density, use plotter.plot_s_sign_lambda2_rho_slice.
This module exposes the same 2D in-plane mathematics for standalone scripts.
"""

import numpy as np
from qdt.analysis import slice as slice_analysis
from qdt.core.density import calculate_density
from qdt.core.grid import evaluate_density

EPS = 1e-15


def compute_density_on_slice(parser, points, data, grid_shape, ext=".wfx"):
    """Electron density on a 2D slice grid."""
    if ext == ".wfx":
        return calculate_density(points, data).reshape(grid_shape)
    if ext == ".cube":
        return evaluate_density(parser, points, ext=".cube").reshape(grid_shape)
    raise ValueError(f"Unsupported file extension: {ext}")


def compute_s_sign_lambda2_times_rho(parser, points, data, grid_shape, uu, vv, ext=".wfx"):
    """
    NCI indicator on a 2D slice: sign(lambda2) * log10(rho * s).

    Parameters
    ----------
    uu, vv : ndarray
        2D meshgrid coordinates in Bohr (in-plane axes).
    """
    rho = compute_density_on_slice(parser, points, data, grid_shape, ext=ext)
    du, dv = slice_analysis.slice_grid_spacing(uu, vv, grid_shape[0])
    return slice_analysis.compute_nci_indicator_2d(rho, du, dv), rho
