"""Shared grid helpers for .cube sampling and density evaluation."""

import numpy as np
import scipy.ndimage
from qdt.core.density import calculate_density


def sample_density_cube(parser, points, order=3):
    """Interpolate electron density from a .cube grid at Cartesian points (Bohr)."""
    points = np.asarray(points, dtype=float)
    origin = parser.origin
    vectors = parser.vectors
    inv_vectors = np.linalg.inv(vectors.T)
    fractional_indices = (points - origin) @ inv_vectors
    return scipy.ndimage.map_coordinates(
        parser.density,
        fractional_indices.T,
        order=order,
        mode="nearest",
    )


def evaluate_density(parser, points, ext=".wfx"):
    """Evaluate total electron density at arbitrary 3D points."""
    ext = ext.lower()
    if ext == ".wfx":
        return calculate_density(points, parser.data)
    if ext == ".cube":
        return sample_density_cube(parser, points)
    raise ValueError(f"Unsupported file extension: '{ext}'. Use '.wfx' or '.cube'.")


def default_padding_bohr(parser, fraction=0.15, minimum=4.0):
    """Automatic padding (Bohr) from molecular span."""
    coords = np.array([n["coords"] for n in parser.data["nuclei"]])
    span = coords.max(axis=0) - coords.min(axis=0)
    return max(fraction * np.linalg.norm(span), minimum)


def resolve_padding(parser, padding):
    """Return padding in Bohr; None triggers automatic value."""
    if padding is None:
        return default_padding_bohr(parser)
    return float(padding)


def resolve_padding_2d(parser, padding_x=None, padding_y=None, padding=None):
    """
    Return (pad_x, pad_y) in Bohr for 2D slice plots.

    Parameters
    ----------
    padding : float, (float, float), or None
        If scalar: same padding on both axes. If (px, py): per-axis values.
        None with padding_x/y None: auto on both axes.
    padding_x, padding_y : float or None
        Per-axis override (plot horizontal / vertical directions).
    """
    auto = default_padding_bohr(parser)
    px, py = None, None

    if padding is not None:
        if isinstance(padding, (tuple, list)) and len(padding) == 2:
            px, py = padding[0], padding[1]
        else:
            px = py = padding

    if padding_x is not None:
        px = padding_x
    if padding_y is not None:
        py = padding_y

    pad_x = auto if px is None else float(px)
    pad_y = auto if py is None else float(py)
    return pad_x, pad_y


def apply_axis_range(axis_min, axis_max, pad, axis_range=None):
    """Use explicit (min, max) in Bohr or extend bounds by padding."""
    if axis_range is not None:
        lo, hi = axis_range
        return float(lo), float(hi)
    return axis_min - pad, axis_max + pad
