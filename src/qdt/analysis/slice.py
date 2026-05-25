"""2D in-plane analysis on a density slice (NCI indicator, RDG, Hessian eigenvalues)."""

import numpy as np
from numpy.linalg import eigvalsh
from qdt.analysis.rdg import compute_s_values

EPS = 1e-15


def slice_grid_spacing(uu, vv, grid_points):
    """Grid spacing along the two in-plane axes (Bohr)."""
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    du = abs(float(uu[0, 1] - uu[0, 0]))
    dv = abs(float(vv[1, 0] - vv[0, 0]))
    if du < EPS:
        du = EPS
    if dv < EPS:
        dv = EPS
    return du, dv


def compute_2d_derivatives(rho, du, dv):
    """First and second derivatives of rho on a 2D slice grid."""
    gx, gy = np.gradient(rho, du, dv)
    dxx, dxy = np.gradient(gx, du, dv)
    dxy2, dyy = np.gradient(gy, du, dv)
    dxy = 0.5 * (dxy + dxy2)
    return gx, gy, dxx, dyy, dxy


def sign_lambda2_2d(dxx, dyy, dxy):
    """sign(lambda2) for the 2x2 Hessian at each grid point (lambda2 = larger eigenvalue)."""
    shape = dxx.shape
    out = np.zeros(shape, dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            hess = np.array([[dxx[i, j], dxy[i, j]], [dxy[i, j], dyy[i, j]]])
            ev = np.sort(eigvalsh(hess))
            out[i, j] = np.sign(ev[1])
    return out


def compute_nci_indicator_2d(rho, du, dv):
    """
    NCI-style indicator on a 2D slice: sign(lambda2) * log10(rho * s).

    Uses in-plane gradient and 2x2 Hessian (appropriate for a 2D cut through 3D rho).
    """
    gx, gy, dxx, dyy, dxy = compute_2d_derivatives(rho, du, dv)
    gz = np.zeros_like(gx)
    s = compute_s_values(rho, gx, gy, gz)
    sign_l2 = sign_lambda2_2d(dxx, dyy, dxy)
    return sign_l2 * np.log10(np.maximum(rho * s, EPS))


def normalize_symmetric(field, clip=True):
    """Scale a signed field to [-1, 1] by its maximum absolute value."""
    absmax = np.nanmax(np.abs(field))
    if not np.isfinite(absmax) or absmax == 0.0:
        return np.zeros_like(field)
    out = field / absmax
    if clip:
        out = np.clip(out, -1.0, 1.0)
    return out
