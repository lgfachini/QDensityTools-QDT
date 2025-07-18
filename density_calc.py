import numpy as np
from numba import njit, prange

EPS = 1e-15  # Small epsilon to avoid numerical issues

@njit(cache=True)
def factorial2_numba(n):
    """
    Compute the double factorial (n!!) for a given integer n using numba for speed.

    The double factorial of n is the product of all integers from n down to 1 (or 2) that have the same parity as n.
    By definition, factorial2(0) = 1 and factorial2(-1) = 1.

    Parameters
    ----------
    n : int
        Integer for which to compute the double factorial.

    Returns
    -------
    float
        The value of n!! as a float.
    """
    if n <= 0:
        return 1.0
    result = 1.0
    for i in range(n, 0, -2):
        result *= i
    return result

@njit(cache=True)
def angular_momentum_numba(prim_type):
    """
    Return the Cartesian angular momentum exponents (l, m, n) for a given primitive type index.

    The function supports primitive types up to G (prim_type 1 to 35) explicitly,
    and computes higher angular momenta (H and beyond) on the fly using combinatorics.

    Parameters
    ----------
    prim_type : int
        Index of primitive type starting at 1.

    Returns
    -------
    tuple of ints
        (l, m, n) angular momentum exponents for the primitive.
        Returns (-1, -1, -1) if prim_type is invalid or not supported.
    """
    # Explicit table of angular momentum exponents for primitives 1 to 35 (up to G)
    table = [
        (0, 0, 0),  # 1  S
        (1, 0, 0),  # 2  PX
        (0, 1, 0),  # 3  PY
        (0, 0, 1),  # 4  PZ
        (2, 0, 0),  # 5  DXX
        (0, 2, 0),  # 6  DYY
        (0, 0, 2),  # 7  DZZ
        (1, 1, 0),  # 8  DXY
        (1, 0, 1),  # 9  DXZ
        (0, 1, 1),  # 10 DYZ
        (3, 0, 0),  # 11 FXXX
        (0, 3, 0),  # 12 FYYY
        (0, 0, 3),  # 13 FZZZ
        (2, 1, 0),  # 14 FXXY
        (2, 0, 1),  # 15 FXXZ
        (0, 2, 1),  # 16 FYYZ
        (1, 2, 0),  # 17 FXYY
        (1, 0, 2),  # 18 FXZZ
        (0, 1, 2),  # 19 FYZZ
        (1, 1, 1),  # 20 FXYZ
        (4, 0, 0),  # 21 GXXXX
        (0, 4, 0),  # 22 GYYYY
        (0, 0, 4),  # 23 GZZZZ
        (3, 1, 0),  # 24 GXXXY
        (3, 0, 1),  # 25 GXXXZ
        (1, 3, 0),  # 26 GXYYY
        (0, 3, 1),  # 27 GYYYZ
        (1, 0, 3),  # 28 GXZZZ
        (0, 1, 3),  # 29 GYZZZ
        (2, 2, 0),  # 30 GXXYY
        (2, 0, 2),  # 31 GXXZZ
        (0, 2, 2),  # 32 GYYZZ
        (2, 1, 1),  # 33 GXXYZ
        (1, 2, 1),  # 34 GXYYZ
        (1, 1, 2)   # 35 GXYZZ
    ]

    if prim_type < 1:
        return -1, -1, -1
    elif prim_type <= len(table):
        return table[prim_type - 1]
    else:
        # For primitive types 36 and beyond (H, I, etc.), generate exponents combinatorially
        prim_index = prim_type - len(table) - 1  # zero-based index beyond explicit table
        current_type = 36
        for L in range(5, 20):  # Angular momentum degree L = 5 (H), 6 (I), etc.
            count = (L + 1) * (L + 2) // 2  # Number of Cartesian functions for angular momentum L
            if prim_index < count:
                idx = 0
                # Enumerate all (lx, ly, lz) such that lx+ly+lz = L
                for lx in range(L + 1):
                    for ly in range(L + 1 - lx):
                        lz = L - lx - ly
                        if idx == prim_index:
                            return lx, ly, lz
                        idx += 1
            else:
                prim_index -= count
                current_type += count
        # Beyond supported angular momenta
        return -1, -1, -1

@njit(parallel=True, fastmath=True)
def calc_density_numba(points, primitive_centers, primitive_types, primitive_exponents,
                      nuclei_coords, mo_coeffs, mo_occupations):
    """
    Calculate the electron density at a set of points given basis set primitives and molecular orbitals.

    Parameters
    ----------
    points : ndarray
        Coordinates where density is evaluated (N_points x 3).
    primitive_centers : ndarray
        Indices (1-based) of atomic centers for each primitive function.
    primitive_types : ndarray
        Angular momentum type index for each primitive.
    primitive_exponents : ndarray
        Gaussian exponents for each primitive function.
    nuclei_coords : ndarray
        Cartesian coordinates of nuclei (atoms).
    mo_coeffs : ndarray
        Molecular orbital coefficients (number of MOs x number of primitives).
    mo_occupations : ndarray
        Occupation numbers for each molecular orbital.

    Returns
    -------
    density : ndarray
        Electron density values at the input points.
    """
    n_points = points.shape[0]
    n_prim = primitive_centers.shape[0]
    n_mo = mo_coeffs.shape[0]

    # Precompute basis function values at all points for all primitives
    basis_values = np.zeros((n_prim, n_points), dtype=np.float64)

    for prim_idx in prange(n_prim):
        center_idx = primitive_centers[prim_idx] - 1  # Convert 1-based to 0-based index
        center = nuclei_coords[center_idx]
        exponent = primitive_exponents[prim_idx]
        prim_type = primitive_types[prim_idx]

        l, m, n = angular_momentum_numba(prim_type)
        if l == -1:
            continue  # Skip invalid primitives

        for i in range(n_points):
            Rx = points[i, 0] - center[0]
            Ry = points[i, 1] - center[1]
            Rz = points[i, 2] - center[2]
            R2 = Rx*Rx + Ry*Ry + Rz*Rz
            # Evaluate primitive Gaussian basis function with Cartesian polynomial prefactor
            basis_values[prim_idx, i] = (Rx**l) * (Ry**m) * (Rz**n) * np.exp(-exponent * R2)

    density = np.zeros(n_points, dtype=np.float64)

    # Sum contributions of all molecular orbitals weighted by their occupation numbers
    for mo_idx in prange(n_mo):
        occ = mo_occupations[mo_idx]
        if abs(occ) < EPS:
            continue  # Skip unoccupied orbitals

        psi = np.zeros(n_points, dtype=np.float64)
        for prim_idx in range(n_prim):
            psi += mo_coeffs[mo_idx, prim_idx] * basis_values[prim_idx]

        density += occ * psi * psi  # Density contribution = occupation * |ψ|^2

    return density

def calculate_density(points, data, spin='total'):
    """
    Compute electron density at given points for specified spin channel or total.

    Parameters
    ----------
    points : ndarray
        Coordinates where density is evaluated.
    data : dict
        Dictionary containing basis set and molecular orbital data.
    spin : str, optional
        Spin channel: 'alpha', 'beta', or 'total' (default).

    Returns
    -------
    ndarray
        Electron density values at input points.
    """
    centers = np.array(data['primitive_centers'])
    types = np.array(data['primitive_types'])
    exps = np.array(data['primitive_exponents'])
    coords = np.array([n['coords'] for n in data['nuclei']])

    if spin == 'alpha':
        return calc_density_numba(points, centers, types, exps, coords,
                                  np.array(data['mo_coefficients_alpha']),
                                  np.array(data['mo_occupations_alpha']))
    elif spin == 'beta':
        return calc_density_numba(points, centers, types, exps, coords,
                                  np.array(data['mo_coefficients_beta']),
                                  np.array(data['mo_occupations_beta']))
    else:
        dens_a = calc_density_numba(points, centers, types, exps, coords,
                                    np.array(data['mo_coefficients_alpha']),
                                    np.array(data['mo_occupations_alpha']))
        if data.get('mo_coefficients_beta', None):
            dens_b = calc_density_numba(points, centers, types, exps, coords,
                                        np.array(data['mo_coefficients_beta']),
                                        np.array(data['mo_occupations_beta']))
            return dens_a + dens_b
        return dens_a
