import numpy as np
from numba import njit, prange

EPS = 1e-15

@njit(cache=True)
def factorial2_numba(n):
    if n <= 0:
        return 1.0
    result = 1.0
    for i in range(n, 0, -2):
        result *= i
    return result

@njit(cache=True)
def angular_momentum_numba(prim_type):
    if prim_type < 1:
        return -1, -1, -1
    idx = 1
    for grau in range(10):
        for l in range(grau + 1):
            for m in range(grau + 1):
                for n in range(grau + 1):
                    if l + m + n == grau:
                        if idx == prim_type:
                            return l, m, n
                        idx += 1
    return -1, -1, -1

@njit(parallel=True, fastmath=True)
def calc_density_numba(points, primitive_centers, primitive_types, primitive_exponents, nuclei_coords, mo_coeffs, mo_occupations):
    n_points = points.shape[0]
    n_prim = primitive_centers.shape[0]
    n_mo = mo_coeffs.shape[0]

    basis_values = np.zeros((n_prim, n_points), dtype=np.float64)

    for prim_idx in prange(n_prim):
        center_idx = primitive_centers[prim_idx] - 1
        center = nuclei_coords[center_idx]
        exponent = primitive_exponents[prim_idx]
        prim_type = primitive_types[prim_idx]

        l, m, n = angular_momentum_numba(prim_type)
        if l == -1:
            continue

        for i in range(n_points):
            Rx = points[i, 0] - center[0]
            Ry = points[i, 1] - center[1]
            Rz = points[i, 2] - center[2]
            R2 = Rx*Rx + Ry*Ry + Rz*Rz
            basis_values[prim_idx, i] = (Rx**l) * (Ry**m) * (Rz**n) * np.exp(-exponent * R2)

    density = np.zeros(n_points, dtype=np.float64)

    for mo_idx in prange(n_mo):
        occ = mo_occupations[mo_idx]
        if abs(occ) < EPS:
            continue

        psi = np.zeros(n_points, dtype=np.float64)
        for prim_idx in range(n_prim):
            psi += mo_coeffs[mo_idx, prim_idx] * basis_values[prim_idx]

        density += occ * psi * psi

    return density

def calculate_density(points, data, spin='total'):
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
        if data['mo_coefficients_beta']:
            dens_b = calc_density_numba(points, centers, types, exps, coords,
                                        np.array(data['mo_coefficients_beta']),
                                        np.array(data['mo_occupations_beta']))
            return dens_a + dens_b
        return dens_a
