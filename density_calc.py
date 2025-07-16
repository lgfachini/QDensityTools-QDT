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
    # Tabela explícita até G (prim_type 1–35)
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
        # A partir de prim_type 36 → orbitais H e superiores
        prim_index = prim_type - len(table) - 1  # índice 0-based a partir do tipo 36
        current_type = 36
        for L in range(5, 20):  # L = grau (5 → H, 6 → I, etc.)
            count = (L + 1) * (L + 2) // 2
            if prim_index < count:
                idx = 0
                for lx in range(L + 1):
                    for ly in range(L + 1 - lx):
                        lz = L - lx - ly
                        if idx == prim_index:
                            return lx, ly, lz
                        idx += 1
            else:
                prim_index -= count
                current_type += count
        # muito além (não suportado)
        return -1, -1, -1

    if 1 <= prim_type <= len(table):
        return table[prim_type - 1]
    else:
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
