import numpy as np
import density_calc
import rdg_calc
from scipy.ndimage import gaussian_filter
from numpy.linalg import eigvalsh

EPS = 1e-15

def compute_density_gradient_hessian(parser, points, data, grid_shape, padding=None):
    """
    Calcula densidade e suas derivadas até segunda ordem (gradiente e Hessiana 2D).
    """
    rho = density_calc.calculate_density(points, data).reshape(grid_shape)
    coords = np.array([n['coords'] for n in parser.data['nuclei']])
    if padding is None:
        span = coords.max(axis=0) - coords.min(axis=0)
        padding = max(0.15 * np.linalg.norm(span), 4.0)
        
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding
    nx, ny = grid_shape
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
      
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    print("The gradient grid mesh in bohr:", round(dx,4),"by", round(dy,4))

    # Gradientes (1ª ordem)
    gx, gy = np.gradient(rho, dx, dy)

    # Derivadas de 2ª ordem (Hessiana 2x2)
    dxx = gaussian_filter(rho, sigma=1, order=(2, 0), mode='nearest')
    dyy = gaussian_filter(rho, sigma=1, order=(0, 2), mode='nearest')
    dxy = gaussian_filter(rho, sigma=1, order=(1, 1), mode='nearest')

    return rho, gx, gy, dxx, dyy, dxy

def compute_lambda2_sign(dxx, dyy, dxy):
    """
    Calcula sign(λ2), onde λ2 é o maior autovalor da Hessiana 2x2 em cada ponto.
    """
    shape = dxx.shape
    sign_lambda2 = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            H = np.array([[dxx[i, j], dxy[i, j]],
                          [dxy[i, j], dyy[i, j]]])
            ev = np.sort(eigvalsh(H))  # ordena os autovalores
            lambda2 = ev[1]  # maior em 2D
            sign_lambda2[i, j] = np.sign(lambda2)

    return sign_lambda2


def compute_s_sign_lambda2_times_rho(parser, points, data, grid_shape, vmin=None, vmax=None):
    rho, gx, gy, dxx, dyy, dxy = compute_density_gradient_hessian(parser, points, data, grid_shape)
    gz = np.zeros_like(gx)  # adiciona a componente z nula para compatibilidade
    s = rdg_calc.compute_s_values(rho, gx, gy, gz)
    sign_lambda2 = compute_lambda2_sign(dxx, dyy, dxy)
    result = sign_lambda2 * np.log10(rho * s)

    return result, rho
