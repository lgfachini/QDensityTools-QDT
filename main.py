import os
from parser import WFXParser, CubeParser
import plotter
import utils
import BCPs

def plot_density_slice_wrapper(parser, ext, *, plane=None, atom_indices=None, z_pos=0.0, grid_points=300):
    print("\n→ Starting plot_density_slice...")
    if plane is not None and atom_indices is not None:
        raise ValueError("Specify either plane or atom_indices, not both.")
    if plane is None and atom_indices is None:
        raise ValueError("Must specify either plane or atom_indices.")
    plotter.plot_density_slice(parser, plane=plane, atom_indices=atom_indices, z_pos=z_pos, grid_points=grid_points, ext=ext)
    print("✓ Finished plot_density_slice.")

def plot_gradient_slice_wrapper(parser, ext, *, plane=None, atom_indices=None, z_pos=0.0, grid_points=300):
    print("\n→ Starting plot_gradient_slice...")
    if plane is not None and atom_indices is not None:
        raise ValueError("Specify either plane or atom_indices, not both.")
    if plane is None and atom_indices is None:
        raise ValueError("Must specify either plane or atom_indices.")
    plotter.plot_gradient_magnitude_slice(parser, plane=plane, atom_indices=atom_indices, z_pos=z_pos, grid_points=grid_points, ext=ext)
    print("✓ Finished plot_gradient_slice.")

def plot_laplacian_slice_wrapper(parser, ext, *, plane=None, atom_indices=None, z_pos=0.0, grid_points=300):
    print("\n→ Starting plot_laplacian_slice...")
    if plane is not None and atom_indices is not None:
        raise ValueError("Specify either plane or atom_indices, not both.")
    if plane is None and atom_indices is None:
        raise ValueError("Must specify either plane or atom_indices.")
    plotter.plot_laplacian_slice(parser, plane=plane, atom_indices=atom_indices, z_pos=z_pos, grid_points=grid_points, ext=ext)
    print("✓ Finished plot_laplacian_slice.")

def plot_spin_density_slice_wrapper(parser, ext, *, plane=None, atom_indices=None, z_pos=0.0, grid_points=300):
    print("\n→ Starting plot_spin_density_slice...")
    if plane is not None and atom_indices is not None:
        raise ValueError("Specify either plane or atom_indices, not both.")
    if plane is None and atom_indices is None:
        raise ValueError("Must specify either plane or atom_indices.")
    plotter.plot_spin_density_slice(parser, plane=plane, atom_indices=atom_indices, z_pos=z_pos, grid_points=grid_points, ext=ext)
    print("✓ Finished plot_spin_density_slice.")

def plot_reduced_gradient_slice_wrapper(parser, ext, *, plane=None, atom_indices=None, z_pos=0.0, grid_points=300):
    print("\n→ Starting plot_reduced_gradient_slice...")
    if plane is not None and atom_indices is not None:
        raise ValueError("Specify either plane or atom_indices, not both.")
    if plane is None and atom_indices is None:
        raise ValueError("Must specify either plane or atom_indices.")
    plotter.plot_reduced_gradient_slice(parser, plane=plane, atom_indices=atom_indices, z_pos=z_pos, grid_points=grid_points, ext=ext)
    print("✓ Finished plot_reduced_gradient_slice.")

def plot_s_sign_lambda2_rho_slice_wrapper(parser, ext, *, plane=None, atom_indices=None, z_pos=0.0, grid_points=500):
    print("\n→ Starting plot_s_sign_lambda2_rho_slice...")
    if plane is not None and atom_indices is not None:
        raise ValueError("Specify either plane or atom_indices, not both.")
    if plane is None and atom_indices is None:
        raise ValueError("Must specify either plane or atom_indices.")
    plotter.plot_s_sign_lambda2_rho_slice(parser, plane=plane, atom_indices=atom_indices, z_pos=z_pos, grid_points=grid_points, ext=ext)
    print("✓ Finished plot_s_sign_lambda2_rho_slice.")

def plot_density_gradient_laplacian_along_path_wrapper(parser, ext,
                                                      atom1_index=1, atom2_index=2,
                                                      points_count=1000):
    print("\n→ Starting plot_density_gradient_laplacian_along_path...")
    plotter.plot_density_gradient_laplacian_along_path(
        parser, atom1_index=atom1_index, atom2_index=atom2_index,
        points_count=points_count, ext=ext
    )
    print("✓ Finished plot_density_gradient_laplacian_along_path.")

def run_integration(parser, ext, grid_points=100, padding=4):
    print("\n→ Starting electron density integration...")
    n_electrons = utils.integrate_electron_density(parser, grid_points=grid_points, padding=padding, ext=ext)
    print(f"✓ Total integrated electrons: {n_electrons:.4f}")
    print("✓ Finished electron density integration.")

def run_bcps_search(parser,
                    grid_points=200,
                    grad_tol=1e-5,
                    min_density=1e-3,
                    n_points_per_pair=10,
                    n_jobs=-1,
                    ext='.wfx'):
    BCPs.find_critical_points_from_gradient_flow(
        parser,
        grid_points=grid_points,
        grad_tol=grad_tol,
        min_density=min_density,
        n_points_per_pair=n_points_per_pair,
        n_jobs=n_jobs,
        ext=ext
    )
    print("✓ Finished BCPs critical points search.")

def main(input_file='data/h2o.wfx'): #you can edit your file, .wfx or .cube
    ext = os.path.splitext(input_file)[1].lower()

    if ext == '.wfx':
        parser = WFXParser(input_file)
    elif ext == '.cube':
        parser = CubeParser(input_file, smoothing_sigma=0)  # If your cube is noisy, smooth the cube.
    else:
        raise ValueError(f"Unrecognized file extension: {ext}")

    print(f"\n File recognized as: {ext[1:]}, using corresponding parser.\n")

    ############## EXAMPLES OF CALLS YOU CAN TWEAK:

    plot_density_slice_wrapper(parser, ext, atom_indices=[0, 1, 2], z_pos=0.0, grid_points=200)  # or plane='xy'
    plot_gradient_slice_wrapper(parser, ext, atom_indices=[0, 1, 2], z_pos=0.0, grid_points=200)
    plot_laplacian_slice_wrapper(parser, ext, atom_indices=[0, 1, 2], z_pos=0.0, grid_points=200)
    plot_spin_density_slice_wrapper(parser, ext, atom_indices=[0, 1, 2], z_pos=0.0, grid_points=200)
    plot_reduced_gradient_slice_wrapper(parser, ext, atom_indices=[0, 1, 2], z_pos=0.0, grid_points=200)
    plot_s_sign_lambda2_rho_slice_wrapper(parser, ext, atom_indices=[0, 1, 2], z_pos=0.0, grid_points=200)
    plot_density_gradient_laplacian_along_path_wrapper(parser, ext, atom1_index=1, atom2_index=2, points_count=200)
    run_integration(parser, ext, grid_points=100, padding=4)
    run_bcps_search(parser, grid_points=100, grad_tol=1e-5, min_density=1e-3, n_points_per_pair=10, n_jobs=-1, ext=ext)

    ##############

if __name__ == "__main__":
    main()
