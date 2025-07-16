from parser import WFXParser
import plotter
import utils
import BCPs

def main():
    parser = WFXParser('data/benzene.wfx')
    #plotter.plot_density_slice(parser, plane='xy', z_pos=0.0, grid_points=200)
    plotter.plot_density_slice(parser, atom_indices=[1, 6, 7], grid_points=500)
    
    #plotter.plot_laplacian_slice(parser,plane='yz', z_pos=0.0, grid_points=200)
    plotter.plot_laplacian_slice(parser,atom_indices=[1, 6, 7], z_pos=0.0, grid_points=700)
    
    #plotter.plot_gradient_magnitude_slice(parser,plane='yz', z_pos=0.0, grid_points=200)
    plotter.plot_gradient_magnitude_slice(parser,atom_indices=[1, 6, 7], z_pos=0.0, grid_points=500)
    
    #plotter.plot_spin_density_slice(parser, plane='yz', z_pos=0.0, grid_points=200)
    plotter.plot_spin_density_slice(parser, atom_indices=[1, 6, 7], z_pos=0.0, grid_points=500)
    
    plotter.plot_density_gradient_laplacian_along_path(parser,atom1_index=1, atom2_index=2, points_count=200)
   
    n_electrons = utils.integrate_electron_density(parser, grid_points=100, padding=4)
    print(f"Número total de elétrons integrado: {n_electrons:.4f}")

    BCPs.find_critical_points_from_gradient_flow(
    parser,
    grid_points=100,
    grad_tol=1e-4,
    min_density=1e-2,
    n_points_per_pair=10,
    n_jobs=-1
)
    
if __name__ == "__main__":
    main()
