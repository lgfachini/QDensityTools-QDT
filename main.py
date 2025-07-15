from parser import WFXParser
import plotter
import utils

def main():
    parser = WFXParser('data/h2o.wfx')
    plotter.plot_density_slice(parser, plane='xy', z_pos=0.0)
    plotter.plot_laplacian_slice(parser,plane='xy', z_pos=0.0)
    plotter.plot_gradient_magnitude_slice(parser,plane='xy', z_pos=0.0)
    plotter.plot_density_gradient_laplacian_along_path(parser,atom1_index=1, atom2_index=2, points_count=200)
    plotter.plot_spin_density_slice(parser, plane='xy', z_pos=0.0)
    utils.find_bond_critical_points_and_export_cube(parser, grid_points=80, gradient_threshold=1e-3, min_density=1e-3)
if __name__ == "__main__":
    main()
