# 🧪 WFX Density Analysis Toolkit

This project is a Python-based toolkit for analyzing wavefunction data from `.wfx` files. It allows for visualization and analysis of **electron density**, **density gradients**, **Laplacians**, **bond critical points (BCPs)**, and more.

## 📁 Project Structure

```
wfx_proj/
├── main.py # Main execution script
├── parser.py # Parses .wfx files and extracts wavefunction data
├── plotter.py # Plots density, gradient, and Laplacian (including along atom paths)
├── density_calc.py # Calculates 3D electron density on grids
├── BCPs.py # Identifies Bond Critical Points via gradient path following
├── cube.py # Exports data to Gaussian cube format
├── utils.py # Utility functions for 3D density calculations and support
├── analysis.py # (Under development) Advanced analysis and post-processing functions
├── geometry.py # (Under development) Geometrical calculations and molecule manipulation
├── data/ # Your .wfx files and output data
└── README.md # This file
```

## ⚙️ Features

- 📄 **Reads `.wfx` files** using `parser.py`
- 📈 **Plots**:
  - Electron density
  - Gradient of the density
  - Laplacian of the density
  - Along any path between two selected atoms
- 🧊 **Calculates 3D electron density** and exports as `.cube`
- 🧠 **Identifies BCPs (Bond Critical Points)** from the electron density field
- 🚀 Everything is executed via `main.py`

## ▶️ How to Use

### 1. Install Requirements

```bash
pip install numpy matplotlib scipy numba joblib tqdm
```

### 2. Add Your `.wfx` File

Place your `.wfx` file inside the `data/` folder. Example:

```
data/molecule.wfx
```

### 3. Run the Project

Adjust parameters in main.py (paths, atoms for paths, grid size, etc.) and run:

```bash
python main.py
```

This will:

- Parse the `.wfx` file
- Calculate and save 3D electron density as `density.cube`
- Generate and save plots for density, gradient magnitude, and Laplacian slices
- Plot density-related properties along a chosen interatomic path
- Detect and save Bond Critical Points (BCPs) coordinates
- Save all outputs in the data/ folder

## 📊 Example Output

- Example calculation and results for water are in /data, like:
- `data/density_gradient_laplacian_path_O_H.png`: Plot of log-scaled density, gradient, and Laplacian between O and H atoms.
- Additional plots for slices in XY, XZ, and YZ planes of the desired density, Laplacian or gradient
- `data/electron_density.cube`: Cube file representing 3D electron density.
- `data/BCPs.xyz`: Coordinates of detected bond critical points

## 📌 Notes

- Paths between atoms are selected via atom indices in `main.py`.
- Coordinates and grids are internally handled in atomic units (Bohr) for precision; all outputs and plots are converted to Angstrom units.
- The density, gradient, and Laplacian are **log10-scaled and normalized** for visual comparison.
- The BCP search is parallelized but can still be computationally demanding for large systems.
- Future modules (analysis.py, geometry.py) provide extensibility for advanced post-processing and molecular manipulations.

## 👤 Author

Lucas Gian Fachini  
[GitHub: lgfachini](https://github.com/lgfachini)

## 📜 License

GPL-3 License
