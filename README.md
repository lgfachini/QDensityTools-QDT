# 🧪 WFX Density Analysis Toolkit

This project is a Python-based toolkit for analyzing wavefunction data from `.wfx` files. It allows for visualization and analysis of **electron density**, **density gradients**, **Laplacians**, **bond critical points (BCPs)**, and other derived quantities such as **reduced density gradient (s)** and **`log10[s \* ρ] \* sign(λ2)`**.

## 📁 Project Structure

```
wfx_proj/
├── main.py                   # Main execution script
├── parser.py                 # Parses .wfx files and extracts wavefunction data
├── plotter.py                # Plots density, gradient, and Laplacian (including along atom paths)
├── density_calc.py           # Calculates 3D electron density on grids
├── rdg_calc.py               # Calculates reduced density gradient (s)
├── s_sign_lambda2_rho_p.py   # Calculates log10[s * ρ] * sign(λ2) and plots it
├── BCPs.py                   # Identifies Bond Critical Points via gradient path following
├── cube.py                   # Exports data to Gaussian cube format
├── utils.py                  # Utility functions for 3D density calculations and support
├── analysis.py               # (Under development) Advanced analysis and post-processing functions
├── geometry.py               # (Under development) Geometrical calculations and molecule manipulation
├── data/                     # Your .wfx files and output data
└── README.md                 # This file
```

## ⚙️ Features

* 📄 **Reads `.wfx` files** using `parser.py`
* 📈 **Plots**:

  * Electron density
  * Gradient of the density
  * Laplacian of the density
  * Reduced density gradient (`s`)
  * `log10[s \* ρ] \* sign(λ2)`
  * Along any path between two selected atoms
* 🧊 **Calculates 3D electron density** and exports as `.cube`
* 🧠 **Identifies BCPs (Bond Critical Points)** from the electron density field
* 🚀 Everything is executed via `main.py`

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

Adjust parameters in `main.py` (paths, atoms for paths, grid size, etc.) and run:

```bash
python main.py
```

This will:

* Parse the `.wfx` file
* Calculate and save 3D electron density as `density.cube`
* Generate and save plots for density, gradient magnitude, and Laplacian slices
* Plot density-related properties along a chosen interatomic path
* Detect and save Bond Critical Points (BCPs) coordinates
* Save all outputs in the `data/` folder

### 4. Additional Modules

#### `rdg_calc.py`

Calculates and plots the **reduced density gradient** (`s`), with logarithmic scaling (`log10(s)`) to enhance visibility across large dynamic ranges.

#### `s_sign_lambda2_rho_p.py`

Computes and plots the field `log10[s * ρ] * sign(λ2)`, commonly used in Non-Covalent Interaction (NCI) analysis to highlight weak interaction regions. The logarithmic term facilitates clearer graphical interpretation.

## 📊 Example Output

* Example calculation and results for a water dimer are in `/data`, including:

  * `data/density_gradient_laplacian_path_O_H.png`: log-scaled density, gradient, and Laplacian between O and H atoms
  * `data/s_field.png`: Reduced density gradient slice in a selected molecular plane
  * `data/s_sign_lambda2_rho_field.png`: Slice of `log10[s * ρ] * sign(λ2)` field
  * `data/electron_density.cube`: Cube file representing 3D electron density
  * `data/BCPs.xyz`: Coordinates of detected bond critical points

## 📌 Notes

* Paths between atoms are selected via atom indices in `main.py`
* Coordinates and grids are internally handled in atomic units (Bohr); outputs are converted to Angstroms
* Density-related quantities are **log10-scaled and sometimes normalized** for visual comparison
* BCP search is parallelized and computationally efficient
* Future modules (e.g., `analysis.py`, `geometry.py`) extend functionality for custom analyses

## 👤 Author

Lucas Gian Fachini
[GitHub: lgfachini](https://github.com/lgfachini)

## 📜 License

GPL-3 License
