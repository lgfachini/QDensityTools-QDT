# 🧪 QDensity Tools (QDT): A WFX/CUBE Density Analysis Toolkit

QDensity Tools (QDT) is a Python-based toolkit for analyzing and visualizing electron density-related properties from quantum chemistry wavefunction data in `.wfx` or `.cube` file formats.

It enables the computation and visualization of the following properties:

* **Electron Density** (`ρ`)
* **Electron Density Gradient Magnitude** (`|∇ρ|`)
* **Laplacian of Electron Density** (`∇²ρ`)
* **Reduced Density Gradient** (`s`)
* **NCI Indicator**: `log10(s · ρ) \* sign(λ₂)`
* **NCI Scatter Plot** (`s` vs `sign(λ₂)ρ`) #in development
* **Bond Critical Points (BCPs)**

\---

## 📁 Structure

```
qdt/
├── main.py		        # Main script, here you tweak your calculation parameters
├── parser.py                  # Parses .wfx and .cube files and stores electronic structure data
├── density.py            	# Computes density from wavefunction files
├── rdg\_calc.py                # Computes Reduced Density Gradient (s)
├── s\_sign\_lambda2\_rho\_p.py    # Computes and plots s \* sign(λ₂) \* ρ
├── BCPs.py                    # Locates Bond Critical Points from density
├── cube.py                    # (Under development) Outputs .cube files
├── plotter.py                 # Generates 2D slices and scatter plots
├── periodic\_table.py          # Periodic table mapping for atomic number ↔ symbol
├── utils                      # Other utilities, like electronic density integration
├── analysis.py                # (Under development) Advanced analysis and post-processing functions
├── geometry.py                # (Under development) Geometrical calculations and molecule manipulation
└── data/                      # Example calculations for water, several other files for test

```

\---

## ⚙️ Features

* 📄 **Reads `.wfx` and `.cube` files** using `parser.py`
* 📈 **Plots**:

  * Electron density
  * Gradient of the density
  * Laplacian of the density
  * Reduced density gradient (`s`)
  * `log10(s · ρ) \* sign(λ₂)`
  * Along any path between two selected atoms
* 🧊 **Calculates 3D electron density** and exports as `.cube`
* 🧠 **Identifies BCPs (Bond Critical Points)** from the electron density field
* 🚀 Everything is executed via `main.py`

\---

## ▶️ How to Use

### 1\. Install Requirements

```bash
pip install numpy matplotlib scipy numba joblib tqdm
```

### 2\. Add Your `.wfx` or `.cube` File

Place your `.wfx` file inside the `data/` folder. Example:

```
data/molecule.wfx
```

### 3\. Run the Project

Adjust parameters in `main.py` (paths, atoms for paths, grid size, etc.) and run:

```bash
python main.py
```

This can:

* Parse the `.wfx` or `.cube` file
* Calculate and save 3D electron density as `density.cube`
* Generate and save plots for density, gradient magnitude, and Laplacian slices
* Plot density-related properties along a chosen interatomic path
* Detect and save Bond Critical Points (BCPs) coordinates
* Save all outputs in the `data/` folder



### 4\. Additional Modules

#### `rdg\_calc.py`

Calculates and plots the **reduced density gradient** (`s`), with logarithmic scaling (`log10(s)`) to enhance visibility across large dynamic ranges.

#### `s\_sign\_lambda2\_rho\_p.py`

Computes and plots the field `log10\[s \* ρ] \* sign(λ2)`, commonly used in Non-Covalent Interaction (NCI) analysis to highlight weak interaction regions. The logarithmic term facilitates clearer graphical interpretation.

## 📊 Example Output

* Example calculation and results for a water molecule are in `/data`, including:

  * `data/density\_gradient\_laplacian\_path\_O\_H.png`: log-scaled density, gradient, and Laplacian between O and H atoms
  * `data/reduced\_gradient\_slice\_custom\_plane\_0\_1\_2.png`: Reduced density gradient slice in a selected molecular plane
  * `data/s\_sign\_lambda2\_rho\_slice\_custom\_plane\_0\_1\_2.png`: Slice of `log10\[s \* ρ] \* sign(λ2)` field
  * `data/h2o.cube`: Cube file representing 3D electron density
  * `data/BCPs.xyz`: Coordinates of detected bond critical points
* Some other results and .wfx files for test

\---

## 📚 Documentation

Each module contains internal docstrings for all public functions. Use:

```bash
pydoc parser
```

Or explore via an IDE like VSCode or PyCharm.

\---

## 📌 Notes

* Paths between atoms are selected via atom indices in `main.py`
* Coordinates and grids are internally handled in atomic units (Bohr); outputs are converted to Angstroms
* Density-related quantities are **log10-scaled and sometimes normalized** for visual comparison
* BCP search is parallelized and computationally efficient
* Future modules (e.g., `analysis.py`, `geometry.py`) extend functionality for custom analyses

## 👨‍🔬 Applications

QDT has been tested for plots involving:

* Bonding analysis in transition-metal and lanthanide complexes
* Non-covalent interaction studies (hydrogen bonding, halogen bonding)

\---

## 👤 Author

Lucas Gian Fachini – *PhD Candidate in Inorganic and Theoretical Chemistry*
[GitHub: lgfachini](https://github.com/lgfachini)

\---

## 📄 License

This project is licensed under the GPL-3 License.

\---

## 💡 Acknowledgments

This project uses concepts from:

* AIM (Atoms in Molecules) theory – Bader
* Non-Covalent Interaction (NCI) analysis – Johnson et al.
* So many other concepts they are hard to list, maybe one day I'll credit it all.

