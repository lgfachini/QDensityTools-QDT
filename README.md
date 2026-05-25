# 🧪 QDensity Tools (QDT): A WFX/CUBE Density Analysis Toolkit

QDensity Tools (QDT) is a Python-based toolkit for analyzing and visualizing electron density-related properties from quantum chemistry wavefunction data in `.wfx` or `.cube` file formats.

It enables the computation and visualization of the following properties:

* **Electron Density** (`ρ`)
* **Electron Density Gradient Magnitude** (`|∇ρ|`)
* **Laplacian of Electron Density** (`∇²ρ`)
* **Reduced Density Gradient** (`s`)
* **NCI Indicator**: `log10(s · ρ) * sign(λ₂)`
* **NCI Scatter Plot** (`s` vs `sign(λ₂)ρ`) *(in development)*
* **Bond Critical Points (BCPs)**

---

## 📁 Structure

```
QDensityTools-QDT/
├── config/
│   ├── settings.py          # ← edit your run configuration here
│   └── settings.example.py
├── data/                    # input .wfx / .cube and generated outputs
├── src/qdt/
│   ├── cli.py               # command-line driver
│   ├── core/
│   │   ├── density.py       # electron density from .wfx (Numba)
│   │   ├── grid.py          # .cube sampling, padding, slice bounds
│   │   └── periodic_table.py
│   ├── io/
│   │   ├── parser.py        # .wfx and .cube parsers
│   │   └── cube.py          # export volumetric .cube files
│   ├── analysis/
│   │   ├── slice.py         # 2D in-plane NCI / Hessian on slices
│   │   ├── rdg.py           # reduced density gradient (s)
│   │   ├── nci.py           # NCI indicator helpers
│   │   ├── bcps.py          # bond critical point search
│   │   └── integration.py   # electron density integration
│   └── viz/
│       └── plotter.py       # 2D slices and path plots
├── tests/
├── main.py                  # shortcut entry: python main.py
├── pyproject.toml
└── requirements.txt
```

---

## ⚙️ Features

* 📄 **Reads `.wfx` and `.cube` files** via `qdt.io.parser`
* 📈 **Plots**:
  * Electron density
  * Gradient of the density
  * Laplacian of the density
  * Reduced density gradient (`s`)
  * `log10(s · ρ) * sign(λ₂)` on 2D slices
  * Along any path between two selected atoms
* 🧊 **Calculates 3D electron density** and exports as `.cube`
* 🧠 **Identifies BCPs (Bond Critical Points)** from the electron density field
* 🚀 Runs from `config/settings.py` via `python main.py`, `python -m qdt`, or `qdt`

---

## ▶️ How to Use

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

Or install the package in editable mode (recommended):

```bash
pip install -e .
```

### 2. Add Your `.wfx` or `.cube` File

Place your file inside the `data/` folder. Example:

```
data/molecule.wfx
```

### 3. Configure and Run

Edit **`config/settings.py`** at the project root:

* Set `INPUT_FILE` (e.g. `data/molecule.wfx`)
* Enable analyses with `RUN_*` flags (`True` = on; `False`, `None`, or `""` = off)
* Set slice plane, atom indices, grid size, and per-axis padding

Then run from the project root:

```bash
python main.py
```

Alternatives:

```bash
python -m qdt
qdt   # after pip install -e .
```

#### Run flags (in `config/settings.py`)

```python
RUN_DENSITY_SLICE = True
RUN_GRADIENT_SLICE = False
RUN_NCI_SLICE = True
RUN_BCP_SEARCH = False
# ... etc.
```

#### 2D slice region (independent X / Y)

Padding per plot axis (Bohr); `None` = automatic from molecular size:

```python
SLICE_PADDING_X = 2.0
SLICE_PADDING_Y = 8.0
```

Or crop to a fixed window (overrides padding on that axis):

```python
SLICE_X_RANGE = (-4.0, 12.0)   # xy: Cartesian x; custom plane: in-plane u
SLICE_Y_RANGE = (-2.0, 6.0)     # xy: Cartesian y; custom plane: in-plane v
```

Atom overlays on slice plots can be adjusted from settings:

```python
SHOW_ATOM_LABELS = True       # False hides element labels
ATOM_LABEL_COLOR = "black"
ATOM_LABEL_SIZE = 12
ATOM_LABEL_OFFSET = 0.05      # Angstrom
ATOM_MARKER_COLOR = "black"
ATOM_MARKER_SIZE = 3
```

Standard plane: `PLANE = 'xy'` and `Z_POS = 0.0`.  
Custom plane through three atoms: `PLANE = None` and `ATOM_INDICES = [0, 1, 2]`.

This can:

* Parse the `.wfx` or `.cube` file
* Calculate and save 3D electron density as `density.cube`
* Generate and save plots for density, gradient magnitude, and Laplacian slices
* Plot density-related properties along a chosen interatomic path
* Detect and save Bond Critical Points (BCPs) coordinates
* Save outputs next to the input file (typically under `data/`)

---

### 4. Additional Modules

#### `qdt.analysis.rdg`

Calculates the **reduced density gradient** (`s`), with logarithmic scaling (`log10(s)`) in plots to enhance visibility across large dynamic ranges.

#### `qdt.analysis.slice` / NCI plots

Computes and plots the field `log10[s * ρ] * sign(λ₂)` on **2D slices** (in-plane gradient and Hessian), commonly used in Non-Covalent Interaction (NCI) analysis to highlight weak interaction regions. The logarithmic term facilitates clearer graphical interpretation.

---

## 📊 Example Output

Example calculation and results for a water molecule can be placed in `data/`, including:

* `data/density_gradient_laplacian_path_O_H.png`: log-scaled density, gradient, and Laplacian between O and H atoms
* `data/reduced_gradient_slice_custom_plane_0_1_2.png`: reduced density gradient slice in a selected molecular plane
* `data/nci_indicator_slice_custom_plane_0_1_2.png`: slice of `log10[s * ρ] * sign(λ₂)` field
* `data/h2o.cube`: cube file representing 3D electron density
* `data/BCPs.xyz`: coordinates of detected bond critical points

Other results and `.wfx` files may be kept in `data/` for testing.

---

## 📚 Documentation

Each module contains internal docstrings for public functions. Use:

```bash
pydoc qdt.io.parser
pydoc qdt.viz.plotter
```

Or explore via an IDE (VS Code, PyCharm, etc.).

---

## 📌 Notes

* Paths between atoms are selected via atom indices in `config/settings.py` (`PATH_ATOM1`, `PATH_ATOM2`)
* Coordinates and grids are internally handled in atomic units (Bohr); plot axes are shown in Angstroms
* Density-related quantities are **log10-scaled and sometimes normalized** for visual comparison
* BCP search is parallelized (joblib) and computationally efficient
* Wavefunction (`.wfx`) files already include normalization constants in the primitive coefficients

---

## 👨‍🔬 Applications

QDT has been tested for plots involving:

* Bonding analysis in transition-metal and lanthanide complexes
* Non-covalent interaction studies (hydrogen bonding, halogen bonding)

---

## 👤 Author

Lucas Gian Fachini – *PhD Candidate in Inorganic and Theoretical Chemistry*  
[GitHub: lgfachini](https://github.com/lgfachini)

---

## 📄 License

This project is licensed under the GPL-3 License. See [LICENSE](LICENSE).

---

## 💡 Acknowledgments

This project uses concepts from:

* AIM (Atoms in Molecules) theory – Bader
* Non-Covalent Interaction (NCI) analysis – Johnson et al.
* So many other concepts they are hard to list; maybe one day I'll credit them all.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```
