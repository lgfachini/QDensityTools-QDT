# QDensity Tools (QDT)

Python toolkit for analyzing and visualizing electron density from quantum chemistry `.wfx` and `.cube` files.

## Project layout

```
QDensityTools-QDT/
├── config/
│   ├── settings.py          # ← edit your run configuration here
│   └── settings.example.py  # template
├── data/                    # input wavefunctions / output figures
├── src/qdt/                 # installable package
│   ├── cli.py               # command-line driver
│   ├── core/                # density, grids, periodic table
│   ├── io/                  # .wfx / .cube parsers and writers
│   ├── analysis/            # NCI, RDG, BCPs, integration
│   └── viz/                 # matplotlib plotting
├── tests/
├── main.py                  # shortcut: python main.py
├── pyproject.toml
└── requirements.txt
```

## Install

From the project root:

```bash
pip install -e .
```

Or dependencies only:

```bash
pip install -r requirements.txt
```

## Configure and run

1. Place your `.wfx` or `.cube` in `data/`.
2. Edit `config/settings.py` (input path, `RUN_*` flags, padding, planes, BCP options).
3. Run:

```bash
python main.py
# or
python -m qdt
# or (after pip install -e .)
qdt
```

### Run flags

Set each analysis to `True` to enable; `False`, `None`, or `""` disables it:

```python
RUN_NCI_SLICE = True
RUN_BCP_SEARCH = False
```

### 2D slice region

Independent padding per plot axis (Bohr):

```python
SLICE_PADDING_X = 2.0
SLICE_PADDING_Y = 8.0
```

Or crop to a fixed window (overrides padding on that axis):

```python
SLICE_X_RANGE = (-4.0, 12.0)
SLICE_Y_RANGE = (-2.0, 6.0)
```

## Features

- Electron density, gradient, Laplacian, spin density (`.wfx`)
- Reduced density gradient and NCI indicator on 2D slices
- Density profiles along atom–atom paths
- Electron count integration
- Bond critical point (BCP) search with `.xyz` export

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
