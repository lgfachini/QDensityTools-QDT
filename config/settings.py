"""
User configuration — edit this file for your calculations.

Run from project root:
  python main.py
  python -m qdt
"""

# Input file (.wfx or .cube), relative to project root
INPUT_FILE = "data/your_molecule.wfx"

# --- Analyses (True / False / None / "" = off) ---
RUN_DENSITY_SLICE = None
RUN_GRADIENT_SLICE = None
RUN_LAPLACIAN_SLICE = None
RUN_SPIN_DENSITY_SLICE = None
RUN_REDUCED_GRADIENT_SLICE = None
RUN_NCI_SLICE = True
RUN_PATH_PLOT = None
RUN_INTEGRATION = None
RUN_BCP_SEARCH = None

# --- 2D slice grid ---
SLICE_GRID_POINTS = 300
FD_STEP = 0.05
NORMALIZE_DIVERGING = True

SLICE_PADDING_X = None
SLICE_PADDING_Y = None
SLICE_X_RANGE = None
SLICE_Y_RANGE = None

PLANE = None
Z_POS = 0.0
ATOM_INDICES = [0, 1, 2]

PATH_ATOM1 = 0
PATH_ATOM2 = 1
PATH_POINTS = 500

INTEGRATION_GRID = 80
INTEGRATION_PADDING = None

BCP_GRID_POINTS = 120
BCP_PADDING = None
BCP_GRAD_TOL = 1e-5
BCP_MIN_DENSITY = 1e-3
BCP_POINTS_PER_PAIR = 15
BCP_MAX_DISTANCE_BOHR = 8.0
BCP_DUPLICATE_THRESHOLD = 0.2
BCP_MAX_ITER = 300
BCP_N_JOBS = -1
BCP_EXPORT_CUBE = True

CUBE_SMOOTHING_SIGMA = 0.0
