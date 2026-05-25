"""
User configuration — edit this file for your calculations.

Run from project root:
  python main.py
  python -m qdt
"""

# Input file (.wfx or .cube), relative to project root
INPUT_FILE = "data/benzeno.wfx"

# --- Analyses (True / False / None / "" = off) ---
RUN_DENSITY_SLICE = True
RUN_GRADIENT_SLICE = True
RUN_LAPLACIAN_SLICE = True
RUN_SPIN_DENSITY_SLICE = True
RUN_REDUCED_GRADIENT_SLICE = True
RUN_NCI_SLICE = True
RUN_PATH_PLOT = True
RUN_INTEGRATION = True
RUN_BCP_SEARCH = True

# --- 2D slice grid ---
SLICE_GRID_POINTS = 200
FD_STEP = 0.05
NORMALIZE_DIVERGING = True

SLICE_PADDING_X = None
SLICE_PADDING_Y = None
SLICE_X_RANGE = None
SLICE_Y_RANGE = None

# --- Atom overlays on slice plots ---
SHOW_ATOM_LABELS = True
ATOM_LABEL_COLOR = "black"
ATOM_LABEL_SIZE = 12
ATOM_LABEL_OFFSET = 0.05  # Angstrom
ATOM_MARKER_COLOR = "black"
ATOM_MARKER_SIZE = 3

PLANE = None
Z_POS = 0.0
ATOM_INDICES = [0, 1, 2]

PATH_ATOM1 = 0
PATH_ATOM2 = 11
PATH_POINTS = 500

INTEGRATION_GRID = 80
INTEGRATION_PADDING = None

BCP_GRID_POINTS = 120
BCP_PADDING = None
BCP_GRAD_TOL = 1e-6
BCP_MIN_DENSITY = 1e-2
BCP_POINTS_PER_PAIR = 15
BCP_MAX_DISTANCE_BOHR = 4.0
BCP_DUPLICATE_THRESHOLD = 0.2
BCP_MAX_ITER = 300
BCP_N_JOBS = -1
BCP_EXPORT_CUBE = True

CUBE_SMOOTHING_SIGMA = 0.0
