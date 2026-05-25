"""Command-line driver for QDensity Tools."""

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Optional

from qdt.analysis.bcps import find_critical_points_from_gradient_flow
from qdt.analysis.integration import integrate_electron_density
from qdt.io.parser import CubeParser, WFXParser
from qdt.viz import plotter


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_settings():
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return import_module("config.settings")


def cfg_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "":
            return False
        return v in ("1", "true", "yes", "on")
    return bool(value)


def _resolve_input_path(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = _project_root() / p
    return str(p.resolve())


def _grid_kwargs(settings):
    return dict(
        plane=settings.PLANE,
        atom_indices=settings.ATOM_INDICES if settings.PLANE is None else None,
        z_pos=settings.Z_POS,
        grid_points=settings.SLICE_GRID_POINTS,
        padding_x=settings.SLICE_PADDING_X,
        padding_y=settings.SLICE_PADDING_Y,
        x_range=settings.SLICE_X_RANGE,
        y_range=settings.SLICE_Y_RANGE,
        show_atom_labels=cfg_bool(getattr(settings, "SHOW_ATOM_LABELS", True)),
        atom_label_color=getattr(settings, "ATOM_LABEL_COLOR", "black"),
        atom_label_size=getattr(settings, "ATOM_LABEL_SIZE", 12),
        atom_marker_color=getattr(settings, "ATOM_MARKER_COLOR", "black"),
        atom_marker_size=getattr(settings, "ATOM_MARKER_SIZE", 3),
        atom_label_offset=getattr(settings, "ATOM_LABEL_OFFSET", 0.05),
    )


def _validate_slice_mode(settings):
    if settings.PLANE is not None and settings.ATOM_INDICES:
        raise ValueError("Set either PLANE or ATOM_INDICES, not both.")
    if settings.PLANE is None and not settings.ATOM_INDICES:
        raise ValueError("Set PLANE ('xy'/'xz'/'yz') or ATOM_INDICES (three indices).")


def _run_slice(name, func, parser, ext, settings, **extra):
    print(f"\n-> {name}...")
    _validate_slice_mode(settings)
    func(parser, ext=ext, **_grid_kwargs(settings), **extra)
    print("done.")


def main(input_file: Optional[str] = None) -> None:
    settings = _load_settings()
    input_file = input_file or settings.INPUT_FILE
    input_file = _resolve_input_path(input_file)
    ext = os.path.splitext(input_file)[1].lower()

    if ext == ".wfx":
        parser = WFXParser(input_file)
    elif ext == ".cube":
        parser = CubeParser(input_file, smoothing_sigma=settings.CUBE_SMOOTHING_SIGMA)
    else:
        raise ValueError(f"Unrecognized extension: {ext}")

    print(f"\nUsing {ext[1:]} parser: {input_file}\n")
    ran = False
    fd = dict(fd_step=settings.FD_STEP)
    norm = dict(normalize_diverging=settings.NORMALIZE_DIVERGING)

    if cfg_bool(settings.RUN_DENSITY_SLICE):
        _run_slice("density slice", plotter.plot_density_slice, parser, ext, settings)
        ran = True
    if cfg_bool(settings.RUN_GRADIENT_SLICE):
        _run_slice("gradient slice", plotter.plot_gradient_magnitude_slice, parser, ext, settings, **fd)
        ran = True
    if cfg_bool(settings.RUN_LAPLACIAN_SLICE):
        _run_slice("Laplacian slice", plotter.plot_laplacian_slice, parser, ext, settings, **fd, **norm)
        ran = True
    if cfg_bool(settings.RUN_SPIN_DENSITY_SLICE):
        _run_slice("spin density slice", plotter.plot_spin_density_slice, parser, ext, settings)
        ran = True
    if cfg_bool(settings.RUN_REDUCED_GRADIENT_SLICE):
        _run_slice("reduced gradient slice", plotter.plot_reduced_gradient_slice, parser, ext, settings, **fd)
        ran = True
    if cfg_bool(settings.RUN_NCI_SLICE):
        _run_slice("NCI indicator slice", plotter.plot_s_sign_lambda2_rho_slice, parser, ext, settings, **norm)
        ran = True

    if cfg_bool(settings.RUN_PATH_PLOT):
        print("\n-> path plot...")
        plotter.plot_density_gradient_laplacian_along_path(
            parser,
            ext=ext,
            atom1_index=settings.PATH_ATOM1,
            atom2_index=settings.PATH_ATOM2,
            points_count=settings.PATH_POINTS,
            fd_step=settings.FD_STEP,
        )
        print("done.")
        ran = True

    if cfg_bool(settings.RUN_INTEGRATION):
        print("\n-> electron density integration...")
        n = integrate_electron_density(
            parser,
            ext=ext,
            grid_points=settings.INTEGRATION_GRID,
            padding=settings.INTEGRATION_PADDING,
        )
        print(f"Integrated electrons: {n:.4f}")
        ran = True

    if cfg_bool(settings.RUN_BCP_SEARCH):
        find_critical_points_from_gradient_flow(
            parser,
            ext=ext,
            grid_points=settings.BCP_GRID_POINTS,
            padding=settings.BCP_PADDING,
            grad_tol=settings.BCP_GRAD_TOL,
            min_density=settings.BCP_MIN_DENSITY,
            n_points_per_pair=settings.BCP_POINTS_PER_PAIR,
            max_distance_bohr=settings.BCP_MAX_DISTANCE_BOHR,
            duplicate_threshold=settings.BCP_DUPLICATE_THRESHOLD,
            max_optimizer_iter=settings.BCP_MAX_ITER,
            n_jobs=settings.BCP_N_JOBS,
            export_cube=settings.BCP_EXPORT_CUBE,
        )
        print("BCP search finished.")
        ran = True

    if not ran:
        print("No analysis enabled. Set at least one RUN_* flag to True in config/settings.py.")
