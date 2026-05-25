"""Smoke tests: package imports and public API."""


def test_version():
    import qdt

    assert qdt.__version__


def test_parsers():
    from qdt import CubeParser, WFXParser

    assert WFXParser is not None
    assert CubeParser is not None


def test_analysis_exports():
    from qdt.analysis import compute_s_values, integrate_electron_density

    assert callable(compute_s_values)
    assert callable(integrate_electron_density)


def test_settings_loads():
    from importlib import import_module

    settings = import_module("config.settings")
    assert hasattr(settings, "INPUT_FILE")
    assert hasattr(settings, "RUN_NCI_SLICE")
