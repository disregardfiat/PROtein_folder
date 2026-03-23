# horizon_physics: HQIV first-principles physics (lattice, horizon scalar, proteins).
# MIT License. Python 3.10+. No external deps beyond numpy.

try:
    from importlib.metadata import version as _distribution_version

    __version__ = _distribution_version("protein-folder")
except Exception:
    __version__ = "0.1.0"
