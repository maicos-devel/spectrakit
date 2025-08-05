"""spectrakit: Compute spectra from MD simulation data."""

__authors__ = "MAICoS Developer Team"

from ._version import __version__  # noqa: F401
from .dielectricspectrum import DielectricSpectrum

__all__ = ["DielectricSpectrum"]
