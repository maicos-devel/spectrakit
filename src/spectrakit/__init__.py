"""spectrakit: Compute spectra from MD simulation data."""

__authors__ = "MAICoS Developer Team"

from ._version import __version__  # noqa: F401
from .dielectricspectrum import (
    DielectricSpectrum,
    calculate_spectrum_from_dipole,
)

from .dielectricspectrum_flux import (
    DielectricSpectrumFlux,
    calculate_spectrum_from_flux,
)

__all__ = ["DielectricSpectrum", "calculate_spectrum_from_dipole",
           "DielectricSpectrumFlux", "calculate_spectrum_from_flux"]
