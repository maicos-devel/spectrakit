#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing dielectric spectra for bulk systems from the flux."""

import logging
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import scipy.constants
from maicos.core import AnalysisBase
from .lib.math import FT, iFT
from maicos.lib.util import (
    bin,
    charge_neutral,
    citation_reminder,
    get_compound,
    render_docs,
)


def calculate_spectrum_from_flux(
    flux: np.ndarray,
    dt: float,
    volume: float,
    temperature: float,
) -> dict[str, np.ndarray]:
    """Calculate dielectric spectrum from flux time series.

    This function computes the complex dielectric susceptibility from a flux time series
    using the Fluctuation-Dissipation theorem. It can be used
    independently of MDAnalysis trajectories.

    Parameters
    ----------
    flux : np.ndarray
        flux time series with shape (n_frames, 3) in units of e·Å / ps
    dt : float
        Time step between frames in picoseconds.
    volume : float
        Average system volume in Angstrom^3.
    temperature : float
        System temperature in Kelvin.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - 'nu': frequency array (THz)
        - 'susc': complex susceptibility

    Notes
    -----
    The algorithm is based on the Fluctuation Dissipation Relation:
    χ(f) = -1/(3 V k_B T ε_0) L[θ(t) ⟨P(0) dP(t)/dt⟩]
    where L is the Laplace transformation.

    """
    P_dot = flux.copy()
    n_frames = len(P_dot)

    # Prefactor for susceptibility
    # Polarization: eÅ / ps 
    pref = (scipy.constants.e) ** 2 * scipy.constants.angstrom**2 / scipy.constants.pico**2 
    # Volume: Ų to m³
    pref /= 3 * volume * scipy.constants.angstrom**3
    pref /= scipy.constants.k * temperature
    pref /= scipy.constants.epsilon_0
    pref /= 4 * np.pi # from the time derivative calculation 

    # Create time array 
    t = dt * np.arange(n_frames)

    # Initialize arrays
    susc = np.zeros(n_frames, dtype=complex)

    # Get the real part of the susceptibility by Wiener-Khinchin
    f_nu_squared = 0 + 0j

    # Loop over x, y, z
    for i in range(3):
        nu, FP_dot = FT(t, P_dot[:, i], True)
        f_nu_squared += np.abs(FP_dot)**2

    susc = f_nu_squared * 1j / nu

    # handle the zero frequency bin
    # TODO: find a better way to do this
    susc[nu == 0] = 0

    # Get the real part by Kramers-Kronig
    kramers_kronig: np.ndarray = iFT(
        t,
        1j * np.sign(nu) * FT(nu, susc, False),
        False,
    )
    susc.real = kramers_kronig.imag

    # Only keep positive frequencies
    pos_mask = nu >= 0
    nu = nu[pos_mask]
    susc = susc[pos_mask]

    results = {
        "t": t, #TODO: should t be calculated outside of this function?
        "nu": nu,
        "susc": susc,
    }
    return results

@render_docs
@charge_neutral(filter="error")
class DielectricSpectrumFlux(AnalysisBase):
    r"""Linear dielectric spectrum from the flux.

    This module, given a molecular dynamics trajectory, produces a `.txt` file
    containing the complex dielectric function as a function of the (linear, not radial
    - i.e., :math:`\nu` or :math:`f`, rather than :math:`\omega`) frequency, along with
    the associated standard deviations. The algorithm is based on the Fluctuation
    Dissipation Relation: :math:`\chi(f) = -1/(3 V k_B T \varepsilon_0)
    \mathcal{L}[\theta(t) \langle P(0) dP(t)/dt\rangle]`, where :math:`\mathcal{L}` is
    the Laplace transformation.

    .. note::
        The polarization time series and the average system volume are also saved.

    Please read and cite :footcite:p:`carlsonExploringAbsorptionSpectrum2020`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    ${TEMPERATURE_PARAMETER}
    ${OUTPUT_PREFIX_PARAMETER}

    Attributes
    ----------
    results

    References
    ----------
    .. footbibliography::

    """

    # TODO(@hejamu): set up script to calc spectrum at intervals while calculating
    # polarization for very big-data trajectories
    # TODO(@PicoCentauri): merge with molecular version?
    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = True,
        pack: bool = True,
        concfreq: int = 0,
        temperature: float = 300,
        output_prefix: str = "",
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        wrap_compound = get_compound(atomgroup)
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            concfreq=concfreq,
            wrap_compound=wrap_compound,
            jitter=jitter,
        )
        self.temperature = temperature
        self.output_prefix = output_prefix

    def _prepare(self) -> None:
        logging.info("Analysis of the linear dielectric spectrum.")
        # Print the Shane Carlson citation
        logging.info(citation_reminder("10.1021/acs.jpca.0c04063"))

        if len(self.output_prefix) > 0:
            self.output_prefix += "_"

        self.dt = self._trajectory.dt * self.step
        self.V = 0
        self.P_dot = np.zeros((self.n_frames, 3))

    def _single_frame(self) -> None:
        self.V += self._ts.volume
        self.P_dot[self._frame_index, :] = np.dot(
            self.atomgroup.charges, self.atomgroup.velocities
        )

    def _conclude(self) -> None:
        self.results.V = self.V / self._index
        self.results.P_dot = self.P_dot

        logging.info("Calculating susceptibility and errors...")

        # Calculate spectrum using the decoupled function
        spectrum_results = calculate_spectrum_from_flux(
            flux=self.results.P_dot,
            dt=self.dt,
            volume=self.results.V,
            temperature=self.temperature,
        )

        # Store results
        self.results.t  = spectrum_results["t"]
        self.results.nu = spectrum_results["nu"]
        self.results.susc = spectrum_results["susc"]

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        np.save(self.output_prefix + "tseries.npy", self.results.t)

        with Path(self.output_prefix + "V.txt").open(mode="w") as Vfile:
            Vfile.write(str(self.results.V))

        np.save(self.output_prefix + "P_tseries.npy", self.results.P_dot)

        suscfilename = "{}{}".format(self.output_prefix, "susc.dat")
        self.savetxt(
            suscfilename,
            np.transpose(
                [
                    self.results.nu,
                    self.results.susc.real,
                    self.results.susc.imag,
                ]
            ),
            columns=["ν [THz]", "real(χ)", "imag(χ)"],
        )

        logging.info("Susceptibility data saved as {suscfilename}")

