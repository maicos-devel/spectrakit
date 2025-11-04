#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing dielectric spectra for bulk systems."""

import logging
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import scipy.constants
from maicos.core import AnalysisBase
from maicos.lib.math import FT, iFT
from maicos.lib.util import (
    bin,
    charge_neutral,
    citation_reminder,
    get_compound,
    render_docs,
)


def calculate_spectrum_from_dipole(
    dipole_moment: np.ndarray,
    dt: float,
    volume: float,
    temperature: float,
    segs: int | None = None,
    df: float | None = None,
    bins: int = 200,
    binafter: float = 20,
    nobin: bool = False,
) -> dict[str, np.ndarray]:
    """Calculate dielectric spectrum from dipole moment time series.

    This function computes the complex dielectric susceptibility from a dipole
    moment time series using the Fluctuation-Dissipation theorem. It can be used
    independently of MDAnalysis trajectories.

    Parameters
    ----------
    dipole_moment : np.ndarray
        Dipole moment time series with shape (n_frames, 3) in units of e·Å.
    dt : float
        Time step between frames in picoseconds.
    volume : float
        Average system volume in Ų.
    temperature : float
        System temperature in Kelvin.
    segs : int, optional
        Number of segments to break the trajectory into. If None and df is None,
        defaults to 20.
    df : float, optional
        Desired frequency spacing in THz. Overrides segs if provided.
    bins : int, default=200
        Number of bins for data averaging (logarithmic binning).
    binafter : float, default=20
        Number of low-frequency data points left unbinned.
    nobin : bool, default=False
        If True, prevents data binning.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - 't': time array
        - 'nu': frequency array (THz)
        - 'susc': complex susceptibility
        - 'dsusc': standard deviation of susceptibility
        - 'nu_binned': binned frequencies (if binning applied)
        - 'susc_binned': binned susceptibility (if binning applied)
        - 'dsusc_binned': binned std deviation (if binning applied)

    Notes
    -----
    The algorithm is based on the Fluctuation Dissipation Relation:
    χ(f) = -1/(3 V k_B T ε_0) L[θ(t) ⟨P(0) dP(t)/dt⟩]
    where L is the Laplace transformation.

    """
    P = dipole_moment.copy()
    n_frames = len(P)

    # Determine number of segments
    segs = np.max([int(n_frames * dt * df), 2]) if df is not None else 20

    if df is not None:
        segs = np.max([int(n_frames * dt * df), 2])

    seglen = int(n_frames / segs)

    # Prefactor for susceptibility
    # Polarization: eÅ² to e m²
    pref = (scipy.constants.e) ** 2 * scipy.constants.angstrom**2
    # Volume: Ų to m³
    pref /= 3 * volume * scipy.constants.angstrom**3
    pref /= scipy.constants.k * temperature
    pref /= scipy.constants.epsilon_0

    # Create time array
    t = dt * np.arange(n_frames)

    # If t too short to simply truncate
    if len(t) < 2 * seglen:
        t = np.append(t, t + t[-1] + dt)

    # Truncate t array
    t = t[: 2 * seglen]

    # Get frequencies
    nu = FT(
        t,
        np.append(P[:seglen, 0], np.zeros(seglen)),
    )[0]

    # Initialize arrays
    susc = np.zeros(seglen, dtype=complex)
    dsusc = np.zeros(seglen, dtype=complex)
    ss = np.zeros((2 * seglen), dtype=complex)

    # Loop over segments
    for s in range(0, segs):
        logging.info(f"\rSegment {s + 1} of {segs}")
        ss = 0 + 0j

        # Loop over x, y, z
        for i in range(3):
            FP: np.ndarray = FT(
                t,
                np.append(
                    P[s * seglen : (s + 1) * seglen, i],
                    np.zeros(seglen),
                ),
                False,
            )
            ss += FP.real * FP.real + FP.imag * FP.imag

        ss *= nu * 1j

        # Get the real part by Kramers-Kronig
        ift: np.ndarray = iFT(
            t,
            1j * np.sign(nu) * FT(nu, ss, False),
            False,
        )
        ss.real = ift.imag

        if s == 0:
            susc += ss[seglen:]
        else:
            ds = ss[seglen:] - (susc / s)
            susc += ss[seglen:]
            dif = ss[seglen:] - (susc / (s + 1))
            ds.real *= dif.real
            ds.imag *= dif.imag
            # Variance by Welford's Method
            dsusc += ds

    dsusc.real = np.sqrt(dsusc.real)
    dsusc.imag = np.sqrt(dsusc.imag)

    # 1/2 b/c it's the full FT, not only half-domain
    susc *= pref / (2 * seglen * segs * dt)
    dsusc *= pref / (2 * seglen * segs * dt)

    # Discard negative-frequency data
    nu = nu[seglen:] / (2 * np.pi)

    results = {
        "t": t,
        "nu": nu,
        "susc": susc,
        "dsusc": dsusc,
    }

    logging.info(f"Length of segments:    {seglen} frames, {seglen * dt:.0f} ps")
    logging.info(f"Frequency spacing:    ~ {segs / (n_frames * dt):.5f} THz")

    # Bin data if there are too many points
    if not (nobin or seglen <= bins):
        bin_indices = np.logspace(
            np.log(binafter) / np.log(10),
            np.log(len(susc)) / np.log(10),
            bins - binafter + 1,
        ).astype(int)
        bin_indices = np.unique(np.append(np.arange(binafter), bin_indices))[:-1]

        results["nu_binned"] = bin(nu, bin_indices)
        results["susc_binned"] = bin(susc, bin_indices)
        results["dsusc_binned"] = bin(dsusc, bin_indices)

        logging.info(f"Binning data above datapoint {binafter} in log-spaced bins")
        logging.info(f"Binned data consists of {len(susc)} datapoints")
    else:
        logging.info(f"Not binning data: there are {len(susc)} datapoints")

    return results


@render_docs
@charge_neutral(filter="error")
class DielectricSpectrum(AnalysisBase):
    r"""Linear dielectric spectrum.

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
    segs : int
        Sets the number of segments the trajectory is broken into.
    df : float
        The desired frequency spacing in THz. This determines the minimum frequency
        about which there is data. Overrides `segs` option.
    bins : int
        Determines the number of bins used for data averaging; (this parameter sets the
        upper limit). The data are by default binned logarithmically. This helps to
        reduce noise, particularly in the high-frequency domain, and also prevents plot
        files from being too large.
    binafter : int
        The number of low-frequency data points that are left unbinned.
    nobin : bool
        Prevents the data from being binned altogether. This can result in very large
        plot files and errors.

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
        segs: int = 20,
        df: float | None = None,
        bins: int = 200,
        binafter: float = 20,
        nobin: bool = False,
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
        self.segs = segs
        self.df = df
        self.bins = bins
        self.binafter = binafter
        self.nobin = nobin

    def _prepare(self) -> None:
        logging.info("Analysis of the linear dielectric spectrum.")
        # Print the Shane Carlson citation
        logging.info(citation_reminder("10.1021/acs.jpca.0c04063"))

        if len(self.output_prefix) > 0:
            self.output_prefix += "_"

        self.dt = self._trajectory.dt * self.step
        self.V = 0
        self.P = np.zeros((self.n_frames, 3))

    def _single_frame(self) -> None:
        self.V += self._ts.volume
        self.P[self._frame_index, :] = np.dot(
            self.atomgroup.charges, self.atomgroup.positions
        )

    def _conclude(self) -> None:
        self.results.V = self.V / self._index
        self.results.P = self.P

        logging.info("Calculating susceptibility and errors...")

        # Calculate spectrum using the decoupled function
        spectrum_results = calculate_spectrum_from_dipole(
            dipole_moment=self.results.P,
            dt=self.dt,
            volume=self.results.V,
            temperature=self.temperature,
            segs=self.segs,
            df=self.df,
            bins=self.bins,
            binafter=self.binafter,
            nobin=self.nobin,
        )

        # Store results
        self.results.t = spectrum_results["t"]
        self.results.nu = spectrum_results["nu"]
        self.results.susc = spectrum_results["susc"]
        self.results.dsusc = spectrum_results["dsusc"]

        if "nu_binned" in spectrum_results:
            self.results.nu_binned = spectrum_results["nu_binned"]
            self.results.susc_binned = spectrum_results["susc_binned"]
            self.results.dsusc_binned = spectrum_results["dsusc_binned"]

        # Store seglen for compatibility
        self.seglen = int(self.n_frames / self.segs)

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        np.save(self.output_prefix + "tseries.npy", self.results.t)

        with Path(self.output_prefix + "V.txt").open(mode="w") as Vfile:
            Vfile.write(str(self.results.V))

        np.save(self.output_prefix + "P_tseries.npy", self.results.P)

        suscfilename = "{}{}".format(self.output_prefix, "susc.dat")
        self.savetxt(
            suscfilename,
            np.transpose(
                [
                    self.results.nu,
                    self.results.susc.real,
                    self.results.dsusc.real,
                    self.results.susc.imag,
                    self.results.dsusc.imag,
                ]
            ),
            columns=["ν [THz]", "real(χ)", " Δ real(χ)", "imag(χ)", "Δ imag(χ)"],
        )

        logging.info("Susceptibility data saved as {suscfilename}")

        if not (self.nobin or self.seglen <= self.bins):
            suscfilename = "{}{}".format(self.output_prefix, "susc_binned.dat")
            self.savetxt(
                suscfilename,
                np.transpose(
                    [
                        self.results.nu_binned,
                        self.results.susc_binned.real,
                        self.results.dsusc_binned.real,
                        self.results.susc_binned.imag,
                        self.results.dsusc_binned.imag,
                    ]
                ),
                columns=["ν [THz]", "real(χ)", " Δ real(χ)", "imag(χ)", "Δ imag(χ)"],
            )

            logging.info("Binned susceptibility data saved as {suscfilename}")
