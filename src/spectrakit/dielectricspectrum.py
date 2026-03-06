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
from maicos.lib.util import (
    charge_neutral,
    citation_reminder,
    get_compound,
    render_docs,
)

from spectrakit.lib.math import FT, iFT, powerspectrum_from_timeseries, hilbert_transform
from spectrakit.lib.util import bin

logger = logging.getLogger(__name__)


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
        Average system volume in Angstrom^3.
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
    if df is not None:
        segs = np.max([int(n_frames * dt * df), 2])
    elif segs is None:
        segs = 20

    seglen = int(n_frames / segs)

    # Prefactor for susceptibility
    # Polarization: eÅ² to e m²
    pref = (scipy.constants.e) ** 2 * scipy.constants.angstrom**2
    # Volume: Ų to m³
    pref /= 3 * volume * scipy.constants.angstrom**3
    pref /= scipy.constants.k * temperature
    pref /= scipy.constants.epsilon_0

    # Create time array for segment (no padding)
    t = dt * np.arange(seglen)

    # Get frequencies (no padding)
    nu = FT(t, P[:seglen, 0])[0]

    # Initialize arrays
    susc = np.zeros(seglen, dtype=complex)
    dsusc = np.zeros(seglen, dtype=complex)

    # Loop over segments
    for s in range(0, segs):
        logging.info(f"\rSegment {s + 1} of {segs}")
        ss = 0 + 0j

        # Loop over x, y, z
        for i in range(3):
            FP: np.ndarray = FT(
                t,
                P[s * seglen : (s + 1) * seglen, i],
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
            susc += ss
        else:
            ds = ss - (susc / s)
            susc += ss
            dif = ss - (susc / (s + 1))
            ds.real *= dif.real
            ds.imag *= dif.imag
            # Variance by Welford's Method
            dsusc += ds

    dsusc.real = np.sqrt(dsusc.real)
    dsusc.imag = np.sqrt(dsusc.imag)

    # Normalization factor
    susc *= pref / (seglen * segs * dt)
    dsusc *= pref / (seglen * segs * dt)

    # Convert to THz
    nu = nu / (2 * np.pi)

    # Only keep positive frequencies
    pos_mask = nu >= 0
    nu = nu[pos_mask]
    susc = susc[pos_mask]
    dsusc = dsusc[pos_mask]

    results = {
        "t": t,
        "nu": nu,
        "susc": susc,
        "dsusc": dsusc,
    }

    logging.info(f"Length of segments:    {seglen} frames, {seglen * dt:.0f} ps")
    logging.info(f"Frequency spacing:    ~ {segs / (n_frames * dt):.5f} THz")

    # Bin data if there are too many points
    if not nobin and seglen > bins:
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
        pass

    return results


def calculate_spectrum_from_current(
    current: np.ndarray,
    dt: float,
    volume: float,
    temperature: float,
    segs: int | None = None,
    df: float | None = None,
    bins: int = 200,
    binafter: float = 20,
    nobin: bool = False,
) -> dict[str, np.ndarray]:
    """Calculate dielectric spectrum from current (dipole derivative) time series.

    This function computes the complex dielectric susceptibility from a current
    time series J(t) = dP/dt using the Fluctuation-Dissipation theorem. It is
    equivalent to :func:`calculate_spectrum_from_dipole` but works directly with
    the time derivative of the dipole moment, avoiding the need to integrate
    the current back to a dipole moment.

    Since FT(dP/dt) = iν·FT(P), we have |FT(J)|² = ν²·|FT(P)|², so
    |FT(J)|²/ν yields the same spectrum as |FT(P)|²·ν.

    Parameters
    ----------
    current : np.ndarray
        Current (dipole derivative) time series with shape (n_frames, 3)
        in units of e·Å/ps.
    dt : float
        Time step between frames in picoseconds.
    volume : float
        Average system volume in Angstrom^3.
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

    """
    J = current.copy()
    n_frames = len(J)

    # Determine number of segments
    if df is not None:
        segs = np.max([int(n_frames * dt * df), 2])
    elif segs is None:
        segs = 20

    seglen = int(n_frames / segs)

    # Prefactor for susceptibility (same as dipole version)
    pref = (scipy.constants.e) ** 2 * scipy.constants.angstrom**2
    pref /= 3 * volume * scipy.constants.angstrom**3
    pref /= scipy.constants.k * temperature
    pref /= scipy.constants.epsilon_0

    # Create time array for segment
    t = dt * np.arange(seglen)

    # Get frequencies
    nu = FT(t, J[:seglen, 0])[0]

    # Initialize arrays
    susc = np.zeros(seglen, dtype=complex)
    dsusc = np.zeros(seglen, dtype=complex)

    # Loop over segments
    for s in range(0, segs):
        logging.info(f"\rSegment {s + 1} of {segs}")
        ss = 0 + 0j

        # Loop over x, y, z
        for i in range(3):
            FJ: np.ndarray = FT(
                t,
                J[s * seglen : (s + 1) * seglen, i],
                False,
            )
            ss += FJ.real * FJ.real + FJ.imag * FJ.imag

        # Divide by ν instead of multiplying (since J = dP/dt)
        # Avoid division by zero at ν=0
        with np.errstate(divide="ignore", invalid="ignore"):
            ss_weighted = np.where(nu != 0, ss * 1j / nu, 0 + 0j)

        # Get the real part by Kramers-Kronig
        ift: np.ndarray = iFT(
            t,
            1j * np.sign(nu) * FT(nu, ss_weighted, False),
            False,
        )
        ss_weighted.real = ift.imag

        if s == 0:
            susc += ss_weighted
        else:
            ds = ss_weighted - (susc / s)
            susc += ss_weighted
            dif = ss_weighted - (susc / (s + 1))
            ds.real *= dif.real
            ds.imag *= dif.imag
            dsusc += ds

    dsusc.real = np.sqrt(dsusc.real)
    dsusc.imag = np.sqrt(dsusc.imag)

    # Normalization factor
    susc *= pref / (seglen * segs * dt)
    dsusc *= pref / (seglen * segs * dt)

    # Convert to THz
    nu = nu / (2 * np.pi)

    # Only keep positive frequencies
    pos_mask = nu >= 0
    nu = nu[pos_mask]
    susc = susc[pos_mask]
    dsusc = dsusc[pos_mask]

    results = {
        "t": t,
        "nu": nu,
        "susc": susc,
        "dsusc": dsusc,
    }

    logging.info(f"Length of segments:    {seglen} frames, {seglen * dt:.0f} ps")
    logging.info(f"Frequency spacing:    ~ {segs / (n_frames * dt):.5f} THz")

    # Bin data if there are too many points
    if not nobin and seglen > bins:
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


class Polarization(AnalysisBase):
    r"""Dipole moment time series from md trajectory.

    This module, given a molecular dynamics trajectory, produces a time series of the
    total dipole moment of the atomgroup and the whole system. It repairs molecules that
    are broken across periodic boundaries by reassembling them before calculating the
    dipole moment. The dipole moment is calculated as
    :math:`\mathbf{P} = \sum_i q_i \mathbf{r}_i`, where :math:`q_i`
    and :math:`\mathbf{r}_i` are the charge and position of atom :math:`i`, respectively.

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

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool = True,
        pack: bool = False,
        concfreq: int = 0,
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        wrap_compound = get_compound(atomgroup)
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=None,
            concfreq=concfreq,
            wrap_compound=wrap_compound,
            jitter=jitter,
        )

    def _prepare(self) -> None:
        self.dt = self._trajectory.dt * self.step
        self._obs.V = 0
        # TODO(@hejamu) Abstract this away
        self.P = np.zeros((self.n_frames, 3))
        self.P_total = np.zeros((self.n_frames, 3))
        self.V = np.zeros(self.n_frames)

    def _single_frame(self) -> None:
        self._obs.V += self._ts.volume
        self.P[self._frame_index, :] = np.dot(
            self.atomgroup.charges, self.atomgroup.positions
        )
        self.P_total[self._frame_index, :] = np.dot(
            self._ts.atoms.charges, self._ts.atoms.positions
        )
        self.V[self._frame_index] = self._ts.volume

    def _conclude(self) -> None:
        self.results.V = self.mean.V
        self.results.P = self.P
        self.results.P_total = self.P_total

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        # TODO(@hejamu): how to save both P and P_total?
        np.save(self.output_prefix + "tseries.npy", self.results.t)

        with Path(self.output_prefix + "V.txt").open(mode="w") as Vfile:
            Vfile.write(str(self.results.V))


class Current(AnalysisBase):
    r"""Current time series from md trajectory.

    This module, given a molecular dynamics trajectory, produces a time series of the
    curren of the atomgroup and the whole system.

    :math:`\mathbf{P} = \sum_i q_i \mathbf{r}_i`, where :math:`q_i`
    and :math:`\mathbf{r}_i` are the charge and position of atom :math:`i`, respectively.

    Please read and cite Rinne paper...

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

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool = False,
        pack: bool = False,
        concfreq: int = 0,
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        wrap_compound = get_compound(atomgroup)
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=None,
            concfreq=concfreq,
            wrap_compound=wrap_compound,
            jitter=jitter,
        )

    def _prepare(self) -> None:
        self.dt = self._trajectory.dt * self.step
        self._obs.V = 0
        # TODO(@hejamu) Abstract this away
        self.J = np.zeros((self.n_frames, 3))
        self.J_total = np.zeros((self.n_frames, 3))
        self.V = np.zeros(self.n_frames)

    def _single_frame(self) -> None:
        self._obs.V += self._ts.volume
        self.J[self._frame_index, :] = np.dot(
            self.atomgroup.charges, self.atomgroup.velocities
        )
        self.J_total[self._frame_index, :] = np.dot(
            self._ts.atoms.charges, self._ts.atoms.velocities
        )
        self.V[self._frame_index] = self._ts.volume

    def _conclude(self) -> None:
        self.results.V = self.mean.V
        self.results.J = self.J
        self.results.J_total = self.J_total

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        np.save(self.output_prefix + "tseries.npy", self.results.t)


class VACF(AnalysisBase):
    r"""Velocity autocorrelation function from md trajectory.

    This module, given a molecular dynamics trajectory, produces a time series of the
    velocity autocorrelation function of the atomgroup and the whole system. It repairs
    molecules that are broken across periodic boundaries by reassembling them before
    calculating the dipole moment. The dipole moment is calculated as
    :math:`\mathbf{P} = \sum_i q_i \mathbf{r}_i`, where :math:`q_i`
    and :math:`\mathbf{r}_i` are the charge and position of atom :math:`i`, respectively.


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

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool = True,
        pack: bool = False,
        concfreq: int = 0,
        jitter: float = 0.0,
    ) -> None:
        self._locals = locals()
        wrap_compound = get_compound(atomgroup)
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=None,
            concfreq=concfreq,
            wrap_compound=wrap_compound,
            jitter=jitter,
        )

    def _prepare(self) -> None: ...

    def _single_frame(self) -> None: ...

    def _conclude(self) -> None: ...

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        ...


def WienerKhinchin(time, timeseries, volume, temperature, spectrum_type="dipole"):
    """Calculate dielectric spectrum from timeseries using Wiener-Khinchin theorem.

    Parameters
    ----------
    time : np.ndarray
        Time steps of the data (n_frames) in ps.
    timeseries : np.ndarray
        Time series data with shape (n_frames, n_coordinates).
    volume : float
        System volume in Angstrom^3.
    temperature: float
        System temperature in Kelvin.
    spectrum_type : str, default="dipole"
        Type of spectrum to calculate. Either "dipole" or "flux".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (frequencies, complex dielectric spectrum).

    Notes
    -----
    Units:
    - dipole: e·Å
    - flux: e·Å / ps
    - dt: ps
    - volume: Å³

    References
    ----------
    https://dx.doi.org/10.1021/acs.jpca.0c04063

    """
    n_coordinates = timeseries.shape[1]
    powerspectrum = 0

    for i in range(n_coordinates):
        nu, powerspectrum_i = powerspectrum_from_timeseries(time, timeseries[:, i])
        powerspectrum += powerspectrum_i

    # Unit conversion: dipole in e·Å, volume in ų
    pref = scipy.constants.elementary_charge**2 * scipy.constants.angstrom**2
    pref /= 3 * volume * scipy.constants.angstrom**3
    pref /= scipy.constants.epsilon_0 * scipy.constants.k * temperature

    dielectricspectrum_imag = powerspectrum * pref

    if spectrum_type == "dipole":
        dielectricspectrum_imag *= np.pi * nu
    elif spectrum_type == "flux":
        with np.errstate(divide="ignore", invalid="ignore"):
            dielectricspectrum_imag = np.where(
                nu != 0, dielectricspectrum_imag / (nu * 4 * np.pi), 0.0
            )
    else:
        raise ValueError(f"{spectrum_type} not implemented. Use dipole or flux.")

    # Get the real-part by Kramers-Kronig / Hilbert Transform
    dielectricspectrum_real = - hilbert_transform(nu, dielectricspectrum_imag)

    return nu, dielectricspectrum_real + 1j * dielectricspectrum_imag
