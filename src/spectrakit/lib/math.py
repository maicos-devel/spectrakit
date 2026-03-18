# -*-
#
# Copyright (c) 2025 Authors and contributors (see the AUTHORS.rst file for the full
# list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helper functions for mathematical and physical operations."""

import numpy as np

# Max spacing variation in series that is allowed
dt_dk_tolerance = 1e-8  # (~1e-10 suggested)
dr_tolerance = 1e-6


def FT(
    t: np.ndarray, x: np.ndarray, indvar: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Discrete Fourier transformation using fast Fourier transformation (FFT).

    Parameters
    ----------
    t : numpy.ndarray
        Time values of the time series.
    x : numpy.ndarray
        Function values corresponding to the time series.
    indvar : bool
        If :obj:`True`, returns the FFT and frequency values. If :obj:`False`, returns
        only the FFT.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray) or numpy.ndarray
        If ``indvar`` is :obj:`True`, returns a tuple ``(k, xf2)`` where:
            - ``k`` (numpy.ndarray): Frequency values corresponding to the FFT.
            - ``xf2`` (numpy.ndarray): FFT of the input function, scaled by the time
              range and phase shifted.

        If indvar is :obj:`False`, returns the FFT (``xf2``) directly as a
        :class:`numpy.ndarray`.

    Raises
    ------
    RuntimeError
        If the time series is not equally spaced.

    Example
    -------
    >>> t = np.linspace(0, np.pi, 4)
    >>> x = np.sin(t)
    >>> k, xf2 = FT(t, x)
    >>> k
    array([-3. , -1.5,  0. ,  1.5])
    >>> np.round(xf2, 2)
    array([ 0.  +0.j  , -0.68+0.68j,  1.36+0.j  , -0.68-0.68j])

    See Also
    --------
    :func:`iFT` : For the inverse fourier transform.

    """
    dt_arr = np.diff(t)
    if not np.allclose(dt_arr, dt_arr[0], atol=dt_dk_tolerance):
        raise ValueError(
            "Time series is not equally spaced. Expected spacing "
            f"~ {dt_arr[0]}, got min={dt_arr.min()}, max={dt_arr.max()}."
        )
    dt = dt_arr[0]

    N = len(t)
    # calculate frequency values for FT
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dt) * 2 * np.pi)

    # calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    a, b = np.min(t), np.max(t)
    xf2 = xf * (b - a) / N * np.exp(-1j * k * a)

    if indvar:
        return k, xf2
    return xf2


def iFT(
    k: np.ndarray, xf: np.ndarray, indvar: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Inverse Fourier transformation using fast Fourier transformation (FFT).

    Takes the frequency series and the function as arguments. By default, returns the
    iFT and the time series. Setting indvar=False means the function returns only the
    iFT.

    Parameters
    ----------
    k : numpy.ndarray
        The frequency series.
    xf : numpy.ndarray
        The function series in the frequency domain.
    indvar : bool
        If :obj:`True`, return both the iFT and the time series. If :obj:`False`, return
        only the iFT.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray) or numpy.ndarray
        If indvar is :obj:`True`, returns a tuple containing the time series and the
        iFT. If indvar is :obj:`False`, returns only the iFT.

    Raises
    ------
    RuntimeError
        If the time series is not equally spaced.

    See Also
    --------
    :func:`FT` : For the Fourier transform.

    """
    dk_arr = np.diff(k)
    if not np.allclose(dk_arr, dk_arr[0], atol=dt_dk_tolerance):
        raise ValueError(
            "Frequency series is not equally spaced. Expected spacing ~ "
            f"{dk_arr[0]}, got min={dk_arr.min()}, max={dk_arr.max()}."
        )
    dk = dk_arr[0]

    N = len(k)
    x = np.fft.ifftshift(np.fft.ifft(xf))
    t = np.fft.ifftshift(np.fft.fftfreq(N, d=dk)) * 2 * np.pi
    if N % 2 == 0:
        x2 = x * np.exp(-1j * t * N * dk / 2.0) * N * dk / (2 * np.pi)
    else:
        x2 = x * np.exp(-1j * t * (N - 1) * dk / 2.0) * N * dk / (2 * np.pi)
    if indvar:
        return t, x2
    return x2


def powerspectrum_from_timeseries(t, timeseries):
    r"""Take a timeseries and calculate the power spectrum.

    The power spectrum is defined by
    :math:`\int_{\infty}^{\infty} dt e^{2 \pi i \nu t} \langle f(0) f(t) \rangle = \frac{1}{L_t} | f(\nu) |^2`
    as defined in the SI of Carlson20a.

    This function returns :math:`\frac{1}{L_t} | f(\nu) |^2`
    """
    nu, fourier = FT(t, timeseries, True)
    f_nu_squared = np.abs(fourier) ** 2
    L_t = t[-1]

    return nu, f_nu_squared / L_t


def correlation_function(
    timeseries,
    dt,
):
    r"""Take a timeseries and calculate the autocorrelation function.

    The autocorrelation function is defined by
    :math:`C(t) = \langle f(0) f(t) \rangle`
    as defined in the SI of Carlson20a.

    This function returns :math:`C(t)`.
    """
    n_t = len(timeseries)
    t = dt * np.arange(n_t)
    c_t = 0 + 0j

    for i in range(n_t):
        c_t[i] = np.mean(timeseries[0 : n_t - i] * timeseries[i:n_t])

    return t, c_t


def kramers_kronig(nu, f):
    """Implementation of Kramers Kronig using the Hilbert transform.

    See `https://en.wikipedia.org/wiki/Hilbert_transform#Relationship_with_the_Fourier_transform`.
    """
    time, ft = FT(nu, f, True)
    transformed = iFT(time, -1j * np.sign(nu) * ft, False)
    return transformed
