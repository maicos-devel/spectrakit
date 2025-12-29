import numpy as np
from scipy.fftpack import dst

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
            QUESTION:: WHY WOULD YOU PHASE SHIFT IT? 

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
    dt = (t[-1] - t[0]) / float(len(t) - 1)

    if (abs(np.diff(t) - dt) > dt_dk_tolerance).any():
        raise ValueError("Time series not equally spaced!")

    N = len(t)

    # calculate frequency values for FT
    nu = np.fft.fftshift(np.fft.fftfreq(N, d=dt))

    # calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    xf2 = xf * dt

    if indvar:
        return nu, xf2
    return xf2


def iFT(
    k: np.ndarray, xf: np.ndarray, indvar: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Inverse Fourier transformation using fast Fourier transformation (FFT).

    Takes the frequency series and the function as arguments. By default, returns the
    iFT and the time values. Setting indvar=False means the function returns only the
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
        If indvar is :obj:`True`, returns a tuple containing the time values and the
        iFT. If indvar is :obj:`False`, returns only the iFT.

    Raises
    ------
    RuntimeError
        If the time series is not equally spaced.

    See Also
    --------
    :func:`FT` : For the Fourier transform.

    """
    dk = (k[-1] - k[0]) / float(len(k) - 1)

    if (abs(np.diff(k) - dk) > dt_dk_tolerance).any():
        raise ValueError("Time series not equally spaced!")

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

