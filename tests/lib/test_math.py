import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrakit


def test_FT():
    """Tests for the Fourier transform."""
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = spectrakit.lib.math.FT(x, sin)
    assert_allclose(abs(t[np.argmax(sin_FT)]), 5, rtol=1e-2)


def test_FT_unequal_spacing():
    """Tests for the Fourier transform with unequal spacing."""
    t = np.linspace(-np.pi, np.pi, 500)
    t[0] += 1e-5  # make it unequal
    sin = np.sin(5 * t)
    match = "Time series is not equally spaced."
    with pytest.raises(ValueError, match=match):
        spectrakit.lib.math.FT(t, sin)


def test_iFT():
    """Tests for the inverse Fourier transform."""
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = spectrakit.lib.math.FT(x, sin)
    sin_new = spectrakit.lib.math.iFT(t, sin_FT, indvar=False)
    # Shift to positive y domain to avoid comparing 0
    assert_allclose(2 + sin, 2 + sin_new.real, rtol=1e-1)


def test_iFT_unequal_spacing():
    """Tests for the inverse Fourier transform with unequal spacing."""
    t = np.linspace(-np.pi, np.pi, 500)
    t[0] += 1e-5  # make it unequal
    sin = np.sin(5 * t)
    match = "Frequency series is not equally spaced."
    with spectrakit.raises(ValueError, match=match):
        spectrakit.lib.math.iFT(t, sin)
