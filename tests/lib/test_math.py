#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test for lib."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from spectrakit.lib.math import FT, iFT

sys.path.append(str(Path(__file__).parents[1]))

def test_ft():
    """test the fourier transform."""

    t_max = np.pi*2
    test_t = np.linspace(0, t_max, 1000, endpoint=False)
    dt = test_t[1] - test_t[0]
    test_x = np.sin(test_t)

    nus, ft = FT(test_t, test_x)
    
    dnu = 1 / t_max
    numax = dnu*500

    # check that the frequency is returned as expected    
    assert_allclose(nus, np.arange(-numax, numax, dnu), atol=1e-11)
    # check that the smallest frequency is correct
    assert_allclose(nus[501], 1/2/np.pi)

    # check that the fourier transform is returned as expected (unitary ft with normal frequency)
    peak_height = 1 / dnu / 2j
    assert_allclose(ft[501], peak_height, rtol=1e-10)
    assert_allclose(ft[499], -peak_height, rtol=1e-10)

    # See carlson 2020
    # See https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_including_sampling_interval
    # See https://numpy.org/doc/stable/reference/routines.fft.html

def test_iFT():
    """test the inverse fourier transform."""

    t_max = np.pi*2
    test_t = np.linspace(0, t_max, 1000, endpoint=False)
    test_x = np.sin(test_t)

    nus, ft = FT(test_t, test_x)
    
    ts, xs = iFT(nus, ft)
    print(xs)

    assert_allclose(ts+np.pi, test_t, atol=1e-13)
    assert_allclose(np.roll(xs, 500), test_x, atol=1e-13)
    
