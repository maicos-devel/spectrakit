#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helper and utilities functions that don't fit anywhere else."""

import numpy as np


def hann_window(N: int) -> np.ndarray:
    """Generate a Hann window of length N.

    Parameters
    ----------
    N : int
        Length of the window.

    Returns
    -------
    numpy.ndarray
        The Hann window.

    """
    n = np.arange(N)
    return np.sin(np.pi * n / N) ** 2
