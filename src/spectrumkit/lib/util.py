#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helper and utilities functions that don't fit anywhere else."""

import numpy as np


def bin(a: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Average array values in bins for easier plotting.

    Parameters
    ----------
    a : numpy.ndarray
        The input array to be averaged.
    bins : numpy.ndarray
        The array containing the indices where each bin begins.

    Returns
    -------
    numpy.ndarray
        The averaged array values.

    Notes
    -----
    The "bins" array should contain the INDEX (integer) where each bin begins.

    """
    if np.iscomplex(a).any():
        avg = np.zeros(len(bins), dtype=complex)  # average of data
    else:
        avg = np.zeros(len(bins))

    count = np.zeros(len(bins), dtype=int)
    ic = -1

    for i in range(0, len(a)):
        if i in bins:
            ic += 1  # index for new average
        avg[ic] += a[i]
        count[ic] += 1

    return avg / count
