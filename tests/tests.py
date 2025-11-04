#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DielectricSpectrum class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from spectrakit import DielectricSpectrum, calculate_spectrum_from_dipole

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402


class TestDielectricSpectrum:
    """Tests for the DielectricSpectrum class."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    def test_output_name(self, ag, monkeypatch, tmp_path):
        """Test output name."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag)
        ds.run()
        ds.save()
        with Path("susc.dat").open():
            pass
        with Path("P_tseries.npy").open():
            pass
        with Path("tseries.npy").open():
            pass
        with Path("V.txt").open():
            pass

    def test_output_name_prefix(self, ag, monkeypatch, tmp_path):
        """Test output name with custom prefix."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, output_prefix="foo")
        ds.run()
        ds.save()
        with Path("foo_susc.dat").open():
            pass
        with Path("foo_P_tseries.npy").open():
            pass
        with Path("foo_tseries.npy").open():
            pass
        with Path("foo_V.txt").open():
            pass

    def test_output_name_binned(self, ag, monkeypatch, tmp_path):
        """Test output name of binned data."""
        """
        The parameters are not meant to be sensible,
        but just to force the binned output.
        """
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, bins=5, binafter=0, segs=5)
        ds.run()
        ds.save()
        with Path("susc.dat").open():
            pass
        with Path("susc_binned.dat").open():
            pass
        with Path("P_tseries.npy").open():
            pass
        with Path("tseries.npy").open():
            pass
        with Path("V.txt").open():
            pass

    def test_output(self, ag, monkeypatch, tmp_path):
        """Test output values by comparing with magic numbers."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag)
        ds.run()

        V = 1559814.4
        nu = [0.0, 0.2, 0.5, 0.7, 1.0]
        susc = [27.5 + 0.0j, 2.9 + 22.3j, -5.0 + 3.6j, -0.5 + 10.7j, -16.8 + 3.5j]
        dsusc = [3.4 + 0.0j, 0.4 + 2.9j, 1.0 + 0.5j, 0.3 + 1.5j, 2.0 + 0.6j]

        assert_allclose(ds.V, V, rtol=1e-1)
        assert_allclose(ds.results.nu, nu, rtol=1)
        assert_allclose(ds.results.susc, susc, rtol=1e-1)
        assert_allclose(ds.results.dsusc, dsusc, rtol=1e-1)

    def test_binning(self, ag, monkeypatch, tmp_path):
        """Test binning & seglen case."""
        monkeypatch.chdir(tmp_path)

        ds = DielectricSpectrum(ag, nobin=False, segs=2, bins=49)
        ds.run()
        assert_allclose(np.mean(ds.results.nu_binned), 0.57, rtol=1e-2)
        ds.save()


class TestCalculateSpectrumFromDipole:
    """Tests for the standalone calculate_spectrum_from_dipole function."""

    def test_basic_calculation(self):
        """Test basic spectrum calculation from synthetic dipole data."""
        # Create synthetic dipole data
        n_frames = 100
        dipole_moment = np.random.randn(n_frames, 3) * 10
        dt = 2.0  # ps
        volume = 1000.0  # Ų
        temperature = 300.0  # K

        results = calculate_spectrum_from_dipole(
            dipole_moment=dipole_moment,
            dt=dt,
            volume=volume,
            temperature=temperature,
            segs=5,
            nobin=True,
        )

        # Check that results contain expected keys
        assert "nu" in results
        assert "susc" in results
        assert "dsusc" in results
        assert "t" in results

        # Check shapes and types
        assert isinstance(results["nu"], np.ndarray)
        assert isinstance(results["susc"], np.ndarray)
        assert isinstance(results["dsusc"], np.ndarray)
        assert results["susc"].dtype == complex
        assert results["dsusc"].dtype == complex

    def test_consistency_with_dielectric_spectrum(self, monkeypatch, tmp_path):
        """Test that standalone function gives same results as class method."""
        monkeypatch.chdir(tmp_path)

        # Load test data
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        ag = u.atoms

        # Run DielectricSpectrum
        ds = DielectricSpectrum(ag, segs=1, nobin=True)
        ds.run()

        # Use standalone function with extracted dipole data
        results = calculate_spectrum_from_dipole(
            dipole_moment=ds.results.P,
            dt=ds.dt,
            volume=ds.results.V,
            temperature=ds.temperature,
            segs=1,
            nobin=True,
        )

        # Results should be very similar
        assert_allclose(ds.results.nu, results["nu"], rtol=1e-10)
        assert_allclose(ds.results.susc, results["susc"], rtol=1e-10)
        assert_allclose(ds.results.dsusc, results["dsusc"], rtol=1e-10)
