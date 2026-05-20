CHANGELOG file
--------------

The rules for spectrumkit's CHANGELOG file:

- entries are sorted newest-first.
- summarize sets of changes (don't reproduce every git log comment here).
- don't ever delete anything.
- keep the format consistent (79 char width, Y/M/D date format) and do not
  use tabs but use spaces for formatting

.. inclusion-marker-changelog-start

v0.1.0 (2026/05/20)
-------------------

Initial release of spectrumkit.

- rename spectrakit to spectrumkit (#19).
- documentation (#18).
- Fixed magnitude and Kramers–Kronig sign conventions (#16).
- ``lib/math.py``: phase-correct, physical-units FFT wrappers ``FT``/``iFT``,
  ``powerspectrum_from_timeseries``, and ``kramers_kronig`` (#16).
- ``lib/util.py``: ``bin()`` for logarithmic index-based averaging of arrays
  (#16).
- ``lib/preprocessing.py``: ``hann_window()`` windowing helper (#16).
- Fixed zero-padding behaviour in the FFT to correctly handle segment
  boundaries (#10).
- Standalone ``calculate_spectrum_from_dipole`` and
  ``calculate_spectrum_from_current`` functions decoupled from the MDAnalysis
  workflow, accepting pre-computed numpy arrays directly (#6).
- ``WienerKhinchin`` alternative analysis class for simpler single-segment
  calculations without error estimation (#6).
- GitHub Actions CI pipeline (tests, lint, build, release) (#1).
- ``tox`` environments for tests, lint, formatting, build, and docs.
- seperate spectrum code from maicos.
- ``DielectricSpectrum`` MDAnalysis-based analysis class that computes the
  frequency-dependent dielectric susceptibility from an MD trajectory using
  segmented averaging (Welford online variance) and Fourier transforms.
- Logarithmic frequency binning via ``bins``/``binafter``/``nobin`` parameters.
- Kramers–Kronig reconstruction of the real part of the susceptibility from
  the imaginary part via a Hilbert transform.

.. inclusion-marker-changelog-end
