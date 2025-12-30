#!/usr/bin/env python
"""Example: Using DielectricSpectrum with MDAnalysis (traditional way).

This example shows that the refactored DielectricSpectrum class still works
exactly as before, maintaining backward compatibility.
"""

import MDAnalysis as mda
from spectrakit import DielectricSpectrumFlux
from spectrakit import DielectricSpectrum
import matplotlib.pyplot as plt

# Load your trajectory
u = mda.Universe("trajectories/n_1000/run.tpr", "trajectories/n_1000/run.trr")

# Select atoms (e.g., all atoms in the system)
atomgroup = u.select_atoms("all")

# Create DielectricSpectrum analysis
# This works exactly as before - the dipole calculation and spectrum
# calculation are still integrated in the class
analysis = DielectricSpectrumFlux(
    atomgroup=atomgroup,
    temperature=300,
    output_prefix="water_flux",
)

# Run the analysis
analysis.run(stop=1000)

# Access results
print(f"Average volume: {analysis.results.V:.2f} Ų")
print(f"Number of frames: {len(analysis.results.P_dot)}")
print(f"Frequency points: {len(analysis.results.nu)}")
print(f"Frequency range: {analysis.results.nu[0]:.4f} - "
      f"{analysis.results.nu[-1]:.4f} THz")

# Save results
analysis.save()

print("\nAnalysis complete!")
print(f"Results saved with prefix: {analysis.output_prefix}")

plt.semilogy(analysis.results.nu, analysis.results.susc.imag, label="flux")
plt.savefig("flux_spectrum.png")

analysis = DielectricSpectrum(
    atomgroup=atomgroup,
    temperature=300,
    output_prefix="water",
    segs=1,
    nobin = True,
)

# Run the analysis
analysis.run(stop=1000)
plt.semilogy(analysis.results.nu, analysis.results.susc.imag, label="dipole momemt")
plt.xlabel("$\nu$ in THz")
plt.ylabel("$\chi''$")
plt.legend()
plt.savefig("comparison.png")

