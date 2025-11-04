#!/usr/bin/env python
"""Example: Using DielectricSpectrum with MDAnalysis (traditional way).

This example shows that the refactored DielectricSpectrum class still works
exactly as before, maintaining backward compatibility.
"""

import MDAnalysis as mda
from spectrakit import DielectricSpectrum

# Load your trajectory
u = mda.Universe("water.tpr", "water.trr")

# Select atoms (e.g., all atoms in the system)
atomgroup = u.select_atoms("all")

# Create DielectricSpectrum analysis
# This works exactly as before - the dipole calculation and spectrum
# calculation are still integrated in the class
analysis = DielectricSpectrum(
    atomgroup=atomgroup,
    temperature=300,
    output_prefix="water",
    segs=20,
    bins=200,
    binafter=20,
    nobin=False,
)

# Run the analysis
analysis.run()

# Access results
print(f"Average volume: {analysis.results.V:.2f} Ų")
print(f"Number of frames: {len(analysis.results.P)}")
print(f"Frequency points: {len(analysis.results.nu)}")
print(f"Frequency range: {analysis.results.nu[0]:.4f} - "
      f"{analysis.results.nu[-1]:.4f} THz")

# Save results
analysis.save()

print("\nAnalysis complete!")
print(f"Results saved with prefix: {analysis.output_prefix}")
