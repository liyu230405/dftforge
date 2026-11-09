"""Single source of truth for physical constants and unit conversions.

Values follow CODATA 2018 and the conventions used by Quantum ESPRESSO.
Import from here instead of redefining literals in each module — multiple
slightly-different copies of the same conversion factor have historically
caused unit bugs.
"""

from __future__ import annotations

import math

# Hartree energy in eV (CODATA 2018)
HARTREE_TO_EV = 27.211386245988
# Rydberg energy in eV (1 Ry = 0.5 Ha); QE prints energies in Ry
RY_TO_EV = 13.605693122994
EV_TO_RY = 1.0 / RY_TO_EV

# Bohr radius in Å (CODATA 2018)
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# FWHM / sigma ratio for a pure Gaussian: FWHM = 2 sqrt(2 ln2) sigma
GAUSS_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

# Coulomb constant e^2/(4 pi eps0) in eV·Å
COULOMB_EV_ANGSTROM = 14.3996454784255

# Phonon frequency conversions used by QE/anaddb output
THZ_TO_WAVENUMBER = 33.35640516  # 1 THz = 33.3564 cm^-1


def qe_label_to_element(label: str) -> str:
    """Map a QE species label (Fe1, FeA, O_up, si) to an element symbol.

    QE ATOMIC_SPECIES labels are free-form; AFM runs often emit FeA/FeB.
    The longest element prefix wins (so 'Cl1' -> Cl, not C).
    """
    from ase.data import chemical_symbols

    base = str(label).strip()
    for n in (2, 1):
        cand = base[:n]
        normalized = cand[0].upper() + cand[1:].lower() if n == 2 else cand.upper()
        if normalized in chemical_symbols:
            return normalized
    raise ValueError(f"cannot derive chemical element from QE species label {label!r}")
