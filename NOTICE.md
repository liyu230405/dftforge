# NOTICE

## External Benchmark

DFT-Forge uses the **DFT-Agent Bench** (https://github.com/Pluto235/DFT-Agent) as an external benchmark for evaluation purposes only.

- Fixed benchmark commit: `516d9ab8428535546ddb1af37dac215cbd92ef84`
- DFT-Forge does **not** import, copy, or derive code from DFT-Agent Bench
- DFT-Forge is an independent implementation from scratch
- The benchmark is used only for: task protocols, evaluation metrics, and comparative testing

## Third-Party Dependencies

DFT-Forge depends on the following open-source packages:
- **Quantum ESPRESSO** (GPL) — density functional theory code
- **ASE** (LGPL) — Atomic Simulation Environment
- **pymatgen** (MIT) — Materials genomics library
- **spglib** (BSD) — Space group library
- **seekpath** (BSD) — High-symmetry path generator
- **NumPy**, **SciPy**, **h5py**, **matplotlib** — scientific computing

No code from CatGo or SilicoLab is copied or imported.

## Pseudopotential Library

`assets/pseudos/gbrv/` contains the GBRV ultrasoft pseudopotentials
(USPP, PBE, v1.5) from https://www.physics.rutgers.edu/gbrv/
(SHA256-verified tarball `all_pbe_UPF_v1.5.tar.gz`). License: GNU Public
License. Please cite:

- K.F. Garrity, J.W. Bennett, K.M. Rabe, D. Vanderbilt,
  Comput. Mater. Sci. 81, 446 (2014). DOI: 10.1016/j.commatsci.2013.08.053

Material structures in `dft_forge/catalog/library.py` are derived from
Materials Project ground-state cells (58 MP-verified entries, M1 set);
each entry records its `mp_id`.
