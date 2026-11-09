"""ABINIT engine: a third deterministic DFT engine behind the graph runtime.

Mirrors the QE/VASP engines' writer / parser / tool layering, but for ABINIT's
single ``.abi`` input-file convention (like QE, unlike VASP's four-file
workdir convention) and ABINIT's ``.abo`` text output plus DDB/anaddb phonon
chain. No LLM sits in the input-generation or parsing path.

ABINIT is free and open source (``conda install -c conda-forge abinit``), so
unlike the commercial VASP engine this one can be validated end-to-end on a
user's own machine. Pseudopotentials still fail closed: the writer errors when
a required element's pseudo is missing from the user-supplied directory
(``DFT_FORGE_ABINIT_PP_DIR``); nothing is fabricated.
"""

from dft_forge.engines.abinit.tool import ABINITCalcTool

__all__ = ["ABINITCalcTool"]
