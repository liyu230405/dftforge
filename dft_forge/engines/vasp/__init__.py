"""VASP engine: a second deterministic DFT engine behind the graph runtime.

Mirrors the QE engine's layering (writer / parser / tool) but for VASP's
four-file input convention (INCAR, POSCAR, KPOINTS, POTCAR) and vasprun.xml /
OUTCAR outputs. No LLM sits in the input-generation or parsing path.

POTCAR files are license-restricted and are never bundled or fabricated: the
writer fails closed when a required element's POTCAR is missing from the
user-supplied pseudopotential directory (``DFT_FORGE_VASP_PP_DIR``).
"""

from dft_forge.engines.vasp.tool import VASPCalcTool

__all__ = ["VASPCalcTool"]
