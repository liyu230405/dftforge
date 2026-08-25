#!/usr/bin/env python3
"""Create canonical <Element>.upf symlinks for the GBRV USPP PBE v1.5 library.

Raw files keep their official names (e.g. si_pbe_v1.uspp.F.UPF) under raw/.
This script exposes each element as gbrv/<Element>.upf.
Hf has two variants: Hf.upf -> standard, Hf_plus4.upf -> plus4.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GBRV = HERE.parent / "assets" / "pseudos" / "gbrv"
RAW = GBRV / "raw"

NAME_RE = re.compile(
    r"^(?P<el>[a-z]+)_pbe(?:_(?P<flag>plus\d+))?_v(?P<ver>[\d.]+)\.uspp\.F\.UPF$"
)


def main() -> int:
    if not RAW.is_dir():
        print(f"ERROR: {RAW} not found. See README for download instructions.", file=sys.stderr)
        return 1

    count = 0
    pseudos_root = GBRV.parent
    for f in sorted(RAW.iterdir()):
        m = NAME_RE.match(f.name)
        if not m:
            print(f"skip (unparseable): {f.name}", file=sys.stderr)
            continue
        element = m.group("el").capitalize()
        suffix = "_plus4" if m.group("flag") else ""
        # QE does not accept subdirectories in pseudo filenames, so expose
        # elements both under gbrv/ and at the assets/pseudos top level.
        for link in (GBRV / f"{element}{suffix}.upf", pseudos_root / f"{element}{suffix}.upf"):
            if link.exists() or link.is_symlink():
                link.unlink()
            target = Path("raw") / f.name if link.parent == GBRV else Path("gbrv") / "raw" / f.name
            link.symlink_to(target)
            count += 1

    print(f"linked {count} pseudopotential symlinks under {pseudos_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
