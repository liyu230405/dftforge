"""Module entry point so ``python3 -m dft_forge.cli`` keeps working."""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
