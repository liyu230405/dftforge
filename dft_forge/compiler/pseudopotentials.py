"""Read electron counts and explicit recommendations from pseudopotentials.

Metadata and the content hash come from the same file snapshot.  Missing
electron counts are errors: treating an unreadable header as zero silently
underestimates both the electron count and the required number of bands.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any


def _fortran_float(value: str) -> float:
    return float(value.strip().replace("D", "E").replace("d", "e"))


def _positive_recommendation(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = _fortran_float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _identity(stat: os.stat_result) -> tuple[int, ...]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def read_pseudo_metadata(path: str | Path) -> dict[str, Any]:
    """Return validated metadata for UPF v1/v2 or a QE-compatible GTH file.

    Required keys are ``filename``, ``sha256``, ``z_valence`` and
    ``pseudo_type`` (``None`` when the header does not identify the type).
    ``functional`` and positive ``suggested_ecutwfc`` / ``suggested_ecutrho``
    in Ry are included only when explicitly available.  Zero cutoff values
    in GBRV headers mean that no recommendation was provided.
    """
    path = Path(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        data = stream.read()
        after = os.fstat(stream.fileno())
    if _identity(before) != _identity(after) or _identity(after) != _identity(path.stat()):
        raise ValueError(f"Pseudopotential changed while reading: {path.name}")
    text = data.decode("utf-8", errors="replace")
    metadata: dict[str, Any] = {
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "pseudo_type": None,
    }
    raw_valence: str | None = None
    wfc: float | None = None
    rho: float | None = None
    header = re.search(r"<PP_HEADER\b([^>]*)>", text, re.IGNORECASE)
    if header:
        # Parse just the header attributes: older UPF information sections
        # contain unescaped text and are not necessarily well-formed XML.
        attrs = {
            key.lower(): value.strip()
            for key, _quote, value in re.findall(
                r"([\w]+)\s*=\s*(['\"])(.*?)\2", header.group(1), re.DOTALL
            )
        }
        raw_valence = attrs.get("z_valence")
        metadata["pseudo_type"] = attrs.get("pseudo_type") or None
        if attrs.get("functional"):
            metadata["functional"] = attrs["functional"]
        wfc = _positive_recommendation(attrs.get("wfc_cutoff"))
        # ONCV files can use an isolated rho_cutoff for the radial-grid
        # extent (e.g. Si_r.upf: 15.27).  Only accept this as an energy
        # recommendation alongside a positive wavefunction cutoff.
        if wfc is not None:
            rho = _positive_recommendation(attrs.get("rho_cutoff"))
            if rho is not None and rho < wfc:
                rho = None
        if raw_valence is None:
            end = re.search(r"</PP_HEADER\s*>", text[header.end():], re.IGNORECASE)
            body = text[header.end():header.end() + end.start()] if end else ""
            for line in body.splitlines():
                valence = re.match(r"\s*(\S+)\s+Z\s+valence\b", line, re.IGNORECASE)
                if valence:
                    raw_valence = valence.group(1)
                pseudo_type = re.match(
                    r"\s*(NC|US|USPP|PAW)\s+.*pseudopotential", line, re.IGNORECASE
                )
                if pseudo_type:
                    metadata["pseudo_type"] = pseudo_type.group(1).upper()
                functional = re.match(r"\s*(.*?)\s+Exchange.Correlation functional", line, re.IGNORECASE)
                if functional and functional.group(1):
                    metadata["functional"] = functional.group(1).strip()
                cutoff = re.match(
                    r"\s*(\S+)\s+(\S+)\s+Suggested cutoff for wfc and rho\b",
                    line, re.IGNORECASE,
                )
                if cutoff:
                    wfc = _positive_recommendation(cutoff.group(1))
                    rho = _positive_recommendation(cutoff.group(2))
    else:
        # QE's Goedecker/Hartwigsen format records the ionic charge as
        # zion, after the atomic number.  It is the valence electron count.
        gth = re.search(
            r"^\s*\S+\s+(\S+)\s+\S+\s+zatom\s*,\s*zion\s*,\s*pspdat\b",
            text, re.MULTILINE | re.IGNORECASE,
        )
        if gth:
            raw_valence = gth.group(1)
            metadata["pseudo_type"] = "GTH"

    try:
        valence = _fortran_float(raw_valence) if raw_valence is not None else float("nan")
    except ValueError:
        valence = float("nan")
    if not math.isfinite(valence) or valence <= 0:
        raise ValueError(
            f"Pseudopotential {path.name} has missing or invalid z_valence; "
            "expected a finite positive electron count in a UPF or GTH header"
        )
    metadata["z_valence"] = valence
    if wfc is not None:
        metadata["suggested_ecutwfc"] = wfc
    if rho is not None:
        metadata["suggested_ecutrho"] = rho
    return metadata
