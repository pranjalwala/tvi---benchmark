"""
tvi/io/tiff_io.py
-----------------
Hardware-agnostic TIFF loading with metadata extraction.

LIBRARIES USED
--------------
tifffile   : read TIFF, extract tags             (open-source)
numpy      : array handling                       (open-source)
json       : parse ImageDescription manifest      (stdlib)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


@dataclass
class TiffImage:
    """Container for a loaded TIFF and its metadata."""
    array: np.ndarray          # (H, W) or (H, W, C)  uint8 or uint16
    dpi: float                 # scan / print resolution (dots per inch)
    bit_depth: int             # 8 or 16
    n_channels: int
    manifest: dict[str, Any]   # parsed ImageDescription JSON (may be empty)
    path: Path


def load_tiff(path: str | Path) -> TiffImage:
    """
    Load a TIFF and extract resolution and manifest metadata.

    Resolution is read from XResolution / ResolutionUnit tags.
    If tags are absent, dpi defaults to -1 (unknown).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    TiffImage
    """
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.asarray()

        # --- resolution ---
        dpi = _extract_dpi(tif)

        # --- bit depth ---
        bit_depth = _extract_bit_depth(tif, arr)

        # --- manifest from ImageDescription ---
        manifest = _extract_manifest(tif)

    n_channels = arr.shape[2] if arr.ndim == 3 else 1
    return TiffImage(
        array=arr,
        dpi=dpi,
        bit_depth=bit_depth,
        n_channels=n_channels,
        manifest=manifest,
        path=path,
    )


def save_tiff(
    arr: np.ndarray,
    path: str | Path,
    dpi: float,
    manifest: dict | None = None,
) -> None:
    """Save an array as a TIFF with DPI and optional manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    description = json.dumps(manifest) if manifest else ""
    tifffile.imwrite(
        str(path),
        arr,
        resolution=(dpi, dpi),
        resolutionunit=tifffile.RESUNIT.INCH,
        description=description,
        compression="deflate",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_dpi(tif: tifffile.TiffFile) -> float:
    try:
        page = tif.pages[0]
        tags = page.tags
        x_res_tag = tags.get("XResolution")
        unit_tag = tags.get("ResolutionUnit")
        if x_res_tag is None:
            return -1.0
        x_res = x_res_tag.value
        # x_res may be a rational (tuple) or a float
        if isinstance(x_res, tuple):
            x_res = x_res[0] / x_res[1] if x_res[1] != 0 else x_res[0]
        unit = unit_tag.value if unit_tag else 2  # 2 = inch, 3 = cm
        if unit == 3:
            x_res = x_res * 2.54   # convert cm⁻¹ → inch⁻¹ (dpi)
        return float(x_res)
    except Exception:
        return -1.0


def _extract_bit_depth(tif: tifffile.TiffFile, arr: np.ndarray) -> int:
    try:
        tag = tif.pages[0].tags.get("BitsPerSample")
        if tag:
            val = tag.value
            return int(val[0]) if isinstance(val, (tuple, list)) else int(val)
    except Exception:
        pass
    return int(arr.dtype.itemsize * 8)


def _extract_manifest(tif: tifffile.TiffFile) -> dict:
    try:
        tag = tif.pages[0].tags.get("ImageDescription")
        if tag:
            return json.loads(tag.value)
    except Exception:
        pass
    return {}
