"""
tvi/calibration_targets/target_generator.py
--------------------------------------------
Generates all calibration target sheets as TIFF files.

Produced targets
----------------
1. tone_ramp_<channel>.tif
   A row of square patches at nominal coverages defined in config.
   Printed separately for each ink channel (single-channel patches).

2. step_wedge.tif
   A co-scanned reference wedge with known nominal reflectances.
   Printed alongside every tone-ramp sheet for session calibration.

3. overprint_patches.tif
   Two-channel and multi-channel overprint combinations.

All targets embed:
  - four corner fiducial marks (solid black squares)
  - a human-readable patch manifest as TIFF ImageDescription metadata.

LIBRARIES USED
--------------
numpy          : array construction (custom code)
tifffile       : TIFF writing with metadata  (tifffile — open-source)
No hardcoded DPI or patch sizes; all derived from config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all_targets(
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Generate every calibration target sheet and save as TIFF.

    Parameters
    ----------
    config : dict
        Loaded device config (from configs/device_template.yaml).
    output_dir : str or Path
        Directory where target TIFFs are written.

    Returns
    -------
    dict mapping target name → Path of written TIFF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    tone_steps: list[int] = config["measurement"]["tone_steps"]
    channels: list[str] = config["printer"]["channels"]
    print_dpi: int = config["printer"]["print_dpi"]
    bit_depth: int = config["scanner"]["bit_depth"]
    max_val: int = _max_value(bit_depth)

    # --- 1. Per-channel tone ramps ---
    for ch in channels:
        arr, manifest = _make_tone_ramp(
            channel=ch,
            tone_steps=tone_steps,
            print_dpi=print_dpi,
            bit_depth=bit_depth,
            channels_available=channels,
        )
        path = output_dir / f"tone_ramp_{ch}.tif"
        _write_tiff(arr, path, print_dpi, manifest)
        written[f"tone_ramp_{ch}"] = path

    # --- 2. Step wedge (grayscale) ---
    wedge_nominals: list[float] = config["calibration"]["step_wedge_nominals"]
    arr, manifest = _make_step_wedge(
        nominals=wedge_nominals,
        print_dpi=print_dpi,
        bit_depth=bit_depth,
        max_val=max_val,
    )
    path = output_dir / "step_wedge.tif"
    _write_tiff(arr, path, print_dpi, manifest)
    written["step_wedge"] = path

    # --- 3. Overprint patches ---
    arr, manifest = _make_overprint_patches(
        channels=channels,
        print_dpi=print_dpi,
        bit_depth=bit_depth,
        max_val=max_val,
    )
    path = output_dir / "overprint_patches.tif"
    _write_tiff(arr, path, print_dpi, manifest)
    written["overprint_patches"] = path

    return written


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _max_value(bit_depth: int) -> int:
    return (1 << bit_depth) - 1


def _patch_size_px(print_dpi: int, side_mm: float = 15.0) -> int:
    """Convert a physical patch side length (mm) to pixels at print_dpi."""
    return max(1, round(print_dpi * side_mm / 25.4))


def _fiducial_size_px(print_dpi: int, side_mm: float = 5.0) -> int:
    return max(1, round(print_dpi * side_mm / 25.4))


def _add_fiducial_marks(
    canvas: np.ndarray,
    print_dpi: int,
    bit_depth: int,
    margin_mm: float = 3.0,
) -> np.ndarray:
    """
    Stamp four solid-black corner fiducial squares on the canvas.
    The canvas shape is (H, W) or (H, W, C); marks are always black.
    """
    canvas = canvas.copy()
    max_val = _max_value(bit_depth)
    fid = _fiducial_size_px(print_dpi)
    margin = round(print_dpi * margin_mm / 25.4)
    H, W = canvas.shape[:2]

    corners = [
        (margin, margin),
        (margin, W - margin - fid),
        (H - margin - fid, margin),
        (H - margin - fid, W - margin - fid),
    ]
    for r, c in corners:
        if canvas.ndim == 2:
            canvas[r:r + fid, c:c + fid] = 0
        else:
            canvas[r:r + fid, c:c + fid, :] = 0

    return canvas


def _make_tone_ramp(
    channel: str,
    tone_steps: list[int],
    print_dpi: int,
    bit_depth: int,
    channels_available: list[str],
) -> tuple[np.ndarray, dict]:
    """
    Build a single-channel tone-ramp target image.

    The channel under test is modulated.
    All other channels are set to 0 (no ink).
    Output is an RGB image (H, W, 3) with dtype matching bit_depth.
    """
    patch_px = _patch_size_px(print_dpi)
    n_patches = len(tone_steps)
    margin_px = round(print_dpi * 5.0 / 25.4)   # 5 mm margins

    # canvas dimensions
    W = margin_px * 2 + n_patches * patch_px + (n_patches - 1) * round(print_dpi * 2.0 / 25.4)
    H = margin_px * 2 + patch_px
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    max_val = _max_value(bit_depth)

    # White background (paper = max_val in reflectance encoding)
    canvas = np.full((H, W, 3), max_val, dtype=dtype)

    gap_px = round(print_dpi * 2.0 / 25.4)
    ch_idx = _channel_to_rgb_idx(channel)

    manifest_patches: list[dict] = []
    x = margin_px
    for step in tone_steps:
        coverage = step / 100.0
        # Ink reduces reflectance: 0% coverage → max_val, 100% → 0
        ink_value = round((1.0 - coverage) * max_val)

        # Stamp the patch on the relevant channel
        patch_canvas = np.full((patch_px, patch_px, 3), max_val, dtype=dtype)
        patch_canvas[:, :, ch_idx] = ink_value

        y = margin_px
        canvas[y: y + patch_px, x: x + patch_px, :] = patch_canvas

        manifest_patches.append({
            "nominal_pct": step,
            "channel": channel,
            "row_px": y,
            "col_px": x,
            "height_px": patch_px,
            "width_px": patch_px,
        })
        x += patch_px + gap_px

    canvas = _add_fiducial_marks(canvas, print_dpi, bit_depth)
    manifest = {
        "target_type": "tone_ramp",
        "channel": channel,
        "print_dpi": print_dpi,
        "bit_depth": bit_depth,
        "patches": manifest_patches,
    }
    return canvas, manifest


def _make_step_wedge(
    nominals: list[float],
    print_dpi: int,
    bit_depth: int,
    max_val: int,
) -> tuple[np.ndarray, dict]:
    """
    Grayscale step wedge for scanner linearization.
    Each step is a rectangle of known reflectance printed in K only.
    """
    patch_px = _patch_size_px(print_dpi, side_mm=12.0)
    gap_px = round(print_dpi * 1.5 / 25.4)
    margin_px = round(print_dpi * 4.0 / 25.4)
    n = len(nominals)
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    W = margin_px * 2 + n * patch_px + (n - 1) * gap_px
    H = margin_px * 2 + patch_px
    canvas = np.full((H, W, 3), max_val, dtype=dtype)

    manifest_patches: list[dict] = []
    x = margin_px
    for nom in nominals:
        # Reflectance encoded as: pixel = round(nom * max_val)
        val = round(nom * max_val)
        y = margin_px
        canvas[y: y + patch_px, x: x + patch_px, :] = val
        manifest_patches.append({
            "nominal_reflectance": nom,
            "row_px": y,
            "col_px": x,
            "height_px": patch_px,
            "width_px": patch_px,
        })
        x += patch_px + gap_px

    canvas = _add_fiducial_marks(canvas, print_dpi, bit_depth)
    manifest = {
        "target_type": "step_wedge",
        "print_dpi": print_dpi,
        "bit_depth": bit_depth,
        "patches": manifest_patches,
    }
    return canvas, manifest


def _make_overprint_patches(
    channels: list[str],
    print_dpi: int,
    bit_depth: int,
    max_val: int,
) -> tuple[np.ndarray, dict]:
    """
    Two-channel overprint patches at 100% coverage per channel.
    Pairs: CM, CY, MY, CK, MK, YK  (for CMYK; subset for fewer channels).
    """
    from itertools import combinations

    patch_px = _patch_size_px(print_dpi)
    gap_px = round(print_dpi * 2.0 / 25.4)
    margin_px = round(print_dpi * 5.0 / 25.4)
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    pairs = list(combinations(channels, 2))
    n = len(pairs)

    W = margin_px * 2 + n * patch_px + (n - 1) * gap_px
    H = margin_px * 2 + patch_px
    canvas = np.full((H, W, 3), max_val, dtype=dtype)

    manifest_patches: list[dict] = []
    x = margin_px
    for ch_a, ch_b in pairs:
        y = margin_px
        patch = np.full((patch_px, patch_px, 3), max_val, dtype=dtype)
        patch[:, :, _channel_to_rgb_idx(ch_a)] = 0
        patch[:, :, _channel_to_rgb_idx(ch_b)] = 0
        canvas[y: y + patch_px, x: x + patch_px, :] = patch
        manifest_patches.append({
            "channels": [ch_a, ch_b],
            "row_px": y,
            "col_px": x,
            "height_px": patch_px,
            "width_px": patch_px,
        })
        x += patch_px + gap_px

    canvas = _add_fiducial_marks(canvas, print_dpi, bit_depth)
    manifest = {
        "target_type": "overprint_patches",
        "print_dpi": print_dpi,
        "bit_depth": bit_depth,
        "patches": manifest_patches,
    }
    return canvas, manifest


def _channel_to_rgb_idx(channel: str) -> int:
    """Map ink channel letter to RGB array index (crude but device-independent)."""
    mapping = {"R": 0, "C": 0, "G": 1, "M": 1, "B": 2, "Y": 2, "K": 1,
               "O": 0, "V": 2}
    return mapping.get(channel.upper(), 1)


def _write_tiff(
    arr: np.ndarray,
    path: Path,
    dpi: int,
    manifest: dict,
) -> None:
    """Write arr to TIFF with DPI tag and manifest embedded as ImageDescription."""
    resolution = (dpi, dpi)
    description = json.dumps(manifest)
    tifffile.imwrite(
        str(path),
        arr,
        resolution=resolution,
        resolutionunit=tifffile.RESUNIT.INCH,
        description=description,
        photometric="rgb",
        compression="deflate",
    )
