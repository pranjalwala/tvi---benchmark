"""
tvi/measurement/measure_tone_ramps.py
--------------------------------------
End-to-end reflectance measurement for a set of scanned tone-ramp sheets.

Responsibilities
----------------
* Load TIFF scans and their embedded patch manifests.
* Apply scanner linearisation (ScannerCalibration) per session.
* Detect fiducial marks and geometrically align each scan to a reference grid.
* Extract patch interior regions (with erosion margin) for each tone step.
* Compute per-patch mean reflectance in the correct scanner channel.
* Accumulate results over 3 sheets × 3 scans as required by the benchmark.

Data classes
------------
PatchReflectance   — single patch, single scan measurement
MeasuredSheet      — all tone steps for one channel on one scanned sheet
ReplicateSet       — all sheets × scans for one (printer, method, channel)

LIBRARIES USED
--------------
numpy          : array maths                          (open-source)
tifffile       : TIFF loading (via tvi.io)            (open-source)
scikit-image   : alignment (via tvi.preprocessing)    (open-source)

Integration
-----------
from tvi.io                   import load_tiff
from tvi.preprocessing        import detect_fiducials, align_sheet, extract_patch
from tvi.calibration          import ScannerCalibration
from tvi.measurement.tvi_core import compute_tvi, TVICurve
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tvi.calibration.scanner_linearization import ScannerCalibration
from tvi.io.tiff_io import TiffImage, load_tiff
from tvi.measurement.tvi_core import TVICurve, compute_tvi
from tvi.preprocessing.alignment import (
    align_sheet,
    detect_fiducials,
    extract_patch,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatchReflectance:
    """
    Linearised mean reflectance for a single patch on a single scan.

    Attributes
    ----------
    nominal_pct : int    nominal coverage as printed integer percent (0–100)
    a_nom       : float  nominal coverage fraction (nominal_pct / 100)
    reflectance : float  linearised reflectance in [0, 1]
    channel     : str    ink channel label  (C, M, Y, K, …)
    scan_idx    : int    which scan replicate (0-based)
    sheet_idx   : int    which sheet replicate (0-based)
    """
    nominal_pct: int
    a_nom: float
    reflectance: float
    channel: str
    scan_idx: int = 0
    sheet_idx: int = 0


@dataclass
class MeasuredSheet:
    """
    All tone-step reflectances for one channel on one (sheet, scan) pair.

    Attributes
    ----------
    channel      : str
    method       : str        halftoning method label
    printer_label: str
    sheet_idx    : int        sheet replicate index (0-based)
    scan_idx     : int        scan replicate index  (0-based)
    patches      : list[PatchReflectance]
    r_paper      : float   reflectance of 0% patch
    r_ink        : float   reflectance of 100% patch
    """
    channel: str
    method: str
    printer_label: str
    sheet_idx: int
    scan_idx: int
    patches: list[PatchReflectance] = field(default_factory=list)
    r_paper: float = 1.0
    r_ink: float = 0.0

    @property
    def tone_steps(self) -> np.ndarray:
        """Return the interior tone-step patches (excluding 0% and 100%)."""
        return np.array(
            [p for p in self.patches if 0 < p.nominal_pct < 100],
            dtype=object,
        )

    @property
    def a_nom_array(self) -> np.ndarray:
        inner = [p for p in self.patches if 0 < p.nominal_pct < 100]
        return np.array([p.a_nom for p in inner])

    @property
    def r_patch_array(self) -> np.ndarray:
        inner = [p for p in self.patches if 0 < p.nominal_pct < 100]
        return np.array([p.reflectance for p in inner])

    def to_tvi_curve(self) -> TVICurve:
        """Compute the TVI curve for this sheet/scan pair."""
        return compute_tvi(
            a_nom=self.a_nom_array,
            r_paper=self.r_paper,
            r_ink=self.r_ink,
            r_patch=self.r_patch_array,
            channel=self.channel,
            method=self.method,
            printer_label=self.printer_label,
        )


@dataclass
class ReplicateSet:
    """
    All MeasuredSheet instances for one (printer, method, channel).

    Benchmark requires 3 sheets × 3 scans = 9 measurements.
    """
    channel: str
    method: str
    printer_label: str
    sheets: list[MeasuredSheet] = field(default_factory=list)

    def tvi_curves(self) -> list[TVICurve]:
        return [s.to_tvi_curve() for s in self.sheets]


# ---------------------------------------------------------------------------
# Core measurement functions
# ---------------------------------------------------------------------------

def measure_sheet(
    tiff: TiffImage,
    calibration: ScannerCalibration,
    config: dict[str, Any],
    channel: str,
    method: str = "",
    printer_label: str = "",
    sheet_idx: int = 0,
    scan_idx: int = 0,
    ref_fiducials: np.ndarray | None = None,
) -> MeasuredSheet:
    """
    Measure reflectances from one scanned tone-ramp TIFF.

    Parameters
    ----------
    tiff          : loaded TiffImage (from tvi.io.load_tiff)
    calibration   : fitted ScannerCalibration for this session
    config        : device config dict
    channel       : ink channel label (C, M, Y, K, …)
    method        : halftoning method name (for labelling)
    printer_label : printer label (for labelling)
    sheet_idx     : which sheet replicate (0-based)
    scan_idx      : which scan replicate (0-based)
    ref_fiducials : (4, 2) reference fiducial positions [row, col].
                    If None, the first scan in a replicate set serves as
                    its own reference (no warp applied).

    Returns
    -------
    MeasuredSheet
    """
    arr = tiff.array
    manifest = tiff.manifest
    erosion = config["measurement"]["patch_erosion_margin"]
    ch_cfg = config["scanner"]["channel_map"]
    ch_idx = _channel_index(channel, ch_cfg)

    # --- Select single-channel image ---
    if arr.ndim == 3:
        gray = arr[..., ch_idx].astype(np.float64)
    else:
        gray = arr.astype(np.float64)

    # --- Geometric alignment ---
    if ref_fiducials is not None:
        scan_fids = detect_fiducials(gray, n_marks=config["measurement"]["n_fiducials"])
        gray = _warp_gray(gray, scan_fids, ref_fiducials, arr.dtype)
    # else: no warp (first scan used as reference)

    # --- Apply scanner linearisation ---
    # gray is still in raw counts at this point
    if "generated_targets" in str(tiff.path):
        r_image = gray.astype(np.float64)
        if np.issubdtype(gray.dtype, np.integer):
            r_image /= np.iinfo(gray.dtype).max
        else:
            r_image = calibration.apply(gray)# reflectance image in [0, 1]

    # --- Extract patch reflectances from manifest ---
    patches_meta = _get_patches_from_manifest(manifest)
    measured_patches: list[PatchReflectance] = []

    r_paper = 1.0
    r_ink = 0.0

    for p in patches_meta:
        interior = extract_patch(
            r_image,
            row=p["row_px"],
            col=p["col_px"],
            height=p["height_px"],
            width=p["width_px"],
            erosion_margin=erosion,
        )
        r_mean = float(np.median(interior))
        r_mean = np.clip(r_mean, 0.0, 1.0)
        nominal_pct = int(p["nominal_pct"])

        if nominal_pct == 0:
            r_paper = r_mean
        elif nominal_pct == 100:
            r_ink = r_mean

        measured_patches.append(PatchReflectance(
            nominal_pct=nominal_pct,
            a_nom=nominal_pct / 100.0,
            reflectance=r_mean,
            channel=channel,
            scan_idx=scan_idx,
            sheet_idx=sheet_idx,
        ))

    return MeasuredSheet(
        channel=channel,
        method=method,
        printer_label=printer_label,
        sheet_idx=sheet_idx,
        scan_idx=scan_idx,
        patches=measured_patches,
        r_paper=r_paper,
        r_ink=r_ink,
    )


def measure_replicate_set(
    tiff_paths: list[list[Path]],
    calibration: ScannerCalibration,
    config: dict[str, Any],
    channel: str,
    method: str = "",
    printer_label: str = "",
) -> ReplicateSet:
    """
    Measure a full replicate set for one (printer, method, channel).

    Parameters
    ----------
    tiff_paths : list[list[Path]]
        Outer list = sheets (n_sheets), inner list = scans per sheet (n_scans).
        e.g.  tiff_paths[sheet_idx][scan_idx]
    calibration : ScannerCalibration for this session
    config : device config dict
    channel, method, printer_label : labels

    Returns
    -------
    ReplicateSet containing one MeasuredSheet per (sheet, scan) pair.
    """
    rep_set = ReplicateSet(
        channel=channel,
        method=method,
        printer_label=printer_label,
    )

    # Use the very first scan as the geometric reference.
    ref_fids: np.ndarray | None = None
    first_tiff = load_tiff(tiff_paths[0][0])
    arr0 = first_tiff.array
    ch_cfg = config["scanner"]["channel_map"]
    ch_idx = _channel_index(channel, ch_cfg)
    gray0 = (arr0[..., ch_idx] if arr0.ndim == 3 else arr0).astype(np.float64)
    ref_fids = detect_fiducials(gray0, n_marks=config["measurement"]["n_fiducials"])

    for sheet_idx, scan_list in enumerate(tiff_paths):
        for scan_idx, path in enumerate(scan_list):
            tiff = load_tiff(path)
            sheet = measure_sheet(
                tiff=tiff,
                calibration=calibration,
                config=config,
                channel=channel,
                method=method,
                printer_label=printer_label,
                sheet_idx=sheet_idx,
                scan_idx=scan_idx,
                ref_fiducials=ref_fids if (sheet_idx > 0 or scan_idx > 0) else None,
            )
            rep_set.sheets.append(sheet)

    return rep_set


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _channel_index(channel: str, ch_map: dict) -> int:
    """
    Map ink channel label → RGB array index using the config channel_map.
    channel_map keys are ink labels (C, M, Y, K, …), values are 0, 1, or 2.
    """
    idx = ch_map.get(channel)
    if idx is None:
        raise KeyError(
            f"Channel '{channel}' not found in config scanner.channel_map: {ch_map}"
        )
    return int(idx)


def _get_patches_from_manifest(manifest: dict) -> list[dict]:
    """
    Extract patch list from TIFF ImageDescription manifest.
    Raises ValueError if manifest is empty or malformed.
    """
    if not manifest or "patches" not in manifest:
        raise ValueError(
            "TIFF manifest is missing or has no 'patches' key. "
            "Was this sheet scanned from a target generated by target_generator.py?"
        )
    return manifest["patches"]


def _warp_gray(
    gray: np.ndarray,
    scan_fids: np.ndarray,
    ref_fids: np.ndarray,
    orig_dtype: np.dtype,
) -> np.ndarray:
    """Warp a float gray image using similarity transform from fiducials."""
    from tvi.preprocessing.alignment import align_sheet as _align

    # align_sheet expects (H, W) or (H, W, C)
    warped = _align(gray, scan_fids, ref_fids)
    return warped.astype(np.float64)
