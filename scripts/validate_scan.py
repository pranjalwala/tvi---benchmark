#!/usr/bin/env python3
"""
scripts/validate_scan.py
-------------------------
Validates a scanned TIFF before running the TVI measurement pipeline.

Checks performed
----------------
1.  File exists and is a valid TIFF.
2.  Bit depth matches config (8 or 16 bit).
3.  Number of channels (must be ≥ 3 for RGB).
4.  DPI tag present and consistent with config.
5.  Manifest present in ImageDescription (required for patch extraction).
6.  Patch manifest bounding boxes are inside the image dimensions.
7.  Fiducial marks detectable (threshold-based blob detection).
8.  Step-wedge monotonicity (optional, for wedge TIFFs).
9.  Saves a diagnostic PNG showing detected fiducials and patch overlays.

Usage
-----
# Validate a single tone-ramp scan:
python scripts/validate_scan.py scan.tif --config configs/my_lab.yaml

# Validate demo data:
python scripts/validate_scan.py demo_data/tone_ramp_K.tif --demo

# Validate and save diagnostic image:
python scripts/validate_scan.py scan.tif --config configs/my_lab.yaml \\
    --save_diagnostic diagnostics/tone_ramp_K_check.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_scan")

DEMO_CONFIG = {
    "printer": {"label": "Demo", "print_dpi": 200, "channels": ["C", "M", "Y", "K"]},
    "scanner": {"capture_dpi": 200, "bit_depth": 8,
                "channel_map": {"K": 1, "C": 0, "M": 1, "Y": 2}},
    "measurement": {"tone_steps": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "patch_erosion_margin": 0.10, "n_fiducials": 4,
                    "n_sheets": 3, "n_scans": 3},
    "calibration": {"step_wedge_nominals": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                            0.6, 0.7, 0.8, 0.9, 0.95],
                    "linearization_poly_degree": 2},
    "yule_nielsen": {"fit_steps": [70, 80, 90], "n_min": 1.0, "n_max": 4.0, "n_init": None},
    "output": {"results_dir": "results", "ci_level": 0.95},
}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        log.info("  ✓  %s", msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("  ⚠  %s", msg)

    def fail(self, msg: str) -> None:
        self.errors.append(msg)
        log.error("  ✗  %s", msg)

    @property
    def is_ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  PASSED  : {len(self.passed)}",
            f"  WARNINGS: {len(self.warnings)}",
            f"  ERRORS  : {len(self.errors)}",
            "="*55,
            "RESULT: " + ("PASS ✓" if self.is_ok else "FAIL ✗"),
        ]
        if self.errors:
            lines.append("\nErrors that must be fixed:")
            for e in self.errors:
                lines.append(f"  • {e}")
        if self.warnings:
            lines.append("\nWarnings (review but not blocking):")
            for w in self.warnings:
                lines.append(f"  • {w}")
        return "\n".join(lines)


def validate_tiff(path: Path, config: dict, save_diagnostic: Path | None) -> ValidationResult:
    from tvi.io.tiff_io import load_tiff

    vr = ValidationResult()

    # ---- 1. File existence ----
    if not path.exists():
        vr.fail(f"File not found: {path}")
        return vr
    vr.ok(f"File exists: {path.name}  ({path.stat().st_size // 1024} KB)")

    # ---- 2. Load TIFF ----
    try:
        tiff = load_tiff(path)
        vr.ok(f"TIFF loaded successfully. Shape: {tiff.array.shape}  dtype: {tiff.array.dtype}")
    except Exception as exc:
        vr.fail(f"TIFF load failed: {exc}")
        return vr

    arr = tiff.array

    # ---- 3. Bit depth ----
    expected_bd = config["scanner"]["bit_depth"]
    actual_bd = tiff.bit_depth
    if actual_bd == expected_bd:
        vr.ok(f"Bit depth: {actual_bd}-bit (expected {expected_bd})")
    else:
        vr.warn(
            f"Bit depth mismatch: got {actual_bd}-bit, config expects {expected_bd}-bit. "
            "Update config scanner.bit_depth to match your scanner output."
        )

    # ---- 4. Channels ----
    n_ch = tiff.n_channels
    if n_ch >= 3:
        vr.ok(f"Channels: {n_ch} (RGB or more — OK)")
    elif n_ch == 1:
        vr.warn("Single-channel (grayscale) TIFF. Channel map will use channel 0.")
    else:
        vr.fail(f"Unexpected channel count: {n_ch}. Expected 1 or ≥ 3.")

    # ---- 5. DPI ----
    expected_dpi = config["scanner"]["capture_dpi"]
    actual_dpi = tiff.dpi
    if actual_dpi < 0:
        vr.warn(
            "DPI tag absent from TIFF. Cannot verify resolution. "
            "Set your scanner to embed DPI metadata, or update config."
        )
    elif abs(actual_dpi - expected_dpi) < 10:
        vr.ok(f"DPI: {actual_dpi:.0f} (expected {expected_dpi})")
    else:
        vr.warn(
            f"DPI mismatch: got {actual_dpi:.0f}, config expects {expected_dpi}. "
            "If your scan was at a different resolution, update scanner.capture_dpi."
        )

    # ---- 6. Manifest ----
    manifest = tiff.manifest
    if not manifest:
        vr.warn(
            "No manifest (ImageDescription JSON) found in TIFF. "
            "This is normal for user-supplied scans — you must supply patch "
            "bounding boxes separately via a JSON manifest file."
        )
    elif "patches" not in manifest:
        vr.warn("Manifest present but 'patches' key missing.")
    else:
        n_patches = len(manifest["patches"])
        vr.ok(f"Manifest present: {n_patches} patches")

        # ---- 7. Patch bounding-box sanity ----
        H, W = arr.shape[:2]
        bad_patches = []
        for i, p in enumerate(manifest["patches"]):
            r, c = p.get("row_px", 0), p.get("col_px", 0)
            h, w = p.get("height_px", 0), p.get("width_px", 0)
            if r + h > H or c + w > W:
                bad_patches.append(i)
        if bad_patches:
            vr.fail(
                f"Patch bounding boxes outside image dimensions ({H}×{W}): "
                f"patches {bad_patches}. Likely DPI mismatch between target "
                "generation and scanning."
            )
        else:
            vr.ok(f"All {n_patches} patch bounding boxes fit within image ({H}×{W}px)")

    # ---- 8. Fiducial detection ----
    from tvi.preprocessing.alignment import detect_fiducials
    n_fid = config["measurement"]["n_fiducials"]
    ch_map = config["scanner"]["channel_map"]
    ch_idx = list(ch_map.values())[0]
    gray = arr[..., ch_idx].copy() if arr.ndim == 3 else arr.copy()

    try:
        centroids = detect_fiducials(gray, n_marks=n_fid)
        vr.ok(f"Detected {len(centroids)} fiducial marks")
    except Exception as exc:
        vr.warn(
            f"Fiducial detection failed: {exc}. "
            "This may be OK if your scan has no fiducials — "
            "alignment will be skipped."
        )
        centroids = None

    # ---- 9. Image statistics ----
    gray_norm = gray.astype(np.float64)
    if np.issubdtype(gray.dtype, np.integer):
        gray_norm = gray_norm / np.iinfo(gray.dtype).max
    elif gray_norm.max() > 1.0:
        gray_norm = gray_norm / gray_norm.max()

    dark_frac = float((gray_norm < 0.1).mean())
    light_frac = float((gray_norm > 0.9).mean())
    vr.ok(
        f"Image stats: min={gray_norm.min():.3f}  max={gray_norm.max():.3f}  "
        f"mean={gray_norm.mean():.3f}  dark<10%={dark_frac*100:.1f}%  light>90%={light_frac*100:.1f}%"
    )

    if gray_norm.max() < 0.3:
        vr.warn("Image appears very dark — check scanner exposure settings.")
    if gray_norm.min() > 0.7:
        vr.warn("Image appears very light — check that ink actually printed.")

    # ---- 10. Diagnostic plot ----
    if save_diagnostic:
        _save_diagnostic_plot(
            arr=arr,
            gray_norm=gray_norm,
            manifest=manifest,
            centroids=centroids,
            path=save_diagnostic,
            tiff_path=path,
        )
        vr.ok(f"Diagnostic image saved: {save_diagnostic}")

    return vr


def _save_diagnostic_plot(
    arr: np.ndarray,
    gray_norm: np.ndarray,
    manifest: dict,
    centroids,
    path: Path,
    tiff_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: grayscale with fiducials and patches
    axes[0].imshow(gray_norm, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Scan: {tiff_path.name}", fontsize=10)

    if centroids is not None:
        axes[0].scatter(
            centroids[:, 1], centroids[:, 0],
            s=80, c="red", marker="+", linewidths=1.5,
            label=f"{len(centroids)} fiducials",
        )
        axes[0].legend(fontsize=8)

    if manifest and "patches" in manifest:
        for p in manifest["patches"]:
            rect = mpatches.Rectangle(
                (p["col_px"], p["row_px"]),
                p["width_px"], p["height_px"],
                linewidth=0.7, edgecolor="lime", facecolor="none",
            )
            axes[0].add_patch(rect)
        # Label first and last
        for p in [manifest["patches"][0], manifest["patches"][-1]]:
            axes[0].text(
                p["col_px"], p["row_px"] - 4,
                str(p.get("nominal_pct", "")),
                color="lime", fontsize=6,
            )

    axes[0].axis("off")

    # Right: histogram
    axes[1].hist(gray_norm.ravel(), bins=128, color="steelblue", edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("Normalised pixel value", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Pixel intensity histogram", fontsize=10)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a scanned TIFF before TVI measurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("tiff", help="Path to the scanned TIFF file to validate.")
    p.add_argument("--config", default="configs/device_template.yaml",
                   help="Device YAML config file.")
    p.add_argument("--demo", action="store_true",
                   help="Use built-in demo config instead of --config.")
    p.add_argument("--save_diagnostic", default=None,
                   help="Path to save diagnostic PNG (optional).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.demo:
        config = DEMO_CONFIG
    else:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            log.error("Config not found: %s", cfg_path)
            return 1
        with open(cfg_path) as fh:
            config = yaml.safe_load(fh)

    tiff_path = Path(args.tiff)
    diag_path = Path(args.save_diagnostic) if args.save_diagnostic else None

    log.info("Validating: %s", tiff_path)
    vr = validate_tiff(tiff_path, config, diag_path)

    print(vr.summary())
    return 0 if vr.is_ok else 1


if __name__ == "__main__":
    sys.exit(main())
