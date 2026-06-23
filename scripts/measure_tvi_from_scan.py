#!/usr/bin/env python3
"""
scripts/measure_tvi_from_scan.py
----------------------------------
Full TVI measurement pipeline from a single scanned TIFF.

Pipeline
--------
load scan
  → calibrate scanner (from co-scanned wedge OR identity fallback)
  → detect fiducials + align sheet
  → extract patches
  → compute reflectances
  → Murray-Davies TVI curve
  → fit Yule-Nielsen n
  → save CSV
  → save plots

Usage
-----
# Minimal (with manifest embedded in TIFF):
python scripts/measure_tvi_from_scan.py \\
    --scan demo_data/tone_ramp_K.tif \\
    --channel K \\
    --demo

# With real config and step-wedge:
python scripts/measure_tvi_from_scan.py \\
    --scan scans/method_DBS/K/sheet1_scan1.tif \\
    --channel K \\
    --method DBS \\
    --config configs/my_lab.yaml \\
    --wedge scans/wedges/wedge_session1.tif \\
    --results_dir results/DBS_K/

# Without a step-wedge (identity scanner — for testing only):
python scripts/measure_tvi_from_scan.py \\
    --scan demo_data/tone_ramp_K.tif \\
    --channel K --demo --no_wedge
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("measure_tvi")

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure TVI from a single scanned TIFF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scan", required=True, help="Path to scanned tone-ramp TIFF.")
    p.add_argument("--channel", required=True,
                   help="Ink channel (C, M, Y, K, …).")
    p.add_argument("--method", default="unknown",
                   help="Halftoning method label (for output labelling).")
    p.add_argument("--config", default="configs/device_template.yaml",
                   help="Device YAML config.")
    p.add_argument("--demo", action="store_true",
                   help="Use built-in demo config.")
    p.add_argument("--wedge", default=None,
                   help="Path to co-scanned step-wedge TIFF for scanner calibration.")
    p.add_argument("--no_wedge", action="store_true",
                   help="Skip scanner calibration (identity: raw counts → reflectance directly).")
    p.add_argument("--results_dir", default="results",
                   help="Directory for CSV and plot outputs.")
    p.add_argument("--fit_yn", action="store_true",
                   help="Fit Yule-Nielsen n factor.")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip plot generation.")
    return p.parse_args()


def identity_calibration(bit_depth: int):
    """
    Fall-back scanner calibration when no step wedge is available.

    Assumes raw scanner counts are already linear and maps
    [0, max_val] → [0, 1].  NOT recommended for real measurements;
    use a step wedge whenever possible.
    """
    from tvi.calibration.scanner_linearization import fit_scanner_calibration
    max_val = (1 << bit_depth) - 1
    # Two-point linear: 0 → 0, max_val → 1
    raw = np.array([0.0, float(max_val)])
    nom = np.array([0.0, 1.0])
    return fit_scanner_calibration(raw, nom, degree=1, session_id="identity")


def calibrate_from_wedge(wedge_path: Path, config: dict):
    from tvi.io.tiff_io import load_tiff
    from tvi.calibration.scanner_linearization import (
        fit_scanner_calibration, measure_wedge_counts
    )
    tiff = load_tiff(wedge_path)
    ch_idx = config["scanner"]["channel_map"].get("K", 1)
    nominals = np.array(config["calibration"]["step_wedge_nominals"])
    if not tiff.manifest or "patches" not in tiff.manifest:
        log.warning(
            "Wedge TIFF has no manifest — cannot extract patches automatically. "
            "Using identity calibration instead."
        )
        return identity_calibration(config["scanner"]["bit_depth"])
    raw_counts = measure_wedge_counts(
        tiff.array, tiff.manifest, channel_idx=ch_idx,
        erosion_margin=config["measurement"]["patch_erosion_margin"],
    )
    return fit_scanner_calibration(
        raw_counts=raw_counts,
        nominal_reflectances=nominals,
        degree=config["calibration"]["linearization_poly_degree"],
        session_id=str(wedge_path.stem),
    )


def main() -> int:
    args = parse_args()
    config = DEMO_CONFIG if args.demo else _load_config(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scan_path = Path(args.scan)

    if not scan_path.exists():
        log.error("Scan file not found: %s", scan_path)
        return 1

    # ---- Scanner calibration ----
    if args.no_wedge:
        log.warning("--no_wedge: using identity calibration (not recommended for real data).")
        calibration = identity_calibration(config["scanner"]["bit_depth"])
    elif args.wedge:
        log.info("Calibrating scanner from wedge: %s", args.wedge)
        calibration = calibrate_from_wedge(Path(args.wedge), config)
    else:
        log.warning(
            "No --wedge specified. Attempting auto-discovery of wedge TIFF in same directory."
        )
        wedge_dir = scan_path.parent.parent.parent / "wedges"
        wedge_files = sorted(wedge_dir.glob("*.tif")) + sorted(wedge_dir.glob("*.tiff"))
        if wedge_files:
            log.info("Found wedge: %s", wedge_files[0])
            calibration = calibrate_from_wedge(wedge_files[0], config)
        else:
            log.warning("No wedge found. Using identity calibration.")
            calibration = identity_calibration(config["scanner"]["bit_depth"])

    # ---- Load and measure scan ----
    from tvi.io.tiff_io import load_tiff
    from tvi.measurement.measure_tone_ramps import measure_sheet

    log.info("Loading scan: %s", scan_path)
    tiff = load_tiff(scan_path)
    log.info("  Shape: %s  dtype: %s  dpi: %.0f", tiff.array.shape, tiff.array.dtype, tiff.dpi)

    sheet = measure_sheet(
        tiff=tiff,
        calibration=calibration,
        config=config,
        channel=args.channel,
        method=args.method,
        printer_label=config["printer"]["label"],
        sheet_idx=0,
        scan_idx=0,
        ref_fiducials=None,
    )

    log.info(
        "  r_paper=%.4f  r_ink=%.4f  n_patches=%d",
        sheet.r_paper, sheet.r_ink, len(sheet.patches),
    )

    # ---- TVI curve ----
    curve = sheet.to_tvi_curve()
    log.info("  TVI @ 50%% = %.2f pp", curve.tvi_at_50 * 100)

    # ---- Yule-Nielsen ----
    yn_result = None
    if args.fit_yn:
        from tvi.measurement.tvi_core import fit_yule_nielsen_n
        yn_mask = curve.a_nom >= config["yule_nielsen"]["fit_steps"][0] / 100.0
        yn_result = fit_yule_nielsen_n(
            a_nom=curve.a_nom,
            r_patch=curve.r_patch,
            r_ink=sheet.r_ink,
            r_paper=sheet.r_paper,
            fit_steps_mask=yn_mask,
            n_min=config["yule_nielsen"]["n_min"],
            n_max=config["yule_nielsen"]["n_max"],
            channel=args.channel,
            method=args.method,
        )
        log.info("  Yule-Nielsen n = %.4f  residual = %.6f", yn_result.n, yn_result.residual)

    # ---- Transfer function ----
    from tvi.calibration.transfer_function import fit_transfer_function, save_transfer_function_csv
    tf = fit_transfer_function(
        a_nom=curve.a_nom,
        a_print=curve.a_print,
        channel=args.channel,
        method=args.method,
        printer_label=config["printer"]["label"],
    )
    tf_csv = results_dir / f"transfer_{args.method}_{args.channel}.csv"
    save_transfer_function_csv(tf, tf_csv)
    log.info("  Transfer function saved: %s", tf_csv)

    # ---- CSV export ----
    import csv
    tvi_csv = results_dir / f"tvi_{args.method}_{args.channel}.csv"
    with open(tvi_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["a_nom", "a_print", "tvi_pp", "r_patch", "channel", "method", "printer"]
        if yn_result:
            header += ["yn_n", "yn_residual"]
        w.writerow(header)
        for a_n, a_p, tvi, r_p in zip(curve.a_nom, curve.a_print, curve.tvi, curve.r_patch):
            row = [
                f"{a_n:.4f}", f"{a_p:.4f}", f"{tvi*100:.4f}", f"{r_p:.4f}",
                args.channel, args.method, config["printer"]["label"],
            ]
            if yn_result:
                row += [f"{yn_result.n:.5f}", f"{yn_result.residual:.8f}"]
            w.writerow(row)
    log.info("  TVI CSV saved: %s", tvi_csv)

    # ---- Plots ----
    if not args.no_plots:
        from tvi.aggregation.statistics import aggregate_tvi_curve
        from tvi.visualization.tvi_plots import (
            plot_tvi_curves, plot_transfer_functions, save_figure
        )

        agg = aggregate_tvi_curve([curve])
        fig1 = plot_tvi_curves([agg], title=f"TVI — {args.method} / {args.channel}")
        save_figure(fig1, results_dir / f"tvi_curve_{args.method}_{args.channel}.png")

        fig2 = plot_transfer_functions([tf], title=f"Transfer Function — {args.method} / {args.channel}")
        save_figure(fig2, results_dir / f"transfer_{args.method}_{args.channel}.png")

        log.info("  Plots saved to %s", results_dir)

    log.info("Done.")
    return 0


def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    sys.exit(main())
