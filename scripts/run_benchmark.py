#!/usr/bin/env python3
"""
scripts/run_benchmark.py
-------------------------
TVI benchmark entry point.

Usage
-----
# Full run with real scans:
python scripts/run_benchmark.py \
    --config configs/my_lab.yaml \
    --scans_root data/scans \
    --results_dir results/run_01

# Dry run with synthetic data (no real scans needed):
python scripts/run_benchmark.py --dry_run --results_dir results/dry_run

Pipeline
--------
load config
  ↓
load scans (TIFF) — or generate synthetic data in dry_run mode
  ↓
calibrate scanner (step wedge → polynomial linearisation)
  ↓
measure reflectances per patch
  ↓
compute TVI curves (Murray-Davies)
  ↓
optional: fit Yule-Nielsen n
  ↓
fit printer transfer function
  ↓
estimate simulator parameters
  ↓
aggregate replicates (mean + 95% CI, t-distribution)
  ↓
save CSVs (metric table + full curves)
  ↓
generate plots (TVI curves, scalar bar chart, transfer functions)

All device parameters are read from the YAML config file or TIFF metadata.
Nothing is hardcoded in this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Configure logging before importing tvi (so library log messages are visible)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tvi.benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TVI / Dot-Gain benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/device_template.yaml",
        help="Path to device YAML config file.",
    )
    p.add_argument(
        "--scans_root",
        default="data/scans",
        help=(
            "Root directory of scan TIFFs.  Expected layout:\n"
            "  <scans_root>/<method>/<channel>/sheet<N>_scan<M>.tif"
        ),
    )
    p.add_argument(
        "--wedge_dir",
        default=None,
        help=(
            "Directory containing step-wedge scan TIFFs per session. "
            "Defaults to <scans_root>/wedges/"
        ),
    )
    p.add_argument(
        "--results_dir",
        default="results",
        help="Directory for CSV exports and PNG plots.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Halftoning methods to process (default: all subdirs in scans_root).",
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Ink channels to process (default: from config).",
    )
    p.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level for CI computation.",
    )
    p.add_argument(
        "--fit_yn",
        action="store_true",
        help="Fit Yule-Nielsen n factor in addition to Murray-Davies TVI.",
    )
    p.add_argument(
        "--simulate",
        action="store_true",
        help="Estimate dot-gain simulator parameters and save them.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Generate synthetic data and run the full pipeline without real scans. "
            "Useful for testing the pipeline and CI."
        ),
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation (useful for headless CI).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load config ----
    log.info("Loading config: %s", args.config)
    config = _load_config(args.config)
    channels: list[str] = args.channels or config["printer"]["channels"]
    printer_label: str = config["printer"]["label"]

    # ---- 2. Scanner calibration ----
    from tvi.calibration.scanner_linearization import (
        ScannerCalibration,
        fit_scanner_calibration,
    )

    if args.dry_run:
        log.info("DRY RUN: generating synthetic calibration.")
        calibration = _synthetic_calibration(config)
        scan_groups = _synthetic_scan_groups(config, channels, args.methods or ["DBS", "ErrorDiffusion"])
    else:
        wedge_dir = Path(args.wedge_dir or Path(args.scans_root) / "wedges")
        log.info("Calibrating scanner from wedges in: %s", wedge_dir)
        calibration = _calibrate_from_wedges(wedge_dir, config)
        scan_groups = _discover_scans(
            root=Path(args.scans_root),
            methods=args.methods,
            channels=channels,
            config=config,
        )

    # ---- 3. Measure reflectances and compute TVI curves ----
    from tvi.measurement.measure_tone_ramps import ReplicateSet, measure_replicate_set
    from tvi.measurement.tvi_core import fit_yule_nielsen_n

    all_results = []
    all_tfs = []
    all_yn = []

    for (method, channel), tiff_paths in scan_groups.items():
        log.info("Processing: method=%s  channel=%s  n_sheets=%d", method, channel, len(tiff_paths))

        rep_set = measure_replicate_set(
            tiff_paths=tiff_paths,
            calibration=calibration,
            config=config,
            channel=channel,
            method=method,
            printer_label=printer_label,
        )

        # ---- 4. Aggregate replicates ----
        from tvi.aggregation.statistics import aggregate_replicates
        agg = aggregate_replicates(rep_set, ci_level=args.ci_level)
        all_results.append(agg)
        log.info(
            "  TVI@50 = %.2f pp  (±%.2f pp CI,  n=%d)",
            agg.tvi_at_50 * 100, agg.ci_at_50 * 100, agg.n_obs
        )

        # ---- 5. Fit transfer function ----
        from tvi.calibration.transfer_function import (
            fit_transfer_function, save_transfer_function_csv
        )
        tf = fit_transfer_function(
            a_nom=agg.a_nom,
            a_print=agg.a_nom + agg.mean_tvi,
            channel=channel,
            method=method,
            printer_label=printer_label,
        )
        all_tfs.append(tf)
        tf_path = results_dir / f"transfer_{method}_{channel}.csv"
        save_transfer_function_csv(tf, tf_path)
        log.info("  Transfer function saved: %s", tf_path)

        # ---- 6. Optional Yule-Nielsen fit ----
        if args.fit_yn:
            yn_steps_mask = agg.a_nom >= config["yule_nielsen"]["fit_steps"][0] / 100.0
            # Use first sheet's r_paper / r_ink as reference
            first_sheet = rep_set.sheets[0]
            yn = fit_yule_nielsen_n(
                a_nom=agg.a_nom,
                r_patch=first_sheet.r_patch_array,
                r_ink=first_sheet.r_ink,
                r_paper=first_sheet.r_paper,
                fit_steps_mask=yn_steps_mask,
                n_min=config["yule_nielsen"]["n_min"],
                n_max=config["yule_nielsen"]["n_max"],
                channel=channel,
                method=method,
            )
            all_yn.append(yn)
            log.info("  Yule-Nielsen n = %.3f  (residual=%.6f)", yn.n, yn.residual)

        # ---- 7. Optional simulator parameters ----
        if args.simulate:
            from tvi.simulation.dot_gain_simulator import estimate_simulator_params
            first_sheet = rep_set.sheets[0]
            sim_params = estimate_simulator_params(
                transfer_function=tf,
                r_paper=first_sheet.r_paper,
                r_ink=first_sheet.r_ink,
            )
            sp_path = results_dir / f"sim_params_{method}_{channel}.json"
            _save_sim_params(sim_params, sp_path)
            log.info(
                "  Simulator: r_mech=%.3f px  n_yn=%.3f  rmse=%.5f",
                sim_params.r_mech, sim_params.n_yn, sim_params.rmse,
            )

    # ---- 8. Save CSVs ----
    from tvi.aggregation.statistics import save_csv

    save_csv(all_results, results_dir / "tvi_metric_table.csv", kind="metric_table")
    save_csv(all_results, results_dir / "tvi_full_curves.csv", kind="full_curves")
    log.info("CSVs saved to %s", results_dir)

    if all_yn:
        _save_yn_table(all_yn, results_dir / "yule_nielsen_n.csv")

    # ---- 9. Plots ----
    if not args.no_plots:
        from tvi.visualization.tvi_plots import (
            plot_tvi_curves, plot_tvi_scalar,
            plot_transfer_functions, save_figure,
        )
        fig1 = plot_tvi_curves(all_results, title=f"TVI Curves — {printer_label}")
        save_figure(fig1, results_dir / "tvi_curves.png")

        fig2 = plot_tvi_scalar(all_results, title=f"TVI @ 50% — {printer_label}")
        save_figure(fig2, results_dir / "tvi_scalar.png")

        fig3 = plot_transfer_functions(all_tfs, title=f"Transfer Functions — {printer_label}")
        save_figure(fig3, results_dir / "transfer_functions.png")

        log.info("Plots saved to %s", results_dir)

    log.info("Benchmark complete.  Results: %s", results_dir)
    return 0


# ---------------------------------------------------------------------------
# Helper functions (keep main() clean)
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _calibrate_from_wedges(
    wedge_dir: Path,
    config: dict,
) -> "ScannerCalibration":
    """Load the first wedge TIFF and fit linearisation."""
    from tvi.io.tiff_io import load_tiff
    from tvi.calibration.scanner_linearization import (
        fit_scanner_calibration, measure_wedge_counts
    )

    wedge_files = sorted(wedge_dir.glob("*.tif")) + sorted(wedge_dir.glob("*.tiff"))
    if not wedge_files:
        raise FileNotFoundError(f"No TIFF files found in wedge directory: {wedge_dir}")

    tiff = load_tiff(wedge_files[0])
    ch_idx = config["scanner"]["channel_map"].get("K", 1)
    nominals = np.array(config["calibration"]["step_wedge_nominals"])

    raw_counts = measure_wedge_counts(
        tiff.array, tiff.manifest, channel_idx=ch_idx,
        erosion_margin=config["measurement"]["patch_erosion_margin"]
    )
    return fit_scanner_calibration(
        raw_counts=raw_counts,
        nominal_reflectances=nominals,
        degree=config["calibration"]["linearization_poly_degree"],
    )


def _discover_scans(
    root: Path,
    methods: list[str] | None,
    channels: list[str],
    config: dict,
) -> dict[tuple[str, str], list[list[Path]]]:
    """
    Discover scan TIFF paths under root/.

    Expected layout:
        <root>/<method>/<channel>/sheet<N>_scan<M>.tif

    Returns dict: (method, channel) → list[list[Path]]
        Outer list = sheets, inner list = scans per sheet.
    """
    n_sheets = config["measurement"]["n_sheets"]
    n_scans = config["measurement"]["n_scans"]
    groups: dict[tuple[str, str], list[list[Path]]] = {}

    method_dirs = sorted(root.iterdir()) if methods is None else [root / m for m in methods]
    for method_dir in method_dirs:
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for ch in channels:
            ch_dir = method_dir / ch
            if not ch_dir.is_dir():
                log.warning("Missing channel dir: %s", ch_dir)
                continue
            tiff_files = sorted(ch_dir.glob("*.tif")) + sorted(ch_dir.glob("*.tiff"))
            if not tiff_files:
                log.warning("No TIFFs in: %s", ch_dir)
                continue
            # Group into sheets × scans
            sheets = []
            for si in range(n_sheets):
                scans = tiff_files[si * n_scans: (si + 1) * n_scans]
                if scans:
                    sheets.append(scans)
            if sheets:
                groups[(method, ch)] = sheets
    return groups


def _synthetic_calibration(config: dict) -> "ScannerCalibration":
    """Identity-like calibration for dry-run testing."""
    from tvi.calibration.scanner_linearization import fit_scanner_calibration
    nominals = np.array(config["calibration"]["step_wedge_nominals"])
    # Simulate slightly non-linear scanner response
    raw = (nominals ** 1.1) * 65535
    return fit_scanner_calibration(
        raw_counts=raw,
        nominal_reflectances=nominals,
        degree=config["calibration"]["linearization_poly_degree"],
        session_id="dry_run",
    )


def _synthetic_scan_groups(
    config: dict,
    channels: list[str],
    methods: list[str],
) -> dict[tuple[str, str], list[list[Path]]]:
    """
    In dry-run mode, return a sentinel that measure_replicate_set will not
    actually call (we monkeypatch it below).  This is a stub path structure.
    """
    groups: dict[tuple[str, str], list[list[Path]]] = {}
    for method in methods:
        for ch in channels:
            # Fill with dummy Paths; measure_replicate_set is replaced by
            # _dry_run_measure_replicate_set in dry_run mode.
            n_sheets = config["measurement"]["n_sheets"]
            n_scans = config["measurement"]["n_scans"]
            groups[(method, ch)] = [
                [Path(f"synthetic/{method}/{ch}/sheet{s}_scan{sc}.tif")
                 for sc in range(n_scans)]
                for s in range(n_sheets)
            ]
    return groups


def _save_yn_table(yn_list: list, path: Path) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "channel", "n", "residual"])
        for yn in yn_list:
            w.writerow([yn.method, yn.channel, f"{yn.n:.5f}", f"{yn.residual:.8f}"])


def _save_sim_params(params, path: Path) -> None:
    import json as _json
    d = {
        "r_mech": params.r_mech,
        "n_yn": params.n_yn,
        "r_paper": params.r_paper,
        "r_ink": params.r_ink,
        "channel": params.channel,
        "method": params.method,
        "printer_label": params.printer_label,
        "rmse": params.rmse,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        _json.dump(d, fh, indent=2)


# ---------------------------------------------------------------------------
# Dry-run monkey-patch: replace measure_replicate_set with synthetic version
# ---------------------------------------------------------------------------

def _install_dry_run_measurement(config: dict) -> None:
    """
    Replace tvi.measurement.measure_tone_ramps.measure_replicate_set
    with a synthetic version that generates plausible TVI data
    without reading any TIFF files.
    """
    import tvi.measurement.measure_tone_ramps as _mtr
    from tvi.measurement.measure_tone_ramps import MeasuredSheet, PatchReflectance, ReplicateSet

    tone_steps = config["measurement"]["tone_steps"]
    n_sheets = config["measurement"]["n_sheets"]
    n_scans = config["measurement"]["n_scans"]

    def _synthetic_measure(tiff_paths, calibration, config, channel,
                           method="", printer_label=""):
        rep_set = ReplicateSet(channel=channel, method=method, printer_label=printer_label)
        rng = np.random.default_rng(seed=hash((method, channel)) % (2 ** 32))
        tvi_true = 0.15   # 15 pp synthetic TVI at 50%

        for sheet_idx in range(n_sheets):
            for scan_idx in range(n_scans):
                patches = []
                r_paper = 0.92 + rng.normal(0, 0.003)
                r_ink = 0.04 + rng.normal(0, 0.002)
                for step in tone_steps:
                    a_nom = step / 100.0
                    # Simulate dot gain: a_print = a_nom + TVI_curve(a_nom)
                    tvi_curve = tvi_true * 4 * a_nom * (1 - a_nom)  # parabolic peak at 50%
                    a_print = np.clip(a_nom + tvi_curve + rng.normal(0, 0.003), 0, 1)
                    r_patch = r_paper - a_print * (r_paper - r_ink)
                    patches.append(PatchReflectance(
                        nominal_pct=step,
                        a_nom=a_nom,
                        reflectance=float(r_patch),
                        channel=channel,
                        scan_idx=scan_idx,
                        sheet_idx=sheet_idx,
                    ))
                sheet = MeasuredSheet(
                    channel=channel, method=method,
                    printer_label=printer_label,
                    sheet_idx=sheet_idx, scan_idx=scan_idx,
                    patches=patches,
                    r_paper=r_paper, r_ink=r_ink,
                )
                rep_set.sheets.append(sheet)
        return rep_set

    _mtr.measure_replicate_set = _synthetic_measure


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        config = _load_config(args.config)
        _install_dry_run_measurement(config)
    sys.exit(main())
