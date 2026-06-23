#!/usr/bin/env python3
"""
scripts/batch_measure_tvi.py
------------------------------
Batch TVI measurement from a directory of scanned TIFFs.

Expected input layout
---------------------
scans/
  wedges/                          ← step-wedge scans (one per session)
    session1_wedge.tif
  <method>/
    <channel>/
      sheet1_scan1.tif
      sheet1_scan2.tif
      sheet1_scan3.tif
      sheet2_scan1.tif
      ...

Outputs
-------
results/
  csv/
    tvi_metric_table.csv           ← primary benchmark Table 3
    tvi_full_curves.csv            ← long-format TVI(a_nom) per method/channel
    yule_nielsen_n.csv             ← fitted n per method/channel (if --fit_yn)
  plots/
    tvi_curves.png
    tvi_scalar.png
    transfer_functions.png
  curves/
    transfer_<method>_<ch>.csv     ← one per method/channel
  sim_params/
    sim_params_<method>_<ch>.json  ← if --simulate

Usage
-----
# Full batch run:
python scripts/batch_measure_tvi.py \\
    --config configs/my_lab.yaml \\
    --scans_dir scans/ \\
    --results_dir results/

# Demo mode (no real scans needed):
python scripts/batch_measure_tvi.py --demo --results_dir results/demo/

# Specific methods and channels only:
python scripts/batch_measure_tvi.py \\
    --config configs/my_lab.yaml \\
    --scans_dir scans/ \\
    --methods DBS ErrorDiffusion \\
    --channels K C \\
    --results_dir results/KC_only/
"""

from __future__ import annotations

import argparse
import json
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
log = logging.getLogger("batch_tvi")

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
        description="Batch TVI measurement across methods and channels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/device_template.yaml")
    p.add_argument("--scans_dir", default="scans",
                   help="Root directory of scan TIFFs.")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Halftoning methods to process (default: all subdirs).")
    p.add_argument("--channels", nargs="+", default=None,
                   help="Channels to process (default: from config).")
    p.add_argument("--fit_yn", action="store_true",
                   help="Fit Yule-Nielsen n factor per method/channel.")
    p.add_argument("--simulate", action="store_true",
                   help="Estimate dot-gain simulator parameters.")
    p.add_argument("--ci_level", type=float, default=0.95)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--demo", action="store_true",
                   help="Use synthetic data (no real scans required).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    config = DEMO_CONFIG if args.demo else _load_config(args.config)
    channels: list[str] = args.channels or config["printer"]["channels"]
    printer_label: str = config["printer"]["label"]
    results_dir = Path(args.results_dir)
    csv_dir = results_dir / "csv"
    plots_dir = results_dir / "plots"
    curves_dir = results_dir / "curves"
    for d in [csv_dir, plots_dir, curves_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Scanner calibration ----
    if args.demo:
        calibration = _synthetic_calibration(config)
    else:
        wedge_dir = Path(args.scans_dir) / "wedges"
        calibration = _calibrate_from_wedge_dir(wedge_dir, config)

    # ---- Discover or synthesise scan groups ----
    if args.demo:
        methods = args.methods or ["DBS", "ErrorDiffusion", "OrderedDither"]
        scan_groups = {
            (m, ch): None   # signals synthetic mode
            for m in methods for ch in channels
        }
    else:
        scan_groups = _discover_scans(
            root=Path(args.scans_dir),
            methods=args.methods,
            channels=channels,
            config=config,
        )

    if not scan_groups:
        log.error(
            "No scan groups found in %s. "
            "Check that the directory layout matches: "
            "<scans_dir>/<method>/<channel>/*.tif",
            args.scans_dir,
        )
        return 1

    log.info(
        "Processing %d (method, channel) combinations ...",
        len(scan_groups),
    )

    # ---- Main loop ----
    all_results = []
    all_tfs = []
    all_yn = []

    for (method, channel), tiff_paths in scan_groups.items():
        log.info("  [%s / %s]", method, channel)

        rep_set = _get_replicate_set(
            tiff_paths=tiff_paths,
            calibration=calibration,
            config=config,
            channel=channel,
            method=method,
            printer_label=printer_label,
            demo=args.demo,
        )

        # Aggregate
        from tvi.aggregation.statistics import aggregate_replicates
        agg = aggregate_replicates(rep_set, ci_level=args.ci_level)
        all_results.append(agg)
        log.info(
            "    TVI@50 = %.2f pp  ±%.2f pp  (n=%d)",
            agg.tvi_at_50 * 100, agg.ci_at_50 * 100, agg.n_obs,
        )

        # Transfer function
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
        save_transfer_function_csv(tf, curves_dir / f"transfer_{method}_{channel}.csv")

        # Yule-Nielsen
        if args.fit_yn:
            from tvi.measurement.tvi_core import fit_yule_nielsen_n
            first = rep_set.sheets[0]
            yn_mask = agg.a_nom >= config["yule_nielsen"]["fit_steps"][0] / 100.0
            yn = fit_yule_nielsen_n(
                a_nom=agg.a_nom,
                r_patch=first.r_patch_array,
                r_ink=first.r_ink,
                r_paper=first.r_paper,
                fit_steps_mask=yn_mask,
                n_min=config["yule_nielsen"]["n_min"],
                n_max=config["yule_nielsen"]["n_max"],
                channel=channel,
                method=method,
            )
            all_yn.append(yn)
            log.info("    YN n=%.3f  residual=%.6f", yn.n, yn.residual)

        # Simulator
        if args.simulate:
            from tvi.simulation.dot_gain_simulator import estimate_simulator_params
            first = rep_set.sheets[0]
            sp = estimate_simulator_params(tf, r_paper=first.r_paper, r_ink=first.r_ink)
            sp_dir = results_dir / "sim_params"
            sp_dir.mkdir(exist_ok=True)
            _save_json(
                {"r_mech": sp.r_mech, "n_yn": sp.n_yn,
                 "r_paper": sp.r_paper, "r_ink": sp.r_ink,
                 "channel": sp.channel, "method": sp.method,
                 "printer_label": sp.printer_label, "rmse": sp.rmse},
                sp_dir / f"sim_params_{method}_{channel}.json",
            )
            log.info("    Sim: r_mech=%.3f px  n_yn=%.3f  rmse=%.5f",
                     sp.r_mech, sp.n_yn, sp.rmse)

    # ---- Save CSVs ----
    from tvi.aggregation.statistics import save_csv
    save_csv(all_results, csv_dir / "tvi_metric_table.csv", kind="metric_table")
    save_csv(all_results, csv_dir / "tvi_full_curves.csv", kind="full_curves")
    log.info("CSVs saved to %s", csv_dir)

    if all_yn:
        _save_yn_csv(all_yn, csv_dir / "yule_nielsen_n.csv")
        log.info("Yule-Nielsen table saved.")

    # ---- Plots ----
    if not args.no_plots:
        from tvi.visualization.tvi_plots import (
            plot_tvi_curves, plot_tvi_scalar,
            plot_transfer_functions, save_figure,
        )
        save_figure(
            plot_tvi_curves(all_results, title=f"TVI Curves — {printer_label}"),
            plots_dir / "tvi_curves.png",
        )
        save_figure(
            plot_tvi_scalar(all_results, title=f"TVI @ 50% — {printer_label}"),
            plots_dir / "tvi_scalar.png",
        )
        save_figure(
            plot_transfer_functions(all_tfs, title=f"Transfer Functions — {printer_label}"),
            plots_dir / "transfer_functions.png",
        )
        log.info("Plots saved to %s", plots_dir)

    # ---- Print summary table ----
    from tvi.aggregation.statistics import build_metric_table
    df = build_metric_table(all_results)
    log.info("\n%s\n", df.to_string(index=False))

    log.info("Batch complete. Results: %s", results_dir.resolve())
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _calibrate_from_wedge_dir(wedge_dir: Path, config: dict):
    from tvi.io.tiff_io import load_tiff
    from tvi.calibration.scanner_linearization import (
        fit_scanner_calibration, measure_wedge_counts
    )
    wedge_files = sorted(wedge_dir.glob("*.tif")) + sorted(wedge_dir.glob("*.tiff"))
    if not wedge_files:
        log.warning("No wedge TIFFs in %s — using identity calibration.", wedge_dir)
        return _identity_calibration(config["scanner"]["bit_depth"])
    tiff = load_tiff(wedge_files[0])
    if not tiff.manifest or "patches" not in tiff.manifest:
        log.warning("Wedge has no manifest — using identity calibration.")
        return _identity_calibration(config["scanner"]["bit_depth"])
    ch_idx = config["scanner"]["channel_map"].get("K", 1)
    nominals = np.array(config["calibration"]["step_wedge_nominals"])
    raw = measure_wedge_counts(tiff.array, tiff.manifest, channel_idx=ch_idx,
                               erosion_margin=config["measurement"]["patch_erosion_margin"])
    return fit_scanner_calibration(
        raw, nominals,
        degree=config["calibration"]["linearization_poly_degree"],
    )


def _identity_calibration(bit_depth: int):
    from tvi.calibration.scanner_linearization import fit_scanner_calibration
    max_val = float((1 << bit_depth) - 1)
    return fit_scanner_calibration(
        np.array([0.0, max_val]), np.array([0.0, 1.0]), degree=1,
    )


def _synthetic_calibration(config: dict):
    from tvi.calibration.scanner_linearization import fit_scanner_calibration
    nominals = np.array(config["calibration"]["step_wedge_nominals"])
    raw = (nominals ** 1.1) * 65535
    return fit_scanner_calibration(raw, nominals,
                                   degree=config["calibration"]["linearization_poly_degree"])


def _discover_scans(root: Path, methods, channels, config) -> dict:
    n_sh = config["measurement"]["n_sheets"]
    n_sc = config["measurement"]["n_scans"]
    groups: dict = {}
    method_dirs = sorted(root.iterdir()) if methods is None else [root / m for m in methods]
    for md in method_dirs:
        if not md.is_dir():
            continue
        method = md.name
        for ch in channels:
            ch_dir = md / ch
            if not ch_dir.is_dir():
                log.warning("Missing channel dir: %s", ch_dir)
                continue
            files = sorted(ch_dir.glob("*.tif")) + sorted(ch_dir.glob("*.tiff"))
            if not files:
                log.warning("No TIFFs in %s", ch_dir)
                continue
            sheets = [files[i * n_sc: (i + 1) * n_sc] for i in range(n_sh)]
            sheets = [s for s in sheets if s]
            if sheets:
                groups[(method, ch)] = sheets
    return groups


def _get_replicate_set(
    tiff_paths, calibration, config, channel, method, printer_label, demo
):
    if demo:
        return _synthetic_replicate_set(config, channel, method, printer_label)
    from tvi.measurement.measure_tone_ramps import measure_replicate_set
    return measure_replicate_set(
        tiff_paths=tiff_paths,
        calibration=calibration,
        config=config,
        channel=channel,
        method=method,
        printer_label=printer_label,
    )


def _synthetic_replicate_set(config, channel, method, printer_label):
    """Generate plausible synthetic replicate data for demo/dry-run."""
    from tvi.measurement.measure_tone_ramps import MeasuredSheet, PatchReflectance, ReplicateSet
    n_sheets = config["measurement"]["n_sheets"]
    n_scans = config["measurement"]["n_scans"]
    tone_steps = config["measurement"]["tone_steps"]
    tvi_true = 0.14 + abs(hash(method + channel)) % 5 * 0.01
    rng = np.random.default_rng(seed=abs(hash(method + channel)) % (2**32))

    rep_set = ReplicateSet(channel=channel, method=method, printer_label=printer_label)
    for si in range(n_sheets):
        for sc in range(n_scans):
            r_paper = 0.92 + rng.normal(0, 0.003)
            r_ink = 0.04 + rng.normal(0, 0.002)
            patches = []
            for step in tone_steps:
                a_nom = step / 100.0
                tvi_curve = tvi_true * 4 * a_nom * (1 - a_nom)
                a_print = np.clip(a_nom + tvi_curve + rng.normal(0, 0.003), 0, 1)
                r_patch = r_paper - a_print * (r_paper - r_ink)
                patches.append(PatchReflectance(
                    nominal_pct=step, a_nom=a_nom,
                    reflectance=float(r_patch),
                    channel=channel, scan_idx=sc, sheet_idx=si,
                ))
            rep_set.sheets.append(MeasuredSheet(
                channel=channel, method=method, printer_label=printer_label,
                sheet_idx=si, scan_idx=sc, patches=patches,
                r_paper=r_paper, r_ink=r_ink,
            ))
    return rep_set


def _save_yn_csv(yn_list, path: Path) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "channel", "n", "residual"])
        for yn in yn_list:
            w.writerow([yn.method, yn.channel, f"{yn.n:.5f}", f"{yn.residual:.8f}"])


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
