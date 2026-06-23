#!/usr/bin/env python3
"""
scripts/generate_targets.py
----------------------------
Generate all calibration target TIFFs from a device config.

Output directory layout
-----------------------
<output_dir>/
  tone_ramp_C.tif
  tone_ramp_M.tif
  tone_ramp_Y.tif
  tone_ramp_K.tif
  step_wedge.tif
  overprint_patches.tif
  manifest.json          ← summary of all generated targets

Usage
-----
# Using the template config (edit it first):
python scripts/generate_targets.py --config configs/device_template.yaml

# Override output directory:
python scripts/generate_targets.py --config configs/device_template.yaml \\
    --output_dir generated_targets/

# Generate demo data only (uses minimal built-in config):
python scripts/generate_targets.py --demo --output_dir demo_data/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_targets")

# Minimal demo config — used when --demo flag is set.
# Uses 200 dpi so the files stay small.
DEMO_CONFIG = {
    "printer": {
        "label": "DemoDevice",
        "print_dpi": 200,
        "ink_type": "inkjet_pigment",
        "channels": ["C", "M", "Y", "K"],
    },
    "scanner": {
        "label": "DemoScanner",
        "capture_dpi": 200,
        "bit_depth": 8,
        "channel_map": {"K": 1, "C": 0, "M": 1, "Y": 2},
        "max_raw_value": None,
    },
    "measurement": {
        "tone_steps": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "n_sheets": 3,
        "n_scans": 3,
        "patch_erosion_margin": 0.10,
        "n_fiducials": 4,
    },
    "calibration": {
        "step_wedge_nominals": [0.05, 0.10, 0.20, 0.30, 0.40,
                                0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        "linearization_poly_degree": 2,
    },
    "yule_nielsen": {
        "fit_steps": [70, 80, 90],
        "n_min": 1.0,
        "n_max": 4.0,
        "n_init": None,
    },
    "output": {
        "results_dir": "results",
        "ci_level": 0.95,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate TVI calibration target TIFFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/device_template.yaml",
        help="Path to device YAML config file.",
    )
    p.add_argument(
        "--output_dir",
        default="generated_targets",
        help="Directory where target TIFFs will be written.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Ignore --config and use a built-in minimal demo config. "
            "Produces small (200 dpi, 8-bit) targets for smoke-testing."
        ),
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Override channels from config (e.g. --channels K C).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ---- Load config ----
    if args.demo:
        config = DEMO_CONFIG
        log.info("Using built-in DEMO config (200 dpi, 8-bit).")
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            log.error("Config file not found: %s", config_path)
            return 1
        with open(config_path) as fh:
            config = yaml.safe_load(fh)
        log.info("Loaded config: %s", config_path)

    # Override channels if requested
    if args.channels:
        config["printer"]["channels"] = args.channels

    # ---- Generate targets ----
    from tvi.calibration_targets.target_generator import generate_all_targets

    log.info(
        "Generating targets for channels=%s at %d dpi ...",
        config["printer"]["channels"],
        config["printer"]["print_dpi"],
    )

    written = generate_all_targets(config, output_dir)

    # ---- Write manifest summary ----
    manifest_summary = {
        name: str(path) for name, path in written.items()
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest_summary, fh, indent=2)

    log.info("Generated %d target files:", len(written))
    for name, path in written.items():
        size_kb = path.stat().st_size // 1024
        log.info("  %-30s  %s  (%d KB)", name, path.name, size_kb)
    log.info("Manifest written: %s", manifest_path)
    log.info("Done. Output directory: %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
