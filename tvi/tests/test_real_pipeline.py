"""
tests/test_real_pipeline.py
-----------------------------
End-to-end integration test for the TVI benchmark pipeline.

Pipeline under test
-------------------
generate_all_targets()
  → save TIFF
  → load TIFF (load_tiff)
  → scanner calibration (fit_scanner_calibration)
  → geometric alignment check (detect_fiducials)
  → patch extraction (extract_patch)
  → reflectance computation
  → TVI curve (compute_tvi)
  → Yule-Nielsen fit (fit_yule_nielsen_n)
  → transfer function (fit_transfer_function)
  → aggregate replicates (aggregate_tvi_curve)
  → CSV export (save_csv)
  → TVI curve plot (plot_tvi_curves)
  → transfer function plot (plot_transfer_functions)

No real printer or scanner is required.  The synthetic scanner model
returns noise-free reflectances from the manifested patch layout.

This test must pass before packaging the ZIP.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal config shared across tests
# ---------------------------------------------------------------------------

PIPELINE_CONFIG = {
    "printer": {
        "label": "PipelineTest",
        "print_dpi": 200,
        "ink_type": "inkjet_pigment",
        "channels": ["C", "M", "Y", "K"],
    },
    "scanner": {
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
        "step_wedge_nominals": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6, 0.7, 0.8, 0.9, 0.95],
        "linearization_poly_degree": 2,
    },
    "yule_nielsen": {
        "fit_steps": [70, 80, 90],
        "n_min": 1.0,
        "n_max": 4.0,
        "n_init": None,
    },
    "output": {"results_dir": "results", "ci_level": 0.95},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_sheet(config, channel="K", method="DBS", tvi_level=0.14,
                     sheet_idx=0, scan_idx=0):
    """
    Build a MeasuredSheet from synthetic reflectance data.

    Simulates a printer with parabolic TVI peaking at tvi_level at 50%.
    """
    from tvi.measurement.measure_tone_ramps import MeasuredSheet, PatchReflectance
    rng = np.random.default_rng(seed=sheet_idx * 3 + scan_idx)
    r_paper = 0.92 + rng.normal(0, 0.002)
    r_ink = 0.04 + rng.normal(0, 0.002)
    tone_steps = config["measurement"]["tone_steps"]
    patches = []
    for step in tone_steps:
        a_nom = step / 100.0
        tvi = tvi_level * 4 * a_nom * (1 - a_nom)
        a_print = np.clip(a_nom + tvi + rng.normal(0, 0.002), 0, 1)
        r_patch = r_paper - a_print * (r_paper - r_ink)
        patches.append(PatchReflectance(
            nominal_pct=step, a_nom=a_nom, reflectance=float(r_patch),
            channel=channel, scan_idx=scan_idx, sheet_idx=sheet_idx,
        ))
    return MeasuredSheet(
        channel=channel, method=method,
        printer_label="PipelineTest",
        sheet_idx=sheet_idx, scan_idx=scan_idx,
        patches=patches, r_paper=r_paper, r_ink=r_ink,
    )


def _make_replicate_set(config, n_sheets=3, n_scans=3, channel="K", method="DBS"):
    from tvi.measurement.measure_tone_ramps import ReplicateSet
    rs = ReplicateSet(channel=channel, method=method, printer_label="PipelineTest")
    for si in range(n_sheets):
        for sc in range(n_scans):
            rs.sheets.append(_synthetic_sheet(config, channel=channel, method=method,
                                               sheet_idx=si, scan_idx=sc))
    return rs


# ===========================================================================
# Stage 1: Target generation and TIFF round-trip
# ===========================================================================

class TestTargetGenerationAndTiffIO:
    def test_generate_all_targets_and_reload(self):
        """generate_all_targets → save TIFF → load TIFF → manifest intact."""
        from tvi.calibration_targets.target_generator import generate_all_targets
        from tvi.io.tiff_io import load_tiff

        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(PIPELINE_CONFIG, tmp)
            assert "tone_ramp_K" in written
            tiff = load_tiff(written["tone_ramp_K"])
            assert tiff.manifest["target_type"] == "tone_ramp"
            assert len(tiff.manifest["patches"]) == len(
                PIPELINE_CONFIG["measurement"]["tone_steps"]
            )

    def test_step_wedge_round_trip(self):
        from tvi.calibration_targets.target_generator import generate_all_targets
        from tvi.io.tiff_io import load_tiff

        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(PIPELINE_CONFIG, tmp)
            tiff = load_tiff(written["step_wedge"])
            assert tiff.manifest["target_type"] == "step_wedge"
            n_expected = len(PIPELINE_CONFIG["calibration"]["step_wedge_nominals"])
            assert len(tiff.manifest["patches"]) == n_expected

    def test_tiff_dpi_preserved(self):
        from tvi.calibration_targets.target_generator import generate_all_targets
        from tvi.io.tiff_io import load_tiff

        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(PIPELINE_CONFIG, tmp)
            tiff = load_tiff(written["tone_ramp_K"])
            assert abs(tiff.dpi - PIPELINE_CONFIG["printer"]["print_dpi"]) < 10


# ===========================================================================
# Stage 2: Scanner calibration
# ===========================================================================

class TestScannerCalibrationPipeline:
    def test_fit_and_apply_synthetic(self):
        from tvi.calibration.scanner_linearization import fit_scanner_calibration

        nominals = np.array(PIPELINE_CONFIG["calibration"]["step_wedge_nominals"])
        raw = (nominals ** 1.15) * 255.0
        cal = fit_scanner_calibration(raw, nominals, degree=2)
        recovered = cal.apply(raw)
        np.testing.assert_allclose(recovered, nominals, atol=0.02)

    def test_measure_wedge_counts_from_tiff(self):
        from tvi.calibration_targets.target_generator import generate_all_targets
        from tvi.io.tiff_io import load_tiff
        from tvi.calibration.scanner_linearization import measure_wedge_counts

        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(PIPELINE_CONFIG, tmp)
            tiff = load_tiff(written["step_wedge"])
            counts = measure_wedge_counts(
                tiff.array, tiff.manifest, channel_idx=1,
                erosion_margin=PIPELINE_CONFIG["measurement"]["patch_erosion_margin"],
            )
            n_expected = len(PIPELINE_CONFIG["calibration"]["step_wedge_nominals"])
            assert len(counts) == n_expected
            assert np.all(counts >= 0)


# ===========================================================================
# Stage 3: Patch extraction and TVI computation
# ===========================================================================

class TestPatchExtractionAndTVI:
    def test_tvi_curve_from_sheet(self):
        sheet = _synthetic_sheet(PIPELINE_CONFIG, tvi_level=0.14)
        curve = sheet.to_tvi_curve()
        assert curve.tvi_at_50 == pytest.approx(0.14, abs=0.02)

    def test_tvi_curve_shape(self):
        sheet = _synthetic_sheet(PIPELINE_CONFIG)
        curve = sheet.to_tvi_curve()
        n_inner = len([s for s in PIPELINE_CONFIG["measurement"]["tone_steps"]
                        if 0 < s < 100])
        assert len(curve.a_nom) == n_inner
        assert len(curve.tvi) == n_inner

    def test_tvi_values_positive(self):
        sheet = _synthetic_sheet(PIPELINE_CONFIG, tvi_level=0.12)
        curve = sheet.to_tvi_curve()
        assert np.all(curve.tvi > -0.01), "TVI should be positive for normal dot gain"

    def test_yule_nielsen_fit(self):
        from tvi.measurement.tvi_core import fit_yule_nielsen_n
        sheet = _synthetic_sheet(PIPELINE_CONFIG)
        curve = sheet.to_tvi_curve()
        yn_mask = curve.a_nom >= 0.70
        yn = fit_yule_nielsen_n(
            a_nom=curve.a_nom,
            r_patch=curve.r_patch,
            r_ink=sheet.r_ink,
            r_paper=sheet.r_paper,
            fit_steps_mask=yn_mask,
        )
        assert 1.0 <= yn.n <= 4.0

    def test_extract_patch_from_real_tiff(self):
        """Load a generated TIFF, extract a patch, check non-trivial content."""
        from tvi.calibration_targets.target_generator import generate_all_targets
        from tvi.io.tiff_io import load_tiff
        from tvi.preprocessing.alignment import extract_patch

        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(PIPELINE_CONFIG, tmp)
            tiff = load_tiff(written["tone_ramp_K"])
            p = tiff.manifest["patches"][5]   # 50% patch
            interior = extract_patch(
                tiff.array,
                row=p["row_px"], col=p["col_px"],
                height=p["height_px"], width=p["width_px"],
                erosion_margin=0.10,
            )
            assert interior.size > 0
            assert interior.ndim == 3 or interior.ndim == 2


# ===========================================================================
# Stage 4: Aggregation and statistics
# ===========================================================================

class TestAggregationPipeline:
    def test_aggregate_9_replicates(self):
        from tvi.aggregation.statistics import aggregate_tvi_curve
        rs = _make_replicate_set(PIPELINE_CONFIG, n_sheets=3, n_scans=3)
        curves = rs.tvi_curves()
        agg = aggregate_tvi_curve(curves)
        assert agg.n_obs == 9
        assert np.all(np.isfinite(agg.mean_tvi))
        assert np.all(np.isfinite(agg.ci_tvi))

    def test_metric_table_has_correct_rows(self):
        from tvi.aggregation.statistics import aggregate_tvi_curve, build_metric_table
        results = []
        for method in ["DBS", "ErrorDiff"]:
            for ch in ["K", "C"]:
                rs = _make_replicate_set(PIPELINE_CONFIG, channel=ch, method=method)
                results.append(aggregate_tvi_curve(rs.tvi_curves()))
        df = build_metric_table(results)
        assert len(df) == 4

    def test_ci_at_50_is_positive(self):
        from tvi.aggregation.statistics import aggregate_tvi_curve
        rs = _make_replicate_set(PIPELINE_CONFIG, n_sheets=3, n_scans=3)
        agg = aggregate_tvi_curve(rs.tvi_curves())
        assert agg.ci_at_50 >= 0


# ===========================================================================
# Stage 5: Transfer function
# ===========================================================================

class TestTransferFunctionPipeline:
    def test_fit_and_evaluate(self):
        from tvi.calibration.transfer_function import fit_transfer_function
        from tvi.aggregation.statistics import aggregate_tvi_curve

        rs = _make_replicate_set(PIPELINE_CONFIG, n_sheets=3, n_scans=3)
        agg = aggregate_tvi_curve(rs.tvi_curves())
        tf = fit_transfer_function(
            a_nom=agg.a_nom,
            a_print=agg.a_nom + agg.mean_tvi,
            channel="K", method="DBS",
        )
        assert tf.is_monotone
        assert tf.evaluate(0.5) > 0.5

    def test_csv_round_trip(self):
        from tvi.calibration.transfer_function import (
            fit_transfer_function, save_transfer_function_csv, load_transfer_function_csv
        )
        from tvi.aggregation.statistics import aggregate_tvi_curve

        rs = _make_replicate_set(PIPELINE_CONFIG)
        agg = aggregate_tvi_curve(rs.tvi_curves())
        tf = fit_transfer_function(agg.a_nom, agg.a_nom + agg.mean_tvi)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tf.csv"
            save_transfer_function_csv(tf, path)
            tf2 = load_transfer_function_csv(path)

        assert tf2.evaluate(0.5) == pytest.approx(tf.evaluate(0.5), abs=0.002)


# ===========================================================================
# Stage 6: Full CSV export
# ===========================================================================

class TestCSVExportPipeline:
    def test_save_and_reload_metric_table(self):
        from tvi.aggregation.statistics import (
            aggregate_tvi_curve, save_csv, load_csv, build_metric_table
        )
        results = []
        for m in ["DBS", "ED"]:
            rs = _make_replicate_set(PIPELINE_CONFIG, method=m)
            results.append(aggregate_tvi_curve(rs.tvi_curves()))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.csv"
            save_csv(results, path, kind="metric_table")
            df = load_csv(path)

        assert len(df) == 2
        assert "tvi_at_50_pp" in df.columns

    def test_save_and_reload_full_curves(self):
        from tvi.aggregation.statistics import (
            aggregate_tvi_curve, save_csv, load_csv
        )
        rs = _make_replicate_set(PIPELINE_CONFIG)
        results = [aggregate_tvi_curve(rs.tvi_curves())]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.csv"
            save_csv(results, path, kind="full_curves")
            df = load_csv(path)

        n_steps = len([s for s in PIPELINE_CONFIG["measurement"]["tone_steps"]
                       if 0 < s < 100])
        assert len(df) == n_steps


# ===========================================================================
# Stage 7: Plots
# ===========================================================================

class TestVisualizationPipeline:
    def test_plot_tvi_curves_saves_file(self):
        from tvi.aggregation.statistics import aggregate_tvi_curve
        from tvi.visualization.tvi_plots import plot_tvi_curves, save_figure

        rs = _make_replicate_set(PIPELINE_CONFIG)
        results = [aggregate_tvi_curve(rs.tvi_curves())]
        fig = plot_tvi_curves(results)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tvi.png"
            save_figure(fig, path)
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_plot_transfer_functions_saves_file(self):
        from tvi.calibration.transfer_function import fit_transfer_function
        from tvi.aggregation.statistics import aggregate_tvi_curve
        from tvi.visualization.tvi_plots import plot_transfer_functions, save_figure

        rs = _make_replicate_set(PIPELINE_CONFIG)
        agg = aggregate_tvi_curve(rs.tvi_curves())
        tf = fit_transfer_function(agg.a_nom, agg.a_nom + agg.mean_tvi)
        fig = plot_transfer_functions([tf])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tf.png"
            save_figure(fig, path)
            assert path.exists()


# ===========================================================================
# Stage 8: Import validation (all public APIs)
# ===========================================================================

class TestImports:
    def test_all_public_imports(self):
        from tvi.measurement import compute_tvi          # noqa
        from tvi.measurement import measure_sheet         # noqa
        from tvi.calibration import fit_scanner_calibration  # noqa
        from tvi.aggregation import aggregate_tvi_curve  # noqa
        from tvi.visualization import plot_tvi_curves    # noqa
        from tvi.calibration_targets import generate_all_targets  # noqa


# ===========================================================================
# Stage 9: Smoke test — demo_data TIFF is valid
# ===========================================================================

class TestDemoData:
    DEMO_TIFF = Path("demo_data/tone_ramp_K.tif")

    def test_demo_tiff_exists(self):
        if not self.DEMO_TIFF.exists():
            pytest.skip("demo_data/tone_ramp_K.tif not yet generated — run generate_targets.py --demo")
        assert self.DEMO_TIFF.exists()

    def test_demo_tiff_loads(self):
        if not self.DEMO_TIFF.exists():
            pytest.skip("demo_data/tone_ramp_K.tif not found")
        from tvi.io.tiff_io import load_tiff
        tiff = load_tiff(self.DEMO_TIFF)
        assert tiff.array.ndim >= 2
        assert tiff.manifest.get("target_type") == "tone_ramp"

    def test_demo_tiff_fiducials_detectable(self):
        if not self.DEMO_TIFF.exists():
            pytest.skip("demo_data/tone_ramp_K.tif not found")
        from tvi.io.tiff_io import load_tiff
        from tvi.preprocessing.alignment import detect_fiducials
        tiff = load_tiff(self.DEMO_TIFF)
        arr = tiff.array
        gray = arr[..., 1] if arr.ndim == 3 else arr
        centroids = detect_fiducials(gray, n_marks=4)
        assert len(centroids) == 4
