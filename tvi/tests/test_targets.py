"""
tests/test_targets.py
---------------------
Unit tests for:
  - tvi.calibration_targets.target_generator   (generate_all_targets)
  - tvi.io.tiff_io                             (load_tiff, save_tiff, TiffImage)
  - tvi.calibration.scanner_linearization      (fit_scanner_calibration)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tvi.calibration_targets.target_generator import (
    generate_all_targets,
    _make_tone_ramp,
    _make_step_wedge,
    _patch_size_px,
)
from tvi.io.tiff_io import TiffImage, load_tiff, save_tiff


# ---------------------------------------------------------------------------
# Minimal test config
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "printer": {
        "label": "TestPrinter",
        "print_dpi": 200,        # low DPI for fast tests
        "ink_type": "inkjet_pigment",
        "channels": ["C", "M", "Y", "K"],
    },
    "scanner": {
        "label": "TestScanner",
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
        "step_wedge_nominals": [0.05, 0.2, 0.4, 0.6, 0.8, 0.95],
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


# ---------------------------------------------------------------------------
# Target generator tests
# ---------------------------------------------------------------------------

class TestTargetGenerator:
    def test_generate_all_targets_returns_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(MINIMAL_CONFIG, tmp)
        assert "tone_ramp_K" in written
        assert "step_wedge" in written
        assert "overprint_patches" in written

    def test_generated_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(MINIMAL_CONFIG, tmp)
            for name, path in written.items():
                assert path.exists(), f"File missing: {name} → {path}"

    def test_tone_ramp_has_correct_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(MINIMAL_CONFIG, tmp)
            tiff = load_tiff(written["tone_ramp_K"])
        manifest = tiff.manifest
        assert manifest["target_type"] == "tone_ramp"
        n_steps = len(MINIMAL_CONFIG["measurement"]["tone_steps"])
        assert len(manifest["patches"]) == n_steps

    def test_tone_ramp_nominal_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(MINIMAL_CONFIG, tmp)
            tiff = load_tiff(written["tone_ramp_C"])
        steps_in_manifest = [p["nominal_pct"] for p in tiff.manifest["patches"]]
        assert steps_in_manifest == MINIMAL_CONFIG["measurement"]["tone_steps"]

    def test_step_wedge_nominal_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(MINIMAL_CONFIG, tmp)
            tiff = load_tiff(written["step_wedge"])
        assert len(tiff.manifest["patches"]) == len(MINIMAL_CONFIG["calibration"]["step_wedge_nominals"])

    def test_image_is_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(MINIMAL_CONFIG, tmp)
            tiff = load_tiff(written["tone_ramp_K"])
        assert tiff.array.ndim == 3
        assert tiff.array.shape[2] == 3

    def test_patch_size_scales_with_dpi(self):
        px_200 = _patch_size_px(200, side_mm=15.0)
        px_400 = _patch_size_px(400, side_mm=15.0)
        assert px_400 == pytest.approx(2 * px_200, abs=2)

    def test_make_tone_ramp_shape(self):
        cfg = MINIMAL_CONFIG
        arr, manifest = _make_tone_ramp(
            channel="K",
            tone_steps=[0, 50, 100],
            print_dpi=200,
            bit_depth=8,
            channels_available=["C", "M", "Y", "K"],
        )
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert len(manifest["patches"]) == 3

    def test_all_channels_get_tone_ramps(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = generate_all_targets(MINIMAL_CONFIG, tmp)
        for ch in MINIMAL_CONFIG["printer"]["channels"]:
            assert f"tone_ramp_{ch}" in written


# ---------------------------------------------------------------------------
# TIFF I/O tests
# ---------------------------------------------------------------------------

class TestTiffIO:
    def _make_rgb_array(self, h: int = 32, w: int = 32, dtype=np.uint8) -> np.ndarray:
        rng = np.random.default_rng(0)
        max_val = 255 if dtype == np.uint8 else 65535
        return (rng.random((h, w, 3)) * max_val).astype(dtype)

    def test_save_and_load_uint8(self):
        arr = self._make_rgb_array(dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.tif"
            save_tiff(arr, path, dpi=200)
            tiff = load_tiff(path)
        np.testing.assert_array_equal(tiff.array, arr)

    def test_save_and_load_uint16(self):
        arr = self._make_rgb_array(dtype=np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test16.tif"
            save_tiff(arr, path, dpi=300)
            tiff = load_tiff(path)
        np.testing.assert_array_equal(tiff.array, arr)

    def test_dpi_preserved(self):
        arr = self._make_rgb_array()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dpi_test.tif"
            save_tiff(arr, path, dpi=600)
            tiff = load_tiff(path)
        assert tiff.dpi == pytest.approx(600, abs=1)

    def test_manifest_round_trip(self):
        arr = self._make_rgb_array()
        manifest = {"target_type": "test", "patches": [{"nominal_pct": 50}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest_test.tif"
            save_tiff(arr, path, dpi=200, manifest=manifest)
            tiff = load_tiff(path)
        assert tiff.manifest["target_type"] == "test"
        assert tiff.manifest["patches"][0]["nominal_pct"] == 50

    def test_bit_depth_8(self):
        arr = self._make_rgb_array(dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bd8.tif"
            save_tiff(arr, path, dpi=200)
            tiff = load_tiff(path)
        assert tiff.bit_depth == 8

    def test_n_channels(self):
        arr = self._make_rgb_array()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ch.tif"
            save_tiff(arr, path, dpi=200)
            tiff = load_tiff(path)
        assert tiff.n_channels == 3

    def test_missing_manifest_returns_empty_dict(self):
        arr = self._make_rgb_array()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "no_manifest.tif"
            save_tiff(arr, path, dpi=200, manifest=None)
            tiff = load_tiff(path)
        assert isinstance(tiff.manifest, dict)


# ---------------------------------------------------------------------------
# Scanner linearization tests
# ---------------------------------------------------------------------------

class TestScannerLinearization:
    def test_fit_and_apply(self):
        from tvi.calibration.scanner_linearization import fit_scanner_calibration

        nominals = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
        # Simulate mildly non-linear scanner: raw ∝ nominal^1.2
        raw = (nominals ** 1.2) * 50000.0
        cal = fit_scanner_calibration(raw, nominals, degree=2)

        # Apply to training points — should recover nominals closely
        recovered = cal.apply(raw)
        np.testing.assert_allclose(recovered, nominals, atol=0.02)

    def test_output_clamped(self):
        from tvi.calibration.scanner_linearization import fit_scanner_calibration

        nominals = np.array([0.1, 0.5, 0.9])
        raw = np.array([5000.0, 25000.0, 45000.0])
        cal = fit_scanner_calibration(raw, nominals, degree=1)

        # Extreme extrapolation should be clamped to [0, 1]
        r = cal.apply(np.array([0.0, 100000.0]))
        assert np.all(r >= 0.0)
        assert np.all(r <= 1.0)

    def test_zero_raw_raises(self):
        from tvi.calibration.scanner_linearization import fit_scanner_calibration

        with pytest.raises(ValueError, match="zero"):
            fit_scanner_calibration(
                raw_counts=np.zeros(5),
                nominal_reflectances=np.linspace(0.1, 0.9, 5),
            )
