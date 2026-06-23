"""
tests/test_measurement.py
--------------------------
Unit tests for:
  - tvi.measurement.measure_tone_ramps  (MeasuredSheet, PatchReflectance)
  - tvi.calibration.transfer_function   (TransferFunction, inversion, CSV round-trip)
  - tvi.simulation.dot_gain_simulator   (SimulatorParams, simulate_dot_gain)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tvi.measurement.measure_tone_ramps import MeasuredSheet, PatchReflectance
from tvi.measurement.tvi_core import compute_tvi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_measured_sheet(
    tvi_level: float = 0.15,
    channel: str = "K",
    method: str = "DBS",
    n_steps: int = 9,
) -> MeasuredSheet:
    """
    Construct a synthetic MeasuredSheet with a parabolic TVI at tvi_level at 50%.
    """
    r_paper, r_ink = 0.92, 0.04
    tone_steps = list(range(0, 110, 10))   # 0, 10, ..., 100 %
    patches = []
    for step in tone_steps:
        a_nom = step / 100.0
        tvi = tvi_level * 4 * a_nom * (1 - a_nom)
        a_print = np.clip(a_nom + tvi, 0, 1)
        r_patch = r_paper - a_print * (r_paper - r_ink)
        patches.append(PatchReflectance(
            nominal_pct=step,
            a_nom=a_nom,
            reflectance=float(r_patch),
            channel=channel,
            scan_idx=0,
            sheet_idx=0,
        ))
    return MeasuredSheet(
        channel=channel,
        method=method,
        printer_label="test_printer",
        sheet_idx=0,
        scan_idx=0,
        patches=patches,
        r_paper=r_paper,
        r_ink=r_ink,
    )


# ---------------------------------------------------------------------------
# MeasuredSheet
# ---------------------------------------------------------------------------

class TestMeasuredSheet:
    def test_r_paper_r_ink_set(self):
        sheet = _make_measured_sheet()
        assert sheet.r_paper == pytest.approx(0.92, abs=1e-6)
        assert sheet.r_ink == pytest.approx(0.04, abs=1e-6)

    def test_a_nom_array_excludes_boundaries(self):
        sheet = _make_measured_sheet()
        a_nom = sheet.a_nom_array
        assert 0.0 not in a_nom
        assert 1.0 not in a_nom
        assert len(a_nom) == 9   # 10%–90%

    def test_r_patch_array_length(self):
        sheet = _make_measured_sheet()
        assert len(sheet.r_patch_array) == len(sheet.a_nom_array)

    def test_to_tvi_curve_tvi_at_50(self):
        sheet = _make_measured_sheet(tvi_level=0.15)
        curve = sheet.to_tvi_curve()
        assert curve.tvi_at_50 == pytest.approx(0.15, abs=0.02)

    def test_labels_in_tvi_curve(self):
        sheet = _make_measured_sheet(channel="C", method="ErrorDiffusion")
        curve = sheet.to_tvi_curve()
        assert curve.channel == "C"
        assert curve.method == "ErrorDiffusion"


# ---------------------------------------------------------------------------
# TransferFunction
# ---------------------------------------------------------------------------

class TestTransferFunction:
    def _make_tf(self, tvi: float = 0.12) -> "TransferFunction":
        from tvi.calibration.transfer_function import fit_transfer_function
        a_nom = np.linspace(0.0, 1.0, 11)
        a_print = np.clip(a_nom + tvi * 4 * a_nom * (1 - a_nom), 0, 1)
        return fit_transfer_function(a_nom, a_print, channel="K", method="DBS")

    def test_evaluate_boundaries(self):
        tf = self._make_tf()
        assert tf.evaluate(0.0) == pytest.approx(0.0, abs=0.01)
        assert tf.evaluate(1.0) == pytest.approx(1.0, abs=0.01)

    def test_evaluate_midtone_larger_than_nom(self):
        tf = self._make_tf(tvi=0.12)
        assert tf.evaluate(0.50) > 0.50

    def test_is_monotone(self):
        tf = self._make_tf()
        assert tf.is_monotone is True

    def test_invert_round_trip(self):
        tf = self._make_tf()
        a_print_q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        a_nom_est = tf.invert(a_print_q)
        a_print_back = tf.evaluate(a_nom_est)
        np.testing.assert_allclose(a_print_back, a_print_q, atol=0.01)

    def test_tvi_method(self):
        tf = self._make_tf(tvi=0.12)
        # TVI at 50% should be approximately 0.12
        assert tf.tvi(0.50) == pytest.approx(0.12, abs=0.02)

    def test_csv_round_trip(self):
        from tvi.calibration.transfer_function import (
            load_transfer_function_csv,
            save_transfer_function_csv,
        )
        tf = self._make_tf(tvi=0.10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tf.csv"
            save_transfer_function_csv(tf, path)
            tf2 = load_transfer_function_csv(path)
        # Values at midtone should match closely
        assert tf2.evaluate(0.50) == pytest.approx(tf.evaluate(0.50), abs=0.002)

    def test_save_csv_creates_file(self):
        from tvi.calibration.transfer_function import save_transfer_function_csv
        tf = self._make_tf()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "tf.csv"
            save_transfer_function_csv(tf, path)
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 102   # header + 101 data rows


# ---------------------------------------------------------------------------
# Dot-gain simulator
# ---------------------------------------------------------------------------

class TestDotGainSimulator:
    def _make_params(
        self,
        r_mech: float = 0.8,
        n_yn: float = 1.5,
    ) -> "SimulatorParams":
        from tvi.simulation.dot_gain_simulator import SimulatorParams
        return SimulatorParams(
            r_mech=r_mech,
            n_yn=n_yn,
            r_paper=0.92,
            r_ink=0.04,
            channel="K",
            method="DBS",
        )

    def test_output_shape_preserved(self):
        from tvi.simulation.dot_gain_simulator import simulate_dot_gain
        halftone = np.zeros((32, 32), dtype=np.uint8)
        halftone[8:24, 8:24] = 255
        params = self._make_params()
        result = simulate_dot_gain(halftone, params, output="reflectance")
        assert result.shape == (32, 32)

    def test_reflectance_in_bounds(self):
        from tvi.simulation.dot_gain_simulator import simulate_dot_gain
        rng = np.random.default_rng(0)
        halftone = (rng.random((64, 64)) > 0.5).astype(np.uint8) * 255
        params = self._make_params()
        result = simulate_dot_gain(halftone, params, output="reflectance")
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_uint8_output(self):
        from tvi.simulation.dot_gain_simulator import simulate_dot_gain
        halftone = np.zeros((16, 16), dtype=np.uint8)
        params = self._make_params()
        result = simulate_dot_gain(halftone, params, output="uint8")
        assert result.dtype == np.uint8

    def test_dot_growth_darkens_midtone(self):
        """
        A 50% halftone with dilation should be darker (lower reflectance)
        than the undilated version.
        """
        from tvi.simulation.dot_gain_simulator import simulate_dot_gain, SimulatorParams

        rng = np.random.default_rng(42)
        half = (rng.random((64, 64)) > 0.50).astype(np.uint8)

        params_no_gain = SimulatorParams(r_mech=0.0, n_yn=1.0, r_paper=0.92, r_ink=0.04)
        params_gain = SimulatorParams(r_mech=1.0, n_yn=1.5, r_paper=0.92, r_ink=0.04)

        r_no_gain = simulate_dot_gain(half, params_no_gain, output="reflectance").mean()
        r_gain = simulate_dot_gain(half, params_gain, output="reflectance").mean()

        assert r_gain < r_no_gain, "Dot gain must darken the image"

    def test_estimate_simulator_params_returns_params(self):
        from tvi.calibration.transfer_function import fit_transfer_function
        from tvi.simulation.dot_gain_simulator import estimate_simulator_params

        a_nom = np.linspace(0.0, 1.0, 11)
        a_print = np.clip(a_nom + 0.12 * 4 * a_nom * (1 - a_nom), 0, 1)
        tf = fit_transfer_function(a_nom, a_print, channel="K", method="DBS")

        params = estimate_simulator_params(tf, r_paper=0.92, r_ink=0.04)
        assert params.r_mech >= 0
        assert 1.0 <= params.n_yn <= 4.0
        assert np.isfinite(params.rmse)
