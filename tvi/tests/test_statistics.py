"""
tests/test_statistics.py
-------------------------
Unit tests for tvi.aggregation.statistics:
  - aggregate_tvi_curve()
  - AggregatedResult.tvi_at_50 / ci_at_50
  - build_metric_table()
  - summary_pivot()
  - save_csv() / load_csv()
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tvi.aggregation.statistics import (
    AggregatedResult,
    aggregate_tvi_curve,
    build_metric_table,
    load_csv,
    save_csv,
    summary_pivot,
)
from tvi.measurement.tvi_core import TVICurve, compute_tvi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_curves(
    n: int = 9,
    tvi_mean: float = 0.15,
    noise: float = 0.005,
    channel: str = "K",
    method: str = "DBS",
) -> list[TVICurve]:
    """Generate n synthetic TVICurve objects with Gaussian noise around tvi_mean."""
    r_paper, r_ink = 0.92, 0.04
    a_nom = np.arange(0.1, 1.0, 0.1)
    rng = np.random.default_rng(seed=0)
    curves = []
    for _ in range(n):
        tvi = tvi_mean * 4 * a_nom * (1 - a_nom) + rng.normal(0, noise, size=len(a_nom))
        a_print = np.clip(a_nom + tvi, 0, 1)
        r_patch = r_paper - a_print * (r_paper - r_ink)
        curves.append(compute_tvi(a_nom, r_paper, r_ink, r_patch,
                                  channel=channel, method=method))
    return curves


# ---------------------------------------------------------------------------
# aggregate_tvi_curve
# ---------------------------------------------------------------------------

class TestAggregateTVICurve:
    def test_basic_shape(self):
        curves = _make_curves(n=9)
        result = aggregate_tvi_curve(curves)
        assert result.mean_tvi.shape == curves[0].a_nom.shape
        assert result.ci_tvi.shape == result.mean_tvi.shape

    def test_n_obs(self):
        curves = _make_curves(n=9)
        result = aggregate_tvi_curve(curves)
        assert result.n_obs == 9

    def test_mean_is_close_to_injected(self):
        curves = _make_curves(n=9, tvi_mean=0.15, noise=0.001)
        result = aggregate_tvi_curve(curves)
        # TVI at 50% ≈ 0.15
        assert result.tvi_at_50 == pytest.approx(0.15, abs=0.02)

    def test_ci_is_positive(self):
        curves = _make_curves(n=9, noise=0.010)
        result = aggregate_tvi_curve(curves)
        assert np.all(result.ci_tvi >= 0)

    def test_ci_shrinks_with_more_reps(self):
        """CI half-width should decrease as n increases (law of large numbers)."""
        ci_small = aggregate_tvi_curve(_make_curves(n=3, noise=0.01)).ci_tvi.mean()
        ci_large = aggregate_tvi_curve(_make_curves(n=27, noise=0.01)).ci_tvi.mean()
        assert ci_large < ci_small

    def test_mismatched_a_nom_raises(self):
        curves = _make_curves(n=2)
        # Corrupt the second curve's a_nom
        bad = TVICurve(
            a_nom=np.arange(0.05, 0.95, 0.1),
            a_print=curves[1].a_print,
            tvi=curves[1].tvi,
            r_paper=curves[1].r_paper,
            r_ink=curves[1].r_ink,
            r_patch=curves[1].r_patch,
        )
        with pytest.raises(ValueError, match="a_nom grid"):
            aggregate_tvi_curve([curves[0], bad])

    def test_single_curve_zero_ci(self):
        curves = _make_curves(n=1)
        result = aggregate_tvi_curve(curves)
        np.testing.assert_allclose(result.ci_tvi, np.zeros_like(result.ci_tvi), atol=1e-9)

    def test_raw_tvi_matrix_stored(self):
        curves = _make_curves(n=6)
        result = aggregate_tvi_curve(curves)
        assert result.raw_tvi_matrix is not None
        assert result.raw_tvi_matrix.shape == (6, len(curves[0].a_nom))

    def test_95_ci_level(self):
        """Test that CI is computed with t(0.975, df) multiplier."""
        from scipy.stats import t as t_dist
        n = 9
        curves = _make_curves(n=n, noise=0.01)
        result = aggregate_tvi_curve(curves, ci_level=0.95)
        # Manually compute expected CI at one step
        step_idx = 4  # a_nom = 0.50
        tvi_vals = result.raw_tvi_matrix[:, step_idx]
        t_crit = t_dist.ppf(0.975, df=n - 1)
        expected_ci = t_crit * tvi_vals.std(ddof=1) / np.sqrt(n)
        assert result.ci_tvi[step_idx] == pytest.approx(expected_ci, rel=1e-5)


# ---------------------------------------------------------------------------
# build_metric_table
# ---------------------------------------------------------------------------

class TestBuildMetricTable:
    def _make_results(self) -> list[AggregatedResult]:
        results = []
        for method in ["DBS", "ErrorDiffusion"]:
            for ch in ["C", "K"]:
                curves = _make_curves(n=9, tvi_mean=0.12, channel=ch, method=method)
                results.append(aggregate_tvi_curve(curves))
        return results

    def test_returns_dataframe(self):
        results = self._make_results()
        df = build_metric_table(results)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        results = self._make_results()
        df = build_metric_table(results)
        assert len(df) == 4   # 2 methods × 2 channels

    def test_required_columns(self):
        results = self._make_results()
        df = build_metric_table(results)
        for col in ["method", "channel", "tvi_at_50_pp", "ci_at_50_pp", "n_obs"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_tvi_in_pp(self):
        """tvi_at_50_pp should be in percentage points (0–100 range, not 0–1)."""
        curves = _make_curves(n=9, tvi_mean=0.12)
        result = aggregate_tvi_curve(curves)
        df = build_metric_table([result])
        # 0.12 → 12 pp
        assert df["tvi_at_50_pp"].iloc[0] == pytest.approx(12.0, abs=2.0)


# ---------------------------------------------------------------------------
# summary_pivot
# ---------------------------------------------------------------------------

class TestSummaryPivot:
    def test_pivot_shape(self):
        results = []
        for method in ["DBS", "ErrorDiff", "OrderedDither"]:
            for ch in ["C", "M", "K"]:
                curves = _make_curves(n=3, channel=ch, method=method)
                results.append(aggregate_tvi_curve(curves))
        df = build_metric_table(results)
        pivot = summary_pivot(df)
        assert pivot.shape == (3, 3)   # 3 methods × 3 channels
        assert set(pivot.columns) == {"C", "M", "K"}


# ---------------------------------------------------------------------------
# CSV serialisation
# ---------------------------------------------------------------------------

class TestCSVSerialization:
    def _make_results(self) -> list[AggregatedResult]:
        return [aggregate_tvi_curve(_make_curves(n=6, method=m, channel=c))
                for m in ["DBS", "ErrorDiff"] for c in ["K", "C"]]

    def test_save_and_load_metric_table(self):
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.csv"
            save_csv(results, path, kind="metric_table")
            df = load_csv(path)
        assert isinstance(df, pd.DataFrame)
        assert "tvi_at_50_pp" in df.columns
        assert len(df) == 4

    def test_save_and_load_full_curves(self):
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.csv"
            save_csv(results, path, kind="full_curves")
            df = load_csv(path)
        assert "a_nom" in df.columns
        assert "mean_tvi_pp" in df.columns
        # 4 results × 9 tone steps = 36 rows
        assert len(df) == 4 * 9

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            save_csv([], "x.csv", kind="invalid_kind")
