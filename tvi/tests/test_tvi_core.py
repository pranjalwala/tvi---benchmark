"""
tests/test_tvi_core.py
-----------------------
Unit tests for tvi.measurement.tvi_core:
  - Murray-Davies equation (Eq. 2)
  - TVI computation (Eq. 3)
  - Yule-Nielsen forward model (Eq. 4)
  - Yule-Nielsen inverse
  - Yule-Nielsen n fitting
"""

import numpy as np
import pytest

from tvi.measurement.tvi_core import (
    TVICurve,
    YuleNielsenResult,
    compute_tvi,
    fit_yule_nielsen_n,
    murray_davies,
    yule_nielsen_aeff,
    yule_nielsen_reflectance,
)


# ---------------------------------------------------------------------------
# Murray-Davies
# ---------------------------------------------------------------------------

class TestMurrayDavies:
    def test_zero_coverage(self):
        """0% ink: patch reflectance = paper, a_print should be 0."""
        r_paper, r_ink = 0.90, 0.04
        assert murray_davies(r_paper, r_ink, r_paper) == pytest.approx(0.0, abs=1e-9)

    def test_full_coverage(self):
        """100% ink: patch reflectance = ink, a_print should be 1."""
        r_paper, r_ink = 0.90, 0.04
        assert murray_davies(r_paper, r_ink, r_ink) == pytest.approx(1.0, abs=1e-9)

    def test_midtone(self):
        """50% coverage under linear mixing: a_print = 0.50."""
        r_paper, r_ink = 0.90, 0.04
        r_mid = 0.5 * r_ink + 0.5 * r_paper
        result = murray_davies(r_paper, r_ink, r_mid)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_dot_gain(self):
        """Dot-gained patch (darker than linear) → a_print > a_nom."""
        r_paper, r_ink = 0.90, 0.04
        a_nom = 0.50
        a_print_expected = 0.65          # 15 pp TVI
        r_patch = r_paper - a_print_expected * (r_paper - r_ink)
        result = murray_davies(r_paper, r_ink, r_patch)
        assert result == pytest.approx(a_print_expected, abs=1e-6)

    def test_array_input(self):
        """Works with numpy array input."""
        r_paper, r_ink = 0.90, 0.04
        r_patches = np.array([0.90, 0.47, 0.04])
        result = murray_davies(r_paper, r_ink, r_patches)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(1.0, abs=1e-6)

    def test_degenerate_raises(self):
        """r_paper == r_ink should raise ValueError."""
        with pytest.raises(ValueError, match="r_paper"):
            murray_davies(0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# TVI computation
# ---------------------------------------------------------------------------

class TestComputeTVI:
    def _make_curve(self, tvi_target: float = 0.15) -> TVICurve:
        """Synthetic curve with constant TVI = tvi_target (parabolic peak)."""
        r_paper, r_ink = 0.92, 0.04
        a_nom = np.arange(0.1, 1.0, 0.1)
        # Parabolic TVI: peaks at 50%
        tvi = tvi_target * 4 * a_nom * (1 - a_nom)
        a_print = np.clip(a_nom + tvi, 0, 1)
        r_patch = r_paper - a_print * (r_paper - r_ink)
        return compute_tvi(a_nom, r_paper, r_ink, r_patch,
                           channel="K", method="DBS")

    def test_tvi_shape(self):
        curve = self._make_curve()
        assert curve.tvi.shape == curve.a_nom.shape

    def test_tvi_positive_with_dot_gain(self):
        """TVI must be positive when dots gain area."""
        curve = self._make_curve(tvi_target=0.15)
        assert np.all(curve.tvi >= -1e-6), "TVI should be non-negative for typical dot gain"

    def test_tvi_at_50_approx(self):
        """TVI at 50% should be close to the injected value."""
        curve = self._make_curve(tvi_target=0.15)
        # Parabolic TVI peaks at 0.15 at a_nom=0.5
        assert curve.tvi_at_50 == pytest.approx(0.15, abs=0.01)

    def test_labels_propagate(self):
        curve = self._make_curve()
        assert curve.channel == "K"
        assert curve.method == "DBS"


# ---------------------------------------------------------------------------
# Yule-Nielsen forward model
# ---------------------------------------------------------------------------

class TestYuleNielsenForward:
    def test_n1_equals_murray_davies(self):
        """At n=1, Yule-Nielsen should equal Murray-Davies linear mixing."""
        r_paper, r_ink, n = 0.90, 0.04, 1.0
        a = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        yn = yule_nielsen_reflectance(a, r_ink, r_paper, n)
        linear = a * r_ink + (1 - a) * r_paper
        np.testing.assert_allclose(yn, linear, atol=1e-9)

    def test_n_gt1_darker_than_linear(self):
        """
        For n > 1, effective dot area is higher than linear mixing predicts.
        Reflectance at 50% under Yule-Nielsen should be LOWER (darker) than linear.
        """
        r_paper, r_ink = 0.90, 0.04
        a = 0.50
        yn_n1 = yule_nielsen_reflectance(a, r_ink, r_paper, 1.0)
        yn_n2 = yule_nielsen_reflectance(a, r_ink, r_paper, 2.0)
        assert yn_n2 < yn_n1, "Higher n should give lower reflectance (more optical gain)"

    def test_boundary_values(self):
        r_paper, r_ink = 0.90, 0.04
        assert yule_nielsen_reflectance(0.0, r_ink, r_paper, 2.0) == pytest.approx(r_paper, abs=1e-6)
        assert yule_nielsen_reflectance(1.0, r_ink, r_paper, 2.0) == pytest.approx(r_ink, abs=1e-6)


# ---------------------------------------------------------------------------
# Yule-Nielsen inverse
# ---------------------------------------------------------------------------

class TestYuleNielsenInverse:
    def test_round_trip(self):
        """yule_nielsen_aeff(yule_nielsen_reflectance(a)) ≈ a."""
        r_paper, r_ink, n = 0.90, 0.04, 1.8
        a_orig = np.linspace(0.0, 1.0, 11)
        r = yule_nielsen_reflectance(a_orig, r_ink, r_paper, n)
        a_recovered = yule_nielsen_aeff(r, r_ink, r_paper, n)
        np.testing.assert_allclose(a_recovered, a_orig, atol=1e-7)

    def test_n1_recovers_murray_davies(self):
        """At n=1, inverse should equal Murray-Davies."""
        r_paper, r_ink = 0.90, 0.04
        r_patches = np.array([0.90, 0.60, 0.40, 0.20, 0.04])
        md = murray_davies(r_paper, r_ink, r_patches)
        yn_inv = yule_nielsen_aeff(r_patches, r_ink, r_paper, n=1.0)
        np.testing.assert_allclose(yn_inv, md, atol=1e-7)


# ---------------------------------------------------------------------------
# Yule-Nielsen n fitting
# ---------------------------------------------------------------------------

class TestFitYuleNielsenN:
    def test_recovers_known_n(self):
        """
        Synthesise Yule-Nielsen data at n=1.8, then fit and recover n.
        """
        r_paper, r_ink, n_true = 0.90, 0.04, 1.8
        a_nom = np.arange(0.1, 1.0, 0.1)
        r_patch = yule_nielsen_reflectance(a_nom, r_ink, r_paper, n_true)

        result = fit_yule_nielsen_n(
            a_nom=a_nom,
            r_patch=r_patch,
            r_ink=r_ink,
            r_paper=r_paper,
            n_min=1.0,
            n_max=4.0,
        )
        assert result.n == pytest.approx(n_true, abs=0.05)

    def test_result_type(self):
        r_paper, r_ink = 0.90, 0.04
        a_nom = np.array([0.7, 0.8, 0.9])
        r_patch = yule_nielsen_reflectance(a_nom, r_ink, r_paper, 1.5)
        result = fit_yule_nielsen_n(a_nom, r_patch, r_ink, r_paper)
        assert isinstance(result, YuleNielsenResult)
        assert 1.0 <= result.n <= 4.0

    def test_n1_limit(self):
        """When data matches linear (n=1), fit should return n ≈ 1."""
        r_paper, r_ink = 0.90, 0.04
        a_nom = np.arange(0.1, 1.0, 0.1)
        # Linear mixing = Yule-Nielsen at n=1
        r_patch = a_nom * r_ink + (1 - a_nom) * r_paper
        result = fit_yule_nielsen_n(a_nom, r_patch, r_ink, r_paper)
        assert result.n == pytest.approx(1.0, abs=0.1)
