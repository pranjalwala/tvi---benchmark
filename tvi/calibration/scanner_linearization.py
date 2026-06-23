"""
tvi/calibration/scanner_linearization.py
-----------------------------------------
Fits a polynomial scanner transfer function:

    reflectance = f(raw_scanner_count)

from a co-scanned step wedge with known nominal reflectances.

APPROACH (custom code on top of numpy/scipy)
--------------------------------------------
*  We have pairs (s_i, r_i) where:
     s_i = mean raw scanner count for step i of the wedge
     r_i = nominal reflectance for step i  (from config)
*  We fit a polynomial:
     r̂ = β₀ + β₁·s + β₂·s²  (degree configurable)
   using numpy.polynomial.polynomial.polyfit (least squares).
*  The fitted polynomial is stored and applied per session.

LIBRARIES USED
--------------
numpy    : polyfit, polyval      (open-source)
scipy    : not required here but available for spline alternative
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScannerCalibration:
    """
    Holds the polynomial coefficients that linearize the scanner.

    coefficients[i] is the coefficient of s^i
    (numpy poly1d convention: highest power first → we use polyval convention).
    """
    coefficients: np.ndarray   # shape (degree+1,) lowest power first
    degree: int
    session_id: str = ""

    def apply(self, raw_counts: np.ndarray | float) -> np.ndarray | float:
        """
        Convert raw scanner counts → linearized reflectance.

        raw_counts may be a scalar or an ndarray of any shape.
        Output is clamped to [0, 1].
        """
        r = np.polynomial.polynomial.polyval(raw_counts, self.coefficients)
        return np.clip(r, 0.0, 1.0)


def fit_scanner_calibration(
    raw_counts: np.ndarray,
    nominal_reflectances: np.ndarray,
    degree: int = 2,
    session_id: str = "",
) -> ScannerCalibration:
    """
    Fit a polynomial scanner transfer function.

    Parameters
    ----------
    raw_counts : 1-D array of mean scanner counts per wedge step.
        Values should be in [0, max_raw_value] — NOT yet normalised.
        The function normalises internally to [0, 1] to keep coefficients
        numerically stable; the calibration object handles denormalisation.
    nominal_reflectances : 1-D array of known reflectance values (0–1).
    degree : polynomial degree (default 2).
    session_id : optional string label for logging.

    Returns
    -------
    ScannerCalibration
    """
    raw_counts = np.asarray(raw_counts, dtype=np.float64)
    nominal_reflectances = np.asarray(nominal_reflectances, dtype=np.float64)

    if raw_counts.shape != nominal_reflectances.shape:
        raise ValueError(
            f"raw_counts ({raw_counts.shape}) and nominal_reflectances "
            f"({nominal_reflectances.shape}) must have the same shape."
        )

    # Normalise raw counts to [0, 1] to improve numerical stability.
    max_count = raw_counts.max()
    if max_count == 0:
        raise ValueError("All raw_counts are zero — check the scan.")
    s_norm = raw_counts / max_count

    # Least-squares polynomial fit: s_norm → reflectance
    coeffs = np.polynomial.polynomial.polyfit(s_norm, nominal_reflectances, degree)

    # Wrap so apply() also normalises input by max_count.
    # We do this by composing: s_norm = s / max_count,
    # then evaluate polynomial at s_norm.
    # Store max_count inside the calibration for transparency.

    return _NormalisedCalibration(
        coefficients=coeffs,
        degree=degree,
        session_id=session_id,
        max_count=max_count,
    )


def measure_wedge_counts(
    wedge_scan: np.ndarray,
    manifest: dict,
    channel_idx: int,
    erosion_margin: float = 0.10,
) -> np.ndarray:
    """
    Extract mean scanner counts for each step-wedge patch.

    Parameters
    ----------
    wedge_scan : ndarray (H, W, C) — the scanned step wedge image
    manifest : the parsed patch manifest from the TIFF ImageDescription
    channel_idx : which RGB channel to use (0=R, 1=G, 2=B)
    erosion_margin : interior fraction to use (avoids border artefacts)

    Returns
    -------
    counts : 1-D array of mean raw counts per step
    """
    from tvi.preprocessing.alignment import extract_patch

    patches = manifest["patches"]
    counts = []
    for p in patches:
        interior = extract_patch(
            wedge_scan,
            row=p["row_px"],
            col=p["col_px"],
            height=p["height_px"],
            width=p["width_px"],
            erosion_margin=erosion_margin,
        )
        # Select channel
        if interior.ndim == 3:
            interior = interior[..., channel_idx]
        counts.append(float(interior.mean()))
    return np.array(counts)


# ---------------------------------------------------------------------------
# Private: subclass to carry max_count
# ---------------------------------------------------------------------------

class _NormalisedCalibration(ScannerCalibration):
    def __init__(self, *, max_count: float, **kwargs):
        super().__init__(**kwargs)
        self.max_count = max_count

    def apply(self, raw_counts):
        s_norm = np.asarray(raw_counts, dtype=np.float64) / self.max_count
        r = np.polynomial.polynomial.polyval(s_norm, self.coefficients)
        return np.clip(r, 0.0, 1.0)
