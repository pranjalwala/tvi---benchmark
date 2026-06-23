"""
tvi/calibration/transfer_function.py
--------------------------------------
Printer transfer function: nominal coverage → printed coverage.

This is the device characterisation curve that underlies the TVI simulator.
It lets us predict how a specific printer will reproduce any requested tone
without physically printing every benchmark image.

Mathematical definition
-----------------------
Given a measured TVI curve TVI(a_nom), the printed coverage at any nominal
coverage a is:

    a_print(a) = a_nom + TVI(a_nom)                                  (Eq. 3)

The transfer function f : [0, 1] → [0, 1] is therefore:

    f(a_nom) = a_nom + TVI(a_nom)

We represent f as a 1-D monotone cubic spline fitted through the measured
(a_nom, a_print) pairs.  The spline is then used for:

  * evaluation   : a_nom  → a_print    (forward)
  * inversion    : a_print → a_nom     (inverse, needed for pre-compensation)

Monotonicity
------------
A physical transfer function must be monotonically non-decreasing: darker
input always gives darker output.  We enforce this via a Pchip (Piecewise
Cubic Hermite Interpolating Polynomial) spline, which guarantees monotonicity
within the data range.

CSV serialisation
-----------------
The transfer function is saved as a two-column CSV:
    a_nom, a_print
so it can be loaded without re-running the full measurement pipeline.

LIBRARIES USED
--------------
numpy  : array maths                                     (open-source)
scipy  : PchipInterpolator (monotone spline)             (open-source)
         interp1d (linear fallback)

Custom code: monotonicity check, inversion wrapper, CSV I/O.

Integration
-----------
from tvi.measurement.tvi_core import TVICurve
from tvi.calibration.transfer_function import (
    fit_transfer_function, evaluate_transfer_function,
    invert_transfer_function, TransferFunction,
)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TransferFunction:
    """
    Printer transfer function: a_nom → a_print.

    Attributes
    ----------
    a_nom     : 1-D array of nominal coverage knot points (sorted, [0, 1])
    a_print   : 1-D array of measured printed coverage at each knot
    channel   : ink channel label
    method    : halftoning method label
    printer_label : device label
    is_monotone : whether the curve passed the monotonicity check
    _spline   : fitted PchipInterpolator (not serialised)
    """
    a_nom: np.ndarray
    a_print: np.ndarray
    channel: str = ""
    method: str = ""
    printer_label: str = ""
    is_monotone: bool = True
    _spline: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.a_nom = np.asarray(self.a_nom, dtype=np.float64)
        self.a_print = np.asarray(self.a_print, dtype=np.float64)
        if self._spline is None:
            self._spline = PchipInterpolator(self.a_nom, self.a_print, extrapolate=True)

    # ------------------------------------------------------------------
    def evaluate(self, a: np.ndarray | float) -> np.ndarray:
        """
        Forward evaluation: a_nom → a_print.

        Parameters
        ----------
        a : scalar or array of nominal coverage values in [0, 1]

        Returns
        -------
        a_print : clipped to [0, 1]
        """
        return np.clip(self._spline(np.asarray(a, dtype=np.float64)), 0.0, 1.0)

    def invert(self, a_print_query: np.ndarray | float) -> np.ndarray:
        """
        Inverse evaluation: a_print → a_nom.

        Uses the monotone spline inverse via a dense forward evaluation
        followed by interpolation on the swapped axes.

        Parameters
        ----------
        a_print_query : target printed coverage values

        Returns
        -------
        a_nom_est : estimated nominal coverage to achieve each target
        """
        return invert_transfer_function(self, a_print_query)

    def tvi(self, a: np.ndarray | float) -> np.ndarray:
        """Return TVI = a_print(a) - a at any coverage."""
        a = np.asarray(a, dtype=np.float64)
        return self.evaluate(a) - a


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_transfer_function(
    a_nom: np.ndarray,
    a_print: np.ndarray,
    channel: str = "",
    method: str = "",
    printer_label: str = "",
) -> TransferFunction:
    """
    Fit a monotone cubic spline transfer function from measured knot pairs.

    Parameters
    ----------
    a_nom   : 1-D array of nominal coverage fractions (must include 0 and 1)
    a_print : 1-D array of measured printed coverage  (same length)
    channel, method, printer_label : metadata labels

    Returns
    -------
    TransferFunction

    Notes
    -----
    * Knots are sorted by a_nom before fitting.
    * If boundary values (0%, 100%) are absent they are inserted:
        f(0) = 0,  f(1) = 1  (ideal; real device may deviate slightly).
    * A monotonicity check is run; the result is stored in is_monotone.
      Non-monotone curves are still returned but flagged.
    """
    a_nom = np.asarray(a_nom, dtype=np.float64)
    a_print = np.asarray(a_print, dtype=np.float64)

    if a_nom.shape != a_print.shape or a_nom.ndim != 1:
        raise ValueError("a_nom and a_print must be 1-D arrays of equal length.")

    # Sort by a_nom
    order = np.argsort(a_nom)
    a_nom = a_nom[order]
    a_print = a_print[order]

    # Ensure boundary knots exist
    a_nom, a_print = _ensure_boundaries(a_nom, a_print)

    # Fit spline
    spline = PchipInterpolator(a_nom, a_print, extrapolate=True)

    # Monotonicity check via derivative on a fine grid
    grid = np.linspace(0.0, 1.0, 500)
    deriv = spline(grid, 1)            # first derivative
    is_monotone = bool(np.all(deriv >= -1e-4))

    if not is_monotone:
        import warnings
        warnings.warn(
            f"Transfer function for channel={channel}, method={method} is "
            "non-monotone. TVI data may be noisy or measurement errors exist. "
            "The spline is still returned but flagged.",
            UserWarning,
            stacklevel=2,
        )

    return TransferFunction(
        a_nom=a_nom,
        a_print=a_print,
        channel=channel,
        method=method,
        printer_label=printer_label,
        is_monotone=is_monotone,
        _spline=spline,
    )


def evaluate_transfer_function(
    tf: TransferFunction,
    a_query: np.ndarray | float,
) -> np.ndarray:
    """
    Evaluate the transfer function at arbitrary coverage values.

    Thin wrapper around TransferFunction.evaluate() for a functional style.
    """
    return tf.evaluate(a_query)


def invert_transfer_function(
    tf: TransferFunction,
    a_print_query: np.ndarray | float,
) -> np.ndarray:
    """
    Compute the nominal coverage required to achieve a target printed coverage.

    Strategy
    --------
    1. Evaluate the forward spline on a dense grid of a_nom values.
    2. Fit a linear interpolant on the (a_print → a_nom) direction.
    3. Query it at a_print_query.

    This works correctly for monotone functions.  For non-monotone functions,
    the result at the non-monotone segment is undefined and a warning is issued.

    Parameters
    ----------
    tf : TransferFunction
    a_print_query : target printed coverage scalar or array

    Returns
    -------
    a_nom_est : estimated nominal coverages, clipped to [0, 1]
    """
    from scipy.interpolate import interp1d

    a_print_query = np.asarray(a_print_query, dtype=np.float64)

    # Build dense forward map
    a_nom_dense = np.linspace(0.0, 1.0, 2000)
    a_print_dense = np.clip(tf._spline(a_nom_dense), 0.0, 1.0)

    if not tf.is_monotone:
        import warnings
        warnings.warn(
            "Inverting a non-monotone transfer function — result may be ambiguous.",
            UserWarning,
            stacklevel=2,
        )

    # Unique a_print values for invertible interpolation
    _, unique_idx = np.unique(a_print_dense, return_index=True)
    inv_interp = interp1d(
        a_print_dense[unique_idx],
        a_nom_dense[unique_idx],
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )
    return np.clip(inv_interp(a_print_query), 0.0, 1.0)


# ---------------------------------------------------------------------------
# CSV serialisation
# ---------------------------------------------------------------------------

def save_transfer_function_csv(
    tf: TransferFunction,
    path: str | Path,
    n_eval_points: int = 101,
) -> None:
    """
    Save the transfer function as a two-column CSV.

    Columns: a_nom, a_print
    Evaluates on a uniform grid of n_eval_points in [0, 1].

    Parameters
    ----------
    tf : TransferFunction
    path : output CSV file path
    n_eval_points : number of rows (default 101 → 0.00, 0.01, …, 1.00)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a_nom_grid = np.linspace(0.0, 1.0, n_eval_points)
    a_print_grid = tf.evaluate(a_nom_grid)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "a_nom", "a_print", "tvi",
            "channel", "method", "printer_label",
        ])
        for a_n, a_p in zip(a_nom_grid, a_print_grid):
            writer.writerow([
                f"{a_n:.6f}",
                f"{a_p:.6f}",
                f"{a_p - a_n:.6f}",
                tf.channel,
                tf.method,
                tf.printer_label,
            ])


def load_transfer_function_csv(path: str | Path) -> TransferFunction:
    """
    Load a TransferFunction from a CSV saved by save_transfer_function_csv().

    Parameters
    ----------
    path : CSV file path

    Returns
    -------
    TransferFunction
    """
    path = Path(path)
    a_nom_list, a_print_list = [], []
    channel = method = printer_label = ""

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a_nom_list.append(float(row["a_nom"]))
            a_print_list.append(float(row["a_print"]))
            channel = row.get("channel", "")
            method = row.get("method", "")
            printer_label = row.get("printer_label", "")

    return fit_transfer_function(
        a_nom=np.array(a_nom_list),
        a_print=np.array(a_print_list),
        channel=channel,
        method=method,
        printer_label=printer_label,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_boundaries(
    a_nom: np.ndarray,
    a_print: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert (0, 0) and (1, 1) boundary knots if absent."""
    if a_nom[0] > 1e-6:
        a_nom = np.concatenate([[0.0], a_nom])
        a_print = np.concatenate([[0.0], a_print])
    if a_nom[-1] < 1.0 - 1e-6:
        a_nom = np.concatenate([a_nom, [1.0]])
        a_print = np.concatenate([a_print, [1.0]])
    return a_nom, a_print
