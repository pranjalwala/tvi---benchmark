"""
tvi/measurement/tvi_core.py
-----------------------------
Core TVI computation:

    Murray-Davies  → a_print from reflectances              (Eq. 2)
    TVI            → a_print − a_nom                        (Eq. 3)
    Yule-Nielsen   → fit empirical n from high-coverage steps (Eq. 4)

All equations follow the notation in the Printing Constraints document.

LIBRARIES USED
--------------
numpy  : array maths                             (open-source)
scipy  : minimize_scalar (Yule-Nielsen n fit)    (open-source)

Custom code
-----------
murray_davies()  — trivial formula, 4 lines.
yule_nielsen_aeff()  — invert Eq. 4 numerically (custom root-finding loop).
fit_yule_nielsen_n()  — minimise residuals over n using scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TVICurve:
    """
    TVI(a_nom) for one (printer, method, channel).

    Attributes
    ----------
    a_nom : 1-D array  nominal coverage, e.g. [0.1, 0.2, ..., 0.9]
    a_print : 1-D array  Murray-Davies effective coverage
    tvi : 1-D array  = a_print - a_nom   (percentage points if × 100)
    r_paper : float  reflectance of unprinted substrate
    r_ink : float    reflectance of 100% solid patch
    r_patch : 1-D array  patch reflectances at each a_nom
    """
    a_nom: np.ndarray
    a_print: np.ndarray
    tvi: np.ndarray
    r_paper: float
    r_ink: float
    r_patch: np.ndarray
    channel: str = ""
    method: str = ""
    printer_label: str = ""

    @property
    def tvi_at_50(self) -> float:
        """Primary scalar: TVI at a_nom = 0.50."""
        idx = np.argmin(np.abs(self.a_nom - 0.50))
        return float(self.tvi[idx])


@dataclass
class YuleNielsenResult:
    """Result of fitting the Yule-Nielsen n factor."""
    n: float
    residual: float
    fit_steps: np.ndarray   # a_nom values used in fit
    channel: str = ""
    method: str = ""


# ---------------------------------------------------------------------------
# Murray-Davies (Eq. 2)
# ---------------------------------------------------------------------------

def murray_davies(
    r_paper: float,
    r_ink: float,
    r_patch: np.ndarray | float,
) -> np.ndarray | float:
    """
    Compute effective printed dot area from reflectances.

        a_print = (R_paper - R_patch) / (R_paper - R_ink)          (Eq. 2)

    Parameters
    ----------
    r_paper : reflectance of unprinted substrate  [0, 1]
    r_ink   : reflectance of 100% solid patch     [0, 1]
    r_patch : reflectance of halftone patches     [0, 1], scalar or array

    Returns
    -------
    a_print : effective coverage in [0, 1]
    """
    denom = r_paper - r_ink
    if abs(denom) < 1e-9:
        raise ValueError(
            f"r_paper ({r_paper:.4f}) ≈ r_ink ({r_ink:.4f}): "
            "cannot compute Murray-Davies — check patch reflectances."
        )
    return (r_paper - r_patch) / denom


# ---------------------------------------------------------------------------
# TVI (Eq. 3)
# ---------------------------------------------------------------------------

def compute_tvi(
    a_nom: np.ndarray,
    r_paper: float,
    r_ink: float,
    r_patch: np.ndarray,
    channel: str = "",
    method: str = "",
    printer_label: str = "",
) -> TVICurve:
    """
    Compute the full TVI curve.

        TVI(a_nom) = a_print(a_nom) − a_nom                         (Eq. 3)

    Parameters
    ----------
    a_nom   : 1-D array of nominal coverage fractions (e.g. 0.1 … 0.9)
    r_paper : reflectance of 0% patch (paper)
    r_ink   : reflectance of 100% patch (solid ink)
    r_patch : reflectance of tone-ramp patches at each a_nom step

    Returns
    -------
    TVICurve
    """
    a_nom = np.asarray(a_nom, dtype=np.float64)
    r_patch = np.asarray(r_patch, dtype=np.float64)

    a_print = murray_davies(r_paper, r_ink, r_patch)
    tvi = a_print - a_nom

    return TVICurve(
        a_nom=a_nom,
        a_print=a_print,
        tvi=tvi,
        r_paper=r_paper,
        r_ink=r_ink,
        r_patch=r_patch,
        channel=channel,
        method=method,
        printer_label=printer_label,
    )


# ---------------------------------------------------------------------------
# Yule-Nielsen (Eq. 4)
# ---------------------------------------------------------------------------

def yule_nielsen_reflectance(
    a_eff: float | np.ndarray,
    r_ink: float,
    r_paper: float,
    n: float,
) -> np.ndarray:
    """
    Yule-Nielsen reflectance model (Eq. 4):

        R_YN = [ a_eff * R_ink^(1/n) + (1 - a_eff) * R_paper^(1/n) ]^n

    Parameters
    ----------
    a_eff   : physical dot area (after mechanical spreading)
    r_ink   : spectral reflectance of solid ink patch
    r_paper : spectral reflectance of unprinted paper
    n       : Yule-Nielsen factor  (n=1 → Murray-Davies)

    Returns
    -------
    R_YN : predicted reflectance
    """
    a_eff = np.asarray(a_eff, dtype=np.float64)
    return (
        a_eff * (r_ink ** (1.0 / n)) + (1.0 - a_eff) * (r_paper ** (1.0 / n))
    ) ** n


def yule_nielsen_aeff(
    r_measured: np.ndarray,
    r_ink: float,
    r_paper: float,
    n: float,
) -> np.ndarray:
    """
    Invert the Yule-Nielsen model to recover a_eff from measured reflectance.

    (R_YN)^(1/n) = a_eff * R_ink^(1/n) + (1-a_eff) * R_paper^(1/n)
    → a_eff = [ R_meas^(1/n) - R_paper^(1/n) ] / [ R_ink^(1/n) - R_paper^(1/n) ]

    This is the exact closed-form inverse of Eq. 4.
    """
    r_measured = np.asarray(r_measured, dtype=np.float64)
    ink_n = r_ink ** (1.0 / n)
    paper_n = r_paper ** (1.0 / n)
    meas_n = r_measured ** (1.0 / n)
    denom = ink_n - paper_n
    if abs(denom) < 1e-9:
        raise ValueError(
            "r_ink^(1/n) ≈ r_paper^(1/n): degenerate Yule-Nielsen inversion."
        )
    return np.clip((meas_n - paper_n) / denom, 0.0, 1.0)


def fit_yule_nielsen_n(
    a_nom: np.ndarray,
    r_patch: np.ndarray,
    r_ink: float,
    r_paper: float,
    fit_steps_mask: np.ndarray | None = None,
    n_min: float = 1.0,
    n_max: float = 4.0,
    channel: str = "",
    method: str = "",
) -> YuleNielsenResult:
    """
    Fit the Yule-Nielsen n factor by minimising squared residuals between
    a_eff(n) and a_nom over the high-coverage tone steps (Eq. 4 inverse).

    Optimisation: scipy.optimize.minimize_scalar with bounded Brent method.

    Parameters
    ----------
    a_nom : 1-D nominal coverage array
    r_patch : 1-D measured reflectance array (same length as a_nom)
    r_ink, r_paper : boundary reflectances
    fit_steps_mask : boolean mask selecting which steps to use in fit.
        Defaults to steps ≥ 0.70 (as specified in the pseudocode).
    n_min, n_max : search bounds for n
    """
    a_nom = np.asarray(a_nom, dtype=np.float64)
    r_patch = np.asarray(r_patch, dtype=np.float64)

    if fit_steps_mask is None:
        fit_steps_mask = a_nom >= 0.70

    a_fit = a_nom[fit_steps_mask]
    r_fit = r_patch[fit_steps_mask]

    if len(a_fit) == 0:
        raise ValueError("fit_steps_mask selects no steps; cannot fit n.")

    def residual(n: float) -> float:
        a_eff = yule_nielsen_aeff(r_fit, r_ink, r_paper, n)
        return float(np.sum((a_eff - a_fit) ** 2))

    result = minimize_scalar(
        residual,
        bounds=(n_min, n_max),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 500},
    )

    return YuleNielsenResult(
        n=float(result.x),
        residual=float(result.fun),
        fit_steps=a_fit,
        channel=channel,
        method=method,
    )
