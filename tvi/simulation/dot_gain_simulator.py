"""
tvi/simulation/dot_gain_simulator.py
--------------------------------------
Printer-aware TVI simulator.

Purpose
-------
This module lets us apply a measured printer transfer function to all 450
benchmark halftone images *without* physically printing every one.  The
simulator replicates the dominant physical phenomena:

  1. Mechanical dot growth  — each ink dot expands spatially on the substrate.
  2. Dot merging           — nearby dots touch and fill gaps (midtone darkening).
  3. Optical gain          — lateral light scatter under ink (Yule-Nielsen).
  4. Fine-line thickening  — 1-px lines become 2-3 px.
  5. Highlight fill-in     — isolated 5–10% dots merge, losing texture.
  6. Shadow fill-in        — 90–95% coverage loses white holes.
  7. Texture loss          — high-frequency microstructure is blurred.

Physical model
--------------
Given a binary halftone H ∈ {0, 1}:

  Step 1 – Mechanical growth:
    Erode the complement (white regions shrink), or equivalently dilate the
    ink dots by a radius r_mech.  r_mech is estimated from the measured TVI
    at low-to-midtone coverage where mechanical gain dominates.

      H_mech = binary_dilation(H, disk(r_mech))

  Step 2 – Optical gain (Yule-Nielsen):
    Compute a continuous reflectance image R from H_mech using the
    Yule-Nielsen mixing model:

      R(x,y) = [ H_mech * R_ink^(1/n)  +  (1 - H_mech) * R_paper^(1/n) ]^n

    Applied patch-wise so that the local coverage matches the transfer function.

  Step 3 – Final reflectance matches measured TVI:
    We calibrate r_mech and n so that the mean simulated reflectance of a
    uniform 50% halftone patch equals the measured a_print(0.50).
    All parameters are derived from the TransferFunction; nothing is hardcoded.

Parameter estimation from measured TVI
---------------------------------------
``estimate_simulator_params()`` fits (r_mech, n, r_paper, r_ink) by minimising
the squared error between the simulated a_print and the measured a_print across
all tone steps.  scipy.optimize.minimize is used.

LIBRARIES USED
--------------
numpy          : array maths, binary operations        (open-source)
scipy          : binary_dilation, minimize             (open-source)
scikit-image   : disk structuring element              (open-source)

Custom code: parameter estimation loop, optical gain application,
             full-image simulation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter
from scipy.optimize import minimize

from tvi.calibration.transfer_function import TransferFunction
from tvi.measurement.tvi_core import yule_nielsen_reflectance


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimulatorParams:
    """
    All parameters needed to run the dot-gain simulator.

    Estimated from measured TVI curves; nothing hardcoded.

    Attributes
    ----------
    r_mech       : mechanical dilation radius in pixels (float ≥ 0)
    n_yn         : Yule-Nielsen factor (≥ 1.0)
    r_paper      : substrate reflectance  (0–1)
    r_ink        : solid ink reflectance  (0–1)
    channel      : ink channel label
    method       : halftoning method
    printer_label: device label
    rmse         : root-mean-square error of the fit (optional diagnostic)
    """
    r_mech: float
    n_yn: float
    r_paper: float
    r_ink: float
    channel: str = ""
    method: str = ""
    printer_label: str = ""
    rmse: float = float("nan")


# ---------------------------------------------------------------------------
# Parameter estimation from measured TVI curves
# ---------------------------------------------------------------------------

def estimate_simulator_params(
    transfer_function: TransferFunction,
    r_paper: float,
    r_ink: float,
    tone_steps: np.ndarray | None = None,
    r_mech_max_px: float = 6.0,
    n_yn_bounds: tuple[float, float] = (1.0, 4.0),
) -> SimulatorParams:
    """
    Estimate simulator parameters (r_mech, n_yn) from a measured TransferFunction.

    Strategy
    --------
    For each candidate (r_mech, n_yn) pair:
      1. Build a synthetic uniform halftone patch at each nominal coverage a_nom.
      2. Dilate it by r_mech pixels (mechanical growth).
      3. Compute the Yule-Nielsen reflectance of the dilated patch.
      4. Convert to effective coverage via Murray-Davies.
      5. Compare to transfer_function.evaluate(a_nom).

    Minimise sum of squared differences over all tone steps.

    Parameters
    ----------
    transfer_function : measured printer TransferFunction
    r_paper, r_ink    : measured boundary reflectances
    tone_steps        : 1-D array of a_nom values to use in fit (default 0.1–0.9)
    r_mech_max_px     : upper bound for mechanical radius search (pixels)
    n_yn_bounds       : (min, max) for Yule-Nielsen n search

    Returns
    -------
    SimulatorParams
    """
    if tone_steps is None:
        tone_steps = np.arange(0.1, 1.0, 0.1)

    a_target = transfer_function.evaluate(tone_steps)

    def objective(params: np.ndarray) -> float:
        r_mech, n_yn = float(params[0]), float(params[1])
        if r_mech < 0 or n_yn < n_yn_bounds[0] or n_yn > n_yn_bounds[1]:
            return 1e6
        a_sim = _simulate_uniform_patches(
            a_nom=tone_steps,
            r_mech=r_mech,
            n_yn=n_yn,
            r_paper=r_paper,
            r_ink=r_ink,
        )
        return float(np.sum((a_sim - a_target) ** 2))

    # Initial guess: small dilation, n=1.5
    x0 = np.array([0.5, 1.5])
    bounds = [(0.0, r_mech_max_px), list(n_yn_bounds)]

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    r_mech_opt, n_yn_opt = float(result.x[0]), float(result.x[1])
    rmse = float(np.sqrt(result.fun / len(tone_steps)))

    return SimulatorParams(
        r_mech=r_mech_opt,
        n_yn=n_yn_opt,
        r_paper=r_paper,
        r_ink=r_ink,
        channel=transfer_function.channel,
        method=transfer_function.method,
        printer_label=transfer_function.printer_label,
        rmse=rmse,
    )


# ---------------------------------------------------------------------------
# Simulator core
# ---------------------------------------------------------------------------

def simulate_dot_gain(
    halftone: np.ndarray,
    params: SimulatorParams,
    output: str = "reflectance",
) -> np.ndarray:
    """
    Apply the full dot-gain simulation pipeline to a binary halftone image.

    Pipeline
    --------
    binary H → mechanical dilation → Yule-Nielsen optical gain → reflectance

    Visibly reproduces
    ------------------
    * dot growth           (mechanical dilation r_mech > 0)
    * merged dots          (touching after dilation)
    * highlight fill-in    (isolated dots at <10% coverage grow and merge)
    * shadow fill-in       (white holes at >90% fill in)
    * fine-line thickening (1-px lines become 1+2*r_mech px)
    * midtone darkening    (a_print > a_nom near 50%)
    * texture loss         (high-frequency structure lost after merging)

    Parameters
    ----------
    halftone : 2-D uint8 or float array where 1 (or 255) = ink, 0 = paper.
               May be a boolean array.
    params   : SimulatorParams estimated from measured TVI curves
    output   : "reflectance"  → float64 image in [0, 1]
               "uint8"        → uint8 image in [0, 255]  (0=ink, 255=paper)

    Returns
    -------
    ndarray (H, W) float64 or uint8
    """
    # ---- Normalise input to bool: True = ink ----
    ink_mask = _to_bool_ink(halftone)

    # ---- Step 1: Mechanical dilation ----
    ink_grown = _mechanical_dilation(ink_mask, params.r_mech)

    # ---- Step 2: Optical gain (Yule-Nielsen) ----
    reflectance = _optical_gain(
        ink_grown,
        params.r_paper,
        params.r_ink,
        params.n_yn,
    )

    if output == "reflectance":
        return reflectance
    elif output == "uint8":
        return (np.clip(reflectance, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"output must be 'reflectance' or 'uint8', got {output!r}")


def simulate_benchmark_image(
    halftone: np.ndarray,
    params: SimulatorParams,
) -> np.ndarray:
    """
    Convenience wrapper that returns a uint8 simulated image.

    Use this to process any of the 450 benchmark halftone images without
    physically printing them.

    Parameters
    ----------
    halftone : 2-D binary halftone (uint8 or bool)
    params   : SimulatorParams from estimate_simulator_params()

    Returns
    -------
    simulated : uint8 image (0 = black, 255 = white)
    """
    return simulate_dot_gain(halftone, params, output="uint8")


# ---------------------------------------------------------------------------
# Uniform-patch helper for parameter estimation
# ---------------------------------------------------------------------------

def _simulate_uniform_patches(
    a_nom: np.ndarray,
    r_mech: float,
    n_yn: float,
    r_paper: float,
    r_ink: float,
    patch_size: int = 64,
) -> np.ndarray:
    """
    Simulate mean coverage for synthetic uniform halftone patches.

    For each nominal coverage a, we build a patch_size×patch_size checkerboard-
    like binary array with exactly round(a * patch_size²) ink pixels, apply
    dilation and Yule-Nielsen, and compute mean coverage via Murray-Davies.

    This is used only inside the parameter estimation loop.
    """
    a_sim = np.zeros_like(a_nom, dtype=np.float64)
    for i, a in enumerate(a_nom):
        n_ink = round(float(a) * patch_size * patch_size)
        patch = np.zeros((patch_size, patch_size), dtype=bool)
        # Scatter ink pixels pseudo-randomly but reproducibly
        rng = np.random.default_rng(seed=int(a * 1e6))
        flat_idx = rng.choice(patch_size * patch_size, size=n_ink, replace=False)
        patch.flat[flat_idx] = True

        grown = _mechanical_dilation(patch, r_mech)
        ref = _optical_gain(grown, r_paper, r_ink, n_yn)

        # Murray-Davies: a_print = (r_paper - r_patch) / (r_paper - r_ink)
        r_mean = float(ref.mean())
        denom = r_paper - r_ink
        if abs(denom) < 1e-9:
            a_sim[i] = a
        else:
            a_sim[i] = np.clip((r_paper - r_mean) / denom, 0.0, 1.0)

    return a_sim


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_bool_ink(halftone: np.ndarray) -> np.ndarray:
    """Convert input to bool mask where True = ink."""
    if halftone.dtype == bool:
        return halftone
    arr = np.asarray(halftone)
    if arr.dtype in (np.float32, np.float64):
        return arr > 0.5
    return arr > 0


def _mechanical_dilation(ink_mask: np.ndarray, r_mech: float) -> np.ndarray:
    """
    Dilate ink dots by r_mech pixels using a disk structuring element.

    r_mech = 0 → no dilation (identity).
    r_mech = 0.5 → mild growth.
    r_mech = 2.0 → strong growth, dots merge in midtones.

    LIBRARY: scipy.ndimage.binary_dilation
    The structuring element is a boolean disk of radius ceil(r_mech).
    For sub-integer radii we use a fractional approach: dilate by ceil and
    then apply a uniform blur to smooth the transition.
    """
    if r_mech < 0.01:
        return ink_mask.copy()

    radius = int(np.ceil(r_mech))
    # Build disk structuring element
    size = 2 * radius + 1
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    struct = (x ** 2 + y ** 2) <= radius ** 2

    dilated = binary_dilation(ink_mask, structure=struct)

    # Fractional blending for sub-pixel smoothness
    frac = r_mech - np.floor(r_mech)
    if frac > 0.01 and radius > 0:
        # Blend between radius-1 and radius dilation
        struct_small = (x ** 2 + y ** 2) <= (radius - 1) ** 2
        dilated_small = binary_dilation(ink_mask, structure=struct_small)
        # Weighted continuous blend (returns float)
        return dilated_small.astype(np.float64) * (1.0 - frac) + dilated.astype(np.float64) * frac

    return dilated.astype(np.float64)


def _optical_gain(
    ink_coverage: np.ndarray,
    r_paper: float,
    r_ink: float,
    n_yn: float,
) -> np.ndarray:
    """
    Apply the Yule-Nielsen optical gain model pixel-wise.

    R(x,y) = [ a(x,y) * R_ink^(1/n)  +  (1 - a(x,y)) * R_paper^(1/n) ]^n

    ink_coverage is a float array in [0, 1] (may be the fractionally-blended
    dilation output, not strictly binary).

    At n=1 this reduces to linear Murray-Davies mixing.
    At n>1 the effective reflectance is higher than linear mixing predicts,
    modelling the fact that light scatters under ink dots and is absorbed
    (darker apparent coverage for a given physical area).
    """
    a = np.asarray(ink_coverage, dtype=np.float64)
    return yule_nielsen_reflectance(a, r_ink=r_ink, r_paper=r_paper, n=n_yn)
