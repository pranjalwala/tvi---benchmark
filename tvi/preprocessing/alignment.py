"""
tvi/preprocessing/alignment.py
-------------------------------
Geometric alignment and patch extraction for scanned target sheets.

Pipeline
--------
1. detect_fiducials()  — find the four corner fiducial marks in the scan.
2. align_sheet()       — fit a similarity transform; warp scan to reference grid.
3. extract_patch()     — crop an interior region with configurable erosion margin.

LIBRARIES USED
--------------
numpy          : array operations                         (open-source)
scikit-image   : threshold_otsu, label, regionprops       (open-source)
               : warp, SimilarityTransform                (open-source)
opencv         : findHomography (fallback), connectedComponents (open-source)
scipy          : ndimage.label                            (open-source)

Custom code: fiducial detection heuristic (largest N dark blobs near corners),
             similarity-transform fitting from 4-point correspondences.
"""

from __future__ import annotations

import numpy as np
from skimage import transform as skt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_fiducials(
    gray: np.ndarray,
    n_marks: int = 4,
    dark_fraction: float = 0.15,
) -> np.ndarray:
    """
    Detect fiducial mark centroids in a grayscale scan.

    Strategy: threshold → connected-component labelling → select the N largest
    dark components that lie closest to the four image corners.

    Parameters
    ----------
    gray : ndarray (H, W)  float in [0, 1] or uint8/uint16
    n_marks : int   expected number of fiducials (≥ 4)
    dark_fraction : float
        Components with mean intensity below (dark_fraction * max) are ink.

    Returns
    -------
    centroids : ndarray (n_marks, 2)  [row, col] in pixel coordinates.
    """
    # Normalise to float — handle uint8, uint16, and float arrays from real scanners
    if gray.dtype != np.float64:
        if np.issubdtype(gray.dtype, np.integer):
            gray = gray.astype(np.float64) / np.iinfo(gray.dtype).max
        else:
            # Float TIFF (e.g. from some flatbed drivers): assume [0,1] or [0,255]
            gray = gray.astype(np.float64)
            if gray.max() > 1.0:
                gray = gray / gray.max()

    thresh = threshold_otsu(gray)
    binary_ink = gray < thresh          # ink pixels = True

    labeled = label(binary_ink)
    props = regionprops(labeled)

    if not props:
        raise RuntimeError("No dark components found — check scan or fiducial mark size.")

    # Sort by area descending, keep top 4*n_marks candidates
    props = sorted(props, key=lambda p: p.area, reverse=True)[: 4 * n_marks]

    # Assign each candidate to the nearest image corner
    H, W = gray.shape
    corners = np.array([[0, 0], [0, W], [H, 0], [H, W]], dtype=float)
    centroids = np.array([p.centroid for p in props])  # (N, 2) [row, col]

    chosen: list[np.ndarray] = []
    used = set()
    for corner in corners:
        dists = np.linalg.norm(centroids - corner, axis=1)
        for idx in np.argsort(dists):
            if idx not in used:
                chosen.append(centroids[idx])
                used.add(idx)
                break

    if len(chosen) < 4:
        raise RuntimeError(
            f"Could only locate {len(chosen)} fiducial marks; need at least 4."
        )

    return np.array(chosen[:n_marks])  # (n_marks, 2)


def align_sheet(
    scan: np.ndarray,
    scan_fiducials: np.ndarray,
    ref_fiducials: np.ndarray,
) -> np.ndarray:
    """
    Warp `scan` so its fiducial marks align with `ref_fiducials`.

    Uses a SimilarityTransform (rotation + uniform scale + translation).
    This corrects global skew without introducing anisotropic distortion.

    Parameters
    ----------
    scan : ndarray (H, W) or (H, W, C)
    scan_fiducials : ndarray (N, 2) [row, col]
    ref_fiducials  : ndarray (N, 2) [row, col]

    Returns
    -------
    warped : ndarray, same shape and dtype as scan.

    LIBRARY
    -------
    skimage.transform.SimilarityTransform + warp   (open-source)
    """
    tform = skt.SimilarityTransform()
    # estimate() expects (col, row) i.e. (x, y)
    src = scan_fiducials[:, ::-1].astype(float)   # [col, row]
    dst = ref_fiducials[:, ::-1].astype(float)
    ok = tform.estimate(src, dst)
    if not ok:
        raise RuntimeError("SimilarityTransform estimation failed.")

    if scan.ndim == 2:
        return skt.warp(
            scan.astype(np.float64),
            tform.inverse,
            order=3,
            preserve_range=True,
        ).astype(scan.dtype)

    # Multi-channel: warp each channel separately
    warped = np.zeros_like(scan)
    for c in range(scan.shape[2]):
        warped[..., c] = skt.warp(
            scan[..., c].astype(np.float64),
            tform.inverse,
            order=3,
            preserve_range=True,
        ).astype(scan.dtype)
    return warped


def extract_patch(
    image: np.ndarray,
    row: int,
    col: int,
    height: int,
    width: int,
    erosion_margin: float = 0.10,
) -> np.ndarray:
    """
    Extract a patch interior with an erosion margin.

    Parameters
    ----------
    image : ndarray (H, W) or (H, W, C)
    row, col : top-left corner in pixels
    height, width : patch dimensions in pixels
    erosion_margin : fraction of each side to erode (0.10 = 10%)

    Returns
    -------
    patch : ndarray — interior region only.
    """
    margin_r = max(1, round(height * erosion_margin))
    margin_c = max(1, round(width * erosion_margin))
    r0 = row + margin_r
    r1 = row + height - margin_r
    c0 = col + margin_c
    c1 = col + width - margin_c
    r1 = max(r0 + 1, r1)
    c1 = max(c0 + 1, c1)
    return image[r0:r1, c0:c1]
