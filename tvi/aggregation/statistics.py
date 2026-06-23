"""
tvi/aggregation/statistics.py
-------------------------------
Statistical aggregation of TVI measurements across replicates.

Benchmark protocol (from pseudocode):
    n_obs = n_sheets × n_scans  (e.g. 3 × 3 = 9)
    mean_TVI = mean over replicates
    CI_TVI   = t(0.975, df=n_obs-1) × std(TVI) / sqrt(n_obs)

This module implements:
  AggregatedResult — stores mean TVI curve + CI per (printer, method, channel)
  aggregate_replicates() — collapse a ReplicateSet into AggregatedResult
  aggregate_tvi_curve()  — aggregate raw TVI curves directly
  build_metric_table()   — produce the benchmark Table 3 primary scalar
  summary_pivot()        — pivot for method × channel comparison
  save_csv() / load_csv() — CSV serialisation

LIBRARIES USED
--------------
numpy   : statistics                                    (open-source)
scipy   : t-distribution critical value                 (open-source)
pandas  : DataFrame, pivot, CSV I/O                     (open-source)

Custom code: aggregation logic, CI computation, metric table builder.

Integration
-----------
from tvi.measurement      import ReplicateSet, TVICurve
from tvi.aggregation      import aggregate_replicates, build_metric_table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from tvi.measurement.tvi_core import TVICurve


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AggregatedResult:
    """
    Aggregated TVI curve for one (printer, method, channel).

    Attributes
    ----------
    channel       : str
    method        : str
    printer_label : str
    a_nom         : 1-D array  — nominal coverage steps
    mean_tvi      : 1-D array  — mean TVI(a_nom) over replicates
    ci_tvi        : 1-D array  — half-width of 95% CI at each step
    std_tvi       : 1-D array  — standard deviation over replicates
    n_obs         : int        — number of observations used
    tvi_at_50     : float      — primary scalar: mean TVI at a_nom=0.50
    ci_at_50      : float      — 95% CI half-width for TVI at a_nom=0.50
    raw_tvi_matrix: (n_obs, n_steps) raw TVI values (optional)
    """
    channel: str
    method: str
    printer_label: str
    a_nom: np.ndarray
    mean_tvi: np.ndarray
    ci_tvi: np.ndarray
    std_tvi: np.ndarray
    n_obs: int
    tvi_at_50: float = 0.0
    ci_at_50: float = 0.0
    raw_tvi_matrix: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        idx = int(np.argmin(np.abs(self.a_nom - 0.50)))
        self.tvi_at_50 = float(self.mean_tvi[idx])
        self.ci_at_50 = float(self.ci_tvi[idx])


# ---------------------------------------------------------------------------
# Core aggregation functions
# ---------------------------------------------------------------------------

def aggregate_tvi_curve(
    curves: list[TVICurve],
    ci_level: float = 0.95,
) -> AggregatedResult:
    """
    Aggregate a list of TVICurve objects into a single AggregatedResult.

    All curves must share the same a_nom grid.  The function stacks the tvi
    arrays into an (n_obs, n_steps) matrix and computes mean and CI.

    Parameters
    ----------
    curves    : list of TVICurve (one per replicate)
    ci_level  : confidence level (default 0.95 → two-sided 95% CI)

    Returns
    -------
    AggregatedResult
    """
    if not curves:
        raise ValueError("curves list is empty.")

    # Validate common a_nom grid
    a_nom_ref = curves[0].a_nom
    for c in curves[1:]:
        if not np.allclose(c.a_nom, a_nom_ref, atol=1e-6):
            raise ValueError(
                "All TVI curves must share the same a_nom grid for aggregation."
            )

    tvi_matrix = np.vstack([c.tvi for c in curves])   # (n_obs, n_steps)
    n_obs, n_steps = tvi_matrix.shape
    df = max(n_obs - 1, 1)
    alpha = 1.0 - ci_level
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df=df))

    mean_tvi = tvi_matrix.mean(axis=0)
    std_tvi = tvi_matrix.std(axis=0, ddof=1) if n_obs > 1 else np.zeros(n_steps)
    ci_tvi = t_crit * std_tvi / np.sqrt(n_obs)

    ref = curves[0]
    return AggregatedResult(
        channel=ref.channel,
        method=ref.method,
        printer_label=ref.printer_label,
        a_nom=a_nom_ref.copy(),
        mean_tvi=mean_tvi,
        ci_tvi=ci_tvi,
        std_tvi=std_tvi,
        n_obs=n_obs,
        raw_tvi_matrix=tvi_matrix,
    )


def aggregate_replicates(
    replicate_set,                  # ReplicateSet (avoid circular import with string type)
    ci_level: float = 0.95,
) -> AggregatedResult:
    """
    Aggregate a ReplicateSet (3 sheets × 3 scans) into an AggregatedResult.

    Parameters
    ----------
    replicate_set : tvi.measurement.ReplicateSet
    ci_level      : confidence level

    Returns
    -------
    AggregatedResult
    """
    curves = replicate_set.tvi_curves()
    result = aggregate_tvi_curve(curves, ci_level=ci_level)
    return result


# ---------------------------------------------------------------------------
# Metric table builder (benchmark Table 3)
# ---------------------------------------------------------------------------

def build_metric_table(
    results: list[AggregatedResult],
) -> pd.DataFrame:
    """
    Build the benchmark printing-constraint metric table (Table 3).

    Each row corresponds to one (method, channel) combination and reports
    the TVI primary scalar (TVI at 50% nominal) with its 95% CI.

    Parameters
    ----------
    results : list of AggregatedResult

    Returns
    -------
    pd.DataFrame with columns:
        method, channel, printer_label,
        tvi_at_50, ci_at_50,
        tvi_min, tvi_max, n_obs
    """
    rows = []
    for r in results:
        rows.append({
            "method": r.method,
            "channel": r.channel,
            "printer_label": r.printer_label,
            "tvi_at_50_pp": round(r.tvi_at_50 * 100, 2),   # convert to pp
            "ci_at_50_pp": round(r.ci_at_50 * 100, 2),
            "tvi_min_pp": round(float(r.mean_tvi.min()) * 100, 2),
            "tvi_max_pp": round(float(r.mean_tvi.max()) * 100, 2),
            "n_obs": r.n_obs,
        })
    return pd.DataFrame(rows)


def build_full_curve_table(
    results: list[AggregatedResult],
) -> pd.DataFrame:
    """
    Build a long-format DataFrame with the full TVI curve per method/channel.

    Columns: method, channel, printer_label, a_nom, mean_tvi_pp, ci_tvi_pp, std_tvi_pp
    """
    rows = []
    for r in results:
        for a, m, ci, s in zip(r.a_nom, r.mean_tvi, r.ci_tvi, r.std_tvi):
            rows.append({
                "method": r.method,
                "channel": r.channel,
                "printer_label": r.printer_label,
                "a_nom": round(float(a), 3),
                "mean_tvi_pp": round(float(m) * 100, 3),
                "ci_tvi_pp": round(float(ci) * 100, 3),
                "std_tvi_pp": round(float(s) * 100, 3),
                "n_obs": r.n_obs,
            })
    return pd.DataFrame(rows)


def summary_pivot(
    metric_table: pd.DataFrame,
    value_col: str = "tvi_at_50_pp",
) -> pd.DataFrame:
    """
    Pivot the metric table to a method × channel comparison matrix.

    Parameters
    ----------
    metric_table : output of build_metric_table()
    value_col    : which column to use as cell values

    Returns
    -------
    pd.DataFrame with index=method, columns=channel
    """
    return metric_table.pivot_table(
        index="method",
        columns="channel",
        values=value_col,
        aggfunc="mean",
    )


# ---------------------------------------------------------------------------
# CSV serialisation
# ---------------------------------------------------------------------------

def save_csv(
    results: list[AggregatedResult],
    path: str | Path,
    kind: str = "metric_table",
) -> None:
    """
    Save aggregated results to CSV.

    Parameters
    ----------
    results : list of AggregatedResult
    path    : output CSV file path
    kind    : "metric_table" (primary scalars) or "full_curves"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "metric_table":
        df = build_metric_table(results)
    elif kind == "full_curves":
        df = build_full_curve_table(results)
    else:
        raise ValueError(f"kind must be 'metric_table' or 'full_curves', got {kind!r}")

    df.to_csv(path, index=False)


def load_csv(
    path: str | Path,
    kind: str = "metric_table",
) -> pd.DataFrame:
    """
    Load a CSV saved by save_csv().

    Returns a raw DataFrame; use build_metric_table() for typed aggregation.

    Parameters
    ----------
    path : CSV file path
    kind : "metric_table" or "full_curves" (informational only)

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(Path(path))
