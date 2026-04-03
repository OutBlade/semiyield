"""
Statistical Process Control (SPC) control chart module.

Implements Shewhart control charts and supplementary rules following:
  AIAG Statistical Process Control Reference Manual, 2nd Edition, 2005.
  Montgomery, D.C., "Introduction to Statistical Quality Control," 7th Ed., Wiley.

Supported chart types:
  Xbar-R   : subgroup means and ranges
  Xbar-S   : subgroup means and standard deviations
  I-MR     : individual measurements and moving range
  EWMA     : Exponentially Weighted Moving Average
  CUSUM    : Cumulative Sum

Process capability indices (AIAG SPC Manual, Section IV):
  Cp   = (USL - LSL) / (6 * sigma_within)
  Cpk  = min((USL - mu) / (3 * sigma_within), (mu - LSL) / (3 * sigma_within))
  Pp   = (USL - LSL) / (6 * sigma_overall)
  Ppk  = min((USL - mu) / (3 * sigma_overall), (mu - LSL) / (3 * sigma_overall))
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np

# ------------------------------------------------------------------ #
# Control chart constants (from AIAG SPC tables / d2 factors)         #
# ------------------------------------------------------------------ #

# d2 values for estimating sigma_within from the average range
# Index is subgroup size n (n=2..10)
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
       6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

# c4 for estimating sigma_within from the average standard deviation
_C4 = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400,
       6: 0.9515, 7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727}

# A2 control limit factor for Xbar-R chart
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577,
       6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}

# D3, D4 for R chart control limits
_D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076,
       8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004,
       7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


class ControlChart:
    """Shewhart-family and advanced control charts.

    Supports Xbar-R, Xbar-S, I-MR, EWMA, and CUSUM charts.

    Parameters
    ----------
    chart_type : str
        One of 'XbarR', 'XbarS', 'IMR', 'EWMA', 'CUSUM'.
    subgroup_size : int
        Number of observations per subgroup (only used for XbarR and XbarS).
        Must be 2-10.
    ewma_lambda : float
        Smoothing parameter for EWMA chart (0 < lambda <= 1).
    cusum_k : float
        Reference value k for CUSUM (typically 0.5 sigma).
    cusum_h : float
        Decision interval h for CUSUM (typically 4-5 sigma).
    """

    def __init__(
        self,
        chart_type: Literal["XbarR", "XbarS", "IMR", "EWMA", "CUSUM"] = "IMR",
        subgroup_size: int = 5,
        ewma_lambda: float = 0.2,
        cusum_k: float = 0.5,
        cusum_h: float = 4.0,
    ) -> None:
        valid_types = ("XbarR", "XbarS", "IMR", "EWMA", "CUSUM")
        if chart_type not in valid_types:
            raise ValueError(f"chart_type must be one of {valid_types}.")
        self.chart_type = chart_type
        self.subgroup_size = subgroup_size
        self.ewma_lambda = ewma_lambda
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

        self.cl: float = 0.0
        self.ucl: float = 0.0
        self.lcl: float = 0.0
        self.sigma: float = 0.0
        self._phase1_data: np.ndarray = np.array([])
        self._fitted = False

        # Running state for EWMA / CUSUM
        self._ewma_last: float = 0.0
        self._cusum_plus: float = 0.0
        self._cusum_minus: float = 0.0

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def fit(self, data: np.ndarray) -> "ControlChart":
        """Compute control limits from Phase I data.

        Parameters
        ----------
        data : numpy.ndarray
            For IMR/EWMA/CUSUM: 1D array of individual measurements.
            For XbarR/XbarS: 1D array of individual measurements (will be
            automatically grouped into subgroups of size *subgroup_size*).

        Returns
        -------
        ControlChart
            self
        """
        data = np.asarray(data, dtype=float)
        self._phase1_data = data

        if self.chart_type in ("XbarR", "XbarS"):
            self._fit_subgroup(data)
        elif self.chart_type == "IMR":
            self._fit_imr(data)
        elif self.chart_type == "EWMA":
            self._fit_ewma(data)
        elif self.chart_type == "CUSUM":
            self._fit_cusum(data)

        self._fitted = True
        return self

    def update(self, new_point: float) -> bool:
        """Check whether a new observation is in statistical control.

        Parameters
        ----------
        new_point : float
            New measurement value.

        Returns
        -------
        bool
            True if the point is in control, False if out of control.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before update().")

        if self.chart_type in ("XbarR", "XbarS", "IMR"):
            return self.lcl <= new_point <= self.ucl

        if self.chart_type == "EWMA":
            self._ewma_last = (
                self.ewma_lambda * new_point
                + (1 - self.ewma_lambda) * self._ewma_last
            )
            return self.lcl <= self._ewma_last <= self.ucl

        if self.chart_type == "CUSUM":
            mu = self.cl
            k = self.cusum_k * self.sigma
            h = self.cusum_h * self.sigma
            self._cusum_plus = max(0.0, self._cusum_plus + (new_point - mu) - k)
            self._cusum_minus = max(0.0, self._cusum_minus - (new_point - mu) - k)
            return self._cusum_plus < h and self._cusum_minus < h

        return True

    def chart_data(self) -> dict:
        """Return all chart values for plotting.

        Returns
        -------
        dict
            Keys: 'data', 'cl', 'ucl', 'lcl', 'sigma', 'chart_type'.
        """
        return {
            "data": self._phase1_data.tolist(),
            "cl": self.cl,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "sigma": self.sigma,
            "chart_type": self.chart_type,
        }

    # ---------------------------------------------------------------- #
    # Private fitting routines                                           #
    # ---------------------------------------------------------------- #

    def _fit_subgroup(self, data: np.ndarray) -> None:
        n = self.subgroup_size
        if n < 2 or n > 10:
            raise ValueError("subgroup_size must be between 2 and 10.")

        # Reshape into subgroups (drop incomplete tail)
        n_complete = (len(data) // n) * n
        groups = data[:n_complete].reshape(-1, n)
        x_bar_bar = groups.mean(axis=1).mean()
        self.cl = x_bar_bar

        if self.chart_type == "XbarR":
            ranges = groups.max(axis=1) - groups.min(axis=1)
            R_bar = ranges.mean()
            self.sigma = R_bar / _D2[n]
            self.ucl = x_bar_bar + _A2[n] * R_bar
            self.lcl = x_bar_bar - _A2[n] * R_bar
        else:  # XbarS
            s_bar = groups.std(axis=1, ddof=1).mean()
            self.sigma = s_bar / _C4[n]
            A3 = 3.0 / (_C4[n] * math.sqrt(n))
            self.ucl = x_bar_bar + A3 * s_bar
            self.lcl = x_bar_bar - A3 * s_bar

    def _fit_imr(self, data: np.ndarray) -> None:
        self.cl = float(data.mean())
        moving_ranges = np.abs(np.diff(data))
        MR_bar = moving_ranges.mean()
        self.sigma = MR_bar / _D2[2]  # d2 for n=2
        self.ucl = self.cl + 3.0 * self.sigma
        self.lcl = self.cl - 3.0 * self.sigma

    def _fit_ewma(self, data: np.ndarray) -> None:
        self.cl = float(data.mean())
        self.sigma = float(data.std(ddof=1))
        lam = self.ewma_lambda
        # Asymptotic EWMA control limits
        sigma_ewma = self.sigma * math.sqrt(lam / (2.0 - lam))
        self.ucl = self.cl + 3.0 * sigma_ewma
        self.lcl = self.cl - 3.0 * sigma_ewma
        self._ewma_last = self.cl

    def _fit_cusum(self, data: np.ndarray) -> None:
        self.cl = float(data.mean())
        self.sigma = float(data.std(ddof=1))
        # CUSUM limits are managed in update(); set UCL for plotting reference
        self.ucl = self.cl + self.cusum_h * self.sigma
        self.lcl = self.cl - self.cusum_h * self.sigma
        self._cusum_plus = 0.0
        self._cusum_minus = 0.0


# ------------------------------------------------------------------ #
# Western Electric Rules                                               #
# ------------------------------------------------------------------ #

def western_electric_violations(
    data: np.ndarray,
    ucl: float,
    lcl: float,
    cl: float,
) -> list[tuple[int, int, str]]:
    """Detect Western Electric rule violations in a control chart data series.

    All eight rules from the Western Electric Statistical Quality Control
    Handbook (1956) are implemented.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of chart values (e.g., subgroup means or individual measurements).
    ucl : float
        Upper control limit (3-sigma from CL).
    lcl : float
        Lower control limit (3-sigma from CL, i.e., CL - 3*sigma).
    cl : float
        Centre line (process mean).

    Returns
    -------
    list of (index, rule_number, description)
        Each tuple identifies a violation: the index of the last point in the
        violating run, the rule number (1-8), and a description string.
    """
    data = np.asarray(data, dtype=float)
    sigma = (ucl - cl) / 3.0
    violations: list[tuple[int, int, str]] = []

    for i, x in enumerate(data):
        # Rule 1: One point beyond 3-sigma
        if x > ucl or x < lcl:
            violations.append((
                i, 1,
                f"Rule 1: point {i} ({x:.4f}) beyond 3-sigma control limit"
            ))

    # Rule 2: Two of three consecutive points beyond 2-sigma (same side)
    for i in range(2, len(data)):
        window = data[i - 2: i + 1]
        above = np.sum(window > cl + 2 * sigma)
        below = np.sum(window < cl - 2 * sigma)
        if above >= 2:
            violations.append((i, 2, f"Rule 2: 2/3 consecutive points above +2-sigma near index {i}"))
        if below >= 2:
            violations.append((i, 2, f"Rule 2: 2/3 consecutive points below -2-sigma near index {i}"))

    # Rule 3: Four of five consecutive points beyond 1-sigma (same side)
    for i in range(4, len(data)):
        window = data[i - 4: i + 1]
        above = np.sum(window > cl + sigma)
        below = np.sum(window < cl - sigma)
        if above >= 4:
            violations.append((i, 3, f"Rule 3: 4/5 consecutive points above +1-sigma near index {i}"))
        if below >= 4:
            violations.append((i, 3, f"Rule 3: 4/5 consecutive points below -1-sigma near index {i}"))

    # Rule 4: Eight consecutive points on same side of CL
    for i in range(7, len(data)):
        window = data[i - 7: i + 1]
        if np.all(window > cl):
            violations.append((i, 4, f"Rule 4: 8 consecutive points above CL, ending at index {i}"))
        if np.all(window < cl):
            violations.append((i, 4, f"Rule 4: 8 consecutive points below CL, ending at index {i}"))

    # Rule 5: Six consecutive points trending strictly up or down
    for i in range(5, len(data)):
        window = data[i - 5: i + 1]
        diffs = np.diff(window)
        if np.all(diffs > 0):
            violations.append((i, 5, f"Rule 5: 6 consecutive points trending upward, ending at index {i}"))
        if np.all(diffs < 0):
            violations.append((i, 5, f"Rule 5: 6 consecutive points trending downward, ending at index {i}"))

    # Rule 6: Fifteen consecutive points within 1-sigma of CL (stratification)
    for i in range(14, len(data)):
        window = data[i - 14: i + 1]
        if np.all(np.abs(window - cl) < sigma):
            violations.append((i, 6, f"Rule 6: 15 consecutive points within 1-sigma, ending at index {i}"))

    # Rule 7: Fourteen consecutive points alternating up and down
    for i in range(13, len(data)):
        window = data[i - 13: i + 1]
        diffs = np.diff(window)
        signs = np.sign(diffs)
        alternating = np.all(signs[:-1] * signs[1:] < 0)
        if alternating:
            violations.append((i, 7, f"Rule 7: 14 consecutive alternating points, ending at index {i}"))

    # Rule 8: Eight consecutive points beyond 1-sigma (both sides, mixture)
    for i in range(7, len(data)):
        window = data[i - 7: i + 1]
        beyond_1sigma = np.abs(window - cl) > sigma
        if np.all(beyond_1sigma):
            violations.append((i, 8, f"Rule 8: 8 consecutive points beyond 1-sigma (both sides), ending at index {i}"))

    return violations


# ------------------------------------------------------------------ #
# Process Capability                                                   #
# ------------------------------------------------------------------ #

def process_capability(
    data: np.ndarray,
    usl: float,
    lsl: float,
    subgroup_size: int = 1,
) -> dict[str, float]:
    """Compute process capability and performance indices.

    Definitions follow AIAG SPC Reference Manual, 2nd Edition (2005):

      sigma_within  = R_bar / d2   (from moving ranges for n=1)
      sigma_overall = std(data, ddof=1)

      Cp   = (USL - LSL) / (6 * sigma_within)
      Cpk  = min((USL - mean) / (3*sigma_within), (mean - LSL) / (3*sigma_within))
      Pp   = (USL - LSL) / (6 * sigma_overall)
      Ppk  = min((USL - mean) / (3*sigma_overall), (mean - LSL) / (3*sigma_overall))

    Parameters
    ----------
    data : numpy.ndarray
        Measurement data (individual values or subgroup means).
    usl : float
        Upper specification limit.
    lsl : float
        Lower specification limit.
    subgroup_size : int
        Subgroup size for within-subgroup sigma estimation.
        Use 1 for individual measurements (I-MR chart context).

    Returns
    -------
    dict
        Keys: 'Cp', 'Cpk', 'Pp', 'Ppk', 'sigma_within', 'sigma_overall',
              'mean', 'sigma_level'.
    """
    data = np.asarray(data, dtype=float)
    mean = float(data.mean())
    sigma_overall = float(data.std(ddof=1))

    # Within-subgroup sigma
    if subgroup_size <= 1:
        moving_ranges = np.abs(np.diff(data))
        MR_bar = moving_ranges.mean()
        sigma_within = MR_bar / _D2[2]
    else:
        n = subgroup_size
        n_complete = (len(data) // n) * n
        groups = data[:n_complete].reshape(-1, n)
        ranges = groups.max(axis=1) - groups.min(axis=1)
        R_bar = ranges.mean()
        sigma_within = R_bar / _D2.get(n, _D2[10])

    spec_width = usl - lsl

    def safe_div(a: float, b: float) -> float:
        return a / b if b > 1e-15 else float("inf")

    Cp = safe_div(spec_width, 6.0 * sigma_within)
    Cpk = min(
        safe_div(usl - mean, 3.0 * sigma_within),
        safe_div(mean - lsl, 3.0 * sigma_within),
    )
    Pp = safe_div(spec_width, 6.0 * sigma_overall)
    Ppk = min(
        safe_div(usl - mean, 3.0 * sigma_overall),
        safe_div(mean - lsl, 3.0 * sigma_overall),
    )

    # Sigma level: distance from mean to nearest spec limit in sigma units
    sigma_level = min(
        safe_div(usl - mean, sigma_within),
        safe_div(mean - lsl, sigma_within),
    )

    return {
        "Cp": Cp,
        "Cpk": Cpk,
        "Pp": Pp,
        "Ppk": Ppk,
        "sigma_within": sigma_within,
        "sigma_overall": sigma_overall,
        "mean": mean,
        "sigma_level": sigma_level,
    }
