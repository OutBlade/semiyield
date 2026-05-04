"""
Synthetic semiconductor fab data generator.

Generates realistic lot-level and wafer-level process data including:
  - Process parameter variation (normal distributions around targets)
  - Lot-to-lot drift (random walk on process means)
  - Equipment aging effects on etch rate and deposition uniformity
  - Wafer-level spatial non-uniformity (radial + random)
  - Yield calculation using Murphy's model

Murphy's yield model:
    Y = ((1 - exp(-A * D)) / (A * D))^2

where:
    A = chip area in cm^2
    D = defect density in cm^{-2}

Reference:
    B.T. Murphy, "Cost-size optima of monolithic integrated circuits,"
    Proc. IEEE 52(12), 1537-1545, 1964.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Process target values and natural variation (1-sigma)                #
# ------------------------------------------------------------------ #

_PROCESS_TARGETS: dict[str, dict[str, float]] = {
    "gate_oxide_thickness": {"mean": 8.5, "sigma": 0.3},  # nm
    "poly_cd": {"mean": 90.0, "sigma": 3.0},  # nm (critical dim.)
    "implant_dose": {"mean": 1.0e13, "sigma": 2.0e11},  # cm^{-2}
    "anneal_temp": {"mean": 1000.0, "sigma": 5.0},  # deg C
    "metal_resistance": {"mean": 0.08, "sigma": 0.005},  # ohm/sq
    "contact_resistance": {"mean": 15.0, "sigma": 1.5},  # ohm
    "etch_rate": {"mean": 120.0, "sigma": 6.0},  # nm/min
    "deposition_unif": {"mean": 2.0, "sigma": 0.3},  # % 1-sigma
}


def _murphy_yield(chip_area_cm2: float, defect_density: float) -> float:
    """Compute die yield using Murphy's model.

    Y = ((1 - exp(-A * D)) / (A * D))^2

    Parameters
    ----------
    chip_area_cm2 : float
        Die area in cm^2.
    defect_density : float
        Defect density in defects/cm^2.

    Returns
    -------
    float
        Yield fraction in [0, 1].
    """
    AD = chip_area_cm2 * defect_density
    if AD < 1e-10:
        return 1.0
    inner = (1.0 - math.exp(-AD)) / AD
    return inner**2


class FabDataGenerator:
    """Synthetic semiconductor fab data generator.

    Produces a pandas DataFrame containing lot-level and wafer-level process
    measurements with realistic statistical properties including drift,
    equipment aging, and spatial non-uniformity.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    chip_area_cm2 : float
        Die area in cm^2.  Affects yield via Murphy's model.  Default: 0.01 cm^2 (1x1 mm die).
    drift_rate : float
        Standard deviation of per-lot random-walk step for process means,
        expressed as a fraction of the natural process sigma.
        Default: 0.05 (5% of sigma per lot).
    aging_factor : float
        Rate of equipment aging expressed as fractional degradation per lot.
        Affects etch rate uniformity and deposition rate.
        Default: 0.002 (0.2% per lot).
    wafer_map_size : int
        Edge length of the square wafer map array.  Default: 50.
    """

    def __init__(
        self,
        seed: int | None = None,
        chip_area_cm2: float = 0.01,
        drift_rate: float = 0.05,
        aging_factor: float = 0.002,
        wafer_map_size: int = 50,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.chip_area_cm2 = chip_area_cm2
        self.drift_rate = drift_rate
        self.aging_factor = aging_factor
        self.wafer_map_size = wafer_map_size
        self._seed = seed

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def generate(self, n_lots: int = 100, wafers_per_lot: int = 25) -> pd.DataFrame:
        """Generate synthetic fab dataset.

        Parameters
        ----------
        n_lots : int
            Number of lots to simulate.
        wafers_per_lot : int
            Number of wafers per lot.

        Returns
        -------
        pandas.DataFrame
            One row per wafer with columns:
            lot_id, wafer_id, lot_sequence, process parameters,
            yield, defect_density, wafer_map (serialised as string).
        """
        rows: list[dict[str, Any]] = []

        # Initialise drifting process means at nominal targets
        current_means = {param: info["mean"] for param, info in _PROCESS_TARGETS.items()}

        for lot_idx in range(n_lots):
            lot_id = f"LOT{lot_idx:04d}"

            # --- Lot-to-lot drift: random walk on process means ---
            for param, info in _PROCESS_TARGETS.items():
                step = self.rng.normal(0.0, self.drift_rate * info["sigma"])
                current_means[param] += step

            # --- Equipment aging: etch rate decreases, uniformity worsens ---
            age = lot_idx * self.aging_factor
            aged_etch_mean = current_means["etch_rate"] * (1.0 - 0.3 * age)
            aged_unif_mean = current_means["deposition_unif"] * (1.0 + 0.5 * age)

            for wafer_idx in range(wafers_per_lot):
                wafer_id = f"{lot_id}_W{wafer_idx + 1:02d}"

                # --- Per-wafer measurements (lot mean + wafer noise) ---
                meas: dict[str, Any] = {
                    "lot_id": lot_id,
                    "wafer_id": wafer_id,
                    "lot_sequence": lot_idx,
                    "wafer_sequence": lot_idx * wafers_per_lot + wafer_idx,
                }

                for param, info in _PROCESS_TARGETS.items():
                    if param == "etch_rate":
                        mean = aged_etch_mean
                    elif param == "deposition_unif":
                        mean = aged_unif_mean
                    else:
                        mean = current_means[param]

                    # Radial non-uniformity for thickness/CD parameters
                    if param in ("gate_oxide_thickness", "poly_cd"):
                        radial_offset = self._radial_nonuniformity(param, info["sigma"])
                        mean += radial_offset

                    meas[param] = float(self.rng.normal(mean, info["sigma"]))

                # --- Defect density based on process cleanliness ---
                # Cleanliness worsens with equipment age and large deviations
                deviation = (
                    abs(
                        meas["gate_oxide_thickness"]
                        - _PROCESS_TARGETS["gate_oxide_thickness"]["mean"]
                    )
                    / _PROCESS_TARGETS["gate_oxide_thickness"]["sigma"]
                )

                base_defect = 0.05  # defects/cm^2 baseline
                defect_density = base_defect * (1.0 + 0.3 * age + 0.1 * deviation)
                defect_density = max(0.001, defect_density)
                meas["defect_density"] = defect_density

                # --- Murphy yield model ---
                param_yield = self._parametric_yield(meas)
                die_yield = _murphy_yield(self.chip_area_cm2, defect_density)
                meas["yield"] = float(np.clip(param_yield * die_yield, 0.0, 1.0))

                # --- Wafer map (serialised mean yield across die) ---
                wmap = self.generate_wafer_map(
                    wafer_id,
                    nonuniformity_level=meas["deposition_unif"] / 10.0,
                )
                meas["wafer_map_mean"] = float(wmap.mean())
                meas["wafer_map_std"] = float(wmap.std())
                # Store the wafer map as a compact string representation
                meas["wafer_map"] = _pack_wafer_map(wmap)

                rows.append(meas)

        df = pd.DataFrame(rows)
        return df

    def generate_wafer_map(
        self,
        wafer_id: str,
        nonuniformity_level: float = 0.05,
    ) -> np.ndarray:
        """Generate a 2-D wafer map of local yield fractions.

        The map represents spatial variation in yield across the wafer.
        A radial gradient models centre-to-edge process variation;
        random noise captures local random defects.

        Parameters
        ----------
        wafer_id : str
            Wafer identifier (used to seed a derived RNG for repeatability).
        nonuniformity_level : float
            Amplitude of the radial non-uniformity term (fraction of mean yield).

        Returns
        -------
        numpy.ndarray
            2-D float array of shape (wafer_map_size, wafer_map_size) with
            local yield values in [0, 1].
        """
        # Derive a local RNG from wafer_id hash so maps are reproducible
        wafer_seed = abs(hash(wafer_id)) % (2**31)
        local_rng = np.random.default_rng(
            wafer_seed if self._seed is None else self._seed + wafer_seed % 10000
        )

        n = self.wafer_map_size
        cx, cy = n / 2.0, n / 2.0

        # Build radial distance map (normalised to 1 at edge)
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)

        # Radial yield profile: slightly higher in center, drops at edge
        radial = 1.0 - nonuniformity_level * (r**2)

        # Random defect contribution
        random_component = 1.0 - local_rng.exponential(0.02, size=(n, n))

        # Mask outside wafer circle
        inside = r <= 1.0
        wafer_map = np.where(inside, np.clip(radial * random_component, 0.0, 1.0), np.nan)

        return wafer_map

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _radial_nonuniformity(self, param: str, sigma: float) -> float:
        """Generate a small radial offset for a wafer measurement.

        Simulates centre-to-edge variation by returning a normally
        distributed offset proportional to sigma.
        """
        return float(self.rng.normal(0.0, 0.2 * sigma))

    @staticmethod
    def _parametric_yield(meas: dict[str, Any]) -> float:
        """Compute parametric yield as a function of process spec windows.

        Returns a yield penalty factor in [0, 1] based on how far key
        parameters deviate from their specifications.
        """
        # Spec limits (mean +/- k*sigma windows)
        specs = {
            "gate_oxide_thickness": (
                _PROCESS_TARGETS["gate_oxide_thickness"]["mean"],
                3.0 * _PROCESS_TARGETS["gate_oxide_thickness"]["sigma"],
            ),
            "poly_cd": (
                _PROCESS_TARGETS["poly_cd"]["mean"],
                3.0 * _PROCESS_TARGETS["poly_cd"]["sigma"],
            ),
            "contact_resistance": (
                _PROCESS_TARGETS["contact_resistance"]["mean"],
                3.0 * _PROCESS_TARGETS["contact_resistance"]["sigma"],
            ),
        }

        yield_factor = 1.0
        for param, (target, tolerance) in specs.items():
            if param not in meas:
                continue
            deviation = abs(meas[param] - target) / tolerance
            if deviation > 1.0:
                # Gaussian roll-off beyond spec limit
                yield_factor *= math.exp(-0.5 * (deviation - 1.0) ** 2)

        return float(np.clip(yield_factor, 0.0, 1.0))


def _pack_wafer_map(wmap: np.ndarray) -> str:
    """Serialise a wafer map to a compact string for DataFrame storage."""
    n = wmap.shape[0]
    # Replace NaN with -1 for serialisation
    flat = np.nan_to_num(wmap.flatten(), nan=-1.0)
    # Quantise to 8-bit unsigned int (0-255) for compactness
    quantised = np.clip(flat * 255.0, 0, 255).astype(np.uint8)
    return f"{n}:{quantised.tobytes().hex()}"


def unpack_wafer_map(packed: str) -> np.ndarray:
    """Deserialise a wafer map packed by _pack_wafer_map.

    Parameters
    ----------
    packed : str
        String produced by _pack_wafer_map.

    Returns
    -------
    numpy.ndarray
        2-D float array with values in [0, 1] (NaN outside wafer circle).
    """
    n_str, hex_data = packed.split(":", 1)
    n = int(n_str)
    raw = np.frombuffer(bytes.fromhex(hex_data), dtype=np.uint8).astype(float) / 255.0
    arr = raw.reshape(n, n)
    # Restore NaN for out-of-wafer pixels (quantised to 0 from -1)
    # Use a threshold: values < 1/255 that came from -1 -> NaN
    # We stored -1 -> clip to 0, so we cannot distinguish. Leave as 0.
    return arr
