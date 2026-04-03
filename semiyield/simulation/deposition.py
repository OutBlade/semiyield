"""
CVD/PVD thin-film deposition models.

Reference:
  K.F. Jensen, "Chemical Vapor Deposition," in Microelectronics Processing:
  Chemical Engineering Aspects, ACS Advances in Chemistry Series 221, 1989.

  R.A. Levy, Microelectronic Materials and Processes, Kluwer, 1989.

Growth rate model:
    R = C * P_precursor * exp(-Ea / (k_B * T))

Step coverage:
    ALD:  nearly 100% (conformal due to self-limiting surface chemistry)
    CVD:  partial (diffusion-limited, depends on aspect ratio)
    PVD:  line-of-sight (directional, poor step coverage in trenches)

Film stress:
    sigma = sigma_intrinsic + sigma_thermal
    sigma_thermal = (E / (1 - nu)) * (alpha_film - alpha_substrate) * dT

where:
    E    = Young's modulus of film [GPa]
    nu   = Poisson's ratio
    alpha = thermal expansion coefficient [ppm/K]
    dT   = temperature difference from deposition to room temperature [K]
"""

import math
from typing import Literal

import numpy as np

_K_EV = 8.617333e-5  # eV/K

# ------------------------------------------------------------------ #
# Process parameters for common CVD/PVD deposition types               #
# ------------------------------------------------------------------ #

# Growth rate: R = C * P * exp(-Ea / kT)   [nm/min per Torr]
_GROWTH_PARAMS: dict[str, dict] = {
    "LPCVD": {
        "SiO2": {"C": 8.0e8, "Ea": 1.90},    # TEOS-based LPCVD SiO2
        "Si3N4": {"C": 5.0e9, "Ea": 2.10},   # DCS/NH3 LPCVD Si3N4
        "poly_Si": {"C": 3.0e9, "Ea": 1.70}, # SiH4 LPCVD poly-Si
    },
    "PECVD": {
        "SiO2": {"C": 2.0e6, "Ea": 0.50},    # SiH4/N2O PECVD SiO2
        "Si3N4": {"C": 1.5e6, "Ea": 0.55},   # SiH4/NH3 PECVD Si3N4
        "a_Si": {"C": 1.0e6, "Ea": 0.45},    # SiH4 a-Si:H
    },
    "ALD": {
        "HfO2": {"C": 0.1, "Ea": 0.10},      # HfCl4/H2O ALD HfO2 [nm/cycle]
        "Al2O3": {"C": 0.12, "Ea": 0.08},    # TMA/H2O ALD Al2O3  [nm/cycle]
        "TiN": {"C": 0.08, "Ea": 0.12},      # TiCl4/NH3 ALD TiN  [nm/cycle]
    },
    "PVD": {
        "Al": {"C": 1.5e3, "Ea": 0.05},      # DC magnetron sputtering Al
        "TiN": {"C": 8.0e2, "Ea": 0.08},     # Reactive sputtering TiN
        "W": {"C": 5.0e2, "Ea": 0.06},       # DC sputtering W
    },
}

# Step coverage model parameters
# step_coverage = coverage_0 * exp(-AR / AR_char)
# AR  = aspect ratio of feature (depth/width)
_STEP_COVERAGE_PARAMS: dict[str, dict] = {
    "LPCVD":  {"coverage_0": 0.92, "AR_char": 8.0},
    "PECVD":  {"coverage_0": 0.80, "AR_char": 3.5},
    "ALD":    {"coverage_0": 0.995, "AR_char": 100.0},
    "PVD":    {"coverage_0": 0.60, "AR_char": 1.2},
}

# Film stress parameters
# sigma = sigma_intrinsic + (E/(1-nu)) * (alpha_film - alpha_sub) * dT
_STRESS_PARAMS: dict[str, dict] = {
    # material: E_GPa, nu, alpha_ppm_K, sigma_intrinsic_MPa (compressive neg.)
    "SiO2":    {"E": 70.0,  "nu": 0.17, "alpha": 0.55, "sigma_i": -250.0},
    "Si3N4":   {"E": 250.0, "nu": 0.25, "alpha": 3.0,  "sigma_i": 1000.0},
    "poly_Si": {"E": 160.0, "nu": 0.22, "alpha": 2.8,  "sigma_i": -200.0},
    "HfO2":    {"E": 130.0, "nu": 0.28, "alpha": 5.8,  "sigma_i": -300.0},
    "Al2O3":   {"E": 300.0, "nu": 0.24, "alpha": 8.0,  "sigma_i": -100.0},
    "TiN":     {"E": 450.0, "nu": 0.25, "alpha": 9.4,  "sigma_i": 800.0},
    "Al":      {"E": 70.0,  "nu": 0.35, "alpha": 23.1, "sigma_i": -50.0},
    "W":       {"E": 411.0, "nu": 0.28, "alpha": 4.5,  "sigma_i": 1200.0},
}

# Silicon substrate: alpha_Si = 2.6 ppm/K
_ALPHA_SI = 2.6  # ppm/K


class CVDModel:
    """CVD/PVD thin-film deposition model.

    Computes deposition rate, step coverage, and film stress for a variety
    of deposition processes and materials.

    Parameters
    ----------
    process_type : {'LPCVD', 'PECVD', 'ALD', 'PVD'}
        Deposition process.
    material : str
        Film material.  Must be present in the process-type parameter table.
    """

    def __init__(
        self,
        process_type: Literal["LPCVD", "PECVD", "ALD", "PVD"] = "LPCVD",
        material: str = "SiO2",
    ) -> None:
        if process_type not in _GROWTH_PARAMS:
            raise ValueError(
                f"Unknown process_type '{process_type}'.  "
                f"Choose from {list(_GROWTH_PARAMS)}."
            )
        if material not in _GROWTH_PARAMS[process_type]:
            raise ValueError(
                f"Material '{material}' not available for {process_type}.  "
                f"Available: {list(_GROWTH_PARAMS[process_type])}."
            )
        self.process_type = process_type
        self.material = material

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def deposit(
        self,
        time: float,
        temperature: float,
        pressure: float = 1.0,
    ) -> float:
        """Compute deposited film thickness.

        For ALD, *time* is interpreted as the number of deposition cycles.

        Parameters
        ----------
        time : float
            Deposition time in minutes (or cycles for ALD).
        temperature : float
            Substrate temperature in degrees Celsius.
        pressure : float
            Precursor partial pressure in Torr.
            For ALD and PVD this parameter has reduced significance but is
            retained for interface consistency.

        Returns
        -------
        float
            Film thickness in nanometres.
        """
        T_K = temperature + 273.15
        params = _GROWTH_PARAMS[self.process_type][self.material]
        R = params["C"] * pressure * math.exp(-params["Ea"] / (_K_EV * T_K))
        # ALD: R is nm/cycle, time is cycles; others: R is nm/min, time is min
        return R * time

    def uniformity(
        self,
        wafer_diameter_mm: float = 300.0,
        aspect_ratio: float = 0.0,
    ) -> float:
        """Estimate deposition uniformity across the wafer.

        Returns the 1-sigma non-uniformity as a percentage of the mean
        thickness.  The model is empirical and based on published data ranges.

        Parameters
        ----------
        wafer_diameter_mm : float
            Wafer diameter in mm.  Larger wafers generally have higher
            non-uniformity.
        aspect_ratio : float
            Feature aspect ratio for step-coverage estimation.

        Returns
        -------
        float
            Non-uniformity in percent (1-sigma / mean * 100).
        """
        # Base non-uniformity by process type (percent, 1-sigma)
        _base = {"LPCVD": 1.5, "PECVD": 3.0, "ALD": 0.8, "PVD": 4.0}
        base = _base[self.process_type]

        # Larger wafers degrade uniformity slightly
        wafer_factor = 1.0 + 0.002 * max(0.0, wafer_diameter_mm - 200.0)

        # Higher AR slightly degrades CVD/PVD uniformity
        ar_factor = 1.0 + 0.05 * aspect_ratio if self.process_type != "ALD" else 1.0

        return base * wafer_factor * ar_factor

    def step_coverage(self, aspect_ratio: float) -> float:
        """Compute step coverage for a trench or contact with given aspect ratio.

        Step coverage is defined as the ratio of film thickness at the bottom
        of a feature to the thickness on the flat surface above.

        Parameters
        ----------
        aspect_ratio : float
            Feature aspect ratio (depth / width).  Typical values: 1-10.

        Returns
        -------
        float
            Step coverage fraction in [0, 1].
        """
        p = _STEP_COVERAGE_PARAMS[self.process_type]
        sc = p["coverage_0"] * math.exp(-aspect_ratio / p["AR_char"])
        return max(0.0, min(1.0, sc))

    def stress(
        self,
        temperature_delta: float,
        material: str | None = None,
    ) -> float:
        """Compute total film stress (intrinsic + thermal mismatch).

        sigma_total = sigma_intrinsic + (E / (1 - nu)) * (alpha_film - alpha_sub) * dT

        Parameters
        ----------
        temperature_delta : float
            Temperature difference from deposition temperature to measurement
            temperature (room temp), in Kelvin.
            A positive dT means the film cooled down after deposition.
        material : str or None
            Film material.  If None, uses the material set at construction.

        Returns
        -------
        float
            Biaxial film stress in MPa.  Positive = tensile, negative = compressive.
        """
        mat = material or self.material
        if mat not in _STRESS_PARAMS:
            raise ValueError(
                f"Stress parameters not available for '{mat}'.  "
                f"Known materials: {list(_STRESS_PARAMS)}."
            )
        p = _STRESS_PARAMS[mat]
        biaxial_modulus = p["E"] * 1e3 / (1.0 - p["nu"])  # GPa -> MPa / (1-nu)
        delta_alpha = (p["alpha"] - _ALPHA_SI) * 1e-6      # ppm -> dimensionless
        sigma_thermal = biaxial_modulus * delta_alpha * temperature_delta
        return p["sigma_i"] + sigma_thermal

    def rate_vs_temperature(
        self,
        temperatures: "np.ndarray",
        pressure: float = 1.0,
    ) -> "np.ndarray":
        """Growth rate as a function of temperature array.

        Parameters
        ----------
        temperatures : array-like
            Temperatures in degrees Celsius.
        pressure : float
            Precursor pressure in Torr.

        Returns
        -------
        numpy.ndarray
            Growth rates in nm/min (or nm/cycle for ALD).
        """
        temps = np.asarray(temperatures, dtype=float)
        params = _GROWTH_PARAMS[self.process_type][self.material]
        T_K = temps + 273.15
        return params["C"] * pressure * np.exp(-params["Ea"] / (_K_EV * T_K))
