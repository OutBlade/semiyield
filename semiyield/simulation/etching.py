"""
Langmuir-Hinshelwood etch rate model for plasma and chemical etching.

Reference:
  C.J. Mogab, A.C. Adams, D.L. Flamm, "Plasma etching of Si and SiO2 – the effect
  of oxygen additions to CF4 plasmas," J. Appl. Phys. 49(7), 3796-3803, 1978.

  The Langmuir-Hinshelwood (L-H) mechanism describes surface-reaction-limited
  etching where etch species adsorb on surface sites before reacting.

Single-reactant L-H model:
    theta = (K * P) / (1 + K * P)          surface coverage fraction
    R = k_s * theta                          etch rate [nm/min]

where:
    P    = partial pressure of etchant species [Torr or mTorr]
    K    = adsorption equilibrium constant [Torr^{-1}]
    k_s  = surface reaction rate constant [nm/min]

Both k_s and K follow Arrhenius temperature dependence.

Two-reactant L-H model (plasma etching with independent reactants A and B):
    R = k * (K_A*P_A * K_B*P_B) / (1 + K_A*P_A + K_B*P_B)^2

This form arises when two species must co-adsorb on adjacent sites.
"""

import math
from typing import Literal

# ------------------------------------------------------------------ #
# Tabulated parameters for common plasma etch chemistries              #
#                                                                      #
# Each entry: (k_s0, Ea_ks, K0, Ea_K)                                 #
#   k_s0  [nm/min]   pre-exponential for surface rate constant        #
#   Ea_ks [eV]       activation energy for k_s                        #
#   K0    [mTorr^-1] pre-exponential for adsorption constant          #
#   Ea_K  [eV]       adsorption energy (negative = exothermic ads.)   #
#                                                                      #
# Selectivity pairs tabulated for CHF3/O2 plasma:                     #
#   SiO2 vs Si:  selectivity ~ 10-30 (standard fluorocarbon)          #
#   Si3N4 vs Si: selectivity ~ 2-5   (isotropic fluorine etch)        #
# ------------------------------------------------------------------ #

_K_EV = 8.617333e-5  # eV/K

# Single-reactant parameters (CF4-based chemistry)
_SINGLE_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "SiO2": (
        8.5e3,  # k_s0  [nm/min]
        0.25,  # Ea_ks [eV]
        3.2e-2,  # K0    [mTorr^{-1}]
        -0.12,  # Ea_K  [eV]  (exothermic adsorption -> negative)
    ),
    "Si": (
        5.0e3,
        0.30,
        2.1e-2,
        -0.10,
    ),
    "Si3N4": (
        6.0e3,
        0.28,
        2.6e-2,
        -0.11,
    ),
}

# Two-reactant parameters (CHF3/O2 plasma)
# Reactant A = CHF3 (fluorocarbon film former), reactant B = O2 (polymer cleaner)
_TWO_REACTANT_PARAMS: dict[str, dict] = {
    "SiO2": {
        "k0": 1.2e4,  # nm/min
        "Ea_k": 0.22,  # eV
        "KA0": 4.0e-2,
        "Ea_KA": -0.13,
        "KB0": 1.5e-2,
        "Ea_KB": -0.09,
    },
    "Si": {
        "k0": 4.0e3,
        "Ea_k": 0.35,
        "KA0": 1.8e-2,
        "Ea_KA": -0.08,
        "KB0": 2.0e-2,
        "Ea_KB": -0.11,
    },
    "Si3N4": {
        "k0": 5.5e3,
        "Ea_k": 0.29,
        "KA0": 2.2e-2,
        "Ea_KA": -0.10,
        "KB0": 1.8e-2,
        "Ea_KB": -0.10,
    },
}


def _arr(prefactor: float, activation_ev: float, temperature_k: float) -> float:
    """Arrhenius function: prefactor * exp(-Ea / (k_B * T))."""
    return prefactor * math.exp(-activation_ev / (_K_EV * temperature_k))


class LangmuirHinshelwoodModel:
    """Langmuir-Hinshelwood etch rate model for plasma etching.

    Supports both single-reactant and two-reactant (competitive adsorption)
    L-H kinetics with Arrhenius temperature dependence of rate constants.

    Parameters
    ----------
    mode : {'single', 'two_reactant'}
        Kinetic model variant.  'single' is suitable for simple CF4 etching;
        'two_reactant' represents CHF3/O2 plasma where fluorocarbon polymer
        deposition and reactive ion etching compete.
    """

    def __init__(self, mode: Literal["single", "two_reactant"] = "single") -> None:
        if mode not in ("single", "two_reactant"):
            raise ValueError("mode must be 'single' or 'two_reactant'.")
        self.mode = mode

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def rate(
        self,
        pressure: float,
        temperature: float,
        material: str = "SiO2",
        pressure_b: float | None = None,
    ) -> float:
        """Compute etch rate for *material* under given conditions.

        Parameters
        ----------
        pressure : float
            Etchant partial pressure in mTorr.
            In two-reactant mode this is the pressure of reactant A (CHF3).
        temperature : float
            Substrate temperature in degrees Celsius.
        material : str
            Target material: 'SiO2', 'Si', or 'Si3N4'.
        pressure_b : float or None
            Pressure of reactant B (O2) in mTorr.  Required for two-reactant
            mode; ignored in single-reactant mode.

        Returns
        -------
        float
            Etch rate in nm/min.
        """
        T_K = temperature + 273.15

        if self.mode == "single":
            return self._single_rate(pressure, T_K, material)
        else:
            if pressure_b is None:
                raise ValueError("pressure_b (O2 pressure) is required for two_reactant mode.")
            return self._two_reactant_rate(pressure, pressure_b, T_K, material)

    def selectivity(
        self,
        material_a: str,
        material_b: str,
        pressure: float,
        temperature: float,
        pressure_b: float | None = None,
    ) -> float:
        """Compute etch selectivity of material_a over material_b.

        Selectivity = R(material_a) / R(material_b)

        A value > 1 means material_a etches faster than material_b.

        Parameters
        ----------
        material_a : str
            Faster-etching material (numerator).
        material_b : str
            Reference material (denominator).
        pressure : float
            Etchant pressure in mTorr (reactant A in two-reactant mode).
        temperature : float
            Temperature in degrees Celsius.
        pressure_b : float or None
            Pressure of O2 in mTorr (two-reactant mode only).

        Returns
        -------
        float
            Selectivity ratio R_a / R_b.
        """
        R_a = self.rate(pressure, temperature, material_a, pressure_b)
        R_b = self.rate(pressure, temperature, material_b, pressure_b)
        if R_b == 0.0:
            return float("inf")
        return R_a / R_b

    def coverage(self, pressure: float, temperature: float, material: str = "SiO2") -> float:
        """Fractional surface coverage theta = (K*P) / (1 + K*P).

        Parameters
        ----------
        pressure : float
            Etchant pressure in mTorr.
        temperature : float
            Temperature in degrees Celsius.
        material : str
            Material name.

        Returns
        -------
        float
            Surface coverage fraction in [0, 1].
        """
        T_K = temperature + 273.15
        if material not in _SINGLE_PARAMS:
            raise ValueError(f"Unknown material '{material}'.")
        _, _, K0, Ea_K = _SINGLE_PARAMS[material]
        K = _arr(K0, Ea_K, T_K)
        return (K * pressure) / (1.0 + K * pressure)

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _single_rate(self, pressure: float, T_K: float, material: str) -> float:
        """Single-reactant L-H etch rate."""
        if material not in _SINGLE_PARAMS:
            raise ValueError(f"Unknown material '{material}'.  Choose from {list(_SINGLE_PARAMS)}.")
        ks0, Ea_ks, K0, Ea_K = _SINGLE_PARAMS[material]
        k_s = _arr(ks0, Ea_ks, T_K)
        K = _arr(K0, Ea_K, T_K)
        theta = (K * pressure) / (1.0 + K * pressure)
        return k_s * theta

    def _two_reactant_rate(self, P_A: float, P_B: float, T_K: float, material: str) -> float:
        """Two-reactant competitive adsorption etch rate.

        R = k * (K_A*P_A * K_B*P_B) / (1 + K_A*P_A + K_B*P_B)^2
        """
        if material not in _TWO_REACTANT_PARAMS:
            raise ValueError(
                f"Unknown material '{material}'.  Choose from {list(_TWO_REACTANT_PARAMS)}."
            )
        p = _TWO_REACTANT_PARAMS[material]
        k = _arr(p["k0"], p["Ea_k"], T_K)
        K_A = _arr(p["KA0"], p["Ea_KA"], T_K)
        K_B = _arr(p["KB0"], p["Ea_KB"], T_K)

        numerator = k * K_A * P_A * K_B * P_B
        denominator = (1.0 + K_A * P_A + K_B * P_B) ** 2
        return numerator / denominator
