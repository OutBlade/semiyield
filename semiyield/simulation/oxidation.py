"""
Deal-Grove thermal oxidation model for silicon dioxide growth.

Reference: B.E. Deal and A.S. Grove, "General Relationship for the Thermal Oxidation
of Silicon," Journal of Applied Physics, 36(12), 3770-3778, 1965.

The model describes oxide growth via the linear-parabolic equation:
    x^2 + A*x = B*(t + tau)

where:
    x    = oxide thickness (cm)
    t    = oxidation time (s)
    A    = linear rate constant ratio (cm)
    B    = parabolic rate constant (cm^2/s)
    tau  = time offset accounting for initial oxide (s)

Solving for x:
    x = (A/2) * (sqrt(1 + (t + tau) / (A^2 / 4B)) - 1)
"""

import math
from typing import Literal

import numpy as np

# Physical constants
R_GAS = 8.314  # J/(mol*K)  - universal gas constant

# ------------------------------------------------------------------ #
# Tabulated rate constants from Deal-Grove (1965) and subsequent      #
# literature.  Values are pre-exponential factors and activation      #
# energies for B (parabolic) and B/A (linear) rate constants.         #
#                                                                      #
# B   = B0 * exp(-Ea_B  / (R*T))   [cm^2/s]                          #
# B/A = (B/A)0 * exp(-Ea_BA / (R*T)) [cm/s]                          #
#                                                                      #
# Sources:                                                             #
#   Dry  O2: Deal & Grove (1965), Table II                             #
#   Wet  H2O: Massoud et al. (1985), corrected for steam oxidation    #
# ------------------------------------------------------------------ #

_RATE_CONSTANTS = {
    "dry": {
        # Source: Jaeger, "Introduction to Microelectronic Fabrication", Table 3-1
        # Original values in um^2/h (B) and um/h (B/A); converted to cm^2/s and cm/s:
        #   1 um^2/h = 1e-8 cm^2 / 3600 s  =>  multiply by 2.778e-12
        #   1 um/h   = 1e-4 cm  / 3600 s   =>  multiply by 2.778e-8
        "B0": 7.72e2 * 2.778e-12,  # cm^2/s  pre-exponential parabolic
        "Ea_B": 1.23,  # eV      parabolic activation energy
        "BA0": 6.23e6 * 2.778e-8,  # cm/s    pre-exponential linear
        "Ea_BA": 2.00,  # eV      linear activation energy
    },
    "wet": {
        # Wet (H2O) oxidation from Jaeger Table 3-1
        "B0": 2.14e2 * 2.778e-12,  # cm^2/s
        "Ea_B": 0.71,  # eV
        "BA0": 1.63e8 * 2.778e-8,  # cm/s
        "Ea_BA": 2.05,  # eV
    },
}

# Boltzmann constant in eV/K
_K_EV = 8.617333e-5  # eV/K


def _arrhenius(prefactor: float, activation_ev: float, temperature_k: float) -> float:
    """Evaluate an Arrhenius expression.

    Parameters
    ----------
    prefactor:
        Pre-exponential factor (any units).
    activation_ev:
        Activation energy in electron-volts.
    temperature_k:
        Temperature in Kelvin.

    Returns
    -------
    float
        prefactor * exp(-activation_ev / (k_B * T))
    """
    return prefactor * math.exp(-activation_ev / (_K_EV * temperature_k))


class DealGroveModel:
    """Thermal oxidation of silicon following the Deal-Grove linear-parabolic model.

    The governing equation is:

        x^2 + A*x = B*(t + tau)                                  (1)

    Rearranged to give thickness explicitly:

        x(t) = (A/2) * [sqrt(1 + (t + tau)/(A^2/(4B))) - 1]     (2)

    The rate constants A and B are related by:
        B   = parabolic rate constant  [cm^2/s]
        B/A = linear rate constant     [cm/s]
        A   = B / (B/A)                [cm]

    Both B and B/A follow Arrhenius temperature dependence.

    Parameters
    ----------
    orientation : str
        Silicon crystal orientation.  Currently both <100> and <111>
        use the same tabulated constants (minor difference absorbed into
        rate-constant spread).  Kept as a parameter for future extension.
    """

    def __init__(self, orientation: str = "<100>") -> None:
        self.orientation = orientation

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def grow(
        self,
        time: float,
        temperature: float,
        atmosphere: Literal["dry", "wet"] = "dry",
        initial_thickness_nm: float = 0.0,
    ) -> float:
        """Compute oxide thickness after a given oxidation time.

        Parameters
        ----------
        time : float
            Oxidation time in minutes.
        temperature : float
            Furnace temperature in degrees Celsius.
        atmosphere : {'dry', 'wet'}
            Oxidation ambient.  'wet' is steam (H2O), 'dry' is pure O2.
        initial_thickness_nm : float
            Pre-existing oxide thickness in nanometres (e.g., native oxide ~1 nm).

        Returns
        -------
        float
            Oxide thickness in nanometres after *time* minutes.

        Examples
        --------
        >>> model = DealGroveModel()
        >>> model.grow(60, 1000, "dry")   # 1 h dry at 1000 C
        """
        T_K = temperature + 273.15
        t_s = time * 60.0  # minutes -> seconds

        B, BA = self._rate_constants(atmosphere, T_K)
        A = B / BA

        # tau: time equivalent for initial oxide thickness (cm)
        x0_cm = initial_thickness_nm * 1e-7  # nm -> cm
        tau_s = (x0_cm**2 + A * x0_cm) / B  # from eq. (1) with x=x0, t=0

        # Solve eq. (2)
        discriminant = 1.0 + (t_s + tau_s) / (A**2 / (4.0 * B))
        x_cm = (A / 2.0) * (math.sqrt(discriminant) - 1.0)
        return x_cm * 1e7  # cm -> nm

    def rate(
        self,
        time: float,
        temperature: float,
        atmosphere: Literal["dry", "wet"] = "dry",
        initial_thickness_nm: float = 0.0,
        dt_min: float = 0.1,
    ) -> float:
        """Instantaneous oxide growth rate at a given time.

        Computed as a finite difference dx/dt ~ (x(t+dt) - x(t-dt)) / (2*dt).

        Parameters
        ----------
        time : float
            Current oxidation time in minutes.
        temperature : float
            Furnace temperature in degrees Celsius.
        atmosphere : {'dry', 'wet'}
            Oxidation ambient.
        initial_thickness_nm : float
            Pre-existing oxide thickness in nanometres.
        dt_min : float
            Half-width of finite-difference interval in minutes.

        Returns
        -------
        float
            Growth rate in nm/min.
        """
        t_lo = max(0.0, time - dt_min)
        t_hi = time + dt_min
        x_lo = self.grow(t_lo, temperature, atmosphere, initial_thickness_nm)
        x_hi = self.grow(t_hi, temperature, atmosphere, initial_thickness_nm)
        return (x_hi - x_lo) / (t_hi - t_lo)

    def growth_curve(
        self,
        time_array: "np.ndarray",
        temperature: float,
        atmosphere: Literal["dry", "wet"] = "dry",
        initial_thickness_nm: float = 0.0,
    ) -> "np.ndarray":
        """Compute oxide thickness for an array of times.

        Parameters
        ----------
        time_array : array-like
            Times in minutes.
        temperature : float
            Temperature in degrees Celsius.
        atmosphere : {'dry', 'wet'}
            Oxidation ambient.
        initial_thickness_nm : float
            Initial oxide thickness in nanometres.

        Returns
        -------
        numpy.ndarray
            Oxide thicknesses in nanometres (same shape as *time_array*).
        """
        times = np.asarray(time_array, dtype=float)
        return np.array(
            [self.grow(t, temperature, atmosphere, initial_thickness_nm) for t in times]
        )

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _rate_constants(self, atmosphere: str, temperature_k: float) -> tuple[float, float]:
        """Return (B, B/A) rate constants at the given temperature.

        Parameters
        ----------
        atmosphere : str
            'dry' or 'wet'.
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        tuple[float, float]
            (B [cm^2/s], B/A [cm/s])
        """
        if atmosphere not in _RATE_CONSTANTS:
            raise ValueError(f"Unknown atmosphere '{atmosphere}'.  Choose 'dry' or 'wet'.")
        c = _RATE_CONSTANTS[atmosphere]
        B = _arrhenius(c["B0"], c["Ea_B"], temperature_k)
        BA = _arrhenius(c["BA0"], c["Ea_BA"], temperature_k)
        return B, BA
