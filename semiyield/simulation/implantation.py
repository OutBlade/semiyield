"""
Ion implantation dopant profile models.

Reference:
  J. Lindhard, M. Scharff, H.E. Schiott, "Range Concepts and Heavy Ion Ranges,"
  Mat. Fys. Medd. Dan. Vid. Selsk. 33(14), 1963 (LSS theory).

  W. Hofker et al., "Concentration profiles of boron implantations in amorphous and
  polycrystalline silicon," Applied Physics, 2(5), 265-278, 1973 (Pearson IV).

The primary model is the Gaussian approximation:

    N(x) = (Phi / (sqrt(2*pi) * dRp)) * exp(-(x - Rp)^2 / (2 * dRp^2))

where:
    Phi  = implant dose [cm^{-2}]
    Rp   = projected range [nm]
    dRp  = range straggling (standard deviation) [nm]
    x    = depth into substrate [nm]

For more accurate tail modeling, a Pearson IV distribution is also available.
"""

import math
from typing import Literal

import numpy as np
from scipy import special, stats

# ------------------------------------------------------------------ #
# LSS-theory-based empirical tables for Rp and dRp                    #
# (projected range and straggling in silicon)                          #
#                                                                      #
# Source: Gibbons, Johnson & Mylroie, "Projected Range Statistics     #
#         in Semiconductors," Halsted Press, 1975, Table 2.1          #
#                                                                      #
# Format: energy_keV -> (Rp_nm, dRp_nm)                               #
# ------------------------------------------------------------------ #

_RANGE_TABLE: dict[str, dict[int, tuple[float, float]]] = {
    "boron": {
        10:  (33.0,  17.5),
        20:  (67.0,  28.0),
        30:  (96.0,  36.0),
        50:  (151.0, 49.0),
        80:  (235.0, 65.0),
        100: (290.0, 74.0),
        150: (420.0, 93.0),
        200: (552.0, 108.0),
        300: (800.0, 133.0),
    },
    "phosphorus": {
        10:  (13.0,  6.5),
        20:  (26.0,  12.0),
        30:  (39.0,  17.0),
        50:  (63.0,  25.0),
        80:  (99.0,  36.0),
        100: (120.0, 43.0),
        150: (175.0, 57.0),
        200: (232.0, 69.0),
        300: (350.0, 89.0),
    },
    "arsenic": {
        10:  (7.0,   3.5),
        20:  (14.0,  6.5),
        30:  (21.0,  9.5),
        50:  (32.0,  14.0),
        80:  (50.0,  20.0),
        100: (60.0,  24.0),
        150: (88.0,  32.0),
        200: (116.0, 40.0),
        300: (175.0, 54.0),
    },
    "antimony": {
        10:  (5.5,   2.5),
        20:  (10.0,  4.5),
        30:  (15.0,  6.5),
        50:  (23.0,  10.0),
        80:  (36.0,  14.5),
        100: (44.0,  17.5),
        150: (64.0,  24.0),
        200: (85.0,  30.0),
        300: (128.0, 41.0),
    },
}

_SUPPORTED_IONS = list(_RANGE_TABLE.keys())


def _interpolate_range(species: str, energy_kev: float) -> tuple[float, float]:
    """Interpolate (Rp, dRp) for *species* at *energy_kev* using log-log interpolation.

    Parameters
    ----------
    species : str
        Ion species name (lowercase).
    energy_kev : float
        Implant energy in keV.

    Returns
    -------
    tuple[float, float]
        (Rp_nm, dRp_nm)
    """
    if species not in _RANGE_TABLE:
        raise ValueError(
            f"Unknown species '{species}'.  Supported: {_SUPPORTED_IONS}."
        )
    table = _RANGE_TABLE[species]
    energies = sorted(table.keys())
    rp_vals = [table[e][0] for e in energies]
    drp_vals = [table[e][1] for e in energies]

    # Log-log interpolation gives better accuracy over wide energy ranges
    log_e = np.log(energies)
    log_rp = np.log(rp_vals)
    log_drp = np.log(drp_vals)

    rp_nm = float(np.exp(np.interp(math.log(energy_kev), log_e, log_rp)))
    drp_nm = float(np.exp(np.interp(math.log(energy_kev), log_e, log_drp)))
    return rp_nm, drp_nm


class IonImplantationModel:
    """Gaussian (and Pearson IV) dopant profile from ion implantation into silicon.

    The Gaussian approximation is valid for light ions (boron) and moderate
    energies.  For heavier ions (arsenic, antimony) or precise tail modeling,
    use ``distribution='pearsoniv'``.

    Parameters
    ----------
    distribution : {'gaussian', 'pearsoniv'}
        Profile shape model.  'gaussian' is the standard approximation;
        'pearsoniv' uses a skewed distribution with empirical skewness and
        kurtosis parameters.
    """

    def __init__(self, distribution: Literal["gaussian", "pearsoniv"] = "gaussian") -> None:
        if distribution not in ("gaussian", "pearsoniv"):
            raise ValueError("distribution must be 'gaussian' or 'pearsoniv'.")
        self.distribution = distribution

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def profile(
        self,
        depth_array: "np.ndarray",
        dose: float,
        energy: float,
        species: str,
    ) -> "np.ndarray":
        """Compute dopant concentration profile as a function of depth.

        Parameters
        ----------
        depth_array : array-like
            Depths at which to evaluate N(x), in nanometres.
        dose : float
            Implant dose (fluence) in cm^{-2}.
        energy : float
            Implant energy in keV.
        species : str
            Ion species: 'boron', 'phosphorus', 'arsenic', or 'antimony'.

        Returns
        -------
        numpy.ndarray
            Dopant concentration in cm^{-3} at each depth in *depth_array*.

        Notes
        -----
        The Gaussian profile satisfies integral(N, 0, inf) = dose when
        integrated over the full depth range, assuming Rp >> dRp.
        """
        x = np.asarray(depth_array, dtype=float)  # nm
        Rp, dRp = _interpolate_range(species, energy)

        # Convert dose from cm^{-2} to nm^{-2} for internal computation,
        # then convert concentration back to cm^{-3}
        # 1 nm^{-3} = 1e21 cm^{-3}
        if self.distribution == "gaussian":
            N = self._gaussian_profile(x, dose, Rp, dRp)
        else:
            N = self._pearsoniv_profile(x, dose, Rp, dRp, species)

        return N  # cm^{-3}

    def junction_depth(
        self,
        dose: float,
        energy: float,
        species: str,
        background: float = 1e16,
    ) -> float:
        """Find the p-n junction depth where N(x) equals the background doping.

        The junction depth x_j is the deeper root of N(x_j) = N_background.

        Parameters
        ----------
        dose : float
            Implant dose in cm^{-2}.
        energy : float
            Implant energy in keV.
        species : str
            Ion species name.
        background : float
            Background (substrate) doping concentration in cm^{-3}.
            Default: 1e16 cm^{-3} (lightly doped p-type silicon).

        Returns
        -------
        float
            Junction depth in nanometres.  Returns 0 if peak concentration
            does not exceed *background*.
        """
        Rp, dRp = _interpolate_range(species, energy)

        # Peak concentration (at x=Rp)
        peak = dose * 1e-14 / (math.sqrt(2 * math.pi) * dRp)
        # dose [cm^{-2}] -> dose_nm2 [nm^{-2}] using 1 cm = 1e7 nm
        # dose_nm2 = dose * 1e-14;  peak [nm^{-3}] * 1e21 = cm^{-3}
        peak_cm3 = dose * 1e-14 / (math.sqrt(2 * math.pi) * dRp) * 1e21

        if peak_cm3 <= background:
            return 0.0

        # Solve N(x_j) = background  for x_j > Rp
        # Gaussian: dose_nm2/(sqrt(2pi)*dRp) * exp(-(x-Rp)^2/(2*dRp^2)) = bg_nm3
        dose_nm2 = dose * 1e-14
        bg_nm3 = background * 1e-21  # cm^{-3} -> nm^{-3}

        peak_nm3 = dose_nm2 / (math.sqrt(2 * math.pi) * dRp)
        if peak_nm3 <= bg_nm3:
            return 0.0

        # x_j = Rp + dRp * sqrt(2 * ln(peak / bg))
        x_j = Rp + dRp * math.sqrt(2.0 * math.log(peak_nm3 / bg_nm3))
        return x_j  # nm

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _gaussian_profile(
        x: "np.ndarray", dose: float, Rp: float, dRp: float
    ) -> "np.ndarray":
        """Gaussian concentration profile.

        N(x) = (Phi / (sqrt(2*pi) * dRp)) * exp(-(x - Rp)^2 / (2*dRp^2))

        where the result is in cm^{-3} (dose in cm^{-2}, depths in nm).
        """
        dose_nm2 = dose * 1e-14          # cm^{-2} -> nm^{-2}
        N_nm3 = (
            dose_nm2
            / (math.sqrt(2.0 * math.pi) * dRp)
            * np.exp(-0.5 * ((x - Rp) / dRp) ** 2)
        )
        return N_nm3 * 1e21              # nm^{-3} -> cm^{-3}

    @staticmethod
    def _pearsoniv_profile(
        x: "np.ndarray", dose: float, Rp: float, dRp: float, species: str
    ) -> "np.ndarray":
        """Pearson IV distribution profile with empirical skewness.

        For heavier ions the implant profile is skewed toward the surface
        (negative skewness).  The Pearson IV distribution captures this.

        Skewness parameters from:
          Hofker et al. (1973) and Gibbons tables (1975).
        """
        # Empirical skewness (gamma1) and excess kurtosis (gamma2) by species
        _skew = {
            "boron":      (-0.1, 0.3),
            "phosphorus": (-0.3, 0.5),
            "arsenic":    (-0.5, 1.0),
            "antimony":   (-0.7, 1.5),
        }
        gamma1, gamma2 = _skew.get(species, (0.0, 0.0))

        # Use scipy's Pearson III (gamma) as approximation when |gamma1| > 0.
        # A full Pearson IV requires specialized numerical integration; here we
        # use a shifted-gamma model which captures the leading-order asymmetry.
        if abs(gamma1) < 1e-6:
            return IonImplantationModel._gaussian_profile(x, dose, Rp, dRp)

        # Compute Pearson III parameters from moments
        # skewness: gamma1 = 2/sqrt(a), kurtosis: gamma2 = 6/a
        a = 4.0 / (gamma1 ** 2) if gamma1 != 0 else 1e6
        scale = dRp / math.sqrt(a)
        loc = Rp - a * scale  # mean = loc + a*scale

        # Evaluate PDF (returns values in nm^{-1} when x is in nm)
        pdf_vals = stats.gamma.pdf(x, a=a, loc=loc, scale=scale)

        dose_nm2 = dose * 1e-14
        N_nm3 = dose_nm2 * pdf_vals
        return N_nm3 * 1e21  # nm^{-3} -> cm^{-3}
