"""
SPICE netlist exporter for semiconductor process simulation results.

Converts process parameters (oxide thickness, doping, geometry) into
BSIM3v3 and BSIM4 SPICE model card parameters compatible with ngspice
and LTspice.

Reference:
  BSIM3v3.3.0 MOSFET Model User's Manual, UC Berkeley, 2005.
  BSIM4.8.0 MOSFET Model User's Manual, UC Berkeley, 2013.

Key physical equations:
  Flat-band voltage:
    VFB = phi_ms - Qf/Cox
    phi_ms ~ 0 for symmetric doping (simplified)

  Surface potential at inversion:
    2*phi_F = 2*(kT/q)*ln(NA/ni)
    ni = 1.45e10 cm^{-3} at 300 K

  Threshold voltage (long-channel, n-MOSFET):
    VTH0 = VFB + 2*phi_F + (1/Cox)*sqrt(2*q*eps_Si*NA*(2*phi_F))

  Gate oxide capacitance per unit area:
    Cox = eps_ox / tox    [F/cm^2]
    eps_ox = 3.9 * eps0 = 3.9 * 8.854e-12 F/m

  Surface mobility (empirical Caughey-Thomas-type):
    mu = mu_min + (mu_max - mu_min) / (1 + (NA/Nref)^alpha)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
# Physical constants                                                    #
# ------------------------------------------------------------------ #

_Q = 1.60218e-19  # C   elementary charge
_K_B = 1.38065e-23  # J/K Boltzmann constant
_T = 300.0  # K   room temperature
_EPS0 = 8.85419e-12  # F/m vacuum permittivity
_EPS_OX = 3.9 * _EPS0  # F/m SiO2 permittivity (relative: 3.9)
_EPS_SI = 11.7 * _EPS0  # F/m Si permittivity (relative: 11.7)
_NI = 1.45e10  # cm^{-3} intrinsic carrier concentration at 300 K
_kT_q = _K_B * _T / _Q  # V  thermal voltage (kT/q)


# ------------------------------------------------------------------ #
# Default BSIM parameter values                                        #
# ------------------------------------------------------------------ #

_BSIM3_DEFAULTS: dict[str, float] = {
    "MOBMOD": 1.0,
    "NFMOD": 0.0,
    "BINUNIT": 1.0,
    "LINT": 0.0,
    "WINT": 0.0,
    "TOXE": 3.0e-9,  # equivalent oxide thickness [m]
    "TOXP": 3.0e-9,
    "TOXM": 3.0e-9,
    "DTOX": 0.0,
    "XJ": 1.5e-7,  # source/drain junction depth [m]
    "NCH": 1.7e17,  # channel doping [cm^{-3}]
    "VTH0": 0.45,  # [V]
    "K1": 0.53,  # first-order body effect [V^{0.5}]
    "K2": -0.02,  # second-order body effect
    "K3": 80.0,
    "K3B": 0.0,
    "W0": 2.5e-6,
    "NLX": 1.74e-7,
    "DVT0W": 0.0,
    "DVT1W": 5.3e6,
    "DVT2W": -0.032,
    "DVT0": 2.2,
    "DVT1": 0.53,
    "DVT2": -0.032,
    "U0": 400.0,  # cm^2/(V*s) electron mobility
    "UA": 2.25e-9,
    "UB": 5.87e-19,
    "UC": -4.65e-11,
    "VSAT": 8.0e4,  # m/s saturation velocity
    "A0": 1.0,
    "AGS": 0.0,
    "B0": 0.0,
    "B1": 0.0,
    "KETA": -0.047,
    "A1": 0.0,
    "A2": 1.0,
    "RDSW": 200.0,
    "PRWG": 0.0,
    "PRWB": 0.0,
    "WR": 1.0,
    "DWG": 0.0,
    "DWB": 0.0,
    "VOFF": -0.08,
    "NFACTOR": 1.0,
    "CDSC": 2.4e-4,
    "CDSCD": 0.0,
    "CDSCB": 0.0,
    "CIT": 0.0,
    "ETA0": 0.08,
    "ETAB": -0.07,
    "DSUB": 0.56,
    "PCLM": 1.3,
    "PDIBLC1": 0.39,
    "PDIBLC2": 8.6e-4,
    "PDIBLCB": 0.0,
    "DROUT": 0.56,
    "PSCBE1": 8.14e8,
    "PSCBE2": 1.0e-7,
    "PVAG": 0.0,
    "DELTA": 0.01,
    "RSH": 0.0,
    "PRT": 0.0,
    "UTE": -1.5,
    "KT1": -0.11,
    "KT1L": 0.0,
    "KT2": 0.022,
    "UA1": 4.31e-9,
    "UB1": -7.61e-18,
    "UC1": -5.6e-11,
    "AT": 3.3e4,
    "WL": 0.0,
    "WLN": 1.0,
    "WW": 0.0,
    "WWN": 1.0,
    "WWL": 0.0,
    "LL": 0.0,
    "LLN": 1.0,
    "LW": 0.0,
    "LWN": 1.0,
    "LWL": 0.0,
    "CAPMOD": 2.0,
    "CGDO": 6.0e-10,
    "CGSO": 6.0e-10,
    "CGBO": 2.56e-11,
    "CJ": 1.7e-3,
    "PB": 1.0,
    "MJ": 0.5,
    "CJSW": 2.1e-10,
    "PBSW": 1.0,
    "MJSW": 0.43,
    "CJSWG": 3.3e-10,
    "PBSWG": 1.0,
    "MJSWG": 0.43,
    "CF": 0.0,
    "TNOM": 27.0,
    "JS": 1.0e-4,
    "JSW": 0.0,
    "NJ": 1.0,
    "XTI": 3.0,
    "IJTH": 0.1,
    "TPB": 0.0,
    "TPBSW": 0.0,
    "TPBSWG": 0.0,
    "TCJ": 0.0,
    "TCJSW": 0.0,
    "TCJSWG": 0.0,
    "VERSION": 3.24,
    "LEVEL": 8.0,
}


class SPICEExporter:
    """Convert semiconductor process simulation parameters to SPICE model cards.

    Supports BSIM3v3 parameter generation from physical process data.
    All key parameters are computed from first principles where possible;
    remaining parameters default to literature values for a 90-130 nm node.

    Parameters
    ----------
    model_level : {'bsim3', 'bsim4'}
        Target SPICE model level.  Currently full parameter computation is
        implemented for BSIM3v3; BSIM4 uses BSIM3 parameters with level
        adjusted.
    """

    def __init__(self, model_level: str = "bsim3") -> None:
        if model_level not in ("bsim3", "bsim4"):
            raise ValueError("model_level must be 'bsim3' or 'bsim4'.")
        self.model_level = model_level

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def process_to_spice(
        self,
        process_params: dict[str, Any],
        model_name: str = "nmos_model",
    ) -> dict[str, float]:
        """Compute SPICE model parameters from process data.

        Parameters
        ----------
        process_params : dict
            Process parameters:
              oxide_thickness_nm  : gate oxide thickness [nm]
              channel_length_nm   : effective channel length [nm]
              doping_concentration: substrate acceptor concentration [cm^{-3}]
              junction_depth_nm   : source/drain junction depth [nm]
        model_name : str
            Model identifier string (used in output files).

        Returns
        -------
        dict
            SPICE parameter dictionary.
        """
        tox_nm = float(process_params.get("oxide_thickness_nm", 8.5))
        Lg_nm = float(process_params.get("channel_length_nm", 90.0))
        NA = float(process_params.get("doping_concentration", 1e17))
        xj_nm = float(process_params.get("junction_depth_nm", 50.0))

        tox_m = tox_nm * 1e-9  # nm -> m
        _Lg_m = Lg_nm * 1e-9
        xj_m = xj_nm * 1e-9

        # Gate oxide capacitance per unit area
        Cox = _EPS_OX / tox_m  # F/m^2

        # Surface potential at strong inversion: 2*phi_F
        phi_F = _kT_q * math.log(NA / _NI)
        two_phi_F = 2.0 * phi_F

        # Flat-band voltage (simplified: metal-semiconductor work function ~ 0)
        VFB = -0.05  # V  (typical for n-MOSFET with n+ poly gate)

        # Threshold voltage (long-channel NMOS)
        # VTH0 = VFB + 2*phi_F + (1/Cox)*sqrt(2*q*eps_Si*NA*2*phi_F)
        # Cox here in F/m^2, charge term in C/m^2
        depletion_charge = math.sqrt(
            2.0 * _Q * _EPS_SI * NA * 1e6 * two_phi_F  # NA: cm^{-3} -> m^{-3}
        )
        VTH0 = VFB + two_phi_F + depletion_charge / Cox

        # First-order body-effect coefficient
        # K1 = sqrt(2*q*eps_Si*NA) / Cox  [V^{0.5}]
        K1 = math.sqrt(2.0 * _Q * _EPS_SI * NA * 1e6) / Cox

        # Electron mobility (empirical Caughey-Thomas model)
        # mu = mu_min + (mu_max - mu_min) / (1 + (NA/Nref)^alpha)
        U0 = _mobility_n(NA)

        # Short-channel CDSC (drain-induced charge sharing)
        # Empirical: CDSC ~ 2.4e-4 * (tox/2nm)^0.5 * (90nm/Lg)^0.5
        CDSC = 2.4e-4 * math.sqrt(tox_nm / 2.0) * math.sqrt(90.0 / Lg_nm)
        CDSCD = 0.0  # simplified

        # Build parameter dict starting from defaults
        params = dict(_BSIM3_DEFAULTS)
        params["TOXE"] = tox_m
        params["TOXP"] = tox_m
        params["TOXM"] = tox_m
        params["XJ"] = xj_m
        params["NCH"] = NA
        params["VTH0"] = VTH0
        params["K1"] = K1
        params["U0"] = U0
        params["CDSC"] = CDSC
        params["CDSCD"] = CDSCD

        # Scale VSAT and RDSW with channel length (short-channel enhancement)
        params["VSAT"] = 8.0e4 * (1.0 + 0.1 * (90.0 / max(Lg_nm, 10.0) - 1.0))
        params["RDSW"] = 200.0 * math.sqrt(Lg_nm / 90.0)

        if self.model_level == "bsim4":
            params["VERSION"] = 4.8
            params["LEVEL"] = 14.0

        return params

    def write_subckt(
        self,
        process_params: dict[str, Any],
        model_name: str = "nmos_model",
        output_path: str | Path = "mosfet.subckt",
    ) -> Path:
        """Write a SPICE subcircuit file for the MOSFET model.

        Parameters
        ----------
        process_params : dict
            Process parameters (see process_to_spice).
        model_name : str
            Model identifier.
        output_path : str or Path
            Output file path.

        Returns
        -------
        Path
            Path of the written file.
        """
        params = self.process_to_spice(process_params, model_name)
        output_path = Path(output_path)

        lines = [
            f"* MOSFET subcircuit generated by SemiYield",
            f"* Model: {model_name}",
            f"*",
            f".subckt {model_name} drain gate source bulk",
            f"M1 drain gate source bulk {model_name}_core L=100n W=1u",
            f".ends {model_name}",
            "",
        ]
        lines += self._model_card_lines(params, model_name + "_core")
        lines.append("")

        output_path.write_text("\n".join(lines))
        return output_path

    def write_model_card(
        self,
        process_params: dict[str, Any],
        model_name: str = "nmos_model",
        output_path: str | Path = "model.lib",
    ) -> Path:
        """Write a standalone SPICE .model card.

        Parameters
        ----------
        process_params : dict
            Process parameters.
        model_name : str
            Model identifier.
        output_path : str or Path
            Output file path.

        Returns
        -------
        Path
            Path of the written file.
        """
        params = self.process_to_spice(process_params, model_name)
        output_path = Path(output_path)

        lines = [
            f"* SPICE model card generated by SemiYield",
            f"* Model: {model_name}",
            "",
        ]
        lines += self._model_card_lines(params, model_name)
        lines.append("")

        output_path.write_text("\n".join(lines))
        return output_path

    def write_testbench(
        self,
        model_name: str = "nmos_model",
        output_path: str | Path = "testbench.sp",
        vdd: float = 1.8,
    ) -> Path:
        """Write a simple DC sweep testbench.

        Performs a VGS sweep from 0 to VDD with VDS = VDD/2 and measures
        drain current ID(M1).  Compatible with ngspice and LTspice.

        Parameters
        ----------
        model_name : str
            Model identifier (must match model card).
        output_path : str or Path
            Output file path.
        vdd : float
            Supply voltage in Volts.

        Returns
        -------
        Path
            Path of the written file.
        """
        output_path = Path(output_path)
        vds = vdd / 2.0

        lines = [
            f"* Testbench for {model_name} generated by SemiYield",
            f"* DC sweep: VGS from 0 to {vdd}V, VDS = {vds}V",
            "",
            f".include model.lib",
            "",
            f"M1 drain gate 0 0 {model_name} L=100n W=1u",
            f"VGS gate 0 DC 0",
            f"VDS drain 0 DC {vds}",
            "",
            f".dc VGS 0 {vdd} 0.01",
            f".probe I(VDS)",
            "",
            f".end",
        ]

        output_path.write_text("\n".join(lines))
        return output_path

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _model_card_lines(params: dict[str, float], model_name: str) -> list[str]:
        """Format a .model card block from a parameter dict."""
        level = int(params.get("LEVEL", 8))
        mtype = "nmos"
        lines = [f".model {model_name} {mtype} level={level}"]

        # Format parameters in groups of 4 per line for readability
        param_items = [(k, v) for k, v in params.items() if k not in ("LEVEL",)]
        for i in range(0, len(param_items), 4):
            chunk = param_items[i : i + 4]
            parts = []
            for k, v in chunk:
                if isinstance(v, float) and (abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0)):
                    parts.append(f"+ {k}={v:.6e}")
                else:
                    parts.append(f"+ {k}={v:.6g}")
            lines.append(" ".join(parts))

        return lines


def _mobility_n(NA: float) -> float:
    """Empirical electron mobility as a function of acceptor doping.

    Uses the Caughey-Thomas model (simplified):
      mu = mu_min + (mu_max - mu_min) / (1 + (NA/Nref)^alpha)

    Parameters
    ----------
    NA : float
        Acceptor doping concentration [cm^{-3}].

    Returns
    -------
    float
        Electron mobility in cm^2/(V*s).
    """
    mu_min = 65.0  # cm^2/(V*s)
    mu_max = 1414.0
    Nref = 9.68e16  # cm^{-3}
    alpha = 0.68

    return mu_min + (mu_max - mu_min) / (1.0 + (NA / Nref) ** alpha)
