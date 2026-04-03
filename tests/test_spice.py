"""
Tests for semiyield.spice.SPICEExporter.

Reference:
  BSIM3v3.3 User's Manual, UC Berkeley (2005).
  Threshold voltage formula: VTH0 = VFB + 2*phi_F + (1/Cox)*sqrt(2*q*eps_Si*NA*(2*phi_F))
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from semiyield.spice import SPICEExporter

# Physical constants (duplicated for test verification)
_Q = 1.60218e-19
_K_B = 1.38065e-23
_T = 300.0
_EPS0 = 8.85419e-12
_EPS_OX = 3.9 * _EPS0
_EPS_SI = 11.7 * _EPS0
_NI = 1.45e10
_kT_q = _K_B * _T / _Q


def _reference_vth0(tox_nm: float, NA: float) -> float:
    """Independent calculation of VTH0 for verification."""
    tox_m = tox_nm * 1e-9
    Cox = _EPS_OX / tox_m
    phi_F = _kT_q * math.log(NA / _NI)
    two_phi_F = 2.0 * phi_F
    VFB = -0.05
    depletion_charge = math.sqrt(2.0 * _Q * _EPS_SI * NA * 1e6 * two_phi_F)
    return VFB + two_phi_F + depletion_charge / Cox


class TestSPICEExporter:
    """Tests for SPICEExporter."""

    def _default_params(self) -> dict:
        return {
            "oxide_thickness_nm": 8.5,
            "channel_length_nm": 90.0,
            "doping_concentration": 1e17,
            "junction_depth_nm": 50.0,
        }

    def test_vth0_formula(self) -> None:
        """Computed VTH0 should match the independent analytical calculation."""
        exp = SPICEExporter()
        params = self._default_params()
        spice = exp.process_to_spice(params)
        expected = _reference_vth0(params["oxide_thickness_nm"], params["doping_concentration"])
        assert abs(spice["VTH0"] - expected) < 1e-6, (
            f"VTH0 mismatch: computed {spice['VTH0']:.6f} V, expected {expected:.6f} V"
        )

    def test_vth0_increases_with_doping(self) -> None:
        """Higher substrate doping should increase VTH0 (stronger body effect)."""
        exp = SPICEExporter()
        p_lo = dict(self._default_params(), doping_concentration=1e16)
        p_hi = dict(self._default_params(), doping_concentration=1e18)
        vth_lo = exp.process_to_spice(p_lo)["VTH0"]
        vth_hi = exp.process_to_spice(p_hi)["VTH0"]
        assert vth_hi > vth_lo, (
            f"VTH0 should increase with doping: low={vth_lo:.4f}, high={vth_hi:.4f}"
        )

    def test_tox_from_thickness(self) -> None:
        """TOXE in metres should equal oxide_thickness_nm / 1e9."""
        exp = SPICEExporter()
        tox_nm = 5.0
        params = dict(self._default_params(), oxide_thickness_nm=tox_nm)
        spice = exp.process_to_spice(params)
        expected_m = tox_nm * 1e-9
        assert abs(spice["TOXE"] - expected_m) < 1e-15, (
            f"TOXE={spice['TOXE']:.3e} m does not match {expected_m:.3e} m"
        )

    def test_spice_params_required_bsim_keys(self) -> None:
        """Output dict must contain essential BSIM3 parameters."""
        exp = SPICEExporter()
        spice = exp.process_to_spice(self._default_params())
        required_keys = ["VTH0", "K1", "U0", "TOXE", "NCH", "XJ", "CDSC", "VSAT", "RDSW"]
        for key in required_keys:
            assert key in spice, f"Missing required SPICE parameter: '{key}'"

    def test_write_subckt_creates_file(self) -> None:
        """write_subckt() must create a file at the given path."""
        exp = SPICEExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.subckt"
            result = exp.write_subckt(self._default_params(), "test_nmos", path)
            assert result.exists(), "write_subckt() must create the output file"

    def test_write_subckt_contains_subckt_keyword(self) -> None:
        """Written .subckt file must contain the .subckt keyword."""
        exp = SPICEExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.subckt"
            exp.write_subckt(self._default_params(), "my_model", path)
            content = path.read_text()
            assert ".subckt" in content, "File must contain '.subckt' keyword"
            assert "my_model" in content, "Model name must appear in .subckt file"

    def test_write_subckt_contains_ends_keyword(self) -> None:
        """Written .subckt file must contain the .ends keyword."""
        exp = SPICEExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.subckt"
            exp.write_subckt(self._default_params(), "nmos90", path)
            content = path.read_text()
            assert ".ends" in content, "File must contain '.ends' keyword"

    def test_write_model_card_creates_file(self) -> None:
        """write_model_card() must create a file."""
        exp = SPICEExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.lib"
            result = exp.write_model_card(self._default_params(), "nmos_test", path)
            assert result.exists(), "write_model_card() must create the output file"

    def test_write_model_card_contains_model_keyword(self) -> None:
        """Written model card must contain the .model keyword."""
        exp = SPICEExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.lib"
            exp.write_model_card(self._default_params(), "nmos_test", path)
            content = path.read_text()
            assert ".model" in content, "Model card must contain '.model' keyword"
            assert "nmos" in content.lower(), "Model card must reference nmos type"

    def test_write_testbench_contains_dc_statement(self) -> None:
        """Written testbench must contain a .dc sweep statement."""
        exp = SPICEExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tb.sp"
            exp.write_testbench("nmos_model", path)
            content = path.read_text()
            assert ".dc" in content.lower(), "Testbench must contain '.dc' sweep statement"

    def test_write_testbench_contains_end(self) -> None:
        """Written testbench must contain .end statement."""
        exp = SPICEExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tb.sp"
            exp.write_testbench("nmos_model", path)
            content = path.read_text()
            assert ".end" in content.lower(), "Testbench must contain '.end'"

    def test_mobility_decreases_with_doping(self) -> None:
        """U0 (mobility) should decrease with increasing substrate doping."""
        exp = SPICEExporter()
        p_lo = dict(self._default_params(), doping_concentration=1e15)
        p_hi = dict(self._default_params(), doping_concentration=1e18)
        u0_lo = exp.process_to_spice(p_lo)["U0"]
        u0_hi = exp.process_to_spice(p_hi)["U0"]
        assert u0_lo > u0_hi, (
            f"Mobility should decrease with doping: low doping={u0_lo:.1f}, high doping={u0_hi:.1f}"
        )

    def test_bsim4_level_parameter(self) -> None:
        """BSIM4 model should have LEVEL = 14."""
        exp = SPICEExporter(model_level="bsim4")
        spice = exp.process_to_spice(self._default_params())
        assert spice["LEVEL"] == 14.0, f"BSIM4 should have LEVEL=14, got {spice['LEVEL']}"

    def test_invalid_model_level(self) -> None:
        """Invalid model level should raise ValueError."""
        with pytest.raises(ValueError):
            SPICEExporter(model_level="spice2")  # type: ignore[arg-type]
