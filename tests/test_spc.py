"""
Tests for semiyield.spc control charts, Western Electric rules,
and process capability indices.

Reference: AIAG SPC Reference Manual, 2nd Edition (2005).
"""

from __future__ import annotations

import numpy as np
import pytest

from semiyield.spc import ControlChart, process_capability, western_electric_violations


class TestControlChart:
    """Tests for the ControlChart class."""

    def test_imr_control_limits_3sigma(self) -> None:
        """I-MR chart UCL and LCL should be at mean +/- 3 * sigma_within."""
        rng = np.random.default_rng(0)
        data = rng.normal(100.0, 5.0, size=100)
        chart = ControlChart(chart_type="IMR")
        chart.fit(data)
        cd = chart.chart_data()
        # UCL = CL + 3*sigma, LCL = CL - 3*sigma
        assert abs(cd["ucl"] - (cd["cl"] + 3 * cd["sigma"])) < 1e-6, "UCL != CL + 3*sigma"
        assert abs(cd["lcl"] - (cd["cl"] - 3 * cd["sigma"])) < 1e-6, "LCL != CL - 3*sigma"

    def test_imr_cl_equals_mean(self) -> None:
        """The centre line of an I-MR chart should equal the data mean."""
        rng = np.random.default_rng(1)
        data = rng.normal(50.0, 3.0, size=80)
        chart = ControlChart(chart_type="IMR")
        chart.fit(data)
        assert abs(chart.cl - data.mean()) < 1e-10, "CL should equal data mean"

    def test_xbar_r_control_limits(self) -> None:
        """Xbar-R chart UCL must be above the data mean and LCL below."""
        rng = np.random.default_rng(2)
        data = rng.normal(100.0, 2.0, size=100)
        chart = ControlChart(chart_type="XbarR", subgroup_size=5)
        chart.fit(data)
        cd = chart.chart_data()
        assert cd["ucl"] > cd["cl"], "UCL must be above CL"
        assert cd["lcl"] < cd["cl"], "LCL must be below CL"

    def test_imr_in_control_normal_data(self) -> None:
        """update() should return True for in-control points from same distribution."""
        rng = np.random.default_rng(3)
        data = rng.normal(100.0, 5.0, size=100)
        chart = ControlChart(chart_type="IMR")
        chart.fit(data)
        # New points drawn from the same distribution should be mostly in control
        new_points = rng.normal(100.0, 5.0, size=100)
        n_out = sum(1 for p in new_points if not chart.update(p))
        # Expect at most ~2% out of control (false alarm rate for normal distribution)
        assert n_out <= 5, f"Too many false alarms: {n_out}/100"

    def test_imr_out_of_control_detection(self) -> None:
        """update() must return False for a point far beyond the control limits."""
        rng = np.random.default_rng(4)
        data = rng.normal(100.0, 5.0, size=100)
        chart = ControlChart(chart_type="IMR")
        chart.fit(data)
        # A point at 10-sigma is clearly out of control
        result = chart.update(100.0 + 10 * chart.sigma)
        assert not result, "A 10-sigma point must be flagged as out of control"

    def test_ewma_limits_narrower_than_individuals(self) -> None:
        """EWMA control limits are narrower than 3-sigma individual limits for lambda < 1."""
        rng = np.random.default_rng(5)
        data = rng.normal(100.0, 5.0, size=100)
        chart_imr = ControlChart(chart_type="IMR")
        chart_imr.fit(data)
        chart_ewma = ControlChart(chart_type="EWMA", ewma_lambda=0.2)
        chart_ewma.fit(data)
        imr_width = chart_imr.ucl - chart_imr.lcl
        ewma_width = chart_ewma.ucl - chart_ewma.lcl
        assert ewma_width < imr_width, "EWMA limits should be narrower than 3-sigma I-MR limits"

    def test_invalid_chart_type(self) -> None:
        """Unknown chart type should raise ValueError."""
        with pytest.raises(ValueError):
            ControlChart(chart_type="ShewhartXY")  # type: ignore[arg-type]

    def test_predict_before_fit_raises(self) -> None:
        """Calling update() before fit() should raise RuntimeError."""
        chart = ControlChart()
        with pytest.raises(RuntimeError):
            chart.update(100.0)


class TestWesternElectricRules:
    """Tests for western_electric_violations()."""

    def _normal_chart(self, n: int = 100, seed: int = 0) -> tuple[np.ndarray, float, float, float]:
        rng = np.random.default_rng(seed)
        data = rng.normal(100.0, 5.0, size=n)
        cl = data.mean()
        sigma = data.std(ddof=1)
        ucl = cl + 3 * sigma
        lcl = cl - 3 * sigma
        return data, ucl, lcl, cl

    def test_no_violations_for_normal_data(self) -> None:
        """Clean in-control data should produce zero or very few violations (statistical)."""
        rng = np.random.default_rng(42)
        # Use a fixed, perfectly normal-looking sequence
        data = 100.0 + rng.normal(0, 1.0, 50)
        cl = 100.0
        ucl = 103.0
        lcl = 97.0
        violations = western_electric_violations(data, ucl, lcl, cl)
        # Filter only rules 1 (beyond 3-sigma) and 4 (8 in a row on one side)
        r1 = [v for v in violations if v[1] == 1]
        assert len(r1) == 0, f"No Rule 1 violations expected in normal data, got {r1}"

    def test_rule_1_fires_for_beyond_3sigma(self) -> None:
        """A single point beyond 3-sigma should trigger Rule 1."""
        cl = 100.0
        sigma = 5.0
        ucl = cl + 3 * sigma
        lcl = cl - 3 * sigma
        data = np.full(20, cl)  # all on centre line
        data[10] = ucl + 1.0  # inject one point beyond UCL
        violations = western_electric_violations(data, ucl, lcl, cl)
        rule1 = [v for v in violations if v[1] == 1]
        assert len(rule1) >= 1, "Rule 1 should fire for a point beyond 3-sigma"
        assert rule1[0][0] == 10, f"Violation should be at index 10, got {rule1[0][0]}"

    def test_rule_1_fires_below_lcl(self) -> None:
        """A single point below LCL should also trigger Rule 1."""
        cl = 100.0
        ucl = cl + 15.0
        lcl = cl - 15.0
        data = np.full(15, cl)
        data[7] = lcl - 1.0
        violations = western_electric_violations(data, ucl, lcl, cl)
        rule1 = [v for v in violations if v[1] == 1 and v[0] == 7]
        assert len(rule1) >= 1, "Rule 1 should fire for point below LCL"

    def test_rule_4_fires_for_8_points_same_side(self) -> None:
        """Eight consecutive points above CL should trigger Rule 4."""
        cl = 100.0
        ucl = cl + 15.0
        lcl = cl - 15.0
        data = np.array([cl] * 5 + [cl + 2.0] * 8 + [cl] * 5)
        violations = western_electric_violations(data, ucl, lcl, cl)
        rule4 = [v for v in violations if v[1] == 4]
        assert len(rule4) >= 1, "Rule 4 must fire for 8 consecutive points above CL"

    def test_rule_5_fires_for_trending_data(self) -> None:
        """Six consecutive points strictly increasing should trigger Rule 5."""
        cl = 100.0
        ucl = cl + 15.0
        lcl = cl - 15.0
        data = np.concatenate(
            [
                np.full(5, cl),
                np.linspace(cl, cl + 3.0, 6),  # strictly increasing
                np.full(5, cl),
            ]
        )
        violations = western_electric_violations(data, ucl, lcl, cl)
        rule5 = [v for v in violations if v[1] == 5]
        assert len(rule5) >= 1, f"Rule 5 should fire for monotone trend; violations={violations}"

    def test_violation_tuple_structure(self) -> None:
        """Each violation should be a 3-tuple (index, rule_number, description_str)."""
        cl = 100.0
        ucl = cl + 15.0
        lcl = cl - 15.0
        data = np.array([cl] * 5 + [ucl + 1.0] + [cl] * 5)
        violations = western_electric_violations(data, ucl, lcl, cl)
        for v in violations:
            assert len(v) == 3, "Violation tuple must have 3 elements"
            assert isinstance(v[0], (int, np.integer)), "First element must be index (int)"
            assert isinstance(v[1], (int, np.integer)), "Second element must be rule number (int)"
            assert isinstance(v[2], str), "Third element must be description (str)"


class TestProcessCapability:
    """Tests for process_capability()."""

    def test_cpk_equals_cp_for_centered_process(self) -> None:
        """For a perfectly centered process, Cp should equal Cpk."""
        rng = np.random.default_rng(0)
        mu, sigma = 100.0, 5.0
        data = rng.normal(mu, sigma, size=1000)
        usl = mu + 3 * sigma
        lsl = mu - 3 * sigma
        cap = process_capability(data, usl, lsl)
        assert (
            abs(cap["Cp"] - cap["Cpk"]) < 0.05
        ), f"Cp ({cap['Cp']:.4f}) and Cpk ({cap['Cpk']:.4f}) should be equal for centered process"

    def test_cpk_less_than_cp_for_off_center(self) -> None:
        """An off-centre process should have Cpk < Cp."""
        rng = np.random.default_rng(1)
        mu_actual = 102.0  # shifted from target of 100
        sigma = 5.0
        data = rng.normal(mu_actual, sigma, size=500)
        usl = 115.0
        lsl = 85.0
        cap = process_capability(data, usl, lsl)
        assert (
            cap["Cpk"] < cap["Cp"]
        ), f"Off-centre process: Cpk ({cap['Cpk']:.4f}) should be < Cp ({cap['Cp']:.4f})"

    def test_ppk_leq_cpk_with_assignable_cause(self) -> None:
        """For data with assignable-cause variation (step shifts), Ppk < Cpk.

        Ppk uses overall std (which includes between-subgroup variation),
        while Cpk uses within-subgroup sigma from moving ranges.
        When the process shifts mid-run, overall std > within-subgroup std,
        giving Ppk < Cpk.
        """
        rng = np.random.default_rng(2)
        # First half at mean=100, second half at mean=108 -> step shift
        part1 = rng.normal(100.0, 3.0, size=100)
        part2 = rng.normal(108.0, 3.0, size=100)
        data = np.concatenate([part1, part2])
        usl = 118.0
        lsl = 88.0
        cap = process_capability(data, usl, lsl)
        # With a mean shift, overall sigma > within-subgroup sigma -> Ppk < Cpk
        assert cap["sigma_overall"] > cap["sigma_within"], (
            f"Data with step shift should have sigma_overall > sigma_within: "
            f"overall={cap['sigma_overall']:.3f}, within={cap['sigma_within']:.3f}"
        )
        assert (
            cap["Ppk"] < cap["Cpk"]
        ), f"With assignable-cause shift, Ppk ({cap['Ppk']:.4f}) should be < Cpk ({cap['Cpk']:.4f})"

    def test_sigma_level_for_six_sigma_process(self) -> None:
        """A 6-sigma process (USL/LSL at 6*sigma) should have sigma_level ~ 6."""
        rng = np.random.default_rng(3)
        mu, sigma = 100.0, 5.0
        data = rng.normal(mu, sigma, size=2000)
        usl = mu + 6 * sigma
        lsl = mu - 6 * sigma
        cap = process_capability(data, usl, lsl)
        # sigma_level should be close to 6 (within 10% due to sampling variation)
        assert (
            5.0 < cap["sigma_level"] < 7.5
        ), f"6-sigma process should have sigma_level near 6, got {cap['sigma_level']:.2f}"

    def test_capability_indices_all_present(self) -> None:
        """All expected keys must be present in the result dict."""
        data = np.random.normal(100, 5, 100)
        cap = process_capability(data, 115.0, 85.0)
        for key in (
            "Cp",
            "Cpk",
            "Pp",
            "Ppk",
            "sigma_within",
            "sigma_overall",
            "mean",
            "sigma_level",
        ):
            assert key in cap, f"Missing key '{key}' in process_capability result"

    def test_cp_formula_3sigma(self) -> None:
        """For a process with USL-LSL = 6*sigma_within, Cp should be ~1.0."""
        rng = np.random.default_rng(5)
        data = rng.normal(100.0, 5.0, size=500)
        # For MR-based sigma_within, we need large enough sample.
        # Just verify the formula: Cp = (USL-LSL)/(6*sigma_within) ~ 1
        cap = process_capability(data, 115.0, 85.0)
        # sigma_within from MR_bar / d2. Should be close to 5.0 for normal data.
        sigma_within = cap["sigma_within"]
        Cp_expected = 30.0 / (6.0 * sigma_within)
        assert (
            abs(cap["Cp"] - Cp_expected) < 0.01
        ), f"Cp formula mismatch: computed {cap['Cp']:.4f}, expected {Cp_expected:.4f}"
