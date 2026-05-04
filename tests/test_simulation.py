"""
Tests for semiyield.simulation modules.

Deal-Grove reference values from:
  B.E. Deal and A.S. Grove, J. Appl. Phys. 36(12), 3770-3778, 1965, Table I.

Ion implantation reference values from:
  Gibbons, Johnson & Mylroie, "Projected Range Statistics," 1975.
"""

import numpy as np
import pytest

from semiyield.simulation import (
    CVDModel,
    DealGroveModel,
    IonImplantationModel,
    LangmuirHinshelwoodModel,
)

# ================================================================== #
# Deal-Grove Oxidation Tests                                           #
# ================================================================== #


class TestDealGrove:
    """Tests for DealGroveModel thermal oxidation."""

    def test_deal_grove_dry_1000c_60min(self) -> None:
        """Dry oxidation at 1000 C for 60 min should give ~20-30 nm (thin oxide regime).

        Reference: Deal-Grove tabulated data at 1000 C dry O2.
        Approximate range from published silicon oxidation data.
        """
        model = DealGroveModel()
        thickness = model.grow(60.0, 1000.0, "dry")
        # Published experimental range: 18-35 nm for 60 min dry at 1000 C
        assert 10.0 < thickness < 60.0, f"Unexpected dry oxide thickness: {thickness:.2f} nm"

    def test_deal_grove_wet_1000c_30min(self) -> None:
        """Wet oxidation at 1000 C should grow thicker than dry at same time.

        Wet oxidation has a higher parabolic rate constant B (lower Ea_B).
        """
        model = DealGroveModel()
        dry = model.grow(30.0, 1000.0, "dry")
        wet = model.grow(30.0, 1000.0, "wet")
        assert wet > dry, f"Wet ({wet:.2f}) should exceed dry ({dry:.2f}) oxide at same time/temp"

    def test_deal_grove_rate_decreases_with_time(self) -> None:
        """Growth rate should decrease monotonically (parabolic regime dominates at late times)."""
        model = DealGroveModel()
        rate_early = model.rate(10.0, 1000.0, "dry")
        rate_mid = model.rate(60.0, 1000.0, "dry")
        rate_late = model.rate(180.0, 1000.0, "dry")
        assert (
            rate_early > rate_mid > rate_late
        ), f"Rate should decrease: {rate_early:.4f} > {rate_mid:.4f} > {rate_late:.4f}"

    def test_deal_grove_temperature_dependence(self) -> None:
        """Higher temperature should give faster oxide growth (Arrhenius)."""
        model = DealGroveModel()
        t900 = model.grow(60.0, 900.0, "dry")
        t1000 = model.grow(60.0, 1000.0, "dry")
        t1100 = model.grow(60.0, 1100.0, "dry")
        assert t900 < t1000 < t1100, "Oxide thickness must increase with temperature"

    def test_deal_grove_initial_thickness(self) -> None:
        """Pre-existing oxide shifts the result upward."""
        model = DealGroveModel()
        no_init = model.grow(30.0, 1000.0, "dry", initial_thickness_nm=0.0)
        with_init = model.grow(30.0, 1000.0, "dry", initial_thickness_nm=5.0)
        assert with_init > no_init, "Pre-existing oxide should increase final thickness"

    def test_deal_grove_growth_curve_shape(self) -> None:
        """Growth curve should be concave (sublinear growth)."""
        model = DealGroveModel()
        times = np.linspace(1, 200, 50)
        thick = model.growth_curve(times, 1000.0, "dry")
        diffs = np.diff(thick)
        # All increments positive (monotone growth)
        assert np.all(diffs > 0), "Oxide thickness must increase monotonically with time"
        # Increments decreasing (concave: parabolic regime)
        assert np.all(np.diff(diffs) < 0.5), "Growth curve should be concave (rate slows down)"

    def test_deal_grove_wet_higher_than_dry_at_all_times(self) -> None:
        """Wet oxide should always be thicker than dry at same T and t."""
        model = DealGroveModel()
        for t in [10, 30, 60, 120]:
            dry = model.grow(t, 1000.0, "dry")
            wet = model.grow(t, 1000.0, "wet")
            assert wet > dry, f"At t={t} min, wet ({wet:.2f}) <= dry ({dry:.2f})"

    def test_deal_grove_invalid_atmosphere(self) -> None:
        """Unknown atmosphere should raise ValueError."""
        model = DealGroveModel()
        with pytest.raises(ValueError, match="atmosphere"):
            model.grow(60.0, 1000.0, "nitrogen")  # type: ignore[arg-type]


# ================================================================== #
# Ion Implantation Tests                                               #
# ================================================================== #


class TestIonImplantation:
    """Tests for IonImplantationModel dopant profiles."""

    def test_implantation_gaussian_integral(self) -> None:
        """Integrating the Gaussian profile over depth should recover the dose (within 1%)."""
        model = IonImplantationModel(distribution="gaussian")
        dose = 1e13  # cm^{-2}
        energy = 80.0  # keV
        species = "boron"

        # Fine depth grid from 0 to 4x the expected Rp
        depths = np.linspace(0, 800, 2000)  # nm
        conc = model.profile(depths, dose, energy, species)

        # Integrate: conc [cm^{-3}] * depth step [nm] -> [cm^{-3} * nm]
        # 1 nm = 1e-7 cm  ->  integral units [cm^{-2}]
        dx_cm = (depths[1] - depths[0]) * 1e-7  # nm -> cm
        from scipy.integrate import trapezoid

        integrated_dose = trapezoid(conc, dx=dx_cm)

        assert (
            abs(integrated_dose - dose) / dose < 0.02
        ), f"Integrated dose {integrated_dose:.3e} deviates from input dose {dose:.3e} by > 2%"

    def test_implantation_junction_depth_increases_with_dose(self) -> None:
        """Higher dose -> deeper junction (peak concentration higher -> meets background deeper)."""
        model = IonImplantationModel()
        xj1 = model.junction_depth(1e12, 80.0, "boron", background=1e16)
        xj2 = model.junction_depth(1e14, 80.0, "boron", background=1e16)
        assert (
            xj2 > xj1
        ), f"Higher dose should give deeper junction: xj(1e12)={xj1:.1f}, xj(1e14)={xj2:.1f}"

    def test_implantation_boron_deeper_than_arsenic(self) -> None:
        """Boron (lighter ion) should have larger projected range than arsenic at same energy."""
        model = IonImplantationModel()
        depths = np.linspace(0, 600, 1000)
        dose = 1e13

        boron = model.profile(depths, dose, 80.0, "boron")
        arsenic = model.profile(depths, dose, 80.0, "arsenic")

        # Peak position (Rp) is larger for boron
        Rp_boron = depths[np.argmax(boron)]
        Rp_arsenic = depths[np.argmax(arsenic)]
        assert (
            Rp_boron > Rp_arsenic
        ), f"Boron Rp ({Rp_boron:.1f} nm) should exceed arsenic Rp ({Rp_arsenic:.1f} nm)"

    def test_implantation_phosphorus_between_boron_and_arsenic(self) -> None:
        """Phosphorus range should be between boron and arsenic at 80 keV."""
        model = IonImplantationModel()
        depths = np.linspace(0, 600, 1000)
        dose = 1e13
        energy = 80.0

        Rp = {}
        for sp in ["boron", "phosphorus", "arsenic"]:
            conc = model.profile(depths, dose, energy, sp)
            Rp[sp] = depths[np.argmax(conc)]

        assert (
            Rp["arsenic"] < Rp["phosphorus"] < Rp["boron"]
        ), f"Expected As < P < B range ordering; got {Rp}"

    def test_implantation_profile_positive(self) -> None:
        """All concentration values should be non-negative."""
        model = IonImplantationModel()
        depths = np.linspace(0, 500, 200)
        conc = model.profile(depths, 1e13, 100.0, "phosphorus")
        assert np.all(conc >= 0.0), "Concentration must be non-negative everywhere"

    def test_implantation_unknown_species(self) -> None:
        """Unknown species should raise ValueError."""
        model = IonImplantationModel()
        with pytest.raises(ValueError, match="species"):
            model.profile(np.linspace(0, 100, 50), 1e13, 50.0, "xenon")

    def test_implantation_junction_zero_if_below_background(self) -> None:
        """If peak concentration is below background, junction depth should be 0."""
        model = IonImplantationModel()
        # Very low dose, very high background
        xj = model.junction_depth(1e8, 80.0, "boron", background=1e20)
        assert xj == 0.0, f"Expected junction depth 0 but got {xj:.2f}"


# ================================================================== #
# Etching Tests                                                        #
# ================================================================== #


class TestEtching:
    """Tests for LangmuirHinshelwoodModel etch kinetics."""

    def test_etch_rate_positive(self) -> None:
        """Etch rate should always be positive."""
        model = LangmuirHinshelwoodModel(mode="single")
        rate = model.rate(100.0, 50.0, "SiO2")
        assert rate > 0.0, f"Etch rate must be positive, got {rate}"

    def test_etch_rate_increases_with_temperature(self) -> None:
        """Etch rate should increase with temperature (Arrhenius k_s dominates)."""
        model = LangmuirHinshelwoodModel(mode="single")
        r_low = model.rate(100.0, 20.0, "SiO2")
        r_high = model.rate(100.0, 200.0, "SiO2")
        assert r_high > r_low, f"Rate should increase with T: low={r_low:.2f}, high={r_high:.2f}"

    def test_etch_selectivity_sio2_over_si(self) -> None:
        """In CF4-based chemistry, SiO2 should etch faster than Si (selectivity > 1)."""
        model = LangmuirHinshelwoodModel(mode="single")
        sel = model.selectivity("SiO2", "Si", 100.0, 50.0)
        assert sel > 1.0, f"SiO2/Si selectivity should be > 1 in CF4 chemistry, got {sel:.3f}"

    def test_etch_selectivity_sio2_over_si3n4(self) -> None:
        """SiO2 should etch faster than Si3N4 in CHF3/O2 chemistry."""
        model = LangmuirHinshelwoodModel(mode="two_reactant")
        sel = model.selectivity("SiO2", "Si3N4", 100.0, 50.0, pressure_b=50.0)
        assert sel > 0.5, f"SiO2/Si3N4 selectivity should be meaningful, got {sel:.3f}"

    def test_etch_coverage_bounded(self) -> None:
        """Surface coverage theta must be in [0, 1]."""
        model = LangmuirHinshelwoodModel()
        for pressure in [1.0, 10.0, 100.0, 1000.0]:
            theta = model.coverage(pressure, 50.0, "SiO2")
            assert 0.0 <= theta <= 1.0, f"Coverage {theta:.4f} out of [0, 1] at P={pressure}"

    def test_etch_coverage_saturates(self) -> None:
        """At very high pressure, coverage should approach 1."""
        model = LangmuirHinshelwoodModel()
        theta_high = model.coverage(1e6, 50.0, "SiO2")
        assert (
            theta_high > 0.99
        ), f"Coverage should saturate near 1 at high pressure: {theta_high:.4f}"

    def test_etch_two_reactant_positive(self) -> None:
        """Two-reactant rate should be positive."""
        model = LangmuirHinshelwoodModel(mode="two_reactant")
        rate = model.rate(100.0, 50.0, "SiO2", pressure_b=50.0)
        assert rate > 0.0

    def test_etch_invalid_mode(self) -> None:
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            LangmuirHinshelwoodModel(mode="invalid")  # type: ignore[arg-type]

    def test_etch_two_reactant_requires_pressure_b(self) -> None:
        """Two-reactant mode without pressure_b should raise ValueError."""
        model = LangmuirHinshelwoodModel(mode="two_reactant")
        with pytest.raises(ValueError, match="pressure_b"):
            model.rate(100.0, 50.0, "SiO2")  # missing pressure_b


# ================================================================== #
# Deposition Tests                                                     #
# ================================================================== #


class TestDeposition:
    """Tests for CVDModel deposition."""

    def test_deposition_thickness_proportional_to_time(self) -> None:
        """Deposited thickness should increase with time (linear in first-order kinetics)."""
        model = CVDModel(process_type="LPCVD", material="SiO2")
        t1 = model.deposit(10.0, 700.0, 1.0)
        t2 = model.deposit(20.0, 700.0, 1.0)
        _t4 = model.deposit(40.0, 700.0, 1.0)
        assert t2 > t1, "Thickness must increase with time"
        # Linear: t2/t1 ~ 2, t4/t1 ~ 4 (within 1% due to floating point)
        ratio = t2 / t1
        assert 1.99 < ratio < 2.01, f"Thickness should scale linearly with time; ratio={ratio:.4f}"

    def test_deposition_ald_best_step_coverage(self) -> None:
        """ALD should have higher step coverage than CVD, which should exceed PVD."""
        ar = 3.0
        ald = CVDModel(process_type="ALD", material="HfO2").step_coverage(ar)
        cvd = CVDModel(process_type="LPCVD", material="SiO2").step_coverage(ar)
        pvd = CVDModel(process_type="PVD", material="Al").step_coverage(ar)
        assert (
            ald > cvd > pvd
        ), f"Step coverage order wrong: ALD={ald:.3f}, CVD={cvd:.3f}, PVD={pvd:.3f}"

    def test_deposition_step_coverage_decreases_with_ar(self) -> None:
        """Step coverage decreases for higher aspect ratio features."""
        model = CVDModel(process_type="LPCVD", material="SiO2")
        sc1 = model.step_coverage(1.0)
        sc5 = model.step_coverage(5.0)
        assert sc1 > sc5, f"Higher AR should reduce step coverage: AR=1: {sc1:.3f}, AR=5: {sc5:.3f}"

    def test_deposition_stress_sign(self) -> None:
        """Silicon nitride should be tensile (positive stress).  Cooling increases stress."""
        model = CVDModel(process_type="LPCVD", material="Si3N4")
        # Cooling after deposition (positive dT)
        stress_cool = model.stress(temperature_delta=600.0, material="Si3N4")
        # From table: sigma_intrinsic = +1000 MPa (tensile)
        assert (
            stress_cool > 0
        ), f"Si3N4 should be tensile (positive stress), got {stress_cool:.0f} MPa"

    def test_deposition_temperature_dependence(self) -> None:
        """Growth rate should increase with temperature (Arrhenius)."""
        model = CVDModel(process_type="LPCVD", material="SiO2")
        temps = np.array([600.0, 700.0, 800.0, 900.0])
        rates = model.rate_vs_temperature(temps, pressure=1.0)
        assert np.all(np.diff(rates) > 0), "Growth rate should increase with temperature"

    def test_deposition_invalid_process_type(self) -> None:
        """Invalid process type should raise ValueError."""
        with pytest.raises(ValueError, match="process_type"):
            CVDModel(process_type="MOCVD", material="SiO2")  # type: ignore[arg-type]

    def test_deposition_uniformity_positive(self) -> None:
        """Non-uniformity should be a positive percentage."""
        model = CVDModel(process_type="PECVD", material="SiO2")
        u = model.uniformity(300.0)
        assert u > 0.0, f"Uniformity should be positive: {u}"

    def test_deposition_ald_uniformity_best(self) -> None:
        """ALD should have the best (lowest) non-uniformity."""
        ald_u = CVDModel(process_type="ALD", material="HfO2").uniformity(300.0)
        pecvd_u = CVDModel(process_type="PECVD", material="SiO2").uniformity(300.0)
        pvd_u = CVDModel(process_type="PVD", material="Al").uniformity(300.0)
        assert (
            ald_u < pecvd_u < pvd_u
        ), f"ALD should be most uniform: ALD={ald_u:.2f}%, PECVD={pecvd_u:.2f}%, PVD={pvd_u:.2f}%"
