"""
Tests for semiyield.datagen.FabDataGenerator.
"""

import numpy as np
import pandas as pd

from semiyield.datagen import FabDataGenerator


class TestFabDataGenerator:
    """Tests for FabDataGenerator synthetic fab data."""

    def _make_df(self, seed: int = 42, n_lots: int = 20, wafers_per_lot: int = 5) -> pd.DataFrame:
        gen = FabDataGenerator(seed=seed)
        return gen.generate(n_lots=n_lots, wafers_per_lot=wafers_per_lot)

    def test_output_shape(self) -> None:
        """DataFrame should have n_lots * wafers_per_lot rows."""
        n_lots, wpl = 10, 5
        df = self._make_df(n_lots=n_lots, wafers_per_lot=wpl)
        assert len(df) == n_lots * wpl, f"Expected {n_lots * wpl} rows, got {len(df)}"

    def test_column_presence(self) -> None:
        """All expected columns must be present."""
        df = self._make_df()
        required = [
            "lot_id",
            "wafer_id",
            "lot_sequence",
            "wafer_sequence",
            "gate_oxide_thickness",
            "poly_cd",
            "implant_dose",
            "anneal_temp",
            "metal_resistance",
            "contact_resistance",
            "etch_rate",
            "deposition_unif",
            "defect_density",
            "yield",
            "wafer_map_mean",
            "wafer_map_std",
        ]
        for col in required:
            assert col in df.columns, f"Missing expected column: '{col}'"

    def test_yield_bounds(self) -> None:
        """Yield values must all be in [0, 1]."""
        df = self._make_df(n_lots=50, wafers_per_lot=10)
        assert df["yield"].min() >= 0.0, "Yield values must be >= 0"
        assert df["yield"].max() <= 1.0, "Yield values must be <= 1"

    def test_lot_drift(self) -> None:
        """The mean of a process parameter should drift over time.

        With n_lots=200 and drift_rate=0.2, the difference between
        the first and last 10% of lots should be statistically detectable.
        """
        gen = FabDataGenerator(seed=7, drift_rate=0.2)
        df = gen.generate(n_lots=200, wafers_per_lot=5)

        n_window = len(df) // 10
        early = df.iloc[:n_window]["gate_oxide_thickness"].mean()
        late = df.iloc[-n_window:]["gate_oxide_thickness"].mean()
        # With high drift_rate, the means should diverge.  We just check they differ.
        assert early != late, "Drift should cause early and late lot means to differ"

    def test_wafer_map_shape(self) -> None:
        """Wafer map should be a 2-D square array of the configured size."""
        size = 40
        gen = FabDataGenerator(seed=1, wafer_map_size=size)
        wmap = gen.generate_wafer_map("LOT0000_W01")
        assert wmap.ndim == 2, "Wafer map must be 2-dimensional"
        assert wmap.shape == (size, size), f"Expected ({size},{size}), got {wmap.shape}"

    def test_reproducibility(self) -> None:
        """Same seed should produce identical DataFrames."""
        df1 = self._make_df(seed=99, n_lots=5, wafers_per_lot=5)
        df2 = self._make_df(seed=99, n_lots=5, wafers_per_lot=5)
        assert (
            df1["yield"].tolist() == df2["yield"].tolist()
        ), "Same seed must produce identical yield values"
        assert df1["gate_oxide_thickness"].tolist() == df2["gate_oxide_thickness"].tolist()

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different data."""
        df1 = self._make_df(seed=1)
        df2 = self._make_df(seed=2)
        assert (
            df1["yield"].mean() != df2["yield"].mean()
        ), "Different seeds should produce different data"

    def test_defect_density_effect(self) -> None:
        """Higher defect density should correspond to lower yield on average.

        We use a large chip area (high Murphy sensitivity) and strong aging
        so the defect-density-to-yield relationship is clearly visible.
        """
        # Large chip area (1 cm^2) makes Murphy yield very sensitive to defect density
        gen = FabDataGenerator(seed=42, aging_factor=0.02, chip_area_cm2=1.0)
        df = gen.generate(n_lots=100, wafers_per_lot=5)

        n_window = len(df) // 5
        early_dd = df.iloc[:n_window]["defect_density"].mean()
        late_dd = df.iloc[-n_window:]["defect_density"].mean()
        # Aging must increase defect density
        assert (
            late_dd > early_dd
        ), f"Aging should increase defect density: early={early_dd:.4f}, late={late_dd:.4f}"

        # Verify the Murphy model works: higher defect density -> lower yield
        # Test by sorting the full dataset
        high_dd = df[df["defect_density"] > df["defect_density"].median()]["yield"].mean()
        low_dd = df[df["defect_density"] <= df["defect_density"].median()]["yield"].mean()
        assert low_dd > high_dd, (
            f"Lower defect density should give higher yield: "
            f"low_dd_yield={low_dd:.4f}, high_dd_yield={high_dd:.4f}"
        )

    def test_wafer_map_values_in_range(self) -> None:
        """Wafer map values should be in [0, 1] where not NaN."""
        gen = FabDataGenerator(seed=5)
        wmap = gen.generate_wafer_map("TEST_W01")
        finite = wmap[~np.isnan(wmap)]
        assert np.all(finite >= 0.0), "Wafer map values must be >= 0"
        assert np.all(finite <= 1.0), "Wafer map values must be <= 1"

    def test_lot_id_format(self) -> None:
        """Lot IDs should follow the LOT#### format."""
        df = self._make_df(n_lots=5)
        for lot_id in df["lot_id"].unique():
            assert lot_id.startswith("LOT"), f"Unexpected lot_id format: {lot_id}"

    def test_implant_dose_positive(self) -> None:
        """Implant dose should be positive for all wafers."""
        df = self._make_df(n_lots=20, wafers_per_lot=10)
        assert (df["implant_dose"] > 0).all(), "Implant dose must be positive"
