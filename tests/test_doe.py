"""
Tests for semiyield.doe.ProcessWindowOptimizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from semiyield.doe import ProcessWindowOptimizer


def _quadratic_objective(x: np.ndarray) -> float:
    """Simple 2D quadratic: maximum at (1.5, 2.0), value = 1.0."""
    x0_opt, x1_opt = 1.5, 2.0
    return float(1.0 - 0.5 * (x[0] - x0_opt) ** 2 - 0.3 * (x[1] - x1_opt) ** 2)


class TestProcessWindowOptimizer:
    """Tests for ProcessWindowOptimizer."""

    def _make_optimizer(self) -> ProcessWindowOptimizer:
        opt = ProcessWindowOptimizer(seed=42)
        opt.define_space({"x0": (0.0, 3.0), "x1": (0.0, 4.0)})
        return opt

    def test_define_space_stores_bounds(self) -> None:
        """define_space() should store parameter bounds correctly."""
        opt = ProcessWindowOptimizer()
        bounds = {"gate_oxide_time": (50.0, 200.0), "anneal_temp": (900.0, 1100.0)}
        opt.define_space(bounds)
        assert opt._bounds_dict == bounds, "Bounds dict must match input"
        assert opt._param_names == list(bounds.keys()), "Parameter names must match"

    def test_observe_adds_data(self) -> None:
        """observe() should accumulate observations."""
        opt = self._make_optimizer()
        X1 = np.array([[1.0, 2.0], [1.5, 2.5]])
        y1 = np.array([0.9, 0.85])
        opt.observe(X1, y1)
        assert opt._X_obs.shape == (2, 2), "X_obs should have 2 rows after first batch"

        X2 = np.array([[0.5, 1.0]])
        y2 = np.array([0.7])
        opt.observe(X2, y2)
        assert opt._X_obs.shape == (3, 2), "X_obs should have 3 rows after second batch"

    def test_suggest_returns_correct_shape(self) -> None:
        """suggest() should return an array of shape (n, n_params)."""
        opt = self._make_optimizer()
        X = np.array([[1.0, 2.0], [2.0, 3.0], [0.5, 1.5]])
        y = np.array([0.9, 0.8, 0.7])
        opt.observe(X, y)
        suggestions = opt.suggest(n_suggestions=3)
        assert suggestions.shape == (3, 2), f"Expected (3, 2), got {suggestions.shape}"

    def test_suggest_within_bounds(self) -> None:
        """Suggestions must fall within the defined parameter bounds."""
        opt = self._make_optimizer()
        # Provide minimal observations to allow GP fitting
        rng = np.random.default_rng(0)
        X = rng.uniform([0, 0], [3, 4], size=(10, 2))
        y = np.array([_quadratic_objective(x) for x in X])
        opt.observe(X, y)

        for _ in range(5):
            suggestions = opt.suggest(n_suggestions=1)
            x0, x1 = suggestions[0]
            assert 0.0 <= x0 <= 3.0, f"x0={x0:.4f} outside bounds [0, 3]"
            assert 0.0 <= x1 <= 4.0, f"x1={x1:.4f} outside bounds [0, 4]"

    def test_optimize_finds_approximate_optimum(self) -> None:
        """optimize() on the quadratic objective should find a value close to the true optimum."""
        opt = self._make_optimizer()
        result = opt.optimize(_quadratic_objective, n_iter=20, n_init=8)

        true_optimum = _quadratic_objective(np.array([1.5, 2.0]))
        best = result["best_value"]
        assert best > 0.7, f"Optimizer should find value > 0.7; got {best:.4f}"
        assert abs(best - true_optimum) < 0.25, (
            f"Best value {best:.4f} is too far from true optimum {true_optimum:.4f}"
        )

    def test_optimize_returns_required_keys(self) -> None:
        """optimize() result dict should contain expected keys."""
        opt = self._make_optimizer()
        result = opt.optimize(_quadratic_objective, n_iter=5, n_init=3)
        for key in ("best_params", "best_value", "history_X", "history_y"):
            assert key in result, f"Missing key '{key}' in optimize result"

    def test_optimize_best_params_dict(self) -> None:
        """best_params should be a dict with correct parameter names."""
        opt = self._make_optimizer()
        result = opt.optimize(_quadratic_objective, n_iter=5, n_init=3)
        bp = result["best_params"]
        assert isinstance(bp, dict), "best_params should be a dict"
        assert set(bp.keys()) == {"x0", "x1"}, f"Expected keys {{x0, x1}}, got {set(bp.keys())}"

    def test_optimize_history_length(self) -> None:
        """History should contain n_init + n_iter observations."""
        opt = self._make_optimizer()
        n_init, n_iter = 5, 10
        result = opt.optimize(_quadratic_objective, n_iter=n_iter, n_init=n_init)
        assert len(result["history_y"]) == n_init + n_iter, (
            f"History length should be {n_init + n_iter}, got {len(result['history_y'])}"
        )

    def test_process_window_contains_optimum(self) -> None:
        """The process window should contain the best observed point."""
        opt = self._make_optimizer()
        result = opt.optimize(_quadratic_objective, n_iter=15, n_init=5)

        window = opt.process_window(confidence=0.90)
        best_params = result["best_params"]

        for name, (lo, hi) in window.items():
            x_best = best_params[name]
            assert lo <= x_best <= hi, (
                f"Best param {name}={x_best:.4f} outside window [{lo:.4f}, {hi:.4f}]"
            )

    def test_suggest_before_define_space_raises(self) -> None:
        """suggest() without define_space() should raise RuntimeError."""
        opt = ProcessWindowOptimizer()
        with pytest.raises(RuntimeError):
            opt.suggest()

    def test_lhs_sample_shape(self) -> None:
        """Internal LHS sampler should return correct shape."""
        opt = self._make_optimizer()
        samples = opt._lhs_sample(10)
        assert samples.shape == (10, 2), f"Expected (10, 2), got {samples.shape}"

    def test_lhs_sample_within_bounds(self) -> None:
        """LHS samples must be within bounds."""
        opt = self._make_optimizer()
        samples = opt._lhs_sample(20)
        assert np.all(samples[:, 0] >= 0.0) and np.all(samples[:, 0] <= 3.0)
        assert np.all(samples[:, 1] >= 0.0) and np.all(samples[:, 1] <= 4.0)
