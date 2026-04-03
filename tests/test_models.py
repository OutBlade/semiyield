"""
Tests for semiyield.models.YieldEnsemble and SHAPExplainer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_regression

from semiyield.models import SHAPExplainer, YieldEnsemble


def _make_data(
    n_samples: int = 300, n_features: int = 8, noise: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic regression dataset for testing."""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise * n_samples, random_state=0)
    # Rescale y to [0.4, 0.95] to mimic yield
    y = 0.4 + 0.55 * (y - y.min()) / (y.max() - y.min())

    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def _train_small_model(n_estimators: int = 50) -> tuple[YieldEnsemble, np.ndarray, np.ndarray]:
    """Train a small ensemble model for testing."""
    X_tr, y_tr, X_te, y_te = _make_data()
    val_split = int(0.85 * len(X_tr))
    X_val, y_val = X_tr[val_split:], y_tr[val_split:]
    X_tr, y_tr = X_tr[:val_split], y_tr[:val_split]

    model = YieldEnsemble(n_estimators=n_estimators, lstm_epochs=5, lstm_patience=3, random_state=42)
    model.fit(X_tr, y_tr, X_val, y_val)
    return model, X_te, y_te


class TestYieldEnsemble:
    """Tests for YieldEnsemble."""

    def test_fit_predict_runs(self) -> None:
        """fit() and predict() should complete without errors."""
        model, X_te, y_te = _train_small_model()
        predictions = model.predict(X_te)
        assert isinstance(predictions, np.ndarray)
        assert predictions.dtype == float or np.issubdtype(predictions.dtype, np.floating)
        assert len(predictions) == len(X_te)

    def test_predictions_are_finite(self) -> None:
        """All predictions should be finite floats."""
        model, X_te, _ = _train_small_model()
        preds = model.predict(X_te)
        assert np.all(np.isfinite(preds)), "All predictions must be finite"

    def test_r2_positive(self) -> None:
        """R2 score should be positive on the test set (better than mean predictor)."""
        model, X_te, y_te = _train_small_model()
        metrics = model.score(X_te, y_te)
        assert metrics["R2"] > 0.0, f"R2 should be positive on test set, got {metrics['R2']:.4f}"

    def test_score_returns_required_keys(self) -> None:
        """score() should return R2, RMSE, and MAE."""
        model, X_te, y_te = _train_small_model()
        metrics = model.score(X_te, y_te)
        for key in ("R2", "RMSE", "MAE"):
            assert key in metrics, f"Missing key '{key}' in score dict"
            assert isinstance(metrics[key], float), f"metric['{key}'] must be float"

    def test_uncertainty_returns_mean_and_std(self) -> None:
        """predict_proba() should return two arrays of the same shape."""
        model, X_te, _ = _train_small_model()
        mean, std = model.predict_proba(X_te)
        assert mean.shape == std.shape == (len(X_te),), "mean and std must match input size"

    def test_uncertainty_std_positive(self) -> None:
        """Uncertainty (std) should be non-negative."""
        model, X_te, _ = _train_small_model()
        _, std = model.predict_proba(X_te)
        assert np.all(std >= 0.0), "Standard deviation must be non-negative"

    def test_save_and_load(self) -> None:
        """Saved and reloaded model should produce the same predictions."""
        model, X_te, _ = _train_small_model()
        preds_before = model.predict(X_te)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            path = tmp.name
        try:
            model.save(path)
            loaded = YieldEnsemble.load(path)
            preds_after = loaded.predict(X_te)
        finally:
            Path(path).unlink(missing_ok=True)

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5,
                                   err_msg="Loaded model predictions should match original")

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict() before fit() should raise RuntimeError."""
        model = YieldEnsemble()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 8)))

    def test_3d_input_accepted(self) -> None:
        """3-D input (batch, seq, feat) should work for 2D models (flattened internally)."""
        X_tr, y_tr, X_te, y_te = _make_data(n_samples=200, n_features=6)
        X_tr_3d = X_tr[:, np.newaxis, :]  # (n, 1, feat)
        X_val_3d = X_tr[-20:, np.newaxis, :]
        y_val = y_tr[-20:]
        X_tr_3d = X_tr_3d[:-20]
        y_tr2 = y_tr[:-20]

        model = YieldEnsemble(n_estimators=20, lstm_epochs=2, random_state=0)
        model.fit(X_tr_3d, y_tr2, X_val_3d, y_val)
        preds = model.predict(X_te[:, np.newaxis, :])
        assert len(preds) == len(X_te)


class TestSHAPExplainer:
    """Tests for SHAPExplainer."""

    def test_shap_values_shape(self) -> None:
        """SHAP values should have shape (n_samples, n_features)."""
        model, X_te, _ = _train_small_model()
        explainer = SHAPExplainer()
        shap_vals = explainer.explain(model.rf, X_te)
        assert shap_vals.shape == X_te.shape, (
            f"SHAP values shape {shap_vals.shape} should match X shape {X_te.shape}"
        )

    def test_top_features_length(self) -> None:
        """top_features() should return exactly n items."""
        model, X_te, _ = _train_small_model()
        n_feat = X_te.shape[1]
        feat_names = [f"feature_{i}" for i in range(n_feat)]
        explainer = SHAPExplainer()
        top = explainer.top_features(model.rf, X_te, feat_names, n=5)
        assert len(top) == 5, f"Expected 5 top features, got {len(top)}"

    def test_top_features_sorted(self) -> None:
        """top_features() should be sorted from most to least important."""
        model, X_te, _ = _train_small_model()
        n_feat = X_te.shape[1]
        feat_names = [f"feature_{i}" for i in range(n_feat)]
        explainer = SHAPExplainer()
        top = explainer.top_features(model.rf, X_te, feat_names, n=n_feat)
        importances = [t[1] for t in top]
        assert importances == sorted(importances, reverse=True), (
            "Features should be sorted by descending importance"
        )

    def test_top_features_returns_tuples(self) -> None:
        """Each element of top_features() result should be a (str, float) tuple."""
        model, X_te, _ = _train_small_model()
        feat_names = [f"f{i}" for i in range(X_te.shape[1])]
        explainer = SHAPExplainer()
        top = explainer.top_features(model.rf, X_te, feat_names, n=3)
        for item in top:
            assert len(item) == 2, "Each item must be a 2-tuple"
            assert isinstance(item[0], str), "First element must be feature name (str)"
            assert isinstance(item[1], float), "Second element must be importance (float)"

    def test_sensitivity_returns_dict(self) -> None:
        """process_window_sensitivity() should return dict with required keys."""
        model, X_te, _ = _train_small_model()
        feat_names = [f"f{i}" for i in range(X_te.shape[1])]
        explainer = SHAPExplainer()
        result = explainer.process_window_sensitivity(model.rf, X_te, feat_names, "f0", n_points=20)
        for key in ("param_values", "predictions", "shap_values"):
            assert key in result, f"Missing key '{key}' in sensitivity result"
        assert len(result["param_values"]) == 20
        assert len(result["predictions"]) == 20

    def test_sensitivity_unknown_feature_raises(self) -> None:
        """Requesting sensitivity for an unknown feature should raise ValueError."""
        model, X_te, _ = _train_small_model()
        feat_names = [f"f{i}" for i in range(X_te.shape[1])]
        explainer = SHAPExplainer()
        with pytest.raises(ValueError, match="not in feature_names"):
            explainer.process_window_sensitivity(model.rf, X_te, feat_names, "nonexistent")
