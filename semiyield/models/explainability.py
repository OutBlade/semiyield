"""
SHAP (SHapley Additive exPlanations) explainability wrapper.

Reference:
  S.M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model
  Predictions," NIPS 2017.

  S.M. Lundberg et al., "From local explanations to global understanding with
  explainable AI for trees," Nature Machine Intelligence 2, 56-67, 2020.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    warnings.warn(
        "shap not installed; SHAPExplainer will use a permutation-based fallback.", stacklevel=2
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class SHAPExplainer:
    """SHAP-based model explainability for yield prediction models.

    Automatically selects TreeExplainer for tree-based models (RF, XGBoost)
    and falls back to KernelExplainer for other model types.

    Parameters
    ----------
    background_samples : int
        Number of background samples for KernelExplainer.  Fewer samples are
        faster but less accurate.
    """

    def __init__(self, background_samples: int = 100) -> None:
        self.background_samples = background_samples

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def explain(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """Compute SHAP values for *X* under *model*.

        Parameters
        ----------
        model : estimator
            Fitted scikit-learn or xgboost model, or any callable that
            accepts a 2D numpy array and returns a 1D array of predictions.
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        feature_names : list of str or None
            Feature names for display purposes (optional).

        Returns
        -------
        numpy.ndarray
            SHAP values of shape (n_samples, n_features).
        """
        X = np.asarray(X)
        if not _HAS_SHAP:
            return self._permutation_shap(model, X)

        explainer = self._make_explainer(model, X)
        shap_vals = explainer.shap_values(X)

        # TreeExplainer may return a list (one array per class) for classifiers
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        return np.asarray(shap_vals)

    def top_features(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-*n* features by mean absolute SHAP value.

        Parameters
        ----------
        model : estimator
            Fitted model.
        X : numpy.ndarray
            Input data.
        feature_names : list of str
            Feature names (length must match X.shape[1]).
        n : int
            Number of top features to return.

        Returns
        -------
        list of (feature_name, mean_abs_shap)
            Sorted from most to least important.
        """
        shap_vals = self.explain(model, X, feature_names)
        mean_abs = np.abs(shap_vals).mean(axis=0)

        if len(feature_names) != mean_abs.shape[0]:
            raise ValueError(
                f"len(feature_names)={len(feature_names)} does not match "
                f"SHAP value dimension {mean_abs.shape[0]}."
            )

        ranked = sorted(
            zip(feature_names, mean_abs.tolist(), strict=False),
            key=lambda t: t[1],
            reverse=True,
        )
        return ranked[:n]

    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: list[str] | None = None,
        X: np.ndarray | None = None,
    ) -> Any:
        """Create a SHAP summary bar chart.

        The chart shows mean absolute SHAP value per feature, sorted
        from most to least important.

        Parameters
        ----------
        shap_values : numpy.ndarray
            SHAP values of shape (n_samples, n_features).
        feature_names : list of str or None
            Feature labels for the y-axis.
        X : numpy.ndarray or None
            Original feature matrix (used for beeswarm plots if provided).

        Returns
        -------
        matplotlib.figure.Figure or None
            Figure object.  Returns None if matplotlib is unavailable.
        """
        if not _HAS_MPL:
            warnings.warn("matplotlib not available; cannot create plot.", stacklevel=2)
            return None

        mean_abs = np.abs(shap_values).mean(axis=0)
        n_feat = len(mean_abs)
        labels = feature_names if feature_names is not None else [f"f{i}" for i in range(n_feat)]

        order = np.argsort(mean_abs)
        sorted_vals = mean_abs[order]
        sorted_labels = [labels[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, max(4, n_feat * 0.3)))
        ax.barh(sorted_labels, sorted_vals, color="#1f77b4")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (SHAP)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return fig

    def process_window_sensitivity(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str],
        target_feature: str,
        n_points: int = 50,
    ) -> dict[str, np.ndarray]:
        """Compute sensitivity of model output to a single process parameter.

        Sweeps *target_feature* across its observed range while holding all
        other features at their median values, and records the model
        prediction at each point.

        Parameters
        ----------
        model : estimator
            Fitted model with a ``predict`` method.
        X : numpy.ndarray
            Reference dataset used to determine feature ranges and medians.
        feature_names : list of str
            Feature names.
        target_feature : str
            Name of the feature to sweep.
        n_points : int
            Number of evaluation points across the sweep range.

        Returns
        -------
        dict
            Keys: 'param_values' (sweep values), 'predictions' (model output),
            'shap_values' (SHAP value of target feature at each point).
        """
        if target_feature not in feature_names:
            raise ValueError(f"'{target_feature}' not in feature_names.")

        feat_idx = feature_names.index(target_feature)
        X = np.asarray(X)
        medians = np.median(X, axis=0)

        x_min, x_max = X[:, feat_idx].min(), X[:, feat_idx].max()
        sweep = np.linspace(x_min, x_max, n_points)

        X_sweep = np.tile(medians, (n_points, 1))
        X_sweep[:, feat_idx] = sweep

        if hasattr(model, "predict"):
            predictions = model.predict(X_sweep)
        else:
            predictions = np.array([model(row.reshape(1, -1))[0] for row in X_sweep])

        shap_vals = self.explain(model, X_sweep, feature_names)
        target_shap = shap_vals[:, feat_idx]

        return {
            "param_values": sweep,
            "predictions": np.asarray(predictions),
            "shap_values": target_shap,
        }

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _make_explainer(self, model: Any, X: np.ndarray) -> Any:
        """Select and create the most appropriate SHAP explainer."""
        model_type = type(model).__name__

        # Tree-based models: use fast TreeExplainer
        tree_model_types = (
            "RandomForestRegressor",
            "RandomForestClassifier",
            "XGBRegressor",
            "XGBClassifier",
            "GradientBoostingRegressor",
            "DecisionTreeRegressor",
        )

        if model_type in tree_model_types:
            try:
                return shap.TreeExplainer(model)
            except Exception:
                pass  # Fall through to KernelExplainer

        # Fallback: KernelExplainer with background summary
        n_bg = min(self.background_samples, X.shape[0])
        background = shap.sample(X, n_bg)
        if hasattr(model, "predict"):
            predict_fn = model.predict
        else:
            predict_fn = model
        return shap.KernelExplainer(predict_fn, background)

    @staticmethod
    def _permutation_shap(model: Any, X: np.ndarray) -> np.ndarray:
        """Simple permutation-based feature importance as SHAP fallback.

        Approximates |SHAP| by measuring prediction change when each
        feature column is permuted.  This is not true SHAP but provides
        a rough feature importance ranking when the shap library is absent.
        """
        if not hasattr(model, "predict"):
            return np.zeros((X.shape[0], X.shape[1]))

        baseline = model.predict(X)
        n_samples, n_features = X.shape
        importance = np.zeros(n_features)

        for j in range(n_features):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            perm_pred = model.predict(X_perm)
            importance[j] = np.mean(np.abs(baseline - perm_pred))

        # Return a uniform attribution per sample (each sample gets mean abs)
        shap_approx = np.tile(importance, (n_samples, 1))
        # Adjust sign: positive where feature value is above median
        median_vals = np.median(X, axis=0)
        sign = np.where(X > median_vals, 1.0, -1.0)
        return shap_approx * sign
