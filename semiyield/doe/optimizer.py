"""
Bayesian process window optimizer.

Uses a Gaussian Process (GP) surrogate model with Expected Improvement (EI)
acquisition to find optimal semiconductor process parameters.

Primary backend: BoTorch (https://botorch.org)
Fallback: scipy.optimize.differential_evolution

Reference:
  E. Brochu, V.M. Cora, N. de Freitas, "A Tutorial on Bayesian Optimization of
  Expensive Cost Functions," arXiv:1012.2599, 2010.

  M. Balandat et al., "BoTorch: A Framework for Efficient Monte-Carlo Bayesian
  Optimization," NeurIPS 2020.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

try:
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    _HAS_BOTORCH = True
except ImportError:
    _HAS_BOTORCH = False
    warnings.warn(
        "botorch/gpytorch not available; using scipy.optimize fallback for "
        "Bayesian optimization.  Install botorch for full GP functionality."
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class ProcessWindowOptimizer:
    """Bayesian optimizer for semiconductor process window identification.

    Builds a Gaussian Process surrogate over the process parameter space and
    uses Expected Improvement to suggest the next experiment to run.

    Parameters
    ----------
    noise_level : float
        Assumed observation noise variance (passed to SingleTaskGP).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, noise_level: float = 1e-4, seed: int = 0) -> None:
        self.noise_level = noise_level
        self.seed = seed
        self._bounds_dict: dict[str, tuple[float, float]] = {}
        self._param_names: list[str] = []
        self._X_obs: np.ndarray | None = None  # (n_obs, n_params)
        self._y_obs: np.ndarray | None = None  # (n_obs,)
        self._X_norm: np.ndarray | None = None
        self._y_norm: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._gp_model: Any = None

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def define_space(self, parameter_bounds: dict[str, tuple[float, float]]) -> None:
        """Define the optimisation domain.

        Parameters
        ----------
        parameter_bounds : dict
            Mapping from parameter name to (lower_bound, upper_bound).
            Example: {"gate_oxide_time": (50, 200), "anneal_temp": (900, 1100)}
        """
        self._bounds_dict = dict(parameter_bounds)
        self._param_names = list(parameter_bounds.keys())
        self._X_obs = None
        self._y_obs = None

    def observe(self, X: np.ndarray, y: np.ndarray) -> None:
        """Add new observations to the GP surrogate.

        Parameters
        ----------
        X : numpy.ndarray
            Parameter matrix of shape (n_obs, n_params).
        y : numpy.ndarray
            Objective values of shape (n_obs,).  Higher is better (yield).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()

        if self._X_obs is None:
            self._X_obs = X
            self._y_obs = y
        else:
            self._X_obs = np.vstack([self._X_obs, X])
            self._y_obs = np.concatenate([self._y_obs, y])

        self._update_normalisation()
        if _HAS_BOTORCH:
            self._fit_gp()

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """Suggest next parameter combinations to evaluate.

        Parameters
        ----------
        n_suggestions : int
            Number of candidate points to return.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_suggestions, n_params) with suggested parameter
            values in the original (unnormalised) scale.
        """
        if not self._param_names:
            raise RuntimeError("Call define_space() first.")

        if self._X_obs is None or len(self._X_obs) < 2:
            # Not enough data: use latin hypercube sampling
            return self._lhs_sample(n_suggestions)

        if _HAS_BOTORCH and self._gp_model is not None:
            return self._suggest_botorch(n_suggestions)
        else:
            return self._suggest_random(n_suggestions)

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        n_iter: int = 50,
        n_init: int = 5,
    ) -> dict[str, Any]:
        """Run full Bayesian optimisation loop.

        Parameters
        ----------
        objective_fn : callable
            Function mapping a 1D parameter array to a scalar objective.
        n_iter : int
            Number of optimisation iterations after initial random exploration.
        n_init : int
            Number of random initial evaluations.

        Returns
        -------
        dict
            Keys: 'best_params' (dict), 'best_value' (float),
                  'history_X' (ndarray), 'history_y' (ndarray).
        """
        if not self._param_names:
            raise RuntimeError("Call define_space() first.")

        # Initial random exploration
        X_init = self._lhs_sample(n_init)
        for x in X_init:
            y = float(objective_fn(x))
            self.observe(x.reshape(1, -1), np.array([y]))

        # Bayesian optimisation loop
        for _ in range(n_iter):
            x_next = self.suggest(n_suggestions=1)[0]
            y_next = float(objective_fn(x_next))
            self.observe(x_next.reshape(1, -1), np.array([y_next]))

        best_idx = int(np.argmax(self._y_obs))
        best_x = self._X_obs[best_idx]
        best_y = float(self._y_obs[best_idx])

        best_params = {
            name: float(val)
            for name, val in zip(self._param_names, best_x)
        }

        return {
            "best_params": best_params,
            "best_value": best_y,
            "history_X": self._X_obs.copy(),
            "history_y": self._y_obs.copy(),
        }

    def process_window(
        self,
        center: dict[str, float] | None = None,
        confidence: float = 0.95,
    ) -> dict[str, tuple[float, float]]:
        """Find parameter ranges around the optimum where predicted yield is high.

        The process window is defined as the region where the GP mean prediction
        stays within the *confidence* quantile of the best observed yield.

        Parameters
        ----------
        center : dict or None
            Centre point for the process window search.  If None, uses the
            point with the highest observed yield.
        confidence : float
            Fraction of best_yield below which the window boundary is drawn.
            E.g., 0.95 means the window spans parameter values where predicted
            yield >= 0.95 * best_yield.

        Returns
        -------
        dict
            Mapping from parameter name to (lower_bound, upper_bound) of the
            process window.
        """
        if self._X_obs is None or len(self._X_obs) == 0:
            raise RuntimeError("No observations available.  Call observe() first.")

        best_idx = int(np.argmax(self._y_obs))
        center_x = (
            np.array([center[n] for n in self._param_names])
            if center is not None
            else self._X_obs[best_idx]
        )
        threshold = confidence * self._y_obs[best_idx]

        window: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(self._param_names):
            lo_bound, hi_bound = self._bounds_dict[name]
            x_lo = self._binary_search_window(center_x, i, lo_bound, center_x[i], threshold, direction="left")
            x_hi = self._binary_search_window(center_x, i, center_x[i], hi_bound, threshold, direction="right")
            window[name] = (float(x_lo), float(x_hi))

        return window

    def plot_surface(
        self,
        param1: str,
        param2: str,
        fixed_params: dict[str, float] | None = None,
        resolution: int = 30,
    ) -> Any:
        """Plot the GP mean prediction surface for two parameters.

        Parameters
        ----------
        param1 : str
            Name of first parameter (x-axis).
        param2 : str
            Name of second parameter (y-axis).
        fixed_params : dict or None
            Values for all other parameters.  If None, median observed values
            are used.
        resolution : int
            Grid resolution per axis.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure, or None if matplotlib is unavailable.
        """
        if not _HAS_MPL:
            return None

        if self._X_obs is None:
            raise RuntimeError("No observations.  Call observe() first.")

        fixed = fixed_params or {}
        medians = np.median(self._X_obs, axis=0)

        p1_idx = self._param_names.index(param1)
        p2_idx = self._param_names.index(param2)

        p1_vals = np.linspace(*self._bounds_dict[param1], resolution)
        p2_vals = np.linspace(*self._bounds_dict[param2], resolution)
        P1, P2 = np.meshgrid(p1_vals, p2_vals)

        grid_X = np.tile(medians, (resolution * resolution, 1))
        for name, val in fixed.items():
            if name in self._param_names:
                grid_X[:, self._param_names.index(name)] = val
        grid_X[:, p1_idx] = P1.ravel()
        grid_X[:, p2_idx] = P2.ravel()

        Z_mean, Z_std = self._predict_gp(grid_X)
        Z_mean = Z_mean.reshape(resolution, resolution)
        Z_std = Z_std.reshape(resolution, resolution)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        c1 = axes[0].contourf(P1, P2, Z_mean, levels=20, cmap="viridis")
        plt.colorbar(c1, ax=axes[0])
        axes[0].set_xlabel(param1)
        axes[0].set_ylabel(param2)
        axes[0].set_title("GP Mean Prediction")

        c2 = axes[1].contourf(P1, P2, Z_std, levels=20, cmap="plasma")
        plt.colorbar(c2, ax=axes[1])
        axes[1].set_xlabel(param1)
        axes[1].set_ylabel(param2)
        axes[1].set_title("GP Standard Deviation (Uncertainty)")

        plt.tight_layout()
        return fig

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _update_normalisation(self) -> None:
        """Normalise observations to [0,1] x [0,1] for GP fitting."""
        lowers = np.array([self._bounds_dict[n][0] for n in self._param_names])
        uppers = np.array([self._bounds_dict[n][1] for n in self._param_names])
        ranges = uppers - lowers
        ranges[ranges == 0] = 1.0
        self._X_norm = (self._X_obs - lowers) / ranges

        self._y_mean = float(self._y_obs.mean())
        self._y_std = float(self._y_obs.std()) or 1.0
        self._y_norm = (self._y_obs - self._y_mean) / self._y_std

    def _fit_gp(self) -> None:
        """Fit the BoTorch SingleTaskGP to normalised observations."""
        if not _HAS_BOTORCH:
            return
        X_t = torch.tensor(self._X_norm, dtype=torch.float64)
        y_t = torch.tensor(self._y_norm, dtype=torch.float64).unsqueeze(-1)
        self._gp_model = SingleTaskGP(X_t, y_t)
        mll = ExactMarginalLogLikelihood(self._gp_model.likelihood, self._gp_model)
        fit_gpytorch_mll(mll)
        self._gp_model.eval()

    def _suggest_botorch(self, n: int) -> np.ndarray:
        """Use BoTorch EI acquisition to suggest next points."""
        lowers = np.array([self._bounds_dict[nm][0] for nm in self._param_names])
        uppers = np.array([self._bounds_dict[nm][1] for nm in self._param_names])
        ranges = uppers - lowers
        ranges[ranges == 0] = 1.0

        n_params = len(self._param_names)
        bounds_t = torch.tensor([[0.0] * n_params, [1.0] * n_params], dtype=torch.float64)

        best_f = torch.tensor(float(self._y_norm.max()), dtype=torch.float64)
        EI = ExpectedImprovement(self._gp_model, best_f=best_f)

        candidates, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds_t,
            q=n,
            num_restarts=5,
            raw_samples=64,
        )

        candidates_np = candidates.detach().numpy()
        # Denormalise
        return candidates_np * ranges + lowers

    def _suggest_random(self, n: int) -> np.ndarray:
        """Random suggestions within bounds (fallback)."""
        rng = np.random.default_rng(self.seed)
        lowers = np.array([self._bounds_dict[nm][0] for nm in self._param_names])
        uppers = np.array([self._bounds_dict[nm][1] for nm in self._param_names])
        return rng.uniform(lowers, uppers, size=(n, len(self._param_names)))

    def _lhs_sample(self, n: int) -> np.ndarray:
        """Latin hypercube sampling within bounds."""
        rng = np.random.default_rng(self.seed)
        n_params = len(self._param_names)
        lowers = np.array([self._bounds_dict[nm][0] for nm in self._param_names])
        uppers = np.array([self._bounds_dict[nm][1] for nm in self._param_names])

        # LHS: n intervals, one sample per interval per dimension
        intervals = np.arange(n, dtype=float) / n
        X = np.zeros((n, n_params))
        for j in range(n_params):
            perm = rng.permutation(n)
            X[:, j] = (intervals + rng.uniform(0, 1.0 / n, size=n))[perm]
        return X * (uppers - lowers) + lowers

    def _predict_gp(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std for an array of points in original scale."""
        lowers = np.array([self._bounds_dict[n][0] for n in self._param_names])
        uppers = np.array([self._bounds_dict[n][1] for n in self._param_names])
        ranges = uppers - lowers
        ranges[ranges == 0] = 1.0
        X_norm = (X - lowers) / ranges

        if _HAS_BOTORCH and self._gp_model is not None:
            X_t = torch.tensor(X_norm, dtype=torch.float64)
            with torch.no_grad():
                posterior = self._gp_model.posterior(X_t)
                mean_n = posterior.mean.squeeze(-1).numpy()
                var_n = posterior.variance.squeeze(-1).numpy()
            mean = mean_n * self._y_std + self._y_mean
            std = np.sqrt(np.maximum(var_n, 0.0)) * self._y_std
        else:
            # Fallback: use RBF kernel interpolation via scipy
            mean, std = self._rbf_predict(X_norm)
            mean = mean * self._y_std + self._y_mean
            std = std * self._y_std

        return mean, std

    def _rbf_predict(
        self, X_norm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple RBF kernel interpolation as scipy fallback."""
        from scipy.interpolate import RBFInterpolator

        if self._X_norm is None or self._y_norm is None:
            return np.zeros(len(X_norm)), np.ones(len(X_norm))

        try:
            rbf = RBFInterpolator(self._X_norm, self._y_norm, kernel="thin_plate_spline")
            mean = rbf(X_norm)
        except Exception:
            mean = np.full(len(X_norm), self._y_norm.mean())

        # Uncertainty: distance to nearest training point (normalised)
        dists = np.min(
            np.linalg.norm(X_norm[:, np.newaxis] - self._X_norm[np.newaxis], axis=-1),
            axis=1,
        )
        std = np.clip(dists, 0.0, 1.0)
        return mean, std

    def _binary_search_window(
        self,
        center: np.ndarray,
        param_idx: int,
        lo: float,
        hi: float,
        threshold: float,
        direction: str,
    ) -> float:
        """Binary search for the process window boundary along one dimension."""
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            x_test = center.copy()
            x_test[param_idx] = mid
            pred_mean, _ = self._predict_gp(x_test.reshape(1, -1))
            if float(pred_mean[0]) >= threshold:
                if direction == "left":
                    hi = mid
                else:
                    lo = mid
            else:
                if direction == "left":
                    lo = mid
                else:
                    hi = mid
        return 0.5 * (lo + hi)
