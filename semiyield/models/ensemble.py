"""
Ensemble yield prediction model.

Combines three base learners:
  1. RandomForestRegressor (scikit-learn)
  2. XGBRegressor (xgboost)
  3. LSTMYieldNet (PyTorch LSTM for sequential process step data)

The ensemble weights are learned via Ridge regression on held-out validation
predictions, following the stacking principle (Wolpert, 1992).

Reference:
  D.H. Wolpert, "Stacked Generalization," Neural Networks 5(2), 241-259, 1992.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn("xgboost not found; XGBRegressor will be replaced by a second RF.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    warnings.warn("PyTorch not found; LSTMYieldNet will be disabled.")


# ------------------------------------------------------------------ #
# PyTorch LSTM model definition                                        #
# ------------------------------------------------------------------ #

if _HAS_TORCH:
    class LSTMYieldNet(nn.Module):
        """Small LSTM network for sequential process step data.

        Input shape: (batch, seq_len, n_features)
        Output shape: (batch, 1)
        """

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq, features)
            out, _ = self.lstm(x)
            # Use last time-step output
            last = out[:, -1, :]
            last = self.dropout(last)
            return self.head(last).squeeze(-1)

else:
    class LSTMYieldNet:  # type: ignore[no-redef]
        """Stub when PyTorch is not available."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for LSTMYieldNet.")


class YieldEnsemble:
    """Stacked ensemble for semiconductor yield prediction.

    Base models:
      1. RandomForestRegressor
      2. XGBRegressor (or second RF if xgboost unavailable)
      3. LSTMYieldNet (or disabled if torch unavailable)

    Ensemble meta-learner: Ridge regression over base model predictions.

    Parameters
    ----------
    n_estimators : int
        Number of trees for RF and XGB.
    lstm_epochs : int
        Maximum training epochs for the LSTM.
    lstm_patience : int
        Early-stopping patience for the LSTM (epochs without improvement).
    device : str or None
        PyTorch device string ('cpu', 'cuda', etc.).  If None, auto-detect.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        lstm_epochs: int = 100,
        lstm_patience: int = 10,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.lstm_epochs = lstm_epochs
        self.lstm_patience = lstm_patience
        self.random_state = random_state

        if device is None and _HAS_TORCH:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device or "cpu"

        # Base models
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        if _HAS_XGB:
            self.xgb = XGBRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                tree_method="hist",
                verbosity=0,
            )
        else:
            self.xgb = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state + 1,
                n_jobs=-1,
            )

        self.lstm_net: LSTMYieldNet | None = None
        self.meta: Ridge = Ridge(alpha=1.0)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._fitted = False
        self._use_lstm = _HAS_TORCH

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "YieldEnsemble":
        """Train all base models and the ensemble meta-learner.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training features.  Shape (n_samples, n_features) for RF/XGB or
            (n_samples, seq_len, n_features) for the LSTM.
        y_train : numpy.ndarray
            Training labels (yield values).
        X_val : numpy.ndarray
            Validation features (same shape convention as X_train).
        y_val : numpy.ndarray
            Validation labels.

        Returns
        -------
        YieldEnsemble
            self (for method chaining).
        """
        X2d_train = self._to_2d(X_train)
        X2d_val = self._to_2d(X_val)
        y_train = np.asarray(y_train, dtype=float)
        y_val = np.asarray(y_val, dtype=float)

        # Scale features and targets
        X_tr_s = self.scaler_X.fit_transform(X2d_train)
        X_va_s = self.scaler_X.transform(X2d_val)
        y_tr_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_va_s = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()

        # --- Train Random Forest ---
        self.rf.fit(X_tr_s, y_tr_s)

        # --- Train XGBoost ---
        self.xgb.fit(X_tr_s, y_tr_s)

        # --- Train LSTM ---
        if self._use_lstm:
            X3d_train = self._to_3d(X_train)
            X3d_val = self._to_3d(X_val)
            n_features = X3d_train.shape[-1]
            self.lstm_net = LSTMYieldNet(n_features=n_features).to(self.device)
            self._train_lstm(X3d_train, y_tr_s, X3d_val, y_va_s)

        # --- Build meta-learner on validation predictions ---
        val_stack = self._stack_predict(X_va_s, X_val)
        self.meta.fit(val_stack, y_va_s)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict yield for new observations.

        Parameters
        ----------
        X : numpy.ndarray
            Features (2D or 3D as in fit).

        Returns
        -------
        numpy.ndarray
            Predicted yield values in the original (unscaled) range.
        """
        self._check_fitted()
        X2d = self._to_2d(X)
        X_s = self.scaler_X.transform(X2d)
        stack = self._stack_predict(X_s, X)
        y_s = self.meta.predict(stack)
        return self.scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()

    def predict_proba(self, X: np.ndarray, n_mc: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Predict yield with uncertainty estimate.

        Uncertainty is estimated from:
          - RF: variance of individual tree predictions
          - LSTM: Monte-Carlo dropout variance
        The reported std is the average of the two (or just RF if LSTM absent).

        Parameters
        ----------
        X : numpy.ndarray
            Features.
        n_mc : int
            Number of MC dropout samples for the LSTM.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_predictions, std_predictions) both in original yield scale.
        """
        self._check_fitted()
        X2d = self._to_2d(X)
        X_s = self.scaler_X.transform(X2d)

        # RF uncertainty: std of individual tree predictions
        tree_preds = np.array([tree.predict(X_s) for tree in self.rf.estimators_])
        rf_std = tree_preds.std(axis=0)

        if self._use_lstm and self.lstm_net is not None:
            # MC dropout: keep dropout active during inference
            self.lstm_net.train()
            X3d = self._to_3d(X)
            t = torch.tensor(X3d, dtype=torch.float32).to(self.device)
            mc_preds = []
            with torch.no_grad():
                for _ in range(n_mc):
                    mc_preds.append(self.lstm_net(t).cpu().numpy())
            self.lstm_net.eval()
            mc_arr = np.stack(mc_preds, axis=0)
            lstm_std = mc_arr.std(axis=0)
            combined_std = 0.5 * (rf_std + lstm_std)
        else:
            combined_std = rf_std

        mean_pred = self.predict(X)
        # Scale std from normalised space back to original
        scale = self.scaler_y.scale_[0]
        return mean_pred, combined_std * scale

    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute regression metrics.

        Parameters
        ----------
        X : numpy.ndarray
            Features.
        y : numpy.ndarray
            True yield values.

        Returns
        -------
        dict
            Keys: 'R2', 'RMSE', 'MAE'.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        return {
            "R2": float(r2_score(y, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
            "MAE": float(mean_absolute_error(y, y_pred)),
        }

    def save(self, path: str | Path) -> None:
        """Persist the ensemble to *path*.

        Parameters
        ----------
        path : str or Path
            File path (pickle format).
        """
        path = Path(path)
        state = {
            "rf": self.rf,
            "xgb": self.xgb,
            "meta": self.meta,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "_fitted": self._fitted,
            "_use_lstm": self._use_lstm,
            "device": self.device,
        }
        if self._use_lstm and self.lstm_net is not None:
            state["lstm_state_dict"] = self.lstm_net.state_dict()
            state["lstm_n_features"] = next(iter(self.lstm_net.lstm.parameters())).shape[1]
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "YieldEnsemble":
        """Load a previously saved ensemble.

        Parameters
        ----------
        path : str or Path
            File path written by :meth:`save`.

        Returns
        -------
        YieldEnsemble
            Restored ensemble instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.rf = state["rf"]
        obj.xgb = state["xgb"]
        obj.meta = state["meta"]
        obj.scaler_X = state["scaler_X"]
        obj.scaler_y = state["scaler_y"]
        obj._fitted = state["_fitted"]
        obj._use_lstm = state["_use_lstm"]
        obj.device = state.get("device", "cpu")
        obj.lstm_net = None
        if obj._use_lstm and "lstm_state_dict" in state and _HAS_TORCH:
            n_features = state["lstm_n_features"]
            obj.lstm_net = LSTMYieldNet(n_features=n_features).to(obj.device)
            obj.lstm_net.load_state_dict(state["lstm_state_dict"])
            obj.lstm_net.eval()
        return obj

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

    @staticmethod
    def _to_2d(X: np.ndarray) -> np.ndarray:
        """Flatten a 3D array (batch, seq, feat) to 2D (batch, seq*feat)."""
        X = np.asarray(X)
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    @staticmethod
    def _to_3d(X: np.ndarray) -> np.ndarray:
        """Ensure array is 3D (batch, seq, feat); add seq dim if 2D."""
        X = np.asarray(X)
        if X.ndim == 2:
            return X[:, np.newaxis, :]  # (batch, 1, feat)
        return X

    def _stack_predict(self, X_s: np.ndarray, X_orig: np.ndarray) -> np.ndarray:
        """Build (n_samples, n_base_models) prediction matrix."""
        rf_pred = self.rf.predict(X_s)
        xgb_pred = self.xgb.predict(X_s)

        cols = [rf_pred, xgb_pred]

        if self._use_lstm and self.lstm_net is not None:
            self.lstm_net.eval()
            X3d = self._to_3d(X_orig)
            t = torch.tensor(X3d, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                lstm_pred = self.lstm_net(t).cpu().numpy()
            cols.append(lstm_pred)

        return np.column_stack(cols)

    def _train_lstm(
        self,
        X3d_train: np.ndarray,
        y_train: np.ndarray,
        X3d_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train the LSTM with early stopping."""
        t_X_tr = torch.tensor(X3d_train, dtype=torch.float32).to(self.device)
        t_y_tr = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        t_X_va = torch.tensor(X3d_val, dtype=torch.float32).to(self.device)
        t_y_va = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(t_X_tr, t_y_tr)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimiser = torch.optim.Adam(self.lstm_net.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.lstm_epochs):
            self.lstm_net.train()
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = criterion(self.lstm_net(xb), yb)
                loss.backward()
                optimiser.step()

            # Validation loss
            self.lstm_net.eval()
            with torch.no_grad():
                val_loss = criterion(self.lstm_net(t_X_va), t_y_va).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.lstm_net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.lstm_patience:
                    break

        if best_state is not None:
            self.lstm_net.load_state_dict(best_state)
        self.lstm_net.eval()
