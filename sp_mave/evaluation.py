"""Downstream predictive evaluation on reduced coordinates.

After estimating B, we project both training and test data:
    Z_train = X_train @ B
    Z_test  = X_test  @ B

A downstream model is trained on (Z_train, y_train) and evaluated on
(Z_test, y_test).

Data-flow diagram (leakage control)
------------------------------------
Full dataset
  |-- train --> fit MAVE/SP-MAVE --> B
  |              |                      |
  |              +-- Z_train = X_train @ B --> fit downstream model
  |
  +-- test  --> Z_test = X_test @ B ----------> predict --> MSE

No test data touches weight construction, B estimation, or downstream
model fitting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def _get_model(model_type: str, params: dict[str, Any] | None = None) -> Any:
    """Instantiate a downstream regression model."""
    params = params or {}
    if model_type == "linear":
        return LinearRegression(**params)
    if model_type == "kernel_ridge":
        params.setdefault("kernel", "rbf")
        return KernelRidge(**params)
    if model_type == "random_forest":
        params.setdefault("n_estimators", 100)
        params.setdefault("random_state", 42)
        return RandomForestRegressor(**params)
    raise ValueError(f"Unknown downstream model type: '{model_type}'")


def evaluate_downstream(
    B: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "linear",
    model_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Train a downstream model on reduced train data, evaluate on test.

    Parameters
    ----------
    B : (p, d) estimated projection matrix.
    X_train, y_train : training data (standardised features, raw response).
    X_test, y_test : held-out test data.
    model_type : one of "linear", "kernel_ridge", "random_forest".
    model_params : additional kwargs for the model constructor.

    Returns
    -------
    dict with keys:
      test_mse        – mean squared error on test set
      train_mse       – mean squared error on train set (for diagnostics)
      downstream_model– string name
    """
    Z_train = X_train @ B
    Z_test = X_test @ B

    model = _get_model(model_type, model_params)
    model.fit(Z_train, y_train)

    y_pred_train = model.predict(Z_train)
    y_pred_test = model.predict(Z_test)

    return {
        "test_mse": float(mean_squared_error(y_test, y_pred_test)),
        "train_mse": float(mean_squared_error(y_train, y_pred_train)),
        "downstream_model": model_type,
    }
