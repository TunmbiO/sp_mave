"""Convenience wrapper for downstream evaluation on all three model types."""

from __future__ import annotations

import numpy as np

from sp_mave.evaluation import evaluate_downstream


def evaluate_subspace(
    B: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the estimated subspace B on three downstream model types.

    Projects train and test data onto B, fits linear / kernel_ridge /
    random_forest regressors, and returns train and test MSE for each.

    Parameters
    ----------
    B : (p, d) estimated projection matrix (orthonormal columns).
    X_train, X_test : (n_train, p) and (n_test, p) feature matrices.
    y_train, y_test : (n_train,) and (n_test,) response vectors.

    Returns
    -------
    dict with keys: train_mse_linear, test_mse_linear,
    train_mse_kernel_ridge, test_mse_kernel_ridge,
    train_mse_random_forest, test_mse_random_forest.
    """
    results: dict[str, float] = {}
    for model_type in ("linear", "kernel_ridge", "random_forest"):
        out = evaluate_downstream(B, X_train, y_train, X_test, y_test, model_type)
        results[f"train_mse_{model_type}"] = out["train_mse"]
        results[f"test_mse_{model_type}"] = out["test_mse"]
    return results
