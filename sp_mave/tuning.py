"""Validation-based tuning for SP-MAVE hyperparameters (k and tau).

Tuning procedure
----------------
1. Split the training data into train_inner and validation.
2. Compute the raw cross-fitted RF proximity matrix P once on
   train_inner.  This is the expensive step (RF fitting).
3. For each candidate (k, tau) pair:
   a. Convert P to a weight matrix W via top-k + softmax(tau).
   b. Run the true iterative MAVE pipeline on train_inner with these
      weights (reduced-space local WLS + constrained B-update).
   c. Project train_inner and validation through estimated B.
   d. Fit a downstream model on reduced train_inner, score on validation.
4. Select the (k, tau) pair that minimises validation MSE.
5. Return the best (k, tau) so the caller can refit SP-MAVE on the
   full training set with those values.

Efficiency note: step 2 is O(n * K * T) and dominates.  Steps 3a-3d
are cheap per candidate but step 3b now involves multiple MAVE
iterations (not a one-step OPG).
"""

from __future__ import annotations

from typing import Any

import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from sp_mave.mave import ClassicalMAVE, MAVEBase
from sp_mave.proximities import (
    compute_crossfitted_proximities,
    proximity_to_weights,
)
from sp_mave.utils import check_dimensions, get_logger

logger = get_logger(__name__)


class _TuningSPMAVE(MAVEBase):
    """Lightweight SP-MAVE that accepts a precomputed weight matrix.

    Used internally by the tuner to avoid recomputing RF proximities
    for each (k, tau) candidate.  Inherits the true iterative MAVE
    algorithm (reduced-space local WLS + constrained B-update).
    """

    def __init__(
        self,
        W: np.ndarray,
        d: int = 1,
        max_iters: int = 50,
        tol: float = 1e-6,
        ridge_lambda: float = 1e-6,
        init_method: str = "pca",
        random_state: int = 42,
    ) -> None:
        super().__init__(
            d=d,
            max_iters=max_iters,
            tol=tol,
            ridge_lambda=ridge_lambda,
            init_method=init_method,
            random_state=random_state,
        )
        self._W_fixed = W

    @property
    def method_name(self) -> str:
        return "SPMAVE_tuning"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        return self._W_fixed


def tune_k_tau(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_candidates: list[int],
    tau_candidates: list[float],
    d: int,
    rf_params: dict[str, Any],
    crossfit_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    val_size: float = 0.2,
    downstream_model: str = "linear",
    random_state: int = 42,
    tuning_max_iters: int | None = None,
) -> dict[str, Any]:
    """Select (k, tau) by validation-set MSE.

    Parameters
    ----------
    X_train, y_train : full training data (already standardised).
    k_candidates : list of neighbourhood sizes to evaluate.
    tau_candidates : list of softmax temperatures to evaluate.
    d : reduced dimension.
    rf_params : RandomForest kwargs.
    crossfit_cfg : cross-fitting settings.
    model_cfg : model config dict (for ridge_lambda, max_iters, etc.).
    val_size : fraction of training data held out for validation.
    downstream_model : downstream regressor type.
    random_state : seed for the validation split.
    tuning_max_iters : if set, overrides model_cfg["max_iters"] for
        faster tuning (each candidate uses fewer MAVE iterations).

    Returns
    -------
    dict with keys:
      best_k, best_tau       -- selected hyperparameters
      best_val_mse           -- validation MSE at the optimum
      grid_results           -- list of dicts for every (k, tau) tried
    """
    n, p = check_dimensions(X_train, y_train, context="tune_k_tau")

    # --- Validation split (inside the training portion) ---
    X_inner, X_val, y_inner, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
    )
    n_inner = X_inner.shape[0]
    logger.info(
        "Tuning: train_inner=%d  val=%d  |k_grid|=%d  |tau_grid|=%d",
        n_inner, X_val.shape[0], len(k_candidates), len(tau_candidates),
    )

    # --- Compute raw proximity matrix once on train_inner ---
    P = compute_crossfitted_proximities(
        X_inner, y_inner,
        n_splits=crossfit_cfg.get("n_splits", 5),
        shuffle=crossfit_cfg.get("shuffle", True),
        crossfit_random_state=crossfit_cfg.get("random_state", 42),
        rf_params=rf_params,
    )
    logger.info("Tuning: computed proximity matrix on train_inner (shape %s)", P.shape)

    # Clamp k candidates to valid range
    max_k = n_inner - 1
    k_candidates = [k for k in k_candidates if 2 <= k <= max_k]
    if not k_candidates:
        raise ValueError(
            f"No valid k candidates for n_inner={n_inner}. "
            f"All candidates must satisfy 2 <= k <= {max_k}."
        )

    # --- Grid search ---
    best_val_mse = np.inf
    best_k = k_candidates[0]
    best_tau = tau_candidates[0]
    grid_results: list[dict[str, Any]] = []

    effective_max_iters = (
        tuning_max_iters
        if tuning_max_iters is not None
        else model_cfg.get("max_iters", 50)
    )
    common_model_kw = dict(
        d=d,
        max_iters=effective_max_iters,
        tol=model_cfg.get("tol", 1e-6),
        ridge_lambda=model_cfg.get("ridge_lambda", 0.01),
        init_method=model_cfg.get("init_method", "pca"),
        random_state=model_cfg.get("random_state", 42),
    )

    # Suppress repetitive per-anchor warnings during the grid search;
    # the main fit will log them once if relevant.
    mave_logger = logging.getLogger("sp_mave.mave")
    prev_level = mave_logger.level
    mave_logger.setLevel(logging.ERROR)

    try:
        for k in k_candidates:
            for tau in tau_candidates:
                W = proximity_to_weights(P, k=k, tau=tau)

                estimator = _TuningSPMAVE(W=W, **common_model_kw)
                estimator.fit(X_inner, y_inner)
                B = estimator.B_

                Z_inner = X_inner @ B
                Z_val = X_val @ B

                model = LinearRegression().fit(Z_inner, y_inner)
                val_mse = float(mean_squared_error(y_val, model.predict(Z_val)))

                grid_results.append({
                    "k": k, "tau": tau, "val_mse": val_mse,
                })

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_k = k
                    best_tau = tau
    finally:
        mave_logger.setLevel(prev_level)

    logger.info(
        "Tuning result: best_k=%d  best_tau=%.4f  val_mse=%.4f",
        best_k, best_tau, best_val_mse,
    )

    return {
        "best_k": best_k,
        "best_tau": best_tau,
        "best_val_mse": best_val_mse,
        "grid_results": grid_results,
    }


def tune_k_classical_topk(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_candidates: list[int],
    d: int,
    model_cfg: dict[str, Any],
    val_size: float = 0.2,
    random_state: int = 42,
    tuning_max_iters: int | None = None,
) -> dict[str, Any]:
    """Select *k* for Classical MAVE with top-k Gaussian weights (validation MSE)."""
    n, p = check_dimensions(X_train, y_train, context="tune_k_classical_topk")
    X_inner, X_val, y_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
    )
    n_inner = X_inner.shape[0]
    max_k = n_inner - 1
    k_list = [k for k in k_candidates if 2 <= k <= max_k]
    if not k_list:
        raise ValueError(
            f"No valid k in {k_candidates} for n_inner={n_inner} (need 2 <= k <= {max_k})"
        )

    effective_max_iters = (
        tuning_max_iters
        if tuning_max_iters is not None
        else model_cfg.get("max_iters", 50)
    )
    common_kw = dict(
        d=d,
        max_iters=effective_max_iters,
        tol=model_cfg.get("tol", 1e-6),
        ridge_lambda=model_cfg.get("ridge_lambda", 0.01),
        init_method=model_cfg.get("init_method", "pca"),
        random_state=model_cfg.get("random_state", 42),
    )
    bandwidth = model_cfg.get("bandwidth", "auto")

    best_val_mse = np.inf
    best_k = k_list[0]
    grid_results: list[dict[str, Any]] = []

    mave_logger = logging.getLogger("sp_mave.mave")
    prev_level = mave_logger.level
    mave_logger.setLevel(logging.ERROR)
    try:
        for k in k_list:
            est = ClassicalMAVE(bandwidth=bandwidth, k=k, **common_kw)
            est.fit(X_inner, y_inner)
            B = est.B_
            Z_inner = X_inner @ B
            Z_val = X_val @ B
            lr = LinearRegression().fit(Z_inner, y_inner)
            val_mse = float(mean_squared_error(y_val, lr.predict(Z_val)))
            grid_results.append({"k": k, "val_mse": val_mse})
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_k = k
    finally:
        mave_logger.setLevel(prev_level)

    logger.info(
        "Classical top-k tuning: best_k=%d  val_mse=%.4f", best_k, best_val_mse
    )
    return {
        "best_k": best_k,
        "best_val_mse": best_val_mse,
        "grid_results": grid_results,
    }


def tune_k_classical_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_candidates: list[int],
    d: int,
    model_cfg: dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
    tuning_max_iters: int | None = None,
) -> dict[str, Any]:
    """Select k for ClassicalMAVE top-k via K-fold cross-validation.

    For each candidate k, runs K-fold CV on X_train: fit ClassicalMAVE
    on each training fold, project the validation fold, fit a linear
    regression on the reduced training fold, score on the reduced
    validation fold.  Selects k* = argmin(mean CV-MSE).

    Parameters
    ----------
    X_train, y_train : training data (already standardised).
    k_candidates : neighbourhood sizes to evaluate.
    d : reduced dimension.
    model_cfg : model config dict (ridge_lambda, max_iters, init_method,
        bandwidth, random_state).
    n_splits : number of CV folds (default 5).
    random_state : seed for the KFold splitter.
    tuning_max_iters : if set, overrides model_cfg["max_iters"] for speed.

    Returns
    -------
    dict with keys:
      best_k        -- selected neighbourhood size
      best_cv_mse   -- mean CV-MSE at best_k
      cv_scores     -- {k: [fold_mse, ...]} for all candidates
    """
    n, _ = check_dimensions(X_train, y_train, context="tune_k_classical_cv")

    max_k = n - 1
    k_list = sorted({min(k, max_k) for k in k_candidates if k >= 2})
    if not k_list:
        raise ValueError(
            f"No valid k candidates for n={n}. "
            f"All candidates must satisfy 2 <= k <= {max_k}."
        )

    effective_max_iters = (
        tuning_max_iters if tuning_max_iters is not None
        else model_cfg.get("max_iters", 50)
    )
    bandwidth = model_cfg.get("bandwidth", "auto")
    common_kw = dict(
        d=d,
        max_iters=effective_max_iters,
        tol=model_cfg.get("tol", 1e-6),
        ridge_lambda=model_cfg.get("ridge_lambda", 0.01),
        init_method=model_cfg.get("init_method", "pca"),
        random_state=model_cfg.get("random_state", 42),
    )

    cv_scores: dict[int, list[float]] = {k: [] for k in k_list}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mave_logger = logging.getLogger("sp_mave.mave")
    prev_level = mave_logger.level
    mave_logger.setLevel(logging.ERROR)
    try:
        for fold_train_idx, fold_val_idx in kf.split(X_train):
            X_fold_tr = X_train[fold_train_idx]
            y_fold_tr = y_train[fold_train_idx]
            X_fold_val = X_train[fold_val_idx]
            y_fold_val = y_train[fold_val_idx]
            n_fold_tr = len(fold_train_idx)

            for k in k_list:
                k_eff = min(k, n_fold_tr - 1)
                est = ClassicalMAVE(bandwidth=bandwidth, k=k_eff, **common_kw)
                est.fit(X_fold_tr, y_fold_tr)
                B = est.B_
                Z_tr = X_fold_tr @ B
                Z_val = X_fold_val @ B
                lr = LinearRegression().fit(Z_tr, y_fold_tr)
                fold_mse = float(mean_squared_error(y_fold_val, lr.predict(Z_val)))
                cv_scores[k].append(fold_mse)
    finally:
        mave_logger.setLevel(prev_level)

    mean_cv = {k: float(np.mean(v)) for k, v in cv_scores.items()}
    best_k = min(k_list, key=lambda k: mean_cv[k])

    logger.info(
        "tune_k_classical_cv: best_k=%d  mean_cv_mse=%.4f",
        best_k, mean_cv[best_k],
    )
    return {
        "best_k": best_k,
        "best_cv_mse": mean_cv[best_k],
        "cv_scores": cv_scores,
    }
