"""Preprocessing utilities: standardisation and train/test splitting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sp_mave.semisynthetic import g_friedman_adapted, sample_orthonormal_B
from sp_mave.types import DatasetBundle


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Column-wise zero-mean, unit-variance standardisation.

    Fit on *X_train* only; transform both splits.
    """
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler


def split_bundle(
    bundle: DatasetBundle,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a DatasetBundle into (X_train, X_test, y_train, y_test).

    Standardisation is applied: fitted on training data, applied to both.

    Semi-synthetic bundles (``metadata['semi_synthetic']``) ignore the
    placeholder ``y`` in the bundle: split raw ``X``, scale train-only,
    sample ``B_true``, then set ``y = g(X_std @ B_true) + ε`` on each
    split and assign ``bundle.true_subspace = B_true`` (scaled space).
    """
    if bundle.metadata.get("semi_synthetic"):
        return _split_semisynthetic(bundle, cfg)

    split_cfg = cfg["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    X_train, X_test, _ = standardize(X_train, X_test)
    return X_train, X_test, y_train, y_test


def _split_semisynthetic(
    bundle: DatasetBundle,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    md = bundle.metadata
    d_true = int(md.get("d_true", cfg["model"].get("d", 3)))
    b_true_seed = int(md.get("b_true_seed", 0))
    noise_std = float(md.get("noise_std", 1.0))

    split_cfg = cfg["split"]
    rs = int(split_cfg["random_state"])
    X_train, X_test, _, _ = train_test_split(
        bundle.X,
        bundle.y,
        test_size=split_cfg["test_size"],
        random_state=rs,
    )
    X_train, X_test, _ = standardize(X_train, X_test)
    p = X_train.shape[1]
    rng_b = np.random.default_rng(b_true_seed)
    B_true = sample_orthonormal_B(p, d_true, rng_b)
    Z_tr = X_train @ B_true
    Z_te = X_test @ B_true
    rng_tr = np.random.default_rng(rs + 17_389)
    rng_te = np.random.default_rng(rs + 17_399)
    y_train = g_friedman_adapted(Z_tr) + noise_std * rng_tr.standard_normal(
        Z_tr.shape[0]
    )
    y_test = g_friedman_adapted(Z_te) + noise_std * rng_te.standard_normal(
        Z_te.shape[0]
    )
    bundle.true_subspace = B_true.astype(np.float64)
    return X_train, X_test, y_train, y_test
