"""Preprocessing utilities: standardisation."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Column-wise zero-mean, unit-variance standardisation.

    Fit on *X_train* only; transform both splits.
    """
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler
