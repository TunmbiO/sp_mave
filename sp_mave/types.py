"""Data types for the SP-MAVE project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DatasetBundle:
    """Container for a dataset with optional ground-truth subspace.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    feature_names : list[str]
        Human-readable names for each feature column.
    true_subspace : np.ndarray or None, shape (p, d)
        Orthonormal basis for the true sufficient dimension reduction subspace.
        None when ground truth is unavailable (e.g. real-world data).
    metadata : dict
        Arbitrary dataset-specific information (e.g. generating function,
        noise level, data source).
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    true_subspace: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-d, got shape {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1-d, got shape {self.y.shape}")
        n, p = self.X.shape
        if self.y.shape[0] != n:
            raise ValueError(
                f"X has {n} rows but y has {self.y.shape[0]} entries"
            )
        if len(self.feature_names) != p:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != p={p}"
            )
        if self.true_subspace is not None:
            self.true_subspace = np.asarray(self.true_subspace, dtype=np.float64)
            if self.true_subspace.shape[0] != p:
                raise ValueError(
                    f"true_subspace has {self.true_subspace.shape[0]} rows, "
                    f"expected p={p}"
                )


@dataclass
class MAVEResult:
    """Result container for a fitted MAVE-style estimator.

    Parameters
    ----------
    B : np.ndarray, shape (p, d)
        Estimated projection matrix (orthonormal columns).
    loss_history : list[float]
        MAVE objective value at each iteration.
    n_iters : int
        Number of iterations actually performed.
    weights : np.ndarray, shape (n, n)
        Weight matrix used in the final fit.
    method_name : str
        Identifier string (e.g. "ClassicalMAVE", "SPMAVE").
    extra : dict
        Method-specific diagnostics (e.g. raw proximities for SP-MAVE).
    """

    B: np.ndarray
    loss_history: list[float]
    n_iters: int
    weights: np.ndarray
    method_name: str
    extra: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
