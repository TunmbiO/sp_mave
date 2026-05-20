"""Data types for the SP-MAVE project."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
