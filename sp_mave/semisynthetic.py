"""Semi-synthetic response on real covariates (train-only scaling, known B_true)."""

from __future__ import annotations

import numpy as np


def sample_orthonormal_B(p: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample B ∈ R^{p×d} with orthonormal columns (QR of standard Gaussian)."""
    if d > p:
        raise ValueError(f"d={d} cannot exceed p={p}")
    a = rng.standard_normal((p, d))
    q, _ = np.linalg.qr(a, mode="reduced")
    return q[:, :d].astype(np.float64)


def g_friedman_adapted(Z: np.ndarray) -> np.ndarray:
    """Nonlinear g(Z) with interaction; dimension-aware.

    - d == 2: 10*sin(π z1 z2) + 5*z1^2 + 3*z2
    - d == 3: Friedman-style on (z1,z2,z3): sin term + (z3-0.5)^2 + linear in z1,z2
    - d == 1: 10*sin(π z1) + 5*z1^2
    """
    d = Z.shape[1]
    if d == 1:
        z1 = Z[:, 0]
        return 10.0 * np.sin(np.pi * z1) + 5.0 * z1**2
    if d == 2:
        z1, z2 = Z[:, 0], Z[:, 1]
        return 10.0 * np.sin(np.pi * z1 * z2) + 5.0 * z1**2 + 3.0 * z2
    if d == 3:
        z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]
        return (
            10.0 * np.sin(np.pi * z1 * z2)
            + 20.0 * (z3 - 0.5) ** 2
            + 10.0 * z1
            + 5.0 * z2
        )
    raise ValueError(f"g_friedman_adapted supports d in {{1,2,3}}, got d={d}")
