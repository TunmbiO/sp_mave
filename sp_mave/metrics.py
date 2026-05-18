"""Subspace recovery metrics.

All functions compare an estimated subspace B_est (p x d_est) against a
ground-truth subspace B_true (p x d_true).

Metrics implemented
-------------------
- Projection Frobenius distance
- Grassmann distance (geodesic based on principal angles)
- Principal angles (the raw angles themselves)
- Vector correlation (average cosine of principal angles)

What is treated as truth
------------------------
For Friedman-1 the truth is span{e_1, ..., e_5} because all five
informative features enter the response through nonlinear / interaction
terms that cannot be compressed to fewer independent linear combinations.

Limitation: these metrics measure alignment of *subspaces*, not recovery
of individual directions.  Two orthonormal bases that span the same
subspace are considered identical even if the basis vectors differ.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd


def _projection_matrix(B: np.ndarray) -> np.ndarray:
    """Orthogonal projector P = B (B^T B)^{-1} B^T.

    When B has orthonormal columns this simplifies to B B^T.
    We use the general formula for robustness.
    """
    return B @ np.linalg.solve(B.T @ B, B.T)


def principal_angles(B_est: np.ndarray, B_true: np.ndarray) -> np.ndarray:
    """Compute principal angles (in radians) between two subspaces.

    The principal angles theta_1, ..., theta_min(d1,d2) are defined via
    the SVD of B_est^T B_true.

    Returns sorted angles in ascending order.
    """
    _, s, _ = svd(B_est.T @ B_true, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.sort(np.arccos(s))


def projection_distance(B_est: np.ndarray, B_true: np.ndarray) -> float:
    """Frobenius norm of the difference of projection matrices.

    d_P = || P_est - P_true ||_F

    Range: [0, sqrt(2 * min(d1, d2))].  Zero means perfect alignment.
    """
    P_est = _projection_matrix(B_est)
    P_true = _projection_matrix(B_true)
    return float(np.linalg.norm(P_est - P_true, "fro"))


def grassmann_distance(B_est: np.ndarray, B_true: np.ndarray) -> float:
    """Geodesic distance on the Grassmann manifold.

    d_G = sqrt(sum_i theta_i^2)

    where theta_i are the principal angles.
    """
    angles = principal_angles(B_est, B_true)
    return float(np.sqrt(np.sum(angles ** 2)))


def vector_correlation(B_est: np.ndarray, B_true: np.ndarray) -> float:
    """Average cosine of the principal angles.

    r_V = (1/min(d1,d2)) * sum_i cos(theta_i)

    Range: [0, 1].  One means perfect alignment.
    """
    angles = principal_angles(B_est, B_true)
    if len(angles) == 0:
        return 0.0
    return float(np.mean(np.cos(angles)))


def compute_all_metrics(
    B_est: np.ndarray,
    B_true: np.ndarray,
) -> dict[str, float]:
    """Compute all subspace recovery metrics.

    Returns a dict with keys:
      projection_distance, grassmann_distance, vector_correlation
    """
    return {
        "projection_distance": projection_distance(B_est, B_true),
        "grassmann_distance": grassmann_distance(B_est, B_true),
        "vector_correlation": vector_correlation(B_est, B_true),
    }
