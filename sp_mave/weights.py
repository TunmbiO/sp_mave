"""Gaussian kernel weight construction for Classical MAVE.

Supports full Gaussian weighting (all pairwise) with optional top-k
truncation (zero out all but k nearest neighbours), and bandwidth
selection via the median heuristic.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from sp_mave.utils import topk_indices


def median_bandwidth(X: np.ndarray) -> float:
    """Bandwidth via the median heuristic: h = median(pairwise distances).

    This is a common default that adapts to the scale of the data.
    Using the raw median (not divided by sqrt(2)) because the Gaussian
    kernel already has the factor of 2 in the denominator.
    """
    dists = cdist(X, X, metric="euclidean")
    np.fill_diagonal(dists, np.inf)
    med = np.median(dists[dists < np.inf])
    if med < 1e-12:
        raise ValueError(
            "Median pairwise distance is near zero; data may be degenerate."
        )
    return float(med)


def gaussian_weights(
    X: np.ndarray,
    bandwidth: float | str = "auto",
    k: int | None = None,
) -> np.ndarray:
    """Compute normalised Gaussian kernel weight matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix (typically standardised).
    bandwidth : float or "auto"
        Kernel bandwidth h.  "auto" uses the median heuristic.
    k : int or None
        If not None, truncate to top-k neighbours per anchor (after
        computing full Gaussian weights).  The retained weights are
        re-normalised to sum to 1.

    Returns
    -------
    W : np.ndarray, shape (n, n)
        W[i, j] is the weight of observation i when j is the anchor.
        Each column sums to 1.  W[j, j] = 0 (self-weight excluded).
    """
    n = X.shape[0]
    if isinstance(bandwidth, str):
        if bandwidth != "auto":
            raise ValueError(f"bandwidth must be a float or 'auto', got '{bandwidth}'")
        bandwidth = median_bandwidth(X)

    dists_sq = cdist(X, X, metric="sqeuclidean")
    log_weights = -dists_sq / (2.0 * bandwidth ** 2)

    np.fill_diagonal(log_weights, -np.inf)

    # Per-column (per-anchor) normalisation in log-space for stability
    W = np.zeros_like(log_weights)
    for j in range(n):
        col = log_weights[:, j].copy()

        if k is not None:
            # top-k truncation: keep only k largest weights
            active = topk_indices(col, k)
            mask = np.full(n, -np.inf)
            mask[active] = col[active]
            col = mask

        col_max = col.max()
        if np.isfinite(col_max):
            exp_col = np.exp(col - col_max)
            W[:, j] = exp_col / exp_col.sum()
        # else: all -inf → column stays zero (degenerate, should not happen)

    return W


