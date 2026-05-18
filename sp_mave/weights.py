"""Gaussian kernel weight construction for Classical MAVE.

Supports:
- full Gaussian weighting (all pairwise)
- optional top-k truncation (zero out all but k nearest neighbours)
- bandwidth selection via the median heuristic
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


def gaussian_opg_weights_full(
    X: np.ndarray,
    bandwidth: float | str = "auto",
) -> np.ndarray:
    """Full-space Gaussian weights for OPG: **includes self** in the kernel.

    For anchor *j*,  w_ij ∝ exp(-||x_i - x_j||^2 / (2 h^2)) for all *i*,
    then column *j* is renormalised to sum to 1.

    Convention matches the rest of the package: ``W[i, j]`` is the weight on
    observation *i* when *j* is the anchor (column-normalised).

    Notes
    -----
    This differs from :func:`gaussian_weights`, which zeros the diagonal
    before normalisation (MAVE classical kernel convention).
    """
    n = X.shape[0]
    if isinstance(bandwidth, str):
        if bandwidth != "auto":
            raise ValueError(f"bandwidth must be a float or 'auto', got '{bandwidth}'")
        bandwidth = median_bandwidth(X)

    dists_sq = cdist(X, X, metric="sqeuclidean")
    h2 = 2.0 * float(bandwidth) ** 2
    log_w = -dists_sq / h2
    W = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        col = log_w[:, j]
        cmax = col.max()
        ex = np.exp(col - cmax)
        s = ex.sum()
        if s <= 0:
            raise ValueError("Gaussian OPG weights: degenerate column (sum=0).")
        W[:, j] = ex / s
    return W


def topk_gaussian_opg_weights(
    X: np.ndarray,
    k: int,
    bandwidth: float | str = "auto",
) -> np.ndarray:
    """Top-*k* Gaussian weights for OPG: *k* nearest neighbours **excluding self**.

    For each anchor *j*, among indices ``i != j`` take the *k* smallest
    Euclidean distances, assign unnormalised Gaussian weights on those
    distances, then renormalise the column to sum to 1.  All other entries
    are 0.

    Raises
    ------
    ValueError
        If ``k < 1`` or ``k >= n`` (cannot pick *k* other points).
    """
    n = X.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= n:
        raise ValueError(
            f"topk_gaussian_opg_weights requires k < n to exclude self "
            f"(got k={k}, n={n})"
        )
    if isinstance(bandwidth, str):
        if bandwidth != "auto":
            raise ValueError(f"bandwidth must be a float or 'auto', got '{bandwidth}'")
        bandwidth = median_bandwidth(X)

    dists_sq = cdist(X, X, metric="sqeuclidean")
    h2 = 2.0 * float(bandwidth) ** 2
    W = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        col = dists_sq[:, j].astype(np.float64, copy=True)
        col[j] = np.inf
        # k smallest distances among i != j
        idx = np.argpartition(col, k - 1)[:k]
        d2 = dists_sq[idx, j]
        w = np.exp(-d2 / h2)
        s = w.sum()
        if s <= 0:
            raise ValueError(
                f"Top-k Gaussian weights: zero sum at anchor j={j} (degenerate kNN)."
            )
        W[idx, j] = w / s
    return W


def uniform_knn_opg_weights(X: np.ndarray, k: int) -> np.ndarray:
    """Uniform weights on the *k* nearest neighbours **excluding self**.

    For anchor *j*, ``W[i,j] = 1/k`` for the *k* training points closest to
    ``x_j`` (other than *j*), else 0.  Each column sums to 1.
    """
    n = X.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= n:
        raise ValueError(
            f"uniform_knn_opg_weights requires k < n (got k={k}, n={n})"
        )
    dists_sq = cdist(X, X, metric="sqeuclidean")
    W = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        col = dists_sq[:, j].astype(np.float64, copy=True)
        col[j] = np.inf
        idx = np.argpartition(col, k - 1)[:k]
        W[idx, j] = 1.0 / k
    return W
