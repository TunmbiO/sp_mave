"""Random Forest proximity computation with K-fold cross-fitting.

Cross-fitting procedure
-----------------------
Split the n training observations into K folds.  For each fold k:
  1. Train a RandomForestRegressor on the K-1 other folds.
  2. Apply the trained forest to ALL n observations to obtain leaf
     assignments (n x T matrix, one column per tree).
  3. For each held-out observation i in fold k, compute its proximity
     to every other observation j as the fraction of trees in which
     i and j land in the same terminal leaf.

The resulting proximity matrix is generally NOT symmetric: observation
i's proximities come from the forest that held out fold(i), while
observation j's proximities come from a different forest (unless i and
j belong to the same fold).  For anchor-based MAVE this is acceptable
because we use each anchor j's proximity row independently.

After assembling the raw proximity matrix:
  - Set P[j, j] = 0 for every j (self is not a neighbour; matches classical
    kernel-style weight matrices).
  - For each anchor j, keep only the top-k neighbours (highest proximity among
    i != j).
  - Apply softmax normalisation with temperature tau over the retained
    neighbours.
  - All other weights are set to zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from sp_mave.utils import softmax, topk_indices


def _build_rf(rf_params: dict[str, Any]) -> RandomForestRegressor:
    """Instantiate a RandomForestRegressor with sanitised parameters."""
    params = dict(rf_params)
    params.setdefault("n_jobs", -1)
    return RandomForestRegressor(**params)


def compute_crossfitted_proximities(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    crossfit_random_state: int = 42,
    rf_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute cross-fitted Random Forest proximity matrix.

    Parameters
    ----------
    X : (n, p) feature matrix (training data only).
    y : (n,) response vector.
    n_splits : number of cross-fitting folds.
    shuffle : whether to shuffle before splitting.
    crossfit_random_state : seed for the KFold splitter.
    rf_params : kwargs forwarded to RandomForestRegressor.

    Returns
    -------
    P : (n, n) raw proximity matrix.  P[i, j] is the fraction of trees
        (from the forest that held out fold(i)) in which observations
        i and j land in the same leaf.  Diagonal entries are set to **0**
        after computation (self is excluded from the proximity matrix).
    """
    if rf_params is None:
        rf_params = {}
    n = X.shape[0]
    P = np.zeros((n, n), dtype=np.float64)

    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=crossfit_random_state,
    )

    for train_idx, held_idx in kf.split(X):
        rf = _build_rf(rf_params)
        rf.fit(X[train_idx], y[train_idx])

        # Leaf assignment matrix for ALL observations: (n, n_trees)
        leaves_all = rf.apply(X)

        # For each held-out observation, compute proximity to every other obs
        leaves_held = leaves_all[held_idx]          # (|held|, n_trees)
        for local_i, global_i in enumerate(held_idx):
            # Fraction of trees where global_i and each observation share a leaf
            same_leaf = (leaves_held[local_i] == leaves_all)  # (n, n_trees)
            P[global_i, :] = same_leaf.mean(axis=1)

    np.fill_diagonal(P, 0.0)
    return P


def proximity_to_weights(P: np.ndarray, k: int, tau: float) -> np.ndarray:
    """Convert raw proximity matrix to MAVE weight matrix.

    For each anchor j (column j of the output):
      1. Mask P[j, j] so the anchor never enters its own neighbourhood (same
         convention as :func:`sp_mave.weights.gaussian_weights`).
      2. Select the top-k neighbours by the remaining proximities in row j.
      3. Apply softmax with temperature tau over those k values.
      4. All other entries are zero.

    Parameters
    ----------
    P : (n, n) raw proximity matrix from cross-fitting (diagonal typically 0).
    k : number of neighbours to retain per anchor.
    tau : softmax temperature (smaller = sharper weighting).

    Returns
    -------
    W : (n, n) weight matrix.  W[i, j] is the weight of observation i
        when j is the anchor.  W[j, j] = 0 and column j sums to 1.
    """
    n = P.shape[0]
    W = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        row = P[j, :].astype(np.float64, copy=True)
        row[j] = -np.inf
        active = topk_indices(row, k)
        W[active, j] = softmax(row[active], tau=tau)

    return W


def symmetrise_proximities(P: np.ndarray) -> np.ndarray:
    """Return (P + P^T) / 2.

    Why this exists
    ---------------
    The cross-fitted proximity matrix is asymmetric by construction:
    row i comes from the forest that held out fold(i), which is generally
    different from the forest that held out fold(j).  Symmetrising
    averages the two directional estimates and can reduce variance.

    Why it is NOT enabled by default
    --------------------------------
    Symmetrising mixes in proximity values where neither observation was
    held out from training.  Specifically, P[i,j] is computed from a
    forest that excluded fold(i), so observation j may have been in that
    forest's training set.  P[j,i] comes from a forest that excluded
    fold(j), where observation i may have been in training.  Averaging
    the two does not introduce a *new* leakage path (each half already
    has one "seen" observation), but it blurs the clean cross-fitting
    semantics where each anchor's row comes from a single held-out
    forest.

    For anchor-based MAVE, where each anchor j uses its own row P[j,:]
    independently, the asymmetric version has a cleaner interpretation.
    Symmetrising is offered as an optional variant for ablation.

    Parameters
    ----------
    P : (n, n) raw proximity matrix.

    Returns
    -------
    P_sym : (n, n) symmetric proximity matrix; diagonal is 0 if *P* had zero
        diagonal (the usual cross-fitted output).
    """
    P_sym = (P + P.T) / 2.0
    np.fill_diagonal(P_sym, 0.0)
    return P_sym


def compute_rf_weights(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    tau: float,
    crossfit_cfg: dict[str, Any],
    rf_params: dict[str, Any],
    symmetrise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end: cross-fitted proximities -> top-k softmax weights.

    Parameters
    ----------
    symmetrise : bool
        If True, symmetrise the raw proximity matrix via (P + P^T) / 2
        before applying top-k and softmax.  Default False (see
        ``symmetrise_proximities`` docstring for rationale).

    Returns
    -------
    W : (n, n) final weight matrix (columns sum to 1).
    P : (n, n) raw proximity matrix (for diagnostics / plotting).
    """
    P = compute_crossfitted_proximities(
        X, y,
        n_splits=crossfit_cfg.get("n_splits", 5),
        shuffle=crossfit_cfg.get("shuffle", True),
        crossfit_random_state=crossfit_cfg.get("random_state", 42),
        rf_params=rf_params,
    )
    P_used = symmetrise_proximities(P) if symmetrise else P
    W = proximity_to_weights(P_used, k=k, tau=tau)
    return W, P
