"""SP-MAVE: Supervised Proximity MAVE.

Replaces the Gaussian kernel weights of classical MAVE with Random
Forest proximity weights constructed via K-fold cross-fitting.

Weight construction
-------------------
1. Cross-fit RF proximities within the training sample (K-fold; see
   proximities.py for why cross-fitting is essential — a standard RF
   inflates self-proximity to 1.0 for every training point, which
   distorts the neighbourhood structure for MAVE's anchor-based WLS).
2. For each anchor j, retain only the top-k neighbours by proximity.
3. Normalise the retained proximities with a softmax (temperature tau).
4. Use the resulting weight matrix in the true iterative MAVE pipeline
   (inherited from MAVEBase): reduced-space local WLS + constrained
   B-update.

Iteration behaviour
-------------------
The RF proximity weights are fixed (they do not depend on B), but the
algorithm still genuinely iterates because the reduced-space local
regressions use projected coordinates B^T (X_i - X_j) which change as
B is updated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sp_mave.mave import MAVEBase
from sp_mave.proximities import compute_rf_weights
from sp_mave.utils import get_logger

logger = get_logger(__name__)


class SPMAVE(MAVEBase):
    """True iterative MAVE with cross-fitted Random Forest proximity weights.

    Parameters
    ----------
    d : int
        Target reduced dimension.
    k : int
        Number of neighbours to retain per anchor.  A practical rule is
        k = floor(0.20 × n); experiments use k=20 at n=100, k=40 at n=200,
        k=100 at n=500.
    tau : float
        Softmax temperature for proximity weighting (τ=1.0 in all experiments).
    rf_params : dict
        Keyword arguments for sklearn.ensemble.RandomForestRegressor.
    crossfit_cfg : dict
        Cross-fitting settings (n_splits, shuffle, random_state).
    """

    def __init__(
        self,
        d: int = 1,
        k: int = 20,
        tau: float = 1.0,
        rf_params: dict[str, Any] | None = None,
        crossfit_cfg: dict[str, Any] | None = None,
        symmetrise: bool = False,
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
        self.k = k
        self.tau = tau
        self.rf_params = rf_params or {}
        self.crossfit_cfg = crossfit_cfg or {}
        self.symmetrise = symmetrise

        # Cached weight matrix (computed once, reused every iteration)
        self._W_cache: np.ndarray | None = None
        self._P_raw: np.ndarray | None = None

    @property
    def method_name(self) -> str:
        return "SPMAVE"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        """Return RF proximity weights (computed and cached on first call).

        The weights do not depend on B; caching avoids redundant RF
        training on subsequent iterations.
        """
        if self._W_cache is not None:
            return self._W_cache

        logger.info(
            "Computing cross-fitted RF proximities "
            "(n=%d, k=%d, tau=%.3f, n_splits=%d, n_estimators=%s)",
            X.shape[0],
            self.k,
            self.tau,
            self.crossfit_cfg.get("n_splits", 5),
            self.rf_params.get("n_estimators", "default"),
        )

        W, P = compute_rf_weights(
            X, y,
            k=self.k,
            tau=self.tau,
            crossfit_cfg=self.crossfit_cfg,
            rf_params=self.rf_params,
            symmetrise=self.symmetrise,
        )
        self._W_cache = W
        self._P_raw = P
        return W

    def fit(self, X: np.ndarray, y: np.ndarray) -> SPMAVE:
        """Fit SP-MAVE (invalidates cached weights from prior .fit calls)."""
        self._W_cache = None
        self._P_raw = None
        super().fit(X, y)
        if self.result_ is not None and self._P_raw is not None:
            self.result_.extra["raw_proximities"] = self._P_raw
        return self
