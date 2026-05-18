"""One-step OPG (Outer Product of Gradients) estimator — preserved baseline.

This module contains the *original* estimator that was previously the main
implementation.  It performs full-space local WLS (beta_j in R^p) followed
by an OPG matrix update:

    M = (1/n) sum_j beta_j beta_j^T,     B = top-d eigenvectors of M.

Because the full-space WLS does not depend on B (the design matrix uses
raw differences X_i - X_j, not reduced coordinates), the algorithm
converges in one step when weights are fixed.

This is kept as a separate baseline called "OPG" so that it can be
compared against the true iterative MAVE estimator in ``mave.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sp_mave.proximities import compute_rf_weights
from sp_mave.types import MAVEResult
from sp_mave.utils import check_dimensions, get_logger, orthonormalize
from sp_mave.weights import (
    gaussian_opg_weights_full,
    gaussian_weights,
    topk_gaussian_opg_weights,
    uniform_knn_opg_weights,
)

logger = get_logger(__name__)


# ======================================================================
# OPG Base
# ======================================================================


class OPGBase(ABC):
    """Abstract base for the one-step OPG estimator.

    Subclasses must implement ``_compute_weights``.
    """

    def __init__(
        self,
        d: int = 1,
        max_iters: int = 50,
        tol: float = 1e-6,
        ridge_lambda: float = 1e-6,
        init_method: str = "pca",
        random_state: int = 42,
    ) -> None:
        self.d = d
        self.max_iters = max_iters
        self.tol = tol
        self.ridge_lambda = ridge_lambda
        self.init_method = init_method
        self.random_state = random_state

        self.B_: np.ndarray | None = None
        self.result_: MAVEResult | None = None

    @abstractmethod
    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        """Return (n, n) weight matrix.  Column j is the weight vector for anchor j."""

    @property
    @abstractmethod
    def method_name(self) -> str: ...

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_B(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        if self.d > p:
            raise ValueError(f"d={self.d} exceeds p={p}")
        X_centred = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centred, full_matrices=False)
        B = Vt[: self.d].T
        return orthonormalize(B)

    # ------------------------------------------------------------------
    # Full-space local WLS
    # ------------------------------------------------------------------

    def _local_wls(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-anchor WLS in R^p with adaptive ridge regularisation.

        Returns
        -------
        alphas : (n,)   local intercepts a_j
        betas  : (n, p) local gradient estimates beta_j
        """
        n, p = X.shape
        alphas = np.zeros(n)
        betas = np.zeros((n, p))
        dim = 1 + p
        _warned_underdet = False

        for j in range(n):
            w_col = W[:, j]
            active = np.where(w_col > 0)[0]
            if len(active) < 2:
                continue

            if len(active) < dim and not _warned_underdet:
                logger.warning(
                    "Under-determined local WLS: n_active=%d < dim=%d.  "
                    "Adaptive ridge regularisation is applied.",
                    len(active), dim,
                )
                _warned_underdet = True

            w_active = w_col[active]
            delta = X[active] - X[j]
            ones = np.ones((len(active), 1))
            Z = np.hstack([ones, delta])
            y_active = y[active]

            Zw = Z * w_active[:, None]
            gram = Zw.T @ Z

            trace_gram = max(np.trace(gram), 1e-12)
            lambda_eff = self.ridge_lambda * trace_gram / dim
            ridge = lambda_eff * np.eye(dim)
            ridge[0, 0] = 0.0

            A = gram + ridge
            rhs = Zw.T @ y_active

            try:
                c = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, rhs, rcond=None)[0]

            alphas[j] = c[0]
            betas[j] = c[1:]

        return alphas, betas

    # ------------------------------------------------------------------
    # OPG update
    # ------------------------------------------------------------------

    def _opg_update(self, betas: np.ndarray) -> np.ndarray:
        """B = top-d eigenvectors of (1/n) betas^T betas."""
        n = betas.shape[0]
        M = (betas.T @ betas) / n
        eigvals, eigvecs = np.linalg.eigh(M)
        B = eigvecs[:, -self.d:]
        return orthonormalize(B)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_objective(
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray,
        alphas: np.ndarray,
        betas: np.ndarray,
    ) -> float:
        """L = sum_j sum_i w_ij (y_i - a_j - beta_j^T (x_i - x_j))^2"""
        n = X.shape[0]
        loss = 0.0
        for j in range(n):
            w_col = W[:, j]
            active = np.where(w_col > 0)[0]
            if len(active) == 0:
                continue
            delta = X[active] - X[j]
            residuals = y[active] - alphas[j] - delta @ betas[j]
            loss += np.dot(w_col[active], residuals ** 2)
        return float(loss)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> OPGBase:
        n, p = check_dimensions(X, y, context=self.method_name)
        if self.d > p:
            raise ValueError(f"d={self.d} > p={p}")

        B = self._initialize_B(X)
        loss_history: list[float] = []
        delta_B_history: list[float] = []
        rel_change_history: list[float] = []
        prev_loss = np.inf
        W = np.empty(0)

        for iteration in range(self.max_iters):
            W = self._compute_weights(X, y, B)
            alphas, betas = self._local_wls(X, y, W)
            B_new = self._opg_update(betas)
            loss = self._compute_objective(X, y, W, alphas, betas)
            loss_history.append(loss)

            rel_change = (
                abs(prev_loss - loss) / max(abs(prev_loss), 1e-12)
                if iteration > 0
                else float("inf")
            )
            rel_change_history.append(rel_change)
            delta_B = float(np.linalg.norm(B_new - B, "fro"))
            delta_B_history.append(delta_B)

            logger.debug(
                "%s iter %d: loss=%.6e", self.method_name, iteration, loss
            )

            if iteration > 0 and rel_change < self.tol:
                logger.info(
                    "%s converged at iter %d (rel_change=%.2e)",
                    self.method_name, iteration, rel_change,
                )
                B = B_new
                break

            prev_loss = loss
            B = B_new

        self.B_ = B
        self.result_ = MAVEResult(
            B=B,
            loss_history=loss_history,
            n_iters=len(loss_history),
            weights=W,
            method_name=self.method_name,
            history={
                "objective": list(loss_history),
                "rel_change": rel_change_history,
                "delta_B": delta_B_history,
            },
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.B_ is None:
            raise RuntimeError("Call .fit() before .transform()")
        return X @ self.B_


# ======================================================================
# OPG + Gaussian kernel weights
# ======================================================================


class OPGClassical(OPGBase):
    """OPG estimator with Gaussian kernel weights (one-step baseline).

    Parameters
    ----------
    d : int
        Target reduced dimension.
    bandwidth : float or "auto"
        Gaussian kernel bandwidth.
    k : int or None
        If set, truncate to top-k neighbours per anchor.
    """

    def __init__(
        self,
        d: int = 1,
        bandwidth: float | str = "auto",
        k: int | None = None,
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
        self.bandwidth = bandwidth
        self.k = k

    @property
    def method_name(self) -> str:
        suffix = f"_k{self.k}" if self.k is not None else ""
        return f"OPGClassical{suffix}"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        return gaussian_weights(X, bandwidth=self.bandwidth, k=self.k)


# ======================================================================
# OPG variants (weight construction only; same backbone as OPGBase)
# ======================================================================


class GaussianOPG(OPGBase):
    """OPG with full Gaussian kernel weights (**self included**), column-normalised."""

    def __init__(
        self,
        d: int = 1,
        bandwidth: float | str = "auto",
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
        self.bandwidth = bandwidth

    @property
    def method_name(self) -> str:
        return "GaussianOPG"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        return gaussian_opg_weights_full(X, bandwidth=self.bandwidth)


class TopKGaussianOPG(OPGBase):
    """OPG with *k*NN-truncated Gaussian weights (self excluded from neighbourhood)."""

    def __init__(
        self,
        d: int = 1,
        k: int = 20,
        bandwidth: float | str = "auto",
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
        self.bandwidth = bandwidth

    @property
    def method_name(self) -> str:
        return f"TopKGaussianOPG_k{self.k}"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        return topk_gaussian_opg_weights(
            X, k=self.k, bandwidth=self.bandwidth,
        )


class UniformKNNOPG(OPGBase):
    """OPG with uniform weights on *k* nearest neighbours (self excluded)."""

    def __init__(
        self,
        d: int = 1,
        k: int = 20,
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

    @property
    def method_name(self) -> str:
        return f"UniformKNNOPG_k{self.k}"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        return uniform_knn_opg_weights(X, k=self.k)


class RFProximityOPG(OPGBase):
    """OPG with cross-fitted RF proximities and top-*k* softmax weights.

    Raw proximities have ``P[j, j] = 0``; each anchor's column uses *k*
    neighbours among ``i != j`` only.
    """

    def __init__(
        self,
        d: int = 1,
        k: int = 20,
        tau: float = 0.1,
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

        self._W_cache: np.ndarray | None = None
        self._P_raw: np.ndarray | None = None

    @property
    def method_name(self) -> str:
        return "RFProximityOPG"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        if self._W_cache is not None:
            return self._W_cache

        logger.info(
            "RFProximityOPG: cross-fitted RF proximities (n=%d, k=%d, tau=%.3f)",
            X.shape[0], self.k, self.tau,
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> RFProximityOPG:
        self._W_cache = None
        self._P_raw = None
        super().fit(X, y)
        if self.result_ is not None and self._P_raw is not None:
            self.result_.extra["raw_proximities"] = self._P_raw
        return self


# Backward compatibility alias
OPGSPMAVE = RFProximityOPG
