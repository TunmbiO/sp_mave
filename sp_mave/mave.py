"""True iterative MAVE estimator for sufficient dimension reduction.

Algorithm overview
------------------
Both Classical MAVE and SP-MAVE minimise the MAVE objective

    L(B, {a_j, b_j}) = sum_j sum_i w_ij (Y_i - a_j - b_j^T B^T (X_i - X_j))^2
    subject to  B^T B = I_d

via alternating minimisation:

  1. **Local WLS in reduced space** (given B, solve for {a_j, b_j}):
     For each anchor j, form design matrix [1, B^T(X_i - X_j)] which is
     (n_active x (1+d)), and solve the weighted least squares problem for
     a_j (scalar) and b_j (R^d).  Because the design is (1+d)-dimensional,
     this is always well-conditioned when k > d + 1.

  2. **B-update** (given {a_j, b_j}, solve for B):
     Setting dL/dB = 0 yields the matrix equation

         C = sum_j Q_j B (b_j b_j^T)

     where  C = sum_j (sum_i w_ij r_ij delta_ij) b_j^T    (p x d)
            Q_j = sum_i w_ij delta_ij delta_ij^T            (p x p)
            r_ij = Y_i - a_j
            delta_ij = X_i - X_j

     Vectorising:  vec(C) = H vec(B)  where H is (pd x pd) with blocks
     H_{l,m} = sum_j b_j[l] b_j[m] Q_j.  We solve this system (with
     Tikhonov regularisation) then orthonormalise the result.

Convergence
-----------
Even when weights are fixed (Classical MAVE with Gaussian kernel, or
SP-MAVE with RF proximities), the algorithm genuinely iterates because
the reduced-space local regressions depend on B through the projected
coordinates B^T (X_i - X_j).

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sp_mave.types import MAVEResult
from sp_mave.utils import check_dimensions, get_logger, orthonormalize
from sp_mave.weights import gaussian_weights

logger = get_logger(__name__)


class MAVEBase(ABC):
    """Abstract base for the true iterative MAVE estimator.

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
        """
        Parameters
        ----------
        d : int
            Target reduced dimension (number of sufficient directions).
        max_iters : int
            Maximum number of alternating minimisation iterations.
        tol : float
            Convergence tolerance on the relative change in MAVE loss.
        ridge_lambda : float
            Tikhonov regularisation coefficient for the WLS and B-update solves.
        init_method : str
            Initialisation strategy for B.  Only "pca" is supported.
        random_state : int
            Seed for any internal random operations (not used by PCA init).
        """
        self.d = d
        self.max_iters = max_iters
        self.tol = tol
        self.ridge_lambda = ridge_lambda
        self.init_method = init_method
        self.random_state = random_state

        self.B_: np.ndarray | None = None
        self.result_: MAVEResult | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

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
        """Initialise B via PCA (top-d right singular vectors of centred X)."""
        n, p = X.shape
        if self.d > p:
            raise ValueError(f"d={self.d} exceeds p={p}")
        X_centred = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centred, full_matrices=False)
        B = Vt[: self.d].T  # (p, d)
        return orthonormalize(B)

    # ------------------------------------------------------------------
    # Step 1: local WLS in reduced space
    # ------------------------------------------------------------------

    def _local_wls_reduced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve per-anchor WLS in R^d using current projection B.

        For each anchor j, form the (1+d)-dimensional design matrix
            Z[i,:] = [1,  (B^T (X_i - X_j))^T ]    shape (n_active, 1+d)
        and solve the weighted least squares problem.

        Returns
        -------
        alphas : (n,)   local intercepts a_j
        betas  : (n, d) local slope estimates b_j in R^d
        """
        n, p = X.shape
        d = self.d
        alphas = np.zeros(n)
        betas = np.zeros((n, d))
        dim = 1 + d

        for j in range(n):
            w_col = W[:, j]
            active = np.where(w_col > 0)[0]
            if len(active) < 2:
                continue

            w_active = w_col[active]
            delta = X[active] - X[j]         # (n_active, p)
            Z_red = delta @ B                 # (n_active, d) — reduced coordinates
            ones = np.ones((len(active), 1))
            design = np.hstack([ones, Z_red]) # (n_active, 1+d)
            y_active = y[active]

            Dw = design * w_active[:, None]
            gram = Dw.T @ design              # (1+d, 1+d)

            trace_gram = max(np.trace(gram), 1e-12)
            lambda_eff = self.ridge_lambda * trace_gram / dim
            ridge = lambda_eff * np.eye(dim)
            ridge[0, 0] = 0.0                # don't penalise intercept

            A = gram + ridge
            rhs = Dw.T @ y_active

            try:
                c = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, rhs, rcond=None)[0]

            alphas[j] = c[0]
            betas[j] = c[1:]

        return alphas, betas

    # ------------------------------------------------------------------
    # Step 2: B-update via normal-equation solve + orthonormalisation
    # ------------------------------------------------------------------

    def _update_B(
        self,
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray,
        alphas: np.ndarray,
        betas: np.ndarray,
        B_old: np.ndarray,
    ) -> np.ndarray:
        """Update B by solving the MAVE normal equation then orthonormalising.

        With {a_j, b_j} fixed, setting dL/dB = 0 yields

            vec(C) = H vec(B)

        where H is (pd x pd) with block (l, m) equal to
        sum_j b_j[l] b_j[m] Q_j,  and Q_j = sum_i w_ij delta_ij delta_ij^T.

        We solve this regularised linear system and orthonormalise the result.
        """
        n, p = X.shape
        d = self.d

        C = np.zeros((p, d))
        H_full = np.zeros((p * d, p * d))

        for j in range(n):
            w_col = W[:, j]
            active = np.where(w_col > 0)[0]
            if len(active) == 0:
                continue

            w_active = w_col[active]
            delta = X[active] - X[j]          # (n_active, p)
            r = y[active] - alphas[j]          # (n_active,)
            b_j = betas[j]                     # (d,)

            # Q_j = delta^T diag(w_active) delta   (p, p)
            w_delta = delta * w_active[:, None]
            Q_j = w_delta.T @ delta

            # f_j = delta^T (w_active * r)          (p,)
            f_j = delta.T @ (w_active * r)

            # C += f_j b_j^T                        (p, d)
            C += np.outer(f_j, b_j)

            # H_full += kron(b_j b_j^T, Q_j)        (pd, pd)
            H_full += np.kron(np.outer(b_j, b_j), Q_j)

        # Adaptive Tikhonov regularisation
        trace_H = max(np.trace(H_full), 1e-12)
        reg = self.ridge_lambda * trace_H / (p * d)
        H_reg = H_full + reg * np.eye(p * d)

        C_vec = C.ravel(order="F")  # column-major vectorisation

        try:
            B_vec = np.linalg.solve(H_reg, C_vec)
        except np.linalg.LinAlgError:
            logger.warning("B-update system solve failed; using lstsq fallback")
            B_vec = np.linalg.lstsq(H_reg, C_vec, rcond=None)[0]

        B_new = B_vec.reshape((p, d), order="F")

        if np.linalg.norm(B_new) < 1e-10:
            logger.warning("B-update produced near-zero matrix; keeping B_old")
            return orthonormalize(B_old)

        return orthonormalize(B_new)

    # ------------------------------------------------------------------
    # Objective computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_objective(
        X: np.ndarray,
        y: np.ndarray,
        W: np.ndarray,
        B: np.ndarray,
        alphas: np.ndarray,
        betas: np.ndarray,
    ) -> float:
        """Evaluate MAVE objective with reduced-space coefficients.

        L = sum_j sum_i w_ij (Y_i - a_j - b_j^T B^T (X_i - X_j))^2
        """
        n = X.shape[0]
        loss = 0.0
        for j in range(n):
            w_col = W[:, j]
            active = np.where(w_col > 0)[0]
            if len(active) == 0:
                continue
            delta = X[active] - X[j]
            Z_red = delta @ B                  # (n_active, d)
            residuals = y[active] - alphas[j] - Z_red @ betas[j]
            loss += np.dot(w_col[active], residuals ** 2)
        return float(loss)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> MAVEBase:
        """Estimate the projection matrix B via alternating minimisation.

        Parameters
        ----------
        X : (n, p) feature matrix (should be standardised).
        y : (n,) response vector.
        """
        n, p = check_dimensions(X, y, context=self.method_name)
        if self.d > p:
            raise ValueError(f"d={self.d} > p={p}")

        B = self._initialize_B(X)
        loss_history: list[float] = []
        delta_PB_history: list[float] = []
        rel_change_history: list[float] = []
        prev_loss = np.inf
        W = np.empty(0)

        for iteration in range(self.max_iters):
            W = self._compute_weights(X, y, B)

            # Step 1: local WLS in reduced space
            alphas, betas = self._local_wls_reduced(X, y, W, B)

            # Compute objective at (B, {a_j, b_j})
            loss = self._compute_objective(X, y, W, B, alphas, betas)
            loss_history.append(loss)

            rel_change = (
                abs(prev_loss - loss) / max(abs(prev_loss), 1e-12)
                if iteration > 0
                else float("inf")
            )
            rel_change_history.append(rel_change)

            logger.debug(
                "%s iter %d: loss=%.6e  rel_change=%.2e",
                self.method_name, iteration, loss, rel_change,
            )

            # Step 2: update B with {a_j, b_j} fixed
            B_new = self._update_B(X, y, W, alphas, betas, B)
            delta_PB = float(np.linalg.norm(B_new @ B_new.T - B @ B.T, "fro"))
            delta_PB_history.append(delta_PB)

            if iteration > 0 and rel_change < self.tol:
                logger.info(
                    "%s converged at iter %d (rel_change=%.2e, delta_PB=%.2e)",
                    self.method_name, iteration, rel_change, delta_PB,
                )
                B = B_new
                break

            prev_loss = loss
            B = B_new

        # Convergence diagnostics
        hit_max = len(loss_history) >= self.max_iters
        if hit_max and delta_PB_history and delta_PB_history[-1] > 1e-2:
            logger.warning(
                "%s: delta_PB=%.4f at final iteration (max_iters=%d) "
                "— may not have converged",
                self.method_name, delta_PB_history[-1], self.max_iters,
            )
        if len(delta_PB_history) > 5 and max(delta_PB_history[3:]) < 1e-10:
            logger.warning(
                "%s: delta_PB near zero since iter 3 — possible stagnation",
                self.method_name,
            )

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
                "delta_PB": delta_PB_history,
            },
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the estimated subspace: Z = X @ B."""
        if self.B_ is None:
            raise RuntimeError("Call .fit() before .transform()")
        return X @ self.B_


# ======================================================================
# Classical MAVE (Gaussian kernel weights)
# ======================================================================


class ClassicalMAVE(MAVEBase):
    """True iterative MAVE with Gaussian kernel weights.

    Parameters
    ----------
    d : int
        Target reduced dimension.
    bandwidth : float or "auto"
        Gaussian kernel bandwidth.  "auto" uses the median heuristic.
    k : int or None
        If set, truncate to top-k neighbours per anchor after computing
        Gaussian weights.  Weights are re-normalised after truncation.
        When None, all pairwise weights are used (pure Gaussian MAVE).
        A practical rule is k = floor(0.20 × n).
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
        return "ClassicalMAVE_kNN" if self.k is not None else "ClassicalMAVE"

    def _compute_weights(
        self, X: np.ndarray, y: np.ndarray, B: np.ndarray
    ) -> np.ndarray:
        """Gaussian kernel weights in the original feature space.

        These do not depend on B, but the algorithm still iterates
        because the reduced-space local regressions depend on B.
        """
        return gaussian_weights(X, bandwidth=self.bandwidth, k=self.k)
