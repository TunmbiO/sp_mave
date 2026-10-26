#!/usr/bin/env python3
"""LendingClub loan default experiment.

Applies SP-MAVE to binary classification via local linear probability
approximation: the 0/1 default label is fed directly into the MAVE
local WLS, which approximates P(default=1|X) in a low-dimensional
sufficient subspace.  The SP-MAVE estimator is unchanged from Chapter 1.

Downstream evaluation uses KernelRidge (RBF) fitted on B^T X_train:
  - Brier score : MSE of KRR outputs vs raw 0/1 labels (lower → better)
  - AUC-ROC     : raw KRR outputs used as ranking scores  (higher → better)

Note on y standardisation
--------------------------
y is NOT standardised before MAVE fitting (unlike Chapter 1 continuous
targets).  The 0/1 binary response is on a natural [0, 1] scale and the
local WLS approximates conditional probabilities directly.

Sample design
-------------
  subsampled : n=300,  stratified draw from full pool,  p/n=0.227
  full       : n=2000, stratified draw from full pool,  p/n=0.034
Both the pool-draw and the 80/20 train/test split are stratified on the
`default` column to preserve the 78%/22% class ratio.

Usage
-----
    python run_lendingclub_experiment.py --smoke-test  # seed=0, n=300 only
    python run_lendingclub_experiment.py               # 20 seeds × 2 regimes

Outputs
-------
    results/lendingclub_results.csv
    results/lendingclub_summary.csv
    results/lendingclub_wilcoxon.csv
    results/lendingclub_failures.log

Wilcoxon convention
-------------------
diffs = sp_vals - baseline_vals.
  Brier  (lower is better): rank_biserial > 0 means SP-MAVE tends lower → SP wins.
  AUC    (higher is better): rank_biserial < 0 means SP-MAVE tends higher → SP wins.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.mave import ClassicalMAVE
from sp_mave.preprocessing import standardize
from sp_mave.spmave import SPMAVE
from sp_mave.utils import get_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH    = Path("/home/tunmbi/data/lendingclub/lendingclub_clean.csv")
RESULTS_DIR  = Path(__file__).resolve().parent / "results"
OUT_RESULTS  = RESULTS_DIR / "lendingclub_results.csv"
OUT_SUMMARY  = RESULTS_DIR / "lendingclub_summary.csv"
OUT_WILCOXON = RESULTS_DIR / "lendingclub_wilcoxon.csv"
FAILURES_LOG = RESULTS_DIR / "lendingclub_failures.log"

# ---------------------------------------------------------------------------
# Hyperparameters — identical to Chapter 1
# ---------------------------------------------------------------------------

R_DIM: int = 3
K_FRACTION: float = 0.20
SEEDS: list[int] = list(range(20))

SPMAVE_TAU: float = 1.0
SPMAVE_RF_PARAMS: dict[str, Any] = {
    "n_estimators":     200,
    "max_depth":        None,
    "min_samples_leaf": 5,
    "max_features":     "sqrt",
}
SPMAVE_CROSSFIT_CFG: dict[str, Any] = {
    "n_splits":     5,
    "shuffle":      True,
    "random_state": 42,
}
MAVE_MODEL_CFG: dict[str, Any] = {
    "bandwidth":    "auto",
    "max_iters":    300,
    "tol":          1e-6,
    "ridge_lambda": 0.01,
    "init_method":  "pca",
}

REGIMES: dict[str, int] = {
    "subsampled": 300,
    "full":       2000,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("lendingclub")
logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)
logging.getLogger("sp_mave.spmave").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[np.ndarray, np.ndarray, int]:
    """Load and return (X, y, p).

    X : (n_pool, p) float64 feature matrix, not yet standardised.
    y : (n_pool,)  int   0/1 default indicator.
    p : number of feature columns.
    """
    logger.info("Loading %s ...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    y  = df["default"].values.astype(np.float64)
    X  = df.drop(columns=["default"]).values.astype(np.float64)
    p  = X.shape[1]
    logger.info("Pool: n=%d  p=%d  default_rate=%.3f", len(y), p, y.mean())
    return X, y, p


# ---------------------------------------------------------------------------
# Binary downstream evaluator
# ---------------------------------------------------------------------------

def evaluate_binary(
    B: np.ndarray,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, float]:
    """Evaluate subspace B on the binary classification task.

    Projects data onto B, fits KernelRidge (RBF) on (Z_train, y_train)
    where y is raw 0/1, then computes:
      - Brier score : MSE(y_pred, y_01)   [lower = better]
      - AUC-ROC     : using raw KRR scores [higher = better]
    Both train and test versions are returned for diagnostics.
    """
    Z_tr = X_tr @ B
    Z_te = X_te @ B

    krr = KernelRidge(kernel="rbf")
    krr.fit(Z_tr, y_tr)

    pred_tr = krr.predict(Z_tr)
    pred_te = krr.predict(Z_te)

    return {
        "brier_train": float(np.mean((pred_tr - y_tr) ** 2)),
        "brier_test":  float(np.mean((pred_te - y_te) ** 2)),
        "auc_train":   float(roc_auc_score(y_tr, pred_tr)),
        "auc_test":    float(roc_auc_score(y_te, pred_te)),
    }


# ---------------------------------------------------------------------------
# Shared split function — imported by run_k_sensitivity.py
# ---------------------------------------------------------------------------

def make_split(
    X_full: np.ndarray,
    y_full: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified subsample of n rows then 80/20 stratified split, X standardised.

    Canonical split logic for all LendingClub experiments.  Both the pool
    draw and the train/test split use stratify=y so the 78%/22% class ratio
    is preserved.  X is standardised with scaler fit on train only.
    y is returned as raw 0/1 (not standardised).

    Parameters
    ----------
    X_full, y_full : full pool arrays (not yet standardised).
    n              : subsample size drawn from the pool.
    seed           : controls both the pool draw and the 80/20 split.

    Returns
    -------
    X_tr, X_te, y_tr, y_te : standardised features, raw 0/1 labels.
    """
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    sub_idx, _ = next(sss.split(X_full, y_full))
    X_sub, y_sub = X_full[sub_idx], y_full[sub_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=seed
    )

    X_tr, X_te, _ = standardize(X_tr, X_te)
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_one_cell(
    X_full: np.ndarray,
    y_full: np.ndarray,
    p: int,
    n: int,
    n_regime: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Fit all three methods for one (n, seed) cell."""
    X_tr, X_te, y_tr, y_te = make_split(X_full, y_full, n, seed)
    n_train = len(X_tr)

    k = max(1, math.floor(K_FRACTION * n_train))
    model_cfg = {**MAVE_MODEL_CFG, "random_state": seed}
    rows: list[dict[str, Any]] = []

    # ---- ClassicalMAVE (full Gaussian, no k) --------------------------------
    try:
        t0  = time.time()
        est = ClassicalMAVE(
            d=R_DIM, bandwidth=model_cfg["bandwidth"], k=None,
            max_iters=model_cfg["max_iters"], tol=model_cfg["tol"],
            ridge_lambda=model_cfg["ridge_lambda"],
            init_method=model_cfg["init_method"], random_state=seed,
        )
        est.fit(X_tr, y_tr)
        rows.append(_build_row(
            n=n, n_regime=n_regime, p=p, seed=seed,
            method="ClassicalMAVE", k_used=float("nan"), est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("ClassicalMAVE", n_regime, seed)

    # ---- ClassicalMAVE_kNN --------------------------------------------------
    try:
        t0  = time.time()
        est = ClassicalMAVE(
            d=R_DIM, bandwidth=model_cfg["bandwidth"], k=k,
            max_iters=model_cfg["max_iters"], tol=model_cfg["tol"],
            ridge_lambda=model_cfg["ridge_lambda"],
            init_method=model_cfg["init_method"], random_state=seed,
        )
        est.fit(X_tr, y_tr)
        rows.append(_build_row(
            n=n, n_regime=n_regime, p=p, seed=seed,
            method="ClassicalMAVE_kNN", k_used=k, est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("ClassicalMAVE_kNN", n_regime, seed)

    # ---- SPMAVE -------------------------------------------------------------
    try:
        t0  = time.time()
        est = SPMAVE(
            d=R_DIM, k=k, tau=SPMAVE_TAU,
            rf_params={**SPMAVE_RF_PARAMS, "random_state": seed},
            crossfit_cfg={**SPMAVE_CROSSFIT_CFG, "random_state": seed},
            max_iters=model_cfg["max_iters"], tol=model_cfg["tol"],
            ridge_lambda=model_cfg["ridge_lambda"],
            init_method=model_cfg["init_method"], random_state=seed,
        )
        est.fit(X_tr, y_tr)
        rows.append(_build_row(
            n=n, n_regime=n_regime, p=p, seed=seed,
            method="SPMAVE", k_used=k, est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("SPMAVE", n_regime, seed)

    return rows


def _build_row(
    n: int, n_regime: str, p: int, seed: int,
    method: str, k_used: float, est: Any,
    X_tr: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_te: np.ndarray,
    fit_time: float,
) -> dict[str, Any]:
    result = est.result_
    metrics = evaluate_binary(result.B, X_tr, X_te, y_tr, y_te)
    return {
        "dataset":    "lendingclub",
        "n_obs":      n,
        "n_regime":   n_regime,
        "p_features": p,
        "p_n_ratio":  round(p / n, 4),
        "seed":       seed,
        "method":     method,
        "k_used":     k_used,
        "r_dim":      R_DIM,
        **metrics,       # brier_train, brier_test, auc_train, auc_test
        "fit_time_s": round(fit_time, 4),
        "n_iters":    result.n_iters,
        "final_loss": result.loss_history[-1] if result.loss_history else float("nan"),
    }


def _log_failure(method: str, n_regime: str, seed: int) -> None:
    msg = (
        f"FAILURE  method={method}  n_regime={n_regime}  seed={seed}\n"
        + traceback.format_exc() + "\n"
    )
    logger.error("Cell failed: method=%s  n_regime=%s  seed=%d", method, n_regime, seed)
    with open(FAILURES_LOG, "a") as fh:
        fh.write(msg)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_COLS = ["brier_train", "brier_test", "auc_train", "auc_test",
                "fit_time_s", "n_iters", "final_loss"]


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean ± SE per (n_regime, method) over seeds."""
    regimes = sorted({r["n_regime"] for r in raw_rows})
    methods = sorted({r["method"]   for r in raw_rows})
    agg: list[dict[str, Any]] = []

    for regime in regimes:
        for m in methods:
            subset = [
                r for r in raw_rows
                if r["n_regime"] == regime and r["method"] == m
            ]
            if not subset:
                continue
            entry: dict[str, Any] = {
                "dataset":    "lendingclub",
                "n_regime":   regime,
                "method":     m,
                "n_seeds":    len(subset),
                "n_obs":      subset[0]["n_obs"],
                "p_features": subset[0]["p_features"],
                "p_n_ratio":  subset[0]["p_n_ratio"],
            }
            for col in _METRIC_COLS:
                vals = [
                    float(r[col]) for r in subset
                    if col in r and not _is_nan(r[col])
                ]
                if vals:
                    entry[f"{col}_mean"] = float(np.mean(vals))
                    entry[f"{col}_se"]   = (
                        float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                        if len(vals) > 1 else float("nan")
                    )
            agg.append(entry)
    return agg


# ---------------------------------------------------------------------------
# Wilcoxon
# ---------------------------------------------------------------------------

def run_wilcoxon(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired Wilcoxon: SPMAVE vs each baseline per n_regime, on Brier + AUC.

    diffs = sp_vals - baseline_vals.
    Brier (lower better): rank_biserial > 0 → SP-MAVE tends lower → SP wins.
    AUC   (higher better): rank_biserial < 0 → SP-MAVE tends higher → SP wins.
    """
    from scipy.stats import wilcoxon as scipy_wilcoxon

    regimes   = sorted({r["n_regime"] for r in raw_rows})
    baselines = ["ClassicalMAVE", "ClassicalMAVE_kNN"]
    metrics   = ["brier_test", "auc_test"]
    results: list[dict[str, Any]] = []

    for regime in regimes:
        sp_rows = {
            r["seed"]: r for r in raw_rows
            if r["method"] == "SPMAVE" and r["n_regime"] == regime
        }
        for bl in baselines:
            bl_rows = {
                r["seed"]: r for r in raw_rows
                if r["method"] == bl and r["n_regime"] == regime
            }
            common = sorted(sp_rows.keys() & bl_rows.keys())
            if len(common) < 10:
                logger.warning(
                    "Skipping Wilcoxon regime=%s bl=%s: only %d paired seeds",
                    regime, bl, len(common),
                )
                continue

            for metric in metrics:
                sp_vals = np.array([sp_rows[s][metric] for s in common], dtype=float)
                bl_vals = np.array([bl_rows[s][metric] for s in common], dtype=float)
                diffs   = sp_vals - bl_vals

                if np.all(diffs == 0):
                    stat, pval = float("nan"), float("nan")
                else:
                    try:
                        stat, pval = scipy_wilcoxon(diffs, alternative="two-sided")
                        stat, pval = float(stat), float(pval)
                    except Exception:
                        stat, pval = float("nan"), float("nan")

                n_nz = int(np.sum(np.abs(diffs) > 0))
                r_rb = (
                    1.0 - 2.0 * stat / (n_nz * (n_nz + 1) / 2.0)
                    if n_nz > 0 and not _is_nan(stat) else float("nan")
                )
                results.append({
                    "n_regime":           regime,
                    "baseline":           bl,
                    "metric":             metric,
                    "n_pairs":            len(common),
                    "spmave_mean":        float(np.mean(sp_vals)),
                    "baseline_mean":      float(np.mean(bl_vals)),
                    "wilcoxon_statistic": stat,
                    "p_value":            pval,
                    "rank_biserial":      r_rb,
                })
    return results


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        logger.warning("No rows to write to %s", path)
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(summary: list[dict[str, Any]]) -> None:
    regimes = sorted({r["n_regime"] for r in summary})
    methods = sorted({r["method"]   for r in summary})

    print("\n" + "=" * 90)
    print("LENDINGCLUB — SUMMARY  (mean ± SE over seeds)")
    print("=" * 90)
    for regime in regimes:
        ref = next((r for r in summary if r["n_regime"] == regime), None)
        if ref is None:
            continue
        print(f"\n  {regime}  (n={ref['n_obs']}  p={ref['p_features']}  p/n={ref['p_n_ratio']})")
        print(f"  {'Method':<22s}  {'Brier test':>20s}  {'AUC test':>20s}")
        print("  " + "-" * 66)
        for m in methods:
            row = next(
                (r for r in summary if r["method"] == m and r["n_regime"] == regime),
                None,
            )
            if row is None:
                continue

            def _fmt(key: str) -> str:
                mu = row.get(f"{key}_mean")
                se = row.get(f"{key}_se")
                if mu is None:
                    return "—".rjust(20)
                if se is None or _is_nan(se):
                    return f"{mu:.4f}".rjust(20)
                return f"{mu:.4f}±{se:.4f}".rjust(20)

            print(f"  {m:<22s}  {_fmt('brier_test')}  {_fmt('auc_test')}")

    print()
    print("  Note: Brier lower=better, AUC higher=better.")
    print("  Wilcoxon convention: rank_biserial > 0 → SP-MAVE tends lower (Brier: SP wins).")
    print("                       rank_biserial < 0 → SP-MAVE tends higher (AUC:   SP wins).")
    print("\n" + "=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    open(FAILURES_LOG, "w").close()

    X_full, y_full, p = load_data()

    if smoke_test:
        seeds        = [0]
        regime_names = ["subsampled"]
        logger.info("SMOKE TEST: seed=0, regime=subsampled")
    else:
        seeds        = SEEDS
        regime_names = list(REGIMES.keys())
        logger.info(
            "FULL RUN: %d seeds × %d regimes × 3 methods",
            len(seeds), len(regime_names),
        )

    # Build cell list
    cells: list[tuple[int, str, int]] = [
        (REGIMES[regime], regime, seed)
        for regime in regime_names
        for seed in seeds
    ]

    if _have_tqdm:
        pbar = tqdm(total=len(cells) * 3, desc="lendingclub", unit="fit")

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for n, n_regime, seed in cells:
        t_cell    = time.time()
        cell_rows = run_one_cell(X_full, y_full, p, n, n_regime, seed)
        elapsed   = time.time() - t_cell
        all_rows.extend(cell_rows)
        msg = (
            f"regime={n_regime}  n={n}  seed={seed:2d}  "
            f"{len(cell_rows)}/3 OK  ({elapsed:.1f}s)"
        )
        if _have_tqdm:
            pbar.update(3)
            pbar.set_postfix_str(msg)
        else:
            logger.info(msg)

    if _have_tqdm:
        pbar.close()

    total_elapsed = time.time() - t_start
    logger.info(
        "Completed %d rows in %.1fs (%.1f min)",
        len(all_rows), total_elapsed, total_elapsed / 60,
    )

    if not all_rows:
        logger.error("No rows produced.")
        return

    suffix = "_smoke" if smoke_test else ""
    _write_csv(all_rows,                    RESULTS_DIR / f"lendingclub_results{suffix}.csv")
    summary = aggregate_summary(all_rows)
    _write_csv(summary,                     RESULTS_DIR / f"lendingclub_summary{suffix}.csv")

    try:
        _write_csv(run_wilcoxon(all_rows),  RESULTS_DIR / f"lendingclub_wilcoxon{suffix}.csv")
    except ImportError:
        logger.warning("scipy not available — skipping Wilcoxon tests")

    _print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "SP-MAVE on LendingClub loan default (binary 0/1 via local "
            "linear probability approximation)."
        )
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="One seed, subsampled regime only. Does not write to canonical CSVs.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
