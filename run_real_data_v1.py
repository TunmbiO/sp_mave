#!/usr/bin/env python3
"""Real data experiment v1.

Validates the SP-MAVE regime characterisation (p/n ≲ 0.5) on three real datasets.
Two sample-size regimes per dataset:
  - subsampled: n ≈ 300  (p/n in SP-MAVE's winning range)
  - full:       n = 2000  (large-n reference)

True subspace is unknown, so evaluation is prediction-only: 6 MSE values
(train/test × linear/kernel_ridge/random_forest) via sp_mave/evaluate.py.

Usage
-----
    python run_real_data_v1.py              # full run (~3–9 h)
    python run_real_data_v1.py --smoke-test # superconductor, 2 seeds, subsampled only

Outputs
-------
    results/real_data/real_data_results.csv
    results/real_data/real_data_summary.csv
    results/real_data/real_data_wilcoxon.csv
    results/real_data/failures.log
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.evaluate import evaluate_subspace
from sp_mave.mave import ClassicalMAVE
from sp_mave.preprocessing import standardize
from sp_mave.spmave import SPMAVE
from sp_mave.utils import get_logger

# ---------------------------------------------------------------------------
# Hyperparameters — edit here to vary the SDR dimension or k schedule
# ---------------------------------------------------------------------------

R_DIM: int = 3          # SDR reduced dimension; change to try d=2, d=5, etc.
K_FRACTION: float = 0.20  # k = floor(K_FRACTION × n_train)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

SEEDS: list[int] = list(range(20))

# Dataset registry.
# target: column name of the response variable in the raw data.
# For superconductor the OpenML column name is "class" (= critical temperature).
# For year_prediction_msd the target is the first column (no header).
DATASETS: dict[str, dict[str, Any]] = {
    "ames_housing": {
        "openml_id": 42165,
        "target":    "SalePrice",
        "n_sub":     300,
        "n_full":    2000,
    },
    "superconductor": {
        "openml_id": 43174,
        "target":    "critical_temp",
        "n_sub":     300,
        "n_full":    2000,
    },
    "year_prediction_msd": {
        "openml_id": None,      # not on OpenML; use UCI direct download
        "uci_url":   (
            "https://archive.ics.uci.edu/ml/machine-learning-databases"
            "/00203/YearPredictionMSD.txt.zip"
        ),
        "target":    None,      # first column is year
        "n_sub":     330,
        "n_full":    2000,
    },
}

# SP-MAVE fixed hyperparameters
SPMAVE_TAU: float = 1.0
SPMAVE_RF_PARAMS: dict[str, Any] = {
    "n_estimators":   200,
    "max_depth":      None,
    "min_samples_leaf": 5,
    "max_features":   "sqrt",
}
SPMAVE_CROSSFIT_CFG: dict[str, Any] = {
    "n_splits":    5,
    "shuffle":     True,
    "random_state": 42,
}

# Shared MAVE model settings (random_state overridden per seed)
MAVE_MODEL_CFG: dict[str, Any] = {
    "bandwidth":    "auto",
    "max_iters":    300,
    "tol":          1e-6,
    "ridge_lambda": 0.01,
    "init_method":  "pca",
}

DATA_DIR    = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "real_data"
FAILURES_LOG = RESULTS_DIR / "failures.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("real_data_v1")
logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)
logging.getLogger("sp_mave.spmave").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Dataset loading and preprocessing
# ---------------------------------------------------------------------------

def _fetch_openml(data_id: int) -> pd.DataFrame:
    """Fetch dataset from OpenML as a single DataFrame (features + target)."""
    from sklearn.datasets import fetch_openml
    try:
        bunch = fetch_openml(
            data_id=data_id, as_frame=True, parser="auto",
            target_column=None,
        )
    except TypeError:
        # Older sklearn without parser argument
        bunch = fetch_openml(data_id=data_id, as_frame=True, target_column=None)
    return bunch.data


def _fetch_year_msd(url: str) -> pd.DataFrame:
    """Download Year Prediction MSD from UCI and return as DataFrame."""
    logger.info("Downloading Year Prediction MSD from UCI (~400 MB compressed) ...")
    with urllib.request.urlopen(url, timeout=300) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f, header=None)
    # First column is year (target), rest are features
    n_cols = df.shape[1]
    df.columns = ["year"] + [f"f{i}" for i in range(1, n_cols)]
    logger.info("Year Prediction MSD loaded: %d rows × %d cols", *df.shape)
    return df


def load_and_preprocess(name: str, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """Load, preprocess, and cache a dataset.

    Preprocessing (applied to full dataset before any subsampling):
      1. Drop columns with > 20% missing values.
      2. Impute: numeric columns → column median; categorical → column mode.
      3. One-hot encode all remaining categorical/object columns (drop_first=True).
      4. Standardise response y to zero mean / unit variance on the full dataset
         is NOT done here — y standardisation is fit per train split.

    Results cached to data/{name}_preprocessed.npz to avoid re-downloading.
    """
    cache_path = DATA_DIR / f"{name}_preprocessed.npz"

    if cache_path.exists():
        logger.info("Loading preprocessed cache: %s", cache_path)
        d = np.load(cache_path, allow_pickle=True)
        X = d["X"].astype(np.float64)
        y = d["y"].astype(np.float64)
        p = int(X.shape[1])
        logger.info("  %s: n=%d  p=%d", name, len(y), p)
        return X, y, p

    logger.info("Preprocessing %s (first run — will cache to %s) ...", name, cache_path)

    # ---- Fetch raw data ----
    if cfg["openml_id"] is not None:
        df = _fetch_openml(cfg["openml_id"])
    else:
        df = _fetch_year_msd(cfg["uci_url"])

    # ---- Extract target ----
    target_col = cfg["target"]
    if target_col is None:
        target_col = df.columns[0]

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset '{name}'. "
            f"Available columns: {list(df.columns)}"
        )

    y_raw = df[target_col].astype(float).values
    X_df  = df.drop(columns=[target_col]).copy()

    logger.info("  Raw shape: %d rows × %d feature cols", len(y_raw), X_df.shape[1])

    # ---- Step 1: Drop columns with > 20% missing ----
    miss_frac = X_df.isna().mean()
    drop_cols = miss_frac[miss_frac > 0.20].index.tolist()
    if drop_cols:
        logger.info("  Dropping %d columns with >20%% missing: %s", len(drop_cols), drop_cols)
    X_df = X_df.drop(columns=drop_cols)

    # ---- Step 2: Impute ----
    num_cols = X_df.select_dtypes(include=[np.number]).columns
    cat_cols = X_df.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].median())
    for col in cat_cols:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].mode().iloc[0])

    # ---- Step 3: One-hot encode categorical columns ----
    if len(cat_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=list(cat_cols), drop_first=True)
        logger.info("  After one-hot encoding: %d feature cols", X_df.shape[1])

    # ---- Finalise ----
    X = X_df.values.astype(np.float64)
    p = X.shape[1]

    # Drop any remaining NaN/Inf rows (shouldn't happen, but defensive)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y_raw)
    if not valid.all():
        n_drop = (~valid).sum()
        logger.warning("  Dropping %d rows with non-finite values", n_drop)
        X, y_raw = X[valid], y_raw[valid]

    logger.info("  Final: n=%d  p=%d", len(y_raw), p)

    # For very large datasets, cap pool at 50 000 rows to keep cache manageable.
    # 50 000 >> 20 seeds × 2000 rows, so no practical loss of diversity.
    MAX_POOL = 50_000
    if len(y_raw) > MAX_POOL:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_raw), size=MAX_POOL, replace=False)
        X, y_raw = X[idx], y_raw[idx]
        logger.info("  Pool capped to %d rows for cache size", MAX_POOL)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y_raw)
    logger.info("  Cached to %s", cache_path)

    return X, y_raw, p


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_one_cell(
    name: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    p: int,
    n: int,
    n_regime: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Fit all three methods for one (dataset, n, seed) cell.

    Each seed draws a fresh subsample of n rows, then splits 80/20.
    X and y are standardised: scalers fit on train only.
    k = floor(K_FRACTION × n_train), same for ClassicalMAVE_kNN and SPMAVE.
    """
    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(len(X_full), size=n, replace=False)
    X_sub, y_sub = X_full[sub_idx], y_full[sub_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=seed
    )
    n_train = len(X_tr)

    # Standardise X (fit on train only)
    X_tr, X_te, _ = standardize(X_tr, X_te)

    # Standardise y (fit on train only)
    y_scaler = StandardScaler()
    y_tr = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

    k = max(1, math.floor(K_FRACTION * n_train))

    model_cfg = {**MAVE_MODEL_CFG, "random_state": seed}
    rows: list[dict[str, Any]] = []

    # ---- Method 1: ClassicalMAVE (full Gaussian, no k) --------------------
    try:
        t0 = time.time()
        est = ClassicalMAVE(
            d=R_DIM,
            bandwidth=model_cfg["bandwidth"],
            k=None,
            max_iters=model_cfg["max_iters"],
            tol=model_cfg["tol"],
            ridge_lambda=model_cfg["ridge_lambda"],
            init_method=model_cfg["init_method"],
            random_state=seed,
        )
        est.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        rows.append(_build_row(
            name=name, n=n, n_regime=n_regime, p=p, seed=seed,
            method="ClassicalMAVE", k_used=float("nan"),
            est=est, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=fit_time,
        ))
    except Exception:
        _log_failure("ClassicalMAVE", name, n_regime, seed)

    # ---- Method 2: ClassicalMAVE_kNN (top-k Gaussian) ---------------------
    try:
        t0 = time.time()
        est = ClassicalMAVE(
            d=R_DIM,
            bandwidth=model_cfg["bandwidth"],
            k=k,
            max_iters=model_cfg["max_iters"],
            tol=model_cfg["tol"],
            ridge_lambda=model_cfg["ridge_lambda"],
            init_method=model_cfg["init_method"],
            random_state=seed,
        )
        est.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        rows.append(_build_row(
            name=name, n=n, n_regime=n_regime, p=p, seed=seed,
            method="ClassicalMAVE_kNN", k_used=k,
            est=est, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=fit_time,
        ))
    except Exception:
        _log_failure("ClassicalMAVE_kNN", name, n_regime, seed)

    # ---- Method 3: SPMAVE -------------------------------------------------
    try:
        crossfit_cfg = {**SPMAVE_CROSSFIT_CFG, "random_state": seed}
        rf_params    = {**SPMAVE_RF_PARAMS,    "random_state": seed}
        t0 = time.time()
        est = SPMAVE(
            d=R_DIM,
            k=k,
            tau=SPMAVE_TAU,
            rf_params=rf_params,
            crossfit_cfg=crossfit_cfg,
            max_iters=model_cfg["max_iters"],
            tol=model_cfg["tol"],
            ridge_lambda=model_cfg["ridge_lambda"],
            init_method=model_cfg["init_method"],
            random_state=seed,
        )
        est.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        rows.append(_build_row(
            name=name, n=n, n_regime=n_regime, p=p, seed=seed,
            method="SPMAVE", k_used=k,
            est=est, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=fit_time,
        ))
    except Exception:
        _log_failure("SPMAVE", name, n_regime, seed)

    return rows


def _build_row(
    name: str,
    n: int,
    n_regime: str,
    p: int,
    seed: int,
    method: str,
    k_used: float,
    est: Any,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    fit_time: float,
) -> dict[str, Any]:
    result = est.result_
    assert result is not None
    mse = evaluate_subspace(result.B, X_tr, X_te, y_tr, y_te)
    return {
        "dataset":    name,
        "n_obs":      n,
        "n_regime":   n_regime,
        "p_features": p,
        "p_n_ratio":  round(p / n, 4),
        "seed":       seed,
        "method":     method,
        "k_used":     k_used,
        "r_dim":      R_DIM,
        **mse,
        "fit_time_s": round(fit_time, 4),
        "n_iters":    result.n_iters,
        "final_loss": result.loss_history[-1] if result.loss_history else float("nan"),
    }


def _log_failure(method: str, name: str, n_regime: str, seed: int) -> None:
    msg = (
        f"FAILURE  method={method}  dataset={name}  n_regime={n_regime}  seed={seed}\n"
        + traceback.format_exc() + "\n"
    )
    logger.error(
        "Cell failed: method=%s  dataset=%s  n_regime=%s  seed=%d",
        method, name, n_regime, seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FAILURES_LOG, "a") as fh:
        fh.write(msg)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    "test_mse_linear", "train_mse_linear",
    "test_mse_kernel_ridge", "train_mse_kernel_ridge",
    "test_mse_random_forest", "train_mse_random_forest",
    "fit_time_s", "n_iters", "final_loss",
]


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and standard error per (dataset, n_regime, method)."""
    datasets  = sorted({r["dataset"]  for r in raw_rows})
    regimes   = sorted({r["n_regime"] for r in raw_rows})
    methods   = sorted({r["method"]   for r in raw_rows})
    agg: list[dict[str, Any]] = []

    for ds in datasets:
        for regime in regimes:
            for m in methods:
                subset = [
                    r for r in raw_rows
                    if r["dataset"] == ds and r["n_regime"] == regime and r["method"] == m
                ]
                if not subset:
                    continue
                entry: dict[str, Any] = {
                    "dataset": ds, "n_regime": regime, "method": m,
                    "n_seeds": len(subset),
                    "n_obs":   subset[0]["n_obs"],
                    "p_features": subset[0]["p_features"],
                    "p_n_ratio":  subset[0]["p_n_ratio"],
                }
                for col in _METRIC_COLS:
                    vals = [float(r[col]) for r in subset if col in r and not _is_nan(r[col])]
                    if vals:
                        mean_v = float(np.mean(vals))
                        se_v   = (
                            float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                            if len(vals) > 1 else float("nan")
                        )
                        entry[f"{col}_mean"] = mean_v
                        entry[f"{col}_se"]   = se_v
                agg.append(entry)

    return agg


# ---------------------------------------------------------------------------
# Wilcoxon tests
# ---------------------------------------------------------------------------

def run_wilcoxon(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired Wilcoxon signed-rank: SPMAVE vs each baseline per (dataset, n_regime).

    Tests test_mse_kernel_ridge only (primary metric for real data).
    rank_biserial > 0 means SPMAVE tends to have smaller values than the baseline.
    """
    from scipy.stats import wilcoxon as scipy_wilcoxon

    datasets  = sorted({r["dataset"]  for r in raw_rows})
    regimes   = sorted({r["n_regime"] for r in raw_rows})
    baselines = ["ClassicalMAVE", "ClassicalMAVE_kNN"]
    metric    = "test_mse_kernel_ridge"
    results: list[dict[str, Any]] = []

    for ds in datasets:
        for regime in regimes:
            sp_rows = {
                r["seed"]: r for r in raw_rows
                if r["method"] == "SPMAVE" and r["dataset"] == ds and r["n_regime"] == regime
            }
            for bl in baselines:
                bl_rows = {
                    r["seed"]: r for r in raw_rows
                    if r["method"] == bl and r["dataset"] == ds and r["n_regime"] == regime
                }
                common = sorted(sp_rows.keys() & bl_rows.keys())
                if len(common) < 10:
                    logger.warning(
                        "Skipping Wilcoxon ds=%s regime=%s bl=%s: only %d paired seeds",
                        ds, regime, bl, len(common),
                    )
                    continue
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
                if n_nz > 0 and not _is_nan(stat):
                    r_rb = 1.0 - 2.0 * stat / (n_nz * (n_nz + 1) / 2.0)
                else:
                    r_rb = float("nan")
                results.append({
                    "dataset":           ds,
                    "n_regime":          regime,
                    "baseline":          bl,
                    "metric":            metric,
                    "n_pairs":           len(common),
                    "spmave_mean":       float(np.mean(sp_vals)),
                    "baseline_mean":     float(np.mean(bl_vals)),
                    "wilcoxon_statistic": stat,
                    "p_value":           pval,
                    "rank_biserial":     r_rb,
                })

    return results


# ---------------------------------------------------------------------------
# CSV helpers
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
# Print helpers
# ---------------------------------------------------------------------------

def _print_summary(
    summary: list[dict[str, Any]],
    datasets: list[str],
    regimes: list[str],
) -> None:
    methods = sorted({r["method"] for r in summary})
    print("\n" + "=" * 110)
    print("REAL DATA v1 — SUMMARY  (mean ± SE over seeds)")
    print("=" * 110)
    for ds in datasets:
        for regime in regimes:
            ref = next(
                (r for r in summary if r["dataset"] == ds and r["n_regime"] == regime),
                None,
            )
            if ref is None:
                continue
            n_obs = ref.get("n_obs", "?")
            p     = ref.get("p_features", "?")
            pn    = ref.get("p_n_ratio", "?")
            print(f"\n  {ds}  |  {regime}  (n={n_obs}  p={p}  p/n={pn})")
            print(
                f"  {'Method':<22s}  {'KRR test MSE':>22s}  "
                f"{'Linear test MSE':>22s}  {'RF test MSE':>22s}"
            )
            print("  " + "-" * 92)
            for m in methods:
                row = next(
                    (r for r in summary
                     if r["method"] == m and r["dataset"] == ds and r["n_regime"] == regime),
                    None,
                )
                if row is None:
                    continue

                def _fmt(key: str) -> str:
                    mu = row.get(f"{key}_mean")
                    se = row.get(f"{key}_se")
                    if mu is None:
                        return "—".rjust(22)
                    if se is None or _is_nan(se):
                        return f"{mu:.4f}".rjust(22)
                    return f"{mu:.4f}±{se:.4f}".rjust(22)

                print(
                    f"  {m:<22s}  {_fmt('test_mse_kernel_ridge')}  "
                    f"{_fmt('test_mse_linear')}  {_fmt('test_mse_random_forest')}"
                )
    print("\n" + "=" * 110 + "\n")


def _print_regime_table(
    summary: list[dict[str, Any]],
    datasets: list[str],
    regimes: list[str],
) -> None:
    print("Regime table — SP-MAVE KRR MSE < kNN-MAVE KRR MSE?")
    print(
        f"  {'Dataset':<25s}  {'n_regime':<12s}  {'p/n':>6}  "
        f"{'SPMAVE':>10}  {'kNN-MAVE':>10}  {'SP wins?':>9}"
    )
    print("  " + "-" * 80)
    for ds in datasets:
        for regime in regimes:
            sp_row = next(
                (r for r in summary
                 if r["method"] == "SPMAVE" and r["dataset"] == ds and r["n_regime"] == regime),
                None,
            )
            knn_row = next(
                (r for r in summary
                 if r["method"] == "ClassicalMAVE_kNN" and r["dataset"] == ds
                 and r["n_regime"] == regime),
                None,
            )
            if sp_row is None or knn_row is None:
                continue
            sp_v  = sp_row.get("test_mse_kernel_ridge_mean", float("nan"))
            knn_v = knn_row.get("test_mse_kernel_ridge_mean", float("nan"))
            wins  = "Y" if (not _is_nan(sp_v) and not _is_nan(knn_v) and sp_v < knn_v) else "N"
            pn    = sp_row.get("p_n_ratio", float("nan"))
            print(
                f"  {ds:<25s}  {regime:<12s}  {pn:>6.3f}  "
                f"{sp_v:>10.4f}  {knn_v:>10.4f}  {wins:>9}"
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    # Smoke test: superconductor only, subsampled regime, 2 seeds
    if smoke_test:
        dataset_names = ["superconductor"]
        seeds         = [0, 1]
        regimes_map   = {"superconductor": ["subsampled"]}
        logger.info("SMOKE TEST: dataset=superconductor  regime=subsampled  seeds=%s", seeds)
    else:
        dataset_names = list(DATASETS.keys())
        seeds         = SEEDS
        regimes_map   = {ds: ["subsampled", "full"] for ds in dataset_names}
        logger.info(
            "FULL RUN: %d datasets × 2 regimes × %d seeds × 3 methods",
            len(dataset_names), len(seeds),
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    open(FAILURES_LOG, "w").close()

    # Load and preprocess all datasets
    loaded: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for ds_name in dataset_names:
        cfg = DATASETS[ds_name]
        X, y, p = load_and_preprocess(ds_name, cfg)
        loaded[ds_name] = (X, y, p)
        logger.info("Dataset ready: %s  n_pool=%d  p=%d", ds_name, len(y), p)

    # Build cell list: (dataset_name, n, n_regime, seed)
    cells: list[tuple[str, int, str, int]] = []
    for ds_name in dataset_names:
        cfg = DATASETS[ds_name]
        _, y, _ = loaded[ds_name]
        n_pool   = len(y)
        for regime in regimes_map[ds_name]:
            if regime == "subsampled":
                n = cfg["n_sub"]
            else:
                n = min(cfg["n_full"], n_pool)
            if n > n_pool:
                logger.warning(
                    "%s: requested n=%d > pool=%d; capping to pool size", ds_name, n, n_pool
                )
                n = n_pool
            for seed in seeds:
                cells.append((ds_name, n, regime, seed))

    total_methods = len(cells) * 3
    if _have_tqdm:
        pbar = tqdm(total=total_methods, desc="real-data-v1", unit="fit")

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for ds_name, n, regime, seed in cells:
        X_full, y_full, p = loaded[ds_name]
        t_cell = time.time()
        cell_rows = run_one_cell(ds_name, X_full, y_full, p, n, regime, seed)
        elapsed   = time.time() - t_cell
        all_rows.extend(cell_rows)
        msg = (
            f"ds={ds_name}  regime={regime}  n={n}  seed={seed:2d}  "
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

    suffix = "_smoke" if smoke_test else ""
    raw_path      = RESULTS_DIR / f"real_data_results{suffix}.csv"
    summary_path  = RESULTS_DIR / f"real_data_summary{suffix}.csv"
    wilcoxon_path = RESULTS_DIR / f"real_data_wilcoxon{suffix}.csv"

    _write_csv(all_rows, raw_path)

    summary = aggregate_summary(all_rows)
    _write_csv(summary, summary_path)

    try:
        wilcoxon_rows = run_wilcoxon(all_rows)
        _write_csv(wilcoxon_rows, wilcoxon_path)
    except ImportError:
        logger.warning("scipy not available — skipping Wilcoxon tests")

    all_datasets = sorted({r["dataset"]  for r in all_rows})
    all_regimes  = sorted({r["n_regime"] for r in all_rows})
    _print_summary(summary, all_datasets, all_regimes)
    _print_regime_table(summary, all_datasets, all_regimes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real data experiment v1 — SP-MAVE vs classical MAVE on real datasets."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: superconductor only, subsampled regime, 2 seeds.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
