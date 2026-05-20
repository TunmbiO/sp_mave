#!/usr/bin/env python3
"""Real data experiment v2.

Adds two new datasets to the SP-MAVE real-data experiment and APPENDS
results to the existing v1 CSVs. Identical experimental pipeline to
run_real_data_v1.py in every respect (same methods, seeds, k schedule,
hyperparameters, evaluation).

New datasets
------------
  geo_music    — Geographical Original of Music (UCI, n=1059, p=68)
                 Target: latitude (column index 68). Longitude (col 69) dropped.
  online_news  — Online News Popularity (UCI, n=39644 capped to 10000, p=58)
                 Target: log1p(shares). Columns url and timedelta dropped.

Usage
-----
    python run_real_data_v2.py              # full run (~2–2.5 h)
    python run_real_data_v2.py --smoke-test # geo_music, 2 seeds, subsampled only

Outputs (appended, never overwritten)
--------------------------------------
    results/real_data/real_data_results.csv
    results/real_data/real_data_summary.csv
    results/real_data/real_data_wilcoxon.csv
    results/real_data/failures_v2.log

Smoke-test output (separate files, canonical CSVs untouched)
-------------------------------------------------------------
    results/real_data/real_data_results_v2_smoke.csv
    results/real_data/real_data_summary_v2_smoke.csv
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
# Hyperparameters — identical to v1
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

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "geo_music": {
        "zip_url":  (
            "https://archive.ics.uci.edu/static/public/315/"
            "geographical+original+of+music.zip"
        ),
        "zip_file": "Geographical Original of Music/default_features_1059_tracks.txt",
        "n_sub":    300,
        "n_full":   1059,
        "pool_cap": None,
    },
    "online_news": {
        "zip_url":  (
            "https://archive.ics.uci.edu/static/public/332/"
            "online+news+popularity.zip"
        ),
        "zip_file": "OnlineNewsPopularity/OnlineNewsPopularity.csv",
        "n_sub":    300,
        "n_full":   2000,
        "pool_cap": 10_000,
    },
}

DATA_DIR     = Path(__file__).resolve().parent / "data"
RESULTS_DIR  = Path(__file__).resolve().parent / "results" / "real_data"
FAILURES_LOG = RESULTS_DIR / "failures_v2.log"
RAW_CSV      = RESULTS_DIR / "real_data_results.csv"
SUMMARY_CSV  = RESULTS_DIR / "real_data_summary.csv"
WILCOXON_CSV = RESULTS_DIR / "real_data_wilcoxon.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("real_data_v2")
logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)
logging.getLogger("sp_mave.spmave").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _load_geo_music(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Download and parse Geographical Original of Music.

    File has no header, 70 columns (all float64):
      cols 0–67  audio features (p=68)
      col  68    latitude  ← target
      col  69    longitude ← dropped
    """
    logger.info("Downloading Geographical Original of Music (~600 KB) ...")
    with urllib.request.urlopen(cfg["zip_url"], timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(cfg["zip_file"]) as f:
            df = pd.read_csv(f, header=None)
    y = df.iloc[:, 68].values.astype(np.float64)   # latitude
    X = df.iloc[:, :68].values.astype(np.float64)  # audio features; col 69 never read
    logger.info("geo_music: n=%d  p=%d", len(y), X.shape[1])
    return X, y


def _load_online_news(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Download and parse Online News Popularity.

    Drops non-predictive columns url and timedelta.
    Response: log1p(shares).  All 58 remaining columns are numeric.
    """
    logger.info("Downloading Online News Popularity (~7 MB) ...")
    with urllib.request.urlopen(cfg["zip_url"], timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(cfg["zip_file"]) as f:
            df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    y = np.log1p(df["shares"].values).astype(np.float64)
    X = df.drop(columns=["url", "timedelta", "shares"]).values.astype(np.float64)
    logger.info("online_news: n=%d  p=%d", len(y), X.shape[1])
    return X, y


# ---------------------------------------------------------------------------
# Load, preprocess, cache — same standards as v1
# ---------------------------------------------------------------------------

def load_and_preprocess(
    name: str, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load, preprocess, and cache a dataset.

    Both new datasets are all-numeric with no missing values, so the
    drop-impute-one-hot steps from v1 are no-ops here.  They are applied
    anyway for consistency.  Results cached to data/{name}_preprocessed.npz.
    """
    cache_path = DATA_DIR / f"{name}_preprocessed.npz"
    if cache_path.exists():
        logger.info("Loading preprocessed cache: %s", cache_path)
        d = np.load(cache_path, allow_pickle=True)
        X = d["X"].astype(np.float64)
        y = d["y"].astype(np.float64)
        logger.info("  %s: n=%d  p=%d", name, len(y), X.shape[1])
        return X, y, int(X.shape[1])

    logger.info("Preprocessing %s (first run — will cache) ...", name)

    if name == "geo_music":
        X, y = _load_geo_music(cfg)
    elif name == "online_news":
        X, y = _load_online_news(cfg)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Drop non-finite rows (defensive; neither dataset should trigger this)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not valid.all():
        logger.warning("  Dropping %d non-finite rows", (~valid).sum())
        X, y = X[valid], y[valid]

    # Pool cap for large datasets
    pool_cap = cfg.get("pool_cap")
    if pool_cap and len(y) > pool_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=pool_cap, replace=False)
        X, y = X[idx], y[idx]
        logger.info("  Pool capped to %d rows", pool_cap)

    p = X.shape[1]
    logger.info("  Final: n=%d  p=%d", len(y), p)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    logger.info("  Cached to %s", cache_path)
    return X, y, p


# ---------------------------------------------------------------------------
# Per-cell runner — identical to v1
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
    """Fit all three methods for one (dataset, n, seed) cell."""
    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(len(X_full), size=n, replace=False)
    X_sub, y_sub = X_full[sub_idx], y_full[sub_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=seed
    )
    n_train = len(X_tr)

    X_tr, X_te, _ = standardize(X_tr, X_te)

    y_scaler = StandardScaler()
    y_tr = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

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
            name=name, n=n, n_regime=n_regime, p=p, seed=seed,
            method="ClassicalMAVE", k_used=float("nan"), est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("ClassicalMAVE", name, n_regime, seed)

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
            name=name, n=n, n_regime=n_regime, p=p, seed=seed,
            method="ClassicalMAVE_kNN", k_used=k, est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("ClassicalMAVE_kNN", name, n_regime, seed)

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
            name=name, n=n, n_regime=n_regime, p=p, seed=seed,
            method="SPMAVE", k_used=k, est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("SPMAVE", name, n_regime, seed)

    return rows


def _build_row(
    name: str, n: int, n_regime: str, p: int, seed: int,
    method: str, k_used: float, est: Any,
    X_tr: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_te: np.ndarray,
    fit_time: float,
) -> dict[str, Any]:
    result = est.result_
    mse    = evaluate_subspace(result.B, X_tr, X_te, y_tr, y_te)
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
        f"FAILURE  method={method}  dataset={name}  "
        f"n_regime={n_regime}  seed={seed}\n"
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
# Aggregation — identical to v1
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    "test_mse_linear",        "train_mse_linear",
    "test_mse_kernel_ridge",  "train_mse_kernel_ridge",
    "test_mse_random_forest", "train_mse_random_forest",
    "fit_time_s", "n_iters", "final_loss",
]


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean ± SE per (dataset, n_regime, method)."""
    datasets = sorted({r["dataset"]  for r in raw_rows})
    regimes  = sorted({r["n_regime"] for r in raw_rows})
    methods  = sorted({r["method"]   for r in raw_rows})
    agg: list[dict[str, Any]] = []

    for ds in datasets:
        for regime in regimes:
            for m in methods:
                subset = [
                    r for r in raw_rows
                    if r["dataset"] == ds and r["n_regime"] == regime
                    and r["method"] == m
                ]
                if not subset:
                    continue
                entry: dict[str, Any] = {
                    "dataset":    ds,
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
# Wilcoxon — identical to v1
# ---------------------------------------------------------------------------

def run_wilcoxon(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired Wilcoxon: SPMAVE vs each baseline per (dataset, n_regime).

    Metric: test_mse_kernel_ridge.
    rank_biserial > 0 means SPMAVE tends to have smaller values.
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
                if r["method"] == "SPMAVE"
                and r["dataset"] == ds and r["n_regime"] == regime
            }
            for bl in baselines:
                bl_rows = {
                    r["seed"]: r for r in raw_rows
                    if r["method"] == bl
                    and r["dataset"] == ds and r["n_regime"] == regime
                }
                common = sorted(sp_rows.keys() & bl_rows.keys())
                if len(common) < 10:
                    logger.warning(
                        "Skipping Wilcoxon ds=%s regime=%s bl=%s: %d pairs",
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
                r_rb = (
                    1.0 - 2.0 * stat / (n_nz * (n_nz + 1) / 2.0)
                    if n_nz > 0 and not _is_nan(stat) else float("nan")
                )
                results.append({
                    "dataset":            ds,
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
# CSV helpers
# ---------------------------------------------------------------------------

def _append_csv(new_rows: list[dict[str, Any]], path: Path) -> None:
    """Append rows to an existing CSV, preserving its column order.

    Reads the existing header to get fieldnames, verifies all required fields
    are present in new_rows, then appends without a new header line.
    """
    if not new_rows:
        logger.warning("No rows to append to %s", path)
        return
    with open(path, "r", newline="") as f:
        existing_fields = list(csv.DictReader(f).fieldnames or [])
    if not existing_fields:
        raise ValueError(f"Cannot read header from {path}")

    missing = set(existing_fields) - set(new_rows[0].keys())
    if missing:
        raise ValueError(
            f"New rows are missing fields required by {path}: {missing}"
        )

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=existing_fields, extrasaction="ignore")
        w.writerows(new_rows)
    logger.info("Appended %d rows → %s", len(new_rows), path)


def _write_csv_new(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a fresh CSV (used for smoke-test output only)."""
    if not rows:
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
# Regime table — reads combined summary for all datasets
# ---------------------------------------------------------------------------

def _print_regime_table_combined() -> None:
    """Print regime table from the full (all datasets) summary CSV."""
    with open(SUMMARY_CSV, "r", newline="") as f:
        summary = list(csv.DictReader(f))

    datasets = sorted({r["dataset"]  for r in summary})
    regimes  = sorted({r["n_regime"] for r in summary})

    print("\nRegime table — SP-MAVE KRR MSE < kNN-MAVE KRR MSE? (all datasets)")
    print(
        f"  {'Dataset':<25s}  {'n_regime':<12s}  {'p/n':>6}  "
        f"{'SPMAVE':>10}  {'kNN-MAVE':>10}  {'SP wins?':>9}"
    )
    print("  " + "-" * 84)
    for ds in datasets:
        for regime in regimes:
            sp_row  = next(
                (r for r in summary
                 if r["method"] == "SPMAVE"
                 and r["dataset"] == ds and r["n_regime"] == regime),
                None,
            )
            knn_row = next(
                (r for r in summary
                 if r["method"] == "ClassicalMAVE_kNN"
                 and r["dataset"] == ds and r["n_regime"] == regime),
                None,
            )
            if sp_row is None or knn_row is None:
                continue
            sp_v  = float(sp_row.get("test_mse_kernel_ridge_mean",  "nan"))
            knn_v = float(knn_row.get("test_mse_kernel_ridge_mean", "nan"))
            wins  = "Y" if (not math.isnan(sp_v) and not math.isnan(knn_v)
                            and sp_v < knn_v) else "N"
            pn    = float(sp_row.get("p_n_ratio", "nan"))
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

    if smoke_test:
        dataset_names = ["geo_music"]
        seeds         = [0, 1]
        regimes_map   = {"geo_music": ["subsampled"]}
        logger.info("SMOKE TEST: dataset=geo_music  regime=subsampled  seeds=%s", seeds)
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

    # Verify canonical CSVs exist before starting (full run only)
    if not smoke_test:
        for path in (RAW_CSV, SUMMARY_CSV, WILCOXON_CSV):
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected existing CSV not found: {path}\n"
                    "Run run_real_data_v1.py first."
                )

    # Load and preprocess all datasets
    loaded: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for ds_name in dataset_names:
        X, y, p = load_and_preprocess(ds_name, DATASETS[ds_name])
        loaded[ds_name] = (X, y, p)
        logger.info("Dataset ready: %s  n_pool=%d  p=%d", ds_name, len(y), p)

    # Build cell list: (dataset_name, n, n_regime, seed)
    cells: list[tuple[str, int, str, int]] = []
    for ds_name in dataset_names:
        cfg    = DATASETS[ds_name]
        n_pool = len(loaded[ds_name][1])
        for regime in regimes_map[ds_name]:
            n = cfg["n_sub"] if regime == "subsampled" else min(cfg["n_full"], n_pool)
            if n > n_pool:
                logger.warning(
                    "%s: requested n=%d > pool=%d; capping", ds_name, n, n_pool
                )
                n = n_pool
            for seed in seeds:
                cells.append((ds_name, n, regime, seed))

    if _have_tqdm:
        pbar = tqdm(total=len(cells) * 3, desc="real-data-v2", unit="fit")

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for ds_name, n, regime, seed in cells:
        X_full, y_full, p = loaded[ds_name]
        t_cell    = time.time()
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

    if not all_rows:
        logger.error("No rows produced — nothing to write.")
        return

    if smoke_test:
        _write_csv_new(all_rows,                     RESULTS_DIR / "real_data_results_v2_smoke.csv")
        _write_csv_new(aggregate_summary(all_rows),  RESULTS_DIR / "real_data_summary_v2_smoke.csv")
        logger.info("Smoke test complete — canonical CSVs untouched.")
        return

    # Append to canonical CSVs
    _append_csv(all_rows, RAW_CSV)
    _append_csv(aggregate_summary(all_rows), SUMMARY_CSV)
    try:
        _append_csv(run_wilcoxon(all_rows), WILCOXON_CSV)
    except ImportError:
        logger.warning("scipy not available — skipping Wilcoxon tests")

    _print_regime_table_combined()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Real data experiment v2 — adds geo_music and online_news datasets, "
            "appending results to the existing v1 CSVs."
        )
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: geo_music only, subsampled, 2 seeds. "
             "Writes to separate smoke files; canonical CSVs untouched.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
