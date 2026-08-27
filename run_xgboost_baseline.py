#!/usr/bin/env python3
"""Industry-standard baselines for LendingClub default prediction.

Fits XGBoost and LogisticRegression on the full 94-feature space (no
dimensionality reduction) to contextualise the SP-MAVE 3-d subspace results.
Hyperparameters are fixed across all seeds — consistent with how SP-MAVE's
hyperparameters are fixed.

Split identity
--------------
Uses make_split imported from run_lendingclub_experiment — identical
stratified pool-draw and 80/20 train/test splits, same seeds.  Results are
directly seed-for-seed comparable to lendingclub_results.csv.

Usage
-----
    python run_xgboost_baseline.py --smoke-test  # seed=0, n=300 only
    python run_xgboost_baseline.py               # full 20 seeds × 2 regimes

Output
------
    results/xgboost_baseline_results.csv
    results/xgboost_baseline_summary.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_lendingclub_experiment import load_data, make_split
from sp_mave.utils import get_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR       = Path(__file__).resolve().parent / "results"
OUT_RESULTS       = RESULTS_DIR / "xgboost_baseline_results.csv"
OUT_SUMMARY       = RESULTS_DIR / "xgboost_baseline_summary.csv"
FAILURES_LOG      = RESULTS_DIR / "xgboost_baseline_failures.log"
SPMAVE_RESULTS    = RESULTS_DIR / "lendingclub_results.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS: list[int] = list(range(20))
REGIMES: dict[str, int] = {"subsampled": 300, "full": 2000}

# Fixed XGBoost hyperparameters — not tuned per seed
XGB_PARAMS: dict[str, Any] = {
    "n_estimators":    300,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "eval_metric":     "logloss",
    "verbosity":       0,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("xgb_baseline")
logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------

def _eval_probs(
    probs_tr: np.ndarray,
    probs_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
) -> dict[str, float]:
    return {
        "brier_train": float(np.mean((probs_tr - y_tr) ** 2)),
        "brier_test":  float(np.mean((probs_te - y_te) ** 2)),
        "auc_train":   float(roc_auc_score(y_tr, probs_tr)),
        "auc_test":    float(roc_auc_score(y_te, probs_te)),
    }


def run_one_cell(
    X_full: np.ndarray,
    y_full: np.ndarray,
    n: int,
    n_regime: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Fit XGBoost and LogisticRegression for one (n, seed) cell."""
    X_tr, X_te, y_tr, y_te = make_split(X_full, y_full, n, seed)
    rows: list[dict[str, Any]] = []

    # ---- XGBoost ------------------------------------------------------------
    try:
        t0  = time.time()
        mdl = XGBClassifier(**XGB_PARAMS, random_state=seed)
        mdl.fit(X_tr, y_tr)
        probs_tr = mdl.predict_proba(X_tr)[:, 1]
        probs_te = mdl.predict_proba(X_te)[:, 1]
        rows.append({
            "dataset": "lendingclub", "n_obs": n, "n_regime": n_regime,
            "seed": seed, "method": "XGBoost",
            **_eval_probs(probs_tr, probs_te, y_tr, y_te),
            "fit_time_s": round(time.time() - t0, 4),
        })
    except Exception:
        _log_failure("XGBoost", n_regime, seed)

    # ---- LogisticRegression (features already standardised by make_split) ---
    try:
        t0  = time.time()
        mdl = LogisticRegression(max_iter=1000, random_state=seed)
        mdl.fit(X_tr, y_tr)
        probs_tr = mdl.predict_proba(X_tr)[:, 1]
        probs_te = mdl.predict_proba(X_te)[:, 1]
        rows.append({
            "dataset": "lendingclub", "n_obs": n, "n_regime": n_regime,
            "seed": seed, "method": "LogisticRegression",
            **_eval_probs(probs_tr, probs_te, y_tr, y_te),
            "fit_time_s": round(time.time() - t0, 4),
        })
    except Exception:
        _log_failure("LogisticRegression", n_regime, seed)

    return rows


def _log_failure(method: str, n_regime: str, seed: int) -> None:
    msg = (
        f"FAILURE  method={method}  n_regime={n_regime}  seed={seed}\n"
        + traceback.format_exc() + "\n"
    )
    logger.error("Cell failed: %s  regime=%s  seed=%d", method, n_regime, seed)
    with open(FAILURES_LOG, "a") as fh:
        fh.write(msg)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_COLS = ["brier_train", "brier_test", "auc_train", "auc_test", "fit_time_s"]


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regimes = sorted({r["n_regime"] for r in raw_rows})
    methods = sorted({r["method"]   for r in raw_rows})
    agg: list[dict[str, Any]] = []
    for regime in regimes:
        for m in methods:
            subset = [r for r in raw_rows if r["n_regime"] == regime and r["method"] == m]
            if not subset:
                continue
            entry: dict[str, Any] = {
                "dataset":  "lendingclub",
                "n_regime": regime,
                "method":   m,
                "n_seeds":  len(subset),
                "n_obs":    subset[0]["n_obs"],
            }
            for col in _METRIC_COLS:
                vals = [float(r[col]) for r in subset if col in r and not _is_nan(r[col])]
                if vals:
                    entry[f"{col}_mean"] = float(np.mean(vals))
                    entry[f"{col}_se"]   = (
                        float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                        if len(vals) > 1 else float("nan")
                    )
            agg.append(entry)
    return agg


# ---------------------------------------------------------------------------
# Summary table (all four methods side by side)
# ---------------------------------------------------------------------------

def _print_combined_table(
    baseline_summary: list[dict[str, Any]],
    regimes_run: list[str],
) -> None:
    """Print baseline results alongside SP-MAVE / ClassicalMAVE_kNN from main CSV."""

    # Load existing SP-MAVE results
    spmave_stats: dict[tuple[str, str], dict] = {}   # (regime, method) -> stats
    if SPMAVE_RESULTS.exists():
        import pandas as pd
        df = pd.read_csv(SPMAVE_RESULTS)
        for regime in regimes_run:
            for m in ["SPMAVE", "ClassicalMAVE_kNN", "ClassicalMAVE"]:
                rows = df[(df["n_regime"] == regime) & (df["method"] == m)]
                if rows.empty:
                    continue
                bv = rows["brier_test"].values
                av = rows["auc_test"].values
                spmave_stats[(regime, m)] = {
                    "brier_mean": float(np.mean(bv)),
                    "brier_se":   float(np.std(bv, ddof=1) / math.sqrt(len(bv))),
                    "auc_mean":   float(np.mean(av)),
                    "auc_se":     float(np.std(av, ddof=1) / math.sqrt(len(av))),
                    "n":          len(bv),
                }

    def _fmt(mu: float | None, se: float | None) -> str:
        if mu is None:
            return "—"
        if se is None or _is_nan(se):
            return f"{mu:.4f}"
        return f"{mu:.4f}±{se:.4f}"

    ORDERED_METHODS = [
        "XGBoost", "LogisticRegression",
        "SPMAVE", "ClassicalMAVE_kNN", "ClassicalMAVE",
    ]

    print("\n" + "=" * 92)
    print("LENDINGCLUB — ALL METHODS COMPARISON  (mean ± SE, Brier↓ / AUC↑)")
    print("=" * 92)

    for regime in regimes_run:
        n_obs = next(
            (r["n_obs"] for r in baseline_summary if r["n_regime"] == regime),
            REGIMES.get(regime, "?"),
        )
        print(f"\n  {regime}  (n={n_obs})")
        print(f"  {'Method':<24s}  {'Brier test':>20s}  {'AUC test':>18s}  {'n_seeds':>7}")
        print("  " + "-" * 76)
        for m in ORDERED_METHODS:
            # Try baseline summary first, then SP-MAVE results
            bl_row = next(
                (r for r in baseline_summary if r["n_regime"] == regime and r["method"] == m),
                None,
            )
            if bl_row is not None:
                brier_str = _fmt(bl_row.get("brier_test_mean"), bl_row.get("brier_test_se"))
                auc_str   = _fmt(bl_row.get("auc_test_mean"),   bl_row.get("auc_test_se"))
                n_str     = str(bl_row.get("n_seeds", "?"))
            elif (regime, m) in spmave_stats:
                d         = spmave_stats[(regime, m)]
                brier_str = _fmt(d["brier_mean"], d["brier_se"])
                auc_str   = _fmt(d["auc_mean"],   d["auc_se"])
                n_str     = str(d["n"])
            else:
                continue
            print(f"  {m:<24s}  {brier_str:>20s}  {auc_str:>18s}  {n_str:>7}")

    print()
    print("  Note: Brier lower=better, AUC higher=better.")
    print("  SP-MAVE and ClassicalMAVE results from lendingclub_results.csv.")
    print("=" * 92 + "\n")


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
        regimes_run = ["subsampled"]
        seeds_run   = [0]
        logger.info("SMOKE TEST: regime=subsampled, seed=0, XGBoost + LogisticRegression")
    else:
        regimes_run = list(REGIMES.keys())
        seeds_run   = SEEDS
        logger.info(
            "FULL RUN: %d regimes × %d seeds × 2 methods = %d fits",
            len(regimes_run), len(seeds_run), len(regimes_run) * len(seeds_run) * 2,
        )

    cells = [
        (REGIMES[regime], regime, seed)
        for regime in regimes_run
        for seed in seeds_run
    ]

    if _have_tqdm:
        pbar = tqdm(total=len(cells) * 2, desc="xgb-baseline", unit="fit")

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for n, n_regime, seed in cells:
        t_cell    = time.time()
        cell_rows = run_one_cell(X_full, y_full, n, n_regime, seed)
        elapsed   = time.time() - t_cell
        all_rows.extend(cell_rows)
        msg = f"regime={n_regime}  n={n}  seed={seed:2d}  {len(cell_rows)}/2 OK  ({elapsed:.1f}s)"
        if _have_tqdm:
            pbar.update(2)
            pbar.set_postfix_str(msg)
        else:
            logger.info(msg)

    if _have_tqdm:
        pbar.close()

    total_elapsed = time.time() - t_start
    logger.info("Completed %d rows in %.1fs (%.1f min)",
                len(all_rows), total_elapsed, total_elapsed / 60)

    if not all_rows:
        logger.error("No rows produced.")
        return

    suffix = "_smoke" if smoke_test else ""
    _write_csv(all_rows,                RESULTS_DIR / f"xgboost_baseline_results{suffix}.csv")
    summary = aggregate_summary(all_rows)
    _write_csv(summary,                 RESULTS_DIR / f"xgboost_baseline_summary{suffix}.csv")
    _print_combined_table(summary, regimes_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost + LogisticRegression baselines for LendingClub default.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="seed=0, n=300 only.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
