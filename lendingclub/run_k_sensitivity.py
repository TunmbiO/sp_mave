#!/usr/bin/env python3
"""k-sensitivity sweep for the LendingClub default experiment.

Tests whether top-k truncation (not RF vs Gaussian proximity) explains the
Brier-score gap between kNN methods and untruncated ClassicalMAVE.  Varies
k_frac in [0.35, 0.50, 0.65, 0.80] while holding all other settings fixed.

Split identity guarantee
------------------------
Imports make_split from run_lendingclub_experiment — the IDENTICAL seeded
stratified-subsample + 80/20-split + X-standardisation logic used in the
main experiment.  Results are directly comparable: same seeds, same folds.

Smoke test
----------
Runs seed=0 at k_frac=0.50 only, then verifies the split matches the
original main-experiment run by re-fitting ClassicalMAVE_kNN at k_frac=0.20
and comparing Brier/AUC to the stored value in lendingclub_results.csv.

Usage
-----
    python run_k_sensitivity.py --smoke-test   # 1 seed, k_frac=0.50, verify split
    python run_k_sensitivity.py                # full 4 × 2 × 20 = 160 fits

Output
------
    results/k_sensitivity_results.csv
    results/k_sensitivity_summary.csv   (includes k=0.20 and ClassicalMAVE baselines)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (sp_mave package)

# Import shared split + evaluator from the main runner — do NOT re-implement.
from run_lendingclub_experiment import evaluate_binary, load_data, make_split
from sp_mave.mave import ClassicalMAVE
from sp_mave.spmave import SPMAVE
from sp_mave.utils import get_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR      = Path(__file__).resolve().parent / "results"
OUT_RESULTS      = RESULTS_DIR / "k_sensitivity_results.csv"
OUT_SUMMARY      = RESULTS_DIR / "k_sensitivity_summary.csv"
FAILURES_LOG     = RESULTS_DIR / "k_sensitivity_failures.log"
MAIN_RESULTS_CSV = RESULTS_DIR / "lendingclub_results.csv"

# ---------------------------------------------------------------------------
# Experiment constants — everything not k is identical to the main runner
# ---------------------------------------------------------------------------

N_SUB: int   = 300
R_DIM: int   = 3
SEEDS: list[int] = list(range(20))

K_FRACS: list[float] = [0.35, 0.50, 0.65, 0.80]

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

# ClassicalMAVE (no k) reference from the full 20-seed main run, subsampled regime.
# Hardcoded here for the summary table; sourced from lendingclub_summary.csv.
_CLASSICAL_MAVE_REF = {
    "brier_test_mean": 0.1908, "brier_test_se": 0.0051,
    "auc_test_mean":   0.6183, "auc_test_se":   0.0150,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("k_sensitivity")
logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)
logging.getLogger("sp_mave.spmave").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_one_cell(
    X_full: np.ndarray,
    y_full: np.ndarray,
    p: int,
    k_frac: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Fit ClassicalMAVE_kNN and SPMAVE for one (k_frac, seed) cell.

    Uses make_split (imported from run_lendingclub_experiment) — identical
    split to the main experiment at the same seed and n=300.
    """
    X_tr, X_te, y_tr, y_te = make_split(X_full, y_full, N_SUB, seed)
    n_train = len(X_tr)
    k = max(1, math.floor(k_frac * n_train))

    model_cfg = {**MAVE_MODEL_CFG, "random_state": seed}
    rows: list[dict[str, Any]] = []

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
            p=p, k_frac=k_frac, k=k, seed=seed,
            method="ClassicalMAVE_kNN", est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("ClassicalMAVE_kNN", k_frac, seed)

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
            p=p, k_frac=k_frac, k=k, seed=seed,
            method="SPMAVE", est=est,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            fit_time=time.time() - t0,
        ))
    except Exception:
        _log_failure("SPMAVE", k_frac, seed)

    return rows


def _build_row(
    p: int, k_frac: float, k: int, seed: int, method: str,
    est: Any, X_tr: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray, y_te: np.ndarray, fit_time: float,
) -> dict[str, Any]:
    result  = est.result_
    metrics = evaluate_binary(result.B, X_tr, X_te, y_tr, y_te)
    return {
        "dataset":    "lendingclub",
        "n_obs":      N_SUB,
        "seed":       seed,
        "method":     method,
        "k_used":     k,
        "k_frac":     k_frac,
        "r_dim":      R_DIM,
        **metrics,
        "fit_time_s": round(fit_time, 4),
        "n_iters":    result.n_iters,
        "final_loss": result.loss_history[-1] if result.loss_history else float("nan"),
    }


def _log_failure(method: str, k_frac: float, seed: int) -> None:
    msg = (
        f"FAILURE  method={method}  k_frac={k_frac}  seed={seed}\n"
        + traceback.format_exc() + "\n"
    )
    logger.error("Cell failed: method=%s  k_frac=%.2f  seed=%d", method, k_frac, seed)
    with open(FAILURES_LOG, "a") as fh:
        fh.write(msg)


# ---------------------------------------------------------------------------
# Split verification (smoke test only)
# ---------------------------------------------------------------------------

def _verify_split(X_full: np.ndarray, y_full: np.ndarray, p: int) -> None:
    """Confirm make_split at seed=0 reproduces the main-experiment results.

    Re-fits ClassicalMAVE_kNN at k_frac=0.20 (the main experiment's k) and
    compares Brier/AUC to the stored row in lendingclub_results.csv.
    Agreement to 1e-4 confirms identical splits.
    """
    if not MAIN_RESULTS_CSV.exists():
        logger.warning(
            "Split verification skipped: %s not found "
            "(run the full main experiment first).",
            MAIN_RESULTS_CSV,
        )
        return

    import pandas as pd
    df = pd.read_csv(MAIN_RESULTS_CSV)
    ref = df[
        (df["seed"] == 0)
        & (df["method"] == "ClassicalMAVE_kNN")
        & (df["n_regime"] == "subsampled")
    ]
    if ref.empty:
        logger.warning(
            "Split verification skipped: seed=0 ClassicalMAVE_kNN subsampled "
            "row not found in %s.",
            MAIN_RESULTS_CSV,
        )
        return

    ref_brier = float(ref["brier_test"].iloc[0])
    ref_auc   = float(ref["auc_test"].iloc[0])
    ref_k     = int(ref["k_used"].iloc[0])

    X_tr, X_te, y_tr, y_te = make_split(X_full, y_full, N_SUB, seed=0)
    n_train = len(X_tr)
    k_020   = max(1, math.floor(0.20 * n_train))

    if k_020 != ref_k:
        logger.warning(
            "Split verification: k mismatch — expected %d from main run, got %d",
            ref_k, k_020,
        )
        return

    model_cfg = {**MAVE_MODEL_CFG, "random_state": 0}
    est = ClassicalMAVE(
        d=R_DIM, bandwidth=model_cfg["bandwidth"], k=k_020,
        max_iters=model_cfg["max_iters"], tol=model_cfg["tol"],
        ridge_lambda=model_cfg["ridge_lambda"],
        init_method=model_cfg["init_method"], random_state=0,
    )
    est.fit(X_tr, y_tr)
    metrics = evaluate_binary(est.result_.B, X_tr, X_te, y_tr, y_te)

    brier_ok = abs(metrics["brier_test"] - ref_brier) < 1e-4
    auc_ok   = abs(metrics["auc_test"]   - ref_auc)   < 1e-4

    if brier_ok and auc_ok:
        logger.info(
            "Split verification PASSED — seed=0 k=%.2f: "
            "Brier %.4f≈%.4f, AUC %.4f≈%.4f",
            0.20, metrics["brier_test"], ref_brier,
            metrics["auc_test"], ref_auc,
        )
    else:
        logger.warning(
            "Split verification FAILED — "
            "Brier %.6f vs %.6f, AUC %.6f vs %.6f. "
            "Splits may differ — do not compare results.",
            metrics["brier_test"], ref_brier,
            metrics["auc_test"],   ref_auc,
        )


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
    """Mean ± SE per (method, k_frac) over seeds."""
    k_fracs = sorted({r["k_frac"] for r in raw_rows})
    methods = sorted({r["method"] for r in raw_rows})
    agg: list[dict[str, Any]] = []

    for kf in k_fracs:
        for m in methods:
            subset = [r for r in raw_rows if r["k_frac"] == kf and r["method"] == m]
            if not subset:
                continue
            entry: dict[str, Any] = {
                "method":  m,
                "k_frac":  kf,
                "k_used":  subset[0]["k_used"],
                "n_seeds": len(subset),
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
# Console summary table
# ---------------------------------------------------------------------------

def _print_summary_table(summary: list[dict[str, Any]]) -> None:
    """Print k-sensitivity table with k=0.20 and ClassicalMAVE baselines.

    Reads k=0.20 baseline from lendingclub_results.csv (subsampled regime)
    if available; otherwise omits those rows.
    """
    # --- Pull k=0.20 baseline from main results CSV ---
    k020_rows: dict[str, dict[str, float]] = {}   # method -> {brier_mean, brier_se, auc_mean, auc_se}
    if MAIN_RESULTS_CSV.exists():
        import pandas as pd
        df = pd.read_csv(MAIN_RESULTS_CSV)
        sub = df[df["n_regime"] == "subsampled"]
        for m in ["ClassicalMAVE_kNN", "SPMAVE"]:
            rows = sub[sub["method"] == m]
            if not rows.empty:
                brier_vals = rows["brier_test"].values
                auc_vals   = rows["auc_test"].values
                k020_rows[m] = {
                    "brier_mean": float(np.mean(brier_vals)),
                    "brier_se":   float(np.std(brier_vals, ddof=1) / math.sqrt(len(brier_vals))),
                    "auc_mean":   float(np.mean(auc_vals)),
                    "auc_se":     float(np.std(auc_vals, ddof=1) / math.sqrt(len(auc_vals))),
                }

    methods = sorted({r["method"] for r in summary})

    print("\n" + "=" * 100)
    print("k-SENSITIVITY — n=300 subsampled, r=3  (mean ± SE over 20 seeds)")
    print("Brier lower=better  |  AUC higher=better")
    print("=" * 100)

    hdr = f"  {'k_frac':>7}  {'k_used':>7}"
    for m in methods:
        short = "kNN-MAVE" if "kNN" in m else "SP-MAVE"
        hdr += f"  {short+' Brier':>20}  {short+' AUC':>18}"
    print(hdr)
    print("  " + "-" * 94)

    def _fmt(mu: float | None, se: float | None) -> str:
        if mu is None:
            return "—"
        if se is None or _is_nan(se):
            return f"{mu:.4f}"
        return f"{mu:.4f}±{se:.4f}"

    # Reference rows at k=0.20
    if k020_rows:
        row_str = f"  {'0.20':>7}  {'*':>7}  (from main run)"
        parts = []
        for m in methods:
            d = k020_rows.get(m, {})
            parts.append(
                f"  {_fmt(d.get('brier_mean'), d.get('brier_se')):>20}"
                f"  {_fmt(d.get('auc_mean'),   d.get('auc_se')):>18}"
            )
        print(f"  {'0.20':>7}  {'*':>7}" + "".join(parts) + "  ← main-run k")
    else:
        print("  (k=0.20 baseline unavailable — lendingclub_results.csv not found)")

    # New k_frac rows
    k_fracs = sorted({r["k_frac"] for r in summary})
    for kf in k_fracs:
        k_used_vals = [r["k_used"] for r in summary if r["k_frac"] == kf]
        k_used_str  = str(k_used_vals[0]) if k_used_vals else "?"
        parts = []
        for m in methods:
            row = next((r for r in summary if r["k_frac"] == kf and r["method"] == m), None)
            if row:
                parts.append(
                    f"  {_fmt(row.get('brier_test_mean'), row.get('brier_test_se')):>20}"
                    f"  {_fmt(row.get('auc_test_mean'),   row.get('auc_test_se')):>18}"
                )
            else:
                parts.append(f"  {'—':>20}  {'—':>18}")
        print(f"  {kf:>7.2f}  {k_used_str:>7}" + "".join(parts))

    # ClassicalMAVE (no k) reference
    ref = _CLASSICAL_MAVE_REF
    print("  " + "-" * 94)
    ref_line = (
        f"  {'—':>7}  {'—':>7}  ClassicalMAVE (no k): "
        f"Brier {_fmt(ref['brier_test_mean'], ref['brier_test_se'])}  "
        f"AUC {_fmt(ref['auc_test_mean'], ref['auc_test_se'])}"
    )
    print(ref_line)
    print("=" * 100 + "\n")


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
        k_fracs_run = [0.50]
        seeds_run   = [0]
        logger.info("SMOKE TEST: k_frac=0.50, seed=0 — then split verification")
    else:
        k_fracs_run = K_FRACS
        seeds_run   = SEEDS
        logger.info(
            "FULL RUN: %d k_fracs × %d seeds × 2 methods = %d fits",
            len(k_fracs_run), len(seeds_run), len(k_fracs_run) * len(seeds_run) * 2,
        )

    cells = [
        (kf, seed)
        for kf in k_fracs_run
        for seed in seeds_run
    ]

    if _have_tqdm:
        pbar = tqdm(total=len(cells) * 2, desc="k-sensitivity", unit="fit")

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for k_frac, seed in cells:
        t_cell    = time.time()
        cell_rows = run_one_cell(X_full, y_full, p, k_frac, seed)
        elapsed   = time.time() - t_cell
        all_rows.extend(cell_rows)
        msg = f"k_frac={k_frac:.2f}  seed={seed:2d}  {len(cell_rows)}/2 OK  ({elapsed:.1f}s)"
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
    _write_csv(all_rows, RESULTS_DIR / f"k_sensitivity_results{suffix}.csv")
    summary = aggregate_summary(all_rows)
    _write_csv(summary,   RESULTS_DIR / f"k_sensitivity_summary{suffix}.csv")

    _print_summary_table(summary)

    # Split verification — smoke test only, runs after the main fit
    if smoke_test:
        logger.info("Running split verification against main experiment ...")
        _verify_split(X_full, y_full, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="k-sensitivity sweep for LendingClub default experiment."
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="k_frac=0.50, seed=0 only, then verifies split against main run.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
