#!/usr/bin/env python3
"""KRR tuning robustness check for the real-data experiments.

results/real_data/real_data_results.csv (the PRIMARY real-data results)
uses KernelRidge(kernel="rbf", alpha=1.0, gamma=1/r) with sklearn defaults
throughout, deliberately un-tuned so that MSE differences reflect subspace
quality rather than downstream hyperparameter search. This script adds an
APPENDIX-only robustness check: CV-tuned KRR on the same (dataset, regime,
seed, method) grid.

No B_hat matrices were ever saved to disk from the original runs, so this
script regenerates them by re-running MAVE with the identical seeds and
hyperparameters used in run_real_data_v1.py / run_real_data_v2.py (subsample
-> 80/20 split -> standardize -> fit, all keyed by random_state=seed), then
caches each B_hat to results/real_data/b_hat/ for future reuse. Each
regenerated row's default-KRR MSE is cross-checked against the original CSV
as a correctness guard on the regeneration.

CRITICAL: this script only ever READS real_data_results.csv /
_summary.csv / _wilcoxon.csv. It never writes to them.

Usage
-----
    python run_krr_tuning_robustness.py                # full run, parallel
    python run_krr_tuning_robustness.py --smoke-test    # a handful of cells
    python run_krr_tuning_robustness.py --workers 16    # override worker count

Outputs
-------
    results/real_data/b_hat/{dataset}_{regime}_{method}_seed{seed}.npy
    results/real_data/real_data_tuned_krr.csv
    results/real_data/real_data_tuned_krr_summary.csv
    results/real_data/real_data_tuned_krr_wilcoxon.csv
    results/real_data/real_data_tuned_krr_failures.log
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_real_data_v1 as rv1
import run_real_data_v2 as rv2
from sp_mave.evaluate import evaluate_subspace
from sp_mave.mave import ClassicalMAVE
from sp_mave.preprocessing import standardize
from sp_mave.spmave import SPMAVE
from sp_mave.utils import get_logger

logger = get_logger("krr_tuning")

RESULTS_DIR  = Path(__file__).resolve().parent / "results" / "real_data"
BHAT_DIR     = RESULTS_DIR / "b_hat"
RAW_CSV      = RESULTS_DIR / "real_data_results.csv"   # READ-ONLY input
FAILURES_LOG = RESULTS_DIR / "real_data_tuned_krr_failures.log"

# Guard: these files must never be opened for writing by this script.
PROTECTED_FILES = {
    (RESULTS_DIR / "real_data_results.csv").resolve(),
    (RESULTS_DIR / "real_data_summary.csv").resolve(),
    (RESULTS_DIR / "real_data_wilcoxon.csv").resolve(),
}

METHODS = ["ClassicalMAVE", "ClassicalMAVE_kNN", "SPMAVE"]

KRR_PARAM_GRID: dict[str, list[float]] = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}

_MODULE_BY_DATASET: dict[str, Any] = {}
for _name in rv1.DATASETS:
    _MODULE_BY_DATASET[_name] = rv1
for _name in rv2.DATASETS:
    _MODULE_BY_DATASET[_name] = rv2


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    resolved = path.resolve()
    if resolved in PROTECTED_FILES:
        raise RuntimeError(f"Refusing to write protected primary file: {path}")
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
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows -> %s", len(rows), path)


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Regenerate the exact split used by run_real_data_v1/v2's run_one_cell
# ---------------------------------------------------------------------------

def _regen_split(
    dataset: str, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    mod = _MODULE_BY_DATASET[dataset]
    cfg = mod.DATASETS[dataset]
    X_full, y_full, _p = mod.load_and_preprocess(dataset, cfg)

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

    k = max(1, math.floor(mod.K_FRACTION * n_train))
    return X_tr, X_te, y_tr, y_te, k, mod


def _fit_method(method: str, X_tr: np.ndarray, y_tr: np.ndarray, k: int, seed: int, mod: Any) -> Any:
    r_dim = mod.R_DIM
    cfg = mod.MAVE_MODEL_CFG
    if method == "ClassicalMAVE":
        est = ClassicalMAVE(
            d=r_dim, bandwidth=cfg["bandwidth"], k=None,
            max_iters=cfg["max_iters"], tol=cfg["tol"],
            ridge_lambda=cfg["ridge_lambda"], init_method=cfg["init_method"],
            random_state=seed,
        )
    elif method == "ClassicalMAVE_kNN":
        est = ClassicalMAVE(
            d=r_dim, bandwidth=cfg["bandwidth"], k=k,
            max_iters=cfg["max_iters"], tol=cfg["tol"],
            ridge_lambda=cfg["ridge_lambda"], init_method=cfg["init_method"],
            random_state=seed,
        )
    elif method == "SPMAVE":
        crossfit_cfg = {**mod.SPMAVE_CROSSFIT_CFG, "random_state": seed}
        rf_params = {**mod.SPMAVE_RF_PARAMS, "random_state": seed, "n_jobs": 1}
        est = SPMAVE(
            d=r_dim, k=k, tau=mod.SPMAVE_TAU,
            rf_params=rf_params, crossfit_cfg=crossfit_cfg,
            max_iters=cfg["max_iters"], tol=cfg["tol"],
            ridge_lambda=cfg["ridge_lambda"], init_method=cfg["init_method"],
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    est.fit(X_tr, y_tr)
    return est


# ---------------------------------------------------------------------------
# Per-triple worker: regenerate B_hat for all 3 methods + tune KRR
# ---------------------------------------------------------------------------

def process_triple(
    dataset: str, n_regime: str, n: int, p: int, seed: int,
    default_mse_lookup: dict[tuple[str, str, int, str], float],
) -> tuple[list[dict[str, Any]], list[str]]:
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV, KFold
    import threadpoolctl

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    with threadpoolctl.threadpool_limits(limits=1):
        try:
            X_tr, X_te, y_tr, y_te, k, mod = _regen_split(dataset, n, seed)
        except Exception:
            failures.append(
                f"FAILURE  stage=split  dataset={dataset}  n_regime={n_regime}  seed={seed}\n"
                + traceback.format_exc() + "\n"
            )
            return rows, failures

        for method in METHODS:
            try:
                t0 = time.time()
                est = _fit_method(method, X_tr, y_tr, k, seed, mod)
                fit_time = time.time() - t0
                B_hat = est.result_.B

                bhat_path = BHAT_DIR / f"{dataset}_{n_regime}_{method}_seed{seed}.npy"
                np.save(bhat_path, B_hat)

                # Validation: recompute default-KRR MSE, compare to original CSV.
                default_recomputed = evaluate_subspace(B_hat, X_tr, X_te, y_tr, y_te)[
                    "test_mse_kernel_ridge"
                ]
                default_original = default_mse_lookup.get((dataset, n_regime, seed, method))
                if default_original is not None and not math.isclose(
                    default_recomputed, default_original, rel_tol=1e-3, abs_tol=1e-6
                ):
                    failures.append(
                        f"WARNING  stage=validate  dataset={dataset}  n_regime={n_regime}  "
                        f"seed={seed}  method={method}  recomputed={default_recomputed!r}  "
                        f"original={default_original!r}\n"
                    )

                Z_tr = X_tr @ B_hat
                Z_te = X_te @ B_hat

                gs = GridSearchCV(
                    KernelRidge(kernel="rbf"),
                    param_grid=KRR_PARAM_GRID,
                    scoring="neg_mean_squared_error",
                    cv=KFold(n_splits=5, shuffle=False),
                    n_jobs=1,
                )
                gs.fit(Z_tr, y_tr)
                y_pred_te = gs.best_estimator_.predict(Z_te)
                tuned_mse = float(np.mean((y_te - y_pred_te) ** 2))

                rows.append({
                    "dataset": dataset,
                    "n_regime": n_regime,
                    "n_obs": n,
                    "p_features": p,
                    "seed": seed,
                    "method": method,
                    "r_dim": mod.R_DIM,
                    "test_mse_kernel_ridge_default": (
                        default_original if default_original is not None else default_recomputed
                    ),
                    "test_mse_kernel_ridge_tuned": tuned_mse,
                    "best_alpha": gs.best_params_["alpha"],
                    "best_gamma": gs.best_params_["gamma"],
                    "fit_time_s": round(fit_time, 4),
                })
            except Exception:
                failures.append(
                    f"FAILURE  stage=method  dataset={dataset}  n_regime={n_regime}  "
                    f"seed={seed}  method={method}\n" + traceback.format_exc() + "\n"
                )

    return rows, failures


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({r["dataset"] for r in rows})
    regimes  = sorted({r["n_regime"] for r in rows})
    agg: list[dict[str, Any]] = []
    for ds in datasets:
        for regime in regimes:
            for method in METHODS:
                subset = [
                    r for r in rows
                    if r["dataset"] == ds and r["n_regime"] == regime and r["method"] == method
                ]
                if not subset:
                    continue
                entry: dict[str, Any] = {
                    "dataset": ds, "n_regime": regime, "method": method,
                    "n_seeds": len(subset),
                }
                for col in ("test_mse_kernel_ridge_default", "test_mse_kernel_ridge_tuned"):
                    vals = [float(r[col]) for r in subset if not _is_nan(r[col])]
                    if vals:
                        entry[f"{col}_mean"] = float(np.mean(vals))
                        entry[f"{col}_se"] = (
                            float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                            if len(vals) > 1 else float("nan")
                        )
                agg.append(entry)
    return agg


def run_wilcoxon(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scipy.stats import wilcoxon

    datasets = sorted({r["dataset"] for r in rows})
    regimes  = sorted({r["n_regime"] for r in rows})
    baselines = ["ClassicalMAVE", "ClassicalMAVE_kNN"]
    metric = "test_mse_kernel_ridge_tuned"
    results: list[dict[str, Any]] = []

    for ds in datasets:
        for regime in regimes:
            sp_rows = {
                r["seed"]: r for r in rows
                if r["dataset"] == ds and r["n_regime"] == regime and r["method"] == "SPMAVE"
            }
            for bl in baselines:
                bl_rows = {
                    r["seed"]: r for r in rows
                    if r["dataset"] == ds and r["n_regime"] == regime and r["method"] == bl
                }
                common = sorted(sp_rows.keys() & bl_rows.keys())
                if len(common) < 10:
                    logger.warning(
                        "Skipping Wilcoxon dataset=%s regime=%s bl=%s: only %d paired seeds",
                        ds, regime, bl, len(common),
                    )
                    continue
                sp_vals = np.array([sp_rows[s][metric] for s in common], dtype=float)
                bl_vals = np.array([bl_rows[s][metric] for s in common], dtype=float)
                diffs = sp_vals - bl_vals
                if np.all(diffs == 0):
                    stat, pval = float("nan"), float("nan")
                else:
                    try:
                        stat, pval = wilcoxon(diffs, alternative="two-sided")
                        stat, pval = float(stat), float(pval)
                    except Exception:
                        stat, pval = float("nan"), float("nan")
                n_nz = int(np.sum(np.abs(diffs) > 0))
                r_rb = (
                    1.0 - 2.0 * stat / (n_nz * (n_nz + 1) / 2.0)
                    if n_nz > 0 and not _is_nan(stat) else float("nan")
                )
                results.append({
                    "dataset": ds, "n_regime": regime, "baseline": bl, "metric": metric,
                    "n_pairs": len(common),
                    "spmave_mean": float(np.mean(sp_vals)),
                    "baseline_mean": float(np.mean(bl_vals)),
                    "wilcoxon_statistic": stat, "p_value": pval, "rank_biserial": r_rb,
                })
    return results


# ---------------------------------------------------------------------------
# Side-by-side comparison + ranking-change report
# ---------------------------------------------------------------------------

def print_comparison(summary: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("KRR TUNING ROBUSTNESS — default vs. CV-tuned KRR (mean test MSE across seeds)")
    print("=" * 100)
    header = f"  {'Dataset':<20} {'Regime':<12} {'Method':<20} {'Default MSE':>14} {'Tuned MSE':>14} {'Delta (%)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    datasets = sorted({r["dataset"] for r in summary})
    regimes  = sorted({r["n_regime"] for r in summary})
    n_changed = 0
    n_cells = 0

    for ds in datasets:
        for regime in regimes:
            cell_rows = [r for r in summary if r["dataset"] == ds and r["n_regime"] == regime]
            if not cell_rows:
                continue
            for r in cell_rows:
                d_mse = r.get("test_mse_kernel_ridge_default_mean", float("nan"))
                t_mse = r.get("test_mse_kernel_ridge_tuned_mean", float("nan"))
                delta = (
                    100.0 * (t_mse - d_mse) / d_mse
                    if d_mse and not _is_nan(d_mse) and d_mse != 0 else float("nan")
                )
                print(
                    f"  {ds:<20} {regime:<12} {r['method']:<20} "
                    f"{d_mse:>14.5f} {t_mse:>14.5f} {delta:>+9.2f}%"
                )

            rank_default = sorted(cell_rows, key=lambda r: r.get("test_mse_kernel_ridge_default_mean", math.inf))
            rank_tuned   = sorted(cell_rows, key=lambda r: r.get("test_mse_kernel_ridge_tuned_mean", math.inf))
            order_default = [r["method"] for r in rank_default]
            order_tuned   = [r["method"] for r in rank_tuned]
            n_cells += 1
            if order_default != order_tuned:
                n_changed += 1

    print("\n" + "=" * 100)
    print(f"Method ranking changed between default and tuned KRR in {n_changed}/{n_cells} (dataset, regime) cells.")
    if n_changed == 0:
        print('Conclusion: "Results are robust to KRR hyperparameter choice."')
    else:
        print("Conclusion: ranking is NOT fully stable — see cells above for details.")
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False, n_workers: int | None = None) -> None:
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    if not RAW_CSV.exists():
        raise RuntimeError(f"{RAW_CSV} not found")

    src = pd.read_csv(RAW_CSV)
    default_mse_lookup: dict[tuple[str, str, int, str], float] = {
        (row.dataset, row.n_regime, int(row.seed), row.method): float(row.test_mse_kernel_ridge)
        for row in src.itertuples()
    }

    triples = (
        src[["dataset", "n_regime", "n_obs", "p_features", "seed"]]
        .drop_duplicates()
        .to_records(index=False)
        .tolist()
    )
    triples = [(d, reg, int(n), int(p), int(s)) for d, reg, n, p, s in triples]

    if smoke_test:
        triples = triples[:6]
        logger.info("SMOKE TEST: %d triples", len(triples))
    else:
        logger.info("FULL RUN: %d triples x up to 3 methods", len(triples))

    BHAT_DIR.mkdir(parents=True, exist_ok=True)
    open(FAILURES_LOG, "w").close()

    n_cpu = os.cpu_count() or 4
    workers = n_workers if n_workers is not None else max(1, min(20, n_cpu - 2))
    logger.info("Using %d worker processes (of %d CPUs)", workers, n_cpu)

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    if _have_tqdm:
        pbar = tqdm(total=len(triples), desc="krr-tuning", unit="triple")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_triple, d, reg, n, p, s, default_mse_lookup): (d, reg, s)
            for d, reg, n, p, s in triples
        }
        for fut in as_completed(futures):
            d, reg, s = futures[fut]
            try:
                rows, failures = fut.result()
            except Exception:
                rows, failures = [], [
                    f"FAILURE  stage=worker  dataset={d}  n_regime={reg}  seed={s}\n"
                    + traceback.format_exc() + "\n"
                ]
            all_rows.extend(rows)
            if failures:
                with open(FAILURES_LOG, "a") as fh:
                    fh.writelines(failures)
            if _have_tqdm:
                pbar.update(1)
                pbar.set_postfix_str(f"{d}/{reg}/seed={s} ({len(rows)}/3 OK)")
            else:
                logger.info("%s/%s seed=%d: %d/3 OK", d, reg, s, len(rows))

    if _have_tqdm:
        pbar.close()

    elapsed = time.time() - t_start
    logger.info("Completed %d rows in %.1fs (%.1f min)", len(all_rows), elapsed, elapsed / 60)

    suffix = "_smoke" if smoke_test else ""
    raw_path      = RESULTS_DIR / f"real_data_tuned_krr{suffix}.csv"
    summary_path  = RESULTS_DIR / f"real_data_tuned_krr_summary{suffix}.csv"
    wilcoxon_path = RESULTS_DIR / f"real_data_tuned_krr_wilcoxon{suffix}.csv"

    _write_csv(all_rows, raw_path)
    summary = aggregate_summary(all_rows)
    _write_csv(summary, summary_path)

    try:
        wilcoxon_rows = run_wilcoxon(all_rows)
        _write_csv(wilcoxon_rows, wilcoxon_path)
    except ImportError:
        logger.warning("scipy not available - skipping Wilcoxon tests")

    print_comparison(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV-tuned KRR robustness check for real-data experiments.")
    parser.add_argument("--smoke-test", action="store_true", help="Only the first 6 triples.")
    parser.add_argument("--workers", type=int, default=None, help="Override worker process count.")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, n_workers=args.workers)
