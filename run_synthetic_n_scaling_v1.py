#!/usr/bin/env python3
"""Friedman-1 n-scaling experiment v1.

Tests the hypothesis that SP-MAVE wins when p/n ≲ 0.25 by varying n
at fixed p values.  The n=100 reference comes from v3_fixed results and
is not rerun here.

Usage
-----
    python run_synthetic_n_scaling_v1.py            # full run
    python run_synthetic_n_scaling_v1.py --smoke-test

Outputs
-------
    results/synthetic_n_scaling_v1.csv
    results/synthetic_n_scaling_v1_summary.csv
    results/synthetic_n_scaling_v1_wilcoxon.csv
    results/failures.log
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
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.evaluation import evaluate_downstream
from sp_mave.mave import ClassicalMAVE
from sp_mave.metrics import compute_all_metrics
from sp_mave.preprocessing import standardize
from sp_mave.spmave import SPMAVE
from sp_mave.utils import get_logger

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

N_VALUES: list[int] = [200, 500]
NOISE_DIMS: list[int] = [40, 80, 120]   # p = D_TRUE + noise_dims → 45, 85, 125
SEEDS: list[int] = list(range(20))

D_TRUE: int = 5
FRIEDMAN_NOISE_STD: float = 1.0
TEST_SIZE: float = 0.2

# k scales with n: floor(0.20 * n), identical for ClassicalMAVE_kNN and SPMAVE
K_SCHEDULE: dict[int, int] = {200: 40, 500: 100}

# SP-MAVE fixed hyperparameters
SPMAVE_TAU: float = 1.0
SPMAVE_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}
SPMAVE_CROSSFIT_CFG: dict[str, Any] = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Shared MAVE model settings
MAVE_MODEL_CFG: dict[str, Any] = {
    "d": D_TRUE,
    "bandwidth": "auto",
    "max_iters": 300,
    "tol": 1e-6,
    "ridge_lambda": 0.01,
    "init_method": "pca",
    "random_state": 42,  # overridden per seed inside run_one_cell
}

DOWNSTREAM_MODELS: list[str] = ["linear", "kernel_ridge", "random_forest"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FAILURES_LOG = RESULTS_DIR / "failures.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("n_scaling_v1")

logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)
logging.getLogger("sp_mave.spmave").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def _true_subspace(p: int) -> np.ndarray:
    """Orthonormal basis for span{e_1,...,e_5} in R^p."""
    return np.eye(p, D_TRUE)


def run_one_cell(n: int, noise_dims: int, seed: int) -> list[dict[str, Any]]:
    """Fit all three methods for one (n, noise_dims, seed) cell."""
    np.random.seed(seed)

    p = D_TRUE + noise_dims
    k = K_SCHEDULE[n]

    X, y = make_friedman1(
        n_samples=n,
        n_features=p,
        noise=FRIEDMAN_NOISE_STD,
        random_state=seed,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )
    X_tr, X_te, _ = standardize(X_tr, X_te)

    true_sub = _true_subspace(p)
    model_cfg = {**MAVE_MODEL_CFG, "random_state": seed}

    rows: list[dict[str, Any]] = []

    # ---- Method 1: ClassicalMAVE ----------------------------------------
    try:
        t0 = time.time()
        est = ClassicalMAVE(
            d=model_cfg["d"],
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
            method="ClassicalMAVE",
            est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub,
            n_obs=n, noise_dims=noise_dims, p=p, seed=seed,
            fit_time=fit_time, fixed_k=float("nan"),
        ))
    except Exception:
        _log_failure("ClassicalMAVE", n, noise_dims, seed)

    # ---- Method 2: ClassicalMAVE_kNN (k = K_SCHEDULE[n]) ----------------
    try:
        t0 = time.time()
        est = ClassicalMAVE(
            d=model_cfg["d"],
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
            method="ClassicalMAVE_kNN",
            est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub,
            n_obs=n, noise_dims=noise_dims, p=p, seed=seed,
            fit_time=fit_time, fixed_k=k,
        ))
    except Exception:
        _log_failure("ClassicalMAVE_kNN", n, noise_dims, seed)

    # ---- Method 3: SPMAVE -----------------------------------------------
    try:
        crossfit_cfg = {**SPMAVE_CROSSFIT_CFG, "random_state": seed}
        rf_params = {**SPMAVE_RF_PARAMS, "random_state": seed}
        t0 = time.time()
        est = SPMAVE(
            d=model_cfg["d"],
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
            method="SPMAVE",
            est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub,
            n_obs=n, noise_dims=noise_dims, p=p, seed=seed,
            fit_time=fit_time, fixed_k=float("nan"),
        ))
    except Exception:
        _log_failure("SPMAVE", n, noise_dims, seed)

    return rows


def _build_row(
    method: str,
    est: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    true_sub: np.ndarray,
    n_obs: int,
    noise_dims: int,
    p: int,
    seed: int,
    fit_time: float,
    fixed_k: float,
) -> dict[str, Any]:
    result = est.result_
    assert result is not None

    sub_metrics = compute_all_metrics(result.B, true_sub)

    row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "n_obs": n_obs,
        "noise_dims": noise_dims,
        "p_features": p,
        "fixed_k": fixed_k,
        "fit_time_s": round(fit_time, 4),
        "n_iters": result.n_iters,
        "final_loss": result.loss_history[-1] if result.loss_history else float("nan"),
        **sub_metrics,
    }

    for ds in DOWNSTREAM_MODELS:
        ds_out = evaluate_downstream(
            B=result.B,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            model_type=ds,
        )
        row[f"test_mse_{ds}"] = ds_out["test_mse"]
        row[f"train_mse_{ds}"] = ds_out["train_mse"]

    return row


def _log_failure(method: str, n: int, noise_dims: int, seed: int) -> None:
    msg = (
        f"FAILURE  method={method}  n={n}  noise_dims={noise_dims}  seed={seed}\n"
        + traceback.format_exc()
        + "\n"
    )
    logger.error(
        "Cell failed: method=%s  n=%d  noise_dims=%d  seed=%d",
        method, n, noise_dims, seed,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FAILURES_LOG, "a") as fh:
        fh.write(msg)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    "grassmann_distance",
    "projection_distance",
    "vector_correlation",
    "fit_time_s",
    "n_iters",
    "final_loss",
] + [f"test_mse_{ds}" for ds in DOWNSTREAM_MODELS] + [f"train_mse_{ds}" for ds in DOWNSTREAM_MODELS]


def aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and standard error per (method, n_obs, p_features)."""
    methods = sorted({r["method"] for r in raw_rows})
    n_values = sorted({r["n_obs"] for r in raw_rows})
    p_values = sorted({r["p_features"] for r in raw_rows})
    agg: list[dict[str, Any]] = []

    for m in methods:
        for n in n_values:
            for p in p_values:
                subset = [
                    r for r in raw_rows
                    if r["method"] == m and r["n_obs"] == n and r["p_features"] == p
                ]
                if not subset:
                    continue
                entry: dict[str, Any] = {
                    "method": m, "n_obs": n, "p_features": p, "n_seeds": len(subset),
                }
                for col in _METRIC_COLS:
                    vals = [float(r[col]) for r in subset if col in r and not _is_nan(r[col])]
                    if vals:
                        mean_v = float(np.mean(vals))
                        se_v = (
                            float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                            if len(vals) > 1 else float("nan")
                        )
                        entry[f"{col}_mean"] = mean_v
                        entry[f"{col}_se"] = se_v
                agg.append(entry)

    return agg


def _is_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Wilcoxon tests
# ---------------------------------------------------------------------------

def run_wilcoxon(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired Wilcoxon signed-rank: SPMAVE vs each baseline per (n_obs, p_features).

    rank_biserial > 0 means SPMAVE tends to have smaller values than baseline.
    """
    from scipy.stats import wilcoxon

    n_values = sorted({r["n_obs"] for r in raw_rows})
    p_values = sorted({r["p_features"] for r in raw_rows})
    baselines = ["ClassicalMAVE", "ClassicalMAVE_kNN"]
    test_metrics = ["grassmann_distance", "test_mse_kernel_ridge"]
    results: list[dict[str, Any]] = []

    for n in n_values:
        for p in p_values:
            sp_rows = {
                r["seed"]: r for r in raw_rows
                if r["method"] == "SPMAVE" and r["n_obs"] == n and r["p_features"] == p
            }
            for bl in baselines:
                bl_rows = {
                    r["seed"]: r for r in raw_rows
                    if r["method"] == bl and r["n_obs"] == n and r["p_features"] == p
                }
                common_seeds = sorted(sp_rows.keys() & bl_rows.keys())
                if len(common_seeds) < 10:
                    logger.warning(
                        "Skipping Wilcoxon n=%d p=%d bl=%s: only %d paired seeds",
                        n, p, bl, len(common_seeds),
                    )
                    continue
                for metric in test_metrics:
                    sp_vals = np.array([sp_rows[s][metric] for s in common_seeds], dtype=float)
                    bl_vals = np.array([bl_rows[s][metric] for s in common_seeds], dtype=float)
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
                    if n_nz > 0 and not _is_nan(stat):
                        r_rb = 1.0 - 2.0 * stat / (n_nz * (n_nz + 1) / 2.0)
                    else:
                        r_rb = float("nan")
                    results.append({
                        "n_obs": n,
                        "p_features": p,
                        "baseline": bl,
                        "metric": metric,
                        "n_pairs": len(common_seeds),
                        "spmave_mean": float(np.mean(sp_vals)),
                        "baseline_mean": float(np.mean(bl_vals)),
                        "wilcoxon_statistic": stat,
                        "p_value": pval,
                        "rank_biserial": r_rb,
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

    n_values = [200] if smoke_test else N_VALUES
    noise_dims = [40, 80] if smoke_test else NOISE_DIMS
    seeds = list(range(2)) if smoke_test else SEEDS

    if smoke_test:
        logger.info("SMOKE TEST: n=%s  noise_dims=%s  seeds=%s", n_values, noise_dims, seeds)
    else:
        logger.info(
            "FULL RUN: %d n-values × %d p-values × %d seeds = %d cells, 3 methods each",
            len(n_values), len(noise_dims), len(seeds),
            len(n_values) * len(noise_dims) * len(seeds),
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    open(FAILURES_LOG, "w").close()

    cells = [(n, nd, s) for n in n_values for nd in noise_dims for s in seeds]
    total_methods = len(cells) * 3

    if _have_tqdm:
        pbar = tqdm(total=total_methods, desc="n-scaling-v1", unit="fit")

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for n, nd, seed in cells:
        t_cell = time.time()
        cell_rows = run_one_cell(n, nd, seed)
        elapsed = time.time() - t_cell
        all_rows.extend(cell_rows)
        msg = f"n={n:3d}  noise_dims={nd:3d}  seed={seed:2d}  {len(cell_rows)}/3 OK  ({elapsed:.1f}s)"
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
    raw_path = RESULTS_DIR / f"synthetic_n_scaling_v1{suffix}.csv"
    summary_path = RESULTS_DIR / f"synthetic_n_scaling_v1_summary{suffix}.csv"
    wilcoxon_path = RESULTS_DIR / f"synthetic_n_scaling_v1_wilcoxon{suffix}.csv"

    _write_csv(all_rows, raw_path)

    summary = aggregate_summary(all_rows)
    _write_csv(summary, summary_path)

    try:
        wilcoxon_rows = run_wilcoxon(all_rows)
        _write_csv(wilcoxon_rows, wilcoxon_path)
    except ImportError:
        logger.warning("scipy not available — skipping Wilcoxon tests")

    _print_summary(summary, n_values, noise_dims)
    _print_pn_table(summary, n_values, noise_dims)


def _print_summary(
    summary: list[dict[str, Any]],
    n_values: list[int],
    noise_dims: list[int],
) -> None:
    methods = sorted({r["method"] for r in summary})
    print("\n" + "=" * 100)
    print("N-SCALING v1 — SUMMARY  (mean ± SE over seeds)")
    print("=" * 100)
    for n in n_values:
        for nd in noise_dims:
            p = D_TRUE + nd
            print(f"\n  n={n}  noise_dims={nd}  (p={p}  p/n={p/n:.3f})")
            print(f"  {'Method':<22s}  {'Grass dist':>20s}  {'Proj dist':>20s}  {'Vec corr':>14s}  {'MSE(linear)':>18s}")
            print("  " + "-" * 96)
            for m in methods:
                row = next(
                    (r for r in summary if r["method"] == m and r["n_obs"] == n and r["p_features"] == p),
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

                print(
                    f"  {m:<22s}  {_fmt('grassmann_distance')}  "
                    f"{_fmt('projection_distance')}  {_fmt('vector_correlation'):>14s}  "
                    f"{_fmt('test_mse_linear'):>18s}"
                )
    print("\n" + "=" * 100 + "\n")


def _print_pn_table(
    summary: list[dict[str, Any]],
    n_values: list[int],
    noise_dims: list[int],
) -> None:
    print("p/n ratio table — SP-MAVE Grassmann < kNN-MAVE Grassmann?")
    print(f"  {'n_obs':>6}  {'p':>5}  {'p/n':>6}  {'SPMAVE':>10}  {'kNN-MAVE':>10}  {'SP wins?':>9}")
    print("  " + "-" * 54)
    for n in n_values:
        for nd in noise_dims:
            p = D_TRUE + nd
            sp_row = next(
                (r for r in summary if r["method"] == "SPMAVE" and r["n_obs"] == n and r["p_features"] == p),
                None,
            )
            knn_row = next(
                (r for r in summary if r["method"] == "ClassicalMAVE_kNN" and r["n_obs"] == n and r["p_features"] == p),
                None,
            )
            if sp_row is None or knn_row is None:
                continue
            sp_g = sp_row.get("grassmann_distance_mean", float("nan"))
            knn_g = knn_row.get("grassmann_distance_mean", float("nan"))
            wins = "Y" if (not _is_nan(sp_g) and not _is_nan(knn_g) and sp_g < knn_g) else "N"
            print(
                f"  {n:>6}  {p:>5}  {p/n:>6.3f}  {sp_g:>10.4f}  {knn_g:>10.4f}  {wins:>9}"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Friedman-1 n-scaling experiment v1."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: n=[200], 2 noise levels, 2 seeds only.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
