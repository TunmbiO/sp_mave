#!/usr/bin/env python3
"""Genuine subspace-recovery synthetic experiments v1 (non-axis-aligned DGPs).

Motivation
----------
run_synthetic_unified_v1.py uses Friedman-1, whose true subspace is
span{e_1,...,e_5} — axis-aligned. A method that recovers the five active
variables scores well on Grassmann distance without demonstrating genuine
sufficient dimension reduction (i.e. recovery of a rotated B). This script
adds three DGPs with non-axis-aligned true subspaces:

  1. single_index_rotated — r=1, B = (1/sqrt(p)) * ones(p). No sparse
     structure at all; a variable selector cannot recover it.
  2. li1991_model1        — r=2, B = [b1, b2], first two columns of a
     random orthonormal p x p matrix Q (Li 1991 Model 1 / Xia et al. 2002
     canonical SDR benchmark).
  3. multi_index_random   — r=3, B = first three columns of the same Q.

Q is generated via QR of a Gaussian matrix seeded at 0, once per p value —
so B is identical across all 20 seeds and all three n values; only X, y,
and the train/test split vary with seed. DGP2 and DGP3 share the same Q(p)
(DGP3's B is a superset of DGP2's first two columns).

No standardisation: X ~ N(0, I_p) is already unit-scale by construction.
Per-feature StandardScaler rescaling by the empirical train-sample std
would rotate the effective subspace slightly away from the true B at
finite n — exactly the kind of coordinate-frame artifact this experiment
is designed to eliminate. (This is safe to skip here; it was harmless for
axis-aligned Friedman-1 but is NOT harmless for these rotated DGPs.)

Same three-method infrastructure as unified_v1, plus a fourth baseline,
L1MAVE (Laplacian / L1-distance kernel, top-k truncated).

Usage
-----
    python run_synthetic_dgp_v1.py              # full run
    python run_synthetic_dgp_v1.py --smoke-test

Outputs
-------
    results/synthetic_dgp_v1.csv
    results/synthetic_dgp_v1_summary.csv
    results/synthetic_dgp_v1_wilcoxon.csv
    results/synthetic_dgp_v1_failures.log
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
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.evaluation import evaluate_downstream
from sp_mave.mave import ClassicalMAVE, L1MAVE
from sp_mave.metrics import compute_all_metrics
from sp_mave.spmave import SPMAVE
from sp_mave.utils import get_logger

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

N_VALUES: list[int] = [100, 200, 500]
P_VALUES: list[int] = [15, 25, 45, 85, 125]
SEEDS: list[int] = list(range(20))

DGP_NAMES: list[str] = ["single_index_rotated", "li1991_model1", "multi_index_random"]
R_TRUE: dict[str, int] = {
    "single_index_rotated": 1,
    "li1991_model1": 2,
    "multi_index_random": 3,
}
B_SEED: int = 0
TEST_SIZE: float = 0.2

# k = floor(K_FRACTION * n) — total n (pre-split), matching unified_v1's
# literal behaviour (20 / 40 / 100 at n=100/200/500). Same k for
# ClassicalMAVE_kNN, SPMAVE, and L1MAVE.
K_FRACTION: float = 0.20

# SP-MAVE fixed hyperparameters (identical to unified_v1)
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

# Shared MAVE model settings (d is overridden per DGP to r_true)
MAVE_MODEL_CFG: dict[str, Any] = {
    "bandwidth": "auto",
    "max_iters": 300,
    "tol": 1e-6,
    "ridge_lambda": 0.01,
    "init_method": "pca",
}

DOWNSTREAM_MODELS: list[str] = ["linear", "kernel_ridge", "random_forest"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FAILURES_LOG = RESULTS_DIR / "synthetic_dgp_v1_failures.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = get_logger("dgp_v1")

logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)
logging.getLogger("sp_mave.spmave").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# B(p) cache — Q via QR of a Gaussian matrix seeded at 0, once per p
# ---------------------------------------------------------------------------

_Q_CACHE: dict[int, np.ndarray] = {}


def _get_Q(p: int) -> np.ndarray:
    """Return the p x p orthonormal Q for this p, generated once (seed=0)."""
    if p not in _Q_CACHE:
        rng_B = np.random.default_rng(B_SEED)
        G = rng_B.standard_normal((p, p))
        Q, _ = np.linalg.qr(G)
        _Q_CACHE[p] = Q
    return _Q_CACHE[p]


def true_subspace(dgp_name: str, p: int) -> np.ndarray:
    """Return the (p, r_true) true orthonormal basis for this DGP and p."""
    if dgp_name == "single_index_rotated":
        b = np.ones((p, 1)) / math.sqrt(p)
        return b
    r = R_TRUE[dgp_name]
    Q = _get_Q(p)
    return Q[:, :r]


# ---------------------------------------------------------------------------
# DGP generators
# ---------------------------------------------------------------------------

def _gen_single_index_rotated(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    b = np.ones(p) / math.sqrt(p)
    noise = rng.standard_normal(n)
    y = np.sin(X @ b) + 0.3 * noise
    return X, y


def _gen_li1991_model1(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Q = _get_Q(p)
    b1, b2 = Q[:, 0], Q[:, 1]
    noise = rng.standard_normal(n)
    u1 = X @ b1
    u2 = X @ b2
    y = u1 / (0.5 + (u2 + 1.5) ** 2) + 0.5 * noise
    return X, y


def _gen_multi_index_random(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Q = _get_Q(p)
    b1, b2, b3 = Q[:, 0], Q[:, 1], Q[:, 2]
    noise = rng.standard_normal(n)
    u1 = X @ b1
    u2 = X @ b2
    u3 = X @ b3
    y = u1 ** 2 + u1 * u2 + 0.5 * u3 + 0.3 * noise
    return X, y


DGP_GENERATORS: dict[str, Callable[[int, int, int], tuple[np.ndarray, np.ndarray]]] = {
    "single_index_rotated": _gen_single_index_rotated,
    "li1991_model1": _gen_li1991_model1,
    "multi_index_random": _gen_multi_index_random,
}


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_one_cell(dgp_name: str, n: int, p: int, seed: int) -> list[dict[str, Any]]:
    """Fit all four methods for one (dgp_name, n, p, seed) cell."""
    r_true = R_TRUE[dgp_name]
    k = max(1, math.floor(K_FRACTION * n))

    X, y = DGP_GENERATORS[dgp_name](n, p, seed)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed
    )
    # No standardisation: X ~ N(0, I_p) already unit-scale; rescaling by
    # empirical train-sample std would distort the non-axis-aligned B_true.

    true_sub = true_subspace(dgp_name, p)
    cfg = MAVE_MODEL_CFG

    rows: list[dict[str, Any]] = []

    # ---- Method 1: ClassicalMAVE (full Gaussian, no truncation) ----------
    try:
        t0 = time.time()
        est = ClassicalMAVE(
            d=r_true,
            bandwidth=cfg["bandwidth"],
            k=None,
            max_iters=cfg["max_iters"],
            tol=cfg["tol"],
            ridge_lambda=cfg["ridge_lambda"],
            init_method=cfg["init_method"],
            random_state=seed,
        )
        est.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        rows.append(_build_row(
            dgp_name=dgp_name, method="ClassicalMAVE", est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub, r_true=r_true,
            n_obs=n, p=p, seed=seed,
            fit_time=fit_time, fixed_k=float("nan"),
        ))
    except Exception:
        _log_failure("ClassicalMAVE", dgp_name, n, p, seed)

    # ---- Method 2: ClassicalMAVE_kNN --------------------------------------
    try:
        t0 = time.time()
        est = ClassicalMAVE(
            d=r_true,
            bandwidth=cfg["bandwidth"],
            k=k,
            max_iters=cfg["max_iters"],
            tol=cfg["tol"],
            ridge_lambda=cfg["ridge_lambda"],
            init_method=cfg["init_method"],
            random_state=seed,
        )
        est.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        rows.append(_build_row(
            dgp_name=dgp_name, method="ClassicalMAVE_kNN", est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub, r_true=r_true,
            n_obs=n, p=p, seed=seed,
            fit_time=fit_time, fixed_k=k,
        ))
    except Exception:
        _log_failure("ClassicalMAVE_kNN", dgp_name, n, p, seed)

    # ---- Method 3: SPMAVE -------------------------------------------------
    try:
        crossfit_cfg = {**SPMAVE_CROSSFIT_CFG, "random_state": seed}
        rf_params = {**SPMAVE_RF_PARAMS, "random_state": seed}
        t0 = time.time()
        est = SPMAVE(
            d=r_true,
            k=k,
            tau=SPMAVE_TAU,
            rf_params=rf_params,
            crossfit_cfg=crossfit_cfg,
            max_iters=cfg["max_iters"],
            tol=cfg["tol"],
            ridge_lambda=cfg["ridge_lambda"],
            init_method=cfg["init_method"],
            random_state=seed,
        )
        est.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        rows.append(_build_row(
            dgp_name=dgp_name, method="SPMAVE", est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub, r_true=r_true,
            n_obs=n, p=p, seed=seed,
            fit_time=fit_time, fixed_k=float("nan"),
        ))
    except Exception:
        _log_failure("SPMAVE", dgp_name, n, p, seed)

    # ---- Method 4: L1MAVE -------------------------------------------------
    try:
        t0 = time.time()
        est = L1MAVE(
            d=r_true,
            bandwidth=cfg["bandwidth"],
            k=k,
            max_iters=cfg["max_iters"],
            tol=cfg["tol"],
            ridge_lambda=cfg["ridge_lambda"],
            init_method=cfg["init_method"],
            random_state=seed,
        )
        est.fit(X_tr, y_tr)
        fit_time = time.time() - t0
        rows.append(_build_row(
            dgp_name=dgp_name, method="L1MAVE", est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub, r_true=r_true,
            n_obs=n, p=p, seed=seed,
            fit_time=fit_time, fixed_k=k,
        ))
    except Exception:
        _log_failure("L1MAVE", dgp_name, n, p, seed)

    return rows


def _build_row(
    dgp_name: str,
    method: str,
    est: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    true_sub: np.ndarray,
    r_true: int,
    n_obs: int,
    p: int,
    seed: int,
    fit_time: float,
    fixed_k: float,
) -> dict[str, Any]:
    result = est.result_
    assert result is not None

    sub_metrics = compute_all_metrics(result.B, true_sub)

    row: dict[str, Any] = {
        "dgp_name": dgp_name,
        "method": method,
        "seed": seed,
        "n_obs": n_obs,
        "p_features": p,
        "r_true": r_true,
        "b_seed": B_SEED,
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


def _log_failure(method: str, dgp_name: str, n: int, p: int, seed: int) -> None:
    msg = (
        f"FAILURE  method={method}  dgp={dgp_name}  n={n}  p={p}  seed={seed}\n"
        + traceback.format_exc()
        + "\n"
    )
    logger.error(
        "Cell failed: method=%s  dgp=%s  n=%d  p=%d  seed=%d",
        method, dgp_name, n, p, seed,
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
    """Mean and standard error per (dgp_name, method, n_obs, p_features)."""
    dgp_names = sorted({r["dgp_name"] for r in raw_rows})
    methods = sorted({r["method"] for r in raw_rows})
    n_values = sorted({r["n_obs"] for r in raw_rows})
    p_values = sorted({r["p_features"] for r in raw_rows})
    agg: list[dict[str, Any]] = []

    for dgp in dgp_names:
        for m in methods:
            for n in n_values:
                for p in p_values:
                    subset = [
                        r for r in raw_rows
                        if r["dgp_name"] == dgp and r["method"] == m
                        and r["n_obs"] == n and r["p_features"] == p
                    ]
                    if not subset:
                        continue
                    entry: dict[str, Any] = {
                        "dgp_name": dgp, "method": m, "n_obs": n, "p_features": p,
                        "n_seeds": len(subset),
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
    """Paired Wilcoxon signed-rank: SPMAVE vs each baseline per (dgp, n_obs, p_features).

    rank_biserial > 0 means SPMAVE tends to have smaller values than baseline.
    """
    from scipy.stats import wilcoxon

    dgp_names = sorted({r["dgp_name"] for r in raw_rows})
    n_values = sorted({r["n_obs"] for r in raw_rows})
    p_values = sorted({r["p_features"] for r in raw_rows})
    baselines = ["ClassicalMAVE", "ClassicalMAVE_kNN", "L1MAVE"]
    test_metrics = ["grassmann_distance", "test_mse_kernel_ridge"]
    results: list[dict[str, Any]] = []

    for dgp in dgp_names:
        for n in n_values:
            for p in p_values:
                sp_rows = {
                    r["seed"]: r for r in raw_rows
                    if r["dgp_name"] == dgp and r["method"] == "SPMAVE"
                    and r["n_obs"] == n and r["p_features"] == p
                }
                for bl in baselines:
                    bl_rows = {
                        r["seed"]: r for r in raw_rows
                        if r["dgp_name"] == dgp and r["method"] == bl
                        and r["n_obs"] == n and r["p_features"] == p
                    }
                    common_seeds = sorted(sp_rows.keys() & bl_rows.keys())
                    if len(common_seeds) < 10:
                        logger.warning(
                            "Skipping Wilcoxon dgp=%s n=%d p=%d bl=%s: only %d paired seeds",
                            dgp, n, p, bl, len(common_seeds),
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
                            "dgp_name": dgp,
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

    dgp_names = DGP_NAMES
    n_values = [100, 200] if smoke_test else N_VALUES
    p_values = [15, 45] if smoke_test else P_VALUES
    seeds = list(range(2)) if smoke_test else SEEDS

    if smoke_test:
        logger.info(
            "SMOKE TEST: dgps=%s  n=%s  p=%s  seeds=%s",
            dgp_names, n_values, p_values, seeds,
        )
    else:
        logger.info(
            "FULL RUN: %d dgps × %d n-values × %d p-values × %d seeds = %d cells, 4 methods each",
            len(dgp_names), len(n_values), len(p_values), len(seeds),
            len(dgp_names) * len(n_values) * len(p_values) * len(seeds),
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    open(FAILURES_LOG, "w").close()

    cells = [
        (dgp, n, p, s)
        for dgp in dgp_names for n in n_values for p in p_values for s in seeds
    ]
    total_methods = len(cells) * 4

    if _have_tqdm:
        pbar = tqdm(total=total_methods, desc="dgp-v1", unit="fit")

    all_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for dgp, n, p, seed in cells:
        t_cell = time.time()
        cell_rows = run_one_cell(dgp, n, p, seed)
        elapsed = time.time() - t_cell
        all_rows.extend(cell_rows)
        msg = (
            f"dgp={dgp:<22s}  n={n:3d}  p={p:3d}  seed={seed:2d}  "
            f"{len(cell_rows)}/4 OK  ({elapsed:.1f}s)"
        )
        if _have_tqdm:
            pbar.update(4)
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
    raw_path      = RESULTS_DIR / f"synthetic_dgp_v1{suffix}.csv"
    summary_path  = RESULTS_DIR / f"synthetic_dgp_v1_summary{suffix}.csv"
    wilcoxon_path = RESULTS_DIR / f"synthetic_dgp_v1_wilcoxon{suffix}.csv"

    _write_csv(all_rows, raw_path)

    summary = aggregate_summary(all_rows)
    _write_csv(summary, summary_path)

    try:
        wilcoxon_rows = run_wilcoxon(all_rows)
        _write_csv(wilcoxon_rows, wilcoxon_path)
    except ImportError:
        logger.warning("scipy not available — skipping Wilcoxon tests")

    _print_pn_table(summary, dgp_names, n_values, p_values)


def _print_pn_table(
    summary: list[dict[str, Any]],
    dgp_names: list[str],
    n_values: list[int],
    p_values: list[int],
) -> None:
    print("\n" + "=" * 110)
    print("DGP v1 — p/n regime table: SP-MAVE vs kNN-MAVE vs L1-MAVE (Grassmann distance)")
    print("=" * 110)
    header = (
        f"  {'DGP':<22}  {'n':>5}  {'p':>5}  {'p/n':>6}  "
        f"{'SPMAVE':>10}  {'kNN':>10}  {'L1':>10}  {'vs kNN?':>8}  {'vs L1?':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    def _row(dgp: str, method: str, n: int, p: int) -> dict[str, Any] | None:
        return next(
            (r for r in summary if r["dgp_name"] == dgp and r["method"] == method
             and r["n_obs"] == n and r["p_features"] == p),
            None,
        )

    for dgp in dgp_names:
        for n in n_values:
            for p in p_values:
                sp = _row(dgp, "SPMAVE", n, p)
                knn = _row(dgp, "ClassicalMAVE_kNN", n, p)
                l1 = _row(dgp, "L1MAVE", n, p)
                if sp is None or knn is None or l1 is None:
                    continue
                sp_g = sp.get("grassmann_distance_mean", float("nan"))
                knn_g = knn.get("grassmann_distance_mean", float("nan"))
                l1_g = l1.get("grassmann_distance_mean", float("nan"))
                wins_knn = "Y" if (not _is_nan(sp_g) and not _is_nan(knn_g) and sp_g < knn_g) else "N"
                wins_l1 = "Y" if (not _is_nan(sp_g) and not _is_nan(l1_g) and sp_g < l1_g) else "N"
                print(
                    f"  {dgp:<22}  {n:>5}  {p:>5}  {p/n:>6.3f}  "
                    f"{sp_g:>10.4f}  {knn_g:>10.4f}  {l1_g:>10.4f}  "
                    f"{wins_knn:>8}  {wins_l1:>7}"
                )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Non-axis-aligned subspace-recovery synthetic experiments v1."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: n=[100,200], p=[15,45], 2 seeds only.",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
