#!/usr/bin/env python3
"""Friedman-only OPG weight-scheme comparison (Gaussian / kNN / RF proximity).

Four estimators share the same OPG backbone in ``sp_mave.opg``; only weight
construction differs.  Outputs go under ``results/opg/``.

Usage
-----
    .venv/bin/python run_opg_experiment.py
    .venv/bin/python run_opg_experiment.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.datasets import load_friedman1
from sp_mave.evaluation import evaluate_downstream
from sp_mave.metrics import compute_all_metrics
from sp_mave.opg import (
    GaussianOPG,
    RFProximityOPG,
    TopKGaussianOPG,
    UniformKNNOPG,
)
from sp_mave.plots import plot_noise_scaling_curves, plot_weight_histograms
from sp_mave.preprocessing import standardize
from sp_mave.types import MAVEResult

RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "opg"

NOISE_DIMS_DEFAULT = [0, 10, 20, 40, 80, 120]
N_SEEDS_DEFAULT = 20
N_SAMPLES = 500 #100
TEST_SIZE = 0.3
D = 5
K_NN = 20
TAU_RF = 1.0
MAX_ITERS = 50
RIDGE = 0.01

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
}
CROSSFIT_CFG: dict[str, Any] = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}


def _assert_column_stochastic(W: np.ndarray, ctx: str) -> None:
    col_sums = W.sum(axis=0)
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"{ctx}: W not square, shape={W.shape}")
    if np.any(W < -1e-12):
        raise ValueError(f"{ctx}: negative weights")
    if not np.allclose(col_sums, 1.0, atol=1e-4):
        bad = np.where(~np.isclose(col_sums, 1.0, atol=1e-4))[0]
        raise ValueError(
            f"{ctx}: columns do not sum to 1 (e.g. col {bad[:5]}...)"
        )


def _make_estimators(seed: int) -> list[Any]:
    common = dict(
        d=D,
        max_iters=MAX_ITERS,
        tol=1e-6,
        ridge_lambda=RIDGE,
        init_method="pca",
        random_state=seed,
    )
    return [
        GaussianOPG(bandwidth="auto", **common),
        TopKGaussianOPG(k=K_NN, bandwidth="auto", **common),
        UniformKNNOPG(k=K_NN, **common),
        RFProximityOPG(
            k=K_NN,
            tau=TAU_RF,
            rf_params={**RF_PARAMS, "random_state": seed},
            crossfit_cfg={**CROSSFIT_CFG, "random_state": seed},
            **common,
        ),
    ]


def _aggregate(
    raw: list[dict[str, Any]],
    noise_dims: list[int],
    methods: list[str],
) -> list[dict[str, Any]]:
    metrics = [
        "projection_distance",
        "grassmann_distance",
        "vector_correlation",
        "test_mse_linear",
        "test_mse_random_forest",
        "fit_time_s",
        "n_iters",
    ]
    out: list[dict[str, Any]] = []
    for nd in noise_dims:
        for m in methods:
            rows = [r for r in raw if r["noise_dims"] == nd and r["method"] == m]
            if not rows:
                continue
            entry: dict[str, Any] = {
                "noise_dims": nd,
                "p_features": 5 + nd,
                "method": m,
                "n_seeds": len(rows),
            }
            for key in metrics:
                vals = [float(r[key]) for r in rows if key in r]
                if vals:
                    entry[f"{key}_mean"] = float(np.mean(vals))
                    entry[f"{key}_std"] = float(np.std(vals))
            out.append(entry)
    out.sort(key=lambda x: (x["noise_dims"], x["method"]))
    return out


def _plot_weight_diagnostics_at_noise(
    noise_dims: int,
    seed: int,
    fig_path: Path,
) -> None:
    bundle = load_friedman1(
        n_samples=N_SAMPLES,
        n_noise=noise_dims,
        noise=1.0,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=TEST_SIZE,
        random_state=seed,
    )
    X_train, X_test, _ = standardize(X_train, X_test)
    results: dict[str, MAVEResult] = {}
    for est in _make_estimators(seed):
        est.fit(X_train, y_train)
        assert est.result_ is not None
        results[est.result_.method_name] = est.result_
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_weight_histograms(results, save_path=fig_path)


def _print_final_report(
    agg: list[dict[str, Any]],
    noise_dims: list[int],
    methods: list[str],
) -> None:
    print("\n" + "=" * 90)
    print("OPG WEIGHT-SCHEME EXPERIMENT — FINAL REPORT (Friedman-1)")
    print("=" * 90)

    def mean_over_noise(method: str, key_mean: str) -> float:
        vals = [
            float(r[key_mean])
            for r in agg
            if r["method"] == method and key_mean in r
        ]
        return float(np.mean(vals)) if vals else float("nan")

    # 1) Best average subspace recovery (lower grassmann, higher vec corr)
    grass_rank = sorted(
        methods,
        key=lambda m: mean_over_noise(m, "grassmann_distance_mean"),
    )
    vec_rank = sorted(
        methods,
        key=lambda m: -mean_over_noise(m, "vector_correlation_mean"),
    )
    print("\n1) Subspace recovery (averaged over noise levels, then over seeds):")
    print(f"   Best Grassmann distance (lowest): {grass_rank[0]}")
    print(f"   Best vector correlation (highest): {vec_rank[0]}")

    mse_rank = sorted(
        methods,
        key=lambda m: mean_over_noise(m, "test_mse_linear_mean"),
    )
    print("\n2) Prediction — linear downstream MSE (mean over noise × seeds):")
    for m in mse_rank:
        v = mean_over_noise(m, "test_mse_linear_mean")
        print(f"   {m:<22s}  {v:.4f}")
    print(f"   Best (lowest MSE): {mse_rank[0]}")

    mse_rf_rank = sorted(
        methods,
        key=lambda m: mean_over_noise(m, "test_mse_random_forest_mean"),
    )
    print("\n   RandomForest downstream MSE (mean over noise × seeds):")
    for m in mse_rf_rank:
        v = mean_over_noise(m, "test_mse_random_forest_mean")
        print(f"   {m:<22s}  {v:.4f}")
    print(f"   Best (lowest MSE): {mse_rf_rank[0]}")

    # 3) RF trend vs noise
    rf = "RFProximityOPG"
    lin_g = "GaussianOPG"
    tkg = [r for r in agg if r["method"] == rf]
    tkg.sort(key=lambda x: x["noise_dims"])
    if len(tkg) >= 2:
        mses = [r["test_mse_linear_mean"] for r in tkg]
        low, high = mses[0], mses[-1]
        mid = mses[len(mses) // 2]
        print("\n3) RFProximityOPG vs noise (linear test MSE mean):")
        print(
            f"   noise={tkg[0]['noise_dims']}: {low:.4f}  →  "
            f"mid≈{mid:.4f}  →  noise={tkg[-1]['noise_dims']}: {high:.4f}"
        )
        if high < low * 0.95:
            print("   Trend: improves vs low-noise endpoint at high noise.")
        elif high > low * 1.05:
            print("   Trend: degrades at high noise vs low-noise endpoint.")
        else:
            print("   Trend: relatively stable across noise levels.")

    # 4) Pairwise contrasts (mean linear MSE over noise)
    def avg_mse(a: str) -> float:
        return mean_over_noise(a, "test_mse_linear_mean")

    print("\n4) Pairwise contrasts (mean linear test MSE over noise levels):")
    g, tg, u, r = (
        "GaussianOPG",
        f"TopKGaussianOPG_k{K_NN}",
        f"UniformKNNOPG_k{K_NN}",
        "RFProximityOPG",
    )
    print(
        f"   Gaussian vs Top-k Gaussian:  {avg_mse(g):.4f} vs {avg_mse(tg):.4f}  "
        f"(truncation Δ = {avg_mse(tg) - avg_mse(g):+.4f})"
    )
    print(
        f"   Top-k Gaussian vs Uniform kNN: {avg_mse(tg):.4f} vs {avg_mse(u):.4f}  "
        f"(kernel shape Δ = {avg_mse(u) - avg_mse(tg):+.4f})"
    )
    print(
        f"   Uniform kNN vs RF proximity: {avg_mse(u):.4f} vs {avg_mse(r):.4f}  "
        f"(supervision Δ = {avg_mse(r) - avg_mse(u):+.4f})"
    )
    print("\n" + "=" * 90 + "\n")


def run_experiment(quick: bool = False) -> None:
    noise_dims = [0, 10] if quick else NOISE_DIMS_DEFAULT
    seeds = [0, 1] if quick else list(range(N_SEEDS_DEFAULT))

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_ROOT / "experiment.log"
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_log.addHandler(fh)

    raw_rows: list[dict[str, Any]] = []
    t0 = time.time()

    for nd in noise_dims:
        for seed in seeds:
            bundle = load_friedman1(
                n_samples=N_SAMPLES,
                n_noise=nd,
                noise=1.0,
                random_state=seed,
            )
            assert bundle.true_subspace is not None
            X_train, X_test, y_train, y_test = train_test_split(
                bundle.X,
                bundle.y,
                test_size=TEST_SIZE,
                random_state=seed,
            )
            X_train, X_test, _ = standardize(X_train, X_test)

            for est in _make_estimators(seed):
                t_fit = time.time()
                est.fit(X_train, y_train)
                fit_time = time.time() - t_fit
                res = est.result_
                assert res is not None
                _assert_column_stochastic(res.weights, res.method_name)

                B = res.B
                sub = compute_all_metrics(B, bundle.true_subspace)
                row: dict[str, Any] = {
                    "noise_dims": nd,
                    "p_features": 5 + nd,
                    "seed": seed,
                    "method": res.method_name,
                    "n_iters": res.n_iters,
                    "fit_time_s": round(fit_time, 4),
                    **sub,
                }
                for model_type in ("linear", "random_forest"):
                    ev = evaluate_downstream(
                        B=B,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        model_type=model_type,
                    )
                    row[f"test_mse_{model_type}"] = ev["test_mse"]
                raw_rows.append(row)
            logging.getLogger("opg_exp").info(
                "noise_dims=%d seed=%d done", nd, seed,
            )

    methods = sorted({r["method"] for r in raw_rows})
    agg = _aggregate(raw_rows, noise_dims, methods)

    # CSVs
    if raw_rows:
        keys = sorted(set().union(*(r.keys() for r in raw_rows)))
        with open(RESULTS_ROOT / "summary_by_noise.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(raw_rows)

    if agg:
        keys2 = sorted(set().union(*(r.keys() for r in agg)))
        with open(RESULTS_ROOT / "summary_mean_std.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys2)
            w.writeheader()
            w.writerows(agg)

    payload = {
        "noise_dims": noise_dims,
        "seeds": seeds,
        "n_samples": N_SAMPLES,
        "test_size": TEST_SIZE,
        "d": D,
        "k_nn": K_NN,
        "tau_rf": TAU_RF,
        "wall_time_s": round(time.time() - t0, 2),
        "summary_mean_std": agg,
    }
    with open(RESULTS_ROOT / "config_snapshot.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    fig_dir = RESULTS_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y_specs = [
        ("projection_distance", "Projection distance", False),
        ("grassmann_distance", "Grassmann distance", False),
        ("vector_correlation", "Vector correlation", False),
        ("test_mse_linear", "Test MSE (linear)", False),
        ("test_mse_random_forest", "Test MSE (RandomForest)", False),
        ("fit_time_s", "Fit time (s)", True),
    ]
    plot_noise_scaling_curves(
        agg,
        noise_dims=noise_dims,
        y_keys=y_specs,
        save_dir=fig_dir,
        title_suffix="(OPG, Friedman-1)",
        file_prefix="opg_",
    )

    if not quick:
        _plot_weight_diagnostics_at_noise(
            noise_dims=40,
            seed=0,
            fig_path=fig_dir / "opg_weight_histograms_noise40.png",
        )

    _print_final_report(agg, noise_dims, methods)
    logging.getLogger("opg_exp").info(
        "Finished in %.1fs; outputs under %s", time.time() - t0, RESULTS_ROOT,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="OPG weight-scheme Friedman experiment.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="2 noise levels × 2 seeds (smoke test).",
    )
    args = p.parse_args()
    run_experiment(quick=args.quick)


if __name__ == "__main__":
    main()
