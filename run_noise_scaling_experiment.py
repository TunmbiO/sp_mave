#!/usr/bin/env python3
"""Friedman-1 noise-dimension scaling experiment (isolated from other datasets).

Varies only the number of appended U(0,1) noise features; signal and sample
size stay fixed.  For each noise level and seed, runs the same methods and
evaluation as the main pipeline (no data leakage; tuning per seed on train).

Usage
-----
    python run_noise_scaling_experiment.py --config configs/friedman_noise_scaling.yaml

Outputs (under results/friedman_noise_scaling/):
    tables/noise_scaling_raw.csv
    tables/noise_scaling_summary.csv
    summaries/noise_scaling.json
    figures/noise_scaling_*.png
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.config import load_config
from sp_mave.plots import plot_noise_scaling_curves
from sp_mave.utils import get_logger

from run_experiment import RESULTS_DIR, run_single_seed

OUTPUT_ROOT = RESULTS_DIR / "friedman_noise_scaling"


def _aggregate(
    raw_rows: list[dict[str, Any]],
    noise_dims: list[int],
    methods: list[str],
    downstream_models: list[str],
) -> list[dict[str, Any]]:
    """Mean ± std per (noise_dims, method)."""
    subspace_keys = [
        "projection_distance",
        "grassmann_distance",
        "vector_correlation",
    ]
    agg: list[dict[str, Any]] = []

    for nd in noise_dims:
        for m in methods:
            rows = [r for r in raw_rows if r["noise_dims"] == nd and r["method"] == m]
            if not rows:
                continue
            entry: dict[str, Any] = {
                "noise_dims": nd,
                "p_features": 5 + nd,
                "method": m,
                "n_seeds": len(rows),
            }
            for key in subspace_keys:
                vals = [float(r[key]) for r in rows if key in r]
                if vals:
                    entry[f"{key}_mean"] = float(np.mean(vals))
                    entry[f"{key}_std"] = float(np.std(vals))
            for key in ("fit_time_s", "n_iters"):
                vals = [float(r[key]) for r in rows if key in r]
                if vals:
                    entry[f"{key}_mean"] = float(np.mean(vals))
                    entry[f"{key}_std"] = float(np.std(vals))
            for ds in downstream_models:
                col = f"test_mse_{ds}"
                vals = [float(r[col]) for r in rows if col in r]
                if vals:
                    entry[f"{col}_mean"] = float(np.mean(vals))
                    entry[f"{col}_std"] = float(np.std(vals))
            agg.append(entry)

    agg.sort(key=lambda x: (x["noise_dims"], x["method"]))
    return agg


def _print_summary_table(
    agg: list[dict[str, Any]],
    noise_dims: list[int],
    methods: list[str],
) -> None:
    print("\n" + "=" * 100)
    print("NOISE SCALING SUMMARY  (mean ± std over seeds)")
    print("=" * 100)

    for nd in noise_dims:
        print(f"\n  Irrelevant noise features = {nd}  (p = {5 + nd})")
        print(
            f"  {'Method':<22s}  {'Proj dist':>18s}  {'Grass dist':>18s}  "
            f"{'Vec corr':>14s}  {'MSE(lin)':>16s}  {'Time (s)':>12s}"
        )
        print("  " + "-" * 96)
        for m in methods:
            row = next(
                (r for r in agg if r["noise_dims"] == nd and r["method"] == m),
                None,
            )
            if row is None:
                print(f"  {m:<22s}  {'—':>18s}  {'—':>18s}  {'—':>14s}  {'—':>16s}  {'—':>12s}")
                continue
            def fmt(b: str) -> str:
                mu, sd = row.get(f"{b}_mean"), row.get(f"{b}_std")
                if mu is None:
                    return "—"
                return f"{mu:.4f}±{sd:.4f}"

            print(
                f"  {m:<22s}  {fmt('projection_distance'):>18s}  "
                f"{fmt('grassmann_distance'):>18s}  {fmt('vector_correlation'):>14s}  "
                f"{fmt('test_mse_linear'):>16s}  {fmt('fit_time_s'):>12s}"
            )

    print("\n" + "=" * 100 + "\n")


def run_noise_scaling(cfg: dict[str, Any]) -> None:
    ns = cfg.get("noise_scaling_experiment") or {}
    noise_dims = list(ns.get("noise_dims", [0, 10, 20, 40, 80, 120]))
    n_seeds = int(ns.get("n_seeds", 10))

    if cfg.get("dataset", {}).get("name") != "friedman1":
        raise ValueError(
            "noise_scaling_experiment is only defined for dataset.name == friedman1"
        )

    downstream_models = cfg["evaluation"].get(
        "downstream_models", ["linear"]
    )

    logger = get_logger("noise_scaling")
    mave_logger = logging.getLogger("sp_mave.mave")
    spmave_logger = logging.getLogger("sp_mave.spmave")
    prev_m, prev_s = mave_logger.level, spmave_logger.level
    mave_logger.setLevel(logging.WARNING)
    spmave_logger.setLevel(logging.WARNING)

    raw_rows: list[dict[str, Any]] = []
    t0 = time.time()

    try:
        for nd in noise_dims:
            for seed in range(n_seeds):
                cfg_run = copy.deepcopy(cfg)
                cfg_run["dataset"]["n_noise"] = int(nd)
                cfg_run["dataset"]["random_state"] = seed
                cfg_run["split"]["random_state"] = seed
                cfg_run["model"]["random_state"] = seed
                cfg_run["crossfit"]["random_state"] = seed
                cfg_run["random_forest"]["random_state"] = seed
                cfg_run["tuning"]["random_state"] = seed

                t_cell = time.time()
                out = run_single_seed(cfg_run, seed=seed, logger_obj=logger)
                elapsed = time.time() - t_cell

                for row in out["summary_rows"]:
                    r = dict(row)
                    r["noise_dims"] = nd
                    r["p_features"] = 5 + nd
                    raw_rows.append(r)

                logger.info(
                    "  noise_dims=%d  seed=%d  done in %.1fs",
                    nd, seed, elapsed,
                )
    finally:
        mave_logger.setLevel(prev_m)
        spmave_logger.setLevel(prev_s)

    total_s = time.time() - t0
    n_cells = len(noise_dims) * n_seeds
    logger.info("Finished %d (noise×seed) cells in %.1fs", n_cells, total_s)

    methods = sorted({r["method"] for r in raw_rows})
    agg = _aggregate(raw_rows, noise_dims, methods, downstream_models)

    out_dir = OUTPUT_ROOT
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    summ_dir = out_dir / "summaries"
    for d in (tables_dir, fig_dir, summ_dir):
        d.mkdir(parents=True, exist_ok=True)

    if raw_rows:
        keys = sorted(set().union(*(r.keys() for r in raw_rows)))
        with open(tables_dir / "noise_scaling_raw.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(raw_rows)

    if agg:
        keys2 = sorted(set().union(*(r.keys() for r in agg)))
        with open(tables_dir / "noise_scaling_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys2)
            w.writeheader()
            w.writerows(agg)

    payload = {
        "noise_dims": noise_dims,
        "n_seeds": n_seeds,
        "methods": methods,
        "method_configs": cfg.get("methods", []),
        "downstream_models": downstream_models,
        "n_samples": cfg["dataset"]["n_samples"],
        "friedman_noise_std": cfg["dataset"].get("noise", 1.0),
        "total_wall_time_s": round(total_s, 2),
        "summary": agg,
    }
    with open(summ_dir / "noise_scaling.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    y_specs: list[tuple[str, str, bool]] = [
        ("projection_distance", "Projection distance", False),
        ("grassmann_distance", "Grassmann distance", False),
        ("vector_correlation", "Vector correlation", False),
        ("test_mse_linear", "Test MSE (linear downstream)", False),
        ("fit_time_s", "Fit time (s)", True),
    ]
    for ds in downstream_models:
        if ds == "linear":
            continue
        y_specs.append(
            (
                f"test_mse_{ds}",
                f"Test MSE ({ds})",
                False,
            )
        )

    plot_noise_scaling_curves(
        agg,
        noise_dims=noise_dims,
        y_keys=y_specs,
        save_dir=fig_dir,
        title_suffix="(Friedman-1)",
    )

    _print_summary_table(agg, noise_dims, methods)
    logger.info("Wrote %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Friedman-1 noise-dimension scaling experiment."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/friedman_noise_scaling.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_noise_scaling(cfg)


if __name__ == "__main__":
    main()
