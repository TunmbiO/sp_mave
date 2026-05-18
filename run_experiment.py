#!/usr/bin/env python3
"""Run SP-MAVE / Classical MAVE comparison experiments.

Supports single-seed and multi-seed modes, multiple downstream models,
convergence diagnostics, and automated final report generation.

Usage
-----
    python run_experiment.py --config configs/friedman_final.yaml

Outputs default to results/<dataset.name>/ or, if ``experiment.results_track``
is set, to results/mave/<track>/<slug>/ (see configs/mave_*.yaml).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.config import load_config
from sp_mave.datasets import load_from_config
from sp_mave.evaluation import evaluate_downstream
from sp_mave.experiment_paths import resolve_results_dir
from sp_mave.mave import ClassicalMAVE
from sp_mave.method_labels import aggregation_group_key, paper_method_name
from sp_mave.metrics import compute_all_metrics
from sp_mave.opg import OPGClassical, RFProximityOPG
from sp_mave.plots import (
    plot_objective_convergence,
    plot_raw_proximities,
    plot_subspace_change,
    plot_weight_histograms,
)
from sp_mave.plots_paper import plot_subspace_metrics_by_method, plot_test_mse_by_method
from sp_mave.preprocessing import split_bundle
from sp_mave.spmave import SPMAVE
from sp_mave.tuning import tune_k_classical_topk, tune_k_tau
from sp_mave.types import MAVEResult
from sp_mave.utils import get_logger

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _dataset_results_dir(cfg: dict[str, Any]) -> Path:
    """Output directory: legacy ``results/<name>/`` or ``results/mave/<track>/<slug>/``."""
    return resolve_results_dir(cfg, RESULTS_DIR)


# ======================================================================
# Estimator factory
# ======================================================================

def _build_estimator(method: str, cfg: dict[str, Any]) -> Any:
    """Instantiate an estimator from a method name and config."""
    mcfg = cfg["model"]
    common = dict(
        d=mcfg["d"],
        max_iters=mcfg["max_iters"],
        tol=mcfg["tol"],
        ridge_lambda=mcfg["ridge_lambda"],
        init_method=mcfg["init_method"],
        random_state=mcfg["random_state"],
    )

    if method == "classical_mave":
        return ClassicalMAVE(bandwidth=mcfg["bandwidth"], k=None, **common)
    if method == "classical_mave_topk":
        return ClassicalMAVE(bandwidth=mcfg["bandwidth"], k=mcfg["k"], **common)
    if method == "sp_mave":
        return SPMAVE(
            k=mcfg["k"], tau=mcfg["tau"],
            rf_params=cfg["random_forest"], crossfit_cfg=cfg["crossfit"],
            **common,
        )
    if method == "opg_classical":
        return OPGClassical(bandwidth=mcfg["bandwidth"], k=None, **common)
    if method == "opg_classical_topk":
        return OPGClassical(bandwidth=mcfg["bandwidth"], k=mcfg["k"], **common)
    if method == "opg_sp":
        return RFProximityOPG(
            k=mcfg["k"], tau=mcfg["tau"],
            rf_params=cfg["random_forest"], crossfit_cfg=cfg["crossfit"],
            **common,
        )
    raise ValueError(f"Unknown method: '{method}'")


# ======================================================================
# Seed override
# ======================================================================

def _override_seed(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a deep copy of cfg with all random states set to *seed*."""
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["random_state"] = seed
    cfg["split"]["random_state"] = seed
    cfg["model"]["random_state"] = seed
    cfg["crossfit"]["random_state"] = seed
    cfg["random_forest"]["random_state"] = seed
    cfg["tuning"]["random_state"] = seed
    return cfg


# ======================================================================
# Single-seed runner
# ======================================================================

def run_single_seed(
    cfg: dict[str, Any],
    seed: int | None = None,
    logger_obj: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run one experiment (one seed).

    Returns dict with keys: seed, summary_rows, results, tuning_report.
    """
    if seed is not None:
        cfg = _override_seed(cfg, seed)
    log = logger_obj or get_logger("experiment")

    bundle = load_from_config(cfg)
    X_train, X_test, y_train, y_test = split_bundle(bundle, cfg)

    methods = cfg.get("methods", ["classical_mave", "sp_mave"])
    downstream_models = cfg["evaluation"].get("downstream_models", ["linear"])

    tuning_cfg = cfg.get("tuning", {})
    tuning_report: dict[str, Any] | None = None
    tuning_report_classical: dict[str, Any] | None = None
    tuned_k: int | None = None
    tuned_tau: float | None = None
    tuned_k_classical: int | None = None

    if tuning_cfg.get("enabled", False):
        if "sp_mave" in methods:
            tuning_report = tune_k_tau(
                X_train,
                y_train,
                k_candidates=tuning_cfg["k_candidates"],
                tau_candidates=tuning_cfg["tau_candidates"],
                d=cfg["model"]["d"],
                rf_params=cfg["random_forest"],
                crossfit_cfg=cfg["crossfit"],
                model_cfg=cfg["model"],
                val_size=tuning_cfg.get("val_size", 0.2),
                downstream_model="linear",
                random_state=tuning_cfg.get("random_state", 42),
                tuning_max_iters=tuning_cfg.get("max_iters"),
            )
            tuned_k = tuning_report["best_k"]
            tuned_tau = tuning_report["best_tau"]
            log.info(
                "Seed %s: tuned SP-MAVE k=%d, tau=%.4f (val_mse=%.4f)",
                seed,
                tuned_k,
                tuned_tau,
                tuning_report["best_val_mse"],
            )

        if "classical_mave_topk" in methods:
            matched = tuning_cfg.get("matched_k_for_classical_topk", True)
            if matched and tuned_k is not None:
                tuned_k_classical = tuned_k
                log.info(
                    "Seed %s: classical top-k k matched to SP-MAVE k=%d",
                    seed,
                    tuned_k_classical,
                )
            elif tuning_cfg.get("tune_classical_topk_k", False):
                tuning_report_classical = tune_k_classical_topk(
                    X_train,
                    y_train,
                    k_candidates=tuning_cfg["k_candidates"],
                    d=cfg["model"]["d"],
                    model_cfg=cfg["model"],
                    val_size=tuning_cfg.get("val_size", 0.2),
                    random_state=tuning_cfg.get("random_state", 42),
                    tuning_max_iters=tuning_cfg.get("max_iters"),
                )
                tuned_k_classical = tuning_report_classical["best_k"]
                log.info(
                    "Seed %s: tuned classical top-k k=%d (val_mse=%.4f)",
                    seed,
                    tuned_k_classical,
                    tuning_report_classical["best_val_mse"],
                )

    all_results: dict[str, MAVEResult] = {}
    summary_rows: list[dict[str, Any]] = []

    for method_name in methods:
        t0 = time.time()
        build_cfg = cfg
        if (
            (method_name == "sp_mave" and tuned_k is not None)
            or (
                method_name == "classical_mave_topk"
                and tuned_k_classical is not None
            )
        ):
            build_cfg = copy.deepcopy(cfg)
        if method_name == "sp_mave" and tuned_k is not None:
            build_cfg["model"]["k"] = tuned_k
            build_cfg["model"]["tau"] = tuned_tau
        if method_name == "classical_mave_topk" and tuned_k_classical is not None:
            build_cfg["model"]["k"] = tuned_k_classical
        estimator = _build_estimator(method_name, build_cfg)
        estimator.fit(X_train, y_train)
        elapsed = time.time() - t0
        result = estimator.result_
        assert result is not None
        all_results[result.method_name] = result

        row: dict[str, Any] = {
            "method": result.method_name,
            "method_paper": paper_method_name(result.method_name),
            "seed": seed,
            "n_iters": result.n_iters,
            "final_loss": result.loss_history[-1] if result.loss_history else None,
            "fit_time_s": round(elapsed, 3),
        }

        # Subspace metrics
        if bundle.true_subspace is not None:
            sub_metrics = compute_all_metrics(result.B, bundle.true_subspace)
            row.update(sub_metrics)

        # Downstream prediction — one column per downstream model
        for ds_model in downstream_models:
            ds = evaluate_downstream(
                B=result.B,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                model_type=ds_model,
            )
            row[f"test_mse_{ds_model}"] = ds["test_mse"]
            row[f"train_mse_{ds_model}"] = ds["train_mse"]

        summary_rows.append(row)

    out: dict[str, Any] = {
        "seed": seed,
        "summary_rows": summary_rows,
        "results": all_results,
        "tuning_report": tuning_report,
    }
    if tuning_report_classical is not None:
        out["tuning_report_classical"] = tuning_report_classical
    return out


# ======================================================================
# Multi-seed runner
# ======================================================================

def run_multi_seed_experiment(cfg: dict[str, Any]) -> None:
    """Run across multiple seeds, aggregate, generate all outputs."""
    logger = get_logger("experiment")
    dataset_name = cfg["dataset"]["name"]
    results_dir = _dataset_results_dir(cfg)

    ms_cfg = cfg.get("multi_seed", {})
    n_seeds = ms_cfg.get("n_seeds", 20)
    seeds = list(range(n_seeds))

    logger.info("=" * 60)
    logger.info("MULTI-SEED EXPERIMENT: %s  (%d seeds)", dataset_name, n_seeds)
    logger.info("=" * 60)

    # Suppress per-iteration logs during bulk runs
    mave_logger = logging.getLogger("sp_mave.mave")
    spmave_logger = logging.getLogger("sp_mave.spmave")
    prev_mave = mave_logger.level
    prev_spmave = spmave_logger.level
    mave_logger.setLevel(logging.WARNING)
    spmave_logger.setLevel(logging.WARNING)

    all_raw_rows: list[dict[str, Any]] = []
    representative_results: dict[str, MAVEResult] | None = None
    representative_tuning: dict[str, Any] | None = None

    t_total = time.time()
    try:
        for i, seed in enumerate(seeds):
            t0 = time.time()
            seed_out = run_single_seed(cfg, seed=seed, logger_obj=logger)
            elapsed = time.time() - t0
            all_raw_rows.extend(seed_out["summary_rows"])
            if i == 0:
                representative_results = seed_out["results"]
                representative_tuning = seed_out["tuning_report"]
            logger.info(
                "  Seed %2d/%d done in %.1fs", i + 1, n_seeds, elapsed
            )
    finally:
        mave_logger.setLevel(prev_mave)
        spmave_logger.setLevel(prev_spmave)

    total_elapsed = time.time() - t_total
    logger.info("All %d seeds completed in %.1fs", n_seeds, total_elapsed)

    # ---- Outputs ----
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "configs").mkdir(parents=True, exist_ok=True)
    cfg_snap = cfg.get("__config_path__")
    if cfg_snap:
        try:
            shutil.copy2(cfg_snap, results_dir / "configs" / Path(cfg_snap).name)
        except OSError as exc:
            logger.warning("Could not copy config to results: %s", exc)

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Convergence and subspace-change plots from representative seed (seed 0)
    if representative_results:
        plot_objective_convergence(representative_results, save_dir=fig_dir)
        plot_subspace_change(representative_results, save_dir=fig_dir)
        plot_weight_histograms(
            representative_results,
            save_path=fig_dir / "weight_distributions.png",
        )
        for name, res in representative_results.items():
            if "raw_proximities" in res.extra:
                plot_raw_proximities(
                    res.extra["raw_proximities"],
                    save_path=fig_dir / f"raw_proximities_{name}.png",
                )
        logger.info("Saved convergence, subspace-change, and weight figures.")

    # Aggregate across seeds (by paper bucket: all ClassicalMAVE_k* -> ClassicalMAVE_kNN)
    downstream_models = cfg["evaluation"].get("downstream_models", ["linear"])
    summary = _aggregate_multi_seed(all_raw_rows, downstream_models)
    methods = sorted({aggregation_group_key(r) for r in all_raw_rows})

    # Save CSVs
    _save_multi_seed_csv(all_raw_rows, summary, results_dir)

    # Save JSON
    _save_multi_seed_json(
        summary, cfg, representative_tuning, n_seeds, results_dir
    )

    exp_meta = cfg.get("experiment", {})
    if exp_meta.get("paper_plots", True):
        ds_title = str(cfg.get("dataset", {}).get("name", ""))
        if downstream_models:
            plot_test_mse_by_method(
                summary,
                methods,
                downstream_models[0],
                fig_dir / f"paper_test_mse_{downstream_models[0]}.png",
                title=ds_title,
            )
        if any(
            "projection_distance_mean" in summary.get(m, {}) for m in methods
        ):
            plot_subspace_metrics_by_method(
                summary,
                methods,
                fig_dir / "paper_subspace_metrics.png",
                title=ds_title,
            )
        logger.info("Saved paper-style figures under %s", fig_dir)

    # Print final report
    _print_final_report(
        summary, methods, downstream_models, n_seeds,
        max_iters=cfg["model"]["max_iters"],
        dataset_name=dataset_name,
    )


# ======================================================================
# Aggregation
# ======================================================================

_SUBSPACE_METRICS = ["projection_distance", "grassmann_distance", "vector_correlation"]


def _aggregate_multi_seed(
    raw_rows: list[dict[str, Any]],
    downstream_models: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute mean ± std across seeds per *paper* method bucket.

    Groups rows with :func:`sp_mave.method_labels.aggregation_group_key` so
    ``ClassicalMAVE_k20``, ``ClassicalMAVE_k25``, … merge into
    ``ClassicalMAVE_kNN`` (same role as a single ``RFWeightedMAVE`` bucket
    for SP-MAVE regardless of tuned *k*).

    Raw per-seed CSV still lists the internal ``method`` string with *k*.

    Returns {paper_method: {metric_mean, metric_std, ...}}
    """
    summary: dict[str, dict[str, Any]] = {}
    group_keys = sorted({aggregation_group_key(r) for r in raw_rows})

    for gkey in group_keys:
        rows = [r for r in raw_rows if aggregation_group_key(r) == gkey]
        if not rows:
            continue
        entry: dict[str, Any] = {"n_seeds": len(rows)}

        for m in _SUBSPACE_METRICS:
            vals = [r[m] for r in rows if m in r]
            if vals:
                entry[f"{m}_mean"] = float(np.mean(vals))
                entry[f"{m}_std"] = float(np.std(vals))

        for ds in downstream_models:
            key = f"test_mse_{ds}"
            vals = [r[key] for r in rows if key in r]
            if vals:
                entry[f"{key}_mean"] = float(np.mean(vals))
                entry[f"{key}_std"] = float(np.std(vals))

        iters = [r["n_iters"] for r in rows]
        entry["n_iters_mean"] = float(np.mean(iters))
        entry["n_iters_std"] = float(np.std(iters))

        times = [r["fit_time_s"] for r in rows]
        entry["fit_time_mean"] = float(np.mean(times))

        summary[gkey] = entry

    return summary


# ======================================================================
# I/O helpers
# ======================================================================

def _save_multi_seed_csv(
    raw_rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    results_dir: Path,
) -> None:
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Raw per-seed rows
    if raw_rows:
        raw_path = tables_dir / "multi_seed_raw.csv"
        keys = list(raw_rows[0].keys())
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(raw_rows)

    # Aggregated summary (method column = paper bucket, e.g. ClassicalMAVE_kNN)
    if summary:
        agg_path = tables_dir / "multi_seed_summary.csv"
        first = next(iter(summary.values()))
        fields = ["method"] + list(first.keys())
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for paper_method, entry in summary.items():
                w.writerow({"method": paper_method, **entry})


def _save_multi_seed_json(
    summary: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    tuning_report: dict[str, Any] | None,
    n_seeds: int,
    results_dir: Path,
) -> None:
    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": cfg,
        "n_seeds": n_seeds,
        "summary": summary,
    }
    if tuning_report is not None:
        payload["tuning_seed0"] = tuning_report
    with open(summaries_dir / "multi_seed_summary.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


# ======================================================================
# Printing
# ======================================================================

def _fmt(mean: float, std: float, width: int = 14) -> str:
    return f"{mean:.4f}±{std:.4f}".rjust(width)


def _print_final_report(
    summary: dict[str, dict[str, Any]],
    methods: list[str],
    downstream_models: list[str],
    n_seeds: int,
    max_iters: int = 300,
    dataset_name: str = "",
) -> None:
    print("\n" + "=" * 90)
    title = f"MULTI-SEED EXPERIMENT SUMMARY  ({n_seeds} seeds)"
    if dataset_name:
        title += f"  —  {dataset_name}"
    print(title)
    print("=" * 90)

    has_subspace = any(
        "projection_distance_mean" in e for e in summary.values()
    )

    # --- Subspace recovery (only for synthetic datasets with known truth) ---
    if has_subspace:
        print("\n  Subspace Recovery (mean ± std):")
        hdr = (
            f"  {'Method':<20s}  {'Proj Dist':>14s}  {'Grass Dist':>14s}"
            f"  {'Vec Corr':>14s}  {'Iters':>10s}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for m in methods:
            e = summary.get(m)
            if not e:
                continue
            pd_s = _fmt(
                e["projection_distance_mean"], e["projection_distance_std"]
            )
            gd_s = _fmt(
                e["grassmann_distance_mean"], e["grassmann_distance_std"]
            )
            vc_s = _fmt(
                e["vector_correlation_mean"], e["vector_correlation_std"]
            )
            it_s = f"{e['n_iters_mean']:.0f}±{e['n_iters_std']:.0f}".rjust(10)
            print(f"  {m:<20s}  {pd_s}  {gd_s}  {vc_s}  {it_s}")
    else:
        print(
            "\n  (Subspace recovery metrics not available"
            " — true subspace unknown for this dataset)"
        )

    # --- Downstream prediction ---
    print("\n  Downstream Prediction — Test MSE (mean ± std):")
    print(
        "  (Second-stage models estimating g after B is learned by MAVE/SP-MAVE)"
    )
    ds_hdr_parts = [f"{'Method':<20s}"]
    for ds in downstream_models:
        ds_hdr_parts.append(f"{ds:>14s}")
    ds_hdr_parts.append(f"{'Iters':>10s}")
    print("  " + "  ".join(ds_hdr_parts))
    print("  " + "-" * (20 + 16 * len(downstream_models) + 12))
    for m in methods:
        e = summary.get(m)
        if not e:
            continue
        parts = [f"{m:<20s}"]
        for ds in downstream_models:
            k_mean = f"test_mse_{ds}_mean"
            k_std = f"test_mse_{ds}_std"
            if k_mean in e:
                parts.append(_fmt(e[k_mean], e[k_std]))
            else:
                parts.append("N/A".rjust(14))
        it_s = f"{e['n_iters_mean']:.0f}±{e['n_iters_std']:.0f}".rjust(10)
        parts.append(it_s)
        print("  " + "  ".join(parts))

    # --- Analysis ---
    print("\n" + "-" * 90)
    print("  ANALYSIS")
    print("-" * 90)

    section = 1

    # Best method per downstream model
    print(f"\n  {section}. Best method by test MSE (lowest mean):")
    for ds in downstream_models:
        k = f"test_mse_{ds}_mean"
        best_m = min(
            methods,
            key=lambda m: summary.get(m, {}).get(k, float("inf")),
        )
        e = summary.get(best_m, {})
        std_k = f"test_mse_{ds}_std"
        print(
            f"     {ds:>14s}: {best_m}  "
            f"({e.get(k, 0):.4f} ± {e.get(std_k, 0):.4f})"
        )
    section += 1

    # Best method by subspace recovery (only if available)
    if has_subspace:
        print(f"\n  {section}. Best method by subspace recovery:")
        for metric, label, lower_is_better in [
            ("projection_distance", "Proj Dist", True),
            ("grassmann_distance", "Grass Dist", True),
            ("vector_correlation", "Vec Corr", False),
        ]:
            k = f"{metric}_mean"
            if lower_is_better:
                best_m = min(
                    methods,
                    key=lambda m: summary.get(m, {}).get(k, float("inf")),
                )
            else:
                best_m = max(
                    methods,
                    key=lambda m: summary.get(m, {}).get(k, float("-inf")),
                )
            e = summary.get(best_m, {})
            std_k = f"{metric}_std"
            print(
                f"     {label:>14s}: {best_m}  "
                f"({e.get(k, 0):.4f} ± {e.get(std_k, 0):.4f})"
            )
        section += 1

    # Consistency: coefficient of variation
    ref_ds = downstream_models[0] if downstream_models else "linear"
    print(
        f"\n  {section}. Consistency across seeds "
        f"(CV = std/mean for test_mse_{ref_ds}):"
    )
    for m in methods:
        e = summary.get(m, {})
        mean_v = e.get(f"test_mse_{ref_ds}_mean", 0)
        std_v = e.get(f"test_mse_{ref_ds}_std", 0)
        cv = std_v / mean_v if mean_v > 0 else float("nan")
        print(f"     {m:<20s}  CV = {cv:.3f}")
    section += 1

    # Nonlinear downstream effect
    if len(downstream_models) > 1:
        print(f"\n  {section}. Effect of nonlinear second-stage models:")
        for m in methods:
            e = summary.get(m, {})
            linear_mean = e.get("test_mse_linear_mean", float("inf"))
            for ds in downstream_models:
                if ds == "linear":
                    continue
                ds_mean = e.get(f"test_mse_{ds}_mean", float("inf"))
                delta = linear_mean - ds_mean
                pct = 100 * delta / linear_mean if linear_mean > 0 else 0
                direction = "improvement" if delta > 0 else "degradation"
                print(
                    f"     {m:<20s}  linear→{ds}: "
                    f"{delta:+.4f} ({pct:+.1f}% {direction})"
                )
        section += 1

    # Convergence summary
    print(f"\n  {section}. Convergence observations:")
    for m in methods:
        e = summary.get(m, {})
        avg_it = e.get("n_iters_mean", 0)
        if avg_it >= max_iters * 0.95:
            print(
                f"     {m:<20s}  avg {avg_it:.0f} iters — "
                f"hitting max_iters={max_iters}, may need more iterations"
            )
        else:
            print(
                f"     {m:<20s}  avg {avg_it:.0f} iters — "
                f"converged well within budget"
            )

    print("\n" + "=" * 90 + "\n")


# ======================================================================
# Legacy single-seed entry point
# ======================================================================

def run_experiment(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch: multi-seed if configured, otherwise single seed."""
    ms_cfg = cfg.get("multi_seed", {})
    if ms_cfg.get("enabled", False):
        run_multi_seed_experiment(cfg)
        return []

    # Single seed (backward compatible)
    logger = get_logger("experiment")
    results_dir = _dataset_results_dir(cfg)
    (results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (results_dir / "configs").mkdir(parents=True, exist_ok=True)
    cfg_snap = cfg.get("__config_path__")
    if cfg_snap:
        try:
            shutil.copy2(cfg_snap, results_dir / "configs" / Path(cfg_snap).name)
        except OSError as exc:
            logger.warning("Could not copy config to results: %s", exc)

    seed_out = run_single_seed(cfg, seed=None, logger_obj=logger)
    rows = seed_out["summary_rows"]
    results = seed_out["results"]
    tuning_report = seed_out["tuning_report"]

    if cfg["evaluation"].get("save_figures", True):
        fig_dir = results_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_objective_convergence(results, save_dir=fig_dir)
        plot_subspace_change(results, save_dir=fig_dir)
        plot_weight_histograms(results, save_path=fig_dir / "weight_distributions.png")
        for name, res in results.items():
            if "raw_proximities" in res.extra:
                plot_raw_proximities(
                    res.extra["raw_proximities"],
                    save_path=fig_dir / f"raw_proximities_{name}.png",
                )

    # Save summary
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with open(tables_dir / "comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    summaries_dir = results_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"config": cfg, "results": rows}
    if tuning_report is not None:
        payload["tuning"] = tuning_report
    with open(summaries_dir / "experiment_summary.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    _print_single_seed_summary(rows, tuning_report)
    return rows


def _print_single_seed_summary(
    rows: list[dict[str, Any]],
    tuning_report: dict[str, Any] | None = None,
) -> None:
    if not rows:
        return
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    if tuning_report is not None:
        print(
            f"  Tuned SP-MAVE: k={tuning_report['best_k']}  "
            f"tau={tuning_report['best_tau']:.4f}  "
            f"(val_mse={tuning_report['best_val_mse']:.4f})"
        )
        print()

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    display_cols: list[tuple[str, str, int, str]] = [
        ("method", "Method", 18, "s"),
        ("projection_distance", "Proj Dist", 11, ".4f"),
        ("grassmann_distance", "Grass Dist", 11, ".4f"),
        ("vector_correlation", "Vec Corr", 10, ".4f"),
        ("n_iters", "Iters", 6, "d"),
        ("fit_time_s", "Time (s)", 9, ".2f"),
    ]
    for key in sorted(all_keys):
        if key.startswith("test_mse_"):
            ds_name = key.replace("test_mse_", "")
            display_cols.append((key, f"MSE({ds_name})", 14, ".4f"))

    cols = [(k, h, w, f) for k, h, w, f in display_cols if k in all_keys]

    header = "  ".join(
        f"{h:<{w}}" if f == "s" else f"{h:>{w}}" for _, h, w, f in cols
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for key, _hdr, width, fmt in cols:
            val = row.get(key)
            if val is None:
                parts.append(f"{'N/A':>{width}}")
            elif fmt == "s":
                parts.append(f"{val:<{width}}")
            else:
                parts.append(f"{val:>{width}{fmt}}")
        print("  ".join(parts))
    print("=" * 80 + "\n")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAVE / SP-MAVE comparison experiment."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["__config_path__"] = str(Path(args.config).resolve())
    run_experiment(cfg)


if __name__ == "__main__":
    main()
