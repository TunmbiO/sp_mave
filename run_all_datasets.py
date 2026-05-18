#!/usr/bin/env python3
"""Run MAVE / SP-MAVE experiments across all configured datasets.

Executes each dataset's config sequentially, then produces a cross-dataset
comparison table summarising whether SP-MAVE's advantage generalises
beyond synthetic data.

Usage
-----
    python run_all_datasets.py
    python run_all_datasets.py --configs configs/diabetes.yaml configs/tecator.yaml
    python run_all_datasets.py --skip-errors

Outputs are written to results/<dataset_name>/ for each dataset and
results/cross_dataset/ for the combined summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.config import load_config
from run_experiment import run_experiment, RESULTS_DIR

DEFAULT_CONFIGS = [
    "configs/friedman_final.yaml",
    "configs/diabetes.yaml",
    "configs/tecator.yaml",
    "configs/communities.yaml",
]


# ======================================================================
# Cross-dataset summary
# ======================================================================

def _load_dataset_summary(dataset_name: str) -> dict[str, Any] | None:
    """Load the saved multi-seed JSON summary for a dataset."""
    path = RESULTS_DIR / dataset_name / "summaries" / "multi_seed_summary.json"
    if not path.exists():
        path = RESULTS_DIR / dataset_name / "summaries" / "experiment_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _print_cross_dataset_summary(
    all_summaries: dict[str, dict[str, Any]],
) -> None:
    """Print a comparison table across all datasets."""
    print("\n" + "=" * 100)
    print("CROSS-DATASET COMPARISON")
    print("=" * 100)

    ds_names = list(all_summaries.keys())

    # Collect all methods and downstream models across datasets
    all_methods: set[str] = set()
    all_ds_models: set[str] = set()
    for ds_info in all_summaries.values():
        summary = ds_info.get("summary", {})
        all_methods.update(summary.keys())
        for entry in summary.values():
            for k in entry:
                if k.startswith("test_mse_") and k.endswith("_mean"):
                    model = k.replace("test_mse_", "").replace("_mean", "")
                    all_ds_models.add(model)

    methods = sorted(all_methods)
    ds_models = sorted(all_ds_models)

    # --- Best method per dataset ---
    print("\n  Best method per dataset (by test MSE, linear downstream):\n")
    print(f"  {'Dataset':<25s}  {'Best Method':<22s}  {'MSE (mean ± std)':>20s}")
    print("  " + "-" * 71)
    for ds_name in ds_names:
        summary = all_summaries[ds_name].get("summary", {})
        best_m = None
        best_mse = float("inf")
        for m, e in summary.items():
            mse = e.get("test_mse_linear_mean", float("inf"))
            if mse < best_mse:
                best_mse = mse
                best_m = m
        if best_m is not None:
            e = summary[best_m]
            std = e.get("test_mse_linear_std", 0)
            print(
                f"  {ds_name:<25s}  {best_m:<22s}"
                f"  {best_mse:.4f} ± {std:.4f}"
            )

    # --- Full comparison table per downstream model ---
    for ds_model in ds_models:
        print(f"\n\n  Test MSE — {ds_model} downstream (mean ± std):\n")
        hdr = f"  {'Method':<22s}" + "".join(
            f"  {d:>20s}" for d in ds_names
        )
        print(hdr)
        print("  " + "-" * (22 + 22 * len(ds_names)))

        for m in methods:
            parts = [f"  {m:<22s}"]
            for ds_name in ds_names:
                summary = all_summaries[ds_name].get("summary", {})
                e = summary.get(m, {})
                mean_k = f"test_mse_{ds_model}_mean"
                std_k = f"test_mse_{ds_model}_std"
                mean_v = e.get(mean_k)
                std_v = e.get(std_k, 0)
                if mean_v is not None:
                    parts.append(f"{mean_v:.4f}±{std_v:.4f}".rjust(22))
                else:
                    parts.append("N/A".rjust(22))
            print("".join(parts))

    # --- SP-MAVE vs ClassicalMAVE relative comparison ---
    print("\n\n  SP-MAVE vs ClassicalMAVE — relative test MSE change (linear):\n")
    print(
        f"  {'Dataset':<25s}  {'SP-MAVE MSE':>14s}  {'Classical MSE':>14s}"
        f"  {'Δ%':>8s}  {'Verdict':<12s}"
    )
    print("  " + "-" * 77)
    for ds_name in ds_names:
        summary = all_summaries[ds_name].get("summary", {})
        sp = summary.get("SPMAVE", {}).get("test_mse_linear_mean")
        cl = summary.get("ClassicalMAVE", {}).get("test_mse_linear_mean")
        if sp is not None and cl is not None and cl > 0:
            pct = 100 * (cl - sp) / cl
            verdict = "SP better" if pct > 1 else (
                "SP worse" if pct < -1 else "comparable"
            )
            print(
                f"  {ds_name:<25s}  {sp:>14.4f}  {cl:>14.4f}"
                f"  {pct:>+7.1f}%  {verdict:<12s}"
            )
        else:
            print(f"  {ds_name:<25s}  {'N/A':>14s}  {'N/A':>14s}  {'N/A':>8s}")

    # --- SP-MAVE vs ClassicalMAVE_k20 ---
    sp_vs_topk_available = False
    for ds_name in ds_names:
        summary = all_summaries[ds_name].get("summary", {})
        if "ClassicalMAVE_k20" in summary and "SPMAVE" in summary:
            sp_vs_topk_available = True
            break

    if sp_vs_topk_available:
        print(
            "\n\n  SP-MAVE vs ClassicalMAVE_k20"
            " — relative test MSE change (linear):\n"
        )
        print(
            f"  {'Dataset':<25s}  {'SP-MAVE MSE':>14s}"
            f"  {'CM_k20 MSE':>14s}  {'Δ%':>8s}  {'Verdict':<12s}"
        )
        print("  " + "-" * 77)
        for ds_name in ds_names:
            summary = all_summaries[ds_name].get("summary", {})
            sp = summary.get("SPMAVE", {}).get("test_mse_linear_mean")
            topk = summary.get("ClassicalMAVE_k20", {}).get(
                "test_mse_linear_mean"
            )
            if sp is not None and topk is not None and topk > 0:
                pct = 100 * (topk - sp) / topk
                verdict = "SP better" if pct > 1 else (
                    "SP worse" if pct < -1 else "comparable"
                )
                print(
                    f"  {ds_name:<25s}  {sp:>14.4f}  {topk:>14.4f}"
                    f"  {pct:>+7.1f}%  {verdict:<12s}"
                )
            else:
                print(
                    f"  {ds_name:<25s}  {'N/A':>14s}"
                    f"  {'N/A':>14s}  {'N/A':>8s}"
                )

    # --- Nonlinear downstream: does it help SP-MAVE more? ---
    if "random_forest" in ds_models and "linear" in ds_models:
        print(
            "\n\n  Nonlinear downstream benefit"
            " (linear → random_forest MSE change):\n"
        )
        print(
            f"  {'Dataset':<25s}  {'Method':<22s}"
            f"  {'Linear MSE':>12s}  {'RF MSE':>12s}  {'Δ%':>8s}"
        )
        print("  " + "-" * 83)
        for ds_name in ds_names:
            summary = all_summaries[ds_name].get("summary", {})
            for m in methods:
                e = summary.get(m, {})
                lin = e.get("test_mse_linear_mean")
                rf = e.get("test_mse_random_forest_mean")
                if lin is not None and rf is not None and lin > 0:
                    pct = 100 * (lin - rf) / lin
                    print(
                        f"  {ds_name:<25s}  {m:<22s}"
                        f"  {lin:>12.4f}  {rf:>12.4f}  {pct:>+7.1f}%"
                    )

    print("\n" + "=" * 100 + "\n")


def _save_cross_dataset_csv(
    all_summaries: dict[str, dict[str, Any]],
) -> None:
    """Save cross-dataset summary to CSV."""
    output_dir = RESULTS_DIR / "cross_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ds_name, ds_info in all_summaries.items():
        summary = ds_info.get("summary", {})
        n_seeds = ds_info.get("n_seeds", "?")
        for method, entry in summary.items():
            row: dict[str, Any] = {
                "dataset": ds_name,
                "method": method,
                "n_seeds": n_seeds,
            }
            row.update(entry)
            rows.append(row)

    if rows:
        all_keys: set[str] = set()
        for r in rows:
            all_keys.update(r.keys())
        fields = ["dataset", "method", "n_seeds"] + sorted(
            all_keys - {"dataset", "method", "n_seeds"}
        )

        csv_path = output_dir / "cross_dataset_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        print(f"  Cross-dataset CSV saved to {csv_path}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MAVE / SP-MAVE experiments across all datasets."
    )
    parser.add_argument(
        "--configs",
        "-c",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config files to run (default: all datasets).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help=(
            "Skip datasets that fail (e.g. download errors) "
            "instead of aborting."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Load existing results and produce cross-dataset summary "
            "without re-running experiments."
        ),
    )
    args = parser.parse_args()

    completed: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    t_total = time.time()

    if args.summary_only:
        # Load existing results without re-running
        for config_path in args.configs:
            config_file = Path(config_path)
            if not config_file.exists():
                continue
            try:
                cfg = load_config(config_path)
                dataset_name = cfg["dataset"]["name"]
                summary = _load_dataset_summary(dataset_name)
                if summary is not None:
                    completed[dataset_name] = summary
                    print(f"  Loaded: {dataset_name}")
                else:
                    print(f"  No saved results for: {dataset_name}")
            except Exception as exc:
                print(f"  SKIP {config_path}: {exc}")
    else:
        for config_path in args.configs:
            config_file = Path(config_path)
            if not config_file.exists():
                print(
                    f"\n  WARNING: config not found: {config_path}, skipping."
                )
                failed.append(config_path)
                continue

            dataset_name = "unknown"
            try:
                cfg = load_config(config_path)
                dataset_name = cfg["dataset"]["name"]
                print(f"\n{'#' * 70}")
                print(f"  DATASET: {dataset_name}  (config: {config_path})")
                print(f"{'#' * 70}\n")

                t0 = time.time()
                run_experiment(cfg)
                elapsed = time.time() - t0
                print(f"\n  {dataset_name} completed in {elapsed:.1f}s")

                summary = _load_dataset_summary(dataset_name)
                if summary is not None:
                    completed[dataset_name] = summary
                else:
                    print(f"  WARNING: no summary found for {dataset_name}")

            except Exception as exc:
                print(f"\n  FAILED: {config_path} — {exc}")
                failed.append(config_path)
                if not args.skip_errors:
                    raise
                continue

    total_elapsed = time.time() - t_total
    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(f"  Completed: {len(completed)} datasets")
    if failed:
        print(f"  Failed: {len(failed)} — {failed}")

    if len(completed) >= 1:
        _save_cross_dataset_csv(completed)
    if len(completed) >= 2:
        _print_cross_dataset_summary(completed)
    elif len(completed) == 1:
        print(
            "\n  Only one dataset completed; "
            "cross-dataset comparison requires at least two."
        )


if __name__ == "__main__":
    main()
