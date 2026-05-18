#!/usr/bin/env python3
"""Run the three MAVE paper-track configs and refresh ``master_summary.csv``.

Usage
-----
    python run_mave_tracks.py
    python run_mave_tracks.py --configs configs/mave_energy_ccpp.yaml
    python run_mave_tracks.py --quick   # 1 seed, smaller forests (edits cfg copy)

Each config should set ``experiment.results_track`` and ``multi_seed.enabled``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.config import load_config
from sp_mave.experiment_paths import resolve_results_dir

from run_experiment import RESULTS_DIR, run_multi_seed_experiment

DEFAULT_CONFIGS = [
    "configs/mave_semi_synthetic_california.yaml",
    "configs/mave_energy_ccpp.yaml",
    "configs/mave_housing_california.yaml",
]

MASTER_CSV = RESULTS_DIR / "mave" / "master_summary.csv"


def _apply_quick(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("multi_seed", {})["n_seeds"] = 1
    cfg.setdefault("random_forest", {})["n_estimators"] = 50
    cfg.setdefault("model", {})["max_iters"] = 15
    t = cfg.setdefault("tuning", {})
    t["max_iters"] = 15
    return cfg


def _append_master_row(
    writer: csv.DictWriter,
    track: str,
    slug: str,
    config_path: str,
    summary_path: Path,
) -> None:
    if not summary_path.is_file():
        return
    with open(summary_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            writer.writerow(
                {
                    "track": track,
                    "results_slug": slug,
                    "config": config_path,
                    **row,
                }
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Run MAVE paper-track YAML configs.")
    p.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="YAML configs to run (default: all three paper tracks).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Single seed, small forest (smoke test).",
    )
    args = p.parse_args()

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    master_fields = [
        "track",
        "results_slug",
        "config",
        "method",
        "n_seeds",
        "test_mse_linear_mean",
        "test_mse_linear_std",
        "test_mse_random_forest_mean",
        "test_mse_random_forest_std",
        "projection_distance_mean",
        "grassmann_distance_mean",
        "vector_correlation_mean",
        "n_iters_mean",
        "fit_time_mean",
    ]

    with open(MASTER_CSV, "w", newline="") as mf:
        master_writer = csv.DictWriter(mf, fieldnames=master_fields, extrasaction="ignore")
        master_writer.writeheader()

        for rel in args.configs:
            cfg_path = Path(rel).resolve()
            cfg = load_config(str(cfg_path))
            cfg["__config_path__"] = str(cfg_path)
            if args.quick:
                cfg = _apply_quick(cfg)
            exp = cfg.get("experiment") or {}
            track = str(exp.get("results_track", "legacy"))
            slug = str(exp.get("results_slug", cfg["dataset"]["name"]))

            print(f"\n=== Running {rel}  ->  mave/{track}/{slug}/ ===\n")
            run_multi_seed_experiment(cfg)

            out_dir = resolve_results_dir(cfg, RESULTS_DIR)
            summ_csv = out_dir / "tables" / "multi_seed_summary.csv"
            _append_master_row(
                master_writer,
                track,
                slug,
                str(rel),
                summ_csv,
            )
            mf.flush()

    print(f"\nWrote aggregated summary: {MASTER_CSV}")


if __name__ == "__main__":
    main()
