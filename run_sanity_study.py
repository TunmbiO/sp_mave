#!/usr/bin/env python3
"""Sanity study: compare methods across different numbers of noise features.

Default noise levels are [0, 20], testing whether SP-MAVE's advantage
grows as irrelevant dimensions increase.

Usage
-----
    python run_sanity_study.py --config configs/sanity_noise_study.yaml
    python run_sanity_study.py  # uses default config
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sp_mave.config import load_config
from sp_mave.plots import plot_sanity_study
from sp_mave.utils import get_logger

from run_experiment import run_experiment

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_sanity_study(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the experiment for each noise level and collect results."""
    logger = get_logger("sanity_study")

    noise_levels = cfg.get("sanity_study", {}).get("noise_levels", [0, 20])
    logger.info("Noise levels to test: %s", noise_levels)

    all_rows: list[dict[str, Any]] = []

    for n_noise in noise_levels:
        logger.info("===== n_noise = %d =====", n_noise)
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["dataset"]["n_noise"] = n_noise
        rows = run_experiment(cfg_copy)
        for row in rows:
            row["n_noise"] = n_noise
        all_rows.extend(rows)

    # ---- Sanity study plots ----
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_sanity_study(all_rows, save_path=fig_dir / "sanity_noise_study.png")
    logger.info("Saved sanity study figure.")

    # ---- Save summary ----
    summaries_dir = RESULTS_DIR / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    json_path = summaries_dir / "sanity_study_summary.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    logger.info("Saved sanity study summary to %s", json_path)

    # ---- Console summary ----
    _print_sanity_summary(all_rows)

    return all_rows


def _print_sanity_summary(rows: list[dict[str, Any]]) -> None:
    """Pretty-print the sanity study comparison."""
    print("\n" + "=" * 80)
    print("SANITY STUDY SUMMARY")
    print("=" * 80)
    header = (
        f"{'n_noise':>8}  {'Method':<18}  {'Proj Dist':>10}  "
        f"{'Grass Dist':>10}  {'Vec Corr':>10}  {'Test MSE':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r.get("n_noise", 0), r["method"])):
        print(
            f"{row.get('n_noise', '?'):>8}  "
            f"{row['method']:<18}  "
            f"{row.get('projection_distance', float('nan')):>10.4f}  "
            f"{row.get('grassmann_distance', float('nan')):>10.4f}  "
            f"{row.get('vector_correlation', float('nan')):>10.4f}  "
            f"{row.get('test_mse', float('nan')):>10.4f}"
        )
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run noise-dimension sanity study."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/sanity_noise_study.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_sanity_study(cfg)


if __name__ == "__main__":
    main()
