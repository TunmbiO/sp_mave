#!/usr/bin/env python3
"""Supplementary run: fit L1MAVE on the existing Friedman1 grid.

run_synthetic_unified_v1.py predates L1MAVE (added for the new rotated-B
DGP experiments), so results/synthetic_unified_v1.csv has only 3 methods.
This script fits L1MAVE on the identical (n, noise_dims, seed) grid and
merges the results into synthetic_unified_v1.csv / _summary.csv so the
figures can show a real 4-method comparison on Friedman1 too.

Usage
-----
    python run_l1mave_on_friedman1.py              # full run
    python run_l1mave_on_friedman1.py --smoke-test

Outputs (merged in place)
--------------------------
    results/synthetic_unified_v1.csv          (+360 L1MAVE rows)
    results/synthetic_unified_v1_summary.csv  (regenerated for 4 methods)
    results/synthetic_unified_v1_l1mave_failures.log
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

import run_synthetic_unified_v1 as base
from sp_mave.mave import L1MAVE
from sp_mave.preprocessing import standardize
from sp_mave.utils import get_logger

logger = get_logger("l1mave_friedman1")
logging.getLogger("sp_mave.mave").setLevel(logging.WARNING)

FAILURES_LOG = base.RESULTS_DIR / "synthetic_unified_v1_l1mave_failures.log"


def run_one_cell(n: int, noise_dims: int, seed: int) -> dict[str, Any] | None:
    np.random.seed(seed)
    p = base.D_TRUE + noise_dims
    k = max(1, math.floor(base.K_FRACTION * n))

    X, y = make_friedman1(
        n_samples=n, n_features=p, noise=base.FRIEDMAN_NOISE_STD, random_state=seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=base.TEST_SIZE, random_state=seed
    )
    X_tr, X_te, _ = standardize(X_tr, X_te)

    true_sub = base._true_subspace(p)
    cfg = base.MAVE_MODEL_CFG

    try:
        t0 = time.time()
        est = L1MAVE(
            d=cfg["d"],
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
        return base._build_row(
            method="L1MAVE",
            est=est,
            X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
            true_sub=true_sub,
            n_obs=n, noise_dims=noise_dims, p=p, seed=seed,
            fit_time=fit_time, fixed_k=k,
        )
    except Exception:
        msg = (
            f"FAILURE  method=L1MAVE  n={n}  noise_dims={noise_dims}  seed={seed}\n"
            + traceback.format_exc() + "\n"
        )
        logger.error("Cell failed: n=%d noise_dims=%d seed=%d", n, noise_dims, seed)
        base.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FAILURES_LOG, "a") as fh:
            fh.write(msg)
        return None


def _cast_row(r: dict[str, str]) -> dict[str, Any]:
    """Re-cast numeric fields read back from CSV (DictReader yields strings)."""
    int_cols = {"seed", "n_obs", "noise_dims", "p_features", "n_iters"}
    float_cols = {
        "fixed_k", "fit_time_s", "final_loss",
        "grassmann_distance", "projection_distance", "vector_correlation",
    } | {
        f"{split}_mse_{ds}" for split in ("train", "test") for ds in base.DOWNSTREAM_MODELS
    }
    out: dict[str, Any] = dict(r)
    for c in int_cols:
        if c in out:
            out[c] = int(out[c])
    for c in float_cols:
        if c in out:
            out[c] = float(out[c])
    return out


def main(smoke_test: bool = False) -> None:
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    n_values = [100, 200] if smoke_test else base.N_VALUES
    noise_dims = [0, 40] if smoke_test else base.NOISE_DIMS
    seeds = list(range(2)) if smoke_test else base.SEEDS

    cells = [(n, nd, s) for n in n_values for nd in noise_dims for s in seeds]
    logger.info("Fitting L1MAVE on %d Friedman1 cells", len(cells))

    base.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    open(FAILURES_LOG, "w").close()

    new_rows: list[dict[str, Any]] = []
    t_start = time.time()

    if _have_tqdm:
        pbar = tqdm(total=len(cells), desc="l1mave-friedman1", unit="fit")

    for n, nd, seed in cells:
        t_cell = time.time()
        row = run_one_cell(n, nd, seed)
        elapsed = time.time() - t_cell
        if row is not None:
            new_rows.append(row)
        msg = f"n={n:3d}  noise_dims={nd:3d}  seed={seed:2d}  ({elapsed:.1f}s)"
        if _have_tqdm:
            pbar.update(1)
            pbar.set_postfix_str(msg)
        else:
            logger.info(msg)

    if _have_tqdm:
        pbar.close()

    elapsed = time.time() - t_start
    logger.info(
        "Fit %d/%d L1MAVE cells in %.1fs (%.1f min)",
        len(new_rows), len(cells), elapsed, elapsed / 60,
    )

    suffix = "_smoke" if smoke_test else ""
    raw_path = base.RESULTS_DIR / f"synthetic_unified_v1{suffix}.csv"
    summary_path = base.RESULTS_DIR / f"synthetic_unified_v1_summary{suffix}.csv"

    existing_rows: list[dict[str, Any]] = []
    if raw_path.exists():
        with open(raw_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("method") == "L1MAVE":
                    continue  # drop stale L1MAVE rows from a prior partial run
                existing_rows.append(_cast_row(r))

    all_rows = existing_rows + new_rows
    base._write_csv(all_rows, raw_path)

    summary = base.aggregate_summary(all_rows)
    base._write_csv(summary, summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit L1MAVE on the existing Friedman1 grid and merge results."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: n=[100,200], 2 noise levels, 2 seeds only (writes *_smoke files).",
    )
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
