"""YAML-based experiment configuration with defaults and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "dataset": {
        "name": "friedman1",
        "n_samples": 100,
        "n_noise": 20,
        "random_state": 42,
    },
    "model": {
        "d": 5,
        "bandwidth": "auto",
        "k": 20,
        "tau": 0.1,
        "max_iters": 50,
        "tol": 1e-6,
        "ridge_lambda": 1e-6,
        "init_method": "pca",
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "random_state": 42,
    },
    "crossfit": {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
    "evaluation": {
        "downstream_models": ["linear"],
        "metrics": ["projection_distance", "grassmann_distance", "test_mse"],
        "save_figures": True,
    },
    "split": {
        "test_size": 0.3,
        "random_state": 42,
    },
    "methods": ["classical_mave", "sp_mave"],
    "tuning": {
        "enabled": False,
        "val_size": 0.2,
        "k_candidates": [10, 15, 20, 25, 30],
        "tau_candidates": [0.05, 0.1, 0.2, 0.5],
        "random_state": 42,
        "max_iters": 50,
    },
    "multi_seed": {
        "enabled": False,
        "n_seeds": 1,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and merge with defaults.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user_cfg)

    # Backward compatibility: downstream_model (singular) -> downstream_models (plural)
    eval_cfg = cfg["evaluation"]
    if "downstream_model" in eval_cfg and "downstream_models" not in user_cfg.get("evaluation", {}):
        eval_cfg["downstream_models"] = [eval_cfg.pop("downstream_model")]
    elif "downstream_model" in eval_cfg:
        eval_cfg.pop("downstream_model", None)

    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Basic sanity checks on a resolved config."""
    d = cfg["model"]["d"]
    if not isinstance(d, int) or d < 1:
        raise ValueError(f"model.d must be a positive integer, got {d}")
    k = cfg["model"]["k"]
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"model.k must be a positive integer, got {k}")
    tau = cfg["model"]["tau"]
    if not isinstance(tau, (int, float)) or tau <= 0:
        raise ValueError(f"model.tau must be positive, got {tau}")
    tol = cfg["model"]["tol"]
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError(f"model.tol must be positive, got {tol}")
    ts = cfg["split"]["test_size"]
    if not (0 < ts < 1):
        raise ValueError(f"split.test_size must be in (0, 1), got {ts}")
    n_splits = cfg["crossfit"]["n_splits"]
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"crossfit.n_splits must be >= 2, got {n_splits}")
