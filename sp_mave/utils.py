"""Shared utilities: orthonormalisation, dimension checks, logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import svd


def get_logger(
    name: str = "sp_mave",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a configured logger that writes to stderr and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def orthonormalize(B: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the column space of B via thin SVD.

    Parameters
    ----------
    B : np.ndarray, shape (p, d)

    Returns
    -------
    np.ndarray, shape (p, d)  with orthonormal columns.
    """
    U, _, _ = svd(B, full_matrices=False)
    return U


def check_dimensions(
    X: np.ndarray,
    y: np.ndarray,
    B: np.ndarray | None = None,
    context: str = "",
) -> tuple[int, int]:
    """Assert shapes are consistent and return (n, p).

    Raises ValueError with a descriptive message on mismatch.
    """
    prefix = f"[{context}] " if context else ""
    if X.ndim != 2:
        raise ValueError(f"{prefix}X must be 2-d, got shape {X.shape}")
    n, p = X.shape
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(
            f"{prefix}y shape {y.shape} incompatible with X ({n}, {p})"
        )
    if B is not None:
        if B.ndim != 2 or B.shape[0] != p:
            raise ValueError(
                f"{prefix}B shape {B.shape} incompatible with p={p}"
            )
    return n, p


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the *k* largest entries in a 1-d array.

    Uses argpartition for O(n) average complexity.
    """
    if k >= len(values):
        return np.arange(len(values))
    idx = np.argpartition(values, -k)[-k:]
    return idx[np.argsort(values[idx])[::-1]]


def softmax(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature *tau*.

    Parameters
    ----------
    logits : np.ndarray, shape (m,)
    tau : float, positive temperature.

    Returns
    -------
    np.ndarray, shape (m,) summing to 1.
    """
    scaled = logits / tau
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()
