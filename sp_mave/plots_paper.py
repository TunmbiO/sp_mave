"""Publication-style figures for MAVE comparison experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sp_mave.method_labels import paper_method_name


def plot_test_mse_by_method(
    summary: dict[str, dict[str, Any]],
    methods_order: list[str],
    downstream: str = "linear",
    save_path: str | Path | None = None,
    title: str = "",
) -> None:
    """Bar chart: mean test MSE with error bars (std over seeds)."""
    key_mean = f"test_mse_{downstream}_mean"
    key_std = f"test_mse_{downstream}_std"
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    for m in methods_order:
        e = summary.get(m)
        if not e or key_mean not in e:
            continue
        labels.append(paper_method_name(m))
        means.append(float(e[key_mean]))
        stds.append(float(e.get(key_std, 0.0)))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="0.35", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(f"Test MSE ({downstream})")
    ax.set_title(title or f"Test MSE by method ({downstream})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subspace_metrics_by_method(
    summary: dict[str, dict[str, Any]],
    methods_order: list[str],
    save_path: str | Path | None = None,
    title: str = "",
) -> None:
    """Grouped bars for projection / Grassmann / vector correlation (if present)."""
    metrics = [
        ("projection_distance_mean", "projection_distance_std", "Proj. dist."),
        ("grassmann_distance_mean", "grassmann_distance_std", "Grassmann"),
        ("vector_correlation_mean", "vector_correlation_std", "Vec. corr."),
    ]
    present = [
        (km, ks, lab)
        for km, ks, lab in metrics
        if any(km in summary.get(m, {}) for m in methods_order)
    ]
    if not present:
        return

    n_m = len(present)
    fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 4), squeeze=False)
    for ax, (km, ks, lab) in zip(axes[0], present):
        labels: list[str] = []
        means: list[float] = []
        stds: list[float] = []
        for m in methods_order:
            e = summary.get(m, {})
            if km not in e:
                continue
            labels.append(paper_method_name(m))
            means.append(float(e[km]))
            stds.append(float(e.get(ks, 0.0)))
        if not labels:
            continue
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color="0.45", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(lab)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title or "Subspace recovery (mean ± std)")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
