"""Visualisation of weight distributions and experiment results.

Produces interpretable, non-decorative figures suitable for a methods
paper appendix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sp_mave.types import MAVEResult


def plot_weight_histograms(
    results: dict[str, MAVEResult],
    save_path: str | Path | None = None,
    bins: int = 50,
) -> None:
    """Side-by-side histograms of nonzero weight values for each method.

    Parameters
    ----------
    results : dict mapping method name -> MAVEResult.
    save_path : if provided, save figure to this path.
    bins : number of histogram bins.
    """
    methods = list(results.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4),
                             squeeze=False)
    axes = axes.ravel()

    for idx, name in enumerate(methods):
        W = results[name].weights
        nonzero = W[W > 0].ravel()
        ax = axes[idx]
        ax.hist(nonzero, bins=bins, edgecolor="black", linewidth=0.5,
                alpha=0.8)
        ax.set_title(f"{name}\n(n_nonzero={len(nonzero):,})", fontsize=10)
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

    fig.suptitle("Distribution of Nonzero Weights", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raw_proximities(
    raw_P: np.ndarray,
    save_path: str | Path | None = None,
    bins: int = 50,
) -> None:
    """Histogram of raw RF proximity values (before softmax).

    Shows the distribution of all nonzero, off-diagonal entries.
    """
    mask = np.ones_like(raw_P, dtype=bool)
    np.fill_diagonal(mask, False)
    vals = raw_P[mask & (raw_P > 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=bins, edgecolor="black", linewidth=0.5, alpha=0.8,
            color="steelblue")
    ax.set_title("Raw RF Proximity Distribution (before softmax)")
    ax.set_xlabel("Proximity value (fraction of shared leaves)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_objective_convergence(
    results: dict[str, MAVEResult],
    save_dir: str | Path | None = None,
) -> None:
    """Per-method and combined objective convergence plots.

    Generates one plot per method (results/figures/objective_convergence_<method>.png)
    plus a combined overlay.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    colors = _method_colors(list(results.keys()))

    for name, res in results.items():
        obj = res.history.get("objective", res.loss_history)
        if not obj:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(len(obj)), obj, marker=".", markersize=3, color=colors[name])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective value")
        ax.set_title(f"Objective Convergence — {name}")
        ax.set_yscale("log")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"objective_convergence_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in results.items():
        obj = res.history.get("objective", res.loss_history)
        if obj:
            ax.plot(range(len(obj)), obj, marker=".", markersize=3,
                    label=f"{name} ({len(obj)} iters)", color=colors[name])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("Objective Convergence — All Methods")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "objective_convergence_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subspace_change(
    results: dict[str, MAVEResult],
    save_dir: str | Path | None = None,
) -> None:
    """Per-method and combined ||B_new - B_old||_F plots.

    Generates one plot per method (results/figures/subspace_change_<method>.png)
    plus a combined overlay.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    colors = _method_colors(list(results.keys()))

    for name, res in results.items():
        db = res.history.get("delta_B", [])
        if not db:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(len(db)), db, marker=".", markersize=3, color=colors[name])
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\|B_{new} - B_{old}\|_F$")
        ax.set_title(f"Subspace Movement — {name}")
        ax.set_yscale("log")
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"subspace_change_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in results.items():
        db = res.history.get("delta_B", [])
        if db:
            ax.plot(range(len(db)), db, marker=".", markersize=3,
                    label=f"{name} ({len(db)} iters)", color=colors[name])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|B_{new} - B_{old}\|_F$")
    ax.set_title("Subspace Movement — All Methods")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "subspace_change_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _method_colors(method_names: list[str]) -> dict[str, str]:
    """Assign consistent colours to method names."""
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return {name: palette[i % len(palette)] for i, name in enumerate(method_names)}


def plot_sanity_study(
    study_results: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> None:
    """Bar chart of subspace-recovery metrics across noise levels.

    Parameters
    ----------
    study_results : list of dicts, each with keys:
        n_noise, method, projection_distance, grassmann_distance, test_mse
    """
    noise_levels = sorted(set(r["n_noise"] for r in study_results))
    methods = sorted(set(r["method"] for r in study_results))

    metric_keys = ["projection_distance", "grassmann_distance", "test_mse"]
    metric_labels = ["Projection Distance", "Grassmann Distance", "Test MSE"]

    fig, axes = plt.subplots(1, len(metric_keys),
                             figsize=(5 * len(metric_keys), 4),
                             squeeze=False)
    axes = axes.ravel()
    bar_width = 0.35

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        for m_idx, method in enumerate(methods):
            vals = [
                next(r[mkey] for r in study_results
                     if r["n_noise"] == nl and r["method"] == method)
                for nl in noise_levels
            ]
            x = np.arange(len(noise_levels)) + m_idx * bar_width
            ax.bar(x, vals, bar_width, label=method, alpha=0.85)
        ax.set_xticks(np.arange(len(noise_levels)) + bar_width / 2)
        ax.set_xticklabels([str(nl) for nl in noise_levels])
        ax.set_xlabel("Number of noise features")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.legend(fontsize=8)

    fig.suptitle("Sanity Study: Effect of Noise Dimensions", fontsize=12,
                 y=1.02)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_scaling_curves(
    agg_rows: list[dict[str, Any]],
    noise_dims: list[int],
    y_keys: list[tuple[str, str, bool]],
    save_dir: str | Path | None = None,
    title_suffix: str = "",
    file_prefix: str = "noise_scaling_",
) -> None:
    """Line plots: metric (mean ± std) vs number of irrelevant noise features.

    Parameters
    ----------
    agg_rows : aggregated dicts with keys noise_dims, method, and
        ``{base}_mean`` / ``{base}_std`` for each metric in *y_keys*.
    noise_dims : sorted x-axis values.
    y_keys : list of (column_base, y-axis label, use_log_scale).
    save_dir : directory for output PNGs (``{file_prefix}<base>.png``).
    title_suffix : appended to each subplot title.
    file_prefix : filename prefix before the metric slug.
    """
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({r["method"] for r in agg_rows})
    colors = _method_colors(methods)

    for base, ylabel, logy in y_keys:
        mean_k = f"{base}_mean"
        std_k = f"{base}_std"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in methods:
            xs: list[int] = []
            means: list[float] = []
            stds: list[float] = []
            for nd in noise_dims:
                row = next(
                    (r for r in agg_rows
                     if r["noise_dims"] == nd and r["method"] == m),
                    None,
                )
                if row is None or mean_k not in row:
                    continue
                xs.append(nd)
                means.append(float(row[mean_k]))
                stds.append(float(row.get(std_k, 0.0)))
            if not xs:
                continue
            xs_arr = np.array(xs, dtype=float)
            m_arr = np.array(means)
            s_arr = np.array(stds)
            ax.errorbar(
                xs_arr, m_arr, yerr=s_arr,
                marker="o", capsize=3, linewidth=1.5, markersize=5,
                label=m, color=colors[m],
            )
        ax.set_xlabel("Number of irrelevant (noise) features")
        ax.set_ylabel(ylabel)
        ttl = f"{ylabel} vs noise dimensions"
        if title_suffix:
            ttl = f"{ttl} {title_suffix}"
        ax.set_title(ttl)
        if logy:
            ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
        ax.set_xticks(noise_dims)
        fig.tight_layout()
        safe = base.replace(" ", "_")
        fig.savefig(
            save_dir / f"{file_prefix}{safe}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
