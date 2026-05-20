#!/usr/bin/env python3
"""Publication figures for SP-MAVE synthetic experiments (JCGS submission).

Reads  results/synthetic_unified_v1_summary.csv
Saves  results/figures/fig1_grassmann_by_n.{pdf,png}
       results/figures/fig2_spmave_advantage.{pdf,png}
       results/figures/fig3_regime_heatmap.{pdf,png}
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE        = Path(__file__).resolve().parent          # results/figures/
SUMMARY_CSV = HERE.parent / "synthetic_unified_v1_summary.csv"
OUT_DIR     = HERE
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
})

# ---------------------------------------------------------------------------
# Aesthetics constants
# ---------------------------------------------------------------------------
COLORS = {
    "ClassicalMAVE":     "#D85A30",
    "ClassicalMAVE_kNN": "#888780",
    "SPMAVE":            "#1D9E75",
}
LINESTYLES = {
    "ClassicalMAVE":     "-",
    "ClassicalMAVE_kNN": "--",
    "SPMAVE":            "-",
}
DISPLAY = {
    "ClassicalMAVE":     "ClassicalMAVE",
    "ClassicalMAVE_kNN": "ClassicalMAVE (kNN)",
    "SPMAVE":            "SP-MAVE",
}
METHODS  = ["ClassicalMAVE", "ClassicalMAVE_kNN", "SPMAVE"]
N_VALUES = [100, 200, 500]
P_PLOT   = [15, 25, 45, 85, 125]   # p=5 excluded (Grassmann ≈ 0 for all)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
df = pd.read_csv(SUMMARY_CSV)
df["n_obs"]      = df["n_obs"].astype(int)
df["p_features"] = df["p_features"].astype(int)


def get_ms(method: str, n: int, p: int, metric: str) -> tuple[float, float]:
    """Return (mean, se) for a given (method, n, p, metric) cell."""
    row = df[
        (df["method"] == method) &
        (df["n_obs"] == n) &
        (df["p_features"] == p)
    ]
    if row.empty:
        return math.nan, math.nan
    return float(row[f"{metric}_mean"].iloc[0]), float(row[f"{metric}_se"].iloc[0])


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Three-panel Grassmann distance line chart
# ---------------------------------------------------------------------------
def make_fig1() -> None:
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7, 3))
    fig.subplots_adjust(wspace=0.10, bottom=0.32)

    for ax_idx, (ax, n) in enumerate(zip(axes, N_VALUES)):
        for method in METHODS:
            means = [get_ms(method, n, p, "grassmann_distance")[0] for p in P_PLOT]
            ses   = [get_ms(method, n, p, "grassmann_distance")[1] for p in P_PLOT]
            color = COLORS[method]
            ls    = LINESTYLES[method]

            ax.plot(P_PLOT, means,
                    color=color, ls=ls, lw=1.5,
                    marker="o", markersize=5, zorder=3)
            ax.errorbar(P_PLOT, means, yerr=ses,
                        fmt="none", ecolor=color, alpha=0.6,
                        capsize=3, capthick=1.0, lw=1.0, zorder=2)

        ax.set_xticks(P_PLOT)
        ax.set_xlabel("p (ambient dimension)", fontsize=10)
        ax.set_title(f"n = {n}", fontsize=10, fontweight="normal", pad=4)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, color="gray")
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.set_ylabel("Grassmann distance", fontsize=10)

    # Single shared legend below all three panels
    legend_handles = [
        Line2D([0], [0],
               color=COLORS[m], lw=1.5, ls=LINESTYLES[m],
               marker="o", markersize=5, label=DISPLAY[m])
        for m in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9, frameon=False,
    )

    save(fig, "fig1_grassmann_by_n")


# ---------------------------------------------------------------------------
# Figure 2 — SP-MAVE advantage bar chart
# ---------------------------------------------------------------------------
def make_fig2() -> None:
    BAR_COLORS = {100: "#888780", 200: "#378ADD", 500: "#1D9E75"}
    W    = 0.25          # bar width
    GAP  = 0.02          # gap between adjacent bars in a group
    STEP = W + GAP       # center-to-center offset = 0.27

    # Offsets from group centre: n=100 left, n=200 centre, n=500 right
    offsets = {100: -STEP, 200: 0.0, 500: STEP}
    x = np.arange(len(P_PLOT))   # integer group positions 0–4

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.subplots_adjust(bottom=0.20)

    for n in N_VALUES:
        advantages, se_diffs = [], []
        for p in P_PLOT:
            knn_m, knn_s = get_ms("ClassicalMAVE_kNN", n, p, "grassmann_distance")
            sp_m,  sp_s  = get_ms("SPMAVE",            n, p, "grassmann_distance")
            advantages.append(knn_m - sp_m)
            se_diffs.append(math.sqrt(knn_s**2 + sp_s**2))

        xs    = x + offsets[n]
        color = BAR_COLORS[n]
        ax.bar(xs, advantages,
               width=W, color=color, edgecolor=color,
               linewidth=0.5, label=f"n = {n}", zorder=3)
        ax.errorbar(xs, advantages, yerr=se_diffs,
                    fmt="none", ecolor="#444444", alpha=0.75,
                    capsize=3, capthick=0.8, lw=0.8, zorder=4)

    ax.axhline(0, color="black", linewidth=0.8, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(P_PLOT)
    ax.set_xlabel("p (ambient dimension)", fontsize=10)
    ax.set_ylabel("Grassmann distance advantage of SP-MAVE", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, color="gray")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    save(fig, "fig2_spmave_advantage")


# ---------------------------------------------------------------------------
# Figure 3 — p/n regime heatmap
# ---------------------------------------------------------------------------
def make_fig3() -> None:
    # Custom diverging colormap: coral (kNN wins) → white → teal (SP-MAVE wins)
    cmap = LinearSegmentedColormap.from_list(
        "sp_div", ["#D85A30", "white", "#1D9E75"], N=256
    )

    # Build advantage matrix: rows = n (top→bottom: 100, 200, 500), cols = p
    data = np.full((len(N_VALUES), len(P_PLOT)), math.nan)
    for i, n in enumerate(N_VALUES):
        for j, p in enumerate(P_PLOT):
            knn_m, _ = get_ms("ClassicalMAVE_kNN", n, p, "grassmann_distance")
            sp_m,  _ = get_ms("SPMAVE",            n, p, "grassmann_distance")
            data[i, j] = knn_m - sp_m

    vmax = float(np.nanmax(np.abs(data)))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    sm   = ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # White cell gridlines via minor ticks
    ax.set_xticks(np.arange(-0.5, len(P_PLOT),   1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(N_VALUES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # Major tick labels
    ax.set_xticks(range(len(P_PLOT)))
    ax.set_xticklabels([f"p = {p}" for p in P_PLOT], fontsize=9)
    ax.set_yticks(range(len(N_VALUES)))
    ax.set_yticklabels([f"n = {n}" for n in N_VALUES], fontsize=9)

    # Cell annotations: value (line 1) + p/n (line 2)
    for i, n in enumerate(N_VALUES):
        for j, p in enumerate(P_PLOT):
            val = data[i, j]
            if math.isnan(val):
                continue
            rgba      = sm.to_rgba(val)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if luminance < 0.45 else "#333333"
            pn = p / n
            ax.text(j, i - 0.17, f"{val:.3f}",
                    ha="center", va="center", fontsize=9, color=tc)
            ax.text(j, i + 0.22, f"p/n = {pn:.2f}",
                    ha="center", va="center", fontsize=7.5, color=tc)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_label("SP-MAVE advantage (↑ better)", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    # Remove all box spines (imshow re-enables them)
    for spine in ax.spines.values():
        spine.set_visible(False)

    save(fig, "fig3_regime_heatmap")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    print("All figures saved to", OUT_DIR)
