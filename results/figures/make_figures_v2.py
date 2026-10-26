#!/usr/bin/env python3
"""Publication figures v2 for SP-MAVE synthetic experiments (JCGS submission).

Extends make_figures.py to a 4-method comparison (adds L1MAVE) and adds
two new DGP figures for the non-axis-aligned subspace-recovery DGPs.

Reads  results/synthetic_unified_v1_summary.csv   (Friedman1, 4 methods
         after run_l1mave_on_friedman1.py merged in L1MAVE)
       results/synthetic_dgp_v1_summary.csv       (3 rotated DGPs, 4 methods)
Saves  results/figures/fig1_grassmann_by_n.{pdf,png}
       results/figures/fig2_spmave_advantage.{pdf,png}
       results/figures/fig3_regime_heatmap.{pdf,png}
       results/figures/fig4_li1991_grassmann.{pdf,png}
       results/figures/fig5_rotated_grassmann.{pdf,png}
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent          # results/figures/
UNIFIED_CSV  = HERE.parent / "synthetic_unified_v1_summary.csv"
DGP_CSV      = HERE.parent / "synthetic_dgp_v1_summary.csv"
OUT_DIR      = HERE
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
    "L1MAVE":            "#378ADD",
    "SPMAVE":            "#1D9E75",
}
LINESTYLES = {
    "ClassicalMAVE":     "-",
    "ClassicalMAVE_kNN": "--",
    "L1MAVE":            ":",
    "SPMAVE":            "-",
}
MARKERS = {
    "ClassicalMAVE":     "o",
    "ClassicalMAVE_kNN": "s",
    "L1MAVE":            "^",
    "SPMAVE":             "o",
}
DISPLAY = {
    "ClassicalMAVE":     "ClassicalMAVE",
    "ClassicalMAVE_kNN": "ClassicalMAVE (kNN)",
    "L1MAVE":            "L1MAVE",
    "SPMAVE":            "SP-MAVE",
}
METHODS_4 = ["ClassicalMAVE", "ClassicalMAVE_kNN", "L1MAVE", "SPMAVE"]
METHODS_3 = ["ClassicalMAVE", "ClassicalMAVE_kNN", "SPMAVE"]
N_VALUES  = [100, 200, 500]
P_PLOT    = [15, 25, 45, 85, 125]   # p=5 excluded (Grassmann ≈ 0 for all, Friedman1 only)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
unified_df = pd.read_csv(UNIFIED_CSV)
unified_df["n_obs"]      = unified_df["n_obs"].astype(int)
unified_df["p_features"] = unified_df["p_features"].astype(int)

dgp_df = pd.read_csv(DGP_CSV)
dgp_df["n_obs"]      = dgp_df["n_obs"].astype(int)
dgp_df["p_features"] = dgp_df["p_features"].astype(int)

if "L1MAVE" not in unified_df["method"].unique():
    raise RuntimeError(
        "L1MAVE missing from synthetic_unified_v1_summary.csv — "
        "run run_l1mave_on_friedman1.py first."
    )


def get_ms(df: pd.DataFrame, method: str, n: int, p: int, metric: str,
           dgp_name: str | None = None) -> tuple[float, float]:
    """Return (mean, se) for a given (method, n, p, metric[, dgp_name]) cell."""
    mask = (df["method"] == method) & (df["n_obs"] == n) & (df["p_features"] == p)
    if dgp_name is not None:
        mask &= (df["dgp_name"] == dgp_name)
    row = df[mask]
    if row.empty:
        return math.nan, math.nan
    return float(row[f"{metric}_mean"].iloc[0]), float(row[f"{metric}_se"].iloc[0])


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures 1 / 4 / 5 — Three-panel Grassmann distance line chart (4 methods)
# ---------------------------------------------------------------------------
def make_grassmann_by_n(
    df: pd.DataFrame,
    stem: str,
    dgp_name: str | None = None,
    figsize: tuple[float, float] = (8, 3.2),
    caption_note: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize)
    fig.subplots_adjust(wspace=0.10, bottom=0.34)

    for ax_idx, (ax, n) in enumerate(zip(axes, N_VALUES)):
        for method in METHODS_4:
            means, ses = [], []
            for p in P_PLOT:
                mu, se = get_ms(df, method, n, p, "grassmann_distance", dgp_name)
                means.append(mu)
                ses.append(se)
            color = COLORS[method]

            ax.plot(P_PLOT, means,
                    color=color, ls=LINESTYLES[method], lw=1.5,
                    marker=MARKERS[method], markersize=5, zorder=3)
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

    legend_handles = [
        Line2D([0], [0],
               color=COLORS[m], lw=1.5, ls=LINESTYLES[m],
               marker=MARKERS[m], markersize=5, label=DISPLAY[m])
        for m in METHODS_4
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9, frameon=False,
    )

    if caption_note:
        fig.suptitle(caption_note, fontsize=8, style="italic", y=1.03)

    save(fig, stem)


# ---------------------------------------------------------------------------
# Figure 2 — SP-MAVE / L1MAVE advantage bar chart (Friedman1)
# ---------------------------------------------------------------------------
def make_fig2() -> None:
    BAR_COLORS = {100: "#888780", 200: "#378ADD", 500: "#1D9E75"}
    BAR_W      = 0.11   # width of a single (SP or L1) bar
    INNER_GAP  = 0.015  # gap between SP and L1 bars within the same n-group
    GROUP_GAP  = 0.04   # gap between different n-groups
    GROUP_W    = 2 * BAR_W + INNER_GAP
    TOTAL_W    = 3 * GROUP_W + 2 * GROUP_GAP

    x = np.arange(len(P_PLOT))

    # Left edge of each n-group, centred on the tick
    group_lefts = []
    cursor = -TOTAL_W / 2
    for _ in N_VALUES:
        group_lefts.append(cursor)
        cursor += GROUP_W + GROUP_GAP

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    fig.subplots_adjust(bottom=0.22, right=0.72)

    for gi, n in enumerate(N_VALUES):
        sp_adv, sp_se, l1_adv, l1_se = [], [], [], []
        for p in P_PLOT:
            knn_m, knn_s = get_ms(unified_df, "ClassicalMAVE_kNN", n, p, "grassmann_distance")
            sp_m,  sp_s  = get_ms(unified_df, "SPMAVE",            n, p, "grassmann_distance")
            l1_m,  l1_s  = get_ms(unified_df, "L1MAVE",            n, p, "grassmann_distance")
            sp_adv.append(knn_m - sp_m)
            sp_se.append(math.sqrt(knn_s**2 + sp_s**2))
            l1_adv.append(knn_m - l1_m)
            l1_se.append(math.sqrt(knn_s**2 + l1_s**2))

        left  = group_lefts[gi]
        sp_x  = x + left + BAR_W / 2
        l1_x  = x + left + BAR_W + INNER_GAP + BAR_W / 2
        color = BAR_COLORS[n]

        ax.bar(sp_x, sp_adv, width=BAR_W, color=color, edgecolor=color,
               linewidth=0.5, zorder=3)
        ax.errorbar(sp_x, sp_adv, yerr=sp_se, fmt="none", ecolor="#444444",
                    alpha=0.75, capsize=3, capthick=0.8, lw=0.8, zorder=4)

        ax.bar(l1_x, l1_adv, width=BAR_W, color=color, edgecolor=color,
               linewidth=0.5, hatch="//", zorder=3)
        ax.errorbar(l1_x, l1_adv, yerr=l1_se, fmt="none", ecolor="#444444",
                    alpha=0.75, capsize=3, capthick=0.8, lw=0.8, zorder=4)

    ax.axhline(0, color="black", linewidth=0.8, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(P_PLOT)
    ax.set_xlabel("p (ambient dimension)", fontsize=10)
    ax.set_ylabel("Grassmann distance advantage vs. kNN-MAVE", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, color="gray")
    ax.set_axisbelow(True)

    legend_handles = [
        Rectangle((0, 0), 1, 1, color=BAR_COLORS[n], label=f"SP-MAVE adv. (n={n})")
        for n in N_VALUES
    ] + [
        Rectangle((0, 0), 1, 1, facecolor="#888780", edgecolor="#888780",
                  hatch="//", label="L1MAVE adv."),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, frameon=False, borderaxespad=0)

    save(fig, "fig2_spmave_advantage")


# ---------------------------------------------------------------------------
# Figure 3 — p/n regime heatmap (unchanged: 3-method, Friedman1)
# ---------------------------------------------------------------------------
def make_fig3() -> None:
    cmap = LinearSegmentedColormap.from_list(
        "sp_div", ["#D85A30", "white", "#1D9E75"], N=256
    )

    data = np.full((len(N_VALUES), len(P_PLOT)), math.nan)
    for i, n in enumerate(N_VALUES):
        for j, p in enumerate(P_PLOT):
            knn_m, _ = get_ms(unified_df, "ClassicalMAVE_kNN", n, p, "grassmann_distance")
            sp_m,  _ = get_ms(unified_df, "SPMAVE",            n, p, "grassmann_distance")
            data[i, j] = knn_m - sp_m

    vmax = float(np.nanmax(np.abs(data)))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    sm   = ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(-0.5, len(P_PLOT),   1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(N_VALUES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(len(P_PLOT)))
    ax.set_xticklabels([f"p = {p}" for p in P_PLOT], fontsize=9)
    ax.set_yticks(range(len(N_VALUES)))
    ax.set_yticklabels([f"n = {n}" for n in N_VALUES], fontsize=9)

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

    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_label("SP-MAVE advantage (↑ better)", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)

    save(fig, "fig3_regime_heatmap")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_grassmann_by_n(unified_df, "fig1_grassmann_by_n", dgp_name=None)
    make_fig2()
    make_fig3()
    make_grassmann_by_n(
        dgp_df, "fig4_li1991_grassmann", dgp_name="li1991_model1",
        caption_note="Li (1991) model 1 benchmark",
    )
    make_grassmann_by_n(
        dgp_df, "fig5_rotated_grassmann", dgp_name="single_index_rotated",
        caption_note=r"Single-index rotated DGP ($b = p^{-1/2}\mathbf{1}_p$, $r=1$)",
    )
    print("All figures saved to", OUT_DIR)
