# SP-MAVE Synthetic Experiments

SP-MAVE is a sufficient dimension reduction method that replaces the Gaussian kernel weights in classical MAVE with Random Forest leaf-co-occurrence proximities computed via cross-fitting. These experiments evaluate whether the RF proximity mechanism improves subspace recovery over classical kNN-MAVE as the number of observations n and noise dimensionality p vary, using the Friedman-1 benchmark (true intrinsic dimension d=5).

## Canonical result files

| File | Experiment | n | p values | Seeds | Key design choices |
|---|---|---|---|---|---|
| `synthetic_noise_scaling_v2.csv` | Noise-scaling (appendix) | 100 | 5, 15, 25, 45, 85, 125 | 20 | k=20 fixed for SP-MAVE; k CV-selected for kNN-MAVE (asymmetric — appendix only) |
| `synthetic_noise_scaling_v2_summary.csv` | " | " | " | " | " |
| `synthetic_noise_scaling_v2_wilcoxon.csv` | " | " | " | " | " |
| `synthetic_noise_scaling_v3_fixed.csv` | Noise-scaling (primary) | 100 | 5, 15, 25, 45, 85, 125 | 20 | k=20 fixed for both kNN-MAVE and SP-MAVE; max_iters=300; convergence on delta_PB |
| `synthetic_noise_scaling_v3_fixed_summary.csv` | " | " | " | " | " |
| `synthetic_noise_scaling_v3_fixed_wilcoxon.csv` | " | " | " | " | " |
| `synthetic_n_scaling_v1.csv` | n-scaling | 200, 500 | 45, 85, 125 | 20 | k=floor(0.20·n) for both kNN-MAVE and SP-MAVE; max_iters=300; convergence on delta_PB |
| `synthetic_n_scaling_v1_summary.csv` | " | " | " | " | " |
| `synthetic_n_scaling_v1_wilcoxon.csv` | " | " | " | " | " |

| `synthetic_unified_v1.csv` (+ `_summary`, `_wilcoxon`) | Friedman-1 grid, 4 methods (incl. L1MAVE via `run_l1mave_on_friedman1.py`) | see script | | 20 | Feeds fig1-fig3 |
| `synthetic_dgp_v1.csv` (+ `_summary`, `_wilcoxon`) | Non-axis-aligned (rotated) true-subspace DGPs | see script | | | Feeds fig4-fig5 |

LendingClub results live in [`../lendingclub/results/`](../lendingclub/results/).

## Key design decisions

- **Fixed k, same for both kNN methods**: `k=20` at n=100; `k=floor(0.20·n)` at n=200,500. This isolates the proximity mechanism (Gaussian distance vs RF proximity) as the sole variable between ClassicalMAVE_kNN and SP-MAVE.
- **Convergence criterion**: primary stopping rule is relative change in the MAVE loss (`rel_change < tol=1e-6`). The diagnostic warning reports `delta_PB = ||B_new @ B_new.T − B_old @ B_old.T||_F`, which is invariant to sign flips and column permutations of B (unlike the naive `||B_new − B_old||_F` used in earlier versions).
- **max_iters=300**: gives high-p, high-n runs room to converge on the loss criterion without hitting the iteration cap prematurely.
- **RF cross-fitting**: 5-fold, 200 trees, `min_samples_leaf=5`, `max_features="sqrt"`, softmax temperature τ=1.0.

## Regime finding

SP-MAVE improves over kNN-MAVE on Grassmann distance when p/n ≲ 0.25–0.43, with the advantage growing as n increases (RF proximity estimates improve with more data).

## Runners

| Script | Produces |
|---|---|
| `run_synthetic_noise_scaling_v3_fixed.py` | `synthetic_noise_scaling_v3_fixed.*` |
| `run_synthetic_n_scaling_v1.py` | `synthetic_n_scaling_v1.*` |
| `run_synthetic_unified_v1.py`, `run_l1mave_on_friedman1.py` | `synthetic_unified_v1.*` |
| `run_synthetic_dgp_v1.py` | `synthetic_dgp_v1.*` |
| `figures/make_figures.py`, `figures/make_figures_v2.py` | `figures/fig1-fig5` |

## Next phase

Real-data results are written to `results/real_data/` (git-ignored). Bundled datasets are in `data/`.
