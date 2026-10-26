# SP-MAVE: Supervised Proximity Minimum Average Variance Estimation

A research-grade Python implementation comparing **Classical MAVE** (Gaussian kernel weights) and **SP-MAVE** (Random Forest proximity weights with cross-fitting) for sufficient dimension reduction.

## Model

Both methods estimate a projection matrix B in R^{p x d} under the sufficient dimension reduction model

    E[Y | X] = g(B^T X)

where g is unknown and d << p.  The goal is to recover the column space of B (the central subspace) without estimating g.

## Methods

### Classical MAVE

Uses Gaussian kernel weights based on Euclidean distance:

    w_ij = exp(-||X_i - X_j||^2 / (2h^2)) / Z_j

Optionally supports top-k truncation (keeping only the k nearest neighbours per anchor and re-normalising weights).

### SP-MAVE

Replaces Gaussian weights with supervised Random Forest proximity weights:

1. **Cross-fitting**: Split training data into K folds.  For each fold, train a RandomForestRegressor on the remaining folds and compute leaf co-occurrence proximities for the held-out observations.
2. **Top-k restriction**: For each anchor, retain only the k neighbours with highest proximity.
3. **Softmax normalisation**: Apply softmax with temperature tau over retained proximities.

## Optimisation

Both methods use the same alternating optimisation structure:

1. **Initialise** B via PCA (top-d principal components).
2. **Local WLS**: For each anchor j, solve a weighted least-squares problem in R^p with Tikhonov regularisation to obtain a local gradient estimate beta_j.
3. **OPG update**: Form the Outer Product of Gradients matrix M = (1/n) sum_j beta_j beta_j^T and set B = top-d eigenvectors of M.
4. **Convergence**: Check relative loss improvement; stop when below tolerance or max iterations reached.

### Approximations

1. **Tikhonov regularisation in WLS**: When the neighbourhood size k is smaller than p+1, the full-space weighted least-squares system is under-determined.  A small ridge penalty (default 1e-6) is applied to the slope coefficients (not the intercept) as a numerical safeguard.

2. **One-step convergence**: Both Classical MAVE (with full-space Gaussian kernel) and SP-MAVE use weights that do not depend on B.  As a result, the full-space WLS produces identical gradient estimates at every iteration and the algorithm converges in one step.  One-step OPG is a consistent estimator of the central subspace.

3. **Cross-fitted proximity asymmetry**: The proximity matrix from K-fold cross-fitting is not exactly symmetric because each observation's proximities come from the forest that excluded its fold.  This is acceptable for anchor-based MAVE where we use each row independently.

4. **True subspace for Friedman-1**: Defined as span{e_1,...,e_5}.  This is the minimal coordinate subspace containing the signal.  The five informative features enter through nonlinear/interaction terms (sin(pi*x1*x2), (x3-0.5)^2, etc.) that cannot be compressed to fewer than five independent linear combinations.

## Project Structure

```
sp_mave/                Core package (importable: pip install -e .)
  types.py              DatasetBundle, MAVEResult dataclasses
  preprocessing.py      Standardisation and train/test splitting
  weights.py            Gaussian and Laplacian kernel weights
  proximities.py        Cross-fitted RF proximity + top-k + softmax
  mave.py               MAVEBase (shared WLS + OPG), ClassicalMAVE, L1MAVE
  spmave.py             SPMAVE estimator
  metrics.py            Subspace distance metrics
  evaluate.py           evaluate_subspace: downstream prediction on B^T X
  evaluation.py         Downstream prediction pipeline
  utils.py              Orthonormalisation, dimension checks, logging

run_synthetic_*.py      Synthetic experiments (Friedman-1 grids, rotated DGPs)
run_l1mave_on_friedman1.py   Adds L1MAVE to the Friedman-1 grid
run_real_data_v1.py, run_real_data_v2.py   Real-data benchmarks (data/*.npz)
run_krr_tuning_robustness.py  Appendix: CV-tuned KRR robustness check

results/                Canonical result CSVs (see results/README_experiments.md)
results/figures/        Publication figures (make_figures.py, make_figures_v2.py)
data/                   Small bundled datasets (.npz / .txt)

lendingclub/            Self-contained LendingClub default-prediction
                        application, with its own README (see below)
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .          # makes `import sp_mave` work from anywhere
```

Requires Python >= 3.9. Core dependencies: numpy, scipy, scikit-learn, pandas.
Figures additionally need matplotlib.

## Running experiments

Each runner is a standalone script; most accept `--smoke-test` for a quick check.

```bash
python run_synthetic_noise_scaling_v3_fixed.py
python run_synthetic_n_scaling_v1.py
python run_synthetic_unified_v1.py
python run_synthetic_dgp_v1.py          # non-axis-aligned (rotated) true subspaces
python run_real_data_v1.py              # then run_real_data_v2.py (appends)
python results/figures/make_figures_v2.py
```

Outputs are written to `results/`. Real-data outputs go to `results/real_data/`,
which is git-ignored; regenerate them with the runners above.

## Applications

- **LendingClub loan-default prediction** (binary response, p=94):
  see [lendingclub/README.md](lendingclub/README.md). It is kept separate from
  the continuous-response experiments above and reuses only the `sp_mave` package.

## Evaluation Metrics

- **Projection distance**: Frobenius norm of the difference of projection matrices.
- **Grassmann distance**: Geodesic on the Grassmann manifold (sqrt of sum of squared principal angles).
- **Vector correlation**: Average cosine of principal angles (1 = perfect alignment).
- **Test MSE**: Downstream regression MSE on held-out test data after dimension reduction.

## Known Limitations

- **Small sample size**: With n=100 and p=25, estimation is inherently noisy.  Results should be interpreted as indicative, not definitive.
- **Bandwidth sensitivity**: The median heuristic for Gaussian MAVE bandwidth is a reasonable default but not optimal.  Results may vary with different bandwidth choices.
- **One-step convergence**: The primary algorithm variant converges in one step (by design).  An iterative variant using reduced-space kernel distances is conceptually possible but not implemented here.
- **Cross-fitting asymmetry**: The RF proximity matrix is not symmetric.  Symmetrising (e.g. averaging P and P^T) is a possible improvement.
- **No automatic tuning**: k and tau are set manually.  Validation-based tuning would strengthen results.
- **Subspace truth for Friedman-1**: The true central subspace is span{e_1,...,e_5} under our definition.  For other definitions of "truth" the metrics would differ.
