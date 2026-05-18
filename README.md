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
sp_mave/
  __init__.py        Package init
  types.py           DatasetBundle, MAVEResult dataclasses
  config.py          YAML config loading with defaults and validation
  datasets.py        Dataset loaders (Friedman-1; extensible registry)
  preprocessing.py   Standardisation and train/test splitting
  weights.py         Gaussian kernel weight computation
  proximities.py     Cross-fitted RF proximity + top-k + softmax
  mave.py            MAVEBase (shared WLS + OPG) and ClassicalMAVE
  spmave.py          SPMAVE estimator
  metrics.py         Subspace distance metrics
  evaluation.py      Downstream prediction pipeline
  plots.py           Weight distribution and sanity study figures
  utils.py           Orthonormalisation, dimension checks, logging

configs/
  default.yaml            Base defaults
  friedman_dev.yaml       Fast development config
  friedman_final.yaml     Full experiment config
  sanity_noise_study.yaml Noise ablation config

run_experiment.py     Main experiment CLI
run_sanity_study.py   Noise ablation study CLI
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python >= 3.10.  Dependencies: numpy, scipy, scikit-learn, matplotlib, pyyaml.

## Usage

### Single experiment

```bash
python run_experiment.py --config configs/friedman_dev.yaml
python run_experiment.py --config configs/friedman_final.yaml
```

### Sanity study (0 vs 20 noise features)

```bash
python run_sanity_study.py --config configs/sanity_noise_study.yaml
```

### Adding a new dataset

1. Write a loader function in `sp_mave/datasets.py`:

```python
def load_my_dataset(**kwargs) -> DatasetBundle:
    X, y = ...  # load your data
    return DatasetBundle(
        X=X, y=y,
        feature_names=[...],
        true_subspace=None,  # or provide if known
        metadata={"source": "..."},
    )
```

2. Register it in the `REGISTRY` dict at the bottom of `datasets.py`:

```python
REGISTRY["my_dataset"] = load_my_dataset
```

3. Create a config file referencing it:

```yaml
dataset:
  name: my_dataset
  # any kwargs your loader accepts
```

4. Run:

```bash
python run_experiment.py --config configs/my_dataset.yaml
```

## Outputs

| Path | Contents |
|------|----------|
| `results/figures/weight_distributions.png` | Histograms of nonzero weights per method |
| `results/figures/raw_proximities_SPMAVE.png` | RF proximity distribution before softmax |
| `results/figures/sanity_noise_study.png` | Metrics vs noise dimension |
| `results/tables/comparison.csv` | Method comparison table |
| `results/summaries/experiment_summary.json` | Full results + config |
| `results/summaries/sanity_study_summary.json` | Sanity study results |

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
