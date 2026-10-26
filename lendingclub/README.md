# SP-MAVE on LendingClub loan default

Application of SP-MAVE to **binary** default prediction. It is independent of the
continuous-response experiments in the repo root and imports only the shared
`sp_mave` package.

## Setup
The 0/1 label is fed directly into the MAVE local weighted least squares, so the
fit approximates P(default=1 | X) in a 3-dimensional subspace (`R_DIM=3`, `k = 0.20 n`).
Downstream, an RBF `KernelRidge` is fitted on `B^T X` and scored on a held-out 20% split:

- **Brier score** (lower is better) and **AUC-ROC** (higher is better).
- Methods compared: `SPMAVE`, `ClassicalMAVE` (Gaussian kernel), `ClassicalMAVE_kNN`.
- Two regimes, both stratified on `default` (about 78/22 class ratio):
  `subsampled` (n=300, p/n≈0.31) and `full` (n=2000, p/n≈0.047), 20 seeds each.
- Features: 94 preprocessed columns plus a `default` column.

## Data
The cleaned file is not distributed with this repo. Provide a CSV with a 0/1
`default` column and 94 numeric feature columns, and point to it with:

```bash
export LENDINGCLUB_DATA=/path/to/lendingclub_clean.csv
```

## Running
From the repo root (after `pip install -e .`):

```bash
python lendingclub/run_lendingclub_experiment.py --smoke-test   # 1 seed, n=300
python lendingclub/run_lendingclub_experiment.py                # 20 seeds x 2 regimes
python lendingclub/run_k_sensitivity.py                         # k_frac in {0.35,0.50,0.65,0.80}, n=300
python lendingclub/run_xgboost_baseline.py                      # XGBoost / LogisticRegression, all 94 features
```

`run_k_sensitivity.py` and `run_xgboost_baseline.py` reuse `make_split` from the main runner,
so all results are seed-for-seed comparable. XGBoost needs `pip install xgboost`.
Outputs go to `lendingclub/results/`.

## Results (test set, mean over 20 seeds)

| Regime | Method | Brier ↓ | AUC ↑ |
|---|---|---|---|
| n=300 | SP-MAVE | 0.204 | 0.622 |
| n=300 | Classical MAVE | 0.191 | 0.618 |
| n=300 | Classical MAVE (kNN) | 0.197 | 0.594 |
| n=300 | Logistic regression (94 features) | 0.211 | 0.618 |
| n=300 | XGBoost (94 features) | 0.188 | 0.624 |
| n=2000 | SP-MAVE | 0.165 | 0.669 |
| n=2000 | Classical MAVE | 0.163 | 0.669 |
| n=2000 | Classical MAVE (kNN) | 0.165 | 0.653 |
| n=2000 | Logistic regression (94 features) | 0.162 | 0.676 |
| n=2000 | XGBoost (94 features) | 0.165 | 0.666 |

Reading these honestly: the signal in this task is weak (AUC about 0.62–0.68 for every method).
SP-MAVE is not significantly better than Classical MAVE on Brier, and is worse at n=300
(Wilcoxon p=0.006). Its clearest gain is in AUC against kNN-MAVE at n=2000 (p=0.010).
SP-MAVE is also faster to fit than Classical MAVE at n=2000 (about 18 s vs 122 s).
The 3-d subspace is roughly on par with models on all 94 features.
See `results/lendingclub_wilcoxon.csv` and `results/k_sensitivity_summary.csv` for tests and the k sweep.

## Files
- `run_lendingclub_experiment.py` — main experiment and shared `load_data` / `make_split`
- `run_k_sensitivity.py` — neighbourhood-size sensitivity
- `run_xgboost_baseline.py` — full-feature baselines
- `results/` — canonical CSVs; smoke-test outputs and logs are git-ignored
