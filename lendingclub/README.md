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

## Results (full sample, n=2000; test AUC, mean over 20 seeds)

| Method | AUC ↑ |
|---|---|
| SP-MAVE (3-d subspace) | 0.669 |
| Classical MAVE (3-d subspace) | 0.669 |
| Classical MAVE (kNN) (3-d subspace) | 0.653 |
| XGBoost (all 94 features) | 0.666 |

The signal in this task is weak: every method sits around AUC 0.65–0.67.
SP-MAVE significantly beats kNN-MAVE on AUC (Wilcoxon p=0.010) and ties Classical MAVE (p=0.99).
It also fits faster than Classical MAVE (about 18 s vs 122 s per fit).
The 3-d subspace matches XGBoost on the full 94 features.
Full tests are in `results/lendingclub_wilcoxon.csv`.

## Files
- `run_lendingclub_experiment.py` — main experiment and shared `load_data` / `make_split`
- `run_k_sensitivity.py` — neighbourhood-size sensitivity
- `run_xgboost_baseline.py` — full-feature baselines
- `results/` — canonical CSVs; smoke-test outputs and logs are git-ignored
