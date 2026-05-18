# MAVE paper experiment tracks

## What was added

- **Datasets** in `sp_mave/datasets.py`: `california_housing`, `combined_cycle_power_plant` (UCI 294 CSV), `semi_synthetic_real_x` (real `X`, synthetic `y` after split).
- **Semi-synthetic pipeline** in `sp_mave/preprocessing.py` + `sp_mave/semisynthetic.py`: train/test split on raw `X`, `StandardScaler` fit on train only, orthonormal `B_true` (QR of Gaussian, seed `b_true_seed`), `Z = X_std @ B_true`, `y = g_friedman_adapted(Z) + ε` on each split.
- **Results layout** via `experiment.results_track` / `results_slug` → `results/mave/<track>/<slug>/` (`sp_mave/experiment_paths.py`). Legacy configs omit these keys and still use `results/<dataset.name>/`.
- **Tuning**: `tune_k_classical_topk` in `sp_mave/tuning.py`; `tuning.matched_k_for_classical_topk` reuses SP-MAVE’s tuned `k` for classical top-k.
- **Paper labels**: `method_paper` column (`sp_mave/method_labels.py`); figures `paper_test_mse_*.png`, `paper_subspace_metrics.png` (`sp_mave/plots_paper.py`).
- **Orchestrator**: `run_mave_tracks.py` → `results/mave/master_summary.csv`.

## Exact datasets

| Track            | Registry name              | Source |
|------------------|----------------------------|--------|
| Semi-synthetic   | `semi_synthetic_real_x`    | `base: california_housing` (sklearn); same pattern can use `ccpp`. |
| Energy           | `combined_cycle_power_plant` | https://archive.ics.uci.edu/static/public/294/data.csv (AT, V, AP, RH → PE). |
| Housing          | `california_housing`       | `sklearn.datasets.fetch_california_housing`. |

## Semi-synthetic construction

- `g_friedman_adapted(Z)` for `d_true=3`: `10*sin(π*z1*z2) + 20*(z3-0.5)^2 + 10*z1 + 5*z2`.
- `B_true`: `p×d_true` orthonormal from QR of `N(0,1)`, RNG `numpy.random.default_rng(b_true_seed)`.
- Noise: independent `N(0, noise_std^2)` on train and test after scaling.

## Preprocessing

- Real tracks: `split_bundle` → train/test split, scaler fit on train only (unchanged).
- `max_samples` subsamples rows before splitting (memory cap for dense `n×n` weights).

## Limitations

- Iterative MAVE / SP-MAVE use **dense** `(n,n)` weights; **full** California (~20k) is not targeted—configs cap `max_samples`.
- Semi-synthetic `y` depends on **scaled** `X` with train-only scaler; `true_subspace` is **B_true in scaled space** — compare estimated `B` on the same scaled training data.
- **OPG** experiments remain under `results/opg/`, not merged into MAVE paper CSVs.

## Suggested paper placement

- **Main text:** semi-synthetic + energy (CCPP).
- **Appendix / sanity:** California housing real `y`.

## Configs

- `configs/mave_semi_synthetic_california.yaml`
- `configs/mave_energy_ccpp.yaml`
- `configs/mave_housing_california.yaml`

Run: `python run_mave_tracks.py` or `python run_experiment.py --config <yaml>`.
