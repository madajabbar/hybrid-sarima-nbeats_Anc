# Air Pollution Forecasting: SARIMA + N-BEATS Hybrid (method-swap replication)

Reproduction-with-swap of:

> Necula, C., Hauer, M., Fotache, D., Hurbean, E. (2025). *Advanced Hybrid Models for Air Pollution Forecasting: Combining SARIMA and BiLSTM Architectures.* Electronics 14(3), 549, MDPI. https://doi.org/10.3390/electronics14030549

Original SARIMA+BiLSTM pipeline kept as reference; the **BiLSTM component is replaced by N-BEATS** (Oreshkin et al., 2020) and compared on accuracy (RMSE/MAE/MAPE, horizons 1/7/30 days) **and computational cost** (training time, parameter count) — the accuracy-vs-complexity trade-off.

## Structure
- `data/transformed_data.csv` — original dataset (Box-Cox transformed, 1,826 daily rows, Romania, 2019–2023; PM10/PM2.5/NO2). From [elephant2015/Code4PollutantsBiLSTM](https://github.com/elephant2015/Code4PollutantsBiLSTM).
- `reference/` — verbatim copy of the original notebooks (read-only, unmodified).
- `notebooks/01_baseline_*.ipynb` — original notebooks, paths adapted to run from this repo (Colab/VS Code kernel).
- `notebooks/02_nbeats_swap.ipynb` — our experiment: SARIMA → residual → N-BEATS hybrid, same split & horizons.
- `notebooks/03_final_recipe.ipynb` — reads the CSVs in `results/` and prints the final tables (no retraining).
- `notebooks/04_end_to_end.ipynb` — full pipeline from raw data in one runnable notebook: Box-Cox → SARIMA (weekly) → residual → Ridge correction → gated volume (alpha) → RMSE comparison, step by step with explanations. ~30 s for one pollutant/fold.
- `notebooks/05_bilstm_all_folds.ipynb` — **GPU (Kaggle/Colab)**: runs the reference BiLSTM arm on all 5 folds x 3 pollutants (`scripts/13_bilstm_all_folds.py`, ~45 min on T4), then `scripts/14_vs_bilstm_all_folds.py` produces the head-to-head table vs our recipe on identical folds/origins.
- `results/` — metric tables (CSV).

## Run
CPU only (laptop), in order:
1. `python scripts/12_full_final.py` → frozen recipe, 9 cells x 5 folds (~4 min)
2. `notebooks/04_end_to_end.ipynb` → step-by-step walkthrough of the same pipeline (~30 s)
3. `notebooks/03_final_recipe.ipynb` → print final tables from the CSVs (no retraining)

GPU (Kaggle T4 / Colab, ~45 min) — head-to-head vs the reference paper:
4. `notebooks/05_bilstm_all_folds.ipynb` → reference BiLSTM arm on all folds, then the comparison table
   (equivalently: `python scripts/13_bilstm_all_folds.py --tune off` then `python scripts/14_vs_bilstm_all_folds.py`)

Original reference notebooks (unchanged): `notebooks/01_baseline_bilstm.ipynb`, `01_baseline_attention_final.ipynb`, `01_baseline_fourier_final.ipynb`, `02_nbeats_swap.ipynb`.

Coursework: Algorithms & Complexity, S2 Informatics, Universitas Nusa Mandiri.
