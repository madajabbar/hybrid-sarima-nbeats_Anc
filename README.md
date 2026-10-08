# Air Pollution Forecasting: SARIMA + N-BEATS Hybrid (method-swap replication)

Reproduction-with-swap of:

> Necula, C., Hauer, M., Fotache, D., Hurbean, E. (2025). *Advanced Hybrid Models for Air Pollution Forecasting: Combining SARIMA and BiLSTM Architectures.* Electronics 14(3), 549, MDPI. https://doi.org/10.3390/electronics14030549

Original SARIMA+BiLSTM pipeline kept as reference; the **BiLSTM component is replaced by N-BEATS** (Oreshkin et al., 2020) and compared on accuracy (RMSE/MAE/MAPE, horizons 1/7/30 days) **and computational cost** (training time, parameter count) — the accuracy-vs-complexity trade-off.

## Structure
- `data/transformed_data.csv` — original dataset (Box-Cox transformed, 1,826 daily rows, Romania, 2019–2023; PM10/PM2.5/NO2). From [elephant2015/Code4PollutantsBiLSTM](https://github.com/elephant2015/Code4PollutantsBiLSTM).
- `reference/` — verbatim copy of the original notebooks (read-only, unmodified).
- `notebooks/01_baseline_*.ipynb` — original notebooks, paths adapted to run from this repo (Colab/VS Code kernel).
- `notebooks/02_nbeats_swap.ipynb` — our experiment: SARIMA → residual → N-BEATS hybrid, same split & horizons.
- `results/` — metric tables (CSV).

## Run
VS Code + Colab kernel (T4 GPU) or Jupyter:
1. `notebooks/01_baseline_bilstm.ipynb` → reproduce BiLSTM baseline
2. `notebooks/02_nbeats_swap.ipynb` → N-BEATS swap + comparison table

Coursework: Algorithms & Complexity, S2 Informatics, Universitas Nusa Mandiri.
