# Air Pollution Forecasting: Method-Swap Replication (Necula 2025 → Ensemble LGBM + Theta)

Reproduction-with-swap of:

> Necula, C., Hauer, M., Fotache, D., Hurbean, E. (2025). *Advanced Hybrid Models for Air Pollution Forecasting: Combining SARIMA and BiLSTM Architectures.* Electronics 14(3), 549, MDPI. https://doi.org/10.3390/electronics14030549

Coursework: **Algorithms & Complexity**, S2 Informatika, Universitas Nusa Mandiri.

## Final result (locked 16 Oct 2026)

The paper's full method (SARIMA s=12 + BiLSTM correction) is **entirely replaced** by one
SARIMA-free, neural-free method:

**`ens_lgbm_theta` — equal-weight average of LGBM (direct tabular learner on lag/calendar/weekly-Fourier features) and Theta (classical trend + weekly seasonal extrapolation).**

| Metric (RMSE, Box-Cox scale) | Value |
|---|---|
| Aggregate lead vs paper's full method (9 cells) | **−5.46%** |
| Cells won (of 9 = 3 pollutants × horizons 1/7/30) | **7/9** (5 wins statistically significant, Diebold–Mariano) |
| Statistically significant losses | **0** |
| Largest edge | PM2.5, 30-day horizon: **−20.1%** |
| Honest weakness | NO2 short horizon (h=1/7) stays slightly behind (not significant) |
| Compute | CPU-only laptop; full bake-off ±17 min; no GPU |

Full protocol: 5 rolling folds (720-day test window, shifted 120), 24 forecasting origins
per fold, horizons 1/7/30, identical masks for every arm, Diebold–Mariano tests.
The paper's BiLSTM was re-run on **all 5 folds** (the paper reports only fold-0).

## Repository map

| Path | Content |
|---|---|
| `data/transformed_data.csv` | Romania air quality, 1,826 daily rows (NO2/PM10/PM2.5, Box-Cox). Source: [elephant2015/Code4PollutantsBiLSTM](https://github.com/elephant2015/Code4PollutantsBiLSTM) |
| `reference/` | Verbatim copy of the original paper notebooks (read-only) |
| `notebooks/` | Work notebooks — **`00_final_method_ens_lgbm_theta.ipynb` = final bake-off + readable results (start here)** · `unused/` = superseded replication notebooks + stale training checkpoints (N-BEATS arc, old recipe; kept for history) |
| `scripts/` | One-phase scripts `03`–`19` (each maps to a phase in `03_Catatan/`); `18_fig_ensemble.py` = duel figures, `19_laporan_fase20.py` = formal report |
| `results/` | Frozen CSVs per phase + `figures/` + `run_20.log` |
| `03_Catatan/` | Phase findings (Indonesian), incl. `HASIL_BAKEOFF3.md` (fase 20 verdict) and the formal report `Laporan_Fase20_Ensemble_LGBM_Theta.pdf` |

## How to run

```bash
# readable results notebook (skips training if results/20_bakeoff3_predictions.csv exists):
#   open notebooks/00_final_method_ens_lgbm_theta.ipynb
#   (full run from scratch: delete that CSV, ~17 min CPU)

# regenerate duel figures + formal report:
python scripts/18_fig_ensemble.py
python scripts/19_laporan_fase20.py && soffice --headless --convert-to pdf 03_Catatan/Laporan_Fase20_Ensemble_LGBM_Theta.docx --outdir 03_Catatan
```

Reference BiLSTM arm (all folds) needs a GPU session (Kaggle T4, ~45 min):
`notebooks/unused/05_bilstm_all_folds.ipynb` (or `scripts/13_bilstm_all_folds.py` + `scripts/14_vs_bilstm_all_folds.py`).

## Method-selection trail (why ens_lgbm_theta)

Fase 20 screened 12 SARIMA-free arms + 5 ensembles under the identical protocol
(`results/20_bakeoff3_*.csv`). Runner-up `ens_lgbm_ets` edges the aggregate (−5.64%) but
**loses significantly at NO2 h=1 (p=0.023)** → rejected. `ens_lgbm_theta` has zero significant
losses. Earlier candidates (N-BEATS, SARIMA+LGBM ensemble, router) are documented in
`03_Catatan/` as the negative results behind this choice.
