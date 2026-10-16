# Hasil fase 16 — LightGBM sebagai penambal (gerbang identik)

Protokol resep beku: SARIMA (2,1,1)(1,0,1,7), 5 lipatan (720 hari, geser 120),
gerbang alfa 120-hari per model. LightGBM: 200 pohon, lr 0.05, leaves 15, seed 42.

| polutan | h | SARIMA | Ridge | LightGBM | Δ LGBM vs Ridge | p (DM) |
|---|---|---|---|---|---|---|
| NO2 | 1 | 0.5962 | 0.5925 | 0.6016 | +1.5% | 0.2356 |
| NO2 | 7 | 0.8890 | 0.9017 | 0.8848 | -1.9% | 0.0020 |
| NO2 | 30 | 0.7303 | 0.7040 | 0.6994 | -0.7% | 0.1458 |
| PM10 | 1 | 0.1145 | 0.1114 | 0.1134 | +1.8% | 0.2761 |
| PM10 | 7 | 0.1536 | 0.1499 | 0.1524 | +1.7% | 0.2263 |
| PM10 | 30 | 0.1525 | 0.1514 | 0.1503 | -0.7% | 0.3606 |
| PM25 | 1 | 0.0650 | 0.0638 | 0.0646 | +1.2% | 0.2500 |
| PM25 | 7 | 0.0754 | 0.0753 | 0.0739 | -2.0% | 0.0000 |
| PM25 | 30 | 0.0914 | 0.0891 | 0.0899 | +0.9% | 0.1793 |

Menang LGBM vs Ridge: 4/9 sel; alfa LGBM rata-rata 0.12.

Lampiran: results/16_lgbm_gate_predictions.csv | 16_lgbm_gate_metrics.csv | 16_lgbm_gate_dm.csv
