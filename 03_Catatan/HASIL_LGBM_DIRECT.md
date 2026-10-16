# Hasil fase 17 — LightGBM telanjang (tanpa SARIMA)

Ablasion: LGBM meramal level langsung; fitur sama dg penambal fase 16;
dilatih ulang per origin (jendela melebar); tanpa gerbang.

| polutan | h | SARIMA | Ridge gate | LGBM gate | LGBM langsung |
|---|---|---|---|---|---|
| NO2 | 1 | 0.6007 | 0.5925 | 0.6016 | 0.8853 |
| NO2 | 7 | 0.8926 | 0.9017 | 0.8848 | 1.0213 |
| NO2 | 30 | 0.7066 | 0.7040 | 0.6994 | 0.7657 |
| PM10 | 1 | 0.1148 | 0.1114 | 0.1134 | 0.1021 |
| PM10 | 7 | 0.1529 | 0.1499 | 0.1524 | 0.1371 |
| PM10 | 30 | 0.1524 | 0.1514 | 0.1503 | 0.1406 |
| PM25 | 1 | 0.0654 | 0.0638 | 0.0646 | 0.0636 |
| PM25 | 7 | 0.0740 | 0.0753 | 0.0739 | 0.0683 |
| PM25 | 30 | 0.0903 | 0.0891 | 0.0899 | 0.0708 |

Sel lgbm_direct nyata vs SARIMA (p<0,05): 1/9. Lampiran: 17_lgbm_direct_*.csv
