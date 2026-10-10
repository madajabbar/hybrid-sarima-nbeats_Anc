# Temuan Fase 3 — perbaikan protokol hybrid (Jalan B)

Pertanyaan: apakah hybrid kalah **karena koreksinya dipakai penuh tanpa penyusutan** (alfa=1)?
Uji: koreksi terkondisi baik (lag residu 1–7, rata-rata 7/30, std 30, lag 365) dengan dua
pembelajar murah — Ridge dan HistGradientBoosting — lalu bobot **alfa dipilih di data validasi**
(90 hari ekor train), bukan di test. Arm "full" = alfa dipaksa 1 (cara paper).

Protokol identik notebook 01/02: NO2, Box-Cox, train 1106 / test 720 hari, origin tiap 30 hari
(h1: 24 origin, h7: 24, h30: 23), SARIMA (2,1,1)(0,0,0,12).
Skrip: `scripts/03_fix_hybrid.py` (jalan di CPU, 3,6 detik).

## Hasil (RMSE, skala Box-Cox)

| horizon | SARIMA | ridge (alfa) | gbm (alfa) | gbm full (alfa=1) |
|---|---|---|---|---|
| 1 | 0.5951 | **0.5578** (1.00) | 0.5925 (0.28) | 0.7098 |
| 7 | **0.7545** | 0.7789 (1.00) | 0.7932 (0.41) | 0.8896 |
| 30 | **0.6087** | 0.6096 (0.17) | **0.6077** (0.02) | 0.6816 |

MAE h=1: SARIMA 0.4847 → ridge 0.4510. MAPE h=1: 6.51% → 6.03%.
Uji-t berpasangan atas kuadrat galat: tidak ada yang signifikan (p = 0.11–0.92) — hanya 23–24 origin.

## Kesimpulan

1. **Mekanisme kerugian terkonfirmasi**: koreksi tanpa penyusutan (alfa=1) selalu memperburuk
   (gbm full: +0.115 / +0.135 / +0.073 RMSE). Penyusutan memulihkan sebagian besar kerugian.
2. **Perbaikan nyata hanya di h=1**: ridge memangkas RMSE 6,3% dan MAPE 0,48 poin.
3. h=7 dan h=30 masih kalah tipis. Sisa kesalahan SARIMA praktis tak punya struktur yang bisa
   dipelajari — konsisten dengan korelasi −0,33 (fase 1) dan kegagalan N-BEATS (fase 2).
4. Alfa validasi tidak selalu jujur: ridge memilih alfa=1 untuk h=7, tetapi di test justru
   merugikan → ekor validasi 90 hari terlalu pendek/berisik. Perlu validasi bergulir.

## Catatan keterbatasan

- 1 seed, 1 polutan (NO2). Paper mengklaim hybrid menang justru di PM2.5/PM10 — belum diuji.
- Belum ada uji Diebold–Mariano formal (dipakai uji-t berpasangan; hasilnya sama-sama belum signifikan).
- Bobot alfa belum distabilkan (lihat poin 4).

## Berkas

- `results/03_fix_hybrid_metrics.csv`, `results/03_fix_hybrid_metrics_predictions.csv`
- `scripts/03_fix_hybrid.py`

## Langkah berikut yang paling berpeluang menang

1. Alfa dipilih dengan validasi bergulir (expanding window), bukan satu ekor tetap.
2. Uji PM10 & PM2.5 — di situlah paper mengklaim hybrid unggul.
3. Gabungan berbobot berbasis validasi (SARIMA + koreksi) — secara definisi tidak lebih buruk
   dari SARIMA pada data validasi.