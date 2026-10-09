# Temuan Fase 2 — swap BiLSTM → N-BEATS (notebook 02)

Protokol: identik dengan blok orizont notebook 01 (target **NO2**, split 720 hari uji,
SARIMA **(2,1,1)(0,0,0,12)**, rolling-origin 1 origin tiap 30 hari, `input_size=90`).
Dijalankan di Kaggle T4 (1 GPU dipaksa, `devices=1`), `neuralforecast 3.3.0`.

## Hasil (RMSE, skala Box-Cox)

| horizon | SARIMA | SARIMA+BiLSTM | SARIMA+N-BEATS | Δ N-BEATS − BiLSTM |
|---|---|---|---|---|
| 1 | 0.5951 | **0.5915** | 0.7162 | +0.1247 |
| 7 | **0.7545** | 0.7603 | 0.8923 | +0.1319 |
| 30 | **0.6087** | 0.6289 | 0.8372 | +0.2083 |

MAPE N-BEATS: 7.94 % / 8.60 % / 9.00 % (BiLSTM: 6.38 % / 7.63 % / 7.29 %).

## Biaya

- BiLSTM: **544.513** parameter, latih **247,3 s** (Kaggle T4).
- N-BEATS: **881.243** parameter (h=1), **1.739.026** parameter (h=30) — **3,2× lebih besar**.
  Latih **5,5 s** untuk ketiga horizon (h=1 saja 2,3 s).

## Kesimpulan

1. **N-BEATS kalah akurasinya di semua horizon** — hipotesis "N-BEATS lebih baik" **gugur**.
2. **Hipotesis "N-BEATS lebih ringan" juga gugur**: parameternya 3,2× BiLSTM.
3. Yang menang **waktu latih**: ~45× lebih cepat — tapi **anggaran latihnya tidak setara**
   (N-BEATS `max_steps=200` ≈ 6 epoch; BiLSTM sampai `early_stopping` pada epoch 30 dari 100).
   Jadi perbandingan waktu ini belum apple-to-apple; angka akurasi tetap sah.
4. Pola paradoks paper Necula **terkonfirmasi**: menambah komponen neural justru memperburuk
   ramalan NO2. SARIMA saja terbaik di h=7 dan h=30.
5. Konsekuensi metodologis: residual SARIMA pada NO2 praktis **tidak punya struktur yang bisa
   dipelajari** — konsisten dengan korelasi prediksi BiLSTM vs residual = **−0,33** di fase 1.

## Catatan keterbatasan (jujur untuk laporan)

- Hanya 1 seed, 1 polutan (NO2), 23–24 origin per horizon.
- `max_steps=200` untuk N-BEATS = anggaran awal, bukan hasil tuning; tuning bisa mengubah angka
  (tapi arah "lebih besar + lebih buruk" kemungkinan tidak berbalik).
- Nilai ekstrem & uji Diebold–Mariano belum dijalankan untuk N-BEATS.

## Berkas

- `results/02_nbeats_vs_baseline.csv` — tabel di atas.
- `results/02_nbeats_swap/02_nbeats_swap_executed.ipynb` — notebook ter-eksekusi (bukti).
- `results/02_nbeats_swap/cost_summary.txt` — ringkasan biaya.