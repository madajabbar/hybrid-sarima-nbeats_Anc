# Temuan Fase 5 — satu resep tetap, 3 lipatan, uji Diebold–Mariano

Permintaan dosen: eksperimen untuk mencari hasil terbaik. Nomor yang dikerjakan: (1) banyak lipatan,
(2) uji Diebold–Mariano, (4) satu resep tetap untuk semua sel.

Skrip: `scripts/05_final_recipe.py` (hitung ulang ±9 menit di CPU) → `scripts/05b_significance.py`
(hitungan ulang uji, 1,5 detik, dari berkas prediksi tersimpan).

## Resep tetap

```
ramalan akhir = SARIMA + alfa * koreksi_ridge
koreksi_ridge : Ridge pada fitur residu (lag 1–7, rata 7/30, std 30, lag 365, Fourier tahunan, hari-minggu)
alfa          : dipilih di jendela validasi 120 hari sebelum titik ujian (0–100%)
```

3 lipatan: jendela ujian 720 hari digeser 0, 180, 360 hari ke belakang. Titik ujian: 72 (h=1,7) dan 69 (h=30).
Arm "kontrol_koreksi_penuh" = koreksi ditambahkan penuh (alfa = 1, cara Necula 2025).

## Hasil (RMSE, 3 lipatan digabung)

| polutan | h | SARIMA | resep kita | Δ | DM p | arah |
|---|---|---|---|---|---|---|
| NO2 | 1 | 0.6453 | 0.6166 | **−4.4%** | 0.0247 | menang nyata |
| NO2 | 7 | 0.8921 | 0.9149 | +2.6% | 0.357 | kalah (tak nyata) |
| NO2 | 30 | 0.7045 | 0.6993 | −0.7% | 0.682 | seri |
| PM10 | 1 | 0.1178 | 0.1122 | **−4.7%** | 0.0311 | menang nyata |
| PM10 | 7 | 0.1524 | 0.1478 | −3.0% | 0.104 | menang (tak nyata) |
| PM10 | 30 | 0.1503 | 0.1497 | −0.4% | 0.377 | seri |
| PM25 | 1 | 0.0663 | 0.0645 | −2.6% | 0.0708 | menang (mendekati) |
| PM25 | 7 | 0.0731 | 0.0745 | **+1.9%** | 0.0084 | **kalah nyata** |
| PM25 | 30 | 0.0890 | 0.0877 | **−1.5%** | 0.0030 | menang nyata |

Ringkasan resep: menang 7/9 sel · rata-rata −1,44% · terbaik −4,71% · terburuk +2,55%.
Kemenangan nyata: NO2 h=1 (−4,4%), PM10 h=1 (−4,7%), PM25 h=30 (−1,5%).
Kekalahan nyata: PM25 h=7 (+1,9%).

## Temuan penting (jujur, mengubah arah)

**Kontrol "koreksi penuh" (cara paper, alfa=1) justru menang nyata di 5 sel — dan lebih besar:**

| polutan | h | Δ | DM p |
|---|---|---|---|
| NO2 | 1 | −5.4% | 0.024 |
| PM10 | 1 | −5.2% | 0.021 |
| PM10 | 30 | −2.4% | 0.0011 |
| PM25 | 1 | −3.8% | 0.028 |
| PM25 | 30 | −7.2% | 5.3e-07 |

Artinya: dengan 3 lipatan (data lebih banyak), **kehati-hatian resep kita berlebihan di partikulat** —
kita menahan koreksi justru di tempat koreksi terbukti berguna. Di NO2, kehati-hatian itu memang
menyelamatkan (NO2 h=7: kontrol penuh +4,0% vs resep kita +2,6%).

## Kesimpulan & langkah berikut

1. Resep tetap sekarang: aman (tidak pernah kalah nyata kecuali PM25 h=7) tetapi **terlalu konservatif**.
2. Resep berikutnya (Fase 6): **dua tingkat, keputusan murni dari data validasi** —
   jika koreksi terbukti menolong di validasi → pakai penuh (alfa=1); jika tidak → alfa=0.
   Ini menyatukan keunggulan kontrol penuh di PM dengan perlindungan di NO2, tanpa memilih
   berdasarkan data ujian.
3. Catatan metodologis: angka 3 lipatan tidak bisa langsung dibandingkan dengan angka Fase 1
   (yang memakai fold-0 saja). Perbandingan langsung dengan BiLSTM harus dihitung pada fold-0.

## Keterbatasan

- Seed hanya bervariasi pada arm GBM (3 seed); resep utama memakai Ridge (deterministik, tanpa variasi seed).
- GBM hanya dijalankan pada lipatan pertama (biaya), jadi angka GBM tidak digabung 3 lipatan.
- Interval keyakinan bootstrap memakai resampling titik ujian; korelasi antar-lipatan tidak dimodelkan.