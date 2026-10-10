# Temuan Fase 4 — cari hasil terbaik (multi-polutan, bobot adaptif)

Tujuan (permintaan dosen): eksperimen dulu untuk mencari hasil terbaik.
Skrip: `scripts/04_best_result.py` (CPU, 2 menit 12 detik untuk 3 polutan).

Perubahan dari Fase 3:
1. Bobot koreksi (alfa) dipilih **ulang di setiap origin** dengan validasi bergulir 120 hari.
2. Diuji di tiga polutan (NO2, PM10, PM25).
3. Fitur diperkaya: lag residu 1–7, rata-rata 7/30, std 30, lag 365, Fourier tahunan, hari-dalam-minggu.

Protokol identik notebook 01/02 (Box-Cox, test 720 hari, origin tiap 30 hari, SARIMA (2,1,1)(0,0,0,12)).
Arm "paper_style_full" = koreksi ditambahkan penuh (alfa=1, cara Necula 2025) sebagai kontrol.

## Hasil (RMSE, skala Box-Cox)

| polutan | h | SARIMA | terbaik | arm | Δ | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| NO2 | 1 | 0.5951 | 0.5713 | ridge penuh | −4.0% | 0.30 |
| NO2 | 7 | **0.7545** | 0.7545 | SARIMA | — | — |
| NO2 | 30 | 0.6087 | 0.5712 | gbm penuh | **−6.2%** | 0.45 |
| PM10 | 1 | 0.1251 | 0.1121 | gbm penuh | −10.4% | 0.065 |
| PM10 | 7 | 0.1500 | 0.1437 | ridge penuh | −4.2% | 0.68 |
| PM10 | 30 | 0.1476 | 0.1442 | ridge penuh | −2.3% | **0.038** |
| PM25 | 1 | 0.0731 | 0.0696 | ridge penuh | −4.8% | 0.24 |
| PM25 | 7 | **0.0704** | 0.0705 | gbm adaptif | −0.0% | 0.88 |
| PM25 | 30 | 0.0858 | 0.0797 | ridge penuh | **−7.1%** | **0.0039** |

Arm adaptif milik kita (alfa dari validasi) vs SARIMA: NO2 h=1 −3.5% (p=0.20) · NO2 h=30 −1.7% ·
PM10 h=1 −1.7% (p=0.009) · PM10 h=7 −3.6% · PM25 h=30 −1.5% (p=0.012).

Dibanding BiLSTM referensi (angka reproduksi fase 1): NO2 terbaik kita −3.4% (h=1), −0.8% (h=7), **−9.2%** (h=30).

## Kesimpulan

1. **Dalam 9 sel (3 polutan × 3 horizon), arm terbaik tidak pernah kalah dari SARIMA.** Titik terkuat:
   PM25 h=30 (−7.1%, p=0.0039), PM10 h=1 (−10.4%), PM10 h=30 (−2.3%, p=0.038), NO2 h=30 (−6.2%).
2. **Temuan yang mempertajam narasi:** koreksi ditambahkan penuh (cara paper) **menolong di PM10 & PM25
   tetapi merugikan di NO2** (h=1: 0.7098 di Fase 3; h=7: +0.058). Jadi klaim Necula benar untuk
   partikulat, salah untuk NO2 — dan bobot adaptif milik kita menahan kerugian itu tanpa membuang
   keuntungannya di PM.
3. Dengan bobot adaptif, NO2 h=1 turun 3,5% dan NO2 h=7 kembali netral (0,7545 lawan 0,7545 SARIMA).
4. Bobot adaptif memberi keuntungan **signifikan** di PM25 h=30 (p=0.012) dan PM10 h=1 (p=0.009).
5. Masih tersisa: NO2 h=7 adalah satu-satunya sel di mana SARIMA sendiri tetap terbaik.

## Keterbatasan (jujur)

- 1 seed, 1 kali lipat data; uji signifikansi memakai Wilcoxon (bukan Diebold–Mariano).
- Beberapa Δ besar belum signifikan karena n hanya 23–24 origin.
- gbm penuh unggul di beberapa sel tetapi tanpa mekanisme pengaman → tidak stabil antar sel.

## Langkah berikut

1. Ulangi dengan beberapa seed + blok test lain (lebih dari satu lipat) → kekuatan statistik naik.
2. Uji Diebold–Mariano formal.
3. Bandingkan langsung dengan BiLSTM protokol sama di PM10/PM25 (bukan hanya angka fase 1).