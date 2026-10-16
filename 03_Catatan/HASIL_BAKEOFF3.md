# Hasil fase 20 — bake-off TANPA-SARIMA: mengganti seluruh metode Necula

_Dicatat: 16 Okt 2026. Arahan Tuan: metode pengganti harus mengganti SARIMA **juga** (bukan
hanya BiLSTM); naskah hanya menulis satu metode terbaik. Router ditiadakan dari status
kandidat. Protokol identik fase 18/19: 5 lipatan (geser 120), 24 origin, h=1/7/30,
RMSE skala Box-Cox, mask `t+h ≤ t0−1`, DM classic. Script `notebooks/08_models_bakeoff3.ipynb`
(dieksekusi; penyokong CLI hilang — versi awal `.py` ada di riwayat Git),
artefak `results/20_*.csv`._

## Lengan baru (semua bebas SARIMA)

Direkt/fitur (lag1-7, roll, Fourier tahunan+mingguan, dow, lag365):
**svr**, **knn**, **ridge_langsung**, **lgbm_mingguan** (LGBM skrip 17 + Fourier mingguan
+ harmonik 2), **fourier_ols** (regresi kalender murni). Deret-waktu: **ucm** (structural
level+musiman mingguan+eksog tahunan), **stl_ets**. Ensemble rata-rata sama besar:
**ens_lgbm_theta, ens_lgbm_ets, ens_lgbm_rf, ens_trio (lgbm+theta+rf), ens_lgbmwk_theta**.

## Papan skor 9 sel (% vs metode lengkap paper — SARIMA s12 + BiLSTM)

| lengan | skor | menang/9 | catatan DM |
|---|---|---|---|
| ens_lgbm_ets | **−5,64%** | 7/9 | kalah nyata 1 (NO2-h1 +12,3% p=0,023) |
| ens_lgbm_theta | **−5,46%** | 7/9 | **kalah nyata 0** (NO2-h1 +10,5% p=0,054; h7 +3,0% ns) |
| ens_trio | −4,76% | 6/9 | |
| ens_lgbmwk_theta | −4,73% | 6/9 | |
| ridge_gate (ber-SARIMA, acuan) | −3,56% | 8/9 | |
| SARIMA mingguan kita | −2,90% | 7/9 | |
| lgbm_langsung | −2,27% | 6/9 | NO2-h1 nyata +32,8% |
| knn_langsung | +0,49% | 6/9 | seri agregat |
| ridge_langsung | −0,47% | 5/9 | |
| ucm | +2,78% | 3/9 | |
| rf_langsung | +2,56% | 6/9 | jago PM jarak panjang |
| theta | +2,63% | 3/9 | |
| svr_langsung | +4,71% | 3/9 | |
| fourier_ols | +4,02% | 5/9 | |
| stl_ets | +17,28% | 0/9 | gagal |

Detail sel ens_lgbm_theta vs paper: NO2 h1 +10,5% (ns) • h7 +3,0% (ns) • h30 −6,8%
(p=0,0003) • PM10 h1 −10,6%* • h7 −9,5%* • h30 −10,1%* • PM25 h1 −5,6% (p=0,06) •
h7 −1,7% (ns) • h30 −20,1%*. (* nyata; menang 5 nyata, 2 kalah tak-nyata.)

## Temuan steroid:

1. **Ensemble LGBM+Theta = kandidat metode pengganti terkuat**: rata2 −5,5% lawan
   metode lengkap paper tanpa satu pun kekalahan nyata. LGBM+ETS lebih cantik rata2nya
   (−5,64) tapi kalah nyata di NO2-h1 → profil keandalan lebih buruk.
2. **Dinding tunggal: NO2 jarak pendek.** Semua lengan bebas-SARIMA kalah di NO2-h1
   (dan h7 tipis). Fourier mingguan sebagai fitur TIDAK menyelamatkan (lgbm_mingguan:
   NO2-h1 0,808 — membaik dari 0,885 tapi masih jauh dari paper 0,667). Pelajaran:
   musiman mingguan NO2 butuh *struktur statstik* (SARIMA/UCM), bukan sekadar fitur.
3. **UCM** per-origin: NO2-h1 0,6619 (hanya 0,5% dari paper, ns) — jagoan pendek NO2
   di kalangan bebas-SARIMA, tapi agregat lemah (+5,9%). Bahan future work/kalibrasi.
4. SVR (+7,97% vs SARIMA), fourier_ols (+7,40), stl_ets (+20,8) — bukan jalannya.
5. Bug tercatat: statsmodels `UnobservedComponents.forecast()` wajib `exog` masa depan;
   run pertama UCM gagal total (semua NaN) — diperbaiki, tidak mengubah lengan lain
   (semua deterministic, angka ulang identik).

## Status keputusan

Menunggu Tuan memilih resep metode final: **ens_lgbm_theta (rekomendasi)** atau
ens_lgbm_ets. Setelah dipilih → naskah ditulis murni "metode baru menggantikan seluruh
metode Necula", sisanya masuk bab perbandingan/bake-off.
Artefak `20_bakeoff3_predictions.csv` (360 baris) + metrics + dm. **Belum di-commit**
(menunggu persetujuan, bersama artefak fase 18–19).
