# Temuan Fase 6 — aturan dua tingkat + adu langsung dengan BiLSTM

## A. Adu langsung dengan BiLSTM (script `07_vs_bilstm.py`)

Memakai berkas prediksi tersimpan (tanpa latih ulang) sehingga arm BiLSTM dan resep kita
berada pada **titik ujian yang persis sama** (diverifikasi: selisih nilai aktual = 0,00), NO2, fold-0.

| h | SARIMA | SARIMA+BiLSTM (paper) | resep kita | Δ vs BiLSTM | p |
|---|---|---|---|---|---|
| 1 | 0.5951 | 0.5915 | **0.5745** | −2.9% | 0.407 |
| 7 | 0.7545 | 0.7603 | 0.8042 | **+5.8%** | **0.048** |
| 30 | 0.6087 | 0.6289 | **0.5981** | −4.9% | 0.618 |

Menang 2/3 jarak. **Jarak 7 hari kalah nyata** — ini titik lemah yang harus dibereskan.

Catatan penting: dibanding SARIMA, BiLSTM paper hanya −0,6% (h=1), **+0,8%** (h=7), **+3,3%** (h=30).
Artinya koreksi BiLSTM mereka merugikan di jarak menengah–jauh, dan itu yang kita perbaiki.

Perbaikan teknis: uji Diebold–Mariano kini membatasi jumlah lag (n kecil) dan memakai varians
tanpa autokorelasi bila varians HAC ≤ 0 — sebelumnya h=30 menghasilkan nilai p kosong.

## B. Aturan dua tingkat (script `06_final_rule.py`)

Aturan: pakai koreksi **penuh** kalau di jendela validasi (120 hari sebelum origin) koreksi terbukti
menurunkan RMSE; kalau tidak, jangan pakai (alfa = 0). Keputusan tidak pernah melihat data ujian.

| arm | menang | rata-rata | terbaik | terburuk | menang nyata | kalah nyata |
|---|---|---|---|---|---|---|
| adaptif (alfa dari validasi) | 7/9 | −1.44% | −4.71% | +2.55% | 3 | 1 |
| dua tingkat + alfa | 7/9 | −1.44% | −4.71% | +2.55% | 3 | 1 |
| dua tingkat + penuh | 6/9 | −1.40% | −5.25% | +3.97% | 2 | 1 |

**Hasilnya: aturan dua tingkat tidak memperbaiki apa pun.** "Dua tingkat + alfa" identik dengan
arm adaptif — artinya gerbang validasi hampir selalu menjawab "pakai", termasuk di PM25 h=7 di mana
data ujian kemudian membuktikan koreksi justru merugikan (+1,9%, p=0,0084). Kesimpulan jujur:
**jendela validasi 120 hari terlalu berisik untuk dijadikan penentu tunggal.**

## C. Hipotesis berikutnya (calon Fase 7)

Akar masalah jarak 7 hari kemungkinan besar **bukan** pada koreksi, tetapi pada SARIMA-nya:
paper memakai `seasonal_order = (0,0,0,12)` pada data **harian** — angka 12 di situ tidak mewakili
siklus mingguan. Pola mingguan (kerja vs akhir pekan) karena itu tidak dimodelkan, dan galat
jarak 7 hari membengkak di ketiga polutan.

Uji yang diusulkan: tambahkan musiman mingguan — `seasonal_order=(0,0,0,7)` atau suku Fourier
mingguan sebagai variabel eksogen — lalu jalankan ulang seluruh protokol. Murah, dan bila benar
akan memperbaiki h=7 untuk ketiga polutan sekaligus.

## Status ringkas

- Resep tetap terbaik saat ini: **SARIMA + alfa * koreksi Ridge** (alfa dari validasi).
- Menang 7/9 sel, menang nyata di NO2 h=1, PM10 h=1, PM25 h=30; kalah nyata hanya PM25 h=7.
- Sisa pekerjaan: perbaiki h=7 (Fase 7), lalu tulis naskah dengan satu tabel perbandingan final.