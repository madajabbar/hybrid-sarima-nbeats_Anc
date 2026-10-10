# Temuan Fase 9 & 10 — kriteria penerimaan koreksi + pelatihan penuh (resep beku)

## Fase 9 — apakah gerbang penerimaan yang lebih ketat diperlukan? (`scripts/11_acceptance.py`)

Base SARIMA = musiman mingguan (2,1,1)(1,0,1,7). Tiga arm, 3 lipatan, 72 titik ujian per sel:

| arm | menang | rata-rata | terbaik | terburuk | menang nyata | kalah nyata |
|---|---|---|---|---|---|---|
| adaptif (alfa 1 jendela 120 hari) | **8/9** | **−2.43%** | −7.33% | +0.02% | 2 | **0** |
| gerbang 3 blok + ambang 5% | 3/9 | −1.34% | −7.76% | +0.70% | 0 | 0 |
| gerbang 3 blok + koreksi penuh | 3/9 | −1.34% | −7.76% | +0.70% | 0 | 0 |

**Kesimpulan: gerbang tidak diperlukan.** Setelah musiman dibetulkan (Fase 7), resep adaptif biasa
sudah menang di 8 dari 9 sel **tanpa satu pun kekalahan nyata** — termasuk PM25 jarak 7 yang tadinya
rugi nyata (+1.9%) kini justru **menang nyata** (−0.93%, p=0.0096). Gerbang hanya mematikan koreksi
di 6 sel dan membuang keuntungan. Gerbang tetap dilaporkan sebagai varian konservatif di naskah.

Efek perbaikan musiman terhadap sel yang tadinya gagal:

| sel | sebelum (musiman 12) | sesudah (musiman 7) |
|---|---|---|
| PM25 h=7 | +1.9% (kalah nyata, p=0.0084) | −0.93% (menang nyata, p=0.0096) |
| NO2 h=7 | +6.6% (kalah, p=0.08) | −7.33% (menang) |

## Fase 10 — pelatihan penuh versi final (`scripts/12_full_final.py`)

Resep beku: **SARIMA (2,1,1)(1,0,1,7) + alfa × koreksi Ridge**, alfa dari proyeksi validasi 120 hari.
Lipatan diperbanyak dari 3 menjadi **5** (jendela ujian 720 hari, geser 0/120/240/360/480 hari)
→ 112–120 titik ujian per sel. Waktu jalan 64 detik di laptop (CPU).

| polutan | h | SARIMA mingguan | RESEP FINAL | Δ | p (DM) | alfa rata2 |
|---|---|---|---|---|---|---|
| NO2 | 1 | 0.5962 | 0.5925 | −0.6% | 0.213 | 0.69 |
| NO2 | 7 | 0.8890 | 0.9017 | +1.4% | 0.439 | 0.26 |
| NO2 | 30 | 0.7303 | 0.7040 | −3.6% | 0.327 | 0.16 |
| PM10 | 1 | 0.1145 | 0.1114 | −2.7% | **0.048** | 0.50 |
| PM10 | 7 | 0.1536 | 0.1499 | −2.4% | 0.108 | 0.34 |
| PM10 | 30 | 0.1525 | 0.1514 | −0.8% | **0.0043** | 0.19 |
| PM25 | 1 | 0.0650 | 0.0638 | −1.8% | **0.031** | 0.42 |
| PM25 | 7 | 0.0754 | 0.0753 | −0.1% | **0.000004** | 0.41 |
| PM25 | 30 | 0.0914 | 0.0891 | −2.5% | **0.0009** | 0.37 |

**Ringkasan final: menang 8/9 sel · rata-rata −1.46% · terbaik −3.60% · menang nyata 5 sel ·
kalah nyata 0 sel.** Satu-satunya sel yang kalah: NO2 jarak 7 (+1.4%, p=0.44 — tidak nyata).

Tabel siap-tempel untuk naskah: `03_Catatan/HASIL_FINAL.md`.

## Yang belum: latihan ulang baseline jaringan di GPU

Perbandingan dengan BiLSTM masih memakai angka BiLSTM yang dilatih dengan protokol lama
(musiman 12). Wajib dilatih ulang dengan **musiman 7** agar perbandingan final apple-to-apple;
itu pekerjaan GPU (Kaggle T4), ±20–40 menit per polutan × jarak termasuk pencarian hyperparameter.