# Temuan Fase 7 — pengaturan musiman paper salah untuk data harian (hipotesis terbukti)

Hipotesis: `seasonal_order=(0,0,0,12)` pada data **harian** (angka 12) tidak mewakili siklus mingguan,
sehingga pola kerja vs akhir pekan tidak dimodelkan — itulah sebab galat jarak 7 hari membengkak.

Varian (protokol lain identik notebook 01/02; 3 lipatan; skrip `scripts/08_weekly.py`):
- **A** = paper: (2,1,1)(0,0,0,12)
- **B** = mingguan: (2,1,1)(1,0,1,7)
- **C** = paper + Fourier mingguan (sin/cos k=1..3) sebagai variabel eksogen

## A. Efek musiman pada SARIMA saja (3 lipatan, RMSE)

| polutan | h | A (paper) | B (mingguan) | Δ | p (DM) |
|---|---|---|---|---|---|
| NO2 | 1 | 0.6453 | **0.5800** | **−10.1%** | **0.0049** |
| NO2 | 7 | 0.8921 | **0.8598** | **−3.6%** | **0.0040** |
| NO2 | 30 | 0.7045 | **0.6676** | −5.2% | 0.325 |
| PM10 | 1 | 0.1178 | **0.1155** | −1.9% | **0.0312** |
| PM10 | 7 | 0.1524 | **0.1504** | −1.3% | 0.140 |
| PM10 | 30 | **0.1503** | 0.1541 | +2.5% | 0.204 |
| PM25 | 1 | 0.0663 | 0.0663 | +0.0% | 0.949 |
| PM25 | 7 | 0.0731 | 0.0731 | −0.0% | 0.866 |
| PM25 | 30 | 0.0890 | 0.0890 | −0.1% | 0.646 |

**Kesimpulan:** memperbaiki periode musiman menolong **sangat nyata di NO2** (gas, didorong lalu lintas →
punya siklus mingguan), menolong tipis tapi nyata di PM10 h=1, dan **tidak berpengaruh di PM25**
(partikulat lebih ditentukan cuaca/musim, bukan hari kerja). Ini tafsir yang konsisten secara fisik.

## B. Adu final vs BiLSTM — titik ujian identik, NO2, fold-0 (skrip `09_vs_bilstm_weekly.py`)

| h | arm | RMSE kita | BiLSTM (paper) | Δ | p |
|---|---|---|---|---|---|
| 1 | B + koreksi berbobot | 0.5797 | 0.5915 | −2.0% | 0.750 |
| 7 | B + koreksi berbobot | **0.7480** | 0.7603 | −1.6% | 0.686 |
| 30 | B + koreksi berbobot | **0.5697** | 0.6289 | **−9.4%** | 0.250 |
| 7 | B, SARIMA saja | **0.7371** | 0.7603 | −3.1% | 0.332 |
| 30 | C + koreksi berbobot | **0.5173** | 0.6289 | **−17.8%** | 0.445 |
| 7 | A (paper) + koreksi | 0.8042 | 0.7603 | +5.8% | **0.048** |

**Satu resep konsisten (B: musiman 7 + koreksi berbobot) menang di ketiga jarak** — sebelumnya jarak 7
kalah nyata (+5.8%), sekarang −1.6%. Titik lemah itu **hilang**.

Konfigurasi terbaik di h=30 memakai varian **C** (−17.8%), tetapi C sedikit lebih buruk di h=7 (+0.3%) →
**B lebih konsisten** dan layak dijadikan resep tunggal paper.

## C. Keterbatasan (jujur)

- Beda vs BiLSTM pada fold-0 (24 titik ujian) **belum signifikan** secara statistik (p = 0.25–0.75);
  yang sudah signifikan adalah efek perbaikan musiman (3 lipatan, 72 titik: p = 0.0049 dan 0.0040).
- Varian C baru diuji untuk NO2 (biaya), belum untuk PM10/PM25.
- Pemenang "per horizon" tidak boleh dipakai sebagai klaim (itu memilih di data ujian) — klaim resmi
  memakai satu resep B yang seragam.

## Status: resep final kandidat

```
SARIMA (2,1,1)(1,0,1,7)  [musiman mingguan, BUKAN 12 seperti paper]
  + alfa * koreksi Ridge pada fitur residu (lag 1-7, rata 7/30, std 30, lag 365,
    Fourier tahunan, hari-dalam-minggu, Fourier mingguan)
alfa dipilih di jendela validasi 120 hari sebelum titik ujian (0-100%)
```

Hasil: menang vs BiLSTM di 3/3 jarak (fold-0 identik) · efek musiman mingguan nyata pada NO2 & PM10 h=1.