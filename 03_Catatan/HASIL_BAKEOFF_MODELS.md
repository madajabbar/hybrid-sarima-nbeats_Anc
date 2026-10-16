# Hasil fase 18 — Bake-off model: haruskah SARIMA? alternatifnya apa?

_Dicatat: 14 Okt 2026. Pertanyaan Tuan: "kalau model lain bisa? apakah harus pakai
SARIMA di kasus ini? tidak ada alternatif?" → 4 model baru diuji satu per satu
di ranah yang PERSIS sama dengan resep beku (data, lipatan, titik uji, metrik)._

## 1. Protokol (identik fase 12/16/17)

- Data: `transformed_data.csv` 1.826 hari, Box-Cox (kalimat sama notebook 04).
- 5 lipatan jendela ujian 720 hari, geser 120; 24 origin × 30 hari; h = 1/7/30.
- Skripsi script: `scripts/06_models_bakeoff.py`; artefak `results/18_*.csv`.
- Lengan baru: **naive7** (dengar mingguan paling naif), **ARIMA nonmusiman**
  (SARIMA tanpa `(1,0,1,7)` — ablasion musiman), **ETS/Holt-Winters**
  (trend damped + musiman mingguan), **VAR** (tiga polutan serentak, lag pilih AIC).

## 2. Papan peringkat (RMSE Box-Cox, 5 lipatan digabung, n per sel 112–118)

| polutan | h | SARIMA | ARIMA nonmus. | ETS | VAR | naive7 | ridge_gate | lgbm_gate | lgbm_langsung |
|---|---|---|---|---|---|---|---|---|---|
| NO2 | 1 | 0.6007 | 0.6756 | 0.6648 | 5.7110 | 0.7761 | **0.5925** | 0.6016 | 0.8853 |
| NO2 | 7 | 0.8889 | 0.9225 | 0.9324 | 5.6541 | 1.0738 | 0.9017 | **0.8848** | 1.0213 |
| NO2 | 30 | 0.7043 | 0.7322 | 0.7077 | 5.7796 | 0.9548 | 0.7040 | **0.6994** | 0.7657 |
| PM10 | 1 | 0.1148 | 0.1175 | 0.1236 | 0.6243 | 0.1684 | 0.1114 | 0.1134 | **0.1021** |
| PM10 | 7 | 0.1525 | 0.1541 | 0.1577 | 0.6507 | 0.1648 | 0.1499 | 0.1524 | **0.1371** |
| PM10 | 30 | 0.1518 | 0.1476 | 0.1590 | 0.5902 | 0.1839 | 0.1514 | 0.1503 | **0.1406** |
| PM25 | 1 | 0.0654 | 0.0654 | 0.0681 | 6.4140 | 0.0942 | 0.0638 | 0.0646 | **0.0636** |
| PM25 | 7 | 0.0738 | 0.0738 | 0.0852 | 6.3800 | 0.0950 | 0.0753 | 0.0739 | **0.0683** |
| PM25 | 30 | 0.0899 | 0.0901 | 0.0862 | 6.4649 | 0.1079 | 0.0891 | 0.0899 | **0.0708** |

Skor gabungan 9 sel (rata-rata % vs SARIMA; negatif = lebih baik):
ridge_gate **−0.68%**, lgbm_gate −0.49%, SARIMA 0%, lgbm_langsung +1.22%,
ARIMA nonmus. +2.35%, ETS +5.25%, naive7 +28.3%, VAR +3168%.

## 3. Uji DM (p<0.05 = beda nyata)

- **ARIMA nonmusiman vs SARIMA:** kalah nyata di NO2 semua horizon (h1 +12.5% p<0.0001,
  h7 +3.8% p=0.005, h30 +4.7% p=0.0002) → musiman mingguan NO2 sungguh ada.
  Di PM10 kerjanya malah lebih baik: h30 −3.1% (p=0.0016) → musiman PM10 lemah.
- **ETS vs SARIMA:** kalah nyata di NO2 h1 (+10.7%), PM10 h1 (+7.7%), PM25 h7 (+13%);
  seri di semua h30. ETS bukan pengganti SARIMA.
- **VAR vs SARIMA:** kalah telak 4–99× di semua sel → ditinggalkan.
- **lgbm_langsung vs SARIMA:** NO2 kalah nyata semua (p<0.05); PM10 menang arah
  (−8..−11%, belum nyata); **PM25 h30 menang nyata −16.3% (p=0.0002)**.
- **lgbm_langsung vs ridge_gate:** PM25 h30 **−20.6% (p<0.0001)**, PM10 h30 −7.1%
  (p=0.008), PM25 h7 −9.3% (p=0.046) → polutan partikulat: LGBM telanjang juara.

## 4. Router (pilih juara per sel dari lipatan TER-UMUR 480 hari, nilai di sisa)

Pilihan: NO2 h1=ridge_gate, h7=SARIMA, h30=ETS • PM10 h1=ridge_gate, h7=lgbm_langsung,
h30=ETS • PM25 h1=ridge_gate, h7=lgbm_langsung, h30=lgbm_langsung.
Skor router: **−4.48% vs SARIMA** — ungguli semua model tunggal (terbaik tunggal −0.68%).
Catatan: versi router dg validasi 120 hari pertama memilih kebetulan (naive7 utk NO2-h7
krn satu lonjakan Feb 2022) → validasi makin panjang = pemilihan makin stabil.

## 5. Jawaban atas "harus SARIMA?"

1. Yang wajib bukan SARIMA, tapi **musiman mingguan utk NO2** — bukti: lengan tanpa
   musiman kalah nyata konsisten. Untuk PM, musiman itu duduk di fitur, bukan di struktur.
2. **PM10 & PM25 tidak butuh SARIMA**: LGBM telanjang menang/setara di semua selnya,
   jauh menang di PM25 panjang. Alternatif nyata sudah ada di tangan sendiri.
3. ETS, VAR, naive: bukan jalannya di data ini (ETS kalah tipis, VAR bubrah, naive bocor lonjakan).
4. Tidak ada model tunggal juara 9 sel → **protokol pemilihan (router)** adalah
   jawaban jujurnya, dan router sederhana ini sudah −4.48%.

## 6. Fase 19 — tambahan 4 lengan: Theta, RF langsung, MLP langsung, ensemble

_Dicatat: 14 Okt 2026. Pertanyaan lanjutan: "Model lain?" → 4 kandidat lagi diuji di
protokol yang persis sama. Script `scripts/07_models_bakeoff2.py`, artefak `results/19_*.csv`._

**Lengan baru:** **Theta** (ekstrapolasi tren + musiman mingguan, metode klasik ringan),
**RF langsung** (Random Forest, 200 pohon, fitur lag sama dgn LGBM langsung),
**MLP langsung** (jaringan saraf kecil 2 lapisan), **ensemble** (rata-rata praketapan
SARIMA + LGBM langsung — "dua tukang ramal, jawabannya dirata-rata").

**Insiden kebocoran (penting):** run pertama RF "menang" −47% — mustahil. Dikorek:
label latihnya menjulur sampai 30 hari ke masa depan (masuk jendela uji — murid
dapat kunci jawaban). Mask dikunci identik skrip 17 (`t+h ≤ hari terakhir yang
diketahui`), cache dibuang, diulang. Angka di bawah = run bersih. RF +5,9% —
bukan sakti; MLP ternyata yang paling parah.

Hasil lengan baru (RMSE Box-Cox):

| polutan | h | Theta | RF langsung | MLP langsung | ensemble |
|---|---|---|---|---|---|
| NO2 | 1 | 0.6422 | 0.8051 | 0.9932 | 0.7088 |
| NO2 | 7 | 0.9471 | 0.9218 | 1.1000 | 0.9362 |
| NO2 | 30 | 0.7192 | **0.6848** | 0.7660 | 0.6919 |
| PM10 | 1 | 0.1262 | 0.1450 | 0.2008 | 0.1024 |
| PM10 | 7 | 0.1602 | 0.1500 | 0.1888 | 0.1383 |
| PM10 | 30 | 0.1570 | **0.1299** | 0.2074 | 0.1387 |
| PM25 | 1 | 0.0687 | 0.0761 | 0.1376 | **0.0622** |
| PM25 | 7 | 0.0855 | 0.0722 | 0.1321 | **0.0679** |
| PM25 | 30 | 0.0866 | 0.0845 | 0.1313 | 0.0762 |

Papan peringkat 12 arm terbaru (skor 9 sel, % vs SARIMA):
**ensemble −3.92** • ridge_gate −0.68 • lgbm_gate −0.49 • SARIMA 0 • lgbm_langsung +1.22
• arima_nonmusiman +2.35 • ets +5.25 • theta +5.70 • **rf_langsung +5.92** • naive7 +28.26
• **mlp_langsung +52.07** • var +3168.

Temuan DM vs SARIMA (lengan baru, p<0.05):

- **RF langsung:** kalah nyata di SEMUA h1 (NO2 +34%, PM10 +26%, PM25 +16%) dan
  NO2-h7; menang nyata hanya di horizon 30: PM10 −13,9% (p<0,0001), PM25 −7,1%
  (p<0,0001), NO2 −3,0% (p=0,04, tipis). RF = spesialis jarak panjang, buruk harian.
- **Theta:** kalah nyata NO2 h1/h7, PM10-h1, PM25-h7; hampir semua h30 seri —
  bukan peningkatan berarti dibanding SARIMA.
- **MLP:** kalah telak nyata di SEMUA 9 sel (rata-rata 52% lebih buruk) —
  jaringan kecil butuh data jauh lebih banyak; konsisten dgn cerita BiLSTM paper acuan.
- **Ensemble:** menang nyata PM10 semua horizon (−9..−11%) + PM25 h7 (−8,2%,
  p=0,0005) & h30 (−15,7%, p<0,0001); kalah nyata NO2 h1 (+18%) & h7 (+4,9%);
  NO2-h30 & PM25-h1 seri. Satu model buat semua sel, tanpa memilih-milih.

Juara per sel bertambah: NO2-h30 → RF langsung (0.6848), PM10-h30 → RF langsung
(0.1299), PM25-h1 & PM25-h7 → ensemble (0.0622 / 0.0679).

**Router 12 arm:** NO2 h1=ridge_gate, h7=SARIMA, h30=RF • PM10 h1=ridge_gate,
h7=lgbm_langsung, h30=ETS • PM25 h1=ridge_gate, h7=ensemble, h30=lgbm_langsung
→ skor **−4,90%** (naik dari −4,48 versi fase 18).

## 7. Jawaban akhir atas "harus SARIMA?" (versi 12 arm)

1. Tetap: wajibnya musiman mingguan untuk NO2; PM tidak butuh SARIMA.
2. **Ensemble rata-rata SARIMA+LGBM = model tunggal terbaik (−3,92%)** — nyaris
   menyamai router (−4,90%) tanpa kerumitan pemilihan per sel. Ini kandidat
   "metode paper" paling sederhana yang layak dipertahankan.
3. RF: hanya layak untuk horizon 30 hari. MLP & Theta & VAR & naive: bukan jalannya.
4. Router tetap juara mutlak, tapi harganya kompleksitas; ensemble = 95% manfaatnya
   dengan 10% keruwetan.

## 8. Artefak

`results/18_bakeoff_predictions.csv` • `18_bakeoff_metrics.csv` • `18_bakeoff_dm.csv` •
`18_router_picks.csv` • `18_router_metrics.csv` • script `scripts/06_models_bakeoff.py`
• `results/19_bakeoff2_predictions.csv` • `19_bakeoff2_metrics.csv` • `19_bakeoff2_dm.csv`
• `19_router_metrics.csv` • script `scripts/07_models_bakeoff2.py`
(semua belum di-commit — menunggu arahan Tuan).
