# Hasil final (resep beku)

SARIMA (2,1,1)(1,0,1,7) + koreksi Ridge bergerbang.
Lipatan: 5 (jendela ujian 720 hari, geser 120 hari).

| polutan | h | SARIMA mingguan | RESEP FINAL | Δ | p (DM) | alfa rata2 |
|---|---|---|---|---|---|---|
| NO2 | 1 | 0.5962 | 0.5925 | -0.6% | 0.2130 | 0.69 |
| NO2 | 7 | 0.8890 | 0.9017 | +1.4% | 0.4391 | 0.26 |
| NO2 | 30 | 0.7303 | 0.7040 | -3.6% | 0.3275 | 0.16 |
| PM10 | 1 | 0.1145 | 0.1114 | -2.7% | 0.0479 | 0.50 |
| PM10 | 7 | 0.1536 | 0.1499 | -2.4% | 0.1081 | 0.34 |
| PM10 | 30 | 0.1525 | 0.1514 | -0.8% | 0.0043 | 0.19 |
| PM25 | 1 | 0.0650 | 0.0638 | -1.8% | 0.0312 | 0.42 |
| PM25 | 7 | 0.0754 | 0.0753 | -0.1% | 0.0000 | 0.41 |
| PM25 | 30 | 0.0914 | 0.0891 | -2.5% | 0.0009 | 0.37 |

## Pembanding protokol paper (SARIMA musiman 12, NO2, 5 lipatan)

| h | SARIMA musiman 12 | SARIMA mingguan | resep final |
|---|---|---|---|
| 1 | 0.6716 | 0.5962 | 0.5925 |
| 7 | 0.9207 | 0.8890 | 0.9017 |
| 30 | 0.7633 | 0.7303 | 0.7040 |

Catatan: angka BiLSTM paper (fold-0 saja): h1=0.5915, h7=0.7603, h30=0.6289 — perbandingan langsung harus dihitung pada lipatan yang sama.
