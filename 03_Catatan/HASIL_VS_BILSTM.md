# Adu final: resep kita vs metode paper (SARIMA + BiLSTM)

Lipatan & origin identik untuk semua lengan (5 lipatan, hingga 120 origin per sel).

| polutan | h | paper SARIMA s12 | paper +BiLSTM | kita SARIMA s7 | resep kita | kita vs paper |
|---|---|---|---|---|---|---|
| NO2 | 1 | 0.6716 | 0.6622 | 0.5962 | 0.5925 | -10.5% |
| NO2 | 7 | 0.9207 | 0.9310 | 0.8890 | 0.9017 | -3.1% |
| NO2 | 30 | 0.7633 | 0.7753 | 0.7303 | 0.7040 | -9.2% |
| PM10 | 1 | 0.1172 | 0.1187 | 0.1145 | 0.1114 | -6.1% |
| PM10 | 7 | 0.1553 | 0.1546 | 0.1536 | 0.1499 | -3.0% |
| PM10 | 30 | 0.1498 | 0.1541 | 0.1525 | 0.1514 | -1.8% |
| PM25 | 1 | 0.0650 | 0.0648 | 0.0650 | 0.0638 | -1.5% |
| PM25 | 7 | 0.0755 | 0.0753 | 0.0754 | 0.0753 | +0.1% |
| PM25 | 30 | 0.0915 | 0.0918 | 0.0914 | 0.0891 | -2.9% |

## Uji Diebold-Mariano (p < 0,05 = beda nyata)

| polutan | h | uji | delta | p | nyata |
|---|---|---|---|---|---|
| NO2 | 1 | kita vs paper(BiLSTM) | -11.13% | 0.0001 | ya |
| NO2 | 1 | kita vs paper(SARIMA s12) | -12.30% | 0.0001 | ya |
| NO2 | 1 | paper: BiLSTM vs SARIMA-nya sendiri | -1.39% | 0.1314 | tidak |
| NO2 | 1 | kita vs SARIMA mingguan kita | -1.37% | 0.2130 | tidak |
| NO2 | 7 | kita vs paper(BiLSTM) | -3.77% | 0.1483 | tidak |
| NO2 | 7 | kita vs paper(SARIMA s12) | -2.67% | 0.2322 | tidak |
| NO2 | 7 | paper: BiLSTM vs SARIMA-nya sendiri | +1.12% | 0.0017 | ya |
| NO2 | 7 | kita vs SARIMA mingguan kita | +1.02% | 0.4391 | tidak |
| NO2 | 30 | kita vs paper(BiLSTM) | -6.13% | 0.0413 | ya |
| NO2 | 30 | kita vs paper(SARIMA s12) | -4.87% | 0.0959 | tidak |
| NO2 | 30 | paper: BiLSTM vs SARIMA-nya sendiri | +1.57% | 0.0618 | tidak |
| NO2 | 30 | kita vs SARIMA mingguan kita | -0.36% | 0.3275 | tidak |
| PM10 | 1 | kita vs paper(BiLSTM) | -6.66% | 0.0000 | ya |
| PM10 | 1 | kita vs paper(SARIMA s12) | -5.23% | 0.0044 | ya |
| PM10 | 1 | paper: BiLSTM vs SARIMA-nya sendiri | +1.22% | 0.2599 | tidak |
| PM10 | 1 | kita vs SARIMA mingguan kita | -2.99% | 0.0479 | ya |
| PM10 | 7 | kita vs paper(BiLSTM) | -2.78% | 0.0453 | ya |
| PM10 | 7 | kita vs paper(SARIMA s12) | -3.04% | 0.0821 | tidak |
| PM10 | 7 | paper: BiLSTM vs SARIMA-nya sendiri | -0.46% | 0.5235 | tidak |
| PM10 | 7 | kita vs SARIMA mingguan kita | -1.95% | 0.1081 | tidak |
| PM10 | 30 | kita vs paper(BiLSTM) | -1.59% | 0.0005 | ya |
| PM10 | 30 | kita vs paper(SARIMA s12) | +1.17% | 0.0989 | tidak |
| PM10 | 30 | paper: BiLSTM vs SARIMA-nya sendiri | +2.85% | 0.0068 | ya |
| PM10 | 30 | kita vs SARIMA mingguan kita | -0.68% | 0.0043 | ya |
| PM25 | 1 | kita vs paper(BiLSTM) | -2.17% | 0.0361 | ya |
| PM25 | 1 | kita vs paper(SARIMA s12) | -2.45% | 0.0379 | ya |
| PM25 | 1 | paper: BiLSTM vs SARIMA-nya sendiri | -0.27% | 0.7226 | tidak |
| PM25 | 1 | kita vs SARIMA mingguan kita | -2.39% | 0.0312 | ya |
| PM25 | 7 | kita vs paper(BiLSTM) | +2.08% | 0.0180 | ya |
| PM25 | 7 | kita vs paper(SARIMA s12) | +1.82% | 0.0000 | ya |
| PM25 | 7 | paper: BiLSTM vs SARIMA-nya sendiri | -0.26% | 0.5159 | tidak |
| PM25 | 7 | kita vs SARIMA mingguan kita | +1.84% | 0.0000 | ya |
| PM25 | 30 | kita vs paper(BiLSTM) | -1.83% | 0.0085 | ya |
| PM25 | 30 | kita vs paper(SARIMA s12) | -1.46% | 0.0012 | ya |
| PM25 | 30 | paper: BiLSTM vs SARIMA-nya sendiri | +0.38% | 0.5746 | tidak |
| PM25 | 30 | kita vs SARIMA mingguan kita | -1.37% | 0.0009 | ya |

Catatan: selisih pada uji DM dihitung hanya pada origin yang berpasangan (kedua lengan sama-sama tersedia), sehingga bisa sedikit berbeda dari selisih kolom RMSE di tabel atas (yang memakai seluruh origin tiap lengan).

Ringkasan: resep kita vs paper+BiLSTM menang 8/9 sel (rata-rata -4.25%), menang nyata 7 sel, kalah nyata 1 sel. Lengan BiLSTM paper tidak memperbaiki SARIMA-nya sendiri secara nyata di sel mana pun.
