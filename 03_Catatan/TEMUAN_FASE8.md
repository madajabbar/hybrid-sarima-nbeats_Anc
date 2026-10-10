# Temuan Fase 8 — varian Fourier diuji semua polutan + diagnosis jarak 7 hari

## A. Varian C (paper + Fourier mingguan) pada ketiga polutan

Efek pada SARIMA saja (Δ vs varian A), 3 lipatan:

| polutan | h | Δ RMSE |
|---|---|---|
| NO2 | 1 | **−7.2%** |
| NO2 | 7 | **−2.7%** |
| NO2 | 30 | **−8.9%** |
| PM10 | 1 | +0.6% |
| PM10 | 7 | +0.4% |
| PM10 | 30 | +1.2% |
| PM25 | 1 | +1.8% |
| PM25 | 7 | +1.2% |
| PM25 | 30 | +1.1% |

**Kesimpulan:** siklus mingguan sebagai variabel eksogen hanya menolong **NO2**; di PM10/PM25 malah sedikit
merugikan. Tafsir fisik konsisten: NO2 digerakkan lalu lintas (pola mingguan kuat), partikulat lebih
ditentukan cuaca/musim. Jadi varian C **bukan** resep umum — hanya alternatif khusus NO2.

## B. Diagnosis jarak 7 hari (`scripts/10_diagnose_h7.py`, 72 titik ujian per sel)

| polutan | varian | SARIMA | resep | koreksi penuh | batas terbaik (alfa sempurna) |
|---|---|---|---|---|---|
| NO2 | A (paper) | 0.8921 | 0.9149 | 0.9275 | 0.8020 |
| NO2 | B (mingguan) | **0.8598** | 0.8698 | 0.9036 | 0.8207 |
| PM25 | A (paper) | 0.0731 | 0.0745 | 0.0750 | 0.0696 |
| PM25 | B (mingguan) | 0.0731 | 0.0744 | 0.0749 | 0.0695 |

Rincian per titik ujian:

| polutan | varian | membaik | alfa=0 | korelasi(alfa, kerugian) | korelasi(|koreksi|, kerugian) | celah ke batas |
|---|---|---|---|---|---|---|---|
| NO2 | A | 30/72 | 0 | +0.04 | −0.08 | 12.3% |
| NO2 | B | 24/72 | 26 | −0.08 | +0.10 | **5.6%** |
| PM25 | A | 20/72 | 21 | +0.30 | **+0.47** | 6.6% |
| PM25 | B | 19/72 | 21 | +0.29 | **+0.46** | 6.6% |

**Temuan diagnosis:**

1. **PM25 jarak 7:** hanya 20 dari 72 titik ujian terbantu. Korelasi antara besar koreksi dan kerugian
   **positif kuat (+0.47)** — artinya koreksi residu di PM25 bukan sekadar tidak informatif, ia
   cenderung menyesatkan, dan bobot alfa tetap memberinya ruang (rata-rata 0.40). Jadi masalahnya
   **bukan pada bobotnya, tetapi pada koreksinya**: tidak ada informasi yang bisa dipelajari dari
   sisa kesalahan SARIMA PM25 pada jarak 7 hari.
2. **NO2 jarak 7:** dengan varian A, validasi **selalu** menjawab "pakai" (0 dari 72 titik beralfa 0)
   tetapi celah ke batas terbaik 12.3% — pemilihan bobotnya terlalu permisif. Setelah musiman
   dibetulkan (varian B), 26 titik beralfa 0 dan celah menyempit ke **5.6%** → perbaikan musiman juga
   memperbaiki disiplin bobot.
3. "Batas terbaik" (alfa dipilih sempurna per titik) hanya 6.6% lebih baik dari resep untuk PM25 jarak 7 —
   dan itu **tidak dapat diprediksi dari data masa lalu**. Kesimpulan praktis: untuk PM25 jarak 7,
   keputusan yang benar adalah **jangan pakai koreksi (alfa = 0)**, bukan mencari bobot lebih pintar.

## C. Perbaikan yang diusulkan (calon Fase 9, murah)

Kriteria penerimaan koreksi diganti dari satu jendela validasi (120 hari) menjadi **beberapa jendela
bergulir + suara mayoritas**, dengan syarat tambahan: koreksi hanya dipakai bila penurunan RMSE di
validasi melewati ambang (mis. > 5%) — supaya koreksi yang tak informatif (PM25 h=7) otomatis dimatikan.

## D. Yang masih kurang: pelatihan penuh

1. **Pelatihan penuh versi final (laptop, hitungan menit)** — resep beku × 3 polutan × 3 jarak,
   jumlah lipatan diperbanyak, ditambah ulangan seed, menghasilkan tabel final naskah.
2. **Latih ulang baseline jaringan (BiLSTM; opsional N-BEATS) di GPU Kaggle dengan protokol terkoreksi
   (musiman 7)** — wajib, karena perbandingan saat ini memakai BiLSTM yang masih memakai protokol lama
   (musiman 12) sehingga belum benar-benar apple-to-apple di sisi protokol musiman.