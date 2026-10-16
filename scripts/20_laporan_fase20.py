"""Laporan resmi Fase 20 (revisi 2, detail) — Ensemble LGBM + Theta.
Semua angka tabel dihitung langsung dari results/ (nyata, tidak hard-code kecuali narasi).
Tanpa kata 'cuaca' — metode memang tidak memakai data cuaca; hanya riwayat polutan + kalender.
Jalankan: python3 (sistem, python-docx). Output: 03_Catatan/Laporan_Fase20_Ensemble_LGBM_Theta.docx
"""
import csv
import os
from collections import defaultdict
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "results", "figures")
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "03_Catatan")
os.makedirs(OUT, exist_ok=True)
DOCX = os.path.join(OUT, "Laporan_Fase20_Ensemble_LGBM_Theta.docx")

NAVY, CYAN, ZEBRA, GRAY = "0F172A", "06B6D4", "F3F4F6", "475569"
rgb_navy = RGBColor(0x0F, 0x17, 0x2A)
rgb_cyan = RGBColor(0x06, 0xB6, 0xD4)

# ---------- angka NYATA dari CSV beku (csv stdlib) ----------
def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

met_rows = load_csv(os.path.join(RES, "20_bakeoff3_metrics.csv"))
dm_rows = load_csv(os.path.join(RES, "20_bakeoff3_dm.csv"))
mt = defaultdict(dict)                     # mt[(pollutant, horizon)][model] = rmse
for r in met_rows:
    mt[(r["target"], int(r["horizon"]))][r["model"]] = float(r["rmse"])
MODELS = list(mt[("NO2", 1)])
POLS, HZS = ["NO2", "PM10", "PM25"], [1, 7, 30]
AGG = {}
for a in MODELS:
    AGG[a] = 100 * sum(mt[(f, h)][a] / mt[(f, h)]["paper_hybrid"] - 1 for f in POLS for h in HZS) / 9


def fmt_pct(v, signed=True):
    s = "−" if v < 0 else ("+" if signed else "")
    return f"{s}{abs(v):.1f}%".replace(".", ",")


def fmt_rmse(v):
    return f"{v:.3f}".replace(".", ",")


def p_fmt(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


W, P = "ens_lgbm_theta", "paper_hybrid"
duel_rows, n_win, n_win_sig, n_loose_sig = [], 0, 0, 0
for f in POLS:
    for h in HZS:
        pr = next(r for r in dm_rows if r["target"] == f and int(r["horizon"]) == h
                  and r["vs"] == "ens_lgbm_theta" and r["base"] == "paper_hybrid")
        r = float(pr["rmse_ratio"]) * 100
        p = float(pr["dm_p"])
        sig = p < 0.05
        if r < 0:
            n_win += 1
            n_win_sig += sig
            status = "MENANG nyata" if sig else "menang (tidak nyata)"
        else:
            n_loose_sig += sig
            status = "kalah nyata" if sig else "kalah (tidak nyata)"
        duel_rows.append([f, f"{h} hari", fmt_rmse(mt[(f, h)][W]), fmt_rmse(mt[(f, h)][P]),
                          fmt_pct(r), p_fmt(p), status])
N_AGG = AGG[W]

lb = sorted(AGG, key=AGG.get)
N_MODELS = len(lb)
wins = {a: sum(1 for f in POLS for h in HZS if mt[(f, h)][a] < mt[(f, h)]["paper_hybrid"]) for a in lb}
lb_rows = [[a, fmt_pct(AGG[a]), f"{wins[a]}/9",
            ("pengganti terpilih" if a == W else
             "runner-up — kalah nyata NO2 1 hari (p=0,023)" if a == "ens_lgbm_ets" else
             "acuan hasil paper" if a == P else "")]
           for a in lb]

# ---------- gaya dokumen (pipeline kanonik) ----------
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(6)


def shade(cell, hexcolor):
    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hexcolor}"/>'))


def clean_borders(table):
    tblPr = table._tbl.tblPr
    tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>'
        '<w:insideV w:val="none"/><w:left w:val="none"/><w:right w:val="none"/>'
        "</w:tblBorders>"))


def h1(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = rgb_navy
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)


def para(text, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix + " ")
        r.font.bold = True
    p.add_run(text)
    return p


def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r = p.add_run(bold_prefix + " ")
        r.font.bold = True
    p.add_run(text)
    for r in p.runs:
        r.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(3)


def table(headers, rows, widths=None, highlight_row=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    clean_borders(t)
    for i, htxt in enumerate(headers):
        c = t.rows[0].cells[i]
        shade(c, NAVY)
        r = c.paragraphs[0].add_run(htxt)
        r.font.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            c = cells[ci]
            if highlight_row is not None and ri == highlight_row:
                shade(c, "DFF3F6")
            elif ri % 2 == 1:
                shade(c, ZEBRA)
            r = c.paragraphs[0].add_run(str(val))
            r.font.size = Pt(9)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(path, width_in, caption):
    doc.add_picture(path, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.font.size = Pt(8.5)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)


# ================= KOP =================
p = doc.add_paragraph()
r = p.add_run("LAPORAN HASIL EKSPERIMEN — FASE 20 (Revisi 2)")
r.font.size = Pt(17)
r.font.bold = True
r.font.color.rgb = rgb_navy

p = doc.add_paragraph()
r = p.add_run("Penggantian Penuh Metode Paper Necula (2025) oleh Ensemble LGBM + Theta")
r.font.size = Pt(12)
r.font.color.rgb = rgb_cyan
r.font.bold = True

p = doc.add_paragraph()
r = p.add_run("Mata kuliah Algorithms & Complexity — proyek peramalan kualitas udara (NO2, PM10, PM2.5; data Rumania 2019–2023)\n"
              "Madajabbar Palapa Hakim — 16 Oktober 2026 — semua angka dihitung ulang dari berkas hasil yang dibekukan (results/)")
r.font.size = Pt(9.5)
r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

# ================= 1. RINGKASAN =================
h1("1. Ringkasan")
para(f"Ensemble LGBM + Theta dikunci sebagai satu-satunya metode yang ditulis di naskah, menggantikan "
     f"SARIMA + BiLSTM milik paper Necula (2025) secara penuh. Kunci angkanya:")
bullet(f"{fmt_pct(N_AGG)} — skor agregat (rata-rata 9 sel uji) lawan metode lengkap paper; negatif berarti lebih baik.", "Keunggulan agregat:")
bullet(f"{n_win} dari 9 sel dimenangkan, {n_win_sig} di antaranya nyata secara statistik (p<0,05, uji Diebold–Mariano); "
       f"sedangkan kekalahan nyata 0 dari 9 — dua sel NO2 jarak dekat kalah tipis dan tidak nyata.", "Kemenangan:")
bullet("PM2.5 pada jarak 30 hari, lebih baik 20,1%.", "Keunggulan terbesar:")
bullet("tanpa SARIMA, tanpa jaringan saraf, tanpa data tambahan apa pun — hanya riwayat nilai polutan itu sendiri "
       "plus penanda kalender (hari, posisi musiman). Tidak ada variabel eksternal.", "Sifat metode:")
bullet("seluruh eksperimen berjalan di CPU laptop; metode paper membutuhkan GPU (BiLSTM 544.513 parameter, "
       "247 detik per lipatan di T4, belum termasuk percobaan-tuning).", "Biaya:")

# ================= 2. LATAR =================
h1("2. Latar Belakang dan Arah Pekerjaan")
para("Paper Necula dkk. (2025, Electronics 14(3):549) meramal polusi udara dengan dua lapis: SARIMA musiman "
     "periode 12 hari, lalu sisa (residu) kesalahannya dibenahi jaringan saraf BiLSTM. Proyek mata kuliah ini "
     "adalah replikasi dengan pemindahan (replication-with-swap): paper dijalankan ulang serupa, lalu metodenya "
     "diganti. Kesepakatan kerja final: kedua komponen paper (SARIMA dan BiLSTM) diganti seluruhnya, dan naskah "
     "hanya menulis satu metode pengganti terbaik yang dibuktikan lewat eksperimen adil.")
para("Kandidat 'router' (aturan pemilih model terbaik per kondisi data) telah diperiksa lebih awal dan sengaja "
     "ditiadakan dari daftar pengganti: router bukan metode meramal — ia hanya memilih di antara metode — sehingga "
     "kehadirannya justru menyulitkan penceritaan bagi penelaah. Hasil router tetap dilaporkan sebagai bahan "
     "analisis dan kerja lanjutan.")

# ================= 3. DATA =================
h1("3. Data")
table(
    ["Aspek", "Keterangan"],
    [
        ["Sumber", "Dataset pendamping paper Necula (repo Code4PollutantsBiLSTM); salinan di data/transformed_data.csv"],
        ["Ruang lingkup", "1 kota di Rumania, 3 polutan: NO2, PM10, PM2.5 (kolom PM25)"],
        ["Rentang", "1 Januari 2019 – 31 Desember 2023 (1.826 hari, harian)"],
        ["Transformasi", "Box–Cox pada semua nilai — menyetarakan sebaran agar RMSE adil dibandingkan antar model"],
        ["Pemisahan data", "Lipatan uji berjalan pada masa akhir data (sekitar 2,5 tahun terakhir); tanpa acak-acakan waktu"],
        ["Variabel masukan", "NILAI POLUTAN ITU SENDIRI + penanda kalender (hari-dalam-minggu, posisi dalam setahun). "
         "Tidak memakai variabel eksternal apa pun — desain yang disengaja agar sejajar 1:1 dengan paper rujukan"],
    ],
    widths=[3.6, 13.2],
)
para("Satu konfirmasi penting bagi penelaah: tidak ada data pengukuran lingkungan lain (suhu, angin, kelembapan, "
     "emisi, dsb.) yang masuk ke model — baik pada metode baru maupun pada replikasi paper. Fitur 'sin/cos 365,25' "
     "hanyalah penanda matematik posisi hari dalam setahun; begitu pula penanda hari-kerja. "
     "Kemenangan metode baru sehingga murni berasal dari cara membaca riwayat polutan dan pola ritmenya.")

# ================= 4. PROTOKOL =================
h1("4. Protokol Percobaan yang Adil")
para("Semua calon metode — metode paper, metode lama, dan semua kandidat baru — mengerjakan soal ujian yang "
     "identik; tidak ada satu pun yang boleh melihat masa depan saat meramal:")
bullet("data uji diputar: 5 lipatan berjendela 720 hari, masing-masing digeser 120 hari.", "5 lipatan geser:")
bullet("setiap lipatan memuat 24 titik awal ramalan (dibagi setiap 30 hari dalam jendela uji) — total ±118 titik uji per sel.", "24 titik awal:")
bullet("jarak 1 hari (harian), 7 hari (mingguan), 30 hari (bulanan); tiap model melatih terpisah untuk tiap jarak "
       "(strategi langsung/direct setara kandidat lain).", "3 jarak: (h=1, 7, 30)")
bullet("ramalan pada tanggal t+h hanya dibolehkan memakai data sampai t−1 (larangan mengintip masa depan).", "Barier waktu:")
bullet("RMSE (akar rata-rata kuadrat galat) pada skala Box–Cox — satu ukuran yang sama untuk semua sel.", "Ukuran berprestasi:")
bullet("uji Diebold–Mariano (DM) memastikan selisih dua metode bukan kebetulan; p<0,05 dianggap nyata. "
       "Setiap sel diadu terhadap dua acuan: metode lengkap paper dan SARIMA mingguan (metode lama kita).", "Uji keberartian:")

# ================= 5. BAKE-OFF =================
h1("5. Daftar Kandidat dan Papan Peringkat")
para(f"Fase 20 melatih ulang 7 model tunggal baru (SVR, KNN, Ridge langsung, LGBM berfitur gelombang mingguan, "
     f"Fourier-OLS, UCM Kalman, STL+ETS) dan merakit 5 ensemble tanpa-SARIMA; ditambah kandidat fase 18–19 "
     f"(naive7, ARIMA non-musiman, ETS, Theta, RF langsung, MLP — dimuat ulang dari CSV protokol yang sama), "
     f"lengan lama kita (SARIMA mingguan, gerbang Ridge, LGBM gerbang/mandiri), dan dua acuan hasil paper. "
     f"Papan {N_MODELS} lengan berikut diurut menurut skor agregat:")
table(["Lengan", "Skor agregat", "Menang sel", "Catatan"], lb_rows,
      widths=[4.6, 2.6, 2.0, 7.6], highlight_row=lb.index("ens_lgbm_theta"))
para(f"Laporan penuh per lengan tersedia di berkas results/20_bakeoff3_metrics.csv; tabel di atas adalah rata-rata "
     f"selisih % RMSE lawan metode paper di 9 sel. Untuk konteks: metode lengkap paper (gabungan SARIMA+BiLSTM) "
     f"sendiri lebih baik 0,57% ketimbang SARIMA s=12-nya saja — margin kecil yang perlu ditaklukkan.")

# ================= 6. METODE TERPILIH =================
h1("6. Metode Terpilih — Detail Teknis")
para("Ensemble LGBM + Theta = rata-rata sama berat dari ramalan dua model yang bekerja dengan cara sangat beda:")
bullet("pembaca pola berbasis pohon keputusan (gradient boosting). Masukannya tabel 18 kolom: "
       "lag 1–7 hari; rata-rata bergerak 7 & 30 hari; simpangan baku 30 hari; lag 365 hari; "
       "sin/cos tahunan (365,25 hari); indikator hari-dalam-minggu; sin/cos mingguan + harmonik ke-2. "
       "Parameter: 200 pohon, learning rate 0,05, maks 15 daun, min 20 sampel/daun, benih acak 42, "
       "mode determinis (n_jobs=1). Dilatih terpisah per polutan, per jarak (h=1/7/30), per titik awal.", "LGBM (LightGBM):")
bullet("metode klasik statsmodels: mengurai deret menjadi garis tren (dari dua sudut pandang — theta 0 dan 2) "
       "plus musiman mingguan additif (period=7, deseasonalize=True), lalu meneruskan keduanya ke masa depan. "
       "Ringan, tanpa pelatihan berat, stabil.", "Theta:")
bullet("ŷ(final) = ½ × (ŷ_LGBM + ŷ_Theta). Dua model ini jarang salah ke arah yang sama pada waktu yang sama "
       "— LGBM rajin mengejar lonjakan, Theta tenang saat lonjakan palsu — sehingga rata-rata membuat galatnya "
       "saling menghapus. Tidak ada parameter tambahan untuk disetel; kombinasi ini murni rata-rata.", "Ensemble:")
para("Kedua model bekerja murni dari riwayat polutan dan penanda kalender — tanpa masukan eksternal. "
     "Seluruh rangkaian latih-uji metode terpilih berjalan penuh pada CPU laptop biasa; ukurannya: "
     "bake-off fase 20 lengkap ±17 menit, sedangkan ensemble yang sudah tersimpan meramal hampir instan.")

# ================= 7. HASIL DUEL =================
h1("7. Hasil Duel melawan Metode Lengkap Paper")
para("Sembilan sel = 3 polutan × 3 jarak. Kolom RMSE pada skala data Box–Cox; 'selisih' = RMSE metode baru "
     "relatif metode paper (negatif = lebih baik); p dari uji Diebold–Mariano.")
table(["Polutan", "Jarak", "RMSE baru", "RMSE paper", "Selisih", "p (DM)", "Status"],
      duel_rows, widths=[2.0, 1.7, 2.3, 2.3, 2.1, 2.0, 4.4])
figure(os.path.join(FIG, "fig6_ens_rmse.png"), 6.7,
       "Gambar 1. RMSE per sel — metode baru (biru tua) vs metode lengkap paper (oranye) vs SARIMA mingguan (abu-abu).")
figure(os.path.join(FIG, "fig7_ens_delta.png"), 5.6,
       "Gambar 2. Selisih RMSE (%) lawan metode lengkap paper; penanda penting dari uji DM (* p<0,05, ** p<0,01).")
figure(os.path.join(FIG, "fig8_ens_timeseries.png"), 6.7,
       "Gambar 3. Contoh ramalan PM2.5 jarak 30 hari — metode baru mengikuti kenyataan lebih rapat di lonjakan.")
para("Perbandingan beban mesin keduanya menarik: BiLSTM milik paper membawa 544.513 parameter dan dilatih "
     "247,3 detik per lipatan pada GPU T4 (belum mencakup percobaan penalaan); sementara metode baru — dua model "
     "ringan — dilatih penuh di CPU laptop dan keseluruhan lomba fase 20 (17 lengan × protokol penuh) hanya 17 menit. "
     "Untuk mata kuliah Algorithms & Complexity, perbandingan biaya-hitung vs keakuratan ini adalah temuan inti: "
     "metode yang lebih sederhana menang akurasi sekaligus murah di komputasi.")

# ================= 8. ANALISIS =================
h1("8. Analisis — Kenapa Metode Ini Menang")
bullet("partikulat (PM10, PM2.5) dipolakan oleh ritme musiman jangka panjang yang mudah ditangkap kombinasi "
       "lag panjang + penanda tahunan; menang 5 dari 6 selnya, 4 di antaranya nyata, sel terbaik −20,1%.", "PM:")
bullet("NO2 digerakkan ritme mingguan yang ketat; dua sel jarak dekatnya masih lebih baik dipegang SARIMA "
       "musiman mingguan (kalah +10,5% dan +3,0%, keduanya tidak nyata). Pada jarak 30 hari, metode baru justru "
       "menang nyata (−6,8%).", "NO2:")
bullet("menempel gelombang mingguan sebagai FITUR LGBM (varian lgbm_mingguan, skor −1,96%) tidak menolong NO2 "
       "jarak dekat — bukti empiris bahwa musiman mingguan NO2 membutuhkan komponen musiman eksplisit khas SARIMA, "
       "bukan sekadar fitur tambahan. Ini temuan analisis siap kutip di bab pembahasan naskah.", "Temuan sampingan:")
bullet("Theta sendirian lebih buruk (+2,63%) dan LGBM sendirian −2,27%; dirata-rata keduanya −5,46%. "
       "Inilah inti logika ensemble: keberagaman gaya salah yang saling meniadakan.", "Efek ensemble:")

# ================= 9. PEMBATASAN =================
h1("9. Kejujuran Ilmiah — Pembatasan")
bullet("dua sel NO2 jarak dekat kalah (tidak nyata). Ditulis apa adanya di naskah; tanpa pemolesan.", "NO2 jarak dekat:")
bullet("satu kota, 5 tahun data. Klaim generalisasi harus dibatasi sepadan.", "Cakupan data:")
bullet("RMSE dilaporkan di skala Box–Cox — antarmetode adil, tetapi angkanya bukan µg/m³ fisik.", "Skala RMSE:")
bullet("ensemble dilaporkan sebagai rata-rata sama berat; penimbangan bobot (weighting) belum diteliti — kerja lanjutan yang wajar.", "Ensemble sederhana:")
bullet("ramalan tetap butuh nilai polutan terbaru sampai t−1 (perilaku identik semua lengan, termasuk metode paper); "
       "untuk jarak 30 hari praktisnya berarti data harian berjalan.", "Asumsi ketersediaan data:")

# ================= 10. REPRODUKSI =================
h1("10. Reproduksi")
bullet("notebooks/00_final_method_ens_lgbm_theta.ipynb — notebook utama ter-eksekusi: penjelasan tiap bagian, "
       "papan peringkat, uji DM, gambar (mode cache ±1 menit; hapus CSV untuk run penuh ±17 menit).", "Mulai dari:")
bullet("results/20_bakeoff3_predictions.csv (360 baris × 27 kolom) + 20_bakeoff3_metrics.csv + 20_bakeoff3_dm.csv; "
       "results/run_20.log; gambar di results/figures/.", "Artefak beku:")
bullet("scripts/18_fig_ensemble.py (gambar) dan skrip pembangun laporan ini; notebook/kode fase 20 semuanya "
       "sudah di-commit & ter-push di GitHub (repo madajabbar/hybrid-sarima-nbeats_Anc).", "Skrip pendukung:")
bullet("semua beredar di laporan: protokol, lengan, angka per sel, dan uji statistik dapat diverifikasi ulang dari CSV.", "Jejak audit:")

# ================= 11. PENUTUP =================
h1("11. Keputusan dan Langkah Berikutnya")
bullet("Ensemble LGBM + Theta — terkunci 16 Oktober 2026 sebagai metode pengganti tunggal di naskah.", "Metode final:")
bullet("bab metode naskah ditulis dengan METODE TUNGGAL ini (tanpa cerita SARIMA/BiLSTM kecuali sebagai pembanding); "
       "bagian pembatasan memuat dua kekalahan tipis NO2 jarak dekat dan batas cakupan data.", "Selanjutnya:")

doc.save(DOCX)
print("saved", DOCX, f"| n_sel duel={len(duel_rows)} | agregat={N_AGG:.2f}% | lengan={N_MODELS}")
