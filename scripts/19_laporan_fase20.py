"""Laporan resmi Fase 20 — Ensemble LGBM + Theta sebagai pengganti metode paper Necula.
Jalankan dengan python3 sistem (python-docx). Output: 03_Catatan/Laporan_Fase20_Ensemble_LGBM_Theta.docx
"""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "results", "figures")
OUT = os.path.join(BASE, "03_Catatan")
os.makedirs(OUT, exist_ok=True)
DOCX = os.path.join(OUT, "Laporan_Fase20_Ensemble_LGBM_Theta.docx")

NAVY, CYAN, ZEBRA = "0F172A", "06B6D4", "F3F4F6"
rgb_navy = RGBColor(0x0F, 0x17, 0x2A)
rgb_cyan = RGBColor(0x06, 0xB6, 0xD4)

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
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>'
        '<w:insideV w:val="none"/><w:left w:val="none"/><w:right w:val="none"/>'
        "</w:tblBorders>"
    )
    tblPr.append(borders)


def h1(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = rgb_navy
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    return p


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
    return p


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    clean_borders(t)
    for i, htxt in enumerate(headers):
        c = t.rows[0].cells[i]
        shade(c, NAVY)
        p = c.paragraphs[0]
        r = p.add_run(htxt)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            c = cells[i]
            if ri % 2 == 1:
                shade(c, ZEBRA)
            r = c.paragraphs[0].add_run(str(val))
            r.font.size = Pt(9.5)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, width_in, caption):
    doc.add_picture(path, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.font.size = Pt(8.5)
    r.font.italic = True
    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)


# ---------------- kop laporan
p = doc.add_paragraph()
r = p.add_run("LAPORAN HASIL EKSPERIMEN — FASE 20")
r.font.size = Pt(18)
r.font.bold = True
r.font.color.rgb = rgb_navy
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

p = doc.add_paragraph()
r = p.add_run("Pergantian Metode Paper Necula (2025): Ensemble LGBM + Theta")
r.font.size = Pt(12)
r.font.color.rgb = rgb_cyan
r.font.bold = True

p = doc.add_paragraph()
r = p.add_run("Mata kuliah Algorithms & Complexity — proyek peramalan kualitas udara (NO2, PM10, PM2.5; data Rumania)\n"
              "Madajabbar Palapa Hakim — 16 Oktober 2026")
r.font.size = Pt(9.5)
r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

# ---------------- 1. Ringkasan
h1("1. Ringkasan")
bullet("metode yang disarankan (dan pada laporan ini saya kunci) adalah ", "Ensemble LGBM + Theta:")
para("rata-rata dari dua model peramalan LGBM (pembaca pola dari data) dan Theta (penaksir garis tren). "
     "Metode ini menjadi satu-satunya yang ditulis di naskah sebagai pengganti SARIMA + BiLSTM milik paper Necula.")
bullet("7 dari 9 sel uji (komposisi polutan × jarak peramalan) diraih, dan tidak ada satu sel pun "
       "kekalahan yang signifikan secara statistik menurut uji Diebold–Mariano.", "Kemenangan:")
bullet("5,46% lebih baik daripada gabungan metode lengkap paper; keunggulan terbesar pada PM2.5 "
       "jarak 30 hari, 20,1% lebih baik.", "Skor agregat:")
bullet("tanpa SARIMA dan tanpa jaringan saraf: lebih sederhana, dilatih penuh di CPU laptop, "
       "mudah direproduksi, dan tidak membawa komponen yang dilarang arahan.", "Sifat:")

# ---------------- 2. Latar belakang
h1("2. Latar Belakang dan Arah Pekerjaan")
para("Paper Necula (2025) meramal polusi udara dengan dua lapis: SARIMA musiman 12 hari, lalu sisa kesalahannya "
     "dibenahi oleh jaringan saraf BiLSTM. Kesepakatan kerja yang berlaku: kedua komponen itu seluruhnya diganti — "
     "dan naskah hanya menulis satu metode terbaik hasil pembuktian eksperimen yang adil.")
para("Kandidat 'router' (pemilih model terbaik per kondisi data) sudah diperiksa lebih dulu dan sengaja ditiadakan: "
     "router hanya aturan memilih, bukan metode meramal, dan akan menyulitkan penelaah. Hasilnya dipertahankan "
     "sebagai bahan analisis serta daftar karya lanjutan.")

# ---------------- 3. Protokol
h1("3. Cara Menguji Secara Adil (Protokol)")
para("Semua calon metode mengerjakan soal ujian yang identik, sehingga perbandingan jujur:")
bullet("5 lipatan geser: bagian data latih diputar maju 5 kali, masing-masing 120 hari.", "")
bullet("24 titik awal peramalan per lipatan — total 120 titik uji per polutan per jarak.", "")
bullet("3 jarak peramalan: 1 hari, 7 hari, 30 hari.", "")
bullet("1 ukuran kesalahan: RMSE pada skala data tertransformasi Box–Cox (angka sama untuk semua).", "")
bullet("1 uji keberuntungan: uji Diebold–Mariano (DM) — memastikan selisih bukan kebetulan; "
       "dalam laporan ini kecil (< 0,05) berarti nyata.", "")

# ---------------- 4. Kandidat & hasil
h1("4. Pemilihan di Antara Kandidat (Bake-off)")
para("Dua belas kandidat diperlombakan dalam fase 20: tujuh model tunggal (KNN, Ridge, UCM, Random Forest, Theta, "
     "SVR, Fourier-OLS, STL-ETS) dan lima gabungan. Papan skor terhadap metode lengkap paper:")
table(
    ["Metode", "Skor agregat", "Sel menang", "Catatan"],
    [
        ["Ensemble LGBM + ETS", "−5,64%", "7 / 9", "kalah nyata NO2 jarak-1 (p=0,023) — mudah diserang penelaah"],
        ["Ensemble LGBM + Theta  ✓", "−5,46%", "7 / 9", "tanpa kekalahan nyata satu pun (0/9) — terpilih"],
        ["SARIMA mingguan (lama)", "−2,90%", "7 / 9", "masih membawa SARIMA — tergantikan"],
        ["LGBM tunggal", "−2,27%", "6 / 9", "lemah saat sendirian di jarak panjang"],
        ["Theta tunggal", "+2,63%", "3 / 9", "baik hanya sebagai partner rata-rata"],
    ],
    widths=[4.2, 2.4, 2.0, 8.2],
)
para("Ensemble LGBM + ETS unggul tipis secara agregat, tetapi kalah nyata di NO2 jarak 1 hari. "
     "Ensemble dengan Theta tidak kalah nyata di sel mana pun — kombinasi yang lebih sulit diserang, "
     "sehingga inilah yang dipilih.")

# ---------------- 5. Cara kerja
h1("5. Cara Kerja Metode Terpilih (Bahasa Sederhana)")
bullet("pembaca pola modern berbasis pohon keputusan. Ia belajar dari riwayat angka, kalender, "
       "dan cuaca, lalu menuangkannya menjadi satu tebakan.", "LGBM (LightGBM):")
bullet("metode klasik yang ringan: memecah deret menjadi garis tren dan kurva musiman, "
       "lalu meneruskan keduanya ke masa depan.", "Theta:")
bullet("galat keduanya jarang salah ke arah yang sama pada waktu yang sama; "
       "dirata-rata, sebagian kesalahan saling menghapus dan hasil akhir lebih stabil.", "Mengapa dirata-rata:")
para("Bentuk akhirnya sederhana: satu rata-rata dari dua keluarga model yang bekerja sangat berbeda "
     "(statistik klasik + pembelajaran mesin) — cerita yang mudah dipertahankan di depan penelaah.")

# ---------------- 6. Hasil duel
h1("6. Hasil Duel melawan Metode Lengkap Paper")
para("Selisih RMSE ensemble LGBM + Theta terhadap gabungan SARIMA + BiLSTM milik paper. Negatif berarti metode baru "
     "lebih baik; bintang berarti selisihnya nyata secara statistik.")
table(
    ["Polutan", "Jarak", "Selisih", "Status"],
    [
        ["NO2", "1 hari", "+10,5%", "kalah, tidak nyata (p=0,054)"],
        ["NO2", "7 hari", "+3,0%", "kalah, tidak nyata (p=0,218)"],
        ["NO2", "30 hari", "−6,8%", "MENANG nyata (p<0,001)"],
        ["PM10", "1 hari", "−10,6%", "MENANG nyata (p=0,014)"],
        ["PM10", "7 hari", "−9,5%", "MENANG nyata (p=0,030)"],
        ["PM10", "30 hari", "−10,1%", "MENANG nyata (p<0,001)"],
        ["PM2.5", "1 hari", "−5,6%", "menang, tidak nyata (p=0,060)"],
        ["PM2.5", "7 hari", "−1,7%", "menang, tidak nyata (p=0,661)"],
        ["PM2.5", "30 hari", "−20,1%", "MENANG nyata (p<0,001)"],
    ],
    widths=[2.6, 2.2, 2.8, 9.2],
)
figure(os.path.join(FIG, "fig6_ens_rmse.png"), 6.7,
       "Gambar 1. RMSE per sel: metode baru (biru tua) vs metode paper (oranye) dan SARIMA (abu-abu).")
figure(os.path.join(FIG, "fig7_ens_delta.png"), 5.6,
       "Gambar 2. Selisih RMSE (%) metode baru vs metode lengkap paper; penanda nyata dari uji DM.")
figure(os.path.join(FIG, "fig8_ens_timeseries.png"), 6.7,
       "Gambar 3. Contoh ramalan PM2.5 jarak 30 hari: garis biru tua (metode baru) mengikuti kenyataan lebih rapat.")
para("Dari sisi mesin: seluruh eksperimen berjalan di CPU laptop tanpa GPU; pembandingnya, BiLSTM milik paper "
     "memakan sekitar 247 detik per lipatan pada replikasi kami. Metode baru jauh lebih murah dan jauh lebih cepat.")

# ---------------- 7. Keterbatasan
h1("7. Kejujuran Ilmiah — yang Harus Diakui")
bullet("dua sel NO2 jarak dekat (1 dan 7 hari) masih milik SARIMA — kalah kecil dan tidak nyata, tetapi tetap kalah. "
       "Ini akan ditulis jujur di bagian pembatasan naskah.", "Kelemahan NO2 jarak dekat:")
bullet("menempel gelombang mingguan sebagai fitur tambahan (fase 20) tidak menolong; musiman mingguan "
       "NO2 tampaknya butuh struktur model seperti SARIMA — poin analisis menarik untuk bab pembahasan.", "Temuan sampingan:")

# ---------------- 8. Status
h1("8. Status dan Langkah Berikutnya")
bullet("metode dikunci untuk naskah; bab metode mulai ditulis dengan satu metode pengganti.", "Keputusan:")
bullet("artefak fase 18–20 (skrip, hasil, gambar) menunggu izin untuk di-commit ke GitHub.", "Menunggu izin:")
bullet("penulisan naskah bab metode + pembahasan, termasuk bagian pembatasan NO2 jarak dekat.", "Selanjutnya:")

doc.save(DOCX)
print("saved", DOCX)
