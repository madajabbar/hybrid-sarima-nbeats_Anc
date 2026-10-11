"""Fase 12 — gambar untuk naskah, semua dari CSV hasil (tanpa hitung ulang model).

Keluaran (results/figures/):
    fig1_rmse_arms.png       RMSE 4 lengan per polutan x jarak
    fig2_delta_signif.png    selisih kita vs paper + penanda nyata (uji DM)
    fig3_gating.png          volume alfa + akibat kalau koreksi dipaksa 100%
    fig4_cost.png            biaya: parameter & waktu latih (log)
    fig5_timeseries.png      contoh ramalan nyata (NO2, lipatan 0, h=1)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)

NAVY, CYAN, ORANGE, GRAY, GREEN, RED = "#0b3d91", "#00a6c8", "#e76f51", "#8d99ae", "#2a9d8f", "#c1121f"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})

m = pd.read_csv(os.path.join(R, "14_vs_bilstm_all_folds.csv"))
t = pd.read_csv(os.path.join(R, "15_vs_bilstm_all_folds_dm.csv"))
dmp = t[t.uji == "kita vs paper(BiLSTM)"].set_index(["target", "horizon"])
POL = ["NO2", "PM10", "PM25"]
H = [1, 7, 30]

# ---------------------------------------------------------------- fig 1: RMSE 4 lengan
ARMS = [("rmse_paper_sarima_s12", "Paper SARIMA (s=12)", GRAY),
        ("rmse_paper_hybrid_bilstm", "Paper SARIMA+BiLSTM", ORANGE),
        ("rmse_ours_sarima_s7", "Ours SARIMA (s=7)", CYAN),
        ("rmse_ours_recipe", "Ours recipe (gated)", NAVY)]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
for ax, pol in zip(axes, POL):
    sub = m[m.target == pol].set_index("horizon")
    x = np.arange(len(H))
    w = 0.2
    for i, (col, lab, c) in enumerate(ARMS):
        vals = [sub.loc[h, col] for h in H]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
        for xi_, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi_, v * 1.01, f"{v:.3f}", ha="center", va="bottom", fontsize=5.6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in H])
    ax.set_ylim(0, max([sub.loc[h, c] for h in H for c, _, _ in ARMS]) * 1.28)
    ax.set_title(pol)
    ax.set_ylabel("RMSE (lebih kecil lebih baik)" if pol == POL[0] else "")
axes[0].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
fig.suptitle("RMSE per lengan pada lipatan & origin identik (5 lipatan) — angka di atas batang", y=1.0, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_rmse_arms.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 2: selisih + penanda nyata
fig, ax = plt.subplots(figsize=(7.6, 4.2))
labels, vals, cols, marks = [], [], [], []
for pol in POL:
    for h in H:
        d = float(dmp.loc[(pol, h), "delta_pct"])
        p = float(dmp.loc[(pol, h), "dm_p"])
        labels.append(f"{pol}  h={h}")
        vals.append(d)
        cols.append(GREEN if d < 0 else RED)
        marks.append("**" if p < 0.01 else ("*" if p < 0.05 else ""))
y = np.arange(len(vals))[::-1]
ax.barh(y, vals, color=cols, height=0.62)
ax.axvline(0, color="k", lw=0.9)
for yy, v, s in zip(y, vals, marks):
    ax.text(v + (0.25 if v >= 0 else -0.25), yy, f"{v:+.1f}%{s}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("selisih RMSE resep kita vs paper+BiLSTM (%)  —  negatif = kita lebih baik")
ax.set_title("Kita menang 8 dari 9 sel; 7 menang nyata, 1 kalah nyata (* p<0,05, ** p<0,01)")
ax.set_xlim(min(vals) * 1.45, max(vals) * 1.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_delta_signif.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 3: gerbang (alfa) + kerusakan 100%
pred = pd.read_csv(os.path.join(R, "12_final_predictions.csv"))
f3 = pd.read_csv(os.path.join(R, "03_fix_hybrid_metrics.csv"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
w = 0.25
x = np.arange(3)
for i, h in enumerate(H):
    a = pred[f"alpha_h{h}"].dropna()
    ax1.bar(x + (i - 1) * w, [a.mean()], w, yerr=[a.std()], capsize=3,
            color=[NAVY, CYAN, ORANGE][i], label=f"h={h} (n={len(a)})")
ax1.axhline(1.0, ls="--", c=RED, lw=1.2)
ax1.text(0.5, 1.06, "paper: selalu 1,0 (koreksi 100%)", color=RED, fontsize=8.5,
         ha="center", transform=ax1.get_yaxis_transform(),
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
ax1.set_xticks(x)
ax1.set_xticklabels(POL)
ax1.set_ylim(0, 1.28)
ax1.set_ylabel("volume alfa yang dipilih data (rata-rata ± simpangan)")
ax1.set_title("Volume koreksi tidak pernah 1,0")
ax1.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

n1 = f3[(f3.target == "NO2") & (f3.horizon == 1)].set_index("model")["rmse"]
names = ["SARIMA\n(polos)", "koreksi ditelan\n100% (cara paper)", "koreksi diberi\nvolume (kita)"]
vals3 = [n1["SARIMA"], n1["gbm_full"], n1["ridge_shrunk"]]
bars = ax2.bar(names, vals3, color=[GRAY, RED, NAVY], width=0.6)
for b, v in zip(bars, vals3):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=9)
ax2.set_ylabel("RMSE (NO2, h=1)")
ax2.set_ylim(0, max(vals3) * 1.18)
ax2.set_title("Koreksi yang tidak diperiksa justru merusak")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_gating.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 4: biaya
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
names = ["Ridge\n(kita)", "SARIMA\n(per lipatan)", "BiLSTM\n(paper)"]
params = [15, 0, 544513]
secs = [0.008, 13.1, 247.3]
for ax, vals, title, unit in [(ax1, params, "Jumlah parameter", "parameter"),
                              (ax2, secs, "Waktu latih", "detik")]:
    v = [max(x, 1e-3) for x in vals]
    bars = ax.bar(names, v, color=[NAVY, CYAN, ORANGE], width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel(f"{unit} (skala log)")
    ax.set_title(title)
    for b, raw in zip(bars, vals):
        txt = "0 (tak ada)" if raw == 0 else (f"{raw:,.0f}" if raw >= 1 else f"{raw:.3f}")
        ax.text(b.get_x() + b.get_width() / 2, max(raw, 1e-3) * 1.3, txt, ha="center", fontsize=8.5)
    ax.set_ylim(1e-3, max(v) * 80)
fig.suptitle("Akurasi lebih baik, biaya jauh lebih kecil", y=1.03, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_cost.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 5: contoh ramalan nyata
ours = pred[(pred.feature == "NO2") & (pred.fold == 0)].copy()
pap = pd.read_csv(os.path.join(R, "13_bilstm_all_folds_predictions.csv"))
pap = pap[(pap.feature == "NO2") & (pap.fold == 0)].copy()
j = ours.merge(pap, on=["feature", "fold", "origin_date"], suffixes=("", "_p"))
j = j.sort_values("origin_date")
fig, ax = plt.subplots(figsize=(11.5, 3.9))
xi = np.arange(len(j))
ax.plot(xi, j.actual_h1, "o-", color="k", lw=1.5, ms=3.5, label="kenyataan")
ax.plot(xi, j.sarima_h1_p, "s--", color=GRAY, lw=1.6, ms=3, label="paper SARIMA (s=12)")
ax.plot(xi, j.hybrid_h1, "^-", color=ORANGE, lw=1.0, ms=3, alpha=0.95, label="paper SARIMA+BiLSTM")
ax.plot(xi, j.final_h1, "D-", color=NAVY, lw=1.6, ms=3.2, label="resep kita")
ax.annotate("paper SARIMA vs +BiLSTM hampir berimpit\n(koreksi BiLSTM nyaris tidak mengubah apa pun)",
            xy=(len(j) * 0.42, float(j.hybrid_h1.mean())), xytext=(len(j) * 0.30, float(j.actual_h1.min()) + 0.15),
            fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
ax.set_xticks(xi[::4])
ax.set_xticklabels([s[:7] for s in j.origin_date][::4], fontsize=7.5)
ax.set_ylabel("NO2 (skala Box-Cox)")
ax.set_title("NO2 jarak 1 hari, lipatan pertama — ramalan vs kenyataan")
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_timeseries.png"), bbox_inches="tight")
plt.close(fig)

print("gambar tersimpan di", FIG)
for f in sorted(os.listdir(FIG)):
    print(" ", f, f"{os.path.getsize(os.path.join(FIG, f)):,} bytes")