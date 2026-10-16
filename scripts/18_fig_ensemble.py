"""Fase 20 — grafik duel: ensemble LGBM+Theta vs metode lengkap paper Necula.

Keluaran (results/figures/):
    fig6_ens_rmse.png      RMSE 4 lengan per polutan x jarak
    fig7_ens_delta.png     selisih % ensemble vs paper+BiLSTM + penanda DM
    fig8_ens_timeseries.png contoh ramalan (PM25 h=30, lipatan pertama)
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

m = pd.read_csv(os.path.join(R, "20_bakeoff3_metrics.csv"))
dm = pd.read_csv(os.path.join(R, "20_bakeoff3_dm.csv"))
dmp = dm[(dm.vs == "ens_lgbm_theta") & (dm.base == "paper_hybrid")].set_index(["target", "horizon"])
POL, H = ["NO2", "PM10", "PM25"], [1, 7, 30]

ARMS = [("paper_sarima_s12", "Paper SARIMA (s=12)", GRAY),
        ("paper_hybrid", "Paper SARIMA+BiLSTM", ORANGE),
        ("SARIMA_mingguan", "SARIMA mingguan (ku)", CYAN),
        ("ens_lgbm_theta", "Ensemble LGBM+Theta (metode baru)", NAVY)]

# --------------------------------------------- fig 6: RMSE per lengan
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
for ax, pol in zip(axes, POL):
    sub = m[m.target == pol].pivot_table(index="horizon", columns="model", values="rmse")
    x = np.arange(len(H))
    w = 0.2
    for i, (c, lab, col) in enumerate(ARMS):
        vals = [sub.loc[h, c] for h in H]
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col)
        for xi_, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi_, v * 1.01, f"{v:.3f}", ha="center", va="bottom", fontsize=5.6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in H])
    ax.set_ylim(0, max([sub.loc[h, c] for h in H for c, _, _ in ARMS]) * 1.28)
    ax.set_title(pol)
    ax.set_ylabel("RMSE (lebih kecil lebih baik)" if pol == POL[0] else "")
axes[0].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
fig.suptitle("Metode baru (tanpa SARIMA, tanpa jaringan saraf) vs metode paper — angka di atas batang",
             y=1.0, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_ens_rmse.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------- fig 7: selisih + penanda nyata
fig, ax = plt.subplots(figsize=(7.6, 4.2))
labels, vals, cols, marks = [], [], [], []
for pol in POL:
    for h in H:
        r = dmp.loc[(pol, h)]
        d = float(r.rmse_ratio) * 100.0
        p = float(r.dm_p)
        labels.append(f"{pol}  h={h}")
        vals.append(d)
        cols.append(GREEN if d < 0 else RED)
        marks.append("**" if p < 0.01 else ("*" if p < 0.05 else " (tak nyata)" if p >= 0.05 else ""))
y = np.arange(len(vals))[::-1]
ax.barh(y, vals, color=cols, height=0.62)
ax.axvline(0, color="k", lw=0.9)
for yy, v, s in zip(y, vals, marks):
    ax.text(v + (0.5 if v >= 0 else -0.5), yy, f"{v:+.1f}%{s}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("selisih RMSE ensemble LGBM+Theta vs metode lengkap paper (%)  —  negatif = metode baru lebih baik")
ax.set_title("Menang 7 dari 9 sel (5 nyata), kekalahan tunggal: NO2 jarak-1/7 — tidak nyata (* p<0,05, ** p<0,01)")
lim = max(abs(min(vals)), abs(max(vals)))
ax.set_xlim(-lim * 1.75, lim * 1.75)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig7_ens_delta.png"), bbox_inches="tight")
plt.close(fig)

# --------------------------------------------- fig 8: contoh ramalan PM25 h=30
def slim(df, prefixes):
    keep = ["feature", "fold", "origin_date"] + [f"{p}_h{h}" for p in prefixes for h in H]
    out = df[keep].copy()
    out["origin_date"] = out["origin_date"].astype(str)
    return out

b20 = pd.read_csv(os.path.join(R, "20_bakeoff3_predictions.csv"))
b20["fold"] = b20["fold"].astype(int)
o19 = slim(pd.read_csv(os.path.join(R, "19_bakeoff2_predictions.csv")), ["theta"])
o19["fold"] = o19["fold"].astype(int) * 120
o17 = slim(pd.read_csv(os.path.join(R, "17_lgbm_direct_predictions.csv")), ["lgbmdir"])
pap = slim(pd.read_csv(os.path.join(R, "13_bilstm_all_folds_predictions.csv")), ["hybrid", "sarima"])
j = (b20.merge(o19, on=["feature", "fold", "origin_date"])
        .merge(o17, on=["feature", "fold", "origin_date"])
        .merge(pap, on=["feature", "fold", "origin_date"]))
j["ens"] = (j.lgbmdir_h30 + j.theta_h30) / 2.0
j = j[(j.feature == "PM25") & (j.fold == 0)].sort_values("origin_date").dropna(subset=["ens", "hybrid_h30", "actual_h30"])
fig, ax = plt.subplots(figsize=(11.5, 3.9))
xi = np.arange(len(j))
ax.plot(xi, j.actual_h30, "o-", color="k", lw=1.5, ms=3.5, label="kenyataan")
ax.plot(xi, j.sarima_h30, "s--", color=GRAY, lw=1.3, ms=2.8, label="paper SARIMA (s=12)")
ax.plot(xi, j.hybrid_h30, "^-", color=ORANGE, lw=1.0, ms=3, alpha=0.95, label="paper SARIMA+BiLSTM")
ax.plot(xi, j.ens, "D-", color=NAVY, lw=1.6, ms=3.2, label="ensemble LGBM+Theta (metode baru)")
ax.set_xticks(xi[::4])
ax.set_xticklabels([s[:7] for s in j.origin_date][::4], fontsize=7.5)
ax.set_ylabel("PM2.5 (skala Box-Cox)")
ax.set_title("PM2.5 jarak 30 hari, lipatan pertama — ramalan vs kenyataan (sel dengan keunggulan besar: −20%)")
ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig8_ens_timeseries.png"), bbox_inches="tight")
plt.close(fig)

print("gambar tersimpan di", FIG)
for f in sorted(os.listdir(FIG)):
    if f.startswith(("fig6", "fig7", "fig8")):
        print(" ", f, f"{os.path.getsize(os.path.join(FIG, f)):,} bytes")

print("\nSelisih ens_lgbm_theta vs paper_hybrid per sel (%%, dari 20_bakeoff3_dm):")
print(dmp[["rmse_ratio", "dm_p", "n"]].round(4).to_string())
