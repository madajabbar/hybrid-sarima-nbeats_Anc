# -*- coding: utf-8 -*-
"""Menghasilkan gambar + angka tabel bergaya paper rujukan (Necula 2025).
Semua dihitung dari repo (raw, prediksi beku). Output: 04_Naskah/fig_m/*.png + tables.json
Jalankan dengan venv (pandas, scipy, statsmodels, matplotlib).
"""
import json
import os
from scipy.stats import boxcox, jarque_bera
from scipy.special import inv_boxcox
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
OUTD = os.path.join(REPO, "04_Naskah", "fig_m")
os.makedirs(OUTD, exist_ok=True)
POL = ["PM10", "PM25", "NO2"]
HZS = [1, 7, 30]
meta = {}

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 300,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

# ---------- data mentah + Box-Cox (rumus persis script 06) ----------
raw = pd.read_csv(os.path.join(REPO, "data", "transformed_data.csv"))
raw.columns = [c.strip("\ufeff") for c in raw.columns]
raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%y")
raw = raw.sort_values("Date").set_index("Date")
shift, lam, bc = {}, {}, {}
for c in POL:
    shift[c] = abs(raw[c].min()) + 1
    bc[c], lam[c] = boxcox(raw[c] + shift[c])
bcs = pd.DataFrame(bc, index=raw.index)

# ---------- T1/T2: deskriptif ----------
def likely_extremes(s):
    thr = s.quantile(0.90)
    return s[s >= thr], thr

desc = {}
for c in POL:
    s = raw[c]
    desc[c] = {"mean": float(s.mean()), "min": float(s.min()), "max": float(s.max()),
               "std": float(s.std()), "skew": float(s.skew()), "kurt": float(s.kurt()),
               "median": float(s.median()), "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75))}
meta["desc"] = desc

# ---------- T3: ADF & KPSS ----------
tests = {}
for c in POL:
    x = bc[c]
    ad = adfuller(x, autolag="AIC")
    kp = kpss(x, regression="c", nlags="auto")
    tests[c] = {"adf_stat": ad[0], "adf_p": ad[1], "kpss_stat": kp[0], "kpss_p": kp[1]}
meta["stationarity"] = tests

# ---------- FIG1: dataset time series (3 panel) ----------
fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=True)
for ax, c in zip(axes, POL):
    ax.plot(raw.index, raw[c], lw=0.55, color="#1f4e79")
    ax.set_ylabel(f"{c} (µg/m³)")
axes[0].set_title("Daily mean concentrations, 2019–2023")
fig.tight_layout()
fig.savefig(os.path.join(OUTD, "fig1_dataset.png"))
plt.close(fig)

# ---------- FIG2: decomposition (a,b,c per polutan; 4 komponen) ----------
fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.8), sharex="col")
for j, c in enumerate(POL):
    dec = seasonal_decompose(raw[c], period=365, model="additive", extrapolate_trend="freq")
    for i, (comp, col) in enumerate([(dec.observed, "#1f4e79"), (dec.trend, "#2e8bcd"),
                                     (dec.seasonal, "#e5813c"), (dec.resid, "#777777")]):
        axes[i, j].plot(raw.index, comp, lw=0.5, color=col)
        if j == 0:
            axes[i, j].set_ylabel(["Observed", "Trend", "Seasonal", "Residual"][i], fontsize=8)
        if i == 0:
            axes[i, j].set_title(c, fontsize=10)
        for lbl in axes[i, j].get_xticklabels():
            lbl.set_fontsize(7)
fig.suptitle("Seasonal–trend decomposition (period = 365 days)", fontsize=10, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUTD, "fig2_decompose.png"))
plt.close(fig)

# ---------- FIG3: Box-Cox transformed (a,b,c) ----------
fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
for ax, c in zip(axes, POL):
    ax.plot(raw.index, bc[c], lw=0.55, color="#1f4e79")
    ax.set_ylabel(f"{c}\nBox-Cox")
fig.suptitle("After Box-Cox transformation", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(OUTD, "fig3_boxcox.png"))
plt.close(fig)

# ---------- FIG4: ACF & PACF ----------
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
NL = 40
for j, c in enumerate(POL):
    a_ = acf(bc[c], nlags=NL)
    p_ = pacf(bc[c], nlags=NL)
    conf = 1.96 / np.sqrt(len(bc[c]))
    axes[0, j].bar(range(NL + 1), a_, width=0.55, color="#2e8bcd")
    axes[1, j].bar(range(NL + 1), p_, width=0.55, color="#e5813c")
    for i in range(2):
        axes[i, j].axhline(conf, ls="--", lw=0.8, color="#444444")
        axes[i, j].axhline(-conf, ls="--", lw=0.8, color="#444444")
        axes[i, j].set_ylim(-0.2, 1.05)
    axes[0, j].set_title(c, fontsize=10)
    if j == 0:
        axes[0, j].set_ylabel("ACF", fontsize=9)
        axes[1, j].set_ylabel("PACF", fontsize=9)
fig.suptitle("ACF and PACF of the transformed series", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUTD, "fig4_acfpacf.png"))
plt.close(fig)

# ---------- prediksi per sel (hybrid paper, SARIMA, theta, ens) ----------
def load_pred(path, prefix):
    df = pd.read_csv(path)
    df["origin_date"] = df["origin_date"].astype(str)
    return df

p13 = load_pred(os.path.join(RES, "13_bilstm_all_folds_predictions.csv"), "hybrid")
p12 = load_pred(os.path.join(RES, "12_final_predictions.csv"), "sarima")
p19 = load_pred(os.path.join(RES, "19_bakeoff2_predictions.csv"), "theta")
p19["fold"] = p19["fold"].astype(int) * 120  # 19 file memakai indeks 0-4; samakan dengan skala offset
p17 = load_pred(os.path.join(RES, "17_lgbm_direct_predictions.csv"), "lgbmdir")

KEYS = ["feature", "fold", "origin_date"]
def wide(df, pfx):
    keep = KEYS + [f"actual_h{h}" for h in HZS] + [f"{pfx}_h{h}" for h in HZS]
    return df[keep].copy()

base = wide(p12, "sarima").merge(wide(p13, "hybrid"), on=KEYS, suffixes=("", "_y")).merge(
    wide(p19, "theta"), on=KEYS, suffixes=("", "_y")).merge(wide(p17, "lgbmdir"), on=KEYS, suffixes=("", "_y"))
base = base.loc[:, ~base.columns.str.endswith("_y")]
for h in HZS:
    base[f"ens_h{h}"] = (base[f"theta_h{h}"] + base[f"lgbmdir_h{h}"]) / 2
base["actual"] = base["actual_h7"]  # bcs actual sama utk semua h
base["raw_actual"] = base.apply(lambda r: inv_boxcox(r["actual_h7"], lam[r["feature"]]) - shift[r["feature"]], axis=1)

MODELS = [("hybrid", "Full hybrid (SARIMA+BiLSTM+Fourier+Attention)"),
          ("sarima", "SARIMA (weekly)"),
          ("theta", "Theta"),
          ("ens", "LightGBM + Theta ensemble")]

# regenerasi lambda-shift pada prediksi agar balik ke skala mentah konsisten
shift_bc = {}
for c in POL:
    shift_bc[c] = float(abs(raw[c].min()) + 1)

def to_raw(vals, c):
    return inv_boxcox(np.asarray(vals, float), lam[c]) - shift_bc[c]

# ---------- metrik RMSE/MAE (BC) dan MAPE (mentah) ----------
perf = []
for f in POL:
    d = base[base.feature == f]
    for h in HZS:
        a_bc = d[f"actual_h{h}"].values
        a_raw = to_raw(a_bc, f)
        for arm, lbl in MODELS:
            p_bc = d[f"{arm}_h{h}"].values
            ok = np.isfinite(a_bc) & np.isfinite(p_bc)
            e_bc = a_bc[ok] - p_bc[ok]
            p_raw = to_raw(p_bc[ok], f)
            mape = float(np.mean(np.abs((a_raw[ok] - p_raw) / a_raw[ok])) * 100)
            perf.append({"target": f, "horizon": h, "model": lbl, "key": arm,
                         "rmse": float(np.sqrt(np.mean(e_bc ** 2))),
                         "mae": float(np.mean(np.abs(e_bc))),
                         "mape": mape, "n": int(ok.sum())})
meta["perf"] = perf

# ---------- T9: diagnostik residual (Ljung-Box, JB) ----------
diag = []
for f in POL:
    d = base[base.feature == f].sort_values(["fold", "origin_date"])
    for h in HZS:
        for arm, lbl in MODELS:
            e = d[f"actual_h{h}"] - d[f"{arm}_h{h}"]
            e = e[np.isfinite(e)]
            lb = acorr_ljungbox(e, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
            jb = jarque_bera(e)
            diag.append({"target": f, "horizon": h, "model": lbl, "key": arm,
                         "resid_mean": float(e.mean()), "resid_std": float(e.std()),
                         "lb_p": float(lb), "jb_p": float(jb[1]), "n": len(e)})
meta["diag"] = diag

# ---------- T10: penanganan nilai ekstrem (MAPE pada dekil atas) ----------
ext = []
for f in POL:
    thr = raw[f].quantile(0.90)
    d = base[base.feature == f].copy()
    d["raw"] = to_raw(d["actual_h7"].values, f)
    extrows = d[d["raw"] >= thr]
    for h in HZS:
        for arm, lbl in MODELS:
            a_raw = to_raw(extrows[f"actual_h{h}"].values, f)
            p_raw = to_raw(extrows[f"{arm}_h{h}"].values, f)
            m = np.isfinite(a_raw) & np.isfinite(p_raw)
            ext.append({"target": f, "horizon": h, "model": lbl, "key": arm,
                        "mape": float(np.mean(np.abs((a_raw[m] - p_raw[m]) / a_raw[m])) * 100),
                        "n": int(m.sum()), "thr": float(thr), "max": float(raw[f].max())})
meta["extreme"] = ext

# ---------- FIG5: RMSE per sel (bar, hybrid vs ens) ----------
fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4))
for ax, f in zip(axes, POL):
    xs = np.arange(3)
    rh = [p for p in perf if p["target"] == f and p["key"] == "hybrid"]
    re_ = [p for p in perf if p["target"] == f and p["key"] == "ens"]
    ax.bar(xs - 0.17, [r["rmse"] for r in rh], 0.34, color="#c0574f", label="Full hybrid")
    ax.bar(xs + 0.17, [r["rmse"] for r in re_], 0.34, color="#2e8bcd", label="Ensemble (ours)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["1-day", "7-day", "30-day"])
    ax.set_ylabel(f"{f}\nRMSE (Box-Cox)")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("RMSE by pollutant and horizon", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUTD, "fig5_rmse.png"))
plt.close(fig)

# ---------- Lampiran A: A1–A9 prediksi aktual vs hybrid vs ensemble ----------
aindex = 0
for f in POL:
    d = base[base.feature == f].sort_values(["fold", "origin_date"])
    for h in HZS:
        aindex += 1
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        x = np.arange(len(d))
        ax.plot(x, d[f"actual_h{h}"], lw=0.8, color="#333333", label="Actual")
        ax.plot(x, d[f"hybrid_h{h}"], lw=0.8, color="#e5813c", label="Full hybrid")
        ax.plot(x, d[f"ens_h{h}"], lw=0.8, color="#2e8bcd", label="Ensemble (ours)")
        ax.set_xlabel("Forecast origin (chronological)")
        ax.set_ylabel(f"{f} (Box-Cox)")
        leg = ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTD, f"figA{aindex}_{f}_h{h}.png"))
        plt.close(fig)
meta["lambda"] = lam
meta["n"] = {f: len(raw) for f in POL}
with open(os.path.join(REPO, "04_Naskah", "tables.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
print("figures:", sorted(os.listdir(OUTD)))
print("lambda:", {k: round(v, 4) for k, v in lam.items()})
