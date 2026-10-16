# Fase 18 — Bake-off model tambahan (ranah persis resep beku fase 12/16/17)
# Lengan baru: naive7 (naif musiman), arima_nonmusiman (ablasion musiman),
# ets (Holt-Winters), var (multivariat 3 polutan). Metrik skala Box-Cox, idem arm lain.
import warnings, math, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import boxcox, norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR

REPO = "/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/air-pollution-hybrid-nbeats"
RES = os.path.join(REPO, "results")
FEATURES = ["NO2", "PM10", "PM25"]
HORIZONS = [1, 7, 30]
ORDER = (2, 1, 1)          # idem notebook 04
TEST_DAYS, STEP, MAX_H = 720, 30, 30
NFOLDS = 5                  # geser 120 hari
OUTP = os.path.join(RES, "18_bakeoff_predictions.csv")

# ---------- data & Box-Cox: kalimat sama persis dengan notebook 04 ----------
raw = pd.read_csv(os.path.join(REPO, "data", "transformed_data.csv"))
raw.columns = [c.strip("\ufeff") for c in raw.columns]
raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%y")
raw = raw.sort_values("Date").set_index("Date")
series = pd.DataFrame(index=raw.index)
for c in raw.columns:
    series[c], _ = boxcox(raw[c] + abs(raw[c].min()) + 1)

def rmse(a, p): return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))

def dm_p(e1, e2, h):
    """DM klasik, loss kuadrat, HAC lag=h (Bartlett). Return (p dua-sisi, n berpasangan)."""
    ok = np.isfinite(e1) & np.isfinite(e2)
    d = np.asarray(e1, float)[ok] ** 2 - np.asarray(e2, float)[ok] ** 2
    n = len(d)
    if n < 10:
        return np.nan, n
    dbar = d.mean()
    din = d - dbar
    L = max(h, 1)
    g0 = np.dot(din, din) / n
    v = g0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        v += 2.0 * w * (np.dot(din[l:], din[:-l]) / n)
    se = math.sqrt(max(v, 1e-300) / n)
    if se == 0:
        return np.nan, n
    return float(2 * (1 - norm.cdf(abs(dbar / se)))), n

# ---------- lengan baru (dihitung sekali; jika CSV sudah ada, pakai kembali) ----------
if os.path.exists(OUTP):
    newp = pd.read_csv(OUTP)
    print("lanjut dari ramalan tersimpan:", newp.shape)
else:
    rows = []
    for k in range(NFOLDS):
        off = k * 120
        end = len(series) - off
        start = end - TEST_DAYS
        test = series.iloc[start:end]
        for f in FEATURES:
            train = series[f].iloc[:start]
            test_f = test[f]
            nons = SARIMAX(train, order=ORDER, seasonal_order=(0, 0, 0, 0)).fit(disp=False)
            for origin in range(0, len(test) - min(HORIZONS), STEP):
                hist = pd.concat([train, test_f.iloc[:origin]])
                hist3 = series.iloc[: start + origin]
                rec = {"feature": f, "fold": k, "origin_date": str(test.index[origin].date())}
                fc_naive = np.resize(hist.iloc[-7:].values, MAX_H)      # naif musiman
                try:
                    fc_ets = np.asarray(ExponentialSmoothing(
                        hist, trend="add", damped_trend=True, seasonal="add",
                        seasonal_periods=7, initialization_method="estimated"
                    ).fit(optimized=True).forecast(MAX_H), float)
                except Exception:
                    fc_ets = np.full(MAX_H, np.nan)
                try:
                    fit_v = VAR(hist3).fit(maxlags=14, ic="aic")
                    fc_var = fit_v.forecast(hist3.values[-fit_v.k_ar:], MAX_H)[:, FEATURES.index(f)]
                except Exception:
                    fc_var = np.full(MAX_H, np.nan)
                try:
                    fc_ar = np.asarray(nons.apply(hist).forecast(steps=MAX_H), float)
                except Exception:
                    fc_ar = np.full(MAX_H, np.nan)
                for h in HORIZONS:
                    io = origin + h - 1
                    if io >= len(test_f):
                        continue
                    rec[f"actual_h{h}"] = float(test_f.iloc[io])
                    rec[f"naive7_h{h}"] = float(fc_naive[h - 1])
                    rec[f"ets_h{h}"] = float(fc_ets[h - 1])
                    rec[f"var_h{h}"] = float(fc_var[h - 1])
                    rec[f"arima_nn_h{h}"] = float(fc_ar[h - 1])
                rows.append(rec)
        print(f"lipatan {k+1}/{NFOLDS} selesai", flush=True)
    newp = pd.DataFrame(rows)
    newp.to_csv(OUTP, index=False)
    print("ramalan baru tersimpan:", newp.shape)

# ---------- gabungkan dengan arm lama (fold file lama = offset geser 0/120/.../480) ----------
KEYS = ["feature", "fold", "origin_date"]
def slim(df, prefixes):
    keep = KEYS + [f"{p}_h{h}" for p in prefixes for h in HORIZONS]
    out = df[keep].copy()
    out["origin_date"] = out["origin_date"].astype(str)
    return out
newp = newp.copy()
newp["fold"] = newp["fold"].astype(int) * 120
newp["origin_date"] = newp["origin_date"].astype(str)
o12 = slim(pd.read_csv(os.path.join(RES, "12_final_predictions.csv")), ["sarima", "final", "alpha"])
o12raw = pd.read_csv(os.path.join(RES, "12_final_predictions.csv"))
chk = newp.merge(o12raw[KEYS + [f"actual_h{h}" for h in HORIZONS]], on=KEYS)
for h in HORIZONS:
    diff = float(np.abs(chk[f"actual_h{h}_x"] - chk[f"actual_h{h}_y"]).max())
    assert diff < 1e-6, f"actual beda dgn file lama: {diff}"
print("cek: actual baru == actual lama (maks selisih < 1e-6) OK")
o16 = slim(pd.read_csv(os.path.join(RES, "16_lgbm_gate_predictions.csv")), ["ridge", "lgbm"])
o17 = slim(pd.read_csv(os.path.join(RES, "17_lgbm_direct_predictions.csv")), ["lgbmdir"])
merged = newp.merge(o12, on=KEYS, how="inner").merge(o16, on=KEYS, how="inner").merge(o17, on=KEYS, how="inner")
print("gabungan baris:", len(merged), "| per fitur:", merged.groupby("feature").size().to_dict())

ARMS = {"naive7": "naive7", "arima_nonmusiman": "arima_nn", "ets": "ets", "var": "var",
        "SARIMA": "sarima", "ridge_gate": "final", "lgbm_gate": "lgbm", "lgbm_langsung": "lgbmdir"}

met, dmrows = [], []
for f in FEATURES:
    for h in HORIZONS:
        d = merged[merged.feature == f].copy()
        errs = {}
        for arm, pre in ARMS.items():
            col = d[f"actual_h{h}"] - d[f"{pre}_h{h}"]
            ok = np.isfinite(d[f"actual_h{h}"]) & np.isfinite(d[f"{pre}_h{h}"])
            errs[arm] = np.where(ok, col, np.nan)
            met.append({"target": f, "horizon": h, "model": arm,
                        "rmse": float(np.nanmean(col[ok] ** 2) ** 0.5),
                        "mae": float(np.nanmean(np.abs(col[ok]))),
                        "n": int(ok.sum())})
        sar = errs["SARIMA"]
        for base_name, base_err in (("SARIMA", sar), ("ridge_gate", errs["ridge_gate"])):
            for arm in ARMS:
                p, n = dm_p(errs[arm], base_err, h)
                ok = np.isfinite(errs[arm]) & np.isfinite(base_err)
                rr = float(np.sqrt(np.nanmean(errs[arm][ok] ** 2)) / np.sqrt(np.nanmean(base_err[ok] ** 2)) - 1) if ok.sum() else np.nan
                dmrows.append({"target": f, "horizon": h, "vs": arm, "base": base_name,
                               "rmse_ratio": rr, "dm_p": p, "n": n})
        print(f"{f} h={h} selesai", flush=True)

metdf = pd.DataFrame(met)
dm = pd.DataFrame(dmrows)
metdf.to_csv(os.path.join(RES, "18_bakeoff_metrics.csv"), index=False)
dm.to_csv(os.path.join(RES, "18_bakeoff_dm.csv"), index=False)

print("\n== PAPAN PERINGKAT (RMSE skala Box-Cox, seluruh lipatan digabung) ==")
piv = metdf.pivot_table(index=["target", "horizon"], columns="model", values="rmse").round(4)
print(piv.to_string())
print("\n== JUARA PER SEL ==")
print(piv.idxmin(axis=1).to_string())
print("\n== DM vs SARIMA (adam_p<0.05 = beda nyata; rmse_ratio<0 = lebih baik) ==")
print(dm[dm.base == "SARIMA"].round(4).to_string(index=False))
print("\n== DM vs ridge_gate ==")
print(dm[dm.base == "ridge_gate"].round(4).to_string(index=False))

# ---------- skor rata-rata ternormalisasi (vs SARIMA, 9 sel) ----------
m2 = metdf.copy()
m2["ratio_sarima"] = m2.apply(lambda r: r["rmse"] / metdf[(metdf.target == r.target) & (metdf.horizon == r.horizon) & (metdf.model == "SARIMA")]["rmse"].iloc[0] - 1, axis=1)
sc = m2.groupby("model")["ratio_sarima"].mean().sort_values().round(4)
print("\n== SKOR GABUNGAN 9 SEL (rata-rata %% vs SARIMA; negatif = lebih baik) ==")
print((sc * 100).to_string())

# ---------- ROUTER: pilih juara per sel memakai lipatan TER-UMUR (offset 480), nilai di sisa ----------
picks, router_rows = {}, []
for f in FEATURES:
    dval = merged[(merged.feature == f) & (merged.fold == 480)]
    for h in HORIZONS:
        best, best_rm = None, np.inf
        for arm, pre in ARMS.items():
            ok = np.isfinite(dval[f"actual_h{h}"]) & np.isfinite(dval[f"{pre}_h{h}"])
            r = float(np.sqrt(np.mean((dval.loc[ok, f"actual_h{h}"] - dval.loc[ok, f"{pre}_h{h}"]) ** 2)))
            if r < best_rm:
                best, best_rm = arm, r
        picks[(f, h)] = best
rows = []
for f in FEATURES:
    d = merged[merged.feature == f]
    for h in HORIZONS:
        arm = picks[(f, h)]
        e = d[f"actual_h{h}"] - d[f"{ARMS[arm]}_h{h}"]
        ok = np.isfinite(e)
        rows.append({"target": f, "horizon": h, "router_arm": arm,
                     "router_rmse": float(np.sqrt(np.mean(e[ok] ** 2)))})
r = pd.DataFrame(rows)
print("\n== ROUTER (pilihan dari 120-hari-pertama, nilai di semua lipatan) ==")
print(r.to_string(index=False))
print("\npilihan router:", {f"{k[0]}-h{k[1]}": v for k, v in picks.items()})
# rasio bebas skala: RMSE arm / RMSE SARIMA di sel yang sama
sar_rm = {(t, h): metdf[(metdf.target == t) & (metdf.horizon == h) & (metdf.model == "SARIMA")]["rmse"].iloc[0]
          for t in FEATURES for h in HORIZONS}
rr_router = float((r.apply(lambda x: x.router_rmse / sar_rm[(x.target, x.horizon)], axis=1)).mean() - 1)
print(f"skor router (rata-rata % vs SARIMA): {rr_router * 100:+.2f}%")
print("skor arm lain (idem):", (sc * 100).round(2).to_dict())
pd.DataFrame([{"target": k[0], "horizon": k[1], "pilih": v} for k, v in picks.items()]).to_csv(
    os.path.join(RES, "18_router_picks.csv"), index=False)
r.to_csv(os.path.join(RES, "18_router_metrics.csv"), index=False)
