# Fase 19 — perluasan bake-off: Theta, RF langsung, MLP langsung, ensemble(SARIMA+LGBM)
# protokol identik fase 18 (lipatan geser 120, 24 origin, h=1/7/30, skala Box-Cox).
# Lengan fase 18 dipakai ulang dari results/18_bakeoff_predictions.csv (tanpa hitung ulang).
import warnings, math, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import boxcox, norm
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

REPO = "/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/air-pollution-hybrid-nbeats"
RES = os.path.join(REPO, "results")
FEATURES, HORIZONS = ["NO2", "PM10", "PM25"], [1, 7, 30]
TEST_DAYS, STEP, MAX_H, NFOLDS = 720, 30, 30, 5
OUTP = os.path.join(RES, "19_bakeoff2_predictions.csv")

raw = pd.read_csv(os.path.join(REPO, "data", "transformed_data.csv"))
raw.columns = [c.strip("\ufeff") for c in raw.columns]
raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%y")
raw = raw.sort_values("Date").set_index("Date")
series = pd.DataFrame(index=raw.index)
for c in raw.columns:
    series[c], _ = boxcox(raw[c] + abs(raw[c].min()) + 1)
series.index = pd.DatetimeIndex(series.index, freq="D")

def rmse(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((np.asarray(a)[ok] - np.asarray(b)[ok]) ** 2)))

def dm_p(e1, e2, h):
    ok = np.isfinite(e1) & np.isfinite(e2)
    d = np.asarray(e1, float)[ok] ** 2 - np.asarray(e2, float)[ok] ** 2
    n = len(d)
    if n < 10:
        return np.nan, n
    din = d - d.mean()
    g0 = np.dot(din, din) / n
    v = g0
    for l in range(1, max(h, 1) + 1):
        v += 2.0 * (1 - l / (max(h, 1) + 1)) * np.dot(din[l:], din[:-l]) / n
    se = math.sqrt(max(v, 1e-300) / n)
    return (np.nan, n) if se == 0 else (float(2 * (1 - norm.cdf(abs(d.mean() / se)))), n)

# ---------- fitur langsung (direct): lag1-7 + roll3/7/30 + hari-dalam-minggu ----------
def feat_matrix(f):
    y = series[f]
    X = pd.DataFrame(index=series.index)
    for i in range(1, 8):
        X[f"lag{i}"] = y.shift(i)
    for w in (3, 7, 30):
        X[f"roll{w}"] = y.shift(1).rolling(w).mean()
    X["dow"] = series.index.dayofweek
    return X, y

FEATS = {f: feat_matrix(f) for f in FEATURES}

# ---------- lengan baru (cache) ----------
if os.path.exists(OUTP):
    b2 = pd.read_csv(OUTP)
    print("lanjut dari cache fase 19:", b2.shape)
else:
    rows, rf_seed = [], 13
    for k in range(NFOLDS):
        off = k * 120
        end = len(series) - off
        start = end - TEST_DAYS
        test = series.iloc[start:end]
        for f in FEATURES:
            X, y = FEATS[f]
            for origin in range(0, len(test) - min(HORIZONS), STEP):
                t_ab = start + origin                       # batas histori (absolut)
                rec = {"feature": f, "fold": k, "origin_date": str(test.index[origin].date())}
                hist = series[f].iloc[:t_ab]
                try:
                    fc_th = np.asarray(ThetaModel(hist, period=7, deseasonalize=True,
                                                  method="additive").fit().forecast(MAX_H), float)
                except Exception:
                    fc_th = np.full(MAX_H, np.nan)
                fcs = {"rf": [], "mlp": []}
                bnd = series.index[t_ab - 1]                 # hari terakhir yg diketahui (t0-1)
                Xtr = X.loc[X.index <= bnd]
                for h in HORIZONS:
                    ytr = y.shift(-h)                        # direct: y(t+h)
                    ytr_x = ytr.loc[Xtr.index]
                    # label t+h <= t0-1 (identik skrip 17: "label murni masa lalu, tanpa bocor")
                    oktr = (Xtr.notna().all(axis=1)
                            & (Xtr.index <= bnd - pd.Timedelta(days=h)) & ytr_x.notna())
                    Xa, ya = Xtr[oktr], ytr_x[oktr]
                    xrow_feat_end = test.index[origin] - pd.Timedelta(days=1)  # issued at t → target t+h
                    xrow = X.loc[[xrow_feat_end]]
                    if xrow.isna().values.any():
                        for mn in fcs:
                            fcs[mn].append(np.nan)
                        continue
                    mrf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                n_jobs=-1, random_state=rf_seed)
                    mrf.fit(Xa.values, ya.values)
                    fcs["rf"].append(float(mrf.predict(xrow.values)[0]))
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler
                    mmlp = make_pipeline(StandardScaler(),
                                         MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=150,
                                                      random_state=rf_seed))
                    mmlp.fit(Xa.values, ya.values)
                    fcs["mlp"].append(float(mmlp.predict(xrow.values)[0]))
                for h in HORIZONS:
                    io = origin + h - 1
                    if io >= len(test):
                        continue
                    rec[f"actual_h{h}"] = float(test[f].iloc[io])
                    rec[f"theta_h{h}"] = float(fc_th[h - 1])
                    rec[f"rf_h{h}"] = float(fcs["rf"][HORIZONS.index(h)])
                    rec[f"mlp_h{h}"] = float(fcs["mlp"][HORIZONS.index(h)])
                rows.append(rec)
        print(f"lipatan {k+1}/{NFOLDS} selesai", flush=True)
    b2 = pd.DataFrame(rows)
    b2.to_csv(OUTP, index=False)
    print("ramalan fase 19 tersimpan:", b2.shape)

# ---------- gabungkan semua lengan ----------
KEYS = ["feature", "fold", "origin_date"]
def slim(df, prefixes):
    keep = KEYS + [f"{p}_h{h}" for p in prefixes for h in HORIZONS]
    out = df[keep].copy()
    out["origin_date"] = out["origin_date"].astype(str)
    return out
b2 = b2.copy()
b2["fold"] = b2["fold"].astype(int) * 120
b2["origin_date"] = b2["origin_date"].astype(str)
o12raw = pd.read_csv(os.path.join(RES, "12_final_predictions.csv"))
chk = b2.merge(o12raw[KEYS + [f"actual_h{h}" for h in HORIZONS]], on=KEYS)
for h in HORIZONS:
    assert float(np.abs(chk[f"actual_h{h}_x"] - chk[f"actual_h{h}_y"]).max()) < 1e-6
o12 = slim(o12raw, ["sarima", "final", "alpha"])
o16 = slim(pd.read_csv(os.path.join(RES, "16_lgbm_gate_predictions.csv")), ["ridge", "lgbm"])
o17 = slim(pd.read_csv(os.path.join(RES, "17_lgbm_direct_predictions.csv")), ["lgbmdir"])
merged = b2.merge(o12, on=KEYS).merge(o16, on=KEYS).merge(o17, on=KEYS)
for h in HORIZONS:                                   # ensemble rata-rata SARIMA+LGBM telanjang
    merged[f"ens_h{h}"] = (merged[f"sarima_h{h}"] + merged[f"lgbmdir_h{h}"]) / 2
o18 = slim(pd.read_csv(os.path.join(RES, "18_bakeoff_predictions.csv")),
           ["naive7", "ets", "var", "arima_nn"])
o18["fold"] = o18["fold"].astype(int) * 120            # 18 file: fold indeks 0..4
merged = merged.merge(o18, on=KEYS, how="inner")
print("gabungan:", len(merged), "| per fitur:", merged.groupby("feature").size().to_dict())

ARMS = {"theta": "theta", "rf_langsung": "rf", "mlp_langsung": "mlp", "ensemble": "ens",
        "naive7": "naive7", "arima_nonmusiman": "arima_nn", "ets": "ets", "var": "var",
        "SARIMA": "sarima", "ridge_gate": "final", "lgbm_gate": "lgbm", "lgbm_langsung": "lgbmdir"}
met, dmrows = [], []
for f in FEATURES:
    for h in HORIZONS:
        d = merged[merged.feature == f]
        errs = {}
        for arm, pre in ARMS.items():
            ok = np.isfinite(d[f"actual_h{h}"]) & np.isfinite(d[f"{pre}_h{h}"])
            e = (d[f"actual_h{h}"] - d[f"{pre}_h{h}"])[ok].values
            errs[arm] = np.where(np.isfinite(d[f"{pre}_h{h}"]), d[f"actual_h{h}"] - d[f"{pre}_h{h}"], np.nan)
            met.append({"target": f, "horizon": h, "model": arm, "rmse": float(np.sqrt(np.mean(e ** 2))),
                        "mae": float(np.mean(np.abs(e))), "n": int(ok.sum())})
        for base_name in ("SARIMA", "ridge_gate"):
            for arm in ARMS:
                p, n = dm_p(errs[arm], errs[base_name], h)
                ok = np.isfinite(errs[arm]) & np.isfinite(errs[base_name])
                rr = float(np.sqrt(np.mean(errs[arm][ok] ** 2)) / np.sqrt(np.mean(errs[base_name][ok] ** 2)) - 1)
                dmrows.append({"target": f, "horizon": h, "vs": arm, "base": base_name,
                               "rmse_ratio": rr, "dm_p": p, "n": n})
        print(f"{f} h={h} selesai", flush=True)
metdf = pd.DataFrame(met)
dm = pd.DataFrame(dmrows)
metdf.to_csv(os.path.join(RES, "19_bakeoff2_metrics.csv"), index=False)
dm.to_csv(os.path.join(RES, "19_bakeoff2_dm.csv"), index=False)

pd.set_option("display.width", 250)
print("\n== PAPAN PERINGKAT 12 ARM (RMSE Box-Cox) ==")
print(metdf.pivot_table(index=["target", "horizon"], columns="model", values="rmse").round(4).to_string())
m2 = metdf.copy()
m2["ratio_sarima"] = m2.apply(lambda r: r["rmse"] / metdf[(metdf.target == r.target) & (metdf.horizon == r.horizon) & (metdf.model == "SARIMA")]["rmse"].iloc[0] - 1, axis=1)
sc = (m2.groupby("model")["ratio_sarima"].mean() * 100).sort_values().round(2)
print("\n== SKOR GABUNGAN 9 SEL (% vs SARIMA; negatif = lebih baik) ==")
print(sc.to_string())
print("\n== DM vs SARIMA (baru saja) ==")
print(dm[(dm.base == "SARIMA") & dm.vs.isin(["theta", "rf_langsung", "mlp_langsung", "ensemble"])].round(4).to_string(index=False))

# ---------- router 12 arm (pilih dari lipatan ter-umur, nilai di semua) ----------
picks = {}
for f in FEATURES:
    dval = merged[(merged.feature == f) & (merged.fold == 480)]
    for h in HORIZONS:
        best, best_rm = None, np.inf
        for arm, pre in ARMS.items():
            ok = np.isfinite(dval[f"actual_h{h}"]) & np.isfinite(dval[f"{pre}_h{h}"])
            r = float(np.sqrt(np.mean((dval.loc[ok, f"actual_h{h}"] - dval.loc[ok, f"{pre}_h{h}"]) ** 2)))
            if r < best_rm:
                best, best_rm = arm, r
        picks[(f, h)] = (best, best_rm)
rrows = []
for f in FEATURES:
    d = merged[merged.feature == f]
    for h in HORIZONS:
        arm = picks[(f, h)][0]
        e = (d[f"actual_h{h}"] - d[f"{ARMS[arm]}_h{h}"]).dropna()
        rrows.append({"target": f, "horizon": h, "router_arm": arm, "router_rmse": float(np.sqrt(np.mean(e ** 2)))})
router = pd.DataFrame(rrows)
sar_rm = {(t, h): metdf[(metdf.target == t) & (metdf.horizon == h) & (metdf.model == "SARIMA")]["rmse"].iloc[0] for t in FEATURES for h in HORIZONS}
rr = float((router.apply(lambda x: x.router_rmse / sar_rm[(x.target, x.horizon)], axis=1)).mean() - 1)
print("\n== ROUTER 12 ARM ==")
print(router.to_string(index=False))
print(f"skor router: {rr * 100:+.2f}% | arm:", sc.to_dict())
router.to_csv(os.path.join(RES, "19_router_metrics.csv"), index=False)
