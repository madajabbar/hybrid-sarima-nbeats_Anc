# Fase 20 — perburuan metode TANPA-SARIMA (ganti penuh metode Necula 2025).
# Protokol identik fase 18/19 (5 lipatan geser 120, 24 origin, h=1/7/30, skala Box-Cox).
# Lengan baru (semua tanpa SARIMA):
#   langsung/fitur  : svr, knn, ridge_langsung, lgbm_mingguan (LGBM 17 + Fourier mingguan),
#                     fourier_ols (regresi kalender murni)
#   deret-waktu     : ucm (structural level+musiman mingguan+eksog tahunan), stl_ets
#   ensemble        : ens_lgbm_theta, ens_lgbm_ets, ens_lgbm_rf, ens_trio, ens_lgbmwk_theta
# Lengan lama dipakai ulang dari 12/16/17/18/19 tanpa hitung ulang. Router sengaja ditiadakan.
# Adu utama = lawan metode lengkap paper (sarima s12 + BiLSTM, dari file 13).
import warnings, math, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import boxcox, norm
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression
from lightgbm import LGBMRegressor

REPO = "/home/mada/Documents/Kuliah/Algorithms and Complexity/02_Kode/air-pollution-hybrid-nbeats"
RES = os.path.join(REPO, "results")
FEATURES, HORIZONS = ["NO2", "PM10", "PM25"], [1, 7, 30]
TEST_DAYS, STEP, MAX_H, NFOLDS = 720, 30, 30, 5
OUTP = os.path.join(RES, "20_bakeoff3_predictions.csv")
LGBM_KW = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
               random_state=42, n_jobs=1, verbosity=-1, deterministic=True)   # identik skrip 17

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

# ---------- fitur langsung (lag1-7 + roll + kalender tahunan + Fourier MINGGUAN) ----------
def feat_matrix(f):
    y = series[f]
    X = pd.DataFrame(index=series.index)
    for i in range(1, 8):
        X[f"lag{i}"] = y.shift(i)
    for w in (7, 30):
        X[f"roll{w}"] = y.shift(1).rolling(w).mean()
    X["std30"] = y.shift(1).rolling(30).std()
    X["lag365"] = y.shift(365)
    doy = series.index.dayofyear
    X["sin365"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos365"] = np.cos(2 * np.pi * doy / 365.25)
    dow = series.index.dayofweek
    X["dow"] = dow                                  # kategori utk LGBM
    X["sin7"] = np.sin(2 * np.pi * dow / 7)         # siklus mingguan eksplisit (kunci NO2)
    X["cos7"] = np.cos(2 * np.pi * dow / 7)
    X["sin7b"] = np.sin(4 * np.pi * dow / 7)        # harmonik ke-2 mingguan
    X["cos7b"] = np.cos(4 * np.pi * dow / 7)
    return X, y

FEATS = {f: feat_matrix(f) for f in FEATURES}

# fourier_ols: hanya kalender + tren, tanpa lag
TREND = np.arange(len(series), dtype=float)
FCAL = pd.DataFrame(index=series.index)
FCAL["t"] = TREND / 1000.0
for k in (1, 2):
    FCAL[f"sy{k}"] = np.sin(2 * np.pi * k * series.index.dayofyear / 365.25)
    FCAL[f"cy{k}"] = np.cos(2 * np.pi * k * series.index.dayofyear / 365.25)
FCAL["sw1"] = np.sin(2 * np.pi * series.index.dayofweek / 7)
FCAL["cw1"] = np.cos(2 * np.pi * series.index.dayofweek / 7)
FCAL["sw2"] = np.sin(4 * np.pi * series.index.dayofweek / 7)
FCAL["cw2"] = np.cos(4 * np.pi * series.index.dayofweek / 7)

if os.path.exists(OUTP):
    b3 = pd.read_csv(OUTP)
    print("lanjut dari cache fase 20:", b3.shape)
else:
    rows = []
    for k in range(NFOLDS):
        off = k * 120
        end = len(series) - off
        start = end - TEST_DAYS
        test = series.iloc[start:end]
        for f in FEATURES:
            X, y = FEATS[f]
            for origin in range(0, len(test) - min(HORIZONS), STEP):
                t_ab = start + origin
                rec = {"feature": f, "fold": off, "origin_date": str(test.index[origin].date())}
                hist = series[f].iloc[:t_ab]
                # deret-waktu: ucm (level+musiman mingguan+eksog tahunan), stl_ets
                exog = FCAL[["sy1", "cy1"]].iloc[:t_ab]
                exf = FCAL[["sy1", "cy1"]].iloc[t_ab:t_ab + MAX_H]
                try:
                    fc_ucm = np.asarray(UnobservedComponents(
                        hist, level="local level", seasonal=7, stochastic_seasonal=False,
                        exog=exog).fit(disp=False).forecast(MAX_H, exog=exf), float)
                except Exception:
                    fc_ucm = np.full(MAX_H, np.nan)
                try:
                    fc_stl = np.asarray(STLForecast(
                        hist, ExponentialSmoothing, period=7,
                        model_kwargs=dict(trend="add", damped_trend=True,
                                          initialization_method="estimated")
                    ).fit().forecast(MAX_H), float)
                except Exception:
                    fc_stl = np.full(MAX_H, np.nan)
                bnd = series.index[t_ab - 1]
                Xtr = X.loc[X.index <= bnd]
                Ftr = FCAL.loc[FCAL.index <= bnd]
                fcs = {m: [] for m in ("svr", "knn", "ridge", "lgbmwk", "fourier")}
                for h in HORIZONS:
                    ytr = y.shift(-h)
                    ytr_x = ytr.loc[Xtr.index]
                    oktr = (Xtr.notna().all(axis=1)
                            & (Xtr.index <= bnd - pd.Timedelta(days=h)) & ytr_x.notna())
                    xrow_feat_end = test.index[origin] - pd.Timedelta(days=1)
                    xrow = X.loc[[xrow_feat_end]]
                    if xrow.isna().values.any():
                        for m in fcs:
                            fcs[m].append(np.nan)
                        continue
                    Xa, ya = Xtr[oktr], ytr_x[oktr]
                    msvr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01, gamma="scale"))
                    mknn = make_pipeline(StandardScaler(),
                                         KNeighborsRegressor(n_neighbors=25, weights="distance"))
                    mrid = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    mlgb = LGBMRegressor(**LGBM_KW)
                    okf = Ftr.index.isin(Xa.index)         # fourier_ols: tanggal sama dgn oktr
                    for name, mod in (("svr", msvr), ("knn", mknn), ("ridge", mrid),
                                      ("lgbmwk", mlgb)):
                        mod.fit(Xa.values, ya.values)
                        fcs[name].append(float(mod.predict(xrow.values)[0]))
                    mf = LinearRegression().fit(Ftr[okf].values, ya.values)
                    frow = FCAL.loc[[xrow_feat_end]]
                    fcs["fourier"].append(float(mf.predict(frow.values)[0]))
                for h in HORIZONS:
                    io = origin + h - 1
                    if io >= len(test):
                        continue
                    rec[f"actual_h{h}"] = float(test[f].iloc[io])
                    rec[f"ucm_h{h}"] = float(fc_ucm[h - 1])
                    rec[f"stl_h{h}"] = float(fc_stl[h - 1])
                    for m in fcs:
                        rec[f"{m}_h{h}"] = float(fcs[m][HORIZONS.index(h)])
                rows.append(rec)
        print(f"lipatan {k+1}/{NFOLDS} selesai", flush=True)
    b3 = pd.DataFrame(rows)
    b3.to_csv(OUTP, index=False)
    print("ramalan fase 20 tersimpan:", b3.shape)

# ---------- gabungkan semua lengan + metode paper ----------
KEYS = ["feature", "fold", "origin_date"]
def slim(df, prefixes):
    keep = KEYS + [f"{p}_h{h}" for p in prefixes for h in HORIZONS]
    out = df[keep].copy()
    out["origin_date"] = out["origin_date"].astype(str)
    return out

b3["fold"] = b3["fold"].astype(int)
b3["origin_date"] = b3["origin_date"].astype(str)
o12 = slim(pd.read_csv(os.path.join(RES, "12_final_predictions.csv")), ["sarima", "final"])
o16 = slim(pd.read_csv(os.path.join(RES, "16_lgbm_gate_predictions.csv")), ["lgbm"])
o17 = slim(pd.read_csv(os.path.join(RES, "17_lgbm_direct_predictions.csv")), ["lgbmdir"])
o19 = slim(pd.read_csv(os.path.join(RES, "19_bakeoff2_predictions.csv")), ["theta", "rf"])
o19["fold"] = o19["fold"].astype(int) * 120
o18 = slim(pd.read_csv(os.path.join(RES, "18_bakeoff_predictions.csv")), ["naive7", "ets", "arima_nn"])
o18["fold"] = o18["fold"].astype(int) * 120
p13 = pd.read_csv(os.path.join(RES, "13_bilstm_all_folds_predictions.csv"))
p13["origin_date"] = p13["origin_date"].astype(str)
paper = p13[KEYS + [f"actual_h{h}" for h in HORIZONS]
            + [f"sarima_h{h}" for h in HORIZONS] + [f"hybrid_h{h}" for h in HORIZONS]].rename(
    columns={f"sarima_h{h}": f"paper_s12_h{h}" for h in HORIZONS} |
            {f"hybrid_h{h}": f"paper_hybrid_h{h}" for h in HORIZONS})
merged = (b3.merge(paper.drop(columns=[f"actual_h{h}" for h in HORIZONS]).drop_duplicates(KEYS), on=KEYS)
             .merge(o12, on=KEYS).merge(o16, on=KEYS).merge(o17, on=KEYS)
             .merge(o19, on=KEYS).merge(o18, on=KEYS, how="inner"))
for h in HORIZONS:                                    # ensemble tanpa-SARIMA (rata-rata sama)
    merged[f"ens_lgbm_theta_h{h}"] = (merged[f"lgbmdir_h{h}"] + merged[f"theta_h{h}"]) / 2
    merged[f"ens_lgbm_ets_h{h}"] = (merged[f"lgbmdir_h{h}"] + merged[f"ets_h{h}"]) / 2
    merged[f"ens_lgbm_rf_h{h}"] = (merged[f"lgbmdir_h{h}"] + merged[f"rf_h{h}"]) / 2
    merged[f"ens_trio_h{h}"] = (merged[f"lgbmdir_h{h}"] + merged[f"theta_h{h}"] + merged[f"rf_h{h}"]) / 3
    merged[f"ens_lgbmwk_theta_h{h}"] = (merged[f"lgbmwk_h{h}"] + merged[f"theta_h{h}"]) / 2
print("gabungan:", len(merged), "| per fitur:", merged.groupby("feature").size().to_dict())

ARMS = {
    "paper_hybrid": "paper_hybrid", "paper_sarima_s12": "paper_s12",
    "SARIMA_mingguan": "sarima", "ridge_gate": "final", "lgbm_gate": "lgbm",
    "lgbm_langsung": "lgbmdir", "svr_langsung": "svr", "knn_langsung": "knn",
    "ridge_langsung": "ridge", "lgbm_mingguan": "lgbmwk", "fourier_ols": "fourier",
    "ucm": "ucm", "stl_ets": "stl", "theta": "theta", "rf_langsung": "rf",
    "ets": "ets", "arima_nonmusiman": "arima_nn", "naive7": "naive7",
    "ens_lgbm_theta": "ens_lgbm_theta", "ens_lgbm_ets": "ens_lgbm_ets",
    "ens_lgbm_rf": "ens_lgbm_rf", "ens_trio": "ens_trio", "ens_lgbmwk_theta": "ens_lgbmwk_theta",
}
NO_SARIMA = ["lgbm_langsung", "svr_langsung", "knn_langsung", "ridge_langsung", "lgbm_mingguan",
             "fourier_ols", "ucm", "stl_ets", "theta", "rf_langsung", "ets",
             "ens_lgbm_theta", "ens_lgbm_ets", "ens_lgbm_rf", "ens_trio", "ens_lgbmwk_theta"]
met, dmrows = [], []
for f in FEATURES:
    for h in HORIZONS:
        d = merged[merged.feature == f]
        errs = {}
        for arm, pre in ARMS.items():
            ok = np.isfinite(d[f"actual_h{h}"]) & np.isfinite(d[f"{pre}_h{h}"])
            e = (d[f"actual_h{h}"] - d[f"{pre}_h{h}"])[ok].values
            errs[arm] = np.where(np.isfinite(d[f"{pre}_h{h}"]),
                                 d[f"actual_h{h}"] - d[f"{pre}_h{h}"], np.nan)
            met.append({"target": f, "horizon": h, "model": arm,
                        "rmse": float(np.sqrt(np.mean(e ** 2))),
                        "mae": float(np.mean(np.abs(e))), "n": int(ok.sum())})
        for base_name in ("SARIMA_mingguan", "paper_hybrid"):
            for arm in ARMS:
                if arm == base_name:
                    continue
                p, n = dm_p(errs[arm], errs[base_name], h)
                ok = np.isfinite(errs[arm]) & np.isfinite(errs[base_name])
                rr = float(np.sqrt(np.mean(errs[arm][ok] ** 2)) /
                           np.sqrt(np.mean(errs[base_name][ok] ** 2)) - 1)
                dmrows.append({"target": f, "horizon": h, "vs": arm, "base": base_name,
                               "rmse_ratio": rr, "dm_p": p, "n": n})
metdf = pd.DataFrame(met)
dm = pd.DataFrame(dmrows)
metdf.to_csv(os.path.join(RES, "20_bakeoff3_metrics.csv"), index=False)
dm.to_csv(os.path.join(RES, "20_bakeoff3_dm.csv"), index=False)

pd.set_option("display.width", 250)
print("\n== PAPAN PERINGKAT (RMSE Box-Cox) ==")
print(metdf.pivot_table(index=["target", "horizon"], columns="model", values="rmse").round(4).to_string())

def score(base):
    m2 = metdf.copy()
    m2["ratio"] = m2.apply(lambda r: r["rmse"] /
                           metdf[(metdf.target == r.target) & (metdf.horizon == r.horizon) &
                                 (metdf.model == base)]["rmse"].iloc[0] - 1, axis=1)
    return (m2.groupby("model")["ratio"].mean() * 100).sort_values().round(2)

print("\n== SKOR GABUNGAN 9 SEL (% vs SARIMA mingguan kita) ==")
print(score("SARIMA_mingguan").to_string())
print("\n== SKOR GABUNGAN 9 SEL (% vs metode lengkap paper) ==")
print(score("paper_hybrid").to_string())

print("\n== MENANG/9 lawan metode lengkap paper (lower RMSE) ==")
wins = {}
for arm in ARMS:
    w = 0
    for f in FEATURES:
        for h in HORIZONS:
            a = metdf[(metdf.target == f) & (metdf.horizon == h) & (metdf.model == arm)]["rmse"].iloc[0]
            b = metdf[(metdf.target == f) & (metdf.horizon == h) & (metdf.model == "paper_hybrid")]["rmse"].iloc[0]
            w += int(a < b)
    wins[arm] = w
for arm, w in sorted(wins.items(), key=lambda x: -x[1]):
    print(f"  {arm}: {w}/9")

print("\n== DM lengan baru/ensemble vs metode lengkap paper & vs SARIMA kita (p<0.05 = nyata) ==")
print(dm[dm.vs.isin(NO_SARIMA) & ((dm.base == "paper_hybrid") | (dm.base == "SARIMA_mingguan"))]
      .round(4).to_string(index=False))
