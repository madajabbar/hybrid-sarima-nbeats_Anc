"""Fase 9 — perbaiki kriteria penerimaan koreksi.

Diagnosis Fase 8: di PM25 jarak 7 hari koreksi justru menyesatkan (korelasi besar-koreksi vs
kerugian +0,47), tetapi validasi satu jendela (120 hari) tetap menjawab "pakai". Perbaikan:
keputusan diambil dari **tiga blok validasi bergulir (60 hari masing-masing)** dengan syarat
(a) mayoritas blok menolong, dan (b) rata-rata penurunan RMSE minimal THRESH persen.

Base SARIMA = varian mingguan (2,1,1)(1,0,1,7) — hasil Fase 7.

Arm:
  adaptif_120 : cara lama (satu jendela 120 hari, alfa dari proyeksi)
  ketat       : gerbang 3 blok + alfa dari proyeksi jendela 180 hari
  ketat_penuh : gerbang 3 blok + alfa = 1
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import boxcox, t as tdist
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'transformed_data.csv')
OUT = os.path.join(ROOT, 'results', '11_acceptance_metrics.csv')

FEATURES = ['NO2', 'PM10', 'PM25']
FOLD_OFFSETS = [0, 180, 360]
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
ORDER, SEASONAL = (2, 1, 1), (1, 0, 1, 7)
LAGS, MIN_TRAIN = [1, 2, 3, 4, 5, 6, 7], 400
BLOCK, NBLOCK, THRESH = 60, 3, 5.0        # 3 blok x 60 hari, ambang penurunan 5%
ARMS = {'SARIMA': 'sarima_h{}', 'adaptif_120': 'adapt_h{}', 'ketat': 'ketat_h{}', 'ketat_penuh': 'ketatf_h{}'}


def load_all():
    raw = pd.read_csv(DATA)
    raw.columns = [c.strip('\ufeff') for c in raw.columns]
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d-%m-%y')
    raw = raw.sort_values('Date').set_index('Date')
    out = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        out[c], _ = boxcox(raw[c] + abs(raw[c].min()) + 1)
    return out


def featurize(r):
    X = pd.DataFrame(index=r.index)
    for L in LAGS:
        X[f'lag{L}'] = r.shift(L)
    X['roll7'] = r.shift(1).rolling(7).mean()
    X['roll30'] = r.shift(1).rolling(30).mean()
    X['std30'] = r.shift(1).rolling(30).std()
    X['lag365'] = r.shift(365)
    doy = r.index.dayofyear
    X['sin365'] = np.sin(2 * np.pi * doy / 365.25)
    X['cos365'] = np.cos(2 * np.pi * doy / 365.25)
    X['dow'] = r.index.dayofweek
    return X


def ridge_fit(X, y):
    sc, m = StandardScaler(), Ridge(alpha=1.0)
    sc.fit(X); m.fit(sc.transform(X), y)
    return lambda Z: m.predict(sc.transform(Z))


def rmse(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def dm(e1, e2, h):
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    dbar = float(np.mean(d))
    g0 = v = float(np.mean((d - dbar) ** 2))
    for k in range(1, min(h - 1, max(1, n // 4)) + 1):
        v += 2 * float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
    if not np.isfinite(v) or v <= 0:
        v = g0
    stat = (dbar / np.sqrt(v / n)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, float(2 * (1 - tdist.cdf(abs(stat), n - 1)))


def run_fold(all_data, feat, offset):
    end = len(all_data) - offset
    start = end - TEST_SIZE_DAYS
    tr, te = all_data[feat].iloc[:start], all_data[feat].iloc[start:end]
    if len(tr) < MIN_TRAIN:
        return pd.DataFrame()
    sarima = SARIMAX(tr, order=ORDER, seasonal_order=SEASONAL).fit(disp=False)
    rows = []
    for origin in range(0, len(te) - min(HORIZONS), ORIGIN_STEP):
        hist = pd.concat([tr, te.iloc[:origin]])
        res = sarima.apply(hist)
        fc = np.asarray(res.forecast(steps=MAX_H), float)
        r_hist = (hist - res.fittedvalues).dropna()
        Xnow = featurize(r_hist).iloc[-1:].fillna(0.0)
        rec = {'feature': feat, 'fold': offset, 'origin_date': str(te.index[origin].date())}
        for h in HORIZONS:
            if origin >= len(te) - h:
                continue
            base = float(fc[h - 1])
            rec[f'actual_h{h}'] = float(te.iloc[origin + h - 1])
            rec[f'sarima_h{h}'] = base
            y_all = r_hist.shift(-h).dropna()
            X_all = featurize(r_hist)
            idx = X_all.dropna().index.intersection(y_all.index)
            need = BLOCK * NBLOCK + 200
            if len(idx) < need:
                continue
            X_h, y_h = X_all.loc[idx], y_all.loc[idx]
            yv = y_h.values
            # arm lama: satu jendela 120 hari
            f120 = X_h.iloc[-120:]
            y120 = yv[-120:]
            p120 = ridge_fit(X_h.iloc[:-120], yv[:-120])(f120)
            d120 = float(np.dot(p120, p120))
            a120 = float(np.clip(np.dot(p120, y120) / d120, 0.0, 1.0)) if d120 > 1e-12 else 0.0
            # arm ketat: 3 blok x 60 hari
            imps, alphas = [], []
            for b in range(NBLOCK):
                s = -(b + 1) * BLOCK
                e = -b * BLOCK if b else None
                Xb, yb = X_h.iloc[s:e], yv[s:e]
                pb = ridge_fit(X_h.iloc[:s], yv[:s])(Xb)
                dz = rmse(yb, np.zeros_like(yb))
                imps.append((dz - rmse(yb, pb)) / dz * 100 if dz > 0 else 0.0)
                dd = float(np.dot(pb, pb))
                alphas.append(float(np.clip(np.dot(pb, yb) / dd, 0.0, 1.0)) if dd > 1e-12 else 0.0)
            p180 = ridge_fit(X_h.iloc[:-180], yv[:-180])(X_h.iloc[-180:])
            d180 = float(np.dot(p180, p180))
            a180 = float(np.clip(np.dot(p180, yv[-180:]) / d180, 0.0, 1.0)) if d180 > 1e-12 else 0.0
            gate = (sum(1 for i in imps if i > 0) >= 2) and (float(np.mean(imps)) >= THRESH)
            corr = float(ridge_fit(X_h, y_h)(Xnow.values)[0])
            rec[f'adapt_h{h}'] = base + a120 * corr
            rec[f'ketat_h{h}'] = base + (a180 if gate else 0.0) * corr
            rec[f'ketatf_h{h}'] = base + (1.0 if gate else 0.0) * corr
            rec[f'gate_h{h}'] = gate
            rec[f'imp_mean_h{h}'] = float(np.mean(imps))
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    all_data = load_all()
    frames = []
    for feat in FEATURES:
        for off in FOLD_OFFSETS:
            df = run_fold(all_data, feat, off)
            if not df.empty:
                frames.append(df)
                print(f'{feat} fold+{off}: {len(df)} origin | gerbang "pakai" h1 {int(df.gate_h1.sum())}/{df.gate_h1.notna().sum()} '
                      f'h7 {int(df.gate_h7.sum())}/{df.gate_h7.notna().sum()} h30 {int(df.gate_h30.sum())}/{df.gate_h30.notna().sum()}', flush=True)
    big = pd.concat(frames, ignore_index=True)
    rows, dm_rows = [], []
    for feat in FEATURES:
        for h in HORIZONS:
            ac = f'actual_h{h}'
            d = big[(big.feature == feat) & big[ac].notna()]
            base_rmse = rmse(d[ac], d[f'sarima_h{h}'])
            for name, pat in ARMS.items():
                col = pat.format(h)
                if col not in d.columns:
                    continue
                dd = d[d[col].notna()]
                got = rmse(dd[ac], dd[col])
                rows.append({'target': feat, 'horizon': h, 'model': name, 'rmse': got,
                             'mae': float(np.mean(np.abs(dd[ac] - dd[col]))),
                             'mape': float(np.mean(np.abs((dd[ac] - dd[col]) / dd[ac])) * 100),
                             'delta_pct': (got - base_rmse) / base_rmse * 100, 'n': len(dd)})
                if col != f'sarima_h{h}':
                    _, p = dm(dd[ac].values - dd[f'sarima_h{h}'].values, dd[ac].values - dd[col].values, h)
                    dm_rows.append({'target': feat, 'horizon': h, 'model': name,
                                    'delta_pct': (got - base_rmse) / base_rmse * 100, 'dm_p': p,
                                    'nyata05': 'ya' if (p == p and p < 0.05) else 'tidak'})
    s, t = pd.DataFrame(rows), pd.DataFrame(dm_rows)
    s.to_csv(OUT, index=False); t.to_csv(OUT.replace('.csv', '_dm.csv'), index=False)
    big.to_csv(OUT.replace('.csv', '_predictions.csv'), index=False)
    print('\n===== semua arm (base SARIMA musiman mingguan) =====')
    print(s.to_string(index=False))
    print('\n===== uji vs SARIMA =====')
    print(t.to_string(index=False))
    print('\n===== ringkas =====')
    for name in ['adaptif_120', 'ketat', 'ketat_penuh']:
        r = s[s.model == name]
        d_ = t[t.model == name]
        print(f'{name:12s}: menang {int((r.delta_pct < 0).sum())}/{len(r)} | rata2 {r.delta_pct.mean():+.2f}% | '
              f'terbaik {r.delta_pct.min():+.2f}% | terburuk {r.delta_pct.max():+.2f}% | '
              f'menang nyata {int(((d_.nyata05 == "ya") & (d_.delta_pct < 0)).sum())} | '
              f'kalah nyata {int(((d_.nyata05 == "ya") & (d_.delta_pct > 0)).sum())}')


if __name__ == '__main__':
    main()