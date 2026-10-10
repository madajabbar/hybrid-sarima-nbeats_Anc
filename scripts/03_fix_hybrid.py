"""Jalan B — perbaiki protokol hybrid, lalu adu ulang dengan aturan adil.

Pertanyaan: koreksi residu yang salah (dipakai penuh, tanpa penyusutan) itukah yang
membuat hybrid kalah dari SARIMA? Uji: latih koreksi yang terkondisi baik (lag residu),
pilih bobot penyusutan alfa di data validasi, lalu adu di 24 origin rolling-origin.

Protokol identik notebook 01/02: NO2, Box-Cox, test 720 hari, 1 origin tiap 30 hari,
SARIMA (2,1,1)(0,0,0,12). Arm: SARIMA saja | koreksi penuh (alfa=1) | ridge+alfa | GBM+alfa.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import boxcox, ttest_rel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'transformed_data.csv')
OUT = os.path.join(ROOT, 'results', '03_fix_hybrid_metrics.csv')

FEATURE = 'NO2'
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H = 720, 30, 30
HORIZONS = [1, 7, 30]
SARIMA_ORDER, SEASONAL_ORDER = (2, 1, 1), (0, 0, 0, 12)
VAL_TAIL = 90          # ekor train untuk memilih alfa
LAGS = [1, 2, 3, 4, 5, 6, 7]


def load():
    raw = pd.read_csv(DATA)
    raw.columns = [c.strip('\ufeff') for c in raw.columns]
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d-%m-%y')
    raw = raw.sort_values('Date').set_index('Date')
    out = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        out[c], _ = boxcox(raw[c] + abs(raw[c].min()) + 1)   # sama persis notebook 01/02
    tr = out.iloc[:-TEST_SIZE_DAYS]
    te = out.iloc[-TEST_SIZE_DAYS:]
    return tr[FEATURE], te[FEATURE]


def featurize(r):
    """r: deret residu 1-langkah (sudah selaras waktu). Fitur dari masa lalu saja."""
    X = pd.DataFrame(index=r.index)
    for L in LAGS:
        X[f'lag{L}'] = r.shift(L)
    X['roll7'] = r.shift(1).rolling(7).mean()
    X['roll30'] = r.shift(1).rolling(30).mean()
    X['std30'] = r.shift(1).rolling(30).std()
    X['lag365'] = r.shift(365)
    return X


def fit_arm(X, y, kind):
    m = Ridge(alpha=1.0) if kind == 'ridge' else HistGradientBoostingRegressor(max_iter=200, random_state=0)
    if kind == 'ridge':
        sc = StandardScaler().fit(X)
        m.fit(sc.transform(X), y)
        return lambda Z: m.predict(sc.transform(Z))
    m.fit(X, y)
    return lambda Z: m.predict(Z)


def pick_alpha(pred_val, y_val):
    """Bobot penyusutan terbaik di validasi: proyeksi terbatas [0,1]."""
    denom = float(np.dot(pred_val, pred_val))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(pred_val, y_val) / denom, 0.0, 1.0))


def metr(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.sqrt(np.mean((a - p) ** 2))), float(np.mean(np.abs(a - p))),
            float(np.mean(np.abs((a - p) / a)) * 100))


def main():
    tr, te = load()
    print(f'train {len(tr)} | test {len(te)} | fitur residu NO2 (skala Box-Cox)')
    sarima = SARIMAX(tr, order=SARIMA_ORDER, seasonal_order=SEASONAL_ORDER).fit(disp=False)
    r_train = (tr - sarima.fittedvalues).dropna()
    Xtr = featurize(r_train)
    print(f'residu train: {len(r_train)} baris | fitur tersedia: {Xtr.notna().sum().max()}')

    # latih per horizon + pilih alfa di ekor train (validasi), bukan di test
    arms = {}
    for h in HORIZONS:
        y = r_train.shift(-h).dropna()
        idx = Xtr.dropna().index.intersection(y.index)
        Xh, yh = Xtr.loc[idx], y.loc[idx]
        cut = len(idx) - VAL_TAIL
        for kind in ('ridge', 'gbm'):
            pred = fit_arm(Xh.iloc[:cut], yh.iloc[:cut], kind)
            a = pick_alpha(pred(Xh.iloc[cut:]), yh.iloc[cut:].values)
            arms[(kind, h)] = (pred, a)
            print(f'  h={h:2d} {kind:5s}: alfa validasi = {a:.3f}')

    # rolling-origin: satu apply per origin (dipakai ulang lintas horizon, sama seperti notebook 01:
    # origin per horizon = range(0, len(test)-h, 30) sehingga jumlah origin berbeda tiap horizon)
    cache = {}
    for origin in range(0, len(te) - min(HORIZONS), ORIGIN_STEP):
        hist = pd.concat([tr, te.iloc[:origin]])
        res = sarima.apply(hist)
        fc = np.asarray(res.forecast(steps=MAX_H), float)
        Xnow = featurize((hist - res.fittedvalues).dropna()).iloc[-1:].fillna(0.0)
        cache[origin] = (fc, Xnow)

    rows = []
    for h in HORIZONS:
        for origin in sorted(o for o in cache if o < len(te) - h):
            fc, Xnow = cache[origin]
            rec = {'horizon': h, 'origin': str(te.index[origin].date()),
                   'target': str(te.index[origin + h - 1].date()),
                   'actual': float(te.iloc[origin + h - 1]), 'sarima': float(fc[h - 1])}
            for kind in ('ridge', 'gbm'):
                pred, a = arms[(kind, h)]
                c = float(pred(Xnow.values)[0])
                rec[f'{kind}_corr'] = c
                rec[f'{kind}_full'] = rec['sarima'] + c          # alfa=1 (cara paper)
                rec[f'{kind}_shrunk'] = rec['sarima'] + a * c     # alfa dari validasi
            rows.append(rec)
    df = pd.DataFrame(rows)

    res_rows, dm_rows = [], []
    for h in HORIZONS:
        d = df[df.horizon == h]
        cols = {'SARIMA': 'sarima', 'ridge_full': 'ridge_full', 'ridge_shrunk': 'ridge_shrunk',
                'gbm_full': 'gbm_full', 'gbm_shrunk': 'gbm_shrunk'}
        base = d['actual'] - d['sarima']
        for name, c in cols.items():
            rmse, mae, mape = metr(d.actual, d[c])
            res_rows.append({'target': FEATURE, 'horizon': h, 'model': name,
                             'rmse': rmse, 'mae': mae, 'mape': mape,
                             'n_origins': len(d)})
            if c != 'sarima':
                l1 = (d['actual'] - d['sarima']) ** 2
                l2 = (d['actual'] - d[c]) ** 2
                p = float(ttest_rel(l1, l2).pvalue)
                dm_rows.append({'horizon': h, 'model': name, 'rmse_delta': rmse - metr(d.actual, d['sarima'])[0],
                                'dm_pvalue': p})
    out = pd.DataFrame(res_rows)
    out.to_csv(OUT, index=False)
    df.to_csv(OUT.replace('.csv', '_predictions.csv'), index=False)

    print('\n=== RMSE / MAE / MAPE per horizon (rata-rata 24 origin) ===')
    print(out.to_string(index=False))
    print('\n=== uji beda vs SARIMA (uji-t berpasangan atas kuadrat galat) ===')
    print(pd.DataFrame(dm_rows).to_string(index=False))

    # cek protokol: arm SARIMA harus mereproduksi blok horizon notebook 01
    ref = {1: 0.595098, 7: 0.754475, 30: 0.608706}
    for h in HORIZONS:
        got = float(out[(out.horizon == h) & (out.model == 'SARIMA')].rmse.iloc[0])
        assert abs(got - ref[h]) < 0.01, f'protokol tidak cocok h={h}: {got:.4f} vs {ref[h]}'
    print('\nOK: protokol SARIMA cocok dengan notebook 01 (toleransi 0,01)')


if __name__ == '__main__':
    main()