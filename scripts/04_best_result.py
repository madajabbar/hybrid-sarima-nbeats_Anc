"""Lanjutan Jalan B — cari hasil terbaik (permintaan dosen: eksperimen dulu).

Tiga perubahan dari eksperimen 03:
1. Bobot koreksi (alfa) dipilih ulang di setiap origin pakai validasi bergulir — bukan satu potongan tetap.
2. Diuji di tiga polutan (NO2, PM10, PM25) — paper mengklaim hybrid menang di PM2.5/PM10.
3. Arm tambahan: gabungan berbobot (SARIMA + koreksi) dengan bobot dari validasi.

Protokol tetap identik notebook 01/02: Box-Cox, train = sisa, test 720 hari, 1 origin tiap 30 hari,
SARIMA (2,1,1)(0,0,0,12), arm "full" (alfa=1) sebagai kontrol cara paper.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import boxcox, wilcoxon
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'transformed_data.csv')
OUT = os.path.join(ROOT, 'results', '04_best_result_metrics.csv')

FEATURES = ['NO2', 'PM10', 'PM25']
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
SARIMA_ORDER, SEASONAL_ORDER = (2, 1, 1), (0, 0, 0, 12)
VAL_WINDOW = 120      # panjang jendela validasi (hari) untuk memilih alfa, dihitung mundur dari origin
LAGS = [1, 2, 3, 4, 5, 6, 7]

# angka referensi (fase 1): reproduksi kita = paper; dipakai untuk membandingkan arm BiLSTM
REF_BILSTM = {('NO2', 1): 0.591472967107877, ('NO2', 7): 0.760324, ('NO2', 30): 0.628948}


def load_all():
    raw = pd.read_csv(DATA)
    raw.columns = [c.strip('\ufeff') for c in raw.columns]
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d-%m-%y')
    raw = raw.sort_values('Date').set_index('Date')
    out = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        out[c], _ = boxcox(raw[c] + abs(raw[c].min()) + 1)
    tr = out.iloc[:-TEST_SIZE_DAYS]
    te = out.iloc[-TEST_SIZE_DAYS:]
    return tr, te


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


def make(kind):
    if kind == 'ridge':
        m, sc = Ridge(alpha=1.0), StandardScaler()
        return lambda X, y: (sc.fit(X), m.fit(sc.transform(X), y)) and (lambda Z: m.predict(sc.transform(Z)))
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=0)
    return lambda X, y: (m.fit(X, y), lambda Z: m.predict(Z))[1]


def alpha_from_val(pred, y):
    d = float(np.dot(pred, pred))
    return float(np.clip(np.dot(pred, y) / d, 0.0, 1.0)) if d > 1e-12 else 0.0


def metr(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.sqrt(np.mean((a - p) ** 2))), float(np.mean(np.abs(a - p))),
            float(np.mean(np.abs((a - p) / a)) * 100))


def run_feature(feat, tr_all, te_all):
    tr, te = tr_all[feat], te_all[feat]
    sarima = SARIMAX(tr, order=SARIMA_ORDER, seasonal_order=SEASONAL_ORDER).fit(disp=False)
    r_train = (tr - sarima.fittedvalues).dropna()
    rows = []
    for origin in range(0, len(te) - min(HORIZONS), ORIGIN_STEP):
        hist = pd.concat([tr, te.iloc[:origin]])
        res = sarima.apply(hist)
        fc = np.asarray(res.forecast(steps=MAX_H), float)
        r_hist = (hist - res.fittedvalues).dropna()
        Xnow = featurize(r_hist).iloc[-1:].fillna(0.0)
        for h in HORIZONS:
            if origin >= len(te) - h:
                continue
            rec = {'feature': feat, 'horizon': h, 'origin': str(te.index[origin].date()),
                   'actual': float(te.iloc[origin + h - 1]), 'sarima': float(fc[h - 1])}
            # target & fitur pada riwayat yang tersedia di origin ini (residu h-langkah)
            y_all = r_hist.shift(-h).dropna()
            X_all = featurize(r_hist)
            idx = X_all.dropna().index.intersection(y_all.index)
            X_h, y_h = X_all.loc[idx], y_all.loc[idx]
            if len(idx) < VAL_WINDOW + 200:
                continue
            for kind in ('ridge', 'gbm'):
                f_train, f_val = X_h.iloc[:-VAL_WINDOW], X_h.iloc[-VAL_WINDOW:]
                y_train, y_val = y_h.iloc[:-VAL_WINDOW].values, y_h.iloc[-VAL_WINDOW:].values
                pred_val = make(kind)(f_train, y_train)(f_val)
                a = alpha_from_val(pred_val, y_val)
                pred_now = float(make(kind)(X_h, y_h)(Xnow.values)[0])
                rec[f'{kind}_alpha'] = a
                rec[f'{kind}_corr'] = pred_now
                rec[f'{kind}_full'] = rec['sarima'] + pred_now
                rec[f'{kind}_alph'] = rec['sarima'] + a * pred_now
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    tr, te = load_all()
    all_rows, summary, tests = [], [], []
    for feat in FEATURES:
        df = run_feature(feat, tr, te)
        all_rows.append(df)
        print(f'\n===== {feat} | {len(df[df.horizon == 1])} / {len(df[df.horizon == 7])} / {len(df[df.horizon == 30])} origin')
        arms = {'SARIMA': 'sarima', 'paper_style_fullridge': 'ridge_full', 'paper_style_fullgbm': 'gbm_full',
                'ours_ridge_alpha': 'ridge_alph', 'ours_gbm_alpha': 'gbm_alph'}
        for h in HORIZONS:
            d = df[df.horizon == h]
            if d.empty:
                continue
            base = metr(d.actual, d.sarima)[0]
            for name, col in arms.items():
                rmse, mae, mape = metr(d.actual, d[col])
                summary.append({'target': feat, 'horizon': h, 'model': name, 'rmse': rmse,
                                'mae': mae, 'mape': mape, 'delta_vs_sarima': rmse - base, 'n': len(d)})
                if col != 'sarima':
                    l1, l2 = (d.actual - d.sarima) ** 2, (d.actual - d[col]) ** 2
                    p = float(wilcoxon(l1, l2).pvalue) if len(d) >= 8 and np.any(l1 != l2) else np.nan
                    a_mean = float(d.get(f'{col.split("_")[0]}_alpha', pd.Series([np.nan])).mean())
                    tests.append({'target': feat, 'horizon': h, 'model': name,
                                  'rmse_delta': rmse - base, 'wilcoxon_p': p, 'alpha_rata2': a_mean})
        s = pd.DataFrame(summary)
        print(s[s.target == feat].to_string(index=False))
    pd.DataFrame(summary).to_csv(OUT, index=False)
    pd.concat(all_rows).to_csv(OUT.replace('.csv', '_predictions.csv'), index=False)
    print('\n===== uji beda vs SARIMA (Wilcoxon berpasangan atas kuadrat galat) =====')
    print(pd.DataFrame(tests).to_string(index=False))

    best = pd.DataFrame(summary).sort_values(['target', 'horizon', 'rmse'])
    g = best.groupby(['target', 'horizon']).first().reset_index()
    print('\n===== arm terbaik per polutan x horizon =====')
    print(g[['target', 'horizon', 'model', 'rmse', 'mape']].to_string(index=False))
    for _, r in g.iterrows():
        ref = REF_BILSTM.get((r['target'], r['horizon']))
        if ref:
            print(f"  {r['target']} h={r['horizon']}: terbaik {r['rmse']:.4f} vs BiLSTM referensi {ref:.4f} "
                  f"({(r['rmse'] - ref) / ref * 100:+.1f}%)")


if __name__ == '__main__':
    main()