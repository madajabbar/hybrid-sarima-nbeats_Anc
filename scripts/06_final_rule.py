"""Fase 6 — aturan dua tingkat: keputusan pakai/tidak koreksi diambil dari data validasi saja.

Tiga arm (semua memakai Ridge; tanpa GBM agar cepat):
  1. adaptif   : ramalan = SARIMA + alfa * koreksi            (alfa dari proyeksi di validasi)
  2. dua_tingkat_alfa : SARIMA + (alfa kalau koreksi menolong di validasi, else 0) * koreksi
  3. dua_tingkat_penuh: SARIMA + (1    kalau koreksi menolong di validasi, else 0) * koreksi

"Menolong" = RMSE koreksi < RMSE tanpa koreksi pada jendela validasi (120 hari sebelum origin).
Keputusan tidak pernah melihat data ujian. Protokol lain identik notebook 01/02.
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
OUT = os.path.join(ROOT, 'results', '06_final_rule_metrics.csv')

FEATURES = ['NO2', 'PM10', 'PM25']
FOLD_OFFSETS = [0, 180, 360]
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
SARIMA_ORDER, SEASONAL_ORDER = (2, 1, 1), (0, 0, 0, 12)
VAL_WINDOW, LAGS, MIN_TRAIN = 120, [1, 2, 3, 4, 5, 6, 7], 400
ARMS = {'SARIMA': 'sarima_h{}', 'adaptif': 'adapt_h{}',
        'dua_tingkat_alfa': 'tier_a_h{}', 'dua_tingkat_penuh': 'tier_f_h{}'}


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
    sarima = SARIMAX(tr, order=SARIMA_ORDER, seasonal_order=SEASONAL_ORDER).fit(disp=False)
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
            rec[f'actual_h{h}'] = float(te.iloc[origin + h - 1])
            rec[f'sarima_h{h}'] = float(fc[h - 1])
            y_all = r_hist.shift(-h).dropna()
            X_all = featurize(r_hist)
            idx = X_all.dropna().index.intersection(y_all.index)
            if len(idx) < VAL_WINDOW + 200:
                continue
            X_h, y_h = X_all.loc[idx], y_all.loc[idx]
            f_tr, f_val = X_h.iloc[:-VAL_WINDOW], X_h.iloc[-VAL_WINDOW:]
            y_tr, y_val = y_h.iloc[:-VAL_WINDOW].values, y_h.iloc[-VAL_WINDOW:].values
            pred_val = ridge_fit(f_tr, y_tr)(f_val)
            d = float(np.dot(pred_val, pred_val))
            a = float(np.clip(np.dot(pred_val, y_val) / d, 0.0, 1.0)) if d > 1e-12 else 0.0
            helps = rmse(y_val, pred_val * a) < rmse(y_val, np.zeros_like(y_val))   # keputusan dari validasi
            corr = float(ridge_fit(X_h, y_h)(Xnow.values)[0])
            base = rec[f'sarima_h{h}']
            rec[f'adapt_h{h}'] = base + a * corr
            rec[f'tier_a_h{h}'] = base + (a if helps else 0.0) * corr
            rec[f'tier_f_h{h}'] = base + (1.0 if helps else 0.0) * corr
            rec[f'help_h{h}'] = helps
            rec[f'alpha_h{h}'] = a
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
                print(f'{feat} fold+{off}: {len(df)} origin | keputusan "pakai" h=1: '
                      f"{int(df.help_h1.sum())}/{df.help_h1.notna().sum()}")
    big = pd.concat(frames, ignore_index=True)
    rows, dm_rows = [], []
    for feat in FEATURES:
        for h in HORIZONS:
            ac, sc_ = f'actual_h{h}', f'sarima_h{h}'
            d = big[(big.feature == feat) & big[ac].notna()]
            base_rmse = rmse(d[ac], d[sc_])
            for name, pat in ARMS.items():
                col = pat.format(h)
                if col not in d.columns:
                    continue
                dd = d[d[col].notna()]
                got = rmse(dd[ac], dd[col])
                mae = float(np.mean(np.abs(dd[ac] - dd[col])))
                mape = float(np.mean(np.abs((dd[ac] - dd[col]) / dd[ac])) * 100)
                rows.append({'target': feat, 'horizon': h, 'model': name, 'rmse': got, 'mae': mae,
                             'mape': mape, 'delta_pct_vs_sarima': (got - base_rmse) / base_rmse * 100,
                             'n': len(dd), 'n_folds': dd.fold.nunique()})
                if col != sc_:
                    _, p = dm(dd[ac].values - dd[sc_].values, dd[ac].values - dd[col].values, h)
                    dm_rows.append({'target': feat, 'horizon': h, 'model': name,
                                    'delta_pct': (got - base_rmse) / base_rmse * 100, 'dm_p': p,
                                    'nyata05': 'ya' if (p == p and p < 0.05) else 'tidak'})
    s, t = pd.DataFrame(rows), pd.DataFrame(dm_rows)
    s.to_csv(OUT, index=False); t.to_csv(OUT.replace('.csv', '_dm.csv'), index=False)
    big.to_csv(OUT.replace('.csv', '_predictions.csv'), index=False)
    print('\n===== semua arm =====')
    print(s.to_string(index=False))
    print('\n===== signifikansi (vs SARIMA) =====')
    print(t.to_string(index=False))
    print('\n===== ringkas per arm: jumlah sel menang / menang nyata / kalah nyata =====')
    for name in ['adaptif', 'dua_tingkat_alfa', 'dua_tingkat_penuh']:
        r = s[s.model == name]
        d_ = t[t.model == name]
        print(f'{name:18s}: menang {int((r.delta_pct_vs_sarima < 0).sum())}/{len(r)} | '
              f"rata2 {r.delta_pct_vs_sarima.mean():+.2f}% | terbaik {r.delta_pct_vs_sarima.min():+.2f}% | "
              f"terburuk {r.delta_pct_vs_sarima.max():+.2f}% | menang nyata {int(((d_.nyata05 == 'ya') & (d_.delta_pct < 0)).sum())} | "
              f"kalah nyata {int(((d_.nyata05 == 'ya') & (d_.delta_pct > 0)).sum())}")


if __name__ == '__main__':
    main()