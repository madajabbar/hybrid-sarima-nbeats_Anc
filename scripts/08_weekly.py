"""Fase 7 — apakah pengaturan musiman paper salah untuk data harian?

Paper memakai SARIMA (2,1,1)(0,0,0,12): angka 12 pada data HARIAN tidak mewakili siklus mingguan,
sehingga pola kerja vs akhir pekan tidak dimodelkan. Dugaan: itulah sebab galat jarak 7 hari membengkak.

Varian yang diuji (protokol lain identik notebook 01/02):
  A. paper        : (2,1,1)(0,0,0,12)
  B. mingguan     : (2,1,1)(1,0,1,7)
  C. paper + Fou  : (2,1,1)(0,0,0,12) + Fourier mingguan (sin/cos k=1..3) sebagai variabel eksogen
Dua arm dievaluasi: SARIMA saja (untuk memisahkan efek musiman) dan resep kita (SARIMA + alfa*koreksi Ridge).

C hanya dijalankan untuk NO2 (biaya); A dan B dijalankan untuk ketiga polutan.
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
OUT = os.path.join(ROOT, 'results', '08_weekly_metrics.csv')

VARIANTS = {
    'A_paper_s12': {'order': (2, 1, 1), 'seasonal': (0, 0, 0, 12), 'weekly_exog': False},
    'B_mingguan_s7': {'order': (2, 1, 1), 'seasonal': (1, 0, 1, 7), 'weekly_exog': False},
    'C_paper_fourier': {'order': (2, 1, 1), 'seasonal': (0, 0, 0, 12), 'weekly_exog': True},
}
PLAN = {'NO2': ['A_paper_s12', 'B_mingguan_s7', 'C_paper_fourier'],
        'PM10': ['A_paper_s12', 'B_mingguan_s7'],
        'PM25': ['A_paper_s12', 'B_mingguan_s7'],
        }
FOLD_OFFSETS = [0, 180, 360]
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
VAL_WINDOW, LAGS, MIN_TRAIN = 120, [1, 2, 3, 4, 5, 6, 7], 400


def load_all():
    raw = pd.read_csv(DATA)
    raw.columns = [c.strip('\ufeff') for c in raw.columns]
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d-%m-%y')
    raw = raw.sort_values('Date').set_index('Date')
    out = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        out[c], _ = boxcox(raw[c] + abs(raw[c].min()) + 1)
    return out


def weekly_exog(idx, k=3):
    doy = idx.dayofweek.values + idx.hour.values / 24.0 if hasattr(idx, 'hour') else idx.dayofweek.values
    w = 2 * np.pi * doy / 7.0
    return pd.DataFrame({f'{pre}{j}': (np.sin if pre == 'sin' else np.cos)(j * w)
                         for j in range(1, k + 1) for pre in ('sin', 'cos')}, index=idx)


def featurize(r, exog=None):
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
    if exog is not None:
        for c in exog.columns:
            X[f'w_{c}'] = exog[c]
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


def run(all_data, feat, offset, vname, cfg):
    end = len(all_data) - offset
    start = end - TEST_SIZE_DAYS
    tr, te = all_data[feat].iloc[:start], all_data[feat].iloc[start:end]
    if len(tr) < MIN_TRAIN:
        return pd.DataFrame()
    ex_tr = weekly_exog(tr.index) if cfg['weekly_exog'] else None
    kw = {'exog': ex_tr} if ex_tr is not None else {}
    m = SARIMAX(tr, order=cfg['order'], seasonal_order=cfg['seasonal'], **kw).fit(disp=False)
    rows = []
    for origin in range(0, len(te) - min(HORIZONS), ORIGIN_STEP):
        hist = pd.concat([tr, te.iloc[:origin]])
        kw2 = {'exog': weekly_exog(hist.index)} if cfg['weekly_exog'] else {}
        res = m.apply(hist, **kw2)
        kwe = {}
        if cfg['weekly_exog']:
            fut = pd.date_range(hist.index[-1] + pd.Timedelta(days=1), periods=MAX_H, freq='D')
            kwe = {'exog': weekly_exog(fut)}
        fc = np.asarray(res.forecast(steps=MAX_H, **kwe), float)
        r_hist = (hist - res.fittedvalues).dropna()
        ex_now = weekly_exog(r_hist.index) if cfg['weekly_exog'] else None
        Xnow = featurize(r_hist, ex_now).iloc[-1:].fillna(0.0)
        rec = {'variant': vname, 'feature': feat, 'fold': offset, 'origin_date': str(te.index[origin].date())}
        for h in HORIZONS:
            if origin >= len(te) - h:
                continue
            rec[f'actual_h{h}'] = float(te.iloc[origin + h - 1])
            rec[f'sarima_h{h}'] = float(fc[h - 1])
            y_all = r_hist.shift(-h).dropna()
            ex_all = weekly_exog(r_hist.index) if cfg['weekly_exog'] else None
            X_all = featurize(r_hist, ex_all)
            idx = X_all.dropna().index.intersection(y_all.index)
            if len(idx) < VAL_WINDOW + 200:
                continue
            X_h, y_h = X_all.loc[idx], y_all.loc[idx]
            f_tr, f_val = X_h.iloc[:-VAL_WINDOW], X_h.iloc[-VAL_WINDOW:]
            y_tr, y_val = y_h.iloc[:-VAL_WINDOW].values, y_h.iloc[-VAL_WINDOW:].values
            pv = ridge_fit(f_tr, y_tr)(f_val)
            dd = float(np.dot(pv, pv))
            a = float(np.clip(np.dot(pv, y_val) / dd, 0.0, 1.0)) if dd > 1e-12 else 0.0
            corr = float(ridge_fit(X_h, y_h)(Xnow.values)[0])
            rec[f'recipe_h{h}'] = rec[f'sarima_h{h}'] + a * corr
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    all_data = load_all()
    frames = []
    for feat, vlist in PLAN.items():
        for vname in vlist:
            for off in FOLD_OFFSETS:
                df = run(all_data, feat, off, vname, VARIANTS[vname])
                if not df.empty:
                    frames.append(df); print(f'{feat:5s} {vname:16s} fold+{off}: {len(df)} origin', flush=True)
    big = pd.concat(frames, ignore_index=True)
    rows = []
    for feat in PLAN:
        for vname in PLAN[feat]:
            for h in HORIZONS:
                ac, sc_ = f'actual_h{h}', f'sarima_h{h}'
                d = big[(big.feature == feat) & (big.variant == vname) & big[ac].notna()]
                if d.empty:
                    continue
                for arm, col in [('SARIMA', sc_), ('resep', f'recipe_h{h}')]:
                    dd = d[d[col].notna()]
                    r_ = rmse(dd[ac], dd[col])
                    rows.append({'variant': vname, 'target': feat, 'horizon': h, 'arm': arm, 'rmse': r_,
                                 'mae': float(np.mean(np.abs(dd[ac] - dd[col]))),
                                 'mape': float(np.mean(np.abs((dd[ac] - dd[col]) / dd[ac])) * 100),
                                 'n': len(dd), 'n_folds': dd.fold.nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    big.to_csv(OUT.replace('.csv', '_predictions.csv'), index=False)

    print('\n===== SARIMA saja: efek pengaturan musiman (RMSE) =====')
    piv = out[out.arm == 'SARIMA'].pivot_table(index=['target', 'horizon'], columns='variant', values='rmse')
    piv['B_vs_A_%'] = (piv['B_mingguan_s7'] - piv['A_paper_s12']) / piv['A_paper_s12'] * 100
    if 'C_paper_fourier' in piv:
        piv['C_vs_A_%'] = (piv['C_paper_fourier'] - piv['A_paper_s12']) / piv['A_paper_s12'] * 100
    print(piv.to_string())

    print('\n===== resep kita per varian =====')
    print(out[out.arm == 'resep'].pivot_table(index=['target', 'horizon'], columns='variant', values='rmse').to_string())

    print('\n===== uji B vs A (SARIMA saja, per polutan x horizon) =====')
    for feat in PLAN:
        for h in HORIZONS:
            ac, sc_ = f'actual_h{h}', f'sarima_h{h}'
            a = big[(big.feature == feat) & (big.variant == 'A_paper_s12')][['origin_date', 'fold', ac, sc_]]
            b = big[(big.feature == feat) & (big.variant == 'B_mingguan_s7')][['origin_date', 'fold', ac, sc_]]
            m = a.merge(b, on=['origin_date', 'fold'], suffixes=('_a', '_b')).dropna()
            if len(m) < 10:
                continue
            e1 = m[f'{ac}_a'].values - m[f'{sc_}_a'].values
            e2 = m[f'{ac}_b'].values - m[f'{sc_}_b'].values
            stat, p = dm(e1, e2, h)
            ra, rb = rmse(m[f'{ac}_a'], m[f'{sc_}_a']), rmse(m[f'{ac}_b'], m[f'{sc_}_b'])
            print(f'  {feat:5s} h={h:2d}: A {ra:.4f} -> B {rb:.4f} ({(rb - ra) / ra * 100:+.1f}%) | p={p:.4f}'
                  f' {"NYATA" if p < 0.05 else ""}')


if __name__ == '__main__':
    main()