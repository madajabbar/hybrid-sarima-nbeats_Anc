"""Fase 8 — diagnosis: mengapa PM25 jarak 7 hari tetap kalah?

Membedah per titik ujian (semua 3 lipatan): berapa bobot alfa yang dipilih, seberapa besar koreksinya,
apakah koreksi menolong atau merugikan, dan berapa RMSE seandainya bobot dipilih sempurna (oracle)
— untuk tahu apakah masalahnya di bobot, di koreksinya, atau di data.

Kontrol: NO2 (kasus di mana resep menang) dijalankan dengan cara yang sama.
Varian: A (paper, s=12) dan B (mingguan, s=7). Horizon 7 hari.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'transformed_data.csv')
OUT = os.path.join(ROOT, 'results', '10_diagnose_h7.csv')

VARIANTS = {'A_paper_s12': ((2, 1, 1), (0, 0, 0, 12)), 'B_mingguan_s7': ((2, 1, 1), (1, 0, 1, 7))}
FOLD_OFFSETS = [0, 180, 360]
TEST_SIZE_DAYS, ORIGIN_STEP, H, MAX_H = 720, 30, 7, 7
VAL_WINDOW, LAGS, MIN_TRAIN = 120, [1, 2, 3, 4, 5, 6, 7], 400
GRID = np.linspace(0, 1.5, 31)


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


def run(all_data, feat, vname):
    order, seas = VARIANTS[vname]
    rows = []
    for off in FOLD_OFFSETS:
        end = len(all_data) - off
        start = end - TEST_SIZE_DAYS
        tr, te = all_data[feat].iloc[:start], all_data[feat].iloc[start:end]
        if len(tr) < MIN_TRAIN:
            continue
        m = SARIMAX(tr, order=order, seasonal_order=seas).fit(disp=False)
        for origin in range(0, len(te) - H, ORIGIN_STEP):
            hist = pd.concat([tr, te.iloc[:origin]])
            res = m.apply(hist)
            fc = np.asarray(res.forecast(steps=H), float)
            r_hist = (hist - res.fittedvalues).dropna()
            y_all = r_hist.shift(-H).dropna()
            X_all = featurize(r_hist)
            idx = X_all.dropna().index.intersection(y_all.index)
            if len(idx) < VAL_WINDOW + 200:
                continue
            X_h, y_h = X_all.loc[idx], y_all.loc[idx]
            f_tr, f_val = X_h.iloc[:-VAL_WINDOW], X_h.iloc[-VAL_WINDOW:]
            y_tr, y_val = y_h.iloc[:-VAL_WINDOW].values, y_h.iloc[-VAL_WINDOW:].values
            pv = ridge_fit(f_tr, y_tr)(f_val)
            d = float(np.dot(pv, pv))
            a = float(np.clip(np.dot(pv, y_val) / d, 0.0, 1.0)) if d > 1e-12 else 0.0
            Xnow = featurize(r_hist).iloc[-1:].fillna(0.0)
            corr = float(ridge_fit(X_h, y_h)(Xnow.values)[0])
            act = float(te.iloc[origin + H - 1])
            sar = float(fc[H - 1])
            best = min(GRID, key=lambda g: (act - (sar + g * corr)) ** 2)
            rows.append({'target': feat, 'variant': vname, 'fold': off,
                         'origin': str(te.index[origin].date()), 'actual': act, 'sarima': sar,
                         'corr': corr, 'alpha': a, 'recipe': sar + a * corr, 'full': sar + corr,
                         'oracle_alpha': float(best), 'oracle': sar + float(best) * corr,
                         'e_sarima': act - sar, 'e_recipe': act - (sar + a * corr)})
    return pd.DataFrame(rows)


def main():
    all_data = load_all()
    frames = []
    for feat in ['PM25', 'NO2']:
        for v in VARIANTS:
            df = run(all_data, feat, v)
            frames.append(df); print(f'{feat} {v}: {len(df)} titik ujian', flush=True)
    big = pd.concat(frames, ignore_index=True)
    big.to_csv(OUT, index=False)

    def r_(a, p):
        return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))
    print('\n===== RMSE jarak 7 hari =====')
    for (feat, v), d in big.groupby(['target', 'variant']):
        print(f'{feat:5s} {v:16s} SARIMA {r_(d.actual, d.sarima):.4f} | resep {r_(d.actual, d.recipe):.4f} '
              f'| koreksi penuh {r_(d.actual, d.full):.4f} | seandainya bobot sempurna {r_(d.actual, d.oracle):.4f}')
    print('\n===== rincian per titik ujian =====')
    for (feat, v), d in big.groupby(['target', 'variant']):
        harm = d.e_recipe.abs() - d.e_sarima.abs()
        print(f'{feat:5s} {v:16s} membaik {int((harm < 0).sum())}/{len(d)} | alfa rata2 {d.alpha.mean():.2f} '
              f'| alfa=0 kalau {int((d.alpha < 0.05).sum())} titik | korelasi(alfa, kerugian) {np.corrcoef(d.alpha, harm)[0, 1]:+.2f} '
              f'| korelasi(|koreksi|, kerugian) {np.corrcoef(d['corr'].abs(), harm)[0, 1]:+.2f}')
        w = d.assign(harm=harm).nlargest(3, 'harm')[['origin', 'actual', 'sarima', 'corr', 'alpha', 'recipe', 'harm']]
        print(w.to_string(index=False))
    print('\n===== kesimpulan otomatis =====')
    for (feat, v), d in big.groupby(['target', 'variant']):
        rec, orc = r_(d.actual, d.recipe), r_(d.actual, d.oracle)
        print(f'{feat:5s} {v:16s}: celah ke batas terbaik = {(rec - orc) / rec * 100:.1f}% '
              f'(resep {rec:.4f} vs batas {orc:.4f})')


if __name__ == '__main__':
    main()