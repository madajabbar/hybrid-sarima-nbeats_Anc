"""Fase 5 — satu resep tetap + uji Diebold-Mariano + banyak lipatan (permintaan dosen).

Resep tetap (dipakai di semua polutan & jarak):
    ramalan final = SARIMA + alfa * koreksi_ridge
    - koreksi: Ridge pada fitur residu (lag 1-7, rata 7/30, std 30, lag 365, Fourier tahunan, hari-minggu)
    - alfa: dipilih di jendela validasi 120 hari terakhir sebelum titik ujian (bukan di data ujian)

Lipatan (3): jendela ujian 720 hari digeser 180 hari ke belakang, sehingga titik ujian bertambah
dan klaim tidak bergantung pada satu periode saja. Fold-0 identik notebook 01 (train 1106 / test 720).

Uji: Diebold-Mariano (1995) dengan koreksi sampel kecil Harvey-Leybourne-Newbold dan
varians HAC (lag = h-1), plus interval keyakinan bootstrap atas selisih RMSE.

Skrip dijalankan di CPU. Rekaan - GBM (3 seed, di-rata) hanya di fold-0 sebagai pembanding.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import boxcox, t as tdist
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'transformed_data.csv')
OUT = os.path.join(ROOT, 'results', '05_final_recipe_metrics.csv')
OUT_DM = os.path.join(ROOT, 'results', '05_final_recipe_dm.csv')

FEATURES = ['NO2', 'PM10', 'PM25']
FOLD_OFFSETS = [0, 180, 360]        # geser jendela ujian ke belakang (hari)
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H = 720, 30, 30
HORIZONS = [1, 7, 30]
SARIMA_ORDER, SEASONAL_ORDER = (2, 1, 1), (0, 0, 0, 12)
VAL_WINDOW, LAGS, MIN_TRAIN = 120, [1, 2, 3, 4, 5, 6, 7], 400
REF_BILSTM_FOLD0 = {('NO2', 1): 0.591472967107877, ('NO2', 7): 0.760324, ('NO2', 30): 0.628948}


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


def gbm_fit(X, y, seed):
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=seed)
    m.fit(X, y)
    return lambda Z: m.predict(Z)


def alpha_val(pred, y):
    d = float(np.dot(pred, pred))
    return float(np.clip(np.dot(pred, y) / d, 0.0, 1.0)) if d > 1e-12 else 0.0


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.sqrt(np.mean((a - p) ** 2))), float(np.mean(np.abs(a - p))),
            float(np.mean(np.abs((a - p) / a)) * 100))


def dm_test(e1, e2, h):
    """Diebold-Mariano dua sisi, varians HAC (lag h-1), koreksi sampel kecil HLN."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    dbar = float(np.mean(d))
    g0 = float(np.mean((d - dbar) ** 2))
    v = g0
    for k in range(1, h):
        gk = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        v += 2 * gk
    if v <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(v / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return stat, float(2 * (1 - tdist.cdf(abs(stat), n - 1)))


def boot_ci(e1, e2, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    idx = rng.integers(0, n, size=(reps, n))
    diffs = np.sqrt(np.mean(d[idx], axis=1))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_fold(all_data, feat, offset, use_gbm=False):
    end = len(all_data) - offset
    start = end - TEST_SIZE_DAYS
    tr = all_data[feat].iloc[:start]
    te = all_data[feat].iloc[start:end]
    if len(tr) < MIN_TRAIN:
        return pd.DataFrame()
    sarima = SARIMAX(tr, order=SARIMA_ORDER, seasonal_order=SEASONAL_ORDER).fit(disp=False)
    rows = []
    origin_step = ORIGIN_STEP
    origin_range = range(0, len(te) - min(HORIZONS), origin_step)
    for origin in origin_range:
        h0 = 0  # setiap origin dipakai untuk semua horizon yang tersedia
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
            a = alpha_val(ridge_fit(f_tr, y_tr)(f_val), y_val)
            corr = float(ridge_fit(X_h, y_h)(Xnow.values)[0])
            rec[f'recipe_h{h}'] = rec[f'sarima_h{h}'] + a * corr
            rec[f'full_h{h}'] = rec[f'sarima_h{h}'] + corr
            rec[f'alpha_h{h}'] = a
            if use_gbm:
                cs = [float(gbm_fit(X_h, y_h, s)(Xnow.values)[0]) for s in (0, 1, 2)]
                rec[f'gbm_h{h}'] = rec[f'sarima_h{h}'] + np.mean(cs)
                rec[f'alpha_gbm_h{h}'] = alpha_val(np.mean([gbm_fit(f_tr, y_tr, s)(f_val) for s in (0, 1, 2)], axis=0), y_val)
                rec[f'gbmalpha_h{h}'] = rec[f'sarima_h{h}'] + rec[f'alpha_gbm_h{h}'] * np.mean(cs)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    all_data = load_all()
    frames = []
    for feat in FEATURES:
        for off in FOLD_OFFSETS:
            df = run_fold(all_data, feat, off, use_gbm=(off == 0))
            if df.empty:
                continue
            frames.append(df)
            print(f'{feat} fold+{off}: {len(df)} origin')
    big = pd.concat(frames, ignore_index=True)

    arms = {'SARIMA': 'sarima_h{}', 'RESEP_kita_ridge_alpha': 'recipe_h{}',
            'kontrol_koreksi_penuh': 'full_h{}'}
    if 'gbm_h1' in big.columns:
        arms['pembanding_gbm_3seed'] = 'gbm_h{}'
        arms['gbm_alpha'] = 'gbmalpha_h{}'

    summary, dm_rows = [], []
    for feat in FEATURES:
        for h in HORIZONS:
            ac, sc_ = f'actual_h{h}', f'sarima_h{h}'
            d = big[(big.feature == feat) & big[sc_].notna() & big[ac].notna()]
            if d.empty:
                continue
            base_rmse = metrics(d[ac], d[sc_])[0]
            for name, pat in arms.items():
                col = pat.format(h)
                if col not in d.columns or d[col].isna().all():
                    continue
                dd = d[d[col].notna()]
                rmse, mae, mape = metrics(dd[ac], dd[col])
                summary.append({'target': feat, 'horizon': h, 'model': name, 'rmse': rmse, 'mae': mae,
                                'mape': mape, 'delta_pct_vs_sarima': (rmse - base_rmse) / base_rmse * 100,
                                'n': len(dd), 'n_folds': dd.fold.nunique()})
                if col != sc_:
                    e1, e2 = dd[ac].values - dd[sc_].values, dd[ac].values - dd[col].values
                    stat, p = dm_test(e1, e2, h)
                    lo, hi = boot_ci(e1, e2)
                    dm_rows.append({'target': feat, 'horizon': h, 'model': name, 'rmse_delta': rmse - base_rmse,
                                    'dm_stat': stat, 'dm_pvalue': p, 'ci95_lo': lo, 'ci95_hi': hi, 'n': len(dd)})
    s, t = pd.DataFrame(summary), pd.DataFrame(dm_rows)
    s.to_csv(OUT, index=False); t.to_csv(OUT_DM, index=False)
    big.to_csv(OUT.replace('.csv', '_predictions.csv'), index=False)

    print('\n===== RESEP TETAP vs SARIMA (3 lipatan, titik ujian digabung) =====')
    print(s[s.model != 'kontrol_koreksi_penuh'].to_string(index=False))
    print('\n===== Diebold-Mariano =====')
    print(t.to_string(index=False))
    r = s[s.model == 'RESEP_kita_ridge_alpha']
    print('\n===== ringkasan resep =====')
    print(f"sel menang : {(r.delta_pct_vs_sarima < 0).sum()} / {len(r)}")
    print(f"rata-rata perbaikan RMSE: {r.delta_pct_vs_sarima.mean():+.2f}% | terbaik {r.delta_pct_vs_sarima.min():+.2f}%")
    sig = t[(t.model == 'RESEP_kita_ridge_alpha') & (t.dm_pvalue < 0.05)]
    print(f"signifikan (DM p<0,05): {len(sig)} sel -> " + ', '.join(f"{x.target} h={x.horizon}" for _, x in sig.iterrows()))
    for feat in ['NO2']:
        for h in HORIZONS:
            ref = REF_BILSTM_FOLD0.get((feat, h))
            row = r[(r.target == feat) & (r.horizon == h)]
            if ref and not row.empty:
                print(f"  NO2 h={h}: resep {row.rmse.iloc[0]:.4f} vs BiLSTM referensi {ref:.4f} "
                      f"({(row.rmse.iloc[0] - ref) / ref * 100:+.1f}%)")


if __name__ == '__main__':
    main()