"""Fase 16 — ganti penambal: Ridge vs LightGBM (gerbang alfa identik, protokol resep beku).

Pertanyaan eksperimen: kalau penambal koreksi SARIMA diganti dari Ridge (15 param)
ke LightGBM (ML boosting modern), apakah akurasi naik di data kecil (1.826 hari)?

Desain (salinan 12_full_final.py, hanya babak koreksi yang bertambah):
    SARIMA (2,1,1)(1,0,1,7)  — dasar sama persis
    + koreksi Ridge          — resep beku (registrasi ulang: harus cocok dgn 12_final_metrics)
    + koreksi LightGBM       — fitur & gerbang alfa identik dgn Ridge
    final = base + alfa * koreksi ; alfa dari proyeksi validasi 120 hari tiap model sendiri.
Lipatan: 5 (720 hari, geser 0/120/240/360/480). Seed LightGBM dipatok (deterministik).
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import boxcox, t as tdist
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'transformed_data.csv')
R = os.path.join(ROOT, 'results')
OUT_P = os.path.join(R, '16_lgbm_gate_predictions.csv')
OUT_M = os.path.join(R, '16_lgbm_gate_metrics.csv')
OUT_D = os.path.join(R, '16_lgbm_gate_dm.csv')
OUT_MD = os.path.join(ROOT, '03_Catatan', 'HASIL_LGBM.md')

FEATURES = ['NO2', 'PM10', 'PM25']
FOLD_OFFSETS = [0, 120, 240, 360, 480]
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
ORDER, SEASONAL = (2, 1, 1), (1, 0, 1, 7)
LAGS, MIN_TRAIN = [1, 2, 3, 4, 5, 6, 7], 400
LGBM_KW = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
               random_state=42, n_jobs=1, verbosity=-1, deterministic=True)


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


def lgbm_fit(X, y):
    m = LGBMRegressor(**LGBM_KW)
    m.fit(X, y)
    return lambda Z: m.predict(Z)


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


def gate_alpha(p120, yv_tail):
    d120 = float(np.dot(p120, p120))
    return float(np.clip(np.dot(p120, yv_tail) / d120, 0.0, 1.0)) if d120 > 1e-12 else 0.0


def run_fold(all_data, feat, offset):
    end = len(all_data) - offset
    start = end - TEST_SIZE_DAYS
    tr, te = all_data[feat].iloc[:start], all_data[feat].iloc[start:end]
    if len(tr) < MIN_TRAIN:
        return pd.DataFrame()
    m = SARIMAX(tr, order=ORDER, seasonal_order=SEASONAL).fit(disp=False)
    rows = []
    for origin in range(0, len(te) - min(HORIZONS), ORIGIN_STEP):
        hist = pd.concat([tr, te.iloc[:origin]])
        res = m.apply(hist)
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
            if len(idx) < 320:
                continue
            X_h, y_h = X_all.loc[idx], y_all.loc[idx]
            yv = y_h.values
            # --- Ridge (resep beku) ---
            p120_r = ridge_fit(X_h.iloc[:-120], yv[:-120])(X_h.iloc[-120:])
            ar = gate_alpha(p120_r, yv[-120:])
            corr_r = float(ridge_fit(X_h, y_h)(Xnow.values)[0])
            rec[f'ridge_h{h}'] = base + ar * corr_r
            rec[f'alpha_ridge_h{h}'] = ar
            # --- LightGBM (calon baru) ---
            p120_l = lgbm_fit(X_h.iloc[:-120], yv[:-120])(X_h.iloc[-120:])
            al = gate_alpha(p120_l, yv[-120:])
            corr_l = float(lgbm_fit(X_h, y_h)(Xnow.values)[0])
            rec[f'lgbm_h{h}'] = base + al * corr_l
            rec[f'alpha_lgbm_h{h}'] = al
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    all_data = load_all()
    print('=== fase 16: Ridge vs LightGBM (5 lipatan) ===', flush=True)
    frames = []
    for feat in FEATURES:
        for off in FOLD_OFFSETS:
            df = run_fold(all_data, feat, off)
            if not df.empty:
                frames.append(df)
                print(f'  {feat} fold+{off}: {len(df)} origin', flush=True)
    big = pd.concat(frames, ignore_index=True)
    big.to_csv(OUT_P, index=False)

    rows, dm_rows = [], []
    ARMS = [('SARIMA', 'sarima'), ('ridge_gate', 'ridge'), ('lgbm_gate', 'lgbm')]
    for feat in FEATURES:
        for h in HORIZONS:
            ac = f'actual_h{h}'
            d = big[(big.feature == feat) & big[ac].notna()]
            dd = d[d[f'ridge_h{h}'].notna()]
            rms = {}
            for label, col in ARMS:
                sub = dd if label != 'SARIMA' else d
                r_ = rmse(sub[ac], sub[f'{col}_h{h}'])
                rms[label] = r_
                rows.append({'target': feat, 'horizon': h, 'model': label, 'rmse': r_,
                             'mae': float(np.mean(np.abs(sub[ac] - sub[f'{col}_h{h}']))),
                             'n': len(sub), 'n_folds': sub.fold.nunique()})
            dm_rows.append({'target': feat, 'horizon': h,
                            'delta_pct_lgbm_vs_ridge': (rms['lgbm_gate'] - rms['ridge_gate']) / rms['ridge_gate'] * 100,
                            'dm_p_lgbm_vs_ridge': dm(dd[ac].values - dd['ridge_h' + str(h)].values,
                                                     dd[ac].values - dd[f'lgbm_h{h}'].values, h)[1],
                            'delta_pct_lgbm_vs_sarima': (rms['lgbm_gate'] - rms['SARIMA']) / rms['SARIMA'] * 100})
    s, t = pd.DataFrame(rows), pd.DataFrame(dm_rows)
    s.to_csv(OUT_M, index=False); t.to_csv(OUT_D, index=False)
    print('\n===== TABEL =====')
    print(s.pivot(index=['target', 'horizon'], columns='model', values='rmse').round(4).to_string())
    print('\n===== LGBM vs menantu-kandung (Ridge) & SARIMA =====')
    print(t.round(4).to_string(index=False))

    # pemeriksaan reproduksi: lengan Ridge harus cocok dgn 12_final_metrics
    ref_path = os.path.join(R, '12_final_metrics.csv')
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        diffs = []
        for _, x in ref[ref.model == 'RESEP_FINAL'].iterrows():
            got = s[(s.target == x.target) & (s.horizon == x.horizon) & (s.model == 'ridge_gate')].rmse.iloc[0]
            diffs.append(abs(got - x.rmse))
        print(f"\nreproduksi Ridge vs fase-12: max selisih RMSE = {max(diffs):.6f}")

    win = int((t.delta_pct_lgbm_vs_ridge < 0).sum())
    alpha_mean = float(pd.concat([big[f'alpha_lgbm_h{h}'].dropna() for h in HORIZONS]).mean())
    lines = ['# Hasil fase 16 — LightGBM sebagai penambal (gerbang identik)', '',
             'Protokol resep beku: SARIMA (2,1,1)(1,0,1,7), 5 lipatan (720 hari, geser 120),',
             'gerbang alfa 120-hari per model. LightGBM: 200 pohon, lr 0.05, leaves 15, seed 42.', '',
             '| polutan | h | SARIMA | Ridge | LightGBM | Δ LGBM vs Ridge | p (DM) |', '|---|---|---|---|---|---|---|']
    for _, x in t.iterrows():
        g = s[(s.target == x.target) & (s.horizon == x.horizon)]
        b = g[g.model == 'SARIMA'].rmse.iloc[0]
        rr = g[g.model == 'ridge_gate'].rmse.iloc[0]
        ll = g[g.model == 'lgbm_gate'].rmse.iloc[0]
        lines.append(f'| {x.target} | {x.horizon} | {b:.4f} | {rr:.4f} | {ll:.4f} | '
                     f'{x.delta_pct_lgbm_vs_ridge:+.1f}% | {x.dm_p_lgbm_vs_ridge:.4f} |')
    lines += ['', f'Menang LGBM vs Ridge: {win}/{len(t)} sel; alfa LGBM rata-rata {alpha_mean:.2f}.',
              '', 'Lampiran: results/16_lgbm_gate_predictions.csv | 16_lgbm_gate_metrics.csv | 16_lgbm_gate_dm.csv']
    open(OUT_MD, 'w').write('\n'.join(lines) + '\n')
    print(f'\ncatatan: {OUT_MD}')


if __name__ == '__main__':
    main()
