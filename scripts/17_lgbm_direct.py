"""Fase 17 — Desain B: LightGBM telanjang (TANPA SARIMA).

Pertanyaan ablasion: jika LGBM meramal langsung dari nol (lag, kalender, tahun-lalu),
apakah kerangka SARIMA masih dibutuhkan sama sekali?

Protokol identik fase 16:
    Box-Cox, 5 lipatan (ujian 720 hari, geser 0/120/240/360/480), origin tiap 30 hari,
    horizon 1/7/30, fitur = lag1-7, roll7/30, std30, lag365, Fourier tahunan, dow.
Perbedaan: model meramal LEVEL (bukan residu), dilatih ulang per origin dgn jendela
melebar (setara cara SARIMA mencerna sejarah baru), tanpa gerbang (tak ada yang ditimbang).
Seed LightGBM 42, n_jobs=1, deterministic.
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import boxcox, t as tdist
from lightgbm import LGBMRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data', 'transformed_data.csv')
R = os.path.join(ROOT, 'results')
OUT_P = os.path.join(R, '17_lgbm_direct_predictions.csv')
OUT_M = os.path.join(R, '17_lgbm_direct_metrics.csv')
OUT_D = os.path.join(R, '17_lgbm_direct_dm.csv')
OUT_MD = os.path.join(ROOT, '03_Catatan', 'HASIL_LGBM_DIRECT.md')

FEATURES = ['NO2', 'PM10', 'PM25']
FOLD_OFFSETS = [0, 120, 240, 360, 480]
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
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
    series = pd.concat([tr, te])
    X_all = featurize(series)
    pos = {d: i for i, d in enumerate(series.index)}
    rows = []
    for origin in range(0, len(te) - min(HORIZONS), ORIGIN_STEP):
        t0 = te.index[origin]
        cut = pos[t0]                                             # s.d. hari ini saja
        hist = series.iloc[:cut + 1]                               # sejarah yg "sudah terjadi"
        rec = {'feature': feat, 'fold': offset, 'origin_date': str(t0.date())}
        for h in HORIZONS:
            if origin >= len(te) - h:
                continue
            y_all = hist.shift(-h)                                 # label murni masa lalu (t+h <= t0), tanpa bocor
            valid = X_all.iloc[:cut + 1].dropna().index.intersection(y_all.dropna().index)
            if len(valid) < 320:
                continue
            rec[f'actual_h{h}'] = float(te.iloc[origin + h - 1])
            m = LGBMRegressor(**LGBM_KW)
            m.fit(X_all.loc[valid], y_all.loc[valid])
            rec[f'lgbmdir_h{h}'] = float(m.predict(X_all.loc[[t0]])[0])
        if any(k.startswith('lgbmdir') for k in rec):
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    all_data = load_all()
    print('=== fase 17: LightGBM telanjang tanpa SARIMA (5 lipatan) ===', flush=True)
    frames = []
    for feat in FEATURES:
        for off in FOLD_OFFSETS:
            df = run_fold(all_data, feat, off)
            if not df.empty:
                frames.append(df)
                print(f'  {feat} fold+{off}: {len(df)} origin', flush=True)
    big = pd.concat(frames, ignore_index=True)
    big.to_csv(OUT_P, index=False)

    path_a = os.path.join(R, '16_lgbm_gate_predictions.csv')
    pa = pd.read_csv(path_a) if os.path.exists(path_a) else pd.DataFrame()
    if pa.empty:
        print('fase 16 belum selesai — tabel adu gabungan dilewati (jalankan ulang setelahnya)')
        return
    key = ['feature', 'fold', 'origin_date']
    bring = [c for c in pa.columns if c.split('_h')[0] in ('sarima', 'ridge', 'lgbm')]
    merged = big.merge(pa[key + bring], on=key, how='inner')

    arms = [('SARIMA', 'sarima'), ('ridge_gate', 'ridge'), ('lgbm_gate', 'lgbm'),
            ('lgbm_direct', 'lgbmdir')]
    rows, dm_rows = [], []
    for feat in FEATURES:
        for h in HORIZONS:
            ac = f'actual_h{h}'
            d = merged[(merged.feature == feat) & merged[ac].notna()].copy()
            d = d[d[f'lgbmdir_h{h}'].notna()]
            rms = {}
            for label, col in arms:
                r_ = rmse(d[ac], d[f'{col}_h{h}'])
                rms[label] = r_
                rows.append({'target': feat, 'horizon': h, 'model': label, 'rmse': r_,
                             'mae': float(np.mean(np.abs(d[ac] - d[f'{col}_h{h}']))), 'n': len(d)})
            base_err = d[ac].values - d[f'lgbmdir_h{h}'].values
            other_map = {'sarima': 'SARIMA', 'ridge': 'ridge_gate', 'lgbm': 'lgbm_gate'}
            for col, label in other_map.items():
                other_err = d[ac].values - d[f'{col}_h{h}'].values
                p_ = dm(other_err, base_err, h)[1]
                dm_rows.append({'target': feat, 'horizon': h, 'vs': label,
                                'delta_pct': (rms[label] - rms['lgbm_direct']) / rms[label] * 100,
                                'dm_p': p_})
    s, t = pd.DataFrame(rows), pd.DataFrame(dm_rows)
    s.to_csv(OUT_M, index=False); t.to_csv(OUT_D, index=False)
    print('\n===== TABEL (tanpa SARIMA vs kerangka penuh) =====')
    print(s.pivot(index=['target', 'horizon'], columns='model', values='rmse')[['SARIMA', 'ridge_gate', 'lgbm_gate', 'lgbm_direct']].round(4).to_string())
    print('\n===== DM: lgbm_direct vs lengan lain =====')
    print(t.round(4).to_string(index=False))

    lines = ['# Hasil fase 17 — LightGBM telanjang (tanpa SARIMA)', '',
             'Ablasion: LGBM meramal level langsung; fitur sama dg penambal fase 16;',
             'dilatih ulang per origin (jendela melebar); tanpa gerbang.', '',
             '| polutan | h | SARIMA | Ridge gate | LGBM gate | LGBM langsung |', '|---|---|---|---|---|---|']
    piv = s.pivot(index=['target', 'horizon'], columns='model', values='rmse')
    for (feat, h), x in piv.iterrows():
        lines.append(f'| {feat} | {h} | {x["SARIMA"]:.4f} | {x["ridge_gate"]:.4f} | '
                     f'{x["lgbm_gate"]:.4f} | {x["lgbm_direct"]:.4f} |')
    nlgbm = int(((t.vs == 'SARIMA') & (t.dm_p < 0.05) & (t.delta_pct > 0)).sum())
    lines += ['', f'Sel lgbm_direct nyata vs SARIMA (p<0,05): {nlgbm}/9. Lampiran: 17_lgbm_direct_*.csv']
    open(OUT_MD, 'w').write('\n'.join(lines) + '\n')
    print(f'\ncatatan: {OUT_MD}')


if __name__ == '__main__':
    main()
