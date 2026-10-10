"""Fase 10 — pelatihan penuh versi final (resep beku).

Resep beku:
    SARIMA (2,1,1)(1,0,1,7)                      <- musiman mingguan (temuan Fase 7)
    + KOREKSI BERBOBOT: alfa dari proyeksi validasi 120 hari (0-100%), bukan dari data ujian
      (gerbang 3 blok diuji di Fase 9 dan TIDAK diperlukan: tanpa gerbang sudah 8/9 sel menang,
       nol kekalahan nyata -- gerbang hanya membuang keuntungan)
    koreksi = Ridge pada fitur residu (lag 1-7, rata 7/30, std 30, lag 365,
              Fourier tahunan, hari-dalam-minggu)

Lipatan diperbanyak (5): jendela ujian 720 hari digeser 0/120/240/360/480 hari.
Keluaran: tabel metrik final + uji Diebold-Mariano + tabel siap-tempel markdown untuk naskah.
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
R = os.path.join(ROOT, 'results')
OUT_M = os.path.join(R, '12_final_metrics.csv')
OUT_D = os.path.join(R, '12_final_dm.csv')
OUT_MD = os.path.join(ROOT, '03_Catatan', 'HASIL_FINAL.md')

FEATURES = ['NO2', 'PM10', 'PM25']
FOLD_OFFSETS = [0, 120, 240, 360, 480]
TEST_SIZE_DAYS, ORIGIN_STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
ORDER, SEASONAL = (2, 1, 1), (1, 0, 1, 7)
ORDER_REF, SEASONAL_REF = (2, 1, 1), (0, 0, 0, 12)      # protokol paper, untuk pembanding
LAGS, MIN_TRAIN = [1, 2, 3, 4, 5, 6, 7], 400
REF_BILSTM_FOLD0 = {('NO2', 1): 0.591473, ('NO2', 7): 0.760324, ('NO2', 30): 0.628948}


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


def run_fold(all_data, feat, offset, order, seasonal, use_correction=True):
    end = len(all_data) - offset
    start = end - TEST_SIZE_DAYS
    tr, te = all_data[feat].iloc[:start], all_data[feat].iloc[start:end]
    if len(tr) < MIN_TRAIN:
        return pd.DataFrame()
    m = SARIMAX(tr, order=order, seasonal_order=seasonal).fit(disp=False)
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
            if not use_correction:
                continue
            y_all = r_hist.shift(-h).dropna()
            X_all = featurize(r_hist)
            idx = X_all.dropna().index.intersection(y_all.index)
            if len(idx) < 320:
                continue
            X_h, y_h = X_all.loc[idx], y_all.loc[idx]
            yv = y_h.values
            p120 = ridge_fit(X_h.iloc[:-120], yv[:-120])(X_h.iloc[-120:])
            d120 = float(np.dot(p120, p120))
            a = float(np.clip(np.dot(p120, yv[-120:]) / d120, 0.0, 1.0)) if d120 > 1e-12 else 0.0
            corr = float(ridge_fit(X_h, y_h)(Xnow.values)[0])
            rec[f'final_h{h}'] = base + a * corr
            rec[f'alpha_h{h}'] = a
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    all_data = load_all()
    print('=== resep final (5 lipatan) ===', flush=True)
    frames = []
    for feat in FEATURES:
        for off in FOLD_OFFSETS:
            df = run_fold(all_data, feat, off, ORDER, SEASONAL)
            if not df.empty:
                frames.append(df)
                print(f'  {feat} fold+{off}: {len(df)} origin', flush=True)
    big = pd.concat(frames, ignore_index=True)
    big.to_csv(os.path.join(R, '12_final_predictions.csv'), index=False)

    print('=== pembanding: protokol paper (musiman 12), SARIMA saja, 5 lipatan ===', flush=True)
    ref_frames = []
    for feat in ['NO2']:
        for off in FOLD_OFFSETS:
            df = run_fold(all_data, feat, off, ORDER_REF, SEASONAL_REF, use_correction=False)
            if not df.empty:
                ref_frames.append(df)
    ref = pd.concat(ref_frames, ignore_index=True) if ref_frames else pd.DataFrame()
    if not ref.empty:
        ref.to_csv(os.path.join(R, '12_final_reference_predictions.csv'), index=False)

    rows, dm_rows = [], []
    for feat in FEATURES:
        for h in HORIZONS:
            ac = f'actual_h{h}'
            d = big[(big.feature == feat) & big[ac].notna()]
            b_sar = rmse(d[ac], d[f'sarima_h{h}'])
            dd = d[d[f'final_h{h}'].notna()]
            f_rmse = rmse(dd[ac], dd[f'final_h{h}'])
            rows.append({'target': feat, 'horizon': h, 'model': 'SARIMA_mingguan', 'rmse': b_sar,
                         'mae': float(np.mean(np.abs(d[ac] - d[f'sarima_h{h}']))),
                         'mape': float(np.mean(np.abs((d[ac] - d[f'sarima_h{h}']) / d[ac])) * 100),
                         'n': len(d), 'n_folds': d.fold.nunique()})
            rows.append({'target': feat, 'horizon': h, 'model': 'RESEP_FINAL', 'rmse': f_rmse,
                         'mae': float(np.mean(np.abs(dd[ac] - dd[f'final_h{h}']))),
                         'mape': float(np.mean(np.abs((dd[ac] - dd[f'final_h{h}']) / dd[ac])) * 100),
                         'n': len(dd), 'n_folds': dd.fold.nunique()})
            _, p = dm(dd[ac].values - dd[f'sarima_h{h}'].values, dd[ac].values - dd[f'final_h{h}'].values, h)
            dm_rows.append({'target': feat, 'horizon': h, 'delta_pct': (f_rmse - b_sar) / b_sar * 100,
                            'dm_p': p, 'nyata05': 'ya' if (p == p and p < 0.05) else 'tidak',
                            'n': len(dd), 'n_folds': dd.fold.nunique(),
                            'alpha_mean': float(dd[f'alpha_h{h}'].mean())})
    s, t = pd.DataFrame(rows), pd.DataFrame(dm_rows)
    s.to_csv(OUT_M, index=False); t.to_csv(OUT_D, index=False)
    print('\n===== TABEL FINAL =====')
    print(s.to_string(index=False))
    print('\n===== RESEP FINAL vs SARIMA mingguan =====')
    print(t.to_string(index=False))

    lines = ['# Hasil final (resep beku)', '', 'SARIMA (2,1,1)(1,0,1,7) + koreksi Ridge bergerbang.',
             f'Lipatan: {len(FOLD_OFFSETS)} (jendela ujian 720 hari, geser 120 hari).', '',
             '| polutan | h | SARIMA mingguan | RESEP FINAL | Δ | p (DM) | alfa rata2 |',
             '|---|---|---|---|---|---|---|']
    for _, x in t.iterrows():
        b = s[(s.target == x.target) & (s.horizon == x.horizon) & (s.model == 'SARIMA_mingguan')].rmse.iloc[0]
        f_ = s[(s.target == x.target) & (s.horizon == x.horizon) & (s.model == 'RESEP_FINAL')].rmse.iloc[0]
        lines.append(f'| {x.target} | {x.horizon} | {b:.4f} | {f_:.4f} | {x.delta_pct:+.1f}% | '
                     f'{x.dm_p:.4f} | {x.alpha_mean:.2f} |')
    if not ref.empty:
        lines += ['', '## Pembanding protokol paper (SARIMA musiman 12, NO2, 5 lipatan)', '',
                  '| h | SARIMA musiman 12 | SARIMA mingguan | resep final |', '|---|---|---|---|']
        for h in HORIZONS:
            ac = f'actual_h{h}'
            rr = ref[ref[ac].notna()]
            b12 = rmse(rr[ac], rr[f'sarima_h{h}'])
            b7 = s[(s.target == 'NO2') & (s.horizon == h) & (s.model == 'SARIMA_mingguan')].rmse.iloc[0]
            ff = s[(s.target == 'NO2') & (s.horizon == h) & (s.model == 'RESEP_FINAL')].rmse.iloc[0]
            bl = REF_BILSTM_FOLD0.get(('NO2', h))
            lines.append(f'| {h} | {b12:.4f} | {b7:.4f} | {ff:.4f} |')
        lines += ['', 'Catatan: angka BiLSTM paper (fold-0 saja): '
                  + ', '.join(f'h{h}={v:.4f}' for (f_, h), v in REF_BILSTM_FOLD0.items())
                  + ' — perbandingan langsung harus dihitung pada lipatan yang sama.']
    open(OUT_MD, 'w').write('\n'.join(lines) + '\n')
    print(f'\ntabel markdown: {OUT_MD}')
    r = t
    print(f"\nmenang {int((r.delta_pct < 0).sum())}/{len(r)} sel | rata2 {r.delta_pct.mean():+.2f}% | "
          f"terbaik {r.delta_pct.min():+.2f}% | terburuk {r.delta_pct.max():+.2f}% | "
          f"menang nyata {int(((r.nyata05 == 'ya') & (r.delta_pct < 0)).sum())} | "
          f"kalah nyata {int(((r.nyata05 == 'ya') & (r.delta_pct > 0)).sum())}")


if __name__ == '__main__':
    main()