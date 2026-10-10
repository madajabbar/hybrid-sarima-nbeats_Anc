"""Adu langsung dengan BiLSTM pada protokol yang sama (lipatan pertama).

Memakai berkas prediksi yang sudah ada — tidak ada pelatihan ulang:
- BiLSTM   : results/01_baseline_horizon_predictions.csv (hasil reproduksi notebook 01, per origin, fold-0)
- Resep kita: results/05_final_recipe_metrics_predictions.csv (fold 0)

Semua arm dihitung pada titik ujian yang sama, jadi perbandingannya apple-to-apple.
Uji: Diebold-Mariano vs SARIMA dan vs BiLSTM.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, 'results')
OUT = os.path.join(R, '07_vs_bilstm.csv')
HORIZONS = [1, 7, 30]


def dm(e1, e2, h):
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    dbar = float(np.mean(d))
    g0 = v = float(np.mean((d - dbar) ** 2))
    for k in range(1, min(h - 1, max(1, n // 4)) + 1):   # lag dibatasi: n kecil (h=30, n=23)
        v += 2 * float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
    if not np.isfinite(v) or v <= 0:
        v = g0          # HAC bisa negatif di n kecil -> pakai varians tanpa autokorelasi
    stat = (dbar / np.sqrt(v / n)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, float(2 * (1 - tdist.cdf(abs(stat), n - 1)))


def metr(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.sqrt(np.mean((a - p) ** 2))), float(np.mean(np.abs(a - p))),
            float(np.mean(np.abs((a - p) / a)) * 100))


def main():
    bil = pd.read_csv(os.path.join(R, '01_baseline_horizon_predictions.csv'))
    our = pd.read_csv(os.path.join(R, '05_final_recipe_metrics_predictions.csv'))
    our = our[(our.fold == 0) & (our.feature == 'NO2')].copy()   # BiLSTM hanya tersedia untuk NO2
    our['origin_date'] = our.origin_date.astype(str)
    bil['origin_date'] = bil.origin_date.astype(str)

    rows = []
    for h in HORIZONS:
        b = bil[bil.horizon == h][['origin_date', 'actual', 'sarima', 'hybrid']].rename(
            columns={'actual': 'a_bil', 'sarima': 's_bil', 'hybrid': 'bilstm'})
        o = our[['origin_date', f'actual_h{h}', f'sarima_h{h}', f'recipe_h{h}', f'full_h{h}']].rename(
            columns={f'actual_h{h}': 'a_our', f'sarima_h{h}': 's_our', f'recipe_h{h}': 'recipe',
                     f'full_h{h}': 'full'})
        m = b.merge(o, on='origin_date', how='inner')
        # konsistensi protokol: actual SARIMA harus sama
        drift = float(np.max(np.abs(m.a_bil - m.a_our)))
        print(f'h={h}: {len(m)} origin dipadankan | selisih maksimum nilai aktual = {drift:.2e}')
        assert drift < 1e-6, 'data aktual tidak sama -> protokol tidak sebanding'
        base = m.a_bil.values
        arms = {'SARIMA': m.s_bil.values, 'SARIMA+BiLSTM(paper)': m.bilstm.values,
                'resep_kita': m.recipe.values, 'koreksi_penuh': m.full.values}
        rm = {k: metr(base, v) for k, v in arms.items()}
        for k, (r_, ma, mp) in rm.items():
            row = {'horizon': h, 'model': k, 'rmse': r_, 'mae': ma, 'mape': mp, 'n': len(m)}
            for ref, col in [('sarima', 'SARIMA'), ('bilstm', 'SARIMA+BiLSTM(paper)')]:
                if k == col:
                    row[f'dm_p_vs_{ref}'] = np.nan
                    row[f'delta_pct_vs_{ref}'] = 0.0
                    continue
                e1 = base - arms[col]
                e2 = base - arms[k]
                _, p = dm(e1, e2, h)
                row[f'delta_pct_vs_{ref}'] = (r_ - rm[col][0]) / rm[col][0] * 100
                row[f'dm_p_vs_{ref}'] = p
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print()
    print(out.to_string(index=False))
    print('\n--- ringkas: resep kita vs BiLSTM paper (protokol sama, fold-0) ---')
    r = out[out.model == 'resep_kita']
    for _, x in r.iterrows():
        print(f"  h={x.horizon:2d}: RMSE {x.rmse:.4f} vs BiLSTM {out[(out.horizon == x.horizon) & (out.model == 'SARIMA+BiLSTM(paper)')].rmse.iloc[0]:.4f} "
              f"({x.delta_pct_vs_bilstm:+.1f}%, p={x.dm_p_vs_bilstm:.3f}) | vs SARIMA {x.delta_pct_vs_sarima:+.1f}% (p={x.dm_p_vs_sarima:.3f})")
    print(f"sel menang vs BiLSTM: {int((r.delta_pct_vs_bilstm < 0).sum())}/{len(r)}")


if __name__ == '__main__':
    main()