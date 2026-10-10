"""Adu final vs BiLSTM pada protokol yang sama, memakai varian musiman mingguan.

Memakai berkas prediksi tersimpan (tanpa latih ulang):
- BiLSTM          : results/01_baseline_horizon_predictions.csv (fold-0, hasil reproduksi)
- varian A/B/C    : results/08_weekly_metrics_predictions.csv
Semua arm dihitung pada titik ujian yang identik (diverifikasi: selisih nilai aktual = 0).
"""
import os
import numpy as np, pandas as pd
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), 'results')
OUT = os.path.join(R, '09_vs_bilstm_weekly.csv')
CASES = [('B_mingguan_s7', 'recipe', 'B: mingguan s=7 + resep'),
         ('B_mingguan_s7', 'sarima', 'B: mingguan s=7, SARIMA saja'),
         ('C_paper_fourier', 'recipe', 'C: paper + Fourier mingguan + resep'),
         ('A_paper_s12', 'recipe', 'A: paper (s=12) + resep')]


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
    return float(2 * (1 - tdist.cdf(abs(stat), n - 1)))


def main():
    big = pd.read_csv(os.path.join(R, '08_weekly_metrics_predictions.csv'))
    big['od'] = big.origin_date.astype(str)
    bil = pd.read_csv(os.path.join(R, '01_baseline_horizon_predictions.csv'))
    bil['od'] = bil.origin_date.astype(str)

    rows = []
    for h in (1, 7, 30):
        b = bil[bil.horizon == h][['od', 'actual', 'hybrid']].rename(columns={'actual': 'a_bil', 'hybrid': 'bil'})
        for vname, arm, label in CASES:
            col = f'{arm}_h{h}'
            d = big[(big.fold == 0) & (big.feature == 'NO2') & (big.variant == vname)][['od', f'actual_h{h}', col]]
            d = d.rename(columns={f'actual_h{h}': 'a_our', col: 'p_our'}).dropna()
            j = b.merge(d, on='od')
            drift = float(np.max(np.abs(j.a_bil - j.a_our)))
            assert drift < 1e-9, f'titik ujian tidak identik (drift {drift:.2e})'
            r_our, r_bil = rmse(j.a_bil, j.p_our), rmse(j.a_bil, j.bil)
            rows.append({'horizon': h, 'arm': label, 'rmse_kita': r_our, 'rmse_bilstm': r_bil,
                         'delta_pct_vs_bilstm': (r_our - r_bil) / r_bil * 100,
                         'mape_kita': float(np.mean(np.abs((j.a_bil - j.p_our) / j.a_bil)) * 100),
                         'dm_p_vs_bilstm': dm(j.a_bil - j.bil, j.a_bil - j.p_our, h), 'n': len(j)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print('\n--- ringkas: satu resep (B: mingguan s=7 + koreksi berbobot) ---')
    r = out[out.arm == 'B: mingguan s=7 + resep']
    for _, x in r.iterrows():
        print(f"  h={x.horizon:2d}: {x.rmse_kita:.4f} vs BiLSTM {x.rmse_bilstm:.4f} "
              f"({x.delta_pct_vs_bilstm:+.1f}% | p={x.dm_p_vs_bilstm:.3f})")
    print(f"sel menang vs BiLSTM: {int((r.delta_pct_vs_bilstm < 0).sum())}/{len(r)}")


if __name__ == '__main__':
    main()