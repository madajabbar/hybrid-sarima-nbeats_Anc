"""Perbaiki uji signifikansi Fase 5 (tanpa melatih ulang) — hitung dari berkas prediksi tersimpan.

Dua perbaikan atas skrip 05:
1. Interval bootstrap dihitung pada selisih RMSE (bukan akar dari rata-rata selisih kuadrat —
   cara lama menghasilkan NaN saat perbaikan negatif).
2. Ringkasan signifikansi dipisah: menang nyata vs kalah nyata (arah diperhatikan).
"""
import os
import numpy as np, pandas as pd
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRED = os.path.join(ROOT, 'results', '05_final_recipe_metrics_predictions.csv')
OUT = os.path.join(ROOT, 'results', '05_final_recipe_dm.csv')
HORIZONS = [1, 7, 30]
ARMS = {'RESEP_kita_ridge_alpha': 'recipe_h{}', 'kontrol_koreksi_penuh': 'full_h{}',
        'pembanding_gbm_3seed': 'gbm_h{}', 'gbm_alpha': 'gbmalpha_h{}'}


def dm_test(e1, e2, h):
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    dbar = float(np.mean(d))
    v = float(np.mean((d - dbar) ** 2))
    for k in range(1, h):
        v += 2 * float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
    if v <= 0:
        return np.nan, np.nan
    stat = (dbar / np.sqrt(v / n)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, float(2 * (1 - tdist.cdf(abs(stat), n - 1)))


def rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))


def boot_ci(e1, e2, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    n = len(e1)
    idx = rng.integers(0, n, size=(reps, n))
    d = np.sqrt(np.mean(e1[idx] ** 2, axis=1)) - np.sqrt(np.mean(e2[idx] ** 2, axis=1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    big = pd.read_csv(PRED)
    rows = []
    for feat in big.feature.unique():
        for h in HORIZONS:
            ac, sc_ = f'actual_h{h}', f'sarima_h{h}'
            d = big[(big.feature == feat) & big[ac].notna() & big[sc_].notna()]
            base = rmse(d[ac], d[sc_])
            for name, pat in ARMS.items():
                col = pat.format(h)
                if col not in d.columns:
                    continue
                dd = d[d[col].notna()]
                if dd.empty:
                    continue
                got = rmse(dd[ac], dd[col])
                e1, e2 = dd[ac].values - dd[sc_].values, dd[ac].values - dd[col].values
                stat, p = dm_test(e1, e2, h)
                lo, hi = boot_ci(e1, e2)
                rows.append({'target': feat, 'horizon': h, 'model': name, 'n': len(dd),
                             'n_folds': dd.fold.nunique(), 'rmse_sarima': base, 'rmse_arm': got,
                             'delta_pct': (got - base) / base * 100, 'dm_stat': stat, 'dm_pvalue': p,
                             'ci95_lo_pct': lo / base * 100, 'ci95_hi_pct': hi / base * 100,
                             'arah': 'menang' if got < base else ('kalah' if got > base else 'seri'),
                             'nyata05': 'ya' if (p == p and p < 0.05) else 'tidak'})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    rec = out[out.model == 'RESEP_kita_ridge_alpha']
    print('\n--- RESEP_kita (ridge + alfa adaptif) ---')
    print(f"menang {int((rec.arah == 'menang').sum())}/{len(rec)} sel | rata-rata {rec.delta_pct.mean():+.2f}% "
          f"| terbaik {rec.delta_pct.min():+.2f}% | terburuk {rec.delta_pct.max():+.2f}%")
    w = rec[(rec['nyata05'] == 'ya') & (rec.arah == 'menang')]
    l = rec[(rec['nyata05'] == 'ya') & (rec.arah == 'kalah')]
    print('menang nyata :', ', '.join(f'{r.target} h={r.horizon} ({r.delta_pct:+.1f}%)' for _, r in w.iterrows()) or '-')
    print('kalah nyata  :', ', '.join(f'{r.target} h={r.horizon} ({r.delta_pct:+.1f}%)' for _, r in l.iterrows()) or '-')
    ctrl = out[out.model == 'kontrol_koreksi_penuh']
    print('\n--- kontrol koreksi penuh (alfa=1, cara paper) ---')
    for _, r in ctrl[ctrl['nyata05'] == 'ya'].iterrows():
        print(f"  {r.target} h={r.horizon}: {r.delta_pct:+.1f}% nyata ({r.arah})")


if __name__ == '__main__':
    main()