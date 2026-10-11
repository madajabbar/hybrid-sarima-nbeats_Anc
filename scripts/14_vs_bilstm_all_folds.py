"""Fase 11b — adu murni: resep kita vs metode paper (SARIMA+BiLSTM) pada lipatan & origin identik.

Masukan:
    results/12_final_predictions.csv            (resep kita: SARIMA mingguan + Ridge bergerbang)
    results/13_bilstm_all_folds_predictions.csv (lengan paper: SARIMA(0,0,0,12) + BiLSTM, fase 11)

Keluaran:
    results/14_vs_bilstm_all_folds.csv          tabel metrik 4 lengan
    results/15_vs_bilstm_all_folds_dm.csv       uji Diebold-Mariano
    03_Catatan/HASIL_VS_BILSTM.md               tabel siap tempel untuk naskah

Pemakaian:
    python scripts/14_vs_bilstm_all_folds.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import t as tdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OURS = os.path.join(R, "12_final_predictions.csv")
PAPER = os.path.join(R, "13_bilstm_all_folds_predictions.csv")
OUT_M = os.path.join(R, "14_vs_bilstm_all_folds.csv")
OUT_D = os.path.join(R, "15_vs_bilstm_all_folds_dm.csv")
OUT_MD = os.path.join(ROOT, "03_Catatan", "HASIL_VS_BILSTM.md")
HORIZONS = [1, 7, 30]


def rmse(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def dm(e1, e2, h):
    """Diebold-Mariano dengan koreksi sampel kecil (sama seperti scripts/12)."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    dbar = float(np.mean(d))
    g0 = v = float(np.mean((d - dbar) ** 2))
    for k in range(1, min(h - 1, max(1, n // 4)) + 1):
        v += 2 * float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
    if not np.isfinite(v) or v <= 0:
        v = g0
    stat = (dbar / np.sqrt(v / n)) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - tdist.cdf(abs(stat), n - 1)))


def main(ours_path=OURS, paper_path=PAPER, out_m=OUT_M, out_d=OUT_D, out_md=OUT_MD):
    if not os.path.exists(paper_path):
        print(f"BELUM ADA: {paper_path}\n")
        print("Lengan BiLSTM semua lipatan belum dihitung. Jalankan dulu (butuh GPU):")
        print("    python scripts/13_bilstm_all_folds.py --tune off     # ±1 jam di T4")
        print("lalu ulangi skrip ini.")
        return 0
    ours = pd.read_csv(ours_path)
    paper = pd.read_csv(paper_path)
    key = ["feature", "fold", "origin_date"]
    pcols = key + ["bilstm_corr"] + [c for c in paper.columns
                                     if c.startswith(("actual_h", "sarima_h", "hybrid_h"))]
    d = ours.merge(paper[pcols], on=key, how="inner", suffixes=("", "_p"))
    print(f"origin kita {len(ours)} | origin paper {len(paper)} | pasangan cocok {len(d)}")
    if len(d) < len(ours):
        miss = ours.merge(paper[key], on=key, how="left", indicator=True)
        print(f"  catatan: {int((miss._merge=='left_only').sum())} origin kita belum ada padanannya")

    recs, dm_recs = [], []
    for feature in sorted(d.feature.unique()):
        for h in HORIZONS:
            ac = f"actual_h{h}"
            x = d[(d.feature == feature) & d[ac].notna()].copy()
            if x.empty:
                continue
            arms = {
                "paper_sarima_s12": x[f"sarima_h{h}_p"],
                "paper_hybrid_bilstm": x[f"hybrid_h{h}"],
                "ours_sarima_s7": x[f"sarima_h{h}"],
                "ours_recipe": x[f"final_h{h}"],
            }
            row = {"target": feature, "horizon": h, "n": len(x), "n_folds": x.fold.nunique()}
            mask = {name: (x[ac].notna() & pd.Series(p).notna()).values
                    for name, p in arms.items()}
            for name, p in arms.items():
                msk = mask[name]
                row[f"rmse_{name}"] = rmse(x.loc[msk, ac], p[msk])
                row[f"n_{name}"] = int(msk.sum())
                # pembanding adil: SARIMA mingguan kita pada origin yang sama persis
                row[f"delta_{name}_pct"] = (row[f"rmse_{name}"]
                                            / rmse(x.loc[msk, ac], x.loc[msk, f"sarima_h{h}"]) - 1) * 100
            row["delta_ours_vs_paper_pct"] = (row["rmse_ours_recipe"]
                                              / row["rmse_paper_hybrid_bilstm"] - 1) * 100
            recs.append(row)

            for label, col_ref, col_new in [
                ("kita vs paper(BiLSTM)", "final", "hybrid"),
                ("kita vs paper(SARIMA s12)", "final", "sarima_p"),
                ("paper: BiLSTM vs SARIMA-nya sendiri", "hybrid", "sarima_p"),
                ("kita vs SARIMA mingguan kita", "final", "sarima"),
            ]:
                ref = x[f"{col_ref}_h{h}"] if col_ref != "sarima_p" else x[f"sarima_h{h}_p"]
                new = x[f"{col_new}_h{h}"] if col_new != "sarima_p" else x[f"sarima_h{h}_p"]
                msk = (x[ac].notna() & ref.notna() & new.notna()).values
                r_ref = rmse(x.loc[msk, ac], ref[msk])
                r_new = rmse(x.loc[msk, ac], new[msk])
                _, p = dm(x[ac].values[msk] - np.asarray(ref, float)[msk],
                          x[ac].values[msk] - np.asarray(new, float)[msk], h)
                dm_recs.append({"target": feature, "horizon": h, "uji": label, "n": int(msk.sum()),
                                "delta_pct": (r_new / r_ref - 1) * 100,
                                "dm_p": p, "nyata05": "ya" if (p == p and p < 0.05) else "tidak"})

    m, t = pd.DataFrame(recs), pd.DataFrame(dm_recs)
    m.to_csv(out_m, index=False)
    t.to_csv(out_d, index=False)

    print("\n===== RMSE per lengan (lipatan & origin identik) =====")
    show = m[["target", "horizon", "n", "n_folds", "rmse_paper_sarima_s12", "rmse_paper_hybrid_bilstm",
              "rmse_ours_sarima_s7", "rmse_ours_recipe", "delta_ours_vs_paper_pct"]]
    print(show.round(4).to_string(index=False))
    print("\n===== uji Diebold-Mariano =====")
    print(t.round(4).to_string(index=False))

    k = m.delta_ours_vs_paper_pct
    print(f"\nkita vs paper(BiLSTM): menang {int((k < 0).sum())}/{len(k)} sel | rata2 {k.mean():+.2f}% "
          f"| terbaik {k.min():+.2f}% | terburuk {k.max():+.2f}%")
    kj = t[t.uji == "kita vs paper(BiLSTM)"]
    sig = kj.dm_p < 0.05
    print(f"menang nyata (p<0,05 & delta<0): {int(((sig) & (kj.delta_pct < 0)).sum())} sel | "
          f"kalah nyata (p<0,05 & delta>0): {int(((sig) & (kj.delta_pct > 0)).sum())} sel | "
          f"p terkecil {kj.dm_p.min():.4f}")

    lines = ["# Adu final: resep kita vs metode paper (SARIMA + BiLSTM)", "",
             f"Lipatan & origin identik untuk semua lengan ({m.n_folds.max()} lipatan, "
             f"hingga {int(m.n.max())} origin per sel).", "",
             "| polutan | h | paper SARIMA s12 | paper +BiLSTM | kita SARIMA s7 | resep kita | kita vs paper |",
             "|---|---|---|---|---|---|---|"]
    for _, x in m.iterrows():
        lines.append(f"| {x.target} | {x.horizon} | {x.rmse_paper_sarima_s12:.4f} | "
                     f"{x.rmse_paper_hybrid_bilstm:.4f} | {x.rmse_ours_sarima_s7:.4f} | "
                     f"{x.rmse_ours_recipe:.4f} | {x.delta_ours_vs_paper_pct:+.1f}% |")
    lines += ["", "## Uji Diebold-Mariano (p < 0,05 = beda nyata)", "",
              "| polutan | h | uji | delta | p | nyata |", "|---|---|---|---|---|---|"]
    for _, x in t.iterrows():
        p = "n/a" if x.dm_p != x.dm_p else f"{x.dm_p:.4f}"
        lines.append(f"| {x.target} | {x.horizon} | {x.uji} | {x.delta_pct:+.2f}% | {p} | {x.nyata05} |")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print(f"\ntabel markdown: {out_md}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=OURS)
    ap.add_argument("--paper", default=PAPER)
    ap.add_argument("--out-m", default=OUT_M)
    ap.add_argument("--out-d", default=OUT_D)
    ap.add_argument("--out-md", default=OUT_MD)
    a = ap.parse_args()
    sys.exit(main(a.ours, a.paper, a.out_m, a.out_d, a.out_md))