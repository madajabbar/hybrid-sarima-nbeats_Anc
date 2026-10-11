"""Fase 11 — lengan BiLSTM RUJUKAN pada semua lipatan, protokol paper apa adanya.

Tujuan: adu murni resep kita vs metode paper pada lipatan & origin yang identik.
Protokol paper (dari notebooks/01_baseline_bilstm.ipynb — tidak diubah):
    SARIMA (2,1,1)(0,0,0,12) -> residu in-sample
    MinMaxScaler(0,1)        -> sekuens 90 langkah
    BiLSTM 2 lapis + Dense   -> koreksi 1 langkah ke depan (satu model per polutan/lipatan)
    hybrid = SARIMA + koreksi

Perbedaan yang disengaja & dicatat di naskah:
  - scaler dilatih hanya dari residu blok latih (paper melatihnya dari residu yang sama,
    kita perketat supaya tidak ada informasi masa depan);
  - koreksi diterapkan pada tiap origin tanpa melatih ulang (sama seperti paper).

Pemakaian di Kaggle/Colab (GPU T4):
    python scripts/13_bilstm_all_folds.py                 # semua polutan, 5 lipatan
    python scripts/13_bilstm_all_folds.py --smoke         # uji cepat 2 epoch
Cekpoint per (polutan, lipatan) -> sesi Kaggle yang mati bisa dilanjutkan, bukan mulai nol.
"""
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "transformed_data.csv")
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(RESULTS, "lstm_folds")
OUT = os.path.join(RESULTS, "13_bilstm_all_folds_predictions.csv")

FEATURES = ["NO2", "PM10", "PM25"]
FOLD_OFFSETS = [0, 120, 240, 360, 480]
TEST_DAYS, STEP, MAX_H, HORIZONS = 720, 30, 30, [1, 7, 30]
ORDER, SEASONAL = (2, 1, 1), (0, 0, 0, 12)      # protokol paper


def load_all():
    raw = pd.read_csv(DATA)
    raw.columns = [c.strip("\ufeff") for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%y")
    raw = raw.sort_values("Date").set_index("Date")
    out = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        out[c], _ = boxcox(raw[c] + abs(raw[c].min()) + 1)
    return out


def make_sequences(a, steps):
    X = np.stack([a[i:i + steps] for i in range(len(a) - steps)])[..., None]
    y = a[steps:]
    return X, y


def build_lstm(steps):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    m = Sequential([
        Bidirectional(LSTM(192, return_sequences=True), input_shape=(steps, 1)),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="linear"),
    ])
    m.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
              loss=tf.keras.losses.Huber(delta=1.0))
    return m


def build_lstm_tuner(hp, steps):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    m = Sequential([
        Bidirectional(LSTM(hp.Int("units_1", 64, 512, step=64), return_sequences=True),
                      input_shape=(steps, 1)),
        Dropout(hp.Float("dropout_1", 0.1, 0.5, step=0.1)),
        Bidirectional(LSTM(hp.Int("units_2", 64, 256, step=64))),
        Dense(hp.Int("dense_units", 64, 256, step=64), activation="relu"),
        Dropout(hp.Float("dropout_2", 0.1, 0.5, step=0.1)),
        Dense(1, activation="linear"),
    ])
    m.compile(optimizer=tf.keras.optimizers.AdamW(
        learning_rate=hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5]),
        weight_decay=hp.Float("weight_decay", 1e-6, 1e-4, sampling="log")),
        loss=tf.keras.losses.Huber(delta=1.0))
    return m


def tune(feature, fold, X, y, steps, max_trials, epochs, batch):
    """Cari hiperparameter (protokol paper). Hasil disimpan ke JSON supaya bisa dipakai ulang."""
    path = os.path.join(CKPT, f"tuned_{feature}.json")
    if os.path.exists(path):                       # hasil tuning yang sudah ada
        return json.load(open(path))
    from keras_tuner import RandomSearch
    from tensorflow.keras.callbacks import EarlyStopping
    t = RandomSearch(lambda hp: build_lstm_tuner(hp, steps), objective="val_loss",
                     max_trials=max_trials, executions_per_trial=1, overwrite=True,
                     directory=os.path.join(CKPT, f"tuner_{feature}_fold{fold}"),
                     project_name="sarima_lstm")
    t.search(X, y, validation_split=0.2, epochs=epochs, batch_size=batch,
             callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
             verbose=0)
    hp = t.get_best_hyperparameters(1)[0]
    best = {k: hp.get(k) for k in ["units_1", "units_2", "dense_units",
                                   "dropout_1", "dropout_2", "learning_rate", "weight_decay"]}
    json.dump(best, open(path, "w"), indent=1)
    print(f"  [tune {feature}] {best}", flush=True)
    return best


def fit_resumable(model, tag, X, y, epochs, batch):
    """Latih; bobot terbaik disimpan per (polutan, lipatan) supaya sesi yang mati bisa dilanjutkan."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    os.makedirs(os.path.join(CKPT, "weights"), exist_ok=True)
    w = os.path.join(CKPT, "weights", f"{tag}.weights.h5")
    if os.path.exists(w):
        model.build((None,) + tuple(X.shape[1:]))
        model.load_weights(w)
        print(f"  [{tag}] bobot terbaik dari sesi sebelumnya dipakai, latih dilewati", flush=True)
        return model
    cbs = [ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=0),
           EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
           ModelCheckpoint(w, monitor="val_loss", save_best_only=True,
                           save_weights_only=True, verbose=0)]
    t0 = time.time()
    model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch, verbose=0, callbacks=cbs)
    model.save_weights(w)
    print(f"  [{tag}] latih maks {epochs} epoch: {time.time()-t0:.0f}s "
          f"({model.count_params():,} parameter)", flush=True)
    return model


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.sqrt(mean_squared_error(a, p))), float(mean_absolute_error(a, p)),
            float(np.mean(np.abs((a - p) / a)) * 100))


def run(features, offsets, epochs, batch, tune_mode, max_trials, steps, seed, limit_origins=None,
        out=None):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    os.makedirs(CKPT, exist_ok=True)
    out = out or OUT
    all_data = load_all()
    rows = []
    if os.path.exists(out):                       # lanjutkan sesi sebelumnya
        rows = pd.read_csv(out).to_dict("records")
        print(f"melanjutkan: {len(rows)} baris dari {out}", flush=True)

    for feature in features:
        for off in offsets:
            done = {(r["feature"], r["fold"]) for r in rows}
            if (feature, off) in done:
                print(f"== {feature} fold+{off}: sudah ada, dilewati", flush=True)
                continue
            print(f"== {feature} fold+{off}", flush=True)
            end, start = len(all_data) - off, len(all_data) - off - TEST_DAYS
            tr = all_data[feature].iloc[:start]
            te = all_data[feature].iloc[start:end]

            sar = SARIMAX(tr, order=ORDER, seasonal_order=SEASONAL).fit(disp=False)
            resid = (tr - sar.fittedvalues).dropna().values
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(resid.reshape(-1, 1))
            scaled = scaler.transform(resid.reshape(-1, 1)).ravel()
            X, y = make_sequences(scaled, steps)

            best = None
            if tune_mode == "each" or (tune_mode == "once" and not os.path.exists(
                    os.path.join(CKPT, f"tuned_{feature}.json"))):
                best = tune(feature, off, X, y, steps, max_trials, epochs, batch)
            elif tune_mode == "once":
                best = json.load(open(os.path.join(CKPT, f"tuned_{feature}.json")))

            model = build_lstm(steps)
            if best:                                          # pakai hiperparameter hasil tuning
                model = None
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
                model = Sequential([
                    Bidirectional(LSTM(best["units_1"], return_sequences=True), input_shape=(steps, 1)),
                    Dropout(best["dropout_1"]),
                    Bidirectional(LSTM(best["units_2"])),
                    Dense(best["dense_units"], activation="relu"),
                    Dropout(best["dropout_2"]),
                    Dense(1, activation="linear"),
                ])
                model.compile(optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=best["learning_rate"], weight_decay=best["weight_decay"]),
                    loss=tf.keras.losses.Huber(delta=1.0))
            model = fit_resumable(model, f"{feature}_fold{off}", X, y, epochs, batch)

            n_origin = 0
            for origin in range(0, len(te) - min(HORIZONS), STEP):
                if limit_origins is not None and n_origin >= limit_origins:
                    break
                hist = pd.concat([tr, te.iloc[:origin]])
                try:
                    res = sar.apply(hist)
                except Exception as e:                        # jangan matikan seluruh run
                    print(f"   origin {origin} gagal: {e}", flush=True)
                    continue
                fc = np.asarray(res.forecast(steps=MAX_H), float)
                rh = (hist - res.fittedvalues).dropna().values
                if len(rh) < steps:
                    continue
                seq = scaler.transform(rh[-steps:].reshape(-1, 1)).ravel()
                corr = float(scaler.inverse_transform(
                    model.predict(seq.reshape(1, steps, 1), verbose=0))[0, 0])
                rec = {"feature": feature, "fold": off, "origin_date": str(te.index[origin].date()),
                       "bilstm_corr": corr}
                for h in HORIZONS:
                    if origin >= len(te) - h:
                        continue
                    base = float(fc[h - 1])
                    rec[f"actual_h{h}"] = float(te.iloc[origin + h - 1])
                    rec[f"sarima_h{h}"] = base
                    rec[f"hybrid_h{h}"] = base + corr          # metode paper: koreksi 100%
                rows.append(rec)
                n_origin += 1

            pd.DataFrame(rows).to_csv(out, index=False)        # simpan tiap lipatan selesai
            print(f"   {n_origin} origin -> {out}", flush=True)

    big = pd.DataFrame(rows)
    if big.empty:
        print("tidak ada baris hasil"); return
    print("\n===== lengan paper: SARIMA(0,0,0,12) vs +BiLSTM =====")
    for feature in features:
        for h in HORIZONS:
            d = big[(big.feature == feature) & big[f"actual_h{h}"].notna()]
            if d.empty:
                continue
            b = metrics(d[f"actual_h{h}"], d[f"sarima_h{h}"])
            hy = metrics(d[f"actual_h{h}"], d[f"hybrid_h{h}"])
            print(f"  {feature:<5} h={h:>2} SARIMA {b[0]:.4f} | +BiLSTM {hy[0]:.4f} "
                  f"| Δ {(hy[0]/b[0]-1)*100:+.2f}% | n={len(d)} | lipatan={d.fold.nunique()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="*", default=FEATURES)
    ap.add_argument("--offsets", nargs="*", type=int, default=FOLD_OFFSETS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tune", choices=["off", "once", "each"], default="off")
    ap.add_argument("--max-trials", type=int, default=20)
    ap.add_argument("--lstm-steps", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true", help="uji cepat: 2 epoch, 3 origin, NO2 fold+0")
    a = ap.parse_args()
    if a.smoke:
        a.features, a.offsets, a.epochs = ["NO2"], [0], 2
        run(a.features, a.offsets, a.epochs, a.batch, "off", a.max_trials, a.lstm_steps,
            a.seed, limit_origins=3, out=a.out)
    else:
        run(a.features, a.offsets, a.epochs, a.batch, a.tune, a.max_trials, a.lstm_steps, a.seed,
            out=a.out)