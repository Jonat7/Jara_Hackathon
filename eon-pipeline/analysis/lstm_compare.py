#!/usr/bin/env python3
"""An LSTM, scored against the tree models on identical data and the identical split.

Framing matters for fairness. A day-ahead bid is 96 numbers committed at once, so this is
a direct multi-output model, not a recursive one-step-ahead model whose errors compound:

    encoder   LSTM over the past 7.5 days of (load, temperature) at hourly resolution,
              ending at 11:45 on D-1 - exactly gate closure, so no future leaks in
    decoder   last hidden state, concatenated with day D's weather forecast and calendar,
              through a small MLP to 96 outputs

Scaling statistics come from the training period only. The validation slice used for early
stopping is the last 30 days of training data, cut chronologically - never at random.

The binding constraint is sample size: one training example per DAY, so roughly 300 of
them. That is the honest reason to expect the trees to win, and it is reported rather
than hidden.

    python lstm_compare.py

Writes into outputs/ (see config.py):
    metrics_lstm.csv       same metric set as metrics_level0.csv, for direct comparison
    lstm_predictions.csv   actual and forecast per test slot
"""
import os

import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("torch is not installed - skipping the LSTM comparison.\n"
                     "  pip install torch   (CPU build is enough)")

SLOTS = 96
HIST_DAYS = 7           # whole days of history before D-1
HIST_HOURS = HIST_DAYS * 24 + 12   # plus D-1 up to 11:45 = gate closure
HIDDEN = 64
EPOCHS = 400
PATIENCE = 40
SEED = 0


def build_samples(df):
    """One sample per target day: (history sequence, exogenous vector, 96 targets)."""
    df = df.sort_index()
    hourly = pd.DataFrame({
        "load": df["kwh_total"].resample("h").mean(),
        "temp": df["Temperature_avg_hourly"].resample("h").mean(),
    })
    hourly = hourly.ffill().bfill()

    dates, X_hist, X_exo, Y, is_test = [], [], [], [], []
    for day, g in df.groupby(df.index.normalize()):
        if len(g) != SLOTS or g["kwh_total"].isna().any():
            continue
        end = day - pd.Timedelta(hours=12)          # 11:45 on D-1, rounded to the hour
        start = end - pd.Timedelta(hours=HIST_HOURS)
        h = hourly.loc[(hourly.index >= start) & (hourly.index < end)]
        if len(h) != HIST_HOURS:
            continue

        # Day D's forecast weather, hourly, plus calendar. Known at gate closure.
        d = g.iloc[::4]
        if len(d) != 24:
            continue
        exo = np.concatenate([
            d["Temperature_avg_hourly"].to_numpy(),
            d["hdd"].to_numpy(),
            d["Sunshine_duration_hourly"].to_numpy(),
            np.eye(7)[int(g["day_of_week"].iloc[0])],
            [g["doy_sin"].iloc[0], g["doy_cos"].iloc[0]],
        ])
        if np.isnan(exo).any():
            continue

        dates.append(day)
        X_hist.append(h[["load", "temp"]].to_numpy())
        X_exo.append(exo)
        Y.append(g["kwh_total"].to_numpy())
        is_test.append(int(g["is_test"].iloc[0] == 1))

    return (np.array(dates), np.asarray(X_hist, dtype=np.float32),
            np.asarray(X_exo, dtype=np.float32), np.asarray(Y, dtype=np.float32),
            np.asarray(is_test, dtype=bool))


class Net(nn.Module):
    def __init__(self, exo_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + exo_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, SLOTS))

    def forward(self, hist, exo):
        _, (h, _) = self.lstm(hist)
        return self.head(torch.cat([h[-1], exo], dim=1))


def scores(y, p):
    y, p = y.ravel(), p.ravel()
    e = p - y
    n = len(y) // SLOTS
    dy, dp = y.reshape(n, SLOTS).sum(1), p.reshape(n, SLOTS).sum(1)
    ss = ((y - y.mean()) ** 2).sum()
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAPE_%": (np.abs(e) / y).mean() * 100,
            "R2": 1 - (e ** 2).sum() / ss,
            "bias": e.mean(),
            "daily_MAE": np.abs(dp - dy).mean(),
            "daily_MAPE_%": (np.abs(dp - dy) / dy).mean() * 100,
            "n": float(len(y))}


def run_once(Xh, Xe, Y, tr_mask, te_mask, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Scale on training data only.
    h_mu, h_sd = Xh[tr_mask].mean((0, 1)), Xh[tr_mask].std((0, 1)) + 1e-8
    e_mu, e_sd = Xe[tr_mask].mean(0), Xe[tr_mask].std(0) + 1e-8
    y_mu, y_sd = Y[tr_mask].mean(), Y[tr_mask].std() + 1e-8
    Xh = (Xh - h_mu) / h_sd
    Xe = (Xe - e_mu) / e_sd
    Yz = (Y - y_mu) / y_sd

    # Chronological validation slice: the last 30 training days.
    tr_idx = np.where(tr_mask)[0]
    val_idx, fit_idx = tr_idx[-30:], tr_idx[:-30]
    t = lambda a: torch.tensor(a)
    Xh_f, Xe_f, Y_f = t(Xh[fit_idx]), t(Xe[fit_idx]), t(Yz[fit_idx])
    Xh_v, Xe_v, Y_v = t(Xh[val_idx]), t(Xe[val_idx]), t(Yz[val_idx])
    Xh_t, Xe_t = t(Xh[te_mask]), t(Xe[te_mask])

    model = Net(Xe.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.L1Loss()          # matches the MAE the trees are judged on

    best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xh_f))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = lossf(model(Xh_f[b], Xe_f[b]), Y_f[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xh_v, Xe_v), Y_v).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("  early stop at epoch %d (best val L1 %.4f)" % (ep, best))
                break
        if ep % 50 == 0:
            print("  epoch %3d  val L1 %.4f" % (ep, v), flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xh_t, Xe_t).numpy() * y_sd + y_mu

    return pred


def main():
    df = pd.read_csv(os.path.join(config.OUTPUTS, "features.csv"),
                     index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, utc=True)
    dates, Xh, Xe, Y, te_mask = build_samples(df)
    tr_mask = ~te_mask
    print("Samples: %d days total -> %d train, %d test"
          % (len(dates), tr_mask.sum(), te_mask.sum()))
    print("One training example per DAY: the LSTM sees ~%d examples, the "
          "tree models see %d rows." % (tr_mask.sum(), tr_mask.sum() * SLOTS))

    # Several seeds, because a 5%% gap on one run is not a result.
    actual = Y[te_mask]
    preds, rows = [], []
    for seed in (0, 1, 2, 3, 4):
        p = run_once(Xh.copy(), Xe.copy(), Y.copy(), tr_mask, te_mask, seed)
        preds.append(p)
        r = scores(actual, p); r["seed"] = seed
        rows.append(r)
        print("  seed %d -> MAE %.2f" % (seed, r["MAE"]), flush=True)
    per_seed = pd.DataFrame(rows).set_index("seed")
    print("")
    print("Per-seed MAE: mean %.2f, sd %.2f, range %.2f-%.2f"
          % (per_seed.MAE.mean(), per_seed.MAE.std(),
             per_seed.MAE.min(), per_seed.MAE.max()))
    per_seed.round(3).to_csv(os.path.join(config.OUTPUTS, "lstm_seeds.csv"))

    pred = np.mean(preds, axis=0)   # seed-averaged ensemble
    row = scores(actual, pred)
    row = {k: (float(v) if not isinstance(v, str) else v)
           for k, v in row.items()}
    row["model"] = "lstm"
    tab = pd.DataFrame([row]).set_index("model")
    print("\n=== LSTM on the test window ===")
    print(tab.round(2).to_string())

    ref = os.path.join(config.OUTPUTS, "metrics_level0.csv")
    if os.path.exists(ref):
        other = pd.read_csv(ref, index_col=0)
        both = pd.concat([other, tab]).sort_values("MAE")
        print("\n=== All models, same data and split ===")
        print(both[["MAE", "RMSE", "MAPE_%", "R2", "daily_MAPE_%"]].round(2).to_string())

    tab.to_csv(os.path.join(config.OUTPUTS, "metrics_lstm.csv"))
    idx = pd.to_datetime(np.repeat(dates[te_mask], SLOTS)) + pd.to_timedelta(
        np.tile(np.arange(SLOTS) * 15, te_mask.sum()), unit="m")
    pd.DataFrame({"actual": actual.ravel(), "lstm": pred.ravel()}, index=idx).to_csv(
        os.path.join(config.OUTPUTS, "lstm_predictions.csv"))
    print("\nWrote metrics_lstm.csv and lstm_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
