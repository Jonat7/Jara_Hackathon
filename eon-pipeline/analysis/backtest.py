#!/usr/bin/env python3
"""Rolling-origin backtest: is the single-split result stable, or a lucky window?

run_level0.py reports one train/test split. One number from one window says nothing about
variance, so this refits the model at successive origins, each time training only on data
before the fold and predicting the month after it. This is the time-series equivalent of
cross-validation; k-fold would shuffle the future into the past and is not usable here.

It also carves a chronological VALIDATION fold out of the training data, so hyperparameters
can be chosen without ever touching the test window.

    python backtest.py

Writes into outputs/ (see config.py):
    backtest_folds.csv   per-fold MAE for model and naive baseline
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

# Each fold: train on everything before `start`, predict [start, end).
# Summer folds matter as much as winter ones: load is a third of its winter level, the
# heat-pump signal that dominates the model is largely absent, and PV output peaks. A
# model validated only on winter says nothing about the rest of the year.
FOLDS = [
    ("2023-06-01", "2023-07-01", "summer"),
    ("2023-07-01", "2023-08-01", "summer"),
    ("2023-08-01", "2023-09-01", "summer"),
    ("2023-09-01", "2023-10-01", "shoulder"),
    ("2023-10-01", "2023-11-01", "shoulder"),
    ("2023-11-01", "2023-12-01", "winter"),
    ("2023-12-01", "2024-01-01", "winter"),
    ("2024-01-01", "2024-02-01", "winter"),
    ("2024-02-01", "2024-02-28", "winter"),
]


def fit_predict(tr, te, feats):
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=0)
    m.fit(tr[feats], tr["kwh_total"])
    return pd.Series(m.predict(te[feats]), index=te.index)


def mae(y, p):
    ok = y.notna() & p.notna()
    return (p[ok] - y[ok]).abs().mean()


def main():
    df = pd.read_csv(os.path.join(config.OUTPUTS, "features.csv"),
                     index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df["kwh_total"].notna()]
    feats = config.feature_list()

    rows = []
    for start, end, season in FOLDS:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        tr = df[df.index < s]
        te = df[(df.index >= s) & (df.index < e)]
        if len(te) == 0 or len(tr) < 5000:
            continue
        pred = fit_predict(tr, te, feats)
        rows.append({
            "fold": start[:7],
            "season": season,
            "train_slots": len(tr),
            "test_slots": len(te),
            "gbm_MAE": mae(te["kwh_total"], pred),
            "naive7d_MAE": mae(te["kwh_total"], te["lag_7d"]),
            "mean_load": te["kwh_total"].mean(),
        })
        print("  fold %s (%s): trained on %d slots, tested on %d"
              % (start[:7], season, len(tr), len(te)), flush=True)

    out = pd.DataFrame(rows)
    out["gbm_MAPE_%"] = out["gbm_MAE"] / out["mean_load"] * 100
    out["improvement_%"] = (1 - out["gbm_MAE"] / out["naive7d_MAE"]) * 100

    print("\n=== Rolling-origin backtest ===")
    print(out.round(2).to_string(index=False))
    print("\ngbm MAE across folds: mean %.2f, sd %.2f, range %.2f-%.2f"
          % (out.gbm_MAE.mean(), out.gbm_MAE.std(),
             out.gbm_MAE.min(), out.gbm_MAE.max()))
    print("improvement over naive: mean %.1f%%, worst fold %.1f%%"
          % (out["improvement_%"].mean(), out["improvement_%"].min()))

    print("")
    print("By season:")
    by = out.groupby("season").agg(
        folds=("fold", "count"), mean_load=("mean_load", "mean"),
        gbm_MAE=("gbm_MAE", "mean"), MAPE=("gbm_MAPE_%", "mean"),
        improvement=("improvement_%", "mean"))
    print(by.round(2).to_string())

    # A validation fold carved from training data, for any tuning that follows.
    val_start = pd.Timestamp("2023-10-15", tz="UTC")
    test_start = pd.Timestamp(config.TEST_START, tz="UTC")
    inner_tr = df[df.index < val_start]
    val = df[(df.index >= val_start) & (df.index < test_start)]
    print("\n=== Tuning split (test window never touched) ===")
    print("inner train %d slots (.. %s) | validation %d slots (%s .. %s)"
          % (len(inner_tr), val_start.date(), len(val),
             val_start.date(), test_start.date()))
    print("validation MAE at current settings: %.2f"
          % mae(val["kwh_total"], fit_predict(inner_tr, val, feats)))

    out.to_csv(os.path.join(config.OUTPUTS, "backtest_folds.csv"), index=False)
    print("\nWrote backtest_folds.csv to outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
