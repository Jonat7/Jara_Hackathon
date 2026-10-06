#!/usr/bin/env python3
"""Level 0: baselines vs one model, scored on the held-out winter window.

A forecast is only meaningful against a baseline, so the naive methods are scored first and
reported alongside. The lag-1d naive is labelled INFEASIBLE because that reading does not
exist yet at gate closure - it is shown for reference, not as a legitimate contender.

    python run_level0.py

Writes into outputs/ (see config.py):
    metrics_level0.csv      one row per method
    predictions_test.csv    actual and every forecast, per test slot
    feature_importance.csv  permutation importance, in kWh of MAE
    fig_example_days.png    coldest and mildest test day
    fig_error_structure.png error across the day, and daily procured volume
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config


def scores(y, yhat):
    """Per-slot accuracy plus the daily totals that procurement actually buys."""
    ok = y.notna() & yhat.notna()
    y, yhat = y[ok], yhat[ok]
    err = yhat - y
    daily = pd.DataFrame({"y": y, "yhat": yhat}).groupby(y.index.date).sum()
    return {
        "MAE": err.abs().mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAPE_%": (err.abs() / y).mean() * 100,
        "R2": r2_score(y, yhat),
        "bias": err.mean(),
        "daily_MAE": (daily.yhat - daily.y).abs().mean(),
        "daily_MAPE_%": ((daily.yhat - daily.y).abs() / daily.y).mean() * 100,
        "n": len(y),
    }


def main():
    path = os.path.join(config.OUTPUTS, "features.csv")
    if not os.path.exists(path):
        raise SystemExit("features.csv not found - run the data_prep stages first.")
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df["kwh_total"].notna()]

    feats = config.feature_list()
    train, test = df[df.is_test == 0], df[df.is_test == 1]
    Xtr, ytr = train[feats], train["kwh_total"]
    Xte, yte = test[feats], test["kwh_total"]
    print("Train %d slots -> Test %d slots (%s .. %s)"
          % (len(train), len(test), test.index.min().date(), test.index.max().date()))

    pred = {
        "naive_seasonal_lag7d": test["lag_7d"],
        "naive_lag2d": test["lag_2d"],
        "naive_lag1d_INFEASIBLE": test["lag_1d"],
    }

    # Early stopping is off on purpose: its validation split is random, which would leak
    # future slots into model selection. Iterations are fixed instead.
    gbm = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=0)
    gbm.fit(Xtr, ytr)
    pred["gbm"] = pd.Series(gbm.predict(Xte), index=test.index)

    # Random forest: the challenge's own reading list points at it, so it is worth
    # having a measured answer rather than an opinion. Averaged independent trees
    # rather than boosted sequential ones; imputed because RF has no native NaN path
    # for this estimator's defaults.
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                             n_jobs=-1, random_state=0))
    rf.fit(Xtr, ytr)
    pred["random_forest"] = pd.Series(rf.predict(Xte), index=test.index)

    # Same features, linear: the gap to the gbm is the evidence that the
    # temperature-to-load relationship is not a straight line.
    ridge = make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(Xtr, ytr)
    pred["ridge"] = pd.Series(ridge.predict(Xte), index=test.index)

    # Plain OLS, no penalty. Ridge is already a linear model, so the gap between these
    # two is what the L2 regularisation buys on these 17 features - not a new family.
    ols = make_pipeline(SimpleImputer(strategy="median"),
                        StandardScaler(), LinearRegression())
    ols.fit(Xtr, ytr)
    pred["linear_regression"] = pd.Series(ols.predict(Xte), index=test.index)

    table = pd.DataFrame({k: scores(yte, v) for k, v in pred.items()}).T
    table = table.sort_values("MAE")
    print("\n=== Test-set performance (kWh per 15-min slot) ===")
    print(table.round(2).to_string())

    best = table.index[0]
    naive = table.loc["naive_seasonal_lag7d", "MAE"]
    print("\nBest: %s  |  MAE improvement vs seasonal naive: %.1f%%"
          % (best, (1 - table.loc[best, "MAE"] / naive) * 100))

    pi = permutation_importance(gbm, Xte, yte, n_repeats=5,
                                random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    print("\n=== Permutation importance (MAE kWh lost if feature shuffled) ===")
    print(imp.round(2).to_string())

    out = pd.DataFrame(pred)
    out["actual"] = yte
    imp.to_csv(os.path.join(config.OUTPUTS, "feature_importance.csv"))
    table.to_csv(os.path.join(config.OUTPUTS, "metrics_level0.csv"))
    out.to_csv(os.path.join(config.OUTPUTS, "predictions_test.csv"))

    daily_temp = test.groupby(test.index.date)["temp_day_mean"].mean()
    cold = pd.Timestamp(daily_temp.idxmin(), tz="UTC")
    mild = pd.Timestamp(daily_temp.idxmax(), tz="UTC")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, day, label in ((axes[0], cold, "coldest"), (axes[1], mild, "mildest")):
        sl = out.loc[day:day + pd.Timedelta("23:45:00")]
        ax.plot(sl.index, sl["actual"], label="actual", color="black", lw=2)
        ax.plot(sl.index, sl["gbm"], label="gbm", color="tab:blue")
        ax.plot(sl.index, sl["naive_seasonal_lag7d"], label="naive lag-7d",
                color="tab:orange", ls="--")
        ax.set_title("%s test day: %s (%.1f degC)"
                     % (label, day.date(), daily_temp.loc[day.date()]))
        ax.set_xlabel("time (UTC)")
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("kWh per 15-min slot (%d households)" % int(df["n_reporting"].max()))
    axes[0].legend()
    fig.tight_layout()
    config.savefig(fig, "fig_example_days.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    by_slot = (out["gbm"] - out["actual"]).abs().groupby(test["slot_of_day"]).mean()
    by_slot_n = (out["naive_seasonal_lag7d"] - out["actual"]).abs().groupby(
        test["slot_of_day"]).mean()
    axes[0].plot(by_slot.index, by_slot.values, label="gbm")
    axes[0].plot(by_slot_n.index, by_slot_n.values, label="naive lag-7d", ls="--")
    axes[0].set_xlabel("slot of day (0 = 00:00, 95 = 23:45)")
    axes[0].set_ylabel("MAE (kWh)")
    axes[0].set_title("Where the error sits across the day")
    axes[0].legend()

    d = pd.DataFrame({"y": out["actual"], "p": out["gbm"]}).groupby(out.index.date).sum()
    axes[1].plot(d.index, d["y"], label="actual", color="black")
    axes[1].plot(d.index, d["p"], label="gbm", color="tab:blue")
    axes[1].set_title("Daily procured volume: actual vs forecast")
    axes[1].set_ylabel("kWh per day")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    fig.tight_layout()
    config.savefig(fig, "fig_error_structure.png", dpi=130)

    print("\nWrote metrics, predictions, importance and 2 figures to outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
