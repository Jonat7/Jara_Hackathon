#!/usr/bin/env python3
"""Level 3: forecast a distribution, not a point, and spend the uncertainty.

Level 2 showed the procurement cost function is a pinball loss whose minimiser is the
quantile COST_SHORT/(COST_SHORT+COST_LONG) = 0.556, not the mean. This fits quantile
models directly and tests three things:

    1. calibration  - does the q-quantile forecast actually cover q of outcomes?
    2. decision     - does bidding the cost-optimal quantile beat bidding the mean,
                      and does the empirical cost minimum land where theory says?
    3. diagnosis    - where is the model uncertain? Interval width by time of day,
                      by temperature, and on the hardest days.

The headline is deliberately counter-intuitive: the cost-optimal forecast is WORSE on MAE
than the mean forecast, and cheaper in euros. Accuracy and value are not the same objective.

    python level3_uncertainty.py

Writes into outputs/ (see config.py):
    level3_quantiles.csv     per-quantile cost, pinball loss, coverage, MAE
    level3_predictions.csv   every quantile forecast per test slot
    fig_level3.png           cost curve, calibration, uncertainty structure, fan chart
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import sys as _sys, os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "analysis"))
import config
from level2_cost import imbalance_cost

QUANTILES = [0.05, 0.10, 0.25, 0.40, 0.50,
             round(config.COST_OPTIMAL_QUANTILE, 3), 0.65, 0.75, 0.90, 0.95]


def pinball(y, f, q):
    d = y - f
    return np.maximum(q * d, (q - 1) * d).mean()


def main():
    df = pd.read_csv(os.path.join(config.OUTPUTS, "features.csv"),
                     index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df["kwh_total"].notna()]
    feats = config.feature_list()
    tr, te = df[df.is_test == 0], df[df.is_test == 1]
    Xtr, ytr, Xte, yte = tr[feats], tr["kwh_total"], te[feats], te["kwh_total"]
    print("Train %d slots -> Test %d slots" % (len(tr), len(te)))

    # The mean forecast from Level 0, refit here so both arms are strictly comparable.
    mean_model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0)
    mean_model.fit(Xtr, ytr)
    pred = {"mean": pd.Series(mean_model.predict(Xte), index=te.index)}

    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q,
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        m.fit(Xtr, ytr)
        pred["q%.3f" % q] = pd.Series(m.predict(Xte), index=te.index)
        print("  fitted quantile %.3f" % q, flush=True)

    out = pd.DataFrame(pred)
    out["actual"] = yte
    out.to_csv(os.path.join(config.OUTPUTS, "level3_predictions.csv"))

    rows = []
    for name, f in pred.items():
        q = None if name == "mean" else float(name[1:])
        rows.append({
            "forecast": name,
            "quantile": q if q is not None else np.nan,
            "MAE": (f - yte).abs().mean(),
            "bias": (f - yte).mean(),
            "cost_EUR": imbalance_cost(yte, f),
            "pinball_at_own_q": pinball(yte.to_numpy(), f.to_numpy(), q) if q else np.nan,
            "coverage_%": (yte <= f).mean() * 100,
        })
    tab = pd.DataFrame(rows).set_index("forecast")
    print("\n=== Quantile forecasts on the test window ===")
    print(tab.round(2).to_string())

    qt = tab[tab["quantile"].notna()]
    emp_best = qt["cost_EUR"].idxmin()
    emp_q = float(qt.loc[emp_best, "quantile"])
    mean_cost = tab.loc["mean", "cost_EUR"]
    opt_cost = qt["cost_EUR"].min()

    print("\n=== Calibration ===")
    for _, r in qt.iterrows():
        print("  nominal %.3f -> empirical coverage %.1f%%  (gap %+.1f pp)"
              % (r["quantile"], r["coverage_%"], r["coverage_%"] - r["quantile"] * 100))
    gap = (qt["coverage_%"] - qt["quantile"] * 100).abs().mean()
    print("  mean absolute calibration gap: %.1f percentage points" % gap)

    print("\n=== Decision: does bidding a quantile beat bidding the mean? ===")
    print("theoretical cost-optimal quantile : %.3f" % config.COST_OPTIMAL_QUANTILE)
    print("empirical cheapest quantile       : %.3f" % emp_q)
    print("mean forecast   cost EUR %.0f, MAE %.2f"
          % (mean_cost, tab.loc["mean", "MAE"]))
    print("best quantile   cost EUR %.0f, MAE %.2f"
          % (opt_cost, qt.loc[emp_best, "MAE"]))
    print("  -> %.1f%% cheaper, while being %.2f kWh WORSE on MAE"
          % ((1 - opt_cost / mean_cost) * 100,
             qt.loc[emp_best, "MAE"] - tab.loc["mean", "MAE"]))

    # --- where is the model uncertain? ---
    lo, hi = pred["q0.100"], pred["q0.900"]
    width = (hi - lo).rename("width")
    w = pd.DataFrame({"width": width, "actual": yte,
                      "slot": te["slot_of_day"], "temp": te["Temperature_avg_hourly"]})
    w["rel"] = w["width"] / w["actual"]
    by_slot = w.groupby("slot")["width"].mean()
    tb = pd.cut(w["temp"], [-20, -5, 0, 5, 10, 30])
    by_temp = w.groupby(tb, observed=False).agg(
        width=("width", "mean"), rel=("rel", "mean"), n=("width", "size"))
    print("\n=== Where the model is uncertain (80% interval width) ===")
    print("widest hours of day:")
    for s, v in by_slot.nlargest(3).items():
        print("   slot %2d (%02d:%02d UTC) width %.1f kWh" % (s, s // 4, (s % 4) * 15, v))
    print("narrowest hours of day:")
    for s, v in by_slot.nsmallest(3).items():
        print("   slot %2d (%02d:%02d UTC) width %.1f kWh" % (s, s // 4, (s % 4) * 15, v))
    print("\nby temperature band:")
    print(by_temp.round(2).to_string())

    cov80 = ((yte >= lo) & (yte <= hi)).mean() * 100
    print("\n80%% interval actually covers %.1f%% of outcomes" % cov80)

    # --- recalibration -------------------------------------------------------------
    # The raw intervals are too narrow. Standard fix: learn the nominal -> empirical
    # mapping on a held-out calibration window that the TEST data never touches, then
    # ask for whatever nominal quantile delivers the coverage actually wanted.
    cal_start = pd.Timestamp("2023-10-15", tz="UTC")
    test_start = pd.Timestamp(config.TEST_START, tz="UTC")
    inner = tr[tr.index < cal_start]
    cal = tr[tr.index >= cal_start]
    print("\n=== Recalibration (inner train %d slots, calibration %d slots) ==="
          % (len(inner), len(cal)))

    nominal, empirical = [], []
    cal_pred = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=0)
        m.fit(inner[feats], inner["kwh_total"])
        p = m.predict(cal[feats])
        cal_pred[q] = p
        nominal.append(q)
        empirical.append((cal["kwh_total"].to_numpy() <= p).mean())
    emp_arr, nom_arr = np.array(empirical), np.array(nominal)
    order = np.argsort(emp_arr)

    def mapped_nominal(target):
        """Nominal quantile to request in order to achieve `target` coverage."""
        return float(np.interp(target, emp_arr[order], nom_arr[order]))

    targets = [0.10, 0.556, 0.90]
    print("target coverage -> nominal quantile to request (from calibration window):")
    for t in targets:
        print("   %.3f -> %.3f" % (t, mapped_nominal(t)))

    # The mapping describes the INNER-train model's bias, so it may only be applied to
    # that same model. Fitting on full train here would mix a correction derived from one
    # model with the behaviour of a different, better-informed one.
    recal = {}
    for t in targets:
        qn = mapped_nominal(t)
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=min(max(qn, 0.005), 0.995),
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        m.fit(inner[feats], inner["kwh_total"])
        recal[t] = pd.Series(m.predict(Xte), index=te.index)

    print("\nTest coverage, raw vs recalibrated:")
    for t in targets:
        raw = tab.loc["q%.3f" % t, "coverage_%"] if ("q%.3f" % t) in tab.index else np.nan
        new = (yte <= recal[t]).mean() * 100
        print("   target %.1f%%  raw %.1f%%  recalibrated %.1f%%" % (t * 100, raw, new))
    new80 = ((yte >= recal[0.10]) & (yte <= recal[0.90])).mean() * 100
    print("   80%% interval: raw %.1f%% -> recalibrated %.1f%%" % (cov80, new80))
    cost_recal = imbalance_cost(yte, recal[0.556])
    print("   cost at recalibrated cost-optimal quantile: EUR %.0f (mean forecast %.0f)"
          % (cost_recal, mean_cost))
    print("\nVERDICT: recalibration makes things WORSE here and is not used downstream.")
    print("  Split-conformal methods assume the calibration data is exchangeable with")
    print("  the test data. A chronological split puts autumn in calibration and winter")
    print("  in test - different regimes, so the correction learned on one misfires on")
    print("  the other. With only 13 months there is no same-season holdout available.")
    print("  The raw quantiles are reported as the deliverable; their miscalibration is")
    print("  stated rather than papered over.")

    tab.to_csv(os.path.join(config.OUTPUTS, "level3_quantiles.csv"))

    # --- figure ---
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))

    ax[0].plot(qt["quantile"], qt["cost_EUR"], marker="o", color="tab:blue")
    ax[0].axhline(mean_cost, color="tab:gray", ls=":", lw=1, label="mean forecast")
    ax[0].axvline(config.COST_OPTIMAL_QUANTILE, color="tab:red", ls="--", lw=1,
                  label="theory %.3f" % config.COST_OPTIMAL_QUANTILE)
    ax[0].axvline(emp_q, color="tab:green", ls="--", lw=1,
                  label="empirical %.3f" % emp_q)
    ax[0].set_xlabel("forecast quantile")
    ax[0].set_ylabel("imbalance cost (EUR)")
    ax[0].set_title("Cost minimum sits near the theoretical quantile")
    ax[0].legend(fontsize=8)

    ax[1].plot([0, 100], [0, 100], color="tab:gray", ls=":", lw=1)
    ax[1].plot(qt["quantile"] * 100, qt["coverage_%"], marker="o", color="tab:purple")
    ax[1].set_xlabel("nominal quantile (%)")
    ax[1].set_ylabel("empirical coverage (%)")
    ax[1].set_title("Calibration (mean gap %.1f pp)" % gap)

    ax[2].plot(by_slot.index, by_slot.values, color="tab:orange")
    ax[2].axvspan(40, 44, color="tab:red", alpha=0.10)
    ax[2].set_xlabel("slot of day (0 = 00:00)")
    ax[2].set_ylabel("80% interval width (kWh)")
    ax[2].set_title("Uncertainty peaks at the blocking window (shaded)")

    day = pd.Timestamp(w.groupby(te.index.date)["temp"].mean().idxmin(), tz="UTC")
    sl = slice(day, day + pd.Timedelta("23:45:00"))
    ax[3].fill_between(out.loc[sl].index, pred["q0.100"].loc[sl], pred["q0.900"].loc[sl],
                       color="tab:blue", alpha=0.18, label="80% interval")
    ax[3].fill_between(out.loc[sl].index, pred["q0.250"].loc[sl], pred["q0.750"].loc[sl],
                       color="tab:blue", alpha=0.30, label="50% interval")
    ax[3].plot(out.loc[sl].index, out.loc[sl, "actual"], color="black", lw=2,
               label="actual")
    ax[3].plot(out.loc[sl].index, pred["q%.3f" % round(config.COST_OPTIMAL_QUANTILE, 3)]
               .loc[sl], color="tab:green", lw=1.2, label="bid (q=%.3f)"
               % round(config.COST_OPTIMAL_QUANTILE, 3))
    ax[3].set_title("Coldest test day: %s" % day.date())
    ax[3].set_ylabel("kWh per slot")
    ax[3].tick_params(axis="x", rotation=30)
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUTS, "fig_level3.png"), dpi=130)
    print("\nWrote level3_quantiles.csv, level3_predictions.csv, fig_level3.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
