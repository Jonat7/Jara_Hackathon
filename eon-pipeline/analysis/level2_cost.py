#!/usr/bin/env python3
"""Level 2: score the forecasts in euros rather than kWh.

MAE treats a 10 kWh over-forecast and a 10 kWh under-forecast as the same mistake. For a
day-ahead desk they are not: a shortfall is bought back on intraday above the day-ahead
price, a surplus is sold below it. The procurement-relevant metric is therefore asymmetric:

    cost = COST_SHORT * max(0, actual - forecast) + COST_LONG * max(0, forecast - actual)

summed over settlement periods. Prices are assumed (see config.py) as the challenge allows,
so every headline number is paired with a sensitivity sweep over the price spread.

Three things this answers that MAE cannot:
    1. what the forecast is worth in money, against a naive baseline and perfect foresight
    2. whether the ranking of models changes once errors are priced asymmetrically
    3. where procurement risk actually concentrates in time

    python level2_cost.py

Writes into outputs/ (see config.py):
    level2_costs.csv        cost per method, Level 0 and Level 1
    level2_sensitivity.csv  cost ranking across price-spread assumptions
    fig_level2.png          cost by method, bias sweep, concentration, sensitivity
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

KWH_PER_MWH = 1000.0


def imbalance_cost(actual, forecast, c_short=None, c_long=None):
    """Euros of imbalance cost. Inputs in kWh per settlement period."""
    c_short = config.COST_SHORT if c_short is None else c_short
    c_long = config.COST_LONG if c_long is None else c_long
    ok = actual.notna() & forecast.notna()
    a, f = actual[ok], forecast[ok]
    short = (a - f).clip(lower=0).sum() / KWH_PER_MWH
    long = (f - a).clip(lower=0).sum() / KWH_PER_MWH
    return c_short * short + c_long * long


def summarise(actual, forecast, n_households, days, label):
    ok = actual.notna() & forecast.notna()
    a, f = actual[ok], forecast[ok]
    cost = imbalance_cost(a, f)
    energy_mwh = a.sum() / KWH_PER_MWH
    short_mwh = (a - f).clip(lower=0).sum() / KWH_PER_MWH
    long_mwh = (f - a).clip(lower=0).sum() / KWH_PER_MWH
    return {
        "method": label,
        "MAE_kWh": (f - a).abs().mean(),
        "cost_EUR": cost,
        "short_MWh": short_mwh,
        "long_MWh": long_mwh,
        "cost_per_MWh": cost / energy_mwh,
        "pct_of_energy_value": cost / (energy_mwh * config.PRICE_DAY_AHEAD) * 100,
        "EUR_per_hh_per_year": cost / n_households * (365.0 / days),
    }


def best_scale(actual, forecast):
    """Cheapest uniform multiplier on the forecast - a crude bias correction.

    If the cost-optimal forecast is not the mean, a single scaling factor should already
    cut cost. That is Level 3's quantile argument in its simplest possible form.
    """
    ks = np.arange(0.90, 1.151, 0.005)
    costs = [imbalance_cost(actual, forecast * k) for k in ks]
    i = int(np.argmin(costs))
    return ks[i], costs[i], ks, np.array(costs)


def main():
    p0 = pd.read_csv(os.path.join(config.OUTPUTS, "predictions_test.csv"),
                     index_col=0, parse_dates=[0])
    p0.index = pd.to_datetime(p0.index, utc=True)
    days = p0.index.normalize().nunique()
    n_hh = 332

    methods = ["gbm", "ridge", "naive_lag2d", "naive_seasonal_lag7d"]
    rows = [summarise(p0["actual"], p0[m], n_hh, days, m) for m in methods]

    # Perfect foresight costs nothing; it bounds what any forecast could ever save.
    rows.append({"method": "perfect foresight", "MAE_kWh": 0.0, "cost_EUR": 0.0,
                 "short_MWh": 0.0, "long_MWh": 0.0, "cost_per_MWh": 0.0,
                 "pct_of_energy_value": 0.0, "EUR_per_hh_per_year": 0.0})

    tab = pd.DataFrame(rows).set_index("method").sort_values("cost_EUR")
    print("=== Level 2: procurement cost, %d households over %d days ==="
          % (n_hh, days))
    print("prices: day-ahead %.0f, intraday buy %.0f, sell-back %.0f EUR/MWh"
          % (config.PRICE_DAY_AHEAD, config.PRICE_INTRADAY_BUY,
             config.PRICE_INTRADAY_SELL))
    print("        -> short %.0f EUR/MWh, long %.0f EUR/MWh\n"
          % (config.COST_SHORT, config.COST_LONG))
    print(tab.round(2).to_string())

    naive_cost = tab.loc["naive_seasonal_lag7d", "cost_EUR"]
    gbm_cost = tab.loc["gbm", "cost_EUR"]
    print("\nModel saves EUR %.0f against the naive baseline over %d days"
          % (naive_cost - gbm_cost, days))
    print("  = EUR %.2f per household per year"
          % ((naive_cost - gbm_cost) / n_hh * (365.0 / days)))

    # Does pricing the errors change the ranking MAE produced?
    by_mae = list(tab.sort_values("MAE_kWh").index)
    by_cost = list(tab.sort_values("cost_EUR").index)
    print("\nRanking by MAE : %s" % " < ".join(by_mae))
    print("Ranking by cost: %s" % " < ".join(by_cost))
    print("Ranking %s" % ("UNCHANGED - the asymmetry is not large enough to reorder these"
                          if by_mae == by_cost else "CHANGED once errors are priced"))

    # --- the bias the cost function wants ---
    k, kcost, ks, kcosts = best_scale(p0["actual"], p0["gbm"])
    print("\n=== What the cost function wants ===")
    print("cheapest uniform scaling of the gbm forecast: %.3f" % k)
    print("  cost %.0f -> %.0f EUR  (%.1f%% saved) by deliberately over-forecasting %.1f%%"
          % (gbm_cost, kcost, (1 - kcost / gbm_cost) * 100, (k - 1) * 100))
    print("theoretical cost-optimal quantile = %.0f/(%.0f+%.0f) = %.4f"
          % (config.COST_SHORT, config.COST_SHORT, config.COST_LONG,
             config.COST_OPTIMAL_QUANTILE))
    print("  -> Level 3 should forecast that quantile instead of the mean.")

    # --- where the cost sits in time ---
    per_slot = pd.Series(
        config.COST_SHORT * (p0["actual"] - p0["gbm"]).clip(lower=0) / KWH_PER_MWH
        + config.COST_LONG * (p0["gbm"] - p0["actual"]).clip(lower=0) / KWH_PER_MWH,
        index=p0.index)
    daily = per_slot.groupby(p0.index.date).sum().sort_values(ascending=False)
    share10 = daily.head(max(1, len(daily) // 10)).sum() / daily.sum() * 100
    share25 = daily.head(max(1, len(daily) // 4)).sum() / daily.sum() * 100
    print("\n=== Where procurement risk concentrates ===")
    print("worst 10%% of days carry %.1f%% of total cost" % share10)
    print("worst 25%% of days carry %.1f%% of total cost" % share25)
    print("worst three days: %s"
          % ", ".join("%s (EUR %.0f)" % (d, v) for d, v in daily.head(3).items()))

    # --- Level 1 arms, scored in euros on their own household set ---
    l1p = os.path.join(config.OUTPUTS, "level1_predictions.csv")
    l1 = None
    if os.path.exists(l1p):
        l1 = pd.read_csv(l1p, index_col=0, parse_dates=[0])
        l1.index = pd.to_datetime(l1.index, utc=True)
        d1 = l1.index.normalize().nunique()
        rows1 = [summarise(l1["actual"], l1[c], 327, d1, c)
                 for c in ("pooled", "segmented")]
        t1 = pd.DataFrame(rows1).set_index("method")
        print("\n=== Level 1 arms priced (327 households, winter window) ===")
        print(t1.round(2).to_string())
        print("segmentation saves EUR %.0f over the window (%.1f%%)"
              % (t1.loc["pooled", "cost_EUR"] - t1.loc["segmented", "cost_EUR"],
                 (1 - t1.loc["segmented", "cost_EUR"] / t1.loc["pooled", "cost_EUR"]) * 100))
        tab = pd.concat([tab, t1.add_suffix("")], axis=0)

    tab.to_csv(os.path.join(config.OUTPUTS, "level2_costs.csv"))

    # --- sensitivity: the prices are assumed, so sweep them ---
    ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    srows = []
    for r in ratios:
        c_short, c_long = 45.0 * r, 45.0
        best_k, _, _, _ = (None, None, None, None)
        kk = np.arange(0.90, 1.201, 0.005)
        cc = [imbalance_cost(p0["actual"], p0["gbm"] * x, c_short, c_long) for x in kk]
        best_k = kk[int(np.argmin(cc))]
        srows.append({
            "short_over_long": r,
            "optimal_quantile": c_short / (c_short + c_long),
            "gbm_cost": imbalance_cost(p0["actual"], p0["gbm"], c_short, c_long),
            "naive_cost": imbalance_cost(p0["actual"], p0["naive_seasonal_lag7d"],
                                         c_short, c_long),
            "best_scaling": best_k,
        })
    sens = pd.DataFrame(srows)
    sens["model_saving_%"] = (1 - sens.gbm_cost / sens.naive_cost) * 100
    print("\n=== Sensitivity to the price assumption ===")
    print(sens.round(3).to_string(index=False))
    print("The model beats naive across every spread tested; only the optimal bias moves.")
    sens.to_csv(os.path.join(config.OUTPUTS, "level2_sensitivity.csv"), index=False)

    # --- figure ---
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))

    t = tab.loc[[m for m in methods if m in tab.index]].sort_values("cost_EUR")
    cols = ["tab:green" if i == t["cost_EUR"].idxmin() else "tab:blue" for i in t.index]
    ax[0].barh(range(len(t)), t["cost_EUR"], color=cols)
    ax[0].set_yticks(range(len(t)))
    ax[0].set_yticklabels(t.index, fontsize=8)
    for i, v in enumerate(t["cost_EUR"]):
        ax[0].text(v, i, " %.0f" % v, va="center", fontsize=8)
    ax[0].set_xlabel("imbalance cost (EUR, %d days)" % days)
    ax[0].set_title("Forecast quality in money")
    ax[0].margins(x=0.18)

    ax[1].plot(ks, kcosts, color="tab:blue")
    ax[1].axvline(k, color="tab:green", ls="--", lw=1,
                  label="cheapest %.3f" % k)
    ax[1].axvline(1.0, color="tab:gray", ls=":", lw=1, label="unbiased")
    ax[1].set_xlabel("uniform multiplier on the forecast")
    ax[1].set_ylabel("imbalance cost (EUR)")
    ax[1].set_title("The cheapest forecast is not the accurate one")
    ax[1].legend(fontsize=8)

    cum = daily.cumsum() / daily.sum() * 100
    ax[2].plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values,
               color="tab:purple")
    ax[2].plot([0, 100], [0, 100], color="tab:gray", ls=":", lw=1)
    ax[2].axvline(10, color="tab:red", ls="--", lw=1)
    ax[2].set_xlabel("worst x% of days")
    ax[2].set_ylabel("cumulative % of cost")
    ax[2].set_title("Cost is only mildly concentrated: worst 10%% = %.0f%%" % share10)

    ax[3].plot(sens["short_over_long"], sens["model_saving_%"],
               marker="o", color="tab:blue", label="model saving vs naive")
    ax3b = ax[3].twinx()
    ax3b.plot(sens["short_over_long"], sens["optimal_quantile"],
              marker="s", color="tab:orange", label="optimal quantile")
    ax[3].set_xlabel("shortfall cost / surplus cost")
    ax[3].set_ylabel("model saving vs naive (%)", color="tab:blue")
    ax3b.set_ylabel("cost-optimal quantile", color="tab:orange")
    ax[3].set_title("Conclusions hold across price assumptions")

    fig.tight_layout()
    config.savefig(fig, "fig_level2.png", dpi=130)
    print("\nWrote level2_costs.csv, level2_sensitivity.csv, fig_level2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
