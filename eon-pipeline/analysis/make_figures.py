#!/usr/bin/env python3
"""Presentation figures: the data, the key relationship, the models, the validation.

run_level0.py already writes the two model-output figures. This adds the ones that carry
the argument: what the data looks like, why temperature is the driver, how the methods
compare, and how performance moves across the year.

    python make_figures.py

Writes into outputs/ (see config.py):
    fig_data_overview.png        daily profile by season, monthly level, panel coverage
    fig_load_vs_temperature.png  the kink that justifies a nonlinear model, and residuals
    fig_model_comparison.png     MAE by method, permutation importance
    fig_backtest.png             per-fold accuracy and improvement, by season
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

SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
SCOL = {"winter": "tab:blue", "spring": "tab:green",
        "summer": "tab:orange", "autumn": "tab:brown",
        "shoulder": "tab:green"}


def load():
    df = pd.read_csv(os.path.join(config.OUTPUTS, "features.csv"),
                     index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df["kwh_total"].notna()]
    df["season"] = df["month"].map(SEASON)
    return df


def fig_data_overview(df):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    for s in ("winter", "spring", "summer", "autumn"):
        sub = df[df["season"] == s]
        prof = sub.groupby("slot_of_day")["kwh_total"].mean()
        ax[0].plot(prof.index, prof.values, label=s, color=SCOL[s])
    # Slots 40-43 = 10:00-11:00 UTC = 11:00-12:00 CET. Load drops 15-22% below trend
    # in winter and not at all in summer: a heat-pump blocking window (Sperrzeit), where
    # supply is contractually interrupted in exchange for a cheaper tariff.
    ax[0].axvspan(40, 44, color="tab:red", alpha=0.10)
    ax[0].set_title("Daily profile by season (shaded: heat-pump blocking window)")
    ax[0].set_xlabel("slot of day (0 = 00:00, 95 = 23:45)")
    ax[0].set_ylabel("kWh per 15-min slot (332 households)")
    ax[0].legend()

    months = df.index.tz_convert(None).to_period("M")
    m = df.groupby(months)["kwh_total"].mean()
    cols = [SCOL[SEASON[p.month]] for p in m.index]
    ax[1].bar(range(len(m)), m.values, color=cols)
    ax[1].set_xticks(range(len(m)))
    ax[1].set_xticklabels([str(p)[2:] for p in m.index], rotation=60, fontsize=8)
    ax[1].set_title("Monthly mean load: a 3.5x swing")
    ax[1].set_ylabel("kWh per 15-min slot")

    daily = df.groupby(df.index.date)["n_reporting"].mean()
    daily.index = pd.to_datetime(list(daily.index))
    # Reindex onto every calendar day so the absent day renders as a break in the line
    # rather than being silently bridged across.
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    ax[2].plot(daily.index, daily.values, color="tab:purple", lw=1)
    ax[2].axvline(pd.Timestamp("2023-10-29"), color="tab:red", ls="--", lw=1)
    ax[2].annotate("2023-10-29: no data\nin any household file",
                   xy=(pd.Timestamp("2023-10-29"), daily.max()),
                   xytext=(pd.Timestamp("2023-03-20"), daily.max() - 2.0),
                   fontsize=8, color="tab:red",
                   arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax[2].set_title("Reporting households per slot")
    ax[2].set_ylabel("households")
    ax[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    config.savefig(fig, "fig_data_overview.png", dpi=130)
    plt.close(fig)


def fig_load_vs_temperature(df):
    pred = pd.read_csv(os.path.join(config.OUTPUTS, "predictions_test.csv"),
                       index_col=0, parse_dates=[0])
    pred.index = pd.to_datetime(pred.index, utc=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    t, y = df["Temperature_avg_hourly"], df["kwh_total"]
    ok = t.notna() & y.notna()
    ax[0].scatter(t[ok], y[ok], s=2, alpha=0.06, color="tab:blue", edgecolors="none")
    bins = np.arange(np.floor(t[ok].min()), np.ceil(t[ok].max()) + 1, 1.0)
    mid = (bins[:-1] + bins[1:]) / 2
    binned = y[ok].groupby(pd.cut(t[ok], bins, labels=mid), observed=False).mean()
    ax[0].plot(binned.index.astype(float), binned.values, color="black", lw=2,
               label="1 degC binned mean")
    ax[0].axvline(config.HDD_BASE, color="tab:red", ls="--", lw=1,
                  label="HDD base (%g degC)" % config.HDD_BASE)
    ax[0].set_title("Load vs temperature: the kink a linear model cannot fit")
    ax[0].set_xlabel("temperature (degC)")
    ax[0].set_ylabel("kWh per 15-min slot")
    ax[0].legend()

    err = (pred["gbm"] - pred["actual"]).rename("err")
    te = df.loc[pred.index, "Temperature_avg_hourly"]
    ok2 = err.notna() & te.notna()
    ax[1].scatter(te[ok2], err[ok2], s=2, alpha=0.08, color="tab:blue", edgecolors="none")
    b2 = np.arange(np.floor(te[ok2].min()), np.ceil(te[ok2].max()) + 1, 1.0)
    m2 = (b2[:-1] + b2[1:]) / 2
    be = err[ok2].groupby(pd.cut(te[ok2], b2, labels=m2), observed=False).mean()
    ax[1].plot(be.index.astype(float), be.values, color="black", lw=2,
               label="binned mean error")
    ax[1].axhline(0, color="tab:red", ls="--", lw=1)
    ax[1].set_title("Forecast error vs temperature (test window)")
    ax[1].set_xlabel("temperature (degC)")
    ax[1].set_ylabel("forecast minus actual (kWh)")
    ax[1].legend()

    fig.tight_layout()
    config.savefig(fig, "fig_load_vs_temperature.png", dpi=130)
    plt.close(fig)


def fig_model_comparison():
    met = pd.read_csv(os.path.join(config.OUTPUTS, "metrics_level0.csv"), index_col=0)
    imp = pd.read_csv(os.path.join(config.OUTPUTS, "feature_importance.csv"),
                      index_col=0).iloc[:, 0]

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))

    best = met["MAE"].idxmin()
    met = met.sort_values("MAE", ascending=False)
    cols = ["tab:green" if i == best
            else ("tab:red" if "INFEASIBLE" in i else "tab:blue") for i in met.index]
    ax[0].barh(range(len(met)), met["MAE"].values, color=cols)
    ax[0].set_yticks(range(len(met)))
    ax[0].set_yticklabels(met.index, fontsize=9)
    for i, v in enumerate(met["MAE"].values):
        ax[0].text(v + 0.4, i, "%.2f" % v, va="center", fontsize=9)
    # The target is the SUMMED load of 332 households, so every kWh figure here is a
    # portfolio quantity. Said on the axis because a bare "MAE kWh" reads per-household.
    ax[0].set_xlabel("MAE, kWh per 15-min slot across all 332 households")
    ax[0].set_title("Error: portfolio mean load is 144.9 kWh/slot")
    ax[0].set_xlim(0, met["MAE"].max() * 1.15)

    # R-squared is scale-free, so unlike MAE it stays comparable across seasons.
    if "R2" in met.columns:
        ax[1].barh(range(len(met)), met["R2"].values, color=cols)
        ax[1].set_yticks(range(len(met)))
        ax[1].set_yticklabels(met.index, fontsize=9)
        for i, v in enumerate(met["R2"].values):
            ax[1].text(v + 0.01, i, "%.2f" % v, va="center", fontsize=9)
        ax[1].set_xlabel("R2 - higher is better")
        ax[1].set_title("Variance explained (scale-free)")
        ax[1].set_xlim(0, 1.08)

    top = imp.sort_values(ascending=True).tail(10)
    ax[2].barh(range(len(top)), top.values, color="tab:purple")
    ax[2].set_yticks(range(len(top)))
    ax[2].set_yticklabels(top.index, fontsize=9)
    ax[2].set_xlabel("MAE increase when shuffled (kWh)")
    ax[2].set_title("Permutation importance: temperature dominates")

    fig.tight_layout()
    config.savefig(fig, "fig_model_comparison.png", dpi=130)
    plt.close(fig)


def fig_backtest():
    bt = pd.read_csv(os.path.join(config.OUTPUTS, "backtest_folds.csv"))
    x = np.arange(len(bt))
    cols = [SCOL.get(s, "tab:gray") for s in bt["season"]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))

    ax[0].bar(x - 0.2, bt["naive7d_MAE"], width=0.4, color="tab:orange",
              label="naive lag-7d")
    ax[0].bar(x + 0.2, bt["gbm_MAE"], width=0.4, color="tab:blue", label="gbm")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(bt["fold"], rotation=45, fontsize=8)
    ax[0].set_ylabel("MAE (kWh per slot)")
    ax[0].set_title("Rolling-origin backtest: error per fold")
    ax[0].legend()

    ax[1].bar(x, bt["improvement_%"], color=cols)
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(bt["fold"], rotation=45, fontsize=8)
    ax[1].set_ylabel("improvement over naive (%)")
    ax[1].set_title("The model is a winter model")
    for i, v in enumerate(bt["improvement_%"]):
        ax[1].text(i, v + (1.5 if v >= 0 else -3.0), "%.0f" % v,
                   ha="center", va="bottom" if v >= 0 else "top",
                   fontsize=8)
    ax[1].margins(y=0.18)
    seasons = list(dict.fromkeys(bt["season"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=SCOL.get(s, "tab:gray"))
               for s in seasons]
    ax[1].legend(handles, seasons, fontsize=8)

    fig.tight_layout()
    config.savefig(fig, "fig_backtest.png", dpi=130)
    plt.close(fig)


NICE_NAME = {
    "gbm": "gradient boosting",
    "random_forest": "random forest",
    "ridge": "ridge regression",
    "linear_regression": "linear regression (OLS)",
    "naive_lag1d_INFEASIBLE": "naive, same slot 1 day back (not feasible)",
    "naive_lag2d": "naive, same slot 2 days back",
    "naive_seasonal_lag7d": "naive, same slot 7 days back",
}
# (source column, printed header, lower is better)
TABLE_COLS = [("MAE", "MAE kWh\n(all 332 households)", True),
              ("RMSE", "RMSE", True),
              ("MAPE_%", "MAPE %", True),
              ("R2", "R2", False),
              ("daily_MAPE_%", "daily MAPE %", True)]


def fig_model_table():
    """The results table rendered as an image, for slides and documents.

    The same numbers as metrics_level0.csv. Kept as a figure because a table pasted
    into a deck loses its formatting, and the best value per column is worth marking.
    """
    met = pd.read_csv(os.path.join(config.OUTPUTS, "metrics_level0.csv"), index_col=0)
    met = met.sort_values("MAE")

    header = ["model"] + [h for _, h, _ in TABLE_COLS]
    body, best_cells = [], set()
    for col_i, (src, _, lower_better) in enumerate(TABLE_COLS, start=1):
        if src not in met.columns:
            continue
        winner = met[src].idxmin() if lower_better else met[src].idxmax()
        best_cells.add((list(met.index).index(winner) + 1, col_i))
    for name, row in met.iterrows():
        cells = [NICE_NAME.get(name, name)]
        for src, _, _ in TABLE_COLS:
            cells.append("%.2f" % row[src] if src in met.columns else "-")
        body.append(cells)

    fig, ax = plt.subplots(figsize=(12.5, 0.46 * (len(body) + 1) + 1.5))
    ax.axis("off")
    # bbox makes the table fill its axes, so row height follows the figure height
    # instead of the font metrics - otherwise it floats with dead space beneath.
    tbl = ax.table(cellText=body, colLabels=header, cellLoc="left",
                   colWidths=[0.34, 0.17, 0.11, 0.11, 0.09, 0.14],
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.8)
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor("#ececec")
            cell.set_text_props(weight="bold")
        elif "not feasible" in body[r - 1][0]:
            cell.set_facecolor("#fdeeee")      # the forecast that could not be placed
        elif r == 1:
            cell.set_facecolor("#eaf5ea")      # best model
        if (r, c) in best_cells:
            cell.set_text_props(weight="bold")

    ax.set_title("Model comparison on the test window", fontweight="bold",
                 loc="left", pad=16, fontsize=13)
    fig.text(0.012, 0.02,
             "Test window 2023-12-15 to 2024-02-27, 7,200 slots. Every kWh figure is a "
             "portfolio total across all 332 households\n(mean load 144.9 kWh per 15-min "
             "slot), not a per-household value. Lower is better except R2.",
             fontsize=8.5, color="#444444", va="bottom")
    fig.tight_layout(rect=[0, 0.14, 1, 0.92])
    config.savefig(fig, "fig_comparison_model_table.png", dpi=160, facecolor="white")
    plt.close(fig)


def main():
    df = load()
    fig_data_overview(df)
    fig_load_vs_temperature(df)
    fig_model_comparison()
    fig_model_table()
    fig_backtest()
    print("Wrote 5 figures to outputs/:")
    for f in ("fig_data_overview.png", "fig_load_vs_temperature.png",
              "fig_model_comparison.png", "fig_comparison_model_table.png",
              "fig_backtest.png"):
        print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
