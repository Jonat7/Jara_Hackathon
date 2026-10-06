#!/usr/bin/env python3
"""Level 1: separate forecasting models for PV and non-PV households.

Takes the PV assignment from pv_detect.py (surveyed where known, predicted otherwise),
splits the panel in two, and forecasts each sub-portfolio with its own model. The summed
prediction is compared against a single pooled model over the identical household set -
so the comparison answers one question: does segmenting actually buy anything?

Both arms are built from the same households, the same features and the same chronological
split. Only the segmentation differs.

    python level1_segmented.py

Writes into outputs/ (see config.py):
    level1_metrics.csv       pooled vs segmented, and each group on its own
    level1_predictions.csv   actual and forecast per test slot, both approaches
    fig_level1.png           group profiles and the comparison
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
_sys.path.insert(0, _os.path.join(_ROOT, "data_prep"))
import config
import build_features as BF

START = pd.Timestamp(config.PANEL_START, tz="UTC")
END = pd.Timestamp(config.PANEL_END, tz="UTC") + pd.Timedelta(days=1)


def aggregate_groups(panel, groups):
    """One pass over the files, accumulating each household into its group and the total.

    `groups` maps Household_ID -> group name. Returns {group: DataFrame} plus "all".
    """
    index = pd.date_range(START, END - pd.Timedelta("15min"), freq="15min", tz="UTC")
    pos = pd.Series(np.arange(len(index)), index=index)
    names = sorted(set(groups.values())) + ["all"]
    tot = {g: np.zeros(len(index)) for g in names}
    cnt = {g: np.zeros(len(index), dtype=int) for g in names}

    ids = [h for h in panel["Household_ID"] if h in groups]
    for i, hid in enumerate(ids, 1):
        df = pd.read_csv(os.path.join(config.D15, "%s.csv" % hid), sep=";",
                         usecols=["Timestamp", "kWh_received_Total"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="ISO8601")
        df = df.dropna(subset=["kWh_received_Total"])
        df = df.drop_duplicates(subset="Timestamp", keep="first")
        df = df[(df["Timestamp"] >= START) & (df["Timestamp"] < END)]

        idx = pos.reindex(df["Timestamp"]).to_numpy()
        ok = ~np.isnan(idx)
        idx = idx[ok].astype(int)
        vals = df["kWh_received_Total"].to_numpy()[ok]
        for g in (groups[hid], "all"):
            np.add.at(tot[g], idx, vals)
            np.add.at(cnt[g], idx, 1)
        if i % 50 == 0:
            print("  read %d/%d households" % (i, len(ids)), flush=True)

    sizes = pd.Series(list(groups.values())).value_counts().to_dict()
    sizes["all"] = len(ids)
    out = {}
    for g in names:
        d = pd.DataFrame({"kwh_raw": tot[g], "n_reporting": cnt[g]}, index=index)
        d["kwh_total"] = np.where(d["n_reporting"] > 0,
                                  d["kwh_raw"] * sizes[g] / d["n_reporting"], np.nan)
        d.index.name = "Timestamp"
        out[g] = d
    return out, sizes


def fit_one(feat, feats, test_start, test_end):
    """Train on everything before test_start, predict [test_start, test_end)."""
    feat = feat[feat["kwh_total"].notna()]
    s = pd.Timestamp(test_start, tz="UTC")
    e = pd.Timestamp(test_end, tz="UTC")
    tr = feat[feat.index < s]
    te = feat[(feat.index >= s) & (feat.index < e)]
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=0)
    m.fit(tr[feats], tr["kwh_total"])
    return pd.Series(m.predict(te[feats]), index=te.index), te["kwh_total"]


def scores(y, p):
    ok = y.notna() & p.notna()
    y, p = y[ok], p[ok]
    e = p - y
    daily = pd.DataFrame({"y": y, "p": p}).groupby(y.index.date).sum()
    return {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAPE_%": (e.abs() / y).mean() * 100, "bias": e.mean(),
            "mean_load": y.mean(),
            "daily_MAPE_%": ((daily.p - daily.y).abs() / daily.y).mean() * 100}


def main():
    config.require_data()
    panel = pd.read_csv(os.path.join(config.OUTPUTS, "panel_households.csv"))
    pv = pd.read_csv(os.path.join(config.OUTPUTS, "pv_predictions.csv"),
                     index_col=0)
    groups = {int(h): ("pv" if bool(v) else "nopv")
              for h, v in pv["pv"].items()}
    print("Household groups: %d pv, %d nopv (of %d panel households)"
          % (sum(v == "pv" for v in groups.values()),
             sum(v == "nopv" for v in groups.values()), len(panel)))

    aggs, sizes = aggregate_groups(panel, groups)

    # Each group gets weather weighted by its OWN station mix.
    sub = {g: panel[panel["Household_ID"].isin(
               [h for h, gg in groups.items() if gg == g])]
           for g in ("pv", "nopv")}
    sub["all"] = panel[panel["Household_ID"].isin(groups)]

    feats = config.feature_list()
    feat = {g: BF.make_features(aggs[g], sub[g]) for g in ("pv", "nopv", "all")}

    # PV output is near zero in the winter test window, so judging segmentation only
    # there would understate it. Summer is where a PV split should earn its keep.
    WINDOWS = [("winter", config.TEST_START, "2024-02-28"),
               ("summer", "2023-06-01", "2023-09-01")]

    tables, keep = {}, None
    for label, s, e in WINDOWS:
        pred, actual = {}, {}
        for g in ("pv", "nopv", "all"):
            pred[g], actual[g] = fit_one(feat[g], feats, s, e)
        seg = pred["pv"].add(pred["nopv"], fill_value=np.nan)
        tot_actual = actual["all"]

        table = pd.DataFrame({
            "pooled (one model, all households)": scores(tot_actual, pred["all"]),
            "segmented (pv + nopv summed)": scores(tot_actual, seg),
            "  group: pv only": scores(actual["pv"], pred["pv"]),
            "  group: nopv only": scores(actual["nopv"], pred["nopv"]),
        }).T
        tables[label] = table

        pooled_mae = table.loc["pooled (one model, all households)", "MAE"]
        seg_mae = table.loc["segmented (pv + nopv summed)", "MAE"]
        delta = (1 - seg_mae / pooled_mae) * 100
        print("\n=== Level 1, %s window (%s .. %s) ===" % (label, s, e))
        print(table.round(2).to_string())
        print("segmented %.2f vs pooled %.2f kWh MAE  (%+.1f%%)  -> %s"
              % (seg_mae, pooled_mae, delta,
                 "segmentation helps" if delta > 1 else "no material gain"))

        if label == "winter":
            keep = pd.DataFrame({"actual": tot_actual, "pooled": pred["all"],
                                 "segmented": seg, "pv_actual": actual["pv"],
                                 "pv_pred": pred["pv"],
                                 "nopv_actual": actual["nopv"],
                                 "nopv_pred": pred["nopv"]})
            pooled_mae_w, seg_mae_w = pooled_mae, seg_mae

    out_tab = pd.concat(tables, names=["window"])
    out_tab.to_csv(os.path.join(config.OUTPUTS, "level1_metrics.csv"))
    keep.to_csv(os.path.join(config.OUTPUTS, "level1_predictions.csv"))
    pooled_mae, seg_mae = pooled_mae_w, seg_mae_w

    # --- figure ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    for g, c in (("pv", "tab:orange"), ("nopv", "tab:blue")):
        f = feat[g]
        f = f[f["kwh_total"].notna()]
        summer = f[f["month"].isin([5, 6, 7, 8])]
        prof = summer.groupby("slot_of_day")["kwh_total"].mean()
        prof = prof / prof.mean()
        ax[0].plot(prof.index, prof.values, color=c, label="%s (n=%d)" % (g, sizes[g]))
    ax[0].axvspan(40, 57, color="gold", alpha=0.15)
    ax[0].set_title("Summer profile, normalised (shaded: midday)")
    ax[0].set_xlabel("slot of day")
    ax[0].set_ylabel("load / own daily mean")
    ax[0].legend()

    sg = pd.read_csv(os.path.join(config.OUTPUTS, "pv_signatures.csv"), index_col=0)
    lab = sg[sg["Installation_HasPVSystem"].notna()]
    for v, c, nm in ((False, "tab:blue", "no PV"), (True, "tab:orange", "PV")):
        d = lab[lab["Installation_HasPVSystem"].astype(bool) == v]["summer_midday_ratio"]
        ax[1].hist(d.dropna(), bins=24, alpha=0.6, color=c, label=nm)
    ax[1].set_title("The detection signature (surveyed households)")
    ax[1].set_xlabel("summer midday load / shoulder load")
    ax[1].set_ylabel("households")
    ax[1].legend()

    wins = ["winter", "summer"]
    x = np.arange(len(wins))
    pooled_v = [tables[w].loc["pooled (one model, all households)", "MAE"] for w in wins]
    seg_v = [tables[w].loc["segmented (pv + nopv summed)", "MAE"] for w in wins]
    ax[2].bar(x - 0.2, pooled_v, width=0.4, color="tab:blue", label="pooled")
    ax[2].bar(x + 0.2, seg_v, width=0.4, color="tab:green", label="segmented")
    for i in range(len(wins)):
        gain = (1 - seg_v[i] / pooled_v[i]) * 100
        ax[2].text(i, max(pooled_v[i], seg_v[i]) + 0.2, "%+.1f%%" % gain,
                   ha="center", fontsize=9)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["winter window", "summer window"])
    ax[2].set_ylabel("MAE on the total (kWh per slot)")
    ax[2].set_title("Segmentation pays off only when PV generates")
    ax[2].legend()
    ax[2].margins(y=0.18)

    fig.tight_layout()
    config.savefig(fig, "fig_level1.png", dpi=130)
    print("\nWrote level1_metrics.csv, level1_predictions.csv, fig_level1.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
