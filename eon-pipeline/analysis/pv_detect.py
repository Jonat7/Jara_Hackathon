#!/usr/bin/env python3
"""Level 1b: identify PV owners from their consumption pattern alone.

The dataset carries no generation signal - no kWh_returned, no PV column. A rooftop system
shows up only as *suppressed net consumption* around midday on sunny days, floored at zero
because export is invisible here. So the detectable signature is a midday dip that deepens
with sunshine and is absent in winter.

139 of the 332 panel households have no PV flag; 193 are labelled (99 yes, 94 no). The
labelled households train and validate a classifier, which then labels the unknown 139.

Validation is stratified k-fold over HOUSEHOLDS, not over time: this is a cross-sectional
classification, so the temporal rules that govern the forecasting task do not apply. What
does apply is that no household may appear in both train and test folds.

    python pv_detect.py

Writes into outputs/ (see config.py):
    pv_signatures.csv    per-household signature features
    pv_predictions.csv   label (known or predicted), probability, and source
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

START = pd.Timestamp(config.PANEL_START, tz="UTC")
END = pd.Timestamp(config.PANEL_END, tz="UTC") + pd.Timedelta(days=1)

MIDDAY = range(40, 57)      # 10:00-13:45 UTC, solar noon in central Europe
MORNING = range(24, 36)     # 06:00-08:45
EVENING = range(68, 80)     # 17:00-19:45
NIGHT = list(range(0, 16)) + list(range(88, 96))   # 22:00-04:00
SUMMER = (4, 5, 6, 7, 8, 9)
WINTER = (11, 12, 1, 2)


def household_signature(hid, sun_daily):
    """Shape features for one household. Returns None if coverage is too thin."""
    df = pd.read_csv(os.path.join(config.D15, "%s.csv" % hid), sep=";",
                     usecols=["Timestamp", "kWh_received_Total"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="ISO8601")
    df = df.dropna(subset=["kWh_received_Total"])
    df = df[(df["Timestamp"] >= START) & (df["Timestamp"] < END)]
    if len(df) < 10000:
        return None

    ts = df["Timestamp"]
    slot = ts.dt.hour * 4 + ts.dt.minute // 15
    month = ts.dt.month
    date = ts.dt.date
    y = df["kWh_received_Total"].to_numpy()

    def mean_of(mask):
        return y[mask].mean() if mask.any() else np.nan

    s_sum = month.isin(SUMMER).to_numpy()
    s_win = month.isin(WINTER).to_numpy()
    mid = slot.isin(MIDDAY).to_numpy()
    mor = slot.isin(MORNING).to_numpy()
    eve = slot.isin(EVENING).to_numpy()
    nig = slot.isin(NIGHT).to_numpy()

    sig = {}
    # Core signature: midday net load relative to the shoulders either side of it.
    # A PV owner's own generation eats the midday peak; a non-owner's does not.
    shoulder_s = np.nanmean([mean_of(s_sum & mor), mean_of(s_sum & eve)])
    sig["summer_midday_ratio"] = mean_of(s_sum & mid) / shoulder_s
    shoulder_w = np.nanmean([mean_of(s_win & mor), mean_of(s_win & eve)])
    sig["winter_midday_ratio"] = mean_of(s_win & mid) / shoulder_w
    # The dip should be a summer phenomenon. Dividing the two cancels anything
    # structural about the household that applies all year.
    sig["midday_season_contrast"] = (sig["summer_midday_ratio"]
                                     / sig["winter_midday_ratio"])
    sig["summer_midday_vs_night"] = mean_of(s_sum & mid) / mean_of(s_sum & nig)

    # Does the midday dip deepen on sunny days? Generation scales with irradiance,
    # so a PV owner's midday ratio should correlate negatively with sunshine.
    d = pd.DataFrame({"date": date, "slot": slot, "y": y, "summer": s_sum})
    d = d[d["summer"]]
    daily_mid = d[d["slot"].isin(MIDDAY)].groupby("date")["y"].mean()
    daily_base = d[d["slot"].isin(list(MORNING) + list(EVENING))].groupby("date")["y"].mean()
    ratio = (daily_mid / daily_base).replace([np.inf, -np.inf], np.nan).dropna()
    sun = sun_daily.reindex(ratio.index).astype(float)
    ok = sun.notna() & ratio.notna()
    sig["sun_ratio_corr"] = (np.corrcoef(sun[ok], ratio[ok])[0, 1]
                             if ok.sum() > 30 else np.nan)
    # Sunniest-quartile days versus dullest-quartile days, same household.
    if ok.sum() > 30:
        q1, q3 = sun[ok].quantile(0.25), sun[ok].quantile(0.75)
        sig["sunny_vs_dull_ratio"] = (ratio[ok][sun[ok] >= q3].mean()
                                      / ratio[ok][sun[ok] <= q1].mean())
    else:
        sig["sunny_vs_dull_ratio"] = np.nan

    sig["summer_mean"] = mean_of(s_sum)
    sig["winter_summer_ratio"] = mean_of(s_win) / mean_of(s_sum)
    return sig


def daily_sunshine():
    """Mean daily sunshine across stations - a proxy for how sunny each day was."""
    frames = []
    for f in sorted(os.listdir(config.WEATHER)):
        w = pd.read_csv(os.path.join(config.WEATHER, f), sep=";")
        w["Timestamp"] = pd.to_datetime(w["Timestamp"], utc=True, format="ISO8601")
        if "Sunshine_duration_hourly" in w.columns:
            frames.append(w[["Timestamp", "Sunshine_duration_hourly"]])
    allw = pd.concat(frames)
    allw = allw.dropna(subset=["Sunshine_duration_hourly"])
    return allw.groupby(allw["Timestamp"].dt.date)["Sunshine_duration_hourly"].sum()


FEATURES = ["summer_midday_ratio", "winter_midday_ratio", "midday_season_contrast",
            "summer_midday_vs_night", "sun_ratio_corr", "sunny_vs_dull_ratio",
            "summer_mean", "winter_summer_ratio"]


def main():
    config.require_data()
    panel = pd.read_csv(os.path.join(config.OUTPUTS, "panel_households.csv"))
    sun = daily_sunshine()

    rows = []
    for i, hid in enumerate(panel["Household_ID"], 1):
        sig = household_signature(hid, sun)
        if sig is None:
            continue
        sig["Household_ID"] = hid
        rows.append(sig)
        if i % 50 == 0:
            print("  signed %d/%d households" % (i, len(panel)), flush=True)

    sg = pd.DataFrame(rows).set_index("Household_ID")
    sg = sg.join(panel.set_index("Household_ID")["Installation_HasPVSystem"])
    sg.to_csv(os.path.join(config.OUTPUTS, "pv_signatures.csv"))

    lab = sg[sg["Installation_HasPVSystem"].notna()].copy()
    unk = sg[sg["Installation_HasPVSystem"].isna()].copy()
    ylab = lab["Installation_HasPVSystem"].astype(bool).astype(int)
    print("\nLabelled %d (%d PV, %d no PV), unlabelled %d"
          % (len(lab), ylab.sum(), (1 - ylab).sum(), len(unk)))

    print("\nSignature means by label (labelled households only):")
    print(lab.groupby("Installation_HasPVSystem")[FEATURES].mean().round(3).to_string())

    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                         max_leaf_nodes=15, min_samples_leaf=10,
                                         l2_regularization=1.0, random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    prob = cross_val_predict(clf, lab[FEATURES], ylab, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    print("\n=== Cross-validated detection (5-fold over households) ===")
    print("accuracy %.3f    ROC AUC %.3f" % (accuracy_score(ylab, pred),
                                             roc_auc_score(ylab, prob)))
    cm = confusion_matrix(ylab, pred)
    print("confusion matrix [rows actual no/yes, cols predicted no/yes]:")
    print(cm)
    base = max(ylab.mean(), 1 - ylab.mean())
    print("majority-class baseline accuracy: %.3f" % base)

    clf.fit(lab[FEATURES], ylab)
    out = pd.DataFrame(index=sg.index)
    out["known_label"] = sg["Installation_HasPVSystem"]
    out["prob_pv"] = np.nan
    out.loc[lab.index, "prob_pv"] = prob
    if len(unk):
        out.loc[unk.index, "prob_pv"] = clf.predict_proba(unk[FEATURES])[:, 1]
    out["source"] = np.where(out["known_label"].notna(), "surveyed", "predicted")
    out["pv"] = np.where(out["known_label"].notna(),
                         out["known_label"].astype("boolean").fillna(False),
                         out["prob_pv"] >= 0.5)
    out.to_csv(os.path.join(config.OUTPUTS, "pv_predictions.csv"))

    print("\nFinal assignment across the panel:")
    print(out.groupby(["source", "pv"]).size().to_string())
    print("\nWrote pv_signatures.csv and pv_predictions.csv to outputs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
