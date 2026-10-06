#!/usr/bin/env python3
"""Select the balanced panel and aggregate it into one 15-minute series.

The 410 households enter and leave the dataset at different dates, so summing all of them
would show panel growth as if it were demand growth. We therefore keep only households with
unbroken coverage across the study window, and carry a per-slot count of how many actually
reported so that metering gaps can be corrected rather than learned.

    python build_panel.py

Writes into outputs/ (see config.py):
    panel_aggregate.csv    one row per 15-minute slot - the forecasting target
    panel_households.csv   who is in the panel, with their weather station
and prints diagnostics worth reading before trusting anything downstream.
"""
import os

import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

START = pd.Timestamp(config.PANEL_START, tz="UTC")
END = pd.Timestamp(config.PANEL_END, tz="UTC") + pd.Timedelta(days=1)


def select_panel():
    """Households whose 15-min coverage spans the whole window."""
    ov = pd.read_csv(os.path.join(config.META, "smart_meter_data_15min_overview.csv"),
                     sep=";")
    for col in ("SMD_15min_TimeAvailable_EarliestTimestamp",
                "SMD_15min_TimeAvailable_LatestTimestamp"):
        ov[col] = pd.to_datetime(ov[col], utc=True, format="ISO8601")

    if config.PANEL_MODE == "all":
        # Any household overlapping the window. n_reporting rescaling absorbs the
        # varying membership; see config.PANEL_MODE for what that assumes.
        keep = ov[
            (ov["SMD_15min_TimeAvailable_EarliestTimestamp"] < END)
            & (ov["SMD_15min_TimeAvailable_LatestTimestamp"] > START)
        ].copy()
    else:
        keep = ov[
            (ov["SMD_15min_TimeAvailable_EarliestTimestamp"] <= START)
            & (ov["SMD_15min_TimeAvailable_LatestTimestamp"] >= END - pd.Timedelta("15min"))
        ].copy()

    hh = pd.read_csv(os.path.join(config.META, "households.csv"), sep=";")
    keep = keep.merge(hh[["Household_ID", "Group", "Weather_ID",
                          "Installation_HasPVSystem"]],
                      on="Household_ID", how="left")
    return keep


def aggregate(panel):
    """Sum the panel onto a complete 15-minute calendar, counting contributors."""
    # Build the full calendar first: anything absent then shows as a hole rather than
    # silently not existing. This is how the missing 2023-10-29 was found.
    index = pd.date_range(START, END - pd.Timedelta("15min"), freq="15min", tz="UTC")
    total = np.zeros(len(index))
    count = np.zeros(len(index), dtype=int)
    pos = pd.Series(np.arange(len(index)), index=index)

    for i, hid in enumerate(panel["Household_ID"], 1):
        df = pd.read_csv(os.path.join(config.D15, "%s.csv" % hid), sep=";",
                         usecols=["Timestamp", "kWh_received_Total"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="ISO8601")
        df = df.dropna(subset=["kWh_received_Total"])
        # Defensive: none found in a 60-file sample, but a repeated timestamp
        # would be silently double-counted by the scatter-add below.
        df = df.drop_duplicates(subset="Timestamp", keep="first")
        df = df[(df["Timestamp"] >= START) & (df["Timestamp"] < END)]

        idx = pos.reindex(df["Timestamp"]).to_numpy()
        ok = ~np.isnan(idx)
        idx = idx[ok].astype(int)
        np.add.at(total, idx, df["kWh_received_Total"].to_numpy()[ok])
        np.add.at(count, idx, 1)

        if i % 50 == 0:
            print("  read %d/%d households" % (i, len(panel)), flush=True)

    out = pd.DataFrame({"Timestamp": index, "kwh_raw": total, "n_reporting": count})
    n_panel = len(panel)
    # Scale to full panel size so metering gaps do not look like demand drops. Assumes
    # absent households are average. NaN where nobody reported - zero would be a lie.
    out["kwh_total"] = np.where(out["n_reporting"] > 0,
                                out["kwh_raw"] * n_panel / out["n_reporting"],
                                np.nan)
    out["panel_size"] = n_panel
    return out


def main():
    config.require_data()
    panel = select_panel()
    print("Panel (%s): %d households covering %s..%s"
          % (config.PANEL_MODE, len(panel), config.PANEL_START, config.PANEL_END))
    print(panel["Installation_HasPVSystem"].value_counts(dropna=False).to_string())
    print("Weather stations:", panel["Weather_ID"].value_counts().to_dict())

    agg = aggregate(panel)

    n = agg["n_reporting"]
    print("\nReporting households per slot: min=%d p01=%.0f median=%.0f max=%d"
          % (n.min(), n.quantile(0.01), n.median(), n.max()))
    print("Slots below 95%% of panel: %.2f%%" % ((n < 0.95 * len(panel)).mean() * 100))
    print("Slots with zero reporters: %d" % (n == 0).sum())
    print("\nAggregate kWh/slot: mean=%.1f min=%.1f max=%.1f"
          % (agg["kwh_total"].mean(), agg["kwh_total"].min(), agg["kwh_total"].max()))

    panel.to_csv(os.path.join(config.OUTPUTS, "panel_households.csv"), index=False)
    agg.to_csv(os.path.join(config.OUTPUTS, "panel_aggregate.csv"), index=False)
    print("\nWrote panel_aggregate.csv (%d slots) to outputs/" % len(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
