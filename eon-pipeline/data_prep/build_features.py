#!/usr/bin/env python3
"""Join weather, engineer features, mark the chronological split.

Every feature here must be knowable at day-ahead gate closure (noon on D-1). The dataset
carries no weather forecasts, only actuals, so weather for day D stands in for a forecast:
that makes results optimistic and must be stated in the write-up.

Lags are computed on the COMPLETE calendar, before rows with no target are dropped. Drop
first and shift() would hop across the missing 2023-10-29, misaligning every lag after it.

    python build_features.py

Writes into outputs/ (see config.py):
    features.csv    one row per 15-minute slot: target, features, train/test flag
"""
import os

import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

SLOTS_PER_DAY = 96
# Levels interpolate between hourly readings; totals are split across the hour's slots.
INTERP = ["Temperature_avg_hourly", "DewPoint_hourly",
          "Humidity_avg_hourly", "WindSpeed_hourly"]
SPREAD = ["Sunshine_duration_hourly", "Precipitation_total_hourly"]


def panel_weather(panel, index):
    """Household-weighted mean across the stations the panel actually maps to.

    Three of the eight stations carry no sunshine or pressure data, so each variable is
    averaged only over the stations that have it, reweighted among themselves.
    """
    weights = panel["Weather_ID"].value_counts()
    frames, cols = {}, INTERP + SPREAD

    for wid, w in weights.items():
        df = pd.read_csv(os.path.join(config.WEATHER, "%s.csv" % wid), sep=";")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="ISO8601")
        df = df.set_index("Timestamp").sort_index()
        frames[wid] = (df.reindex(columns=cols).astype(float), w)

    hourly_index = pd.date_range(index[0].floor("h"), index[-1].ceil("h"),
                                 freq="h", tz="UTC")
    out = pd.DataFrame(index=hourly_index)
    for col in cols:
        num = pd.Series(0.0, index=hourly_index)
        den = pd.Series(0.0, index=hourly_index)
        for wid, (df, w) in frames.items():
            s = df[col].reindex(hourly_index)
            ok = s.notna()
            num = num.add((s.fillna(0.0) * w).where(ok, 0.0))
            den = den.add(pd.Series(float(w), index=hourly_index).where(ok, 0.0))
        out[col] = num / den.replace(0.0, np.nan)

    fine = out.reindex(out.index.union(index)).sort_index()
    fine[INTERP] = fine[INTERP].interpolate(method="time", limit_direction="both")
    # An hourly total copied to four slots must be quartered, or the hour is counted 4x.
    fine[SPREAD] = fine[SPREAD].ffill() / 4.0
    return fine.reindex(index)


def build():
    agg = pd.read_csv(os.path.join(config.OUTPUTS, "panel_aggregate.csv"),
                      parse_dates=["Timestamp"])
    agg["Timestamp"] = pd.to_datetime(agg["Timestamp"], utc=True)
    agg = agg.set_index("Timestamp").sort_index()
    panel = pd.read_csv(os.path.join(config.OUTPUTS, "panel_households.csv"))
    return make_features(agg, panel)


def make_features(agg, panel):
    """Feature table for any aggregate series, not just the full panel.

    `agg` is indexed by timestamp with kwh_total and n_reporting; `panel` supplies the
    Weather_ID mix used to weight the stations. Level 1 calls this once per household
    group, so the two sub-portfolios get features built exactly the same way.
    """
    df = agg[["kwh_total", "n_reporting"]].copy()
    df = df.join(panel_weather(panel, df.index))

    ts = df.index
    df["slot_of_day"] = ts.hour * 4 + ts.minute // 15
    df["day_of_week"] = ts.dayofweek
    df["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    df["month"] = ts.month
    # Day-of-year on a circle, so 31 Dec sits next to 1 Jan instead of 364 units away.
    doy = ts.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    temp = df["Temperature_avg_hourly"]
    # Heating demand switches on below a threshold and is flat above it.
    df["hdd"] = (config.HDD_BASE - temp).clip(lower=0.0)
    # Buildings have thermal mass: yesterday's cold still matters today.
    df["temp_roll24h"] = temp.rolling(SLOTS_PER_DAY, min_periods=4).mean()
    df["temp_day_mean"] = temp.groupby(ts.date).transform("mean")

    # Same-slot lags. 96 slots = 1 day, so a d-day lag is 96*d rows.
    df["lag_7d"] = df["kwh_total"].shift(7 * SLOTS_PER_DAY)
    df["lag_2d"] = df["kwh_total"].shift(2 * SLOTS_PER_DAY)
    df["lag_1d"] = df["kwh_total"].shift(1 * SLOTS_PER_DAY)
    df["lag7_roll4"] = pd.concat(
        [df["kwh_total"].shift(k * SLOTS_PER_DAY) for k in (7, 14, 21, 28)],
        axis=1).mean(axis=1)

    # Mean load over 00:00-11:45 of D-1: the freshest data available at gate closure.
    morning = df.loc[df["slot_of_day"] < 48, "kwh_total"].groupby(
        lambda t: t.date()).mean()
    prev_date = pd.Series(ts.date, index=ts) - pd.Timedelta(days=1)
    df["d1_morning_mean"] = prev_date.map(morning).to_numpy()

    df["is_test"] = (ts >= pd.Timestamp(config.TEST_START, tz="UTC")).astype(int)
    return df


def main():
    config.require_data()
    df = build()
    before = len(df)
    df = df[df["kwh_total"].notna()]
    print("Dropped %d slots with no panel data (the missing 2023-10-29)"
          % (before - len(df)))

    feats = config.feature_list()
    print("\nStrict gate closure: %s -> %d features"
          % (config.STRICT_GATE_CLOSURE, len(feats)))
    print("Train: %d slots  Test: %d slots"
          % ((df.is_test == 0).sum(), (df.is_test == 1).sum()))
    print("Test window: %s .. %s" % (df[df.is_test == 1].index.min(), df.index.max()))

    # Expected: first week has no history, plus fallout from the missing October day.
    miss = df[feats].isna().mean()
    print("\nFeature missingness (non-zero only):")
    print((miss[miss > 0] * 100).round(2).to_string() or "  none")

    print("\nTarget by season (kWh/slot):")
    print(df.groupby(df.index.month)["kwh_total"].mean().round(1).to_string())

    df.to_csv(os.path.join(config.OUTPUTS, "features.csv"))
    print("\nWrote features.csv to outputs/  shape=%s" % (df.shape,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
