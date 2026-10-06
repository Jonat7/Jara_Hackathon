#!/usr/bin/env python3
"""Verify that no feature uses information unavailable at day-ahead gate closure.

The no-leakage claim is the one that decides whether the reported scores mean anything,
so it is asserted here rather than argued in prose. Checks:

    1. every same-slot lag equals the target that many days earlier (alignment)
    2. d1_morning_mean reproduces from D-1 morning slots only
    3. no feature in the strict set is derived from day D's own consumption
    4. the train/test boundary is a clean date cut with no overlap

    python check_leakage.py

Exits non-zero on any failure.
"""
import os

import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

SLOTS_PER_DAY = 96
fails = []


def check(name, ok, detail=""):
    print("  %-52s %s" % (name, "PASS" if ok else "FAIL"), ("  " + detail) if detail else "")
    if not ok:
        fails.append(name)


def main():
    df = pd.read_csv(os.path.join(config.OUTPUTS, "features.csv"),
                     index_col=0, parse_dates=[0])
    df.index = pd.to_datetime(df.index, utc=True)
    y = df["kwh_total"]

    print("1. Lag alignment (lag_Nd[t] must equal kwh_total[t - N days])")
    for days, col in ((2, "lag_2d"), (7, "lag_7d"), (1, "lag_1d")):
        want = y.reindex(df.index - pd.Timedelta(days=days)).to_numpy()
        got = df[col].to_numpy()
        both = ~np.isnan(want) & ~np.isnan(got)
        worst = np.abs(want[both] - got[both]).max() if both.any() else 0.0
        check("%s aligned on %d of %d comparable rows" % (col, both.sum(), len(df)),
              worst < 1e-9, "max abs diff %.2e" % worst)

    print("\n2. d1_morning_mean rebuilt from D-1 00:00-11:45 only")
    morning = y[df["slot_of_day"] < 48].groupby(lambda t: t.date()).mean()
    want = (pd.Series(df.index.date, index=df.index)
            - pd.Timedelta(days=1)).map(morning).to_numpy()
    got = df["d1_morning_mean"].to_numpy()
    both = ~np.isnan(want) & ~np.isnan(got)
    worst = np.abs(want[both] - got[both]).max() if both.any() else 0.0
    check("reproduces exactly", worst < 1e-9, "max abs diff %.2e" % worst)

    print("\n3. No strict feature is a function of day D's own consumption")
    # A feature built from day D's load would reproduce the target's within-day shape
    # far too well. Correlating each feature's daily profile against the target's is a
    # blunt but effective tripwire for an accidental same-day lag.
    feats = config.feature_list()
    same_day = []
    for col in feats:
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        s = df[col]
        if s.nunique() < 3:
            continue
        # Residual within-day variation, after removing each day's mean.
        fr = (s - s.groupby(df.index.date).transform("mean"))
        yr = (y - y.groupby(df.index.date).transform("mean"))
        ok = fr.notna() & yr.notna()
        if ok.sum() > 1000 and fr[ok].std() > 0:
            r = np.corrcoef(fr[ok], yr[ok])[0, 1]
            if abs(r) > 0.95:
                same_day.append((col, r))
    check("no feature tracks same-day load with |r| > 0.95", not same_day,
          str(same_day) if same_day else "")
    check("lag_1d excluded when STRICT_GATE_CLOSURE",
          (not config.STRICT_GATE_CLOSURE) or ("lag_1d" not in feats))

    print("\n4. Train/test split is a clean date cut")
    tr, te = df[df.is_test == 0].index, df[df.is_test == 1].index
    check("no timestamp on both sides", len(tr.intersection(te)) == 0)
    check("every train slot precedes every test slot", tr.max() < te.min(),
          "train ends %s, test starts %s" % (tr.max(), te.min()))
    check("test boundary matches config.TEST_START",
          te.min() >= pd.Timestamp(config.TEST_START, tz="UTC"))

    print("\n%s" % ("ALL CHECKS PASSED" if not fails
                    else "FAILED: %s" % ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
