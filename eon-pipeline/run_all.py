#!/usr/bin/env python3
"""Run the E.ON day-ahead forecasting pipeline end to end.

    python run_all.py                # panel -> features -> level 0
    python run_all.py --from level0  # start at a given stage

Stages, in order:

    panel     data_prep/build_panel.py      332 households -> one 15-min series
    features  data_prep/build_features.py   weather join, calendar, lags, split
    check     analysis/check_leakage.py     asserts no feature sees past gate closure
    level0    analysis/run_level0.py        baselines, model, metrics, figures
    backtest  analysis/backtest.py          rolling-origin folds + tuning split
    figures   analysis/make_figures.py      data, relationship, comparison, validation
    pv        analysis/pv_detect.py         classify PV owners from load shape
    level1    analysis/level1_segmented.py  separate models per group, pooled vs split
    level2    analysis/level2_cost.py       score the forecasts in euros, not kWh
    level3    analysis/level3_uncertainty.py quantile forecasts and interval width

Stops at the first failing stage. `panel` failing usually means the challenge data is not
where config.py expects it - the error message says where it looked. `check` failing means
a feature can see past gate closure, which invalidates every score downstream of it.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ('panel', ['data_prep/build_panel.py']),
    ('features', ['data_prep/build_features.py']),
    ('check', ['analysis/check_leakage.py']),
    ('level0', ['analysis/run_level0.py']),
    ('backtest', ['analysis/backtest.py']),
    ('figures', ['analysis/make_figures.py']),
    ('pv', ['analysis/pv_detect.py']),
    ('level1', ['analysis/level1_segmented.py']),
    ('level2', ['analysis/level2_cost.py']),
    ('level3', ['analysis/level3_uncertainty.py']),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', choices=[s for s, _ in STAGES],
                    help='start at this stage')
    args = ap.parse_args()

    stages = list(STAGES)
    if args.start:
        names = [s for s, _ in stages]
        stages = stages[names.index(args.start):]

    for name, cmd in stages:
        print('\n' + '=' * 70)
        print('STAGE: %s  (%s)' % (name, ' '.join(cmd)))
        print('=' * 70, flush=True)
        rc = subprocess.call([sys.executable] + cmd, cwd=HERE)
        if rc != 0:
            print('\nstage %r exited %d - stopping.' % (name, rc))
            return rc

    print('\nall stages complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
