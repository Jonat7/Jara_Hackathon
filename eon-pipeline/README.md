# E.ON day-ahead load forecasting — RWTH hackathon

Forecasting engine for the **day-ahead procurement stage**: at noon on day D-1, predict all
96 fifteen-minute consumption slots of day D for a portfolio of heat-pump households, so the
right volume can be bought on the day-ahead market.

Challenge repo: https://github.com/ArsamAryandoust/rwth_hackathon

## Getting the data

The challenge data is ~1.7 GB and is **not** kept in this folder, because this tree sits
inside Sciebo and would sync. Either clone it next to this pipeline:

    cd "C:\Users\adama\Sciebo\PhD RWTH Aachen\Summer Schools\JARA Energy"
    git clone https://github.com/ArsamAryandoust/rwth_hackathon.git

(then add it to Sciebo's exclusion list), or keep it on a local disk and point the pipeline
at it:

    $env:EON_HACKATHON_DATA = "D:\data\rwth_hackathon"     # PowerShell

## Running

    cd eon-pipeline
    python run_all.py

About two minutes, most of it reading 332 CSVs. `python run_all.py --from level0` skips
straight to modelling once `outputs/features.csv` exists. Needs pandas, numpy,
scikit-learn and matplotlib.

## Layout

    eon-pipeline/
        config.py          every path and every experiment choice; scripts import it
        run_all.py         stages in order, stops at first failure
        data_prep/
            build_panel.py      332 households -> one 15-min series
            build_features.py   weather join, calendar, lags, chronological split
        analysis/
            check_leakage.py    asserts no feature sees past gate closure
            pv_detect.py        Level 1b: classify PV owners from load shape
            level1_segmented.py Level 1: per-group models vs pooled
            level2_cost.py      Level 2: asymmetric procurement cost
            level3_uncertainty.py Level 3: quantile forecasts and intervals
            run_level0.py       baselines, model, metrics, figures
            backtest.py         rolling-origin folds + tuning split
        outputs/           derived; safe to delete and regenerate

Scripts never compute paths from their own location — they open with the three-line
bootstrap and `import config`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
```

New analysis goes in `analysis/`, follows the same bootstrap, and is added to `STAGES` in
`run_all.py` if it belongs in the standard run.

## What the pipeline does

**1. Balanced panel.** Households enter and leave the dataset at different dates, so summing
all 410 would show panel growth as demand growth. Keeping only those with unbroken coverage
over 2023-02-01 → 2024-02-27 leaves **332 households**. A per-slot count of reporting
households is carried alongside, and the total is scaled by `332 / n_reporting` so metering
gaps are corrected rather than learned.

**2. Features.** Weather is averaged across the 8 stations weighted by how many panel
households map to each, skipping variables a station lacks. Hourly weather is upsampled to
15 minutes with two rules: *levels* (temperature, humidity, wind) interpolate; *totals*
(sunshine, precipitation) are divided by four. Then calendar terms, heating degrees, a
24-hour rolling temperature, and same-slot lags at D-2, D-7 and a 4-week mean.

**3. Level 0.** Three naive baselines, a Ridge and a gradient-boosted tree, all scored on
the held-out window with a chronological split.

**4. Validation.** A leakage test that fails the build, and a rolling-origin backtest over
four monthly folds, plus a validation split for tuning that never touches the test window.

## Results (test window 2023-12-15 → 2024-02-27)

| model | MAE | RMSE | MAPE % | R2 | daily MAPE % |
|---|---|---|---|---|---|
| **LSTM (5-seed ensemble)** | **10.90** | **14.43** | **6.64** | **0.92** | **3.24** |
| gbm | 11.97 | 15.59 | 7.10 | 0.91 | 3.88 |
| random forest | 14.52 | 19.01 | 9.03 | 0.86 | 4.43 |
| ridge | 15.04 | 18.96 | 9.70 | 0.86 | 5.25 |
| linear regression (OLS) | 15.04 | 18.96 | 9.70 | 0.86 | 5.25 |
| naive lag-1d *(infeasible)* | 21.91 | 28.38 | 13.35 | 0.70 | 7.47 |
| naive lag-2d | 29.36 | 38.46 | 17.85 | 0.44 | 12.07 |
| naive seasonal lag-7d | 36.78 | 47.44 | 21.44 | 0.15 | 17.43 |

Panel is **407 households** (`PANEL_MODE = "all"`): every household with data in the
window. Three stopped reporting before it opens and cannot be included at all.

The LSTM wins across five seeds (per-seed mean 11.26, sd 0.18, range 10.96-11.42 - every
seed beat the gbm). Caveat: it early-stops on a chronological validation slice while the
gbm runs fixed iterations, so a tuned model is being compared with an untuned one.

Plain OLS and ridge are identical to two decimal places. With 17 features and 30,336
training rows there is nothing for an L2 penalty to prevent, so the regularisation buys
nothing here - worth knowing rather than assuming it helps.

**67.5% better than seasonal naive on this window**; daily procured volume within 3.7%.
That single-window figure is a winter result and is not representative of the year - see
the backtest below, where summer performance collapses to roughly break-even.

### Rolling-origin backtest

One split is one number. Refitting at nine successive origins, each fold training only on
data before it:

| fold | season | train slots | gbm MAE | naive MAE | mean load | MAPE % | improvement % |
|---|---|---|---|---|---|---|---|
| 2023-06 | summer | 11,520 | 5.10 | 4.71 | 48.3 | 10.56 | **-8.2** |
| 2023-07 | summer | 14,400 | 4.29 | 4.91 | 46.7 | 9.19 | 12.7 |
| 2023-08 | summer | 17,376 | 5.69 | 7.60 | 51.9 | 10.97 | 25.2 |
| 2023-09 | shoulder | 20,352 | 5.99 | 8.20 | 53.0 | 11.29 | 27.0 |
| 2023-10 | shoulder | 23,232 | 8.10 | 15.18 | 73.3 | 11.05 | 46.6 |
| 2023-11 | winter | 26,112 | 14.27 | 23.79 | 130.2 | 10.95 | 40.0 |
| 2023-12 | winter | 28,992 | 11.12 | 34.64 | 156.1 | 7.13 | 67.9 |
| 2024-01 | winter | 31,968 | 11.76 | 42.47 | 167.0 | 7.04 | 72.3 |
| 2024-02 | winter | 34,944 | 7.95 | 18.74 | 117.6 | 6.76 | 57.6 |

| season | folds | mean load | gbm MAE | MAPE % | improvement % |
|---|---|---|---|---|---|
| summer | 3 | 49.0 | 5.03 | 10.24 | **9.9** |
| shoulder | 2 | 63.2 | 7.04 | 11.17 | 36.8 |
| winter | 4 | 142.7 | 11.27 | 7.97 | **59.5** |

**This is the most important result in the Level 0 work, and it is not the headline number.**

The model is a winter model. In winter it beats the naive baseline by ~60%. In summer it
beats it by ~10%, and in June 2023 it is **8% worse than doing nothing**. The reason is
structural: temperature is the dominant feature (10.18 kWh of permutation importance), and
in summer there is no heating load for temperature to explain. What remains is a low, flat
profile whose weekly pattern the lag-7d naive already captures.

Note MAPE is *worse* in summer (10.2%) than winter (8.0%) despite the much lower absolute
error, because summer load is a third of winter load.

One confound to state: the early folds have both summer weather and the least training
history (June trains on 11,520 slots, February on 34,944), so season and sample size cannot
be fully separated here. The direction is clear, the magnitude is not precise.

For day-ahead procurement this may be acceptable - winter is where the volume and the money
are - but it must be said out loud rather than buried under an annual average. Across all
nine folds the mean improvement is 37.9%, which is a number that describes no actual season.

### Tuning split

`backtest.py` also carves a chronological validation fold (2023-10-15 to 2023-12-15) out of
the training data, so hyperparameters can be chosen without touching the test window.
Validation MAE at the current settings is 16.41 - substantially worse than the test figure,
because that window is the autumn transition with less history behind it. Tune against that
number, not against the test set.

**The current hyperparameters are untuned** - sensible defaults, chosen by hand, never
optimised. That is a deliberate position (nothing has been fitted to the test window), not
a finished tuning exercise.

Random forest is included because the challenge's reading list points at it. It lands
between ridge and the gbm, **23% worse than gradient boosting**. Both are tree ensembles, so
the gap is boosting against bagging: boosting fits each tree to the running error and keeps
pushing into the sparsely-sampled cold end, while a forest averages leaf values it has
already seen and cannot extrapolate past them. The test window is colder than most of
training, which is where that distinction bites.

Permutation importance puts **temperature first (10.18 kWh)**, ahead of `slot_of_day` (6.08)
and `hdd` (4.21), with the first lag only fourth (3.17). That is the justification for
treating this as tabular regression on weather rather than as a classical time-series
problem: ARIMA and friends see only the series' own past, and here the dominant driver is
outside it.

## Figures

Six figures, written by `run_level0.py` and `make_figures.py`:

| file | what it shows |
|---|---|
| `fig_data_overview.png` | daily profile by season, monthly level, panel coverage |
| `fig_load_vs_temperature.png` | the kink that justifies a nonlinear model; residuals vs temperature |
| `fig_model_comparison.png` | MAE by model, R2 by model, permutation importance |
| `fig_comparison_model_table.png` | the full results table rendered as an image, for slides |
| `fig_backtest.png` | per-fold error and improvement, coloured by season |
| `fig_example_days.png` | coldest and mildest test day, actual vs model vs naive |
| `fig_error_structure.png` | error across the day; daily procured volume |
| `fig_level1.png` | PV vs non-PV summer profiles; detection signature; segmentation gain |
| `fig_level2.png` | cost by method; bias sweep; cost concentration; price sensitivity |
| `fig_level3.png` | cost vs quantile; calibration; interval width by hour; fan chart |

## A structural feature found in the daily profile

Slots 40-43 (10:00-11:00 UTC, **11:00-12:00 CET**) sit **15-22% below the surrounding
trend in winter**, recovering completely at slot 44. Row counts across those slots are
identical, so it is not a data-availability artifact. In summer the same slots are slightly
*above* their neighbours - the dip is absent entirely.

A fixed daily hour where load drops sharply, but only when there is heating demand to
interrupt, is the signature of a **heat-pump blocking window** (Sperrzeit): supply is
contractually interrupted at contracted hours in exchange for a cheaper tariff. It is
shaded in `fig_data_overview.png`.

Worth knowing for three reasons: it is deterministic and calendar-driven, so it is
genuinely predictable; it explains part of why `slot_of_day` ranks second in importance
(the tree can already learn it from that feature, so an explicit flag would likely add
little accuracy); and the rebound at slot 44 is a real procurement feature, not noise.

## Level 1: PV detection and segmented forecasting

### 1b - identifying PV owners from load shape alone

The dataset has no generation signal. A rooftop system is visible only as **suppressed net
consumption around midday on sunny days**, floored at zero because export is invisible here.
`pv_detect.py` builds a per-household signature from that: summer midday load relative to
the morning and evening shoulders, the same ratio in winter as a control, how the ratio
responds to sunshine, and the winter/summer level ratio.

The separation is stark. Mean summer midday-to-shoulder ratio:

| | surveyed no PV | surveyed PV |
|---|---|---|
| summer midday ratio | 0.837 | **0.412** |
| midday season contrast | 1.09 | **0.70** |
| correlation with sunshine | +0.054 | **-0.195** |

Classifier: gradient-boosted trees, 5-fold stratified cross-validation **over households**
(this is cross-sectional, so no household appears in both folds; the temporal rules that
govern forecasting do not apply here).

**Accuracy 0.858, ROC AUC 0.909**, against a majority-class baseline of 0.516. Confusion
matrix: 82/92 non-PV and 81/98 PV correct. It then labels the 137 unsurveyed households -
22 PV, 115 not.

Worth flagging: the surveyed households are 52% PV, the unsurveyed are predicted at only
16%. Either the unsurveyed genuinely differ, or the classifier is conservative off its
training distribution. With no labels there, this cannot be resolved - state it rather than
assume the split is right.

### 1a/1b - does segmenting actually improve the forecast?

`level1_segmented.py` splits the 327 households into PV and non-PV, forecasts each
sub-portfolio with its own model, sums the two, and compares against a single pooled model
over the identical household set. Same features, same split, same hyperparameters - only
the segmentation differs.

| window | pooled MAE | segmented MAE | gain | pv group MAPE | nopv group MAPE |
|---|---|---|---|---|---|
| winter (Dec 15 - Feb 27) | 9.76 | 9.69 | **+0.8%** | 15.66% | 6.21% |
| summer (Jun - Aug) | 5.66 | 5.14 | **+9.1%** | 23.51% | 9.67% |

**Segmentation pays off only when PV is generating.** In winter the split is worth nothing -
a rooftop system produces almost nothing, so the two groups behave alike and a pooled model
already captures the mix. In summer it is worth 9.1% on MAE and 22% on daily volume error
(6.33% vs 8.08%).

Judging this on the Level 0 test window alone would have concluded that segmentation does
not work. It does - the winter window simply cannot show it.

The group figures also confirm the premise: **PV households are roughly 2.4x harder to
forecast per unit of load** (23.5% vs 9.7% MAPE in summer). Their absolute error is smaller
only because they consume less.

## Level 2: scoring the forecast in euros

MAE prices a 10 kWh over-forecast and a 10 kWh under-forecast identically. A procurement
desk does not: a shortfall is bought back on intraday above the day-ahead price, a surplus
is sold below it. So the business metric is asymmetric:

    cost = COST_SHORT * max(0, actual - forecast) + COST_LONG * max(0, forecast - actual)

Assumed prices (in `config.py`, as the challenge permits): day-ahead 100, intraday buy 150,
sell-back 60 EUR/MWh, giving **50 EUR/MWh for being short and 40 for being long**.

### What the forecast is worth

332 households, 75-day test window:

| method | MAE kWh | cost EUR | short MWh | long MWh | EUR/household/year |
|---|---|---|---|---|---|
| perfect foresight | 0.00 | 0 | 0.0 | 0.0 | 0.00 |
| **gbm** | 9.90 | **3,253** | 40.1 | 31.2 | **47.68** |
| ridge | 12.59 | 3,993 | 36.8 | 53.9 | 58.53 |
| naive lag-2d | 24.40 | 7,876 | 84.8 | 90.9 | 115.45 |
| naive lag-7d | 30.44 | 9,782 | 101.5 | 117.7 | 143.38 |

The model saves **EUR 6,529 over 75 days against the naive baseline - EUR 95.70 per
household per year**. Imbalance cost is 3.12% of the energy value procured, against 9.37%
for the naive baseline.

### Does pricing change the ranking?

**No.** Ordering by cost reproduces the ordering by MAE exactly. A 50/40 asymmetry is not
severe enough to reorder methods this far apart. Reporting that honestly is better than
implying the business metric overturned the statistical one - what it adds is *magnitude*,
not a different winner.

The short/long split is still informative: ridge over-forecasts badly (53.9 MWh long against
36.8 short) while the gbm leans the other way (40.1 short, 31.2 long).

### What the cost function wants

The cheapest uniform multiplier on the gbm forecast is **1.020** - deliberately
over-forecasting by 2% saves 2.6% of cost. Two effects push the same way: the model's own
negative bias, and the fact that being short costs more than being long.

This is the Level 3 argument in its crudest form. The cost function above is a pinball loss,
and its minimiser is not the mean but the quantile **50/(50+40) = 0.556**. Level 3 should
forecast that quantile directly rather than scaling a mean forecast.

### Where the risk sits

The worst 10% of days carry **18.3%** of total cost, the worst 25% carry 37.4%. That is only
mildly concentrated - a uniform spread would give 10% and 25%. The worst single day is
2024-01-20 at EUR 121, the coldest day in the window.

Caveat: the test window is entirely winter, so this measures concentration *within* the
heating season. Seasonal concentration across a full year is a separate question this
window cannot answer.

### Sensitivity

The prices are assumed, so the spread is swept from 0.5x to 4x:

| short/long | optimal quantile | best scaling | model saving vs naive |
|---|---|---|---|
| 0.5 | 0.333 | 0.970 | 69.6% |
| 1.0 | 0.500 | 1.010 | 67.5% |
| 2.0 | 0.667 | 1.045 | 65.3% |
| 4.0 | 0.800 | 1.080 | 63.4% |

**The model beats the naive baseline by 63-70% under every assumption tested.** Only the
optimal bias moves. Note that even at a symmetric 1.0 ratio the best scaling is 1.010, not
1.000 - that residual is the model's own cold-weather bias, not the price asymmetry.

### Level 1 arms priced

Segmentation saves EUR 2 over the winter window (0.1%), consistent with its MAE result.
Its value is a summer phenomenon and this window cannot show it.

## Level 3: uncertainty, and spending it

Level 2 showed the cost function is a pinball loss whose minimiser is the quantile 0.556,
not the mean. `level3_uncertainty.py` fits quantile models directly at ten quantiles and
asks three questions.

### Does bidding a quantile beat bidding the mean?

Yes, modestly. Cost falls from **EUR 3,253 to EUR 3,148 (-3.2%)** - while MAE gets *worse*
(9.97 vs 9.90 kWh). That is the Level 3 point in one line: **the cheapest forecast is not
the most accurate one**, because over- and under-forecasting are not equally expensive.

The empirical cost minimum is at quantile **0.650**, against a theoretical **0.556**. Close,
and the cost curve is flat between them, but not identical - the gap is explained by the
calibration problem below.

### Are the intervals calibrated?

**No, and this is the honest headline.** Mean absolute calibration gap is **6.0 percentage
points**, and it is systematic: low quantiles over-cover, high quantiles under-cover.

| nominal | empirical coverage |
|---|---|
| 0.05 | 12.4% |
| 0.10 | 17.8% |
| 0.50 | 47.8% |
| 0.90 | 80.3% |
| 0.95 | 86.0% |

The nominal **80% interval covers only 62.5%** of outcomes. The intervals are too narrow,
which matters: anyone sizing a risk buffer off them would be under-hedged.

The cause is the same extrapolation that produces the cold-day bias - the test window is
colder than most of training, so the model is confident in a regime it has barely seen.

### The recalibration attempt failed

The textbook fix is to learn the nominal-to-empirical mapping on a held-out calibration
window and request whatever nominal quantile delivers the coverage wanted. Implemented on an
autumn calibration window (2023-10-15 to 2023-12-15), it made things **worse**: the 80%
interval fell from 62.5% to 48.2% coverage, and cost rose from EUR 3,253 to EUR 4,294.

Split-conformal methods assume the calibration data is **exchangeable** with the test data.
A chronological split puts autumn in calibration and winter in test - different regimes - so
a correction learned on one misfires on the other. With 13 months of data there is no
same-season holdout to calibrate against.

This is reported as a negative result and the raw quantiles are the deliverable. A fix needs
either a second winter in the data, or a method that adapts online as the season progresses.

### Where is the model uncertain?

Not where I expected. The widest 80% intervals are at **slots 40-42, 10:00-10:45 UTC** -
**the heat-pump blocking window** - at 34.8 kWh, against 17.2 kWh at 03:45. The model knows
the block is coming but not exactly how deep it goes or how the rebound lands, and that
varies by household and day. The two discoveries in this project turn out to be the same
phenomenon seen from different angles.

By temperature the picture splits:

| temperature | interval width | width / load |
|---|---|---|
| below -5 degC | 28.4 kWh | **0.12** |
| -5 to 0 | 26.7 | 0.14 |
| 0 to 5 | 21.4 | 0.15 |
| 5 to 10 | 21.7 | 0.19 |
| above 10 | 19.6 | **0.22** |

Absolute uncertainty rises with cold; **relative** uncertainty falls. In cold weather the
heat pump dominates and behaves predictably; in mild weather what remains is discretionary
household behaviour, which is proportionally far noisier.

### How procurement would use this

Bid the cost-optimal quantile rather than the mean, and treat interval width as a
risk signal: wide intervals at the blocking window and on cold evenings are where intraday
exposure concentrates. The caveat is unavoidable though - the intervals are too narrow, so
widen them before sizing any buffer off them.

## Things that will bite you

- **Leakage is asserted, not assumed.** `analysis/check_leakage.py` runs as a pipeline
  stage and fails the build if any lag is misaligned, if `d1_morning_mean` cannot be
  rebuilt from D-1 mornings alone, if any strict feature tracks same-day load, or if the
  train/test cut overlaps. All currently pass, with exact lag alignment.
- **Never split randomly.** Test is the most recent slice, by date. This also applies to
  the model's own internals: `early_stopping` is off in `run_level0.py` because sklearn's
  validation split is random and would leak future slots into model selection.
- **2023-10-29 is missing from every household file** — 96 empty slots, the European DST
  fall-back date, dropped by the source export during a timezone conversion. The weather
  files have it. Lags are therefore computed on the complete calendar *before* empty rows
  are dropped; reverse that order and `shift()` hops the hole and misaligns every lag after
  October.
- **Gate closure.** `STRICT_GATE_CLOSURE = True` in `config.py` excludes the same-slot D-1
  lag, because for an evening slot that reading lands after the market closed. Flip it to
  measure what the optimistic assumption is worth.
- **No weather forecasts exist in this dataset**, only actuals, so actuals stand in for a
  forecast. This makes results optimistic and must be said out loud in the write-up.
- **No PV production signal** — no `kWh_returned`, no generation column. PV shows only as
  suppressed midday net consumption, and the PV flag is missing for 139 of the 332 panel
  households. That gap is what Level 1b is really about.

## Known limitation

The test window is the coldest part of the year, so the model partly extrapolates beyond
its training temperature range. It **under-forecasts the coldest day** (2024-01-20, −5.4 °C)
by roughly 30 kWh/slot at midday, and overall bias is −1.24 kWh/slot. Under-forecasting
means buying short and topping up on the expensive intraday market — the costly direction.

That is the bridge to the next levels: Level 2 scores models with an asymmetric cost
function instead of MAE, and Level 3 corrects the bias deliberately by forecasting a
cost-optimal quantile rather than the mean. Neither needs any change to `features.csv`.
