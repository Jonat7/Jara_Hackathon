r"""Paths and experiment settings for the E.ON day-ahead forecasting pipeline.

Every script imports from here rather than computing paths from its own location, so the
hackathon data can live anywhere without touching the analysis code.

Layout this assumes:

    JARA Energy/
        rwth_hackathon/            <- the E.ON challenge repo (data), read-only to us
        eon-pipeline/
            config.py              <- you are here
            data_prep/             <- panel selection and feature engineering
            analysis/              <- models and scoring
            outputs/               <- everything derived; safe to delete and regenerate

The data repo is ~1.7 GB and is NOT kept in this folder by default, because this tree sits
inside Sciebo and would sync. Either clone it next to this pipeline:

    git clone https://github.com/ArsamAryandoust/rwth_hackathon.git

or keep it anywhere you like and point this at it:

    set EON_HACKATHON_DATA=D:\somewhere\rwth_hackathon      (Windows, cmd)
    $env:EON_HACKATHON_DATA = "D:\somewhere\rwth_hackathon" (PowerShell)
"""
import os
import sys
import time

PIPELINE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(PIPELINE)

# --- where the challenge data lives -------------------------------------------------
REPO = os.environ.get("EON_HACKATHON_DATA") or os.path.join(PROJECT, "rwth_hackathon")
DATA = os.path.join(REPO, "data")
D15 = os.path.join(DATA, "15min")
META = os.path.join(DATA, "smart_meter_meta_data")
WEATHER = os.path.join(DATA, "weather_data_hourly")

# Everything the pipeline produces. Regenerable - nothing here is a source of truth.
OUTPUTS = os.path.join(PIPELINE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)


def require_data():
    """Fail early and legibly if the challenge repo is not where we think it is."""
    if not os.path.isdir(D15):
        sys.exit(
            "Cannot find the challenge data.\n"
            f"  looked for: {D15}\n\n"
            "Clone the repo next to this pipeline:\n"
            f"  cd \"{PROJECT}\"\n"
            "  git clone https://github.com/ArsamAryandoust/rwth_hackathon.git\n\n"
            "or point EON_HACKATHON_DATA at an existing copy (see config.py docstring)."
        )


# --- the choices that define the experiment -----------------------------------------

# Balanced panel: households with unbroken coverage across this whole window (332 of 410).
PANEL_START = "2023-02-01"
PANEL_END = "2024-02-27"

# Which households form the panel.
#   "balanced" - only those with unbroken coverage across the whole window (332 of 410).
#                The aggregate is then a like-for-like series with no composition drift.
#   "all"      - every household with any data in the window (410). Coverage runs 96-98%
#                except Feb 2023 at 88%, so the n_reporting rescaling stays a small
#                correction; it does assume absent households are average.
PANEL_MODE = "all"

# Chronological split. Test is the last ~20% of the window - never split randomly.
TEST_START = "2023-12-15"

# Heating degree base, SIA 12/20 convention as used by the dataset's own weather variables.
HDD_BASE = 12.0

# True  -> only features knowable at day-ahead gate closure (noon on D-1).
#          Same-slot lag is D-2; D-1 contributes its morning only.
# False -> also allow the same-slot D-1 lag. Optimistic: that reading does not exist
#          yet for afternoon slots. Usable, but say so in the write-up.
STRICT_GATE_CLOSURE = True

# --- Level 2: procurement prices (EUR/MWh) ------------------------------------------
# The challenge permits assumed prices. These are plausible central-European figures, not
# measured ones, so every cost result is reported alongside a sensitivity sweep.
# Buying short on intraday costs more than day-ahead; selling surplus back recovers less.
PRICE_DAY_AHEAD = 100.0
PRICE_INTRADAY_BUY = 150.0    # covering a shortfall
PRICE_INTRADAY_SELL = 60.0    # offloading a surplus

# Penalty per MWh of forecast error, by direction.
COST_SHORT = PRICE_INTRADAY_BUY - PRICE_DAY_AHEAD    # 50: under-forecast
COST_LONG = PRICE_DAY_AHEAD - PRICE_INTRADAY_SELL    # 40: over-forecast

# The cost function above is a pinball loss. Its minimiser is not the mean but this
# quantile - the theoretical bridge from Level 2 into Level 3.
COST_OPTIMAL_QUANTILE = COST_SHORT / (COST_SHORT + COST_LONG)

FEATURES_STRICT = [
    "slot_of_day", "day_of_week", "is_weekend", "month", "doy_sin", "doy_cos",
    "Temperature_avg_hourly", "hdd", "temp_roll24h", "temp_day_mean",
    "Humidity_avg_hourly", "WindSpeed_hourly", "Sunshine_duration_hourly",
    "lag_2d", "lag_7d", "lag7_roll4", "d1_morning_mean",
]
FEATURES_LOOSE = FEATURES_STRICT + ["lag_1d"]


def feature_list():
    return FEATURES_STRICT if STRICT_GATE_CLOSURE else FEATURES_LOOSE


def savefig(fig, filename, dpi=130, **kw):
    """Save a figure into OUTPUTS, retrying through transient cloud-sync locks.

    Sciebo grabs newly written files to upload them, which surfaces as
    OSError EINVAL part-way through a write and has killed whole pipeline runs.
    Retrying a moment later succeeds.
    """
    path = os.path.join(OUTPUTS, filename)
    last = None
    for attempt in range(6):
        try:
            fig.savefig(path, dpi=dpi, **kw)
            return path
        except OSError as exc:
            last = exc
            time.sleep(1.5)
    raise last
