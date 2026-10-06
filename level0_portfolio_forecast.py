from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =====================================================================
# PATHS
# =====================================================================

DATA_ROOT = Path(
    r"C:\Users\pu65fyqu\Desktop\rwth_hackathon-main\rwth_hackathon-main\data"
)

LOAD_DIR = DATA_ROOT / "15min"
SMART_METER_META_DIR = DATA_ROOT / "smart_meter_meta_data"
WEATHER_DIR = DATA_ROOT / "weather_data_hourly"

# This spelling matches the folder name supplied by the user.
WEATHER_OVERVIEW_DIR = DATA_ROOT / "weather_data_overview"

HOUSEHOLDS_FILE = SMART_METER_META_DIR / "households.csv"
META_DATA_FILE = SMART_METER_META_DIR / "meta_data.csv"
META_DATA_VARIABLES_FILE = SMART_METER_META_DIR / "meta_data_variables.csv"
SMART_METER_OVERVIEW_FILE = (
    SMART_METER_META_DIR / "smart_meter_data_15min_overview.csv"
)
WEATHER_VARIABLES_FILE = WEATHER_OVERVIEW_DIR / "weather_variables.csv"
WEATHER_AVAILABILITY_FILE = (
    WEATHER_OVERVIEW_DIR / "weather_variables_availability.csv"
)
OUTPUT_DIR = DATA_ROOT / "level0_results"


# =====================================================================
# CONFIGURATION
# =====================================================================

TARGET_COLUMN = "kWh_received_Total"
TRAIN_FRACTION = 0.80
INTERVALS_PER_DAY = 96
INTERVALS_PER_WEEK = 7 * INTERVALS_PER_DAY
RANDOM_STATE = 42

ANALYSIS_WINDOW_DAYS = 730
MINIMUM_STABLE_HOUSEHOLDS = 20
MINIMUM_HOUSEHOLD_COMPLETENESS = 0.95
MINIMUM_PORTFOLIO_COVERAGE = 0.95
MINIMUM_WEATHER_COMPLETENESS = 0.90
SELECTED_PLOT_DAY = None


# =====================================================================
# BASIC UTILITIES
# =====================================================================

def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV while automatically detecting comma or semicolon separators."""
    return pd.read_csv(path, sep=None, engine="python", **kwargs)


def parse_boolean_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype("boolean")
    )


def validate_paths() -> None:
    required_directories = [
        DATA_ROOT,
        LOAD_DIR,
        SMART_METER_META_DIR,
        WEATHER_DIR,
        WEATHER_OVERVIEW_DIR,
    ]
    required_files = [
        HOUSEHOLDS_FILE,
        META_DATA_FILE,
        META_DATA_VARIABLES_FILE,
        SMART_METER_OVERVIEW_FILE,
        WEATHER_VARIABLES_FILE,
        WEATHER_AVAILABILITY_FILE,
    ]

    missing = [str(p) for p in required_directories if not p.is_dir()]
    missing += [str(p) for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "The following required paths do not exist:\n\n" + "\n".join(missing)
        )

    load_files = list(LOAD_DIR.glob("*.csv"))
    weather_files = list(WEATHER_DIR.glob("*.csv"))
    if not load_files:
        raise FileNotFoundError(f"No CSV files found in {LOAD_DIR}")
    if not weather_files:
        raise FileNotFoundError(f"No CSV files found in {WEATHER_DIR}")

    print("Path validation passed.")
    print(f"Load files found: {len(load_files)}")
    print(f"Weather files found: {len(weather_files)}")


# =====================================================================
# REGISTRY AND AVAILABILITY DATA
# =====================================================================

def load_household_registry() -> pd.DataFrame:
    households = read_csv_auto(
        HOUSEHOLDS_FILE,
        dtype={"Household_ID": "string", "Weather_ID": "string"},
    )
    required = {
        "Household_ID",
        "Weather_ID",
        "SmartMeterData_Available_15min",
    }
    missing = required - set(households.columns)
    if missing:
        raise ValueError(
            "Missing household-register columns: " + ", ".join(sorted(missing))
        )

    households["SmartMeterData_Available_15min"] = parse_boolean_series(
        households["SmartMeterData_Available_15min"]
    )
    households["Household_ID"] = households["Household_ID"].str.strip()
    households["Weather_ID"] = households["Weather_ID"].str.strip()
    return households.drop_duplicates("Household_ID", keep="first")


def load_smart_meter_overview() -> pd.DataFrame:
    overview = read_csv_auto(
        SMART_METER_OVERVIEW_FILE,
        dtype={"Household_ID": "string"},
    )
    required = {
        "Household_ID",
        "SMD_15min_TimeAvailable_EarliestTimestamp",
        "SMD_15min_TimeAvailable_LatestTimestamp",
        "SMD_15min_MeasurementsAvailable_Total",
    }
    missing = required - set(overview.columns)
    if missing:
        raise ValueError(
            "Missing smart-meter overview columns: " + ", ".join(sorted(missing))
        )

    overview["Household_ID"] = overview["Household_ID"].str.strip()
    overview["earliest_timestamp"] = pd.to_datetime(
        overview["SMD_15min_TimeAvailable_EarliestTimestamp"],
        utc=True,
        errors="coerce",
    )
    overview["latest_timestamp"] = pd.to_datetime(
        overview["SMD_15min_TimeAvailable_LatestTimestamp"],
        utc=True,
        errors="coerce",
    )
    overview["has_total_measurements"] = parse_boolean_series(
        overview["SMD_15min_MeasurementsAvailable_Total"]
    )
    return overview


def define_analysis_period(overview: pd.DataFrame):
    usable = overview.loc[
        overview["has_total_measurements"].fillna(False)
        & overview["latest_timestamp"].notna()
    ].copy()
    if usable.empty:
        raise ValueError("No household has total 15-minute measurements.")

    analysis_end = (
        usable["latest_timestamp"].max().normalize()
        + pd.Timedelta(hours=23, minutes=45)
    )
    analysis_start = analysis_end.normalize() - pd.Timedelta(
        days=ANALYSIS_WINDOW_DAYS - 1
    )
    return analysis_start, analysis_end


def select_stable_households(
    households: pd.DataFrame,
    overview: pd.DataFrame,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> pd.DataFrame:
    selection = households.merge(
        overview[
            [
                "Household_ID",
                "earliest_timestamp",
                "latest_timestamp",
                "has_total_measurements",
            ]
        ],
        on="Household_ID",
        how="inner",
        validate="one_to_one",
    )

    selection = selection.loc[
        selection["SmartMeterData_Available_15min"].fillna(False)
        & selection["has_total_measurements"].fillna(False)
        & selection["Weather_ID"].notna()
        & (selection["earliest_timestamp"] <= analysis_start)
        & (selection["latest_timestamp"] >= analysis_end)
    ].copy()

    selection["load_file"] = selection["Household_ID"].map(
        lambda household_id: LOAD_DIR / f"{household_id}.csv"
    )
    selection = selection.loc[selection["load_file"].map(Path.is_file)].copy()
    selection = selection.sort_values("Household_ID").reset_index(drop=True)

    if len(selection) < MINIMUM_STABLE_HOUSEHOLDS:
        raise ValueError(
            f"Only {len(selection)} households cover the complete analysis period. "
            f"Reduce ANALYSIS_WINDOW_DAYS, currently {ANALYSIS_WINDOW_DAYS}."
        )
    return selection


# =====================================================================
# LOAD DATA
# =====================================================================

def read_household_load(
    row,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
):
    household_id = row["Household_ID"]
    frame = read_csv_auto(
        Path(row["load_file"]),
        usecols=["Timestamp", TARGET_COLUMN],
    )
    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"], utc=True, errors="coerce"
    )
    frame[TARGET_COLUMN] = pd.to_numeric(
        frame[TARGET_COLUMN], errors="coerce"
    )
    frame = frame.loc[
        (frame["Timestamp"] >= analysis_start)
        & (frame["Timestamp"] <= analysis_end)
    ].dropna(subset=["Timestamp"])

    series = (
        frame.groupby("Timestamp")[TARGET_COLUMN]
        .mean()
        .sort_index()
    )
    complete_index = pd.date_range(
        analysis_start, analysis_end, freq="15min", tz="UTC"
    )
    series = series.reindex(complete_index)
    series.name = household_id
    return series, series.notna().mean()


def load_stable_portfolio(
    selected_households: pd.DataFrame,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
):
    household_series = []
    quality_rows = []

    for position, row in selected_households.iterrows():
        series, completeness = read_household_load(
            row, analysis_start, analysis_end
        )
        quality_rows.append(
            {
                "Household_ID": row["Household_ID"],
                "Weather_ID": row["Weather_ID"],
                "completeness": completeness,
            }
        )
        if completeness >= MINIMUM_HOUSEHOLD_COMPLETENESS:
            household_series.append(series)

        if (position + 1) % 25 == 0 or position + 1 == len(selected_households):
            print(
                f"Processed {position + 1} of "
                f"{len(selected_households)} load files."
            )

    quality_report = pd.DataFrame(quality_rows)
    accepted_ids = set(
        quality_report.loc[
            quality_report["completeness"] >= MINIMUM_HOUSEHOLD_COMPLETENESS,
            "Household_ID",
        ]
    )
    accepted_households = selected_households.loc[
        selected_households["Household_ID"].isin(accepted_ids)
    ].copy().reset_index(drop=True)

    if len(accepted_households) < MINIMUM_STABLE_HOUSEHOLDS:
        raise ValueError(
            f"Only {len(accepted_households)} households meet the completeness "
            "requirement. Reduce ANALYSIS_WINDOW_DAYS or the minimum thresholds."
        )

    load_matrix = pd.concat(household_series, axis=1)
    reporting_count = load_matrix.notna().sum(axis=1)
    total_load = load_matrix.sum(axis=1, min_count=1)
    expected_count = len(accepted_households)
    coverage_ratio = reporting_count / expected_count
    total_load.loc[coverage_ratio < MINIMUM_PORTFOLIO_COVERAGE] = np.nan

    portfolio = pd.DataFrame(
        {
            "load_kwh": total_load,
            "households_reporting": reporting_count,
            "expected_households": expected_count,
            "coverage_ratio": coverage_ratio,
        }
    )
    portfolio.index.name = "Timestamp"
    return portfolio, accepted_households, quality_report


# =====================================================================
# WEATHER DATA
# =====================================================================

def load_weather_definitions():
    variables = read_csv_auto(WEATHER_VARIABLES_FILE)
    availability = read_csv_auto(
        WEATHER_AVAILABILITY_FILE,
        dtype={"Weather_ID": "string"},
    )
    availability["Weather_ID"] = availability["Weather_ID"].str.strip()
    for column in availability.columns:
        if column != "Weather_ID":
            availability[column] = parse_boolean_series(availability[column])
    return variables, availability


def determine_weather_variables(
    weather_ids,
    variables: pd.DataFrame,
    availability: pd.DataFrame,
):
    variable_name_column = next(
        (c for c in ["VariableName", "Variable", "variable"] if c in variables.columns),
        None,
    )
    resolution_column = next(
        (c for c in ["Resolution", "resolution"] if c in variables.columns),
        None,
    )
    if variable_name_column is None:
        raise ValueError(
            "Could not identify the variable-name column in weather_variables.csv. "
            f"Columns are: {list(variables.columns)}"
        )

    if resolution_column is None:
        candidate_variables = set(variables[variable_name_column].dropna().astype(str))
    else:
        candidate_variables = set(
            variables.loc[
                variables[resolution_column]
                .astype("string")
                .str.lower()
                .eq("hourly"),
                variable_name_column,
            ].dropna().astype(str)
        )

    station_availability = availability.loc[
        availability["Weather_ID"].isin(weather_ids)
    ].copy()
    missing_stations = set(weather_ids) - set(station_availability["Weather_ID"])
    if missing_stations:
        raise ValueError(
            "No weather availability entry found for: "
            + ", ".join(sorted(missing_stations))
        )

    selected = []
    for variable in sorted(candidate_variables):
        if variable in station_availability.columns and station_availability[
            variable
        ].fillna(False).all():
            selected.append(variable)

    if not selected:
        # Robust fallback: select columns marked available for all stations,
        # even if the description file uses unexpected column labels.
        ignored = {"Weather_ID"}
        for column in station_availability.columns:
            if column not in ignored and station_availability[column].fillna(False).all():
                selected.append(column)

    if not selected:
        raise ValueError(
            "No hourly weather variable is available for every required station."
        )
    return selected


def load_weather_station(
    weather_id: str,
    selected_variables,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
):
    file_path = WEATHER_DIR / f"{weather_id}.csv"
    if not file_path.is_file():
        raise FileNotFoundError(f"Weather file not found: {file_path}")

    header = read_csv_auto(file_path, nrows=0).columns.tolist()
    available_columns = [v for v in selected_variables if v in header]
    if not available_columns:
        raise ValueError(
            f"None of the selected weather variables occur in {file_path.name}. "
            f"Columns are: {header}"
        )

    usecols = ["Timestamp", *available_columns]
    if "Weather_ID" in header:
        usecols.insert(0, "Weather_ID")
    frame = read_csv_auto(file_path, usecols=usecols)
    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"], utc=True, errors="coerce"
    )
    frame = frame.loc[
        (frame["Timestamp"] >= analysis_start)
        & (frame["Timestamp"] <= analysis_end.floor("h"))
    ].copy()
    for column in available_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["Timestamp"])
        .groupby("Timestamp")[available_columns]
        .mean()
        .sort_index()
    )


def load_weighted_portfolio_weather(
    accepted_households: pd.DataFrame,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
):
    variables, availability = load_weather_definitions()
    station_weights = (
        accepted_households["Weather_ID"].value_counts(normalize=True).sort_index()
    )
    weather_ids = station_weights.index.tolist()
    selected_variables = determine_weather_variables(
        weather_ids, variables, availability
    )

    print("\nWeather station weights")
    print(station_weights.round(4))
    print("\nSelected hourly weather variables")
    for variable in selected_variables:
        print(f"  {variable}")

    station_data = {
        weather_id: load_weather_station(
            weather_id, selected_variables, analysis_start, analysis_end
        )
        for weather_id in weather_ids
    }
    hourly_index = pd.date_range(
        analysis_start.floor("h"), analysis_end.floor("h"), freq="h", tz="UTC"
    )
    weighted_weather = pd.DataFrame(index=hourly_index)
    retained_variables = []

    for variable in selected_variables:
        weighted_sum = pd.Series(0.0, index=hourly_index)
        available_weight = pd.Series(0.0, index=hourly_index)
        for weather_id, station_frame in station_data.items():
            if variable not in station_frame.columns:
                continue
            weight = station_weights.loc[weather_id]
            values = station_frame[variable].reindex(hourly_index)
            valid = values.notna()
            weighted_sum.loc[valid] += values.loc[valid] * weight
            available_weight.loc[valid] += weight

        portfolio_variable = weighted_sum / available_weight.replace(0, np.nan)
        if portfolio_variable.notna().mean() >= MINIMUM_WEATHER_COMPLETENESS:
            weighted_weather[f"weather_{variable}"] = portfolio_variable
            retained_variables.append(variable)

    if weighted_weather.empty:
        raise ValueError("No weather variable passed the completeness threshold.")

    quarter_hour_index = pd.date_range(
        analysis_start, analysis_end, freq="15min", tz="UTC"
    )
    expanded_index = weighted_weather.index.union(quarter_hour_index).sort_values()
    weighted_weather = (
        weighted_weather.reindex(expanded_index)
        .ffill()
        .reindex(quarter_hour_index)
    )
    weighted_weather.index.name = "Timestamp"
    return weighted_weather, station_weights, retained_variables


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    quarter_hour = frame.index.hour * 4 + frame.index.minute // 15
    weekday = frame.index.dayofweek
    day_of_year = frame.index.dayofyear

    frame["quarter_hour"] = quarter_hour
    frame["day_of_week"] = weekday
    frame["month"] = frame.index.month
    frame["is_weekend"] = (weekday >= 5).astype(int)
    frame["time_sin"] = np.sin(2 * np.pi * quarter_hour / 96)
    frame["time_cos"] = np.cos(2 * np.pi * quarter_hour / 96)
    frame["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    frame["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return frame


def build_features(
    portfolio: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    load = portfolio["load_kwh"]
    frame = pd.DataFrame(index=portfolio.index)
    frame["target"] = load
    frame = add_calendar_features(frame)

    for days in [1, 2, 3, 7, 14, 21, 28]:
        frame[f"load_lag_{days}_days"] = load.shift(days * INTERVALS_PER_DAY)

    weekly_lags = pd.concat(
        [load.shift(week * INTERVALS_PER_WEEK) for week in range(1, 5)],
        axis=1,
    )
    frame["load_same_time_4_week_mean"] = weekly_lags.mean(axis=1)
    frame["load_same_time_4_week_std"] = weekly_lags.std(axis=1)

    known_load = load.shift(INTERVALS_PER_DAY)
    frame["load_previous_day_mean"] = known_load.rolling(
        INTERVALS_PER_DAY, min_periods=INTERVALS_PER_DAY
    ).mean()
    frame["load_previous_day_max"] = known_load.rolling(
        INTERVALS_PER_DAY, min_periods=INTERVALS_PER_DAY
    ).max()
    frame["load_previous_7_days_mean"] = known_load.rolling(
        INTERVALS_PER_WEEK, min_periods=INTERVALS_PER_DAY
    ).mean()
    frame["load_previous_7_days_std"] = known_load.rolling(
        INTERVALS_PER_WEEK, min_periods=INTERVALS_PER_DAY
    ).std()

    for column in weather.columns:
        series = weather[column]
        for days in [1, 2, 3, 7]:
            frame[f"{column}_lag_{days}_days"] = series.shift(
                days * INTERVALS_PER_DAY
            )

        known_weather = series.shift(INTERVALS_PER_DAY)
        frame[f"{column}_previous_day_mean"] = known_weather.rolling(
            INTERVALS_PER_DAY, min_periods=INTERVALS_PER_DAY
        ).mean()
        frame[f"{column}_previous_day_min"] = known_weather.rolling(
            INTERVALS_PER_DAY, min_periods=INTERVALS_PER_DAY
        ).min()
        frame[f"{column}_previous_day_max"] = known_weather.rolling(
            INTERVALS_PER_DAY, min_periods=INTERVALS_PER_DAY
        ).max()
        frame[f"{column}_previous_7_days_mean"] = known_weather.rolling(
            INTERVALS_PER_WEEK, min_periods=INTERVALS_PER_DAY
        ).mean()
        frame[f"{column}_previous_7_days_std"] = known_weather.rolling(
            INTERVALS_PER_WEEK, min_periods=INTERVALS_PER_DAY
        ).std()

    return frame


# =====================================================================
# SPLIT, METRICS, AND PLOTTING
# =====================================================================

def get_complete_valid_days(feature_data: pd.DataFrame) -> pd.DatetimeIndex:
    day_summary = (
        feature_data.assign(day=feature_data.index.normalize())
        .groupby("day")
        .agg(intervals=("target", "size"), valid_targets=("target", "count"))
    )
    days = day_summary.loc[
        (day_summary["intervals"] == INTERVALS_PER_DAY)
        & (day_summary["valid_targets"] == INTERVALS_PER_DAY)
    ].index
    return pd.DatetimeIndex(days).sort_values()


def temporal_split(feature_data: pd.DataFrame):
    complete_days = get_complete_valid_days(feature_data)
    split_position = int(np.floor(len(complete_days) * TRAIN_FRACTION))
    if split_position < 30:
        raise ValueError("The training period contains too few complete days.")
    if split_position >= len(complete_days):
        raise ValueError("No evaluation days remain after splitting.")
    return complete_days[:split_position], complete_days[split_position:]


def calculate_metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    valid = actual.notna() & prediction.notna()
    actual = actual.loc[valid]
    prediction = prediction.loc[valid]
    if actual.empty:
        return {
            "observations": 0,
            "MAE_kWh": np.nan,
            "RMSE_kWh": np.nan,
            "WAPE_percent": np.nan,
            "energy_bias_kWh": np.nan,
            "relative_bias_percent": np.nan,
        }

    error = prediction - actual
    denominator = np.abs(actual).sum()
    energy = actual.sum()
    return {
        "observations": len(actual),
        "MAE_kWh": mean_absolute_error(actual, prediction),
        "RMSE_kWh": np.sqrt(mean_squared_error(actual, prediction)),
        "WAPE_percent": (
            100 * np.abs(error).sum() / denominator if denominator > 0 else np.nan
        ),
        "energy_bias_kWh": error.sum(),
        "relative_bias_percent": (
            100 * error.sum() / energy if energy != 0 else np.nan
        ),
    }


def plot_day(predictions: pd.DataFrame, target_day) -> None:
    target_day = pd.Timestamp(target_day)
    if target_day.tzinfo is None:
        target_day = target_day.tz_localize("UTC")
    else:
        target_day = target_day.tz_convert("UTC")
    target_day = target_day.normalize()
    day_end = target_day + pd.Timedelta(days=1)
    day_data = predictions.loc[
        (predictions.index >= target_day) & (predictions.index < day_end)
    ]
    if day_data.empty:
        print(f"No forecast exists for {target_day.date()}.")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(day_data.index, day_data["actual_kwh"], label="Actual load", linewidth=2)
    plt.plot(
        day_data.index,
        day_data["model_forecast_kwh"],
        label="Gradient boosting",
        linewidth=2,
    )
    plt.plot(
        day_data.index,
        day_data["weekly_persistence_kwh"],
        label="Same quarter-hour one week earlier",
        linestyle="--",
        alpha=0.8,
    )
    plt.xlabel("UTC timestamp")
    plt.ylabel("Portfolio energy per 15 minutes [kWh]")
    plt.title(f"Level 0 portfolio forecast: {target_day.date()}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"level0_forecast_{target_day:%Y-%m-%d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_level0():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validate_paths()

    households = load_household_registry()
    overview = load_smart_meter_overview()
    analysis_start, analysis_end = define_analysis_period(overview)

    print("\nAnalysis period")
    print("-" * 60)
    print(f"Start: {analysis_start}")
    print(f"End:   {analysis_end}")

    selected_households = select_stable_households(
        households, overview, analysis_start, analysis_end
    )
    print(
        f"\nHouseholds covering the official period: {len(selected_households)}"
    )

    portfolio, accepted_households, household_quality = load_stable_portfolio(
        selected_households, analysis_start, analysis_end
    )
    print(
        "Households retained after file validation: "
        f"{len(accepted_households)}"
    )

    weather, station_weights, weather_variables = load_weighted_portfolio_weather(
        accepted_households, analysis_start, analysis_end
    )
    print("\nCreating day-ahead-safe features...")
    feature_data = build_features(portfolio, weather)
    feature_columns = [c for c in feature_data.columns if c != "target"]

    unlagged_weather = [
        c
        for c in feature_columns
        if c.startswith("weather_")
        and "_lag_" not in c
        and "_previous_" not in c
    ]
    if unlagged_weather:
        raise ValueError(
            "Target-day measured weather leakage detected:\n"
            + "\n".join(unlagged_weather)
        )

    train_days, evaluation_days = temporal_split(feature_data)
    print("\nTemporal split")
    print("-" * 60)
    print(f"First training day:   {train_days.min().date()}")
    print(f"Last training day:    {train_days.max().date()}")
    print(f"First evaluation day: {evaluation_days.min().date()}")
    print(f"Last evaluation day:  {evaluation_days.max().date()}")
    print(f"Training days: {len(train_days)}")
    print(f"Evaluation days: {len(evaluation_days)}")

    train_mask = feature_data.index.normalize().isin(train_days)
    training_data = feature_data.loc[train_mask].dropna(subset=["target"]).copy()
    X_train = training_data[feature_columns]
    y_train = training_data["target"]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    print("\nTraining model once on the first 80%...")
    model.fit(X_train, y_train)
    print("Training complete. Model is frozen.")

    prediction_frames = []
    metric_rows = []
    for position, target_day in enumerate(evaluation_days, start=1):
        day_end = target_day + pd.Timedelta(days=1)
        day_data = feature_data.loc[
            (feature_data.index >= target_day) & (feature_data.index < day_end)
        ].copy()
        if len(day_data) != INTERVALS_PER_DAY:
            continue

        actual = day_data["target"]
        forecast = pd.Series(
            model.predict(day_data[feature_columns]),
            index=day_data.index,
            name="model_forecast_kwh",
        ).clip(lower=0)
        weekly = day_data["load_lag_7_days"].rename("weekly_persistence_kwh")
        day_predictions = pd.DataFrame(
            {
                "actual_kwh": actual,
                "model_forecast_kwh": forecast,
                "weekly_persistence_kwh": weekly,
                "forecast_day": target_day.date(),
            }
        )
        prediction_frames.append(day_predictions)

        model_metrics = calculate_metrics(actual, forecast)
        benchmark_metrics = calculate_metrics(actual, weekly)
        metric_rows.append(
            {
                "forecast_day": target_day.date(),
                "actual_energy_kWh": actual.sum(),
                "forecast_energy_kWh": forecast.sum(),
                "model_MAE_kWh": model_metrics["MAE_kWh"],
                "model_RMSE_kWh": model_metrics["RMSE_kWh"],
                "model_WAPE_percent": model_metrics["WAPE_percent"],
                "model_energy_bias_kWh": model_metrics["energy_bias_kWh"],
                "model_relative_bias_percent": model_metrics[
                    "relative_bias_percent"
                ],
                "benchmark_MAE_kWh": benchmark_metrics["MAE_kWh"],
                "benchmark_RMSE_kWh": benchmark_metrics["RMSE_kWh"],
                "benchmark_WAPE_percent": benchmark_metrics["WAPE_percent"],
                "benchmark_energy_bias_kWh": benchmark_metrics[
                    "energy_bias_kWh"
                ],
            }
        )
        if position % 25 == 0 or position == len(evaluation_days):
            print(
                f"Predicted {position} of {len(evaluation_days)} evaluation days."
            )

    if not prediction_frames:
        raise ValueError("No complete evaluation day could be predicted.")

    predictions = pd.concat(prediction_frames).sort_index()
    daily_metrics = pd.DataFrame(metric_rows)
    overall_metrics = pd.DataFrame(
        {
            "Gradient boosting": calculate_metrics(
                predictions["actual_kwh"], predictions["model_forecast_kwh"]
            ),
            "Weekly persistence": calculate_metrics(
                predictions["actual_kwh"], predictions["weekly_persistence_kwh"]
            ),
        }
    ).T

    split_summary = pd.DataFrame(
        {
            "period": ["training", "evaluation"],
            "first_day": [train_days.min().date(), evaluation_days.min().date()],
            "last_day": [train_days.max().date(), evaluation_days.max().date()],
            "days": [len(train_days), len(evaluation_days)],
        }
    )
    station_weight_output = station_weights.rename(
        "portfolio_weight"
    ).reset_index()
    station_weight_output.columns = ["Weather_ID", "portfolio_weight"]

    portfolio.to_csv(OUTPUT_DIR / "portfolio_load.csv")
    weather.to_csv(OUTPUT_DIR / "portfolio_weather_historical.csv")
    accepted_households.to_csv(
        OUTPUT_DIR / "stable_household_portfolio.csv", index=False
    )
    household_quality.to_csv(
        OUTPUT_DIR / "household_data_quality.csv", index=False
    )
    station_weight_output.to_csv(
        OUTPUT_DIR / "weather_station_weights.csv", index=False
    )
    predictions.to_csv(OUTPUT_DIR / "level0_predictions.csv")
    daily_metrics.to_csv(OUTPUT_DIR / "level0_daily_metrics.csv", index=False)
    overall_metrics.to_csv(OUTPUT_DIR / "level0_overall_metrics.csv")
    split_summary.to_csv(OUTPUT_DIR / "level0_temporal_split.csv", index=False)
    pd.DataFrame({"weather_variable": weather_variables}).to_csv(
        OUTPUT_DIR / "weather_variables_used.csv", index=False
    )

    print("\nOverall metrics")
    print("-" * 60)
    print(overall_metrics.round(3))
    print("\nResults written to")
    print(OUTPUT_DIR.resolve())

    plot_target = evaluation_days[0] if SELECTED_PLOT_DAY is None else SELECTED_PLOT_DAY
    plot_day(predictions, plot_target)

    return {
        "model": model,
        "portfolio": portfolio,
        "weather": weather,
        "feature_data": feature_data,
        "feature_columns": feature_columns,
        "accepted_households": accepted_households,
        "train_days": train_days,
        "evaluation_days": evaluation_days,
        "predictions": predictions,
        "daily_metrics": daily_metrics,
        "overall_metrics": overall_metrics,
    }


if __name__ == "__main__":
    results = run_level0()
