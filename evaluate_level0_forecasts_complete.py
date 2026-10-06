from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================================
# PATHS
# =====================================================================

# Update this path if you changed the data location in the training script.
DATA_ROOT = Path(
    r"C:\Users\pu65fyqu\Desktop\rwth_hackathon-main"
    r"\rwth_hackathon-main\data"
)

RESULTS_DIR = DATA_ROOT / "level0_results"
PREDICTIONS_FILE = RESULTS_DIR / "level0_predictions.csv"
DAILY_PLOTS_DIR = RESULTS_DIR / "daily_forecast_plots"


# =====================================================================
# CONFIGURATION
# =====================================================================

EXPECTED_INTERVALS_PER_DAY = 96
SAVE_PNG_DPI = 150
SHOW_PLOTS = False


# =====================================================================
# LOAD SAVED FORECASTS
# =====================================================================

def load_predictions() -> pd.DataFrame:
    """
    Load predictions created by level0_portfolio_forecast.py.

    This script does not train, refit, update, or recalibrate the model.
    It only evaluates the already saved forecasts for the final 20%.
    """
    if not PREDICTIONS_FILE.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}\n"
            "Run level0_portfolio_forecast.py first, or correct DATA_ROOT."
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)

    timestamp_candidates = ["Timestamp", "timestamp", "Unnamed: 0"]
    timestamp_column = next(
        (column for column in timestamp_candidates if column in predictions.columns),
        None,
    )

    if timestamp_column is None:
        raise ValueError(
            "Could not identify the timestamp column. "
            f"Available columns: {list(predictions.columns)}"
        )

    predictions[timestamp_column] = pd.to_datetime(
        predictions[timestamp_column],
        utc=True,
        errors="coerce",
    )
    predictions = predictions.dropna(subset=[timestamp_column])
    predictions = predictions.set_index(timestamp_column).sort_index()
    predictions.index.name = "Timestamp"

    required_columns = {
        "actual_kwh",
        "model_forecast_kwh",
        "weekly_persistence_kwh",
    }
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    for column in required_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    predictions["forecast_day"] = predictions.index.normalize()
    return predictions


# =====================================================================
# PERFORMANCE METRICS
# =====================================================================

def calculate_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """Calculate interval, energy, and peak forecast metrics."""
    valid = actual.notna() & forecast.notna()
    actual = actual.loc[valid].astype(float)
    forecast = forecast.loc[valid].astype(float)

    if actual.empty:
        return {
            "Observations": 0,
            "MAE_kWh": np.nan,
            "RMSE_kWh": np.nan,
            "WAPE_percent": np.nan,
            "MAPE_percent": np.nan,
            "sMAPE_percent": np.nan,
            "R2": np.nan,
            "Energy_bias_kWh": np.nan,
            "Relative_bias_percent": np.nan,
            "Actual_energy_kWh": np.nan,
            "Forecast_energy_kWh": np.nan,
            "Absolute_energy_error_kWh": np.nan,
            "Actual_peak_kWh": np.nan,
            "Forecast_peak_kWh": np.nan,
            "Peak_error_kWh": np.nan,
            "Peak_time_error_minutes": np.nan,
        }

    error = forecast - actual
    absolute_error = error.abs()
    actual_energy = actual.sum()
    forecast_energy = forecast.sum()
    energy_bias = forecast_energy - actual_energy

    nonzero_actual = actual.abs() > 1e-12
    mape = (
        (absolute_error.loc[nonzero_actual] / actual.loc[nonzero_actual].abs()).mean()
        * 100
        if nonzero_actual.any()
        else np.nan
    )

    smape_denominator = actual.abs() + forecast.abs()
    valid_smape = smape_denominator > 1e-12
    smape = (
        (
            2
            * absolute_error.loc[valid_smape]
            / smape_denominator.loc[valid_smape]
        ).mean()
        * 100
        if valid_smape.any()
        else np.nan
    )

    actual_peak_time = actual.idxmax()
    forecast_peak_time = forecast.idxmax()
    peak_time_error_minutes = abs(
        (forecast_peak_time - actual_peak_time).total_seconds() / 60
    )

    return {
        "Observations": len(actual),
        "MAE_kWh": mean_absolute_error(actual, forecast),
        "RMSE_kWh": np.sqrt(mean_squared_error(actual, forecast)),
        "WAPE_percent": (
            absolute_error.sum() / actual.abs().sum() * 100
            if actual.abs().sum() > 0
            else np.nan
        ),
        "MAPE_percent": mape,
        "sMAPE_percent": smape,
        "R2": r2_score(actual, forecast) if len(actual) >= 2 else np.nan,
        "Energy_bias_kWh": energy_bias,
        "Relative_bias_percent": (
            energy_bias / actual_energy * 100
            if abs(actual_energy) > 1e-12
            else np.nan
        ),
        "Actual_energy_kWh": actual_energy,
        "Forecast_energy_kWh": forecast_energy,
        "Absolute_energy_error_kWh": abs(energy_bias),
        "Actual_peak_kWh": actual.max(),
        "Forecast_peak_kWh": forecast.max(),
        "Peak_error_kWh": forecast.max() - actual.max(),
        "Peak_time_error_minutes": peak_time_error_minutes,
    }


def calculate_daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create one performance row for every day in the test set."""
    rows = []

    for target_day, day_data in predictions.groupby("forecast_day"):
        model_metrics = calculate_metrics(
            day_data["actual_kwh"],
            day_data["model_forecast_kwh"],
        )
        benchmark_metrics = calculate_metrics(
            day_data["actual_kwh"],
            day_data["weekly_persistence_kwh"],
        )

        row = {
            "forecast_day": target_day.date(),
            "intervals_in_file": len(day_data),
            "complete_96_interval_day": (
                len(day_data) == EXPECTED_INTERVALS_PER_DAY
            ),
        }
        row.update(
            {f"model_{key}": value for key, value in model_metrics.items()}
        )
        row.update(
            {f"benchmark_{key}": value for key, value in benchmark_metrics.items()}
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("forecast_day")


def calculate_overall_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate metrics over every valid 15-minute observation in the
    complete final-20-percent test set.
    """
    model_metrics = calculate_metrics(
        predictions["actual_kwh"],
        predictions["model_forecast_kwh"],
    )
    benchmark_metrics = calculate_metrics(
        predictions["actual_kwh"],
        predictions["weekly_persistence_kwh"],
    )

    overall = pd.DataFrame(
        [model_metrics, benchmark_metrics],
        index=["Gradient boosting", "Weekly persistence"],
    )
    overall.index.name = "Method"
    return overall


# =====================================================================
# DAILY FORECAST PLOTS
# =====================================================================

def save_daily_forecast_plots(predictions: pd.DataFrame) -> None:
    """Save forecast-versus-actual plots for every test-set day."""
    DAILY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    grouped = list(predictions.groupby("forecast_day"))

    for number, (target_day, day_data) in enumerate(grouped, start=1):
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(
            day_data.index,
            day_data["actual_kwh"],
            label="Actual load",
            linewidth=2.2,
            color="black",
        )
        ax.plot(
            day_data.index,
            day_data["model_forecast_kwh"],
            label="Model forecast",
            linewidth=1.9,
            color="tab:blue",
        )
        ax.plot(
            day_data.index,
            day_data["weekly_persistence_kwh"],
            label="Weekly persistence",
            linewidth=1.4,
            linestyle="--",
            color="tab:orange",
            alpha=0.85,
        )

        ax.set_title(f"15-minute portfolio load forecast: {target_day.date()}")
        ax.set_xlabel("UTC time")
        ax.set_ylabel("Energy per 15-minute interval [kWh]")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()

        output_path = DAILY_PLOTS_DIR / f"forecast_{target_day:%Y-%m-%d}.png"
        fig.savefig(output_path, dpi=SAVE_PNG_DPI, bbox_inches="tight")

        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)

        if number % 25 == 0 or number == len(grouped):
            print(f"Saved {number} of {len(grouped)} daily forecast plots.")


# =====================================================================
# SUMMARY PLOTS
# =====================================================================

def save_performance_over_time_plot(daily_metrics: pd.DataFrame) -> None:
    """Plot daily MAE, WAPE, MAPE, and energy bias."""
    dates = pd.to_datetime(daily_metrics["forecast_day"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True)

    axes[0].plot(dates, daily_metrics["model_MAE_kWh"], label="Model MAE")
    axes[0].plot(
        dates,
        daily_metrics["benchmark_MAE_kWh"],
        label="Benchmark MAE",
        alpha=0.8,
    )
    axes[0].set_ylabel("MAE [kWh/15 min]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        dates,
        daily_metrics["model_WAPE_percent"],
        label="Model WAPE",
    )
    axes[1].plot(
        dates,
        daily_metrics["benchmark_WAPE_percent"],
        label="Benchmark WAPE",
        alpha=0.8,
    )
    axes[1].set_ylabel("WAPE [%]")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        dates,
        daily_metrics["model_MAPE_percent"],
        label="Model MAPE",
    )
    axes[2].plot(
        dates,
        daily_metrics["benchmark_MAPE_percent"],
        label="Benchmark MAPE",
        alpha=0.8,
    )
    axes[2].set_ylabel("MAPE [%]")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    axes[3].axhline(0, color="black", linewidth=1)
    axes[3].plot(
        dates,
        daily_metrics["model_Energy_bias_kWh"],
        label="Model daily energy bias",
    )
    axes[3].plot(
        dates,
        daily_metrics["benchmark_Energy_bias_kWh"],
        label="Benchmark daily energy bias",
        alpha=0.8,
    )
    axes[3].set_ylabel("Energy bias [kWh/day]")
    axes[3].set_xlabel("Forecast day")
    axes[3].legend()
    axes[3].grid(alpha=0.25)

    fig.suptitle("Forecast performance across the final 20%")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "level0_performance_over_time.png",
        dpi=SAVE_PNG_DPI,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_daily_mape_plot(daily_metrics: pd.DataFrame) -> None:
    """Save a standalone daily MAPE plot with mean-MAPE reference lines."""
    dates = pd.to_datetime(daily_metrics["forecast_day"])

    model_mean_mape = daily_metrics["model_MAPE_percent"].mean()
    benchmark_mean_mape = daily_metrics["benchmark_MAPE_percent"].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        dates,
        daily_metrics["model_MAPE_percent"],
        label="Model MAPE",
        linewidth=1.8,
        color="tab:blue",
    )
    ax.plot(
        dates,
        daily_metrics["benchmark_MAPE_percent"],
        label="Weekly persistence MAPE",
        linewidth=1.5,
        linestyle="--",
        color="tab:orange",
        alpha=0.85,
    )
    ax.axhline(
        model_mean_mape,
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean daily model MAPE: {model_mean_mape:.2f}%",
    )
    ax.axhline(
        benchmark_mean_mape,
        color="tab:orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean daily benchmark MAPE: {benchmark_mean_mape:.2f}%",
    )

    ax.set_title("Daily Mean Absolute Percentage Error, final 20%")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("MAPE [%]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "level0_daily_mape.png",
        dpi=SAVE_PNG_DPI,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_forecast_vs_actual_scatter(predictions: pd.DataFrame) -> None:
    """Save a scatter plot of all test-set predictions versus actual values."""
    valid = (
        predictions["actual_kwh"].notna()
        & predictions["model_forecast_kwh"].notna()
    )
    actual = predictions.loc[valid, "actual_kwh"]
    forecast = predictions.loc[valid, "model_forecast_kwh"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, forecast, s=8, alpha=0.25)

    lower = min(actual.min(), forecast.min())
    upper = max(actual.max(), forecast.max())
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linestyle="--",
        label="Perfect forecast",
    )

    ax.set_xlabel("Actual load [kWh/15 min]")
    ax.set_ylabel("Forecast load [kWh/15 min]")
    ax.set_title("Forecast versus actual load, final 20%")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "level0_forecast_vs_actual_scatter.png",
        dpi=SAVE_PNG_DPI,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_overall_metrics_table(overall_metrics: pd.DataFrame) -> None:
    """Save a readable PNG table of overall test-set performance metrics."""
    selected_columns = [
        "MAE_kWh",
        "RMSE_kWh",
        "WAPE_percent",
        "MAPE_percent",
        "sMAPE_percent",
        "R2",
        "Energy_bias_kWh",
        "Relative_bias_percent",
    ]

    display_table = overall_metrics[selected_columns].copy()
    display_table.columns = [
        "MAE\n[kWh]",
        "RMSE\n[kWh]",
        "WAPE\n[%]",
        "MAPE\n[%]",
        "sMAPE\n[%]",
        "R²",
        "Energy bias\n[kWh]",
        "Relative bias\n[%]",
    ]
    display_table = display_table.round(3)

    fig, ax = plt.subplots(figsize=(15, 3.4))
    ax.axis("off")
    table = ax.table(
        cellText=display_table.values,
        rowLabels=display_table.index,
        colLabels=display_table.columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        if column == -1:
            cell.set_text_props(weight="bold")

    ax.set_title(
        "Overall performance metrics across the complete final-20% test set",
        fontsize=13,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "level0_overall_metrics_table.png",
        dpi=SAVE_PNG_DPI,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    daily_metrics = calculate_daily_metrics(predictions)
    overall_metrics = calculate_overall_metrics(predictions)

    # Machine-readable metric tables.
    daily_metrics.to_csv(
        RESULTS_DIR / "level0_daily_metrics_extended.csv",
        index=False,
    )
    overall_metrics.to_csv(
        RESULTS_DIR / "level0_overall_metrics_extended.csv"
    )

    # Plots and visual performance table.
    save_daily_forecast_plots(predictions)
    save_performance_over_time_plot(daily_metrics)
    save_daily_mape_plot(daily_metrics)
    save_forecast_vs_actual_scatter(predictions)
    save_overall_metrics_table(overall_metrics)

    display_columns = [
        "Observations",
        "MAE_kWh",
        "RMSE_kWh",
        "WAPE_percent",
        "MAPE_percent",
        "sMAPE_percent",
        "R2",
        "Energy_bias_kWh",
        "Relative_bias_percent",
    ]

    print("\nOverall performance across the complete final-20% test set")
    print("-" * 90)
    print(overall_metrics[display_columns].round(4).to_string())

    print("\nEvaluation coverage")
    print("-" * 90)
    print(f"First forecast day: {predictions.index.min().date()}")
    print(f"Last forecast day:  {predictions.index.max().date()}")
    print(f"Forecast days:      {predictions['forecast_day'].nunique()}")
    print(f"15-minute rows:     {len(predictions)}")
    print(
        "Complete 96-interval days: "
        f"{int(daily_metrics['complete_96_interval_day'].sum())}"
    )

    print("\nNo model training was performed by this evaluation script.")
    print("It evaluated only the forecasts already stored in level0_predictions.csv.")

    print("\nFiles written to")
    print("-" * 90)
    print(RESULTS_DIR.resolve())
    print(DAILY_PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()
