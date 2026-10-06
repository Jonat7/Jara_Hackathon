from pathlib import Path
import base64
import html
import mimetypes
import numpy as np
import pandas as pd

# Apply the same corrected root path used in your forecasting scripts.
DATA_ROOT = Path(r"C:\Users\pu65fyqu\Desktop\rwth_hackathon-main\rwth_hackathon-main\data")
RESULTS_DIR = DATA_ROOT / "level0_results"
OUTPUT_HTML = RESULTS_DIR / "level0_recap.html"


def required(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run level0_portfolio_forecast.py and "
            "evaluate_level0_forecasts_complete.py first."
        )


def optional_csv(name, **kwargs):
    path = RESULTS_DIR / name
    return pd.read_csv(path, **kwargs) if path.is_file() else pd.DataFrame()


def fmt(value, digits=2, suffix=""):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}" if np.isfinite(value) else "n/a"


def esc(value):
    return html.escape(str(value))


def image_uri(path):
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def figure(path, caption):
    uri = image_uri(path)
    if not uri:
        return ""
    return (
        "<figure>"
        f"<img src='{uri}' alt='{esc(caption)}'>"
        f"<figcaption>{esc(caption)}</figcaption>"
        "</figure>"
    )


def load_predictions():
    path = RESULTS_DIR / "level0_predictions.csv"
    required(path)
    df = pd.read_csv(path)
    timestamp = next(
        (c for c in ["Timestamp", "timestamp", "Unnamed: 0"] if c in df.columns),
        None,
    )
    if timestamp is None:
        raise ValueError(f"No timestamp column in {path.name}: {list(df.columns)}")
    df[timestamp] = pd.to_datetime(df[timestamp], utc=True, errors="coerce")
    return df.dropna(subset=[timestamp]).set_index(timestamp).sort_index()


def metric_row(overall, phrase):
    for index in overall.index:
        if phrase.lower() in str(index).lower():
            return overall.loc[index]
    raise KeyError(f"No row matching {phrase!r}; rows are {list(overall.index)}")


def skill(model_value, reference_value):
    model_value = float(model_value)
    reference_value = float(reference_value)
    return 1 - model_value / reference_value if reference_value else np.nan


def df_html(df, index=True):
    if df.empty:
        return "<p class='muted'>Not available.</p>"
    return df.to_html(index=index, border=0, escape=True, na_rep="n/a")


def daily_example(daily, which):
    metric = next(
        (c for c in ["model_MAPE_percent", "model_WAPE_percent", "model_MAE_kWh"] if c in daily.columns),
        None,
    )
    if metric is None or daily.empty:
        return None
    ordered = daily.dropna(subset=[metric]).sort_values(metric)
    if ordered.empty:
        return None
    row = ordered.iloc[0] if which == "best" else ordered.iloc[-1] if which == "worst" else ordered.iloc[len(ordered)//2]
    day = pd.Timestamp(row["forecast_day"]).strftime("%Y-%m-%d")
    path = RESULTS_DIR / "daily_forecast_plots" / f"forecast_{day}.png"
    return path if path.is_file() else None


def build_report():
    overall_path = RESULTS_DIR / "level0_overall_metrics_extended.csv"
    daily_path = RESULTS_DIR / "level0_daily_metrics_extended.csv"
    required(overall_path)
    required(daily_path)

    predictions = load_predictions()
    overall = pd.read_csv(overall_path, index_col=0)
    daily = pd.read_csv(daily_path)
    split = optional_csv("level0_temporal_split.csv")
    panel = optional_csv("stable_household_portfolio.csv")
    quality = optional_csv("household_data_quality.csv")
    weights = optional_csv("weather_station_weights.csv")
    weather_vars = optional_csv("weather_variables_used.csv")

    model = metric_row(overall, "Gradient boosting")
    baseline = metric_row(overall, "Weekly persistence")

    def m(row, key):
        return row.get(key, np.nan)

    rmse_skill = skill(m(model, "RMSE_kWh"), m(baseline, "RMSE_kWh"))
    mae_skill = skill(m(model, "MAE_kWh"), m(baseline, "MAE_kWh"))
    mase = float(m(model, "MAE_kWh")) / float(m(baseline, "MAE_kWh"))

    metric_columns = [
        "MAE_kWh", "RMSE_kWh", "WAPE_percent", "MAPE_percent",
        "sMAPE_percent", "R2", "Energy_bias_kWh", "Relative_bias_percent"
    ]
    metric_table = overall[[c for c in metric_columns if c in overall.columns]].copy().round(3)
    metric_table.columns = [c.replace("_percent", " [%]").replace("_kWh", " [kWh]").replace("_", " ") for c in metric_table.columns]

    daily_cols = [c for c in [
        "model_MAE_kWh", "model_RMSE_kWh", "model_WAPE_percent",
        "model_MAPE_percent", "model_Energy_bias_kWh",
        "model_Peak_error_kWh", "model_Peak_time_error_minutes"
    ] if c in daily.columns]
    daily_summary = daily[daily_cols].agg(["mean", "median", "min", "max"]).T.round(3) if daily_cols else pd.DataFrame()
    if not daily_summary.empty:
        daily_summary.columns = ["Mean", "Median", "Minimum", "Maximum"]

    if not weights.empty:
        wc = next((c for c in ["portfolio_weight", "weight", "proportion"] if c in weights.columns), None)
        if wc:
            weights[wc] = (pd.to_numeric(weights[wc], errors="coerce") * 100).round(2)
            weights = weights.rename(columns={wc: "Panel weight [%]"})

    split_html = df_html(split, index=False)
    weather_list = ", ".join(weather_vars.iloc[:, 0].dropna().astype(str)) if not weather_vars.empty else "n/a"
    panel_count = len(panel) if not panel.empty else np.nan
    station_count = weights["Weather_ID"].nunique() if "Weather_ID" in weights.columns else len(weights)
    variable_count = len(weather_vars)
    completeness = quality["completeness"].mean() * 100 if "completeness" in quality.columns else np.nan

    start = predictions.index.min().date()
    end = predictions.index.max().date()
    days = predictions.index.normalize().nunique()

    source_figures = "".join([
        figure(RESULTS_DIR / "level0_performance_over_time.png", "Daily MAE, WAPE, MAPE, and energy bias across the final 20%."),
        figure(RESULTS_DIR / "level0_daily_mape.png", "Daily MAPE for the model and weekly-persistence benchmark."),
        figure(RESULTS_DIR / "level0_forecast_vs_actual_scatter.png", "All test-set forecasts versus actual portfolio load."),
        figure(RESULTS_DIR / "level0_overall_metrics_table.png", "Overall performance metrics across the complete test set."),
    ])

    examples = ""
    for which, caption in [
        ("best", "Low-error forecast day from the final 20%."),
        ("median", "Median-error forecast day from the final 20%."),
        ("worst", "High-error forecast day from the final 20%."),
    ]:
        path = daily_example(daily, which)
        if path:
            examples += figure(path, caption)

    css = '''
    :root{--fg:#111827;--muted:#374151;--line:#e5e7eb;--accent:#1f77b4;--bg:#fff;--soft:#f8fafc}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);max-width:980px;margin:0 auto;padding:32px 20px 80px;line-height:1.56}
    h1{font-size:29px;margin:.2em 0 .1em}h2{font-size:21px;margin:1.8em 0 .4em;padding-top:.5em;border-top:2px solid var(--line)}h3{font-size:16px;color:var(--muted)}.sub,.muted{color:var(--muted)}
    code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:13px}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}th{background:var(--soft)}tr:nth-child(even) td{background:#fcfdff}
    figure{margin:18px 0;padding:10px;border:1px solid var(--line);border-radius:10px;background:var(--soft)}figure img,figure svg{width:100%;height:auto;display:block;background:#fff;border-radius:6px}figcaption{font-size:13px;color:var(--muted);margin-top:8px;text-align:center}
    .toc{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:14px 18px}.toc a{color:var(--accent);text-decoration:none}.key{background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:10px 14px;margin:12px 0}.warn{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:10px 14px;margin:12px 0}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin:15px 0}.card{border:1px solid var(--line);border-radius:9px;padding:12px;background:var(--soft)}.card b{display:block;font-size:21px;color:var(--accent)}.card span{font-size:12px;color:var(--muted)}
    @media(max-width:650px){body{padding:20px 12px 60px}table{display:block;overflow-x:auto}.metrics{grid-template-columns:1fr 1fr}}
    '''

    pipeline_svg = '''
    <figure><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 410">
    <defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#64748b"/></marker><style>.b{rx:10;stroke-width:1.5}.t{font:700 15px sans-serif;fill:#111827}.s{font:12px sans-serif;fill:#374151}.e{stroke:#64748b;stroke-width:1.6;fill:none;marker-end:url(#a)}</style></defs>
    <rect width="1000" height="410" fill="#fff"/><rect class="b" x="35" y="35" width="270" height="70" fill="#eef2ff" stroke="#c7d2fe"/><text x="52" y="64" class="t">15-minute household load</text><text x="52" y="87" class="s">fixed stable panel</text>
    <rect class="b" x="365" y="35" width="270" height="70" fill="#eef2ff" stroke="#c7d2fe"/><text x="382" y="64" class="t">Household metadata</text><text x="382" y="87" class="s">Household_ID → Weather_ID</text>
    <rect class="b" x="695" y="35" width="270" height="70" fill="#eef2ff" stroke="#c7d2fe"/><text x="712" y="64" class="t">Hourly measured weather</text><text x="712" y="87" class="s">station-weighted, forward-filled</text>
    <rect class="b" x="240" y="155" width="520" height="70" fill="#e0f2fe" stroke="#7dd3fc"/><text x="260" y="184" class="t">Leakage-safe feature matrix</text><text x="260" y="207" class="s">calendar + load/weather information from previous days only</text>
    <path class="e" d="M170,105 L380,153"/><path class="e" d="M500,105 L500,153"/><path class="e" d="M830,105 L620,153"/>
    <rect class="b" x="60" y="285" width="390" height="80" fill="#dbeafe" stroke="#93c5fd"/><text x="80" y="316" class="t">First 80%: train once</text><text x="80" y="340" class="s">HistGradientBoosting, no shuffle</text>
    <rect class="b" x="550" y="285" width="390" height="80" fill="#dcfce7" stroke="#86efac"/><text x="570" y="316" class="t">Final 20%: frozen evaluation</text><text x="570" y="340" class="s">96-point daily forecasts, no refit</text>
    <path class="e" d="M430,225 L275,283"/><path class="e" d="M570,225 L725,283"/><path class="e" d="M450,325 L548,325"/>
    </svg><figcaption>Level 0 data pipeline and chronological evaluation protocol.</figcaption></figure>
    '''

    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Level 0 Forecasting — Project Recap</title><style>{css}</style></head><body>
    <h1>Level 0 Day-Ahead Load Forecasting — Full Project Recap</h1><p class="sub">Aggregate 15-minute portfolio forecast. The model is trained once on the first 80% and frozen for evaluation on every complete day in the final 20%.</p>
    <div class="toc"><b>Contents</b><br><a href="#problem">1. Problem</a> · <a href="#decisions">2. Decisions</a> · <a href="#data">3. Data</a> · <a href="#pipeline">4. Pipeline</a> · <a href="#features">5. Features</a> · <a href="#split">6. Split</a> · <a href="#metrics">7. Metrics</a> · <a href="#daily">8. Daily results</a> · <a href="#figures">9. Figures</a> · <a href="#findings">10. Findings</a> · <a href="#caveats">11. Caveats</a> · <a href="#next">12. Next steps</a></div>

    <h2 id="problem">1. The problem</h2><p>Forecast the complete next-day load curve before the day begins. Each evaluated day contains 96 quarter-hour predictions. Errors matter both for total energy allocation and for peak magnitude and timing.</p>

    <h2 id="decisions">2. Locked decisions</h2><table><tr><th>Decision</th><th>Choice</th><th>Reason</th></tr>
    <tr><td>Scope</td><td><b>Level 0 aggregate</b></td><td>One overall portfolio series.</td></tr><tr><td>Target</td><td><code>kWh_received_Total</code></td><td>Measured energy received per 15-minute interval.</td></tr>
    <tr><td>Panel</td><td><b>{fmt(panel_count,0)} stable households</b></td><td>Prevents household entry and exit from looking like demand changes.</td></tr><tr><td>Split</td><td>Chronological 80/20</td><td>Test only on later observations.</td></tr>
    <tr><td>Model</td><td>HistGradientBoosting</td><td>Nonlinear tree-boosting baseline.</td></tr><tr><td>Test protocol</td><td>Frozen model</td><td>No training, tuning, or calibration with the final 20%.</td></tr>
    <tr><td>Weather</td><td>Prior measured days only</td><td>Measured target-day weather would not be available at issue time.</td></tr></table>

    <h2 id="data">3. Data</h2><div class="metrics"><div class="card"><b>{fmt(panel_count,0)}</b><span>stable households</span></div><div class="card"><b>{fmt(station_count,0)}</b><span>weather stations</span></div><div class="card"><b>{fmt(variable_count,0)}</b><span>weather variables</span></div><div class="card"><b>{fmt(completeness,2,'%')}</b><span>mean household completeness</span></div></div><p><b>Weather variables used:</b> {esc(weather_list)}</p><h3>Weather station representation</h3>{df_html(weights, index=False)}

    <h2 id="pipeline">4. Pipeline</h2>{pipeline_svg}<ol><li>Select a fixed household panel covering the analysis window.</li><li>Aggregate valid <code>kWh_received_Total</code> records to one portfolio series.</li><li>Weight weather stations according to the retained household mapping.</li><li>Forward-fill hourly weather to 15-minute timestamps without future interpolation.</li><li>Create leakage-safe calendar, load-history, and weather-history features.</li><li>Fit once on the first 80%; predict each final-20% day without any further fit call.</li></ol>

    <h2 id="features">5. Features and leakage</h2><table><tr><th>Group</th><th>Examples</th><th>Rule</th></tr><tr><td>Calendar</td><td>quarter-hour, weekday, weekend, month, cyclical encodings</td><td>Known in advance.</td></tr><tr><td>Load history</td><td>1/2/3/7/14/21/28-day lags and rolling summaries</td><td>Shifted by at least one full day.</td></tr><tr><td>Weather history</td><td>1/2/3/7-day lags and prior-day/week summaries</td><td>No measured target-day weather.</td></tr></table><div class="key"><b>Leakage rule:</b> for day D, the model may use calendar information for D and load/weather measurements strictly before D. Test targets are used only for scoring.</div>

    <h2 id="split">6. Temporal split</h2>{split_html}<p>Exported predictions cover <b>{start}</b> through <b>{end}</b>: <b>{days}</b> forecast days and <b>{len(predictions):,}</b> quarter-hour rows.</p>

    <h2 id="metrics">7. Overall performance on the complete test set</h2><div class="metrics"><div class="card"><b>{fmt(m(model,'MAE_kWh'),2)}</b><span>MAE [kWh/15 min]</span></div><div class="card"><b>{fmt(m(model,'RMSE_kWh'),2)}</b><span>RMSE [kWh/15 min]</span></div><div class="card"><b>{fmt(m(model,'MAPE_percent'),2,'%')}</b><span>MAPE</span></div><div class="card"><b>{fmt(m(model,'WAPE_percent'),2,'%')}</b><span>WAPE</span></div><div class="card"><b>{fmt(m(model,'R2'),3)}</b><span>R²</span></div><div class="card"><b>{fmt(rmse_skill*100,1,'%')}</b><span>RMSE skill vs weekly persistence</span></div></div>{df_html(metric_table, index=True)}<p><b>Derived benchmark comparisons:</b> MASE-like MAE ratio = <b>{fmt(mase,3)}</b>; MAE skill = <b>{fmt(mae_skill,3)}</b>; RMSE skill = <b>{fmt(rmse_skill,3)}</b>.</p>
    <h3>Definitions</h3><table><tr><th>Metric</th><th>Meaning</th></tr><tr><td>MAE</td><td>Average absolute 15-minute error.</td></tr><tr><td>RMSE</td><td>Square-error metric that penalizes large misses.</td></tr><tr><td>WAPE</td><td>Total absolute error divided by total actual load.</td></tr><tr><td>MAPE</td><td>Mean interval percentage error; sensitive to small actual values.</td></tr><tr><td>sMAPE</td><td>Symmetric percentage error.</td></tr><tr><td>R²</td><td>Share of target variance explained.</td></tr><tr><td>Energy bias</td><td>Total forecast energy minus total actual energy.</td></tr><tr><td>Skill</td><td>1 − model metric / reference metric; positive is better.</td></tr></table>

    <h2 id="daily">8. Daily performance distribution</h2>{df_html(daily_summary, index=True)}{examples}
    <h2 id="figures">9. Evaluation figures</h2>{source_figures}

    <h2 id="findings">10. Headline findings</h2><div class="key"><ul><li>Test-set MAPE: <b>{fmt(m(model,'MAPE_percent'),2,'%')}</b>; WAPE: <b>{fmt(m(model,'WAPE_percent'),2,'%')}</b>.</li><li>MAE: <b>{fmt(m(model,'MAE_kWh'),2)} kWh</b>; RMSE: <b>{fmt(m(model,'RMSE_kWh'),2)} kWh</b> per quarter-hour.</li><li>RMSE skill versus weekly persistence: <b>{fmt(rmse_skill,3)}</b>.</li><li>Overall energy bias: <b>{fmt(m(model,'Energy_bias_kWh'),2)} kWh</b>, or <b>{fmt(m(model,'Relative_bias_percent'),2,'%')}</b>.</li><li>All headline results come from the final 20%, not the training period.</li></ul></div>

    <h2 id="caveats">11. Caveats</h2><div class="warn"><ul><li>No actual weather forecast is supplied; only previous measured weather is available.</li><li>The fixed-panel result applies to the retained cohort.</li><li>The 20% remains a valid holdout only if it was not repeatedly used for model selection.</li><li>MAPE should be interpreted together with WAPE, MAE, RMSE, R², and bias.</li><li>No walk-forward validation or multi-model tournament is included yet.</li></ul></div>

    <h2 id="next">12. Next steps</h2><ol><li>Use rolling-origin validation strictly inside the first 80%.</li><li>Add daily persistence, linear regression, and random forest comparisons.</li><li>Add archived day-ahead weather forecasts when available.</li><li>Analyse seasonal, weekday/weekend, peak-hour, and temperature-regime errors.</li><li>Add quantile forecasts or prediction intervals for risk-aware allocation.</li></ol>
    <p class="sub">Generated automatically from <code>{esc(RESULTS_DIR)}</code>. This report generator performs no model training.</p></body></html>'''


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(build_report(), encoding="utf-8")
    print(f"Created: {OUTPUT_HTML.resolve()}")
    print("No model training was performed.")


if __name__ == "__main__":
    main()
