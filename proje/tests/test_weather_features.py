from pathlib import Path

import pandas as pd

from scripts.weather_features import (
    WEATHER_MODEL_FEATURES,
    add_time_safe_weather_features,
    load_weather_reference,
)


ROOT = Path(__file__).resolve().parents[1]


def test_weather_climatology_never_reads_target_or_future_weather():
    observed, forecasts = load_weather_reference(
        ROOT / "data" / "reference"
    )
    frame = pd.DataFrame(
        {
            "forecast_origin": pd.to_datetime(
                ["2023-02-01", "2026-07-21", "2026-07-21"]
            ),
            "target_date": pd.to_datetime(
                ["2023-02-08", "2026-07-28", "2027-01-02"]
            ),
            "lead_days": [7, 7, 165],
        }
    )
    result = add_time_safe_weather_features(frame, observed, forecasts)
    assert result[WEATHER_MODEL_FEATURES].notna().all(axis=None)
    assert (
        pd.to_datetime(result["weather_source_max_date"])
        < result["forecast_origin"]
    ).all()
    assert result["target_weather_is_fixed_lead_forecast"].eq(0).all()
    assert result["weather_feature_source"].eq("time_safe_climatology").all()


def test_changing_post_origin_observations_does_not_change_features():
    observed, forecasts = load_weather_reference(
        ROOT / "data" / "reference"
    )
    frame = pd.DataFrame(
        {
            "forecast_origin": [pd.Timestamp("2025-01-15")],
            "target_date": [pd.Timestamp("2025-02-15")],
            "lead_days": [31],
        }
    )
    baseline = add_time_safe_weather_features(frame, observed, forecasts)
    changed = observed.copy()
    changed.loc[changed["date"].ge("2025-01-15"), "temperature_mean_c"] = 9999
    repeated = add_time_safe_weather_features(frame, changed, forecasts)
    pd.testing.assert_series_equal(
        baseline.loc[0, WEATHER_MODEL_FEATURES],
        repeated.loc[0, WEATHER_MODEL_FEATURES],
        check_names=False,
    )
