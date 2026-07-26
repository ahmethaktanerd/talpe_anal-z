"""Bursa hava feature'larını tahmin-anı güvenli biçimde üretir."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scripts.project_config import REFERENCE_DIR


WEATHER_FEATURE_VERSION = "BURSA_WEATHER_FEATURES_V1"
WEATHER_FORECAST_MAX_LEAD_DAYS = 7
CLIMATOLOGY_WINDOW_DAYS = 15
MIN_CLIMATOLOGY_SAMPLES = 90

WEATHER_VALUE_FEATURES = [
    "target_weather_temperature_mean_c",
    "target_weather_temperature_max_c",
    "target_weather_precipitation_mm",
    "target_weather_snowfall_cm",
    "target_weather_cloud_cover_pct",
    "target_weather_wind_max_kmh",
    "target_weather_solar_radiation_mj_m2",
]
WEATHER_MODEL_FEATURES = WEATHER_VALUE_FEATURES

OBSERVED_TO_FEATURE = {
    "temperature_mean_c": "target_weather_temperature_mean_c",
    "temperature_max_c": "target_weather_temperature_max_c",
    "precipitation_mm": "target_weather_precipitation_mm",
    "snowfall_cm": "target_weather_snowfall_cm",
    "cloud_cover_pct": "target_weather_cloud_cover_pct",
    "wind_max_kmh": "target_weather_wind_max_kmh",
    "solar_radiation_mj_m2": "target_weather_solar_radiation_mj_m2",
}


def load_weather_reference(
    reference_dir: Path = REFERENCE_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(
        reference_dir / "bursa_weather_observed_era5.csv",
        parse_dates=["date"],
    )
    forecasts = pd.read_csv(
        reference_dir / "bursa_weather_previous_runs.csv",
        parse_dates=["target_date"],
    )
    expected_observed = {"date", *OBSERVED_TO_FEATURE}
    expected_forecast = {"target_date", "lead_days", *WEATHER_VALUE_FEATURES}
    if not expected_observed.issubset(observed.columns):
        raise ValueError("ERA5 Bursa hava referans kolonları eksik.")
    if not expected_forecast.issubset(forecasts.columns):
        raise ValueError("Bursa previous-runs hava referans kolonları eksik.")
    if not forecasts.empty and forecasts.duplicated(
        ["target_date", "lead_days"]
    ).any():
        raise ValueError("Hava forecast referansında target_date/lead_days tekil değil.")
    return observed.sort_values("date"), forecasts.sort_values(
        ["target_date", "lead_days"]
    )


def _climatology_day(values: pd.Series) -> np.ndarray:
    values = pd.to_datetime(values)
    anchor = pd.to_datetime(
        {
            "year": np.full(len(values), 2000),
            "month": values.dt.month,
            "day": values.dt.day,
        }
    )
    return anchor.dt.dayofyear.to_numpy(dtype=np.int16)


def _circular_distance(days: np.ndarray, target_day: int) -> np.ndarray:
    direct = np.abs(days.astype(int) - int(target_day))
    return np.minimum(direct, 366 - direct)


def _build_climatology_rows(
    keys: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    weather = observed.copy()
    weather["climatology_day"] = _climatology_day(weather["date"])
    output = []
    cache: Dict[Tuple[pd.Timestamp, int], Dict] = {}

    for row in keys.itertuples(index=False):
        origin = pd.Timestamp(row.forecast_origin).normalize()
        target = pd.Timestamp(row.target_date).normalize()
        target_day = int(
            _climatology_day(pd.Series([target], dtype="datetime64[ns]"))[0]
        )
        cache_key = (origin, target_day)
        if cache_key not in cache:
            history = weather.loc[weather["date"].lt(origin)]
            distances = _circular_distance(
                history["climatology_day"].to_numpy(), target_day
            )
            sample = history.loc[distances <= CLIMATOLOGY_WINDOW_DAYS]
            if len(sample) < MIN_CLIMATOLOGY_SAMPLES:
                raise ValueError(
                    f"Yetersiz time-safe hava klimatolojisi: origin={origin.date()}, "
                    f"target_day={target_day}, n={len(sample)}"
                )
            feature_values = {
                feature: float(sample[source].mean())
                for source, feature in OBSERVED_TO_FEATURE.items()
            }
            cache[cache_key] = {
                **feature_values,
                "target_weather_is_fixed_lead_forecast": 0,
                "weather_feature_source": "time_safe_climatology",
                "weather_source_max_date": sample["date"].max(),
                "weather_climatology_sample_days": int(len(sample)),
            }
        output.append(
            {
                "forecast_origin": origin,
                "target_date": target,
                "lead_days": int(row.lead_days),
                **cache[cache_key],
            }
        )
    return pd.DataFrame(output)


def add_time_safe_weather_features(
    frame: pd.DataFrame,
    observed: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Hedef-gün gerçekleşen hava değerini hiçbir koşulda kullanmaz."""
    required = {"forecast_origin", "target_date", "lead_days"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Hava feature girdisinde kolon eksik: {sorted(required)}")
    result = frame.copy()
    result["forecast_origin"] = pd.to_datetime(result["forecast_origin"]).dt.normalize()
    result["target_date"] = pd.to_datetime(result["target_date"]).dt.normalize()
    result["lead_days"] = result["lead_days"].astype(int)

    keys = result[list(required)].drop_duplicates().reset_index(drop=True)
    climatology = _build_climatology_rows(keys, observed)
    exact_forecasts = forecasts.copy()
    exact_forecasts["target_date"] = pd.to_datetime(
        exact_forecasts["target_date"]
    ).dt.normalize()

    safe_forecasts = keys.merge(
        exact_forecasts,
        on=["target_date", "lead_days"],
        how="inner",
        validate="one_to_one",
    )
    safe_forecasts = safe_forecasts.loc[
        safe_forecasts["lead_days"].between(1, WEATHER_FORECAST_MAX_LEAD_DAYS)
        & (
            safe_forecasts["target_date"]
            - pd.to_timedelta(safe_forecasts["lead_days"], unit="D")
        ).eq(safe_forecasts["forecast_origin"])
    ].copy()
    safe_forecasts["target_weather_is_fixed_lead_forecast"] = 1
    safe_forecasts["weather_feature_source"] = "archived_fixed_lead_forecast"
    safe_forecasts["weather_source_max_date"] = safe_forecasts["forecast_origin"]
    safe_forecasts["weather_climatology_sample_days"] = 0

    if not safe_forecasts.empty:
        replacement_keys = set(
            zip(
                safe_forecasts["forecast_origin"],
                safe_forecasts["target_date"],
                safe_forecasts["lead_days"],
            )
        )
        climatology_key = list(
            zip(
                climatology["forecast_origin"],
                climatology["target_date"],
                climatology["lead_days"],
            )
        )
        climatology = climatology.loc[
            [key not in replacement_keys for key in climatology_key]
        ]
    if safe_forecasts.empty:
        weather_features = climatology
    else:
        weather_features = pd.concat(
            [climatology, safe_forecasts[climatology.columns]],
            ignore_index=True,
        )
    result = result.merge(
        weather_features,
        on=["forecast_origin", "target_date", "lead_days"],
        how="left",
        validate="many_to_one",
    )
    if result[WEATHER_MODEL_FEATURES].isna().any().any():
        raise AssertionError("Time-safe hava feature alanlarında eksik değer bulundu.")
    if not pd.to_datetime(result["weather_source_max_date"]).le(
        result["forecast_origin"]
    ).all():
        raise AssertionError("Hava feature kaynak tarihi forecast origin sonrasına geçti.")
    return result


def weather_context_from_row(row: pd.Series) -> Dict:
    source = str(row["weather_feature_source"])
    return {
        "location": "Bursa/Osmangazi şehir merkezi",
        "source": source,
        "is_fixed_lead_forecast": bool(
            row["target_weather_is_fixed_lead_forecast"]
        ),
        "temperature_mean_c": round(
            float(row["target_weather_temperature_mean_c"]), 1
        ),
        "temperature_max_c": round(
            float(row["target_weather_temperature_max_c"]), 1
        ),
        "precipitation_mm": round(
            float(row["target_weather_precipitation_mm"]), 2
        ),
        "snowfall_cm": round(float(row["target_weather_snowfall_cm"]), 2),
        "cloud_cover_pct": round(
            float(row["target_weather_cloud_cover_pct"]), 1
        ),
        "wind_max_kmh": round(float(row["target_weather_wind_max_kmh"]), 1),
        "solar_radiation_mj_m2": round(
            float(row["target_weather_solar_radiation_mj_m2"]), 2
        ),
        "climatology_sample_days": int(row["weather_climatology_sample_days"]),
    }
