"""Bursa hava verisini tekrarlanabilir biçimde indirir ve kaynak manifesti üretir.

İki ayrı veri sözleşmesi vardır:

1. ERA5 günlük gerçekleşen hava: yalnız iklim normali ve betimsel EDA için.
2. Previous Runs sabit-ufuk tahmini: hedef gün için 1–7 gün önceden yayımlanmış
   tahmin. Model eğitiminde gerçekleşmiş hedef-gün havası kesinlikle kullanılmaz.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from scripts.data_utils import clean_sales_records, file_sha256, read_raw_sales
from scripts.project_config import REFERENCE_DIR, ensure_project_dirs


WEATHER_DATA_VERSION = "BURSA_WEATHER_2026_07_V1"
LOCATION = {
    "name": "Bursa",
    "admin2": "Osmangazi",
    "country_code": "TR",
    "latitude": 40.19559,
    "longitude": 29.06013,
    "elevation_m": 155.0,
    "timezone": "Europe/Istanbul",
    "geocoding_id": 750269,
}
OBSERVED_START_DATE = pd.Timestamp("2018-01-01")
FORECAST_ARCHIVE_START_DATE = pd.Timestamp("2024-01-01")
FORECAST_LEADS = tuple(range(1, 8))

OBSERVED_DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "snowfall_sum",
    "cloud_cover_mean",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
]

FORECAST_HOURLY_BASES = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]
FORECAST_OUTPUT_COLUMNS = [
    "target_date",
    "lead_days",
    "target_weather_temperature_mean_c",
    "target_weather_temperature_max_c",
    "target_weather_precipitation_mm",
    "target_weather_snowfall_cm",
    "target_weather_cloud_cover_pct",
    "target_weather_wind_max_kmh",
    "target_weather_solar_radiation_mj_m2",
]


def _request_json(base_url: str, params: Dict[str, object], retries: int = 4) -> Dict:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{base_url}?{query}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "BursaRetailDemandForecast/1.0 "
                        "(reproducible ML weather feature pipeline)"
                    )
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # pragma: no cover - yalnız ağ hatasında çalışır
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Hava API isteği başarısız: {url}") from last_error


def _year_intervals(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple]:
    cursor = start.normalize()
    while cursor <= end:
        interval_end = min(pd.Timestamp(cursor.year, 12, 31), end)
        yield cursor, interval_end
        cursor = interval_end + pd.Timedelta(days=1)


def fetch_observed_weather(end_date: pd.Timestamp) -> pd.DataFrame:
    payload = _request_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": LOCATION["latitude"],
            "longitude": LOCATION["longitude"],
            "start_date": OBSERVED_START_DATE.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "timezone": LOCATION["timezone"],
            "models": "era5",
            "daily": ",".join(OBSERVED_DAILY_VARIABLES),
        },
    )
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "ERA5 isteği başarısız."))
    daily = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename(
        columns={
            "temperature_2m_mean": "temperature_mean_c",
            "temperature_2m_max": "temperature_max_c",
            "precipitation_sum": "precipitation_mm",
            "snowfall_sum": "snowfall_cm",
            "cloud_cover_mean": "cloud_cover_pct",
            "wind_speed_10m_max": "wind_max_kmh",
            "shortwave_radiation_sum": "solar_radiation_mj_m2",
        }
    )
    value_columns = [column for column in daily.columns if column != "date"]
    # ERA5 yaklaşık beş gün gecikmeli yayımlanır; API son istenen güne boş bir
    # satır döndürebilir. Kısmi satırı doldurmak yerine tümüyle dışarıda bırakırız.
    daily = daily.dropna(subset=value_columns)
    if daily.empty or daily["date"].max() < end_date - pd.Timedelta(days=10):
        raise ValueError("ERA5 hava verisi beklenenden fazla gecikmeli veya boş.")
    return daily.sort_values("date").reset_index(drop=True)


def _forecast_hourly_variables() -> List[str]:
    return [
        f"{base}_previous_day{lead}"
        for base in FORECAST_HOURLY_BASES
        for lead in FORECAST_LEADS
    ]


def _aggregate_forecast_payload(payload: Dict) -> pd.DataFrame:
    hourly = pd.DataFrame(payload["hourly"]).rename(columns={"time": "timestamp"})
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    hourly["target_date"] = hourly["timestamp"].dt.normalize()
    daily_frames = []
    for lead in FORECAST_LEADS:
        suffix = f"_previous_day{lead}"
        selected = hourly[
            [
                "target_date",
                f"temperature_2m{suffix}",
                f"precipitation{suffix}",
                f"snowfall{suffix}",
                f"cloud_cover{suffix}",
                f"wind_speed_10m{suffix}",
                f"shortwave_radiation{suffix}",
            ]
        ].copy()
        selected = selected.rename(
            columns={
                f"temperature_2m{suffix}": "temperature",
                f"precipitation{suffix}": "precipitation",
                f"snowfall{suffix}": "snowfall",
                f"cloud_cover{suffix}": "cloud_cover",
                f"wind_speed_10m{suffix}": "wind_speed",
                f"shortwave_radiation{suffix}": "shortwave_radiation",
            }
        )
        grouped = (
            selected.groupby("target_date", as_index=False)
            .agg(
                target_weather_temperature_mean_c=("temperature", "mean"),
                target_weather_temperature_max_c=("temperature", "max"),
                target_weather_precipitation_mm=("precipitation", "sum"),
                target_weather_snowfall_cm=("snowfall", "sum"),
                target_weather_cloud_cover_pct=("cloud_cover", "mean"),
                target_weather_wind_max_kmh=("wind_speed", "max"),
                shortwave_radiation_hourly_sum=("shortwave_radiation", "sum"),
                non_null_hours=("temperature", "count"),
            )
        )
        # Hourly W/m² değerlerini günlük enerji toplamına (MJ/m²) dönüştürür.
        grouped["target_weather_solar_radiation_mj_m2"] = (
            grouped.pop("shortwave_radiation_hourly_sum") * 0.0036
        )
        grouped["lead_days"] = lead
        daily_frames.append(grouped)
    return pd.concat(daily_frames, ignore_index=True)


def fetch_archived_forecasts(end_date: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for start, end in _year_intervals(FORECAST_ARCHIVE_START_DATE, end_date):
        payload = _request_json(
            "https://previous-runs-api.open-meteo.com/v1/forecast",
            {
                "latitude": LOCATION["latitude"],
                "longitude": LOCATION["longitude"],
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "timezone": LOCATION["timezone"],
                "hourly": ",".join(_forecast_hourly_variables()),
            },
        )
        if payload.get("error"):
            raise RuntimeError(payload.get("reason", "Previous Runs isteği başarısız."))
        frames.append(_aggregate_forecast_payload(payload))
    result = pd.concat(frames, ignore_index=True)
    result = result.loc[result["non_null_hours"].eq(24)].drop(
        columns="non_null_hours"
    )
    feature_columns = [
        column
        for column in result.columns
        if column not in {"target_date", "lead_days"}
    ]
    result = result.dropna(subset=feature_columns)
    return result.sort_values(["target_date", "lead_days"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-archived-forecasts",
        action="store_true",
        help=(
            "Previous Runs 1–7 günlük arşivini de indirir. Büyük ve yavaş bir "
            "istektir; V1 üretim modeli yalnız time-safe klimatoloji kullanır."
        ),
    )
    args = parser.parse_args()
    ensure_project_dirs()
    raw = read_raw_sales()
    data_end = clean_sales_records(raw)["date"].max()
    if pd.isna(data_end):
        raise ValueError("Ham satış verisinden son tarih okunamadı.")

    observed = fetch_observed_weather(data_end)
    if args.include_archived_forecasts:
        forecast_end = data_end + pd.Timedelta(days=max(FORECAST_LEADS))
        forecasts = fetch_archived_forecasts(forecast_end)
    else:
        forecasts = pd.DataFrame(columns=FORECAST_OUTPUT_COLUMNS)

    observed_path = REFERENCE_DIR / "bursa_weather_observed_era5.csv"
    forecasts_path = REFERENCE_DIR / "bursa_weather_previous_runs.csv"
    observed.to_csv(observed_path, index=False)
    forecasts.to_csv(forecasts_path, index=False)

    manifest = {
        "weather_data_version": WEATHER_DATA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "location": LOCATION,
        "location_assumption": (
            "Mağaza açık adresi verilmediği için Bursa/Osmangazi şehir merkezi "
            "koordinatı kullanıldı."
        ),
        "observed_weather": {
            "purpose": "time-safe climatology and descriptive EDA only",
            "provider": "Open-Meteo Historical Weather API",
            "upstream_model": "ERA5",
            "endpoint": "https://archive-api.open-meteo.com/v1/archive",
            "start": observed["date"].min().date().isoformat(),
            "end": observed["date"].max().date().isoformat(),
            "file": str(observed_path.relative_to(REFERENCE_DIR.parent.parent)),
            "sha256": file_sha256(observed_path),
        },
        "archived_forecasts": {
            "enabled_in_current_bundle": bool(args.include_archived_forecasts),
            "purpose": (
                "optional future forecast-as-of model features for fixed 1-7 "
                "day leads; disabled in V1 to avoid incomplete archive and "
                "train-serving skew"
            ),
            "provider": "Open-Meteo Previous Model Runs API",
            "endpoint": "https://previous-runs-api.open-meteo.com/v1/forecast",
            "lead_days": list(FORECAST_LEADS),
            "start": (
                forecasts["target_date"].min().date().isoformat()
                if not forecasts.empty
                else None
            ),
            "end": (
                forecasts["target_date"].max().date().isoformat()
                if not forecasts.empty
                else None
            ),
            "file": str(forecasts_path.relative_to(REFERENCE_DIR.parent.parent)),
            "sha256": file_sha256(forecasts_path),
        },
        "geocoding_source": (
            "https://geocoding-api.open-meteo.com/v1/search"
            "?name=Bursa&count=5&language=tr&countryCode=TR"
        ),
        "documentation": [
            "https://open-meteo.com/en/docs/historical-weather-api",
            "https://open-meteo.com/en/docs/previous-runs-api",
            "https://open-meteo.com/en/docs",
        ],
        "leakage_contract": (
            "Observed target-date weather is never a model feature. Current V1 uses "
            "only climatology calculated from weather dates strictly before "
            "forecast_origin. Optional fixed-lead forecasts may be enabled only after "
            "complete train/validation archive verification."
        ),
    }
    manifest_path = REFERENCE_DIR / "weather_sources.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"ERA5: {len(observed):,} gün | Previous Runs: {len(forecasts):,} "
        f"hedef-gün/lead | son satış tarihi: {data_end.date()}"
    )


if __name__ == "__main__":
    main()
