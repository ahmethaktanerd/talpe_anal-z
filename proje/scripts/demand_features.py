from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from scripts.turkey_calendar import (
    CALENDAR_MODEL_COLUMNS,
    build_turkey_calendar,
)
from scripts.weather_features import (
    WEATHER_MODEL_FEATURES,
    add_time_safe_weather_features,
    load_weather_reference,
)


FEATURE_BUILDER_VERSION = "3.0.0"

SNAPSHOT_FEATURES = [
    "product_age_days",
    "days_since_last_sale",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_sum_7",
    "rolling_mean_7",
    "rolling_positive_rate_7",
    "rolling_sum_14",
    "rolling_mean_14",
    "rolling_positive_rate_14",
    "rolling_sum_28",
    "rolling_mean_28",
    "rolling_median_28",
    "rolling_std_28",
    "rolling_positive_rate_28",
    "rolling_sum_90",
    "rolling_mean_90",
    "rolling_positive_rate_90",
    "expanding_mean_demand",
    "expanding_positive_rate",
    "expanding_positive_mean",
]

BASE_TARGET_DATE_FEATURES = [
    "lead_days",
    "target_day_of_week",
    "target_month",
    "target_is_weekend",
    "target_day_of_year_sin",
    "target_day_of_year_cos",
]

SPECIAL_CALENDAR_FEATURES = [
    f"target_{column}" for column in CALENDAR_MODEL_COLUMNS
]

DEPLOYED_TARGET_DATE_FEATURES = BASE_TARGET_DATE_FEATURES + SPECIAL_CALENDAR_FEATURES
TARGET_DATE_FEATURES = DEPLOYED_TARGET_DATE_FEATURES + WEATHER_MODEL_FEATURES

# Hava alanları model-ready tabloda aday olarak korunur; validation ablation
# sonucunda final V3 karar modelinden çıkarılmıştır.
MODEL_FEATURES = SNAPSHOT_FEATURES + DEPLOYED_TARGET_DATE_FEATURES
ALL_CANDIDATE_FEATURES = MODEL_FEATURES + WEATHER_MODEL_FEATURES


def add_snapshot_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    demand = group["daily_demand"].astype(float)
    past = demand.shift(1)

    group["product_age_days"] = (group["date"] - group["date"].min()).dt.days
    last_sale_date = group["date"].where(demand.gt(0)).ffill()
    group["days_since_last_sale"] = (group["date"] - last_sale_date).dt.days
    group["days_since_last_sale"] = group["days_since_last_sale"].fillna(
        group["product_age_days"] + 1
    )

    for lag in (1, 7, 14, 28):
        group[f"lag_{lag}"] = demand.shift(lag)

    for window in (7, 14, 28):
        rolling = past.rolling(window, min_periods=window)
        group[f"rolling_sum_{window}"] = rolling.sum()
        group[f"rolling_mean_{window}"] = rolling.mean()
        group[f"rolling_positive_rate_{window}"] = (
            past.gt(0).rolling(window, min_periods=window).mean()
        )

    rolling_28 = past.rolling(28, min_periods=28)
    group["rolling_median_28"] = rolling_28.median()
    group["rolling_std_28"] = rolling_28.std().fillna(0.0)
    rolling_90 = past.rolling(90, min_periods=28)
    group["rolling_sum_90"] = rolling_90.sum()
    group["rolling_mean_90"] = rolling_90.mean()
    group["rolling_positive_rate_90"] = (
        past.gt(0).rolling(90, min_periods=28).mean()
    )
    group["expanding_mean_demand"] = past.expanding(min_periods=28).mean()
    group["expanding_positive_rate"] = (
        past.gt(0).expanding(min_periods=28).mean()
    )
    group["expanding_positive_mean"] = (
        past.where(past.gt(0)).expanding(min_periods=1).mean()
    )
    return group


def build_snapshot_table(panel: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, group in panel.groupby("product_id", sort=False):
        frames.append(add_snapshot_features(group))
    result = pd.concat(frames, ignore_index=True)
    result["feature_available"] = result[SNAPSHOT_FEATURES].notna().all(axis=1)
    return result


def add_target_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    target_date = pd.to_datetime(frame["target_date"])
    frame["target_day_of_week"] = target_date.dt.dayofweek
    frame["target_month"] = target_date.dt.month
    frame["target_is_weekend"] = target_date.dt.dayofweek.ge(5).astype(int)
    day_of_year = target_date.dt.dayofyear
    frame["target_day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["target_day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    calendar = build_turkey_calendar(target_date.min(), target_date.max()).set_index(
        "date"
    )
    for column in CALENDAR_MODEL_COLUMNS:
        mapped = target_date.map(calendar[column])
        if mapped.isna().any():
            raise ValueError(f"Takvim feature'ı eşleşmedi: {column}")
        frame[f"target_{column}"] = mapped.to_numpy(dtype=np.int16)
    return frame


def build_direct_horizon_examples(
    snapshots: pd.DataFrame,
    lead_day_grid: Iterable[int],
    observed_weather: pd.DataFrame,
    forecast_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Her snapshot için deterministik bir lead seçer ve günlük geleceği hedefler."""
    base = snapshots.loc[
        snapshots["feature_available"] & snapshots["store_observed"].eq(1)
    ].copy()
    leads = np.asarray(list(lead_day_grid), dtype=int)
    product_hash = pd.util.hash_pandas_object(base["product_id"], index=False).to_numpy()
    date_ordinal = base["date"].map(pd.Timestamp.toordinal).to_numpy(dtype=np.int64)
    selector = (
        (product_hash.astype(np.uint64) + date_ordinal.astype(np.uint64))
        % np.uint64(len(leads))
    ).astype(np.int64)
    base["lead_days"] = leads[selector]
    base["forecast_origin"] = base["date"]
    base["target_date"] = base["forecast_origin"] + pd.to_timedelta(
        base["lead_days"], unit="D"
    )

    target_lookup = snapshots[
        ["product_id", "date", "daily_demand", "store_observed"]
    ].rename(
        columns={
            "date": "target_date",
            "daily_demand": "target_demand",
            "store_observed": "target_store_observed",
        }
    )
    examples = base.merge(
        target_lookup, on=["product_id", "target_date"], how="left", validate="many_to_one"
    )
    examples = examples.loc[
        examples["target_store_observed"].eq(1)
        & examples["target_demand"].notna()
        & examples["target_date"].gt(examples["forecast_origin"])
    ].copy()
    examples["demand_occurs"] = examples["target_demand"].gt(0).astype(int)
    examples = add_target_calendar_features(examples)
    examples = add_time_safe_weather_features(
        examples,
        observed=observed_weather,
        forecasts=forecast_weather,
    )
    return examples


def build_inference_features(
    snapshots: pd.DataFrame,
    product_catalog: pd.DataFrame,
    forecast_origin: pd.Timestamp,
) -> pd.DataFrame:
    latest = snapshots.loc[
        snapshots["date"].eq(pd.Timestamp(forecast_origin))
        & snapshots["feature_available"]
        & snapshots["store_observed"].eq(1)
    ].copy()
    if "current_status" not in latest.columns:
        catalog_cols = ["product_id", "current_status"]
        latest = latest.merge(product_catalog[catalog_cols], on="product_id", how="left")
    return latest


def prepare_single_target_date(
    snapshot_row: pd.DataFrame,
    target_date: pd.Timestamp,
    observed_weather: pd.DataFrame = None,
    forecast_weather: pd.DataFrame = None,
) -> pd.DataFrame:
    if len(snapshot_row) != 1:
        raise ValueError("Tek ürün için tam olarak bir snapshot satırı gerekir.")
    frame = snapshot_row.copy()
    frame["forecast_origin"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.Timestamp(target_date)
    frame["lead_days"] = (
        frame["target_date"] - frame["forecast_origin"]
    ).dt.days.astype(int)
    frame = add_target_calendar_features(frame)
    if observed_weather is None or forecast_weather is None:
        observed_weather, forecast_weather = load_weather_reference()
    return add_time_safe_weather_features(
        frame,
        observed=observed_weather,
        forecasts=forecast_weather,
    )
