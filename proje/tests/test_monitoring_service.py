import csv

from app.services.monitoring_service import (
    LOG_COLUMNS,
    log_forecast,
    read_forecast_log,
)


LEGACY_COLUMNS = [
    "status",
    "warning_codes",
    "product_id",
    "product_name",
    "unit",
    "forecast_origin",
    "target_date",
    "lead_days",
    "demand_expected",
    "demand_probability",
    "decision_threshold",
    "conditional_quantity_prediction",
    "demand_prediction",
    "display_quantity",
    "stock_zero_transfer_quantity",
    "transfer_assumption",
    "model_version",
    "feature_builder_version",
    "data_freshness",
    "message",
    "created_at",
]


def _base_values():
    return [
        "forecast_ready",
        '["UNCALIBRATED_PROBABILITY"]',
        "100",
        "Test Ürün",
        "ADT",
        "2026-07-21",
        "2026-07-31",
        "10",
        "False",
        "0.1",
        "0.47",
        "1.2",
        "0",
        "0",
        "0",
        "Stok varsayımı",
        "v1",
        "1.0.0",
        "2026-07-21",
        "Talep beklenmiyor.",
        "2026-07-26T10:00:00+03:00",
    ]


def test_legacy_and_context_rows_are_migrated_without_loss(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "forecast_log.csv"
    old = _base_values()
    context = old[:-2] + [
        "{'school_status': 'summer_break'}",
        "{'source': 'time_safe_climatology', 'used_by_model': False}",
    ] + old[-2:]
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(LEGACY_COLUMNS)
        writer.writerow(old)
        writer.writerow(context)

    frame = read_forecast_log(tmp_path)
    assert len(frame) == 2
    assert list(frame.columns) == LOG_COLUMNS
    assert (log_dir / "forecast_log.pre_schema_v2_backup.csv").is_file()
    assert "time_safe_climatology" in frame.loc[1, "weather_context"]


def test_new_forecasts_always_use_fixed_schema(tmp_path):
    result = {
        "status": "forecast_ready",
        "warning_codes": [],
        "product_id": "100",
        "product_name": "Test Ürün",
        "unit": "KG",
        "forecast_origin": "2026-07-21",
        "target_date": "2026-08-01",
        "lead_days": 11,
        "demand_expected": False,
        "demand_probability": 0.2,
        "calendar_context": {"school_status": "summer_break"},
        "weather_context": {
            "source": "time_safe_climatology",
            "used_by_model": False,
        },
        "message": "Talep beklenmiyor.",
    }
    log_forecast(tmp_path, result)
    log_forecast(tmp_path, {**result, "extra_future_field": "ignored safely"})
    frame = read_forecast_log(tmp_path)
    assert len(frame) == 2
    assert list(frame.columns) == LOG_COLUMNS
