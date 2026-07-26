import ast
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


LOG_SCHEMA_VERSION = "2"
LOG_COLUMNS = [
    "log_schema_version",
    "created_at",
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
    "calendar_context",
    "weather_context",
    "message",
]


def _json_text(value, empty_value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        parsed = empty_value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value
    else:
        parsed = value
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def _canonical_record(record: Dict) -> Dict:
    output = {column: record.get(column) for column in LOG_COLUMNS}
    output["log_schema_version"] = LOG_SCHEMA_VERSION
    output["warning_codes"] = _json_text(record.get("warning_codes"), [])
    output["calendar_context"] = _json_text(record.get("calendar_context"), {})
    output["weather_context"] = _json_text(record.get("weather_context"), {})
    return output


def _read_rows_with_schema_recovery(log_path: Path) -> List[Dict]:
    """Eski 21 kolonlu ve yeni bağlam alanlı 23 kolonlu satırları kurtarır."""
    with log_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return []
        rows = []
        for line_number, values in enumerate(reader, start=2):
            if not values or not any(str(value).strip() for value in values):
                continue
            if header == LOG_COLUMNS and len(values) == len(LOG_COLUMNS):
                row_header = LOG_COLUMNS
            elif len(values) == len(header):
                row_header = header
            elif (
                len(values) == len(header) + 2
                and header[-2:] == ["message", "created_at"]
            ):
                # V3 servis sonucu iki yeni dict alanı içeriyordu; eski CSV başlığı
                # değişmeden append edildiği için pandas dosyayı okuyamıyordu.
                row_header = (
                    header[:-2]
                    + ["calendar_context", "weather_context"]
                    + header[-2:]
                )
            else:
                raise ValueError(
                    "Tahmin logu kurtarılamayan bir satır içeriyor: "
                    f"satır={line_number}, başlık={len(header)}, alan={len(values)}"
                )
            rows.append(_canonical_record(dict(zip(row_header, values))))
    return rows


def _rewrite_canonical_log(log_path: Path, rows: List[Dict]) -> None:
    frame = pd.DataFrame(rows, columns=LOG_COLUMNS)
    temporary_path = log_path.with_suffix(".schema_v2.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(log_path)


def _ensure_log_schema(log_path: Path) -> None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return
    with log_path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream), [])
    if header == LOG_COLUMNS:
        return

    rows = _read_rows_with_schema_recovery(log_path)
    backup_path = log_path.with_name("forecast_log.pre_schema_v2_backup.csv")
    if not backup_path.exists():
        shutil.copy2(log_path, backup_path)
    _rewrite_canonical_log(log_path, rows)


def log_forecast(project_root: Path, result: Dict) -> None:
    log_path = Path(project_root) / "logs" / "forecast_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_log_schema(log_path)
    row = _canonical_record(
        {
            **result,
            "created_at": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(),
        }
    )
    frame = pd.DataFrame([row], columns=LOG_COLUMNS)
    frame.to_csv(
        log_path,
        mode="a" if log_path.exists() else "w",
        header=not log_path.exists(),
        index=False,
    )


def read_forecast_log(project_root: Path) -> pd.DataFrame:
    log_path = Path(project_root) / "logs" / "forecast_log.csv"
    if not log_path.exists():
        return pd.DataFrame()
    _ensure_log_schema(log_path)
    return pd.read_csv(log_path)
