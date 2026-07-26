from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from app.services.bundle_loader import load_metadata, load_unit_models
from app.services.input_validator import validate_target_date, validate_unit
from models.demand_forecasting_bundle.feature_builder import (
    FEATURE_BUILDER_VERSION,
    MODEL_FEATURES,
    describe_calendar_date,
    prepare_single_target_date,
)
from scripts.weather_features import (
    WEATHER_FEATURE_VERSION,
    load_weather_reference,
    weather_context_from_row,
)


class DemandForecastService:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.bundle_dir = self.project_root / "models" / "demand_forecasting_bundle"
        self.metadata = load_metadata(self.bundle_dir)
        self.snapshot = pd.read_csv(
            self.project_root / "data" / "model_ready" / "inference_snapshot.csv",
            dtype={"product_id": "string"},
            parse_dates=["date"],
        )
        self.catalog = pd.read_csv(
            self.bundle_dir / "product_catalog.csv",
            dtype={"product_id": "string"},
            parse_dates=["first_sale", "last_sale"],
        )
        self.observed_weather, self.forecast_weather = load_weather_reference(
            self.bundle_dir
        )
        self.models = {
            unit: load_unit_models(self.bundle_dir, self.metadata, unit)
            for unit in self.metadata["units"]
        }
        self._daily_history = None
        expected_features = list(self.metadata["feature_columns"])
        if expected_features != MODEL_FEATURES:
            raise ValueError("Bundle feature listesi kod sürümüyle uyuşmuyor.")
        if self.metadata["feature_builder_version"] != FEATURE_BUILDER_VERSION:
            raise ValueError("Bundle feature builder sürümü metadata ile uyuşmuyor.")
        if self.metadata["weather_feature_version"] != WEATHER_FEATURE_VERSION:
            raise ValueError("Bundle hava feature sürümü kodla uyuşmuyor.")

    def product_options(self) -> pd.DataFrame:
        available_ids = set(self.snapshot["product_id"].astype(str))
        options = self.catalog[
            [
                "product_id",
                "product_name",
                "unit",
                "current_status",
                "history_days",
                "days_since_last_sale",
            ]
        ].copy()
        options["forecast_available"] = options["product_id"].astype(str).isin(
            available_ids
        )
        options["label"] = (
            options["product_name"]
            + " · "
            + options["unit"]
            + " · ID "
            + options["product_id"].astype(str)
        )
        options["status_order"] = np.where(
            options["current_status"].eq("active"), 0, 1
        )
        return options.sort_values(
            ["forecast_available", "status_order", "product_name"],
            ascending=[False, True, True],
        ).drop(columns="status_order")

    def product_history(self, product_id: str, days: int = 180) -> pd.DataFrame:
        """Arayüz grafiği için ürünün gözlenen son günlük talep geçmişini döndürür."""
        if days < 7 or days > 730:
            raise ValueError("Geçmiş grafik aralığı 7–730 gün olmalıdır.")
        if self._daily_history is None:
            self._daily_history = pd.read_csv(
                self.project_root / "data" / "processed" / "daily_product_demand.csv",
                usecols=[
                    "date",
                    "product_id",
                    "unit",
                    "daily_demand",
                    "store_observed",
                ],
                dtype={"product_id": "string"},
                parse_dates=["date"],
            )
        origin = pd.Timestamp(self.metadata["forecast_origin"])
        start = origin - pd.Timedelta(days=days - 1)
        history = self._daily_history.loc[
            self._daily_history["product_id"].astype(str).eq(str(product_id))
            & self._daily_history["date"].between(start, origin)
            & self._daily_history["store_observed"].eq(1)
        ].copy()
        return history.sort_values("date").reset_index(drop=True)

    def forecast(self, product_id: str, target_date: date) -> Dict:
        product_id = str(product_id)
        lead_days = validate_target_date(target_date, self.metadata)
        catalog_row = self.catalog.loc[
            self.catalog["product_id"].astype(str).eq(product_id)
        ]
        if catalog_row.empty:
            raise ValueError("Ürün kataloğunda bu product_id bulunamadı.")
        catalog_row = catalog_row.iloc[0]
        unit = validate_unit(catalog_row["unit"])

        snapshot_row = self.snapshot.loc[
            self.snapshot["product_id"].astype(str).eq(product_id)
        ]
        if snapshot_row.empty:
            return {
                "status": "insufficient_history",
                "warning_codes": ["INSUFFICIENT_28_DAY_HISTORY"],
                "product_id": product_id,
                "product_name": catalog_row["product_name"],
                "unit": unit,
                "target_date": target_date.isoformat(),
                "forecast_origin": self.metadata["forecast_origin"],
                "lead_days": lead_days,
                "message": "Bu ürün için güvenli tahmin üretmeye yetecek 28 günlük geçmiş yok.",
            }
        if len(snapshot_row) != 1:
            raise ValueError("Inference snapshot içinde ürün satırı tekil değil.")

        model_frame = prepare_single_target_date(
            snapshot_row,
            pd.Timestamp(target_date),
            observed_weather=self.observed_weather,
            forecast_weather=self.forecast_weather,
        )
        x = model_frame[self.metadata["feature_columns"]].astype(np.float32)
        probability = float(self.models[unit]["occurrence"].predict_proba(x)[0, 1])
        conditional_quantity = float(
            np.clip(self.models[unit]["quantity"].predict(x)[0], 0, None)
        )
        threshold = float(self.metadata["occurrence_threshold"][unit])
        demand_expected = probability >= threshold
        raw_prediction = conditional_quantity if demand_expected else 0.0
        display_quantity = (
            int(max(1, round(raw_prediction))) if unit == "ADT" and demand_expected else round(raw_prediction, 3)
        )
        calendar_context = describe_calendar_date(pd.Timestamp(target_date))
        weather_context = weather_context_from_row(model_frame.iloc[0])
        weather_context["used_by_model"] = bool(self.metadata["weather_features"])
        warnings: List[str] = []
        if catalog_row["current_status"] != "active":
            warnings.append("POSSIBLY_INACTIVE_PRODUCT")
        if lead_days > 90:
            warnings.append("LONG_RANGE_FORECAST")
        if not self.metadata["probability_calibrated"][unit]:
            warnings.append("UNCALIBRATED_PROBABILITY")

        status = "forecast_ready_with_warning" if warnings else "forecast_ready"
        message = (
            f"Talep bekleniyor; mağaza talebi yaklaşık {display_quantity} {unit}."
            if demand_expected
            else "Bu hedef gün için talep beklenmiyor."
        )
        return {
            "status": status,
            "warning_codes": warnings,
            "product_id": product_id,
            "product_name": catalog_row["product_name"],
            "unit": unit,
            "forecast_origin": self.metadata["forecast_origin"],
            "target_date": target_date.isoformat(),
            "lead_days": lead_days,
            "demand_expected": bool(demand_expected),
            "demand_probability": probability,
            "decision_threshold": threshold,
            "conditional_quantity_prediction": conditional_quantity,
            "demand_prediction": raw_prediction,
            "display_quantity": display_quantity,
            "stock_zero_transfer_quantity": display_quantity,
            "transfer_assumption": (
                "Mağaza stoku, yoldaki sevkiyat ve emniyet stoku sıfır kabul edilir."
            ),
            "model_version": self.metadata["model_version"],
            "feature_builder_version": self.metadata["feature_builder_version"],
            "data_freshness": self.metadata["forecast_origin"],
            "calendar_context": calendar_context,
            "weather_context": weather_context,
            "message": message,
        }

    def batch_forecast(self, requests: pd.DataFrame) -> pd.DataFrame:
        required = {"product_id", "target_date"}
        missing = required - set(requests.columns)
        if missing:
            raise ValueError(f"Eksik toplu tahmin kolonları: {sorted(missing)}")
        results = []
        for row in requests.itertuples(index=False):
            try:
                target = pd.Timestamp(row.target_date).date()
                results.append(self.forecast(str(row.product_id), target))
            except Exception as error:
                results.append(
                    {
                        "product_id": str(row.product_id),
                        "target_date": str(row.target_date),
                        "status": "error",
                        "message": str(error),
                    }
                )
        return pd.DataFrame(results)
