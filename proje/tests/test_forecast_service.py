from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.services.forecast_service import DemandForecastService


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def service():
    return DemandForecastService(ROOT)


def test_kg_and_adt_forecast(service):
    origin = pd.Timestamp(service.metadata["forecast_origin"]).date()
    options = service.product_options()
    for unit in ("KG", "ADT"):
        row = options.loc[
            options["unit"].eq(unit) & options["forecast_available"]
        ].iloc[0]
        result = service.forecast(
            str(row["product_id"]), origin + timedelta(days=164)
        )
        assert result["unit"] == unit
        assert result["lead_days"] == 164
        assert 0 <= result["demand_probability"] <= 1
        assert result["demand_prediction"] >= 0
        assert result["weather_context"]["source"] == "time_safe_climatology"
        assert not result["weather_context"]["is_fixed_lead_forecast"]
        assert not result["weather_context"]["used_by_model"]


def test_user_can_choose_second_january_2027(service):
    options = service.product_options()
    row = options.loc[
        options["current_status"].eq("active") & options["forecast_available"]
    ].iloc[0]
    result = service.forecast(
        str(row["product_id"]), pd.Timestamp("2027-01-02").date()
    )
    assert result["target_date"] == "2027-01-02"
    assert result["lead_days"] == 165
    assert result["unit"] in {"KG", "ADT"}
    assert result["display_quantity"] >= 0
    assert result["stock_zero_transfer_quantity"] == result["display_quantity"]
    assert result["weather_context"]["climatology_sample_days"] >= 90


def test_unknown_product(service):
    origin = pd.Timestamp(service.metadata["forecast_origin"]).date()
    with pytest.raises(ValueError):
        service.forecast("__UNKNOWN__", origin + timedelta(days=7))


def test_product_history_for_interactive_chart(service):
    options = service.product_options()
    row = options.loc[options["forecast_available"]].iloc[0]
    history = service.product_history(str(row["product_id"]), days=90)
    assert not history.empty
    assert history["product_id"].astype(str).eq(str(row["product_id"])).all()
    assert history["date"].is_monotonic_increasing
    assert len(history) <= 90


def test_short_history_returns_status(service):
    origin = pd.Timestamp(service.metadata["forecast_origin"]).date()
    options = service.product_options()
    unavailable = options.loc[~options["forecast_available"]]
    if unavailable.empty:
        pytest.skip("Bütün ürünlerde yeterli snapshot geçmişi var.")
    result = service.forecast(
        str(unavailable.iloc[0]["product_id"]), origin + timedelta(days=7)
    )
    assert result["status"] == "insufficient_history"
