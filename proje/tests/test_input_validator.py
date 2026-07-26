from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.services.bundle_loader import load_metadata
from app.services.input_validator import validate_target_date, validate_unit


ROOT = Path(__file__).resolve().parents[1]
METADATA = load_metadata(ROOT / "models" / "demand_forecasting_bundle")
ORIGIN = pd.Timestamp(METADATA["forecast_origin"]).date()


def test_target_date_range():
    assert validate_target_date(ORIGIN + timedelta(days=1), METADATA) == 1
    assert validate_target_date(ORIGIN + timedelta(days=180), METADATA) == 180
    with pytest.raises(ValueError):
        validate_target_date(ORIGIN, METADATA)
    with pytest.raises(ValueError):
        validate_target_date(ORIGIN + timedelta(days=181), METADATA)


def test_units():
    assert validate_unit("kg") == "KG"
    assert validate_unit("ADT") == "ADT"
    with pytest.raises(ValueError):
        validate_unit("LITRE")
