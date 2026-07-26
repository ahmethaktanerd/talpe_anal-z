from pathlib import Path

import json

from app.services.bundle_loader import load_metadata, load_unit_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "models" / "demand_forecasting_bundle"


def test_metadata_and_checksums_load():
    metadata = load_metadata(BUNDLE_DIR)
    assert metadata["forecast_type"] == "direct_multi_horizon_daily"
    assert set(metadata["units"]) == {"KG", "ADT"}
    checksums = json.loads(
        (BUNDLE_DIR / "checksums.json").read_text(encoding="utf-8")
    )
    assert "feature_builder.py" in checksums


def test_both_unit_models_load():
    metadata = load_metadata(BUNDLE_DIR)
    for unit in ("KG", "ADT"):
        models = load_unit_models(BUNDLE_DIR, metadata, unit)
        assert hasattr(models["occurrence"], "predict_proba")
        assert hasattr(models["quantity"], "predict")
