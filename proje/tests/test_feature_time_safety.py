from pathlib import Path

import pandas as pd

from models.demand_forecasting_bundle.feature_builder import (
    MODEL_FEATURES,
    prepare_single_target_date,
)


ROOT = Path(__file__).resolve().parents[1]


def test_target_date_features_do_not_change_snapshot_history():
    snapshot = pd.read_csv(
        ROOT / "data" / "model_ready" / "inference_snapshot.csv",
        dtype={"product_id": "string"},
        parse_dates=["date"],
    ).head(1)
    target = snapshot["date"].iloc[0] + pd.Timedelta(days=164)
    prepared = prepare_single_target_date(snapshot, target)
    assert prepared["forecast_origin"].iloc[0] < prepared["target_date"].iloc[0]
    assert prepared["lead_days"].iloc[0] == 164
    assert prepared[MODEL_FEATURES].notna().all(axis=None)
