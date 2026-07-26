import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.data_utils import file_sha256
from scripts.demand_features import (
    FEATURE_BUILDER_VERSION,
    MODEL_FEATURES,
    SPECIAL_CALENDAR_FEATURES,
)
from scripts.project_config import (
    BUNDLE_DIR,
    FIGURES_DIR,
    MAX_FORECAST_LEAD_DAYS,
    MIN_HISTORY_DAYS,
    MODEL_READY_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REFERENCE_DIR,
    REPORT_CSV_DIR,
    REPORT_MD_DIR,
    ensure_project_dirs,
)
from scripts.turkey_calendar import (
    CALENDAR_MAX_DATE,
    CALENDAR_MIN_DATE,
    CALENDAR_VERSION,
)
from scripts.weather_features import (
    WEATHER_FEATURE_VERSION,
    WEATHER_MODEL_FEATURES,
)


KEY_COLUMNS = [
    "forecast_origin",
    "target_date",
    "product_id",
    "product_name",
    "unit",
    "segment",
]


def load_split(name: str) -> pd.DataFrame:
    features = pd.read_csv(
        MODEL_READY_DIR / f"demand_features_{name}.csv",
        dtype={"product_id": "string"},
        parse_dates=["forecast_origin", "target_date"],
    )
    targets = pd.read_csv(
        MODEL_READY_DIR / f"demand_targets_{name}.csv",
        dtype={"product_id": "string"},
        parse_dates=["forecast_origin", "target_date"],
    )
    merged = features.merge(
        targets,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(targets):
        raise AssertionError(f"{name} feature/target eşleşmesi bire bir değil.")
    return merged


def optimize_threshold(y_true: np.ndarray, probability: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probability >= threshold, zero_division=0) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> Dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probability)),
        "positive_prediction_rate": float(prediction.mean()),
    }


def quantity_metrics(y_true: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=float), 0, None)
    y_true = np.asarray(y_true, dtype=float)
    denominator = float(np.abs(y_true).sum())
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "wape": float(np.abs(y_true - prediction).sum() / denominator)
        if denominator > 0
        else np.nan,
        "bias": float(np.mean(prediction - y_true)),
    }


def occurrence_models() -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.07,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=20,
            max_features=0.8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def quantity_models() -> Dict[str, object]:
    return {
        "hist_gradient_boosting_poisson": HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=140,
            learning_rate=0.07,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=12,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def build_occurrence_model(name: str) -> object:
    return occurrence_models()[name]


def build_quantity_model(name: str) -> object:
    return quantity_models()[name]


def fit_validate_unit(
    unit: str, train: pd.DataFrame, validation: pd.DataFrame
) -> Tuple[dict, list]:
    train_unit = train.loc[train["unit"].eq(unit)].copy()
    validation_unit = validation.loc[validation["unit"].eq(unit)].copy()
    x_train = train_unit[MODEL_FEATURES].astype(np.float32)
    y_train = train_unit["demand_occurs"].to_numpy(dtype=int)
    x_validation = validation_unit[MODEL_FEATURES].astype(np.float32)
    y_validation = validation_unit["demand_occurs"].to_numpy(dtype=int)

    rows = []
    occurrence_fitted = {}

    baseline_probability = np.clip(
        validation_unit["rolling_positive_rate_28"].to_numpy(dtype=float), 0, 1
    )
    threshold, _ = optimize_threshold(y_validation, baseline_probability)
    metrics = classification_metrics(y_validation, baseline_probability, threshold)
    rows.append(
        {
            "unit": unit,
            "target": "occurrence",
            "model": "historical_frequency_baseline",
            "threshold": threshold,
            "fit_seconds": 0.0,
            **metrics,
        }
    )

    for name, model in occurrence_models().items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        probability = model.predict_proba(x_validation)[:, 1]
        threshold, _ = optimize_threshold(y_validation, probability)
        metrics = classification_metrics(y_validation, probability, threshold)
        occurrence_fitted[name] = (model, threshold, probability, metrics)
        rows.append(
            {
                "unit": unit,
                "target": "occurrence",
                "model": name,
                "threshold": threshold,
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )

    positive_train = train_unit.loc[train_unit["target_demand"].gt(0)]
    positive_validation = validation_unit.loc[validation_unit["target_demand"].gt(0)]
    x_train_positive = positive_train[MODEL_FEATURES].astype(np.float32)
    y_train_positive = positive_train["target_demand"].to_numpy(dtype=float)
    x_validation_positive = positive_validation[MODEL_FEATURES].astype(np.float32)
    y_validation_positive = positive_validation["target_demand"].to_numpy(dtype=float)

    baseline_prediction = np.maximum(
        positive_validation["rolling_mean_28"].to_numpy(dtype=float), 0
    )
    baseline_metrics = quantity_metrics(y_validation_positive, baseline_prediction)
    rows.append(
        {
            "unit": unit,
            "target": "quantity_positive",
            "model": "rolling_mean_28_baseline",
            "threshold": np.nan,
            "fit_seconds": 0.0,
            **baseline_metrics,
        }
    )

    quantity_fitted = {}
    for name, model in quantity_models().items():
        start = time.perf_counter()
        model.fit(x_train_positive, y_train_positive)
        fit_seconds = time.perf_counter() - start
        prediction = np.clip(model.predict(x_validation_positive), 0, None)
        metrics = quantity_metrics(y_validation_positive, prediction)
        quantity_fitted[name] = (model, prediction, metrics)
        rows.append(
            {
                "unit": unit,
                "target": "quantity_positive",
                "model": name,
                "threshold": np.nan,
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )

    occurrence_name = max(
        occurrence_fitted,
        key=lambda name: (
            occurrence_fitted[name][3]["pr_auc"],
            occurrence_fitted[name][3]["f1"],
            -occurrence_fitted[name][3]["brier"],
        ),
    )
    quantity_name = min(
        quantity_fitted,
        key=lambda name: (
            quantity_fitted[name][2]["mae"],
            quantity_fitted[name][2]["wape"],
        ),
    )
    occurrence_model, selected_threshold, occurrence_probability, occurrence_metric = (
        occurrence_fitted[occurrence_name]
    )
    quantity_model = quantity_fitted[quantity_name][0]
    conditional_prediction = np.clip(
        quantity_model.predict(x_validation), 0, None
    )
    end_to_end_prediction = np.where(
        occurrence_probability >= selected_threshold, conditional_prediction, 0.0
    )
    end_to_end_metrics = quantity_metrics(
        validation_unit["target_demand"].to_numpy(dtype=float), end_to_end_prediction
    )
    rows.append(
        {
            "unit": unit,
            "target": "end_to_end",
            "model": f"{occurrence_name}+{quantity_name}",
            "threshold": selected_threshold,
            "fit_seconds": np.nan,
            **end_to_end_metrics,
        }
    )
    selection = {
        "occurrence_name": occurrence_name,
        "quantity_name": quantity_name,
        "threshold": selected_threshold,
        "validation_occurrence_metrics": occurrence_metric,
        "validation_quantity_metrics": end_to_end_metrics,
    }
    return selection, rows


def rolling_backtest(
    unit: str,
    combined: pd.DataFrame,
    occurrence_name: str,
    quantity_name: str,
) -> list:
    unit_data = combined.loc[combined["unit"].eq(unit)].sort_values("target_date")
    final_end = unit_data["target_date"].max().normalize()
    fold_ends = [
        final_end - pd.Timedelta(days=112),
        final_end - pd.Timedelta(days=56),
        final_end,
    ]
    results = []
    for fold_number, fold_end in enumerate(fold_ends, start=1):
        fold_start = fold_end - pd.Timedelta(days=41)
        train_end = fold_start - pd.Timedelta(days=8)
        train_fold = unit_data.loc[unit_data["target_date"].le(train_end)]
        validation_fold = unit_data.loc[
            unit_data["target_date"].between(fold_start, fold_end)
        ]
        if len(train_fold) < 1000 or len(validation_fold) < 100:
            continue
        x_train = train_fold[MODEL_FEATURES].astype(np.float32)
        x_validation = validation_fold[MODEL_FEATURES].astype(np.float32)
        y_train = train_fold["demand_occurs"].to_numpy(dtype=int)
        y_validation = validation_fold["demand_occurs"].to_numpy(dtype=int)
        occurrence = build_occurrence_model(occurrence_name)
        occurrence.fit(x_train, y_train)
        probability = occurrence.predict_proba(x_validation)[:, 1]
        threshold, _ = optimize_threshold(y_validation, probability)
        occurrence_metric = classification_metrics(y_validation, probability, threshold)

        positive_train = train_fold.loc[train_fold["target_demand"].gt(0)]
        quantity = build_quantity_model(quantity_name)
        quantity.fit(
            positive_train[MODEL_FEATURES].astype(np.float32),
            positive_train["target_demand"].to_numpy(dtype=float),
        )
        conditional = np.clip(quantity.predict(x_validation), 0, None)
        final_prediction = np.where(probability >= threshold, conditional, 0.0)
        amount_metric = quantity_metrics(
            validation_fold["target_demand"].to_numpy(dtype=float), final_prediction
        )
        results.append(
            {
                "unit": unit,
                "fold": fold_number,
                "train_end": train_end,
                "validation_start": fold_start,
                "validation_end": fold_end,
                "train_rows": len(train_fold),
                "validation_rows": len(validation_fold),
                "threshold": threshold,
                **{f"occurrence_{k}": v for k, v in occurrence_metric.items()},
                **{f"quantity_{k}": v for k, v in amount_metric.items()},
            }
        )
    return results


def segment_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> list:
    output = []
    working = frame.copy()
    working["prediction"] = prediction
    for (unit, segment), subset in working.groupby(["unit", "segment"]):
        metrics = quantity_metrics(
            subset["target_demand"].to_numpy(dtype=float),
            subset["prediction"].to_numpy(dtype=float),
        )
        output.append(
            {
                "unit": unit,
                "segment": segment,
                "rows": len(subset),
                "products": subset["product_id"].nunique(),
                **metrics,
            }
        )
    return output


def evaluate_calendar_ablation(
    unit: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selection: dict,
) -> list:
    train_unit = train.loc[train["unit"].eq(unit)]
    validation_unit = validation.loc[validation["unit"].eq(unit)]
    feature_sets = {
        "full_calendar": MODEL_FEATURES,
        "special_calendar_removed": [
            feature
            for feature in MODEL_FEATURES
            if feature not in SPECIAL_CALENDAR_FEATURES
        ],
    }
    rows = []
    for feature_set_name, features in feature_sets.items():
        occurrence = build_occurrence_model(selection["occurrence_name"])
        occurrence.fit(
            train_unit[features].astype(np.float32),
            train_unit["demand_occurs"].to_numpy(dtype=int),
        )
        probability = occurrence.predict_proba(
            validation_unit[features].astype(np.float32)
        )[:, 1]
        threshold, _ = optimize_threshold(
            validation_unit["demand_occurs"].to_numpy(dtype=int), probability
        )
        occurrence_metric = classification_metrics(
            validation_unit["demand_occurs"].to_numpy(dtype=int),
            probability,
            threshold,
        )

        positive_train = train_unit.loc[train_unit["target_demand"].gt(0)]
        quantity = build_quantity_model(selection["quantity_name"])
        quantity.fit(
            positive_train[features].astype(np.float32),
            positive_train["target_demand"].to_numpy(dtype=float),
        )
        conditional = np.clip(
            quantity.predict(validation_unit[features].astype(np.float32)),
            0,
            None,
        )
        final_prediction = np.where(
            probability >= threshold, conditional, 0.0
        )
        amount_metric = quantity_metrics(
            validation_unit["target_demand"].to_numpy(dtype=float),
            final_prediction,
        )
        rows.append(
            {
                "unit": unit,
                "feature_set": feature_set_name,
                "feature_count": len(features),
                "occurrence_model": selection["occurrence_name"],
                "quantity_model": selection["quantity_name"],
                "threshold": threshold,
                **{
                    f"occurrence_{key}": value
                    for key, value in occurrence_metric.items()
                },
                **{
                    f"quantity_{key}": value
                    for key, value in amount_metric.items()
                },
            }
        )
    return rows


def evaluate_weather_ablation(
    unit: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selection: dict,
) -> list:
    train_unit = train.loc[train["unit"].eq(unit)]
    validation_unit = validation.loc[validation["unit"].eq(unit)]
    feature_sets = {
        "candidate_with_weather": MODEL_FEATURES + WEATHER_MODEL_FEATURES,
        "deployed_weather_removed": MODEL_FEATURES,
    }
    rows = []
    for feature_set_name, features in feature_sets.items():
        occurrence = build_occurrence_model(selection["occurrence_name"])
        occurrence.fit(
            train_unit[features].astype(np.float32),
            train_unit["demand_occurs"].to_numpy(dtype=int),
        )
        probability = occurrence.predict_proba(
            validation_unit[features].astype(np.float32)
        )[:, 1]
        threshold, _ = optimize_threshold(
            validation_unit["demand_occurs"].to_numpy(dtype=int), probability
        )
        occurrence_metric = classification_metrics(
            validation_unit["demand_occurs"].to_numpy(dtype=int),
            probability,
            threshold,
        )
        positive_train = train_unit.loc[train_unit["target_demand"].gt(0)]
        quantity = build_quantity_model(selection["quantity_name"])
        quantity.fit(
            positive_train[features].astype(np.float32),
            positive_train["target_demand"].to_numpy(dtype=float),
        )
        conditional = np.clip(
            quantity.predict(validation_unit[features].astype(np.float32)),
            0,
            None,
        )
        final_prediction = np.where(
            probability >= threshold, conditional, 0.0
        )
        amount_metric = quantity_metrics(
            validation_unit["target_demand"].to_numpy(dtype=float),
            final_prediction,
        )
        rows.append(
            {
                "unit": unit,
                "feature_set": feature_set_name,
                "feature_count": len(features),
                "occurrence_model": selection["occurrence_name"],
                "quantity_model": selection["quantity_name"],
                "threshold": threshold,
                **{
                    f"occurrence_{key}": value
                    for key, value in occurrence_metric.items()
                },
                **{
                    f"quantity_{key}": value
                    for key, value in amount_metric.items()
                },
            }
        )
    return rows


def calendar_segment_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> list:
    working = frame.copy()
    working["probability"] = probability
    working["prediction"] = prediction
    special_union = (
        working["target_is_public_holiday"].eq(1)
        | working["target_is_pre_holiday_3d"].eq(1)
        | working["target_is_post_holiday_3d"].eq(1)
        | working["target_is_ramadan"].eq(1)
        | working["target_is_religious_special_day"].eq(1)
        | working["target_is_school_break"].eq(1)
        | working["target_is_weekend"].eq(1)
    )
    contexts = {
        "public_holiday": working["target_is_public_holiday"].eq(1),
        "pre_holiday_3d": working["target_is_pre_holiday_3d"].eq(1),
        "post_holiday_3d": working["target_is_post_holiday_3d"].eq(1),
        "ramadan": working["target_is_ramadan"].eq(1),
        "religious_special_day": working[
            "target_is_religious_special_day"
        ].eq(1),
        "school_break": working["target_is_school_break"].eq(1),
        "weekend": working["target_is_weekend"].eq(1),
        "ordinary_day": ~special_union,
    }
    output = []
    for context, mask in contexts.items():
        subset = working.loc[mask]
        if subset.empty:
            continue
        y_true = subset["demand_occurs"].to_numpy(dtype=int)
        if np.unique(y_true).size > 1:
            occurrence_metric = classification_metrics(
                y_true,
                subset["probability"].to_numpy(dtype=float),
                threshold,
            )
        else:
            occurrence_metric = {
                "pr_auc": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "brier": np.nan,
                "positive_prediction_rate": float(
                    subset["probability"].ge(threshold).mean()
                ),
            }
        amount_metric = quantity_metrics(
            subset["target_demand"].to_numpy(dtype=float),
            subset["prediction"].to_numpy(dtype=float),
        )
        output.append(
            {
                "unit": subset["unit"].iloc[0],
                "calendar_context": context,
                "rows": len(subset),
                "products": subset["product_id"].nunique(),
                "actual_positive_rate": subset["demand_occurs"].mean(),
                **{
                    f"occurrence_{key}": value
                    for key, value in occurrence_metric.items()
                },
                **{
                    f"quantity_{key}": value
                    for key, value in amount_metric.items()
                },
            }
        )
    return output


def main() -> None:
    ensure_project_dirs()
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    comparison_rows = []
    selections = {}
    for unit in ("KG", "ADT"):
        selection, rows = fit_validate_unit(unit, train, validation)
        selections[unit] = selection
        comparison_rows.extend(rows)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REPORT_CSV_DIR / "model_comparison_results.csv", index=False)

    combined_train_validation = pd.concat([train, validation], ignore_index=True)
    ablation_rows = []
    for unit in ("KG", "ADT"):
        ablation_rows.extend(
            evaluate_calendar_ablation(
                unit,
                train,
                validation,
                selections[unit],
            )
        )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(
        REPORT_CSV_DIR / "calendar_feature_ablation.csv", index=False
    )
    weather_ablation_rows = []
    for unit in ("KG", "ADT"):
        weather_ablation_rows.extend(
            evaluate_weather_ablation(
                unit,
                train,
                validation,
                selections[unit],
            )
        )
    weather_ablation = pd.DataFrame(weather_ablation_rows)
    weather_ablation.to_csv(
        REPORT_CSV_DIR / "weather_feature_ablation.csv", index=False
    )

    backtest_rows = []
    for unit in ("KG", "ADT"):
        backtest_rows.extend(
            rolling_backtest(
                unit,
                combined_train_validation,
                selections[unit]["occurrence_name"],
                selections[unit]["quantity_name"],
            )
        )
    backtest = pd.DataFrame(backtest_rows)
    backtest.to_csv(REPORT_CSV_DIR / "backtest_results.csv", index=False)

    test_prediction_frames = []
    segment_rows = []
    test_metrics = {}
    calendar_metric_rows = []
    unit_model_map = {}
    fitted_assets = {}

    for unit in ("KG", "ADT"):
        train_unit = combined_train_validation.loc[
            combined_train_validation["unit"].eq(unit)
        ]
        test_unit = test.loc[test["unit"].eq(unit)].copy()
        x_train = train_unit[MODEL_FEATURES].astype(np.float32)
        x_test = test_unit[MODEL_FEATURES].astype(np.float32)

        occurrence = build_occurrence_model(selections[unit]["occurrence_name"])
        occurrence.fit(x_train, train_unit["demand_occurs"].to_numpy(dtype=int))
        positive_train = train_unit.loc[train_unit["target_demand"].gt(0)]
        quantity = build_quantity_model(selections[unit]["quantity_name"])
        quantity.fit(
            positive_train[MODEL_FEATURES].astype(np.float32),
            positive_train["target_demand"].to_numpy(dtype=float),
        )

        probability = occurrence.predict_proba(x_test)[:, 1]
        conditional = np.clip(quantity.predict(x_test), 0, None)
        threshold = float(selections[unit]["threshold"])
        final_prediction = np.where(probability >= threshold, conditional, 0.0)

        occurrence_metric = classification_metrics(
            test_unit["demand_occurs"].to_numpy(dtype=int), probability, threshold
        )
        amount_metric = quantity_metrics(
            test_unit["target_demand"].to_numpy(dtype=float), final_prediction
        )
        positive_mask = test_unit["target_demand"].gt(0).to_numpy()
        positive_metric = quantity_metrics(
            test_unit.loc[positive_mask, "target_demand"].to_numpy(dtype=float),
            conditional[positive_mask],
        )
        test_metrics[unit] = {
            "occurrence": occurrence_metric,
            "quantity_end_to_end": amount_metric,
            "quantity_positive_conditional": positive_metric,
        }

        test_unit["demand_probability"] = probability
        test_unit["conditional_quantity_prediction"] = conditional
        test_unit["demand_prediction"] = final_prediction
        test_unit["selected_threshold"] = threshold
        test_prediction_frames.append(test_unit)
        segment_rows.extend(segment_metrics(test_unit, final_prediction))
        calendar_metric_rows.extend(
            calendar_segment_metrics(
                test_unit,
                probability,
                final_prediction,
                threshold,
            )
        )

        occurrence_path = BUNDLE_DIR / f"occurrence_model_{unit.lower()}.pkl"
        quantity_path = BUNDLE_DIR / f"quantity_model_{unit.lower()}.pkl"
        joblib.dump(occurrence, occurrence_path)
        joblib.dump(quantity, quantity_path)
        fitted_assets[unit] = (occurrence, quantity)
        unit_model_map[unit] = {
            "occurrence_model_path": occurrence_path.name,
            "quantity_model_path": quantity_path.name,
            "pipeline_path": None,
            "occurrence_model_name": selections[unit]["occurrence_name"],
            "quantity_model_name": selections[unit]["quantity_name"],
        }

    test_predictions = pd.concat(test_prediction_frames, ignore_index=True)
    test_predictions.to_csv(REPORT_CSV_DIR / "test_predictions.csv", index=False)
    segment_frame = pd.DataFrame(segment_rows)
    segment_frame.to_csv(REPORT_CSV_DIR / "segment_metrics.csv", index=False)
    calendar_metric_frame = pd.DataFrame(calendar_metric_rows)
    calendar_metric_frame.to_csv(
        REPORT_CSV_DIR / "calendar_segment_metrics.csv", index=False
    )

    catalog = pd.read_csv(PROCESSED_DIR / "product_catalog.csv", dtype={"product_id": "string"})
    catalog.to_csv(BUNDLE_DIR / "product_catalog.csv", index=False)
    shutil.copy2(
        REFERENCE_DIR / "bursa_weather_observed_era5.csv",
        BUNDLE_DIR / "bursa_weather_observed_era5.csv",
    )
    shutil.copy2(
        REFERENCE_DIR / "bursa_weather_previous_runs.csv",
        BUNDLE_DIR / "bursa_weather_previous_runs.csv",
    )

    model_version = "2026.07.26-weather-v3"
    metadata = {
        "model_version": model_version,
        "created_at": datetime.now().astimezone().isoformat(),
        "forecast_type": "direct_multi_horizon_daily",
        "forecast_origin": pd.to_datetime(
            pd.read_csv(MODEL_READY_DIR / "inference_snapshot.csv", nrows=1)["date"].iloc[0]
        ).date().isoformat(),
        "max_forecast_lead_days": MAX_FORECAST_LEAD_DAYS,
        "decision_grain": "product-target_date",
        "targets": ["demand_occurs", "target_demand"],
        "units": ["KG", "ADT"],
        "model_layout": "per_unit",
        "unit_model_map": unit_model_map,
        "required_history_days": MIN_HISTORY_DAYS,
        "feature_columns": MODEL_FEATURES,
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "calendar_version": CALENDAR_VERSION,
        "calendar_features": SPECIAL_CALENDAR_FEATURES,
        "calendar_coverage": {
            "start": CALENDAR_MIN_DATE.date().isoformat(),
            "end": CALENDAR_MAX_DATE.date().isoformat(),
        },
        "calendar_source_manifest": "data/reference/calendar_sources.json",
        "calendar_code_sha256": file_sha256(
            PROJECT_ROOT / "scripts" / "turkey_calendar.py"
        ),
        "calendar_source_manifest_sha256": file_sha256(
            REFERENCE_DIR / "calendar_sources.json"
        ),
        "weather_feature_version": WEATHER_FEATURE_VERSION,
        "weather_candidate_features": WEATHER_MODEL_FEATURES,
        "weather_features": [],
        "weather_location": "Bursa/Osmangazi city centre (40.19559, 29.06013)",
        "weather_policy": (
            "V1 uses expanding seasonal climatology from ERA5 dates strictly "
            "before forecast_origin. Observed target-date weather is EDA-only. "
            "Candidate weather fields were excluded from deployed estimators after "
            "validation ablation showed no robust cross-unit gain."
        ),
        "weather_deployment_decision": (
            "excluded_after_validation_ablation_no_robust_gain"
        ),
        "weather_source_manifest": "data/reference/weather_sources.json",
        "weather_code_sha256": file_sha256(
            PROJECT_ROOT / "scripts" / "weather_features.py"
        ),
        "weather_source_manifest_sha256": file_sha256(
            REFERENCE_DIR / "weather_sources.json"
        ),
        "weather_observed_bundle_file": "bursa_weather_observed_era5.csv",
        "weather_observed_bundle_sha256": file_sha256(
            BUNDLE_DIR / "bursa_weather_observed_era5.csv"
        ),
        "weather_forecast_bundle_file": "bursa_weather_previous_runs.csv",
        "weather_forecast_bundle_sha256": file_sha256(
            BUNDLE_DIR / "bursa_weather_previous_runs.csv"
        ),
        "training_end_date": combined_train_validation["target_date"].max().date().isoformat(),
        "validation_metrics": {
            unit: {
                "occurrence": selections[unit]["validation_occurrence_metrics"],
                "quantity_end_to_end": selections[unit]["validation_quantity_metrics"],
            }
            for unit in ("KG", "ADT")
        },
        "test_metrics": test_metrics,
        "occurrence_threshold": {
            unit: float(selections[unit]["threshold"]) for unit in ("KG", "ADT")
        },
        "probability_calibrated": {"KG": False, "ADT": False},
        "quantity_target_transform": "none",
        "quantity_postprocess": "clip_to_zero_then_threshold_gate",
        "combination_rule": (
            "If demand_probability >= unit threshold, use non-negative conditional "
            "quantity prediction; otherwise return zero."
        ),
        "zero_demand_policy": (
            "Observed store day and product panel window without a sale is zero; "
            "globally missing store day remains unobserved."
        ),
        "cold_start_policy": (
            "Products without 28-day snapshot return insufficient_history; "
            "possibly inactive products are forecast with a visible warning."
        ),
        "known_limitations": [
            "No price, promotion, stockout, or store-specific closure features.",
            (
                "Weather is Bursa city-centre time-safe climatology, not the "
                "realised weather of the future target day."
            ),
            (
                "Exact store coordinates and a complete archived/live forecast "
                "contract are not yet available."
            ),
            "School calendar is national; local closures and store opening hours are absent.",
            "Forecasts near 180 days are less certain than near-term forecasts.",
            "Probability is not calibrated and must not be labelled confidence.",
            "Demand forecast is not a replenishment order without inventory inputs.",
        ],
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (BUNDLE_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checksums = {}
    for path in sorted(BUNDLE_DIR.iterdir()):
        if path.is_file() and path.name != "checksums.json":
            checksums[path.name] = file_sha256(path)
    (BUNDLE_DIR / "checksums.json").write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, unit in zip(axes, ("KG", "ADT")):
        subset = comparison.loc[
            comparison["unit"].eq(unit) & comparison["target"].eq("occurrence")
        ].sort_values("pr_auc")
        axis.barh(subset["model"], subset["pr_auc"], color="#6366f1")
        axis.set_title(f"{unit} — Validation PR-AUC")
        axis.set_xlabel("PR-AUC")
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "model_occurrence_validation_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, unit in zip(axes, ("KG", "ADT")):
        subset = ablation.loc[ablation["unit"].eq(unit)].copy()
        positions = np.arange(len(subset))
        axis.bar(
            positions - 0.18,
            subset["occurrence_pr_auc"],
            width=0.36,
            label="Occurrence PR-AUC",
            color="#6366f1",
        )
        secondary = axis.twinx()
        secondary.bar(
            positions + 0.18,
            subset["quantity_mae"],
            width=0.36,
            label="Miktar MAE",
            color="#f59e0b",
            alpha=0.75,
        )
        axis.set_xticks(positions, subset["feature_set"], rotation=12)
        axis.set_title(f"{unit} — Özel Takvim Ablation")
        axis.set_ylabel("PR-AUC")
        secondary.set_ylabel(f"MAE ({unit})")
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "model_calendar_feature_ablation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, unit in zip(axes, ("KG", "ADT")):
        subset = test_predictions.loc[test_predictions["unit"].eq(unit)]
        sample = subset.sample(min(4000, len(subset)), random_state=RANDOM_STATE)
        axis.scatter(
            sample["target_demand"],
            sample["demand_prediction"],
            alpha=0.25,
            s=12,
            color="#22d3ee" if unit == "KG" else "#fbbf24",
        )
        maximum = max(sample["target_demand"].max(), sample["demand_prediction"].max())
        axis.plot([0, maximum], [0, maximum], "--", color="#ef4444", linewidth=1)
        axis.set_title(f"{unit} — Test Gerçekleşen ve Tahmin")
        axis.set_xlabel(f"Gerçek {unit}")
        axis.set_ylabel(f"Tahmin {unit}")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "model_test_prediction_vs_actual.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    report_lines = [
        "# Model Değerlendirme Raporu",
        "",
        "## Problem",
        "",
        "Seçilen ürün ve gelecek hedef gün için günlük talep oluşumu ile ürünün kendi "
        "birimindeki miktar doğrudan tahmin edilir. Model 1–180 günlük lead aralığı için "
        "eğitilmiştir.",
        "",
        f"Model `{CALENDAR_VERSION}` takvimiyle resmî/dinî tatil, Ramazan, "
        "tatil önce/sonra pencereleri ve MEB okul dönemlerini kullanır.",
        "",
        "## Seçilen modeller",
        "",
    ]
    for unit in ("KG", "ADT"):
        report_lines.extend(
            [
                f"### {unit}",
                "",
                f"- Occurrence: `{selections[unit]['occurrence_name']}`",
                f"- Quantity: `{selections[unit]['quantity_name']}`",
                f"- Validation threshold: `{selections[unit]['threshold']:.3f}`",
                f"- Test occurrence PR-AUC: `{test_metrics[unit]['occurrence']['pr_auc']:.4f}`",
                f"- Test occurrence F1: `{test_metrics[unit]['occurrence']['f1']:.4f}`",
                f"- Test quantity MAE: `{test_metrics[unit]['quantity_end_to_end']['mae']:.4f}`",
                f"- Test quantity WAPE: `{test_metrics[unit]['quantity_end_to_end']['wape']:.4f}`",
                f"- Test bias: `{test_metrics[unit]['quantity_end_to_end']['bias']:.4f}`",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Değerlendirme ilkeleri",
            "",
            "- Model seçimi validation sonuçlarıyla yapılmış, test final ölçüm için kullanılmıştır.",
            "- KG ve ADT ayrı model ve metriklerle değerlendirilmiştir.",
            "- Olasılıklar kalibre değildir; kullanıcıya güven skoru olarak sunulmaz.",
            "- Tahmin stok/sipariş önerisi değildir.",
            "- Özel takvim alanlarının katkısı validation ablation ile ölçülmüştür.",
            "- Hava klimatolojisinin katkısı validation ablation ile ölçülmüştür.",
            "- Takvim ablation sonucu: `reports/csv/calendar_feature_ablation.csv`.",
            "- Hava ablation sonucu: `reports/csv/weather_feature_ablation.csv`.",
            "- Özel tarih test segmentleri: `reports/csv/calendar_segment_metrics.csv`.",
            "",
            "## Kalan riskler",
            "",
            *[f"- {item}" for item in metadata["known_limitations"]],
        ]
    )
    report_text = "\n".join(report_lines)
    (REPORT_MD_DIR / "MODEL_EVALUATION_REPORT.md").write_text(
        report_text, encoding="utf-8"
    )

    handoff = f"""# Model Expert Handoff — Deployment

## Tahmin sözleşmesi

- Forecast type: `direct_multi_horizon_daily`
- Forecast origin: {metadata['forecast_origin']}
- Kullanıcı girdisi: ürün ve hedef tarih
- Doğrulanmış lead aralığı: 1–{MAX_FORECAST_LEAD_DAYS} gün
- Hedef: seçili günde talep olasılığı ve günlük KG/ADT miktarı

## Bundle

- Sürüm: `{model_version}`
- Layout: `per_unit`
- Feature builder: `{FEATURE_BUILDER_VERSION}`
- Takvim: `{CALENDAR_VERSION}`
- Hava: `{WEATHER_FEATURE_VERSION}` — Bursa/Osmangazi time-safe klimatoloji
- Metadata: `models/demand_forecasting_bundle/model_metadata.json`
- Checksum: `models/demand_forecasting_bundle/checksums.json`

## Birim yönlendirmesi

{json.dumps(unit_model_map, ensure_ascii=False, indent=2)}

## Güvenli sonuç sunumu

Olasılık kalibre değildir. Sonuç “modelin talep olasılığı” olarak gösterilmelidir.
Miktar ürün birimindedir. `ADT` sonuç ekranda tam sayıya yuvarlanabilir; ham tahmin
logda korunur. Modelin miktar tahmini sipariş önerisi değildir.

## İzleme

Tahmin kaydı ürün, forecast origin, target date, lead days, unit, model sürümü,
olasılık, miktar ve uyarı kodlarını içermelidir. Hedef gün geçtikten sonra gerçek
satışla ürün-tarih-birim anahtarında eşleştirilmelidir.
"""
    (REPORT_MD_DIR / "MODEL_EXPERT_HANDOFF.md").write_text(handoff, encoding="utf-8")

    impact = pd.read_csv(REPORT_CSV_DIR / "calendar_demand_impact.csv")
    calendar_report_lines = [
        "# Özel Takvim Feature Raporu",
        "",
        "## Kapsam",
        "",
        f"- Takvim sürümü: `{CALENDAR_VERSION}`",
        f"- Feature builder: `{FEATURE_BUILDER_VERSION}`",
        f"- Özel takvim feature sayısı: **{len(SPECIAL_CALENDAR_FEATURES)}**",
        "- Kaynak manifesti: `data/reference/calendar_sources.json`",
        "- İçerik: resmî/dinî tatil, yarım gün, Ramazan, kandil, tatilden "
        "1/3 gün önce-sonra, MEB okul dönemi ve tatilleri.",
        "",
        "## Validation ablation",
        "",
        "| Birim | Full PR-AUC | Takvimsiz PR-AUC | Δ PR-AUC | Full MAE | Takvimsiz MAE | Δ MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for unit in ("KG", "ADT"):
        full = ablation.loc[
            ablation["unit"].eq(unit)
            & ablation["feature_set"].eq("full_calendar")
        ].iloc[0]
        removed = ablation.loc[
            ablation["unit"].eq(unit)
            & ablation["feature_set"].eq("special_calendar_removed")
        ].iloc[0]
        calendar_report_lines.append(
            f"| {unit} | {full['occurrence_pr_auc']:.4f} | "
            f"{removed['occurrence_pr_auc']:.4f} | "
            f"{full['occurrence_pr_auc'] - removed['occurrence_pr_auc']:+.4f} | "
            f"{full['quantity_mae']:.4f} | {removed['quantity_mae']:.4f} | "
            f"{full['quantity_mae'] - removed['quantity_mae']:+.4f} |"
        )
    calendar_report_lines.extend(
        [
            "",
            "Pozitif Δ PR-AUC iyileşme; negatif Δ MAE iyileşme anlamına gelir. "
            "Bu karşılaştırma aynı model ailesi ve aynı validation dönemiyle yapılmıştır.",
            "",
            "## Gözlenen takvim etkisi",
            "",
            "| Birim | Bağlam | Pozitif talep oranı / normal gün | Ortalama miktar / normal gün |",
            "|---|---|---:|---:|",
        ]
    )
    for unit in ("KG", "ADT"):
        for context in (
            "public_holiday",
            "pre_holiday_3d",
            "post_holiday_3d",
            "ramadan_nonholiday",
            "school_midterm_break",
            "school_semester_break",
            "school_summer_break",
            "weekend_nonholiday",
        ):
            row = impact.loc[
                impact["unit"].eq(unit)
                & impact["calendar_context"].eq(context)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            calendar_report_lines.append(
                f"| {unit} | {context} | "
                f"{row['positive_rate_ratio_vs_ordinary']:.3f} | "
                f"{row['mean_demand_ratio_vs_ordinary']:.3f} |"
            )
    calendar_report_lines.extend(
        [
            "",
            "## Yorum sınırı",
            "",
            "Bu oranlar betimseldir; kampanya, fiyat, stokta yokluk ve ürün portföyü "
            "değişimini tek başına kontrol etmez. Model katkısı için esas kanıt "
            "validation ablation ve zaman sıralı test segmentleridir.",
        ]
    )
    (REPORT_MD_DIR / "CALENDAR_FEATURE_REPORT.md").write_text(
        "\n".join(calendar_report_lines),
        encoding="utf-8",
    )

    weather_impact = pd.read_csv(REPORT_CSV_DIR / "weather_demand_impact.csv")
    weather_report_lines = [
        "# Bursa Hava Durumu Feature Raporu",
        "",
        "## Güvenli veri sözleşmesi",
        "",
        f"- Hava feature sürümü: `{WEATHER_FEATURE_VERSION}`",
        "- Konum: Bursa/Osmangazi şehir merkezi (40.19559, 29.06013).",
        "- Gerçekleşmiş hedef-gün havası model girdisi değildir; yalnız EDA'dadır.",
        "- Her satırda ERA5 hava tarihleri `forecast_origin` gününden kesinlikle "
        "önce filtrelenir.",
        "- Hedef mevsim için ±15 günlük dairesel pencere klimatolojisi kullanılır.",
        "- Kaynak manifesti: `data/reference/weather_sources.json`.",
        "",
        "## Validation ablation",
        "",
        "| Birim | Aday havalı PR-AUC | Deploy PR-AUC | Δ PR-AUC | Aday havalı MAE | Deploy MAE | Δ MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for unit in ("KG", "ADT"):
        full = weather_ablation.loc[
            weather_ablation["unit"].eq(unit)
            & weather_ablation["feature_set"].eq("candidate_with_weather")
        ].iloc[0]
        removed = weather_ablation.loc[
            weather_ablation["unit"].eq(unit)
            & weather_ablation["feature_set"].eq("deployed_weather_removed")
        ].iloc[0]
        weather_report_lines.append(
            f"| {unit} | {full['occurrence_pr_auc']:.4f} | "
            f"{removed['occurrence_pr_auc']:.4f} | "
            f"{full['occurrence_pr_auc'] - removed['occurrence_pr_auc']:+.4f} | "
            f"{full['quantity_mae']:.4f} | {removed['quantity_mae']:.4f} | "
            f"{full['quantity_mae'] - removed['quantity_mae']:+.4f} |"
        )
    weather_report_lines.extend(
        [
            "",
            "Pozitif Δ PR-AUC ve negatif Δ MAE iyileşme demektir. Ablation aynı "
            "validation dönemi ve aynı seçilmiş model aileleriyle yapılmıştır.",
            "",
            "## Deployment kararı",
            "",
            "**Hava alanları final estimator feature listesinden çıkarıldı.** KG'de "
            "çok küçük occurrence farkı diğer metriklerde ve ADT'de doğrulanmadığı "
            "için sonuç sağlam bir genel kazanım sayılmadı. Aday alanlar model-ready "
            "tabloda, EDA'da ve kullanıcıya açıklayıcı hava bağlamında korunur.",
            "",
            "## Betimsel gerçekleşen hava ilişkileri",
            "",
            "| Birim | Koşul | Gün | Ortalama talep oranı (koşul / diğer günler) |",
            "|---|---|---:|---:|",
        ]
    )
    for row in weather_impact.itertuples(index=False):
        weather_report_lines.append(
            f"| {row.unit} | {row.weather_condition} | "
            f"{int(row.condition_days)} | {row.mean_total_demand_ratio:.3f} |"
        )
    weather_report_lines.extend(
        [
            "",
            "## Yorum sınırı",
            "",
            "Gerçekleşen hava oranları nedensellik kanıtı değildir; mevsim, ürün "
            "portföyü, fiyat, kampanya ve stok durumu ile karışabilir. Modeldeki esas "
            "karar kanıtı zaman sıralı validation ablation'dır. Bu kanıt zayıf "
            "olduğu için hava feature grubu final estimator'lardan çıkarılmıştır.",
        ]
    )
    (REPORT_MD_DIR / "WEATHER_FEATURE_REPORT.md").write_text(
        "\n".join(weather_report_lines),
        encoding="utf-8",
    )

    print(comparison.to_string(index=False))
    print("\nFinal test metrikleri:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
