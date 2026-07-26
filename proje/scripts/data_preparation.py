import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.data_utils import clean_sales_records, file_sha256, read_raw_sales, valid_sales
from scripts.demand_features import (
    ALL_CANDIDATE_FEATURES,
    FEATURE_BUILDER_VERSION,
    MODEL_FEATURES,
    SNAPSHOT_FEATURES,
    SPECIAL_CALENDAR_FEATURES,
    TARGET_DATE_FEATURES,
    build_direct_horizon_examples,
    build_inference_features,
    build_snapshot_table,
)
from scripts.project_config import (
    ACTIVE_TAIL_DAYS,
    LEAD_DAY_GRID,
    MAX_FORECAST_LEAD_DAYS,
    MIN_HISTORY_DAYS,
    MODEL_READY_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    PROCESSED_DIR,
    RAW_DATA,
    REFERENCE_DIR,
    REPORT_CSV_DIR,
    REPORT_MD_DIR,
    ensure_project_dirs,
)
from scripts.turkey_calendar import (
    CALENDAR_VERSION,
    build_turkey_calendar,
    calendar_reference_metadata,
)
from scripts.weather_features import (
    WEATHER_FEATURE_VERSION,
    WEATHER_MODEL_FEATURES,
    load_weather_reference,
)


KEY_COLUMNS = [
    "forecast_origin",
    "target_date",
    "product_id",
    "product_name",
    "unit",
    "segment",
]
TARGET_COLUMNS = ["demand_occurs", "target_demand"]


def build_product_catalog(valid: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    catalog = (
        valid.groupby(["product_id", "product_name", "unit"], as_index=False)
        .agg(
            first_sale=("date", "min"),
            last_sale=("date", "max"),
            positive_sales_days=("date", "nunique"),
            total_quantity=("quantity", "sum"),
            median_positive_quantity=("quantity", "median"),
        )
    )
    catalog["history_days"] = (catalog["last_sale"] - catalog["first_sale"]).dt.days + 1
    catalog["active_ratio"] = catalog["positive_sales_days"] / catalog["history_days"]
    catalog["days_since_last_sale"] = (data_end - catalog["last_sale"]).dt.days
    catalog["current_status"] = np.where(
        catalog["days_since_last_sale"].le(ACTIVE_TAIL_DAYS), "active", "possibly_inactive"
    )
    catalog["segment"] = np.select(
        [
            catalog["history_days"].lt(30),
            catalog["current_status"].eq("possibly_inactive"),
            catalog["active_ratio"].ge(0.50),
            catalog["active_ratio"].ge(0.10),
        ],
        [
            "new_short_history",
            "possibly_inactive",
            "continuous",
            "intermittent",
        ],
        default="sparse",
    )
    return catalog


def build_daily_panel(
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
    data_end: pd.Timestamp,
    extend_all_to_end: bool = False,
) -> pd.DataFrame:
    daily_sales = (
        valid.groupby(
            ["date", "product_id", "product_name", "unit"], as_index=False
        )
        .agg(daily_demand=("quantity", "sum"), source_row_count=("source_row", "count"))
    )
    observed_dates = pd.Index(valid["date"].unique())
    frames = []
    for row in catalog.itertuples(index=False):
        active_end = (
            data_end
            if extend_all_to_end
            else min(row.last_sale + pd.Timedelta(days=ACTIVE_TAIL_DAYS), data_end)
        )
        dates = pd.date_range(row.first_sale, active_end, freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "product_id": row.product_id,
                    "product_name": row.product_name,
                    "unit": row.unit,
                    "segment": row.segment,
                    "current_status": row.current_status,
                    "first_sale": row.first_sale,
                    "last_sale": row.last_sale,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    panel["store_observed"] = panel["date"].isin(observed_dates).astype(int)
    panel = panel.merge(
        daily_sales,
        on=["date", "product_id", "product_name", "unit"],
        how="left",
        validate="one_to_one",
    )
    panel["source_row_count"] = panel["source_row_count"].fillna(0).astype(int)
    panel["daily_demand"] = np.where(
        panel["store_observed"].eq(0),
        np.nan,
        panel["daily_demand"].fillna(0.0),
    )
    panel["observation_status"] = np.select(
        [
            panel["store_observed"].eq(0),
            panel["source_row_count"].gt(0),
        ],
        ["missing_or_unobserved", "observed_positive"],
        default="observed_zero",
    )
    return panel.sort_values(["product_id", "date"]).reset_index(drop=True)


def choose_splits(examples: pd.DataFrame) -> dict:
    target_end = examples["target_date"].max().normalize()
    test_start = target_end - pd.Timedelta(days=83)
    validation_end = test_start - pd.Timedelta(days=8)
    validation_start = validation_end - pd.Timedelta(days=83)
    train_end = validation_start - pd.Timedelta(days=8)
    return {
        "train": {"start": examples["target_date"].min().normalize(), "end": train_end},
        "validation": {"start": validation_start, "end": validation_end},
        "test": {"start": test_start, "end": target_end},
        "gap_1": {
            "start": train_end + pd.Timedelta(days=1),
            "end": validation_start - pd.Timedelta(days=1),
        },
        "gap_2": {
            "start": validation_end + pd.Timedelta(days=1),
            "end": test_start - pd.Timedelta(days=1),
        },
    }


def split_examples(examples: pd.DataFrame, split_dates: dict) -> dict:
    result = {}
    for split_name in ("train", "validation", "test"):
        start = split_dates[split_name]["start"]
        end = split_dates[split_name]["end"]
        result[split_name] = examples.loc[
            examples["target_date"].between(start, end)
        ].copy()
    return result


def save_model_ready(split_frames: dict) -> pd.DataFrame:
    summaries = []
    for split_name, frame in split_frames.items():
        features = frame[KEY_COLUMNS + ALL_CANDIDATE_FEATURES].copy()
        targets = frame[KEY_COLUMNS + TARGET_COLUMNS].copy()
        features.to_csv(
            MODEL_READY_DIR / f"demand_features_{split_name}.csv", index=False
        )
        targets.to_csv(
            MODEL_READY_DIR / f"demand_targets_{split_name}.csv", index=False
        )
        for unit, subset in frame.groupby("unit"):
            summaries.append(
                {
                    "split": split_name,
                    "unit": unit,
                    "rows": len(subset),
                    "products": subset["product_id"].nunique(),
                    "target_start": subset["target_date"].min(),
                    "target_end": subset["target_date"].max(),
                    "positive_rate": subset["demand_occurs"].mean(),
                    "target_quantity_sum": subset["target_demand"].sum(),
                    "median_lead_days": subset["lead_days"].median(),
                }
            )
    return pd.DataFrame(summaries)


def build_calendar_impact_report(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    calendar_columns = [
        "date",
        "is_public_holiday",
        "is_pre_holiday_3d",
        "is_post_holiday_3d",
        "is_ramadan",
        "is_religious_special_day",
        "is_midterm_break",
        "is_semester_break",
        "is_summer_break",
    ]
    working = panel.loc[panel["store_observed"].eq(1)].merge(
        calendar[calendar_columns],
        on="date",
        how="left",
        validate="many_to_one",
    )
    working["calendar_context"] = np.select(
        [
            working["is_public_holiday"].eq(1),
            working["is_pre_holiday_3d"].eq(1),
            working["is_post_holiday_3d"].eq(1),
            working["is_ramadan"].eq(1),
            working["is_religious_special_day"].eq(1),
            working["is_midterm_break"].eq(1),
            working["is_semester_break"].eq(1),
            working["is_summer_break"].eq(1),
            working["date"].dt.dayofweek.ge(5),
        ],
        [
            "public_holiday",
            "pre_holiday_3d",
            "post_holiday_3d",
            "ramadan_nonholiday",
            "religious_special_day",
            "school_midterm_break",
            "school_semester_break",
            "school_summer_break",
            "weekend_nonholiday",
        ],
        default="ordinary_day",
    )
    working["demand_occurs"] = working["daily_demand"].gt(0).astype(int)
    summary = (
        working.groupby(["unit", "calendar_context"], as_index=False)
        .agg(
            rows=("daily_demand", "size"),
            products=("product_id", "nunique"),
            positive_rate=("demand_occurs", "mean"),
            mean_daily_demand=("daily_demand", "mean"),
            median_daily_demand=("daily_demand", "median"),
            total_demand=("daily_demand", "sum"),
        )
    )
    ordinary = summary.loc[
        summary["calendar_context"].eq("ordinary_day"),
        ["unit", "positive_rate", "mean_daily_demand"],
    ].rename(
        columns={
            "positive_rate": "ordinary_positive_rate",
            "mean_daily_demand": "ordinary_mean_daily_demand",
        }
    )
    summary = summary.merge(ordinary, on="unit", how="left", validate="many_to_one")
    summary["positive_rate_ratio_vs_ordinary"] = (
        summary["positive_rate"] / summary["ordinary_positive_rate"]
    )
    summary["mean_demand_ratio_vs_ordinary"] = (
        summary["mean_daily_demand"] / summary["ordinary_mean_daily_demand"]
    )
    return summary.sort_values(["unit", "calendar_context"]).reset_index(drop=True)


def build_weather_impact_report(
    panel: pd.DataFrame,
    observed_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Gerçekleşen havayı yalnız betimsel EDA için satışla eşler."""
    daily = (
        panel.loc[panel["store_observed"].eq(1)]
        .groupby(["date", "unit"], as_index=False)
        .agg(
            total_demand=("daily_demand", "sum"),
            products_with_demand=("daily_demand", lambda values: values.gt(0).sum()),
            active_products=("product_id", "nunique"),
        )
    )
    daily = daily.merge(
        observed_weather,
        on="date",
        how="inner",
        validate="many_to_one",
    )
    conditions = {
        "rainy_1mm_plus": daily["precipitation_mm"].ge(1.0),
        "heavy_rain_10mm_plus": daily["precipitation_mm"].ge(10.0),
        "snow_day": daily["snowfall_cm"].gt(0),
        "hot_30c_plus": daily["temperature_max_c"].ge(30.0),
        "cold_below_10c": daily["temperature_max_c"].lt(10.0),
        "sunny_dry": daily["precipitation_mm"].lt(1.0)
        & daily["cloud_cover_pct"].lt(40.0),
        "windy_30kmh_plus": daily["wind_max_kmh"].ge(30.0),
    }
    rows = []
    for unit in ("KG", "ADT"):
        unit_daily = daily.loc[daily["unit"].eq(unit)]
        for condition, mask in conditions.items():
            unit_mask = mask.loc[unit_daily.index]
            exposed = unit_daily.loc[unit_mask]
            comparison = unit_daily.loc[~unit_mask]
            if exposed.empty or comparison.empty:
                continue
            rows.append(
                {
                    "unit": unit,
                    "weather_condition": condition,
                    "condition_days": len(exposed),
                    "comparison_days": len(comparison),
                    "condition_mean_total_demand": exposed["total_demand"].mean(),
                    "comparison_mean_total_demand": comparison[
                        "total_demand"
                    ].mean(),
                    "mean_total_demand_ratio": (
                        exposed["total_demand"].mean()
                        / comparison["total_demand"].mean()
                    ),
                    "condition_mean_products_with_demand": exposed[
                        "products_with_demand"
                    ].mean(),
                    "comparison_mean_products_with_demand": comparison[
                        "products_with_demand"
                    ].mean(),
                    "observed_start": unit_daily["date"].min(),
                    "observed_end": unit_daily["date"].max(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_project_dirs()
    raw = read_raw_sales()
    cleaned = clean_sales_records(raw)
    valid = valid_sales(cleaned)
    if len(valid) != len(raw):
        raise ValueError("Geçersiz kayıtlar çözülmeden model paneli oluşturulamaz.")

    data_end = valid["date"].max()
    catalog = build_product_catalog(valid, data_end)
    panel = build_daily_panel(valid, catalog, data_end)
    calendar_end = data_end + pd.Timedelta(days=MAX_FORECAST_LEAD_DAYS)
    calendar = build_turkey_calendar(valid["date"].min(), calendar_end)
    calendar.to_csv(PROCESSED_DIR / "turkey_calendar_daily.csv", index=False)
    (REFERENCE_DIR / "calendar_sources.json").write_text(
        json.dumps(calendar_reference_metadata(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    calendar_impact = build_calendar_impact_report(panel, calendar)
    calendar_impact.to_csv(
        REPORT_CSV_DIR / "calendar_demand_impact.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, unit in zip(axes, ("KG", "ADT")):
        subset = calendar_impact.loc[
            calendar_impact["unit"].eq(unit)
            & ~calendar_impact["calendar_context"].eq("ordinary_day")
        ].sort_values("mean_demand_ratio_vs_ordinary")
        positions = np.arange(len(subset))
        axis.barh(
            positions - 0.18,
            subset["mean_demand_ratio_vs_ordinary"],
            height=0.36,
            label="Ortalama miktar oranı",
            color="#6366f1",
        )
        axis.barh(
            positions + 0.18,
            subset["positive_rate_ratio_vs_ordinary"],
            height=0.36,
            label="Pozitif talep oranı",
            color="#22c55e",
        )
        axis.axvline(1.0, color="#ef4444", linestyle="--", linewidth=1)
        axis.set_yticks(positions, subset["calendar_context"])
        axis.set_title(f"{unit} — Normal Güne Göre Takvim Etkisi")
        axis.set_xlabel("Oran (1 = normal gün)")
        axis.grid(axis="x", alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "calendar_demand_impact.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    observed_weather, forecast_weather = load_weather_reference()
    weather_impact = build_weather_impact_report(panel, observed_weather)
    weather_impact.to_csv(
        REPORT_CSV_DIR / "weather_demand_impact.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, unit in zip(axes, ("KG", "ADT")):
        subset = weather_impact.loc[weather_impact["unit"].eq(unit)].sort_values(
            "mean_total_demand_ratio"
        )
        axis.barh(
            subset["weather_condition"],
            subset["mean_total_demand_ratio"],
            color="#0ea5e9",
        )
        axis.axvline(1.0, color="#ef4444", linestyle="--", linewidth=1)
        axis.set_title(f"{unit} — Gözlenen Hava / Günlük Talep İlişkisi")
        axis.set_xlabel("Koşullu gün / diğer gün ortalama talep oranı")
        axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "weather_demand_impact.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    cleaned.to_csv(PROCESSED_DIR / "sales_cleaned.csv", index=False)
    panel.to_csv(PROCESSED_DIR / "daily_product_demand.csv", index=False)
    catalog.to_csv(PROCESSED_DIR / "product_catalog.csv", index=False)

    high_value_thresholds = valid.groupby("unit")["quantity"].quantile(0.999)
    quality_issues = valid.loc[
        valid.apply(
            lambda row: row["quantity"] > high_value_thresholds.loc[row["unit"]],
            axis=1,
        )
    ].copy()
    quality_issues["issue_type"] = "high_value_review_candidate"
    quality_issues.to_csv(REPORT_CSV_DIR / "data_quality_issues.csv", index=False)
    catalog.to_csv(REPORT_CSV_DIR / "product_history_summary.csv", index=False)

    snapshots = build_snapshot_table(panel)
    examples = build_direct_horizon_examples(
        snapshots,
        LEAD_DAY_GRID,
        observed_weather=observed_weather,
        forecast_weather=forecast_weather,
    )
    examples["source_max_date"] = examples["forecast_origin"]
    if not examples["source_max_date"].le(examples["forecast_origin"]).all():
        raise AssertionError("Feature kaynak tarihi forecast origin sonrasına taşıyor.")
    if not examples["target_date"].gt(examples["forecast_origin"]).all():
        raise AssertionError("Hedef tarihi forecast origin sonrasında olmalıdır.")
    if examples[ALL_CANDIDATE_FEATURES].isna().any().any():
        raise AssertionError("Model feature alanlarında eksik değer bulundu.")
    if not pd.to_datetime(examples["weather_source_max_date"]).le(
        examples["forecast_origin"]
    ).all():
        raise AssertionError("Hava feature kaynağı forecast origin sonrasına geçti.")

    split_dates = choose_splits(examples)
    split_frames = split_examples(examples, split_dates)
    split_summary = save_model_ready(split_frames)
    split_summary.to_csv(REPORT_CSV_DIR / "data_prep_summary.csv", index=False)

    inference_panel = build_daily_panel(
        valid, catalog, data_end, extend_all_to_end=True
    )
    inference_snapshots = build_snapshot_table(inference_panel)
    inference = build_inference_features(inference_snapshots, catalog, data_end)
    inference_columns = [
        "date",
        "product_id",
        "product_name",
        "unit",
        "segment",
        "current_status",
    ] + SNAPSHOT_FEATURES
    inference[inference_columns].to_csv(
        MODEL_READY_DIR / "inference_snapshot.csv", index=False
    )

    feature_specification = {
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "forecast_type": "direct_multi_horizon_daily",
        "forecast_origin_policy": "latest_observed_store_date",
        "target": {
            "occurrence": "selected product has demand > 0 on target_date",
            "quantity": "daily quantity on target_date in product unit",
        },
        "max_forecast_lead_days": MAX_FORECAST_LEAD_DAYS,
        "lead_day_grid": list(LEAD_DAY_GRID),
        "required_history_days": MIN_HISTORY_DAYS,
        "model_features": MODEL_FEATURES,
        "candidate_features": ALL_CANDIDATE_FEATURES,
        "snapshot_features": SNAPSHOT_FEATURES,
        "target_date_features": TARGET_DATE_FEATURES,
        "special_calendar_features": SPECIAL_CALENDAR_FEATURES,
        "calendar_version": CALENDAR_VERSION,
        "calendar_reference": "data/reference/calendar_sources.json",
        "calendar_coverage": {
            "start": calendar["date"].min().date().isoformat(),
            "end": calendar["date"].max().date().isoformat(),
        },
        "weather_feature_version": WEATHER_FEATURE_VERSION,
        "weather_features": WEATHER_MODEL_FEATURES,
        "weather_reference": "data/reference/weather_sources.json",
        "weather_policy": (
            "Observed target-date weather is EDA-only. Candidate fields use a "
            "15-day seasonal climatology estimated from ERA5 dates strictly "
            "before each forecast_origin; deployment requires validation ablation."
        ),
        "weather_deployment_decision": (
            "candidate_only_excluded_after_validation_ablation"
        ),
        "zero_demand_policy": (
            "Observed store day and product active window without sale = 0; "
            "globally missing store day = missing_or_unobserved."
        ),
        "activity_policy": (
            f"Panel starts at first sale and extends at most {ACTIVE_TAIL_DAYS} days "
            "after last sale, capped at dataset end."
        ),
        "leakage_rule": "source_max_date <= forecast_origin < target_date",
    }
    (MODELS_DIR / "feature_specification.json").write_text(
        json.dumps(feature_specification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split_metadata = {
        "forecast_type": "direct_multi_horizon_daily",
        "forecast_origin_latest": data_end.date().isoformat(),
        "max_forecast_lead_days": MAX_FORECAST_LEAD_DAYS,
        "lead_day_grid": list(LEAD_DAY_GRID),
        "split_basis": "target_date",
        "temporal_gap_days": 7,
        "splits": {
            key: {
                nested_key: nested_value.date().isoformat()
                for nested_key, nested_value in value.items()
            }
            for key, value in split_dates.items()
        },
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "calendar_version": CALENDAR_VERSION,
        "calendar_reference": "data/reference/calendar_sources.json",
        "weather_feature_version": WEATHER_FEATURE_VERSION,
        "weather_reference": "data/reference/weather_sources.json",
        "raw_sha256": file_sha256(RAW_DATA),
        "zero_demand_policy": feature_specification["zero_demand_policy"],
        "rows_by_split_and_unit": split_summary.to_dict(orient="records"),
    }
    (MODEL_READY_DIR / "split_metadata.json").write_text(
        json.dumps(split_metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    actions = pd.DataFrame(
        [
            {
                "Aşama": "Parsing",
                "Sorun": "Türkçe sayı biçimi",
                "Karar": "Nokta binlik, virgül ondalık olarak ayrıştırıldı",
                "Gerekçe": "Ham veri sözleşmesi ve üç karma biçimli kayıt",
                "Risk": "Düşük",
            },
            {
                "Aşama": "Günlük panel",
                "Sorun": "Satışsız gün",
                "Karar": "Gözlenen mağaza gününde aktif pencerede 0; genel eksik gün NaN",
                "Gerekçe": "Sessiz sıfır atamasını engellemek",
                "Risk": "Orta",
            },
            {
                "Aşama": "Hedef",
                "Sorun": "Kullanıcı seçili gelecek tarih istiyor",
                "Karar": "Doğrudan çok-ufuklu günlük hedef",
                "Gerekçe": f"1–{MAX_FORECAST_LEAD_DAYS} gün arası seçilebilir tarih",
                "Risk": "Orta",
            },
            {
                "Aşama": "Split",
                "Sorun": "Zaman sızıntısı",
                "Karar": "Target-date tabanlı kronolojik train/validation/test",
                "Gerekçe": "Gelecek hedef günleri eğitimden ayırmak",
                "Risk": "Düşük",
            },
            {
                "Aşama": "Özel takvim",
                "Sorun": "Bayram, tatil ve okul dönemlerinde talep rejimi değişebilir",
                "Karar": (
                    "Resmî/dinî tatil, Ramazan, tatil önce-sonra pencereleri ve "
                    "MEB okul dönemleri target-date feature'ı yapıldı"
                ),
                "Gerekçe": "Tahmin anında önceden bilinen dışsal takvim bilgisi",
                "Risk": "Yerel okul kapanışları ve mağaza özel çalışma saatleri eksik",
            },
            {
                "Aşama": "Hava durumu",
                "Sorun": "Hedef gün gerçekleşen hava bilgisi gelecekte bilinmez",
                "Karar": (
                    "2018'den başlayan ERA5 geçmişinden, her forecast origin öncesi "
                    "±15 günlük mevsimsel klimatoloji üretildi"
                ),
                "Gerekçe": (
                    "1–180 gün boyunca train/serving eşitliği ve hedef-gün hava "
                    "sızıntısını engellemek"
                ),
                "Risk": (
                    "Mağaza açık adresi yerine Bursa/Osmangazi merkez koordinatı; "
                    "ani gelecek hava değişimi uzun ufukta bilinemez"
                ),
            },
        ]
    )
    actions.to_csv(REPORT_CSV_DIR / "dataprep_actions.csv", index=False)

    report = f"""# DataPrep Handoff — Seçili Tarih Günlük Talep Tahmini

## Veri durumu

- Ham SHA-256: `{file_sha256(RAW_DATA)}`
- Temiz kayıt: {len(valid):,}
- Günlük panel satırı: {len(panel):,}
- Model örneği: {len(examples):,}
- Aktif inference ürünü: {len(inference):,}
- Son bilinen tarih / forecast origin: {data_end.date()}

## Tahmin sözleşmesi

Kullanıcı bir ürün ve gelecek `target_date` seçer. Model `lead_days`, hedef günün
takvim özellikleri ve son bilinen tarihteki ürün snapshot feature'larıyla doğrudan
o günün talep oluşumunu ve günlük miktarını tahmin eder. Doğrulanmış aralık
1–{MAX_FORECAST_LEAD_DAYS} gündür.

## Sıfır ve aktiflik politikası

Gözlenen mağaza gününde ürünün aktif panel penceresinde satış satırı yoksa günlük
talep `0` kabul edilir. Mağaza genelinde hiç kayıt olmayan gün bilinmeyen kalır.
Panel ilk satıştan başlar ve son satıştan sonra en fazla {ACTIVE_TAIL_DAYS} gün uzar.

## Feature sözleşmesi

Snapshot feature'ları yalnız `forecast_origin` ve öncesini kullanır. Hedef güne ait
yalnız önceden bilinen takvim alanları ve `lead_days` eklenir. Takvim sürümü
`{CALENDAR_VERSION}`; resmî/dinî tatil, Ramazan, tatilden 1/3 gün önce-sonra,
MEB okul dönemi, ara/yarıyıl/yaz tatili ve olağanüstü okul kapanışı feature'larını
içerir. Kaynaklar `data/reference/calendar_sources.json`, model alanları
`models/feature_specification.json` dosyasındadır.

Hava katmanı `{WEATHER_FEATURE_VERSION}` sürümündedir. Gerçekleşen hedef-gün havası
yalnız EDA'da kullanılır; aday alanlar her satır için yalnız `forecast_origin`
öncesindeki ERA5 geçmişinden hedef mevsime ait ±15 günlük iklim normalini hesaplar. Bursa/
Osmangazi merkez koordinatı kullanılmıştır. Kaynak ve checksum bilgisi
`data/reference/weather_sources.json` içindedir.

## Split

Split `target_date` temelindedir:

- Train: {split_dates['train']['start'].date()} – {split_dates['train']['end'].date()}
- Validation: {split_dates['validation']['start'].date()} – {split_dates['validation']['end'].date()}
- Test: {split_dates['test']['start'].date()} – {split_dates['test']['end'].date()}

Aralarda 7 günlük tampon bulunur. Test model, threshold veya feature seçimi için
kullanılmayacaktır.

## Leakage

`source_max_date <= forecast_origin < target_date` kontrolü tüm örneklerde geçti.
KG ve ADT hedefleri ayrı modelleme için korunmuştur.

## Kalan riskler

- Kampanya, fiyat, stokta yokluk ve mağaza özel çalışma saatleri veri setinde yoktur.
- Aday hava alanları kesin hedef-gün gerçekleşmesi değil time-safe Bursa iklim normalidir;
  final kullanım kararı validation ablation'a bağlıdır.
- Takvim Türkiye geneli resmî kaynakları kullanır; yerel okul/mağaza kapanışları yoktur.
- 180 güne yaklaşan tahminler yakın tarihlerden daha belirsizdir.
- Son 90 günde satılmamış ürünler inference kataloğunda pasif kabul edilir.
"""
    (REPORT_MD_DIR / "DATA_PREP_HANDOFF.md").write_text(report, encoding="utf-8")

    print(split_summary.to_string(index=False))
    print(f"\nPanel: {len(panel):,} | Örnek: {len(examples):,} | Inference: {len(inference):,}")


if __name__ == "__main__":
    main()
