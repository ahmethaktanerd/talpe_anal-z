"""Projenin veri, zaman güvenliği, bundle, notebook ve servis sözleşmesini doğrular."""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import date
from typing import Dict, List

import pandas as pd

from app.services.bundle_loader import load_metadata, verify_checksums
from app.services.forecast_service import DemandForecastService
from models.demand_forecasting_bundle.feature_builder import MODEL_FEATURES
from scripts.demand_features import SPECIAL_CALENDAR_FEATURES
from scripts.data_utils import file_sha256
from scripts.project_config import (
    BUNDLE_DIR,
    MODEL_READY_DIR,
    PROJECT_ROOT,
    RAW_DATA,
    REPORT_CSV_DIR,
    REPORT_MD_DIR,
)
from scripts.turkey_calendar import CALENDAR_VERSION, describe_calendar_date
from scripts.weather_features import (
    WEATHER_FEATURE_VERSION,
    WEATHER_MODEL_FEATURES,
    load_weather_reference,
)


EXPECTED_RAW_SHA256 = (
    "b674355d6d3783792d9c6a4cdfb3a0098db3702b8e0e725aca0ff35138e2cb75"
)


def main() -> None:
    results: List[Dict[str, str]] = []

    def record(group: str, check: str, passed: bool, evidence: str) -> None:
        results.append(
            {
                "group": group,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    raw_hash = file_sha256(RAW_DATA)
    record(
        "raw_data",
        "immutable_source_checksum",
        raw_hash == EXPECTED_RAW_SHA256,
        raw_hash,
    )

    profile = pd.read_csv(REPORT_CSV_DIR / "eda_data_profile.csv")
    profile_map = dict(zip(profile["metric"], profile["value"]))
    record(
        "raw_data",
        "all_rows_valid",
        int(profile_map["raw_rows"]) == int(profile_map["valid_rows"]) == 183_184,
        f"{profile_map['valid_rows']}/{profile_map['raw_rows']} satır",
    )
    record(
        "raw_data",
        "product_and_unit_coverage",
        int(profile_map["unique_products"]) == 713
        and int(profile_map["kg_rows"]) > 0
        and int(profile_map["adt_rows"]) > 0,
        (
            f"713 ürün; KG={profile_map['kg_rows']}, "
            f"ADT={profile_map['adt_rows']}"
        ),
    )

    calendar_manifest_path = (
        PROJECT_ROOT / "data" / "reference" / "calendar_sources.json"
    )
    calendar_manifest = json.loads(
        calendar_manifest_path.read_text(encoding="utf-8")
    )
    record(
        "calendar",
        "official_source_manifest",
        calendar_manifest["calendar_version"] == CALENDAR_VERSION
        and len(calendar_manifest["sources"]) >= 10,
        (
            f"{calendar_manifest['calendar_version']}; "
            f"{len(calendar_manifest['sources'])} kaynak kaydı"
        ),
    )
    second_january_context = describe_calendar_date(pd.Timestamp("2027-01-02"))
    record(
        "calendar",
        "arbitrary_target_date_context",
        second_january_context["days_since_public_holiday"] == 1
        and second_january_context["school_status"] == "in_session",
        "2027-01-02: yılbaşından 1 gün sonra, okul dönemi",
    )

    weather_manifest_path = (
        PROJECT_ROOT / "data" / "reference" / "weather_sources.json"
    )
    weather_manifest = json.loads(
        weather_manifest_path.read_text(encoding="utf-8")
    )
    observed_weather, archived_weather = load_weather_reference(
        PROJECT_ROOT / "data" / "reference"
    )
    record(
        "weather",
        "source_and_location_contract",
        weather_manifest["weather_data_version"].startswith("BURSA_WEATHER")
        and weather_manifest["location"]["name"] == "Bursa"
        and len(observed_weather) >= 3000,
        (
            f"Bursa/Osmangazi; ERA5={len(observed_weather)} gün; "
            f"previous-runs={len(archived_weather)} satır"
        ),
    )
    record(
        "weather",
        "v1_no_incomplete_forecast_archive",
        not weather_manifest["archived_forecasts"]["enabled_in_current_bundle"]
        and archived_weather.empty,
        "Eksik forecast arşivi estimator'a bağlanmadı; climatology-only aday katman",
    )

    split_ranges = {}
    for split in ("train", "validation", "test"):
        feature_path = MODEL_READY_DIR / f"demand_features_{split}.csv"
        target_path = MODEL_READY_DIR / f"demand_targets_{split}.csv"
        feature_rows = 0
        target_rows = sum(1 for _ in target_path.open(encoding="utf-8")) - 1
        complete = True
        time_safe = True
        target_min = None
        target_max = None
        usecols = ["forecast_origin", "target_date", *MODEL_FEATURES]
        for chunk in pd.read_csv(
            feature_path,
            usecols=usecols,
            parse_dates=["forecast_origin", "target_date"],
            chunksize=50_000,
        ):
            feature_rows += len(chunk)
            complete = complete and bool(chunk[MODEL_FEATURES].notna().all(axis=None))
            time_safe = time_safe and bool(
                chunk["target_date"].gt(chunk["forecast_origin"]).all()
            )
            chunk_min = chunk["target_date"].min()
            chunk_max = chunk["target_date"].max()
            target_min = chunk_min if target_min is None else min(target_min, chunk_min)
            target_max = chunk_max if target_max is None else max(target_max, chunk_max)
        split_ranges[split] = (target_min, target_max)
        record(
            "model_ready",
            f"{split}_row_alignment",
            feature_rows == target_rows,
            f"features={feature_rows}, targets={target_rows}",
        )
        record(
            "model_ready",
            f"{split}_feature_completeness",
            complete,
            f"{len(MODEL_FEATURES)} model feature'ı",
        )
        record(
            "model_ready",
            f"{split}_time_safety",
            time_safe,
            f"{target_min.date()}–{target_max.date()}",
        )

    chronological = (
        split_ranges["train"][1] < split_ranges["validation"][0]
        and split_ranges["validation"][1] < split_ranges["test"][0]
    )
    record(
        "model_ready",
        "chronological_split_order",
        chronological,
        "train < validation < test ve aralarda zaman tamponu",
    )

    verify_checksums(BUNDLE_DIR)
    metadata = load_metadata(BUNDLE_DIR)
    checksum_entries = json.loads(
        (BUNDLE_DIR / "checksums.json").read_text(encoding="utf-8")
    )
    bundle_files = {
        path.name
        for path in BUNDLE_DIR.iterdir()
        if path.is_file() and path.name != "checksums.json"
    }
    record(
        "bundle",
        "checksum_verification",
        set(checksum_entries) == bundle_files,
        f"{len(checksum_entries)} dosya doğrulandı",
    )
    record(
        "bundle",
        "forecast_contract",
        metadata["forecast_type"] == "direct_multi_horizon_daily"
        and int(metadata["max_forecast_lead_days"]) == 180,
        "ürün + seçilen tek hedef gün; lead=1–180",
    )
    record(
        "bundle",
        "calendar_feature_contract",
        metadata["calendar_version"] == CALENDAR_VERSION
        and metadata["calendar_features"] == SPECIAL_CALENDAR_FEATURES
        and len(metadata["feature_columns"]) == len(MODEL_FEATURES),
        (
            f"{metadata['calendar_version']}; "
            f"{len(metadata['calendar_features'])} özel / "
            f"{len(metadata['feature_columns'])} toplam feature"
        ),
    )
    record(
        "bundle",
        "weather_feature_governance",
        metadata["weather_feature_version"] == WEATHER_FEATURE_VERSION
        and metadata["weather_candidate_features"] == WEATHER_MODEL_FEATURES
        and metadata["weather_features"] == []
        and metadata["weather_deployment_decision"]
        == "excluded_after_validation_ablation_no_robust_gain",
        (
            f"{len(metadata['weather_candidate_features'])} aday; "
            f"{len(metadata['weather_features'])} deploy; "
            f"{metadata['weather_deployment_decision']}"
        ),
    )

    notebook_path = (
        PROJECT_ROOT / "notebooks" / "retail_demand_forecasting_story.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    notebook_errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    notebook_executed = all(
        cell.get("execution_count") is not None for cell in code_cells
    )
    record(
        "notebook",
        "executed_without_errors",
        notebook_executed and not notebook_errors,
        f"{len(code_cells)} kod hücresi, {len(notebook_errors)} hata",
    )

    service = DemandForecastService(PROJECT_ROOT)
    options = service.product_options()
    examples = []
    for unit in ("KG", "ADT"):
        product = options.loc[
            options["unit"].eq(unit)
            & options["forecast_available"]
            & options["current_status"].eq("active")
        ].iloc[0]
        forecast = service.forecast(
            str(product["product_id"]), date(2027, 1, 2)
        )
        examples.append(forecast)
        record(
            "forecast_service",
            f"{unit.lower()}_2027_01_02",
            forecast["lead_days"] == 165
            and forecast["unit"] == unit
            and 0 <= forecast["demand_probability"] <= 1
            and forecast["display_quantity"] >= 0,
            (
                f"{forecast['product_name']}; talep={forecast['demand_expected']}; "
                f"miktar={forecast['display_quantity']} {unit}"
            ),
        )
        record(
            "forecast_service",
            f"{unit.lower()}_calendar_context",
            forecast["calendar_context"]["calendar_version"] == CALENDAR_VERSION
            and forecast["calendar_context"]["school_status"] == "in_session",
            "takvim bağlamı servis sonucunda mevcut",
        )
        record(
            "forecast_service",
            f"{unit.lower()}_weather_context",
            forecast["weather_context"]["source"] == "time_safe_climatology"
            and not forecast["weather_context"]["is_fixed_lead_forecast"]
            and not forecast["weather_context"]["used_by_model"]
            and forecast["weather_context"]["climatology_sample_days"] >= 90,
            (
                f"{forecast['weather_context']['location']}; "
                f"n={forecast['weather_context']['climatology_sample_days']}"
            ),
        )

    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    pytest_summary = (pytest_result.stdout or pytest_result.stderr).strip().splitlines()
    record(
        "automated_tests",
        "pytest_suite",
        pytest_result.returncode == 0,
        pytest_summary[-1] if pytest_summary else "çıktı yok",
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        streamlit_port = probe.getsockname()[1]
    streamlit_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(streamlit_port),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    http_status = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{streamlit_port}", timeout=2
                ) as response:
                    http_status = response.status
                break
            except Exception:
                if streamlit_process.poll() is not None:
                    break
                time.sleep(0.5)
    finally:
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            streamlit_process.kill()
            streamlit_process.wait(timeout=5)
    record(
        "streamlit",
        "http_smoke_test",
        http_status == 200,
        f"HTTP {http_status}" if http_status is not None else "HTTP yanıtı alınamadı",
    )

    result_frame = pd.DataFrame(results)
    result_path = REPORT_CSV_DIR / "final_validation_results.csv"
    result_frame.to_csv(result_path, index=False)

    metrics = metadata["test_metrics"]
    ablation = pd.read_csv(
        REPORT_CSV_DIR / "calendar_feature_ablation.csv"
    )
    ablation_lines = []
    for unit in ("KG", "ADT"):
        full = ablation.loc[
            ablation["unit"].eq(unit)
            & ablation["feature_set"].eq("full_calendar")
        ].iloc[0]
        removed = ablation.loc[
            ablation["unit"].eq(unit)
            & ablation["feature_set"].eq("special_calendar_removed")
        ].iloc[0]
        ablation_lines.append(
            f"- {unit}: validation PR-AUC farkı "
            f"**{full['occurrence_pr_auc'] - removed['occurrence_pr_auc']:+.4f}**, "
            f"MAE farkı **{full['quantity_mae'] - removed['quantity_mae']:+.4f}**."
        )
    ablation_summary = "\n".join(ablation_lines)
    weather_ablation = pd.read_csv(
        REPORT_CSV_DIR / "weather_feature_ablation.csv"
    )
    weather_ablation_lines = []
    for unit in ("KG", "ADT"):
        candidate = weather_ablation.loc[
            weather_ablation["unit"].eq(unit)
            & weather_ablation["feature_set"].eq("candidate_with_weather")
        ].iloc[0]
        deployed = weather_ablation.loc[
            weather_ablation["unit"].eq(unit)
            & weather_ablation["feature_set"].eq("deployed_weather_removed")
        ].iloc[0]
        weather_ablation_lines.append(
            f"- {unit}: aday hava PR-AUC farkı "
            f"**{candidate['occurrence_pr_auc'] - deployed['occurrence_pr_auc']:+.4f}**, "
            f"MAE farkı "
            f"**{candidate['quantity_mae'] - deployed['quantity_mae']:+.4f}**."
        )
    weather_ablation_summary = "\n".join(weather_ablation_lines)
    example_lines = "\n".join(
        (
            f"| {item['product_name']} | {item['unit']} | {item['target_date']} | "
            f"{item['lead_days']} | "
            f"{'Evet' if item['demand_expected'] else 'Hayır'} | "
            f"{item['demand_probability']:.4f} | "
            f"{item['display_quantity']} {item['unit']} |"
        )
        for item in examples
    )
    report = f"""# Proje Final Doğrulama Raporu

## Teslim kararı

**{int(result_frame['status'].eq('PASS').sum())}/{len(result_frame)} kontrol geçti.**
Proje; kullanıcı tarafından seçilen ürün ve desteklenen herhangi bir hedef gün için
günlük talep oluşumu ile ürünün kendi birimindeki miktarı üretir.

## Tahmin sözleşmesi

- Son veri / forecast origin: **{metadata['forecast_origin']}**
- Desteklenen hedef tarih aralığı: **2026-07-22–2027-01-17**
- Karar tanesi: **ürün + kullanıcının seçtiği tek hedef gün**
- Desteklenen birimler: **KG ve ADT**; ürünün katalog birimi değiştirilmez.
- 2 Ocak 2027, origin'den **165 gün** ileridedir ve desteklenir.

## 2 Ocak 2027 servis kanıtı

| Ürün | Birim | Hedef tarih | Lead | Talep bekleniyor | Olasılık | Tahmini günlük miktar |
|---|---:|---:|---:|---:|---:|---:|
{example_lines}

Bu satırlar örnek ürünler içindir. Streamlit'te seçilen ürün değiştiğinde aynı servis
o ürüne özel sonucu üretir.

## Final test performansı

| Birim | Occurrence PR-AUC | Occurrence F1 | Günlük miktar MAE | WAPE | Bias |
|---|---:|---:|---:|---:|---:|
| KG | {metrics['KG']['occurrence']['pr_auc']:.4f} | {metrics['KG']['occurrence']['f1']:.4f} | {metrics['KG']['quantity_end_to_end']['mae']:.4f} | {metrics['KG']['quantity_end_to_end']['wape']:.4f} | {metrics['KG']['quantity_end_to_end']['bias']:.4f} |
| ADT | {metrics['ADT']['occurrence']['pr_auc']:.4f} | {metrics['ADT']['occurrence']['f1']:.4f} | {metrics['ADT']['quantity_end_to_end']['mae']:.4f} | {metrics['ADT']['quantity_end_to_end']['wape']:.4f} | {metrics['ADT']['quantity_end_to_end']['bias']:.4f} |

## Özel takvim

- Takvim sürümü: **{metadata['calendar_version']}**
- Özel takvim feature'ı: **{len(metadata['calendar_features'])}**
- Toplam model feature'ı: **{len(metadata['feature_columns'])}**
- Kapsam: resmî/dinî tatil, yarım gün, Ramazan, kandil, tatilden önce/sonra
  pencereleri ve MEB okul/ara/yarıyıl/yaz tatili.
- Validation ablation: `reports/csv/calendar_feature_ablation.csv`
- Takvim segment testleri: `reports/csv/calendar_segment_metrics.csv`
{ablation_summary}

## Bursa hava katmanı

- Hava feature sürümü: **{metadata['weather_feature_version']}**
- Konum: **{metadata['weather_location']}**
- Gerçekleşen hedef-gün havası yalnız EDA'dadır; estimator'a verilmez.
- ERA5 geçmişinden origin öncesi ±15 günlük klimatolojiyle 7 aday alan üretildi.
- Validation'da sağlam/birimler arası tutarlı kazanım görülmediği için deploy
  edilen hava feature sayısı **{len(metadata['weather_features'])}**;
  karar: `{metadata['weather_deployment_decision']}`.
- Adaylar model-ready tabloda ve açıklayıcı arayüz bağlamında korunur.
- Ablation: `reports/csv/weather_feature_ablation.csv`
{weather_ablation_summary}

## Operasyonel yorum

Tahmini miktar, seçilen gündeki **brüt mağaza talebi** tahminidir. Elde mağaza stoku,
yoldaki sevkiyat ve emniyet stoku olmadığı için tek başına kesin net depo transfer
emri değildir. Stok sıfır kabul edilirse tahmini miktar depodan çekilecek başlangıç
miktarı olarak kullanılabilir.

## Bilinen riskler

- Fiyat, kampanya, stokta yokluk ve mağaza özel kapanışı verileri yoktur.
- Hava bağlamı Bursa merkez klimatolojisidir; geleceğin kesin hedef-gün havası değildir.
- Takvim Türkiye genelidir; yerel okul ve mağaza çalışma değişikliklerini içermez.
- 90–180 günlük tahminler yakın vadeden daha belirsizdir.
- Gösterilen olasılık kalibre edilmiş güven skoru değildir.
- ADT miktar WAPE değeri yüksek olduğundan adet tahminleri operasyonel izlemede
  özellikle takip edilmelidir.

Makine-okunur kontrol sonucu: `reports/csv/final_validation_results.csv`
"""
    report_path = REPORT_MD_DIR / "PROJECT_FINAL_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    failures = result_frame.loc[result_frame["status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(
            f"{len(failures)} final kontrol başarısız. Ayrıntı: {result_path}"
        )
    print(
        f"Final doğrulama başarılı: {len(result_frame)}/{len(result_frame)} PASS\n"
        f"Rapor: {report_path}"
    )


if __name__ == "__main__":
    main()
