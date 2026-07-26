import hashlib
import json
from pathlib import Path
from typing import Dict

import joblib


REQUIRED_METADATA = {
    "model_version",
    "forecast_type",
    "forecast_origin",
    "max_forecast_lead_days",
    "targets",
    "units",
    "model_layout",
    "unit_model_map",
    "required_history_days",
    "feature_columns",
    "feature_builder_version",
    "calendar_version",
    "calendar_features",
    "calendar_coverage",
    "calendar_code_sha256",
    "calendar_source_manifest",
    "calendar_source_manifest_sha256",
    "weather_feature_version",
    "weather_features",
    "weather_candidate_features",
    "weather_deployment_decision",
    "weather_location",
    "weather_policy",
    "weather_source_manifest",
    "weather_code_sha256",
    "weather_source_manifest_sha256",
    "weather_observed_bundle_file",
    "weather_observed_bundle_sha256",
    "weather_forecast_bundle_file",
    "weather_forecast_bundle_sha256",
    "occurrence_threshold",
    "zero_demand_policy",
    "cold_start_policy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_bundle_file(bundle_dir: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Bundle dosya yolu boş veya geçersiz.")
    root = bundle_dir.resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(
            f"Geçersiz veya bulunamayan bundle dosyası: {relative_path}"
        )
    return path


def verify_checksums(bundle_dir: Path) -> None:
    checksum_path = bundle_dir / "checksums.json"
    if not checksum_path.is_file():
        raise FileNotFoundError("checksums.json bulunamadı.")
    expected = json.loads(checksum_path.read_text(encoding="utf-8"))
    for name, expected_hash in expected.items():
        path = resolve_bundle_file(bundle_dir, name)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum uyuşmazlığı: {name}")


def load_metadata(bundle_dir: Path) -> Dict:
    verify_checksums(bundle_dir)
    metadata_path = resolve_bundle_file(bundle_dir, "model_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"Eksik metadata alanları: {sorted(missing)}")
    if set(metadata["units"]) != set(metadata["unit_model_map"]):
        raise ValueError("units ve unit_model_map uyuşmuyor.")
    if metadata["forecast_type"] != "direct_multi_horizon_daily":
        raise ValueError("Desteklenmeyen forecast_type.")
    project_root = bundle_dir.resolve().parents[1]
    calendar_code = project_root / "scripts" / "turkey_calendar.py"
    calendar_manifest = project_root / metadata["calendar_source_manifest"]
    if not calendar_code.is_file() or (
        sha256(calendar_code) != metadata["calendar_code_sha256"]
    ):
        raise ValueError("Takvim kodu model metadata checksum'u ile uyuşmuyor.")
    if not calendar_manifest.is_file() or (
        sha256(calendar_manifest)
        != metadata["calendar_source_manifest_sha256"]
    ):
        raise ValueError("Takvim kaynak manifesti checksum'u ile uyuşmuyor.")
    weather_code = project_root / "scripts" / "weather_features.py"
    weather_manifest = project_root / metadata["weather_source_manifest"]
    if not weather_code.is_file() or (
        sha256(weather_code) != metadata["weather_code_sha256"]
    ):
        raise ValueError("Hava feature kodu model metadata checksum'u ile uyuşmuyor.")
    if not weather_manifest.is_file() or (
        sha256(weather_manifest)
        != metadata["weather_source_manifest_sha256"]
    ):
        raise ValueError("Hava kaynak manifesti checksum'u ile uyuşmuyor.")
    for file_key, hash_key in (
        ("weather_observed_bundle_file", "weather_observed_bundle_sha256"),
        ("weather_forecast_bundle_file", "weather_forecast_bundle_sha256"),
    ):
        weather_file = resolve_bundle_file(bundle_dir, metadata[file_key])
        if sha256(weather_file) != metadata[hash_key]:
            raise ValueError(f"Hava bundle dosyası checksum uyuşmazlığı: {file_key}")
    return metadata


def load_unit_models(bundle_dir: Path, metadata: Dict, unit: str) -> Dict:
    if unit not in metadata["unit_model_map"]:
        raise ValueError(f"Desteklenmeyen birim: {unit}")
    mapping = metadata["unit_model_map"][unit]
    occurrence_path = resolve_bundle_file(
        bundle_dir, mapping["occurrence_model_path"]
    )
    quantity_path = resolve_bundle_file(bundle_dir, mapping["quantity_model_path"])
    return {
        "occurrence": joblib.load(occurrence_path),
        "quantity": joblib.load(quantity_path),
    }
