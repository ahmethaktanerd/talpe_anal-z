import hashlib
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from scripts.project_config import RAW_DATA


RAW_COLUMNS = ["satıs_tarıhı", "urun_ıd", "urun_ad", "satılan_mıktar"]
QUANTITY_PATTERN = re.compile(r"^\s*([+-]?\d[\d.,]*)\s*([A-Za-zÇĞİÖŞÜçğıöşü]+)\s*$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_turkish_number(value: str) -> float:
    """Türkçe biçimde virgülü ondalık, noktayı binlik ayıracı kabul eder."""
    text = str(value).strip()
    if not text:
        return np.nan
    normalized = text.replace(".", "").replace(",", ".")
    return float(normalized)


def split_quantity_and_unit(value: str) -> Tuple[float, str, str]:
    match = QUANTITY_PATTERN.match(str(value))
    if not match:
        return np.nan, None, "parse_error"
    number_text, unit = match.groups()
    try:
        quantity = parse_turkish_number(number_text)
    except (TypeError, ValueError):
        return np.nan, unit.upper(), "parse_error"
    return quantity, unit.upper(), "ok"


def read_raw_sales(path: Path = RAW_DATA) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype={"urun_ıd": "string", "urun_ad": "string", "satılan_mıktar": "string"},
    )
    missing_columns = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Eksik ham veri kolonları: {missing_columns}")
    return raw[RAW_COLUMNS].copy()


def clean_sales_records(raw: pd.DataFrame) -> pd.DataFrame:
    parsed = raw["satılan_mıktar"].map(split_quantity_and_unit)
    parsed_frame = pd.DataFrame(
        parsed.tolist(), columns=["quantity", "unit", "parse_status"], index=raw.index
    )
    cleaned = pd.DataFrame(
        {
            "source_row": raw.index + 2,
            "date": pd.to_datetime(raw["satıs_tarıhı"], dayfirst=True, errors="coerce"),
            "product_id": raw["urun_ıd"].astype("string").str.strip(),
            "product_name": raw["urun_ad"].astype("string").str.strip(),
            "raw_quantity": raw["satılan_mıktar"],
        }
    )
    cleaned = pd.concat([cleaned, parsed_frame], axis=1)
    cleaned["date_status"] = np.where(cleaned["date"].isna(), "parse_error", "ok")
    cleaned["record_status"] = np.select(
        [
            cleaned["date"].isna(),
            cleaned["quantity"].isna(),
            ~cleaned["unit"].isin(["KG", "ADT"]),
            cleaned["quantity"] < 0,
            cleaned["quantity"] == 0,
        ],
        [
            "invalid_date",
            "invalid_quantity",
            "invalid_unit",
            "negative_quantity",
            "zero_quantity",
        ],
        default="valid",
    )
    return cleaned


def valid_sales(cleaned: pd.DataFrame) -> pd.DataFrame:
    return cleaned.loc[cleaned["record_status"].eq("valid")].copy()
