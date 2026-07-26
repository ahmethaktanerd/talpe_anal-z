from datetime import date
from typing import Dict

import pandas as pd


def validate_target_date(target_date: date, metadata: Dict) -> int:
    origin = pd.Timestamp(metadata["forecast_origin"]).date()
    if target_date <= origin:
        raise ValueError(f"Hedef tarih {origin} tarihinden sonra olmalıdır.")
    lead_days = (target_date - origin).days
    maximum = int(metadata["max_forecast_lead_days"])
    if lead_days > maximum:
        max_date = origin + pd.Timedelta(days=maximum)
        raise ValueError(
            f"Model en fazla {maximum} gün ileri için doğrulandı. "
            f"Seçilebilecek en ileri tarih: {max_date}."
        )
    return lead_days


def validate_unit(unit: str) -> str:
    normalized = str(unit).upper().strip()
    if normalized not in {"KG", "ADT"}:
        raise ValueError(f"Desteklenmeyen ürün birimi: {unit}")
    return normalized
