"""Türkiye resmî tatil, dinî gün ve MEB okul takvimi feature'ları.

Bu modüldeki tarihler tahmin anında önceden bilinen dışsal takvim bilgisidir.
Satıştan türetilmez ve bu nedenle doğru kullanıldığında zaman sızıntısı yaratmaz.
Kaynaklar `CALENDAR_SOURCES` içinde sürümlü olarak tutulur.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


CALENDAR_VERSION = "TR_CALENDAR_2023_2027_V1"
CALENDAR_MIN_DATE = pd.Timestamp("2023-01-01")
CALENDAR_MAX_DATE = pd.Timestamp("2027-12-31")

CALENDAR_SOURCES = {
    "public_holiday_law_basis": (
        "2429 sayılı Ulusal Bayram ve Genel Tatiller Hakkında Kanun; "
        "Diyanet resmî tatil listelerinde belirtilen yasal dayanak"
    ),
    "diyanet_2023": "https://vakithesaplama.diyanet.gov.tr/icerik.php?icerik=150",
    "diyanet_2024": "https://vakithesaplama.diyanet.gov.tr/dinigunler.php?yil=2024",
    "diyanet_2025": "https://vakithesaplama.diyanet.gov.tr/dinigunler.php?yil=2025",
    "diyanet_2026": "https://vakithesaplama.diyanet.gov.tr/dinigunler.php?yil=2026",
    "diyanet_2027": "https://vakithesaplama.diyanet.gov.tr/icerik.php?icerik=154",
    "meb_2022_2023": (
        "https://www.meb.gov.tr/2022-2023-egitim-ogretim-yilinin-ilk-"
        "ara-tatili-basliyor/haber/28115/tr"
    ),
    "meb_2023_2024": (
        "https://meb.gov.tr/2023-2024-egitim-ogretim-yilina-ait-calisma-"
        "takvimi-aciklandi/haber/30337/tr"
    ),
    "meb_2024_2025": (
        "https://www.meb.gov.tr/2024-2025-egitim-ogretim-yili-takvimi-"
        "aciklandi/haber/33888/tr"
    ),
    "meb_2025_2026": (
        "https://meb.gov.tr/2025-2026-egitim-ogretim-yili-takvimi-"
        "aciklandi/haber/37198/ar"
    ),
    "meb_2026_2027": (
        "https://meb.gov.tr/2026-2027-egitim-ogretim-yili-takvimi-"
        "aciklandi/haber/41057/tr"
    ),
}


RELIGIOUS_HOLIDAYS: Dict[int, Dict[str, Iterable[str]]] = {
    2023: {
        "ramadan_eve": ["2023-04-20"],
        "ramadan_feast": ["2023-04-21", "2023-04-22", "2023-04-23"],
        "sacrifice_eve": ["2023-06-27"],
        "sacrifice_feast": [
            "2023-06-28",
            "2023-06-29",
            "2023-06-30",
            "2023-07-01",
        ],
    },
    2024: {
        "ramadan_eve": ["2024-04-09"],
        "ramadan_feast": ["2024-04-10", "2024-04-11", "2024-04-12"],
        "sacrifice_eve": ["2024-06-15"],
        "sacrifice_feast": [
            "2024-06-16",
            "2024-06-17",
            "2024-06-18",
            "2024-06-19",
        ],
    },
    2025: {
        "ramadan_eve": ["2025-03-29"],
        "ramadan_feast": ["2025-03-30", "2025-03-31", "2025-04-01"],
        "sacrifice_eve": ["2025-06-05"],
        "sacrifice_feast": [
            "2025-06-06",
            "2025-06-07",
            "2025-06-08",
            "2025-06-09",
        ],
    },
    2026: {
        "ramadan_eve": ["2026-03-19"],
        "ramadan_feast": ["2026-03-20", "2026-03-21", "2026-03-22"],
        "sacrifice_eve": ["2026-05-26"],
        "sacrifice_feast": [
            "2026-05-27",
            "2026-05-28",
            "2026-05-29",
            "2026-05-30",
        ],
    },
    2027: {
        "ramadan_eve": ["2027-03-08"],
        "ramadan_feast": ["2027-03-09", "2027-03-10", "2027-03-11"],
        "sacrifice_eve": ["2027-05-15"],
        "sacrifice_feast": [
            "2027-05-16",
            "2027-05-17",
            "2027-05-18",
            "2027-05-19",
        ],
    },
}

RAMADAN_INTERVALS: List[Tuple[str, str]] = [
    ("2023-03-23", "2023-04-20"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-29"),
    ("2026-02-19", "2026-03-19"),
    ("2027-02-08", "2027-03-08"),
]

RELIGIOUS_SPECIAL_DAYS: Dict[str, str] = {
    "2023-01-26": "Regaib Kandili",
    "2023-02-17": "Miraç Kandili",
    "2023-03-06": "Berat Kandili",
    "2023-04-17": "Kadir Gecesi",
    "2023-09-26": "Mevlid Kandili",
    "2024-01-11": "Regaib Kandili",
    "2024-02-06": "Miraç Kandili",
    "2024-02-24": "Berat Kandili",
    "2024-04-05": "Kadir Gecesi",
    "2024-09-14": "Mevlid Kandili",
    "2025-01-02": "Regaib Kandili",
    "2025-01-26": "Miraç Kandili",
    "2025-02-13": "Berat Kandili",
    "2025-03-26": "Kadir Gecesi",
    "2025-09-03": "Mevlid Kandili",
    "2025-12-25": "Regaib Kandili",
    "2026-01-15": "Miraç Kandili",
    "2026-02-02": "Berat Kandili",
    "2026-03-16": "Kadir Gecesi",
    "2026-08-24": "Mevlid Kandili",
    "2026-12-10": "Regaib Kandili",
    "2027-01-04": "Miraç Kandili",
    "2027-01-22": "Berat Kandili",
    "2027-03-05": "Kadir Gecesi",
    "2027-08-13": "Mevlid Kandili",
    "2027-12-02": "Regaib Kandili",
    "2027-12-24": "Miraç Kandili",
}

# Hafta sonları, öğrenci davranışındaki gerçek tatil penceresini temsil etmesi için
# MEB'in ilan ettiği hafta içi ara/yarıyıl tatillerinin çevresine dahil edilmiştir.
SCHOOL_IN_SESSION_INTERVALS: List[Tuple[str, str]] = [
    ("2023-01-01", "2023-01-20"),
    ("2023-02-06", "2023-06-16"),
    ("2023-09-11", "2024-01-19"),
    ("2024-02-05", "2024-06-14"),
    ("2024-09-09", "2025-01-17"),
    ("2025-02-03", "2025-06-20"),
    ("2025-09-08", "2026-01-16"),
    ("2026-02-02", "2026-06-26"),
    ("2026-09-14", "2027-01-22"),
    ("2027-02-08", "2027-06-25"),
]

SCHOOL_MIDTERM_BREAKS: List[Tuple[str, str]] = [
    ("2023-04-15", "2023-04-23"),
    ("2023-11-11", "2023-11-19"),
    ("2024-04-06", "2024-04-14"),
    ("2024-11-09", "2024-11-17"),
    ("2025-03-29", "2025-04-06"),
    ("2025-11-08", "2025-11-16"),
    ("2026-03-14", "2026-03-22"),
    ("2026-11-14", "2026-11-22"),
    ("2027-03-06", "2027-03-14"),
]

SCHOOL_SEMESTER_BREAKS: List[Tuple[str, str]] = [
    ("2023-01-21", "2023-02-05"),
    ("2024-01-20", "2024-02-04"),
    ("2025-01-18", "2025-02-02"),
    ("2026-01-17", "2026-02-01"),
    ("2027-01-23", "2027-02-07"),
]

SCHOOL_SUMMER_BREAKS: List[Tuple[str, str]] = [
    ("2023-06-17", "2023-09-10"),
    ("2024-06-15", "2024-09-08"),
    ("2025-06-21", "2025-09-07"),
    ("2026-06-27", "2026-09-13"),
    ("2027-06-26", "2027-12-31"),
]

EXTRAORDINARY_SCHOOL_CLOSURES: List[Tuple[str, str]] = [
    ("2023-02-06", "2023-02-19"),
]

CALENDAR_MODEL_COLUMNS = [
    "is_public_holiday",
    "is_religious_holiday",
    "is_national_holiday",
    "is_half_day_holiday",
    "is_ramadan",
    "is_religious_special_day",
    "days_to_public_holiday",
    "days_since_public_holiday",
    "is_pre_holiday_1d",
    "is_pre_holiday_3d",
    "is_post_holiday_1d",
    "is_post_holiday_3d",
    "school_in_session",
    "is_school_break",
    "is_midterm_break",
    "is_semester_break",
    "is_summer_break",
    "is_extraordinary_school_closure",
]


def _mark_interval(frame: pd.DataFrame, start: str, end: str, column: str) -> None:
    mask = frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    frame.loc[mask, column] = 1


def _append_name(current: str, name: str) -> str:
    return name if not current else f"{current} | {name}"


def _mark_public_holiday(
    frame: pd.DataFrame,
    event_date: str,
    name: str,
    category: str,
    half_day: bool = False,
) -> None:
    mask = frame["date"].eq(pd.Timestamp(event_date))
    if not mask.any():
        return
    frame.loc[mask, "is_public_holiday"] = 1
    frame.loc[mask, "is_half_day_holiday"] = int(half_day)
    if category == "religious":
        frame.loc[mask, "is_religious_holiday"] = 1
    else:
        frame.loc[mask, "is_national_holiday"] = 1
    current = str(frame.loc[mask, "public_holiday_name"].iloc[0])
    frame.loc[mask, "public_holiday_name"] = _append_name(current, name)


def build_turkey_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start < CALENDAR_MIN_DATE or end > CALENDAR_MAX_DATE or start > end:
        raise ValueError(
            f"Takvim kapsamı {CALENDAR_MIN_DATE.date()}–"
            f"{CALENDAR_MAX_DATE.date()}; istek {start.date()}–{end.date()}."
        )

    frame = pd.DataFrame(
        {"date": pd.date_range(CALENDAR_MIN_DATE, CALENDAR_MAX_DATE, freq="D")}
    )
    for column in CALENDAR_MODEL_COLUMNS:
        frame[column] = 0
    frame["public_holiday_name"] = ""
    frame["religious_special_name"] = ""

    fixed_events = [
        ("01-01", "Yılbaşı", False),
        ("04-23", "Ulusal Egemenlik ve Çocuk Bayramı", False),
        ("05-01", "Emek ve Dayanışma Günü", False),
        ("05-19", "Atatürk'ü Anma Gençlik ve Spor Bayramı", False),
        ("07-15", "Demokrasi ve Millî Birlik Günü", False),
        ("08-30", "Zafer Bayramı", False),
        ("10-28", "Cumhuriyet Bayramı Arifesi", True),
        ("10-29", "Cumhuriyet Bayramı", False),
    ]
    for year in range(2023, 2028):
        for month_day, name, half_day in fixed_events:
            _mark_public_holiday(
                frame,
                f"{year}-{month_day}",
                name,
                category="national",
                half_day=half_day,
            )

    for year, events in RELIGIOUS_HOLIDAYS.items():
        for event_date in events["ramadan_eve"]:
            _mark_public_holiday(
                frame,
                event_date,
                "Ramazan Bayramı Arifesi",
                category="religious",
                half_day=True,
            )
        for day_number, event_date in enumerate(events["ramadan_feast"], start=1):
            _mark_public_holiday(
                frame,
                event_date,
                f"Ramazan Bayramı {day_number}. Gün",
                category="religious",
            )
        for event_date in events["sacrifice_eve"]:
            _mark_public_holiday(
                frame,
                event_date,
                "Kurban Bayramı Arifesi",
                category="religious",
                half_day=True,
            )
        for day_number, event_date in enumerate(events["sacrifice_feast"], start=1):
            _mark_public_holiday(
                frame,
                event_date,
                f"Kurban Bayramı {day_number}. Gün",
                category="religious",
            )

    for interval in RAMADAN_INTERVALS:
        _mark_interval(frame, *interval, "is_ramadan")

    for event_date, name in RELIGIOUS_SPECIAL_DAYS.items():
        mask = frame["date"].eq(pd.Timestamp(event_date))
        frame.loc[mask, "is_religious_special_day"] = 1
        frame.loc[mask, "religious_special_name"] = name

    for interval in SCHOOL_IN_SESSION_INTERVALS:
        _mark_interval(frame, *interval, "school_in_session")
    for interval in SCHOOL_MIDTERM_BREAKS:
        _mark_interval(frame, *interval, "is_midterm_break")
    for interval in SCHOOL_SEMESTER_BREAKS:
        _mark_interval(frame, *interval, "is_semester_break")
    for interval in SCHOOL_SUMMER_BREAKS:
        _mark_interval(frame, *interval, "is_summer_break")
    for interval in EXTRAORDINARY_SCHOOL_CLOSURES:
        _mark_interval(frame, *interval, "is_extraordinary_school_closure")

    frame["is_school_break"] = (
        frame[
            ["is_midterm_break", "is_semester_break", "is_summer_break"]
        ]
        .max(axis=1)
        .astype(int)
    )
    not_in_session = (
        frame["is_school_break"].eq(1)
        | frame["is_extraordinary_school_closure"].eq(1)
    )
    frame.loc[not_in_session, "school_in_session"] = 0

    holiday_dates = (
        frame.loc[frame["is_public_holiday"].eq(1), "date"]
        .sort_values()
        .to_numpy(dtype="datetime64[D]")
    )
    date_values = frame["date"].to_numpy(dtype="datetime64[D]")
    insert_positions = np.searchsorted(holiday_dates, date_values, side="left")
    next_positions = np.clip(insert_positions, 0, len(holiday_dates) - 1)
    previous_insert_positions = (
        np.searchsorted(holiday_dates, date_values, side="right") - 1
    )
    previous_positions = np.clip(
        previous_insert_positions, 0, len(holiday_dates) - 1
    )
    next_days = (holiday_dates[next_positions] - date_values).astype(int)
    previous_days = (date_values - holiday_dates[previous_positions]).astype(int)
    next_days[insert_positions >= len(holiday_dates)] = 31
    previous_days[previous_insert_positions < 0] = 31
    frame["days_to_public_holiday"] = np.clip(next_days, 0, 31)
    frame["days_since_public_holiday"] = np.clip(previous_days, 0, 31)
    frame["is_pre_holiday_1d"] = frame["days_to_public_holiday"].eq(1).astype(int)
    frame["is_pre_holiday_3d"] = (
        frame["days_to_public_holiday"].between(1, 3).astype(int)
    )
    frame["is_post_holiday_1d"] = (
        frame["days_since_public_holiday"].eq(1).astype(int)
    )
    frame["is_post_holiday_3d"] = (
        frame["days_since_public_holiday"].between(1, 3).astype(int)
    )

    frame["school_status"] = np.select(
        [
            frame["is_extraordinary_school_closure"].eq(1),
            frame["is_midterm_break"].eq(1),
            frame["is_semester_break"].eq(1),
            frame["is_summer_break"].eq(1),
            frame["school_in_session"].eq(1),
        ],
        [
            "extraordinary_closure",
            "midterm_break",
            "semester_break",
            "summer_break",
            "in_session",
        ],
        default="outside_defined_school_period",
    )
    integer_columns = CALENDAR_MODEL_COLUMNS
    frame[integer_columns] = frame[integer_columns].astype(np.int16)
    return frame.loc[frame["date"].between(start, end)].reset_index(drop=True)


def describe_calendar_date(target_date: pd.Timestamp) -> Dict:
    row = build_turkey_calendar(target_date, target_date).iloc[0]
    status_labels = {
        "in_session": "Okul dönemi",
        "midterm_break": "Ara tatil",
        "semester_break": "Yarıyıl tatili",
        "summer_break": "Yaz tatili",
        "extraordinary_closure": "Olağanüstü okul kapanışı",
        "outside_defined_school_period": "Tanımlı okul dönemi dışında",
    }
    return {
        "calendar_version": CALENDAR_VERSION,
        "public_holiday_name": row["public_holiday_name"] or None,
        "religious_special_name": row["religious_special_name"] or None,
        "is_public_holiday": bool(row["is_public_holiday"]),
        "is_ramadan": bool(row["is_ramadan"]),
        "school_status": row["school_status"],
        "school_status_label": status_labels[row["school_status"]],
        "days_to_public_holiday": int(row["days_to_public_holiday"]),
        "days_since_public_holiday": int(row["days_since_public_holiday"]),
    }


def calendar_reference_metadata() -> Dict:
    return {
        "calendar_version": CALENDAR_VERSION,
        "coverage_start": CALENDAR_MIN_DATE.date().isoformat(),
        "coverage_end": CALENDAR_MAX_DATE.date().isoformat(),
        "sources": CALENDAR_SOURCES,
        "notes": [
            "MEB ara ve yarıyıl tatilleri öğrenci davranışını temsil etmek için "
            "komşu hafta sonlarını kapsayacak şekilde genişletildi.",
            "28 Ekim ve dinî bayram arifeleri günlük veride half-day bayrağı taşır.",
            "Okul takvimi Türkiye geneli varsayımıdır; il/ilçe özel kapanışlarını içermez.",
        ],
    }
