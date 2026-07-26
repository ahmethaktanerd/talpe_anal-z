import pandas as pd

from scripts.demand_features import (
    SPECIAL_CALENDAR_FEATURES,
    add_target_calendar_features,
)
from scripts.turkey_calendar import (
    CALENDAR_VERSION,
    build_turkey_calendar,
    describe_calendar_date,
)


def test_overlapping_national_and_religious_holiday():
    row = build_turkey_calendar("2023-04-23", "2023-04-23").iloc[0]
    assert row["is_public_holiday"] == 1
    assert row["is_national_holiday"] == 1
    assert row["is_religious_holiday"] == 1
    assert "Ulusal Egemenlik" in row["public_holiday_name"]
    assert "Ramazan Bayramı" in row["public_holiday_name"]


def test_pre_post_holiday_windows_and_school_break():
    calendar = build_turkey_calendar("2023-04-18", "2023-04-24").set_index("date")
    assert calendar.loc[pd.Timestamp("2023-04-18"), "is_pre_holiday_3d"] == 1
    assert calendar.loc[pd.Timestamp("2023-04-24"), "is_post_holiday_3d"] == 1
    assert calendar.loc[pd.Timestamp("2023-04-20"), "is_midterm_break"] == 1


def test_second_january_2027_calendar_context():
    context = describe_calendar_date(pd.Timestamp("2027-01-02"))
    assert context["calendar_version"] == CALENDAR_VERSION
    assert context["days_since_public_holiday"] == 1
    assert context["school_status"] == "in_session"


def test_target_calendar_features_are_complete():
    frame = pd.DataFrame(
        {
            "target_date": pd.to_datetime(
                ["2026-11-16", "2027-01-01", "2027-01-04"]
            )
        }
    )
    result = add_target_calendar_features(frame)
    assert result[SPECIAL_CALENDAR_FEATURES].notna().all(axis=None)
    assert result.loc[0, "target_is_midterm_break"] == 1
    assert result.loc[1, "target_is_public_holiday"] == 1
    assert result.loc[2, "target_is_religious_special_day"] == 1
