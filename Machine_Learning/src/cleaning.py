
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from dateutil import parser


MISSING_MARKERS = {
    "",
    "na",
    "n/a",
    "n.a",
    "null",
    "none",
    "nan",
    "unknown",
    "?",
    "not recorded",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "none": 0,
    "many": 8,
    "multiple": 3,
}


@dataclass
class CleaningAudit:
    raw_rows: int
    duplicate_rows_removed: int
    rows_after_dedup: int
    rows_with_missing_target_dropped: int
    final_modeling_rows: int
    date_inconsistencies_fixed: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "raw_rows": self.raw_rows,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "rows_after_dedup": self.rows_after_dedup,
            "rows_with_missing_target_dropped": self.rows_with_missing_target_dropped,
            "final_modeling_rows": self.final_modeling_rows,
            "date_inconsistencies_fixed": self.date_inconsistencies_fixed,
        }


def _normalize_string(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return np.nan
    return text


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def map_vehicle_model(value: Any) -> str | float:
    text = _normalize_string(value)
    if pd.isna(text):
        return np.nan
    mapping = {
        "car": "Car",
        "suv": "SUV",
        "truck": "Truck",
        "van": "Van",
        "bus": "Bus",
        "motorcycle": "Motorcycle",
        "motor cycle": "Motorcycle",
    }
    return mapping.get(_normalize_space(str(text)), np.nan)


def map_maintenance_history(value: Any) -> str | float:
    text = _normalize_string(value)
    if pd.isna(text):
        return np.nan
    mapping = {
        "good": "Good",
        "average": "Average",
        "avg": "Average",
        "poor": "Poor",
        "p00r": "Poor",
    }
    return mapping.get(_normalize_space(str(text)), np.nan)


def map_fuel_type(value: Any) -> str | float:
    text = _normalize_string(value)
    if pd.isna(text):
        return np.nan
    mapping = {
        "petrol": "Petrol",
        "gasoline": "Petrol",
        "diesel": "Diesel",
        "dsl": "Diesel",
        "electric": "Electric",
        "ev": "Electric",
        "hybrid": "Other",
        "cng": "Other",
    }
    return mapping.get(_normalize_space(str(text)), np.nan)


def map_transmission(value: Any) -> str | float:
    text = _normalize_string(value)
    if pd.isna(text):
        return np.nan
    mapping = {
        "manual": "Manual",
        "automatic": "Automatic",
        "auto": "Automatic",
    }
    return mapping.get(_normalize_space(str(text)), np.nan)


def map_owner_type(value: Any) -> str | float:
    text = _normalize_string(value)
    if pd.isna(text):
        return np.nan
    mapping = {
        "first": "First",
        "1st": "First",
        "first owner": "First",
        "second": "Second",
        "2nd": "Second",
        "second owner": "Second",
        "third": "Third",
        "3rd": "Third",
        "third owner": "Third",
    }
    return mapping.get(_normalize_space(str(text)), np.nan)


def map_condition(value: Any, weak_allowed: bool = False) -> str | float:
    text = _normalize_string(value)
    if pd.isna(text):
        return np.nan
    cleaned = re.sub(r"[_-]+", " ", str(text))
    cleaned = _normalize_space(cleaned)
    if weak_allowed:
        mapping = {"new": "New", "good": "Good", "weak": "Weak"}
    else:
        mapping = {"new": "New", "good": "Good", "worn out": "Worn Out"}
    return mapping.get(cleaned, np.nan)


def map_target(value: Any) -> float:
    text = _normalize_string(value)
    if pd.isna(text):
        return np.nan
    mapping = {
        "1": 1.0,
        "yes": 1.0,
        "true": 1.0,
        "0": 0.0,
        "no": 0.0,
        "false": 0.0,
    }
    return mapping.get(str(text).strip().lower(), np.nan)


def parse_numeric(value: Any, field: str | None = None) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return np.nan

    lowered = text.lower().strip()
    if lowered in NUMBER_WORDS:
        return float(NUMBER_WORDS[lowered])

    if re.fullmatch(r"-?\d+,\d+", lowered):
        lowered = lowered.replace(",", ".")

    has_liter_unit = bool(re.search(r"(?<![a-z])[0-9.]+\s*l\b", lowered)) or lowered.endswith("l")

    cleaned = lowered.replace("$", "").replace(",", "")
    cleaned = cleaned.replace("km/l", "").replace("km", "").replace("cc", "").strip()
    cleaned = cleaned.replace("l", "").strip()

    try:
        number = float(cleaned)
    except ValueError:
        return np.nan

    if field == "Engine_Size" and has_liter_unit and abs(number) <= 10:
        number *= 1000.0

    return number


def parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return pd.NaT

    for dayfirst in (False, True):
        try:
            parsed = parser.parse(text, dayfirst=dayfirst, fuzzy=False)
            if 2000 <= parsed.year <= 2030:
                return pd.Timestamp(parsed.date())
        except Exception:
            continue

    return pd.NaT


VALID_RANGES = {
    "Mileage": (0, 300000),
    "Reported_Issues": (0, 20),
    "Vehicle_Age": (0, 30),
    "Engine_Size": (500, 8000),
    "Odometer_Reading": (0, 500000),
    "Insurance_Premium": (0, 100000),
    "Service_History": (0, 30),
    "Accident_History": (0, 20),
    "Fuel_Efficiency": (5, 40),
}


def clean_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningAudit]:
    df = raw_df.copy()
    raw_rows = len(df)
    duplicate_rows_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    df["Vehicle_Model"] = df["Vehicle_Model"].apply(map_vehicle_model)
    df["Maintenance_History"] = df["Maintenance_History"].apply(map_maintenance_history)
    df["Fuel_Type"] = df["Fuel_Type"].apply(map_fuel_type)
    df["Transmission_Type"] = df["Transmission_Type"].apply(map_transmission)
    df["Owner_Type"] = df["Owner_Type"].apply(map_owner_type)
    df["Tire_Condition"] = df["Tire_Condition"].apply(lambda x: map_condition(x, weak_allowed=False))
    df["Brake_Condition"] = df["Brake_Condition"].apply(lambda x: map_condition(x, weak_allowed=False))
    df["Battery_Status"] = df["Battery_Status"].apply(lambda x: map_condition(x, weak_allowed=True))
    df["Need_Maintenance"] = df["Need_Maintenance"].apply(map_target)

    numeric_columns = [
        "Mileage",
        "Reported_Issues",
        "Vehicle_Age",
        "Engine_Size",
        "Odometer_Reading",
        "Insurance_Premium",
        "Service_History",
        "Accident_History",
        "Fuel_Efficiency",
    ]
    for column in numeric_columns:
        df[column] = df[column].apply(lambda x, c=column: parse_numeric(x, field=c))
        lower, upper = VALID_RANGES[column]
        df.loc[(df[column] < lower) | (df[column] > upper), column] = np.nan

    date_columns = ["Last_Service_Date", "Warranty_Expiry_Date"]
    for column in date_columns:
        df[column] = df[column].apply(parse_date)

    inconsistent_dates = (
        df["Warranty_Expiry_Date"].notna()
        & df["Last_Service_Date"].notna()
        & (df["Warranty_Expiry_Date"] < df["Last_Service_Date"])
    )
    inconsistent_count = int(inconsistent_dates.sum())
    df.loc[inconsistent_dates, "Warranty_Expiry_Date"] = pd.NaT

    missing_target_rows = int(df["Need_Maintenance"].isna().sum())
    df = df.dropna(subset=["Need_Maintenance"]).reset_index(drop=True)
    df["Need_Maintenance"] = df["Need_Maintenance"].astype(int)

    audit = CleaningAudit(
        raw_rows=raw_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        rows_after_dedup=raw_rows - duplicate_rows_removed,
        rows_with_missing_target_dropped=missing_target_rows,
        final_modeling_rows=len(df),
        date_inconsistencies_fixed=inconsistent_count,
    )
    return df, audit
