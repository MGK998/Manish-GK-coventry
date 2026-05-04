
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_NUMERIC = [
    "Mileage",
    "Reported_Issues",
    "Vehicle_Age",
    "Engine_Size",
    "Odometer_Reading",
    "Insurance_Premium",
    "Service_History",
    "Accident_History",
    "Fuel_Efficiency",
    "days_since_last_service",
    "days_to_warranty_expiry",
]

BASE_CATEGORICAL = [
    "Vehicle_Model",
    "Maintenance_History",
    "Fuel_Type",
    "Transmission_Type",
    "Owner_Type",
    "Tire_Condition",
    "Brake_Condition",
    "Battery_Status",
]

ENGINEERED_NUMERIC = [
    "warranty_expired_flag",
    "mileage_per_year",
    "issues_per_year",
    "service_per_year",
    "accidents_per_year",
    "condition_score",
]


@dataclass
class FeatureBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series


def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    engineered = df.copy()
    reference_date = engineered["Last_Service_Date"].max()

    engineered["days_since_last_service"] = (reference_date - engineered["Last_Service_Date"]).dt.days
    engineered["days_to_warranty_expiry"] = (engineered["Warranty_Expiry_Date"] - reference_date).dt.days
    engineered["warranty_expired_flag"] = (engineered["days_to_warranty_expiry"] < 0).astype(float)

    vehicle_age = engineered["Vehicle_Age"].replace(0, np.nan)
    engineered["mileage_per_year"] = engineered["Mileage"] / vehicle_age
    engineered["issues_per_year"] = engineered["Reported_Issues"] / vehicle_age
    engineered["service_per_year"] = engineered["Service_History"] / vehicle_age
    engineered["accidents_per_year"] = engineered["Accident_History"] / vehicle_age

    tire_map = {"New": 0, "Good": 1, "Worn Out": 2}
    brake_map = {"New": 0, "Good": 1, "Worn Out": 2}
    battery_map = {"New": 0, "Good": 1, "Weak": 2}
    engineered["tire_condition_score"] = engineered["Tire_Condition"].map(tire_map)
    engineered["brake_condition_score"] = engineered["Brake_Condition"].map(brake_map)
    engineered["battery_condition_score"] = engineered["Battery_Status"].map(battery_map)
    engineered["condition_score"] = engineered[
        ["tire_condition_score", "brake_condition_score", "battery_condition_score"]
    ].sum(axis=1, min_count=1)

    return engineered, reference_date


def build_feature_bundles(df: pd.DataFrame) -> tuple[dict[str, FeatureBundle], pd.Timestamp]:
    engineered_df, reference_date = add_engineered_features(df)
    target = "Need_Maintenance"

    baseline_columns = BASE_NUMERIC + BASE_CATEGORICAL
    full_columns = BASE_NUMERIC + ENGINEERED_NUMERIC + BASE_CATEGORICAL
    leakage_columns = [
        column
        for column in full_columns
        if column not in {"Tire_Condition", "Brake_Condition", "Battery_Status", "condition_score"}
    ]

    bundles = {
        "baseline_cleaned_lr": FeatureBundle(
            name="baseline_cleaned_lr",
            X=engineered_df[baseline_columns].copy(),
            y=engineered_df[target].copy(),
        ),
        "engineered_full_lr": FeatureBundle(
            name="engineered_full_lr",
            X=engineered_df[full_columns].copy(),
            y=engineered_df[target].copy(),
        ),
        "leakage_aware_lr": FeatureBundle(
            name="leakage_aware_lr",
            X=engineered_df[leakage_columns].copy(),
            y=engineered_df[target].copy(),
        ),
    }
    return bundles, reference_date
