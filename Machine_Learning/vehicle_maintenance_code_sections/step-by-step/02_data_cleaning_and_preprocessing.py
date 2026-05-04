# ==============================================================================
# 02 Data Cleaning And Preprocessing
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 1) Load the dirty dataset
# ############################################################################

# ==============================================================================
# 1) Load the dirty dataset
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 3 ---
# ## 1) Load the dirty dataset
# Unclean dataset with duplicates, missing-like markers, inconsistent categories, mixed numeric formats, invalid dates, and outliers.

# --- Code cell 4 ---
raw_df = pd.read_csv(DATA_PATH)

print("Raw shape:", raw_df.shape)
display(raw_df.head(10))
display(raw_df.dtypes.rename("raw_dtype").to_frame().T)



# ############################################################################
# Included section: 2) Quick dirty-data audit
# ############################################################################

# ==============================================================================
# 2) Quick dirty-data audit
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 5 ---
# ## 2) Quick dirty-data audit
# for **auditing** the dirty dataset before we clean it.

# --- Code cell 6 ---
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

categorical_cols = [
    "Vehicle_Model",
    "Maintenance_History",
    "Fuel_Type",
    "Transmission_Type",
    "Owner_Type",
    "Tire_Condition",
    "Brake_Condition",
    "Battery_Status",
    "Need_Maintenance",
]

numeric_cols = [
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

date_cols = ["Last_Service_Date", "Warranty_Expiry_Date"]


def is_missing_like(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_MARKERS


def preview_top_labels(df: pd.DataFrame, column: str, top_n: int = 12) -> pd.DataFrame:
    return (
        df[column]
        .astype(str)
        .value_counts(dropna=False)
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )



# ############################################################################
# Included section: 2.1 Dirty dataset overview
# ############################################################################

# ==============================================================================
# 2.1 Dirty dataset overview
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 7 ---
# ### 2.1 Dirty dataset overview

# --- Code cell 8 ---
dirty_overview = pd.DataFrame(
    [
        {"metric": "Rows", "value": len(raw_df)},
        {"metric": "Columns", "value": raw_df.shape[1]},
        {"metric": "Exact duplicate rows", "value": int(raw_df.duplicated().sum())},
        {"metric": "Rows with missing-like target labels", "value": int(raw_df["Need_Maintenance"].apply(is_missing_like).sum())},
    ]
)
display(dirty_overview)



# ############################################################################
# Included section: 2.2 Missing-like values in the dirty dataset
# ############################################################################

# ==============================================================================
# 2.2 Missing-like values in the dirty dataset
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 9 ---
# ### 2.2 Missing-like values in the dirty dataset

# --- Code cell 10 ---
dirty_missing = pd.DataFrame(
    {
        "column": raw_df.columns,
        "dirty_missing_like_count": [int(raw_df[c].apply(is_missing_like).sum()) for c in raw_df.columns],
    }
).sort_values("dirty_missing_like_count", ascending=False)

display(dirty_missing)

plot_df = dirty_missing[dirty_missing["dirty_missing_like_count"] > 0].copy()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(plot_df["column"], plot_df["dirty_missing_like_count"])
ax.set_title("Missing-like values in the dirty dataset")
ax.set_ylabel("Count")
ax.set_xticklabels(plot_df["column"], rotation=70, ha="right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "dirty_missing_like_counts.png", bbox_inches="tight")
plt.show()



# ############################################################################
# Included section: 2.3 Examples of messy labels before cleaning
# ############################################################################

# ==============================================================================
# 2.3 Examples of messy labels before cleaning
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 11 ---
# ### 2.3 Examples of messy labels before cleaning
# The next output makes the problem visible. Notice mixed casing, spelling variants, and target labels such as `yes`, `True`, and `0/1`.

# --- Code cell 12 ---
for col in ["Fuel_Type", "Transmission_Type", "Vehicle_Model", "Need_Maintenance"]:
    print(f"\nTop raw labels for {col}")
    display(preview_top_labels(raw_df, col))



# ############################################################################
# Included section: 3) Complete cleaning code
# ############################################################################

# ==============================================================================
# 3) Complete cleaning code
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 13 ---
# ## 3) Complete cleaning code
#  **full dataset cleaning logic** for this project

# --- Code cell 14 ---
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



# ############################################################################
# Included section: 4) Apply the cleaning pipeline
# ############################################################################

# ==============================================================================
# 4) Apply the cleaning pipeline
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 15 ---
# ## 4) Apply the cleaning pipeline
# This step removes duplicates, standardizes missing values, normalizes categories, parses numbers and dates, fixes invalid date logic, and drops rows whose target label cannot be recovered.

# --- Code cell 16 ---
cleaned_df, cleaning_audit = clean_dataset(raw_df)

display(pd.DataFrame([cleaning_audit.as_dict()]))
print("Cleaned shape:", cleaned_df.shape)
display(cleaned_df.head(10))



# ############################################################################
# Included section: 5) Compare the dirty and cleaned datasets
# ############################################################################

# ==============================================================================
# 5) Compare the dirty and cleaned datasets
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 17 ---
# ## 5) Compare the dirty and cleaned datasets
#  **What changed after cleaning?**

# --- Code cell 18 ---
def count_raw_numeric_issues(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        parsed = df[col].apply(lambda x, c=col: parse_numeric(x, field=c))
        lower, upper = VALID_RANGES[col]
        rows.append(
            {
                "column": col,
                "dirty_missing_or_unparsed": int(parsed.isna().sum()),
                "dirty_out_of_range": int(((parsed < lower) | (parsed > upper)).fillna(False).sum()),
                "clean_missing_after_rules": int(cleaned_df[col].isna().sum()),
                "clean_out_of_range": int(((cleaned_df[col] < lower) | (cleaned_df[col] > upper)).fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def count_date_issues(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    parsed_dates = {}
    rows = []
    for col in date_cols:
        parsed = df[col].apply(parse_date)
        parsed_dates[col] = parsed
        rows.append(
            {
                "column": col,
                "dirty_missing_or_invalid": int(parsed.isna().sum()),
                "clean_missing_after_rules": int(cleaned_df[col].isna().sum()),
            }
        )

    dirty_inconsistent = int(
        (
            parsed_dates["Warranty_Expiry_Date"].notna()
            & parsed_dates["Last_Service_Date"].notna()
            & (parsed_dates["Warranty_Expiry_Date"] < parsed_dates["Last_Service_Date"])
        ).sum()
    )
    clean_inconsistent = int(
        (
            cleaned_df["Warranty_Expiry_Date"].notna()
            & cleaned_df["Last_Service_Date"].notna()
            & (cleaned_df["Warranty_Expiry_Date"] < cleaned_df["Last_Service_Date"])
        ).sum()
    )
    return pd.DataFrame(rows), dirty_inconsistent, clean_inconsistent



# ############################################################################
# Included section: 5.2 Unique category levels before and after cleaning
# ############################################################################

# ==============================================================================
# 5.2 Unique category levels before and after cleaning
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 21 ---
# ### 5.2 Unique category levels before and after cleaning
# A successful cleaning step should collapse messy labels into a much smaller, valid label set.

# --- Code cell 22 ---
category_compare = pd.DataFrame(
    {
        "column": categorical_cols,
        "dirty_unique_labels": [raw_df[c].astype(str).nunique(dropna=True) for c in categorical_cols],
        "clean_unique_labels": [cleaned_df[c].nunique(dropna=True) for c in categorical_cols],
    }
).sort_values("dirty_unique_labels", ascending=False)

display(category_compare)
category_compare.to_csv(OUTPUT_DIR / "category_unique_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(category_compare))
width = 0.38
ax.bar(x - width / 2, category_compare["dirty_unique_labels"], width=width, label="Dirty")
ax.bar(x + width / 2, category_compare["clean_unique_labels"], width=width, label="Cleaned")
ax.set_xticks(x, category_compare["column"], rotation=70, ha="right")
ax.set_ylabel("Unique labels")
ax.set_title("Category cleanup: unique labels before vs after")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "category_unique_before_after.png", bbox_inches="tight")
plt.show()



# ############################################################################
# Included section: 5.3 Numeric quality before and after cleaning
# ############################################################################

# ==============================================================================
# 5.3 Numeric quality before and after cleaning
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 23 ---
# ### 5.3 Numeric quality before and after cleaning
# This shows how many numeric values were either unreadable or outside the valid range before cleaning, versus after cleaning.

# --- Code cell 24 ---
numeric_quality = count_raw_numeric_issues(raw_df)
display(numeric_quality)
numeric_quality.to_csv(OUTPUT_DIR / "numeric_quality_comparison.csv", index=False)

plot_numeric = numeric_quality.set_index("column")[["dirty_out_of_range", "clean_out_of_range"]]

fig, ax = plt.subplots(figsize=(10, 5))
plot_numeric.plot(kind="bar", ax=ax)
ax.set_title("Out-of-range numeric values before vs after cleaning")
ax.set_ylabel("Count")
ax.set_xticklabels(plot_numeric.index, rotation=70, ha="right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "numeric_out_of_range_before_after.png", bbox_inches="tight")
plt.show()



# ############################################################################
# Included section: 5.4 Date quality before and after cleaning
# ############################################################################

# ==============================================================================
# 5.4 Date quality before and after cleaning
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 25 ---
# ### 5.4 Date quality before and after cleaning

# --- Code cell 26 ---
display(date_quality_df)
date_quality_df.to_csv(OUTPUT_DIR / "date_quality_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
date_plot = date_quality_df.set_index("column")
date_plot.plot(kind="bar", ax=ax)
ax.set_title("Date parsing issues before vs after cleaning")
ax.set_ylabel("Count")
ax.set_xticklabels(date_plot.index, rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "date_quality_before_after.png", bbox_inches="tight")
plt.show()



# ############################################################################
# Included section: 5.5 category labels before and after cleaning
# ############################################################################

# ==============================================================================
# 5.5 category labels before and after cleaning
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 27 ---
# ### 5.5 category labels before and after cleaning

# --- Code cell 28 ---
for col in ["Fuel_Type", "Transmission_Type", "Vehicle_Model", "Need_Maintenance"]:
    print(f"\n{col} BEFORE cleaning")
    display(preview_top_labels(raw_df, col))
    print(f"{col} AFTER cleaning")
    display(cleaned_df[col].astype(str).value_counts(dropna=False).head(12).rename_axis(col).reset_index(name="count"))



# ############################################################################
# Included section: 5.6 raw vs cleaned numeric distribution
# ############################################################################

# ==============================================================================
# 5.6 raw vs cleaned numeric distribution
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 29 ---
# ### 5.6 raw vs cleaned numeric distribution
# The raw data contains messy formatting and outliers. After cleaning, the distribution becomes more realistic.

# --- Code cell 30 ---
raw_mileage_parsed = raw_df["Mileage"].apply(lambda x: parse_numeric(x, field="Mileage"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(raw_mileage_parsed.dropna(), bins=40)
axes[0].set_title("Dirty mileage after loose parsing")
axes[0].set_xlabel("Mileage")
axes[0].set_ylabel("Frequency")

axes[1].hist(cleaned_df["Mileage"].dropna(), bins=40)
axes[1].set_title("Cleaned mileage")
axes[1].set_xlabel("Mileage")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "dirty_vs_cleaned_mileage.png", bbox_inches="tight")
plt.show()

# --- Code cell 31 ---
display(Markdown(
    f'''
### What changes can we see after cleaning?

- **{cleaning_audit.duplicate_rows_removed:,} exact duplicates** were removed.
- **{cleaning_audit.rows_with_missing_target_dropped:,} rows** were dropped because the target label could not be recovered.
- Invalid target-label variants were standardized into just **2 valid classes**: 0 and 1.
- Category explosion was reduced. For example:
  - Fuel type: **{int(category_compare.loc[category_compare["column"]=="Fuel_Type", "dirty_unique_labels"].iloc[0])} -> {int(category_compare.loc[category_compare["column"]=="Fuel_Type", "clean_unique_labels"].iloc[0])}**
  - Transmission type: **{int(category_compare.loc[category_compare["column"]=="Transmission_Type", "dirty_unique_labels"].iloc[0])} -> {int(category_compare.loc[category_compare["column"]=="Transmission_Type", "clean_unique_labels"].iloc[0])}**
  - Vehicle model: **{int(category_compare.loc[category_compare["column"]=="Vehicle_Model", "dirty_unique_labels"].iloc[0])} -> {int(category_compare.loc[category_compare["column"]=="Vehicle_Model", "clean_unique_labels"].iloc[0])}**
- Numeric columns no longer contain impossible negative values or extreme out-of-range values after the cleaning rules.
- Date logic improved: records with warranty dates earlier than the last service date were fixed by setting the invalid warranty date to missing.
'''
))

