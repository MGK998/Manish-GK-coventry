# ==============================================================================
# 01 Setup And Dataset Loading
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: Intro and setup
# ############################################################################

# ==============================================================================
# Intro and setup
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 0 ---
# # Vehicle Maintenance Logistic Regression Walkthrough



# ############################################################################
# Included section: 0) Setup
# ############################################################################

# ==============================================================================
# 0) Setup
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 1 ---
# ## 0) Setup
# Imports the libraries, finds the dataset automatically, and creates a folder where outputs will be saved.

# --- Code cell 2 ---
from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from dateutil import parser
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
plt.rcParams["figure.dpi"] = 140


def resolve_paths(start: Path = Path.cwd()) -> tuple[Path, Path]:
    candidates = []
    for p in [start, *start.parents]:
        candidates.append(p)
        candidates.append(p / "vehicle_maintenance_logreg_project")

    candidates.extend(
        [
            Path(r"C:\Users\Dell\Desktop\vehicle_maintenance_logreg_project"),
            Path("/mnt/data/vehicle_maintenance_logreg_project"),
            Path("/mnt/data"),
        ]
    )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for root in unique_candidates:
        path1 = root / "data" / "vehicle_maintenance_data_unclean.csv"
        path2 = root / "vehicle_maintenance_data_unclean.csv"
        if path1.exists():
            return root, path1
        if path2.exists():
            return root, path2

    raise FileNotFoundError(
        "Could not find vehicle_maintenance_data_unclean.csv. "
        "Place the notebook inside the extracted project folder, "
        "or keep the CSV in a nearby data folder."
    )


PROJECT_DIR, DATA_PATH = resolve_paths()
OUTPUT_DIR = PROJECT_DIR / "notebook_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project directory:", PROJECT_DIR)
print("Dirty dataset path:", DATA_PATH)
print("Notebook output folder:", OUTPUT_DIR)



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
# Included section: 9.1) Experimental setup
# ############################################################################

# ==============================================================================
# 9.1) Experimental setup
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 42 ---
# ## 9.1) Experimental setup 
#
#  It summarizes:
#
# - **data pre-processing**
# - **feature extraction / engineering**
# - **feature selection**
# - **classification parameters**
# - **clustering parameters** (**not applicable** here because this study is a supervised **binary classification** problem, not a clustering task)
#  it documents *how* the Logistic Regression experiment was designed and validated.

# --- Code cell 43 ---
bundle_rationale = {
    "baseline_cleaned_lr": "Cleaned baseline variables only; no engineered temporal/usage features.",
    "engineered_full_lr": "All cleaned variables plus engineered temporal/usage features from the proposal.",
    "leakage_aware_lr": "Engineered model with condition-heavy variables removed to test leakage-aware realism.",
}

bundle_feature_summary = pd.DataFrame(
    [
        {
            "bundle": name,
            "raw_input_features": bundle.X.shape[1],
            "numeric_features": int(sum(pd.api.types.is_numeric_dtype(bundle.X[c]) for c in bundle.X.columns)),
            "categorical_features": int(sum(not pd.api.types.is_numeric_dtype(bundle.X[c]) for c in bundle.X.columns)),
            "rationale": bundle_rationale[name],
        }
        for name, bundle in feature_bundles.items()
    ]
)

preprocessing_summary = pd.DataFrame(
    [
        {
            "component": "Dirty-data cleaning",
            "implemented": "Yes",
            "details": "Removed exact duplicates; standardized missing-like markers; cleaned categorical labels; parsed mixed numeric strings; parsed mixed-format dates; fixed invalid date order; standardized Need_Maintenance target.",
        },
        {
            "component": "Missing-data handling",
            "implemented": "Yes",
            "details": "Median imputation for numeric features and most-frequent imputation for categorical features inside the ML pipeline.",
        },
        {
            "component": "Categorical encoding",
            "implemented": "Yes",
            "details": "OneHotEncoder(handle_unknown='ignore') for categorical variables.",
        },
        {
            "component": "Numeric scaling",
            "implemented": "Yes",
            "details": "StandardScaler() applied to numeric features for stable Logistic Regression optimization.",
        },
        {
            "component": "Data splitting",
            "implemented": "Yes",
            "details": "Stratified split into train/validation/test = 64% / 16% / 20% using train_test_split twice.",
        },
        {
            "component": "Class-imbalance handling",
            "implemented": "Yes",
            "details": "LogisticRegression(class_weight='balanced') on the training workflow.",
        },
        {
            "component": "Thresholding / output design",
            "implemented": "Yes",
            "details": "Validation-based threshold search from 0.20 to 0.80, plus Low/Medium/High risk bands (<0.40, 0.40-0.70, >0.70).",
        },
    ]
)

feature_engineering_summary = pd.DataFrame(
    [
        {"engineered_feature": "days_since_last_service", "source": "Last_Service_Date", "purpose": "Recency of maintenance"},
        {"engineered_feature": "days_to_warranty_expiry", "source": "Warranty_Expiry_Date", "purpose": "Remaining warranty window"},
        {"engineered_feature": "warranty_expired_flag", "source": "Warranty_Expiry_Date", "purpose": "Expired-vs-active warranty indicator"},
        {"engineered_feature": "mileage_per_year", "source": "Mileage / Vehicle_Age", "purpose": "Usage intensity"},
        {"engineered_feature": "issues_per_year", "source": "Reported_Issues / Vehicle_Age", "purpose": "Issue density"},
        {"engineered_feature": "service_per_year", "source": "Service_History / Vehicle_Age", "purpose": "Service frequency"},
        {"engineered_feature": "accidents_per_year", "source": "Accident_History / Vehicle_Age", "purpose": "Accident burden"},
        {"engineered_feature": "condition_score", "source": "Tire_Condition + Brake_Condition + Battery_Status", "purpose": "Compact aggregate condition measure"},
    ]
)

feature_selection_summary = bundle_feature_summary.rename(columns={"rationale": "selection_strategy"}).copy()
feature_selection_summary["selection_type"] = [
    "Baseline feature set",
    "Full engineered feature set",
    "Leakage-aware reduced feature set",
]
feature_selection_summary = feature_selection_summary[
    ["bundle", "selection_type", "raw_input_features", "numeric_features", "categorical_features", "selection_strategy"]
]

feature_selection_validation_summary = pd.DataFrame(
    [
        {
            "method": "Feature-bundle comparison",
            "section": "9.1",
            "purpose": "Compares baseline cleaned, engineered full, and leakage-aware feature sets.",
            "output": "experimental_feature_selection_summary.csv",
        },
        {
            "method": "L1 sparse Logistic Regression",
            "section": "9.2",
            "purpose": "Automatically shrinks weak transformed-feature coefficients to zero.",
            "output": "validation_l1_sparse_sweep.csv and validation_l1_selected_features.csv",
        },
        {
            "method": "Recursive Feature Elimination (RFE)",
            "section": "9.3",
            "purpose": "Applies RFE separately to baseline cleaned, engineered full, and leakage-aware bundles; then ranks the best RFE result by validation F1 and ROC-AUC.",
            "output": "validation_rfe_feature_selection_all_bundles.csv, validation_rfe_best_by_bundle.csv, validation_rfe_overall_best_bundle.csv, and validation_rfe_selected_features_by_bundle.csv",
        },
    ]
)

classification_parameter_summary = pd.DataFrame(
    [
        {"parameter": "Primary algorithm", "value": "LogisticRegression", "notes": "Binary classification target"},
        {"parameter": "Primary deployed solver", "value": "liblinear", "notes": "Stable for regularized Logistic Regression on tabular data"},
        {"parameter": "Primary deployed penalty", "value": "l2", "notes": "Default predictive model in the main pipeline"},
        {"parameter": "Primary deployed C", "value": "1.0", "notes": "Regularization strength used in the deployed predictive pipeline"},
        {"parameter": "Max iterations", "value": "3000 / 5000", "notes": "3000 in the main pipeline; 5000 in the RFE and stricter GridSearchCV cells to reduce convergence risk"},
        {"parameter": "Class imbalance handling", "value": "class_weight='balanced'", "notes": "Main workflow uses balanced class weights"},
        {"parameter": "Strict GridSearchCV parameter tuning", "value": "solver × penalty × C × class_weight", "notes": "Section 9.4 adds a formal grid over solver, penalty, regularization strength, and class weighting"},
        {"parameter": "GridSearchCV solvers", "value": "liblinear, lbfgs, saga", "notes": "Only valid solver/penalty combinations are included"},
        {"parameter": "GridSearchCV penalties", "value": "l1, l2", "notes": "l1 is used for sparse selection-capable models; l2 is used for standard regularized LR"},
        {"parameter": "GridSearchCV C values", "value": "0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00", "notes": "C controls inverse regularization strength"},
        {"parameter": "GridSearchCV class_weight values", "value": "None, balanced", "notes": "Compares unweighted LR against imbalance-aware LR"},
        {"parameter": "Train / validation / test split", "value": "64% / 16% / 20%", "notes": "All splits are stratified"},
        {"parameter": "Threshold search", "value": "0.20 to 0.80 (step 0.005)", "notes": "Best threshold chosen on validation set by F1, then applied to test set"},
        {"parameter": "Risk bands", "value": "Low <0.40, Medium 0.40-0.70, High >0.70", "notes": "Probability-based output design from the proposal"},
        {"parameter": "Evaluation metrics", "value": "Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier, Confusion Matrix", "notes": "Plus odds ratios, goodness-of-fit, RFE validation metrics, and GridSearchCV CV metrics"},
    ]
)

clustering_parameter_summary = pd.DataFrame(
    [
        {
            "item": "Clustering algorithm / parameters",
            "status": "Not applicable",
            "reason": "This project is a supervised binary-classification study using Logistic Regression; no clustering model is part of the proposed methodology.",
        }
    ]
)

experimental_setup_checklist = pd.DataFrame(
    [
        {"rubric_item": "Data pre-processing", "implemented": "Yes", "evidence_in_notebook": "Sections 3, 4, 5, and 8"},
        {"rubric_item": "Feature extraction / engineering", "implemented": "Yes", "evidence_in_notebook": "Sections 6 and 7"},
        {"rubric_item": "Feature selection", "implemented": "Yes", "evidence_in_notebook": "Section 9.1 tables + Section 9.2 L1 sparse validation + Section 9.3 RFE comparison across all three feature bundles"},
        {"rubric_item": "Classification parameters", "implemented": "Yes", "evidence_in_notebook": "Section 8 model code + Section 9.1 parameter table + Sections 9.2 and 9.4 validation sweeps"},
        {"rubric_item": "Clustering parameters", "implemented": "N/A", "evidence_in_notebook": "Section 9.1 explicitly marks clustering as not applicable"},
    ]
)

display(Markdown("### Experimental setup checklist"))
display(experimental_setup_checklist)

display(Markdown("### Data pre-processing summary"))
display(preprocessing_summary)

display(Markdown("### Feature extraction / engineering summary"))
display(feature_engineering_summary)

display(Markdown("### Feature selection summary across model variants"))
display(feature_selection_summary)

display(Markdown("### Formal feature-selection validation methods"))
display(feature_selection_validation_summary)

display(Markdown("### Classification parameters"))
display(classification_parameter_summary)

display(Markdown("### Clustering parameters"))
display(clustering_parameter_summary)

experimental_setup_checklist.to_csv(OUTPUT_DIR / "experimental_setup_checklist.csv", index=False)
preprocessing_summary.to_csv(OUTPUT_DIR / "experimental_preprocessing_summary.csv", index=False)
feature_engineering_summary.to_csv(OUTPUT_DIR / "experimental_feature_engineering_summary.csv", index=False)
feature_selection_summary.to_csv(OUTPUT_DIR / "experimental_feature_selection_summary.csv", index=False)
feature_selection_validation_summary.to_csv(OUTPUT_DIR / "experimental_feature_selection_validation_methods.csv", index=False)
classification_parameter_summary.to_csv(OUTPUT_DIR / "experimental_classification_parameters.csv", index=False)
clustering_parameter_summary.to_csv(OUTPUT_DIR / "experimental_clustering_parameters.csv", index=False)

