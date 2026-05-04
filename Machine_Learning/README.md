
# Vehicle Maintenance Risk Prediction Using Logistic Regression

This project implements the revised proposal on the uploaded dirty vehicle-maintenance dataset.

## What it does

- cleans the dirty CSV
- standardizes missing values and categories
- converts mixed-format numeric fields back to numeric
- parses mixed-format dates
- removes exact duplicates
- drops rows where the target label is missing
- engineers the proposal's temporal and usage-intensity features
- trains three Logistic Regression variants:
  - `baseline_cleaned_lr`
  - `engineered_full_lr`
  - `leakage_aware_lr`
- tunes the classification threshold on a validation split
- exports metrics, risk bands, plots, and coefficient tables

## Project structure

- `main.py` - end-to-end runner
- `src/cleaning.py` - data cleaning and audit logic
- `src/features.py` - feature engineering and model feature sets
- `src/modeling.py` - preprocessing pipeline, training, threshold tuning, evaluation, and plots
- `data/vehicle_maintenance_data_unclean.csv` - dirty practice dataset
- `outputs/` - generated cleaned data, engineered data, metrics, plots, and summaries

## Virtual environment setup

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --input data/vehicle_maintenance_data_unclean.csv --output outputs
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --input data\vehicle_maintenance_data_unclean.csv --output outputs
```

## Main feature engineering used

The code implements the features recommended by the revised proposal:

- `days_since_last_service`
- `days_to_warranty_expiry`
- `warranty_expired_flag`
- `mileage_per_year`
- `issues_per_year`
- `service_per_year`
- `accidents_per_year`
- `condition_score`

## Modeling notes

- Logistic Regression uses `class_weight='balanced'`
- splitting is stratified
- threshold tuning happens on the validation split, not the test split
- missing-value imputation is done **inside** the pipeline to avoid train/test leakage
- categorical variables use one-hot encoding
- numeric variables are standardized

## Key output files

After running the project, check these first:

- `outputs/results_summary.md`
- `outputs/test_tuned_summary.csv`
- `outputs/all_model_metrics.csv`
- `outputs/all_model_risk_bands.csv`
- `outputs/engineered_full_lr/top_coefficients.png`
- `outputs/leakage_aware_lr/top_coefficients.png`

## Research-gap mapping

This implementation addresses the revised proposal's gaps by:

1. explicitly evaluating Logistic Regression on the binary target
2. generating probability-based risk bands instead of only hard labels
3. adding temporal and usage-based feature engineering
4. reporting precision, recall, F1, ROC-AUC, PR-AUC, Brier score, and confusion-matrix counts
5. comparing a full model against a leakage-aware model to expose the effect of condition-heavy variables
