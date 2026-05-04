# Vehicle Maintenance Logistic Regression Results

## Cleaning audit

- Raw rows: 50,900
- Exact duplicates removed: 900
- Rows after deduplication: 50,000
- Rows dropped because the target label was missing: 180
- Date inconsistencies fixed (warranty earlier than service date): 239
- Final modeling rows: 49,820
- Feature-engineering reference date: 2025-01-15

## How the proposal was operationalized

1. The dirty CSV was cleaned by removing duplicates, standardizing missing values, normalizing categories, parsing mixed numeric text, parsing mixed-format dates, and dropping rows with missing target labels.
2. Feature engineering added the proposal's temporal and usage-intensity features: days since last service, days to warranty expiry, warranty-expired flag, mileage per year, issues per year, service frequency, accident burden, and condition score.
3. Logistic Regression was trained with one-hot encoding, median/mode imputation inside the pipeline, feature scaling for numeric columns, stratified splitting, and class_weight='balanced'.
4. Threshold tuning was done on a validation split instead of the test split to keep the test evaluation leakage-aware.

## Test-set results at tuned thresholds

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline cleaned LR | 0.39 | 0.9503 | 0.9899 | 0.9483 | 0.9687 | 0.9833 | 0.9962 | 0.0454 |
| Engineered full LR | 0.39 | 0.9507 | 0.9908 | 0.9479 | 0.9689 | 0.9833 | 0.9962 | 0.0453 |
| Leakage-aware LR | 0.20 | 0.7874 | 0.8416 | 0.9084 | 0.8737 | 0.8179 | 0.9541 | 0.1810 |

## Interpretation

- The engineered full model reached ROC-AUC **0.983** and F1 **0.969**, showing that Logistic Regression is a strong and interpretable baseline on this binary maintenance task.
- Feature engineering provided a small but positive uplift over the cleaned baseline model (F1 0.9687 -> 0.9689) while keeping the model simple.
- The leakage-aware model dropped to ROC-AUC **0.818** and F1 **0.874**. This is the most important research-gap result: condition-heavy variables inflate apparent performance, so the leakage-aware evaluation is more realistic for deployment.
- Threshold tuning materially improved the practical operating point. For the engineered full model, the selected threshold was **0.39** instead of 0.50, which improved the recall/precision balance for decision support.
- Probability outputs were converted into low/medium/high maintenance-risk bands and exported as CSV files for direct use in a fleet-prioritization workflow.

## Key files

- `outputs/all_model_metrics.csv` - detailed metrics for all models and thresholds
- `outputs/all_model_risk_bands.csv` - probability band summaries
- `outputs/engineered_full_lr/coefficients.csv` - interpretable logistic coefficients
- `outputs/engineered_full_lr/top_coefficients.png` - strongest coefficients plot
- `outputs/leakage_aware_lr/top_coefficients.png` - leakage-aware coefficient plot
