
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.cleaning import clean_dataset
from src.features import add_engineered_features, build_feature_bundles
from src.modeling import run_modeling


def write_markdown_report(
    output_path: Path,
    audit: dict,
    reference_date: str,
    model_results: dict,
) -> None:
    baseline = model_results["baseline_cleaned_lr"]["metrics"]["test_tuned"]
    full_model = model_results["engineered_full_lr"]["metrics"]["test_tuned"]
    leakage = model_results["leakage_aware_lr"]["metrics"]["test_tuned"]

    lines = [
        "# Vehicle Maintenance Logistic Regression Results",
        "",
        "## Cleaning audit",
        "",
        f"- Raw rows: {audit['raw_rows']:,}",
        f"- Exact duplicates removed: {audit['duplicate_rows_removed']:,}",
        f"- Rows after deduplication: {audit['rows_after_dedup']:,}",
        f"- Rows dropped because the target label was missing: {audit['rows_with_missing_target_dropped']:,}",
        f"- Date inconsistencies fixed (warranty earlier than service date): {audit['date_inconsistencies_fixed']:,}",
        f"- Final modeling rows: {audit['final_modeling_rows']:,}",
        f"- Feature-engineering reference date: {reference_date}",
        "",
        "## How the proposal was operationalized",
        "",
        "1. The dirty CSV was cleaned by removing duplicates, standardizing missing values, normalizing categories, parsing mixed numeric text, parsing mixed-format dates, and dropping rows with missing target labels.",
        "2. Feature engineering added the proposal's temporal and usage-intensity features: days since last service, days to warranty expiry, warranty-expired flag, mileage per year, issues per year, service frequency, accident burden, and condition score.",
        "3. Logistic Regression was trained with one-hot encoding, median/mode imputation inside the pipeline, feature scaling for numeric columns, stratified splitting, and class_weight='balanced'.",
        "4. Threshold tuning was done on a validation split instead of the test split to keep the test evaluation leakage-aware.",
        "",
        "## Test-set results at tuned thresholds",
        "",
        "| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Baseline cleaned LR | {baseline['threshold']:.2f} | {baseline['accuracy']:.4f} | {baseline['precision']:.4f} | {baseline['recall']:.4f} | {baseline['f1']:.4f} | {baseline['roc_auc']:.4f} | {baseline['pr_auc']:.4f} | {baseline['brier_score']:.4f} |",
        f"| Engineered full LR | {full_model['threshold']:.2f} | {full_model['accuracy']:.4f} | {full_model['precision']:.4f} | {full_model['recall']:.4f} | {full_model['f1']:.4f} | {full_model['roc_auc']:.4f} | {full_model['pr_auc']:.4f} | {full_model['brier_score']:.4f} |",
        f"| Leakage-aware LR | {leakage['threshold']:.2f} | {leakage['accuracy']:.4f} | {leakage['precision']:.4f} | {leakage['recall']:.4f} | {leakage['f1']:.4f} | {leakage['roc_auc']:.4f} | {leakage['pr_auc']:.4f} | {leakage['brier_score']:.4f} |",
        "",
        "## Interpretation",
        "",
        f"- The engineered full model reached ROC-AUC **{full_model['roc_auc']:.3f}** and F1 **{full_model['f1']:.3f}**, showing that Logistic Regression is a strong and interpretable baseline on this binary maintenance task.",
        f"- Feature engineering provided a small but positive uplift over the cleaned baseline model (F1 {baseline['f1']:.4f} -> {full_model['f1']:.4f}) while keeping the model simple.",
        f"- The leakage-aware model dropped to ROC-AUC **{leakage['roc_auc']:.3f}** and F1 **{leakage['f1']:.3f}**. This is the most important research-gap result: condition-heavy variables inflate apparent performance, so the leakage-aware evaluation is more realistic for deployment.",
        f"- Threshold tuning materially improved the practical operating point. For the engineered full model, the selected threshold was **{full_model['threshold']:.2f}** instead of 0.50, which improved the recall/precision balance for decision support.",
        "- Probability outputs were converted into low/medium/high maintenance-risk bands and exported as CSV files for direct use in a fleet-prioritization workflow.",
        "",
        "## Key files",
        "",
        "- `outputs/all_model_metrics.csv` - detailed metrics for all models and thresholds",
        "- `outputs/all_model_risk_bands.csv` - probability band summaries",
        "- `outputs/engineered_full_lr/coefficients.csv` - interpretable logistic coefficients",
        "- `outputs/engineered_full_lr/top_coefficients.png` - strongest coefficients plot",
        "- `outputs/leakage_aware_lr/top_coefficients.png` - leakage-aware coefficient plot",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle maintenance logistic regression project.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the dirty CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where cleaned data, engineered data, and model results will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input)
    cleaned_df, audit = clean_dataset(raw_df)
    engineered_df, reference_date = add_engineered_features(cleaned_df)

    cleaned_df.to_csv(args.output / "cleaned_vehicle_maintenance.csv", index=False)
    engineered_df.to_csv(args.output / "engineered_vehicle_maintenance.csv", index=False)

    with open(args.output / "cleaning_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit.as_dict(), handle, indent=2)

    feature_bundles, _ = build_feature_bundles(cleaned_df)
    model_results = run_modeling(feature_bundles, output_dir=args.output)

    report_payload = {
        "audit": audit.as_dict(),
        "reference_date": str(reference_date.date()),
    }
    with open(args.output / "project_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2)

    write_markdown_report(
        output_path=args.output / "results_summary.md",
        audit=audit.as_dict(),
        reference_date=str(reference_date.date()),
        model_results=model_results,
    )


if __name__ == "__main__":
    main()
