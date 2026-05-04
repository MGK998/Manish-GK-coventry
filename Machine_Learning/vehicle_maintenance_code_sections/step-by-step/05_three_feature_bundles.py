# ==============================================================================
# 05 Three Feature Bundles
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 9) Build the three model variants
# ############################################################################

# ==============================================================================
# 9) Build the three model variants
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 40 ---
# ## 9) Build the three model variants
# - **baseline_cleaned_lr**: cleaned data, no extra engineered features  
# - **engineered_full_lr**: cleaned data + engineered temporal/usage features  
# - **leakage_aware_lr**: engineered data but with condition-heavy variables removed

# --- Code cell 41 ---
feature_bundles, _ = build_feature_bundles(cleaned_df)

results = {}
for name, bundle in feature_bundles.items():
    print(f"Training {name} ...")
    results[name] = run_bundle(name, bundle.X, bundle.y)

model_summary = pd.DataFrame(
    [
        {
            "model": name,
            "threshold": res["tuned_metrics"]["threshold"],
            "accuracy": res["tuned_metrics"]["accuracy"],
            "precision": res["tuned_metrics"]["precision"],
            "recall": res["tuned_metrics"]["recall"],
            "f1": res["tuned_metrics"]["f1"],
            "roc_auc": res["tuned_metrics"]["roc_auc"],
            "pr_auc": res["tuned_metrics"]["pr_auc"],
            "brier_score": res["tuned_metrics"]["brier_score"],
        }
        for name, res in results.items()
    ]
).sort_values(["f1", "roc_auc"], ascending=False)

display(model_summary.round(4))
model_summary.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)

