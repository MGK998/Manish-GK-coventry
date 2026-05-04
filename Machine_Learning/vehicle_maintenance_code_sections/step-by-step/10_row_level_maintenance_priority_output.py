# ==============================================================================
# 10 Row Level Maintenance Priority Output
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 15) Risk-band output and per-row maintenance priority
# ############################################################################

# ==============================================================================
# 15) Risk-band output and per-row maintenance priority
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 60 ---
# ## 15) Risk-band output and per-row maintenance priority
# The proposal asked for probability-based output that can be translated into **Low / Medium / High** maintenance priority instead of using only a hard yes/no decision.
#
# This section now produces two outputs:
#
# 1. the original **risk-band summary** table, which counts vehicles in Low / Medium / High bands; and
# 2. a new **row-level engineered_full_lr maintenance-priority table**, where every row in the cleaned modeling dataset receives:
#    - predicted maintenance probability
#    - Low / Medium / High `maintenance_priority`
#    - tuned yes/no prediction using the engineered-full Logistic Regression threshold
#    - train / validation / test split marker
#
# The row-level table is saved as `engineered_full_maintenance_priority_by_row.csv`.

# --- Code cell 61 ---
# Keep the aggregate risk-band summary for all three model variants.
all_risk_bands = pd.concat([res["risk_bands"] for res in results.values()], ignore_index=True)
display(Markdown("#### Held-out test-set risk-band summary for all model variants"))
display(all_risk_bands.round(4))
all_risk_bands.to_csv(OUTPUT_DIR / "all_model_risk_bands_notebook.csv", index=False)

# Focus on engineered_full_lr because this was the best-performing deployed feature bundle.
full_risk = results["engineered_full_lr"]["risk_bands"].copy()
display(Markdown("#### Held-out test-set risk-band summary for engineered_full_lr"))
display(full_risk.round(4))
full_risk.to_csv(OUTPUT_DIR / "engineered_full_risk_band_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.bar(full_risk["band"].astype(str), full_risk["vehicles"])
ax.set_title("Engineered full LR: vehicles in each risk band")
ax.set_ylabel("Vehicles")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "engineered_full_risk_bands.png", bbox_inches="tight")
plt.show()

# New requirement: score every row of the cleaned modeling dataset with engineered_full_lr
# and store the Low / Medium / High band as a row-level maintenance priority.
engineered_full_bundle = feature_bundles["engineered_full_lr"]
engineered_full_result = results["engineered_full_lr"]
engineered_full_pipeline = engineered_full_result["pipeline"]
engineered_full_threshold = engineered_full_result["threshold_info"].threshold

# Recreate the deterministic split used earlier so the row-level file shows whether a row was
# part of the train, validation, or test subset. This is for traceability only; the test-set
# metrics above remain the unbiased evaluation output.
X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full = split_data(
    engineered_full_bundle.X,
    engineered_full_bundle.y,
    random_state=42,
)

split_labels = pd.Series(index=engineered_full_bundle.X.index, dtype="object")
split_labels.loc[X_train_full.index] = "train"
split_labels.loc[X_val_full.index] = "validation"
split_labels.loc[X_test_full.index] = "test"

# Predict maintenance probability for every cleaned modeling row using the fitted engineered-full pipeline.
# This creates the practical priority output requested by the proposal.
row_probabilities = engineered_full_pipeline.predict_proba(engineered_full_bundle.X)[:, 1]
row_priority = engineered_df.loc[engineered_full_bundle.X.index].copy()

row_priority.insert(0, "dataset_row_id", np.arange(1, len(row_priority) + 1))
row_priority.insert(1, "data_split", split_labels.to_numpy())
row_priority.insert(2, "engineered_full_lr_probability", row_probabilities)
row_priority.insert(3, "maintenance_priority", pd.cut(row_probabilities, bins=RISK_BINS, labels=RISK_LABELS).astype(str))
row_priority.insert(4, "engineered_full_lr_threshold", engineered_full_threshold)
row_priority.insert(
    5,
    "engineered_full_lr_predicted_need_maintenance",
    (row_probabilities >= engineered_full_threshold).astype(int),
)
row_priority.insert(6, "actual_need_maintenance", engineered_full_bundle.y.astype(int).to_numpy())

# A compact preview is displayed in the notebook; the saved CSV contains all available cleaned/engineered columns.
priority_preview_columns = [
    "dataset_row_id",
    "data_split",
    "engineered_full_lr_probability",
    "maintenance_priority",
    "engineered_full_lr_predicted_need_maintenance",
    "actual_need_maintenance",
    "Vehicle_Model",
    "Vehicle_Age",
    "Mileage",
    "Reported_Issues",
    "Service_History",
    "Accident_History",
    "days_since_last_service",
    "days_to_warranty_expiry",
    "condition_score",
]
priority_preview_columns = [c for c in priority_preview_columns if c in row_priority.columns]

# Summary confirms that every cleaned modeling row received exactly one priority band.
row_priority_summary = (
    row_priority.groupby("maintenance_priority", observed=False)
    .agg(
        vehicles=("dataset_row_id", "size"),
        avg_probability=("engineered_full_lr_probability", "mean"),
        actual_positive_rate=("actual_need_maintenance", "mean"),
    )
    .reindex(RISK_LABELS)
    .reset_index()
)

row_priority_sorted = row_priority.sort_values(
    ["engineered_full_lr_probability", "dataset_row_id"],
    ascending=[False, True],
).reset_index(drop=True)

display(Markdown("#### Engineered_full_lr maintenance priority preview: top 20 highest-probability rows"))
display(row_priority_sorted[priority_preview_columns].head(20).round(4))

display(Markdown("#### Row-level maintenance-priority distribution"))
display(row_priority_summary.round(4))

row_priority.to_csv(OUTPUT_DIR / "engineered_full_maintenance_priority_by_row.csv", index=False)
row_priority_sorted.to_csv(OUTPUT_DIR / "engineered_full_maintenance_priority_ranked.csv", index=False)
row_priority_summary.to_csv(OUTPUT_DIR / "engineered_full_maintenance_priority_summary.csv", index=False)

print(
    f"Saved {len(row_priority):,} row-level engineered_full_lr maintenance-priority predictions to "
    f"{OUTPUT_DIR / 'engineered_full_maintenance_priority_by_row.csv'}"
)

