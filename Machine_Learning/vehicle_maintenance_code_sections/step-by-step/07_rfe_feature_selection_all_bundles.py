# ==============================================================================
# 07 Rfe Feature Selection All Bundles
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 9.3) Recursive Feature Elimination (RFE) across all feature bundles
# ############################################################################

# ==============================================================================
# 9.3) Recursive Feature Elimination (RFE) across all feature bundles
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 46 ---
# ### 9.3) Recursive Feature Elimination (RFE) across all feature bundles
#
# This stricter feature-selection section now applies RFE to **all three feature bundles** instead of only the engineered-full feature set:
#
# 1. `baseline_cleaned_lr`
# 2. `engineered_full_lr`
# 3. `leakage_aware_lr`
#
# RFE is useful here because it uses Logistic Regression itself to repeatedly remove the weakest transformed features after imputation, scaling, and one-hot encoding.
# The comparison is made at two levels:
#
# - the best selected-feature count **within each bundle**
# - the overall best RFE-supported bundle across all three bundles
#
# The selected-feature counts are **transformed features**, not raw columns, because categorical variables are expanded by one-hot encoding.
#
# When the original dirty CSV is available and the notebook is run end-to-end, this cell saves:
#
# - `validation_rfe_feature_selection_all_bundles.csv`
# - `validation_rfe_best_by_bundle.csv`
# - `validation_rfe_overall_best_bundle.csv`
# - `validation_rfe_selected_features_by_bundle.csv`
# - `validation_rfe_selected_features.csv` as a compatibility alias for the overall best selected-feature table

# --- Code cell 47 ---
from sklearn.feature_selection import RFE


def build_rfe_lr_pipeline(X: pd.DataFrame, n_features_to_select: int, *, C: float = 1.0) -> Pipeline:
    """Build an impute/scale/encode -> RFE -> Logistic Regression pipeline."""
    numeric_columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    rfe_estimator = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        C=C,
        random_state=42,
    )

    selector = RFE(
        estimator=rfe_estimator,
        n_features_to_select=int(n_features_to_select),
        step=0.20,
    )

    final_model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        C=C,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", final_model),
        ]
    )


rfe_requested_counts = [5, 10, 15, 20, 25, 30, 40, 50]
rfe_rows = []
rfe_selected_feature_rows = []
rfe_threshold_rows = []
rfe_feature_sets = {}

rfe_bundle_reason = {
    "baseline_cleaned_lr": (
        "This would mean the cleaned original columns already contain most of the useful signal, "
        "so extra engineered variables add little validation benefit."
    ),
    "engineered_full_lr": (
        "This means the proposal-driven temporal, usage-intensity, and condition features add useful predictive signal "
        "while retaining the strongest condition indicators."
    ),
    "leakage_aware_lr": (
        "This would mean the reduced feature set gives the best validation trade-off while avoiding condition-heavy variables "
        "that may inflate apparent performance."
    ),
}

for bundle_name, bundle in feature_bundles.items():
    X_train_rfe, X_val_rfe, X_test_rfe, y_train_rfe, y_val_rfe, y_test_rfe = split_data(
        bundle.X,
        bundle.y,
        random_state=42,
    )

    # Count transformed features after the same preprocessing used by Logistic Regression.
    rfe_probe_preprocessor = build_lr_pipeline_for_validation(bundle.X, penalty="l2", C=1.0).named_steps[
        "preprocessor"
    ]
    rfe_probe_preprocessor.fit(X_train_rfe, y_train_rfe)
    rfe_all_feature_names = rfe_probe_preprocessor.get_feature_names_out()
    rfe_total_transformed_features = len(rfe_all_feature_names)

    # Evaluate several subset sizes and include the all-feature case as a no-elimination reference.
    rfe_feature_counts = sorted(
        {
            int(min(count, rfe_total_transformed_features))
            for count in rfe_requested_counts
            if count <= rfe_total_transformed_features
        }
    )
    if rfe_total_transformed_features not in rfe_feature_counts:
        rfe_feature_counts.append(int(rfe_total_transformed_features))
    if not rfe_feature_counts:
        rfe_feature_counts = [int(rfe_total_transformed_features)]

    for selected_count in rfe_feature_counts:
        rfe_pipe = build_rfe_lr_pipeline(bundle.X, selected_count, C=1.0)
        rfe_pipe.fit(X_train_rfe, y_train_rfe)

        rfe_val_prob = rfe_pipe.predict_proba(X_val_rfe)[:, 1]
        best_rfe_threshold, rfe_threshold_table = search_threshold(y_val_rfe, rfe_val_prob)
        rfe_metrics = evaluate_validation_metrics(
            y_val_rfe,
            rfe_val_prob,
            threshold=best_rfe_threshold.threshold,
        )

        transformed_feature_names = rfe_pipe.named_steps["preprocessor"].get_feature_names_out()
        selected_mask = rfe_pipe.named_steps["selector"].support_
        selected_feature_names = transformed_feature_names[selected_mask]
        selected_coefficients = rfe_pipe.named_steps["model"].coef_[0]

        selected_features_df = (
            pd.DataFrame(
                {
                    "bundle": bundle_name,
                    "selected_transformed_features": int(selected_count),
                    "total_transformed_features": int(rfe_total_transformed_features),
                    "feature": selected_feature_names,
                    "coefficient": selected_coefficients,
                    "abs_coefficient": np.abs(selected_coefficients),
                }
            )
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )
        selected_features_df.insert(3, "feature_rank_within_candidate", np.arange(1, len(selected_features_df) + 1))

        rfe_feature_sets[(bundle_name, int(selected_count))] = selected_features_df
        rfe_selected_feature_rows.append(selected_features_df)

        rfe_threshold_table = rfe_threshold_table.copy()
        rfe_threshold_table.insert(0, "bundle", bundle_name)
        rfe_threshold_table.insert(1, "selected_transformed_features", int(selected_count))
        rfe_threshold_rows.append(rfe_threshold_table)

        rfe_metrics.update(
            {
                "bundle": bundle_name,
                "selected_transformed_features": int(selected_count),
                "total_transformed_features": int(rfe_total_transformed_features),
                "feature_fraction": float(selected_count / rfe_total_transformed_features),
                "raw_input_features": int(bundle.X.shape[1]),
            }
        )
        rfe_rows.append(rfe_metrics)

rfe_validation = (
    pd.DataFrame(rfe_rows)
    .sort_values(
        ["f1", "roc_auc", "precision", "recall", "selected_transformed_features"],
        ascending=[False, False, False, False, True],
    )
    .reset_index(drop=True)
)

rfe_best_by_bundle = (
    rfe_validation
    .sort_values(
        ["bundle", "f1", "roc_auc", "precision", "recall", "selected_transformed_features"],
        ascending=[True, False, False, False, False, True],
    )
    .groupby("bundle", as_index=False, sort=False)
    .head(1)
    .sort_values(
        ["f1", "roc_auc", "precision", "recall", "selected_transformed_features"],
        ascending=[False, False, False, False, True],
    )
    .reset_index(drop=True)
)
rfe_best_by_bundle.insert(0, "bundle_rank", np.arange(1, len(rfe_best_by_bundle) + 1))

best_rfe_row = rfe_best_by_bundle.iloc[0]
best_rfe_bundle = str(best_rfe_row["bundle"])
best_rfe_feature_count = int(best_rfe_row["selected_transformed_features"])
best_rfe_features = rfe_feature_sets[(best_rfe_bundle, best_rfe_feature_count)].copy()

rfe_selected_features_all = pd.concat(rfe_selected_feature_rows, ignore_index=True)
rfe_thresholds_all = pd.concat(rfe_threshold_rows, ignore_index=True)

rfe_overall_best_bundle = pd.DataFrame(
    [
        {
            "best_bundle": best_rfe_bundle,
            "selected_transformed_features": best_rfe_feature_count,
            "total_transformed_features": int(best_rfe_row["total_transformed_features"]),
            "threshold": float(best_rfe_row["threshold"]),
            "accuracy": float(best_rfe_row["accuracy"]),
            "precision": float(best_rfe_row["precision"]),
            "recall": float(best_rfe_row["recall"]),
            "specificity": float(best_rfe_row["specificity"]),
            "f1": float(best_rfe_row["f1"]),
            "roc_auc": float(best_rfe_row["roc_auc"]),
            "pr_auc": float(best_rfe_row["pr_auc"]),
            "reason": rfe_bundle_reason.get(best_rfe_bundle, "Selected by the highest validation F1, with ROC-AUC as a tie-breaker."),
        }
    ]
)

rfe_display_columns = [
    "bundle",
    "selected_transformed_features",
    "total_transformed_features",
    "feature_fraction",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
]

display(Markdown("#### RFE validation table across all three feature bundles"))
display(rfe_validation[rfe_display_columns].round(4))

display(Markdown("#### Best RFE result within each feature bundle"))
display(rfe_best_by_bundle[["bundle_rank"] + rfe_display_columns].round(4))

display(Markdown(
    f"#### Overall best RFE-supported bundle: **{best_rfe_bundle}**\n\n"
    f"It selected **{best_rfe_feature_count}** of "
    f"**{int(best_rfe_row['total_transformed_features'])}** transformed features, "
    f"with validation F1 = **{best_rfe_row['f1']:.4f}** and "
    f"ROC-AUC = **{best_rfe_row['roc_auc']:.4f}**.\n\n"
    f"**Why this bundle is best:** {rfe_bundle_reason.get(best_rfe_bundle, 'Selected by highest validation F1, with ROC-AUC as a tie-breaker.')}"
))
display(best_rfe_features.head(25).round(4))

# New all-bundle outputs.
rfe_validation.to_csv(OUTPUT_DIR / "validation_rfe_feature_selection_all_bundles.csv", index=False)
rfe_best_by_bundle.to_csv(OUTPUT_DIR / "validation_rfe_best_by_bundle.csv", index=False)
rfe_overall_best_bundle.to_csv(OUTPUT_DIR / "validation_rfe_overall_best_bundle.csv", index=False)
rfe_selected_features_all.to_csv(OUTPUT_DIR / "validation_rfe_selected_features_by_bundle.csv", index=False)
rfe_thresholds_all.to_csv(OUTPUT_DIR / "validation_rfe_thresholds_by_bundle.csv", index=False)

# Compatibility aliases used by the previous patch.
rfe_validation.to_csv(OUTPUT_DIR / "validation_rfe_feature_selection.csv", index=False)
best_rfe_features.to_csv(OUTPUT_DIR / "validation_rfe_selected_features.csv", index=False)

