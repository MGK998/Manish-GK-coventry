# ==============================================================================
# 06 L2 L1 Parameter Validation
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 9.2) Formal feature-selection and classification-parameter validation
# ############################################################################

# ==============================================================================
# 9.2) Formal feature-selection and classification-parameter validation
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 44 ---
# ### 9.2) Formal feature-selection and classification-parameter validation
#
# It uses three **feature bundles** and a fixed Logistic Regression configuration.  
# To make the experimental setup more explicit, the code below adds two formal validation steps on the **engineered full** feature set:
#
# 1. **L2 Logistic Regression hyperparameter sweep** over several `C` values on the validation split.  
# 2. **L1 Logistic Regression sparse-model sweep** to provide a formal feature-selection check by shrinking weak coefficients to zero.
#
# These validation tables do **not replace** the main reported models above.  
# They document the experimental setup more formally.

# --- Code cell 45 ---
def build_lr_pipeline_for_validation(X: pd.DataFrame, *, penalty: str = "l2", C: float = 1.0) -> Pipeline:
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

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        penalty=penalty,
        C=C,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_validation_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.50) -> dict:
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def coefficient_table_from_pipeline(pipe: Pipeline) -> pd.DataFrame:
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipe.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return coef_df.reset_index(drop=True)


validation_bundle = feature_bundles["engineered_full_lr"]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(validation_bundle.X, validation_bundle.y, random_state=42)

l2_rows = []
for C in [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00]:
    pipe = build_lr_pipeline_for_validation(validation_bundle.X, penalty="l2", C=C)
    pipe.fit(X_train, y_train)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    metrics = evaluate_validation_metrics(y_val, val_prob, threshold=0.50)
    metrics.update({"penalty": "l2", "C": C})
    l2_rows.append(metrics)

l2_validation = (
    pd.DataFrame(l2_rows)
    .sort_values(["f1", "roc_auc", "precision"], ascending=False)
    .reset_index(drop=True)
)

l1_rows = []
l1_feature_sets = {}
for C in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
    pipe = build_lr_pipeline_for_validation(validation_bundle.X, penalty="l1", C=C)
    pipe.fit(X_train, y_train)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    metrics = evaluate_validation_metrics(y_val, val_prob, threshold=0.50)
    coef_df = coefficient_table_from_pipeline(pipe)
    selected_df = coef_df[coef_df["abs_coefficient"] > 1e-8].copy().reset_index(drop=True)
    l1_feature_sets[C] = selected_df
    metrics.update(
        {
            "penalty": "l1",
            "C": C,
            "selected_transformed_features": int(len(selected_df)),
        }
    )
    l1_rows.append(metrics)

l1_validation = (
    pd.DataFrame(l1_rows)
    .sort_values(["f1", "roc_auc", "precision"], ascending=False)
    .reset_index(drop=True)
)

best_l2_row = l2_validation.iloc[0]
best_sparse_row = l1_validation.iloc[0]
best_sparse_features = l1_feature_sets[float(best_sparse_row["C"])].copy()

display(Markdown("#### L2 Logistic Regression validation sweep"))
display(l2_validation.round(4))

display(Markdown("#### L1 sparse Logistic Regression validation sweep (formal feature-selection check)"))
display(l1_validation.round(4))

display(Markdown(
    f"#### Best sparse model selected **{int(best_sparse_row['selected_transformed_features'])}** transformed features "
    f"at **C = {best_sparse_row['C']:.2f}**"
))
display(best_sparse_features.head(25).round(4))

l2_validation.to_csv(OUTPUT_DIR / "validation_l2_hyperparameter_sweep.csv", index=False)
l1_validation.to_csv(OUTPUT_DIR / "validation_l1_sparse_sweep.csv", index=False)
best_sparse_features.to_csv(OUTPUT_DIR / "validation_l1_selected_features.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(l2_validation["C"], l2_validation["f1"], marker="o", label="L2 F1")
ax.plot(l1_validation["C"], l1_validation["f1"], marker="o", label="L1 F1")
ax.set_xscale("log")
ax.set_xlabel("C")
ax.set_ylabel("Validation F1")
ax.set_title("Classification-parameter validation on the validation split")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "validation_parameter_sweep_f1.png", bbox_inches="tight")
plt.show()

