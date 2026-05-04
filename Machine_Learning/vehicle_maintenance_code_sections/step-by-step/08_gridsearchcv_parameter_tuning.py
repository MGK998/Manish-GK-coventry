# ==============================================================================
# 08 Gridsearchcv Parameter Tuning
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 9.4) Full GridSearchCV validation for strict classification-parameter review
# ############################################################################

# ==============================================================================
# 9.4) Full GridSearchCV validation for strict classification-parameter review
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 48 ---
# ### 9.4) Full GridSearchCV validation for strict classification-parameter review
#
# This extra cell is added for a stricter reviewer. It performs a formal `GridSearchCV` over valid Logistic Regression combinations across:
#
# - `solver`
# - `penalty`
# - `C`
# - `class_weight`
#
# It uses stratified 3-fold cross-validation on the training split, refits the best model by F1-score, then evaluates the best estimator on the validation split using the same threshold-search logic already used in the notebook.
#
# When the original dirty CSV is available and the notebook is run end-to-end, this cell saves:
#
# - `validation_gridsearch_parameter_grid_candidates.csv`
# - `validation_gridsearch_logreg.csv`
# - `validation_gridsearch_best_validation_metrics.csv`
# - `validation_gridsearch_thresholds.csv`

# --- Code cell 49 ---
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold


grid_search_C_values = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00]

# Keep only solver/penalty combinations that LogisticRegression supports.
grid_search_param_grid = [
    {
        "model__solver": ["liblinear"],
        "model__penalty": ["l1", "l2"],
        "model__C": grid_search_C_values,
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__solver": ["lbfgs"],
        "model__penalty": ["l2"],
        "model__C": grid_search_C_values,
        "model__class_weight": [None, "balanced"],
    },
    {
        "model__solver": ["saga"],
        "model__penalty": ["l1", "l2"],
        "model__C": grid_search_C_values,
        "model__class_weight": [None, "balanced"],
    },
]

grid_search_candidates = pd.DataFrame(list(ParameterGrid(grid_search_param_grid)))
grid_search_candidates = grid_search_candidates.rename(
    columns={
        "model__solver": "solver",
        "model__penalty": "penalty",
        "model__C": "C",
        "model__class_weight": "class_weight",
    }
)
grid_search_candidates["class_weight"] = grid_search_candidates["class_weight"].fillna("None")

display(Markdown("#### GridSearchCV candidate parameter combinations"))
display(grid_search_candidates)
grid_search_candidates.to_csv(OUTPUT_DIR / "validation_gridsearch_parameter_grid_candidates.csv", index=False)

grid_search_base_pipeline = build_lr_pipeline_for_validation(validation_bundle.X, penalty="l2", C=1.0)
grid_search_base_pipeline.named_steps["model"].set_params(max_iter=5000)

grid_search_scoring = {
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "roc_auc": "roc_auc",
}

grid_search_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

logreg_grid_search = GridSearchCV(
    estimator=grid_search_base_pipeline,
    param_grid=grid_search_param_grid,
    scoring=grid_search_scoring,
    refit="f1",
    cv=grid_search_cv,
    n_jobs=-1,
    return_train_score=False,
    error_score=np.nan,
)

logreg_grid_search.fit(X_train, y_train)

grid_results = pd.DataFrame(logreg_grid_search.cv_results_)
grid_results = grid_results[
    [
        "rank_test_f1",
        "mean_test_f1",
        "std_test_f1",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_roc_auc",
        "param_model__solver",
        "param_model__penalty",
        "param_model__C",
        "param_model__class_weight",
    ]
].rename(
    columns={
        "rank_test_f1": "rank_f1",
        "mean_test_f1": "cv_mean_f1",
        "std_test_f1": "cv_std_f1",
        "mean_test_precision": "cv_mean_precision",
        "mean_test_recall": "cv_mean_recall",
        "mean_test_roc_auc": "cv_mean_roc_auc",
        "param_model__solver": "solver",
        "param_model__penalty": "penalty",
        "param_model__C": "C",
        "param_model__class_weight": "class_weight",
    }
)
grid_results["class_weight"] = grid_results["class_weight"].fillna("None")
grid_results = grid_results.sort_values(["rank_f1", "cv_mean_roc_auc"]).reset_index(drop=True)

best_grid_estimator = logreg_grid_search.best_estimator_
grid_val_prob = best_grid_estimator.predict_proba(X_val)[:, 1]
best_grid_threshold, grid_threshold_table = search_threshold(y_val, grid_val_prob)
best_grid_metrics = evaluate_validation_metrics(y_val, grid_val_prob, threshold=best_grid_threshold.threshold)

best_grid_summary = pd.DataFrame(
    [
        {
            "solver": logreg_grid_search.best_params_["model__solver"],
            "penalty": logreg_grid_search.best_params_["model__penalty"],
            "C": logreg_grid_search.best_params_["model__C"],
            "class_weight": "None"
            if logreg_grid_search.best_params_["model__class_weight"] is None
            else logreg_grid_search.best_params_["model__class_weight"],
            "cv_best_f1": logreg_grid_search.best_score_,
            "validation_threshold": best_grid_metrics["threshold"],
            "validation_accuracy": best_grid_metrics["accuracy"],
            "validation_precision": best_grid_metrics["precision"],
            "validation_recall": best_grid_metrics["recall"],
            "validation_specificity": best_grid_metrics["specificity"],
            "validation_f1": best_grid_metrics["f1"],
            "validation_roc_auc": best_grid_metrics["roc_auc"],
            "validation_pr_auc": best_grid_metrics["pr_auc"],
            "validation_tn": best_grid_metrics["tn"],
            "validation_fp": best_grid_metrics["fp"],
            "validation_fn": best_grid_metrics["fn"],
            "validation_tp": best_grid_metrics["tp"],
        }
    ]
)
best_grid_row = best_grid_summary.iloc[0]

display(Markdown("#### Full GridSearchCV results, sorted by cross-validated F1"))
display(grid_results.head(20).round(4))

display(Markdown("#### Best GridSearchCV model validated on the validation split"))
display(best_grid_summary.round(4))

grid_results.to_csv(OUTPUT_DIR / "validation_gridsearch_logreg.csv", index=False)
best_grid_summary.to_csv(OUTPUT_DIR / "validation_gridsearch_best_validation_metrics.csv", index=False)
grid_threshold_table.to_csv(OUTPUT_DIR / "validation_gridsearch_thresholds.csv", index=False)

