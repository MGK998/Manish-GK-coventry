# ==============================================================================
# 04 Logistic Regression Pipeline And Threshold
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 8) Logistic Regression modeling
# ############################################################################

# ==============================================================================
# 8) Logistic Regression modeling
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 38 ---
# ## 8) Logistic Regression modeling
# To train, Validate, and evaluate the models.

# --- Code cell 39 ---
RISK_BINS = [-np.inf, 0.40, 0.70, np.inf]
RISK_LABELS = ["Low", "Medium", "High"]


@dataclass
class ThresholdInfo:
    threshold: float
    precision: float
    recall: float
    f1: float


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(X: pd.DataFrame) -> Pipeline:
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
        C=1.0,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def search_threshold(y_true: pd.Series, probabilities: np.ndarray):
    thresholds = np.linspace(0.20, 0.80, 121)
    rows = []
    best_row = None

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        candidate = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        rows.append(candidate)

        score = (round(f1, 6), round(recall, 6), -abs(threshold - 0.50))
        if best_row is None or score > best_row[0]:
            best_row = (score, candidate)

    return ThresholdInfo(**best_row[1]), pd.DataFrame(rows)


def evaluate_probability_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float):
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def build_risk_band_summary(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> pd.DataFrame:
    band_df = pd.DataFrame(
        {
            "model": model_name,
            "probability": probabilities,
            "band": pd.cut(probabilities, bins=RISK_BINS, labels=RISK_LABELS),
            "actual": y_true.to_numpy(),
        }
    )
    summary = (
        band_df.groupby(["model", "band"], observed=False)
        .agg(
            vehicles=("actual", "size"),
            avg_probability=("probability", "mean"),
            actual_positive_rate=("actual", "mean"),
        )
        .reset_index()
    )
    return summary


def run_bundle(bundle_name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    threshold_info, threshold_table = search_threshold(y_val, val_prob)
    tuned_metrics = evaluate_probability_metrics(y_test, test_prob, threshold_info.threshold)
    default_metrics = evaluate_probability_metrics(y_test, test_prob, 0.50)
    coef_df = get_coefficients(pipeline)
    risk_df = build_risk_band_summary(y_test, test_prob, bundle_name)

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_prob)

    return {
        "name": bundle_name,
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "threshold_info": threshold_info,
        "threshold_table": threshold_table,
        "tuned_metrics": tuned_metrics,
        "default_metrics": default_metrics,
        "coefficients": coef_df,
        "risk_bands": risk_df,
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
    }

