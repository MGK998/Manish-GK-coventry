
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from .features import FeatureBundle


RISK_BINS = [-np.inf, 0.40, 0.70, np.inf]
RISK_LABELS = ["Low", "Medium", "High"]


@dataclass
class ThresholdInfo:
    threshold: float
    precision: float
    recall: float
    f1: float


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_columns = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

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


def search_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[ThresholdInfo, pd.DataFrame]:
    thresholds = np.linspace(0.20, 0.80, 121)
    rows = []
    best_row = None

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
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

    threshold_table = pd.DataFrame(rows)
    best_candidate = best_row[1]
    return ThresholdInfo(**best_candidate), threshold_table


def evaluate_probability_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
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
    bands = pd.cut(probabilities, bins=RISK_BINS, labels=RISK_LABELS)
    band_df = pd.DataFrame(
        {
            "model": model_name,
            "probability": probabilities,
            "band": bands,
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


def save_confusion_matrix_plot(metrics: dict, output_path: Path, title: str) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_plot(y_true: pd.Series, probabilities: np.ndarray, output_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pr_plot(y_true: pd.Series, probabilities: np.ndarray, output_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_coefficients_plot(coefficients: pd.DataFrame, output_path: Path, title: str, top_n: int = 12) -> None:
    plot_df = coefficients.head(top_n).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(plot_df["feature"], plot_df["coefficient"])
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_bundle(bundle: FeatureBundle, output_dir: Path) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(bundle.X, bundle.y)
    pipeline = build_pipeline(bundle.X)
    pipeline.fit(X_train, y_train)

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    threshold_info, threshold_table = search_threshold(y_val, val_probabilities)

    default_val_metrics = evaluate_probability_metrics(y_val, val_probabilities, threshold=0.50)
    tuned_val_metrics = evaluate_probability_metrics(y_val, val_probabilities, threshold=threshold_info.threshold)
    default_test_metrics = evaluate_probability_metrics(y_test, test_probabilities, threshold=0.50)
    tuned_test_metrics = evaluate_probability_metrics(y_test, test_probabilities, threshold=threshold_info.threshold)

    coefficients = get_coefficients(pipeline)
    risk_bands = build_risk_band_summary(y_test, test_probabilities, model_name=bundle.name)

    model_output_dir = output_dir / bundle.name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    threshold_table.to_csv(model_output_dir / "threshold_search.csv", index=False)
    coefficients.to_csv(model_output_dir / "coefficients.csv", index=False)
    risk_bands.to_csv(model_output_dir / "risk_bands.csv", index=False)

    save_confusion_matrix_plot(
        tuned_test_metrics,
        model_output_dir / "confusion_matrix_tuned.png",
        f"{bundle.name} - Tuned threshold confusion matrix",
    )
    save_roc_plot(y_test, test_probabilities, model_output_dir / "roc_curve.png", f"{bundle.name} - ROC curve")
    save_pr_plot(
        y_test,
        test_probabilities,
        model_output_dir / "precision_recall_curve.png",
        f"{bundle.name} - Precision-recall curve",
    )
    save_top_coefficients_plot(
        coefficients,
        model_output_dir / "top_coefficients.png",
        f"{bundle.name} - strongest logistic coefficients",
    )

    metrics = {
        "model": bundle.name,
        "n_rows": int(len(bundle.X)),
        "n_input_columns": int(bundle.X.shape[1]),
        "positive_rate": float(bundle.y.mean()),
        "selected_threshold": asdict(threshold_info),
        "validation_default_0_50": default_val_metrics,
        "validation_tuned": tuned_val_metrics,
        "test_default_0_50": default_test_metrics,
        "test_tuned": tuned_test_metrics,
    }

    with open(model_output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "metrics": metrics,
        "coefficients": coefficients,
        "risk_bands": risk_bands,
    }


def save_summary_tables(model_results: dict[str, dict], output_dir: Path) -> None:
    rows = []
    risk_rows = []

    for model_name, result in model_results.items():
        metrics = result["metrics"]
        for split_name in ["validation_default_0_50", "validation_tuned", "test_default_0_50", "test_tuned"]:
            split_metrics = metrics[split_name]
            rows.append(
                {
                    "model": model_name,
                    "evaluation_slice": split_name,
                    **split_metrics,
                }
            )
        risk_rows.append(result["risk_bands"])

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "all_model_metrics.csv", index=False)

    risk_df = pd.concat(risk_rows, ignore_index=True)
    risk_df.to_csv(output_dir / "all_model_risk_bands.csv", index=False)

    summary_rows = []
    for model_name, result in model_results.items():
        tuned = result["metrics"]["test_tuned"]
        summary_rows.append(
            {
                "model": model_name,
                "threshold": tuned["threshold"],
                "accuracy": tuned["accuracy"],
                "precision": tuned["precision"],
                "recall": tuned["recall"],
                "f1": tuned["f1"],
                "roc_auc": tuned["roc_auc"],
                "pr_auc": tuned["pr_auc"],
                "brier_score": tuned["brier_score"],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    summary_df.to_csv(output_dir / "test_tuned_summary.csv", index=False)


def run_modeling(feature_bundles: dict[str, FeatureBundle], output_dir: Path) -> dict[str, dict]:
    model_results = {}
    for bundle_name, bundle in feature_bundles.items():
        model_results[bundle_name] = train_and_evaluate_bundle(bundle, output_dir=output_dir)
    save_summary_tables(model_results, output_dir=output_dir)
    return model_results
