"""
02_classification.py
=====================
TASK 2 - CLASSIFICATION: predict accident_severity (minor / major / fatal).

Why Random Forest?
  The feature space mixes ordinal-categorical and numeric variables and the
  drivers of severity interact non-linearly (e.g. fog x night x highway).
  A tree ensemble captures these interactions natively, needs no feature
  scaling or polynomial expansion, and is robust to the moderate class
  imbalance present here. It is preferred over logistic regression for those
  reasons, and over a single decision tree because bagging reduces variance.

Imbalance handling:
  minor 55% / major 30% / fatal 15%. We pass inverse-frequency class weights
  (largest class = 1.0) via weightCol so the minority 'fatal' class is not
  ignored by the model.

Tuning:
  5-fold CrossValidator over {numTrees: 50/100/200, maxDepth: 5/10/15},
  optimising weighted F1.

Run:
    python src/02_classification.py

Outputs in outputs/:
    clf_confusion_matrix.csv   -- actual x predicted counts (Tableau heatmap)
    clf_metrics.csv            -- overall + per-class metrics
    clf_feature_importance.csv -- ranked feature importances
"""

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pandas as pd
import os

import config
from config import CLF_CATEGORICAL, CLF_NUMERIC, RANDOM_SEED
from data_preprocessing import prepare_data, build_feature_stages
from utils import banner, save_pandas, feature_importance_table


# --------------------------------------------------------------------------- #
# Hyperparameter-search configuration
# --------------------------------------------------------------------------- #
# The proposal specifies a 5-fold search over {numTrees: 50/100/200,
# maxDepth: 5/10/15} = 45 model fits. That is compute-heavy on a single
# machine (deep forests dominate the cost). The defaults below are
# laptop-friendly (a few minutes on a multi-core CPU) and still demonstrate a
# genuine tuned search.
#
#   * Just run `python src/02_classification.py` for the balanced default.
#   * Set FULL_GRID = True (or env var RF_FULL_GRID=1) to run the complete
#     proposal grid for the final report.
FULL_GRID = os.environ.get("RF_FULL_GRID", "0") == "1"

if FULL_GRID:
    NUM_TREES_GRID, MAX_DEPTH_GRID, NUM_FOLDS = [50, 100, 200], [5, 10, 15], 5
else:
    NUM_TREES_GRID, MAX_DEPTH_GRID, NUM_FOLDS = [50, 100], [8, 12], 3

# Optional fine-grained overrides (advanced / for quick iteration), e.g.
#   RF_NUM_TREES=20 RF_MAX_DEPTH=5 RF_FOLDS=2 python src/02_classification.py
NUM_TREES_GRID = [int(x) for x in os.environ["RF_NUM_TREES"].split(",")] \
    if "RF_NUM_TREES" in os.environ else NUM_TREES_GRID
MAX_DEPTH_GRID = [int(x) for x in os.environ["RF_MAX_DEPTH"].split(",")] \
    if "RF_MAX_DEPTH" in os.environ else MAX_DEPTH_GRID
NUM_FOLDS = int(os.environ.get("RF_FOLDS", NUM_FOLDS))


def main():
    spark = config.get_spark("Classification")
    df = prepare_data(spark)

    # ----------------------------------------------------------------- #
    # Encode the label and compute inverse-frequency class weights
    # ----------------------------------------------------------------- #
    banner("TASK 2: ACCIDENT SEVERITY CLASSIFICATION (Random Forest)")

    label_indexer = StringIndexer(
        inputCol="accident_severity", outputCol="label",
        stringOrderType="frequencyDesc",
    )
    label_model = label_indexer.fit(df)
    df = label_model.transform(df)
    labels = label_model.labels                      # index -> severity name
    print("Label mapping (index -> severity):",
          {i: name for i, name in enumerate(labels)})

    counts = {r["accident_severity"]: r["count"]
              for r in df.groupBy("accident_severity").count().collect()}
    max_count = max(counts.values())
    weight_col = F.lit(None)
    for sev, cnt in counts.items():
        w = round(max_count / cnt, 4)
        weight_col = F.when(F.col("accident_severity") == sev, F.lit(w)) \
            .otherwise(weight_col)
    df = df.withColumn("weight", weight_col)
    print("Class weights:",
          {sev: round(max_count / cnt, 4) for sev, cnt in counts.items()})

    # ----------------------------------------------------------------- #
    # Stratified 80/20 split (preserves class proportions)
    # ----------------------------------------------------------------- #
    fractions = {sev: 0.8 for sev in counts}
    train = df.sampleBy("accident_severity", fractions, seed=RANDOM_SEED)
    test = df.join(train.select("accident_id"), on="accident_id", how="left_anti")
    train, test = train.cache(), test.cache()
    print(f"\nTrain rows: {train.count():,}   Test rows: {test.count():,}")

    # ----------------------------------------------------------------- #
    # Build pipeline: feature encoding -> Random Forest
    # ----------------------------------------------------------------- #
    feature_stages = build_feature_stages(CLF_CATEGORICAL, CLF_NUMERIC,
                                          output_col="features")
    rf = RandomForestClassifier(
        labelCol="label", featuresCol="features",
        weightCol="weight", seed=RANDOM_SEED,
    )
    pipeline = Pipeline(stages=feature_stages + [rf])

    # ----------------------------------------------------------------- #
    # 5-fold cross-validated hyperparameter search (weighted F1)
    # ----------------------------------------------------------------- #
    grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, NUM_TREES_GRID)
        .addGrid(rf.maxDepth, MAX_DEPTH_GRID)
        .build()
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1",
    )
    cv = CrossValidator(
        estimator=pipeline, estimatorParamMaps=grid,
        evaluator=f1_evaluator, numFolds=NUM_FOLDS,
        parallelism=2, seed=RANDOM_SEED,
    )

    mode = "FULL proposal grid" if FULL_GRID else "balanced default grid"
    print(f"\nRunning {NUM_FOLDS}-fold CV over {len(grid)} parameter "
          f"combinations ({len(grid) * NUM_FOLDS} model fits) [{mode}]...")
    cv_model = cv.fit(train)

    # Report the winning hyperparameters.
    best_idx = max(range(len(cv.getEstimatorParamMaps())),
                   key=lambda i: cv_model.avgMetrics[i])
    best_params = {p.name: v for p, v
                   in cv.getEstimatorParamMaps()[best_idx].items()}
    print("Best hyperparameters:", best_params)
    print(f"Best CV weighted F1: {cv_model.avgMetrics[best_idx]:.4f}")

    best_model = cv_model.bestModel

    # ----------------------------------------------------------------- #
    # Evaluate on the held-out test set
    # ----------------------------------------------------------------- #
    banner("TEST-SET PERFORMANCE")
    pred = best_model.transform(test)

    def ev(metric, **kw):
        return MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction",
            metricName=metric, **kw).evaluate(pred)

    overall = {
        "accuracy": ev("accuracy"),
        "weighted_f1": ev("f1"),
        "weighted_precision": ev("weightedPrecision"),
        "weighted_recall": ev("weightedRecall"),
    }
    for k, v in overall.items():
        print(f"  {k:20s}: {v:.4f}")

    # Per-class precision / recall / F1.
    print("\nPer-class metrics:")
    per_class_rows = []
    for i, name in enumerate(labels):
        p = ev("precisionByLabel", metricLabel=float(i))
        r = ev("recallByLabel", metricLabel=float(i))
        f = ev("fMeasureByLabel", metricLabel=float(i))
        per_class_rows.append({"severity": name, "precision": round(p, 4),
                               "recall": round(r, 4), "f1": round(f, 4)})
        print(f"  {name:6s}  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")

    # ----------------------------------------------------------------- #
    # Confusion matrix (long format for a Tableau heatmap)
    # ----------------------------------------------------------------- #
    banner("EXPORTING RESULTS")
    cm = (pred.groupBy("label", "prediction").count()
              .orderBy("label", "prediction").collect())
    cm_rows = [{
        "actual": labels[int(r["label"])],
        "predicted": labels[int(r["prediction"])],
        "count": r["count"],
    } for r in cm]
    save_pandas(pd.DataFrame(cm_rows), "clf_confusion_matrix")

    # Metrics table (overall rows + per-class rows).
    metrics_rows = [{"metric": k, "value": round(v, 4)}
                    for k, v in overall.items()]
    metrics_df = pd.concat([
        pd.DataFrame(metrics_rows),
        pd.DataFrame(per_class_rows),
    ], axis=0, ignore_index=True)
    save_pandas(metrics_df, "clf_metrics")

    # Feature importances.
    importances = best_model.stages[-1].featureImportances
    fi = feature_importance_table(pred, importances, "features")
    fi_df = pd.DataFrame(fi, columns=["feature", "importance"])
    fi_df["importance"] = fi_df["importance"].round(5)
    save_pandas(fi_df, "clf_feature_importance")
    print("\nTop 8 features:")
    print(fi_df.head(8).to_string(index=False))

    print("\nClassification task complete.")
    spark.stop()


if __name__ == "__main__":
    main()
