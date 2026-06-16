"""
03_regression.py
================
TASK 3 - REGRESSION: model the continuous risk_score (0.0-1.0).

Why Gradient-Boosted Trees (GBT)?
  risk_score is a composite engineered measure whose relationship to the
  environmental and traffic inputs is non-linear and interaction-heavy.
  GBT builds trees sequentially, each correcting the residual errors of the
  previous ones, which captures these interactions far better than linear
  regression while still yielding an interpretable feature-importance ranking.

Run:
    python src/03_regression.py
  (Optional: RGR_MAX_ITER=30 python src/03_regression.py  for a quicker run.)

Outputs in outputs/:
    reg_metrics.csv             -- RMSE / R2 / MAE on the test set
    reg_feature_importance.csv  -- ranked feature importances (Tableau bar chart)
    reg_residuals.csv           -- actual / predicted / residual (Tableau diagnostics)
"""

import os
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

import config
from config import REG_CATEGORICAL, REG_NUMERIC, RANDOM_SEED
from data_preprocessing import prepare_data, build_feature_stages
from utils import banner, save_csv, save_pandas, feature_importance_table

# GBT settings (proposal: maxIter=100, maxDepth=8, stepSize=0.1).
# maxIter is the main cost knob; override with env var for a faster run.
MAX_ITER = int(os.environ.get("RGR_MAX_ITER", "100"))
MAX_DEPTH = int(os.environ.get("RGR_MAX_DEPTH", "8"))
STEP_SIZE = float(os.environ.get("RGR_STEP_SIZE", "0.1"))


def main():
    spark = config.get_spark("Regression")
    df = prepare_data(spark)

    banner("TASK 3: RISK SCORE REGRESSION (Gradient-Boosted Trees)")
    print(f"GBT config -> maxIter={MAX_ITER}, maxDepth={MAX_DEPTH}, "
          f"stepSize={STEP_SIZE}")

    # ----------------------------------------------------------------- #
    # 80/20 split (random; stratification not required for regression)
    # ----------------------------------------------------------------- #
    train, test = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
    train, test = train.cache(), test.cache()
    print(f"Train rows: {train.count():,}   Test rows: {test.count():,}")

    # ----------------------------------------------------------------- #
    # Pipeline: feature encoding -> GBT regressor
    # ----------------------------------------------------------------- #
    feature_stages = build_feature_stages(REG_CATEGORICAL, REG_NUMERIC,
                                          output_col="features")
    gbt = GBTRegressor(
        labelCol="risk_score", featuresCol="features",
        maxIter=MAX_ITER, maxDepth=MAX_DEPTH, stepSize=STEP_SIZE,
        seed=RANDOM_SEED,
    )
    pipeline = Pipeline(stages=feature_stages + [gbt])

    print("\nTraining GBT regressor...")
    model = pipeline.fit(train)
    pred = model.transform(test)

    # ----------------------------------------------------------------- #
    # Evaluate
    # ----------------------------------------------------------------- #
    banner("TEST-SET PERFORMANCE")

    def ev(metric):
        return RegressionEvaluator(
            labelCol="risk_score", predictionCol="prediction",
            metricName=metric).evaluate(pred)

    metrics = {"RMSE": ev("rmse"), "R2": ev("r2"), "MAE": ev("mae")}
    for k, v in metrics.items():
        print(f"  {k:5s}: {v:.4f}")
    print("\n  Interpretation: R2 is the share of risk-score variance the model "
          "explains;\n  RMSE/MAE are the typical prediction error on the "
          "0-1 risk scale.")

    # ----------------------------------------------------------------- #
    # Export results for Tableau
    # ----------------------------------------------------------------- #
    banner("EXPORTING RESULTS")

    save_pandas(pd.DataFrame([{"metric": k, "value": round(v, 4)}
                              for k, v in metrics.items()]), "reg_metrics")

    # Feature importances (which factors drive risk the most).
    importances = model.stages[-1].featureImportances
    fi = feature_importance_table(pred, importances, "features")
    fi_df = pd.DataFrame(fi, columns=["feature", "importance"])
    fi_df["importance"] = fi_df["importance"].round(5)
    save_pandas(fi_df, "reg_feature_importance")
    print("\nTop 8 risk drivers:")
    print(fi_df.head(8).to_string(index=False))

    # Residuals for diagnostic plotting (actual vs predicted vs residual).
    residuals = (pred
                 .withColumn("residual",
                             F.round(F.col("risk_score") - F.col("prediction"), 4))
                 .withColumn("prediction", F.round("prediction", 4))
                 .select("accident_id", "city", "accident_severity",
                         F.col("risk_score").alias("actual"),
                         "prediction", "residual"))
    save_csv(residuals, "reg_residuals")

    print("\nRegression task complete.")
    spark.stop()


if __name__ == "__main__":
    main()
