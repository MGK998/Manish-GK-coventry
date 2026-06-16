"""
04_clustering.py
================
TASK 4 - CLUSTERING: segment accidents into latent risk profiles (unsupervised).

Why KMeans?
  After one-hot encoding the categorical inputs the feature space is fully
  numeric, the dataset size (20k) suits centroid-based partitioning, and the
  resulting centroids are directly interpretable as "risk archetypes"
  (e.g. low-visibility night highway fatals vs. clear-weather urban minors).
  Features are standardised first (StandardScaler) so that no single variable
  dominates the Euclidean distance purely because of its numeric range.

Model selection:
  Fit k = 2..6 and choose k using the Silhouette score (cohesion vs.
  separation), cross-checked against the Elbow of within-cluster SSE (WSSSE).

Run:
    python src/04_clustering.py

Outputs in outputs/:
    clu_k_selection.csv   -- k, silhouette, WSSSE (Tableau elbow + silhouette)
    clu_profiles.csv      -- per-cluster size + average + dominant attributes
    clu_assignments.csv   -- per-accident cluster label (Tableau scatter / map)
"""

import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import config
from config import CLU_CATEGORICAL, CLU_NUMERIC, RANDOM_SEED
from data_preprocessing import prepare_data, build_feature_stages
from utils import banner, save_csv, save_pandas

K_RANGE = [2, 3, 4, 5, 6]


def dominant_category(preds, cat_col):
    """Most frequent value of `cat_col` within each cluster (pure PySpark)."""
    w = Window.partitionBy("prediction").orderBy(F.desc("cnt"))
    return (preds.groupBy("prediction", cat_col).agg(F.count("*").alias("cnt"))
                 .withColumn("rn", F.row_number().over(w))
                 .filter(F.col("rn") == 1)
                 .select("prediction", F.col(cat_col).alias(f"top_{cat_col}")))


def main():
    spark = config.get_spark("Clustering")
    df = prepare_data(spark)

    banner("TASK 4: ACCIDENT RISK-PROFILE CLUSTERING (KMeans)")

    # ----------------------------------------------------------------- #
    # Build + fit the encoding/scaling pipeline ONCE, then reuse
    # ----------------------------------------------------------------- #
    prep_stages = build_feature_stages(CLU_CATEGORICAL, CLU_NUMERIC,
                                       output_col="features", scale=True)
    prep_model = Pipeline(stages=prep_stages).fit(df)
    data = prep_model.transform(df).cache()
    print(f"Prepared {data.count():,} records with standardised features.")

    # ----------------------------------------------------------------- #
    # Search k = 2..6 : Silhouette + WSSSE (elbow)
    # ----------------------------------------------------------------- #
    banner("MODEL SELECTION: SILHOUETTE & ELBOW")
    evaluator = ClusteringEvaluator(featuresCol="features",
                                    predictionCol="prediction",
                                    metricName="silhouette")
    selection, models = [], {}
    for k in K_RANGE:
        km = KMeans(featuresCol="features", predictionCol="prediction",
                    k=k, seed=RANDOM_SEED)
        model = km.fit(data)
        preds = model.transform(data)
        silhouette = evaluator.evaluate(preds)
        wssse = model.summary.trainingCost
        selection.append({"k": k, "silhouette": round(silhouette, 4),
                          "wssse": round(wssse, 2)})
        models[k] = model
        print(f"  k={k}:  silhouette={silhouette:.4f}   WSSSE={wssse:,.1f}")

    save_pandas(pd.DataFrame(selection), "clu_k_selection")

    best_k = max(selection, key=lambda r: r["silhouette"])["k"]
    print(f"\nSelected k = {best_k} (highest silhouette).")

    # ----------------------------------------------------------------- #
    # Final clustering + cluster profiling
    # ----------------------------------------------------------------- #
    banner(f"CLUSTER PROFILES (k = {best_k})")
    best_model = models[best_k]
    clustered = best_model.transform(data)

    # Numeric profile per cluster.
    profile = (clustered.groupBy("prediction")
               .agg(F.count("*").alias("size"),
                    F.round(F.avg("risk_score"), 3).alias("avg_risk_score"),
                    F.round(F.avg("casualties"), 2).alias("avg_casualties"),
                    F.round(F.avg("hour"), 1).alias("avg_hour"),
                    F.round(F.avg("temperature"), 1).alias("avg_temperature"),
                    F.round(F.avg("is_peak_hour"), 2).alias("pct_peak_hour"),
                    F.round(F.avg("is_weekend"), 2).alias("pct_weekend")))

    # Dominant categorical attributes per cluster.
    for cat in ["weather", "visibility", "road_type", "traffic_density"]:
        profile = profile.join(dominant_category(clustered, cat),
                               on="prediction", how="left")

    profile = profile.orderBy("prediction").withColumnRenamed("prediction", "cluster")
    profile.show(truncate=False)
    save_csv(profile, "clu_profiles")

    # Per-accident assignments for the Tableau scatter plot / geospatial map.
    assignments = clustered.select(
        "accident_id", "city", "state", "latitude", "longitude",
        F.col("prediction").alias("cluster"),
        "risk_score", "hour", "casualties", "accident_severity",
        "weather", "visibility", "road_type", "traffic_density",
    )
    save_csv(assignments, "clu_assignments")

    print("\nClustering task complete.")
    spark.stop()


if __name__ == "__main__":
    main()
