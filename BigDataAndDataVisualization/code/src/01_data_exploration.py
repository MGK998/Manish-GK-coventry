"""
01_data_exploration.py
=======================
Exploratory Data Analysis (EDA) on the Indian Road Accident dataset using
PySpark, plus export of clean data sources for the Tableau dashboards.

Run:
    python src/01_data_exploration.py

Produces in outputs/:
    accidents_clean.csv      -- full cleaned + engineered dataset (Tableau master)
    eda_city_severity.csv    -- city x severity counts + avg risk/casualties (map)
    eda_hour_dow.csv         -- hour x day_of_week accident counts (heatmap)
    eda_cause_severity.csv   -- cause x severity counts
"""

from pyspark.sql import functions as F

import config
from data_preprocessing import prepare_data
from utils import banner, save_csv


def main():
    spark = config.get_spark("EDA")
    df = prepare_data(spark)

    # ----------------------------------------------------------------- #
    # Basic shape and schema
    # ----------------------------------------------------------------- #
    banner("1. DATASET OVERVIEW")
    n_rows = df.count()
    n_cols = len(df.columns)
    print(f"Rows: {n_rows:,}   |   Columns (after engineering): {n_cols}")
    print("\nSchema:")
    df.printSchema()

    # ----------------------------------------------------------------- #
    # Data-type variety (assignment brief requires >= 3 types)
    # ----------------------------------------------------------------- #
    banner("2. DATA-TYPE VARIETY")
    type_counts = {}
    for _, dtype in df.dtypes:
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
    for dtype, cnt in sorted(type_counts.items()):
        print(f"  {dtype:10s}: {cnt} column(s)")

    # ----------------------------------------------------------------- #
    # Missing values
    # ----------------------------------------------------------------- #
    banner("3. MISSING VALUES (after cleaning)")
    missing = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
    ]).collect()[0].asDict()
    any_missing = {k: v for k, v in missing.items() if v > 0}
    print(any_missing if any_missing else "  No missing values remain.")

    # ----------------------------------------------------------------- #
    # Numeric summary
    # ----------------------------------------------------------------- #
    banner("4. NUMERIC SUMMARY STATISTICS")
    numeric_cols = ["hour", "lanes", "temperature",
                    "vehicles_involved", "casualties", "risk_score"]
    df.select(numeric_cols).describe().show()

    # ----------------------------------------------------------------- #
    # Categorical distributions
    # ----------------------------------------------------------------- #
    banner("5. KEY CATEGORICAL DISTRIBUTIONS")
    for col in ["accident_severity", "cause", "weather",
                "road_type", "traffic_density", "visibility"]:
        print(f"\n{col}:")
        (df.groupBy(col).count()
           .orderBy(F.desc("count"))
           .show(truncate=False))

    # ----------------------------------------------------------------- #
    # Target relationship: risk_score by severity
    # ----------------------------------------------------------------- #
    banner("6. AVG RISK SCORE & CASUALTIES BY SEVERITY")
    (df.groupBy("accident_severity")
       .agg(F.round(F.avg("risk_score"), 3).alias("avg_risk_score"),
            F.round(F.avg("casualties"), 2).alias("avg_casualties"),
            F.count("*").alias("n"))
       .orderBy(F.desc("avg_risk_score"))
       .show())

    # ----------------------------------------------------------------- #
    # Exports for Tableau
    # ----------------------------------------------------------------- #
    banner("7. EXPORTING TABLEAU DATA SOURCES")

    # Master file for all dashboards (drop helper date column to keep it tidy).
    master = df.drop("date_parsed")
    save_csv(master, "accidents_clean")

    # Dashboard 1: geospatial hotspot map (city x severity).
    city_sev = (df.groupBy("city", "state", "accident_severity")
                  .agg(F.count("*").alias("accidents"),
                       F.round(F.avg("risk_score"), 3).alias("avg_risk_score"),
                       F.round(F.avg("latitude"), 5).alias("avg_lat"),
                       F.round(F.avg("longitude"), 5).alias("avg_lon"))
                  .orderBy("city", "accident_severity"))
    save_csv(city_sev, "eda_city_severity")

    # Dashboard 2: temporal heatmap (hour x day_of_week).
    hour_dow = (df.groupBy("day_of_week", "hour")
                  .agg(F.count("*").alias("accidents"))
                  .orderBy("day_of_week", "hour"))
    save_csv(hour_dow, "eda_hour_dow")

    # Cause x severity breakdown.
    cause_sev = (df.groupBy("cause", "accident_severity")
                   .agg(F.count("*").alias("accidents"))
                   .orderBy("cause", "accident_severity"))
    save_csv(cause_sev, "eda_cause_severity")

    print("\nEDA complete.")
    spark.stop()


if __name__ == "__main__":
    main()
