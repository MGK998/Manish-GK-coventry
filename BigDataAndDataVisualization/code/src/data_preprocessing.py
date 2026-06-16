"""
data_preprocessing.py
=====================
Shared data-preparation layer for all three analysis tasks.

Pipeline (matches the project proposal):
  1. load_data()          -- read CSV with an explicit schema
  2. clean_data()         -- null imputation + outlier handling
  3. engineer_features()  -- date parsing, hour_bin, is_festival flag
  4. build_feature_stages()-- StringIndexer -> OneHotEncoder -> VectorAssembler
                              (+ optional StandardScaler for clustering)

`prepare_data()` chains steps 1-3 and returns one analysis-ready DataFrame so
every task starts from an identical, reproducible base.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer, StandardScaler,
)

import config


# --------------------------------------------------------------------------- #
# 1. LOAD
# --------------------------------------------------------------------------- #
def load_data(spark: SparkSession) -> DataFrame:
    """Read the raw CSV using the explicit schema from config.SCHEMA."""
    df = (
        spark.read
        .option("header", True)
        .schema(config.SCHEMA)
        .csv(str(config.DATA_PATH))
    )
    return df


# --------------------------------------------------------------------------- #
# 2. CLEAN
# --------------------------------------------------------------------------- #
def clean_data(df: DataFrame) -> DataFrame:
    """
    Clean the raw data.

    * festival: ~99% of rows are null (no festival on that day). We map null
      to the explicit category 'None' so the field becomes usable.
    * numeric/categorical null imputation: implemented defensively per the
      proposal (median for numeric, most-frequent for categorical). The supplied
      data is complete, so these are effectively safeguards that keep the
      pipeline correct if future data contains gaps.
    * outliers: risk_score is constrained to [0, 1] and casualties capped at 5
      (its documented maximum), again as a safeguard.
    """
    # festival null -> 'None'
    df = df.withColumn(
        "festival",
        F.when(F.col("festival").isNull(), F.lit("None")).otherwise(F.col("festival")),
    )

    # Numeric median imputation (safeguard)
    numeric_to_impute = ["temperature", "lanes", "vehicles_involved",
                         "casualties", "risk_score"]
    imputer = Imputer(
        inputCols=numeric_to_impute,
        outputCols=numeric_to_impute,
        strategy="median",
    )
    df = imputer.fit(df).transform(df)

    # Categorical mode imputation (safeguard)
    for col in ["weather", "visibility", "traffic_density", "road_type"]:
        mode_row = (
            df.filter(F.col(col).isNotNull())
              .groupBy(col).count()
              .orderBy(F.desc("count"))
              .first()
        )
        if mode_row is not None:
            df = df.fillna({col: mode_row[col]})

    # Outlier handling (safeguard)
    df = df.filter((F.col("risk_score") >= 0.0) & (F.col("risk_score") <= 1.0))
    df = df.withColumn(
        "casualties",
        F.when(F.col("casualties") > 5, F.lit(5)).otherwise(F.col("casualties")),
    )
    return df


# --------------------------------------------------------------------------- #
# 3. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def engineer_features(df: DataFrame) -> DataFrame:
    """
    Derive new analytical features.

    * date  -> proper DateType, then accident_year / accident_month
    * hour  -> hour_bin: night / morning / afternoon / evening
    * festival -> is_festival binary flag (1 if a named festival, else 0)
    """
    # Parse the string date into a real date and extract calendar parts.
    df = df.withColumn("date_parsed", F.to_date(F.col("date"), "yyyy-MM-dd"))
    df = df.withColumn("accident_year", F.year("date_parsed"))
    df = df.withColumn("accident_month", F.month("date_parsed"))

    # Time-of-day buckets from the hour column.
    df = df.withColumn(
        "hour_bin",
        F.when((F.col("hour") >= 5) & (F.col("hour") <= 11), "morning")
         .when((F.col("hour") >= 12) & (F.col("hour") <= 16), "afternoon")
         .when((F.col("hour") >= 17) & (F.col("hour") <= 20), "evening")
         .otherwise("night"),
    )

    # Binary festival indicator.
    df = df.withColumn(
        "is_festival",
        F.when(F.col("festival") == "None", F.lit(0)).otherwise(F.lit(1)),
    )
    return df


def prepare_data(spark: SparkSession) -> DataFrame:
    """Convenience: load -> clean -> engineer, cached for repeated use."""
    df = load_data(spark)
    df = clean_data(df)
    df = engineer_features(df)
    return df.cache()


# --------------------------------------------------------------------------- #
# 4. ENCODING / ASSEMBLY STAGES
# --------------------------------------------------------------------------- #
def build_feature_stages(categorical_cols, numeric_cols,
                         output_col="features", scale=False):
    """
    Return a list of Spark ML pipeline stages that turn the chosen columns into
    a single feature vector:

        StringIndexer (per categorical) -> OneHotEncoder (per categorical)
        -> VectorAssembler -> [optional StandardScaler]

    Parameters
    ----------
    categorical_cols : list[str]  string columns to index + one-hot encode
    numeric_cols     : list[str]  numeric columns used as-is
    output_col       : str        name of the final vector column
    scale            : bool       if True, append a StandardScaler (needed for
                                  distance-based KMeans so no single feature
                                  dominates by virtue of its scale)
    """
    stages = []
    indexed_cols, encoded_cols = [], []

    for col in categorical_cols:
        idx = StringIndexer(inputCol=col, outputCol=f"{col}_idx",
                            handleInvalid="keep")
        ohe = OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_ohe")
        stages += [idx, ohe]
        indexed_cols.append(f"{col}_idx")
        encoded_cols.append(f"{col}_ohe")

    assembler_input = encoded_cols + numeric_cols
    assembled_col = "features_raw" if scale else output_col
    stages.append(
        VectorAssembler(inputCols=assembler_input, outputCol=assembled_col,
                        handleInvalid="keep")
    )

    if scale:
        stages.append(
            StandardScaler(inputCol="features_raw", outputCol=output_col,
                           withMean=True, withStd=True)
        )

    return stages
