"""
config.py
=========
Central configuration for the Urban Road Safety Intelligence project.

Holds:
  * Project paths (resolved relative to this file, so scripts run from anywhere)
  * A single, reusable SparkSession builder tuned for this dataset
  * The dataset schema and the feature lists used by each analysis task

Keeping this in one place makes the three analysis scripts short, consistent
and reproducible (identical Spark config and identical preprocessing inputs).
"""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, StringType,
)

# --------------------------------------------------------------------------- #
# 1. PATHS  (resolved from this file -> robust no matter the working directory)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "indian_roads_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42  # fixed everywhere for reproducibility


# --------------------------------------------------------------------------- #
# 2. SPARK SESSION
# --------------------------------------------------------------------------- #
def get_spark(app_name: str = "RoadSafetyIntelligence") -> SparkSession:
    """
    Build (or fetch) a local SparkSession tuned for a ~20k-row workload.

    Notes on the configuration:
      * master("local[*]")        -> use all available CPU cores
      * shuffle.partitions = 8    -> the default of 200 is wasteful for a small
                                     dataset and makes cross-validation very
                                     slow; 8 keeps shuffles light and fast
      * arrow enabled             -> fast Spark<->pandas conversion on export
      * UI disabled               -> avoids the 4040 port and keeps logs clean
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    # Show only warnings/errors so analysis output is readable.
    spark.sparkContext.setLogLevel("WARN")
    return spark


# --------------------------------------------------------------------------- #
# 3. EXPLICIT SCHEMA
# --------------------------------------------------------------------------- #
# Defining the schema explicitly (instead of inferSchema) is good big-data
# practice: it avoids a full extra pass over the data and guarantees the same
# column types on every run. `date` is read as a string here and converted to a
# proper DateType during feature engineering (robust across Spark versions).
SCHEMA = StructType([
    StructField("accident_id",        IntegerType(), True),
    StructField("city",               StringType(),  True),
    StructField("state",              StringType(),  True),
    StructField("latitude",           DoubleType(),  True),
    StructField("longitude",          DoubleType(),  True),
    StructField("date",               StringType(),  True),   # -> DateType later
    StructField("time",               StringType(),  True),
    StructField("hour",               IntegerType(), True),
    StructField("day_of_week",        StringType(),  True),
    StructField("is_weekend",         IntegerType(), True),   # boolean 0/1
    StructField("road_type",          StringType(),  True),
    StructField("lanes",              IntegerType(), True),
    StructField("traffic_signal",     IntegerType(), True),   # boolean 0/1
    StructField("weather",            StringType(),  True),
    StructField("visibility",         StringType(),  True),
    StructField("temperature",        IntegerType(), True),
    StructField("traffic_density",    StringType(),  True),
    StructField("cause",              StringType(),  True),
    StructField("accident_severity",  StringType(),  True),   # classification target
    StructField("vehicles_involved",  IntegerType(), True),
    StructField("casualties",         IntegerType(), True),
    StructField("is_peak_hour",       IntegerType(), True),   # boolean 0/1
    StructField("festival",           StringType(),  True),   # mostly null
    StructField("risk_score",         DoubleType(),  True),   # regression target
])


# --------------------------------------------------------------------------- #
# 4. FEATURE LISTS PER TASK  (mirrors the project proposal exactly)
# --------------------------------------------------------------------------- #
# Columns engineered in data_preprocessing.engineer_features():
#   hour_bin (string), is_festival (0/1)

# --- Task 2: Classification (predict accident_severity) -------------------- #
CLF_CATEGORICAL = ["visibility", "weather", "traffic_density",
                   "road_type", "hour_bin", "cause"]
CLF_NUMERIC = ["risk_score", "is_peak_hour", "lanes",
               "vehicles_involved", "is_festival"]

# --- Task 3: Regression (predict risk_score) ------------------------------- #
REG_CATEGORICAL = ["weather", "visibility", "traffic_density",
                   "road_type", "hour_bin", "city"]
REG_NUMERIC = ["temperature", "traffic_signal", "lanes",
               "is_peak_hour", "is_weekend", "is_festival"]

# --- Task 4: Clustering (risk-profile segmentation) ------------------------ #
CLU_CATEGORICAL = ["traffic_density", "visibility", "weather", "road_type"]
CLU_NUMERIC = ["risk_score", "hour", "casualties",
               "is_peak_hour", "is_weekend", "temperature"]
