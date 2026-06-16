"""
utils.py
========
Small shared helpers for console output and for exporting results in a
Tableau-friendly form.

Why convert to pandas before writing?
  Spark's native df.write.csv() produces a *directory* of part-files
  (part-00000-*.csv, _SUCCESS, ...), which is awkward to load into Tableau.
  Our result tables are small (<= 20k rows), so collecting to a pandas
  DataFrame and writing one clean .csv is the pragmatic, reproducible choice
  and gives Tableau a single tidy file per output.
"""

from pathlib import Path
import pandas as pd

import config


def banner(title: str) -> None:
    """Print a clearly delimited section header to the console."""
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def save_csv(spark_df, name: str) -> Path:
    """Collect a Spark DataFrame to pandas and write one clean CSV to outputs/."""
    return save_pandas(spark_df.toPandas(), name)


def save_pandas(pdf: pd.DataFrame, name: str) -> Path:
    """Write a pandas DataFrame to outputs/<name>.csv (index dropped)."""
    out_path = config.OUTPUT_DIR / f"{name}.csv"
    pdf.to_csv(out_path, index=False)
    print(f"  -> saved {out_path.relative_to(config.PROJECT_ROOT)}  "
          f"({len(pdf)} rows)")
    return out_path


def feature_importance_table(transformed_df, importances, features_col="features"):
    """
    Map a Spark feature-importance vector back to human-readable feature names.

    Spark stores per-index attribute metadata on the assembled vector column
    (numeric / binary / nominal). We read that metadata to recover names, then
    return a list of (feature_name, importance) sorted high -> low.

    Works for RandomForest and GBT importances alike.
    """
    attrs = transformed_df.schema[features_col].metadata["ml_attr"]["attrs"]
    idx_to_name = {}
    for attr_type in ("numeric", "binary", "nominal"):
        for a in attrs.get(attr_type, []):
            idx_to_name[a["idx"]] = a["name"]

    rows = [(idx_to_name.get(i, f"feature_{i}"), float(importances[i]))
            for i in range(len(importances))]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows
