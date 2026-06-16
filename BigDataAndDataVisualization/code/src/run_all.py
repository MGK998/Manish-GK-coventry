"""
run_all.py
==========
Convenience orchestrator: runs the four analysis stages in order.

    python src/run_all.py

Each stage is launched as its own process (a fresh SparkSession per stage),
which mirrors running the scripts individually and keeps Spark contexts clean.
For the full proposal grid in the classification stage, run with:

    RF_FULL_GRID=1 python src/run_all.py        (macOS / Linux)
    set RF_FULL_GRID=1 && python src/run_all.py  (Windows cmd)
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

STAGES = [
    "01_data_exploration.py",
    "02_classification.py",
    "03_regression.py",
    "04_clustering.py",
]


def main():
    for stage in STAGES:
        print("\n" + "#" * 72)
        print(f"# RUNNING: {stage}")
        print("#" * 72)
        result = subprocess.run([sys.executable, str(SRC_DIR / stage)])
        if result.returncode != 0:
            print(f"\n[ERROR] {stage} exited with code {result.returncode}. "
                  f"Stopping.")
            sys.exit(result.returncode)
    print("\n" + "#" * 72)
    print("# ALL STAGES COMPLETE. See the outputs/ folder for Tableau CSVs.")
    print("#" * 72)


if __name__ == "__main__":
    main()
