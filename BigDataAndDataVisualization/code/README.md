# Urban Road Safety Intelligence
### Big Data Analytics of Indian Road Accident Patterns (2022–2025)

PySpark analysis pipeline for the **ST7082CEM – Big Data Management and Data
Visualisation** coursework. The project applies three complementary machine-
learning tasks to a 20,000-row, 24-column Indian road accident dataset and
exports tidy CSV result tables for Tableau dashboards.

| Task | Technique | Target | Script |
|------|-----------|--------|--------|
| Classification | Random Forest (+ class weighting, 5-fold CV) | `accident_severity` (minor/major/fatal) | `02_classification.py` |
| Regression | Gradient-Boosted Trees | `risk_score` (0.0–1.0) | `03_regression.py` |
| Clustering | KMeans (silhouette + elbow) | risk-profile segments | `04_clustering.py` |

---

## 1. Folder structure

```
road_accident_analysis/
├── README.md
├── requirements.txt
├── data/
│   └── indian_roads_dataset.csv      # the dataset (20,000 x 24)
├── src/
│   ├── config.py                     # paths, Spark session, schema, feature lists
│   ├── data_preprocessing.py         # load + clean + feature engineering + encoding
│   ├── utils.py                      # console + CSV-export helpers
│   ├── 01_data_exploration.py        # EDA + Tableau exploration exports
│   ├── 02_classification.py          # Random Forest severity prediction
│   ├── 03_regression.py              # GBT risk-score regression
│   ├── 04_clustering.py              # KMeans risk-profile clustering
│   └── run_all.py                    # runs all four stages in order
└── outputs/                          # CSV result tables (created when you run)
```

---

## 2. Prerequisites

* **Java JDK 8, 11, 17 or 21** (Spark runs on the JVM).
  Check: `java -version`. If missing, install Temurin/OpenJDK and set
  `JAVA_HOME`.
* **Python 3.8–3.12**. Check: `python --version`.

---

## 3. Setup (virtual environment in VS Code)

Open the `road_accident_analysis` folder in VS Code, then open a terminal
(`Terminal → New Terminal`) and run the commands for your OS.

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
*(If activation is blocked, run once:*
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`*, then retry.)*

**Windows (cmd)**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In VS Code, select the interpreter from `venv` via
**Ctrl/Cmd+Shift+P → "Python: Select Interpreter"** so the editor and the
integrated terminal both use the virtual environment.

---

## 4. Running the analysis

With the virtual environment **activated**, run from the project root:

```bash
python src/01_data_exploration.py     # EDA + base Tableau data
python src/02_classification.py       # Random Forest
python src/03_regression.py           # GBT regression
python src/04_clustering.py           # KMeans clustering
```

Or run everything in one go:

```bash
python src/run_all.py
```

### Classification grid size

`02_classification.py` defaults to a **laptop-friendly** search
(`numTrees ∈ {50,100}`, `maxDepth ∈ {8,12}`, 3-fold CV — a few minutes on a
multi-core CPU). To run the **full grid from the proposal**
(`numTrees ∈ {50,100,200}`, `maxDepth ∈ {5,10,15}`, 5-fold CV — substantially
longer), set an environment variable:

```bash
# macOS / Linux
RF_FULL_GRID=1 python src/02_classification.py
# Windows (cmd)
set RF_FULL_GRID=1 && python src/02_classification.py
# Windows (PowerShell)
$env:RF_FULL_GRID=1; python src/02_classification.py
```

---

## 5. Outputs (load these into Tableau)

Running the scripts populates `outputs/` with single, tidy CSV files:

| File | Produced by | Use in Tableau |
|------|-------------|----------------|
| `accidents_clean.csv` | EDA | Master source: geospatial hotspot map, temporal heatmaps, filters |
| `eda_city_severity.csv` | EDA | City × severity counts + avg risk (map) |
| `eda_hour_dow.csv` | EDA | Hour × day-of-week heatmap |
| `eda_cause_severity.csv` | EDA | Cause × severity breakdown |
| `clf_confusion_matrix.csv` | Classification | Confusion-matrix heatmap |
| `clf_metrics.csv` | Classification | Overall + per-class metric bars |
| `clf_feature_importance.csv` | Classification | Feature-importance bar chart |
| `reg_metrics.csv` | Regression | RMSE / R² / MAE |
| `reg_feature_importance.csv` | Regression | Risk-driver ranking |
| `reg_residuals.csv` | Regression | Residual / actual-vs-predicted diagnostics |
| `clu_k_selection.csv` | Clustering | Elbow (WSSSE) + silhouette curves |
| `clu_profiles.csv` | Clustering | Cluster archetype profile table |
| `clu_assignments.csv` | Clustering | Cluster scatter plot + geospatial map |

---

## 6. Notes

* **Reproducibility.** A fixed random seed (42) is used for every split and
  model, so results are stable across runs.
* **PySpark is the sole analysis engine.** All loading, cleaning, feature
  engineering, modelling and aggregation run in PySpark. pandas/pyarrow are
  used only to write one clean `.csv` per result (Spark's native writer emits
  awkward multi-file directories that are harder to load into Tableau).
* **Windows + Spark.** Local mode normally runs fine. If you see a harmless
  `winutils.exe`/Hadoop warning, you can ignore it; to silence it, install
  `winutils.exe` for your Hadoop version into a folder and set
  `HADOOP_HOME` to its parent. (Outputs are written via pandas, so this is not
  required for the pipeline to work.)
* **Spark UI** is disabled in the session config to keep logs clean; re-enable
  it in `config.get_spark()` if you want the web UI on port 4040.
```
