import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "news_dataset.csv"

def run_benchmark():
    if not DATASET_PATH.exists():
        return []

    df = pd.read_csv(DATASET_PATH)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = [
        ("MultinomialNB", MultinomialNB()),
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("RandomForest", RandomForestClassifier(n_estimators=100)),
        ("SVC", SVC(kernel='linear')),
    ]

    results = []

    for name, clf in models:
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("clf", clf),
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        end_time = time.time()
        
        train_time = end_time - start_time

        metrics = {
            "name": name,
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred, average='weighted', zero_division=0), 3),
            "recall": round(recall_score(y_test, pred, average='weighted', zero_division=0), 3),
            "f1": round(f1_score(y_test, pred, average='weighted', zero_division=0), 3),
            "time": round(train_time, 3)
        }
        results.append(metrics)

    return results

if __name__ == "__main__":
    results = run_benchmark()
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:<10} {r['precision']:<10} {r['recall']:<10} {r['f1']:<10} {r['time']:<10}")
