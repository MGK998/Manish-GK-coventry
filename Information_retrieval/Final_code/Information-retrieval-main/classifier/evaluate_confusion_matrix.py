
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "news_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "model.joblib"

def evaluate():
    print("--- Classifier Confusion Matrix Evaluation ---")
    
    # 1. Check if dataset and model exist
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Please run training first.")
        return

    # 2. Load data
    df = pd.read_csv(DATASET_PATH)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # 3. Split data (using the same random state as train.py for consistency)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4. Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # 5. Predict
    print("Generating predictions on test set...")
    y_pred = model.predict(X_test)

    # 6. Generate Confusion Matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Create a nice DataFrame for the confusion matrix display
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    
    print("\nConfusion Matrix:")
    print(cm_df.to_string())
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate()
