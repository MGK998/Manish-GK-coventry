
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Define paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "news_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "model.joblib"
OUTPUT_PATH = BASE_DIR / "confusion_matrix.png"

def generate_plot():
    print("--- Generating Confusion Matrix Heatmap ---")
    
    # 1. Load data and model
    df = pd.read_csv(DATASET_PATH)
    X = df["text"].astype(str)
    y = df["label"].astype(str)
    model = joblib.load(MODEL_PATH)

    # 2. Split data (using the same random state as training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Predict and get confusion matrix
    y_pred = model.predict(X_test)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 4. Plot using Seaborn
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Create the heatmap
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', 
                xticklabels=labels, yticklabels=labels,
                cbar=True, annot_kws={"size": 16})
    
    # Beautify
    plt.title('News Classification Confusion Matrix', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=15)
    plt.ylabel('True Label', fontsize=14, labelpad=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Heatmap saved to: {OUTPUT_PATH}")
    plt.close()

if __name__ == "__main__":
    generate_plot()
