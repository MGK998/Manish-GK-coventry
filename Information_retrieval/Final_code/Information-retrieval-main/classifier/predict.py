import argparse
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "data" / "model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_label(text: str) -> str:
    model = load_model()
    if model is None:
        return ""
    return str(model.predict([text])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    model = load_model()
    if model is None:
        print("Model not found. Train first: python -m classifier.train")
        return

    print(predict_label(args.text))


if __name__ == "__main__":
    main()
