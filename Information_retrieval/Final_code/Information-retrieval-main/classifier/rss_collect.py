import argparse
import csv
from pathlib import Path
import feedparser

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "news_dataset.csv"

FEEDS = {
    "Business": "https://feeds.bbci.co.uk/news/business/rss.xml?edition=int",
    "Entertainment": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml?edition=int",
    "Health": "https://feeds.bbci.co.uk/news/health/rss.xml?edition=int",
}


def collect(per_class: int = 40) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for label, url in FEEDS.items():
        feed = feedparser.parse(url)
        count = 0
        for entry in feed.entries:
            title = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or "").strip()
            text = (title + ". " + summary).strip()
            if len(text) < 20:
                continue
            rows.append({"label": label, "text": text, "source": url})
            count += 1
            if count >= per_class:
                break

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "text", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved dataset: {CSV_PATH} (rows={len(rows)})")
    print("Note: This stores short RSS summaries only (not full articles).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()
    collect(per_class=args.per_class)


if __name__ == "__main__":
    main()
