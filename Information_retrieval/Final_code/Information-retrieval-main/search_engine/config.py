from dataclasses import dataclass
from pathlib import Path

@dataclass
class CrawlConfig:
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    delay_seconds: float = 2.0
    max_pages: int = 300
    same_domain_only: bool = True
    org: str = ""

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PUBLICATIONS_JSONL = str(DATA_DIR / "publications.jsonl")
INDEX_JSON = str(DATA_DIR / "index.json")
