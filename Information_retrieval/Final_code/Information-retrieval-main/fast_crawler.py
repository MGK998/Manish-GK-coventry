
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from search_engine.parser import parse_publication_page
from search_engine.storage import load_jsonl, append_jsonl
from search_engine.config import PUBLICATIONS_JSONL, INDEX_JSON
from search_engine.indexer import build_documents, build_inverted_index, save_index
import re

PUB_RE = re.compile(r"/en/publications/")
URLS_FILE = "filtered_urls.txt"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

def fetch_and_parse(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            pub = parse_publication_page(url, resp.text)
            if pub.get('title'):
                pub["organisations"] = ["CSM"]
                return pub
    except Exception:
        pass
    return None

def main():
    print(f"Loading URLs from {URLS_FILE}...")
    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Starting parallel crawl of {len(urls)} URLs...")
    new_pubs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_and_parse, urls))
        new_pubs = [r for r in results if r]
        
    print(f"Found {len(new_pubs)} valid publications.")
    
    old = load_jsonl(PUBLICATIONS_JSONL)
    
    # Merge
    by_url = {p.get("publication_url"): p for p in old if p.get("publication_url")}
    for p in new_pubs:
        u = p.get("publication_url")
        if u:
            existing = by_url.get(u, {})
            old_orgs = existing.get("organisations", [])
            new_orgs = p.get("organisations", [])
            merged_orgs = list(set(old_orgs + new_orgs))
            by_url[u] = {**existing, **p, "organisations": merged_orgs}
            
    merged = list(by_url.values())
    print(f"Total merged publications: {len(merged)}")
    
    append_jsonl(PUBLICATIONS_JSONL, merged)
    
    docs = build_documents(merged)
    index, doc_lengths = build_inverted_index(docs)
    save_index(INDEX_JSON, docs, index, doc_lengths)
    print("Done.")

if __name__ == "__main__":
    main()
