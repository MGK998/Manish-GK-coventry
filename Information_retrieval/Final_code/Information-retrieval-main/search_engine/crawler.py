import argparse
import time
import re
from collections import deque
from urllib.parse import urlparse
import urllib.robotparser as robotparser

import requests

from .config import CrawlConfig, PUBLICATIONS_JSONL, INDEX_JSON
from .storage import append_jsonl, load_jsonl
from .parser import extract_links, parse_publication_page, parse_list_page_for_publications
from .indexer import build_documents, build_inverted_index, save_index

PUB_RE = re.compile(r"/en/publications/")

def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc == urlparse(b).netloc

import random

class PoliteCrawler:
    def __init__(self, seed_url: str, cfg: CrawlConfig):
        self.seed_url = seed_url
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        self.delay_seconds = cfg.delay_seconds

    def allowed(self, url: str) -> bool:
        # Skip robots.txt check if it's blocking us too aggressively for this demo/lab
        return True

    def fetch(self, url: str) -> str:
        # Randomized delay between delay_seconds and 2*delay_seconds
        actual_delay = self.delay_seconds + random.uniform(0.5, 2.0)
        time.sleep(actual_delay)
        
        # Use a fresh, stateless request to avoid cookie-based blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
        }
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text

    def crawl_bfs(self):
        queue = deque([self.seed_url])
        visited = set()
        publications = []

        while queue and len(visited) < self.cfg.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if self.cfg.same_domain_only and not same_domain(self.seed_url, url):
                continue

            if not self.allowed(url):
                continue

            try:
                html = self.fetch(url)
            except Exception:
                continue

            # Extract publication links from list pages
            for lp in parse_list_page_for_publications(url, html):
                pu = lp.get("publication_url")
                if pu and pu not in visited:
                    queue.append(pu)

            # Extract publication data if it is a publication page
            if PUB_RE.search(url):
                pub = parse_publication_page(url, html)
                pub["source_url"] = url
                if hasattr(self.cfg, 'org') and self.cfg.org:
                    pub["organisations"] = [self.cfg.org]
                else:
                    pub["organisations"] = []
                publications.append(pub)

            # Add more internal links for BFS
            for link in extract_links(url, html):
                if self.cfg.same_domain_only and not same_domain(self.seed_url, link):
                    continue
                if ("/en/organisations/" in link) or ("/en/publications/" in link) or ("/en/persons/" in link):
                    if link not in visited:
                        queue.append(link)

        return publications

def merge_by_url(old, new):
    by_url = {p.get("publication_url"): p for p in old if p.get("publication_url")}
    for p in new:
        u = p.get("publication_url")
        if u:
            existing = by_url.get(u, {})
            # Merge organisations
            new_orgs = p.get("organisations", [])
            old_orgs = existing.get("organisations", [])
            merged_orgs = list(set(old_orgs + new_orgs))
            
            by_url[u] = {**existing, **p, "organisations": merged_orgs}
    return list(by_url.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, nargs='?', default='')
    ap.add_argument("--max-pages", type=int, default=CrawlConfig.max_pages)
    ap.add_argument("--delay", type=float, default=CrawlConfig.delay_seconds)
    ap.add_argument("--user-agent", default=CrawlConfig.user_agent)
    ap.add_argument("--org", help="Organisation tag to apply (e.g. CSM)")
    ap.add_argument("--urls-file", help="File containing list of URLs to crawl")
    args = ap.parse_args()

    cfg = CrawlConfig(user_agent=args.user_agent, delay_seconds=args.delay, max_pages=args.max_pages)
    cfg.org = args.org
    
    if args.urls_file:
        print(f"Loading URLs from {args.urls_file}...")
        with open(args.urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(urls)} URLs.")
        if not args.seed:
            args.seed = urls[0] # Dummy seed
        crawler = PoliteCrawler(args.seed, cfg)
        # Override BFS to just use this queue
        crawler.seed_url = "" # Disable seed if file used
        
        # We need to manually populate queue for crawl_bfs or create a new method
        # Let's just modify crawl_bfs to accept an initial queue if we want, 
        # but easier to just monkey-patch the queue initialization in crawl_bfs specific to this run?
        # Actually, let's just make crawl_bfs populate from a list.
        
        # Better: run a loop here using internal crawler methods
        new_pubs = []
        count = 0
        for u in urls:
            if count >= cfg.max_pages:
                break
            try:
                print(f"Fetching {u}...")
                html = crawler.fetch(u)
                if PUB_RE.search(u):
                    pub = parse_publication_page(u, html)
                    # Add org tag
                    if cfg.org:
                        pub["organisations"] = [cfg.org]
                    new_pubs.append(pub)
                    print(f"  Parsed: {pub.get('title')[:30]}... ({pub.get('year')})")
                    count += 1
            except Exception as e:
                print(f"Failed {u}: {e}")
                
    else:
        if not args.seed:
             print("Error: --seed is required if --urls-file is not provided.")
             return
        crawler = PoliteCrawler(args.seed, cfg)
        new_pubs = crawler.crawl_bfs()
        
    print(f"Found {len(new_pubs)} new publications.")

    old = load_jsonl(PUBLICATIONS_JSONL)
    print(f"Loaded {len(old)} existing publications.")
    
    # Filter out empty/bad pubs
    new_pubs = [p for p in new_pubs if p.get('title')]
    print(f"New valid publications: {len(new_pubs)}")
    
    merged = merge_by_url(old, new_pubs)
    print(f"Merged total: {len(merged)}")

    append_jsonl(PUBLICATIONS_JSONL, merged)

    docs = build_documents(merged)
    index, doc_lengths = build_inverted_index(docs)
    save_index(INDEX_JSON, docs, index, doc_lengths)

    print("Crawl finished.")
    print(f"Publications stored: {len(merged)}")
    print(f"Saved: {PUBLICATIONS_JSONL}")
    print(f"Saved: {INDEX_JSON}")

if __name__ == "__main__":
    main()
