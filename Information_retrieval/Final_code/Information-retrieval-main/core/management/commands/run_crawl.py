
import os
import re
import requests
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from django.core.management.base import BaseCommand
from django.conf import settings
from search_engine.parser import parse_publication_page
from search_engine.storage import load_jsonl, append_jsonl
from search_engine.config import PUBLICATIONS_JSONL, INDEX_JSON
from search_engine.indexer import build_documents, build_inverted_index, save_index

class Command(BaseCommand):
    help = 'Runs a full end-to-end crawl of CSM publications'

    SITEMAP_URL = "https://pureportal.coventry.ac.uk/sitemap/publications.xml"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    }

    def handle(self, *args, **options):
        self.stdout.write("Starting weekly crawl process...")
        
        # 1. Fetch Sitemap
        self.stdout.write("Fetching sitemap...")
        try:
            resp = requests.get(self.SITEMAP_URL, headers=self.HEADERS, timeout=30)
            resp.raise_for_status()
            content = resp.text
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to fetch sitemap: {e}"))
            return

        # 2. Parse for 200 diverse URLs (sampling strategy)
        urls = re.findall(r'<loc>(.*?)</loc>', content)
        self.stdout.write(f"Found {len(urls)} URLs. Sampling 200...")
        
        # Strategy: Last 100 (newest) + 100 random from the rest
        if len(urls) > 200:
            tail = urls[-100:]
            middle = random.sample(urls[:-100], 100)
            selected = list(set(tail + middle))
        else:
            selected = urls

        # 3. Parallel Crawl
        self.stdout.write(f"Crawling {len(selected)} publications in parallel...")
        new_pubs = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.fetch_and_parse, selected))
            new_pubs = [r for r in results if r]

        self.stdout.write(f"Successfully parsed {len(new_pubs)} publications.")

        # 4. Merge and Save
        old = load_jsonl(PUBLICATIONS_JSONL)
        by_url = {p.get("publication_url"): p for p in old if p.get("publication_url")}
        for p in new_pubs:
            u = p.get("publication_url")
            if u:
                by_url[u] = {**by_url.get(u, {}), **p, "organisations": list(set(by_url.get(u, {}).get("organisations", []) + ["CSM"]))}
        
        merged = list(by_url.values())
        append_jsonl(PUBLICATIONS_JSONL, merged)
        
        # 5. Re-index
        docs = build_documents(merged)
        index, doc_lengths = build_inverted_index(docs)
        save_index(INDEX_JSON, docs, index, doc_lengths)

        self.stdout.write(self.style.SUCCESS(f"Crawl finished. Total publications: {len(merged)}"))

    def fetch_and_parse(self, url):
        try:
            # Slight sleep to be polite
            import time
            time.sleep(random.uniform(0.5, 1.5))
            resp = requests.get(url, headers=self.HEADERS, timeout=20)
            if resp.status_code == 200:
                pub = parse_publication_page(url, resp.text)
                if pub.get('title'):
                    return pub
        except Exception:
            pass
        return None
