
import json
import re
from pathlib import Path
from search_engine.config import PUBLICATIONS_JSONL, INDEX_JSON
from search_engine.storage import load_jsonl, append_jsonl
from search_engine.indexer import build_documents, build_inverted_index, save_index

def create_slug(title, limit=50):
    # Standard slugify logic
    s = title.lower()
    # Replace non-breaking hyphen (U+2011) and other dashes with standard hyphen
    s = s.replace('\u2011', '-')
    s = s.replace('\u2013', '-')
    s = s.replace('\u2014', '-')
    # Remove standard punctuation
    s = re.sub(r'[^\w\s-]', '', s)
    # Replace whitespace with hyphens
    s = re.sub(r'[-\s]+', '-', s).strip('-')
    
    # APPLY 50 CHARACTER TRUNCATION (Definitive for Coventry PurePortal 2025+)
    if len(s) > limit:
        s = s[:limit].strip('-')
    return s

def repair_url(url, title):
    base_portal = "https://pureportal.coventry.ac.uk/en/publications/"
    if not url.startswith(base_portal):
        return url
    
    slug = create_slug(title)
    return f"{base_portal}{slug}/"

def main():
    print(f"Repairing URLs with definitive 50-char truncation in {PUBLICATIONS_JSONL}...")
    pubs = load_jsonl(PUBLICATIONS_JSONL)
    fixed_count = 0
    
    for p in pubs:
        original = p.get('publication_url', '')
        title = p.get('title', '')
        if title:
            repaired = repair_url(original, title)
            if repaired != original:
                p['publication_url'] = repaired
                fixed_count += 1
            
    print(f"Repaired {fixed_count} URLs.")
    append_jsonl(PUBLICATIONS_JSONL, pubs)
    
    print("Re-indexing...")
    docs = build_documents(pubs)
    index, doc_lengths = build_inverted_index(docs)
    save_index(INDEX_JSON, docs, index, doc_lengths)
    print("Done.")

if __name__ == "__main__":
    main()
