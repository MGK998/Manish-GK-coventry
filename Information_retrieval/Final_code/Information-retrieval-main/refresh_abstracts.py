import json
import time
import argparse
from search_engine.parser import parse_publication_page
from search_engine.config import PUBLICATIONS_JSONL, INDEX_JSON
from search_engine.storage import load_jsonl, append_jsonl
from search_engine.indexer import build_documents, build_inverted_index, save_index
import requests

def main():
    parser = argparse.ArgumentParser(description="Refresh abstracts for existing publications.")
    parser.add_argument("--limit", type=int, default=10, help="Limit the number of updates for testing.")
    parser.add_argument("--all", action="store_true", help="Process all publications.")
    args = parser.parse_args()

    print("Loading existing publications...")
    publications = load_jsonl(PUBLICATIONS_JSONL)
    print(f"Loaded {len(publications)} publications.")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    })

    updated_count = 0
    limit = args.limit if not args.all else len(publications)

    for i, pub in enumerate(publications):
        if updated_count >= limit:
            break
        
        url = pub.get("publication_url")
        if not url:
            continue

        # Check if abstract is currently short, authors contain "Profiles", or author_links is missing
        current_abstract = pub.get("abstract", "")
        current_authors = pub.get("authors", [])
        current_author_links = pub.get("author_links", [])
        has_profiles = any(a.lower() == "profiles" for a in current_authors)
        
        needs_refresh = (len(current_abstract) < 500) or has_profiles or (not current_author_links and current_authors) or args.all


        if not needs_refresh:
            continue

        print(f"[{updated_count+1}/{limit}] Refreshing: {pub.get('title')[:50]}...")
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            
            new_data = parse_publication_page(url, r.text)
            new_abstract = new_data.get("abstract", "")
            new_authors = new_data.get("authors", [])
            new_author_links = new_data.get("author_links", [])
            
            changed = False
            if len(new_abstract) > len(current_abstract):
                print(f"  Abstract lengthened: {len(current_abstract)} -> {len(new_abstract)}")
                pub["abstract"] = new_abstract
                changed = True
            
            if new_authors != current_authors or pub.get("author_links") != new_author_links:
                print(f"  Authors updated: {current_authors} -> {new_authors}")
                pub["authors"] = new_authors
                pub["author_links"] = new_author_links
                changed = True

            
            if changed:
                updated_count += 1
            else:
                print("  No changes found.")
            
            # Politeness delay
            time.sleep(1)

        except Exception as e:
            print(f"  Failed to refresh {url}: {e}")

    if updated_count > 0:
        print(f"Saving {len(publications)} records with {updated_count} updates...")
        append_jsonl(PUBLICATIONS_JSONL, publications) # This overwrites because append_jsonl in this codebase seems to be used as save_all in some places? 
        # Wait, let me check search_engine/storage.py
        
        print("Re-building index...")
        docs = build_documents(publications)
        index, doc_lengths = build_inverted_index(docs)
        save_index(INDEX_JSON, docs, index, doc_lengths)
        print("Done.")
    else:
        print("No updates made.")

if __name__ == "__main__":
    main()
